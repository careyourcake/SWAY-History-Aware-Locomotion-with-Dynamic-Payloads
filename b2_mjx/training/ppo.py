"""Brax PPO integration and temporal actor networks."""

from __future__ import annotations

import csv
import functools
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.training import distribution, networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo


class TemporalActor(nn.Module):
    output_size: int
    hidden_size: int = 64
    mlp_layers: tuple[int, ...] = (128, 64)

    @nn.compact
    def __call__(self, current, history):
        sequence = nn.RNN(nn.GRUCell(features=self.hidden_size))(history)
        latent = sequence[..., -1, :]
        x = jnp.concatenate((current, latent), axis=-1)
        for size in self.mlp_layers:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.output_size)(x)


def network_factory(policy_config: Mapping[str, Any], hidden_layers):
    """Creates a Brax-compatible PPO factory for GRU and baseline actors."""
    policy_cfg = dict(policy_config)
    value_layers = tuple(int(size) for size in hidden_layers)

    def make(observation_size, action_size, preprocess_observations_fn=running_statistics.normalize):
        kind = policy_cfg["type"]
        action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
        if kind == "gru":
            module = TemporalActor(
                output_size=action_distribution.param_size,
                hidden_size=int(policy_cfg.get("gru_hidden_size", 64)),
                mlp_layers=tuple(int(x) for x in policy_cfg.get("actor_hidden_layers", [128, 64])),
            )
            state_shape = observation_size["state"] if isinstance(observation_size["state"], tuple) else (int(observation_size["state"]),)
            dummy_current = jnp.zeros((1,) + tuple(state_shape))
            state_last_dim = int(state_shape[-1] if isinstance(state_shape[-1], int) else state_shape[-1][-1])
            if "history_length" in policy_cfg:
                history_length = int(policy_cfg["history_length"])
            else:
                history_length = int(observation_size["history"]) // state_last_dim
            history_shape = (history_length, state_last_dim)
            dummy_history = jnp.zeros((1,) + history_shape)

            def init(key):
                return module.init(key, dummy_current, dummy_history)

            def apply(processor_params, policy_params, obs):
                current = preprocess_observations_fn(obs["state"], networks.normalizer_select(processor_params, "state"))
                history = preprocess_observations_fn(obs["history"], networks.normalizer_select(processor_params, "history"))
                history = history.reshape(history.shape[:-1] + history_shape)
                return module.apply(policy_params, current, history)

            policy_network = networks.FeedForwardNetwork(init=init, apply=apply)
        else:
            obs_key = "stack" if kind == "frame_stack" else "state"
            policy_network = networks.make_policy_network(
                action_distribution.param_size,
                observation_size,
                preprocess_observations_fn=preprocess_observations_fn,
                hidden_layer_sizes=tuple(int(x) for x in policy_cfg.get("actor_hidden_layers", hidden_layers)),
                activation=nn.relu,
                obs_key=obs_key,
            )
        value_network = networks.make_value_network(
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_layers,
            activation=nn.relu,
            obs_key="privileged_state",
        )
        return ppo_networks.PPONetworks(policy_network, value_network, action_distribution)

    return make


def append_metrics(path: Path, step: int, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {"step": int(step)}
    for key, value in metrics.items():
        try:
            flat[key] = float(value)
        except (TypeError, ValueError):
            continue
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def select_best_checkpoint(run_dir: Path) -> Path:
    """Select the best validation checkpoint and persist the selection."""
    metrics_path = run_dir / "metrics.csv"
    checkpoints = {int(p.name): p for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit()}
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            try:
                step = int(float(row["step"]))
            except (KeyError, TypeError, ValueError):
                continue
            if step in checkpoints:
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No evaluated checkpoints found in {run_dir}")
    error_key = "eval/episode_velocity_squared_error"
    length_key = "eval/avg_episode_length"
    if error_key in rows[0] and rows[0][error_key]:
        max_length = max(float(r[length_key]) for r in rows)
        viable = [r for r in rows if float(r[length_key]) >= 0.8 * max_length]
        best = min(viable, key=lambda r: float(r[error_key]) / max(float(r[length_key]), 1.0))
        key = "mean_velocity_squared_error"
        value = float(best[error_key]) / max(float(best[length_key]), 1.0)
    else:
        key = "eval/episode_reward"
        best = max(rows, key=lambda r: float(r[key]))
        value = float(best[key])
    step = int(float(best["step"]))
    manifest = {"step": step, "metric": key, "value": value, "checkpoint": str(checkpoints[step].resolve())}
    with (run_dir / "best_checkpoint.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return checkpoints[step]


def train_ppo(env, config: Mapping[str, Any], run_dir: Path, resume: str | None = None):
    train_cfg = dict(config["training"])
    hidden = train_cfg.pop("network_hidden_layers")
    randomization_fn = train_cfg.pop("randomization_fn", None)
    progress_path = run_dir / "metrics.csv"

    def progress(step, metrics):
        append_metrics(progress_path, step, metrics)
        reward = metrics.get("eval/episode_reward", metrics.get("training/sps", "n/a"))
        print(f"step={step} metric={reward}", flush=True)

    allowed = {
        "num_timesteps", "num_envs", "episode_length", "action_repeat", "num_evals",
        "num_eval_envs", "learning_rate", "entropy_cost", "discounting", "unroll_length",
        "batch_size", "num_minibatches", "num_updates_per_batch", "normalize_observations",
        "reward_scaling", "clipping_epsilon", "gae_lambda", "run_evals",
    }
    kwargs = {key: value for key, value in train_cfg.items() if key in allowed}
    kwargs.update(
        environment=env, seed=int(config["seed"]),
        network_factory=network_factory(config["policy"], hidden), progress_fn=progress,
        save_checkpoint_path=str((run_dir / "checkpoints").resolve()), restore_checkpoint_path=str(Path(resume).resolve()) if resume else None,
        randomization_fn=randomization_fn,
    )
    result = ppo.train(**kwargs)
    if progress_path.exists() and (run_dir / "checkpoints").exists():
        try:
            select_best_checkpoint(run_dir)
        except FileNotFoundError:
            pass
    return result
