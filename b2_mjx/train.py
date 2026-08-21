"""Train PPO on B2PayloadVelocity-v0."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import platform
from importlib.metadata import PackageNotFoundError, version

import jax
import yaml

from b2_mjx.config import load_config, without_internal_keys
from b2_mjx.envs import B2PayloadEnv
from b2_mjx.envs.randomization import make_domain_randomization
from b2_mjx.training.ppo import train_ppo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--run-dir")
    parser.add_argument("--method", choices=["gru_dynamic", "pdb_prc", "mlp_dynamic", "stack5_dynamic", "mlp_static"])
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.method is not None:
        method = args.method
        config["policy"]["type"] = {"gru_dynamic": "gru", "pdb_prc": "pdb_prc", "mlp_dynamic": "mlp", "stack5_dynamic": "frame_stack", "mlp_static": "mlp"}[method]
        config["environment"]["dynamic_payload"]["enabled"] = method != "mlp_static"
        config["experiment_name"] = method
    devices = jax.devices()
    print(f"JAX backend={jax.default_backend()} devices={devices}")
    if config["require_gpu"] and jax.default_backend() != "gpu":
        raise RuntimeError("This configuration requires a CUDA GPU; refusing CPU fallback.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"runs/{stamp}_{config['experiment_name']}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("checkpoints", "evaluation", "videos"):
        (run_dir / directory).mkdir(exist_ok=True)
    config["environment"]["history_length"] = config["policy"]["history_length"]
    config["environment"]["history_stride"] = config["policy"]["history_stride"]
    env = B2PayloadEnv(config["environment"])
    randomization_fn = make_domain_randomization(config["environment"])
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(without_internal_keys(config), stream, sort_keys=False)
    packages = {}
    for package in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "flax", "optax"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "jax_backend": jax.default_backend(), "devices": [str(device) for device in devices],
        "packages": packages,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    config["training"]["randomization_fn"] = randomization_fn
    train_ppo(env, config, run_dir, args.resume)


if __name__ == "__main__":
    main()
