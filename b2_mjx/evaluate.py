"""Evaluate a checkpoint and export reproducible episode metrics."""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy

import jax
import jax.numpy as jp

from b2_mjx.envs import B2PayloadEnv
from b2_mjx.envs.randomization import payload_adjusted_system
from b2_mjx.evaluation.policy import load_checkpoint_policy
from b2_mjx.evaluation.scenarios import SCENARIOS
from b2_mjx.evaluation.metrics import EpisodeTotals


def mechanical_cot(energy: float, mass: float, distance: float) -> float:
    return energy / (mass * 9.81 * distance) if mass > 0 and distance > 1e-6 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--target-distance", type=float, default=10.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    config, run_dir, policy = load_checkpoint_policy(args.checkpoint)
    scenarios = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    rows = []
    dynamic = bool(config["environment"]["dynamic_payload"]["enabled"])
    for scenario_name, scenario in scenarios.items():
        env_config = deepcopy(config["environment"])
        env_config["command_resample_steps"] = env_config["episode_length"] + 1
        env = B2PayloadEnv(env_config)
        payload = jp.array([scenario["mass"], 0.0, 0.0, 0.30])
        env.sys = payload_adjusted_system(env.sys, payload, dynamic=dynamic, length=scenario["length"])
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        for episode in range(args.episodes):
            key = jax.random.PRNGKey(int(config["seed"]) + episode)
            state = reset(key)
            state = env.set_command(state, jp.array(scenario["command"]))
            squared_error = yaw_error = posture = energy = peak_torque = inference_ms = distance = 0.0
            start_x = float(state.pipeline_state.q[0])
            totals = EpisodeTotals(start_x)
            steps = 0
            for _ in range(env_config["episode_length"]):
                key, action_key = jax.random.split(key)
                started = time.perf_counter(); action, _ = policy(state.obs, action_key); jax.block_until_ready(action)
                inference_ms += (time.perf_counter() - started) * 1000.0
                state = step(state, action); jax.block_until_ready(state.reward)
                command = state.info["command"]
                squared_error += float((state.metrics["x_velocity"] - command[0]) ** 2 + (state.metrics["y_velocity"] - command[1]) ** 2)
                yaw_error += float((state.metrics["yaw_rate"] - command[2]) ** 2)
                posture += float(state.metrics["posture_error"])
                totals.update(float(state.pipeline_state.q[0]), float(state.metrics["power"]), env.dt)
                peak_torque = max(peak_torque, float(jp.max(jp.abs(state.info["last_torque"]))))
                distance = float(state.pipeline_state.q[0]) - start_x; steps += 1
                if distance >= args.target_distance or float(state.done): break
            fallen = int(float(state.metrics["fallen"]) > 0.0)
            timeout = int(float(state.info["timeout"]) > 0.0)
            total_mass = float(jp.sum(env.sys.body_mass))
            assert totals.steps == steps
            distance, energy = totals.distance, totals.mechanical_energy
            rows.append({
                "method": config["experiment_name"], "train_seed": config["seed"], "scenario": scenario_name,
                "episode": episode, "mass": scenario["mass"], "length": scenario["length"],
                "success": int(distance >= args.target_distance), "steps": steps, "distance": distance,
                "fallen": fallen, "timeout": timeout,
                "velocity_rmse": (squared_error / steps) ** 0.5, "yaw_rmse": (yaw_error / steps) ** 0.5,
                "posture_error": posture / steps, "mechanical_energy": energy,
                "cot": mechanical_cot(energy, total_mass, distance), "peak_torque": peak_torque,
                "inference_ms": inference_ms / steps,
            })
    output = run_dir / "evaluation" / "metrics.csv" if args.output is None else __import__("pathlib").Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} episode records to {output}")


if __name__ == "__main__": main()
