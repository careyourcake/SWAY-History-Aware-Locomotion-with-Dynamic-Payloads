"""Configuration and validation for capability-gated curriculum stages."""

from __future__ import annotations

from copy import deepcopy
import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from b2_mjx.config import load_config, validate_config, without_internal_keys


METHOD_POLICY = {"mlp_dynamic": "mlp", "stack5_dynamic": "frame_stack", "gru_dynamic": "gru"}


def deep_update(target: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def load_curriculum(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    base_path = Path(spec["base_config"])
    if not base_path.is_absolute():
        candidate = path.parent / base_path
        base_path = candidate if candidate.exists() else path.parent.parent / base_path
    spec["base_config"] = str(base_path.resolve())
    validate_curriculum(spec)
    return spec


def _difficulty(stage: Mapping[str, Any]) -> tuple[float, ...]:
    env = stage["environment"]
    dynamic, randomization = env["dynamic_payload"], env["randomization"]
    mass = dynamic["mass"]
    angles = dynamic["initial_angle_deg"]
    return (
        float(bool(dynamic["enabled"])), float(mass[1]),
        float(dynamic["length"][1]) - float(dynamic["length"][0]),
        max(abs(float(angles[0])), abs(float(angles[1]))),
        float(randomization["observation_noise"]), float(randomization["action_delay_probability"]),
        float(randomization["push_probability"]), float(randomization["push_velocity"]),
        float(randomization["motor_strength"][1]) - float(randomization["motor_strength"][0]),
        float(randomization["friction"][1]) - float(randomization["friction"][0]),
    )


def validate_curriculum(spec: Mapping[str, Any]) -> None:
    stages = spec.get("stages", [])
    if [stage.get("name") for stage in stages] != ["c0_locomotion", "c1_light_swing", "c2_target_swing", "c3_robust_swing"]:
        raise ValueError("Curriculum must define C0-C3 in order")
    budgets = [int(stage["stage_timesteps"]) for stage in stages]
    if any(budget <= 0 for budget in budgets) or sum(budgets) != int(spec["total_timesteps"]):
        raise ValueError("Stage budgets must be positive and sum to total_timesteps")
    difficulties = [_difficulty(stage) for stage in stages]
    for earlier, later in zip(difficulties, difficulties[1:]):
        if any(after < before for before, after in zip(earlier, later)):
            raise ValueError("Curriculum difficulty must be monotonic")


def stage_config(spec: Mapping[str, Any], index: int, method: str, seed: int) -> dict[str, Any]:
    if method not in METHOD_POLICY:
        raise ValueError(f"Unsupported curriculum method: {method}")
    config = load_config(spec["base_config"])
    stage = spec["stages"][index]
    deep_update(config["environment"], stage["environment"])
    config["policy"]["type"] = METHOD_POLICY[method]
    config["experiment_name"] = f"{method}_{stage['name']}"
    config["seed"] = int(seed)
    config["training"]["num_timesteps"] = int(stage["stage_timesteps"])
    config["curriculum"] = {
        "stage_index": index, "stage_name": stage["name"], "stage_timesteps": int(stage["stage_timesteps"]),
        "source": None if index == 0 else spec["stages"][index - 1]["name"],
    }
    validate_config(config)
    return without_internal_keys(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, default=Path("configs/curriculum.yaml"))
    parser.add_argument("--stage", type=int, required=True, choices=range(4))
    parser.add_argument("--method", required=True, choices=METHOD_POLICY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = stage_config(load_curriculum(args.curriculum), args.stage, args.method, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
