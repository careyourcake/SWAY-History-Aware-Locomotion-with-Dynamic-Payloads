"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration {path} must contain a mapping.")
    config = deepcopy(config)
    config["_config_path"] = str(path.resolve())
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"experiment_name", "seed", "require_gpu", "environment", "policy", "training"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    env = config["environment"]
    for section in ("command", "payload", "dynamic_payload", "randomization", "reward"):
        if section not in env:
            raise ValueError(f"Missing environment.{section}")
    if int(env["episode_length"]) <= 0:
        raise ValueError("environment.episode_length must be positive")
    reward = env["reward"]
    for name in ("tracking_linear", "tracking_yaw", "upright", "height", "stand"):
        if float(reward[name]) < 0:
            raise ValueError(f"environment.reward.{name} must be non-negative")
    for name in ("vertical_velocity", "roll_pitch_rate", "torque", "power", "action_rate", "foot_slip", "collision"):
        if float(reward[name]) > 0:
            raise ValueError(f"environment.reward.{name} must be non-positive")
    policy = config["policy"]
    if policy.get("type") not in {"gru", "mlp", "frame_stack"}:
        raise ValueError("policy.type must be gru, mlp, or frame_stack")
    if int(policy.get("history_length", 0)) <= 0 or int(policy.get("history_stride", 0)) <= 0:
        raise ValueError("policy history_length and history_stride must be positive")


def without_internal_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
