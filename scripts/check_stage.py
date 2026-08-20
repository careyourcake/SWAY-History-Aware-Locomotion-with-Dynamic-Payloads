#!/usr/bin/env python3
"""Fail a training pipeline when a baseline stage misses its acceptance gate."""

import argparse
import json
from pathlib import Path

from b2_mjx.evaluation.gates import evaluate_gate, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    parser.add_argument("--max-velocity-rmse", type=float, default=0.45)
    parser.add_argument("--max-fall-rate", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_gate(
        read_csv(args.metrics), expected_episodes=args.episodes,
        min_success_rate=args.min_success_rate, max_velocity_rmse=args.max_velocity_rmse,
        max_fall_rate=args.max_fall_rate,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
