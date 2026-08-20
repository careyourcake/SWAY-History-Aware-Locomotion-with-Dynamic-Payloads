#!/usr/bin/env bash
set -euo pipefail

seed="${1:-0}"
output_root="${2:-runs/ablation}"
config=configs/ppo_b2_payload.yaml

for method in mlp_dynamic stack5_dynamic gru_dynamic; do
  run_dir="$output_root/$method/seed_$seed"
  python -m b2_mjx.train --config "$config" --method "$method" --seed "$seed" --run-dir "$run_dir"
  python -m b2_mjx.evaluate --checkpoint "$run_dir" --episodes 100
done
