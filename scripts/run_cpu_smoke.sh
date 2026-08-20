#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-runs/smoke_matrix}"
for method in gru_dynamic mlp_dynamic stack5_dynamic mlp_static; do
  python -m b2_mjx.train --config configs/ppo_b2_payload_cpu.yaml \
    --method "$method" --seed 0 --run-dir "$output_root/$method/seed_0"
done
