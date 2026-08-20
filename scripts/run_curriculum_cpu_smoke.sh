#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-runs/curriculum_smoke}"
for method in mlp_dynamic stack5_dynamic gru_dynamic; do
  method_root="$output_root/$method"
  python -m b2_mjx.curriculum --curriculum configs/curriculum_cpu.yaml --stage 0 --method "$method" --output "$method_root/c0.yaml"
  python -m b2_mjx.train --config "$method_root/c0.yaml" --run-dir "$method_root/c0"
  checkpoint="$(find "$method_root/c0/checkpoints" -mindepth 1 -maxdepth 1 -type d | sort -n | tail -n 1)"
  test -n "$checkpoint"
  python -m b2_mjx.curriculum --curriculum configs/curriculum_cpu.yaml --stage 1 --method "$method" --output "$method_root/c1.yaml"
  python -m b2_mjx.train --config "$method_root/c1.yaml" --resume "$checkpoint" --run-dir "$method_root/c1"
done
