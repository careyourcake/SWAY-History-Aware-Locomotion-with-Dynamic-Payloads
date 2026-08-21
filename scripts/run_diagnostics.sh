#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_diagnostics.sh B0|B1|B2|B3 [seed] [output_root]}"
seed="${2:-0}"
output_root="${3:-runs/diagnostics}"

case "$stage" in
  B0) config=configs/b0_sanity.yaml; name=b0_sanity; scenario=unloaded ;;
  B1) config=configs/b1_locomotion.yaml; name=b1_no_load_reference; scenario=unloaded ;;
  B2) config=configs/b2_static.yaml; name=b2_static_load_reference; scenario=mass_3 ;;
  B3) config=configs/b3_dynamic_mlp.yaml; name=b3_dynamic_load_reference; scenario=mass_3 ;;
  *) echo "unknown diagnostic stage: $stage" >&2; exit 2 ;;
esac

run_dir="$output_root/$name/seed_$seed"
python -m b2_mjx.train --config "$config" --seed "$seed" --run-dir "$run_dir"
python -m b2_mjx.evaluate --checkpoint "$run_dir" --scenario "$scenario" --episodes 100
python scripts/check_stage.py "$run_dir/evaluation/metrics.csv"
