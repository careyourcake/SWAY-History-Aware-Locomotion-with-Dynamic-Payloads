#!/usr/bin/env bash
set -euo pipefail

method="${1:?usage: run_curriculum.sh METHOD SEED OUTPUT_ROOT}"
seed="${2:?usage: run_curriculum.sh METHOD SEED OUTPUT_ROOT}"
output_root="${3:?usage: run_curriculum.sh METHOD SEED OUTPUT_ROOT}"

case "$method" in
  mlp_dynamic|stack5_dynamic|gru_dynamic) ;;
  *) echo "unsupported method: $method" >&2; exit 2 ;;
esac

python -m b2_mjx.run_curriculum "$method" "$seed" "$output_root"
