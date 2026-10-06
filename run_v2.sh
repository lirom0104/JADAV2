#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
v2_python="${V2_PYTHON:-/home/lf/miniconda3/envs/thermalgaussian/bin/python}"
v2_python_env="$(dirname -- "$(dirname -- "$v2_python")")"
v2_result_dir="$project_dir/output/ir_kernel_structural_v2_full10_01"
v2_output_dir="$project_dir/output/ir_kernel_structural_v2_rerun_$(date +%Y%m%d_%H%M%S)"

# Reuse the CUDA extensions archived with the verified v2 experiment.
export PYTHONPATH="$v2_result_dir/runtime_packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$v2_python_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd -- "$project_dir"
exec "$v2_python" "$project_dir/ir_experiments/run_suite.py" run \
  --repo "$project_dir" \
  --python "$v2_python" \
  --root "$v2_output_dir" \
  --baseline-root "$project_dir/output/odb_30k_20261003_101217_1051565" \
  --iterations 30000 \
  --ir-kernel-ssim-weight 0.01 \
  "$@"
