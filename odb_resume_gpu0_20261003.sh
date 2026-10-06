#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/lf/code/Our_Project-New-1-1"
DATA_ROOT="/home/lf/data/thermal3dgs/RGBT-Scenes"
OUTPUT_ROOT="/home/lf/code/Our_Project-New-1-1/output/odb_30k_20261003_020148_318442"
PYTHON="/home/lf/miniconda3/envs/thermalgaussian/bin/python"
PORT=6010

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"

DETAIL_ARGS=(
    --use_detail_basis
    --detail_basis_mode oriented_hermite
    --detail_basis_scale 0.08
    --detail_basis_thermal_scale 0.06
    --detail_basis_lr 0.001
    --detail_basis_reg_weight 0.005
    --detail_edge_weight 0.05
    --detail_basis_start_iter 26000
)

for scene in Parterre Truck RotaryKiln; do
    result_dir="$OUTPUT_ROOT/$scene"
    if [[ -f "$result_dir/results.json" ]]; then
        echo "[$(date +%T)] SKIP $scene (results.json already exists)"
        continue
    fi

    echo "[$(date +%T)] START $scene on physical GPU 0"
    env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/train.py" \
        -s "$DATA_ROOT/$scene" -m "$result_dir" "${DETAIL_ARGS[@]}" \
        --iterations 30000 --save_iterations 30000 \
        --checkpoint_iterations 30000 --port "$PORT" --quiet
    env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/render.py" \
        -m "$result_dir" --iteration 30000 --skip_train --quiet
    env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/metrics.py" \
        -m "$result_dir"
    echo "[$(date +%T)] DONE $scene"
done

echo "[$(date +%T)] GPU 0 recovery scenes finished"
