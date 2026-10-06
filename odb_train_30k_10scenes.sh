#!/usr/bin/env bash
# Train RGB-only ODB from iteration 26k on all ten RGBT-Scenes scenes.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash odb_train_30k_10scenes.sh [--dry-run | --help]

Train all ten RGBT-Scenes scenes from scratch for exactly 30000 iterations,
then render test views and evaluate RGB/thermal PSNR, SSIM and LPIPS.
RGB-only ODB starts after iteration 26000. No checkpoint is loaded.

Environment settings:
  ODB_GPUS           Space-separated physical GPU IDs (default: "0 1")
  ODB_DATA_ROOT      Dataset directory (default: /home/lf/data/thermal3dgs/RGBT-Scenes)
  ODB_OUTPUT_ROOT    New output directory (default: output/odb_30k_<time>_<pid>)
  ODB_PYTHON         Python executable (default: thermalgaussian conda environment)
  ODB_PORT_BASE      First training GUI port; one per GPU (default: 6010)
  ODB_BASELINE_ROOT  Baseline directory for optional comparison; "none" skips it

Examples:
  bash odb_train_30k_10scenes.sh
  ODB_GPUS="0" bash odb_train_30k_10scenes.sh
  bash odb_train_30k_10scenes.sh --dry-run

--dry-run prints commands without training or creating output directories.
Each GPU processes one scene at a time. Logs: <output>/<scene>.log.
Results: summary.json, metrics_30000.csv, and (if available) comparison_30000.json.
Training settings are fixed; ODB_DETAIL_* and ODB_SCENES overrides are not supported.
HELP
}

DRY_RUN=0
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
PYTHON="${ODB_PYTHON:-/home/lf/miniconda3/envs/thermalgaussian/bin/python}"
PYTHON="$(command -v "$PYTHON")" || { echo "Python executable not found" >&2; exit 2; }
DATA_ROOT="${ODB_DATA_ROOT:-/home/lf/data/thermal3dgs/RGBT-Scenes}"
OUTPUT_ROOT="${ODB_OUTPUT_ROOT:-$REPO_ROOT/output/odb_30k_$(date +%Y%m%d_%H%M%S)_$$}"
BASELINE_ROOT="${ODB_BASELINE_ROOT:-$REPO_ROOT/output/camera_only_20261001_142717/RGBT-Scenes}"
PORT_BASE="${ODB_PORT_BASE:-6010}"
ITERATIONS=30000
SCENES=(Building DailyStuff Dimsum Ebike IronIngot LandScape Parterre RoadBlock RotaryKiln Truck)
read -r -a GPUS <<< "${ODB_GPUS:-0 1}"

# These settings retain the 30k schedule with ODB active only for RGB.
DETAIL_ARGS=(--use_detail_basis --detail_basis_mode oriented_hermite
    --detail_basis_scale 0.08
    --detail_basis_lr 0.001 --detail_basis_reg_weight 0.005
    --detail_edge_weight 0.05 --detail_basis_start_iter 26000)
if [[ "${ODB_ITERATIONS:-30000}" != 30000 || -n "${ODB_SCENES:-}" ]]; then
    echo "This reproduction script requires all ten scenes and exactly 30000 iterations." >&2
    exit 2
fi
if (( ${#GPUS[@]} == 0 )); then echo "ODB_GPUS must not be empty" >&2; exit 2; fi
if [[ ! "$PORT_BASE" =~ ^[1-9][0-9]{0,4}$ ]] || (( PORT_BASE + ${#GPUS[@]} - 1 > 65535 )); then
    echo "Invalid ODB_PORT_BASE" >&2; exit 2
fi
declare -A GPU_SEEN=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ || -n "${GPU_SEEN[$gpu]:-}" ]]; then
        echo "ODB_GPUS must contain unique numeric GPU IDs" >&2; exit 2
    fi
    GPU_SEEN[$gpu]=1
done
for scene in "${SCENES[@]}"; do
    for subdir in rgb/train rgb/test thermal/train thermal/test sparse/0; do
        if [[ ! -d "$DATA_ROOT/$scene/$subdir" ]]; then
            echo "Missing dataset directory: $DATA_ROOT/$scene/$subdir" >&2; exit 2
        fi
    done
done
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Output already exists. Set ODB_OUTPUT_ROOT to a new directory: $OUTPUT_ROOT" >&2
    exit 2
fi
REPORT_ARGS=(--candidate "$OUTPUT_ROOT")
if [[ "$BASELINE_ROOT" != none ]]; then
    baseline_complete=1
    for scene in "${SCENES[@]}"; do
        [[ -f "$BASELINE_ROOT/$scene/results.json" ]] || baseline_complete=0
    done
    if (( baseline_complete )); then
        REPORT_ARGS+=(--baseline "$BASELINE_ROOT")
    else
        echo "No complete baseline found; dataset metrics will still be summarized."
    fi
fi
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "Output: $OUTPUT_ROOT"
echo "GPUs: ${GPUS[*]} | Scenes: ${#SCENES[@]} | Iterations: $ITERATIONS | ODB start: 26000"

# Each worker tracks its current Python process so Ctrl-C also stops training.
run_command() {
    printf '%q ' "$@"
    printf '\n'
    if (( DRY_RUN )); then return 0; fi
    "$@" &
    ACTIVE_PID=$!
    local status=0
    wait "$ACTIVE_PID" || status=$?
    ACTIVE_PID=""
    return "$status"
}

run_scene() {
    local scene="$1" gpu="$2" port="$3"
    local result_dir="$OUTPUT_ROOT/$scene"
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO_ROOT/train.py" \
        -s "$DATA_ROOT/$scene" -m "$result_dir" "${DETAIL_ARGS[@]}" \
        --iterations "$ITERATIONS" --save_iterations "$ITERATIONS" \
        --checkpoint_iterations "$ITERATIONS" --port "$port" --quiet || return $?
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO_ROOT/render.py" \
        -m "$result_dir" --iteration "$ITERATIONS" --skip_train --quiet || return $?
    run_command env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO_ROOT/metrics.py" \
        -m "$result_dir" || return $?
    # metrics.py can catch an evaluation error without a nonzero exit code.
    if (( ! DRY_RUN )); then
        "$PYTHON" -c 'import sys; from pathlib import Path; from odb_report import read_scene; read_scene(Path(sys.argv[1]), sys.argv[2])' "$OUTPUT_ROOT" "$scene"
    fi
}

if (( DRY_RUN )); then
    for i in "${!SCENES[@]}"; do
        slot=$((i % ${#GPUS[@]}))
        run_scene "${SCENES[i]}" "${GPUS[slot]}" "$((PORT_BASE + slot))"
    done
    run_command "$PYTHON" "$REPO_ROOT/odb_report.py" "${REPORT_ARGS[@]}"
    exit 0
fi

GPU_LIST="$(IFS=,; echo "${GPUS[*]}")"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON" -c \
    'import sys, torch; n=int(sys.argv[1]); assert torch.cuda.is_available() and torch.cuda.device_count() == n, "Requested CUDA GPUs are unavailable"' "${#GPUS[@]}"
mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT"
cp "$REPO_ROOT/odb_train_30k_10scenes.sh" "$OUTPUT_ROOT/launch_script.sh"

stop_worker() {
    trap - INT TERM
    if [[ -n "${ACTIVE_PID:-}" ]]; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    exit 130
}
worker() {
    local slot="$1" i
    ACTIVE_PID=""
    trap stop_worker INT TERM
    for ((i=slot; i<${#SCENES[@]}; i+=${#GPUS[@]})); do
        echo "[$(date +%T)] START ${SCENES[i]} on GPU ${GPUS[slot]} -> $OUTPUT_ROOT/${SCENES[i]}.log"
        if run_scene "${SCENES[i]}" "${GPUS[slot]}" "$((PORT_BASE + slot))" >"$OUTPUT_ROOT/${SCENES[i]}.log" 2>&1; then
            echo "[$(date +%T)] DONE  ${SCENES[i]}"
        else
            echo "FAILED ${SCENES[i]}; see $OUTPUT_ROOT/${SCENES[i]}.log" >&2
            return 1
        fi
    done
}
PIDS=()
stop_all() {
    trap - INT TERM
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 130
}
trap stop_all INT TERM
for slot in "${!GPUS[@]}"; do
    worker "$slot" &
    PIDS+=("$!")
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
trap - INT TERM
if (( failed )); then
    echo "Some scenes failed. No ten-scene summary was produced. Logs: $OUTPUT_ROOT" >&2
    exit 1
fi
"$PYTHON" "$REPO_ROOT/odb_report.py" "${REPORT_ARGS[@]}" | tee "$OUTPUT_ROOT/report.log"
echo "Completed all ten scenes: $OUTPUT_ROOT"
