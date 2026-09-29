#!/usr/bin/env bash
# Two GPUs, twenty scenes: original model plus calibrated camera projection.
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CAMERA_ONLY_PYTHON:-/home/lf/miniconda3/envs/JADA/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "找不到 Python: $PYTHON_BIN" >&2; exit 1; }
exec "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/camera_only_run_all.py" "$@"
