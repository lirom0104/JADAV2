#!/usr/bin/env bash
# Train and evaluate every RGBT-Scenes scene sequentially on physical GPU 0.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env ODB_GPUS=0 bash "$REPO_ROOT/odb_train_30k_10scenes.sh" "$@"
