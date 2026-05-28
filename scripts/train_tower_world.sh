#!/usr/bin/env bash
# Flow-JEPA Tower Stage 0: world representation (JEPA + semantic ELF)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TOWER_NO_DEEPSPEED="${TOWER_NO_DEEPSPEED:-1}"
# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"
train_env_setup
train_env_print_training_summary config configs/train/world_pt.yaml
python -m tower.cli train --config configs/train/world_pt.yaml "$@"
