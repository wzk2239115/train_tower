#!/usr/bin/env bash
# Flow-JEPA Tower Stage 0: world representation (JEPA + semantic ELF)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TOWER_NO_DEEPSPEED="${TOWER_NO_DEEPSPEED:-1}"
python -m tower.cli train --config configs/train/world_pt.yaml "$@"
