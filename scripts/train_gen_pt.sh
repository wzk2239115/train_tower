#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "${NUM_GPUS}" -le 1 && "${USE_DEEPSPEED:-0}" != "1" ]]; then
  export TOWER_NO_DEEPSPEED=1
fi
torchrun --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  -m tower.cli train --stage generation_pt "$@"
