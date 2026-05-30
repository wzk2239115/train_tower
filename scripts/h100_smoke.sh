#!/usr/bin/env bash
# 8× H100/H800 smoke: Flow Tower world_pt, 20 steps by default.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=h100_common.sh
source "${ROOT}/scripts/h100_common.sh"
h100_env_setup

export MAX_STEPS="${MAX_STEPS:-20}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/pretrain/world_pt_h800_smoke}"

# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"
train_env_setup
train_env_print_training_summary config "${CONFIG}"

echo "[h100_smoke] NUM_GPUS=${NUM_GPUS} MAX_STEPS=${MAX_STEPS} OUTPUT_DIR=${OUTPUT_DIR}"
echo "[h100_smoke] CONFIG=${CONFIG} MASTER=${MASTER_ADDR}:${MASTER_PORT}"

"${ROOT}/scripts/fetch_tokenizer.sh"

h100_run_torchrun "$@"
