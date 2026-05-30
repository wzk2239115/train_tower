#!/usr/bin/env bash
# 8× H100/H800 full training: Flow Tower world_pt (configs/train/world_pt_h800.yaml).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=h100_common.sh
source "${ROOT}/scripts/h100_common.sh"
h100_env_setup

# Optional overrides: MAX_STEPS, OUTPUT_DIR, DATASETS (see scripts/train_env.sh).
# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"
train_env_setup
train_env_print_training_summary config "${CONFIG}"

echo "[h100_train] NUM_GPUS=${NUM_GPUS} CONFIG=${CONFIG}"
echo "[h100_train] MASTER=${MASTER_ADDR}:${MASTER_PORT} USE_DEEPSPEED=${USE_DEEPSPEED}"
[[ -n "${MAX_STEPS:-}" ]] && echo "[h100_train] MAX_STEPS=${MAX_STEPS}"
[[ -n "${OUTPUT_DIR:-}" ]] && echo "[h100_train] OUTPUT_DIR=${OUTPUT_DIR}"
[[ -n "${DATASETS:-}" ]] && echo "[h100_train] DATASETS=${DATASETS}"

"${ROOT}/scripts/fetch_tokenizer.sh"

h100_run_torchrun "$@"
