#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"
train_env_setup
train_env_print_training_summary stage unified_mt
torchrun --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  -m tower.cli train --stage unified_mt "${TRAIN_ENV_EXTRA[@]}" "$@"
