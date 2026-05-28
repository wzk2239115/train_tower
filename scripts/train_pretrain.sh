#!/usr/bin/env bash
# Full 0→1 scratch pretrain: UW → Gen PT → Uni MT → Uni SFT
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"
train_env_setup

"${ROOT}/scripts/fetch_tokenizer.sh"

run_stage() {
  local config="$1"
  local ckpt_dir="$2"
  shift 2
  echo "==> Training with ${config} (NUM_GPUS=${NUM_GPUS}, TOWER_NO_DEEPSPEED=${TOWER_NO_DEEPSPEED:-0})"
  train_env_print_training_summary config "${config}"
  if [[ "${config}" != *"understanding_warmup"* && ! -d "${ckpt_dir}" ]]; then
    echo "Missing checkpoint: ${ckpt_dir}" >&2
    exit 1
  fi
  torchrun --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    -m tower.cli train --config "${config}" "${TRAIN_ENV_EXTRA[@]}" "$@"
}

run_stage configs/train/understanding_warmup.yaml "" "$@"
run_stage configs/train/generation_pt.yaml outputs/pretrain/uw "$@"
run_stage configs/train/unified_mt.yaml outputs/pretrain/gen_pt "$@"
run_stage configs/train/unified_sft.yaml outputs/pretrain/mt "$@"

echo "==> 0→1 scratch pretrain complete: outputs/pretrain/sft"
