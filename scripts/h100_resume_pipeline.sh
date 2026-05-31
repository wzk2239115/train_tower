#!/usr/bin/env bash
# Continue full pipeline on 8x H100/H800 from world_pt_h800 output:
# world_pt_h800 -> uw_h800 -> gen_pt_h800 -> mt_h800 -> sft_h800
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=h100_common.sh
source "${ROOT}/scripts/h100_common.sh"
h100_env_setup

# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"

WORLD_CKPT="${WORLD_CKPT:-outputs/pretrain/world_pt_h800}"
UW_CONFIG="${UW_CONFIG:-configs/train/understanding_warmup_h800_resume.yaml}"
GEN_PT_CONFIG="${GEN_PT_CONFIG:-configs/train/generation_pt_h800_resume.yaml}"
MT_CONFIG="${MT_CONFIG:-configs/train/unified_mt_h800_resume.yaml}"
SFT_CONFIG="${SFT_CONFIG:-configs/train/unified_sft_h800_resume.yaml}"

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local required_ckpt="${3:-}"

  if [[ -n "${required_ckpt}" && ! -d "${required_ckpt}" ]]; then
    echo "[h100_resume_pipeline] missing required checkpoint dir: ${required_ckpt}" >&2
    exit 1
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "[h100_resume_pipeline] missing config file: ${config_path}" >&2
    exit 1
  fi

  export CONFIG="${config_path}"
  train_env_setup
  train_env_print_training_summary config "${CONFIG}"
  echo "[h100_resume_pipeline] stage=${stage_name} NUM_GPUS=${NUM_GPUS} CONFIG=${CONFIG}"
  echo "[h100_resume_pipeline] MASTER=${MASTER_ADDR}:${MASTER_PORT} USE_DEEPSPEED=${USE_DEEPSPEED}"
  [[ -n "${DATASETS:-}" ]] && echo "[h100_resume_pipeline] DATASETS=${DATASETS}"
  [[ -n "${MAX_STEPS:-}" ]] && echo "[h100_resume_pipeline] MAX_STEPS=${MAX_STEPS}"
  [[ -n "${OUTPUT_DIR:-}" ]] && echo "[h100_resume_pipeline] OUTPUT_DIR=${OUTPUT_DIR}"

  h100_run_torchrun "$@"
}

"${ROOT}/scripts/fetch_tokenizer.sh"

run_stage "understanding_warmup" "${UW_CONFIG}" "${WORLD_CKPT}" "$@"
run_stage "generation_pt" "${GEN_PT_CONFIG}" "outputs/pretrain/uw_h800" "$@"
run_stage "unified_mt" "${MT_CONFIG}" "outputs/pretrain/gen_pt_h800" "$@"
run_stage "unified_sft" "${SFT_CONFIG}" "outputs/pretrain/mt_h800" "$@"

echo "[h100_resume_pipeline] done: outputs/pretrain/sft_h800"
