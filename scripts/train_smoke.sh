#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Smoke uses plain `python` (not torchrun). Limit to one GPU — DataParallel breaks
# SenseNovaTrainModel's custom state_dict / device handling on multi-GPU nodes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOWER_NO_DEEPSPEED=1

"${ROOT}/scripts/fetch_tokenizer.sh"

python << PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/train/understanding_warmup.yaml"))
cfg["max_steps"] = int("${MAX_STEPS:-10}")
cfg["output_dir"] = "${OUTPUT_DIR:-outputs/smoke_uw}"
cfg["datasets"] = "${DATASETS:-blip3o_short_pt}"
cfg["save_steps"] = max(5, cfg["max_steps"])
cfg["deepspeed"] = None
Path("outputs").mkdir(exist_ok=True)
yaml.dump(cfg, open("/tmp/train_tower_smoke.yaml", "w"))
PY

# shellcheck source=train_env.sh
source "${ROOT}/scripts/train_env.sh"
train_env_setup
train_env_print_training_summary config /tmp/train_tower_smoke.yaml

python -m tower.cli train --config /tmp/train_tower_smoke.yaml "$@"
