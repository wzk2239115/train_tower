#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

python -m tower.cli train --config /tmp/train_tower_smoke.yaml "$@"
