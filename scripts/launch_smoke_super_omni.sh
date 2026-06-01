#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Super-Omni Smoke Test — 30min end-to-end pipeline validation
# ============================================================
# Usage:
#   bash scripts/launch_smoke_super_omni.sh [GPU_COUNT]
#
# Prerequisites:
#   1. Process audio data:  python scripts/process_audio.py
#   2. Verify manifest:     cat data/processed/manifest.json | python3 -m json.tool
# ============================================================

GPUS="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  Super-Omni Smoke Test"
echo "  GPUs: $GPUS"
echo "  $(date)"
echo "============================================"

# --- Step 1: Process audio data (if not already done) ---
if ! python3 -c "import json; m=json.load(open('data/processed/manifest.json')); assert 'wavcaps' in m" 2>/dev/null; then
    echo ""
    echo "[Step 1/3] Processing audio data ..."
    python3 scripts/process_audio.py \
        --raw-dir data/raw \
        --processed-dir data/processed \
        --data-root data
else
    echo "[Step 1/3] Audio data already processed, skipping"
fi

echo ""
echo "[Step 2/3] Verifying data ..."
python3 -c "
import json
m = json.load(open('data/processed/manifest.json'))
for k, v in m.items():
    stages = list(v.get('stages', {}).keys())
    samples = v.get('samples', 0)
    print(f'  {k:20s}  stages={stages}  samples={samples:,}')
print(f'\n  Total datasets: {len(m)}')
"

echo ""
echo "[Step 3/3] Launching smoke test ..."
echo ""

if [ "$GPUS" = "1" ]; then
    export TOWER_NO_DEEPSPEED=1
    echo "  Single GPU mode (no DeepSpeed)"
    python3 -m tower.cli train --experiment smoke_super_omni
else
    echo "  Multi-GPU mode ($GPUS GPUs)"
    torchrun \
        --standalone \
        --nproc_per_node="$GPUS" \
        -m tower.cli train \
        --experiment smoke_super_omni
fi

echo ""
echo "============================================"
echo "  Smoke test complete! $(date)"
echo "  Output: outputs/smoke/super_omni/"
echo "============================================"
