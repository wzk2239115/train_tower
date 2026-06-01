#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Super-Omni Smoke Test — 30min end-to-end pipeline validation
# ============================================================
# Usage:
#   bash scripts/launch_smoke_super_omni.sh          # auto-detect GPUs
#   bash scripts/launch_smoke_super_omni.sh 8         # 8× H100/H800
#   bash scripts/launch_smoke_super_omni.sh 1         # single GPU debug
# ============================================================

GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  Super-Omni Smoke Test"
echo "  GPUs: $GPUS"
echo "  $(date)"
echo "============================================"

# --- H100/H800 distributed env ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    for nic in eth0 ens5 enp0s8 bond0; do
        if ip -o link show "$nic" &>/dev/null 2>&1; then
            export NCCL_SOCKET_IFNAME="$nic"
            export GLOO_SOCKET_IFNAME="$nic"
            break
        fi
    done
fi

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

# --- Step 2: Verify data ---
echo ""
echo "[Step 2/3] Verifying data ..."
python3 -c "
import json
m = json.load(open('data/processed/manifest.json'))
for k, v in sorted(m.items()):
    stages = list(v.get('stages', {}).keys())
    samples = v.get('samples', 0)
    print(f'  {k:20s}  stages={stages}  samples={samples:,}')
print(f'\n  Total datasets: {len(m)}')
"

# --- Step 3: Launch ---
echo ""
echo "[Step 3/3] Launching smoke test ..."
echo "  Experiment:  smoke_super_omni"
echo "  Size preset: tiny_smoke (8L @ 256d)"
echo "  Steps:       300 (5 stages × 60 steps)"
echo "  Output:      outputs/smoke/super_omni/"
echo ""

if [ "$GPUS" = "1" ]; then
    export TOWER_NO_DEEPSPEED=1
    echo "  Mode: single GPU (no DeepSpeed)"
    python3 -m tower.cli train --experiment smoke_super_omni
else
    echo "  Mode: $GPUS GPUs + DeepSpeed ZeRO-2"
    torchrun \
        --standalone \
        --nproc_per_node="$GPUS" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m tower.cli train \
        --experiment smoke_super_omni
fi

echo ""
echo "============================================"
echo "  Smoke test complete! $(date)"
echo "  Output: outputs/smoke/super_omni/"
echo ""
echo "  Check results:"
echo "    cat outputs/smoke/super_omni/trainer_state.json | python3 -m json.tool | tail -20"
echo "============================================"
