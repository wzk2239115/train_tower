#!/usr/bin/env bash
# End-to-end smoke test for structgen (complex surface / internal-topology
# generation) on the dev machine. Uses the offline proxy backbone + a tiny
# voxel decoder so it runs on a single GPU in ~1 min.
#
# On the compute box, swap --backbone stepfun --pretrained-path <Step-3.7-Flash>
# to use the real pretrained multimodal backbone.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m structgen.cli train \
    --smoke --smoke-steps 30 \
    --backbone proxy \
    --batch 2 --res 32 --base-ch 32 --mults 1,2,4 --blocks 1 \
    --num-samples 96 --surface-samples 512 \
    --image-size 112 --log-every 5 --save-every 30 \
    --out-dir outputs/structgen

python -m structgen.cli generate \
    --ckpt outputs/structgen/decoder_step30.pt \
    --res 32 --base-ch 32 --mults 1,2,4 --blocks 1 --image-size 112 \
    --prompt "structural cylinder part with internal gyroid infill" \
    --out outputs/structgen/gen_gyroid.stl --sample-steps 30

echo
echo "smoke OK — STL written to outputs/structgen/gen_gyroid.stl"
