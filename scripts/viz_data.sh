#!/usr/bin/env bash
# Training data visualization (terminal). Requires: pip install -e ".[viz]"
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/viz_data.sh <command> [args...]

Commands (forwarded to `tower viz`):
  list-stages
  list-datasets [--stage STAGE]
  metrics --stage STAGE [--datasets k1,k2] [--max-samples N]
  preview --stage STAGE [-n 4] [--seed 42]
  compare [--selections-yaml exports/viz/stage_selections.yml]
  curves [--runs run1,run2] [--metric loss|grad_norm|lr]
  export [--output exports/viz/stage_selections.yml]

Headless: prints tables to stdout, saves PNGs to exports/viz/.

Examples:
  ./scripts/viz_data.sh list-stages
  ./scripts/viz_data.sh metrics --stage understanding_warmup
  ./scripts/viz_data.sh preview --stage unified_mt -n 6
  ./scripts/viz_data.sh compare
  ./scripts/viz_data.sh curves --metric loss
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

exec tower viz "$@"
