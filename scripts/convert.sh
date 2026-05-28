#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIMIT="${LIMIT:-}"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

case "${1:-all}" in
  all)
    python -m tower.cli convert --all "${EXTRA[@]}" "${@:2}"
    ;;
  pt|mt|sft)
    python -m tower.cli convert --stage "$1" "${EXTRA[@]}" "${@:2}"
    ;;
  dry-run)
    python -m tower.cli convert --all --dry-run "${EXTRA[@]}" "${@:2}"
    ;;
  refresh-manifest)
    python -m tower.cli convert --refresh-manifest "${@:2}"
    ;;
  *)
    python -m tower.cli convert --dataset "$1" "${EXTRA[@]}" "${@:2}"
    ;;
esac
