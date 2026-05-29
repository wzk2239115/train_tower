#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIMIT="${LIMIT:-}"
JOBS="${JOBS:-1}"
WORKERS="${WORKERS:-1}"
EXTRA=()
if [[ -n "$LIMIT" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi
if [[ "$JOBS" != "1" ]]; then
  EXTRA+=(--jobs "$JOBS")
fi
if [[ "$WORKERS" != "1" ]]; then
  EXTRA+=(--workers "$WORKERS")
fi
if [[ "${EXTRACT_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--extract-only)
fi
if [[ "${JSONL_ONLY:-0}" == "1" ]]; then
  EXTRA+=(--jsonl-only)
fi
if [[ "${LEGACY_CONVERT:-0}" == "1" ]]; then
  EXTRA+=(--legacy-convert)
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
  EXTRA+=(--verbose)
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
