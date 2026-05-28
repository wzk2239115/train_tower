#!/usr/bin/env bash
# Sync vendored training dependencies into third_party/ for offline / source builds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NEO_REPO="${NEO_REPO:-https://github.com/EvolvingLMMs-Lab/NEO.git}"
SNU_REPO="${SNU_REPO:-https://github.com/OpenSenseNova/SenseNova-U1.git}"
NEO_REF="${NEO_REF:-main}"
SNU_REF="${SNU_REF:-main}"
WORK="${ROOT}/.vendor_cache"

usage() {
  cat <<'EOF'
Usage: ./scripts/vendor_third_party.sh [--from-local PATH_NEO PATH_SNU]

Default: shallow-clone upstream repos into .vendor_cache/, then copy:
  NEO/VLMTrainKit/              -> third_party/NEO/
  SenseNova-U1/src/ + metadata  -> third_party/SenseNova-U1/

Env overrides:
  NEO_REPO, SNU_REPO, NEO_REF, SNU_REF

Local sync example:
  ./scripts/vendor_third_party.sh --from-local /path/to/NEO /path/to/SenseNova-U1
EOF
}

copy_from_paths() {
  local neo_root="$1"
  local snu_root="$2"
  mkdir -p third_party/NEO third_party/SenseNova-U1
  rsync -a --delete "${neo_root}/VLMTrainKit/" third_party/NEO/
  rsync -a --delete "${snu_root}/src/" third_party/SenseNova-U1/src/
  cp "${snu_root}/pyproject.toml" "${snu_root}/LICENSE" third_party/SenseNova-U1/
  write_revisions_from_paths "${neo_root}" "${snu_root}"
}

clone_and_copy() {
  mkdir -p "$WORK"
  if [[ ! -d "${WORK}/NEO/.git" ]]; then
    git clone --depth 1 --branch "$NEO_REF" "$NEO_REPO" "${WORK}/NEO"
  else
    git -C "${WORK}/NEO" fetch --depth 1 origin "$NEO_REF"
    git -C "${WORK}/NEO" checkout "$NEO_REF"
    git -C "${WORK}/NEO" pull --ff-only origin "$NEO_REF" || true
  fi
  if [[ ! -d "${WORK}/SenseNova-U1/.git" ]]; then
    git clone --depth 1 --branch "$SNU_REF" "$SNU_REPO" "${WORK}/SenseNova-U1"
  else
    git -C "${WORK}/SenseNova-U1" fetch --depth 1 origin "$SNU_REF"
    git -C "${WORK}/SenseNova-U1" checkout "$SNU_REF"
    git -C "${WORK}/SenseNova-U1" pull --ff-only origin "$SNU_REF" || true
  fi
  copy_from_paths "${WORK}/NEO" "${WORK}/SenseNova-U1"
}

write_revisions_from_paths() {
  local neo_root="$1"
  local snu_root="$2"
  cat > third_party/VENDOR_REVISIONS <<EOF
# Vendored upstream revisions (updated by scripts/vendor_third_party.sh)
neo_repo: ${NEO_REPO}
neo_path: VLMTrainKit/
neo_commit: $(git -C "$neo_root" rev-parse HEAD 2>/dev/null || echo unknown)
sensenova_repo: ${SNU_REPO}
sensenova_path: src/
sensenova_commit: $(git -C "$snu_root" rev-parse HEAD 2>/dev/null || echo unknown)
updated_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--from-local" ]]; then
  [[ $# -eq 3 ]] || { usage; exit 1; }
  copy_from_paths "$2" "$3"
else
  clone_and_copy
fi

echo "Vendored:"
du -sh third_party/NEO third_party/SenseNova-U1
echo "Revisions: third_party/VENDOR_REVISIONS"
