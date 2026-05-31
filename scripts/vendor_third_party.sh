#!/usr/bin/env bash
# Sync upstream NEO / SenseNova-U1 source into tower/ (in-tree, not third_party/).
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
  NEO/VLMTrainKit/neo/data + neo/train/argument.py  -> tower/neo/
  SenseNova-U1/src/sensenova_u1/models/neo_unify/   -> tower/models/neo_unify/

Env overrides:
  NEO_REPO, SNU_REPO, NEO_REF, SNU_REF

Local sync example:
  ./scripts/vendor_third_party.sh --from-local /path/to/NEO /path/to/SenseNova-U1
EOF
}

copy_from_paths() {
  local neo_root="$1"
  local snu_root="$2"
  mkdir -p tower/neo/train tower/models/neo_unify
  # train_tower: neo/data + neo/train/argument only (no neo/model, no upstream train entrypoint).
  rsync -a --delete \
    --exclude 'neo/model/' \
    --exclude 'neo/train/train.py' \
    --exclude 'scripts/' \
    --exclude 'requirements.txt' \
    --exclude 'README.md' \
    "${neo_root}/VLMTrainKit/neo/data/" tower/neo/data/
  rsync -a \
    "${neo_root}/VLMTrainKit/neo/train/argument.py" tower/neo/train/
  # SenseNova: neo_unify models only.
  rsync -a --delete \
    "${snu_root}/src/sensenova_u1/models/neo_unify/" tower/models/neo_unify/
  mkdir -p scripts
  cp "${snu_root}/LICENSE" scripts/SENSENOVA-U1.LICENSE
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
  cat > scripts/VENDOR_REVISIONS <<EOF
# Upstream revisions synced into tower/ (updated by scripts/vendor_third_party.sh)
neo_repo: ${NEO_REPO}
neo_path: VLMTrainKit/neo/data + neo/train/argument.py -> tower/neo/
neo_commit: $(git -C "$neo_root" rev-parse HEAD 2>/dev/null || echo unknown)
sensenova_repo: ${SNU_REPO}
sensenova_path: src/sensenova_u1/models/neo_unify/ -> tower/models/neo_unify/
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

echo "Synced into tower/:"
du -sh tower/neo tower/models/neo_unify
echo "Revisions: scripts/VENDOR_REVISIONS"
