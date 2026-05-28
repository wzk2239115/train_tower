#!/usr/bin/env bash
# One-shot source setup: pip deps + tokenizer + third_party sanity check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pip install -e ".[train,viz]"

if [[ ! -f third_party/NEO/neo/data/__init__.py ]]; then
  echo "Missing vendored NEO source. Run: ./scripts/vendor_third_party.sh" >&2
  exit 1
fi
if [[ ! -f third_party/SenseNova-U1/src/sensenova_u1/__init__.py ]]; then
  echo "Missing vendored SenseNova-U1 source. Run: ./scripts/vendor_third_party.sh" >&2
  exit 1
fi

"${ROOT}/scripts/fetch_tokenizer.sh"

python -c "
from tower.paths import ensure_train_paths
ensure_train_paths()
import neo.data
print('third_party OK: neo + sensenova source present')
"

echo "Setup complete. Try: MAX_STEPS=10 DATASETS=blip3o_short_pt ./scripts/train_smoke.sh"
