#!/usr/bin/env bash
# One-shot source setup: pip deps + tokenizer + in-tree upstream sanity check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pip install -e ".[train,viz]"

if [[ ! -f tower/neo/data/__init__.py ]]; then
  echo "Missing tower/neo data pipeline. Run: ./scripts/vendor_third_party.sh" >&2
  exit 1
fi
if [[ ! -f tower/models/neo_unify/__init__.py ]]; then
  echo "Missing tower/models/neo_unify. Run: ./scripts/vendor_third_party.sh" >&2
  exit 1
fi

"${ROOT}/scripts/fetch_tokenizer.sh"

python -c "
from tower.unify.backends import import_neo_data, import_neo_chat_config
import_neo_data()
import_neo_chat_config()
print('tower/neo + tower/models/neo_unify OK')
"

echo "Setup complete. Try: MAX_STEPS=10 DATASETS=blip3o_short_pt ./scripts/train_smoke.sh"
