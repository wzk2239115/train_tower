# Third-party notices

Upstream training dependencies are **in-tree** under `tower/` (not runtime-vendored copies).

| Component | In-repo path | Upstream | License |
|-----------|--------------|----------|---------|
| NEO data pipeline | `tower/neo/` | [EvolvingLMMs-Lab/NEO](https://github.com/EvolvingLMMs-Lab/NEO) (`VLMTrainKit/neo/data`, `neo/train/argument.py`) | see upstream |
| SenseNova NEO-Unify MoT | `tower/models/neo_unify/` | [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) | Apache-2.0 (full text: [`scripts/SENSENOVA-U1.LICENSE`](scripts/SENSENOVA-U1.LICENSE)) |

Refresh from upstream: `./scripts/vendor_third_party.sh` (writes pinned commits to `scripts/VENDOR_REVISIONS`).

train_tower-specific patches (block-causal SDPA, transformers compat) live in `tower/unify/attention.py` and `tower/unify/compat.py`, applied at runtime via `tower/unify/backends/sensenova.py`.
