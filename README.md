# train_tower

Data conversion and unified multimodal training pipeline for **SenseNova-U1 MoT** style native models.

## Setup

```bash
cd train_tower
pip install -e ".[train]"
./scripts/fetch_tokenizer.sh              # Qwen 词表 only, no weights
python scripts/estimate_params.py         # verify ~500M param count
```

Training code (**NEO** + **SenseNova-U1**) is **vendored in-repo** under `third_party/` (~1.5 MB source). Copy the repo to an offline server and train — no git submodule or symlink setup required.

To refresh vendored upstream source:

```bash
./scripts/vendor_third_party.sh
# or from local clones:
./scripts/vendor_third_party.sh --from-local /path/to/NEO /path/to/SenseNova-U1
```

Third-party layout:

```
third_party/
├── NEO/              # VLMTrainKit (neo package)
├── SenseNova-U1/     # src/sensenova_u1 (MoT model)
└── VENDOR_REVISIONS  # pinned upstream commits
```

## Data conversion

```bash
python -m tower.cli convert --dataset blip3o_short --limit 100
python -m tower.cli convert --stage sft
python -m tower.cli convert --all
```

Output: `data/processed/{pt,mt,sft}/*.jsonl` + `data/processed/manifest.json`

## 0→1 完全从零预训练（SenseNova MoT · ~500M）

**不下载任何预训练权重**（无 Qwen3-Base / SenseNova checkpoint）。仅下载 Qwen 词表文件；模型从 [`configs/model/sensenova_500m_mot/config.json`](configs/model/sensenova_500m_mot/config.json) 随机初始化。

结构对标 SenseNova-U1-8B-MoT（MoT 双路径 + `fm_modules` + `image_gen_indicators`），规模缩至 ~500M（14 层 × hidden 768）。

```
random init (~500M MoT)
        │
        ▼
  UW (CE, train und path)       ← blip3o PT caption
        │
        ▼
  Gen PT (FM, T2I flip)         ← blip3o PT → text-to-image
        │
        ▼
  Uni MT (CE+FM)                ← llava/sharegpt4v/refcoco/textcaps MT
        │
        ▼
  Uni SFT (CE+FM)               ← docvqa/chartqa + instruction SFT
```

| Stage | Script | Loss |
|-------|--------|------|
| UW | `scripts/train_uw.sh` | CE |
| Gen PT | `scripts/train_gen_pt.sh` | FM |
| Uni MT | `scripts/train_mt.sh` | **Tower 四探针**（见 tower.yml） |
| Uni SFT | `scripts/train_sft.sh` | **Tower 四探针**（见 tower.yml） |

Checkpoints: `outputs/pretrain/{uw,gen_pt,mt,sft}`

### 真·一次训练（continuous run）

用单个 job 跑完整 curriculum（`world_pt -> understanding_warmup -> generation_pt -> unified_mt -> unified_sft`），当前版本只切换 `note/tower.yml` 的 tower loss stage，不切数据集和 freeze 策略：

```bash
chmod +x scripts/train_continuous.sh
./scripts/train_continuous.sh
# 等价：
# torchrun ... -m tower.cli train --config configs/train/continuous.yaml
```

产物会额外导出到 `outputs/pretrain/continuous/checkpoint/`：

```text
checkpoint/
├── backbone.pt
├── world_model.pt
├── semantic_model.pt
├── language_model.pt
└── generator.pt
```

### Flow-JEPA Tower (multi-exit)

Stacked ELF + JEPA at layers 7 / 15 / 21 / 25 (`note/tower.yml`). Enable with `use_flow_tower: true`.

```bash
chmod +x scripts/train_tower_world.sh
./scripts/train_tower_world.sh   # Stage 0: world_pt (JEPA + semantic ELF)
```

See [`idea.md`](idea.md) for the full distillation-tower design.

### Smoke test

```bash
MAX_STEPS=10 DATASETS=blip3o_short_pt ./scripts/train_smoke.sh
```

### Full pipeline

```bash
./scripts/train_pretrain.sh
# or stage-by-stage:
./scripts/train_uw.sh && ./scripts/train_gen_pt.sh && ./scripts/train_mt.sh && ./scripts/train_sft.sh
# or single continuous run:
./scripts/train_continuous.sh
```

**Single GPU (default):** scripts use `torchrun` and auto-set `TOWER_NO_DEEPSPEED=1` to avoid `mpi4py` / NVML issues. Force DeepSpeed with `USE_DEEPSPEED=1` (multi-GPU recommended).

**Multi GPU:** `NUM_GPUS=8 ./scripts/train_pretrain.sh` — DeepSpeed ZeRO-2 from yaml is enabled automatically.

## Data visualization (terminal)

Inspect per-stage datasets, modality coverage, Tower exit weights, and training loss curves. Headless-friendly: terminal tables + PNGs under `exports/viz/` (no GUI).

```bash
pip install -e ".[viz]"

tower viz list-stages
tower viz metrics --stage understanding_warmup
tower viz preview --stage unified_mt -n 6
tower viz compare
tower viz curves --metric loss
tower viz export   # -> exports/viz/stage_selections.yml

# or via helper script:
./scripts/viz_data.sh metrics --stage world_pt
```

Override datasets: `--datasets blip3o_short_pt,llava_pt`. Python API: `tower.viz`.

## Architecture

- **Model**: SenseNova `NEOChatModel` (MoT) via `tower/unify/build.py` + `SenseNovaTrainModel`
- **Flow-JEPA Tower** (optional): `tower/unify/flow_tower.py` — multi-exit JEPA + stacked ELF; see [`idea.md`](idea.md) and [`note/tower.yml`](note/tower.yml)
- **Data**: NEO `LazySupervisedDataset` + packed collator with `image_gen_indicators`
- **Freeze schedule**: `tower/train/freeze.py` (UW → und, Gen PT → gen, MT/SFT → all)
- **Loss**: MT/SFT 默认 **Flow-JEPA Tower** 四探针联合（`use_flow_tower: true`，权重见 `note/tower.yml`）；UW/GenPT 仍为单出口 SenseNova

## Config reference

| Field | Description |
|-------|-------------|
| `init_mode` | `scratch` or `checkpoint` |
| `weight_init` | `random` (no HF weights) for UW |
| `model_config_path` | Local arch config (`configs/model/sensenova_500m_mot`) |
| `tokenizer_name_or_path` | Local Qwen tokenizer dir (`configs/tokenizer/qwen3`) |
| `loss_weights.ce/fm` | CE and FM loss weights |
| `task_override` | Force `t2i` for generation pretrain |

## Known limitations

- **LLaVA/ShareGPT4V**: limited samples until COCO train2017 is downloaded
- **500M capacity**: structural alignment with 8B-MoT, not quality parity
- **FM training**: derived from SenseNova inference logic; may differ from internal training
- Requires **torch>=2.5** with working CUDA for GPU training

## Project layout

```
train_tower/
├── configs/
│   ├── model/sensenova_500m_mot/   # arch config (no weights)
│   ├── tokenizer/qwen3/            # vocab only
│   └── train/                      # stage yaml
├── tower/
│   ├── convert/
│   ├── train/                      # trainer, freeze, dataset
│   ├── viz/                        # CLI + plots (data stats, metrics)
│   └── unify/                      # build, SenseNovaTrainModel
├── exports/viz/                    # saved plots & stage_selections.yml
├── note/train.yml
└── scripts/
```
