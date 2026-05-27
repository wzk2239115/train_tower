# train_tower

Data conversion and unified multimodal training pipeline for **SenseNova-U1 MoT** style native models.

## Setup

```bash
cd /home/wzk/projects/train_tower
git submodule update --init --recursive   # optional
pip install -e ".[train]"
./scripts/fetch_tokenizer.sh              # Qwen 词表 only, no weights
python scripts/estimate_params.py         # verify ~500M param count
```

Third-party layout:

```
third_party/
├── NEO/            -> NEO/VLMTrainKit (data pipeline)
└── SenseNova-U1/   -> SenseNova-U1 (neo_unify MoT model)
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
| Uni MT | `scripts/train_mt.sh` | CE + FM |
| Uni SFT | `scripts/train_sft.sh` | CE + FM |

Checkpoints: `outputs/pretrain/{uw,gen_pt,mt,sft}`

### Smoke test

```bash
MAX_STEPS=10 DATASETS=blip3o_short_pt ./scripts/train_smoke.sh
```

### Full pipeline

```bash
./scripts/train_pretrain.sh
# or stage-by-stage:
./scripts/train_uw.sh && ./scripts/train_gen_pt.sh && ./scripts/train_mt.sh && ./scripts/train_sft.sh
```

**Single GPU (default):** scripts use `torchrun` and auto-set `TOWER_NO_DEEPSPEED=1` to avoid `mpi4py` / NVML issues. Force DeepSpeed with `USE_DEEPSPEED=1` (multi-GPU recommended).

**Multi GPU:** `NUM_GPUS=8 ./scripts/train_pretrain.sh` — DeepSpeed ZeRO-2 from yaml is enabled automatically.

## Architecture

- **Model**: SenseNova `NEOChatModel` (MoT) via `tower/unify/build.py` + `SenseNovaTrainModel`
- **Data**: NEO `LazySupervisedDataset` + packed collator with `image_gen_indicators`
- **Freeze schedule**: `tower/train/freeze.py` (UW → und, Gen PT → gen, MT/SFT → all)
- **Loss**: weighted CE + rectified flow velocity MSE

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
│   └── unify/                      # build, SenseNovaTrainModel
├── note/train.yml
└── scripts/
```
