# train_tower

Data conversion and unified multimodal training pipeline for **SenseNova-U1 MoT** style native models.

## Setup

```bash
cd train_tower
pip install -e ".[train]"
./scripts/fetch_tokenizer.sh              # Qwen 词表 only, no weights
python scripts/estimate_params.py         # verify ~500M param count
```

Training code (**NEO** data + **SenseNova-U1** MoT model) lives **in-tree** under `tower/neo/` and `tower/models/neo_unify/`. Copy the repo to an offline server and train — no submodule or `PYTHONPATH` setup required.

To refresh from upstream:

```bash
./scripts/vendor_third_party.sh
# or from local clones:
./scripts/vendor_third_party.sh --from-local /path/to/NEO /path/to/SenseNova-U1
```

Layout and attribution: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Pinned upstream commits: [`scripts/VENDOR_REVISIONS`](scripts/VENDOR_REVISIONS).

## 开发机与算力平台（当前协作方式）

**现状**：日常改代码在**开发机**（本仓库工作目录，即当前开发环境）完成；改完后由人工 **commit & push** 到 Git 远程；算力平台（如 `h800fast` 上的 `wangzekai/train_tower`）在跑训练前执行 **`git pull`** 拉取同一分支。训练以算力平台上的 **`git rev-parse HEAD`** 为准（与远程不一致则说明未 pull 到最新修复）。

```bash
# 开发机
git add -A && git commit -m "..." && git push

# 算力平台
cd /home/jovyan/h800fast/wangzekai/train_tower   # 按实际路径
git pull
git rev-parse HEAD
```

**跑 H800 续训前建议在算力平台确认**：

```bash
python scripts/h800_preflight_check.py
python scripts/h800_preflight_check.py --run-benchmark   # optional eager vs SDPA VRAM check
```

1. 日志出现 `Block-causal attention: native fused SDPA`（`tower/unify/compat.py` 在训练启动时 monkey-patch；没有则说明未 `git pull` 到最新 commit）。
2. `grep sdpa_block_attention_forward tower/unify/attention.py` 有命中（train_tower 侧 SDPA 实现；vendor `modeling_qwen3.py` 保持 upstream 干净）。
3. `max` profile 时出现 `[vram_tune]`，且 `per_device_train_batch_size` / `max_pixels` 被压到与 `stable` 同量级（pack 条数 + 像素，不是单纯加大 micro-batch）。
4. 算力平台若未 pull 到含修复的 commit：`max` 仍可能 OOM 在 `eager_attention_forward` → `softmax(attn_weights)`（旧版 eager 路径）。

可选：上平台前在开发机或算力平台跑 `python scripts/benchmark_block_attn.py --device cuda --seq-len 8192 --heads 12` 做 eager vs SDPA 数值/显存对拍。

可选环境变量（算力平台 export 或写进启动脚本）：

| 变量 | 含义 |
|------|------|
| `TOWER_H800_VRAM_TUNE` | `1` 启用显存护栏（`H800_PROFILE=max|extreme` 时 pipeline 默认开） |
| `TOWER_TARGET_GLOBAL_BATCH` | `max`/`extreme` 默认 `160`，用 grad_accum 凑全局 batch |
| `TOWER_VRAM_MAX_SCORE` | `extreme` 默认 `3.25`（SDPA + 无 grad_ckpt）；`max` 默认 `0.95` |
| `TOWER_DISABLE_SDPA_BLOCK_ATTN` | `1` 回退旧 eager（仅调试，易 OOM） |
| `TOWER_DATALOADER_PREFLIGHT_STEPS` | 预取 batch 数；排查卡死可设 `0` |
| `TOWER_PACKED_BATCH_LOG` | `1`（默认）packed 序列监控；`0` 关闭 |
| `TOWER_PACKED_BATCH_LOG_EVERY` | 前 20 step 每步；之后每 N step（默认 `50`；`extreme` pipeline 默认 50） |
| `TOWER_STEP_SUMMARY_EVERY` | 简洁 step 摘要间隔（默认 `100`；异常 vram/loss 仍立即打） |

## H800 显存与 attention（2025-05 现状）

**为何 `stable` 能跑、`max` 曾 OOM**：UW / Gen PT 使用 `create_block_causal_mask`；SenseNova 原实现对带 mask 的路径走 **eager attention**，在 `softmax` 前物化完整 `[B, H, L, L]` 权重（`L≈8192` 时单层约 3GiB）。数据侧 `data_flatten` 下 yaml 里的 `per_device_train_batch_size` 表示 **pack 进同一条序列的样本条数**，序列越长显存按约 **L²** 增长；**`gradient_accumulation_steps` 不降低单步峰值显存**。

**本质修复**（`tower/unify/attention.py`，由 `compat.py` 在运行时 patch 到 neo_unify）：带 block mask 的 attention 走 **fused SDPA**（禁止 MATH 后端），`create_block_causal_mask` 产出 bf16 掩码，避免分配 L×L 分数矩阵。vendor `modeling_qwen3.py` 保持 upstream 干净。仅调 yaml batch/pixels 无法根治该 OOM。

**辅助**（`tower/train/vram_tune.py`）：`H800_PROFILE=max` 时将 pack 条数 / `max_pixels` / `max_seq_length` 限制在 `stable` 锚点内，用 **grad_accum** 提高全局 batch（默认目标 160 = 8×8×3）。

## Data conversion

```bash
python -m tower.cli convert --dataset blip3o_short --limit 100
python -m tower.cli convert --stage sft
python -m tower.cli convert --all
```

**blip3o fast convert** (default: bulk `tar -xf` + scan for jsonl; no per-image PIL re-encode):

```bash
# Recommended for blip3o_long (parallel tar extract + jsonl)
WORKERS=16 ./scripts/convert.sh blip3o_long

# Step-by-step (resume-friendly)
EXTRACT_ONLY=1 WORKERS=16 ./scripts/convert.sh blip3o_long
JSONL_ONLY=1  WORKERS=16 ./scripts/convert.sh blip3o_long

# Old slow path (PIL re-encode every image)
LEGACY_CONVERT=1 WORKERS=8 ./scripts/convert.sh blip3o_long
```

Extracted layout: `data/images/blip3o_long/<tar_stem>/{image.jpg, image.txt, ...}`.

**Parallel convert** (multiple datasets at once):

```bash
JOBS=2 WORKERS=16 ./scripts/convert.sh all
python -m tower.cli convert --all --jobs 2 --workers 16
```

`--jobs` / `-j`: parallel datasets. `--workers` / `-w`: parallel tar extract (blip3o) or jsonl scan. `--limit` / `--legacy-convert` use the old slow PIL path.

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

### Resume full pipeline after `world_pt_h800`

If Stage 0 (`world_pt_h800`) already finished on H100/H800 and you want to continue all remaining stages in order:

```bash
chmod +x scripts/h100_resume_pipeline.sh
./scripts/h100_resume_pipeline.sh
```

Default chain:

```text
outputs/pretrain/world_pt_h800
  -> outputs/pretrain/uw_h800
  -> outputs/pretrain/gen_pt_h800
  -> outputs/pretrain/mt_h800
  -> outputs/pretrain/sft_h800
```

Configs used by the script:

- `configs/train/understanding_warmup_h800_resume.yaml`
- `configs/train/generation_pt_h800_resume.yaml`
- `configs/train/unified_mt_h800_resume.yaml`
- `configs/train/unified_sft_h800_resume.yaml`

**H800 profiles** (`H800_PROFILE=…`，见上一节显存说明；开发机 push 后算力平台需 `git pull`)：

| Profile | When to use | UW global batch | Notes |
|---------|-------------|-----------------|-------|
| `stable` (default) | 默认稳妥 | 64 (8×8) | grad checkpointing on |
| `turbo_safe` | 略提速 | 72 (8×9) | |
| `turbo` | 更快，注意显存 | 80 (8×10) | UW/Gen 关 grad ckpt，慎用 |
| `max` | 80GB×8 提高全局 batch | **160 (8×8×3)** | 需 SDPA patch + `vram_tune`；YAML 里可写更大 pack，运行时会钳制 |
| `extreme` | SDPA 已确认后的最高吞吐 | **160 (8×10×2)** | grad ckpt off (UW/Gen)、6M px、16 workers；轻量 vram_tune |

```bash
# 上平台前预检（git / SDPA / GPU）:
python scripts/h800_preflight_check.py
python scripts/h800_preflight_check.py --run-benchmark

# Stage 0 完成后（开发机 push → 算力平台 git pull 再执行）:
chmod +x scripts/h100_resume_pipeline.sh
H800_PROFILE=extreme ./scripts/h100_resume_pipeline.sh

# 或 max（更保守的全局 batch 策略）:
H800_PROFILE=max ./scripts/h100_resume_pipeline.sh

# 短跑验证:
TOWER_DATALOADER_PREFLIGHT_STEPS=0 H800_PROFILE=extreme MAX_STEPS=30 ./scripts/h100_resume_pipeline.sh
```

Optional overrides:

- `WORLD_CKPT` — Stage 0 checkpoint 路径。
- `UW_CONFIG` / `GEN_PT_CONFIG` / `MT_CONFIG` / `SFT_CONFIG` — 单阶段 yaml。
- `TOWER_DATALOADER_NUM_WORKERS` — stable 默认 8，max/turbo 默认 12，extreme 默认 16。
- `DATASETS` / `MAX_STEPS` / `OUTPUT_DIR` — 传给 `tower.cli train`。
- `TOWER_H800_VRAM_TUNE=0` — 关闭自动钳制（不推荐，除非已确认 SDPA patch 生效且要手调 yaml）。

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
- **Data**: NEO `LazySupervisedDataset` via `tower/unify/backends/neo.py` + packed collator with `image_gen_indicators`
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
- **Block-causal + SDPA patch**: 实现位于 `tower/unify/attention.py`，由 `tower/unify/compat.py` 在 `apply_sensenova_transformers_compat()` 时 patch 到 neo_unify；算力平台须 `git pull` 到含该改动的 commit 后才会生效

## Project layout

```
train_tower/
├── configs/
│   ├── model/sensenova_500m_mot/   # arch config (no weights)
│   ├── tokenizer/qwen3/            # vocab only
│   └── train/                      # stage yaml
├── tower/
│   ├── convert/
│   ├── train/                      # trainer, vram_tune, freeze, dataset
│   ├── viz/                        # CLI + plots (data stats, metrics)
│   └── unify/                      # build, attention, compat, backends/{neo,sensenova}
├── exports/viz/                    # saved plots & stage_selections.yml
├── note/train.yml
└── scripts/
```
