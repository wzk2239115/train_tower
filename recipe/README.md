# Super-Omni Data Recipe

Flow-JEPA Tower 项目的完整数据配方，覆盖 text/image/audio/video 全模态理解与交织生成。

## 目录结构

```
recipe/
├── README.md                          ← 你在这里
├── data_registry.py                   核心：读取 pool YAML → 生成 data_dict
│
├── pools/                             9 个数据池定义
│   ├── 00_text.yaml                   文本底座 (~8 datasets)
│   ├── 01_image_understanding.yaml    Image → Text (~8 datasets)
│   ├── 02_image_generation.yaml       Text → Image (~6 datasets)
│   ├── 03_audio_understanding.yaml    Audio → Text (~6 datasets)
│   ├── 04_audio_generation.yaml       Text → Audio/Speech/Music (~10 datasets)
│   ├── 05_video_understanding.yaml    Video → Text (~8 datasets)
│   ├── 06_video_generation.yaml       Text → Video (~6 datasets)
│   ├── 07_av_sync.yaml               Audio-Visual 同步 (~6 datasets)
│   └── 08_interleaved_omni.yaml       交织多模态, 合成 (~5 datasets)
│
├── stages/                            5 个训练阶段配方
│   ├── world_pt.yaml                  Stage 0: JEPA (image+text only)
│   ├── understanding_warmup.yaml      Stage 1: 理解路径 CE (image+text only)
│   ├── generation_pt.yaml             Stage 2: 生成路径 FM (image T2I)
│   ├── unified_mt.yaml                Stage 3: 全模态多任务 (9 pools)
│   └── unified_sft.yaml              Stage 4: 全模态 SFT (9 pools, gold)
│
├── scales/                            模型规模缩放
│   ├── _schema.yaml                   缩放公式文档
│   ├── 500m.yaml                      26L@768d 基线
│   └── 1b.yaml                        32L@1024d 扩展
│
├── format/                            数据格式规范
│   ├── jsonl_schema.md                JSONL 字段说明
│   ├── token_conventions.md           Special token 用法
│   └── examples/                      每种任务的 JSONL 样例
│       ├── text_only.jsonl
│       ├── image_understanding.jsonl
│       ├── t2i.jsonl
│       ├── audio_understanding.jsonl
│       ├── t2a.jsonl
│       ├── t2s.jsonl
│       ├── video_understanding.jsonl
│       ├── t2v.jsonl
│       ├── av_sync.jsonl
│       └── interleaved_omni.jsonl
│
└── synthesis/                         合成数据管线
    ├── interleaved_pipeline.md        交织数据合成流程
    ├── quality_control.md             质量过滤规则
    └── task_type_taxonomy.md          12 类任务分类法
```

## 快速开始

### 1. 查看注册的数据集

```bash
python recipe/data_registry.py
```

输出所有 pool、stage、scale，以及 64 个注册数据集的列表。

### 2. 解析某个 stage 的 datasets 字符串

```python
from recipe.data_registry import resolve_stage_datasets

# unified_mt → 完整的 datasets 字符串 (带采样率)
datasets = resolve_stage_datasets("recipe/stages/unified_mt.yaml")
# 输出: "text__openhermes%5,text__sharegpt%5,...,omni__synth_sft%25"
```

### 3. 应用 scale 缩放

```python
from recipe.data_registry import resolve_full_stage

# 解析 unified_sft + 1b scale
config = resolve_full_stage("unified_sft", scale="1b")
# config["datasets"] = 解析后的 datasets 字符串
# config["hyperparams"]["max_steps"] = 40000  (20000 × 2.0)
# config["hyperparams"]["learning_rate"] = 0.00019  (1.2e-4 × 1.58)
```

### 4. 构建 data_dict

```python
from recipe.data_registry import build_data_dict

data_dict = build_data_dict()
# data_dict["aud_und__audiocaps"] = {
#     "annotation_path": "./data/audiocaps/train.jsonl",
#     "data_path": "./data/audiocaps"
# }
```

## 数据流

```
recipe/pools/*.yaml          recipe/stages/*.yaml
        ↓                            ↓
   data_registry.py ←──── pool_mix + scale
        ↓
   build_data_dict()
        ↓                            ↓
tower/neo/data/__init__.py   configs/train/*.yaml
   data_dict = {...}            datasets: "..."
        ↓                            ↓
   LazySupervisedDataset     TrainConfig.datasets
        ↓                            ↓
   data_list(name_str)       CurriculumRuntime
        ↓
   FlattenedDataCollator
        ↓
   UnifiedCollator (audio/video mask)
        ↓
   FlowJepaTowerTrainModel.forward()
```

## 训练阶段总览

| Stage | 数据 | Loss | Tower Exits | 模态 |
|-------|------|------|-------------|------|
| world_pt | image+text | JEPA | world_elf (1.0) | image |
| understanding_warmup | image+text | CE | understanding_elf (1.0) | image+text |
| generation_pt | image+text (T2I) | FM | generative_elf (1.0) | image |
| **unified_mt** | **9 pools** | **CE + FM** | **all exits** | **all** |
| **unified_sft** | **9 pools (gold)** | **CE + FM** | **all exits** | **all** |

## 数据池混合比例 (unified_mt)

```
text:                5%   ← 保语言能力
image_understanding: 15%  ← 视觉理解
image_generation:    15%  ← T2I 生成
audio_understanding:  8%  ← 音频理解
audio_generation:     8%  ← T2A/T2S/T2M
video_understanding:  8%  ← 视频理解
video_generation:     8%  ← T2V 生成
av_sync:              8%  ← 音视频同步
interleaved_omni:    25%  ← 交织多模态（Super-Omni 核心）
```

## 模型规模缩放

| Scale | Layers | Hidden | 数据量倍率 | 步数倍率 | LR 倍率 | GPU |
|-------|--------|--------|-----------|---------|---------|-----|
| 500m | 26 | 768 | ×1.0 | ×1.0 | ×1.0 | 2×A100 |
| 1b | 32 | 1024 | ×2.5 | ×2.0 | ×1.58 | 4×A100 |

## 数据集统计

共 **64** 个注册数据集，覆盖 9 个数据池：

| Pool | 数据集数 | 总量级 (gold+silver) | 主要来源 |
|------|----------|---------------------|----------|
| text | 8 | ~6M samples | OpenHermes, ShareGPT, UltraChat |
| image_understanding | 8 | ~15M samples | LLaVA-150K, ShareGPT4V, Cambrian |
| image_generation | 6 | ~83M pairs | JourneyDB, LAION-Aesthetic, COYO |
| audio_understanding | 6 | ~2.6M samples | AudioCaps, WavCaps, AudioSet |
| audio_generation | 10 | ~120K hours | Emilia, LibriTTS-R, MusicCaps |
| video_understanding | 8 | ~97M clips | WebVid, Panda-70M, InternVid |
| video_generation | 6 | ~89M pairs | OpenVid, Panda-70M, MiraData |
| av_sync | 6 | ~3.5M clips | VGGSound, AudioSet, AVSpeech |
| interleaved_omni | 5 | ~420K samples (合成) | OmniCorpus, Synth-Omni |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_ROOT` | `./data` | 所有数据集的根目录 |

## 与现有代码集成 (未来)

当准备好集成时，需要修改以下文件：

1. **`tower/neo/data/__init__.py`**: 改为 `from recipe.data_registry import build_data_dict`
2. **`tower/train/config.py`**: `load_train_config()` 支持 recipe/stages/*.yaml
3. **`note/train.yml`**: stages 引用指向 recipe/stages/
