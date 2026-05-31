# 任务类型分类法

## 总览

Super-Omni 支持 12 类任务，按 (输入模态, 输出模态) 分类。每类任务对应不同的 loss 组合和 tower exit 激活模式。

## 任务分类

### 理解类 (输入多模态 → 输出文本)

| Task ID | 输入 | 输出 | Loss | Tower Exits | 描述 |
|---------|------|------|------|-------------|------|
| `IT2T` | Image + Text | Text | CE | understanding_elf | 图像理解/问答 |
| `AT2T` | Audio + Text | Text | CE | understanding_elf | 音频理解/描述 |
| `VT2T` | Video + Text | Text | CE | understanding_elf | 视频理解/描述 |
| `AVT2T` | Audio + Video + Text | Text | CE | understanding_elf | 音视频联合理解 |
| `OMNI2T` | Image + Audio + Video + Text | Text | CE | understanding_elf | 全模态理解 |

### 生成类 (输入文本/多模态 → 输出特定模态)

| Task ID | 输入 | 输出 | Loss | Tower Exits | 描述 |
|---------|------|------|------|-------------|------|
| `T2I` | Text | Image | FM | generative_elf | 文本生成图像 |
| `T2A` | Text | Audio | FM | audio_elf | 文本生成音效 |
| `T2S` | Text | Audio (speech) | FM | audio_elf | 文本转语音 |
| `T2M` | Text | Audio (music) | FM | audio_elf | 文本生成音乐 |
| `T2V` | Text | Video | FM | video_elf | 文本生成视频 |
| `I2V` | Image + Text | Video | FM | video_elf | 图像生成视频 |
| `V2A` | Video + Text | Audio | FM | audio_elf | 视频生成音频 |

### 交织类 (Super-Omni 灵魂)

| Task ID | 输入 | 输出 | Loss | Tower Exits | 描述 |
|---------|------|------|------|-------------|------|
| `T2OMNI` | Text | Text + Image + Audio + Video | CE + FM | all | 文本→交织多模态 |
| `OMNI2OMNI` | Image + Audio + Video + Text | Text + Image + Audio + Video | CE + FM | all | 全模态交织 |

## Loss 路由

### CE Loss (Cross-Entropy)

- 作用于所有 text tokens (非 `<*_CONTEXT>` tokens)
- 仅通过 understanding_elf exit 反传
- 理解类任务和生成类任务的文本部分都使用 CE

### FM Loss (Flow Matching / Rectified Flow)

- 作用于所有 `<*_CONTEXT>` tokens
- 通过对应的 tower exit 反传:
  - `<IMG_CONTEXT>` → generative_elf (L25)
  - `<AUDIO_CONTEXT>` → audio_elf (L11)
  - `<VIDEO_CONTEXT>` → video_elf (L13)
- 在 understanding path 中, 这些 tokens 被 inject 的 clean embedding 替换
- 在 generation path 中, 这些 tokens 是 FM 采样点 z_t

### JEPA Loss

- 仅 world_pt 阶段
- 作用于 understanding path 的 vision embeddings
- 通过 world_elf exit (L7) 反传

## 训练阶段 → 任务映射

| Stage | 活跃任务 | 说明 |
|-------|----------|------|
| world_pt | — (无 CE/FM) | 纯 JEPA, 无对话 |
| understanding_warmup | IT2T | 图像理解 CE |
| generation_pt | T2I | 图像生成 FM |
| unified_mt | IT2T, T2I, AT2T, T2A, T2S, T2M, VT2T, T2V, I2V, V2A, AVT2T, T2OMNI, OMNI2OMNI | 全部 |
| unified_sft | 同 unified_mt (高质量子集) | 全部 |

## Meta Task 字段

在 JSONL 的 `meta.task` 中使用以下值:

```python
TASK_TYPES = {
    # 理解
    "understanding",    # IT2T / AT2T / VT2T / AVT2T / OMNI2T 的统称

    # 单模态生成
    "t2i",              # Text → Image
    "t2a",              # Text → Audio (sound effect)
    "t2s",              # Text → Speech
    "t2m",              # Text → Music
    "t2v",              # Text → Video
    "i2v",              # Image → Video
    "v2v",              # Video → Video (edit)
    "av2v",             # Audio + Video → Video

    # 交织
    "interleave",       # T2OMNI / OMNI2OMNI
}
```

## 采样策略

### unified_mt 阶段

按 pool_mix 比例自动平衡各任务类型:

```
pool_mix 中:
  image_understanding: 0.15  → IT2T 任务
  image_generation:    0.15  → T2I 任务
  audio_understanding: 0.08  → AT2T 任务
  audio_generation:    0.08  → T2A + T2S + T2M 任务 (按子池比例分配)
  video_understanding: 0.08  → VT2T 任务
  video_generation:    0.08  → T2V + I2V 任务
  av_sync:             0.08  → V2A + AVT2T + T2OMNI 任务
  interleaved_omni:    0.25  → T2OMNI + OMNI2OMNI 任务
  text:                0.05  → 纯文本 (CE only)
```

### unified_sft 阶段

同样使用 pool_mix，但启用 quality_filter，只保留 gold 级别数据。

## 任务采样与 Tower Exit Loss 权重的关系

每个 stage YAML 中的 `tower_exit_weights` 决定了不同 exit 的 loss 权重:

```yaml
tower_exit_weights:
  world_elf: 0.05       # JEPA, 几乎关闭
  semantic_elf: 0.15    # 语义 flow
  audio_elf: 0.2        # 音频 FM — 由 T2A/T2S/T2M/V2A 任务触发
  video_elf: 0.2        # 视频 FM — 由 T2V/I2V 任务触发
  understanding_elf: 1.0  # CE — 所有理解任务触发
  generative_elf: 0.3    # 图像 FM — 由 T2I 任务触发
```

当某个 exit 的 weight=0 时, 对应的任务数据仍然可以存在于 batch 中, 但该 exit 不产生 loss 梯度。这意味着:
- audio/video 数据在 world_pt/uw/gen_pt 阶段可以存在但不会影响训练
- 无需严格按 stage 过滤数据, pool_mix 已自然控制
