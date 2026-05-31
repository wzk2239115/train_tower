# Super-Omni JSONL 数据格式规范

## 概述

所有训练数据采用统一的 JSONL 格式，每行一个 JSON 对象，符合 `tower.schema.UnifiedSample` 的定义。

## 基础 Schema

```json
{
  "id": "unique_sample_id",
  "image": "path/to/image.jpg",
  "audio": "path/to/audio.wav",
  "video": "path/to/video.mp4",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ],
  "meta": {
    "task": "understanding",
    "source": "dataset_name"
  }
}
```

## 字段说明

### 必需字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `conversations` | `list[dict]` | 对话轮次, 每轮 `{"from": "human"/"gpt", "value": "..."}` |
| `id` | `string` | 唯一样本 ID (可选, 用于追踪) |

### 模态字段 (按需出现)

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | `string` 或 `list[string]` | 图片路径, 支持单张或多张 |
| `images` | `list[string]` | 多图替代写法 |
| `audio` | `string` 或 `list[string]` | 音频文件路径 (.wav/.flac/.mp3) |
| `audios` | `list[string]` | 多音频替代写法 |
| `video` | `string` 或 `list[string]` | 视频文件路径 (.mp4/.pt/.npy) |
| `videos` | `list[string]` | 多视频替代写法 |

### 预计算特征字段 (可选, 优先于文件路径)

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio_values` | `list[list[float]]` | 预计算的音频 patch features [N_patch, patch_dim] |
| `audio_token_mask` | `list[bool]` | 预计算的音频 token mask |
| `video_values` | `list[list[float]]` | 预计算的视频 features [N_tokens, feat_dim] |
| `video_token_mask` | `list[bool]` | 预计算的视频 token mask |

### 元信息字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `meta.task` | `string` | 任务类型: `understanding`, `t2i`, `t2a`, `t2v`, `interleave` 等 |
| `meta.source` | `string` | 来源数据集名 |
| `meta.quality_tier` | `string` | 质量等级: `gold`, `silver`, `bronze` |
| `width` | `int` | 图片宽度 |
| `height` | `int` | 图片高度 |

## Special Token 在 conversations 中的用法

### 图片

| Token | 用途 |
|-------|------|
| `<image>` | 占位符, 在 tokenization 时展开为 `<img><IMG_CONTEXT×N></img>` |

### 音频

| Token | 用途 |
|-------|------|
| `<audio>` | 占位符, 展开为 `<audio><AUDIO_CONTEXT×N></audio>` |

### 视频

| Token | 用途 |
|-------|------|
| `<video>` | 占位符, 展开为 `<video><VIDEO_CONTEXT×N></video>` |

### 规则

1. **理解任务**: `<image>`/`<audio>`/`<video>` 出现在 human turn
2. **生成任务**: `<image>`/`<audio>`/`<video>` 出现在 gpt turn
3. **多模态输入**: 可在同一 turn 中混合多个模态标记
4. **交织输出**: gpt turn 可包含文本 + `<image>` + `<audio>` + `<video>` 的任意组合
5. 占位符数量必须与 `image`/`audio`/`video` 字段的文件数匹配

## 任务类型与 token 位置

| 任务类型 | human turn | gpt turn | meta.task |
|----------|-----------|----------|-----------|
| Image Understanding | `<image>\n{question}` | `{answer}` | `understanding` |
| T2I | `{caption}` | `<image>` | `t2i` |
| Audio Understanding | `<audio>\n{question}` | `{answer}` | `understanding` |
| T2A | `{instruction}` | `<audio>` | `t2a` |
| T2S (Speech) | `Read aloud: {text}` | `<audio>` | `t2s` |
| T2M (Music) | `Compose: {desc}` | `<audio>` | `t2m` |
| Video Understanding | `<video>\n{question}` | `{answer}` | `understanding` |
| T2V | `{caption}` | `<video>` | `t2v` |
| I2V | `<image>\nAnimate: {desc}` | `<video>` | `i2v` |
| AV Sync | `<video>\nGenerate audio` | `<audio>` | `t2a` |
| Text to AV | `{caption}` | `<video><audio>` | `interleave` |
| Interleaved OMNI | `{mixed_input}` | `{text}<image>{text}<audio>{text}` | `interleave` |

## 文件路径规则

- **相对路径**: 相对于 `DataArguments.data_path` (来自 data_dict 的 `data_path` 字段)
- **绝对路径**: 直接使用
- **环境变量**: `$DATA_ROOT/{dataset_name}/...` 由 data_registry.py 解析

## 行为约定

1. 如果样本包含 `image`/`audio`/`video` 字段但 conversations 中缺少对应的 `<image>`/`<audio>`/`<video>` 标记, `_get_item()` 会自动在首个 human turn 前插入
2. `audio_values`/`video_values` 优先于文件路径加载: 如果提供了预计算特征, 跳过文件读取
3. 视频支持 `.pt`/.npy`/`.npz` 格式的预计算特征文件
4. 音频只支持 `.wav` 格式 (PCM), 其他格式需预先转换
