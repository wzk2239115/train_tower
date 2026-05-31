# Special Token 用法规范

## 注册的 Special Tokens

### 图像相关 (原有)

| Token | 类型 | 用途 |
|-------|------|------|
| `<img>` | wrapper start | 图像生成区域开始, gpt turn 中出现 |
| `</img>` | wrapper end | 图像生成区域结束 |
| `<IMG_CONTEXT>` | context | 图像理解占位, 每个 token 代表一个 image patch |
| `<image>` | 便捷标记 | 在 conversations 中使用, tokenization 时展开 |

展开规则: `<image>` → `<img><IMG_CONTEXT × N></img>`
其中 N = `image_grid_h × image_grid_w` (由图片分辨率和 patch_size 决定)

### 音频相关 (新增)

| Token | 类型 | 用途 |
|-------|------|------|
| `<audio>` | wrapper start + 便捷标记 | 音频区域开始, 理解和生成都使用 |
| `</audio>` | wrapper end | 音频区域结束 |
| `<AUDIO_CONTEXT>` | context | 音频 patch 占位, 每个 token 代表一个 mel-spectrogram patch |

展开规则: `<audio>` → `<audio><AUDIO_CONTEXT × N></audio>`
其中 N = 音频的 patch 数量 (由音频长度和 patch_frames/patch_bins 决定)

### 视频相关 (新增)

| Token | 类型 | 用途 |
|-------|------|------|
| `<video>` | wrapper start + 便捷标记 | 视频区域开始, 理解和生成都使用 |
| `</video>` | wrapper end | 视频区域结束 |
| `<VIDEO_CONTEXT>` | context | 视频 patch 占位, 每个 token 代表一个视频 patch |

展开规则: `<video>` → `<video><VIDEO_CONTEXT × N></video>`
其中 N = `num_frames × patches_per_frame`

### 文本相关 (原有)

| Token | 类型 | 用途 |
|-------|------|------|
| `<|im_start|>` | system | ChatML 格式开始 |
| `<|im_end|>` | system | ChatML 格式结束 |
| `<|endoftext|>` | system | 序列结束 |

## Token ID 注册流程

1. `constants.py` 定义 token 字符串常量和 `ALL_SPECIAL_TOKEN_LIST`
2. `build_tokenizer()` 调用 `tokenizer.add_tokens(ALL_SPECIAL_TOKEN_LIST, special_tokens=True)`
3. `build.py` 从 tokenizer 获取 token ID 并设置到模型属性:
   - `model.audio_context_token_id = tokenizer.convert_tokens_to_ids("<AUDIO_CONTEXT>")`
   - `model.audio_start_token_id = tokenizer.convert_tokens_to_ids("<audio>")`
   - `model.video_context_token_id = tokenizer.convert_tokens_to_ids("<VIDEO_CONTEXT>")`
   - `model.video_start_token_id = tokenizer.convert_tokens_to_ids("<video>")`

## Token 在不同阶段的行为

### 训练时 (tokenization)

```
tokenize_mm_chat_conversations() 处理流程:

1. 遍历 conversations
2. 对每个 turn, 替换:
   <image> → <img><IMG_CONTEXT × num_image_tokens[i]></img>
   <audio> → <audio><AUDIO_CONTEXT × num_audio_tokens[j]></audio>
   <video> → <video><VIDEO_CONTEXT × num_video_tokens[k]></video>
3. 使用 tokenizer 编码文本部分
4. 返回 input_ids, labels, attention_mask
```

### 训练时 (tower forward)

```
FlowJepaTowerTrainModel 处理流程:

1. _audio_token_mask(): 从 input_ids 中找 == audio_context_token_id 的位置
   → 生成 bool mask 标记哪些 token 是音频 patch
2. _extract_audio_clean(): 从 batch["audio_values"] 取音频特征, 投影到 hidden dim
3. _inject_audio(): 将投影后的音频 embedding 替换 hidden states 中对应位置
4. FM loss: 在 tower exit 层对 audio/video context tokens 计算 flow matching loss
```

### 推理时 (super_omni_gen)

```
super_omni_gen() 处理流程:

1. AR 自回归生成 text tokens
2. 遇到 <img>/<audio>/<video> start token 时:
   a. 暂停 AR
   b. 收集 context tokens (直到遇到对应的 end token)
   c. 从噪声开始, 执行 N 步 rectified flow denoising
   d. 用 decoder 将 latent 转为 pixel/waveform/video frame
   e. 将生成结果注入 KV cache
   f. 恢复 AR 继续生成 text
3. 返回最终结果
```

## Token 数量计算

### Image tokens

```python
# 由 dynamic_preprocess_native_resolution() 计算
num_tokens = (resized_w // patch_size) * (resized_h // patch_size)
# 例: 512×512 image, patch_size=16, downsample_ratio=0.5
# → 16×16 = 256 tokens per tile
```

### Audio tokens

```python
# 由 audio_file_to_patch_features() 计算
n_freq_bins = 80       # DataArguments.audio_n_mels
patch_bins = 10         # DataArguments.audio_patch_bins
patch_frames = 8        # DataArguments.audio_patch_frames
n_f = n_freq_bins // patch_bins  # = 8
n_t = num_spectrogram_frames // patch_frames
num_tokens = n_f * n_t  # 上限 max_patches=256
```

### Video tokens

```python
num_frames = 16         # DataArguments.video_num_frames
patches_per_frame = 256 # 默认值, 与图像相同
num_tokens = num_frames * patches_per_frame  # = 4096
```

## 禁忌

1. **不要在代码中硬编码 token ID** — 始终从 tokenizer 获取
2. **不要混用 `<image>` 和 `<img>`** — `<image>` 是数据层标记, `<img>` 是模型层标记
3. **不要在 human turn 中放 `<img>`** — 使用 `<image>`, tokenization 会自动展开
4. **不要在 gpt turn 中放 `<IMG_CONTEXT>`** — 这是理解占位符, 生成用 `<image>` 即可
5. **音频/视频的 end token 不能省略** — `<audio>...</audio>`, `<video>...</video>` 必须配对
