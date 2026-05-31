# 交织多模态合成数据管线设计

## 为什么需要合成数据

公开可用的交织多模态数据（text+image+audio+video 混合输入/输出）极少。现有数据集主要是单一模态对（image-text, audio-text, video-text）。Super-Omni 的核心能力——在单次响应中交织输出多种模态——必须通过合成数据来训练。

## 合成目标

| 任务类型 | 输入 | 输出 | 数据量目标 |
|----------|------|------|-----------|
| T2OMNI | text | text + `<image>` + `<audio>` + `<video>` | 100K |
| OMNI2T | text + `<image>` + `<audio>` + `<video>` | text | 50K |
| OMNI2OMNI | text + `<image>` + `<audio>` + `<video>` | text + `<image>` + `<audio>` + `<video>` | 50K |

## 管线架构

```
Phase 1: 种子收集
├── 从现有数据集选高质量种子
├── image-text pairs (JourneyDB, LAION-Aesthetic)
├── audio-text pairs (AudioCaps, Clotho)
├── video-text pairs (OpenVid, Panda-70M)
└── av pairs (VGGSound, AVSpeech)

Phase 2: 主题对齐
├── 对每种子的 caption 做 topic clustering
├── 相同 topic 的 image/audio/video 种子配对
├── 生成 "故事线" prompt
└── 输出: (caption, image_path, audio_path, video_path) 四元组

Phase 3: LLM 重写
├── 用 LLM 将 caption 重写为自然对话
├── 在对话中插入 <image>/<audio>/<video> 标记
├── 确保标记位置语义合理
└── 输出: conversations 格式的 JSONL

Phase 4: 质量过滤
├── CLIP-Score 检查 image-text 对齐
├── CLAP-Score 检查 audio-text 对齐
├── 人工抽检 5%
└── 去除低分样本

Phase 5: 多样性增强
├── 同一四元组生成多种对话模板
├── 变换语气/风格/长度
├── 中文/英文双语版本
└── 最终输出: interleaved_omni JSONL
```

## Phase 1: 种子收集

### 策略: 跨模态主题匹配

核心思想：从不同模态的数据集中找到**语义相似**的样本，组合成交织数据。

```python
# 伪代码
def collect_seeds():
    # 从每个模态收集 (caption, path) 对
    images = load("journeydb", fields=["caption", "image_path"])
    audios = load("audiocaps", fields=["caption", "audio_path"])
    videos = load("openvid", fields=["caption", "video_path"])

    # 用 CLIP/CLAP encoder 统一 embed
    img_embeds = clip_encode([c for c, _ in images])
    aud_embeds = clap_encode([c for c, _ in audios])
    vid_embeds = clip_encode([c for c, _ in videos])

    # 跨模态最近邻匹配
    # 找到 caption 语义最接近的 (image, audio, video) 三元组
    for i, (img_cap, img_path) in enumerate(images):
        nn_audio = find_nearest(img_embeds[i], aud_embeds)
        nn_video = find_nearest(img_embeds[i], vid_embeds)
        yield {
            "image": img_path,
            "audio": audios[nn_audio][1],
            "video": videos[nn_video][1],
            "theme_caption": img_cap,
        }
```

### 种子数量目标

| 来源 | 数量 | 用途 |
|------|------|------|
| JourneyDB + AudioCaps + OpenVid | 50K 三元组 | 自然场景 |
| LAION-Aesthetic + WavCaps + Panda-70M | 100K 三元组 | 大规模语义 |
| VGGSound | 50K 三元组 | 音视频同步 |
| AVSpeech | 10K 三元组 | 人物说话场景 |

## Phase 2: 主题对齐

对每个种子三元组，生成一个 "故事线" prompt：

```python
storyline_templates = [
    "Imagine a {theme} scene: {image_desc}. The sound of {audio_desc} fills the air. "
    "In the background, {video_desc} unfolds.",

    "Let me paint you a picture. {image_desc}. Listen closely — {audio_desc}. "
    "And watch how {video_desc} plays out over time.",

    "Here's a complete sensory experience of {theme}.\n"
    "Visual: {image_desc}\n"
    "Audio: {audio_desc}\n"
    "Motion: {video_desc}",
]
```

## Phase 3: LLM 重写

用 LLM (如 GPT-4o / Claude) 将种子重写为自然对话：

**System Prompt:**
```
You are a creative AI assistant that can output text, images, audio, and video
in an interleaved manner. When you want to show an image, use <image>.
When you want to play a sound, use <audio>. When you want to show a video,
use <video>. Make the output natural and flowing.
```

**Human Prompt (种子):**
```
Create a vivid, multi-sensory description of this scene:
- Visual: A golden sunset over the ocean
- Sound: Waves crashing on rocks
- Motion: The sun slowly descending

The response should include text, an image, audio, and a video clip in a natural flow.
```

**Expected GPT Output:**
```
Let me share this breathtaking sunset with you.

As the golden sun begins its descent, the sky transforms into a canvas of
warm oranges and deep purples.

<image>

The sound of waves rhythmically crashing against the weathered rocks creates
a meditative backdrop.

<audio>

Watch as the sun slowly dips below the horizon, painting the clouds in
ever-changing hues.

<video>

This is one of those moments where nature reminds us of its timeless beauty.
```

### 重写模板

| 模板类型 | 描述 | 输出模式 |
|----------|------|----------|
| 叙事型 | 讲故事风格, 自然插入多媒体 | text→image→text→audio→text→video→text |
| 描述型 | 逐模态描述场景 | text→image→audio→video→text |
| 教学型 | 解释某个概念时穿插示例 | text→image→text→audio→text→video→text |
| 对话型 | 模拟多轮对话 | Q→A(text+image)→Q→A(text+audio+video) |
| 创作型 | "帮我创作一个多媒体故事" | text→image→text→audio→text→video→text |

## Phase 4: 质量过滤

### 自动过滤规则

```python
def quality_filter(sample):
    # 1. CLIP-Score 检查 image-text 对齐
    clip_score = compute_clip_score(sample["image"], sample["conversations"])
    if clip_score < 0.25:
        return False

    # 2. CLAP-Score 检查 audio-text 对齐
    clap_score = compute_clap_score(sample["audio"], sample["conversations"])
    if clap_score < 0.3:
        return False

    # 3. Special token 数量检查
    gpt_text = get_gpt_turns(sample)
    if gpt_text.count("<image>") > 3:  # 不超过 3 张图
        return False
    if gpt_text.count("<audio>") > 3:
        return False
    if gpt_text.count("<video>") > 2:
        return False

    # 4. 文本长度检查
    total_text = sum(len(t["value"]) for t in sample["conversations"])
    if total_text < 50 or total_text > 5000:
        return False

    # 5. 模态间距检查 (避免连续多个 <image><audio><video> 没有文本)
    if has_consecutive_modality_tokens(gpt_text):
        return False

    return True
```

### 人工抽检

- 随机抽 5% 样本人工审核
- 评估维度: 语义连贯性, 模态插入位置合理性, 信息冗余度
- 不合格率 > 20% 时触发批次重做

## Phase 5: 多样性增强

### 对话模板变换

同一个种子可生成多种对话:

```python
# 原始
{"from": "human", "value": "Describe this sunset scene."}
{"from": "gpt", "value": "Look at this...\n<image>\nAnd listen...\n<audio>\nAnd watch...\n<video>"}

# 变换1: 更详细的指令
{"from": "human", "value": "I'm writing a novel. Can you show me what a sunset over the ocean looks, sounds, and feels like?"}
{"from": "gpt", "value": "Here's a complete sensory portrait...\n\nFirst, the visual...\n<image>\n\nThe soundscape...\n<audio>\n\nAnd the passage of time...\n<video>"}

# 变换2: 中文版
{"from": "human", "value": "描述一下海边的日落场景。"}
{"from": "gpt", "value": "让我带你感受这个场景。\n\n首先是画面...\n<image>\n\n海浪的声音...\n<audio>\n\n日落的延时...\n<video>"}
```

### 多语言

| 语言 | 比例 |
|------|------|
| English | 60% |
| Chinese | 30% |
| 其他 (日/韩/法/德) | 10% |

## 最终输出规格

```json
{
  "id": "synth_omni_00001",
  "image": "$DATA_ROOT/synth_omni_sft/images/00001.jpg",
  "audio": "$DATA_ROOT/synth_omni_sft/audios/00001.wav",
  "video": "$DATA_ROOT/synth_omni_sft/videos/00001.mp4",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "...<image>...<audio>...<video>..."}
  ],
  "meta": {
    "task": "interleave",
    "source": "synth_omni_sft",
    "quality_tier": "gold",
    "synthesis_method": "cross_modal_nn_matching + llm_rewrite",
    "seed_datasets": ["journeydb", "audiocaps", "openvid"]
  }
}
```

## 数据量规划

| 阶段 | 数据量 | 质量 | 用途 |
|------|--------|------|------|
| Synth-PT | 200K | silver | unified_mt 预训练 |
| Synth-SFT | 50K | gold | unified_sft 高质量 |
| Synth-DPO (未来) | 10K | gold | RLHF/偏好对齐 |
