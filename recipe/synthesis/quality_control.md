# 数据质量过滤规则

## 分级标准

### Gold (人工验证 / 高置信度)

- 人工标注或人工审核
- CLIP-Score ≥ 0.30
- 文本流畅、无语法错误
- 模态内容与文本高度一致
- 无水印、无 NSFW、无隐私泄露
- 用于 SFT 阶段

### Silver (弱标注 + 过滤)

- ChatGPT/LLM 弱标注或自动标注
- CLIP-Score ≥ 0.22
- 文本基本流畅，允许轻微错误
- 模态内容与文本大致一致
- 经过去重和基本过滤
- 用于 MT 阶段

### Bronze (原始爬取)

- 自动爬取的原始数据
- 仅做了最基本的格式验证
- 需要强过滤才能使用
- 仅在需要极大规模数据时使用

## 各模态过滤规则

### 图像过滤

| 规则 | 阈值 | 说明 |
|------|------|------|
| 分辨率 | ≥ 256×256 | 过小的图像质量太差 |
| 宽高比 | ≤ 10:1 | 去除全景拼接等异常图 |
| CLIP-Score | ≥ 0.22 (silver), ≥ 0.30 (gold) | 文本-图像对齐度 |
| NSFW 检测 | confidence < 0.5 | 用 CLIP/安全分类器过滤 |
| 人脸检测 | 儿童人脸 blur/mask | LAION 已知问题 |
| 水印检测 | watermark score < 0.3 | 去除明显水印图 |
| 重复检测 | pHash 去重 | 相同图像只保留一条 |
| 文字占比 | text area < 60% | 去除以文字为主的图像 |

### 音频过滤

| 规则 | 阈值 | 说明 |
|------|------|------|
| 时长 | 1s ~ 60s | 太短无意义，太长显存溢出 |
| 采样率 | ≥ 16kHz | 低于此质量不够 |
| SNR | ≥ 10dB | 信噪比过低 |
| 静音占比 | silence < 50% | 去除大量静音的音频 |
| 音量 | RMS > 0.01 | 去除几乎无声的音频 |
| CLAP-Score | ≥ 0.25 (silver), ≥ 0.35 (gold) | 文本-音频对齐度 |
| 语言检测 | (仅 T2S) 匹配目标语言 | 音频内容与目标语言一致 |

### 视频过滤

| 规则 | 阈值 | 说明 |
|------|------|------|
| 分辨率 | ≥ 360p | 最低可接受分辨率 |
| 时长 | 2s ~ 30s | 训练用的视频片段长度 |
| 帧率 | ≥ 10fps | 过低帧率影响质量 |
| 运动幅度 | optical_flow_mean > threshold | 去除静态视频 (如PPT) |
| 黑帧占比 | black_frames < 20% | 去除片头片尾黑屏 |
| CLIP-Score | ≥ 0.22 (silver), ≥ 0.30 (gold) | 文本-视频对齐度 |
| NSFW 检测 | 每帧 confidence < 0.5 | 逐帧检测 |
| 重复检测 | 视频指纹去重 | 相同视频只保留一条 |

### 文本过滤

| 规则 | 阈值 | 说明 |
|------|------|------|
| 长度 | 10 ~ 5000 字符 | 过短无意义，过长超出 context |
| 语言检测 | 匹配标注语言 | 中文标注不应是英文 |
| 重复 n-gram | 重复 4-gram < 30% | 去除机器生成的重复文本 |
| 标点密度 | 标点占比 2% ~ 30% | 异常标点密度说明质量差 |
| 特殊字符 | 非常见 Unicode < 5% | 去除乱码 |
| toxicity | toxicity score < 0.5 | 去除有害内容 |
| PII 检测 | 无邮箱/电话/身份证 | 去除隐私信息 |

## LAION 系列特殊过滤

LAION-5B/Aesthetics 是最大规模图文数据，但存在已知问题，需要以下额外过滤：

```python
def laion_extra_filter(sample):
    # 1. 美学分数
    if sample.get("aesthetic_score", 0) < 6.0:
        return False  # 只用 6.0+ 子集

    # 2. NSFW (使用 LAION 官方 nsfw_score)
    if sample.get("nsfw_score", 0) > 0.5:
        return False

    # 3. 儿童隐私
    if sample.get("has_child_face", False):
        return False

    # 4. 水印
    if sample.get("watermark_score", 0) > 0.5:
        return False

    # 5. 文字图 (meme/截图)
    if sample.get("text_area_ratio", 0) > 0.4:
        return False

    # 6. 去重
    if is_duplicate_phash(sample["image"]):
        return False

    return True
```

## 跨模态一致性过滤

对 AV Sync 和 Interleaved 数据，需要额外的跨模态一致性检查：

| 检查项 | 方法 | 阈值 |
|--------|------|------|
| 音画对应 | Audio-Visual CLIP score | ≥ 0.25 |
| 音频→文本 | CLAP-Score | ≥ 0.30 |
| 视频→文本 | CLIP-Score (帧平均) | ≥ 0.25 |
| 时间对齐 | 音频事件与视频帧匹配 | 人工抽检 |

## 过滤流程

```
原始数据
  ↓ [格式验证] (JSON 合法性, 文件存在性)
  ↓ [基础过滤] (分辨率, 时长, 分辨率, SNR)
  ↓ [质量过滤] (CLIP/CLAP-Score, NSFW, 水印)
  ↓ [去重] (pHash, 文本 simhash, 视频指纹)
  ↓ [PII/toxicity] (隐私和有害内容)
  ↓ [分级标注] (gold/silver/bronze)
  ↓
可用训练数据
```

## 质量指标追踪

在训练过程中应持续追踪以下指标：

| 指标 | 目标 | 监控频率 |
|------|------|----------|
| CLIP-Score (image-text) | ≥ 0.30 | 每 1000 步 |
| CLAP-Score (audio-text) | ≥ 0.30 | 每 1000 步 |
| FID-30K (image gen) | ≤ 15 | 每 5000 步 |
| FVD-30K (video gen) | ≤ 300 | 每 5000 步 |
| FD (audio gen) | ≤ 10 | 每 5000 步 |
| MMBench (image und) | ≥ 70 | 每 stage |
| C-Eval (text) | ≥ 50 | 每 stage |
| AudioCaps-BLEU | ≥ 30 | 每 stage |

当质量指标下降时，检查是否是数据质量问题，触发重新过滤。
