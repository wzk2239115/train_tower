# Flow-JEPA Tower：分层连续世界模型设想

## 0. 核心命题（训练哲学）

> **在塔状结构上进行训练。一个能胜任不同维度任务的表征，一定是好表征。**

这里的「维度」不是向量空间的维数，而是 **任务维度 / 能力维度**：

| 维度 | 塔层出口 | 探针任务 | 对表征的要求 |
|------|----------|----------|--------------|
| 空间/物理 | L0 JEPA | 遮挡区域隐式预测 | 局部几何、物体边界、不变性 |
| 语义/动力学 | L1 ELF-FM | latent 上 flow 去噪 | 概念、属性、短程「变化」 |
| 语言/推理 | L2 CE | VQA、grounding、指令 | 符号接地、跨模态对齐 |
| 生成/合成 | L3 ELF-FM | pixel patch flow | 细粒度视觉细节、可控生成 |

**关键**：这些探针挂在 **同一主干、不同深度**，同时（或按权重联合）施加梯度。  
若 Layer 8 的 hidden 既能支撑 JEPA，Layer 16 又能支撑 CE，Layer 24 还能支撑 FM——  
说明中间表征是 **多用途的**，而不是只为某一个下游任务过拟合。

这与传统 pipeline 的本质区别：

```text
传统：UW(CE) → GenPT(FM) → MT(CE+FM)   # 换任务 ≈ 换目标，表征是否通用未验证
Tower：L0∥L1∥L2∥L3 联合探针同一 backbone   # 表征质量 = 多维度 simultaneously 可解
```

**好表征的操作性定义（本项目的标准）**：

1. 浅层出口在 **无像素重建** 的 JEPA 上 loss 下降；
2. 中层出口在 **embedding 空间 flow** 上可预测 velocity；
3. 中深层出口在 **CE** 上能接地语言；
4. 深层出口在 **pixel flow** 上能生成图像；
5. 且以上能力 **共享参数**，不是四个独立模型拼起来。

训练目标因此不是「先学 A 再学 B」，而是 **让分馏塔每一层都析出真实能力**——  
能力越能在多维度上被探针激活，主干表征越接近「世界模型」而非「任务模型」。

---

## 1. 架构：石油分馏塔

设计一个结合 **JEPA、ELF / Flow Matching、分层多出口架构** 的多模态世界模型。

| 塔层 | 深度（26L 示例） | 出口类型 | 学什么 |
|------|------------------|----------|--------|
| L0 世界表征 | Layer 0–7 | **JEPA** | 遮挡 patch 的隐空间预测，不重建像素 |
| L1 语义/动力学 | Layer 8–15 | **ELF-FM（latent）** | 在 vision embedding 上做 flow，短 caption / 语义 |
| L2 理解 | Layer 16–21 | **CE** | VQA、grounding、指令跟随 |
| L3 生成 | Layer 22–25 | **ELF-FM（pixel）** | T2I patch flow，与 SenseNova 推理对齐 |

- 共享 **一个 LLM 主干**（MoT 双路径保留）
- 在不同深度 **堆叠 ELF flow head**（每层出口 = 该维度的任务探针）
- 底层 JEPA、高层 ELF，中间 CE —— 能力随深度「析出」，**多层同时约束表征**

**当前 `train_tower` 的问题**：只有塔顶一个 `fm_head` + 单一 CE，分阶段换 loss，无法验证「中间层表征是否多维度可用」。

**真正的 Tower**：同一 forward 中，多个出口按 `note/tower.yml` 权重 **联合训练** 共享 backbone。

---

## 2. 借鉴 JEPA 的部分

JEPA 核心：

> 不直接重建像素，而是在隐空间中预测目标区域的表征。

```text
可见 patch → Context（LLM 前 L 层）→ 上下文 hidden
完整 patch → Target Encoder (EMA) → 目标表征
Predictor → 在 pred_mask 位置 MSE(predictor, target)
```

在 Tower L0 出口：

- 输入：understanding path 的 image token hidden（Layer 7 处 hook）
- Mask：随机块级 mask（参考 `omni-jepa/omni_jepa/masking.py`）
- Target：`target_encoder` EMA 跟随 context projector
- Loss：仅在 mask 位置 MSE

数据：BLIP3o PT caption、无标注图像池（COCO / COYO metadata）

---

## 3. 堆叠 ELF 的部分

ELF 在 **连续 latent** 上做 rectified flow，关键技巧（已在 Phase 1 部分接入顶层 FM）：

| 技巧 | L1 semantic ELF | L3 generative ELF |
|------|-----------------|-------------------|
| logit-normal t | ✓ | ✓ |
| resolution noise scale | 固定 / 小 scale | ✓（按 grid_hw） |
| CFG label drop | caption drop | caption drop |
| self-conditioning | Phase 3 可选 | Phase 3 可选 |
| 堆叠 depth | 2× ELF block | 2× ELF block |

**L1 vs L3 的区别**：

```text
L1 semantic_elf:
  clean = vision_tower.gen(image)     # [N_patch, D] embedding
  z = (1-t)*noise*scale + t*clean
  ELF blocks(h_L8, z, t) → v_pred
  loss: velocity MSE in embedding space

L3 generative_elf:
  clean = pixel_values patches        # [N_patch, 3*P*P]
  z = (1-t)*noise*scale + t*clean
  vit_noisy = vision_tower.gen(z)
  h = LLM layers 22–25 + fm_head
  loss: velocity MSE in pixel space（当前 SenseNovaTrainModel._fm_forward）
```

每一层 ELF 出口结构（`tower/unify/tower_exits.py`）：

```text
hidden[Lk] + timestep_embed(t) [+ noise_scale_embed]
    → ElfBlock × depth（RMSNorm + FFN residual）
    → fm_head → x_pred
    → rectified_flow_velocity_loss
```

---

## 4. 与现有 train_tower 的映射

```
现状 (SenseNovaTrainModel)          目标 (FlowJepaTowerTrainModel)
─────────────────────────          ────────────────────────────────
单 fm_head @ LLM 顶层               L1 + L3 两个 ELF 出口（可继续加）
无 JEPA                             L0 JEPA 出口
CE @ lm_head                        L2 CE 出口（同 lm_head）
UW→GenPT→MT→SFT 分阶段              Tower stage 控制各出口 loss 权重
MoT und/gen path                    保留，各出口指定 gen/und indicator
```

配置文件：

- 架构出口定义：`note/tower.yml`
- 训练 stage：`note/tower_train.yml`（待建）
- 代码入口：`tower/unify/flow_tower.py`

---

## 5. 训练课程（Tower curriculum）

训练仍分 stage 是为 **算力与稳定性**，但每一 stage 的语义是「激活哪些维度探针」，而非「只做单一任务」：

```text
Stage 0  world_pt — 打地基：空间 + 语义维度
  激活: jepa=1.0, semantic_elf=0.3
  意图: 浅层必须先成为「可预测的世界」表征，再叠语言/生成

Stage 1  understanding_warmup — 语言维度
  激活: ce=1.0 (L2)，可选 jepa=0.05 正则（防止浅层表征退化）
  意图: 同一 backbone 在 L2 探针上可接地，且不牺牲 L0

Stage 2  generation_pt — 合成维度
  激活: generative_elf=1.0, semantic_elf=0.1
  意图: 深层 pixel flow 与中层 latent flow 一致 → 表征跨尺度连贯

Stage 3  unified_mt / sft — 全塔联合（核心）
  激活: ce=1.0, generative_elf=0.1–0.3, semantic_elf=0.15, jepa=0.05
  意图: **四维度探针同时约束** —— 这才是「好表征」的验收阶段
```

**MT/SFT 阶段最重要**：若只在 Gen PT 训 FM、只在 UW 训 CE，永远是在换探针；  
全塔联合 loss 才是在检验 **一个表征能否同时胜任多维度任务**。

---

## 6. 实现路线

### Phase A（当前）：脚手架

- [x] Phase 1 FM 对齐 SenseNova（logit-normal t, noise scale, CFG drop）
- [x] `FlowJepaTowerTrainModel`：按层 hook + 多出口 loss 聚合
- [x] `note/tower.yml`：26L 出口定义
- [x] `tower/unify/tower_exits.py`：`JepaTowerExit`, `ElfFlowTowerExit`

### Phase B：接通训练

- [x] `TrainConfig.use_flow_tower: bool`
- [x] `note/tower_train.yml` + `scripts/train_tower_world.sh`
- [x] freeze 策略扩展：`tower/train/freeze.py` 按出口名 freeze
- [x] MT/SFT 默认开启全塔联合 loss（四探针同时 >0，`use_flow_tower: true`）

### Phase C：ELF 完整技巧

- [ ] self-conditioning @ L3
- [ ] per-sample CE/FM branch mixing（ELF decoder_prob）
- [ ] EMA eval checkpoint

---

## 7. 参数与算力（500M 量级估算）

26L LLM hidden=768，每层 ELF stack depth=2：

| 出口 | 新增参数量（粗估） |
|------|-------------------|
| jepa (predictor + target) | ~4M |
| semantic_elf (2 blocks + head) | ~6M |
| generative_elf (2 blocks + head) | ~12M（head 到 patch dim） |
| **合计** | ~22M（+4% on 500M base） |

训练：world_pt 可与 UW 合并或前置 20k–50k steps；多出口 loss 需调权重避免 L3 梯度压制 L0。

---

## 8. 参考仓库

| repo | 借鉴 |
|------|------|
| `train_tower/` | MoT 主干、数据管线、rectified flow loss |
| `ELF/` | ELF block、self-cond、CFG training、logit-normal schedule |
| `omni-jepa/` | JEPA target EMA、块级 mask、predictor |
| `SenseNova-U1/` | 推理 CFG、noise scale、timestep embed |
