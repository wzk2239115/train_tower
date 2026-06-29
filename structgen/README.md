# structgen — 复杂曲面结构件生成（几何生成 + 隐式场预测 + 结构约束优化）

> **不是 next-token 预测。** 输入是文本需求 / 草图 / 参考图 / 结构约束，
> 输出是**可制造的 3D 结构件表示**（SDF / occupancy 体素场 → mesh → STL）。

`structgen/` 是 `train_tower` 里的一个独立子系统，复用了项目的 flow-matching
原语（`rectified_flow_velocity_loss` / logit-normal 时间采样 / self-conditioning），
但把输出空间从 pixel-patch 换成了**体素隐式场**。

## 1. 设计：Step-3.7-Flash 当 backbone（组件），不是当 token 生成器

```
文本 / 草图 / 参考图 / 约束
        │
        ▼  ┌────────────────────────────────────────────┐
[Step-3.7-Flash backbone  ← 冻结的预训练多模态组件]        │
        │   vision encoder + MoE LLM → 多模态隐层          │  不生成 token
        ▼                                                  │
   condition tokens (pooled + token seq)                   │
        │                                                  │
        ▼                                                  │
[3D 体素 Geometry Decoder  ← 可训练]                       │
   rectified flow on SDF/occupancy voxel field              │
   条件注入: adaLN(t+pool) + 瓶颈 cross-attn(cond tokens)   │
        │  net 预测 x0 (与项目 rectified_flow_velocity_loss 一致)
        ▼
   SDF grid + occupancy grid
        │
        ▼
   marching cubes → mesh → STL / OBJ / SDF
        │
   多目标 loss: SDF L1/L2 · occupancy BCE · Chamfer ·
               normal consistency · curvature smooth ·
               topology/connectivity · manufacturability(overhang/wall)
```

**关键**：Step-3.7-Flash 只负责"看懂"需求（编码文本/草图/参考结构），把它当成一个
预训练得不错的多模态编码器；真正"画"出几何的是体素 decoder，用 flow matching 生成
SDF 场，再 marching cubes 出 mesh。这正好对应你说的"生成的都不是 token，只是把这个
模型作为经过不错预训练的模型"。

## 2. 在算力机上切到真实 Step-3.7-Flash

开发机本地快照不全，用 `proxy`（离线、可训练的轻量编码器）跑通闭环；算力机有完整权重，
一行切到真实 backbone：

```bash
# 算力机（完整 Step-3.7-Flash + modeling 代码）
git pull
python -m structgen.cli train \
    --backbone stepfun \
    --pretrained-path /path/to/Step-3.7-Flash \
    --batch 8 --steps 50000 --res 64 --base-ch 64
```

`StepfunBackbone`（`structgen/model/backbone.py`）用 `AutoModelForCausalLM(..., trust_remote_code=True)`
加载完整模型并冻结，forward 时抽取 understanding 路径的最后一层 hidden state 作为 condition。
接口（`ConditionOutput{tokens, pooled}`）和 proxy 完全一致，decoder 无需改动。

## 3. 本机冒烟（~1 分钟，单卡）

```bash
pip install scikit-image          # marching cubes
./scripts/structgen_smoke.sh      # proxy backbone + tiny decoder → STL
```

产物：`outputs/structgen/gen_gyroid.stl`（训练 30 步、采样 30 步生成的结构件 mesh）。

## 4. 数据：程序合成的 TPMS / lattice

外壳（box / cylinder / sphere / superellipsoid）与内部拓扑做 CSG（smooth-intersect），
内部拓扑覆盖你点名的全部类型：

| topology | 说明 |
|----------|------|
| `gyroid` | TPMS Gyroid |
| `diamond` | TPMS Diamond |
| `schwarz_p` | TPMS Schwarz-P |
| `graded_gyroid` | 渐变密度 Gyroid（壁厚随 z 变化） |
| `voronoi` | Voronoi 支杆晶格 |
| `lattice` | 三轴 strut 晶格 |

每个样本保存：text prompt（结构化描述）、rendered sketch（三视投影）、shell、SDF grid、
occupancy grid、surface points/normals。合成器在 `structgen/data/tpms.py` +
`structgen/data/sampler.py`，数据集在 `structgen/data/dataset.py`。

> 后续可接 ABC / Fusion360 / DeepCAD 外壳 + 程序合成内部拓扑（`recipe/` 同款思路），
> 框架已留好 `StructGenDataset` 的扩展点。

## 5. 输出表示优先级（已落地）

| 优先级 | 表示 | 实现 |
|--------|------|------|
| 第一 | **SDF / occupancy 隐式场** | decoder 直接输出体素 SDF（channel 0）+ occupancy（channel 1） |
| 第二 | **CAD DSL + TPMS 参数** | `SpecimenParams`（`structgen/data/sampler.py`）作为可解释 GT |
| 第三 | **mesh / STL / OBJ** | marching cubes + 自带 STL/OBJ writer（无需 trimesh） |

## 6. 多目标 loss（`structgen/losses.py`）

| loss | 作用 |
|------|------|
| `fm_velocity_loss` | rectified-flow velocity MSE（**复用** `tower.train.losses`） |
| `sdf_l1` / `sdf_l2` | SDF 回归 |
| `occupancy_bce` | 符号(sdf) 推出的 occupancy BCE（logit=-sdf/eps，autocast 安全） |
| `chamfer_loss_from_field` | 预测场 vs GT surface 的双向 Chamfer |
| `normal_consistency_loss` | 零等值面附近 SDF 梯度(法向)对齐 |
| `curvature_smooth_loss` | SDF 拉普拉斯惩罚 → 光滑可制造面 |
| `topology_loss` | 惩罚过薄实体薄片 → 连通性 |
| `manufacturability_loss` | 沿打印轴的悬垂惩罚 |

权重在 `StructGenConfig.loss_weights`（`LossWeights`）里统一调。

## 7. CLI

```bash
# 训练（本机 smoke）
python -m structgen.cli train --smoke --smoke-steps 30

# 查看合成数据分布（导出 GT STL）
python -m structgen.cli inspect --n 8 --res 64

# 从 prompt 生成结构件
python -m structgen.cli generate --ckpt <decoder.pt> \
    --prompt "structural cylinder part with internal gyroid infill" \
    --out part.stl

# 带草图输入
python -m structgen.cli generate --ckpt <decoder.pt> --sketch my_sketch.png \
    --prompt "lightweight stiff part" --out part.obj
```

## 8. 目录

```
structgen/
├── config.py                     # StructGenConfig (backbone/decoder/flow/loss/train)
├── data/
│   ├── tpms.py                   # SDF 原语 + TPMS + Voronoi + lattice + boolean
│   ├── sampler.py                # 样本构建(shell ∩ topology) + 网格/曲面采样
│   └── dataset.py                # StructGenDataset (field+surface+prompt+sketch)
├── losses.py                     # 多目标几何 loss + rectified-flow 复用
├── model/
│   ├── backbone.py               # BackboneAdapter: Proxy / Qwen / Stepfun
│   ├── voxelnnet.py              # 3D 体素 velocity U-Net (adaLN + cross-attn + self-cond)
│   ├── geometry_decoder.py       # decode_loss(训练) + sample(Euler 流匹配推理)
│   └── meshing.py                # marching cubes → STL/OBJ
├── train.py                      # 训练循环
├── infer.py                      # 推理 / 导出
└── cli.py                        # train / generate / inspect
tests/test_structgen.py           # 合成 / loss / mesh / decoder / backbone / 全闭环
scripts/structgen_smoke.sh        # 本机冒烟
```

## 9. 和项目 Flow-JEPA Tower 的关系

`structgen` 是**独立子系统**（不动现有 omni 训练管线），但**复用**了项目最值得借鉴的
flow-matching 思想：

- `tower/train/losses.py::rectified_flow_velocity_loss` —— 网络预测 x0、velocity 空间 MSE；
- `sample_logit_normal_timesteps` / `sample_flow_batch` —— logit-normal 时间调度；
- `ElfFlowTowerExit` 的 self-conditioning + timestep embed —— 对应这里的 adaLN + self-cond。

区别：Tower 的 flow 在 **pixel/latent patch 序列**上、条件来自 LLM 同一 backbone；
`structgen` 的 flow 在 **3D 体素 SDF 场**上、条件来自 **冻结的 Step-3.7-Flash**。

## 10. 后续迭代方向

- [ ] latent AE：体素场先过 VAE → latent flow（更大分辨率，如 128³）
- [ ] 真实外壳数据：ABC / Fusion360 shell + 程序合成内部拓扑
- [ ] 可选 FEA / 物理 reward（stiffness-to-weight）做强化 / reward-weighted 训练
- [ ] CAD DSL 解码头（TPMS 类型 / 周期 / 壁厚 / 密度梯度 结构化输出）
- [ ] 装配约束：孔位 / 装配面作为额外 condition 通道
