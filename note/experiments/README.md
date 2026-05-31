# 实验日志

记录在算力平台（Notebook / K8s Pod）上跑的训练实验，便于断点续训、换机和写论文/复盘。

## 命名

`YYYY-MM-DD_<stage>_<profile>.md`，例如 `2025-05-31_world_pt_h800.md`。

## 每条日志建议包含

- 平台实例名、GPU 型号与数量、代码与数据路径
- 启动命令、配置文件、Git commit（训练时所在 commit）
- 起止时间、global_step、产出目录与 checkpoint
- 下一阶段入口（checkpoint 路径、续跑脚本）

## 流水线索引

| Stage | 配置 (H800) | 输出目录 | max_steps |
|-------|-------------|----------|-----------|
| 0 world_pt | `configs/train/world_pt_h800.yaml` | `outputs/pretrain/world_pt_h800` | 50,000 |
| 1 understanding_warmup | `configs/train/understanding_warmup_h800_resume.yaml` | `outputs/pretrain/uw_h800` | 200,000 |
| 2 generation_pt | `configs/train/generation_pt_h800_resume.yaml` | `outputs/pretrain/gen_pt_h800` | 100,000 |
| 3 unified_mt | `configs/train/unified_mt_h800_resume.yaml` | `outputs/pretrain/mt_h800` | 50,000 |
| 4 unified_sft | `configs/train/unified_sft_h800_resume.yaml` | `outputs/pretrain/sft_h800` | 4,500 |

Stage 0 完成后续跑：`./scripts/h100_resume_pipeline.sh`  
比 stable 更快且能塞进 80GB 时用：`H800_PROFILE=max ./scripts/h100_resume_pipeline.sh`（8×10×2 + grad ckpt；旧版 8×20 会 OOM）
