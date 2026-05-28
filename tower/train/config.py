from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tower.config import PROJECT_ROOT

TRAIN_YML = PROJECT_ROOT / "note" / "train.yml"


@dataclass
class TrainConfig:
    stage: str = "understanding_warmup"
    init_mode: str = "scratch"
    weight_init: str = "random"
    model_config_path: str | None = "configs/model/sensenova_500m_mot"
    base_model: str | None = None
    model_name_or_path: str | None = None
    llm_model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = "configs/tokenizer/qwen3"
    datasets: str = "blip3o_short_pt"
    loss_weights: dict[str, float] = field(default_factory=lambda: {"ce": 1.0, "fm": 0.0})
    task_override: str | None = None
    train_buffer: bool = False
    extra_num_layers: int = 0
    num_hidden_layers: int = 26
    max_steps: int = 1000
    output_dir: str = "outputs/default"
    deepspeed: str | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    max_seq_length: int = 8192
    max_pixels: int = 262144
    min_pixels: int = 12544
    patch_size: int = 16
    downsample_ratio: float = 0.5
    data_flatten: bool = True
    loss_reduction: str = "square"
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    dataloader_num_workers: int = 2
    gen_image_size: int = 512
    seed: int = 42
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    report_to: str = "tensorboard"
    cfg_label_drop_prob: float = 0.0
    use_flow_tower: bool = False
    tower_self_cond_prob: float = 0.0
    tower_self_cond_cfg_min: float = 1.0
    tower_self_cond_cfg_max: float = 1.0
    tower_decoder_prob: float = 0.0
    audio_context_token_id: int = -1
    audio_patch_dim: int = 80

    @property
    def ce_weight(self) -> float:
        return float(self.loss_weights.get("ce", 1.0))

    @property
    def fm_weight(self) -> float:
        return float(self.loss_weights.get("fm", 0.0))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_pretrain_config(cfg: TrainConfig) -> None:
    if cfg.init_mode != "scratch" or cfg.weight_init != "random":
        return
    if cfg.stage == "understanding_warmup":
        if cfg.model_name_or_path:
            raise ValueError("0→1 scratch UW must not set model_name_or_path")
        if cfg.base_model or cfg.llm_model_name_or_path:
            raise ValueError("0→1 scratch UW must not set base_model / llm_model_name_or_path")
        if not cfg.model_config_path:
            raise ValueError("0→1 scratch UW requires model_config_path")
        if not cfg.tokenizer_name_or_path:
            raise ValueError("0→1 scratch UW requires tokenizer_name_or_path")


def load_train_config(*, config_path: Path | None = None, stage: str | None = None) -> TrainConfig:
    if config_path is not None:
        raw = _load_yaml(config_path)
        cfg = TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig.__dataclass_fields__})
        validate_pretrain_config(cfg)
        return cfg

    if stage is None:
        raise ValueError("Provide config_path or stage")

    train_yml = _load_yaml(TRAIN_YML)
    stage_cfg = train_yml.get("stages", {}).get(stage)
    if stage_cfg is None:
        raise KeyError(f"Unknown stage '{stage}' in {TRAIN_YML}")

    stage_file_map = {
        "world_pt": "world_pt.yaml",
        "understanding_warmup": "understanding_warmup.yaml",
        "generation_pt": "generation_pt.yaml",
        "unified_mt": "unified_mt.yaml",
        "unified_sft": "unified_sft.yaml",
    }
    fname = stage_file_map.get(stage, f"{stage}.yaml")
    defaults_path = PROJECT_ROOT / "configs" / "train" / fname
    base: dict[str, Any] = {"stage": stage}
    if defaults_path.is_file():
        base.update(_load_yaml(defaults_path))
    base.update(stage_cfg)
    cfg = TrainConfig(**{k: v for k, v in base.items() if k in TrainConfig.__dataclass_fields__})
    validate_pretrain_config(cfg)
    return cfg
