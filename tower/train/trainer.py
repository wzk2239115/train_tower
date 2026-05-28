from __future__ import annotations

import os
import pathlib
from dataclasses import asdict

import torch
from transformers import HfArgumentParser, Trainer, set_seed
from transformers.utils import logging

from tower.paths import ensure_train_paths
from tower.train.config import TrainConfig
from tower.train.dataset import make_unified_data_module
from tower.train.freeze import apply_stage_freeze, apply_tower_exit_freeze
from tower.train.registry import inject_data_dict
from tower.unify.build import build_model_and_tokenizer
from tower.unify.flow_tower import FlowJepaTowerTrainModel
from tower.unify.train_model import SenseNovaTrainModel

logger = logging.get_logger(__name__)


class TowerTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save:
            dest = output_dir if output_dir is not None else self.args.output_dir
            self.model.save_pretrained(dest, safe_serialization=False)

    def _save(self, output_dir: str | None = None, state_dict=None):
        self.save_model(output_dir, _internal_call=True)


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def _resolve_deepspeed(cfg: TrainConfig) -> str | None:
    """Pick DeepSpeed config or disable when unsafe on single-process launch."""
    if not cfg.deepspeed:
        return None
    if os.environ.get("TOWER_NO_DEEPSPEED", "0") == "1":
        logger.warning("DeepSpeed disabled (TOWER_NO_DEEPSPEED=1)")
        return None
    if os.environ.get("LOCAL_RANK") is None and int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        logger.warning(
            "DeepSpeed disabled: launch with torchrun for multi-GPU, "
            "or set TOWER_NO_DEEPSPEED=1 for single-GPU without DeepSpeed"
        )
        return None
    return cfg.deepspeed


def run_training(cfg: TrainConfig) -> None:
    ensure_train_paths()
    inject_data_dict()

    from neo.train.argument import DataArguments, TrainingArguments

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.init_mode == "scratch" and cfg.weight_init == "random":
        logger.info(
            "0→1 scratch PT: random init from %s (no pretrained weights)",
            cfg.model_config_path,
        )

    data_kwargs = {
        "dataset_use": cfg.datasets,
        "max_seq_length": cfg.max_seq_length,
        "max_pixels": cfg.max_pixels,
        "min_pixels": cfg.min_pixels,
        "patch_size": cfg.patch_size,
        "downsample_ratio": cfg.downsample_ratio,
        "data_flatten": cfg.data_flatten,
        "loss_reduction": cfg.loss_reduction,
    }

    train_kwargs = {
        "output_dir": cfg.output_dir,
        "max_steps": cfg.max_steps,
        "do_train": True,
        "do_eval": False,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": cfg.warmup_steps,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "bf16": cfg.bf16,
        "remove_unused_columns": False,
        "report_to": cfg.report_to,
        "seed": cfg.seed,
        "lr_scheduler_type": "cosine_with_min_lr",
        "min_lr_ratio": 0.1,
    }
    ds = _resolve_deepspeed(cfg)
    if ds:
        train_kwargs["deepspeed"] = ds

    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_dict({**data_kwargs, **train_kwargs})
    if hasattr(training_args, "save_safetensors"):
        training_args.save_safetensors = False

    set_seed(training_args.seed)

    neo_model, tokenizer = build_model_and_tokenizer(cfg)
    if cfg.bf16:
        neo_model = neo_model.to(dtype=torch.bfloat16)

    model = FlowJepaTowerTrainModel(neo_model, cfg) if cfg.use_flow_tower else SenseNovaTrainModel(neo_model, cfg)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    apply_stage_freeze(model.model, cfg.stage)
    if cfg.use_flow_tower and isinstance(model, FlowJepaTowerTrainModel):
        apply_tower_exit_freeze(model, cfg.stage)

    data_module = make_unified_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        cfg=cfg,
    )

    trainer = TowerTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    ckpt_dirs = sorted(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    resume = ckpt_dirs and (ckpt_dirs[-1] / "pytorch_model.bin").is_file()
    if resume:
        logger.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    meta_path = pathlib.Path(training_args.output_dir) / "train_config.yaml"
    import yaml

    meta_path.write_text(yaml.dump(asdict(cfg), default_flow_style=False), encoding="utf-8")
