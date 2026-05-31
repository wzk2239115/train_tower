from __future__ import annotations

import os
import pathlib
import time
from dataclasses import asdict

import torch
from transformers import HfArgumentParser, Trainer, set_seed
from transformers.optimization import get_scheduler
from transformers.utils import logging

from tower.train.config import TrainConfig
from tower.train.curriculum import CurriculumCallback
from tower.train.dataset import make_unified_data_module
from tower.train.packed_batch_monitor import TowerPackedBatchMonitorCallback
from tower.train.diagnostics import (
    TowerStepSummaryCallback,
    TowerTrainDiagnosticsCallback,
    distributed_barrier,
    distributed_rank,
    log_startup_summary,
    log_training_phase,
)
from tower.train.freeze import apply_stage_freeze, apply_tower_exit_freeze
from tower.train.registry import inject_data_dict
from tower.train.vram_tune import apply_h800_vram_tune
from tower.unify.build import build_model_and_tokenizer
from tower.unify.export import export_multi_artifacts
from tower.unify.flow_tower import FlowJepaTowerTrainModel
from tower.unify.train_model import SenseNovaTrainModel

logger = logging.get_logger(__name__)
WRAPPER_WEIGHTS_NAME = "tower_wrapper.bin"


class TowerTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """DeepSpeedZeroOptimizer lacks ``defaults``; pass ``min_lr_rate`` explicitly."""
        if (
            self.is_deepspeed_enabled
            and self.args.lr_scheduler_type == "cosine_with_min_lr"
            and self.lr_scheduler is None
        ):
            if optimizer is None:
                optimizer = self.optimizer
            kwargs = dict(self.args.lr_scheduler_kwargs or {})
            min_lr = kwargs.pop("min_lr", None)
            lr = float(self.args.learning_rate)
            if min_lr is not None and lr > 0:
                kwargs["min_lr_rate"] = float(min_lr) / lr
            elif "min_lr_rate" not in kwargs:
                kwargs["min_lr_rate"] = float(getattr(self.args, "min_lr_ratio", 0.0))
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=kwargs,
            )
            self._created_lr_scheduler = True
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer=optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        stats = inputs.pop("_tower_batch_stats", None)
        if stats is not None:
            self._last_packed_batch_stats = stats
        if hasattr(model, "set_curriculum_step"):
            model.set_curriculum_step(self.state.global_step)
        if num_items_in_batch is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save:
            dest = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(dest, exist_ok=True)
            self.model.save_pretrained(dest, safe_serialization=False)
            torch.save(self.model.state_dict(), os.path.join(dest, WRAPPER_WEIGHTS_NAME))

    def _save(self, output_dir: str | None = None, state_dict=None):
        if not self.args.should_save:
            return
        dest = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(dest, exist_ok=True)
        self.model.save_pretrained(dest, safe_serialization=False)
        payload = state_dict if state_dict is not None else self.model.state_dict()
        torch.save(payload, os.path.join(dest, WRAPPER_WEIGHTS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target_model = model if model is not None else self.model
        wrapper_ckpt = os.path.join(resume_from_checkpoint, WRAPPER_WEIGHTS_NAME)
        if os.path.isfile(wrapper_ckpt):
            wrapper_state = torch.load(wrapper_ckpt, map_location="cpu")
            target_model.load_state_dict(wrapper_state, strict=False)
            logger.info("Loaded full wrapper state from %s", wrapper_ckpt)


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


def _run_dataloader_preflight(trainer: Trainer, steps: int) -> None:
    """Warm up dataloader on every rank, then barrier before trainer.train().

    Previously only rank 0 prefetched while other ranks entered trainer.train() and
    blocked on DeepSpeed/NCCL collectives — appearing as a hang at 0/30 steps.
    """
    if steps <= 0:
        return
    dl = trainer.get_train_dataloader()
    if dl is None:
        return

    rank = distributed_rank()
    target = int(steps)
    log_training_phase("dataloader preflight start", batches=target)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    it = iter(dl)
    fetched = 0
    t0 = time.monotonic()
    progress = (
        tqdm(total=target, desc=f"DataLoad[r{rank}]", leave=False)
        if tqdm is not None and rank == 0
        else None
    )
    try:
        while fetched < target:
            try:
                _ = next(it)
            except StopIteration:
                break
            fetched += 1
            if progress is not None:
                progress.update(1)
            elif fetched % 10 == 0 or fetched == target:
                logger.info(
                    "[rank %s] Dataloader preflight progress: %s/%s", rank, fetched, target
                )
    finally:
        if progress is not None:
            progress.close()

    log_training_phase(
        "dataloader preflight done",
        fetched=fetched,
        target=target,
        seconds=f"{time.monotonic() - t0:.1f}",
    )
    distributed_barrier("after_dataloader_preflight")


def run_training(cfg: TrainConfig) -> None:
    inject_data_dict()

    from tower.train.size_preset import apply_size_preset_to_train_config, assert_model_tower_layer_consistency
    from tower.unify.backends import import_train_arguments

    DataArguments, TrainingArguments = import_train_arguments()

    cfg = apply_size_preset_to_train_config(cfg)
    if cfg.size_preset:
        assert_model_tower_layer_consistency(cfg)

    cfg = apply_h800_vram_tune(cfg)

    log_startup_summary(cfg)
    log_training_phase(
        "run_training start",
        stage=cfg.stage,
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        datasets=cfg.datasets,
        batch=cfg.per_device_train_batch_size,
        grad_accum=cfg.gradient_accumulation_steps,
        max_pixels=cfg.max_pixels,
        grad_ckpt=cfg.gradient_checkpointing,
    )

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

    dl_workers = int(os.environ.get("TOWER_DATALOADER_NUM_WORKERS", str(cfg.dataloader_num_workers)) or 0)

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
        "dataloader_num_workers": dl_workers,
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
    if dl_workers != cfg.dataloader_num_workers:
        logger.info(
            "Override dataloader_num_workers: cfg=%s env=%s",
            cfg.dataloader_num_workers,
            dl_workers,
        )

    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_dict({**data_kwargs, **train_kwargs})
    if hasattr(training_args, "save_safetensors"):
        training_args.save_safetensors = False

    set_seed(training_args.seed)

    t_build = time.monotonic()
    neo_model, tokenizer = build_model_and_tokenizer(cfg)
    if cfg.bf16:
        neo_model = neo_model.to(dtype=torch.bfloat16)

    model = FlowJepaTowerTrainModel(neo_model, cfg) if cfg.use_flow_tower else SenseNovaTrainModel(neo_model, cfg)
    hidden = getattr(model.config, "llm_config", model.config)
    llm_hidden = getattr(hidden, "hidden_size", getattr(model.config, "hidden_size", "?"))
    log_training_phase(
        "model ready",
        use_flow_tower=cfg.use_flow_tower,
        llm_hidden_size=llm_hidden,
        seconds=f"{time.monotonic() - t_build:.1f}",
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    apply_stage_freeze(model.model, cfg.curriculum_stage_for_step(0) if cfg.curriculum else cfg.stage)
    if cfg.use_flow_tower and isinstance(model, FlowJepaTowerTrainModel):
        apply_tower_exit_freeze(
            model,
            cfg.curriculum_stage_for_step(0) if cfg.curriculum else cfg.stage,
        )

    data_module = make_unified_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        cfg=cfg,
    )
    curriculum_runtime = data_module.pop("curriculum_runtime")
    callbacks = [
        TowerTrainDiagnosticsCallback(),
        TowerPackedBatchMonitorCallback(cfg),
        TowerStepSummaryCallback(cfg),
    ]
    if cfg.curriculum:
        callbacks.append(
            CurriculumCallback(
                curriculum_runtime,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
                train_dataset=data_module["train_dataset"],
            )
        )
        logger.info("Length/resolution curriculum enabled with %s phases", len(cfg.curriculum))

    trainer = TowerTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    preflight_steps = int(os.environ.get("TOWER_DATALOADER_PREFLIGHT_STEPS", "0") or 0)
    _run_dataloader_preflight(trainer, preflight_steps)
    distributed_barrier("before_trainer_train")

    ckpt_dirs = sorted(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    resume = ckpt_dirs and (ckpt_dirs[-1] / "pytorch_model.bin").is_file()
    log_training_phase("trainer.train() enter", resume=resume, preflight_steps=preflight_steps)
    if resume:
        logger.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    log_training_phase("trainer.train() finished")

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    export_multi_artifacts(model, training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    meta_path = pathlib.Path(training_args.output_dir) / "train_config.yaml"
    import yaml

    meta_path.write_text(yaml.dump(asdict(cfg), default_flow_style=False), encoding="utf-8")
