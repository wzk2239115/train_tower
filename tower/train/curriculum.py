from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from tower.train.config import TrainConfig, CURRICULUM_TRAIN_OVERRIDE_KEYS

logger = logging.get_logger(__name__)


@dataclass
class CurriculumRuntime:
    """Mutable training-time view of step-based curriculum data settings."""

    cfg: TrainConfig
    phase_index: int = -1
    settings: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    should_save: bool = True

    def has_curriculum(self) -> bool:
        return bool(self.cfg.curriculum)

    def sync(
        self,
        step: int,
        *,
        data_args,
        training_args,
        tokenizer,
        model=None,
        train_dataset=None,
    ) -> bool:
        """Apply curriculum phase for ``step``; return True when phase changed."""
        settings = self.cfg.curriculum_data_settings_for_step(step)
        new_index = int(settings["phase_index"])
        if new_index == self.phase_index and self.settings:
            return False

        prev = dict(self.settings)
        self.phase_index = new_index
        self.settings = settings

        data_args.max_seq_length = int(settings["max_seq_length"])
        data_args.max_pixels = int(settings["max_pixels"])
        data_args.min_pixels = int(settings["min_pixels"])
        if tokenizer is not None:
            tokenizer.model_max_length = int(settings["max_seq_length"])

        training_args.per_device_train_batch_size = int(settings["per_device_train_batch_size"])
        training_args.gradient_accumulation_steps = int(settings["gradient_accumulation_steps"])
        training_args.learning_rate = float(settings["learning_rate"])

        datasets = str(settings.get("datasets", self.cfg.datasets))
        if train_dataset is not None and prev and prev.get("datasets") != datasets:
            from tower.train.dataset import rebuild_unified_dataset_base

            rebuild_unified_dataset_base(
                train_dataset,
                tokenizer=tokenizer,
                data_args=data_args,
                datasets=datasets,
            )

        loss_weights = settings.get("loss_weights")
        if isinstance(loss_weights, dict):
            self.cfg.loss_weights = dict(loss_weights)

        self.cfg.task_override = settings.get("task_override")
        for key in CURRICULUM_TRAIN_OVERRIDE_KEYS:
            if key in settings:
                setattr(self.cfg, key, float(settings[key]))

        prev_stage = prev.get("stage") if prev else None
        new_stage = settings["stage"]
        if model is not None and (prev_stage is None or new_stage != prev_stage):
            from tower.train.freeze import apply_stage_freeze, apply_tower_exit_freeze
            from tower.unify.flow_tower import FlowJepaTowerTrainModel

            apply_stage_freeze(model.model, new_stage)
            if isinstance(model, FlowJepaTowerTrainModel):
                apply_tower_exit_freeze(model, new_stage)
            logger.info("Applied stage freeze for curriculum stage=%s", new_stage)

        if (
            model is not None
            and prev_stage is not None
            and new_stage != prev_stage
            and os.environ.get("TOWER_EXPORT_STAGE_SNAPSHOTS", "1") == "1"
        ):
            output_dir = getattr(self, "output_dir", None)
            should_save = getattr(self, "should_save", True)
            if output_dir and should_save:
                from tower.unify.export import export_multi_artifacts

                snapshot_dir = str(Path(output_dir) / "artifacts" / prev_stage)
                export_multi_artifacts(model, snapshot_dir)
                logger.info(
                    "Exported stage snapshot for %s at step=%s -> %s",
                    prev_stage,
                    step,
                    snapshot_dir,
                )

        if prev:
            logger.info(
                "Curriculum phase %s -> %s at step=%s: stage=%s seq=%s pixels=%s batch=%s "
                "grad_accum=%s datasets=%s lr=%s",
                prev.get("phase_index"),
                new_index,
                step,
                settings["stage"],
                settings["max_seq_length"],
                settings["max_pixels"],
                settings["per_device_train_batch_size"],
                settings["gradient_accumulation_steps"],
                datasets,
                settings["learning_rate"],
            )
        elif self.has_curriculum():
            logger.info(
                "Curriculum phase %s at step=%s: stage=%s seq=%s pixels=%s batch=%s grad_accum=%s "
                "datasets=%s lr=%s",
                new_index,
                step,
                settings["stage"],
                settings["max_seq_length"],
                settings["max_pixels"],
                settings["per_device_train_batch_size"],
                settings["gradient_accumulation_steps"],
                datasets,
                settings["learning_rate"],
            )

        return True


class CurriculumCallback(TrainerCallback):
    def __init__(
        self,
        runtime: CurriculumRuntime,
        *,
        data_args,
        training_args,
        tokenizer,
        train_dataset=None,
    ):
        self.runtime = runtime
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.runtime.output_dir = training_args.output_dir
        self.runtime.should_save = training_args.should_save

    def _apply(self, trainer, model, step: int) -> None:
        changed = self.runtime.sync(
            step,
            data_args=self.data_args,
            training_args=self.training_args,
            tokenizer=self.tokenizer,
            model=model,
            train_dataset=self.train_dataset,
        )
        if not changed or trainer is None:
            return
        trainer.args.per_device_train_batch_size = int(
            self.runtime.settings["per_device_train_batch_size"]
        )
        trainer.args.gradient_accumulation_steps = int(
            self.runtime.settings["gradient_accumulation_steps"]
        )
        trainer.args.learning_rate = float(self.runtime.settings["learning_rate"])
        if trainer.optimizer is not None:
            lr = float(self.runtime.settings["learning_rate"])
            for group in trainer.optimizer.param_groups:
                group["lr"] = lr
        trainer._train_dataloader = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._apply(kwargs.get("trainer"), kwargs.get("model"), int(state.global_step))

    def on_step_begin(self, args, state, control, **kwargs):
        self._apply(kwargs.get("trainer"), kwargs.get("model"), int(state.global_step))
