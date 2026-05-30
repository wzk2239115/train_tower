from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from tower.train.config import CURRICULUM_OVERRIDE_KEYS, TrainConfig

logger = logging.get_logger(__name__)


@dataclass
class CurriculumRuntime:
    """Mutable training-time view of step-based curriculum data settings."""

    cfg: TrainConfig
    phase_index: int = -1
    settings: dict[str, Any] = field(default_factory=dict)

    def has_curriculum(self) -> bool:
        return bool(self.cfg.curriculum)

    def sync(self, step: int, *, data_args, training_args, tokenizer, model=None) -> bool:
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

        if prev:
            logger.info(
                "Curriculum phase %s -> %s at step=%s: stage=%s seq=%s pixels=%s batch=%s grad_accum=%s",
                prev.get("phase_index"),
                new_index,
                step,
                settings["stage"],
                settings["max_seq_length"],
                settings["max_pixels"],
                settings["per_device_train_batch_size"],
                settings["gradient_accumulation_steps"],
            )
        elif self.has_curriculum():
            logger.info(
                "Curriculum phase %s at step=%s: stage=%s seq=%s pixels=%s batch=%s grad_accum=%s",
                new_index,
                step,
                settings["stage"],
                settings["max_seq_length"],
                settings["max_pixels"],
                settings["per_device_train_batch_size"],
                settings["gradient_accumulation_steps"],
            )

        prev_stage = prev.get("stage") if prev else None
        new_stage = settings["stage"]
        if model is not None and prev_stage is not None and new_stage != prev_stage:
            from tower.train.freeze import apply_stage_freeze, apply_tower_exit_freeze
            from tower.unify.flow_tower import FlowJepaTowerTrainModel

            apply_stage_freeze(model.model, new_stage)
            if isinstance(model, FlowJepaTowerTrainModel):
                apply_tower_exit_freeze(model, new_stage)
            logger.info("Applied stage freeze for curriculum stage=%s", new_stage)

        return True


class CurriculumCallback(TrainerCallback):
    def __init__(
        self,
        runtime: CurriculumRuntime,
        *,
        data_args,
        training_args,
        tokenizer,
    ):
        self.runtime = runtime
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer

    def _apply(self, trainer, model, step: int) -> None:
        changed = self.runtime.sync(
            step,
            data_args=self.data_args,
            training_args=self.training_args,
            tokenizer=self.tokenizer,
            model=model,
        )
        if not changed or trainer is None:
            return
        trainer.args.per_device_train_batch_size = int(
            self.runtime.settings["per_device_train_batch_size"]
        )
        trainer.args.gradient_accumulation_steps = int(
            self.runtime.settings["gradient_accumulation_steps"]
        )
        trainer._train_dataloader = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._apply(kwargs.get("trainer"), kwargs.get("model"), int(state.global_step))

    def on_step_begin(self, args, state, control, **kwargs):
        self._apply(kwargs.get("trainer"), kwargs.get("model"), int(state.global_step))
