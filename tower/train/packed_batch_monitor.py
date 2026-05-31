from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers.trainer_callback import TrainerCallback

from tower.train.config import TrainConfig
from tower.train.diagnostics import distributed_rank, log_training_phase
from tower.train.vram_tune import _get_anchor, peak_vram_score, _CKPT_FACTOR, _FLOW_TOWER_FACTOR, _PIXEL_EXP, _SEQ_EXP


@dataclass(frozen=True)
class PackedBatchStats:
    packed_seq_length: int
    num_packed_samples: int
    num_images: int
    num_vision_tokens: int
    peak_vram_score: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def step_peak_vram_score(
    *,
    packed_seq_length: int,
    num_packed_samples: int,
    cfg: TrainConfig,
) -> float:
    """Relative peak VRAM for this micro-batch (actual L / pack count vs stable anchor)."""
    anchor = _get_anchor(cfg.stage)
    if anchor is None:
        return 0.0
    pack_ratio = num_packed_samples / max(anchor.per_device_train_batch_size, 1)
    pixel_ratio = (cfg.max_pixels / max(anchor.max_pixels, 1)) ** _PIXEL_EXP
    seq_ratio = (packed_seq_length / max(anchor.max_seq_length, 1)) ** _SEQ_EXP
    ckpt_ratio = _CKPT_FACTOR if cfg.gradient_checkpointing else 1.0
    ckpt_anchor = _CKPT_FACTOR if anchor.gradient_checkpointing else 1.0
    flow_ratio = _FLOW_TOWER_FACTOR if cfg.use_flow_tower else 1.0
    flow_anchor = _FLOW_TOWER_FACTOR if anchor.use_flow_tower else 1.0
    return pack_ratio * pixel_ratio * seq_ratio * (ckpt_ratio / ckpt_anchor) * (flow_ratio / flow_anchor)


def compute_packed_batch_stats(
    batch: dict[str, Any],
    cfg: TrainConfig,
    img_context_token_id: int,
) -> PackedBatchStats:
    input_ids = batch["input_ids"]
    if input_ids.ndim == 2:
        packed_seq_length = int(input_ids.shape[1])
        ids = input_ids[0]
    else:
        packed_seq_length = int(input_ids.shape[0])
        ids = input_ids

    num_vision_tokens = int((ids == img_context_token_id).sum().item())

    boundaries = batch.get("seq_boundaries")
    if boundaries is not None:
        if isinstance(boundaries, torch.Tensor):
            num_packed_samples = max(int(boundaries.numel()) - 1, 1)
        else:
            num_packed_samples = max(len(boundaries) - 1, 1)
    else:
        num_packed_samples = max(len(batch.get("tasks", batch.get("is_gen", [1]))), 1)

    num_images = 0
    image_grid_hw = batch.get("image_grid_hw")
    if image_grid_hw and len(image_grid_hw) > 0 and image_grid_hw[0] is not None:
        grid = image_grid_hw[0]
        num_images = int(grid.shape[0]) if isinstance(grid, torch.Tensor) else len(grid)

    peak_score = step_peak_vram_score(
        packed_seq_length=packed_seq_length,
        num_packed_samples=num_packed_samples,
        cfg=cfg,
    )
    return PackedBatchStats(
        packed_seq_length=packed_seq_length,
        num_packed_samples=num_packed_samples,
        num_images=num_images,
        num_vision_tokens=num_vision_tokens,
        peak_vram_score=peak_score,
    )


def attach_packed_batch_stats(batch: dict[str, Any], cfg: TrainConfig, img_context_token_id: int) -> dict[str, Any]:
    batch["_tower_batch_stats"] = compute_packed_batch_stats(batch, cfg, img_context_token_id).to_dict()
    return batch


def _log_interval(global_step: int) -> bool:
    if os.environ.get("TOWER_PACKED_BATCH_LOG", "1") == "0":
        return False
    every = int(os.environ.get("TOWER_PACKED_BATCH_LOG_EVERY", "1") or 1)
    warmup = int(os.environ.get("TOWER_PACKED_BATCH_LOG_WARMUP", "20") or 20)
    if global_step < warmup:
        return True
    return every > 0 and global_step % every == 0


class TowerPackedBatchMonitorCallback(TrainerCallback):
    """Log packed sequence / vision token stats each step (B2 monitoring)."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self._config_peak_vram_score = 0.0
        anchor = _get_anchor(cfg.stage)
        if anchor is not None:
            self._config_peak_vram_score = peak_vram_score(cfg, anchor)

    def on_train_begin(self, args, state, control, **kwargs):
        if distributed_rank() != 0:
            return
        log_training_phase(
            "packed_batch monitor enabled",
            config_peak_vram_score=f"{self._config_peak_vram_score:.3f}",
            max_seq_length=self.cfg.max_seq_length,
            max_pixels=self.cfg.max_pixels,
            pack_samples=self.cfg.per_device_train_batch_size,
        )

    def on_step_begin(self, args, state, control, **kwargs):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if distributed_rank() != 0:
            return
        step = int(state.global_step)
        if not _log_interval(step):
            return

        trainer = kwargs.get("trainer")
        stats = getattr(trainer, "_last_packed_batch_stats", None) if trainer is not None else None
        if not stats:
            return

        extra: dict[str, object] = dict(stats)
        import torch

        if torch.cuda.is_available():
            peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
            extra["cuda_peak_gib"] = f"{peak_gib:.2f}"

        if stats.get("peak_vram_score", 0) > 1.0:
            extra["WARN"] = "peak_vram_score>1.0"

        log_training_phase(f"packed_batch step={step}", **extra)
