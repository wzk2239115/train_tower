from __future__ import annotations

import os
import time

from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

logger = logging.get_logger(__name__)


def distributed_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) or 0)


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1") or 1)


def distributed_barrier(tag: str) -> None:
    """Synchronize all ranks before/after rank-asymmetric work (e.g. preflight)."""
    world = distributed_world_size()
    if world <= 1:
        return
    import torch.distributed as dist

    if not dist.is_available():
        logger.warning("[%s] torch.distributed unavailable; skip barrier", tag)
        return
    if not dist.is_initialized():
        logger.warning(
            "[%s] process group not initialized (world_size=%s); "
            "barrier skipped — multi-GPU preflight may deadlock",
            tag,
            world,
        )
        return
    rank = dist.get_rank()
    t0 = time.monotonic()
    if rank == 0:
        logger.info("[%s] barrier enter (world_size=%s)", tag, world)
    dist.barrier()
    if rank == 0:
        logger.info("[%s] barrier done (%.2fs)", tag, time.monotonic() - t0)


def log_training_phase(phase: str, **extra: object) -> None:
    rank = distributed_rank()
    world = distributed_world_size()
    parts = [f"[train_phase][rank {rank}/{world}] {phase}"]
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))


class TowerTrainDiagnosticsCallback(TrainerCallback):
    """Log first-step timing and train loop entry (helps debug apparent hangs at 0%)."""

    def __init__(self) -> None:
        self._train_begin_at: float | None = None
        self._step0_begin_at: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_begin_at = time.monotonic()
        log_training_phase(
            "on_train_begin",
            max_steps=args.max_steps,
            per_device_batch=args.per_device_train_batch_size,
            grad_accum=args.gradient_accumulation_steps,
            deepspeed=bool(getattr(kwargs.get("trainer"), "is_deepspeed_enabled", False)),
        )

    def on_step_begin(self, args, state, control, **kwargs):
        if int(state.global_step) != 0:
            return
        self._step0_begin_at = time.monotonic()
        since_begin = (
            (self._step0_begin_at - self._train_begin_at) if self._train_begin_at else 0.0
        )
        log_training_phase(
            "step 0 begin (first forward/backward — can take several minutes on large batches)",
            seconds_since_train_begin=f"{since_begin:.1f}",
        )

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) != 1:
            return
        elapsed = (
            (time.monotonic() - self._step0_begin_at) if self._step0_begin_at is not None else 0.0
        )
        log_training_phase("step 1 complete (first optimizer step done)", step0_seconds=f"{elapsed:.1f}")
