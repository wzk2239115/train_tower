from __future__ import annotations

import os
import subprocess
import time

from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging

from tower.train.config import TrainConfig

logger = logging.get_logger(__name__)


def distributed_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) or 0)


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1") or 1)


def git_head_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def sdpa_block_attn_status() -> dict[str, str | bool]:
    """Report SDPA patch state without applying compat."""
    disabled = os.environ.get("TOWER_DISABLE_SDPA_BLOCK_ATTN", "0") == "1"
    patched = False
    try:
        from tower.unify.backends.sensenova import import_modeling_qwen3

        patched = bool(getattr(import_modeling_qwen3(), "_NATIVE_SDPA_BLOCK_ATTN", False))
    except Exception:
        pass
    expected = not disabled
    if disabled:
        state = "disabled"
    elif patched:
        state = "active"
    else:
        state = "pending"
    return {
        "state": state,
        "disabled": disabled,
        "patched": patched,
        "expected": expected,
        "backends": os.environ.get("TOWER_SDPA_BACKENDS", "efficient,cudnn"),
    }


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


def log_startup_summary(cfg: TrainConfig) -> None:
    """One-line startup summary: profile, SDPA, batch, multimodal knobs."""
    if distributed_rank() != 0:
        return

    world = distributed_world_size()
    micro = cfg.per_device_train_batch_size
    accum = cfg.gradient_accumulation_steps
    global_batch = micro * accum * world
    profile = os.environ.get("H800_PROFILE", "custom")
    vram_tune = os.environ.get("TOWER_H800_VRAM_TUNE", "")
    if not vram_tune:
        from tower.train.vram_tune import _vram_tune_enabled

        vram_tune = "1" if _vram_tune_enabled() else "0"

    sdpa = sdpa_block_attn_status()
    dl_workers = int(os.environ.get("TOWER_DATALOADER_NUM_WORKERS", str(cfg.dataloader_num_workers)) or 0)

    log_training_phase(
        "startup",
        git=git_head_short(),
        profile=profile,
        stage=cfg.stage,
        sdpa=sdpa["state"],
        global_batch=global_batch,
        micro=micro,
        accum=accum,
        gpus=world,
        max_seq=cfg.max_seq_length,
        max_px=cfg.max_pixels,
        pack=micro,
        grad_ckpt=cfg.gradient_checkpointing,
        flow_tower=cfg.use_flow_tower,
        vram_tune=vram_tune,
        dl_workers=dl_workers,
    )
    if sdpa["state"] == "pending" and profile in ("max", "extreme", "turbo"):
        logger.warning(
            "[startup] SDPA patch not yet applied — confirm logs show fused SDPA after model build"
        )


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


class TowerStepSummaryCallback(TrainerCallback):
    """Periodic concise step logs; always logs anomalies (high VRAM score / loss spike)."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self._last_loss: float | None = None
        self._warn_vram = float(os.environ.get("TOWER_STEP_SUMMARY_VRAM_WARN", "1.0") or 1.0)
        self._loss_spike = float(os.environ.get("TOWER_STEP_SUMMARY_LOSS_SPIKE", "3.0") or 3.0)

    def _every(self) -> int:
        return int(os.environ.get("TOWER_STEP_SUMMARY_EVERY", "100") or 0)

    def on_step_end(self, args, state, control, **kwargs):
        if distributed_rank() != 0:
            return
        step = int(state.global_step)
        if step <= 0:
            return

        trainer = kwargs.get("trainer")
        stats = getattr(trainer, "_last_packed_batch_stats", None) if trainer else None
        vram_score = float(stats.get("peak_vram_score", 0)) if stats else 0.0

        loss = None
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                loss = float(last["loss"])

        anomaly = vram_score > self._warn_vram
        if loss is not None and self._last_loss is not None and self._last_loss > 0:
            if loss > self._last_loss * self._loss_spike:
                anomaly = True
        if loss is not None:
            self._last_loss = loss

        every = self._every()
        if not anomaly and every > 0 and step % every != 0:
            return

        parts: dict[str, object] = {"step": step}
        if loss is not None:
            parts["loss"] = f"{loss:.4f}"
        if stats:
            parts["L"] = stats.get("packed_seq_length")
            parts["pack"] = stats.get("num_packed_samples")
            parts["imgs"] = stats.get("num_images")
            parts["vram"] = f"{vram_score:.2f}"
        import torch

        if torch.cuda.is_available():
            parts["cuda_gib"] = f"{torch.cuda.max_memory_allocated() / (1024**3):.1f}"
        if anomaly:
            parts["WARN"] = "anomaly"

        log_training_phase("step_summary", **parts)


class TowerLossBreakdownCallback(TrainerCallback):
    """将各模态 loss 分量注入到 HF Trainer 的 log_history 中。"""

    LOSS_NAMES = {
        "ce_loss": "ce_loss",
        "tower_ce_loss": "tower_ce_loss",
        "image_fm_loss": "image_fm_loss",
        "audio_fm_loss": "audio_fm_loss",
        "video_fm_loss": "video_fm_loss",
        "image_jepa_loss": "image_jepa_loss",
        "text_hidden_loss": "text_hidden_loss",
    }

    LOSS_NAMES_CN = {
        "ce_loss": "文本CE",
        "tower_ce_loss": "中层CE",
        "image_fm_loss": "图像FM",
        "audio_fm_loss": "音频FM",
        "video_fm_loss": "视频FM",
        "image_jepa_loss": "图像JEPA",
        "text_hidden_loss": "文本隐藏层",
    }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        breakdown = getattr(trainer, "_last_loss_breakdown", None)
        if not breakdown:
            return
        for key, log_key in self.LOSS_NAMES.items():
            if key in breakdown:
                logs[log_key] = round(breakdown[key], 4)

        if distributed_rank() == 0 and state.log_history:
            parts = []
            for key, cn_name in self.LOSS_NAMES_CN.items():
                val = breakdown.get(key)
                if val is not None and val > 0:
                    parts.append(f"{cn_name}={val:.2f}")
            if parts:
                logger.info("  [Loss分解] %s", " | ".join(parts))

        grad_weights = breakdown.get("grad_norm_weights")
        if isinstance(grad_weights, dict) and distributed_rank() == 0:
            gw_parts = []
            for key, cn_name in self.LOSS_NAMES_CN.items():
                w = grad_weights.get(key)
                if w is not None:
                    gw_parts.append(f"{cn_name}×{w:.4f}")
            if gw_parts:
                logger.info("  [GradNorm权重] %s", " | ".join(gw_parts))


class TowerGradNormCallback(TrainerCallback):
    """Log per-task gradient norms (measured during forward, see flow_tower)."""

    def on_after_backward(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        balancer = getattr(model, "_grad_norm_balancer", None)
        if balancer is None or not balancer.enabled:
            return
        breakdown = getattr(kwargs.get("trainer"), "_last_loss_breakdown", None)
        if not breakdown:
            return
        # Per-task gradient norms are now measured inside
        # FlowJepaTowerTrainModel.forward (where the autograd graph is alive)
        # via _measure_grad_norms, instead of being recomputed here from
        # detached floats — which always yielded zero and never updated weights.
        grad_norms = breakdown.get("grad_norms") or {}
        if grad_norms and distributed_rank() == 0:
            parts = []
            for key, g in grad_norms.items():
                if g > 0:
                    cn = TowerLossBreakdownCallback.LOSS_NAMES_CN.get(key, key)
                    parts.append(f"{cn}‖g‖={g:.2f}")
            if parts:
                logger.info("  [GradNorm梯度] %s", " | ".join(parts))
