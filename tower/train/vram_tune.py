"""H800 VRAM auto-tune: keep peak memory within the stable anchor envelope.

Root cause (why ``max`` OOMs but ``stable`` runs)
-------------------------------------------------
* UW / Gen PT use ``create_block_causal_mask`` → LLM attention is always **eager**.
* Eager attention peak scales ~ ``batch × seq²``; ``seq`` grows with ``max_pixels`` (capped by
  ``max_seq_length``).
* ``gradient_accumulation_steps`` only raises **global** batch over time; each micro-step still
  uses ``per_device_train_batch_size`` activations → **does not** reduce peak VRAM.
* Old ``max`` raised **both** micro-batch (10 vs 8) **and** ``max_pixels`` (6M vs 4M) → ~1.7×
  peak vs stable → OOM on 80GB.

Smart strategy
--------------
* Never exceed the stable profile's peak-VRAM score (per stage).
* Recover throughput via ``gradient_accumulation_steps`` (and optional ``TOWER_TARGET_GLOBAL_BATCH``).
* Force ``gradient_checkpointing`` on eager-attn stages when tuning is active.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from transformers.utils import logging

from tower.train.config import TrainConfig

logger = logging.get_logger(__name__)

# Stable ``*_h800_resume.yaml`` anchors (known-good on 8×80GB).
_STAGE_ANCHORS: dict[str, dict[str, int | bool]] = {
    "understanding_warmup": {
        "per_device_train_batch_size": 8,
        "max_pixels": 4_194_304,
        "max_seq_length": 8192,
        "gradient_checkpointing": True,
        "use_flow_tower": False,
    },
    "generation_pt": {
        "per_device_train_batch_size": 8,
        "max_pixels": 4_194_304,
        "max_seq_length": 8192,
        "gradient_checkpointing": True,
        "use_flow_tower": False,
    },
    "unified_mt": {
        "per_device_train_batch_size": 4,
        "max_pixels": 4_194_304,
        "max_seq_length": 16384,
        "gradient_checkpointing": True,
        "use_flow_tower": True,
    },
    "unified_sft": {
        "per_device_train_batch_size": 4,
        "max_pixels": 4_194_304,
        "max_seq_length": 16384,
        "gradient_checkpointing": True,
        "use_flow_tower": True,
    },
    "world_pt": {
        "per_device_train_batch_size": 4,
        "max_pixels": 8_388_608,
        "max_seq_length": 8192,
        "gradient_checkpointing": True,
        "use_flow_tower": True,
    },
}

# Exponent on pixel ratio: vision tokens grow sub-linearly with max_pixels cap.
_PIXEL_EXP = 0.85
# Exponent on seq cap: attention is quadratic in sequence length.
_SEQ_EXP = 2.0
# Activation savings with gradient checkpointing vs none (eager-attn LLM).
_CKPT_FACTOR = 0.55
# Flow tower hooks add overhead vs plain UW/Gen.
_FLOW_TOWER_FACTOR = 1.25


@dataclass(frozen=True)
class _Anchor:
    per_device_train_batch_size: int
    max_pixels: int
    max_seq_length: int
    gradient_checkpointing: bool
    use_flow_tower: bool


def _get_anchor(stage: str) -> _Anchor | None:
    raw = _STAGE_ANCHORS.get(stage)
    if raw is None:
        return None
    return _Anchor(
        per_device_train_batch_size=int(raw["per_device_train_batch_size"]),
        max_pixels=int(raw["max_pixels"]),
        max_seq_length=int(raw["max_seq_length"]),
        gradient_checkpointing=bool(raw["gradient_checkpointing"]),
        use_flow_tower=bool(raw["use_flow_tower"]),
    )


def peak_vram_score(cfg: TrainConfig, anchor: _Anchor) -> float:
    """Relative peak VRAM estimate; 1.0 ≈ stable anchor on 8×80GB."""
    batch_ratio = cfg.per_device_train_batch_size / max(anchor.per_device_train_batch_size, 1)
    pixel_ratio = (cfg.max_pixels / max(anchor.max_pixels, 1)) ** _PIXEL_EXP
    seq_ratio = (cfg.max_seq_length / max(anchor.max_seq_length, 1)) ** _SEQ_EXP
    ckpt_ratio = _CKPT_FACTOR if cfg.gradient_checkpointing else 1.0
    ckpt_anchor = _CKPT_FACTOR if anchor.gradient_checkpointing else 1.0
    flow_ratio = _FLOW_TOWER_FACTOR if cfg.use_flow_tower else 1.0
    flow_anchor = _FLOW_TOWER_FACTOR if anchor.use_flow_tower else 1.0
    return batch_ratio * pixel_ratio * seq_ratio * (ckpt_ratio / ckpt_anchor) * (flow_ratio / flow_anchor)


def _per_gpu_global_target(cfg: TrainConfig) -> int:
    env_target = int(os.environ.get("TOWER_TARGET_GLOBAL_BATCH", "0") or 0)
    if env_target > 0:
        world = int(os.environ.get("WORLD_SIZE", os.environ.get("NUM_GPUS", "8")) or 8)
        return max(1, (env_target + world - 1) // world)
    return max(1, cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)


def _needs_eager_attn_guard(cfg: TrainConfig) -> bool:
    if cfg.use_flow_tower:
        return True
    return cfg.stage in ("understanding_warmup", "generation_pt")


def _vram_tune_enabled() -> bool:
    explicit = os.environ.get("TOWER_H800_VRAM_TUNE", "")
    if explicit == "0":
        return False
    if explicit == "1":
        return True
    if os.environ.get("H800_PROFILE") == "max":
        return True
    config_path = os.environ.get("CONFIG", "")
    return "_h800_max" in config_path


def apply_h800_vram_tune(cfg: TrainConfig) -> TrainConfig:
    """Clamp peak-VRAM knobs to ≤ stable; rebalance grad_accum for target global batch."""
    if not _vram_tune_enabled():
        return cfg

    anchor = _get_anchor(cfg.stage)
    if anchor is None:
        return cfg

    tuned = copy.copy(cfg)
    before = (
        tuned.per_device_train_batch_size,
        tuned.gradient_accumulation_steps,
        tuned.max_pixels,
        tuned.gradient_checkpointing,
    )
    target_per_gpu = _per_gpu_global_target(cfg)
    initial_score = peak_vram_score(tuned, anchor)

    if _needs_eager_attn_guard(tuned) and not tuned.gradient_checkpointing:
        tuned.gradient_checkpointing = True
        logger.warning(
            "[vram_tune] Forced gradient_checkpointing=true (%s uses block-causal eager attention)",
            tuned.stage,
        )

    # Shrink peak-VRAM knobs until within stable envelope.
    while peak_vram_score(tuned, anchor) > 1.0 + 1e-9:
        if tuned.per_device_train_batch_size > anchor.per_device_train_batch_size:
            tuned.per_device_train_batch_size -= 1
        elif tuned.max_pixels > anchor.max_pixels:
            tuned.max_pixels = max(anchor.max_pixels, int(tuned.max_pixels * 0.85))
        elif tuned.max_seq_length > anchor.max_seq_length:
            tuned.max_seq_length = max(anchor.max_seq_length, int(tuned.max_seq_length * 0.9))
        else:
            tuned.per_device_train_batch_size = max(1, tuned.per_device_train_batch_size - 1)
        if tuned.per_device_train_batch_size < 1:
            break

    micro = max(1, tuned.per_device_train_batch_size)
    tuned.gradient_accumulation_steps = max(1, (target_per_gpu + micro - 1) // micro)

    after_score = peak_vram_score(tuned, anchor)
    if (
        before != (
            tuned.per_device_train_batch_size,
            tuned.gradient_accumulation_steps,
            tuned.max_pixels,
            tuned.gradient_checkpointing,
        )
        or initial_score > 1.0
    ):
        world = int(os.environ.get("WORLD_SIZE", "8") or 8)
        global_batch = tuned.per_device_train_batch_size * tuned.gradient_accumulation_steps * world
        logger.warning(
            "[vram_tune] stage=%s peak_vram_score %.2f → %.2f (anchor=1.00, stable envelope). "
            "batch %s→%s accum %s→%s max_pixels %s→%s grad_ckpt %s→%s | "
            "global_batch≈%s (target_per_gpu=%s). "
            "Root cause: eager attn peak ~ batch×pixels^%.2f×seq^%.0f; accum does not lower peak.",
            tuned.stage,
            initial_score,
            after_score,
            before[0],
            tuned.per_device_train_batch_size,
            before[1],
            tuned.gradient_accumulation_steps,
            before[2],
            tuned.max_pixels,
            before[3],
            tuned.gradient_checkpointing,
            global_batch,
            target_per_gpu,
            _PIXEL_EXP,
            _SEQ_EXP,
        )

    return tuned
