"""H800 VRAM auto-tune (secondary guard; primary fix is fused SDPA in tower.unify.compat).

Essential OOM root cause (UW / Gen PT)
--------------------------------------
* ``data_flatten`` packs ``per_device_train_batch_size`` **samples into one sequence** (length → ``max_seq_length``).
* ``create_block_causal_mask(L)`` + legacy **eager** attention materializes ``[B, H, L, L]`` fp32 weights (~3 GiB at L=8192 **per layer**).
* ``gradient_accumulation_steps`` does **not** lower that peak (only increases global batch over time).

This module caps pack-count / pixels / seq when ``TOWER_H800_VRAM_TUNE=1``. Throughput scaling should prefer
``gradient_accumulation`` after compat SDPA patch is active.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from transformers.utils import logging

from tower.train.config import TrainConfig

logger = logging.get_logger(__name__)

# Stable ``*_h800_resume.yaml`` anchors (known-good on 8×80GB with fused SDPA).
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

_PIXEL_EXP = 0.85
_SEQ_EXP = 2.0
_CKPT_FACTOR = 0.55
_FLOW_TOWER_FACTOR = 1.25
# Headroom below anchor (stable was already near limit on some nodes before SDPA patch).
_MAX_SCORE = float(os.environ.get("TOWER_VRAM_MAX_SCORE", "0.95"))


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
    """Relative peak VRAM; 1.0 ≈ stable anchor (pack-count × pixels × seq²)."""
    pack_ratio = cfg.per_device_train_batch_size / max(anchor.per_device_train_batch_size, 1)
    pixel_ratio = (cfg.max_pixels / max(anchor.max_pixels, 1)) ** _PIXEL_EXP
    seq_ratio = (cfg.max_seq_length / max(anchor.max_seq_length, 1)) ** _SEQ_EXP
    ckpt_ratio = _CKPT_FACTOR if cfg.gradient_checkpointing else 1.0
    ckpt_anchor = _CKPT_FACTOR if anchor.gradient_checkpointing else 1.0
    flow_ratio = _FLOW_TOWER_FACTOR if cfg.use_flow_tower else 1.0
    flow_anchor = _FLOW_TOWER_FACTOR if anchor.use_flow_tower else 1.0
    return pack_ratio * pixel_ratio * seq_ratio * (ckpt_ratio / ckpt_anchor) * (flow_ratio / flow_anchor)


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
    """Clamp pack-count / pixels / seq; rebalance grad_accum for target global batch."""
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
        tuned.max_seq_length,
        tuned.gradient_checkpointing,
    )
    target_per_gpu = _per_gpu_global_target(cfg)
    initial_score = peak_vram_score(tuned, anchor)

    if _needs_eager_attn_guard(tuned) and not tuned.gradient_checkpointing:
        tuned.gradient_checkpointing = True
        logger.warning(
            "[vram_tune] Forced gradient_checkpointing=true (%s)",
            tuned.stage,
        )

    limit = _MAX_SCORE
    while peak_vram_score(tuned, anchor) > limit + 1e-9:
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
    if before != (
        tuned.per_device_train_batch_size,
        tuned.gradient_accumulation_steps,
        tuned.max_pixels,
        tuned.max_seq_length,
        tuned.gradient_checkpointing,
    ) or initial_score > limit:
        world = int(os.environ.get("WORLD_SIZE", "8") or 8)
        global_batch = tuned.per_device_train_batch_size * tuned.gradient_accumulation_steps * world
        logger.warning(
            "[vram_tune] stage=%s score %.2f→%.2f (limit %.2f). "
            "pack %s→%s accum %s→%s pixels %s→%s seq %s→%s | global≈%s. "
            "Note: per_device_train_batch_size = flattened pack count; OOM fix is SDPA block attn in compat.",
            tuned.stage,
            initial_score,
            after_score,
            limit,
            before[0],
            tuned.per_device_train_batch_size,
            before[1],
            tuned.gradient_accumulation_steps,
            before[2],
            tuned.max_pixels,
            before[3],
            tuned.max_seq_length,
            before[4],
            tuned.gradient_checkpointing,
            global_batch,
        )

    return tuned
