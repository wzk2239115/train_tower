from __future__ import annotations

from transformers.utils import logging

logger = logging.get_logger(__name__)


def _is_mot_gen(name: str) -> bool:
    return "_mot_gen" in name


def _is_fm_module(name: str) -> bool:
    return name.startswith("fm_modules.")


def _is_und_vision(name: str) -> bool:
    return name.startswith("vision_model.")


def _is_shared(name: str) -> bool:
    return "embed_tokens" in name or name.endswith("lm_head.weight")


def _is_und_llm(name: str) -> bool:
    return name.startswith("language_model.") and not _is_mot_gen(name) and not _is_shared(name)


def apply_stage_freeze(model, stage: str) -> None:
    """Freeze parameter groups per pretrain stage (replaces NEO train_buffer)."""
    stage = stage.lower()
    for name, param in model.named_parameters():
        trainable = True
        if stage == "understanding_warmup":
            trainable = _is_und_vision(name) or _is_und_llm(name) or _is_shared(name)
        elif stage == "generation_pt":
            trainable = _is_fm_module(name) or _is_mot_gen(name) or _is_shared(name)
        elif stage in ("unified_mt", "unified_sft"):
            trainable = True
        else:
            trainable = True
        param.requires_grad = trainable

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Stage freeze '%s': trainable %.2fM / %.2fM params",
        stage,
        n_train / 1e6,
        n_total / 1e6,
    )
