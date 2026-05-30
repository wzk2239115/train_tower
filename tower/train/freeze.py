from __future__ import annotations

from transformers.utils import logging

from tower.unify.tower_config import load_tower_config

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


def _llm_layer_index(name: str) -> int | None:
    marker = "language_model.model.layers."
    if marker not in name:
        return None
    idx_str = name.split(marker, 1)[1].split(".", 1)[0]
    try:
        return int(idx_str)
    except ValueError:
        return None


def _world_pt_shallow_trainable(name: str, shallow_layers: int) -> bool:
    """Scheme B: train und vision, shared embed, and LLM layers [0, shallow_layers)."""
    if _is_mot_gen(name) or _is_fm_module(name):
        return False
    if _is_und_vision(name) or _is_shared(name):
        return True
    if _is_und_llm(name):
        layer_idx = _llm_layer_index(name)
        return layer_idx is not None and layer_idx < shallow_layers
    return False


def apply_stage_freeze(model, stage: str) -> None:
    """Freeze parameter groups per pretrain stage (replaces NEO train_buffer)."""
    stage = stage.lower()
    tower_cfg = load_tower_config()
    shallow_layers = tower_cfg.shallow_train_layers(stage)

    for name, param in model.named_parameters():
        trainable = True
        if stage == "understanding_warmup":
            trainable = _is_und_vision(name) or _is_und_llm(name) or _is_shared(name)
        elif stage == "generation_pt":
            trainable = _is_fm_module(name) or _is_mot_gen(name) or _is_shared(name)
        elif stage == "world_pt" and shallow_layers is not None:
            trainable = _world_pt_shallow_trainable(name, shallow_layers)
        elif stage in ("unified_mt", "unified_sft"):
            trainable = True
        else:
            trainable = True
        param.requires_grad = trainable

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Stage freeze '%s': trainable %.2fM / %.2fM params"
        + (f" (shallow_layers={shallow_layers})" if stage == "world_pt" and shallow_layers else ""),
        stage,
        n_train / 1e6,
        n_total / 1e6,
    )


def apply_tower_exit_freeze(tower_model, stage: str) -> None:
    """Freeze tower exit modules per note/tower.yml stage_freeze."""
    if not hasattr(tower_model, "tower_exits"):
        return
    tower_cfg = load_tower_config()
    stage = stage.lower()
    spec = tower_cfg.stage_freeze.get(stage)
    if not spec:
        return

    train_names = set(spec.get("train") or [])
    for name, module in tower_model.tower_exits.items():
        trainable = name in train_names
        for param in module.parameters():
            param.requires_grad = trainable

    logger.info(
        "Tower exit freeze '%s': train=%s",
        stage,
        sorted(train_names),
    )
