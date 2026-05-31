from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoTokenizer

from tower.config import PROJECT_ROOT
from tower.train.config import TrainConfig
from tower.train.size_preset import resolve_model_config_dict
from tower.unify.compat import (
    apply_sensenova_transformers_compat,
    fix_llm_config_compat,
    fix_vision_config_compat,
)


def _resolve_attn_implementation(requested: str) -> str:
    impl = (requested or "sdpa").strip().lower()
    if impl in ("flash_attention_2", "flash_attn", "fa2"):
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            from transformers.utils import logging

            logging.get_logger(__name__).warning(
                "flash_attn not installed; falling back to sdpa attention"
            )
            return "sdpa"
        return "flash_attention_2"
    if impl in ("sdpa", "eager"):
        return impl
    return "sdpa"


def _apply_attn_implementation(model, impl: str) -> None:
    resolved = _resolve_attn_implementation(impl)
    model.config.llm_config._attn_implementation = resolved
    from transformers.utils import logging

    log = logging.get_logger(__name__)
    log.info("LLM attention implementation: %s", resolved)
    log.info(
        "Block-causal training uses fused SDPA when a mask is set (compat patch); "
        "flash/sdpa hub path applies only when attention_mask is None"
    )


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def build_tokenizer(cfg: TrainConfig):
    tok_path = _resolve_path(cfg.tokenizer_name_or_path)
    if not tok_path:
        raise ValueError("tokenizer_name_or_path is required")
    from tower.unify.backends.neo import all_special_token_list

    def _load(path: str):
        return AutoTokenizer.from_pretrained(
            path,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False,
        )

    try:
        tokenizer = _load(tok_path)
    except Exception as exc:
        # Some stage outputs may miss complete tokenizer assets; fall back to
        # a known good tokenizer directory so training can continue.
        fallback = _resolve_path(
            os.environ.get("TOWER_TOKENIZER_FALLBACK", "configs/tokenizer/qwen3")
        )
        if fallback and fallback != tok_path and Path(fallback).is_dir():
            from transformers.utils import logging

            log = logging.get_logger(__name__)
            log.warning(
                "Tokenizer load failed at %s (%s). Falling back to %s",
                tok_path,
                exc,
                fallback,
            )
            tokenizer = _load(fallback)
        else:
            raise
    tokenizer.model_max_length = cfg.max_seq_length
    tokenizer.add_tokens(all_special_token_list(), special_tokens=True)
    return tokenizer


def build_scratch_model(cfg: TrainConfig):
    """Instantiate SenseNova MoT NEOChatModel with random weights from local config."""
    apply_sensenova_transformers_compat()
    from tower.unify.backends import import_neo_chat_config, import_neo_chat_model

    NEOChatConfig = import_neo_chat_config()
    NEOChatModel = import_neo_chat_model()

    config_path = _resolve_path(cfg.model_config_path)
    if not config_path:
        raise ValueError("model_config_path is required for scratch init")

    merged = resolve_model_config_dict(cfg)
    config = NEOChatConfig.from_dict(merged)
    fix_llm_config_compat(config)
    fix_vision_config_compat(config)
    model = NEOChatModel(config)
    _apply_attn_implementation(model, cfg.attn_implementation)
    return model


def build_checkpoint_model(cfg: TrainConfig):
    """Load a prior training checkpoint."""
    apply_sensenova_transformers_compat()
    from tower.unify.backends import import_neo_chat_config, import_neo_chat_model

    NEOChatConfig = import_neo_chat_config()
    NEOChatModel = import_neo_chat_model()

    ckpt = _resolve_path(cfg.model_name_or_path)
    if not ckpt:
        raise ValueError("model_name_or_path is required for checkpoint init")
    dtype = "bfloat16" if cfg.bf16 else "float32"
    config = NEOChatConfig.from_pretrained(ckpt)
    fix_llm_config_compat(config)
    fix_vision_config_compat(config)
    model = NEOChatModel.from_pretrained(
        ckpt,
        config=config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    _apply_attn_implementation(model, cfg.attn_implementation)
    return model


def build_model_and_tokenizer(cfg: TrainConfig):
    apply_sensenova_transformers_compat()
    tokenizer = build_tokenizer(cfg)
    if cfg.init_mode == "scratch" and cfg.weight_init == "random":
        model = build_scratch_model(cfg)
    elif cfg.model_name_or_path:
        model = build_checkpoint_model(cfg)
    else:
        raise ValueError(f"Unsupported init: init_mode={cfg.init_mode}, weight_init={cfg.weight_init}")

    from tower.unify.backends.neo import img_token_ids

    model.img_context_token_id, model.img_start_token_id = img_token_ids(tokenizer)
    if getattr(cfg, "audio_context_token_id", -1) >= 0:
        model.audio_context_token_id = int(cfg.audio_context_token_id)
    else:
        c = __import__("tower.neo.data.constants", fromlist=["AUDIO_CONTEXT_TOKEN"])
        model.audio_context_token_id = tokenizer.convert_tokens_to_ids(c.AUDIO_CONTEXT_TOKEN)
    if getattr(cfg, "audio_start_token_id", None) is None:
        c = __import__("tower.neo.data.constants", fromlist=["AUDIO_START_TOKEN"])
        model.audio_start_token_id = tokenizer.convert_tokens_to_ids(c.AUDIO_START_TOKEN)
    if getattr(cfg, "video_context_token_id", -1) >= 0:
        model.video_context_token_id = int(cfg.video_context_token_id)
    else:
        c = __import__("tower.neo.data.constants", fromlist=["VIDEO_CONTEXT_TOKEN"])
        model.video_context_token_id = tokenizer.convert_tokens_to_ids(c.VIDEO_CONTEXT_TOKEN)
    if getattr(cfg, "video_start_token_id", None) is None:
        c = __import__("tower.neo.data.constants", fromlist=["VIDEO_START_TOKEN"])
        model.video_start_token_id = tokenizer.convert_tokens_to_ids(c.VIDEO_START_TOKEN)
    model.config.use_cache = False
    return model, tokenizer
