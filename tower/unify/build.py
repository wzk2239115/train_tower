from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from tower.config import PROJECT_ROOT
from tower.paths import ensure_train_paths
from tower.train.config import TrainConfig
from tower.unify.compat import apply_sensenova_transformers_compat, fix_llm_config_compat


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

    logging.get_logger(__name__).info("LLM attention implementation: %s", resolved)


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def build_tokenizer(cfg: TrainConfig):
    ensure_train_paths()
    tok_path = _resolve_path(cfg.tokenizer_name_or_path)
    if not tok_path:
        raise ValueError("tokenizer_name_or_path is required")
    from neo.data.constants import ALL_SPECIAL_TOKEN_LIST

    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = cfg.max_seq_length
    tokenizer.add_tokens(ALL_SPECIAL_TOKEN_LIST, special_tokens=True)
    return tokenizer


def build_scratch_model(cfg: TrainConfig):
    """Instantiate SenseNova MoT NEOChatModel with random weights from local config."""
    apply_sensenova_transformers_compat()
    ensure_train_paths()
    from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

    config_path = _resolve_path(cfg.model_config_path)
    if not config_path:
        raise ValueError("model_config_path is required for scratch init")

    config = NEOChatConfig.from_pretrained(config_path)
    fix_llm_config_compat(config)
    model = NEOChatModel(config)
    _apply_attn_implementation(model, cfg.attn_implementation)
    return model


def build_checkpoint_model(cfg: TrainConfig):
    """Load a prior training checkpoint."""
    apply_sensenova_transformers_compat()
    ensure_train_paths()
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

    ckpt = _resolve_path(cfg.model_name_or_path)
    if not ckpt:
        raise ValueError("model_name_or_path is required for checkpoint init")
    dtype = "bfloat16" if cfg.bf16 else "float32"
    model = NEOChatModel.from_pretrained(ckpt, torch_dtype=dtype)
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

    from neo.data.constants import IMG_CONTEXT_TOKEN, IMG_START_TOKEN

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    if getattr(cfg, "audio_context_token_id", -1) >= 0:
        model.audio_context_token_id = int(cfg.audio_context_token_id)
    model.config.use_cache = False
    return model, tokenizer
