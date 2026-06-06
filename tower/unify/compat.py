from __future__ import annotations

import os
import inspect
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tower.unify import attention
from tower.unify.backends.sensenova import (
    import_modeling_neo_chat,
    import_modeling_qwen3,
    import_modeling_qwen3_moe,
)

_APPLIED = False
_SDPA_BLOCK_ATTN_PATCHED = False


def sdpa_block_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Fused block-causal SDPA (train_tower implementation in ``tower.unify.attention``)."""
    return attention.sdpa_block_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=dropout,
        **kwargs,
    )


def _call_attention_base(
    base_fn,
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout,
    kwargs,
):
    """Call transformers attention functions across 5.x signature drift."""
    try:
        sig = inspect.signature(base_fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        call_kwargs = dict(kwargs)
        if "scaling" in params or has_var_kw:
            call_kwargs["scaling"] = scaling
        if "dropout" in params or has_var_kw:
            call_kwargs["dropout"] = dropout
        return base_fn(
            module,
            query,
            key,
            value,
            attention_mask,
            **call_kwargs,
        )
    except (TypeError, ValueError):
        # Older vendored implementations used positional scaling before
        # keyword dropout. Keep this fallback for local/offline images.
        return base_fn(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )


def _patch_block_causal_attention() -> None:
    """Route block-causal masks through fused SDPA on all Qwen3 attention dispatch paths."""
    global _SDPA_BLOCK_ATTN_PATCHED
    if _SDPA_BLOCK_ATTN_PATCHED or os.environ.get("TOWER_DISABLE_SDPA_BLOCK_ATTN", "0") == "1":
        return

    from transformers.utils import logging

    sn_qwen3 = import_modeling_qwen3()
    log = logging.get_logger(__name__)

    if getattr(sn_qwen3, "_NATIVE_SDPA_BLOCK_ATTN", False):
        _SDPA_BLOCK_ATTN_PATCHED = True
        log.warning(
            "Block-causal attention: native SDPA already patched "
            "(compat monkey-patch skipped)"
        )
        return

    orig_eager = sn_qwen3.eager_attention_forward
    orig_all = {name: fn for name, fn in ALL_ATTENTION_FUNCTIONS.items()}

    def _wrap(base_fn):
        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if attention_mask is not None:
                return attention.sdpa_block_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling,
                    dropout=dropout,
                    **kwargs,
                )
            return _call_attention_base(
                base_fn,
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout,
                kwargs,
            )

        return wrapped

    sn_qwen3.eager_attention_forward_unpatched = orig_eager
    sn_qwen3.eager_attention_forward = _wrap(orig_eager)
    for name, fn in orig_all.items():
        ALL_ATTENTION_FUNCTIONS[name] = _wrap(fn)

    def _resolve(attention_mask, attn_implementation):
        return attention.resolve_attention_interface(
            attention_mask,
            attn_implementation,
            eager_attention_forward=orig_eager,
        )

    sn_qwen3.sdpa_block_attention_forward = attention.sdpa_block_attention_forward
    sn_qwen3.resolve_attention_interface = _resolve
    sn_qwen3._NATIVE_SDPA_BLOCK_ATTN = True
    _SDPA_BLOCK_ATTN_PATCHED = True
    log.warning(
        "Patched block-causal attention: eager L×L materialization → fused SDPA "
        "(set TOWER_DISABLE_SDPA_BLOCK_ATTN=1 to revert)"
    )


def apply_sensenova_transformers_compat() -> None:
    """Patch SenseNova neo_unify for transformers 5.x (omni-jepa uses 5.9)."""
    global _APPLIED
    if _APPLIED:
        return

    sn_qwen3 = import_modeling_qwen3()

    cls = sn_qwen3.Qwen3RotaryEmbedding
    if not hasattr(cls, "compute_default_rope_parameters"):

        @classmethod
        def compute_default_rope_parameters(cls_, config=None, device=None, seq_len=None):
            inv_freq, attention_factor = sn_qwen3._compute_default_rope_parameters(config, device)
            return inv_freq, attention_factor

        cls.compute_default_rope_parameters = compute_default_rope_parameters

    for model_cls in (sn_qwen3.Qwen3ForCausalLM,):
        tied = getattr(model_cls, "_tied_weights_keys", None)
        if isinstance(tied, list):
            model_cls._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    try:
        sn_moe = import_modeling_qwen3_moe()
        tied = getattr(sn_moe.Qwen3MoeForCausalLM, "_tied_weights_keys", None)
        if isinstance(tied, list):
            sn_moe.Qwen3MoeForCausalLM._tied_weights_keys = {
                "lm_head.weight": "model.embed_tokens.weight"
            }
    except ImportError:
        pass

    try:
        sn_chat = import_modeling_neo_chat()
        if not hasattr(sn_chat.NEOChatModel, "all_tied_weights_keys"):
            sn_chat.NEOChatModel.all_tied_weights_keys = {}
    except ImportError:
        pass

    _patch_block_causal_attention()
    _APPLIED = True


def fix_llm_config_compat(config) -> None:
    """Bridge newer transformers Qwen3Config (rope_parameters) with SenseNova code."""
    apply_sensenova_transformers_compat()
    lc = config.llm_config
    rp = getattr(lc, "rope_parameters", None) or {}
    lc.rope_theta = float(rp.get("rope_theta", 5_000_000.0))
    if not getattr(lc, "layer_types", None):
        use_swa = bool(getattr(lc, "use_sliding_window", False))
        max_window = int(getattr(lc, "max_window_layers", 0) or 0)
        lc.layer_types = [
            "sliding_attention" if (use_swa and i >= max_window) else "full_attention"
            for i in range(lc.num_hidden_layers)
        ]


def fix_vision_config_compat(config) -> None:
    """Normalize SenseNova vision/downsample fields for legacy checkpoints."""
    vc = getattr(config, "vision_config", None)
    if vc is None:
        return

    ds = _unwrap_singleton(getattr(vc, "downsample_ratio", 0.5))
    llm_h = _unwrap_singleton(getattr(vc, "llm_hidden_size", 0))

    vc.downsample_ratio = [float(ds)]
    vc.llm_hidden_size = [int(llm_h)]

    if hasattr(config, "downsample_ratio"):
        config.downsample_ratio = float(_unwrap_singleton(getattr(config, "downsample_ratio")))


def _unwrap_singleton(value):
    """Unwrap nested singleton list/tuple values (e.g. [[0.5]] -> 0.5)."""
    cur = value
    while isinstance(cur, (list, tuple)) and len(cur) == 1:
        cur = cur[0]
    return cur
