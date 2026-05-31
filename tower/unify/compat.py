from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tower.paths import ensure_train_paths

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
    """Fused block-causal attention without materializing ``[B, H, L, L]`` weights.

    SenseNova defaults to ``eager_attention_forward`` whenever a block mask is present.
    That path allocates the full attention matrix in fp32 for softmax — ~3 GiB per layer
    at L=8192 — which is the dominant OOM source on 80GB GPUs, independent of grad_accum.
    """
    from sensenova_u1.models.neo_unify.modeling_qwen3 import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_mask = attention_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:, :, :, : key_states.shape[-2]]
        if attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(dtype=query.dtype)

    dropout_p = float(dropout) if module.training else 0.0
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        scale=scaling,
        is_causal=False,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _patch_block_causal_attention() -> None:
    """Route block-causal masks through fused SDPA instead of eager matmul+softmax."""
    global _SDPA_BLOCK_ATTN_PATCHED
    if _SDPA_BLOCK_ATTN_PATCHED or os.environ.get("TOWER_DISABLE_SDPA_BLOCK_ATTN", "0") == "1":
        return

    from sensenova_u1.models.neo_unify import modeling_qwen3 as sn_qwen3
    from transformers.utils import logging

    log = logging.get_logger(__name__)
    _orig = sn_qwen3.eager_attention_forward

    def _dispatch(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if attention_mask is not None:
            return sdpa_block_attention_forward(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
            )
        return _orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)

    sn_qwen3.eager_attention_forward = _dispatch
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
    ensure_train_paths()

    from sensenova_u1.models.neo_unify import modeling_qwen3 as sn_qwen3

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
        from sensenova_u1.models.neo_unify import modeling_qwen3_moe as sn_moe

        tied = getattr(sn_moe.Qwen3MoeForCausalLM, "_tied_weights_keys", None)
        if isinstance(tied, list):
            sn_moe.Qwen3MoeForCausalLM._tied_weights_keys = {
                "lm_head.weight": "model.embed_tokens.weight"
            }
    except ImportError:
        pass

    # transformers 5.x may access ``all_tied_weights_keys`` on top-level model
    # during loading report finalization. SenseNova's NEOChatModel does not
    # define it, so provide a harmless fallback to avoid AttributeError.
    try:
        from sensenova_u1.models.neo_unify import modeling_neo_chat as sn_chat

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


def _unwrap_singleton(value):
    """Unwrap nested singleton list/tuple values (e.g. [[0.5]] -> 0.5)."""
    cur = value
    while isinstance(cur, (list, tuple)) and len(cur) == 1:
        cur = cur[0]
    return cur


def fix_vision_config_compat(config) -> None:
    """Normalize SenseNova vision/downsample fields for legacy checkpoints."""
    vc = getattr(config, "vision_config", None)
    if vc is None:
        return

    ds = _unwrap_singleton(getattr(vc, "downsample_ratio", 0.5))
    llm_h = _unwrap_singleton(getattr(vc, "llm_hidden_size", 0))

    # modeling_neo_vit indexes these fields with [0]
    vc.downsample_ratio = [float(ds)]
    vc.llm_hidden_size = [int(llm_h)]

    # modeling_neo_chat expects a scalar downsample_ratio on top-level config
    if hasattr(config, "downsample_ratio"):
        config.downsample_ratio = float(_unwrap_singleton(getattr(config, "downsample_ratio")))
