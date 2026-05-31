"""Block-causal fused SDPA for SenseNova Qwen3 (train_tower patches, not vendor)."""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

_SDPA_BLOCK_ATTN_LOGGED = False


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query head count (GQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_block_causal_mask(index: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Build ``(1, 1, L, L)`` block-wise causal additive mask (bf16 by default for training)."""
    length = index.size(0)
    idx_i = index.unsqueeze(1).expand(length, length)
    idx_j = index.unsqueeze(0).expand(length, length)

    arange = torch.arange(length, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

    finfo = torch.finfo(dtype)
    return torch.where(
        mask[None, None, :, :],
        torch.zeros((), device=index.device, dtype=dtype),
        torch.full((), finfo.min, device=index.device, dtype=dtype),
    )


def _sdpa_non_math_context(device: torch.device | None = None):
    """Prefer fused SDPA backends on CUDA; CPU / non-CUDA tensors use default SDPA."""
    if device is None or device.type != "cuda":
        return contextlib.nullcontext()

    backends_env = os.environ.get("TOWER_SDPA_BACKENDS", "efficient,cudnn").lower()
    name_map = {
        "efficient": "EFFICIENT_ATTENTION",
        "mem_efficient": "EFFICIENT_ATTENTION",
        "flash": "FLASH_ATTENTION",
        "cudnn": "CUDNN_ATTENTION",
    }
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        selected = []
        for part in backends_env.split(","):
            key = part.strip()
            if not key:
                continue
            attr = name_map.get(key)
            if attr is not None:
                selected.append(getattr(SDPBackend, attr))
        if selected:
            return sdpa_kernel(selected)
    except ImportError:
        pass
    if torch.cuda.is_available():
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False,
            )
        except Exception:
            pass
    return contextlib.nullcontext()


def _log_block_causal_sdpa_once() -> None:
    global _SDPA_BLOCK_ATTN_LOGGED
    if _SDPA_BLOCK_ATTN_LOGGED:
        return
    _SDPA_BLOCK_ATTN_LOGGED = True
    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) not in ("0", 0):
        return
    from transformers.utils import logging

    log = logging.get_logger(__name__)
    log.warning(
        "Block-causal attention: fused SDPA (TOWER_SDPA_BACKENDS=%s; MATH disabled on CUDA). "
        "Set TOWER_DISABLE_SDPA_BLOCK_ATTN=1 to revert to eager L×L materialization.",
        os.environ.get("TOWER_SDPA_BACKENDS", "efficient,cudnn"),
    )


def sdpa_block_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """Fused block-causal attention without materializing ``[B, H, L, L]`` weights."""
    _log_block_causal_sdpa_once()

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_mask = attention_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:, :, :, : key_states.shape[-2]]
        if attn_mask.dtype != query.dtype:
            attn_mask = attn_mask.to(dtype=query.dtype)

    dropout_p = float(dropout) if module.training else 0.0
    with _sdpa_non_math_context(query.device):
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


def resolve_attention_interface(
    attention_mask: Optional[torch.Tensor],
    attn_implementation: str,
    *,
    eager_attention_forward: Callable,
) -> Callable:
    """Block mask → fused SDPA; otherwise transformers attention dispatch."""
    if attention_mask is not None:
        if os.environ.get("TOWER_DISABLE_SDPA_BLOCK_ATTN", "0") == "1":
            return eager_attention_forward
        return sdpa_block_attention_forward
    if attn_implementation != "eager":
        return ALL_ATTENTION_FUNCTIONS[attn_implementation]
    return eager_attention_forward
