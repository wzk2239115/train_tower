"""Facade for ``tower.models.neo_unify`` — sole SenseNova model import boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tower.unify.attention import create_block_causal_mask, sdpa_block_attention_forward

if TYPE_CHECKING:
    import types

    from tower.models.neo_unify.configuration_neo_chat import NEOChatConfig
    from tower.models.neo_unify.modeling_fm_modules import TimestepEmbedder
    from tower.models.neo_unify.modeling_neo_chat import NEOChatModel
    from tower.models.neo_unify.modeling_neo_vit import build_abs_positions_from_grid_hw


def import_neo_chat_config():
    from tower.models.neo_unify.configuration_neo_chat import NEOChatConfig

    return NEOChatConfig


def import_neo_chat_model():
    from tower.models.neo_unify.modeling_neo_chat import NEOChatModel

    return NEOChatModel


def import_timestep_embedder():
    from tower.models.neo_unify.modeling_fm_modules import TimestepEmbedder

    return TimestepEmbedder


def import_build_abs_positions_from_grid_hw():
    from tower.models.neo_unify.modeling_neo_vit import build_abs_positions_from_grid_hw

    return build_abs_positions_from_grid_hw


def import_modeling_qwen3() -> types.ModuleType:
    from tower.models.neo_unify import modeling_qwen3

    return modeling_qwen3


def import_modeling_qwen3_moe() -> types.ModuleType:
    from tower.models.neo_unify import modeling_qwen3_moe

    return modeling_qwen3_moe


def import_modeling_neo_chat() -> types.ModuleType:
    from tower.models.neo_unify import modeling_neo_chat

    return modeling_neo_chat


def get_eager_attention_forward_unpatched():
    """Original eager attention (before block-causal SDPA wrapping)."""
    sn_qwen3 = import_modeling_qwen3()
    return getattr(sn_qwen3, "eager_attention_forward_unpatched", sn_qwen3.eager_attention_forward)


def get_resolve_attention_interface():
    sn_qwen3 = import_modeling_qwen3()
    return sn_qwen3.resolve_attention_interface


__all__ = [
    "create_block_causal_mask",
    "get_eager_attention_forward_unpatched",
    "get_resolve_attention_interface",
    "import_build_abs_positions_from_grid_hw",
    "import_modeling_neo_chat",
    "import_modeling_qwen3",
    "import_modeling_qwen3_moe",
    "import_neo_chat_config",
    "import_neo_chat_model",
    "import_timestep_embedder",
    "sdpa_block_attention_forward",
]
