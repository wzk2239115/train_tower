"""Migrated upstream import boundaries (NEO data, SenseNova neo_unify).

train_tower must import ``tower.neo.*`` and ``tower.models.neo_unify.*`` only through
this package.
"""

from tower.unify.backends.neo import (
    import_data_constants,
    import_flattened_data_collator,
    import_lazy_supervised_dataset,
    import_neo_data,
    import_train_arguments,
)
from tower.unify.backends.sensenova import (
    create_block_causal_mask,
    get_eager_attention_forward_unpatched,
    get_resolve_attention_interface,
    import_build_abs_positions_from_grid_hw,
    import_modeling_neo_chat,
    import_modeling_qwen3,
    import_modeling_qwen3_moe,
    import_neo_chat_config,
    import_neo_chat_model,
    import_timestep_embedder,
    sdpa_block_attention_forward,
)

__all__ = [
    "create_block_causal_mask",
    "get_eager_attention_forward_unpatched",
    "get_resolve_attention_interface",
    "import_build_abs_positions_from_grid_hw",
    "import_data_constants",
    "import_flattened_data_collator",
    "import_lazy_supervised_dataset",
    "import_modeling_neo_chat",
    "import_modeling_qwen3",
    "import_modeling_qwen3_moe",
    "import_neo_chat_config",
    "import_neo_chat_model",
    "import_neo_data",
    "import_timestep_embedder",
    "import_train_arguments",
    "sdpa_block_attention_forward",
]
