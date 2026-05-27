from __future__ import annotations

from tower.paths import ensure_train_paths

_APPLIED = False


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
