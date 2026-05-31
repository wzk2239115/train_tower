import torch

from tower.unify.compat import apply_sensenova_transformers_compat, sdpa_block_attention_forward


def _fake_module():
    class _Mod:
        num_key_value_groups = 1
        training = True

    return _Mod()


def test_native_sdpa_flag_and_compat_skips_monkey_patch():
    apply_sensenova_transformers_compat()
    from sensenova_u1.models.neo_unify import modeling_qwen3 as sn_qwen3

    assert getattr(sn_qwen3, "_NATIVE_SDPA_BLOCK_ATTN", False)
    # Native path: eager_attention_forward stays the original (returns attn weights).
    mod = _fake_module()
    b, h, l, d = 1, 4, 32, 16
    q = torch.randn(b, h, l, d, dtype=torch.float32)
    k = torch.randn(b, h, l, d, dtype=torch.float32)
    v = torch.randn(b, h, l, d, dtype=torch.float32)
    mask = torch.zeros(1, 1, l, l)

    _, weights = sn_qwen3.eager_attention_forward(mod, q, k, v, mask, scaling=d**-0.5)
    assert weights is not None


def test_resolve_attention_interface_uses_sdpa_with_mask():
    from sensenova_u1.models.neo_unify.modeling_qwen3 import (
        resolve_attention_interface,
        sdpa_block_attention_forward,
    )

    fn = resolve_attention_interface(torch.zeros(1), "eager")
    assert fn is sdpa_block_attention_forward


def test_sdpa_matches_eager_numerically():
    from sensenova_u1.models.neo_unify.modeling_qwen3 import (
        create_block_causal_mask,
        eager_attention_forward,
    )

    mod = _fake_module()
    b, h, l, d = 1, 4, 64, 16
    scaling = d**-0.5
    torch.manual_seed(0)
    q = torch.randn(b, h, l, d, dtype=torch.bfloat16)
    k = torch.randn(b, h, l, d, dtype=torch.bfloat16)
    v = torch.randn(b, h, l, d, dtype=torch.bfloat16)
    index = torch.cat([torch.zeros(l // 2, dtype=torch.long), torch.ones(l - l // 2, dtype=torch.long)])
    mask = create_block_causal_mask(index)

    assert mask.dtype == torch.bfloat16

    out_eager, _ = eager_attention_forward(mod, q, k, v, mask, scaling=scaling)
    out_sdpa, weights = sdpa_block_attention_forward(mod, q, k, v, mask, scaling=scaling)

    assert weights is None
    assert out_eager.shape == out_sdpa.shape
    diff = (out_eager.float() - out_sdpa.float()).abs().max().item()
    assert diff < 0.05
