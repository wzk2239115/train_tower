import torch

from tower.unify.compat import apply_sensenova_transformers_compat, sdpa_block_attention_forward


def test_eager_forward_patched_to_sdpa_when_mask_present():
    apply_sensenova_transformers_compat()
    from sensenova_u1.models.neo_unify import modeling_qwen3 as sn_qwen3

    class _Mod:
        num_key_value_groups = 1
        training = True

    mod = _Mod()
    b, h, l, d = 1, 4, 32, 16
    q = torch.randn(b, h, l, d, dtype=torch.float32)
    k = torch.randn(b, h, l, d, dtype=torch.float32)
    v = torch.randn(b, h, l, d, dtype=torch.float32)
    mask = torch.zeros(1, 1, l, l)

    out, weights = sn_qwen3.eager_attention_forward(mod, q, k, v, mask, scaling=d**-0.5)
    assert weights is None
    assert out.shape == (b, l, h, d)

    out2, _ = sdpa_block_attention_forward(mod, q, k, v, mask, scaling=d**-0.5)
    assert out2.shape == (b, l, h, d)
