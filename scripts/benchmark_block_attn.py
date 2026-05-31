#!/usr/bin/env python3
"""Compare eager vs native SDPA block-causal attention (numerics + peak VRAM).

Usage:
  python scripts/benchmark_block_attn.py
  python scripts/benchmark_block_attn.py --seq-len 8192 --heads 12 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())


def _run_once(fn, mod, q, k, v, mask, scaling, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
    else:
        base = 0

    out, weights = fn(mod, q, k, v, mask, scaling=scaling)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - base
    else:
        peak = 0

    return out, weights, peak


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark block-causal attention paths")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    from tower.unify.compat import apply_sensenova_transformers_compat
    from tower.unify.backends import (
        create_block_causal_mask,
        get_eager_attention_forward_unpatched,
        sdpa_block_attention_forward,
    )

    apply_sensenova_transformers_compat()
    eager_attention_forward = get_eager_attention_forward_unpatched()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)

    class _Mod:
        num_key_value_groups = 1
        training = False

    mod = _Mod()
    b, h, l, d = args.batch, args.heads, args.seq_len, args.head_dim
    scaling = d**-0.5

    torch.manual_seed(0)
    q = torch.randn(b, h, l, d, device=device, dtype=dtype)
    k = torch.randn(b, h, l, d, device=device, dtype=dtype)
    v = torch.randn(b, h, l, d, device=device, dtype=dtype)

    # Sample index: three packed segments
    seg = l // 3
    index = torch.cat(
        [
            torch.zeros(seg, dtype=torch.long),
            torch.ones(seg, dtype=torch.long),
            torch.full((l - 2 * seg,), 2, dtype=torch.long),
        ],
        dim=0,
    ).to(device)
    mask = create_block_causal_mask(index)

    print(f"device={device} dtype={dtype} L={l} H={h} mask_dtype={mask.dtype} mask_bytes={mask.numel() * mask.element_size() / 1e6:.1f} MB")

    out_eager, w_eager, peak_eager = _run_once(
        eager_attention_forward, mod, q, k, v, mask, scaling, device
    )
    out_sdpa, w_sdpa, peak_sdpa = _run_once(
        sdpa_block_attention_forward, mod, q, k, v, mask, scaling, device
    )

    assert w_eager is not None
    assert w_sdpa is None
    assert out_eager.shape == out_sdpa.shape

    diff = (out_eager.float() - out_sdpa.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cos = _cosine(out_eager, out_sdpa)

    print(f"max_abs_diff={max_abs:.6e} mean_abs_diff={mean_abs:.6e} cosine={cos:.8f}")
    if device.type == "cuda":
        print(f"peak_vram_eager={peak_eager / 1e9:.3f} GiB peak_vram_sdpa={peak_sdpa / 1e9:.3f} GiB")
        if peak_sdpa > 0:
            print(f"vram_ratio_sdpa/eager={peak_sdpa / max(peak_eager, 1):.3f}")

    ok = max_abs < 0.05 and cos > 0.999
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
