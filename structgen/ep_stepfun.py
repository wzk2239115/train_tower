"""Expert-parallel (EP) Stepfun: 8 GPUs each hold 288/8 = 36 experts, all busy.

Replaces Step3p7MoEMLP.forward with an EP version: each rank loops only its 36
experts and all-reduces the result. Non-expert parts (attention, router, shared
expert, embed) are REPLICATED on every rank (small ~7B of the 196B) and run
redundantly — cheap, and keeps all ranks in lockstep so the geom-token
inputs_embeds + multi-layer feature hooks work unchanged.

Memory/rank: ~51GB (my experts) + ~14GB (replicated non-expert) ≈ 65GB < 80GB.
This makes the frozen Stepfun forward run on 8 GPUs in parallel (vs pipeline's
1/8 util) → fast enough to validate "frozen Stepfun surfaces 3D structure from
geometry tokens" and to train the residual decoder after it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

N_EXPERTS = 288


def _ep_moe_forward(self, hidden_states):
    """EP replacement for Step3p7MoEMLP.forward. Each rank computes only its
    expert slice; all_reduce sums contributions across ranks."""
    B, S, H = hidden_states.shape
    x = hidden_states.view(-1, H)
    N = x.shape[0]
    # router (replicated, identical on all ranks)
    if getattr(self, "need_fp32_gate", False):
        logits = torch.matmul(x.float(), self.gate.weight.t().float())
    else:
        logits = self.gate(x)
    if self.custom_routing_function is not None:
        rw, sel = self.custom_routing_function(logits, self.top_k, renormalize=True)
    else:
        rw = F.softmax(logits, dim=1, dtype=torch.float)
        rw, sel = torch.topk(rw, self.top_k, dim=-1)
    rw = rw * self.routed_scaling_factor

    final = torch.zeros(N, H, dtype=hidden_states.dtype, device=hidden_states.device)
    mask = F.one_hot(sel, self.num_experts_global).permute(2, 1, 0)  # (288, top_k, N)

    rank = dist.get_rank()
    world = dist.get_world_size()
    per = self.num_experts_global // world
    lo, hi = rank * per, (rank + 1) * per
    for ei in range(lo, hi):                       # only my 36 experts
        idx, top_x = torch.where(mask[ei])
        if top_x.numel() == 0:
            continue
        cur = x[top_x]
        out = self.get_expert_output(cur, ei - lo) * rw[top_x, idx, None]
        final.index_add_(0, top_x, out.to(hidden_states.dtype))
    dist.all_reduce(final, op=dist.ReduceOp.SUM)
    return final.reshape(B, S, H)


def _patch_moe_ep(model):
    """Shard each Step3p7MoEMLP's expert weights to this rank's slice and swap
    in the EP forward. Call AFTER dist.init_process_group."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    per = N_EXPERTS // world
    lo, hi = rank * per, (rank + 1) * per
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Step3p7MoEMLP":
            continue
        # remember the global expert count (for routing/mask), then slice weights
        mod.num_experts_global = N_EXPERTS
        for proj in (mod.up_proj, mod.gate_proj, mod.down_proj):
            w = proj.weight.data                      # (288, out, in)
            proj.weight = nn.Parameter(w[lo:hi].clone().contiguous(), requires_grad=False)
            proj.num_experts = per
        # free router_bias stays (288); gate stays (288); only experts sharded
        mod.forward = _ep_moe_forward.__get__(mod, type(mod))
        n += 1
    return n


def load_stepfun_ep(path: str, max_cpu: str = "1200GiB"):
    """Each rank loads the model to CPU (low_cpu_mem_usage), shards experts to
    its slice, moves (replicated non-expert + my experts) to cuda:local_rank,
    frees the rest. Needs ~400GB CPU RAM per rank during load."""
    import transformers as _tf
    from structgen.model.backbone import _patch_causal_mask_compat, _patch_rope_batched_compat

    rank = dist.get_rank()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    model = _tf.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cpu", low_cpu_mem_usage=True)
    _patch_causal_mask_compat()
    _patch_rope_batched_compat()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = _patch_moe_ep(model)
    if rank == 0:
        print(f"[ep] patched {n} MoE layers → EP (world={dist.get_world_size()}, "
              f"{N_EXPERTS // dist.get_world_size()} experts/rank)")

    # move kept params to this rank's GPU; free CPU
    moved = 0
    for p in model.parameters():
        if p.numel() == 0:
            continue
        p.data = p.data.to(dev, non_blocking=True)
        moved += p.numel()
    if rank == 0:
        print(f"[ep] rank0 loaded + moved {moved/1e9:.1f}B params to GPU")
    torch.cuda.empty_cache()
    return model


def ep_ready() -> bool:
    return dist.is_available() and dist.is_initialized()
