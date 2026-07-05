"""Stepfun-as-denoiser: the 150B model IS the flow-matching backbone.

The geometry latent (from the VAE) is injected as continuous "soft tokens" via
``inputs_embeds``, concatenated after the text embeddings. The Stepfun forward
runs EVERY flow step and its pretrained attention/MLP computation processes the
geometry jointly with the text — we read the geometry-token positions' last-layer
hidden states and a small head maps them back to the latent (predicted x0).

Trainable: geom-token projection (latent_ch→hidden) + output head (hidden→latent_ch).
Stepfun weights frozen (150B full fine-tune is impractical)
optional LoRA later.

Both blockers are resolved on the compute box (verified):
  * inputs_embeds forward works (TEST 1)
  * batched forward works after the RoPE shape patch (TEST 2/3)
"""
from __future__ import annotations

import torch
import torch.nn as nn



def _build_stepfun(path: str, max_mem: dict):
    """Load Stepfun across GPUs, apply mask + RoPE compat patches, freeze."""
    import transformers as _tf
    from structgen.model.backbone import _patch_causal_mask_compat, _patch_rope_batched_compat

    model = _tf.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", max_memory=max_mem, low_cpu_mem_usage=True)
    _patch_causal_mask_compat()
    _patch_rope_batched_compat()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _find_llm_layers(model):
    """The LLM transformer-layer ModuleList (skip vision). Returns the list."""
    import torch.nn as _nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, _nn.ModuleList) and len(mod) >= 10:
            low = name.lower()
            if any(k in low for k in ("vision", "image", "visual", "encoder")):
                continue
            score = len(mod) + (200 if "language" in low else (150 if "text" in low else 0))
            if best is None or score > best[0]:
                best = (score, name, mod)
    return best[2]  # type: ignore[index]


def _find_last_layer(model):
    return _find_llm_layers(model)[-1]


def _find_embed(model):
    import torch.nn as _nn
    for attr in ("embed_tokens", "wte", "shared"):
        for base in (model, getattr(model, "model", None),
                     getattr(getattr(model, "model", None), "language_model", None)):
            if base is not None:
                cand = getattr(base, attr, None)
                if isinstance(cand, _nn.Embedding):
                    return cand
    raise RuntimeError("token embedding not found")


class StepfunGenNet(nn.Module):
    """Geometry latent <-> Stepfun soft tokens -> predicted x0 latent.

    Multi-depth residual read-out: hooks several LLM layers (not just the last)
    and fuses their geom-token representations into the output with learnable
    gates (init 0 → starts as the deepest layer, learns to add shallower ones).
    This ingests the backbone's computation at every depth into the generation.
    """

    def __init__(self, stepfun_model, embed, llm_layers, latent_ch, latent_res,
                 hidden=4096, n_toks=None, hook_fracs=(0.25, 0.5, 0.75, 1.0)):
        super().__init__()
        object.__setattr__(self, "_model", stepfun_model)
        object.__setattr__(self, "_embed", embed)
        self.hidden = hidden
        self.latent_ch = latent_ch
        self.latent_res = latent_res
        self.n_toks = n_toks or (latent_res ** 3)

        # geom-token projection (latent_ch -> hidden) + timestep
        self.geom_in = nn.Linear(latent_ch, hidden)
        self.t_embed = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden))

        # multi-depth hooks: pick layer indices from the LLM ModuleList
        n = len(llm_layers)
        self.hook_idx = [min(int(f * n) - 1, n - 1) for f in hook_fracs]
        self.hook_idx = sorted(set(self.hook_idx))
        self.proj = nn.ModuleList([nn.Linear(hidden, latent_ch) for _ in self.hook_idx])
        # residual gates: init 0 → deepest layer dominates at start, shallower
        # layers are added residually as they're learned. We use the LAST hooked
        # layer as the base (gate fixed at 1); others learn from 0.
        self.gate = nn.Parameter(torch.zeros(len(self.hook_idx)))   # learned for non-base
        self._base = self.hook_idx.index(max(self.hook_idx))         # which hook is the base

        self._dev = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._bufs: list[list] = [[] for _ in self.hook_idx]
        self._cur_text_len = 0
        for bi, idx in enumerate(self.hook_idx):
            llm_layers[idx].register_forward_hook(self._make_hook(bi))

    def _make_hook(self, bi):
        def hk(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            s = self._cur_text_len
            # NOTE: no .detach() — we WANT gradients to flow back through Stepfun
            # (weights frozen) so geom_in / t_embed learn to align soft tokens.
            self._bufs[bi].append(h[:, s:s + self.n_toks])
        return hk

    @property
    def device(self):
        return self._dev.device

    def _timestep(self, t):
        half = self.hidden // 2
        freq = torch.exp(-torch.arange(half, device=t.device) / half * 4.0)
        arg = t[:, None].float() * freq[None]
        emb = torch.cat([torch.cos(arg), torch.sin(arg)], -1)
        return self.t_embed(emb.to(self.t_embed[0].weight.dtype))

    def forward(self, noisy_latent, t, text_ids, text_lens) -> torch.Tensor:
        B = noisy_latent.shape[0]
        L = self.latent_res
        dev = self._embed.weight.device
        for buf in self._bufs:
            buf.clear()
        self._cur_text_len = text_ids.shape[1]

        mdtype = self._embed.weight.dtype              # bf16 (the frozen model's dtype)
        text_emb = self._embed(text_ids.to(dev))
        g = noisy_latent.permute(0, 2, 3, 4, 1).reshape(B, L * L * L, self.latent_ch)
        g_tok = self.geom_in(g.to(dev)) + self._timestep(t).to(dev).unsqueeze(1)
        inputs_embeds = torch.cat([text_emb, g_tok], dim=1).to(mdtype)
        am = torch.ones(B, inputs_embeds.shape[1], dtype=torch.long, device=dev)
        pos = torch.arange(inputs_embeds.shape[1], device=dev).unsqueeze(0).expand(B, -1)
        # NOTE: NOT under no_grad — gradients flow through the frozen Stepfun so
        # geom_in/t_embed learn. Weights stay frozen; only activations are stored.
        self._model(inputs_embeds=inputs_embeds, attention_mask=am,
                    position_ids=pos, use_cache=False)

        # residual fusion of multi-depth geom representations
        head_dev = self.proj[self._base].weight.device
        out = self.proj[self._base](self._bufs[self._base][0].to(head_dev))   # base (deepest)
        gates = self.gate.sigmoid()
        for i, buf in enumerate(self._bufs):
            if i == self._base:
                continue
            out = out + gates[i] * self.proj[i](buf[0].to(head_dev))          # residual adds
        out = out.reshape(B, L, L, L, self.latent_ch).permute(0, 4, 1, 2, 3)
        return out
