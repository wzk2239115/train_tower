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


class _DecBlock(nn.Module):
    """Self-attention + FFN block, adaLN-modulated by the flow timestep."""

    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class GeomDecoder(nn.Module):
    """Large trainable decoder sitting AFTER the frozen Stepfun.

    Operates on the n_toks geom tokens. Stepfun's multi-depth features are
    residually injected: the token sequence is SEEDED by their (projected) sum,
    and re-injected at several block depths. Timestep-conditioned via adaLN-ish
    additive bias. Output head -> latent_ch per token.
    """

    def __init__(self, hidden=4096, dim=1024, n_blocks=12, n_inject=4, latent_ch=32):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.proj_in = nn.ModuleList([nn.Linear(hidden, dim) for _ in range(n_inject)])
        self.t_embed = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_DecBlock(dim) for _ in range(n_blocks)])
        # residual injection points (block indices), one per Stepfun depth
        self.inject_at = [int((i + 1) * n_blocks / (n_inject + 1)) for i in range(n_inject)]
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_ch)

    def _timestep(self, t):
        half = self.dim // 2
        freq = torch.exp(-torch.arange(half, device=t.device) / half * 4.0)
        arg = t[:, None].float() * freq[None]
        emb = torch.cat([torch.cos(arg), torch.sin(arg)], -1)
        return self.t_embed(emb.to(self.t_embed[0].weight.dtype))

    def forward(self, feats, t):
        # feats: list of (B, n_toks, 4096) from Stepfun at several depths
        proj = [p(f.to(p.weight.device).to(p.weight.dtype)) for p, f in zip(self.proj_in, feats)]
        t_emb = self._timestep(t).unsqueeze(1).to(proj[0].dtype)   # (B,1,dim)
        x = sum(proj) + t_emb                                      # residual seed from Stepfun + t
        inj_set = set(self.inject_at)
        for i, blk in enumerate(self.blocks):
            if i in inj_set:
                k = self.inject_at.index(i)
                x = x + proj[k]                                    # residual re-injection
            x = blk(x)
        return self.out(self.out_norm(x))                         # (B, n_toks, latent_ch)


class StepfunGenNet(nn.Module):
    """Geometry latent <-> Stepfun soft tokens -> predicted x0 latent.

    Multi-depth residual read-out: hooks several LLM layers (not just the last)
    and fuses their geom-token representations into the output with learnable
    gates (init 0 → starts as the deepest layer, learns to add shallower ones).
    This ingests the backbone's computation at every depth into the generation.
    """

    def __init__(self, stepfun_model, embed, llm_layers, latent_ch, latent_res,
                 hidden=4096, n_toks=None, hook_fracs=(0.25, 0.5, 0.75, 1.0),
                 dec_dim=1024, dec_blocks=12):
        super().__init__()
        object.__setattr__(self, "_model", stepfun_model)
        object.__setattr__(self, "_embed", embed)
        self.hidden = hidden
        self.latent_ch = latent_ch
        self.latent_res = latent_res
        self.n_toks = n_toks or (latent_res ** 3)

        # geom-token projection (latent_ch -> hidden): inject noisy latent into
        # frozen Stepfun. FROZEN (Stepfun stays frozen to preserve the base).
        self.geom_in = nn.Linear(latent_ch, hidden)
        self.t_embed_sf = nn.Linear(hidden, hidden)
        for mod in (self.geom_in, self.t_embed_sf):
            for p in mod.parameters():
                p.requires_grad_(False)

        # multi-depth hooks on the frozen LLM
        n = len(llm_layers)
        self.hook_idx = sorted({min(int(f * n) - 1, n - 1) for f in hook_fracs})

        # === the LARGE trainable decoder AFTER Stepfun ===
        # a stack of transformer blocks on the geom tokens; Stepfun's multi-depth
        # features are residually injected at several depths. Stepfun is frozen
        # (its pretrained capability preserved); this decoder does the learning.
        self.decoder = GeomDecoder(hidden=hidden, dim=dec_dim, n_blocks=dec_blocks,
                                   n_inject=len(self.hook_idx), latent_ch=latent_ch)

        self._dev = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._bufs: list[list] = [[] for _ in self.hook_idx]
        self._cur_text_len = 0
        for bi, idx in enumerate(self.hook_idx):
            llm_layers[idx].register_forward_hook(self._make_hook(bi))

    def _make_hook(self, bi):
        def hk(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            s = self._cur_text_len
            self._bufs[bi].append(h[:, s:s + self.n_toks].detach())
        return hk

    @property
    def device(self):
        return self._dev.device

    def _sf_timestep(self, t):
        half = self.hidden // 2
        freq = torch.exp(-torch.arange(half, device=t.device) / half * 4.0)
        arg = t[:, None].float() * freq[None]
        emb = torch.cat([torch.cos(arg), torch.sin(arg)], -1)
        return self.t_embed_sf(emb.to(self.t_embed_sf.weight.dtype))

    def forward(self, noisy_latent, t, text_ids, text_lens) -> torch.Tensor:
        B = noisy_latent.shape[0]
        L = self.latent_res
        dev = self._embed.weight.device
        for buf in self._bufs:
            buf.clear()
        self._cur_text_len = text_ids.shape[1]

        mdtype = self._embed.weight.dtype              # bf16 (the frozen model)
        text_emb = self._embed(text_ids.to(dev))
        g = noisy_latent.permute(0, 2, 3, 4, 1).reshape(B, L * L * L, self.latent_ch)
        g_tok = self.geom_in(g.to(dev)) + self._sf_timestep(t).to(dev).unsqueeze(1)
        inputs_embeds = torch.cat([text_emb, g_tok], dim=1).to(mdtype)
        am = torch.ones(B, inputs_embeds.shape[1], dtype=torch.long, device=dev)
        pos = torch.arange(inputs_embeds.shape[1], device=dev).unsqueeze(0).expand(B, -1)
        # FROZEN Stepfun forward (preserves the base; no gradient into the 150B).
        # Its computation transforms the geom tokens across all 45 layers; the
        # large trainable decoder AFTER it reads those features (residually).
        with torch.no_grad():
            self._model(inputs_embeds=inputs_embeds, attention_mask=am,
                        position_ids=pos, use_cache=False)

        # multi-depth geom features from frozen Stepfun -> large trainable decoder
        feats = [buf[0] for buf in self._bufs]         # list of (B, n_toks, hidden)
        out = self.decoder(feats, t.to(self.decoder.proj_in[0].weight.device))
        out = out.reshape(B, L, L, L, self.latent_ch).permute(0, 4, 1, 2, 3)
        return out
