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


def _find_last_layer(model):
    """The LLM transformer-layer ModuleList's last element (skip vision)."""
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
    return best[2][-1]  # type: ignore[index]


class StepfunGenNet(nn.Module):
    """Geometry latent <-> Stepfun soft tokens -> predicted x0 latent.

    forward(noisy_latent, t, text_ids, text_lens) -> predicted clean latent.
    """

    def __init__(self, stepfun_model, embed, last_layer, latent_ch, latent_res,
                 hidden=4096, n_toks=None):
        super().__init__()
        # store the huge model UNTRACKED (object.__setattr__) so .to() won't move it
        object.__setattr__(self, "_model", stepfun_model)
        object.__setattr__(self, "_embed", embed)
        object.__setattr__(self, "_last_layer", last_layer)
        self.hidden = hidden
        self.latent_ch = latent_ch
        self.latent_res = latent_res
        self.n_toks = n_toks or (latent_res ** 3)   # geom tokens = flattened latent grid

        # trainable: latent_ch -> hidden (soft-token projection) + timestep embed
        self.geom_in = nn.Linear(latent_ch, hidden)
        self.t_embed = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden))
        # trainable: hidden -> latent_ch (read-out head)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, latent_ch))
        # placeholder so .to()/device works (the big model is untracked)
        self._dev = nn.Parameter(torch.zeros(1), requires_grad=False)

        # hook the last LLM layer to capture hidden states
        self._captured: list = []
        last_layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        self._captured.clear()
        self._captured.append(h)

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
        """noisy_latent (B,C,L,L,L); text_ids (B,Lt) padded
        text_lens (B,).
        Returns predicted x0 latent (B,C,L,L,L)."""
        B = noisy_latent.shape[0]
        L = self.latent_res
        dev = self._embed.weight.device

        # text embeddings (frozen)
        text_emb = self._embed(text_ids.to(dev))            # (B,Lt,hidden)

        # geom soft tokens: flatten latent grid -> project -> add timestep
        g = noisy_latent.permute(0, 2, 3, 4, 1).reshape(B, L * L * L, self.latent_ch)
        g_tok = self.geom_in(g.to(dev))                     # (B, n_toks, hidden)
        g_tok = g_tok + self._timestep(t).to(g_tok.dtype).unsqueeze(1)

        inputs_embeds = torch.cat([text_emb, g_tok], dim=1)  # (B, Lt+n_toks, hidden)
        am = torch.ones(B, inputs_embeds.shape[1], dtype=torch.long, device=dev)
        pos = torch.arange(inputs_embeds.shape[1], device=dev).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            self._model(inputs_embeds=inputs_embeds, attention_mask=am,
                        position_ids=pos, use_cache=False)
        h = self._captured[0]                                # (B, Lt+n_toks, hidden)
        # slice geom positions (right after text) and read out
        geom_h = h[:, text_ids.shape[1]:text_ids.shape[1] + self.n_toks]
        pred = self.head(geom_h)                             # (B, n_toks, latent_ch)
        pred = pred.reshape(B, L, L, L, self.latent_ch).permute(0, 4, 1, 2, 3)
        return pred
