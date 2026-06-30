"""Multimodal condition backbones.

The backbone is a *pretrained component*: it encodes text / sketches /
reference images into condition tokens that drive the geometry decoder.
It does NOT generate tokens.

Two adapters share the same interface:

* ``ProxyBackbone``  — a small CLIP text+image encoder. Used on the dev machine
  (the local Step-3.7-Flash snapshot is incomplete) to exercise the full
  pipeline end-to-end.
* ``StepfunBackbone`` — loads Step-3.7-Flash (full weights on the compute box),
  runs a forward pass and extracts multimodal hidden states. Frozen.

Both return ``ConditionOutput``:
    tokens : (B, T, cond_dim)   — cross-attention source
    pooled : (B, cond_dim)      — global condition vector (adaLN / modulation)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConditionOutput:
    tokens: torch.Tensor   # (B, T, D)
    pooled: torch.Tensor   # (B, D)


class BackboneAdapter(nn.Module):
    """Interface. Subclasses implement ``forward``."""

    def forward(self, prompt: list[str] | str, sketch: torch.Tensor | None = None,
                ref_image: torch.Tensor | None = None) -> ConditionOutput:
        raise NotImplementedError


class _TextOnlyProjection(nn.Module):
    """Tiny learnable projection so the cond_dim matches the decoder."""

    def __init__(self, in_dim: int, cond_dim: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Linear(in_dim, cond_dim)
        self.query = nn.Parameter(torch.randn(n_tokens, cond_dim) * 0.02)

    def forward(self, feats: torch.Tensor) -> ConditionOutput:
        # feats: (B, K, in_dim) or (B, in_dim)
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)
        feats = self.proj(feats)  # (B, K, D)
        q = self.query.unsqueeze(0).expand(feats.shape[0], -1, -1)
        pooled_tok = q + feats.mean(dim=1, keepdim=True)
        tokens = torch.cat([feats, pooled_tok], dim=1)  # (B, K+n_tokens, D)
        pooled = feats.mean(dim=1)
        return ConditionOutput(tokens=tokens, pooled=pooled)


# --------------------------------------------------------------------------- #
# Self-contained tokenizer for the offline proxy (no HF tokenizer needed)
# --------------------------------------------------------------------------- #

class _HashTokenizer:
    """Word-level hash tokenizer with a fixed vocab. Fully offline."""

    def __init__(self, dim: int = 16, max_len: int = 48):
        self.dim = dim
        self.max_len = max_len

    def __call__(self, texts: list[str], device) -> torch.Tensor:
        import hashlib

        out = torch.zeros(len(texts), self.max_len, self.dim, device=device)
        for i, t in enumerate(texts):
            words = t.lower().replace(",", " ").replace(";", " ").split()
            for j, w in enumerate(words[: self.max_len]):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, j] = (h % 1000) / 1000.0
        return out


class ProxyBackbone(BackboneAdapter):
    """Fully-offline multimodal condition encoder (no pretrained download).

    Text  → hash-tokenized → embedding table (trainable), mean-pooled.
    Sketch→ small CNN (trainable).
    Both project to ``cond_dim`` tokens that drive the geometry decoder. This is
    deliberately lightweight: its only job is to exercise the full pipeline on
    the dev machine. On the compute box use ``StepfunBackbone`` for the real
    pretrained multimodal features.
    """

    def __init__(self, cond_dim: int = 768, n_cond_tokens: int = 32,
                 image_size: int = 224, text_emb: int = 64):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_cond_tokens = n_cond_tokens
        self.image_size = image_size
        self._tok = _HashTokenizer(dim=text_emb, max_len=48)
        self.text_embed = nn.Sequential(
            nn.Linear(text_emb, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim))
        self.sketch_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, cond_dim, 4, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.sketch_proj = nn.Linear(cond_dim, cond_dim)
        self.query = nn.Parameter(torch.randn(n_cond_tokens, cond_dim) * 0.02)

    def _encode_text(self, texts: list[str], device) -> torch.Tensor:
        toks = self._tok(texts, device)  # (B, L, text_emb)
        return self.text_embed(toks)     # (B, L, cond_dim)

    def _encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        if sketch.shape[-1] != self.image_size:
            import torch.nn.functional as F
            sketch = F.interpolate(sketch, size=self.image_size,
                                    mode="bilinear", align_corners=False)
        f = self.sketch_cnn(sketch).flatten(2).transpose(1, 2)  # (B,1,cond_dim)
        return self.sketch_proj(f)

    def forward(self, prompt, sketch=None, ref_image=None) -> ConditionOutput:
        if isinstance(prompt, str):
            prompt = [prompt]
        device = self.text_embed[0].weight.device
        feats = self._encode_text(prompt, device)
        if sketch is not None:
            feats = feats + self._encode_sketch(sketch).expand(-1, feats.shape[1], -1) * 0.1
        q = self.query.unsqueeze(0).expand(feats.shape[0], -1, -1)
        pooled_tok = q + feats.mean(dim=1, keepdim=True)
        tokens = torch.cat([feats, pooled_tok], dim=1)
        pooled = feats.mean(dim=1)
        return ConditionOutput(tokens=tokens, pooled=pooled)


class QwenProxyBackbone(BackboneAdapter):
    """Optional: use a local pretrained text model (e.g. Qwen3-0.6B) as the
    *frozen* condition encoder + a small trainable sketch CNN. Demonstrates the
    "pretrained model as a component" idea with a real backbone locally."""

    def __init__(self, cond_dim: int = 768, n_cond_tokens: int = 32,
                 image_size: int = 224, pretrained_path: str = "Qwen/Qwen3-0.6B"):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_cond_tokens = n_cond_tokens
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        self._model = None
        self._hidden = None
        self._proj = None
        self.sketch_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, cond_dim, 4, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.sketch_proj = nn.Linear(cond_dim, cond_dim)
        self.query = nn.Parameter(torch.randn(n_cond_tokens, cond_dim) * 0.02)

    def _build(self, device):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained(self.pretrained_path,
                                                 torch_dtype=torch.bfloat16)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self._model = m.to(device)
        self._tok = AutoTokenizer.from_pretrained(self.pretrained_path)
        self._hidden = m.config.hidden_size
        self._proj = nn.Linear(self._hidden, self.cond_dim).to(device)

    def forward(self, prompt, sketch=None, ref_image=None):
        self._build(self.query.device)
        device = self.query.device
        texts = [prompt] if isinstance(prompt, str) else list(prompt)
        enc = self._tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = self._model.model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1].float()  # (B, L, hidden)
        feats = self._proj(h)
        if sketch is not None:
            f = self.sketch_cnn(sketch).flatten(2).transpose(1, 2)
            feats = feats + self.sketch_proj(f).expand(-1, feats.shape[1], -1) * 0.1
        q = self.query.unsqueeze(0).expand(feats.shape[0], -1, -1)
        pooled_tok = q + feats.mean(dim=1, keepdim=True)
        tokens = torch.cat([feats, pooled_tok], dim=1)
        return ConditionOutput(tokens=tokens, pooled=feats.mean(dim=1))


def _patch_causal_mask_compat() -> list[str]:
    """Wrap every mask-creation function reachable in ``sys.modules`` so it
    tolerates extra kwargs (e.g. ``cache_position``) that Step-3.7-Flash's
    remote modeling code passes but the installed transformers may not accept.

    Covers ``create_causal_mask``, ``create_sliding_window_causal_mask`` and
    any other ``create_*mask*``/``create_*causal*`` callable. For a full-sequence
    forward (extracting hidden states, not generation) the dropped kwargs don't
    affect the mask. Same spirit as the project's ``tower/unify/compat.py`` shim.
    """
    import functools
    import inspect
    import sys

    patched: list[str] = []

    def _is_target(name: str) -> bool:
        if not name.startswith("create_"):
            return False
        low = name.lower()
        return "mask" in low or "causal" in low

    def _wrap(fn):
        if getattr(fn, "_structgen_mask_patched", False):
            return fn
        try:
            params = set(inspect.signature(fn).parameters)
        except (ValueError, TypeError):
            params = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if params is not None and kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in params}
            return fn(*args, **kwargs)

        wrapper._structgen_mask_patched = True  # type: ignore[attr-defined]
        return wrapper

    for modname, mod in list(sys.modules.items()):
        if mod is None:
            continue
        names = getattr(mod, "__dict__", {})
        for name, fn in list(names.items()):
            if _is_target(name) and callable(fn) and not getattr(fn, "_structgen_mask_patched", False):
                try:
                    setattr(mod, name, _wrap(fn))
                    patched.append(f"{modname}.{name}")
                except Exception:
                    pass
    return patched


class StepfunBackbone(BackboneAdapter):
    """Step-3.7-Flash backbone (full weights on the compute box).

    Loads via ``AutoModelForCausalLM`` (the compute box has the modeling code +
    complete shards). Frozen. Extracts multimodal hidden states and projects
    them to ``cond_dim``.

    On the dev machine the snapshot is incomplete, so this only instantiates a
    *configuration object*; actual loading is deferred and will run on the
    compute box where the full model lives.
    """

    def __init__(self, cond_dim: int = 768, n_cond_tokens: int = 32,
                 pretrained_path: str | None = None, image_size: int = 224):
        super().__init__()
        self.cond_dim = cond_dim
        self.n_cond_tokens = n_cond_tokens
        self.image_size = image_size
        self.pretrained_path = pretrained_path
        # placeholder param so .to()/device-of-module works before lazy build
        self._device_marker = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._model = None
        self._processor = None
        self._proj = None

    def _build(self, device: torch.device) -> None:
        if self._model is not None:
            return
        import time

        import transformers as _tf
        path = self.pretrained_path
        # 198B MoE ≈ 396GB in bf16 → must shard across all visible GPUs.
        # device_map="auto" distributes layer-by-layer across GPUs. Leave more
        # headroom on GPU 0 (hosts the trainable decoder + optimizer).
        n_gpu = torch.cuda.device_count() or 1
        max_memory = {0: "68GiB"}
        for i in range(1, n_gpu):
            max_memory[i] = "78GiB"
        print(f"[stepfun] loading {path} on {n_gpu} GPU(s) "
              f"(device_map=auto, ~396GB bf16, this reads from disk once per run)...")
        t0 = time.time()
        model = _tf.AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True,
        )
        # compat: tolerate extra mask kwargs (cache_position) from remote code
        patched = _patch_causal_mask_compat()
        if patched:
            print(f"[stepfun] patched create_causal_mask in {len(patched)} module(s) "
                  f"(compat: {', '.join(patched[:3])}{'…' if len(patched) > 3 else ''})")
        # report placement: confirm the model actually spans multiple GPUs
        dm = getattr(model, "hf_device_map", {})
        from collections import Counter

        if dm:
            per_gpu = Counter(str(v) for v in dm.values())
            cpu_mods = per_gpu.pop("cpu", 0)
            print(f"[stepfun] loaded in {time.time() - t0:.0f}s, spans {len(per_gpu)} GPU(s): "
                  + ", ".join(f"{k}:{v} mods" for k, v in sorted(per_gpu.items())))
            if cpu_mods:
                print(f"[stepfun] WARNING: {cpu_mods} modules on CPU (forward will be slow); "
                      "consider lowering --res/--batch")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # Store WITHOUT nn.Module tracking (object.__setattr__) so that the
        # module-level .to() in train() never tries to move the 198B weights —
        # device_map already placed them across GPUs.
        object.__setattr__(self, "_model", model)
        try:
            proc = _tf.AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception:
            proc = None
        object.__setattr__(self, "_processor", proc)
        hidden = getattr(model.config, "text_config", model.config).hidden_size
        object.__setattr__(self, "_hidden", hidden)
        # projection is trainable → registered normally, lives on the decoder GPU
        self._proj = _TextOnlyProjection(hidden, self.cond_dim,
                                         self.n_cond_tokens).to(device)
        # output_hidden_states isn't reliably forwarded by the remote code, so
        # capture the LLM's last-layer output via a forward hook instead.
        self._install_last_layer_hook(model)

    def _install_last_layer_hook(self, model) -> None:
        import torch.nn as _nn

        # find the LLM transformer-layer ModuleList (skip the vision encoder):
        # prefer a list whose name contains language/text/llm, length >= 10.
        best = None  # (score, name, module_list)
        for name, mod in model.named_modules():
            if not isinstance(mod, _nn.ModuleList) or len(mod) < 10:
                continue
            low = name.lower()
            if any(k in low for k in ("vision", "image", "visual", "encoder")):
                continue
            score = len(mod) + (200 if "language" in low else
                                (150 if "text" in low or "llm" in low else 0))
            if best is None or score > best[0]:
                best = (score, name, mod)
        if best is None:
            raise RuntimeError("could not locate the LLM layer list for hook")
        last = best[2][-1]
        captured: list = []
        object.__setattr__(self, "_captured_hidden", captured)

        def _hook(_module, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            if hasattr(h, "last_hidden_state"):
                h = h.last_hidden_state
            captured.clear()
            captured.append(h)

        last.register_forward_hook(_hook)
        print(f"[stepfun] hidden-state hook on {best[1]}[{len(best[2]) - 1}]")

    def _masked_mean_pool(self, h: torch.Tensor, attention_mask: torch.Tensor | None,
                          proj_device: torch.device) -> torch.Tensor:
        """Mean-pool (B,L,hidden) → (B,hidden), ignoring padding positions."""
        h = h.float().to(proj_device)
        if attention_mask is None:
            return h.mean(dim=1)
        m = attention_mask.float().to(proj_device).unsqueeze(-1)  # (B,L,1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def forward(self, prompt, sketch=None, ref_image=None) -> ConditionOutput:
        self._build(self._device_marker.device)
        proj_device = self._device_marker.device
        in_device = next(self._model.parameters()).device
        texts = [prompt] if isinstance(prompt, str) else list(prompt)
        B = len(texts)

        messages = []
        for txt in texts:
            content = [{"type": "text", "text": txt}]
            if sketch is not None:
                content.append({"type": "image"})
            messages.append([{"role": "user", "content": content}])
        chat_texts = [self._processor.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False) for m in messages]

        try:
            # ---- fast path: one batched forward (processor pads + masks) ----
            if sketch is not None:
                images = [_to_pil(sketch[i]) for i in range(B)]
                proc_inputs = self._processor(text=chat_texts, images=images,
                                              return_tensors="pt", padding=True)
            else:
                proc_inputs = self._processor(text=chat_texts,
                                              return_tensors="pt", padding=True)
            attn = proc_inputs.get("attention_mask")
            proc_inputs = {k: (v.to(in_device) if torch.is_tensor(v) else v)
                           for k, v in proc_inputs.items()}
            with torch.no_grad():
                self._model(**proc_inputs, use_cache=False)
            pooled = self._masked_mean_pool(self._captured_hidden[0], attn, proj_device)
            return self._proj(pooled)
        except Exception as e:
            # ---- fallback: per-sample forward (robust to batching issues) ----
            print(f"[stepfun] batched forward failed ({e!r}); falling back to per-sample")
            feats = []
            for i in range(B):
                one: dict = {"text": chat_texts[i], "return_tensors": "pt"}
                if sketch is not None:
                    one["images"] = _to_pil(sketch[i])
                inp = self._processor(**one)
                am = inp.get("attention_mask")
                inp = {k: (v.to(in_device) if torch.is_tensor(v) else v) for k, v in inp.items()}
                with torch.no_grad():
                    self._model(**inp, use_cache=False)
                feats.append(self._masked_mean_pool(self._captured_hidden[0], am, proj_device))
            return self._proj(torch.cat(feats, dim=0))


def _to_pil(t: torch.Tensor):
    from PIL import Image
    arr = ((t.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def build_backbone(cfg) -> BackboneAdapter:
    kind = cfg.backbone.kind
    if kind == "proxy":
        return ProxyBackbone(cond_dim=cfg.backbone.cond_dim,
                             n_cond_tokens=cfg.backbone.n_cond_tokens,
                             image_size=cfg.backbone.image_size)
    if kind == "qwen":
        return QwenProxyBackbone(cond_dim=cfg.backbone.cond_dim,
                                 n_cond_tokens=cfg.backbone.n_cond_tokens,
                                 image_size=cfg.backbone.image_size,
                                 pretrained_path=cfg.backbone.pretrained_path or "Qwen/Qwen3-0.6B")
    if kind == "stepfun":
        return StepfunBackbone(cond_dim=cfg.backbone.cond_dim,
                               n_cond_tokens=cfg.backbone.n_cond_tokens,
                               pretrained_path=cfg.backbone.pretrained_path,
                               image_size=cfg.backbone.image_size)
    raise ValueError(f"unknown backbone kind: {kind}")
