#!/usr/bin/env python
"""One-shot diagnostic for Step-3.7-Flash as a structgen backbone.

Run ONCE on the compute box, paste the full output back. It probes everything
needed to make the backbone forward robust:

  * device_map placement summary
  * the LLM transformer-layer ModuleList (hook target)
  * processor output keys + tensor shapes
  * hidden-state hook output (shape/dtype/device)
  * whether output_hidden_states actually works
  * multi-sample sequence-length mismatch (the cat() failure)
  * mean-pool shape (the likely fix)

Usage:
    python scripts/structgen_probe_stepfun.py /home/jovyan/h800fast/wangzekai/Step-3.7-Flash
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn


def main(path: str) -> None:
    import transformers as tf
    print("==== ENV ====")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"n_gpu={torch.cuda.device_count()} transformers={tf.__version__}")

    from structgen.model.backbone import _patch_causal_mask_compat

    print("\n==== LOAD ====")
    n_gpu = torch.cuda.device_count() or 1
    max_memory = {0: "68GiB"}
    for i in range(1, n_gpu):
        max_memory[i] = "78GiB"
    model = tf.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True)
    _patch_causal_mask_compat()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dm = getattr(model, "hf_device_map", {})
    from collections import Counter
    per_gpu = Counter(str(v) for v in dm.values())
    print("device_map:", {k: v for k, v in sorted(per_gpu.items())})

    print("\n==== STRUCTURE (top-level children) ====")
    for name, _ in model.named_children():
        print("  .", name)

    print("\n==== CANDIDATE LAYER LISTS (ModuleList len>=8) ====")
    cands = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            cands.append((name, len(mod)))
    for name, ln in cands:
        flag = "  <-- vision?" if any(k in name.lower() for k in
            ("vision", "image", "visual")) else "  <-- LLM?"
        print(f"  {name}  len={ln}{flag}")

    # locate the LLM list (skip vision)
    best = None
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList) or len(mod) < 8:
            continue
        low = name.lower()
        if any(k in low for k in ("vision", "image", "visual", "encoder")):
            continue
        score = len(mod) + (200 if "language" in low else (150 if "text" in low else 0))
        if best is None or score > best[0]:
            best = (score, name, mod)
    assert best is not None, "no LLM layer list found"
    hook_name, last = best[1], best[2][-1]
    print(f"\n==== HOOK TARGET: {hook_name}[{len(best[2]) - 1}] "
          f"({type(last).__name__}) ====")

    captured: list = []
    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(h, "last_hidden_state"):
            h = h.last_hidden_state
        captured.clear()
        captured.append(h)
    last.register_forward_hook(_hook)

    proc = tf.AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def build_inputs(prompt, with_image=False):
        content = [{"type": "text", "text": prompt}]
        if with_image:
            content.append({"type": "image"})
        msg = [{"role": "user", "content": content}]
        kwargs = dict(
            text=proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False),
            return_tensors="pt", padding=True)
        if with_image:
            from PIL import Image
            kwargs["images"] = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return proc(**kwargs)

    in_device = next(model.parameters()).device
    p1 = "A 3D-printable structural cylinder part with internal gyroid infill."
    p2 = "Box with Schwarz-P lattice."  # different length

    print("\n==== FORWARD #1: text-only (prompt1) ====")
    inp1 = build_inputs(p1, with_image=False)
    print("processor keys:", list(inp1.keys()))
    for k, v in inp1.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    inp1 = {k: (v.to(in_device) if torch.is_tensor(v) else v) for k, v in inp1.items()}
    with torch.no_grad():
        out1 = model(**inp1, use_cache=False)
    print("out type:", type(out1).__name__)
    print("out fields:", [a for a in dir(out1) if not a.startswith("_")][:20])
    print("captured hidden: shape=", tuple(captured[0].shape),
          "dtype=", captured[0].dtype, "device=", captured[0].device)

    print("\n==== PROBE: output_hidden_states=True ====")
    try:
        with torch.no_grad():
            out1h = model(**inp1, use_cache=False, output_hidden_states=True)
        hs = getattr(out1h, "hidden_states", None)
        print("out.hidden_states is None?", hs is None)
        if hs is not None:
            print("  num layers:", len(hs), "layer0 shape:", tuple(hs[0].shape),
                  "last shape:", tuple(hs[-1].shape))
    except Exception as e:
        print("  output_hidden_states raised:", repr(e))

    print("\n==== FORWARD #2: prompt2 (different length) ====")
    inp2 = build_inputs(p2, with_image=False)
    inp2 = {k: (v.to(in_device) if torch.is_tensor(v) else v) for k, v in inp2.items()}
    with torch.no_grad():
        model(**inp2, use_cache=False)
    print("captured hidden #2: shape=", tuple(captured[0].shape),
          "(mismatch with #1 causes torch.cat to fail)")

    print("\n==== MEAN-POOL (likely fix) ====")
    # re-run #1 to refill captured
    with torch.no_grad():
        model(**inp1, use_cache=False)
    h1 = captured[0].float().mean(dim=1)  # (1, hidden)
    with torch.no_grad():
        model(**inp2, use_cache=False)
    h2 = captured[0].float().mean(dim=1)
    pooled = torch.cat([h1, h2], dim=0)
    print("pooled stack shape:", tuple(pooled.shape),
          "hidden_dim=", int(pooled.shape[-1]))

    print("\n==== FORWARD #3: text + image ====")
    try:
        inp3 = build_inputs(p1, with_image=True)
        print("processor keys:", list(inp3.keys()))
        for k, v in inp3.items():
            if torch.is_tensor(v):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        inp3 = {k: (v.to(in_device) if torch.is_tensor(v) else v) for k, v in inp3.items()}
        with torch.no_grad():
            model(**inp3, use_cache=False)
        print("captured hidden #3: shape=", tuple(captured[0].shape))
    except Exception as e:
        print("  image forward raised:", repr(e))

    print("\n==== language_model direct ====")
    if hasattr(model, "language_model"):
        print("model.language_model =", type(model.language_model).__name__)
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "language_model"):
        lm = getattr(getattr(model, "model"), "language_model")
        print("model.model.language_model =", type(lm).__name__)
    print("\nDONE")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/jovyan/h800fast/wangzekai/Step-3.7-Flash"
    main(path)
