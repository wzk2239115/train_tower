"""Probe: can Step3p7 be used as a generative backbone (geom soft-tokens in)?

Checks the 2 unknowns before building 'Stepfun-as-denoiser':
  1. does forward accept `inputs_embeds` (continuous tokens, not input_ids)?
  2. does BATCHED forward work (RoPE bug) with UNIFORM-length inputs_embeds?
  3. can we read hidden states at the injected geom-token positions?
Run ONCE on the compute box, paste output.
"""
import sys
import torch
import torch.nn as nn


def main(path):
    import transformers as tf
    from structgen.model.backbone import _patch_causal_mask_compat

    n_gpu = torch.cuda.device_count() or 1
    max_mem = {0: "68GiB"}
    for i in range(1, n_gpu):
        max_mem[i] = "78GiB"
    print("loading...")
    model = tf.AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto", max_memory=max_mem, low_cpu_mem_usage=True)
    _patch_causal_mask_compat()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = tf.AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", None) or tf.AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # --- locate the input embeddings (token id -> hidden) ---
    embed = None
    for attr in ("embed_tokens", "wte", "shared"):
        m = model
        for part in ("model", "language_model"):
            sub = getattr(m, part, None)
            cand = getattr(sub, attr, None) or getattr(m, attr, None)
            if isinstance(cand, nn.Embedding):
                embed = cand
                break
        if embed:
            break
    assert embed is not None, "could not find token embedding"
    H = embed.embedding_dim
    print(f"[probe] embed found: {type(embed).__name__}, hidden={H}")

    # --- text -> input_ids -> embeds (batch=1) ---
    msg = [{"role": "user", "content": [{"type": "text", "text": "a brown wooden chair"}]}]
    chat = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = tok(chat, return_tensors="pt")["input_ids"].to(model.device)
    Lt = ids.shape[1]
    with torch.no_grad():
        text_emb = embed(ids).to(model.device)          # (1, Lt, H)
    print(f"[probe] text tokens={Lt}, text_emb {tuple(text_emb.shape)}")

    # --- inject geom soft-tokens (random projection stand-in) ---
    Lg = 64                                          # geom latent tokens (e.g. 8^3/8)
    geom = torch.randn(1, Lg, H, device=model.device, dtype=text_emb.dtype) * 0.02
    inputs_embeds = torch.cat([text_emb, geom], dim=1)   # (1, Lt+Lg, H)
    am = torch.ones(1, Lt + Lg, device=model.device, dtype=torch.long)
    pos = torch.arange(Lt + Lg, device=model.device).unsqueeze(0)

    print("\n=== TEST 1: B=1 inputs_embeds forward ===")
    try:
        with torch.no_grad():
            out = model(inputs_embeds=inputs_embeds, attention_mask=am,
                        position_ids=pos, use_cache=False)
        # read hidden states at geom positions
        hs = getattr(out, "hidden_states", None)
        print("  OK. out type:", type(out).__name__,
              "hidden_states:", "None" if hs is None else f"{len(hs)} layers, last {tuple(hs[-1].shape)}")
    except Exception as e:
        print("  inputs_embeds FAILED:", repr(e))

    # --- hook the last LLM layer to read geom-token hidden states (fallback) ---
    captured = []

    def _hk(_m, _i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        captured.clear(); captured.append(h)
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 10 and "language" in name.lower():
            last = mod[-1]
    if last is not None:
        last.register_forward_hook(_hk)
        print(f"[probe] hooked last layer: ...{name}[-1]")
        with torch.no_grad():
            model(inputs_embeds=inputs_embeds, attention_mask=am, position_ids=pos,
                  use_cache=False)
        if captured:
            gh = captured[0][0, Lt:Lt + Lg]   # geom-token hidden states
            print(f"  geom hidden: {tuple(gh.shape)} dtype={gh.dtype} "
                  f"mean={gh.float().mean():.3f}")

    print("\n=== TEST 2: B=2 UNIFORM-length inputs_embeds (RoPE check) ===")
    g2 = torch.randn(2, Lg, H, device=model.device, dtype=text_emb.dtype) * 0.02
    te2 = text_emb.expand(2, -1, -1)               # same text twice, uniform length
    ie2 = torch.cat([te2, g2], dim=1)              # (2, Lt+Lg, H) — uniform
    am2 = torch.ones(2, Lt + Lg, device=model.device, dtype=torch.long)
    pos2 = torch.arange(Lt + Lg, device=model.device).unsqueeze(0).expand(2, -1)
    try:
        with torch.no_grad():
            model(inputs_embeds=ie2, attention_mask=am2, position_ids=pos2, use_cache=False)
        print("  OK — batched uniform-length forward WORKS (RoPE fine)")
    except Exception as e:
        print("  batched FAILED (RoPE bug?):", repr(e))
        print("  → would need batch=1 (slow) or a modeling-code RoPE patch")

    print("\n=== TEST 3: B=2 PADDED (variable text len) inputs_embeds ===")
    ids2 = tok([chat, tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "table"}]}],
        add_generation_prompt=True, tokenize=False)],
        return_tensors="pt", padding=True)["input_ids"].to(model.device)
    with torch.no_grad():
        te_ = embed(ids2)
    Lp = te_.shape[1]
    g_ = torch.randn(2, Lg, H, device=model.device, dtype=te_.dtype) * 0.02
    ie_ = torch.cat([te_, g_], dim=1)
    am_ = torch.cat([torch.ones_like(ids2, dtype=torch.long),
                     torch.ones(2, Lg, device=model.device, dtype=torch.long)], dim=1)
    pos_ = torch.arange(Lp + Lg, device=model.device).unsqueeze(0).expand(2, -1)
    try:
        with torch.no_grad():
            model(inputs_embeds=ie_, attention_mask=am_, position_ids=pos_, use_cache=False)
        print("  OK — batched padded forward WORKS")
    except Exception as e:
        print("  batched padded FAILED:", repr(e))
    print("\nDONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/home/jovyan/h800fast/wangzekai/Step-3.7-Flash")
