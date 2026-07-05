"""Train + sample the Stepfun-as-denoiser (geometry latent flows THROUGH Stepfun).

Stage: VAE must already be trained (occ<->latent). Then this trains the
geom-token projection + read-out head so that injecting the noisy latent into
Stepfun and reading its hidden states predicts the clean latent (x0). Stepfun
runs every step (its computation participates in generation), weights frozen.

Run on the compute box (8xH800) — Stepfun is loaded device_map=auto across GPUs.
"""
from __future__ import annotations

import csv
import glob
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from structgen.config import StructGenConfig
from structgen.model.stepfun_gen import StepfunGenNet, _build_stepfun, _find_embed, _find_llm_layers
from structgen.model.vae import VoxelVAE
from structgen.data.nrrd import read_nrrd_occ


class _TextOcc(Dataset):
    def __init__(self, nrrd_dir, captions_csv, res, tokenizer, max_len=48):
        desc = {}
        with open(captions_csv) as f:
            for r in csv.DictReader(f):
                desc[r["modelId"]] = r["description"]
        self.items = []
        for n in sorted(glob.glob(os.path.join(nrrd_dir, "*.nrrd"))):
            mid = os.path.basename(n)[:-5]
            if mid in desc:
                self.items.append((desc[mid], n))   # (caption, path) — lazy read
        self.res = res
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cap, path = self.items[i]
        occ = read_nrrd_occ(path, self.res)
        ids = self.tok(cap, truncation=True, max_length=self.max_len)["input_ids"]
        return {"occ": torch.from_numpy(occ).float(), "ids": torch.tensor(ids)}


def _collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    ids = pad_sequence([b["ids"] for b in batch], batch_first=True, padding_value=0)
    lens = torch.tensor([b["ids"].numel() for b in batch])
    return {"occ": torch.stack([b["occ"] for b in batch])[:, None],
            "ids": ids, "lens": lens}


def train(cfg: StructGenConfig, vae_path, stepfun_path, nrrd_dir, captions_csv,
          steps=5000, batch=2, out="outputs/structgen/stepfun_gen.pt"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = cfg.decoder.grid_res
    L, C = cfg.decoder.latent_res, cfg.decoder.latent_ch

    # frozen VAE
    vae = VoxelVAE(R, L, C, base=cfg.decoder.vae_base).to(dev)
    vae.load_state_dict(torch.load(vae_path, map_location=dev)["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    n_gpu = torch.cuda.device_count() or 1
    # 8xH800-80GB fully free: give ~76GiB/GPU (weights ~50 + activations).
    # GPU0 keeps a bit more headroom for the trainable proj+head + moved hidden.
    max_mem = {0: "68GiB"}
    for i in range(1, n_gpu):
        max_mem[i] = "76GiB"
    print("[sgen] loading Stepfun (device_map=auto across GPUs)...")
    sf = _build_stepfun(stepfun_path, max_mem)
    embed = _find_embed(sf)
    llm_layers = _find_llm_layers(sf)
    net = StepfunGenNet(sf, embed, llm_layers, C, L).to(dev)
    ntrain = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    print(f"[sgen] Stepfun on {n_gpu} GPU(s); trainable params {ntrain:.2f}M "
          f"(proj+head; 150B frozen)")

    import transformers as _tf
    tok = _tf.AutoTokenizer.from_pretrained(stepfun_path, trust_remote_code=True)
    ds = _TextOcc(nrrd_dir, captions_csv, R, tok)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2,
                        collate_fn=_collate, drop_last=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)

    it = iter(loader)
    step = 0
    t0 = time.time()
    while step < steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        with torch.no_grad():
            mu, _ = vae.enc(b["occ"].to(dev))
            gt = mu                                   # latent target
        t = torch.rand(gt.shape[0], device=dev).clamp(0.02, 0.98)
        noise = torch.randn_like(gt)
        z = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * gt
        x0 = net(z, t, b["ids"], b["lens"])
        loss = F.mse_loss(x0, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 20 == 0 or step == steps:
            print(f"[sgen] step {step}/{steps} mse={loss.item():.4f} "
                  f"dt={(time.time()-t0)/step:.2f}s/it")
        if step % 1000 == 0 or step == steps:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            torch.save({"net": net.state_dict(), "vae_path": vae_path,
                        "cfg": asdict(cfg)}, out)
            print(f"  saved {out}")
    return out


@torch.no_grad()
def generate(cfg: StructGenConfig, vae_path, ckpt_path, stepfun_path, prompt,
             cfg_scale=3.0, n_steps=30):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = cfg.decoder.grid_res
    L, C = cfg.decoder.latent_res, cfg.decoder.latent_ch
    vae = VoxelVAE(R, L, C, base=cfg.decoder.vae_base).to(dev)
    vae.load_state_dict(torch.load(vae_path, map_location=dev)["vae"])
    vae.eval()

    n_gpu = torch.cuda.device_count() or 1
    max_mem = {0: "68GiB"}
    for i in range(1, n_gpu):
        max_mem[i] = "76GiB"
    sf = _build_stepfun(stepfun_path, max_mem)
    net = StepfunGenNet(sf, _find_embed(sf), _find_llm_layers(sf), C, L).to(dev)
    net.load_state_dict(torch.load(ckpt_path, map_location=dev)["net"])
    net.eval()
    import transformers as _tf
    tok = _tf.AutoTokenizer.from_pretrained(stepfun_path, trust_remote_code=True)

    ids = tok([prompt], truncation=True, max_length=48)["input_ids"]
    ids = torch.tensor(ids)
    # null (unconditional) prompt = empty ids
    null_ids = torch.tensor([[tok.eos_token_id or 0]])
    z = torch.randn(1, C, L, L, L, device=dev)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        tt = torch.full((1,), i * dt, device=dev)
        x0c = net(z, tt, ids, torch.tensor([ids.shape[1]]))
        x0u = net(z, tt, null_ids, torch.tensor([1]))
        x0 = x0u + cfg_scale * (x0c - x0u)
        z = z + (x0 - z) * dt
    occ = (torch.sigmoid(vae.dec(z)) > 0.5)[0, 0].float().cpu().numpy()
    print(f"[sgen-gen] solid_frac={occ.mean():.3f}")
    return occ
