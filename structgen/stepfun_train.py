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
import torch.distributed as dist
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
        # PAD to a FIXED length so every rank's forward has identical (B,S,H)
        # — EP's MoE all_reduce requires all ranks reduce same-shape tensors.
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = ids + [pad] * (self.max_len - len(ids))
        return {"occ": torch.from_numpy(occ).float(), "ids": torch.tensor(ids)}


def _collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    ids = pad_sequence([b["ids"] for b in batch], batch_first=True, padding_value=0)
    lens = torch.tensor([b["ids"].numel() for b in batch])
    return {"occ": torch.stack([b["occ"] for b in batch])[:, None],
            "ids": ids, "lens": lens}


def train_ep(cfg: StructGenConfig, vae_path, stepfun_path, nrrd_dir, captions_csv,
             steps=5000, batch=2, out="outputs/structgen/sgen_ep.pt",
             dec_dim=1024, dec_blocks=12):
    """Expert-parallel training: 8 ranks, each holds 36 experts (all GPUs busy),
    decoder is DDP'd. Launched via `torchrun --nproc_per_node=8`."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DistributedSampler
    from structgen.ep_stepfun import load_stepfun_ep
    from structgen.model.stepfun_gen import StepfunGenNet, _find_embed, _find_llm_layers

    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    R = cfg.decoder.grid_res
    L, C = cfg.decoder.latent_res, cfg.decoder.latent_ch

    vae = VoxelVAE(R, L, C, base=cfg.decoder.vae_base).to(dev)
    vae.load_state_dict(torch.load(vae_path, map_location=dev)["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    sf = load_stepfun_ep(stepfun_path)
    net = StepfunGenNet(sf, _find_embed(sf), _find_llm_layers(sf), C, L,
                        dec_dim=dec_dim, dec_blocks=dec_blocks)
    # move ALL tracked params to this rank's GPU (incl. frozen geom_in/t_embed_sf),
    # then DDP only the trainable decoder. The frozen Stepfun is already on dev.
    for p in net.parameters():
        p.data = p.data.to(dev)
    net.decoder = DDP(net.decoder, device_ids=[local])

    import transformers as _tf
    tok = _tf.AutoTokenizer.from_pretrained(stepfun_path, trust_remote_code=True)
    ds = _TextOcc(nrrd_dir, captions_csv, R, tok)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=2,
                        collate_fn=_collate, drop_last=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    if rank == 0:
        ntrain = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        print(f"[ep-train] world={world}, decoder trainable {ntrain:.1f}M, "
              f"{len(ds)} shapes")

    it = iter(loader)
    step = 0
    t0 = time.time()
    while step < steps:
        try:
            b = next(it)
        except StopIteration:
            sampler.set_epoch(step // len(loader) + 1)
            it = iter(loader)
            b = next(it)
        with torch.no_grad():
            mu, _ = vae.enc(b["occ"].to(dev))
            gt = mu
        t = torch.rand(gt.shape[0], device=dev).clamp(0.02, 0.98)
        noise = torch.randn_like(gt)
        z = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * gt
        feats = net.extract(z, t, b["ids"])                       # EP forward (frozen)
        x0 = net.decoder(feats, t).reshape(gt.shape[0], L, L, L, C).permute(0, 4, 1, 2, 3)
        loss = F.mse_loss(x0, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if rank == 0 and (step % 20 == 0 or step == steps):
            print(f"[ep-train] step {step}/{steps} mse={loss.item():.4f} "
                  f"dt={(time.time()-t0)/step:.2f}s/it")
        if rank == 0 and (step % 500 == 0 or step == steps):
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            dec = net.decoder.module if hasattr(net.decoder, "module") else net.decoder
            torch.save({"decoder": dec.state_dict(),
                        "geom_in": net.geom_in.state_dict(),
                        "dec_dim": dec_dim, "dec_blocks": dec_blocks,
                        "vae_path": vae_path, "cfg": asdict(cfg)},
                       out)
            print(f"  saved {out}")
    return out


def train(cfg: StructGenConfig, vae_path, stepfun_path, nrrd_dir, captions_csv,
          steps=5000, batch=2, out="outputs/structgen/stepfun_gen.pt",
          dec_dim=1024, dec_blocks=12):
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
    net = StepfunGenNet(sf, embed, llm_layers, C, L,
                        dec_dim=dec_dim, dec_blocks=dec_blocks).to(dev)
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
def generate_ep(cfg: StructGenConfig, vae_path, ckpt_path, stepfun_path, prompt,
                cfg_scale=3.0, n_steps=30, out="outputs/structgen/gen_ep.stl"):
    """EP generation: torchrun --nproc=8, each rank runs frozen Stepfun EP forward
    (all GPUs busy), shared decoder, rank 0 saves the STL."""
    from structgen.ep_stepfun import load_stepfun_ep
    from structgen.model.meshing import occupancy_to_mesh, export_mesh

    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    R = cfg.decoder.grid_res
    L, C = cfg.decoder.latent_res, cfg.decoder.latent_ch

    vae = VoxelVAE(R, L, C, base=cfg.decoder.vae_base).to(dev)
    vae.load_state_dict(torch.load(vae_path, map_location=dev)["vae"])
    vae.eval()

    sf = load_stepfun_ep(stepfun_path)
    net = StepfunGenNet(sf, _find_embed(sf), _find_llm_layers(sf), C, L)
    for p in net.parameters():
        p.data = p.data.to(dev)
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    dec_sd = {k.replace("module.", ""): v for k, v in state["decoder"].items()}
    net.decoder.load_state_dict(dec_sd)
    net.geom_in.load_state_dict(state["geom_in"])
    net.eval()

    import transformers as _tf
    tok = _tf.AutoTokenizer.from_pretrained(stepfun_path, trust_remote_code=True)
    pad = tok.pad_token_id or 0
    max_len = 48

    def _ids(text):
        ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
        return torch.tensor([ids + [pad] * (max_len - len(ids))])

    ids = _ids(prompt)
    null_ids = _ids("")
    z = torch.randn(1, C, L, L, L, device=dev)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        tt = torch.full((1,), i * dt, device=dev)
        fc = net.decoder(net.extract(z, tt, ids), tt).reshape(1, L, L, L, C).permute(0, 4, 1, 2, 3)
        fu = net.decoder(net.extract(z, tt, null_ids), tt).reshape(1, L, L, L, C).permute(0, 4, 1, 2, 3)
        x0 = fu + cfg_scale * (fc - fu)
        z = z + (x0 - z) * dt
    occ = (torch.sigmoid(vae.dec(z)) > 0.5)[0, 0].float().cpu().numpy()
    if rank == 0:
        print(f"[ep-gen] solid_frac={occ.mean():.3f}")
        mesh = occupancy_to_mesh(occ)
        if mesh is not None:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            export_mesh(mesh, out)
            print(f"[ep-gen] mesh {len(mesh)} faces -> {out}")


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
    state = torch.load(ckpt_path, map_location=dev, weights_only=False)
    dec_dim = state.get("dec_dim", 1024)
    dec_blocks = state.get("dec_blocks", 12)
    net = StepfunGenNet(sf, _find_embed(sf), _find_llm_layers(sf), C, L,
                        dec_dim=dec_dim, dec_blocks=dec_blocks).to(dev)
    if "decoder" in state:                       # EP checkpoint
        dec_sd = {k.replace("module.", ""): v for k, v in state["decoder"].items()}
        net.decoder.load_state_dict(dec_sd)
        net.geom_in.load_state_dict(state["geom_in"])
    else:                                        # legacy single-process ckpt
        net.load_state_dict(state["net"])
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
