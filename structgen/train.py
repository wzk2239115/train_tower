"""Training loop: backbone (frozen component) + voxel geometry decoder.

The backbone encodes text/sketch → condition tokens; the decoder learns the
SDF/occupancy field via rectified flow + multi-objective geometry losses.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from structgen.config import StructGenConfig
from structgen.data.dataset import StructGenDataset, collate_structgen
from structgen.model.backbone import build_backbone
from structgen.model.geometry_decoder import GeometryDecoder


def _lr_schedule(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    # cosine to 0
    prog = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def train(cfg: StructGenConfig, smoke_steps: int | None = None) -> str:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    backbone = build_backbone(cfg).to(device)
    decoder = GeometryDecoder(cfg.decoder).to(device)

    ds = StructGenDataset(
        grid_res=cfg.decoder.grid_res,
        surface_samples=cfg.surface_samples,
        num_samples=cfg.num_samples,
        image_size=cfg.backbone.image_size,
        seed=cfg.seed,
    )
    loader = DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_structgen, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # trainable params: decoder (all) + backbone projection (encoder frozen)
    params = list(decoder.parameters())
    params += [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    max_steps = smoke_steps or cfg.train.max_steps
    step = 0
    backbone.train()
    decoder.train()
    print(f"[structgen] device={device} decoder_params={sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")
    print(f"[structgen] backbone={cfg.backbone.kind} max_steps={max_steps} batch={cfg.train.batch_size} res={cfg.decoder.grid_res}")

    t0 = time.time()
    data_iter = iter(loader)
    running = {}
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        field = batch["field"].to(device)
        surface = batch["surface"].to(device)
        sketch = batch["sketch"].to(device)
        prompts = batch["prompt"]

        for g in opt.param_groups:
            g["lr"] = _lr_schedule(step, cfg.train.warmup_steps, max_steps, cfg.train.lr)

        opt.zero_grad(set_to_none=True)
        amp_enabled = cfg.train.amp and device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            cond = backbone(prompts, sketch=sketch)
            loss, logs = decoder.decode_loss(
                field, cond.pooled, cond.tokens, surface,
                cfg.loss_weights, cfg.flow,
            )
        loss.backward()
        if cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()

        for k, v in logs.items():
            running[k] = running.get(k, 0.0) + v
        step += 1
        if step % cfg.train.log_every == 0 or step == max_steps:
            avg = {k: v / cfg.train.log_every for k, v in running.items()}
            running.clear()
            lr = opt.param_groups[0]["lr"]
            dt = (time.time() - t0) / step
            msg = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(avg.items()) if k != "loss/total")
            print(f"step {step:5d}/{max_steps} lr={lr:.2e} {dt:.2f}s/it total={avg['loss/total']:.4f} {msg}")
        if step % cfg.train.save_every == 0 or step == max_steps:
            ckpt = os.path.join(cfg.train.out_dir, f"decoder_step{step}.pt")
            torch.save({"decoder": decoder.state_dict(),
                        "step": step, "cfg": asdict(cfg)}, ckpt)
            print(f"  saved {ckpt}")
    return os.path.join(cfg.train.out_dir, f"decoder_step{step}.pt")
