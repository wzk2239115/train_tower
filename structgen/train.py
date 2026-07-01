"""Training loop: backbone (frozen component) + voxel geometry decoder.

Single-GPU: ``python -m structgen.cli train ...``
Multi-GPU (DDP, uses all 8 cards for the decoder):
    torchrun --nproc_per_node=8 -m structgen.cli train --backbone cached \
        --text-emb outputs/structgen/text_emb.pt ...
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from structgen.config import StructGenConfig
from structgen.data.dataset import StructGenDataset, collate_structgen
from structgen.model.backbone import build_backbone
from structgen.model.geometry_decoder import GeometryDecoder


def _lr_schedule(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def train(cfg: StructGenConfig, smoke_steps: int | None = None) -> str:
    ddp = _is_ddp()
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if ddp else 0

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if ddp else cfg.train.device)
        if ddp:
            torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")
    torch.manual_seed(cfg.seed + rank)
    is_main = rank == 0
    if is_main:
        os.makedirs(cfg.train.out_dir, exist_ok=True)

    backbone = build_backbone(cfg).to(device)
    decoder = GeometryDecoder(cfg.decoder).to(device)
    if ddp:
        backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)
        decoder = DDP(decoder, device_ids=[local_rank])

    def _bb(m):
        return m.module if isinstance(m, DDP) else m

    ds = StructGenDataset(
        grid_res=cfg.decoder.grid_res,
        surface_samples=cfg.surface_samples,
        num_samples=cfg.num_samples,
        image_size=cfg.backbone.image_size,
        seed=cfg.seed,
        real_data_dir=cfg.real_data_dir,
    )
    if ddp:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=True, seed=cfg.seed)
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, sampler=sampler,
                            num_workers=2, collate_fn=collate_structgen,
                            drop_last=True)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                            num_workers=2, collate_fn=collate_structgen,
                            drop_last=True,
                            generator=torch.Generator().manual_seed(cfg.seed))

    params = list(decoder.parameters())
    params += [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    max_steps = smoke_steps or cfg.train.max_steps
    backbone.train()
    decoder.train()
    if is_main:
        print(f"[structgen] device={device} ddp={ddp} world={world} "
              f"decoder_params={sum(p.numel() for p in _bb(decoder).parameters())/1e6:.2f}M")
        print(f"[structgen] backbone={cfg.backbone.kind} max_steps={max_steps} "
              f"batch/gpu={cfg.train.batch_size} global_batch={cfg.train.batch_size * world} "
              f"res={cfg.decoder.grid_res}")

    t0 = time.time()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)
    running = {}
    step = 0
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
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
            loss, logs = decoder(
                field, cond.pooled, cond.tokens, surface,
                cfg.loss_weights, cfg.flow,
            )
        loss.backward()
        if cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        opt.step()

        if is_main:
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
        step += 1
        if is_main and (step % cfg.train.log_every == 0 or step == max_steps):
            avg = {k: v / cfg.train.log_every for k, v in running.items()}
            running.clear()
            lr = opt.param_groups[0]["lr"]
            dt = (time.time() - t0) / step
            msg = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(avg.items())
                           if k != "loss/total")
            print(f"step {step:5d}/{max_steps} lr={lr:.2e} {dt:.2f}s/it "
                  f"total={avg['loss/total']:.4f} {msg}")
        if is_main and (step % cfg.train.save_every == 0 or step == max_steps):
            ckpt = os.path.join(cfg.train.out_dir, f"decoder_step{step}.pt")
            # Save BOTH the decoder AND the trainable condition-encoder params
            # (text projection + sketch CNN). The frozen 198B is NOT in here
            # (it's stored untracked in StepfunBackbone). Without these, the
            # decoder receives random conditions at generation → fog output.
            torch.save({
                "decoder": _bb(decoder).state_dict(),
                "backbone": _bb(backbone).state_dict(),
                "step": step, "cfg": asdict(cfg),
            }, ckpt)
            print(f"  saved {ckpt}")
        if ddp:
            dist.barrier()
    return os.path.join(cfg.train.out_dir, f"decoder_step{step}.pt")
