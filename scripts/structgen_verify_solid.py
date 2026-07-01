"""End-to-end solid-generation verification on the dev machine.

Trains proxy backbone + tiny decoder long enough to actually FIT the simple
synthetic data, then generates from a prompt and dumps:
  - the SDF field stats (solid_frac must be ~0.2-0.4 for a TPMS-in-box, NOT ~0.5 noise)
  - an ortho slice PNG (visual: structured vs fog)
  - the marched STL
If the slice shows structure (shell boundary + internal topology) and solid_frac
is reasonable, basic solid generation works. If it's noise, there's a real bug.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structgen.config import StructGenConfig, DecoderConfig, BackboneConfig, TrainConfig
from structgen.train import train
from structgen.infer import load_trained, generate
from structgen.data import sampler

os.makedirs("outputs/structgen/verify", exist_ok=True)
cfg = StructGenConfig(
    backbone=BackboneConfig(kind="proxy", image_size=112),
    decoder=DecoderConfig(grid_res=32, base_channels=32, channel_mults=(1, 2, 4),
                          num_blocks=1, field_channels=2, cross_attn=False),
    train=TrainConfig(batch_size=4, lr=2e-4, max_steps=800, warmup_steps=50,
                      log_every=100, save_every=800, device="cuda",
                      out_dir="outputs/structgen/verify"),
)

print("=== TRAIN (proxy, 800 steps) ===")
ckpt = train(cfg)

print("\n=== GENERATE (with the fix: backbone weights loaded from ckpt) ===")
device = torch.device("cuda")
bb, dec = load_trained(cfg, ckpt, device)
rp = sampler.all_recipes()[0]   # box + gyroid
field_arr, mesh = generate(bb, dec, cfg, [rp.prompt], device=device,
                           out_mesh="outputs/structgen/verify/gen.stl")
f = field_arr[0]
print(f"field shape={f.shape} sdf[{f.min():.3f},{f.max():.3f}] "
      f"solid_frac={float((f<0).mean()):.3f}")

# compare to a GT field for the same prompt
gt = sampler.build_field(rp, 32)
print(f"GT       shape={gt.shape} sdf[{gt.min():.3f},{gt.max():.3f}] "
      f"solid_frac={float((gt<0).mean()):.3f}")

# ortho slices: gen vs GT
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow((f < 0)[16], cmap="gray"); axes[0].set_title("gen occ (z mid)")
axes[1].imshow((gt < 0)[16], cmap="gray"); axes[1].set_title("GT occ (z mid)")
axes[2].imshow(f[16], cmap="RdBu"); axes[2].set_title("gen SDF (z mid)")
for a in axes: a.axis("off")
plt.tight_layout()
plt.savefig("outputs/structgen/verify/slices.png", dpi=80)
print("saved slices -> outputs/structgen/verify/slices.png")
print("mesh faces:", len(mesh) if mesh else None)
