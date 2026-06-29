"""Inference: text/sketch → voxel field → mesh → STL/OBJ."""

from __future__ import annotations

import os

import torch
from PIL import Image

from structgen.config import StructGenConfig
from structgen.data.dataset import sketch_to_tensor
from structgen.model.backbone import build_backbone
from structgen.model.geometry_decoder import GeometryDecoder


def load_trained(cfg: StructGenConfig, ckpt_path: str, device: torch.device):
    """Load decoder; prefer the decoder config saved in the checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = state.get("cfg") or {}
    dec_cfg = saved_cfg.get("decoder")
    if isinstance(dec_cfg, dict):
        # adopt shape-affecting fields that must match the saved weights
        cfg.decoder.grid_res = dec_cfg.get("grid_res", cfg.decoder.grid_res)
        cfg.decoder.field_channels = dec_cfg.get("field_channels", cfg.decoder.field_channels)
        cfg.decoder.base_channels = dec_cfg.get("base_channels", cfg.decoder.base_channels)
        cfg.decoder.channel_mults = tuple(dec_cfg.get("channel_mults", cfg.decoder.channel_mults))
        cfg.decoder.num_blocks = dec_cfg.get("num_blocks", cfg.decoder.num_blocks)
    backbone = build_backbone(cfg).to(device)
    decoder = GeometryDecoder(cfg.decoder).to(device)
    decoder.load_state_dict(state["decoder"])
    decoder.eval()
    backbone.eval()
    return backbone, decoder


def generate(backbone, decoder, cfg: StructGenConfig, prompt: str,
             sketch: torch.Tensor | None = None, device: torch.device | None = None,
             out_mesh: str | None = None, n_sample_steps: int | None = None):
    """Generate a structural part. Returns (field np.ndarray, Mesh | None)."""
    device = device or next(decoder.parameters()).device
    if isinstance(prompt, str):
        prompt = [prompt]
    if sketch is not None and sketch.ndim == 3:
        sketch = sketch.unsqueeze(0).to(device)
    elif sketch is not None:
        sketch = sketch.to(device)
    with torch.no_grad():
        cond = backbone(prompt, sketch=sketch)
        flow = cfg.flow
        if n_sample_steps is not None:
            from dataclasses import replace
            flow = replace(flow, n_sample_steps=n_sample_steps)
        field = decoder.sample(cond.pooled, cond.tokens, flow, device=device)
    field_sdf = field[:, 0].float().cpu().numpy()  # (B,D,D,D)
    mesh = None
    if out_mesh is not None:
        from structgen.model.meshing import sdf_to_mesh, export_mesh
        mesh = sdf_to_mesh(field_sdf[0])
        if mesh is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_mesh)) or ".", exist_ok=True)
            export_mesh(mesh, out_mesh)
            print(f"[structgen] mesh exported: {out_mesh} ({len(mesh)} faces)")
    return field_sdf, mesh


def sketch_from_path(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return sketch_to_tensor(img, image_size)
