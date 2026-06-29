"""Synthetic structural-part dataset (field grid + surface pts + prompt + sketch)."""

from __future__ import annotations


import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from structgen.data import sampler
from structgen.data.sampler import SpecimenParams


def _render_sketch(field: np.ndarray, size: int = 128) -> Image.Image:
    """Cheap sketch: 3 orthographic projections of the occupancy, stacked."""
    occ = (field < 0).astype(np.float32)
    proj = [occ.mean(axis=i) for i in (0, 1, 2)]  # 3 gray projections
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for c, pr in enumerate(proj):
        img = np.clip(pr * 255, 0, 255).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize((size, size)))
        canvas[..., c] = img
    return Image.fromarray(canvas)


def sketch_to_tensor(img: Image.Image, image_size: int = 224) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W


class StructGenDataset(Dataset):
    """Yields synthetic specimens with their SDF field, surface pts, prompt, sketch.

    Each ``__getitem__`` returns a dict:
        field:   (C, D, H, W)  — [sdf, occupancy] on the voxel grid
        surface: (S, 3)        — GT surface points for Chamfer
        prompt:  str
        sketch:  (3, image_size, image_size)  — rendered ortho sketch
        params:  SpecimenParams
    """

    def __init__(self, grid_res: int = 64, surface_samples: int = 2048,
                 num_samples: int = 2048, image_size: int = 224,
                 smooth: float = 0.1, seed: int = 0, augment: bool = True):
        super().__init__()
        self.grid_res = grid_res
        self.surface_samples = surface_samples
        self.image_size = image_size
        self.smooth = smooth
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        # recipe bank is the "class" of structures; we index into it (with
        # random parameter jitter) to produce ``num_samples`` items
        self.recipes = sampler.all_recipes()
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def _jittered_recipe(self, idx: int) -> SpecimenParams:
        base = self.recipes[idx % len(self.recipes)]
        rp = SpecimenParams(**{**base.__dict__})
        kw = dict(rp.topology_kw)
        # jitter period / thickness for variety while staying manufacturable
        if "period" in kw:
            kw["period"] = float(np.clip(kw["period"] * self._rng.uniform(0.9, 1.1), 0.4, 1.2))
        if "thickness" in kw:
            kw["thickness"] = float(np.clip(kw["thickness"] * self._rng.uniform(0.85, 1.15), 0.06, 0.2))
        if "strut_r" in kw:
            kw["strut_r"] = float(np.clip(kw["strut_r"] * self._rng.uniform(0.9, 1.1), 0.03, 0.12))
        if "cells" in kw:
            kw["seed"] = int(self._rng.integers(0, 1 << 30))
        rp.topology_kw = kw
        return rp

    def __getitem__(self, idx: int) -> dict:
        rp = self._jittered_recipe(idx)
        field = sampler.build_field(rp, self.grid_res, smooth=self.smooth)
        occ = sampler.occupancy_from_sdf(field)
        field_ch = np.stack([field, occ], axis=0)  # (2,D,D,D)
        surf = sampler.sample_surface(field, self.surface_samples,
                                      rng=self._rng)
        sketch = _render_sketch(field, size=self.image_size)
        sketch_t = sketch_to_tensor(sketch, self.image_size)
        return {
            "field": torch.from_numpy(field_ch).float(),
            "surface": torch.from_numpy(surf).float(),
            "prompt": rp.prompt,
            "sketch": sketch_t,
            "params": rp,
        }


def collate_structgen(batch: list[dict]) -> dict:
    """Collate; prompts kept as list (variable length), params as list."""
    out = {
        "field": torch.stack([b["field"] for b in batch]),
        "surface": torch.stack([b["surface"] for b in batch]),
        "sketch": torch.stack([b["sketch"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
        "params": [b["params"] for b in batch],
    }
    return out
