"""GPU volume rendering via raymarching (torch-only, no OpenGL/display needed).

Renders a 3D occupancy volume from arbitrary viewing angles by:
  1. Shooting rays through each output pixel
  2. Marching along each ray, trilinearly sampling the volume (grid_sample)
  3. First-hit detection for surface extraction
  4. Gradient-based normal estimation + Phong-style shading

Usage:
    from structgen.viz.gpu_render import render_volume_views
    images = render_volume_views(occ, res=768)   # list of (H, W, 3) uint8
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def render_volume_views(
    occ: np.ndarray,
    views: list[tuple[float, float]] | None = None,
    res: int = 768,
    device: str = "cuda",
    n_samples: int = 320,
) -> list[np.ndarray]:
    """Render occupancy from multiple angles on GPU.

    Args:
        occ: (D, H, W) float array in [0, 1]
        views: list of (elev, azim) in degrees. Default: front/side/perspective.
        res: per-view image resolution.
        device: torch device.
        n_samples: raymarch steps (more = sharper surfaces).

    Returns:
        list of (res, res, 3) uint8 arrays.
    """
    if views is None:
        views = [(12, 35), (12, 125), (28, 245)]

    vol = torch.as_tensor(np.ascontiguousarray(occ), dtype=torch.float32, device=device)
    D, H, W = vol.shape
    vol5 = vol[None, None]  # (1, 1, D, H, W)

    # Precompute volume gradient (for surface normals) via central differences
    gx = F.pad(vol5[:, :, 2:, :, :] - vol5[:, :, :-2, :, :], (0, 0, 0, 0, 1, 1)) * 0.5
    gy = F.pad(vol5[:, :, :, 2:, :] - vol5[:, :, :, :-2, :], (0, 0, 1, 1, 0, 0)) * 0.5
    gz = F.pad(vol5[:, :, :, :, 2:] - vol5[:, :, :, :, :-2], (1, 1, 0, 0, 0, 0)) * 0.5
    grad5 = torch.cat([gx, gy, gz], dim=1)  # (1, 3, D, H, W)

    images = []
    for elev, azim in views:
        img = _raymarch_one(vol5, grad5, elev, azim, res, n_samples, device)
        images.append(img)
    return images


@torch.no_grad()
def _raymarch_one(vol5, grad5, elev, azim, res, n_samples, device):
    """Render one view. vol5=(1,1,D,H,W), grad5=(1,3,D,H,W)."""
    er, ar = np.radians(elev), np.radians(azim)
    dist = 3.8

    cam = torch.tensor(
        [dist * np.cos(er) * np.cos(ar),
         dist * np.sin(er),
         dist * np.cos(er) * np.sin(ar)],
        device=device, dtype=torch.float32,
    )
    fwd = F.normalize(-cam, dim=0)
    wup = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = F.normalize(torch.cross(fwd, wup, dim=0), dim=0)
    up = torch.cross(right, fwd, dim=0)

    # --- ray directions -------------------------------------------------------
    half_fov = np.radians(22)
    tan = np.tan(half_fov)
    py, px = torch.meshgrid(
        torch.linspace(1, -1, res, device=device),   # top → bottom
        torch.linspace(-1, 1, res, device=device),
        indexing="ij",
    )
    dirs = (fwd[None, None, :]
            + right[None, None, :] * (px * tan)[..., None]
            + up[None, None, :] * (py * tan)[..., None])
    dirs = F.normalize(dirs, dim=-1)  # (res, res, 3)

    # --- raymarch -------------------------------------------------------------
    # Volume spans [-1,1]^3; cam at distance ~3.8, so rays span [~1.2, ~6.8]
    tn, tf = 1.2, 6.8
    ts = torch.linspace(tn, tf, n_samples, device=device)
    pts = cam.view(1, 1, 1, 3) + dirs[None] * ts.view(-1, 1, 1, 1)
    # pts: (n_samples, res, res, 3)

    # Trilinear sample volume.
    # grid_sample 5D: grid[...,0]→W(last dim), grid[...,1]→H, grid[...,2]→D(first dim)
    # Our volume: dim0=X(world), dim1=Y, dim2=Z.
    # So grid coords = (pz, py, px) = pts.flip(-1)
    grid = pts.flip(-1).view(1, n_samples, res, res, 3)
    samp = F.grid_sample(
        vol5, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0]  # (n_samples, res, res)

    # --- first-hit surface ----------------------------------------------------
    hit = samp > 0.5
    has_hit = hit.any(dim=0)  # (res, res)
    first_idx = hit.float().argmax(dim=0)  # first True along ray
    first_idx = first_idx.clamp(0, n_samples - 1)

    yy, xx = torch.meshgrid(
        torch.arange(res, device=device),
        torch.arange(res, device=device),
        indexing="ij",
    )
    hit_pts = pts[first_idx, yy, xx]  # (res, res, 3)

    # --- normals from gradient ------------------------------------------------
    ngrid = hit_pts.flip(-1).view(1, 1, res, res, 3)
    normals = F.grid_sample(
        grad5, ngrid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, :, 0]  # (3, res, res)
    normals = F.normalize(normals, dim=0, eps=1e-6)

    # Flip normals to face camera
    view_dir = dirs.permute(2, 0, 1)  # (3, res, res)
    facing = (normals * view_dir).sum(0, keepdim=True)
    normals = torch.where(facing > 0, -normals, normals)

    # --- shading (Blinn-Phong-ish) -------------------------------------------
    light1 = F.normalize(torch.tensor([0.5, 0.85, 0.3], device=device), dim=0)
    light2 = F.normalize(torch.tensor([-0.4, 0.4, -0.6], device=device), dim=0)

    diff1 = torch.clamp((normals * light1.view(3, 1, 1)).sum(0), 0, 1)
    diff2 = torch.clamp((normals * light2.view(3, 1, 1)).sum(0), 0, 1)

    # Specular highlight from light1
    half_v = F.normalize(light1 - fwd, dim=0)
    spec = torch.clamp((normals * half_v.view(3, 1, 1)).sum(0), 0, 1) ** 32

    intensity = 0.18 + diff1 * 0.55 + diff2 * 0.18 + spec * 0.35

    # Warm material color
    base = torch.tensor([0.82, 0.72, 0.58], device=device)
    rgb = intensity[None] * base.view(3, 1, 1)  # (3, res, res)
    rgb = rgb * has_hit[None].float()

    # Subtle depth darkening: farther surfaces slightly darker
    depth = first_idx.float() / n_samples
    depth_factor = 1.0 - depth * 0.15
    rgb = rgb * depth_factor[None]

    rgb = rgb.clamp(0, 1).permute(1, 2, 0)  # (res, res, 3)
    return (rgb.cpu().numpy() * 255).astype(np.uint8)
