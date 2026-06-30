"""Convert a triangle mesh to an SDF voxel grid + surface points/normals.

Pipeline (offline, per ABC model):
  1. normalize mesh into the cube [-1, 1]
  2. sample surface points (area-weighted) carrying face normals
  3. unsigned distance from each grid point → nearest surface point (cKDTree)
  4. sign from the nearest face normal (dot(q-p, n)) — robust for closed meshes
  5. SDF = sign * distance; occupancy = sdf < 0

This is the standard "closest face normal" SDF approximation used by many
mesh→SDF toolkits; good enough for CAD parts (watertight-ish).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from structgen.data.stl_io import read_stl


def normalize_mesh(verts: np.ndarray, faces: np.ndarray):
    """Center + uniformly scale verts into [-1, 1]. Returns (verts, faces)."""
    center = (verts.min(0) + verts.max(0)) * 0.5
    v = verts - center
    extent = (v.max(0) - v.min(0)).max()
    if extent < 1e-8:
        extent = 1.0
    v = v / (extent * 0.5)  # → [-1, 1]
    return v.astype(np.float32), faces


def _sample_surface(verts, faces, normals, n_samples, rng):
    """Area-weighted surface samples with per-sample face normals."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    # triangle area (twice, but we only need relative weights)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = np.clip(area, 1e-12, None)
    p_face = area / area.sum()
    idx = rng.choice(len(faces), size=n_samples, replace=True, p=p_face)
    r1 = rng.random(n_samples)[:, None]
    r2 = rng.random(n_samples)[:, None]
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    w = sqrt_r1 * (1 - r2)
    vv = sqrt_r1 * r2
    pts = a[idx] * u + b[idx] * w + c[idx] * vv
    nrm = normals[idx]
    # recompute exact normals for robustness if the STL normals are zero
    bad = np.linalg.norm(nrm, axis=1) < 1e-6
    if bad.any():
        cross = np.cross(b[idx][bad] - a[idx][bad], c[idx][bad] - a[idx][bad])
        cn = np.linalg.norm(cross, axis=1, keepdims=True)
        cn[cn < 1e-12] = 1.0
        nrm[bad] = cross / cn
    return pts.astype(np.float32), nrm.astype(np.float32)


def mesh_to_sdf(path: str, grid_res: int = 64, n_surf: int = 4096,
                bounds=(-1.0, 1.0), seed: int = 0):
    """Return dict(field [R,R,R] f32, surface [S,3], normals [S,3], ok bool)."""
    rng = np.random.default_rng(seed)
    verts, faces, normals = read_stl(path)
    if len(verts) < 3 or len(faces) < 1:
        return {"field": None, "surface": None, "normals": None, "ok": False}
    verts, faces = normalize_mesh(verts, faces)

    surf_pts, surf_nrm = _sample_surface(verts, faces, normals, n_surf, rng)
    tree = cKDTree(surf_pts)

    lin = np.linspace(bounds[0], bounds[1], grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    dist, nn = tree.query(grid_pts, k=1)            # nearest surface distance
    nearest_pt = surf_pts[nn]
    nearest_n = surf_nrm[nn]
    sign = np.sign(np.einsum("ij,ij->i", grid_pts - nearest_pt, nearest_n))
    sign[sign == 0] = 1.0
    sdf = (sign * dist).astype(np.float32).reshape(grid_res, grid_res, grid_res)

    solid_frac = float((sdf < 0).mean())
    ok = 0.01 < solid_frac < 0.99
    return {"field": sdf, "surface": surf_pts, "normals": surf_nrm,
            "solid_frac": solid_frac, "ok": ok}


def prompt_from_stats(field: np.ndarray, verts: np.ndarray) -> str:
    """Synthesize a coarse text prompt from shape statistics (no metadata)."""
    extent = verts.max(0) - verts.min(0)
    extent = extent / max(extent.max(), 1e-8)
    long = extent[0] > 1.6 * extent[1]
    flat = extent[2] < 0.5 * extent[0]
    solid = float((field < 0).mean())
    shape = "long slender" if long else ("thin flat plate" if flat else "compact blocky")
    fill = "solid" if solid > 0.45 else ("hollow / shell-like")
    return (f"A mechanical CAD part, {shape} and {fill}; "
            f"smooth machinable surfaces, 3D-printable envelope.")
