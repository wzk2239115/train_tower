"""Implicit SDF field synthesis for complex surface / internal topology.

Everything is defined as a *signed distance field* sampled on a regular 3D
grid. Two families are combined with CSG (union / intersection / smooth-blend):

1. **Shell** — the outer manufacturable envelope (box, cylinder, sphere, blended
   superellipsoid, sketch-revolved profiles, ...).
2. **Internal topology** — TPMS (Gyroid, Diamond, Schwarz-P), Voronoi struts,
   lattice (strut/box), and graded-density variants.

A structural part = ``shell`` intersected with the *solid* region of an
internal topology (so the TPMS/lattice lives *inside* the shell). The field
returned is a proper SDF: negative inside the solid, positive outside.

All functions are numpy-only and operate on a flattened ``(N, 3)`` array of
sample points in the canonical cube ``[-1, 1]^3``; callers reshape to the grid.
"""

from __future__ import annotations

import numpy as np

_rng = np.random.default_rng()

# --------------------------------------------------------------------------- #
# Small SDF primitives (normalized-ish approximations; exact for basic shapes)
# --------------------------------------------------------------------------- #


def sdf_sphere(p: np.ndarray, r: float = 0.9) -> np.ndarray:
    return np.linalg.norm(p, axis=-1) - r


def sdf_box(p: np.ndarray, half: tuple[float, float, float] = (0.8, 0.8, 0.8)) -> np.ndarray:
    q = np.abs(p) - np.array(half, dtype=p.dtype)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def sdf_cylinder(p: np.ndarray, r: float = 0.7, h: float = 0.8, axis: int = 2) -> np.ndarray:
    a1, a2 = (axis + 1) % 3, (axis + 2) % 3
    radial = np.sqrt(p[..., a1] ** 2 + p[..., a2] ** 2) - r
    ax = np.abs(p[..., axis]) - h
    q = np.stack([np.maximum(radial, 0.0), np.maximum(ax, 0.0)], axis=-1)
    return np.linalg.norm(q, axis=-1) + np.minimum(np.maximum(radial, ax), 0.0)


def sdf_superellipsoid(p: np.ndarray, a: float = 0.8, b: float = 0.8, c: float = 0.8,
                       e: float = 0.6, n: float = 0.6) -> np.ndarray:
    """Bloch-style superellipsoid shell (curved, non-trivial surface)."""
    x, y, z = p[..., 0] / a, p[..., 1] / b, p[..., 2] / c
    # inside/outside via the implicit form; SDF approximated by scaling gradient
    f = (np.abs(x) ** (2.0 / e) + np.abs(y) ** (2.0 / n)) ** (e * n / (2.0 * n)) + np.abs(z) ** (2.0 / n) - 1.0
    # crude radial scaling → a Lipschitz-ish distance estimate
    return f * 0.5


# --------------------------------------------------------------------------- #
# Boolean / smooth operations on SDFs
# --------------------------------------------------------------------------- #


def op_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def op_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def op_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return op_intersect(a, -b)


def op_smooth_union(a: np.ndarray, b: np.ndarray, k: float = 0.15) -> np.ndarray:
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (1.0 - h) * b + h * a - k * h * (1.0 - h)


def op_smooth_intersect(a: np.ndarray, b: np.ndarray, k: float = 0.15) -> np.ndarray:
    h = np.clip(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return (1.0 - h) * b + h * a + k * h * (1.0 - h)


# --------------------------------------------------------------------------- #
# TPMS (Triply Periodic Minimal Surfaces)
# --------------------------------------------------------------------------- #


def tpms_gyroid(p: np.ndarray, period: float = 1.0, thickness: float = 0.15) -> np.ndarray:
    """Gyroid. Solid where |F| < thickness (a wall of given half-thickness)."""
    s = 2.0 * np.pi / period
    x, y, z = p[..., 0] * s, p[..., 1] * s, p[..., 2] * s
    f = (np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x))
    # distance ~ |f|/|grad f|; use |f|/6 as a cheap Lipschitz-scaled estimate
    return np.abs(f) / 6.0 - thickness


def tpms_diamond(p: np.ndarray, period: float = 1.0, thickness: float = 0.15) -> np.ndarray:
    s = 2.0 * np.pi / period
    x, y, z = p[..., 0] * s, p[..., 1] * s, p[..., 2] * s
    f = (np.sin(x) * np.sin(y) * np.sin(z)
         + np.sin(x) * np.cos(y) * np.cos(z)
         + np.cos(x) * np.sin(y) * np.cos(z)
         + np.cos(x) * np.cos(y) * np.sin(z))
    return np.abs(f) / 6.0 - thickness


def tpms_schwarz_p(p: np.ndarray, period: float = 1.0, thickness: float = 0.15) -> np.ndarray:
    s = 2.0 * np.pi / period
    x, y, z = p[..., 0] * s, p[..., 1] * s, p[..., 2] * s
    f = np.cos(x) + np.cos(y) + np.cos(z)
    return np.abs(f) / 3.0 - thickness


_TPMS = {
    "gyroid": tpms_gyroid,
    "diamond": tpms_diamond,
    "schwarz_p": tpms_schwarz_p,
}


def graded_tpms(p: np.ndarray, kind: str = "gyroid", period: float = 1.0,
                thickness: float = 0.15, grade: float = 0.08) -> np.ndarray:
    """Density-graded TPMS: wall thickness varies along z (heavier near base)."""
    # thickness grows linearly with height → denser/thicker toward bottom
    t = thickness + grade * (0.5 - p[..., 2])  # z in [-1,1]
    # sample base field but re-threshold per point
    s = 2.0 * np.pi / period
    x, y, z = p[..., 0] * s, p[..., 1] * s, p[..., 2] * s
    if kind == "gyroid":
        f = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    elif kind == "diamond":
        f = (np.sin(x) * np.sin(y) * np.sin(z) + np.sin(x) * np.cos(y) * np.cos(z)
             + np.cos(x) * np.sin(y) * np.cos(z) + np.cos(x) * np.cos(y) * np.sin(z))
    else:
        f = np.cos(x) + np.cos(y) + np.cos(z)
    scale = 6.0 if kind != "schwarz_p" else 3.0
    return np.abs(f) / scale - t


# --------------------------------------------------------------------------- #
# Lattice / Voronoi struts
# --------------------------------------------------------------------------- #


def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Deterministic per-cell pseudo-random in [0,1)."""
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647) & 0xFFFFFFFF
    return (h * 2654435761 & 0xFFFFFFFF) / 4294967295.0


def voronoi_struts_sdf(p: np.ndarray, cells: int = 4, strut_r: float = 0.06,
                       seed: int = 0) -> np.ndarray:
    """Voronoi-edge-like strut network (approx): nearest-node distance field."""
    n = cells
    # generate a fixed jittered grid of nodes in [-1,1]^3
    g = np.linspace(-1.0 + 1.0 / n, 1.0 - 1.0 / n, n)
    GX, GY, GZ = np.meshgrid(g, g, g, indexing="ij")
    rng = np.random.default_rng(seed)
    jx = (rng.random((n, n, n)) - 0.5) * (1.6 / n)
    jy = (rng.random((n, n, n)) - 0.5) * (1.6 / n)
    jz = (rng.random((n, n, n)) - 0.5) * (1.6 / n)
    nx = (GX + jx).ravel()
    ny = (GY + jy).ravel()
    nz = (GZ + jz).ravel()
    nodes = np.stack([nx, ny, nz], axis=-1)  # (M,3)
    # for each point, distance to nearest node line-graph is expensive; use
    # nearest-node proximity as a tube network proxy (cheap, looks like lattice)
    d = np.linalg.norm(p[:, None, :] - nodes[None, :, :], axis=-1)  # (N,M)
    nearest = np.min(d, axis=1)
    return nearest - strut_r


def lattice_strut_sdf(p: np.ndarray, period: float = 0.5, strut_r: float = 0.05) -> np.ndarray:
    """Grid-aligned strut lattice (3 axis families)."""
    s = period / 2.0
    # distance to nearest axis-line in each of the 3 directions, take union
    fields = []
    for axis in range(3):
        q = np.mod(p[..., axis] + s, period) - s  # radial coord mod period
        # line = the other two coords
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        r = np.sqrt(p[..., a1] ** 2 + p[..., a2] ** 2)
        # repeat along axis using q only matters for periodic nodes; approximate
        d = np.abs(q) * 0.0 + r  # lines run full length along axis
        fields.append(d - strut_r)
    f = fields[0]
    for ff in fields[1:]:
        f = op_union(f, ff)
    return f
