"""Build full structural-part specimens (shell ∩ internal topology).

A specimen combines an outer shell with an internal topology field and
returns a sampled SDF grid + surface points/normals + a text prompt + the
parametric recipe. Used by ``StructGenDataset``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from structgen.data import tpms as T

# canonical sampling cube
CUBE_BOUNDS = (-1.0, 1.0)


@dataclasses.dataclass
class SpecimenParams:
    name: str
    shell: str  # "box" | "cylinder" | "sphere" | "superellipsoid"
    shell_kw: dict[str, Any]
    topology: str  # "gyroid"|"diamond"|"schwarz_p"|"graded_gyroid"|"voronoi"|"lattice"|"none"
    topology_kw: dict[str, Any]
    prompt: str

    def to_prompt(self) -> str:
        return self.prompt


# a curated bank of parametric recipes (deterministic prompts)
def _recipe_bank() -> list[SpecimenParams]:
    bank: list[SpecimenParams] = []
    topos = [
        ("gyroid", dict(period=0.8, thickness=0.12), "gyroid infill"),
        ("diamond", dict(period=0.9, thickness=0.13), "diamond TPMS infill"),
        ("schwarz_p", dict(period=0.7, thickness=0.10), "Schwarz-P infill"),
        ("graded_gyroid", dict(period=0.85, thickness=0.10, grade=0.10), "graded-density gyroid infill"),
        ("voronoi", dict(cells=4, strut_r=0.07, seed=1), "Voronoi strut lattice"),
        ("lattice", dict(period=0.5, strut_r=0.05), "grid strut lattice"),
    ]
    shells = [
        ("box", dict(half=(0.85, 0.85, 0.85)), "box"),
        ("cylinder", dict(r=0.75, h=0.85, axis=2), "cylinder"),
        ("sphere", dict(r=0.9), "sphere"),
        ("superellipsoid", dict(a=0.85, b=0.85, c=0.85, e=0.6, n=0.6), "curved superellipsoid"),
    ]
    for si, (sname, skw, slabel) in enumerate(shells):
        for ti, (tname, tkw, tlabel) in enumerate(topos):
            prompt = (
                f"A 3D-printable structural {slabel} part with internal {tlabel}; "
                f"lightweight, stiff, manufacturable."
            )
            bank.append(SpecimenParams(
                name=f"{sname}_{tname}_{si}_{ti}",
                shell=sname, shell_kw=skw,
                topology=tname, topology_kw=tkw,
                prompt=prompt,
            ))
    return bank


def sample_grid(n: int, bounds: tuple[float, float] = CUBE_BOUNDS) -> np.ndarray:
    """Return (n^3, 3) sample points in the cube."""
    lin = np.linspace(bounds[0], bounds[1], n, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


def _shell_sdf(p: np.ndarray, params: SpecimenParams) -> np.ndarray:
    s = params.shell
    kw = params.shell_kw
    if s == "box":
        return T.sdf_box(p, **kw)
    if s == "cylinder":
        return T.sdf_cylinder(p, **kw)
    if s == "sphere":
        return T.sdf_sphere(p, **kw)
    if s == "superellipsoid":
        return T.sdf_superellipsoid(p, **kw)
    raise ValueError(f"unknown shell {s}")


def _topology_sdf(p: np.ndarray, params: SpecimenParams) -> np.ndarray:
    t = params.topology
    kw = params.topology_kw
    if t == "none":
        return np.full(p.shape[:-1], -1.0, dtype=p.dtype)  # fully solid
    if t in ("gyroid", "diamond", "schwarz_p"):
        return T._TPMS[t](p, **kw)
    if t == "graded_gyroid":
        return T.graded_tpms(p, kind="gyroid", **kw)
    if t == "voronoi":
        return T.voronoi_struts_sdf(p, **kw)
    if t == "lattice":
        return T.lattice_strut_sdf(p, **kw)
    raise ValueError(f"unknown topology {t}")


def build_field(params: SpecimenParams, grid_res: int,
                smooth: float = 0.1) -> np.ndarray:
    """Sample the full structural-part SDF on a (grid_res)^3 grid.

    Part = shell ∩ topology (so the TPMS/lattice is the *solid* interior of the
    shell). Returns shape (grid_res, grid_res, grid_res).
    """
    p = sample_grid(grid_res)
    shell = _shell_sdf(p, params)
    topo = _topology_sdf(p, params)
    # smooth-intersect so the union of shell-boundary and lattice looks organic
    field = T.op_smooth_intersect(shell, topo, k=smooth)
    return field.reshape(grid_res, grid_res, grid_res).astype(np.float32)


def occupancy_from_sdf(field: np.ndarray) -> np.ndarray:
    """Solid mask: 1 inside (sdf<0), 0 outside."""
    return (field < 0).astype(np.float32)


def sample_surface(field: np.ndarray, n: int, bounds: tuple[float, float] = CUBE_BOUNDS,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample points near the zero level set (for Chamfer / normal targets)."""
    rng = rng or np.random.default_rng(0)
    grad = np.stack(np.gradient(field), axis=-1)
    gnorm = np.linalg.norm(grad, axis=-1) + 1e-8
    # distance estimate |sdf|/|grad sdf|
    approx_d = np.abs(field) / gnorm
    # weight near-surface voxels
    w = np.exp(-(approx_d ** 2) / (2 * 0.03 ** 2))
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        # degenerate field → fall back to uniform sampling
        w = np.ones_like(w)
        tot = w.sum()
    w = w / tot
    flat_idx = rng.choice(w.size, size=n, replace=True, p=w.ravel())
    coords = np.stack(np.unravel_index(flat_idx, field.shape), axis=-1)  # (n,3) int
    pts = np.stack([coords[:, 0], coords[:, 1], coords[:, 2]], axis=-1)
    pts = (pts + 0.5) / field.shape[0] * (bounds[1] - bounds[0]) + bounds[0]
    # jitter within voxel
    pts = pts + rng.uniform(-0.5, 0.5, size=pts.shape) * (
        (bounds[1] - bounds[0]) / field.shape[0])
    return pts.astype(np.float32)


def all_recipes() -> list[SpecimenParams]:
    return _recipe_bank()
