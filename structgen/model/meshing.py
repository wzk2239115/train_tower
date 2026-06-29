"""Extract meshes from SDF/occupancy voxel fields and export STL/OBJ.

Uses scikit-image marching cubes on the zero level set of the SDF.
STL/OBJ writers are self-contained (no trimesh dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from skimage import measure as _skm
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


@dataclass
class Mesh:
    vertices: np.ndarray  # (V,3) float
    faces: np.ndarray  # (F,3) int

    def __len__(self) -> int:
        return len(self.faces)


def sdf_to_mesh(field: np.ndarray, level: float = 0.0,
                bounds: tuple[float, float] = (-1.0, 1.0)) -> Mesh | None:
    """Marching cubes on the SDF zero level set. ``field``: (D,H,W)."""
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required for marching cubes")
    lo, hi = bounds
    # pad so closed surfaces at the boundary march correctly
    pad = np.pad(field, 1, mode="edge")
    try:
        verts, faces, _, _ = _skm.marching_cubes(pad, level=level, spacing=(
            (hi - lo) / (field.shape[0]),
            (hi - lo) / (field.shape[1]),
            (hi - lo) / (field.shape[2]),
        ))
    except (RuntimeError, ValueError):
        return None
    # shift because of padding
    verts = verts - (hi - lo) / field.shape[0] + lo
    return Mesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int64))


def occupancy_to_mesh(occ: np.ndarray, bounds: tuple[float, float] = (-1.0, 1.0)) -> Mesh | None:
    """Marching cubes on a binary occupancy grid (level 0.5)."""
    field = (occ.astype(np.float32) - 0.5) * 2.0  # + inside, - outside (level 0)
    # marching_cubes expects higher=inside convention; flip sign so level=0 works
    return sdf_to_mesh(-field, level=0.0, bounds=bounds)


def write_obj(mesh: Mesh, path: str) -> None:
    v, f = mesh.vertices, mesh.faces
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"f {i0 + 1} {i1 + 1} {i2 + 1}" for i0, i1, i2 in f]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_stl(mesh: Mesh, path: str) -> None:
    """Binary STL (self-contained)."""
    v, f = mesh.vertices, mesh.faces
    n = len(f)
    normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(np.array(n, dtype="<u4").tobytes())
        dt = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ])
        recs = np.zeros(n, dtype=dt)
        recs["normal"] = normals
        recs["v0"] = v[f[:, 0]]
        recs["v1"] = v[f[:, 1]]
        recs["v2"] = v[f[:, 2]]
        fh.write(recs.tobytes())


def export_mesh(mesh: Mesh, path: str) -> None:
    """Export by extension (.stl / .obj)."""
    low = path.lower()
    if low.endswith(".stl"):
        write_stl(mesh, path)
    elif low.endswith(".obj"):
        write_obj(mesh, path)
    else:
        raise ValueError(f"unsupported mesh format: {path}")
