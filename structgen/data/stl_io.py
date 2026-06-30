"""Minimal binary/ASCII STL reader (no trimesh dependency).

ABC ``stl2`` files are binary STL (one merged mesh per model). We parse the
mesh into (vertices, faces, face_normals) numpy arrays.
"""

from __future__ import annotations

import numpy as np


def read_stl(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices [V,3], faces [F,3] int, face_normals [F,3])."""
    with open(path, "rb") as f:
        head = f.read(5)
    if head == b"solid":
        # could be ASCII; verify by looking for "facet"
        with open(path, "r", errors="ignore") as f:
            txt = f.read(512)
        if "facet" in txt:
            return _read_ascii_stl(path)
    return _read_binary_stl(path)


def _read_binary_stl(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        f.read(80)  # header
        n_faces = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        dt = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ])
        data = np.frombuffer(f.read(n_faces * dt.itemsize), dtype=dt)
    normals = data["normal"].astype(np.float32)
    v0 = data["v0"].astype(np.float32)
    v1 = data["v1"].astype(np.float32)
    v2 = data["v2"].astype(np.float32)
    # unique vertices via concatenation + unique (good enough for SDF sampling)
    verts = np.concatenate([v0, v1, v2], axis=0)
    uniq, inv = np.unique(verts, axis=0, return_inverse=True)
    faces = inv.reshape(3, n_faces).T  # (F,3)
    return uniq.astype(np.float32), faces.astype(np.int64), normals


def _read_ascii_stl(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    normals: list[np.ndarray] = []
    with open(path, "r", errors="ignore") as f:
        tokens = f.read().split()
    i = 0
    cur_face: list[int] = []
    cur_normal = np.zeros(3, dtype=np.float32)
    while i < len(tokens):
        t = tokens[i]
        if t == "normal":
            cur_normal = np.array([float(tokens[i + 1]), float(tokens[i + 2]),
                                   float(tokens[i + 3])], dtype=np.float32)
            i += 4
        elif t == "vertex":
            verts.append(np.array([float(tokens[i + 1]), float(tokens[i + 2]),
                                   float(tokens[i + 3])], dtype=np.float32))
            i += 4
        elif t == "endfacet":
            if len(cur_face) == 0:
                a, b, c = len(verts) - 3, len(verts) - 2, len(verts) - 1
                faces.append((a, b, c))
            else:
                faces.append(tuple(cur_face[-3:]))
                cur_face = []
            normals.append(cur_normal)
            i += 1
        else:
            i += 1
    v = np.array(verts, dtype=np.float32)
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    f = np.array(faces, dtype=np.int64)
    if f.size == 0:
        f = inv.reshape(3, -1).T
    return uniq.astype(np.float32), f, np.array(normals, dtype=np.float32)
