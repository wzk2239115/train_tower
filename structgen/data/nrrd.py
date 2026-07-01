"""Read real ShapeNet solid voxelizations (NRRD) → occupancy grids.

Format (text2voxel customNrrdWriter):
  type: uint8, dimension: 4, sizes: 4 128 128 128, encoding: gzip
  4 channels = RGBA; alpha is all-zero. Occupancy = max(RGB channels) > 0.
"""
from __future__ import annotations

import gzip

import numpy as np


def read_nrrd_occ(path: str, res: int = 64) -> np.ndarray:
    """Read an NRRD solid voxelization → occupancy grid at ``res``^3 (strided)."""
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep + 2:]), dtype=np.uint8)
    # sizes line tells us the grid; assume 128^3 x 4 channels
    header = raw[:sep].decode(errors="ignore")
    size = 128
    for line in header.splitlines():
        if line.startswith("sizes:"):
            parts = line.split()
            ch, size = int(parts[1]), int(parts[2])
            break
    arr = arr.reshape(ch, size, size, size)
    occ = (arr[:3].max(0) > 0).astype(np.float32)  # {0,1}
    if res < size:
        step = size // res
        occ = occ[::step, ::step, ::step]
    elif res > size:
        raise ValueError(f"res {res} > native {size}")
    return occ  # (res,res,res) float32 in {0,1}
