"""Read real ShapeNet solid voxelizations (NRRD) → occupancy grids.

Format (text2voxel customNrrdWriter):
  type: uint8, dimension: 4, sizes: 4 128 128 128, encoding: gzip
  4 channels = RGBA; alpha is all-zero. Occupancy = max(RGB channels) > 0.
"""
from __future__ import annotations

import gzip

import numpy as np


def read_nrrd_occ(path: str, res: int = 64, threshold: int = 0) -> np.ndarray:
    """Read an NRRD solid voxelization → occupancy grid at ``res``^3 (strided).

    Follows text2shape/lib/nrrd_rw.py convention:
      - channel-first (4, D, H, W) from pynrrd
      - occupancy = max(RGB) > threshold  (alpha is all-zero in our data)
      - two swapaxes to make the model stand up straight
    """
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep + 2:]), dtype=np.uint8)
    header = raw[:sep].decode(errors="ignore")
    size = 128
    ch = 4
    for line in header.splitlines():
        if line.startswith("sizes:"):
            parts = line.split()
            ch, size = int(parts[1]), int(parts[2])
            break
    arr = arr.reshape(ch, size, size, size)  # (4, D, H, W) channel-first
    occ = (arr[:3].max(0) > threshold).astype(np.float32)  # (D, H, W) {0,1}

    # text2shape axis reorientation: swapaxes ×2 to stand up straight
    # Original axes after channel collapse: (Y, Z, X) per space directions
    # After swapaxes(0,1) then swapaxes(0,2): becomes (X, Y, Z)
    occ = np.swapaxes(occ, 0, 1)
    occ = np.swapaxes(occ, 0, 2)

    if res < size:
        step = size // res
        occ = occ[::step, ::step, ::step]
    elif res > size:
        raise ValueError(f"res {res} > native {size}")
    return np.ascontiguousarray(occ)  # (res,res,res) float32 in {0,1}
