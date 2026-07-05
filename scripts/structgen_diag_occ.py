"""Detailed occupancy structure diagnostic.

Run: python scripts/structgen_diag_occ.py [nrrd_dir]
"""
import sys, os, glob, gzip
import numpy as np


def read_nrrd_channels(path):
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep + 2:]), dtype=np.uint8)
    return arr.reshape(4, 128, 128, 128)


def main():
    nrrd_dir = sys.argv[1] if len(sys.argv) > 1 else "data/shapenet/nrrd"
    if os.path.isdir(nrrd_dir):
        files = sorted(glob.glob(os.path.join(nrrd_dir, "*.nrrd")))
    else:
        files = [nrrd_dir]

    for path in files[:3]:
        mid = os.path.basename(path)[:-5]
        print(f"\n{'='*60}")
        print(f"Model: {mid}")
        print(f"{'='*60}")

        chans = read_nrrd_channels(path)
        occ = (chans[:3].max(0) > 0).astype(np.float32)
        D, H, W = occ.shape

        # --- per-slice occupancy profile ---
        print("\nZ-slice occupancy (every 4th slice):")
        for z in range(0, D, 4):
            n = int(occ[z].sum())
            bar = '#' * min(n // 50, 60)
            print(f"  Z={z:3d}: {n:6d} voxels  {bar}")

        # --- center cross-sections ---
        cz, cy, cx = D // 2, H // 2, W // 2

        print(f"\nXY slice at Z={cz} (top-down, center):")
        sl = occ[cz]
        for y in range(0, H, 2):
            row = ''.join(['#' if sl[y, x] > 0 else '.' for x in range(0, W, 1)])
            if any(c == '#' for c in row):
                print(f"  Y={y:3d}: {row}")

        print(f"\nXZ slice at Y={cy} (side view, center):")
        sl = occ[:, cy, :]
        for z in range(0, D, 2):
            row = ''.join(['#' if sl[z, x] > 0 else '.' for x in range(0, W, 1)])
            if any(c == '#' for c in row):
                print(f"  Z={z:3d}: {row}")

        # --- interior fill check ---
        print("\nInterior fill check (center voxel column along X):")
        col = occ[:, cy, cz]
        runs = []
        in_run = False; start = 0
        for i in range(D):
            if col[i] > 0 and not in_run:
                in_run = True; start = i
            elif col[i] == 0 and in_run:
                in_run = False; runs.append((start, i - 1))
        if in_run:
            runs.append((start, D - 1))
        print(f"  Occupied runs along X at (Y={cy}, Z={cz}): {runs}")
        print(f"  Total occupied: {col.sum()}/{D}")

        # --- value histogram ---
        vals = chans[:3].max(0)
        nz = vals[vals > 0]
        if len(nz) > 0:
            print(f"\nNonzero RGB values: min={nz.min()} max={nz.max()} "
                  f"mean={nz.mean():.1f} median={np.median(nz):.0f}")
            hist, edges = np.histogram(nz, bins=10)
            for i in range(10):
                print(f"  [{edges[i]:.0f}-{edges[i+1]:.0f}]: {hist[i]:6d}")


if __name__ == "__main__":
    main()
