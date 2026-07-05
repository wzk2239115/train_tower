"""Diagnose NRRD RGBA channels to find correct occupancy extraction.

Run: python scripts/structgen_diag_nrrd.py data/shapenet/nrrd/XXXXX.nrrd
"""
import sys, gzip, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    import glob, os
    files = sorted(glob.glob("data/shapenet/nrrd/*.nrrd"))
    if not files:
        print("usage: python scripts/structgen_diag_nrrd.py <file.nrrd>"); sys.exit(1)
    path = files[0]

raw = open(path, "rb").read()
sep = raw.index(b"\n\n")
header = raw[:sep].decode(errors="ignore")

print("=" * 60)
print("NRRD HEADER:")
print(header)
print("=" * 60)

# Parse sizes
sizes = None
for line in header.splitlines():
    if line.strip().startswith("sizes:"):
        sizes = [int(x) for x in line.split()[1:]]
    if line.strip().startswith("type:"):
        dtype_str = line.split()[-1]
    if line.strip().startswith("encoding:"):
        enc = line.split()[-1]

print(f"sizes: {sizes}")
print(f"type: {locals().get('dtype_str', '?')}")
print(f"encoding: {locals().get('enc', '?')}")

# Decompress and reshape
body = gzip.decompress(raw[sep + 2:])
arr = np.frombuffer(body, dtype=np.uint8)
print(f"\nraw bytes: {len(arr)}  (= {sizes} product = {np.prod(sizes)})")

arr = arr.reshape(*sizes)
if sizes[0] == 4:
    chans = arr  # (4, D, H, W) channel-first
    layout = "channel-first (4, D, H, W)"
elif sizes[-1] == 4:
    chans = np.transpose(arr, (3, 0, 1, 2))  # (D, H, W, 4) → (4, D, H, W)
    layout = "channel-last (D, H, W, 4)"
else:
    print(f"ERROR: unexpected sizes {sizes}"); sys.exit(1)
print(f"layout: {layout}\n")

# Per-channel stats
labels = ["R (ch0)", "G (ch1)", "B (ch2)", "A (ch3)"]
for i in range(min(4, chans.shape[0])):
    ch = chans[i].astype(np.float32)
    nz = (ch > 0).sum()
    print(f"{labels[i]:12s}: min={ch.min():.0f}  max={ch.max():.0f}  "
          f"mean={ch.mean():.2f}  nonzero={nz:>8d} ({nz/ch.size*100:.2f}%)  "
          f"unique_vals={len(np.unique(ch))}")

print()

# Compare occupancy extraction methods
rgb_occ = chans[:3].max(0) > 0
a_occ = chans[3] > 0 if chans.shape[0] > 3 else None
a_occ_strong = chans[3] > 127 if chans.shape[0] > 3 else None

print(f"RGB>0          : occupied={rgb_occ.sum():>8d}  frac={rgb_occ.mean():.4f}")
if a_occ is not None:
    print(f"Alpha>0        : occupied={a_occ.sum():>8d}  frac={a_occ.mean():.4f}")
    print(f"Alpha>127      : occupied={a_occ_strong.sum():>8d}  frac={a_occ_strong.mean():.4f}")

    rgb_not_a = rgb_occ & ~a_occ          # RGB says occupied, Alpha says empty
    a_not_rgb = a_occ & ~rgb_occ          # Alpha says occupied, RGB says empty
    both = rgb_occ & a_occ

    print(f"\n--- Discrepancy ---")
    print(f"RGB>0 & Alpha=0 (SPURIOUS in RGB): {rgb_not_a.sum():>8d}  frac={rgb_not_a.mean():.4f}")
    print(f"Alpha>0 & RGB=0  (missed by RGB): {a_not_rgb.sum():>8d}  frac={a_not_rgb.mean():.4f}")
    print(f"Both agree                         : {both.sum():>8d}  frac={both.mean():.4f}")

    # Value distribution of alpha where RGB>0 but Alpha=0
    if rgb_not_a.sum() > 0:
        a_vals = chans[3][rgb_not_a]
        print(f"\nAlpha values where RGB>0 but Alpha low: "
              f"min={a_vals.min()} max={a_vals.max()} mean={a_vals.mean():.1f}")

    # Value distribution of RGB where Alpha>0 but RGB=0
    if a_not_rgb.sum() > 0:
        rgb_vals = chans[:3].max(0)[a_not_rgb]
        print(f"RGB values where Alpha>0 but RGB=0: "
              f"min={rgb_vals.min()} max={rgb_vals.max()} mean={rgb_vals.mean():.1f}")

print("\n=> If 'SPURIOUS in RGB' frac is high, switch read_nrrd to use Alpha channel.")
