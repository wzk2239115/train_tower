"""Read NRRD exactly like text2shape does (pynrrd + swapaxes), print ASCII cross-sections.

Run: python scripts/structgen_diag_ascii.py data/shapenet/nrrd
"""
import sys, os, glob, gzip
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/shapenet/nrrd"
if os.path.isdir(path):
    files = sorted(glob.glob(os.path.join(path, "*.nrrd")))
    path = files[0]

mid = os.path.basename(path)[:-5]
print(f"File: {mid}")

# --- Method 1: pynrrd (exactly like text2shape) ---
try:
    import nrrd
    data, header = nrrd.read(path)  # (4, 128, 128, 128) uint8
    print(f"pynrrd shape: {data.shape}, dtype: {data.dtype}")
    
    # text2shape read_nrrd:
    voxel = data.astype(np.float32) / 255.0       # (4, D, H, W)
    voxel = np.rollaxis(voxel, 0, 4)               # (D, H, W, 4) channel-last
    voxel = np.swapaxes(voxel, 0, 1)                # text2shape reorientation
    voxel = np.swapaxes(voxel, 0, 2)
    # Now: (H', W', D', 4) in text2shape convention
    
    alpha = voxel[..., 3]
    rgb_max = voxel[..., :3].max(-1)
    
    print(f"\nAfter text2shape transforms:")
    print(f"  shape: {voxel.shape}")
    print(f"  alpha: min={alpha.min():.3f} max={alpha.max():.3f} "
          f"mean={alpha.mean():.4f} nonzero={((alpha>0).mean()*100):.2f}%")
    print(f"  rgb_max: min={rgb_max.min():.3f} max={rgb_max.max():.3f} "
          f"mean={rgb_max.mean():.4f} nonzero={((rgb_max>0).mean()*100):.2f}%")
    
    # Occupancy from alpha vs RGB
    occ_alpha = (alpha > 0.5).astype(int)
    occ_rgb = (rgb_max > 0.01).astype(int)  # /255 threshold
    
    print(f"\n  occ(alpha>0.5): {occ_alpha.sum()} voxels ({occ_alpha.mean()*100:.2f}%)")
    print(f"  occ(rgb>0.01):  {occ_rgb.sum()} voxels ({occ_rgb.mean()*100:.2f}%)")
    
    # Print ASCII cross-sections for BOTH methods
    D, H, W = occ_rgb.shape
    for name, occ in [("RGB>0", occ_rgb), ("Alpha>0", occ_alpha)]:
        if occ.sum() == 0:
            print(f"\n--- {name}: EMPTY ---")
            continue
        print(f"\n--- {name} cross-sections ---")
        
        # XY plane at center Z
        z = D // 2
        sl = occ[z]
        print(f"XY slice at Z={z} (top-down):")
        for y in range(0, H, 2):
            row = ''.join(['#' if sl[y, x] else '.' for x in range(W)])
            if '#' in row:
                print(f"  {row}")
        
        # XZ plane at center Y
        y = H // 2
        sl = occ[:, y, :]
        print(f"\nXZ slice at Y={y} (side):")
        for z in range(D-1, -1, -2):
            row = ''.join(['#' if sl[z, x] else '.' for x in range(W)])
            if '#' in row:
                print(f"  {row}")
        
        # YZ plane at center X
        x = W // 2
        sl = occ[:, :, x]
        print(f"\nYZ slice at X={x} (front):")
        for z in range(D-1, -1, -2):
            row = ''.join(['#' if sl[z, y] else '.' for y in range(H)])
            if '#' in row:
                print(f"  {row}")

except ImportError:
    print("pynrrd not installed. Install: pip install pynrrd")
    
    # Fallback: manual read
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep+2:]), dtype=np.uint8)
    arr = arr.reshape(4, 128, 128, 128)
    voxel = np.rollaxis(arr.astype(np.float32) / 255.0, 0, 4)
    voxel = np.swapaxes(voxel, 0, 1)
    voxel = np.swapaxes(voxel, 0, 2)
    rgb_max = voxel[..., :3].max(-1)
    occ = (rgb_max > 0.01).astype(int)
    
    print(f"Manual read (no pynrrd)")
    print(f"occ(rgb>0): {occ.sum()} voxels ({occ.mean()*100:.2f}%)")
    
    D, H, W = occ.shape
    z = D // 2
    sl = occ[z]
    print(f"\nXY slice at Z={z}:")
    for y in range(0, H, 2):
        row = ''.join(['#' if sl[y, x] else '.' for x in range(W)])
        if '#' in row:
            print(f"  {row}")
