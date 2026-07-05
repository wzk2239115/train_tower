"""Visualize one ShapeNet data sample: occupancy slices + ortho projections + caption.

Run: python scripts/structgen_viz_data.py [nrrd_dir] [caption_csv]
Outputs PNGs to outputs/structgen/viz/
"""
import sys, os, csv, glob
import numpy as np
import gzip

def read_nrrd(path, res=128):
    """Read NRRD at native resolution (128³) or downsample."""
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep+2:]), dtype=np.uint8)
    header = raw[:sep].decode(errors="ignore")
    size = 128; ch = 4
    for line in header.splitlines():
        if line.startswith("sizes:"):
            parts = line.split(); ch, size = int(parts[1]), int(parts[2]); break
    arr = arr.reshape(ch, size, size, size)
    occ = (arr[:3].max(0) > 0).astype(np.float32)
    if res < size:
        step = size // res
        occ = occ[::step, ::step, ::step]
    return occ

def main():
    nrrd_dir = sys.argv[1] if len(sys.argv) > 1 else "data/shapenet/nrrd"
    cap_csv = sys.argv[2] if len(sys.argv) > 2 else "captions.tablechair.csv"

    desc = {}
    with open(cap_csv) as f:
        for r in csv.DictReader(f):
            desc[r["modelId"]] = r["description"]

    files = sorted(glob.glob(os.path.join(nrrd_dir, "*.nrrd")))
    if not files:
        print(f"No .nrrd in {nrrd_dir}"); return

    # pick first file with a caption
    for path in files:
        mid = os.path.basename(path)[:-5]
        if mid in desc:
            break
    cap = desc.get(mid, "(no caption)")
    occ = read_nrrd(path)
    print(f"model: {mid}")
    print(f"caption: {cap}")
    print(f"occupancy: {occ.shape} solid_frac={occ.mean():.3f}")

    outdir = "outputs/structgen/viz"
    os.makedirs(outdir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = occ.shape[0]
    # 1) Z-slices (8 evenly spaced)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        z = int(i * R / 8)
        ax.imshow(occ[z], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Z={z}/{R}")
        ax.axis("off")
    plt.suptitle(f"Z-slices\n{cap[:60]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/slices.png", dpi=100)
    plt.close()
    print(f"saved {outdir}/slices.png")

    # 2) Ortho projections (what the current "sketch" looks like)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["XY (top view)", "XZ (side view)", "YZ (front view)"]
    for i, ax in enumerate(axes):
        proj = occ.mean(axis=i)
        ax.imshow(proj, cmap="gray")
        ax.set_title(titles[i])
        ax.axis("off")
    plt.suptitle(f"Ortho projections (density)\n{cap[:60]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{outdir}/ortho.png", dpi=100)
    plt.close()
    print(f"saved {outdir}/ortho.png")

    # 3) Smooth 3D surface via marching cubes (native 128³ resolution)
    from skimage import measure as _skm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    occ_full = read_nrrd(path, res=128)  # native resolution
    pad = np.pad(occ_full, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = _skm.marching_cubes(pad, level=0.5,
            spacing=(2/128, 2/128, 2/128))
        verts = verts - 1.0
    except Exception as e:
        print(f"marching cubes failed: {e}"); verts, faces = None, None

    if verts is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                                 subplot_kw={"projection": "3d"})
        views = [(20, 30, "front"), (20, 90, "side"), (40, -60, "perspective")]
        for ax, (elev, azim, label) in zip(axes, views):
            mesh = Poly3DCollection(verts[faces], alpha=0.8)
            mesh.set_facecolor([0.7, 0.5, 0.3, 0.8])
            ax.add_collection3d(mesh)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(label)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.suptitle(f"3D surface (128³ marching cubes)\n{cap[:60]}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{outdir}/surface3d.png", dpi=120)
        plt.close()
        print(f"saved {outdir}/surface3d.png (smooth surface, 3 angles)")
    else:
        print("skipped 3D surface (marching cubes failed)")

    # 4) If there's a .png beside the .nrrd (ShapeNet rendered view)
    png_path = path.replace(".nrrd", ".png")
    if os.path.exists(png_path):
        from PIL import Image
        img = Image.open(png_path)
        img.save(f"{outdir}/original_render.png")
        print(f"saved {outdir}/original_render.png (ShapeNet original render)")
    else:
        print("(no original .png render beside the .nrrd)")

    print(f"\nAll outputs in {outdir}/")

if __name__ == "__main__":
    main()
