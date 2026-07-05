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

    # 3) marching cubes → STL only (instant, open in a 3D viewer locally)
    from skimage import measure as _skm
    from structgen.model.meshing import Mesh, write_stl

    occ_mc = read_nrrd(path, res=64)
    pad = np.pad(occ_mc, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = _skm.marching_cubes(pad, level=0.5,
            spacing=(2/64, 2/64, 2/64))
        write_stl(Mesh(verts.astype(np.float32), faces.astype(np.int64)),
                  f"{outdir}/shape.stl")
        print(f"saved {outdir}/shape.stl ({len(faces)} faces)")
        print(f"  -> download & open at https://www.viewstl.com")
    except Exception as e:
        print(f"marching cubes/STL failed: {e}")

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
