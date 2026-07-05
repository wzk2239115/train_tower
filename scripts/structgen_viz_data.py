"""Visualize one ShapeNet data sample: occupancy slices + ortho projections + caption.

Run: python scripts/structgen_viz_data.py [nrrd_dir] [caption_csv]
Outputs PNGs to outputs/structgen/viz/
"""
import sys, os, csv, glob
import numpy as np
import gzip

def read_nrrd(path, res=128, threshold=0):
    """Read NRRD with text2shape axis convention (swapaxes ×2 to stand up)."""
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep+2:]), dtype=np.uint8)
    header = raw[:sep].decode(errors="ignore")
    size = 128; ch = 4
    for line in header.splitlines():
        if line.startswith("sizes:"):
            parts = line.split(); ch, size = int(parts[1]), int(parts[2]); break
    arr = arr.reshape(ch, size, size, size)
    occ = (arr[:3].max(0) > threshold).astype(np.float32)
    # text2shape reorientation
    occ = np.swapaxes(occ, 0, 1)
    occ = np.swapaxes(occ, 0, 2)
    if res < size:
        step = size // res
        occ = occ[::step, ::step, ::step]
    return np.ascontiguousarray(occ)

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

    # 3) GPU volume rendering — 128³ full resolution, raymarched + shaded
    print("rendering 3D on GPU (128³ raymarch, 3 views)...")
    try:
        from structgen.viz.gpu_render import render_volume_views
        view_imgs = render_volume_views(occ, res=768)
        labels = ["front (12°,35°)", "side (12°,125°)", "perspective (28°,245°)"]
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, img, label in zip(axes, view_imgs, labels):
            ax.imshow(img)
            ax.set_title(label, fontsize=13)
            ax.axis("off")
        plt.suptitle(f"GPU volume render — 128³ full res\n{cap[:60]}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{outdir}/render3d.png", dpi=100)
        plt.close()
        print(f"saved {outdir}/render3d.png")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"GPU render failed: {e}")

    # 4) STL export (full 128³ marching cubes)
    from skimage import measure as _skm
    from structgen.model.meshing import Mesh, write_stl
    pad = np.pad(occ, 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = _skm.marching_cubes(pad, level=0.5,
            spacing=(2/128, 2/128, 2/128))
        verts = verts - 1.0
        write_stl(Mesh(verts.astype(np.float32), faces.astype(np.int64)),
                  f"{outdir}/shape.stl")
        print(f"saved {outdir}/shape.stl ({len(faces)} faces)")
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
