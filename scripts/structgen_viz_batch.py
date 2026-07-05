"""Batch preview multiple ShapeNet samples on GPU.

Run: python scripts/structgen_viz_batch.py [nrrd_dir] [caption_csv] [count]
Outputs: outputs/structgen/viz/batch_grid.png
"""
import sys, os, csv, glob, gzip
import numpy as np


def read_nrrd(path, res=128, threshold=0):
    raw = open(path, "rb").read()
    sep = raw.index(b"\n\n")
    arr = np.frombuffer(gzip.decompress(raw[sep + 2:]), dtype=np.uint8)
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
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    desc = {}
    with open(cap_csv) as f:
        for r in csv.DictReader(f):
            desc[r["modelId"]] = r["description"]

    files = sorted(glob.glob(os.path.join(nrrd_dir, "*.nrrd")))
    if not files:
        print(f"No .nrrd in {nrrd_dir}"); return

    # pick files that have captions
    picked = []
    for path in files:
        mid = os.path.basename(path)[:-5]
        if mid in desc:
            picked.append((path, mid, desc[mid]))
        if len(picked) >= count:
            break

    print(f"Rendering {len(picked)} samples on GPU...")

    import torch
    from structgen.viz.gpu_render import render_volume_views

    outdir = "outputs/structgen/viz"
    os.makedirs(outdir, exist_ok=True)

    imgs = []
    captions = []
    for i, (path, mid, cap) in enumerate(picked):
        occ = read_nrrd(path)
        # one 3/4 perspective view
        view = render_volume_views(occ, views=[(18, 45)], res=512)[0]
        imgs.append(view)
        short_cap = cap[:45] + ("..." if len(cap) > 45 else "")
        captions.append(f"[{i}] {mid[:8]}  solid={occ.mean():.3f}\n{short_cap}")
        print(f"  [{i+1}/{len(picked)}] {mid[:12]} solid_frac={occ.mean():.3f}")

    # grid layout
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(4, len(imgs))
    nrows = (len(imgs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None]
    elif ncols == 1:
        axes = axes[:, None]

    for i, (img, cap) in enumerate(zip(imgs, captions)):
        r, c = i // ncols, i % ncols
        axes[r, c].imshow(img)
        axes[r, c].set_title(cap, fontsize=8, fontfamily="monospace")
        axes[r, c].axis("off")
    # hide unused
    for i in range(len(imgs), nrows * ncols):
        r, c = i // ncols, i % ncols
        axes[r, c].axis("off")

    plt.suptitle(f"ShapeNet data preview — {len(imgs)} samples (GPU raymarch, 128³)", fontsize=13)
    plt.tight_layout()
    outpath = f"{outdir}/batch_grid.png"
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"\nsaved {outpath}")


if __name__ == "__main__":
    main()
