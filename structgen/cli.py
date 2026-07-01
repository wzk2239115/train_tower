"""structgen CLI: train / generate / inspect.

Examples
--------
# local smoke run (proxy CLIP backbone, tiny decoder)
python -m structgen.cli train --smoke

# generate a part from a prompt and export STL
python -m structgen.cli generate --ckpt outputs/structgen/decoder_stepN.pt \\
    --prompt "structural cylinder part with internal gyroid infill" --out part.stl

# on the compute box: switch to the real Step-3.7-Flash backbone
python -m structgen.cli train --backbone stepfun \\
    --pretrained-path /path/to/Step-3.7-Flash --batch 8 --steps 50000
"""

from __future__ import annotations

import argparse
import sys

import torch

from structgen.config import (
    BackboneConfig, DecoderConfig, StructGenConfig, TrainConfig,
)


def _build_cfg(args) -> StructGenConfig:
    cfg = StructGenConfig()
    cfg.backbone = BackboneConfig(
        kind=args.backbone, pretrained_path=args.pretrained_path,
        text_emb_path=getattr(args, "text_emb", None),
        image_size=args.image_size,
    )
    cfg.decoder = DecoderConfig(
        grid_res=args.res, base_channels=args.base_ch,
        channel_mults=tuple(int(m) for m in args.mults.split(",")),
        num_blocks=args.blocks,
    )
    g = lambda n, d: getattr(args, n, d)  # noqa: E731
    cfg.train = TrainConfig(
        batch_size=g("batch", 4), lr=g("lr", 1e-4), max_steps=g("steps", 5000),
        device=args.device, out_dir=g("out_dir", "outputs/structgen"),
        log_every=g("log_every", 20), save_every=g("save_every", 1000),
    )
    cfg.num_samples = g("num_samples", 2048)
    cfg.surface_samples = g("surface_samples", 2048)
    cfg.real_data_dir = g("real_data_dir", None)
    return cfg


def cmd_train(args) -> int:
    from structgen.train import train
    _maybe_init_distributed()
    cfg = _build_cfg(args)
    smoke = args.smoke_steps if args.smoke else None
    ckpt = train(cfg, smoke_steps=smoke)
    if _is_main():
        print(f"\nDone. Final checkpoint: {ckpt}")
    return 0


def _maybe_init_distributed():
    import os
    import torch.distributed as dist
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")


def _is_main() -> bool:
    import torch.distributed as dist
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def cmd_generate(args) -> int:
    from structgen.infer import load_trained, generate, sketch_from_path
    cfg = _build_cfg(args)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    backbone, decoder = load_trained(cfg, args.ckpt, device)
    sketch = sketch_from_path(args.sketch, cfg.backbone.image_size) if args.sketch else None
    field, mesh = generate(
        backbone, decoder, cfg, args.prompt, sketch=sketch,
        device=device, out_mesh=args.out, n_sample_steps=args.sample_steps,
    )
    print(f"field shape={field.shape} sdf[{field.min():.3f},{field.max():.3f}] "
          f"solid_frac={float((field<0).mean()):.3f}")
    return 0


def cmd_convert_shapenet(args) -> int:
    """Extract ShapeNet NRRD solid voxels (+ pair captions) into a flat dir of
    .nrrd for the latent pipeline. Skips models without a caption."""
    import os
    import shutil
    import zipfile

    os.makedirs(args.out_dir, exist_ok=True)
    zf = zipfile.ZipFile(args.input)
    nrrds = [n for n in zf.namelist() if n.endswith(".nrrd")]
    import csv
    have = {r["modelId"] for r in csv.DictReader(open(args.captions))}
    kept = 0
    for n in nrrds:
        mid = os.path.basename(n)[:-5]
        if mid not in have:
            continue
        dst = os.path.join(args.out_dir, mid + ".nrrd")
        if not os.path.exists(dst):
            with zf.open(n) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
        kept += 1
        if kept % 2000 == 0:
            print(f"  extracted {kept}...")
    print(f"[convert-shapenet] {kept} captioned NRRD -> {args.out_dir}")
    return 0


def cmd_train_vae(args) -> int:
    from structgen.latent import train_vae
    _maybe_init_distributed()
    cfg = _build_cfg(args)
    train_vae(cfg, args.nrrd_dir, args.captions, steps=args.steps,
              batch=args.batch, beta=args.beta, out=args.out)
    return 0


def cmd_train_latent(args) -> int:
    from structgen.latent import train_latent
    _maybe_init_distributed()
    cfg = _build_cfg(args)
    train_latent(cfg, args.vae, args.nrrd_dir, args.captions,
                 steps=args.steps, batch=args.batch, out=args.out)
    return 0


def cmd_generate_latent(args) -> int:
    from structgen.infer import sketch_from_path
    from structgen.latent import generate_latent
    from structgen.model.meshing import occupancy_to_mesh, export_mesh
    cfg = _build_cfg(args)
    sk = sketch_from_path(args.sketch, cfg.backbone.image_size) if args.sketch else None
    occ = generate_latent(cfg, args.vae, args.ckpt, args.prompt, sketch=sk,
                          cfg_scale=args.cfg_scale, n_steps=args.sample_steps)
    if args.out:
        mesh = occupancy_to_mesh(occ)
        if mesh is not None:
            export_mesh(mesh, args.out)
            print(f"[gen] mesh {len(mesh)} faces -> {args.out}")
    return 0


def cmd_convert_abc(args) -> int:
    """Convert a directory of ABC STL meshes → per-model .npz (field+surface+
    prompt) ready for StructGenDataset. Extract the ``.7z`` first (py7zr)."""
    import glob
    import os

    import numpy as np

    from structgen.data.mesh_to_sdf import mesh_to_sdf, prompt_from_stats
    from structgen.data.stl_io import read_stl
    from structgen.data.dataset import _render_sketch

    src = args.input
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # accept a .7z directly → extract beside it
    if src.lower().endswith(".7z"):
        import py7zr
        ex = os.path.join(os.path.dirname(src) or ".",
                          os.path.basename(src).replace(".7z", ""))
        if not os.path.isdir(ex):
            print(f"[convert-abc] extracting {src} -> {ex}")
            with py7zr.SevenZipFile(src, mode="r") as z:
                z.extractall(ex)
        src = ex

    stls = sorted(glob.glob(os.path.join(src, "**", "*.stl"), recursive=True))
    print(f"[convert-abc] {len(stls)} STL files under {src} -> {out_dir}")
    ok = 0
    for i, stl in enumerate(stls):
        name = os.path.splitext(os.path.basename(stl))[0]
        out_npz = os.path.join(out_dir, f"{name}.npz")
        if os.path.exists(out_npz) and not args.force:
            ok += 1
            continue
        try:
            r = mesh_to_sdf(stl, grid_res=args.res, n_surf=args.surface_samples,
                            seed=i)
            if not r["ok"]:
                continue
            verts, _, _ = read_stl(stl)
            prompt = prompt_from_stats(r["field"], verts)
            sketch = _render_sketch(r["field"], size=args.image_size)
            import io
            buf = io.BytesIO()
            sketch.save(buf, format="PNG")
            sketch_png = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            np.savez_compressed(out_npz,
                                field=r["field"], surface=r["surface"],
                                normals=r["normals"], prompt=prompt,
                                sketch_png=sketch_png)
            ok += 1
        except Exception as e:  # noqa: BLE001
            if args.verbose:
                print(f"  skip {name}: {e!r}")
            continue
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(stls)}] converted ok={ok}")
    print(f"[convert-abc] done: {ok}/{len(stls)} -> {out_dir}")
    print(f"[convert-abc] train with: --real-data-dir {out_dir}")
    return 0


def cmd_precompute(args) -> int:
    import os
    import torch as _torch
    from structgen.data import sampler
    from structgen.model.backbone import build_backbone
    from structgen.config import StructGenConfig, BackboneConfig

    cfg = StructGenConfig(backbone=BackboneConfig(
        kind="stepfun", pretrained_path=args.pretrained_path,
        image_size=args.image_size))
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    bb = build_backbone(cfg).to(device)
    prompts = sorted({rp.prompt for rp in sampler.all_recipes()})
    print(f"[precompute] encoding {len(prompts)} distinct prompts with 198B...")
    emb: dict[str, _torch.Tensor] = {}
    for i, p in enumerate(prompts):
        bb(p)  # batch=1; raw pooled (1,hidden) is cached in bb._text_cache
        emb[p] = bb._text_cache[p].detach().cpu()
        print(f"  [{i + 1}/{len(prompts)}] {p[:60]}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    _torch.save(emb, args.out)
    print(f"[precompute] saved {len(emb)} embeddings -> {args.out}")
    print(f"[precompute] now train with: --backbone cached --text-emb {args.out}")
    return 0


def cmd_inspect(args) -> int:
    """Render a sample of the synthetic data distribution to disk."""
    import os
    from structgen.data import sampler
    from structgen.model.meshing import sdf_to_mesh, export_mesh
    os.makedirs(args.out_dir, exist_ok=True)
    recipes = sampler.all_recipes()
    n = min(args.n, len(recipes))
    for i in range(n):
        rp = recipes[i]
        field = sampler.build_field(rp, args.res)
        mesh = sdf_to_mesh(field)
        if mesh is None:
            continue
        path = os.path.join(args.out_dir, f"{rp.name}.stl")
        export_mesh(mesh, path)
        print(f"[{i}] {rp.name}: {len(mesh)} faces -> {path}")
        print(f"    prompt: {rp.prompt}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="structgen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--backbone", default="proxy",
                        choices=["proxy", "qwen", "stepfun", "cached"])
        sp.add_argument("--pretrained-path", default=None)
        sp.add_argument("--text-emb", default=None,
                        help="precomputed prompt->emb .pt (for --backbone cached)")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--res", type=int, default=64, help="voxel grid resolution")
        sp.add_argument("--base-ch", type=int, default=64)
        sp.add_argument("--mults", default="1,2,4,8")
        sp.add_argument("--blocks", type=int, default=2)
        sp.add_argument("--image-size", type=int, default=224)

    t = sub.add_parser("train", help="train the geometry decoder")
    add_common(t)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--num-samples", type=int, default=2048)
    t.add_argument("--surface-samples", type=int, default=2048)
    t.add_argument("--real-data-dir", default=None,
                   help="dir of ABC .npz (from convert-abc); overrides synthesis")
    t.add_argument("--out-dir", default="outputs/structgen")
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--smoke", action="store_true", help="tiny smoke run")
    t.add_argument("--smoke-steps", type=int, default=30)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="generate a structural part")
    add_common(g)
    g.add_argument("--ckpt", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--sketch", default=None, help="optional sketch image path")
    g.add_argument("--out", default="outputs/structgen/gen.stl")
    g.add_argument("--sample-steps", type=int, default=50)
    g.set_defaults(func=cmd_generate)

    insp = sub.add_parser("inspect", help="render synthetic GT specimens to STL")
    insp.add_argument("--n", type=int, default=8)
    insp.add_argument("--res", type=int, default=64)
    insp.add_argument("--out-dir", default="outputs/structgen/inspect")
    insp.set_defaults(func=cmd_inspect)

    pre = sub.add_parser("precompute",
                         help="encode all prompts with 198B once → .pt (run before cached+DDP train)")
    pre.add_argument("--pretrained-path", required=True)
    pre.add_argument("--image-size", type=int, default=224)
    pre.add_argument("--out", default="outputs/structgen/text_emb.pt")
    pre.set_defaults(func=cmd_precompute)

    cab = sub.add_parser("convert-abc",
                         help="ABC STL dir/.7z → per-model .npz (field+surface+prompt) for training")
    cab.add_argument("--input", required=True, help="dir of *.stl or a .7z file")
    cab.add_argument("--out-dir", default="data/abc/npz")
    cab.add_argument("--res", type=int, default=64)
    cab.add_argument("--surface-samples", type=int, default=4096)
    cab.add_argument("--image-size", type=int, default=128)
    cab.add_argument("--force", action="store_true", help="reconvert existing")
    cab.add_argument("--verbose", action="store_true")
    cab.set_defaults(func=cmd_convert_abc)

    cs = sub.add_parser("convert-shapenet",
                        help="ShapeNet NRRD zip + captions CSV → flat dir of .nrrd (latent pipeline)")
    cs.add_argument("--input", required=True, help="nrrd_*.zip")
    cs.add_argument("--captions", required=True, help="captions.tablechair.csv")
    cs.add_argument("--out-dir", default="data/shapenet/nrrd")
    cs.set_defaults(func=cmd_convert_shapenet)

    def _latent_common(sp):
        sp.add_argument("--nrrd-dir", default="data/shapenet/nrrd")
        sp.add_argument("--captions", default="captions.tablechair.csv")
        sp.add_argument("--res", type=int, default=64)
        sp.add_argument("--base-ch", type=int, default=64)
        sp.add_argument("--mults", default="1,2,4,8")
        sp.add_argument("--blocks", type=int, default=2)
        sp.add_argument("--image-size", type=int, default=224)
        sp.add_argument("--backbone", default="proxy", choices=["proxy", "qwen", "stepfun", "cached"])
        sp.add_argument("--pretrained-path", default=None)
        sp.add_argument("--text-emb", default=None)

    v = sub.add_parser("train-vae", help="Stage A: train the voxel VAE (occ<->latent)")
    _latent_common(v)
    v.add_argument("--vae-base", type=int, default=24)
    v.add_argument("--steps", type=int, default=20000)
    v.add_argument("--batch", type=int, default=16)
    v.add_argument("--beta", type=float, default=1e-3)
    v.add_argument("--out", default="outputs/structgen/vae.pt")
    v.set_defaults(func=cmd_train_vae)

    f = sub.add_parser("train-latent", help="Stage B: conditional flow in VAE latent (CFG)")
    _latent_common(f)
    f.add_argument("--vae", required=True)
    f.add_argument("--flow-base", type=int, default=192)
    f.add_argument("--latent-res", type=int, default=16)
    f.add_argument("--latent-ch", type=int, default=32)
    f.add_argument("--steps", type=int, default=30000)
    f.add_argument("--batch", type=int, default=8)
    f.add_argument("--out", default="outputs/structgen/flow.pt")
    f.set_defaults(func=cmd_train_latent)

    gl = sub.add_parser("generate-latent", help="sample latent (CFG) → decode → STL")
    _latent_common(gl)
    gl.add_argument("--vae", required=True)
    gl.add_argument("--ckpt", required=True, help="flow.pt")
    gl.add_argument("--prompt", required=True)
    gl.add_argument("--sketch", default=None)
    gl.add_argument("--cfg-scale", type=float, default=4.0)
    gl.add_argument("--sample-steps", type=int, default=50)
    gl.add_argument("--out", default="outputs/structgen/gen_latent.stl")
    gl.set_defaults(func=cmd_generate_latent)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
