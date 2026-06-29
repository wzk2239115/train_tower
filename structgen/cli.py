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
    return cfg


def cmd_train(args) -> int:
    from structgen.train import train
    cfg = _build_cfg(args)
    smoke = args.smoke_steps if args.smoke else None
    ckpt = train(cfg, smoke_steps=smoke)
    print(f"\nDone. Final checkpoint: {ckpt}")
    return 0


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
        sp.add_argument("--backbone", default="proxy", choices=["proxy", "qwen", "stepfun"])
        sp.add_argument("--pretrained-path", default=None)
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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
