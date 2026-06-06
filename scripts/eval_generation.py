#!/usr/bin/env python3
"""Run lightweight generation evals from exported train_tower artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from tower.config import PROJECT_ROOT
from tower.train.experiment_profile import load_train_config_from_experiment
from tower.unify.build import build_model_and_tokenizer
from tower.unify.train_model import SenseNovaTrainModel


DEFAULT_PROMPTS = [
    "a red train tower beside a futuristic railway station, detailed, daylight",
    "a small robot reading a book under a glass dome, cinematic lighting",
    "a watercolor landscape of mountains, lake, and a bright orange sunrise",
    "a product photo of a sleek black smartwatch on a white table",
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _load_backbone_model(args: argparse.Namespace):
    cfg = load_train_config_from_experiment(args.profile)
    cfg.output_dir = str(_resolve(args.run_dir))
    cfg.deepspeed = None
    cfg.gradient_checkpointing = False
    cfg.bf16 = not args.fp32

    model, tokenizer = build_model_and_tokenizer(cfg)

    # Training patched the small-model FM head inside the wrapper; recreate the
    # same module shape before loading exported backbone weights.
    wrapper = SenseNovaTrainModel(model, cfg)
    model = wrapper.model

    backbone = _resolve(args.checkpoint_dir) / "backbone.pt"
    if not backbone.is_file():
        raise FileNotFoundError(f"Missing backbone artifact: {backbone}")

    state = torch.load(backbone, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded backbone: {backbone}")
    print(f"State dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print("  missing preview:", missing[:5])
    if unexpected[:5]:
        print("  unexpected preview:", unexpected[:5])

    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    model.to(device=device, dtype=dtype)
    model.eval()
    model.config.use_cache = True
    if hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()
    return model, tokenizer


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    img = tensor.detach().float().cpu()
    if img.ndim == 4:
        img = img[0]
    # Generation code operates in a roughly [-1, 1] image space.
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(img)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="500m_continuous")
    parser.add_argument("--run-dir", default="outputs/pretrain/500m_super_omni")
    parser.add_argument("--checkpoint-dir", default="outputs/pretrain/500m_super_omni/checkpoint")
    parser.add_argument("--out-dir", default="outputs/eval_generation/500m_super_omni_t2i")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to generate; repeatable")
    parser.add_argument("--size", type=int, default=256, help="Square image size in pixels")
    parser.add_argument("--steps", type=int, default=20, help="Denoising steps")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-norm", default="none", choices=["none", "global", "channel", "cfg_zero_star"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.prompt or DEFAULT_PROMPTS

    model, tokenizer = _load_backbone_model(args)

    rows: list[dict] = []
    for idx, prompt in enumerate(prompts):
        seed = args.seed + idx
        print(f"[{idx + 1}/{len(prompts)}] seed={seed} prompt={prompt!r}", flush=True)
        start = time.time()
        with torch.inference_mode():
            generated = model.t2i_generate(
                tokenizer,
                prompt,
                cfg_scale=args.cfg_scale,
                cfg_norm=args.cfg_norm,
                image_size=(args.size, args.size),
                num_steps=args.steps,
                seed=seed,
            )
        image = _tensor_to_image(generated)
        image_path = out_dir / f"t2i_{idx:03d}_seed{seed}.png"
        image.save(image_path)
        seconds = round(time.time() - start, 2)
        row = {
            "index": idx,
            "prompt": prompt,
            "seed": seed,
            "image": str(image_path),
            "size": args.size,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "cfg_norm": args.cfg_norm,
            "seconds": seconds,
        }
        rows.append(row)
        print(f"  saved {image_path} ({seconds}s)", flush=True)

    manifest = out_dir / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
