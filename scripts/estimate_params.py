#!/usr/bin/env python3
"""Verify SenseNova-500M-MoT scratch model parameter count."""

from __future__ import annotations

import argparse
import sys

from tower.train.config import TrainConfig
from tower.unify.build import build_scratch_model


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-m", type=float, default=450.0)
    parser.add_argument("--max-m", type=float, default=550.0)
    args = parser.parse_args()

    cfg = TrainConfig(
        init_mode="scratch",
        weight_init="random",
        model_config_path="configs/model/sensenova_500m_mot",
    )
    model = build_scratch_model(cfg)
    total, trainable = count_params(model)
    total_m = total / 1e6

    print(f"Total params: {total_m:.2f}M ({total:,})")
    print(f"Trainable:    {trainable / 1e6:.2f}M")

    groups = {
        "vision_model": 0,
        "fm_modules": 0,
        "mot_gen": 0,
        "und_llm": 0,
        "shared": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("vision_model."):
            groups["vision_model"] += n
        elif name.startswith("fm_modules."):
            groups["fm_modules"] += n
        elif "_mot_gen" in name:
            groups["mot_gen"] += n
        elif "embed_tokens" in name or name.endswith("lm_head.weight"):
            groups["shared"] += n
        elif name.startswith("language_model."):
            groups["und_llm"] += n

    for k, v in groups.items():
        print(f"  {k}: {v / 1e6:.2f}M")

    if not (args.min_m <= total_m <= args.max_m):
        print(f"FAIL: expected {args.min_m}M–{args.max_m}M, got {total_m:.2f}M", file=sys.stderr)
        return 1
    print("OK: parameter count within target range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
