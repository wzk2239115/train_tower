from __future__ import annotations

import argparse
import sys

from tower.config import PROJECT_ROOT, load_dataset_specs
from tower.io.writer import write_manifest
from tower.config import URL_ONLY_DATASETS
from tower.registry import get_converter


def _select_datasets(args: argparse.Namespace) -> list[str]:
    specs = load_dataset_specs()
    if args.dataset:
        if args.dataset not in specs:
            raise SystemExit(f"Unknown dataset: {args.dataset}")
        return [args.dataset]
    if args.stage:
        return [k for k, s in specs.items() if args.stage in s.stages and k not in URL_ONLY_DATASETS]
    if args.all:
        return [k for k in specs if k not in URL_ONLY_DATASETS]
    raise SystemExit("Specify --dataset, --stage, or --all")


def cmd_convert(args: argparse.Namespace) -> int:
    keys = _select_datasets(args)
    if not keys:
        print("No datasets matched.", file=sys.stderr)
        return 1

    reports = []
    for key in keys:
        spec = load_dataset_specs()[key]
        converter = get_converter(key)
        print(f"\n{'[dry-run] ' if args.dry_run else ''}Converting {key} (role={spec.role}, stages={spec.stages})")
        try:
            report = converter.convert(spec, limit=args.limit, dry_run=args.dry_run)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  SKIP {key}: {exc}", file=sys.stderr)
            continue

        reports.append(report)
        if report.skipped:
            print(f"  skipped: {dict(report.skipped)}")
        if report.stages:
            print(f"  written per stage: {dict(report.stages)}")
        for stage, path in report.output_files.items():
            print(f"  -> {path}")

    if reports and not args.dry_run:
        write_manifest(
            reports,
            PROJECT_ROOT / "data" / "processed" / "manifest.json",
            PROJECT_ROOT / "note" / "processed_registry.py",
            PROJECT_ROOT,
        )
        print(f"\nManifest: {PROJECT_ROOT / 'data' / 'processed' / 'manifest.json'}")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from pathlib import Path

    from tower.train.config import load_train_config
    from tower.train.trainer import run_training

    if args.config:
        cfg = load_train_config(config_path=Path(args.config))
    elif args.stage:
        cfg = load_train_config(stage=args.stage)
    else:
        raise SystemExit("Specify --config or --stage for training")

    if args.datasets:
        cfg.datasets = args.datasets
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.output_dir:
        cfg.output_dir = args.output_dir

    print(f"Training stage={cfg.stage} datasets={cfg.datasets} -> {cfg.output_dir}")
    run_training(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tower", description="train_tower data conversion and training")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert raw datasets to NEO JSONL")
    convert.add_argument("--dataset", help="Single dataset key from note/dataset.yml")
    convert.add_argument("--stage", choices=["pt", "mt", "sft"], help="Convert all datasets for a stage")
    convert.add_argument("--all", action="store_true", help="Convert all phase-1 datasets")
    convert.add_argument("--limit", type=int, default=None, help="Max samples per dataset")
    convert.add_argument("--dry-run", action="store_true", help="Count samples without writing")
    convert.set_defaults(func=cmd_convert)

    train = sub.add_parser("train", help="Train unified NEO/SenseNova model")
    train.add_argument("--config", help="Path to train yaml config")
    train.add_argument(
        "--stage",
        choices=["world_pt", "understanding_warmup", "generation_pt", "unified_mt", "unified_sft"],
        help="Training stage (reads note/train.yml + configs/train/)",
    )
    train.add_argument("--datasets", help="Override dataset_use comma list")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--output-dir", help="Override output directory")
    train.set_defaults(func=cmd_train)

    viz = sub.add_parser("viz", help="Visualize training data and metrics (terminal)")
    viz.set_defaults(func=cmd_viz)

    return parser


def cmd_viz(args: argparse.Namespace) -> int:
    from tower.viz.cli import main as viz_main

    return viz_main(getattr(args, "viz_argv", None))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and argv[0] == "viz":
        args = parser.parse_args(["viz"])
        args.viz_argv = argv[1:]
        return args.func(args)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
