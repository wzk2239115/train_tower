from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tower.config import PROJECT_ROOT, load_dataset_specs
from tower.io.writer import persist_manifest, refresh_manifest_from_disk, write_manifest
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


def _print_report(key: str, report, *, dry_run: bool) -> None:
    spec = load_dataset_specs()[key]
    print(f"\n{'[dry-run] ' if dry_run else ''}Converting {key} (role={spec.role}, stages={spec.stages})")
    if report.skipped:
        print(f"  skipped: {dict(report.skipped)}")
    if report.stages:
        print(f"  written per stage: {dict(report.stages)}")
    for stage, path in report.output_files.items():
        print(f"  -> {path}")


def cmd_refresh_manifest(_args: argparse.Namespace) -> int:
    manifest = refresh_manifest_from_disk(PROJECT_ROOT)
    if not manifest:
        print("No processed JSONL found under data/processed/.", file=sys.stderr)
        return 1

    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.json"
    persist_manifest(
        manifest,
        manifest_path,
        PROJECT_ROOT / "note" / "processed_registry.py",
        PROJECT_ROOT,
    )
    print("Refreshed manifest from disk:")
    for dataset_key in sorted(manifest):
        entry = manifest[dataset_key]
        stages = ", ".join(sorted(entry.get("stages", {})))
        print(f"  {dataset_key}: samples={entry.get('samples')} stages=[{stages}]")
    print(f"\nManifest: {manifest_path}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    if args.refresh_manifest:
        return cmd_refresh_manifest(args)

    keys = _select_datasets(args)
    if not keys:
        print("No datasets matched.", file=sys.stderr)
        return 1

    reports = []
    if args.jobs <= 1:
        for key in keys:
            spec = load_dataset_specs()[key]
            converter = get_converter(key)
            print(f"\n{'[dry-run] ' if args.dry_run else ''}Converting {key} (role={spec.role}, stages={spec.stages})")
            try:
                report = converter.convert(
                    spec,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    workers=args.workers,
                    extract_only=args.extract_only,
                    jsonl_only=args.jsonl_only,
                    legacy_convert=args.legacy_convert,
                )
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
    else:
        from tower.convert.runner import run_convert_job

        print(f"Parallel convert: jobs={args.jobs}, workers={args.workers}, datasets={len(keys)}")
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(keys))) as pool:
            futures = {
                pool.submit(
                    run_convert_job,
                    key,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    workers=args.workers,
                    extract_only=args.extract_only,
                    jsonl_only=args.jsonl_only,
                    legacy_convert=args.legacy_convert,
                ): key
                for key in keys
            }
            for fut in as_completed(futures):
                key = futures[fut]
                key, report, err = fut.result()
                if err:
                    print(f"  SKIP {key}: {err}", file=sys.stderr)
                    continue
                reports.append(report)
                _print_report(key, report, dry_run=args.dry_run)

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
    convert.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Convert this many datasets in parallel (default: 1)",
    )
    convert.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Workers per dataset (blip3o: parallel tar extract / jsonl dirs; default: 1)",
    )
    convert.add_argument(
        "--extract-only",
        action="store_true",
        help="blip3o only: bulk tar extract to data/images/<dataset>/ (no jsonl)",
    )
    convert.add_argument(
        "--jsonl-only",
        action="store_true",
        help="blip3o only: build jsonl from already extracted images (no tar extract)",
    )
    convert.add_argument(
        "--legacy-convert",
        action="store_true",
        help="blip3o only: slow per-sample PIL re-encode path",
    )
    convert.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild manifest.json from existing data/processed/*/*.jsonl (no re-convert)",
    )
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
