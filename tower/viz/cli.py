"""Terminal CLI for training data visualization (replaces Jupyter notebook)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless server: no GUI backend

from tower.config import PROJECT_ROOT
from tower.viz.plots import (
    plot_modality_breakdown,
    plot_resolution_histogram,
    plot_role_distribution,
    plot_sample_grid,
    plot_stage_comparison,
    plot_tower_exit_weights,
    plot_training_curves,
)
from tower.viz.selection import StageSelection, export_selection_yaml, parse_dataset_keys
from tower.viz.stages import STAGE_ORDER, list_available_datasets, load_stage_configs
from tower.viz.training_metrics import discover_training_runs

DEFAULT_VIZ_DIR = PROJECT_ROOT / "exports" / "viz"
DEFAULT_EXPORT_PATH = DEFAULT_VIZ_DIR / "stage_selections.yml"


def _stage_script_name(stage: str) -> str:
    mapping = {
        "world_pt": "train_tower_world",
        "understanding_warmup": "train_uw",
        "generation_pt": "train_gen_pt",
        "unified_mt": "train_mt",
        "unified_sft": "train_sft",
    }
    return mapping.get(stage, f"train_{stage}")


def _ordered_stages(configs: dict[str, Any]) -> list[str]:
    return [s for s in STAGE_ORDER if s in configs]


def _resolve_out_dir(path: str | None) -> Path:
    out = Path(path) if path else DEFAULT_VIZ_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(fig, *, out_dir: Path, name: str) -> Path:
    import matplotlib.pyplot as plt

    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _selection_for_stage(
    stage: str,
    *,
    datasets: str | None = None,
    max_samples: int | None = None,
) -> StageSelection:
    sel = StageSelection.from_stage_defaults(stage)
    if datasets:
        sel.set_selection(parse_dataset_keys(datasets))
    if max_samples is not None:
        sel.max_samples_per_dataset = max_samples
    return sel


def _load_selections_yaml(path: Path) -> dict[str, StageSelection]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid selections yaml: {path}")

    selections: dict[str, StageSelection] = {}
    for stage, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        keys = payload.get("datasets") or parse_dataset_keys(str(payload.get("datasets_csv", "")))
        sel = StageSelection(stage=str(stage), selected_reg_keys=list(keys))
        limit = payload.get("max_samples_per_dataset")
        if limit is not None:
            sel.max_samples_per_dataset = int(limit)
        selections[str(stage)] = sel
    return selections


def _default_selections() -> dict[str, StageSelection]:
    configs = load_stage_configs()
    return {stage: StageSelection.from_stage_defaults(stage) for stage in _ordered_stages(configs)}


def _resolve_selections(
    *,
    selections_yaml: str | None,
    stage: str | None,
    datasets: str | None,
    max_samples: int | None,
) -> dict[str, StageSelection]:
    if selections_yaml:
        selections = _load_selections_yaml(Path(selections_yaml))
    else:
        selections = _default_selections()

    if stage:
        selections[stage] = _selection_for_stage(stage, datasets=datasets, max_samples=max_samples)
    elif datasets:
        raise SystemExit("--datasets requires --stage when not using --selections-yaml")

    if max_samples is not None and not stage:
        for sel in selections.values():
            sel.max_samples_per_dataset = max_samples
    return selections


def _print_stage_header(stage: str, sel: StageSelection) -> None:
    info = load_stage_configs()[stage]
    print(f"\n=== {stage} ===")
    print(f"  output_dir:      {info.output_dir or 'N/A'}")
    print(f"  use_flow_tower:  {info.use_flow_tower}")
    print(f"  learning_rate:   {info.learning_rate}")
    print(f"  max_steps:       {info.max_steps}")
    print(f"  datasets:        {sel.datasets_csv or '(empty)'}")


def cmd_list_stages(_args: argparse.Namespace) -> int:
    configs = load_stage_configs()
    for stage in _ordered_stages(configs):
        info = configs[stage]
        tower = " [flow-tower]" if info.use_flow_tower else ""
        print(f"{stage:24} datasets={len(info.datasets):2}  lr={info.learning_rate}  steps={info.max_steps}{tower}")
    return 0


def cmd_list_datasets(args: argparse.Namespace) -> int:
    rows = list_available_datasets()
    if args.stage:
        info = load_stage_configs().get(args.stage)
        if not info:
            raise SystemExit(f"Unknown stage: {args.stage}")
        default_stages = {k.rsplit("_", 1)[-1] for k in info.datasets}
        rows = [r for r in rows if not default_stages or r["stage"] in default_stages]

    for row in rows:
        exists = "ok" if row["exists"] else "MISSING"
        print(
            f"{row['reg_key']:32} {row['stage']:3} {row['role']:16} "
            f"n≈{row['manifest_samples']!s:>8}  [{exists}]"
        )
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import pandas as pd

    sel = _selection_for_stage(args.stage, datasets=args.datasets, max_samples=args.max_samples)
    _print_stage_header(args.stage, sel)

    if not sel.selected_reg_keys:
        print("No datasets selected.", file=sys.stderr)
        return 1

    summary = sel.summary()
    df = pd.DataFrame(summary.metrics_table())
    print("\nPer-dataset metrics:")
    print(df.to_string(index=False))

    print(
        f"\nTotal: {summary.total_samples} | valid: {summary.valid_samples} | "
        f"errors: {dict(summary.validation_errors) if summary.validation_errors else 0}"
    )

    info = load_stage_configs()[args.stage]
    out_dir = _resolve_out_dir(args.out_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plot_modality_breakdown(summary.modality, title=f"{args.stage} modality", ax=axes[0])
    plot_role_distribution(summary, title=f"{args.stage} roles", ax=axes[1])
    plot_resolution_histogram(summary.datasets, title=f"{args.stage} resolution", ax=axes[2])
    fig.tight_layout()
    print(f"Saved: {_save_figure(fig, out_dir=out_dir, name=f'{args.stage}_metrics')}")

    if info.use_flow_tower and info.tower_exit_weights:
        fig2 = plot_tower_exit_weights(
            info.tower_exit_weights,
            title=f"{args.stage} tower exit weights",
        )
        print(f"Saved: {_save_figure(fig2, out_dir=out_dir, name=f'{args.stage}_tower_exits')}")
        active = [e for e, w in info.tower_exit_weights.items() if w > 0]
        print(
            f"train_exits={', '.join(info.tower_train_exits) or '—'} | "
            f"freeze_exits={', '.join(info.tower_freeze_exits) or '—'} | "
            f"active_loss={', '.join(active) or '—'}"
        )
    elif info.loss_weights:
        print(f"Classic loss_weights: {info.loss_weights}")

    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    from tower.viz import load_samples

    sel = _selection_for_stage(args.stage, datasets=args.datasets, max_samples=args.max_samples)
    if not sel.selected_reg_keys:
        print("No datasets selected.", file=sys.stderr)
        return 1

    samples = load_samples(
        sel.selected_reg_keys,
        limit_per_dataset=sel.max_samples_per_dataset,
        seed=args.seed,
    )
    if not samples:
        print("No samples available.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    picked = rng.sample(samples, min(args.n, len(samples)))
    fig = plot_sample_grid(picked, max_samples=args.n, title=f"{args.stage} preview")
    out_dir = _resolve_out_dir(args.out_dir)
    print(f"Saved: {_save_figure(fig, out_dir=out_dir, name=f'{args.stage}_preview')}")
    plt.close("all")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt
    import pandas as pd

    selections = _resolve_selections(
        selections_yaml=args.selections_yaml,
        stage=args.stage,
        datasets=args.datasets,
        max_samples=args.max_samples,
    )

    summaries = {}
    for stage, sel in selections.items():
        if sel.selected_reg_keys:
            summaries[stage] = sel.summary()

    if not summaries:
        print("No datasets selected for any stage.", file=sys.stderr)
        return 1

    out_dir = _resolve_out_dir(args.out_dir)
    for metric, title in (("total", "samples per stage (total)"), ("valid", "samples per stage (valid)")):
        fig = plot_stage_comparison(summaries, metric=metric, title=title)
        print(f"Saved: {_save_figure(fig, out_dir=out_dir, name=f'compare_{metric}')}")

    rows = []
    for stage, summary in summaries.items():
        mod = summary.modality.as_dict()
        rows.append(
            {
                "stage": stage,
                "total": summary.total_samples,
                "valid": summary.valid_samples,
                "image+text": mod.get("image+text", 0),
                "audio": mod.get("audio_only", 0)
                + mod.get("image+audio", 0)
                + mod.get("image+text+audio", 0),
            }
        )
    print("\nStage comparison:")
    print(pd.DataFrame(rows).to_string(index=False))
    plt.close("all")
    return 0


def cmd_curves(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    runs = discover_training_runs(Path(args.outputs_dir) if args.outputs_dir else None)
    if not runs:
        print(f"No training runs under {args.outputs_dir or PROJECT_ROOT / 'outputs'}", file=sys.stderr)
        return 1

    names = [r.name for r in runs]
    print(f"Found {len(runs)} run(s): {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")

    if args.runs:
        wanted = {n.strip() for n in args.runs.split(",") if n.strip()}
        chosen = [r for r in runs if r.name in wanted]
        missing = wanted - {r.name for r in chosen}
        if missing:
            print(f"Warning: unknown runs skipped: {', '.join(sorted(missing))}", file=sys.stderr)
    else:
        chosen = runs[: min(4, len(runs))]

    if not chosen:
        print("No runs matched.", file=sys.stderr)
        return 1

    fig = plot_training_curves(chosen, metric=args.metric)
    out_dir = _resolve_out_dir(args.out_dir)
    print(f"Saved: {_save_figure(fig, out_dir=out_dir, name=f'curves_{args.metric}')}")
    plt.close("all")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    selections = _resolve_selections(
        selections_yaml=args.selections_yaml,
        stage=args.stage,
        datasets=args.datasets,
        max_samples=args.max_samples,
    )
    out = export_selection_yaml(selections, path=args.output or DEFAULT_EXPORT_PATH)
    print(f"Exported: {out}")
    print("\nExample training commands:")
    for stage, sel in selections.items():
        if not sel.selected_reg_keys:
            continue
        script = _stage_script_name(stage)
        print(f'  DATASETS="{sel.datasets_csv}" ./scripts/{script}.sh')
    return 0


def build_viz_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tower viz",
        description="Visualize training data on headless servers (text tables + PNG exports)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Directory for plot PNGs (default: {DEFAULT_VIZ_DIR})",
    )
    sub = parser.add_subparsers(dest="viz_command", required=True)

    p = sub.add_parser("list-stages", help="List training stages from note/train.yml")
    p.set_defaults(func=cmd_list_stages)

    p = sub.add_parser("list-datasets", help="List registered datasets")
    p.add_argument("--stage", help="Filter by training stage (pt/mt/sft suffix)")
    p.set_defaults(func=cmd_list_datasets)

    p = sub.add_parser("metrics", help="Per-stage data stats and distribution plots")
    p.add_argument("--stage", required=True, choices=list(STAGE_ORDER))
    p.add_argument("--datasets", help="Comma-separated reg_keys (default: stage config)")
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset (0=unlimited)")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("preview", help="Random sample grid (image/text/audio)")
    p.add_argument("--stage", required=True, choices=list(STAGE_ORDER))
    p.add_argument("--datasets", help="Comma-separated reg_keys")
    p.add_argument("-n", type=int, default=4, help="Number of samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("compare", help="Compare sample counts across stages")
    p.add_argument("--selections-yaml", help="YAML from `tower viz export`")
    p.add_argument("--stage", choices=list(STAGE_ORDER), help="Override one stage")
    p.add_argument("--datasets", help="With --stage: comma-separated reg_keys")
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("curves", help="Plot training loss/grad_norm/lr from outputs/")
    p.add_argument("--runs", help="Comma-separated run dir names (default: first 4)")
    p.add_argument("--metric", choices=["loss", "grad_norm", "lr"], default="loss")
    p.add_argument("--outputs-dir", default=None, help="Root to search (default: outputs/)")
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("export", help="Export stage dataset selections to YAML")
    p.add_argument("--output", default=None, help=f"Output path (default: {DEFAULT_EXPORT_PATH})")
    p.add_argument("--selections-yaml", help="Base selections to merge")
    p.add_argument("--stage", choices=list(STAGE_ORDER), help="Override one stage")
    p.add_argument("--datasets", help="With --stage: comma-separated reg_keys")
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_viz_parser()
    args = parser.parse_args(argv)
    return args.func(args)
