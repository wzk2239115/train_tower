from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from tower.config import PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}. Run `tower convert` first.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def build_data_dict() -> dict[str, dict[str, str]]:
    """Build NEO-style data_dict from processed manifest."""
    manifest = load_manifest()
    data_dict: dict[str, dict[str, str]] = {}
    for dataset_key, entry in manifest.items():
        for stage, rel_path in entry.get("stages", {}).items():
            reg_key = f"{dataset_key}_{stage}"
            abs_path = (PROJECT_ROOT / rel_path).resolve()
            data_dict[reg_key] = {
                "annotation_path": str(abs_path),
                "data_path": "",
            }
    return data_dict


def inject_data_dict() -> None:
    """Register train_tower datasets into NEO neo.data module."""
    from tower.unify.backends import import_neo_data

    neo_data = import_neo_data()

    data_dict = build_data_dict()
    neo_data.data_dict.update(data_dict)


def validate_curriculum_datasets(curriculum: list[dict[str, Any]]) -> None:
    """Pre-flight check: verify every dataset referenced in curriculum exists."""
    manifest = load_manifest()
    available = set()
    for dataset_key, entry in manifest.items():
        for stage in entry.get("stages", {}):
            available.add(f"{dataset_key}_{stage}")

    missing_all: list[tuple[str, str]] = []
    for phase in curriculum:
        stage_name = phase.get("stage", "?")
        datasets_str = phase.get("datasets", "")
        if not datasets_str:
            continue
        for ds in datasets_str.split(","):
            ds = ds.strip()
            if ds and ds not in available:
                missing_all.append((stage_name, ds))

    if missing_all:
        lines = [f"  stage={s}: dataset={d}" for s, d in missing_all]
        raise ValueError(
            "Curriculum references missing datasets (not in manifest):\n"
            + "\n".join(lines)
            + f"\nAvailable: {sorted(available)}"
        )


def print_training_manifest(
    curriculum: list[dict[str, Any]] | None,
    *,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    num_gpus: int,
    loss_weights: dict[str, float] | None = None,
    use_flow_tower: bool = False,
) -> None:
    """Print a human-readable training plan and wait for user confirmation."""
    manifest = load_manifest()

    w = 72
    sep = "=" * w

    lines = [
        sep,
        "  TRAINING PLAN".center(w),
        sep,
        f"  max_steps          : {max_steps}",
        f"  GPUs               : {num_gpus} x batch {batch_size} x accum {grad_accum}"
        f"  = {num_gpus * batch_size * grad_accum} samples/step",
        f"  total samples      : ~{num_gpus * batch_size * grad_accum * max_steps:,}",
        f"  flow_tower         : {use_flow_tower}",
        f"  loss_weights       : {loss_weights or {'default': 1.0}}",
        sep,
    ]

    if curriculum:
        lines.append("  CURRICULUM ({:d} stages)".format(len(curriculum)).center(w))
        lines.append(sep)
        prev_step = 0
        for i, phase in enumerate(curriculum):
            stage = phase.get("stage", "?")
            until = phase.get("until_step", "?")
            datasets_str = phase.get("datasets", "")
            lw = phase.get("loss_weights", {})
            task = phase.get("task_override", "")
            decoder_prob = phase.get("tower_decoder_prob", "")
            cfg_drop = phase.get("cfg_label_drop_prob", "")
            self_cond = phase.get("tower_self_cond_prob", "")

            step_range = f"step {prev_step + 1:>4d} – {until}"
            prev_step = int(until)

            lines.append(f"")
            lines.append(f"  [{i+1}/{len(curriculum)}] {stage}  ({step_range})")
            lines.append(f"    datasets : {datasets_str}")

            ds_entries = []
            total_samples = 0
            for ds in datasets_str.split(","):
                ds = ds.strip()
                if not ds:
                    continue
                base_name = ds.rsplit("_", 1)[0] if "_" in ds else ds
                entry = manifest.get(base_name, {})
                n = entry.get("samples", 0)
                role = entry.get("role", "")
                ds_entries.append((ds, n, role))
                total_samples += n

            for ds, n, role in ds_entries:
                lines.append(f"               {ds:<28s} {n:>8,d} samples  [{role}]")
            lines.append(f"               {'TOTAL':<28s} {total_samples:>8,d}")

            extras = []
            if lw:
                extras.append(f"loss={lw}")
            if task:
                extras.append(f"task={task}")
            if decoder_prob != "":
                extras.append(f"decoder_prob={decoder_prob}")
            if cfg_drop != "":
                extras.append(f"cfg_drop={cfg_drop}")
            if self_cond != "":
                extras.append(f"self_cond={self_cond}")
            if extras:
                lines.append(f"    config   : {', '.join(extras)}")

        lines.append("")
    else:
        datasets_str = manifest.get("datasets", "N/A")
        lines.append(f"  datasets: {datasets_str}")

    lines.append(sep)
    plan_text = "\n".join(lines)
    print(plan_text)

    if int(os.environ.get("TOWER_AUTO_START", "0")):
        print("  TOWER_AUTO_START=1, skipping confirmation.")
        return

    try:
        input("\n  >>> Press ENTER to start training, Ctrl+C to abort >>> ")
    except EOFError:
        pass
