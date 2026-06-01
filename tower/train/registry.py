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
