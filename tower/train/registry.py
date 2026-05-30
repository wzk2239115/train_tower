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
    import neo.data as neo_data

    data_dict = build_data_dict()
    neo_data.data_dict.update(data_dict)
