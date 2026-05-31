"""Super-Omni Data Registry — read recipe/pools/*.yaml and build data_dict.

Usage:
    from recipe.data_registry import build_data_dict, resolve_stage_datasets

    # Build the full data_dict for tower/neo/data/__init__.py
    data_dict = build_data_dict()

    # Resolve a stage YAML's pool_mix into a datasets string
    datasets_str = resolve_stage_datasets("recipe/stages/unified_mt.yaml")

    # Apply scale overrides
    datasets_str = apply_scale(datasets_str, "recipe/scales/1b.yaml")
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

RECIPE_DIR = Path(__file__).resolve().parent
POOLS_DIR = RECIPE_DIR / "pools"
STAGES_DIR = RECIPE_DIR / "stages"
SCALES_DIR = RECIPE_DIR / "scales"

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")


def _resolve_source(source: str) -> str:
    if source.startswith("$DATA_ROOT"):
        return source.replace("$DATA_ROOT", DATA_ROOT)
    return source


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_datasets_from_pool(pool_data: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    datasets = pool_data.get("datasets") or []
    for ds in datasets:
        if not isinstance(ds, dict) or "key" not in ds:
            continue
        results.append(ds)

    sub_pools = pool_data.get("sub_pools") or {}
    for _sub_name, sub_data in sub_pools.items():
        if not isinstance(sub_data, dict):
            continue
        for ds in sub_data.get("datasets") or []:
            if not isinstance(ds, dict) and "key" not in ds:
                continue
            if isinstance(ds, dict):
                results.append(ds)

    return results


def build_data_dict() -> dict[str, dict[str, str]]:
    """Read all pool YAMLs and build data_dict {key: {annotation_path, data_path}}."""
    data_dict: dict[str, dict[str, str]] = {}

    for pool_file in sorted(POOLS_DIR.glob("*.yaml")):
        pool_data = _load_yaml(pool_file)
        for ds in _collect_datasets_from_pool(pool_data):
            key = ds["key"]
            source = _resolve_source(ds.get("source", ""))
            ann_file = ds.get("annotation_file", "train.jsonl")
            data_dict[key] = {
                "annotation_path": str(Path(source) / ann_file),
                "data_path": source,
            }

    return data_dict


def _build_pool_index() -> dict[str, list[dict[str, Any]]]:
    """Map pool_id → list of dataset entries."""
    index: dict[str, list[dict[str, Any]]] = {}
    for pool_file in sorted(POOLS_DIR.glob("*.yaml")):
        pool_data = _load_yaml(pool_file)
        pool_id = pool_data.get("pool_id", pool_file.stem)
        index[pool_id] = _collect_datasets_from_pool(pool_data)
    return index


def resolve_stage_datasets(stage_path: str | Path) -> str:
    """Resolve a stage YAML's pool_mix into a comma-separated datasets string.

    Each dataset key gets a sampling rate suffix %N where
    N = round(pool_mix_ratio × dataset_sampling_rate × 100).
    """
    stage_data = _load_yaml(Path(stage_path))

    if "datasets" in stage_data:
        return stage_data["datasets"]

    pool_mix = stage_data.get("pool_mix")
    if not pool_mix:
        raise ValueError(f"Stage {stage_path} has neither 'datasets' nor 'pool_mix'")

    pool_index = _build_pool_index()
    parts: list[str] = []

    for pool_id, mix_ratio in pool_mix.items():
        datasets = pool_index.get(pool_id, [])
        for ds in datasets:
            key = ds.get("key", "")
            ds_sampling = float(ds.get("sampling_rate", 1.0))
            effective_pct = max(1, round(mix_ratio * ds_sampling * 100))
            parts.append(f"{key}%{effective_pct}")

    return ",".join(parts)


def apply_scale(datasets_str: str, scale_path: str | Path) -> str:
    """Apply scale overrides to a datasets string.

    The scale YAML's pool_sampling section can override individual dataset
    sampling rates. This function parses the datasets string, applies
    overrides, and reconstructs it.
    """
    scale_data = _load_yaml(Path(scale_path))
    pool_sampling = scale_data.get("pool_sampling") or {}
    if not pool_sampling:
        return datasets_str

    overrides: dict[str, int] = {}
    for _pool_id, ds_overrides in pool_sampling.items():
        if not isinstance(ds_overrides, dict):
            continue
        for ds_key, new_rate in ds_overrides.items():
            overrides[ds_key] = max(1, round(float(new_rate) * 100))

    parts = datasets_str.split(",")
    updated: list[str] = []
    for part in parts:
        match = re.match(r"^(.+)%(\d+)$", part.strip())
        if match:
            key = match.group(1)
            new_pct = overrides.get(key, int(match.group(2)))
            updated.append(f"{key}%{new_pct}")
        else:
            updated.append(part.strip())

    return ",".join(updated)


def resolve_full_stage(
    stage_name: str,
    *,
    scale: str | None = None,
    recipe_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve a stage YAML to a full config dict for TrainConfig construction.

    Returns a dict with 'datasets' resolved from pool_mix and scale applied.
    """
    root = recipe_root or RECIPE_DIR
    stage_path = root / "stages" / f"{stage_name}.yaml"
    if not stage_path.is_file():
        raise FileNotFoundError(f"Stage file not found: {stage_path}")

    stage_data = _load_yaml(stage_path)

    if "pool_mix" in stage_data and "datasets" not in stage_data:
        stage_data["datasets"] = resolve_stage_datasets(stage_path)

    if scale:
        scale_path = root / "scales" / f"{scale}.yaml"
        if scale_path.is_file():
            scale_data = _load_yaml(scale_path)
            if "datasets" in stage_data:
                stage_data["datasets"] = apply_scale(stage_data["datasets"], scale_path)

            mult = float(scale_data.get("data_multiplier", 1.0))
            step_mult = float(scale_data.get("step_multiplier", 1.0))
            lr_mult = float(scale_data.get("lr_multiplier", 1.0))

            hp = stage_data.get("hyperparams") or {}
            if "max_steps" in hp:
                hp["max_steps"] = int(hp["max_steps"] * step_mult)
            if "learning_rate" in hp:
                hp["learning_rate"] = hp["learning_rate"] * lr_mult
            if "warmup_steps" in hp:
                hp["warmup_steps"] = max(500, int(hp["max_steps"] * 0.01))
            stage_data["hyperparams"] = hp

            max_seq = scale_data.get("max_seq_length")
            if max_seq and "hyperparams" in stage_data:
                stage_data["hyperparams"]["max_seq_length"] = int(max_seq)

    return stage_data


def list_pools() -> list[str]:
    return sorted(p.stem for p in POOLS_DIR.glob("*.yaml"))


def list_stages() -> list[str]:
    return sorted(p.stem for p in STAGES_DIR.glob("*.yaml"))


def list_scales() -> list[str]:
    return sorted(
        p.stem for p in SCALES_DIR.glob("*.yaml") if not p.name.startswith("_")
    )


if __name__ == "__main__":
    print("=== Data Registry ===")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"\nPools: {list_pools()}")
    print(f"Stages: {list_stages()}")
    print(f"Scales: {list_scales()}")

    dd = build_data_dict()
    print(f"\nRegistered datasets: {len(dd)}")
    for key in sorted(dd.keys())[:10]:
        print(f"  {key}: {dd[key]['annotation_path']}")
    if len(dd) > 10:
        print(f"  ... and {len(dd) - 10} more")

    print("\n=== unified_mt datasets ===")
    mt_path = STAGES_DIR / "unified_mt.yaml"
    if mt_path.is_file():
        ds = resolve_stage_datasets(mt_path)
        print(f"  {ds[:200]}...")

    print("\n=== unified_sft datasets (1b scale) ===")
    sft_path = STAGES_DIR / "unified_sft.yaml"
    if sft_path.is_file():
        full = resolve_full_stage("unified_sft", scale="1b")
        print(f"  datasets: {full.get('datasets', 'N/A')[:200]}...")
        hp = full.get("hyperparams", {})
        print(f"  max_steps: {hp.get('max_steps')}")
        print(f"  learning_rate: {hp.get('learning_rate')}")
