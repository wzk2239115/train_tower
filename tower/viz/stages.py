from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tower.config import NOTE_DIR, PROJECT_ROOT, load_dataset_specs, load_roles
from tower.train.registry import load_manifest

TRAIN_YML = NOTE_DIR / "train.yml"
TOWER_TRAIN_YML = NOTE_DIR / "tower_train.yml"
TOWER_YML = NOTE_DIR / "tower.yml"

# Canonical training stage order (classic + tower world_pt).
STAGE_ORDER = (
    "world_pt",
    "understanding_warmup",
    "generation_pt",
    "unified_mt",
    "unified_sft",
)

DATA_STAGE_LABELS = {
    "pt": "Pretrain",
    "mt": "Multi-task",
    "sft": "SFT",
}


@dataclass
class StageInfo:
    name: str
    datasets: tuple[str, ...]
    output_dir: str | None = None
    use_flow_tower: bool = False
    loss_weights: dict[str, float] = field(default_factory=dict)
    tower_exit_weights: dict[str, float] = field(default_factory=dict)
    tower_train_exits: tuple[str, ...] = ()
    tower_freeze_exits: tuple[str, ...] = ()
    learning_rate: float | None = None
    max_steps: int | None = None
    task_override: str | None = None

    @property
    def datasets_csv(self) -> str:
        return ",".join(self.datasets)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _merge_stage_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _parse_datasets(raw: str | list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return tuple(parts)
    return tuple(str(x).strip() for x in raw if str(x).strip())


def _load_tower_exit_weights(stage: str) -> dict[str, float]:
    tower = _load_yaml(TOWER_YML)
    exits = tower.get("exits", {})
    weights: dict[str, float] = {}
    for exit_name, cfg in exits.items():
        if not isinstance(cfg, dict):
            continue
        stage_weights = cfg.get("loss_weights", {})
        if isinstance(stage_weights, dict) and stage in stage_weights:
            weights[exit_name] = float(stage_weights[stage])
    return weights


def _load_tower_freeze(stage: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tower = _load_yaml(TOWER_YML)
    stage_cfg = tower.get("stage_freeze", {}).get(stage, {})
    if not isinstance(stage_cfg, dict):
        return (), ()
    train = tuple(stage_cfg.get("train", []) or [])
    freeze = tuple(stage_cfg.get("freeze", []) or [])
    return train, freeze


def load_stage_configs(*, include_tower: bool = True) -> dict[str, StageInfo]:
    """Load per-stage training config merged from note/train.yml and tower overrides."""
    train_doc = _load_yaml(TRAIN_YML)
    tower_doc = _load_yaml(TOWER_TRAIN_YML) if include_tower else {}

    base_stages = train_doc.get("stages", {})
    tower_stages = tower_doc.get("stages", {})
    use_flow_tower_global = bool(tower_doc.get("use_flow_tower", False))

    stage_names = set(base_stages) | set(tower_stages)
    if include_tower and "world_pt" in tower_stages:
        stage_names.add("world_pt")

    out: dict[str, StageInfo] = {}
    for name in sorted(stage_names, key=lambda s: (STAGE_ORDER.index(s) if s in STAGE_ORDER else 99, s)):
        base = base_stages.get(name, {})
        override = tower_stages.get(name, {})
        merged = _merge_stage_dict(base if isinstance(base, dict) else {}, override if isinstance(override, dict) else {})

        use_tower = bool(merged.get("use_flow_tower", use_flow_tower_global and name in tower_stages))
        train_exits, freeze_exits = _load_tower_freeze(name) if use_tower else ((), ())

        out[name] = StageInfo(
            name=name,
            datasets=_parse_datasets(merged.get("datasets")),
            output_dir=merged.get("output_dir"),
            use_flow_tower=use_tower,
            loss_weights=dict(merged.get("loss_weights") or {}),
            tower_exit_weights=_load_tower_exit_weights(name) if use_tower else {},
            tower_train_exits=train_exits,
            tower_freeze_exits=freeze_exits,
            learning_rate=merged.get("learning_rate"),
            max_steps=merged.get("max_steps"),
            task_override=merged.get("task_override"),
        )
    return out


def list_available_datasets() -> list[dict[str, Any]]:
    """Return manifest entries enriched with role descriptions."""
    manifest = load_manifest()
    roles = load_roles()
    specs = load_dataset_specs()

    rows: list[dict[str, Any]] = []
    for dataset_key, entry in manifest.items():
        role = entry.get("role", specs.get(dataset_key, None) and specs[dataset_key].role or "")
        role_desc = roles.get(role, {}).get("desc", "")
        for stage, rel_path in entry.get("stages", {}).items():
            reg_key = f"{dataset_key}_{stage}"
            abs_path = (PROJECT_ROOT / rel_path).resolve()
            rows.append(
                {
                    "reg_key": reg_key,
                    "dataset_key": dataset_key,
                    "stage": stage,
                    "stage_label": DATA_STAGE_LABELS.get(stage, stage),
                    "role": role,
                    "role_desc": role_desc,
                    "path": str(abs_path),
                    "exists": abs_path.is_file(),
                    "manifest_samples": entry.get("samples"),
                }
            )
    rows.sort(key=lambda r: (r["stage"], r["dataset_key"]))
    return rows
