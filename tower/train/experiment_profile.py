from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tower.config import PROJECT_ROOT
from tower.train.config import TrainConfig, _normalize_curriculum, validate_pretrain_config
from tower.train.stage_boundaries import STAGE_ORDER, resolve_stage_boundaries

EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"


@dataclass
class ExperimentProfile:
    name: str
    description: str = ""
    size_preset: str = "500m"
    train_config: str = "configs/train/continuous.yaml"
    overrides: dict[str, Any] = field(default_factory=dict)
    curriculum: list[dict[str, Any]] = field(default_factory=list)
    stage_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    viz: dict[str, Any] = field(default_factory=dict)

    @property
    def viz_output_dir(self) -> Path:
        raw = self.viz.get("output_dir") or f"exports/viz/{self.name}"
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def viz_metrics(self) -> list[str]:
        raw = self.viz.get("metrics") or ["loss"]
        return [str(m) for m in raw]


def list_experiment_profiles() -> list[str]:
    return sorted(
        p.stem
        for p in EXPERIMENTS_DIR.glob("*.yaml")
        if p.name != "_schema.yaml" and not p.name.startswith("_")
    )


def load_experiment_profile(name: str) -> ExperimentProfile:
    path = EXPERIMENTS_DIR / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(list_experiment_profiles()) or "(none)"
        raise FileNotFoundError(f"Unknown experiment profile '{name}'. Available: {available}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment profile {path} must be a mapping")

    profile_name = str(raw.get("name") or name)
    stage_overrides = _parse_stage_overrides(raw.get("stages"))
    curriculum = list(raw.get("curriculum") or [])
    if not curriculum and stage_overrides and raw.get("curriculum_from_stages"):
        curriculum = stages_dict_to_curriculum(stage_overrides)

    return ExperimentProfile(
        name=profile_name,
        description=str(raw.get("description") or ""),
        size_preset=str(raw.get("size_preset") or "500m"),
        train_config=str(raw.get("train_config") or "configs/train/continuous.yaml"),
        overrides=dict(raw.get("overrides") or {}),
        curriculum=curriculum,
        stage_overrides=stage_overrides,
        viz=dict(raw.get("viz") or {}),
    )


def _parse_stage_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for stage, spec in raw.items():
        if isinstance(spec, dict):
            out[str(stage)] = dict(spec)
    return out


def stages_dict_to_curriculum(stages: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build curriculum until_step entries from per-stage max_steps specs."""
    curriculum: list[dict[str, Any]] = []
    cumulative = -1
    for stage_name in STAGE_ORDER:
        if stage_name not in stages:
            continue
        spec = stages[stage_name]
        steps = int(spec.get("max_steps", spec.get("steps", 0)))
        if steps <= 0:
            continue
        cumulative += steps
        entry: dict[str, Any] = {"stage": stage_name, "until_step": cumulative}
        if "datasets" in spec:
            ds = spec["datasets"]
            entry["datasets"] = ",".join(ds) if isinstance(ds, list) else str(ds)
        for key in (
            "max_seq_length",
            "max_pixels",
            "min_pixels",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "loss_weights",
            "task_override",
            "tower_decoder_prob",
            "cfg_label_drop_prob",
            "tower_self_cond_prob",
            "tower_self_cond_cfg_min",
            "tower_self_cond_cfg_max",
        ):
            if key in spec:
                entry[key] = spec[key]
        curriculum.append(entry)
    return curriculum


def _merge_curriculum_by_stage(
    base: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not overrides:
        return base
    merged: list[dict[str, Any]] = []
    for item in base:
        entry = dict(item)
        stage = str(entry.get("stage", ""))
        patch = overrides.get(stage)
        if patch:
            for key, value in patch.items():
                if key in ("max_steps", "steps"):
                    continue
                if key == "datasets" and isinstance(value, list):
                    entry["datasets"] = ",".join(value)
                else:
                    entry[key] = value
        merged.append(entry)
    return merged


def load_train_config_from_experiment(
    profile_name: str,
    *,
    size_preset: str | None = None,
) -> TrainConfig:
    from tower.train.config import load_train_config
    from tower.train.size_preset import apply_size_preset_to_train_config

    profile = load_experiment_profile(profile_name)
    config_path = Path(profile.train_config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    preset = (size_preset or profile.size_preset or "").strip() or None
    cfg = load_train_config(config_path=config_path, size_preset=preset)
    cfg.experiment_profile = profile_name

    for key, value in profile.overrides.items():
        if key in TrainConfig.__dataclass_fields__:
            setattr(cfg, key, value)

    if profile.curriculum:
        cfg.curriculum = _normalize_curriculum(copy.deepcopy(profile.curriculum))
    elif profile.stage_overrides:
        cfg.curriculum = _normalize_curriculum(
            _merge_curriculum_by_stage(cfg.curriculum, profile.stage_overrides)
        )

    validate_pretrain_config(cfg)
    return apply_size_preset_to_train_config(cfg, cli_size=size_preset)


def summarize_profile(profile_name: str) -> dict[str, Any]:
    """Return display metadata for `tower experiment list`."""
    profile = load_experiment_profile(profile_name)
    cfg = load_train_config_from_experiment(profile_name)
    boundaries = resolve_stage_boundaries(cfg)
    return {
        "name": profile.name,
        "description": profile.description,
        "size_preset": cfg.size_preset or profile.size_preset,
        "train_config": profile.train_config,
        "max_steps": cfg.max_steps,
        "output_dir": cfg.output_dir,
        "stages": [
            {
                "stage": b.stage,
                "steps": b.step_count,
                "datasets": b.datasets,
            }
            for b in boundaries
        ],
    }
