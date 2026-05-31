from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from tower.config import PROJECT_ROOT
from tower.train.config import TrainConfig

SIZES_DIR = PROJECT_ROOT / "configs" / "sizes"

DEFAULT_EXIT_DEPTH_FRAC: dict[str, float | None] = {
    "world_elf": 0.28,
    "audio_elf": 0.44,
    "semantic_elf": 0.60,
    "understanding_elf": 0.84,
    "generative_elf": None,
}

DEFAULT_STAGE_SHALLOW_WORLD_PT_FRAC = 0.308


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def list_size_presets() -> list[str]:
    return sorted(
        p.stem
        for p in SIZES_DIR.glob("*.yaml")
        if p.name != "_schema.yaml" and not p.name.startswith("_")
    )


def load_size_preset(name: str) -> dict[str, Any]:
    path = SIZES_DIR / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(list_size_presets()) or "(none)"
        raise FileNotFoundError(f"Unknown size preset '{name}'. Available: {available}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Size preset {path} must be a mapping")
    return raw


def exit_after_layer(num_hidden_layers: int, frac: float | None) -> int:
    if frac is None:
        return num_hidden_layers - 1
    return min(num_hidden_layers - 1, max(0, round(frac * (num_hidden_layers - 1))))


def shallow_world_pt_layers(num_hidden_layers: int, frac: float) -> int:
    return max(1, round(frac * num_hidden_layers))


def scale_tower_yaml(
    base_tower: dict[str, Any],
    *,
    num_hidden_layers: int,
    hidden_size: int,
    exit_depth_frac: dict[str, float | None],
    stage_shallow_world_pt_frac: float,
) -> dict[str, Any]:
    tower = copy.deepcopy(base_tower)
    tower["num_hidden_layers"] = num_hidden_layers
    tower["hidden_size"] = hidden_size

    exits = tower.get("exits") or {}
    for name, cfg in exits.items():
        if not isinstance(cfg, dict):
            continue
        frac = exit_depth_frac.get(name, DEFAULT_EXIT_DEPTH_FRAC.get(name))
        cfg["after_layer"] = exit_after_layer(num_hidden_layers, frac)

    shallow = dict(tower.get("stage_shallow_train_layers") or {})
    if "world_pt" in shallow or stage_shallow_world_pt_frac > 0:
        shallow["world_pt"] = shallow_world_pt_layers(num_hidden_layers, stage_shallow_world_pt_frac)
    tower["stage_shallow_train_layers"] = shallow
    return tower


def merge_model_config_dict(base: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    llm_overrides = preset.get("llm") or {}
    vision_overrides = preset.get("vision") or {}
    if llm_overrides:
        merged["llm_config"] = _deep_update(merged.get("llm_config") or {}, llm_overrides)
        layers = llm_overrides.get("num_hidden_layers")
        if layers is not None:
            merged["llm_config"]["max_window_layers"] = llm_overrides.get(
                "max_window_layers", layers
            )
    if vision_overrides:
        merged["vision_config"] = _deep_update(merged.get("vision_config") or {}, vision_overrides)
    if "fm_head_dim" in preset:
        merged["fm_head_dim"] = preset["fm_head_dim"]
    return merged


def _resolve_config_dir(path: str | None) -> Path:
    if not path:
        raise ValueError("model_config_path is required to apply a size preset")
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_base_model_config_json(model_config_path: str) -> dict[str, Any]:
    config_file = _resolve_config_dir(model_config_path) / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing model config: {config_file}")
    with config_file.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_model_config_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Return config.json contents, merged with active size preset when set."""
    base_path = cfg.model_config_path
    if not base_path:
        raise ValueError("model_config_path is required")
    base = load_base_model_config_json(base_path)
    preset_name = _preset_name_from_cfg(cfg)
    if not preset_name:
        return base
    preset = load_size_preset(preset_name)
    return merge_model_config_dict(base, preset)


def _preset_name_from_cfg(cfg: TrainConfig) -> str | None:
    name = (cfg.size_preset or cfg.model_size or "").strip()
    return name or None


def build_scaled_tower_raw(preset: dict[str, Any]) -> dict[str, Any]:
    from tower.unify.tower_config import TOWER_YML

    with TOWER_YML.open(encoding="utf-8") as f:
        base_tower = yaml.safe_load(f) or {}

    llm = preset.get("llm") or {}
    num_hidden_layers = int(llm.get("num_hidden_layers", base_tower.get("num_hidden_layers", 26)))
    hidden_size = int(llm.get("hidden_size", base_tower.get("hidden_size", 768)))

    tower_section = preset.get("tower") or {}
    exit_depth_frac = dict(DEFAULT_EXIT_DEPTH_FRAC)
    exit_depth_frac.update(tower_section.get("exit_depth_frac") or {})
    shallow_frac = float(
        tower_section.get(
            "stage_shallow_world_pt_frac",
            DEFAULT_STAGE_SHALLOW_WORLD_PT_FRAC,
        )
    )
    return scale_tower_yaml(
        base_tower,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        exit_depth_frac=exit_depth_frac,
        stage_shallow_world_pt_frac=shallow_frac,
    )


def apply_size_preset_to_train_config(
    cfg: TrainConfig,
    *,
    cli_size: str | None = None,
) -> TrainConfig:
    """Apply named preset to TrainConfig and register tower overlay for load_tower_config()."""
    from tower.unify.tower_config import clear_active_tower_overlay, set_active_tower_overlay

    preset_name = (cli_size or _preset_name_from_cfg(cfg) or "").strip()
    if not preset_name:
        clear_active_tower_overlay()
        return cfg

    preset = load_size_preset(preset_name)
    if preset.get("model_config_path"):
        cfg.model_config_path = str(preset["model_config_path"])

    llm = preset.get("llm") or {}
    if "num_hidden_layers" in llm:
        cfg.num_hidden_layers = int(llm["num_hidden_layers"])

    cfg.size_preset = preset_name
    tower_raw = build_scaled_tower_raw(preset)
    set_active_tower_overlay(tower_raw)
    return cfg


def assert_model_tower_layer_consistency(cfg: TrainConfig) -> None:
    """Validate LLM depth in merged config.json matches tower overlay."""
    from tower.unify.tower_config import load_tower_config

    model = resolve_model_config_dict(cfg)
    tower = load_tower_config()
    llm_layers = int(model["llm_config"]["num_hidden_layers"])
    if tower.num_hidden_layers != llm_layers:
        raise ValueError(
            f"Tower num_hidden_layers={tower.num_hidden_layers} != "
            f"llm num_hidden_layers={llm_layers}"
        )
    for spec in tower.exits:
        if spec.after_layer >= llm_layers:
            raise ValueError(
                f"Exit {spec.name} after_layer={spec.after_layer} >= num_layers={llm_layers}"
            )
