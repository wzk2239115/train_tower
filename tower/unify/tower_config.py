from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tower.config import PROJECT_ROOT

TOWER_YML = PROJECT_ROOT / "note" / "tower.yml"


@dataclass(frozen=True)
class TowerExitSpec:
    name: str
    after_layer: int
    exit_type: str  # jepa | elf_fm | ce
    latent: str = "pixel_patch"  # vision_embed | pixel_patch | audio_embed | token_hidden
    elf_depth: int = 2
    ema_momentum: float = 0.996
    loss_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class TowerConfig:
    num_hidden_layers: int
    hidden_size: int
    exits: tuple[TowerExitSpec, ...]
    stage_freeze: dict[str, dict[str, list[str]]]

    def exit(self, name: str) -> TowerExitSpec:
        for spec in self.exits:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown tower exit '{name}'")

    def loss_weight(self, exit_name: str, stage: str) -> float:
        spec = self.exit(exit_name)
        return float(spec.loss_weights.get(stage, 0.0))

    def active_exits(self, stage: str) -> list[TowerExitSpec]:
        return [e for e in self.exits if self.loss_weight(e.name, stage) > 0]

    def hook_layers(self) -> list[int]:
        """Sorted unique layer indices where exits attach."""
        return sorted({e.after_layer for e in self.exits})


def load_tower_config(path: Path | None = None) -> TowerConfig:
    p = path or TOWER_YML
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    exits: list[TowerExitSpec] = []
    for name, cfg in (raw.get("exits") or {}).items():
        exits.append(
            TowerExitSpec(
                name=name,
                after_layer=int(cfg["after_layer"]),
                exit_type=str(cfg["type"]),
                latent=str(cfg.get("latent", "pixel_patch")),
                elf_depth=int(cfg.get("elf_depth", 2)),
                ema_momentum=float(cfg.get("ema_momentum", 0.996)),
                loss_weights={str(k): float(v) for k, v in (cfg.get("loss_weights") or {}).items()},
                description=str(cfg.get("description", "")),
            )
        )
    exits.sort(key=lambda e: e.after_layer)

    return TowerConfig(
        num_hidden_layers=int(raw.get("num_hidden_layers", 26)),
        hidden_size=int(raw.get("hidden_size", 768)),
        exits=tuple(exits),
        stage_freeze=raw.get("stage_freeze") or {},
    )
