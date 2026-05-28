from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tower.config import PROJECT_ROOT
from tower.viz.data_stats import summarize_stage_data
from tower.viz.stages import StageInfo, load_stage_configs


def parse_dataset_keys(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


@dataclass
class StageSelection:
    """User-adjustable dataset selection for one training stage."""

    stage: str
    selected_reg_keys: list[str] = field(default_factory=list)
    max_samples_per_dataset: int | None = None

    @classmethod
    def from_stage_defaults(cls, stage: str) -> StageSelection:
        configs = load_stage_configs()
        info = configs.get(stage)
        if not info:
            raise KeyError(f"Unknown stage: {stage}")
        return cls(stage=stage, selected_reg_keys=list(info.datasets))

    @property
    def datasets_csv(self) -> str:
        return ",".join(self.selected_reg_keys)

    def toggle(self, reg_key: str, enabled: bool) -> None:
        if enabled and reg_key not in self.selected_reg_keys:
            self.selected_reg_keys.append(reg_key)
        elif not enabled and reg_key in self.selected_reg_keys:
            self.selected_reg_keys.remove(reg_key)

    def set_selection(self, reg_keys: list[str]) -> None:
        self.selected_reg_keys = list(reg_keys)

    def summary(self):
        return summarize_stage_data(
            self.selected_reg_keys,
            stage=self.stage,
            max_samples_per_dataset=self.max_samples_per_dataset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "datasets": self.selected_reg_keys,
            "datasets_csv": self.datasets_csv,
            "max_samples_per_dataset": self.max_samples_per_dataset,
        }

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return out


def export_selection_yaml(
    selections: dict[str, StageSelection],
    path: Path | str | None = None,
) -> Path:
    """Export multi-stage dataset selections for reproducible training configs."""
    out = Path(path) if path else PROJECT_ROOT / "exports" / "viz" / "stage_selections.yml"
    payload = {stage: sel.to_dict() for stage, sel in selections.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def stage_info_for(stage: str) -> StageInfo:
    configs = load_stage_configs()
    if stage not in configs:
        raise KeyError(f"Unknown stage: {stage}")
    return configs[stage]
