from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tower.config import PROJECT_ROOT


@dataclass
class TrainingRun:
    name: str
    path: Path
    stage: str | None = None
    global_step: int = 0
    max_steps: int | None = None
    train_loss: float | None = None
    log_history: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def loss_curve(self) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        for entry in self.log_history:
            if "loss" in entry and "step" in entry:
                points.append((int(entry["step"]), float(entry["loss"])))
        return points

    @property
    def grad_norm_curve(self) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        for entry in self.log_history:
            if "grad_norm" in entry and "step" in entry:
                points.append((int(entry["step"]), float(entry["grad_norm"])))
        return points

    @property
    def lr_curve(self) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        for entry in self.log_history:
            if "learning_rate" in entry and "step" in entry:
                points.append((int(entry["step"]), float(entry["learning_rate"])))
        return points


def _infer_stage(name: str, config: dict[str, Any]) -> str | None:
    for key in ("stage", "train_stage"):
        if key in config:
            return str(config[key])
    train_cfg = config.get("train_config") or {}
    if isinstance(train_cfg, dict) and "stage" in train_cfg:
        return str(train_cfg["stage"])

    lowered = name.lower()
    for stage in (
        "world_pt",
        "understanding_warmup",
        "generation_pt",
        "unified_mt",
        "unified_sft",
        "uw",
        "gen_pt",
        "mt",
        "sft",
    ):
        if stage in lowered:
            return stage
    return None


def load_training_run(run_dir: Path) -> TrainingRun:
    """Load trainer_state.json (+ optional train_config.yaml) from an output directory."""
    run_dir = run_dir.resolve()
    state_path = run_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"No trainer_state.json in {run_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = {}

    for cfg_name in ("train_config.yaml", "train_config.yml"):
        cfg_path = run_dir / cfg_name
        if cfg_path.is_file():
            import yaml

            with cfg_path.open(encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                config = loaded
            break

    train_loss = None
    for entry in reversed(state.get("log_history", [])):
        if "train_loss" in entry:
            train_loss = float(entry["train_loss"])
            break

    return TrainingRun(
        name=run_dir.name,
        path=run_dir,
        stage=_infer_stage(run_dir.name, config),
        global_step=int(state.get("global_step", 0)),
        max_steps=state.get("max_steps"),
        train_loss=train_loss,
        log_history=list(state.get("log_history", [])),
        config=config,
    )


def discover_training_runs(
    outputs_root: Path | None = None,
    *,
    include_checkpoints: bool = False,
) -> list[TrainingRun]:
    """Find training runs under outputs/."""
    root = (outputs_root or PROJECT_ROOT / "outputs").resolve()
    if not root.is_dir():
        return []

    runs: list[TrainingRun] = []
    for state_path in sorted(root.rglob("trainer_state.json")):
        run_dir = state_path.parent
        if not include_checkpoints and run_dir.name.startswith("checkpoint-"):
            continue
        try:
            runs.append(load_training_run(run_dir))
        except (json.JSONDecodeError, OSError):
            continue

    # Prefer top-level run dirs over nested duplicates (keep longest path depth smallest).
    seen: set[str] = set()
    deduped: list[TrainingRun] = []
    for run in sorted(runs, key=lambda r: (len(r.path.parts), r.path.as_posix())):
        key = run.path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(run)
    return deduped
