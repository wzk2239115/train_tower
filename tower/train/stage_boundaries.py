from __future__ import annotations

from dataclasses import dataclass

from tower.train.config import TrainConfig

STAGE_ORDER = (
    "world_pt",
    "understanding_warmup",
    "generation_pt",
    "unified_mt",
    "unified_sft",
)


@dataclass
class StageBoundary:
    stage: str
    start_step: int
    end_step: int
    datasets: str = ""
    phase_index: int = 0

    @property
    def step_count(self) -> int:
        return self.end_step - self.start_step + 1


def resolve_stage_boundaries(cfg: TrainConfig) -> list[StageBoundary]:
    """Map global_step ranges to curriculum stages."""
    if not cfg.curriculum:
        return [
            StageBoundary(
                stage=cfg.stage,
                start_step=0,
                end_step=max(int(cfg.max_steps) - 1, 0),
                datasets=str(cfg.datasets),
                phase_index=0,
            )
        ]

    boundaries: list[StageBoundary] = []
    prev_until = -1
    for idx, item in enumerate(cfg.curriculum):
        stage = str(item["stage"])
        until = int(item["until_step"])
        start = prev_until + 1
        settings = cfg.curriculum_data_settings_for_step(start)
        boundaries.append(
            StageBoundary(
                stage=stage,
                start_step=start,
                end_step=until,
                datasets=str(settings.get("datasets", cfg.datasets)),
                phase_index=idx,
            )
        )
        prev_until = until
    return boundaries
