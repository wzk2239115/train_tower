from __future__ import annotations

from pathlib import Path

import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)

HEAD_EXPORT_MAP = {
    "world_model.pt": "world_elf",
    "semantic_model.pt": "semantic_elf",
    "language_model.pt": "understanding_elf",
    "generator.pt": "generative_elf",
}


def _to_cpu_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def export_multi_artifacts(model, output_dir: str) -> Path:
    """
    Export a shared backbone plus head-only artifacts.

    Output layout:
      <output_dir>/checkpoint/backbone.pt
      <output_dir>/checkpoint/world_model.pt
      <output_dir>/checkpoint/semantic_model.pt
      <output_dir>/checkpoint/language_model.pt
      <output_dir>/checkpoint/generator.pt
    """
    export_dir = Path(output_dir) / "checkpoint"
    export_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "model"):
        raise ValueError("Expected wrapped training model with .model backbone")

    backbone_path = export_dir / "backbone.pt"
    torch.save(_to_cpu_state(model.model.state_dict()), backbone_path)
    logger.info("Exported backbone artifact: %s", backbone_path)

    tower_exits = getattr(model, "tower_exits", None)
    if tower_exits is None:
        logger.warning("Model has no tower_exits; skipped head artifact exports.")
        return export_dir

    for filename, exit_name in HEAD_EXPORT_MAP.items():
        if exit_name not in tower_exits:
            logger.warning("Missing tower exit '%s'; skip %s", exit_name, filename)
            continue
        exit_module = tower_exits[exit_name]
        payload = {
            "exit_name": exit_name,
            "state_dict": _to_cpu_state(exit_module.state_dict()),
        }
        out_path = export_dir / filename
        torch.save(payload, out_path)
        logger.info("Exported head artifact: %s", out_path)

    return export_dir
