from __future__ import annotations

import copy
from typing import Any

from tower.train.config import TrainConfig


def flip_to_t2i(source: dict[str, Any]) -> dict[str, Any]:
    """Convert caption-style sample to T2I (caption -> image)."""
    rec = copy.deepcopy(source)
    convs = rec.get("conversations") or []
    caption = ""
    for turn in convs:
        if turn.get("from") == "gpt" and turn.get("value"):
            caption = turn["value"].strip()
            break
    if not caption:
        return rec
    rec["conversations"] = [
        {"from": "human", "value": caption},
        {"from": "gpt", "value": "<image>"},
    ]
    meta = dict(rec.get("meta") or {})
    meta["task"] = "t2i"
    rec["meta"] = meta
    return rec


def sample_task(source: dict[str, Any], cfg: TrainConfig) -> str:
    if cfg.task_override:
        return cfg.task_override
    meta = source.get("meta") or {}
    return meta.get("task", "understanding")
