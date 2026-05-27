from __future__ import annotations

import copy
import random
from typing import Any

import torch

from tower.train.config import TrainConfig
from tower.train.tasks import flip_to_t2i, sample_task


class UnifiedTrainDataset:
    """Wrap NEO LazySupervisedDataset with task-aware sample preprocessing."""

    def __init__(self, base_dataset, cfg: TrainConfig):
        self._base = base_dataset
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        num_retries = 3
        for attempt in range(num_retries):
            try:
                return self._fetch(index)
            except Exception:
                if attempt == num_retries - 1:
                    index = random.randint(0, len(self) - 1)
        return self._fetch(index)

    def _fetch(self, index: int) -> dict[str, Any]:
        raw = self._base.list_data_dict[index]
        if isinstance(raw, dict):
            sources = [copy.deepcopy(raw)]
        elif isinstance(raw, list):
            sources = copy.deepcopy(raw)
        else:
            sources = [raw]

        src = sources[0] if isinstance(sources[0], dict) else sources[0]
        task = sample_task(src, self.cfg)
        if task == "t2i":
            sources[0] = flip_to_t2i(src)

        item = self._base._get_item(sources)
        item["task"] = task
        item["is_gen"] = task in ("t2i", "interleave")
        return item


def make_unified_data_module(tokenizer, data_args, training_args, cfg: TrainConfig):
    from neo.data.data_processor import FlattenedDataCollatorForSupervisedDataset, LazySupervisedDataset

    base = LazySupervisedDataset(tokenizer, data_args=data_args)
    train_dataset = UnifiedTrainDataset(base, cfg)
    collator = UnifiedCollator(
        base_collator=FlattenedDataCollatorForSupervisedDataset(
            tokenizer=tokenizer, data_args=data_args, training_args=training_args
        ),
        cfg=cfg,
    )
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=collator)


class UnifiedCollator:
    def __init__(self, base_collator, cfg: TrainConfig):
        self.base_collator = base_collator
        self.cfg = cfg

    def __call__(self, instances):
        batch = self.base_collator(instances)
        batch["tasks"] = [inst.get("task", "understanding") for inst in instances]
        batch["is_gen"] = [bool(inst.get("is_gen", False)) for inst in instances]

        seq_len = batch["input_ids"].shape[1]
        indicators = torch.zeros(seq_len, dtype=torch.bool)
        boundaries = batch.get("seq_boundaries")
        if boundaries is not None:
            if not isinstance(boundaries, torch.Tensor):
                boundaries = torch.tensor(boundaries, dtype=torch.long)
            for i, is_gen in enumerate(batch["is_gen"]):
                if not is_gen:
                    continue
                start = int(boundaries[i].item()) if i < len(boundaries) else 0
                end = int(boundaries[i + 1].item()) if i + 1 < len(boundaries) else seq_len
                indicators[start:end] = True
        elif self.cfg.task_override == "t2i" or all(batch["is_gen"]):
            indicators[:] = True
        batch["image_gen_indicators"] = indicators
        return batch
