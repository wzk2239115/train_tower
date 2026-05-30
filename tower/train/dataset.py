from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import torch

from tower.train.config import TrainConfig
from tower.train.tasks import flip_to_t2i, sample_task
from tower.train.vision_batch import reconcile_vision_inputs
from tower.io.audio import audio_file_to_patch_features


class UnifiedTrainDataset:
    """Wrap NEO LazySupervisedDataset with task-aware sample preprocessing."""

    def __init__(self, base_dataset, cfg: TrainConfig):
        self._base = base_dataset
        self.cfg = cfg
        self._audio_cache: dict[str, torch.Tensor] = {}

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
        if isinstance(src, dict):
            audio_values = self._resolve_audio_values(src)
            if audio_values is not None:
                item["audio_values"] = audio_values
            if "audio_token_mask" in src:
                item["audio_token_mask"] = src.get("audio_token_mask")
        return item

    def _resolve_audio_values(self, src: dict[str, Any]) -> torch.Tensor | list[list[float]] | None:
        if "audio_values" in src and src.get("audio_values") is not None:
            audio_values = src.get("audio_values")
            if isinstance(audio_values, torch.Tensor):
                return audio_values
            return torch.tensor(audio_values, dtype=torch.float32)

        path = src.get("audio") or src.get("audio_path")
        if not isinstance(path, str) or not path.strip():
            return None
        p = Path(path)
        if not p.is_absolute():
            data_path = getattr(getattr(self._base, "data_args", None), "data_path", None)
            base = Path(data_path) if isinstance(data_path, str) and data_path else Path.cwd()
            p = (base / p).resolve()
        cache_key = str(p)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        if not p.is_file():
            return None
        feats = audio_file_to_patch_features(p)
        self._audio_cache[cache_key] = feats
        return feats


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
        batch = self._reconcile_vision_batch(batch)
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

        audio_values = [inst.get("audio_values") for inst in instances]
        if any(v is not None for v in audio_values):
            batch["audio_values"] = audio_values

        audio_masks = [inst.get("audio_token_mask") for inst in instances]
        if any(m is not None for m in audio_masks):
            audio_mask = torch.zeros(seq_len, dtype=torch.bool)
            if boundaries is not None:
                for i, local in enumerate(audio_masks):
                    if local is None:
                        continue
                    local_t = local if isinstance(local, torch.Tensor) else torch.tensor(local)
                    local_t = local_t.to(dtype=torch.bool).view(-1)
                    start = int(boundaries[i].item()) if i < len(boundaries) else 0
                    end = int(boundaries[i + 1].item()) if i + 1 < len(boundaries) else seq_len
                    span = max(end - start, 0)
                    n = min(span, local_t.numel())
                    if n > 0:
                        audio_mask[start : start + n] = local_t[:n]
            else:
                local = audio_masks[0]
                if local is not None:
                    local_t = local if isinstance(local, torch.Tensor) else torch.tensor(local)
                    local_t = local_t.to(dtype=torch.bool).view(-1)
                    n = min(seq_len, local_t.numel())
                    if n > 0:
                        audio_mask[:n] = local_t[:n]
            batch["audio_token_mask"] = audio_mask
        return batch

    def _reconcile_vision_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pixel_values = batch.get("pixel_values")
        image_grid_hw = batch.get("image_grid_hw")
        if (
            pixel_values is None
            or len(pixel_values) == 0
            or pixel_values[0] is None
            or not image_grid_hw
            or len(image_grid_hw) == 0
            or image_grid_hw[0] is None
        ):
            return batch

        flat = pixel_values[0]
        grid_hw = image_grid_hw[0]
        if not isinstance(grid_hw, torch.Tensor):
            grid_hw = torch.tensor(grid_hw, dtype=torch.long)

        num_patches = int(flat.shape[0])
        expected = int((grid_hw[:, 0] * grid_hw[:, 1]).sum().item())
        if expected != num_patches:
            flat, grid_hw = reconcile_vision_inputs(flat, grid_hw)
            batch["pixel_values"] = [flat]
            batch["image_grid_hw"] = [grid_hw]
        return batch
