from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import torch
from transformers.utils import logging

from tower.train.config import TrainConfig
from tower.train.curriculum import CurriculumRuntime
from tower.train.packed_batch_monitor import attach_packed_batch_stats
from tower.train.tasks import flip_to_t2i, is_generation_task, sample_task
from tower.train.vision_batch import reconcile_vision_inputs
from tower.io.audio import audio_file_to_patch_features

logger = logging.get_logger(__name__)


class UnifiedTrainDataset:
    """Wrap NEO LazySupervisedDataset with task-aware sample preprocessing."""

    def __init__(self, base_dataset, cfg: TrainConfig):
        self._base = base_dataset
        self.cfg = cfg
        self._audio_cache: dict[str, torch.Tensor] = {}
        self._video_cache: dict[str, torch.Tensor] = {}

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
        item["is_gen"] = is_generation_task(task)
        if isinstance(src, dict):
            audio_values = self._resolve_audio_values(src)
            if audio_values is not None:
                item["audio_values"] = audio_values
            if "audio_token_mask" in src:
                item["audio_token_mask"] = src.get("audio_token_mask")
            video_values = self._resolve_video_values(src)
            if video_values is not None:
                item["video_values"] = video_values
            if "video_token_mask" in src:
                item["video_token_mask"] = src.get("video_token_mask")
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

    def _resolve_video_values(self, src: dict[str, Any]) -> torch.Tensor | None:
        if "video_values" in src and src.get("video_values") is not None:
            video_values = src.get("video_values")
            if isinstance(video_values, torch.Tensor):
                return video_values.to(dtype=torch.float32)
            return torch.tensor(video_values, dtype=torch.float32)

        path = src.get("video") or src.get("video_path")
        if not isinstance(path, str) or not path.strip():
            return None
        p = Path(path)
        if not p.is_absolute():
            data_path = getattr(getattr(self._base, "data_args", None), "data_path", None)
            base = Path(data_path) if isinstance(data_path, str) and data_path else Path.cwd()
            p = (base / p).resolve()
        cache_key = str(p)
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]
        if not p.is_file():
            return None
        feats = self._load_video_features(p)
        if feats is not None:
            self._video_cache[cache_key] = feats
        return feats

    def _load_video_features(self, path: Path) -> torch.Tensor | None:
        suffix = path.suffix.lower()
        if suffix in (".pt", ".pth", ".bin"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("video_values", "features", "feats", "x"):
                    if key in obj:
                        obj = obj[key]
                        break
            if not isinstance(obj, torch.Tensor):
                obj = torch.tensor(obj, dtype=torch.float32)
            return self._normalize_video_tensor(obj)

        # Lightweight fallback contract:
        # when no decoder dependency is available, consume raw frame features
        # saved as .npy/.npz where axis-0 is temporal and the last axis is feature dim.
        if suffix == ".npy":
            import numpy as np

            arr = np.load(path)
            return self._normalize_video_tensor(torch.from_numpy(arr))
        if suffix == ".npz":
            import numpy as np

            arrs = np.load(path)
            if len(arrs.files) == 0:
                return None
            arr = arrs[arrs.files[0]]
            return self._normalize_video_tensor(torch.from_numpy(arr))

        logger.warning(
            "Unsupported video file '%s'. Provide precomputed features via .pt/.npy/.npz",
            path,
        )
        return None

    def _normalize_video_tensor(self, value: torch.Tensor) -> torch.Tensor:
        video = value.detach().to(dtype=torch.float32)
        if video.ndim == 0:
            video = video.reshape(1, 1)
        elif video.ndim == 1:
            video = video.unsqueeze(0)
        elif video.ndim >= 3:
            video = video.reshape(-1, video.shape[-1])

        target_dim = int(getattr(self.cfg, "video_patch_dim", 1024))
        if target_dim > 0 and video.shape[-1] != target_dim:
            if video.shape[-1] > target_dim:
                video = video[..., :target_dim]
            else:
                video = torch.nn.functional.pad(video, (0, target_dim - video.shape[-1]))
        return video.contiguous()


def rebuild_unified_dataset_base(
    unified: UnifiedTrainDataset,
    *,
    tokenizer,
    data_args,
    datasets: str,
) -> None:
    """Reload LazySupervisedDataset when curriculum switches datasets."""
    from tower.unify.backends import import_lazy_supervised_dataset

    LazySupervisedDataset = import_lazy_supervised_dataset()
    data_args.dataset_use = datasets.strip()
    unified._base = LazySupervisedDataset(tokenizer, data_args=data_args)
    unified._audio_cache.clear()
    unified._video_cache.clear()
    logger.info("Rebuilt train dataset %s (%s samples)", datasets, len(unified))


def make_unified_data_module(tokenizer, data_args, training_args, cfg: TrainConfig):
    from tower.unify.backends import import_flattened_data_collator, import_lazy_supervised_dataset

    LazySupervisedDataset = import_lazy_supervised_dataset()
    FlattenedDataCollatorForSupervisedDataset = import_flattened_data_collator()
    base = LazySupervisedDataset(tokenizer, data_args=data_args)
    train_dataset = UnifiedTrainDataset(base, cfg)
    curriculum_runtime = CurriculumRuntime(cfg)
    collator = UnifiedCollator(
        base_collator=FlattenedDataCollatorForSupervisedDataset(
            tokenizer=tokenizer, data_args=data_args, training_args=training_args
        ),
        cfg=cfg,
        curriculum_runtime=curriculum_runtime,
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        curriculum_runtime=curriculum_runtime,
    )


class UnifiedCollator:
    def __init__(self, base_collator, cfg: TrainConfig, curriculum_runtime: CurriculumRuntime | None = None):
        self.base_collator = base_collator
        self.cfg = cfg
        self.curriculum_runtime = curriculum_runtime

    def __call__(self, instances):
        batch = self.base_collator(instances)
        batch = self._reconcile_vision_batch(batch)
        if self.curriculum_runtime is not None and self.curriculum_runtime.settings:
            batch["curriculum_phase"] = int(self.curriculum_runtime.phase_index)
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
        audio_mask = self._pack_token_mask(audio_masks, boundaries, seq_len)
        if audio_mask is not None:
            batch["audio_token_mask"] = audio_mask

        video_values = [inst.get("video_values") for inst in instances]
        if any(v is not None for v in video_values):
            batch["video_values"] = video_values

        video_masks = [inst.get("video_token_mask") for inst in instances]
        video_mask = self._pack_token_mask(video_masks, boundaries, seq_len)
        if video_mask is not None:
            batch["video_token_mask"] = video_mask

        from tower.unify.backends import import_data_constants

        img_context_token_id = self.base_collator.tokenizer.convert_tokens_to_ids(
            import_data_constants().IMG_CONTEXT_TOKEN
        )
        attach_packed_batch_stats(batch, self.cfg, img_context_token_id)
        return batch

    def _pack_token_mask(
        self,
        local_masks: list[Any],
        boundaries: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        if not any(m is not None for m in local_masks):
            return None
        packed = torch.zeros(seq_len, dtype=torch.bool)
        if boundaries is not None:
            for i, local in enumerate(local_masks):
                if local is None:
                    continue
                local_t = local if isinstance(local, torch.Tensor) else torch.tensor(local)
                local_t = local_t.to(dtype=torch.bool).view(-1)
                start = int(boundaries[i].item()) if i < len(boundaries) else 0
                end = int(boundaries[i + 1].item()) if i + 1 < len(boundaries) else seq_len
                span = max(end - start, 0)
                n = min(span, local_t.numel())
                if n > 0:
                    packed[start : start + n] = local_t[:n]
            return packed

        local = local_masks[0]
        if local is None:
            return packed
        local_t = local if isinstance(local, torch.Tensor) else torch.tensor(local)
        local_t = local_t.to(dtype=torch.bool).view(-1)
        n = min(seq_len, local_t.numel())
        if n > 0:
            packed[:n] = local_t[:n]
        return packed

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
        merge = max(1, int(round(1 / float(self.cfg.downsample_ratio))))
        flat, grid_hw = reconcile_vision_inputs(flat, grid_hw, spatial_merge=merge)
        batch["pixel_values"] = [flat]
        batch["image_grid_hw"] = [grid_hw]
        return batch
