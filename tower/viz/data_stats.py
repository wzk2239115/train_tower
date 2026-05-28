from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tower.config import PROJECT_ROOT, load_roles
from tower.schema import UnifiedSample, count_image_tags, validate_sample
from tower.train.registry import load_manifest


def iter_jsonl(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream JSONL records."""
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def _to_sample(record: dict[str, Any]) -> UnifiedSample:
    return UnifiedSample(
        id=str(record.get("id", "")),
        image=record.get("image", ""),
        conversations=list(record.get("conversations") or []),
        width=record.get("width"),
        height=record.get("height"),
        audio=record.get("audio"),
        audio_values=record.get("audio_values"),
        audio_token_mask=record.get("audio_token_mask"),
        meta=dict(record.get("meta") or {}),
    )


def load_samples(
    reg_keys: list[str] | tuple[str, ...],
    *,
    limit_per_dataset: int | None = None,
    seed: int = 42,
) -> list[UnifiedSample]:
    """Load samples from one or more registered dataset keys (e.g. blip3o_short_pt)."""
    manifest = load_manifest()
    samples: list[UnifiedSample] = []
    rng = random.Random(seed)

    for reg_key in reg_keys:
        dataset_key, _, stage = reg_key.partition("_")
        if stage not in ("pt", "mt", "sft"):
            # Handle keys like blip3o_short_pt where dataset_key has underscore.
            parts = reg_key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            dataset_key, stage = parts

        entry = manifest.get(dataset_key)
        if not entry:
            continue
        rel_path = entry.get("stages", {}).get(stage)
        if not rel_path:
            continue
        path = (PROJECT_ROOT / rel_path).resolve()
        if not path.is_file():
            continue

        records = list(iter_jsonl(path, limit=limit_per_dataset))
        if limit_per_dataset is not None and len(records) > limit_per_dataset:
            records = rng.sample(records, limit_per_dataset)
        samples.extend(_to_sample(r) for r in records)
    return samples


@dataclass
class ModalityBreakdown:
    image_only: int = 0
    audio_only: int = 0
    image_text: int = 0
    image_audio: int = 0
    image_text_audio: int = 0
    text_only: int = 0
    other: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "image_only": self.image_only,
            "audio_only": self.audio_only,
            "image+text": self.image_text,
            "image+audio": self.image_audio,
            "image+text+audio": self.image_text_audio,
            "text_only": self.text_only,
            "other": self.other,
        }

    @property
    def total(self) -> int:
        return sum(self.as_dict().values())


def _has_image(sample: UnifiedSample) -> bool:
    if isinstance(sample.image, str):
        return bool(sample.image)
    return bool(sample.image)


def _has_audio(sample: UnifiedSample) -> bool:
    if sample.audio_values:
        return True
    if sample.audio and Path(sample.audio).is_file():
        return True
    return False


def _has_text(sample: UnifiedSample) -> bool:
    if not sample.conversations:
        return False
    text = " ".join(t.get("value", "") for t in sample.conversations).strip()
    return len(text.replace("<image>", "").strip()) > 0


def _classify_modality(sample: UnifiedSample) -> str:
    img = _has_image(sample)
    aud = _has_audio(sample)
    txt = _has_text(sample)
    if img and txt and aud:
        return "image_text_audio"
    if img and aud:
        return "image_audio"
    if img and txt:
        return "image_text"
    if img:
        return "image_only"
    if aud:
        return "audio_only"
    if txt:
        return "text_only"
    return "other"


@dataclass
class DatasetStats:
    reg_key: str
    dataset_key: str
    stage: str
    role: str
    path: str
    total: int = 0
    valid: int = 0
    validation_errors: Counter[str] = field(default_factory=Counter)
    modality: ModalityBreakdown = field(default_factory=ModalityBreakdown)
    caption_lengths: list[int] = field(default_factory=list)
    turn_counts: list[int] = field(default_factory=list)
    image_tag_counts: list[int] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    audio_patch_counts: list[int] = field(default_factory=list)
    audio_mask_true_counts: list[int] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        avg_caption = (
            sum(self.caption_lengths) / len(self.caption_lengths) if self.caption_lengths else 0.0
        )
        avg_turns = sum(self.turn_counts) / len(self.turn_counts) if self.turn_counts else 0.0
        return {
            "reg_key": self.reg_key,
            "dataset_key": self.dataset_key,
            "stage": self.stage,
            "role": self.role,
            "total": self.total,
            "valid": self.valid,
            "invalid": self.total - self.valid,
            "avg_caption_len": round(avg_caption, 1),
            "avg_turns": round(avg_turns, 1),
            "has_audio_pct": round(
                100.0
                * (
                    self.modality.image_audio
                    + self.modality.image_text_audio
                    + self.modality.audio_only
                )
                / max(self.total, 1),
                1,
            ),
        }


def compute_dataset_stats(
    reg_key: str,
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> DatasetStats:
    """Compute statistics for a single registered dataset."""
    manifest = load_manifest()
    parts = reg_key.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid reg_key: {reg_key}")
    dataset_key, stage = parts
    entry = manifest.get(dataset_key)
    if not entry:
        raise KeyError(f"Unknown dataset in manifest: {dataset_key}")

    rel_path = entry.get("stages", {}).get(stage)
    if not rel_path:
        raise KeyError(f"Stage '{stage}' not found for dataset '{dataset_key}'")

    path = (PROJECT_ROOT / rel_path).resolve()
    role = entry.get("role", "")

    stats = DatasetStats(
        reg_key=reg_key,
        dataset_key=dataset_key,
        stage=stage,
        role=role,
        path=str(path),
    )
    if not path.is_file():
        return stats

    records = list(iter_jsonl(path))
    if max_samples is not None and len(records) > max_samples:
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)

    for record in records:
        sample = _to_sample(record)
        stats.total += 1

        err = validate_sample(sample)
        if err:
            stats.validation_errors[err] += 1
        else:
            stats.valid += 1

        bucket = _classify_modality(sample)
        setattr(stats.modality, bucket, getattr(stats.modality, bucket) + 1)

        gpt_text = " ".join(
            t.get("value", "") for t in sample.conversations if t.get("from") == "gpt"
        ).strip()
        if gpt_text:
            stats.caption_lengths.append(len(gpt_text))

        stats.turn_counts.append(len(sample.conversations))
        stats.image_tag_counts.append(count_image_tags(sample.conversations))

        if sample.width:
            stats.widths.append(int(sample.width))
        if sample.height:
            stats.heights.append(int(sample.height))
        if sample.audio_values:
            stats.audio_patch_counts.append(len(sample.audio_values))
        if sample.audio_token_mask:
            stats.audio_mask_true_counts.append(sum(1 for x in sample.audio_token_mask if x))

    return stats


@dataclass
class StageDataSummary:
    stage: str
    reg_keys: tuple[str, ...]
    datasets: list[DatasetStats] = field(default_factory=list)
    role_counts: Counter[str] = field(default_factory=Counter)
    modality: ModalityBreakdown = field(default_factory=ModalityBreakdown)
    validation_errors: Counter[str] = field(default_factory=Counter)

    @property
    def total_samples(self) -> int:
        return sum(d.total for d in self.datasets)

    @property
    def valid_samples(self) -> int:
        return sum(d.valid for d in self.datasets)

    def metrics_table(self) -> list[dict[str, Any]]:
        return [d.to_row() for d in self.datasets]


def summarize_stage_data(
    reg_keys: list[str] | tuple[str, ...],
    *,
    stage: str = "",
    max_samples_per_dataset: int | None = None,
    seed: int = 42,
) -> StageDataSummary:
    """Aggregate stats across selected datasets for a training stage."""
    summary = StageDataSummary(stage=stage, reg_keys=tuple(reg_keys))
    roles = load_roles()

    for reg_key in reg_keys:
        try:
            ds = compute_dataset_stats(reg_key, max_samples=max_samples_per_dataset, seed=seed)
        except (KeyError, ValueError):
            continue
        summary.datasets.append(ds)
        summary.role_counts[ds.role] += ds.total
        for k, v in ds.validation_errors.items():
            summary.validation_errors[k] += v
        for field_name in ModalityBreakdown.__dataclass_fields__:
            summary_val = getattr(summary.modality, field_name)
            ds_val = getattr(ds.modality, field_name)
            setattr(summary.modality, field_name, summary_val + ds_val)

    return summary
