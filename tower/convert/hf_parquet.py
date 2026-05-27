from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from tower.config import DatasetSpec
from tower.convert.base import BaseConverter
from tower.io.images import bytes_to_jpeg_path, image_size
from tower.schema import UnifiedSample, caption_conversation, qa_conversation


def _first_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "none":
            return None
        return s
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            s = _first_str(item)
            if s:
                return s
    return None


def _join_list(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        parts = [_first_str(x) for x in value]
        parts = [p for p in parts if p]
        if parts:
            return ", ".join(parts)
    return _first_str(value)


def _image_from_row(row: pd.Series) -> tuple[bytes, str] | None:
    image = row.get("image")
    if isinstance(image, dict) and image.get("bytes"):
        path = image.get("path") or "image.jpg"
        return image["bytes"], str(path)
    return None


class HfParquetConverter(BaseConverter):
    def _parquet_files(self, spec: DatasetSpec) -> list[Path]:
        patterns = ["**/*.parquet"]
        files: list[Path] = []
        for pat in patterns:
            files.extend(spec.raw_dir.glob(pat))
        # exclude huggingface cache metadata paths if any slip through
        files = [p for p in files if ".cache" not in p.parts]
        return sorted(set(files))

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        files = self._parquet_files(spec)
        if not files:
            raise FileNotFoundError(f"No parquet files under {spec.raw_dir}")

        count = 0
        images_dir = spec.images_dir
        images_dir.mkdir(parents=True, exist_ok=True)

        for parquet_path in files:
            df = pd.read_parquet(parquet_path)
            for idx, row in df.iterrows():
                parsed = _image_from_row(row)
                if parsed is None:
                    continue
                img_bytes, img_name = parsed
                stem = Path(str(img_name)).stem or f"{spec.key}_{idx}"
                dest = images_dir / f"{stem}.jpg"
                bytes_to_jpeg_path(img_bytes, dest)
                w, h = image_size(dest)

                sample = self._row_to_sample(spec, row, idx, str(dest.resolve()), w, h, stem)
                if sample is None:
                    continue

                yield sample
                count += 1
                if limit is not None and count >= limit:
                    return

    def _row_to_sample(
        self,
        spec: DatasetSpec,
        row: pd.Series,
        idx: Any,
        image_path: str,
        width: int,
        height: int,
        stem: str,
    ) -> UnifiedSample | None:
        role = spec.role

        if role == "grounding":
            question = _first_str(row.get("question")) or "Locate the referred object."
            answer = _first_str(row.get("answer"))
            if not answer:
                return None
            bbox = row.get("bbox")
            bbox_str = ""
            if bbox is not None and hasattr(bbox, "tolist"):
                bbox_str = ", ".join(f"{x:.1f}" for x in bbox.tolist())
            gpt = answer
            if bbox_str:
                gpt = f"{answer} [bbox: {bbox_str}]"
            conversations = qa_conversation(f"Locate: {question}", gpt)

        elif role == "ocr_caption":
            caption = _first_str(row.get("reference_strs")) or _first_str(row.get("caption_str"))
            if not caption:
                classes = _join_list(row.get("image_classes"))
                if classes:
                    caption = f"Scene containing: {classes}."
            if not caption:
                return None
            conversations = caption_conversation(
                caption,
                human_prompt="<image>\nDescribe the text in this image.",
            )

        elif role == "document_vqa":
            question = _first_str(row.get("question"))
            answer = _first_str(row.get("answers"))
            if not question or not answer:
                return None
            conversations = qa_conversation(question, answer)

        elif role == "chart_vqa":
            question = _first_str(row.get("query"))
            answer = _first_str(row.get("label"))
            if not question or not answer:
                return None
            conversations = qa_conversation(question, answer)

        else:
            return None

        source_id = _first_str(row.get("question_id")) or _first_str(row.get("questionId")) or stem
        return UnifiedSample(
            id=f"{spec.key}_{source_id}",
            image=image_path,
            width=width,
            height=height,
            conversations=conversations,
            meta={"dataset": spec.key, "role": role, "source_id": str(source_id)},
        )
