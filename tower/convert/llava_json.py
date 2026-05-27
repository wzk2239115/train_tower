from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tower.config import DatasetSpec
from tower.convert.base import BaseConverter
from tower.io.images import copy_or_link_image, image_size, resolve_coco_image
from tower.io.json_stream import iter_json_array
from tower.schema import UnifiedSample


class LlavaJsonConverter(BaseConverter):
    def _find_json(self, spec: DatasetSpec) -> Path:
        files = sorted(spec.raw_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON annotation under {spec.raw_dir}")
        return files[0]

    def _coco_root(self) -> Path:
        from tower.config import RAW_DIR

        return RAW_DIR / "OpenDataLab___COCO_2017"

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        json_path = self._find_json(spec)
        coco_root = self._coco_root()
        images_dir = spec.images_dir
        images_dir.mkdir(parents=True, exist_ok=True)

        records = iter_json_array(json_path)

        count = 0
        for rec in records:
            rel_image = rec.get("image")
            if not rel_image:
                continue

            if isinstance(rel_image, list):
                resolved: list[str] = []
                for rel in rel_image:
                    src = resolve_coco_image(rel, coco_root)
                    if src is None:
                        resolved = []
                        break
                    dest = images_dir / Path(rel).name
                    copy_or_link_image(src, dest)
                    resolved.append(str(dest.resolve()))
                if not resolved:
                    continue
                image_field: str | list[str] = resolved if len(resolved) > 1 else resolved[0]
                primary = Path(resolved[0])
            else:
                src = resolve_coco_image(rel_image, coco_root)
                if src is None:
                    continue
                dest = images_dir / Path(rel_image).name
                copy_or_link_image(src, dest)
                image_field = str(dest.resolve())
                primary = dest

            conversations = rec.get("conversations") or []
            if not conversations:
                continue

            w, h = image_size(primary)
            source_id = rec.get("id", Path(str(rel_image)).stem)
            yield UnifiedSample(
                id=f"{spec.key}_{source_id}",
                image=image_field,
                width=w,
                height=h,
                conversations=conversations,
                meta={"dataset": spec.key, "role": spec.role, "source_id": str(source_id)},
            )

            count += 1
            if limit is not None and count >= limit:
                return
