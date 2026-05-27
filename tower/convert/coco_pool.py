from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tower.config import DatasetSpec
from tower.convert.base import BaseConverter
from tower.io.images import copy_or_link_image, image_size
from tower.schema import UnifiedSample


class CocoPoolConverter(BaseConverter):
    def _image_dirs(self, spec: DatasetSpec) -> list[Path]:
        root = spec.raw_dir
        candidates = [root / "val2017", root / "train2017", root / "sample" / "image"]
        return [d for d in candidates if d.is_dir()]

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        images_dir = spec.images_dir
        images_dir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        count = 0

        for img_dir in self._image_dirs(spec):
            for src in sorted(img_dir.glob("*.jpg")):
                if src.name in seen:
                    continue
                seen.add(src.name)

                dest = images_dir / src.name
                copy_or_link_image(src, dest)
                w, h = image_size(dest)
                stem = src.stem

                yield UnifiedSample(
                    id=f"{spec.key}_{stem}",
                    image=str(dest.resolve()),
                    width=w,
                    height=h,
                    conversations=[],
                    meta={"dataset": spec.key, "role": spec.role, "source_id": stem},
                )

                count += 1
                if limit is not None and count >= limit:
                    return
