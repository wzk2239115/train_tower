from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Iterator

from tower.config import DatasetSpec
from tower.convert.base import BaseConverter
from tower.io.images import bytes_to_jpeg_path, image_size
from tower.schema import UnifiedSample, caption_conversation


class Blip3oTarConverter(BaseConverter):
    def _tar_files(self, spec: DatasetSpec) -> list[Path]:
        return sorted(spec.raw_dir.glob("*.tar"))

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        tar_paths = self._tar_files(spec)
        if not tar_paths:
            raise FileNotFoundError(f"No .tar files under {spec.raw_dir}")

        count = 0
        images_dir = spec.images_dir

        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                stems = sorted({Path(n).stem for n in members if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))})

                for stem in stems:
                    img_name = next(
                        (n for n in members if Path(n).stem == stem and n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))),
                        None,
                    )
                    txt_name = next((n for n in members if Path(n).stem == stem and n.lower().endswith(".txt")), None)
                    if not img_name:
                        continue

                    img_member = members[img_name]
                    img_data = tar.extractfile(img_member)
                    if img_data is None:
                        continue
                    img_bytes = img_data.read()

                    caption = ""
                    if txt_name:
                        txt_file = tar.extractfile(members[txt_name])
                        if txt_file:
                            caption = txt_file.read().decode("utf-8", errors="replace").strip()

                    if not caption:
                        continue

                    dest = bytes_to_jpeg_path(img_bytes, images_dir / f"{stem}.jpg")

                    w, h = image_size(dest)
                    sample_id = f"{spec.key}_{stem}"
                    yield UnifiedSample(
                        id=sample_id,
                        image=str(dest.resolve()),
                        width=w,
                        height=h,
                        conversations=caption_conversation(caption),
                        meta={"dataset": spec.key, "role": spec.role, "source_id": stem},
                    )

                    count += 1
                    if limit is not None and count >= limit:
                        return
