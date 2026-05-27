from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path

from PIL import Image


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def save_image_bytes(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    dest.write_bytes(data)
    return dest


def copy_or_link_image(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    shutil.copy2(src, dest)
    return dest


def bytes_to_jpeg_path(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        rgb.save(dest, format="JPEG", quality=95)
    return dest


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def resolve_coco_image(relative_path: str, coco_root: Path) -> Path | None:
    """Resolve LLaVA-style paths like coco/train2017/000000033471.jpg."""
    name = Path(relative_path).name
    candidates = [
        coco_root / relative_path,
        coco_root / "train2017" / name,
        coco_root / "val2017" / name,
        coco_root / "sample" / "image" / name,
        coco_root / "raw" / "Images" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None
