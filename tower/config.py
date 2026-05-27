from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTE_DIR = PROJECT_ROOT / "note"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

DATASET_YML = NOTE_DIR / "dataset.yml"
ROLE_YML = NOTE_DIR / "role.yml"

# dataset.yml key -> raw directory name under data/raw/
RAW_DIR_MAP: dict[str, str] = {
    "laion400m": "relaion400m",
    "coyo700m": "coyo-700m",
    "blip3o_long": "BLIP3o-Pretrain-Long-Caption",
    "blip3o_short": "BLIP3o-Pretrain-Short-Caption",
    "coco2017": "OpenDataLab___COCO_2017",
    "wukong": "wukong100m",
    "sharegpt4v": "ShareGPT4V",
    "llava150k": "LLaVA-Instruct-150K",
    "refcoco": "RefCOCO",
    "textcaps": "TextCaps",
    "docvqa": "DocVQA",
    "chartqa": "ChartQA",
}

URL_ONLY_DATASETS = frozenset(
    {"laion400m", "coyo700m", "wukong", "openimages", "textvqa", "gqa"}
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    url: str
    stages: tuple[str, ...]
    role: str
    raw_dir: Path

    @property
    def images_dir(self) -> Path:
        return IMAGES_DIR / self.key


def load_roles() -> dict[str, dict]:
    with ROLE_YML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("roles", {})


def load_dataset_specs() -> dict[str, DatasetSpec]:
    with DATASET_YML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    specs: dict[str, DatasetSpec] = {}
    for key, cfg in data.get("datasets", {}).items():
        raw_name = RAW_DIR_MAP.get(key, key)
        raw_dir = RAW_DIR / raw_name
        specs[key] = DatasetSpec(
            key=key,
            url=cfg.get("url", ""),
            stages=tuple(cfg.get("stages", [])),
            role=cfg.get("role", ""),
            raw_dir=raw_dir,
        )
    return specs


def get_spec(key: str) -> DatasetSpec:
    specs = load_dataset_specs()
    if key not in specs:
        raise KeyError(f"Unknown dataset '{key}'. Available: {', '.join(sorted(specs))}")
    return specs[key]
