from __future__ import annotations

from tower.config import URL_ONLY_DATASETS
from tower.convert.base import BaseConverter, UrlOnlyConverter
from tower.convert.blip3o_tar import Blip3oTarConverter
from tower.convert.coco_pool import CocoPoolConverter
from tower.convert.hf_parquet import HfParquetConverter
from tower.convert.llava_json import LlavaJsonConverter

_CONVERTERS: dict[str, BaseConverter] = {
    "blip3o_long": Blip3oTarConverter(),
    "blip3o_short": Blip3oTarConverter(),
    "llava150k": LlavaJsonConverter(),
    "sharegpt4v": LlavaJsonConverter(),
    "coco2017": CocoPoolConverter(),
    "refcoco": HfParquetConverter(),
    "textcaps": HfParquetConverter(),
    "docvqa": HfParquetConverter(),
    "chartqa": HfParquetConverter(),
}

_URL_STUB = UrlOnlyConverter()
for key in URL_ONLY_DATASETS:
    _CONVERTERS.setdefault(key, _URL_STUB)


def get_converter(dataset_key: str) -> BaseConverter:
    if dataset_key in _CONVERTERS:
        return _CONVERTERS[dataset_key]
    if dataset_key in URL_ONLY_DATASETS:
        return _URL_STUB
    raise KeyError(f"No converter registered for dataset '{dataset_key}'")


def convertible_datasets() -> list[str]:
    from tower.config import load_dataset_specs

    specs = load_dataset_specs()
    return [k for k in specs if k not in URL_ONLY_DATASETS or k in _CONVERTERS]
