"""Facade for ``tower.neo`` data pipeline — sole NEO import boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tower.neo.data.data_processor import FlattenedDataCollatorForSupervisedDataset, LazySupervisedDataset
    from tower.neo.train.argument import DataArguments, TrainingArguments


def import_neo_data():
    import tower.neo.data as neo_data

    return neo_data


def import_data_constants():
    from tower.neo.data import constants

    return constants


def import_lazy_supervised_dataset():
    from tower.neo.data.data_processor import LazySupervisedDataset

    return LazySupervisedDataset


def import_flattened_data_collator():
    from tower.neo.data.data_processor import FlattenedDataCollatorForSupervisedDataset

    return FlattenedDataCollatorForSupervisedDataset


def import_train_arguments():
    from tower.neo.train.argument import DataArguments, TrainingArguments

    return DataArguments, TrainingArguments


def import_smart_resize():
    from tower.neo.data.utils import smart_resize

    return smart_resize


def all_special_token_list() -> list[str]:
    """NEO special tokens for tokenizer extension (re-export of ``constants``)."""
    return list(import_data_constants().ALL_SPECIAL_TOKEN_LIST)


def img_token_ids(tokenizer) -> tuple[int, int]:
    """``(img_context_token_id, img_start_token_id)`` from NEO constants + tokenizer."""
    c = import_data_constants()
    return (
        tokenizer.convert_tokens_to_ids(c.IMG_CONTEXT_TOKEN),
        tokenizer.convert_tokens_to_ids(c.IMG_START_TOKEN),
    )


__all__ = [
    "all_special_token_list",
    "img_token_ids",
    "import_data_constants",
    "import_flattened_data_collator",
    "import_lazy_supervised_dataset",
    "import_neo_data",
    "import_smart_resize",
    "import_train_arguments",
]
