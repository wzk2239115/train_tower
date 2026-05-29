from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def shard_path(output_path: Path, shard_id: int) -> Path:
    return output_path.with_name(f"{output_path.name}.part{shard_id:04d}")


def merge_jsonl_shards(shard_paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as out_fp:
        for shard in shard_paths:
            if not shard.is_file():
                continue
            with shard.open(encoding="utf-8") as in_fp:
                shutil.copyfileobj(in_fp, out_fp)
            shard.unlink()


def merge_stage_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = defaultdict(int)
    for part in counts:
        for stage, n in part.items():
            merged[stage] += n
    return dict(merged)


def merge_skip_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = defaultdict(int)
    for part in counts:
        for reason, n in part.items():
            merged[reason] += n
    return dict(merged)


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
