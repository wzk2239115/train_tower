#!/usr/bin/env python3
"""Process raw video datasets (MSR-VTT) into JSONL for training.

Usage:
    python scripts/process_video.py --raw-dir data/raw --processed-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_LOG_T0 = time.time()


def _log(msg: str) -> None:
    elapsed = time.time() - _LOG_T0
    print(f"  [{elapsed:6.1f}s] {msg}", flush=True)


def _find_video_files(dir_path: Path) -> dict[str, Path]:
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    result = {}
    if not dir_path.is_dir():
        return result
    for f in dir_path.rglob("*"):
        if f.suffix.lower() in video_exts:
            result[f.stem] = f
    return result


def process_msr_vtt(raw_dir: Path, processed_dir: Path) -> dict:
    msr_dir = raw_dir / "MSR-VTT"
    if not msr_dir.is_dir():
        _log(f"MSR-VTT not found at {msr_dir}, skipping")
        return {}

    _log(f"Processing MSR-VTT from {msr_dir} ...")

    video_dir = msr_dir / "video"
    video_files = _find_video_files(video_dir)
    _log(f"Found {len(video_files)} video files in {video_dir}")

    for ann_name in ["msrvtt_train_9k.json", "msrvtt_train_7k.json"]:
        ann_path = msr_dir / ann_name
        if ann_path.is_file():
            _log(f"Using annotations: {ann_name}")
            break
    else:
        _log("No train annotation file found, skipping")
        return {}

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _log(f"Loaded {len(data)} annotation entries")
    _log(f"  Sample keys: {list(data[0].keys())}")
    _log(f"  Sample: {json.dumps(data[0], ensure_ascii=False)[:200]}")

    records = []
    matched = 0
    skipped_no_video = 0
    for item in data:
        video_name = item.get("video", "")
        captions = item.get("caption", [])
        video_id = item.get("video_id", "")

        stem = Path(video_name).stem if video_name else video_id
        if stem not in video_files:
            if video_name and Path(video_name).stem not in video_files:
                skipped_no_video += 1
                continue
            stem = Path(video_name).stem

        caption = captions[0] if isinstance(captions, list) and captions else ""
        if not caption:
            continue

        video_path = video_files[stem]
        records.append({
            "id": f"msrvtt_{stem}",
            "video": str(video_path),
            "conversations": [
                {"from": "human", "value": "<video>\nDescribe this video in detail."},
                {"from": "gpt", "value": caption},
            ],
            "meta": {
                "dataset": "msrvtt",
                "role": "video_caption",
                "source_id": stem,
                "all_captions": captions[:5] if isinstance(captions, list) else [],
            },
        })
        matched += 1

    _log(f"Matched: {matched}  |  Skipped (no video): {skipped_no_video}")

    if not records:
        _log("WARNING: No MSR-VTT records produced")
        return {}

    out_path = processed_dir / "mt" / "msrvtt.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(f"Wrote {len(records)} records ({size_mb:.1f} MB) to {out_path}")
    _log(f"  Sample: {json.dumps(records[0], ensure_ascii=False)[:200]}")

    sft_path = processed_dir / "sft" / "msrvtt.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _log(f"Also wrote {len(records)} records to {sft_path}")

    return {
        "msrvtt": {
            "stages": {
                "mt": str(out_path),
                "sft": str(sft_path),
            },
            "samples": len(records),
            "role": "video_caption",
            "skipped": {"no_video": skipped_no_video},
        }
    }


def update_manifest(processed_dir: Path, new_entries: dict) -> None:
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    _log(f"Manifest before: {len(manifest)} datasets")
    manifest.update(new_entries)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _log(f"Manifest after:  {len(manifest)} datasets")
    for k in new_entries:
        _log(f"  + {k}: {new_entries[k]['samples']:,} samples, role={new_entries[k]['role']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    _log(f"Raw dir: {args.raw_dir.resolve()}")
    _log(f"Processed dir: {args.processed_dir.resolve()}")

    all_new = {}
    all_new.update(process_msr_vtt(args.raw_dir, args.processed_dir))

    if all_new:
        update_manifest(args.processed_dir, all_new)
        _log(f"Done. Added {len(all_new)} video datasets.")
    else:
        _log("No video data processed.")
    _log(f"Total elapsed: {time.time() - _LOG_T0:.1f}s")


if __name__ == "__main__":
    main()
