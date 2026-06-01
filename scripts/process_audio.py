#!/usr/bin/env python3
"""Process raw audio datasets (WavCaps, ClothoAQA) into JSONL for training.

Usage:
    python scripts/process_audio.py --raw-dir data/raw --processed-dir data/processed --data-root data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_audio_files(dir_path: Path) -> dict[str, Path]:
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    result = {}
    if not dir_path.is_dir():
        return result
    for f in dir_path.rglob("*"):
        if f.suffix.lower() in audio_exts:
            result[f.stem] = f
    return result


def process_wavcaps(raw_dir: Path, processed_dir: Path, data_root: Path) -> dict:
    wavcaps_dir = raw_dir / "WavCaps"
    if not wavcaps_dir.is_dir():
        print(f"  WavCaps not found at {wavcaps_dir}, skipping")
        return {}

    print(f"  Processing WavCaps from {wavcaps_dir} ...")

    audio_files = _find_audio_files(wavcaps_dir)
    print(f"  Found {len(audio_files)} audio files")

    annotation_files = list(wavcaps_dir.rglob("*.json")) + list(wavcaps_dir.rglob("*.csv"))
    print(f"  Found {len(annotation_files)} annotation files")

    records = []
    if annotation_files:
        for ann_file in annotation_files:
            try:
                with open(ann_file, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                audio_path = item.get("audio", item.get("path", item.get("file", "")))
                caption = item.get("caption", item.get("text", item.get("description", "")))
                if not audio_path or not caption:
                    continue

                if isinstance(audio_path, list):
                    audio_path = audio_path[0] if audio_path else ""

                stem = Path(audio_path).stem if audio_path else ""
                resolved = None
                if stem in audio_files:
                    resolved = audio_files[stem]
                elif audio_path:
                    candidates = list(wavcaps_dir.rglob(Path(audio_path).name))
                    if candidates:
                        resolved = candidates[0]

                if resolved and resolved.is_file():
                    records.append({
                        "id": f"wavcaps_{stem}",
                        "audio": str(resolved),
                        "conversations": [
                            {"from": "human", "value": "<audio>\nDescribe the sound you hear."},
                            {"from": "gpt", "value": caption},
                        ],
                        "meta": {"dataset": "wavcaps", "role": "audio_caption", "source_id": stem},
                    })
    else:
        for stem, path in list(audio_files.items())[:50000]:
            name_parts = stem.replace("_", " ").replace("-", " ")
            records.append({
                "id": f"wavcaps_{stem}",
                "audio": str(path),
                "conversations": [
                    {"from": "human", "value": "<audio>\nDescribe the sound you hear."},
                    {"from": "gpt", "value": name_parts},
                ],
                "meta": {"dataset": "wavcaps", "role": "audio_caption_auto", "source_id": stem},
            })

    if not records:
        print(f"  WARNING: No WavCaps records produced (no matching annotations)")
        return {}

    out_path = processed_dir / "pt" / "wavcaps.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {out_path}")
    return {
        "wavcaps": {
            "stages": {"pt": str(out_path), "mt": str(out_path)},
            "samples": len(records),
            "role": "audio_caption",
            "skipped": {},
        }
    }


def process_clothoaqa(raw_dir: Path, processed_dir: Path, data_root: Path) -> dict:
    clotho_dir = raw_dir / "ClothoAQA"
    if not clotho_dir.is_dir():
        print(f"  ClothoAQA not found at {clotho_dir}, skipping")
        return {}

    print(f"  Processing ClothoAQA from {clotho_dir} ...")

    audio_files = _find_audio_files(clotho_dir)
    print(f"  Found {len(audio_files)} audio files")

    annotation_files = list(clotho_dir.rglob("*.json")) + list(clotho_dir.rglob("*.csv"))
    print(f"  Found {len(annotation_files)} annotation files")

    records = []
    for ann_file in annotation_files:
        try:
            with open(ann_file, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        items = data if isinstance(data, list) else data.get("data", [])
        for item in items:
            audio_path = item.get("audio", item.get("path", item.get("file", "")))
            question = item.get("question", item.get("prompt", ""))
            answer = item.get("answer", item.get("response", item.get("text", "")))
            if not audio_path or not answer:
                continue

            stem = Path(audio_path).stem if audio_path else ""
            resolved = None
            if stem in audio_files:
                resolved = audio_files[stem]
            elif audio_path:
                candidates = list(clotho_dir.rglob(Path(audio_path).name))
                if candidates:
                    resolved = candidates[0]

            if resolved and resolved.is_file():
                q = question if question else "What do you hear in this audio?"
                records.append({
                    "id": f"clothoaqa_{stem}",
                    "audio": str(resolved),
                    "conversations": [
                        {"from": "human", "value": f"<audio>\n{q}"},
                        {"from": "gpt", "value": answer},
                    ],
                    "meta": {"dataset": "clothoaqa", "role": "audio_qa", "source_id": stem},
                })

    if not records:
        print(f"  WARNING: No ClothoAQA records produced")
        return {}

    out_path = processed_dir / "sft" / "clothoaqa.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {out_path}")
    return {
        "clothoaqa": {
            "stages": {"sft": str(out_path), "mt": str(out_path)},
            "samples": len(records),
            "role": "audio_qa",
            "skipped": {},
        }
    }


def update_manifest(processed_dir: Path, new_entries: dict) -> None:
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest.update(new_entries)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated manifest: {manifest_path}")
    print(f"  Total datasets: {len(manifest)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    all_new = {}
    all_new.update(process_wavcaps(args.raw_dir, args.processed_dir, args.data_root))
    all_new.update(process_clothoaqa(args.raw_dir, args.processed_dir, args.data_root))

    if all_new:
        update_manifest(args.processed_dir, all_new)
        print(f"\nDone. Added {len(all_new)} audio datasets to manifest.")
    else:
        print("\nNo audio data processed. Check raw directory structure.")


if __name__ == "__main__":
    main()
