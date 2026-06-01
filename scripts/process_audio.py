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
import time
from pathlib import Path

_LOG_T0 = time.time()


def _log(msg: str) -> None:
    elapsed = time.time() - _LOG_T0
    print(f"  [{elapsed:6.1f}s] {msg}", flush=True)


def _find_audio_files(dir_path: Path) -> dict[str, Path]:
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    result = {}
    if not dir_path.is_dir():
        return result
    for f in dir_path.rglob("*"):
        if f.suffix.lower() in audio_exts:
            result[f.stem] = f
    return result


def _peek_raw_structure(raw_dir: Path) -> None:
    _log("=" * 50)
    _log("Scanning raw data structure ...")
    for ds_dir in sorted(raw_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        subdirs = [d.name for d in ds_dir.iterdir() if d.is_dir()]
        jsonl_files = list(ds_dir.rglob("*.jsonl"))
        json_files = list(ds_dir.rglob("*.json"))
        csv_files = list(ds_dir.rglob("*.csv"))
        audio_files = [f for f in ds_dir.rglob("*") if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
        img_files = [f for f in ds_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        _log(f"  {ds_dir.name}/")
        if subdirs:
            _log(f"    subdirs ({len(subdirs)}): {', '.join(subdirs[:5])}{'...' if len(subdirs) > 5 else ''}")
        if jsonl_files:
            _log(f"    JSONL: {len(jsonl_files)} files")
        if json_files:
            _log(f"    JSON:  {len(json_files)} files — {', '.join(f.name for f in json_files[:3])}")
        if csv_files:
            _log(f"    CSV:   {len(csv_files)} files")
        if audio_files:
            _log(f"    Audio: {len(audio_files)} files — ext samples: {', '.join(set(f.suffix for f in audio_files[:100]))}")
        if img_files:
            _log(f"    Image: {len(img_files)} files")
    _log("=" * 50)


def process_wavcaps(raw_dir: Path, processed_dir: Path, data_root: Path) -> dict:
    wavcaps_dir = raw_dir / "WavCaps"
    if not wavcaps_dir.is_dir():
        _log(f"WavCaps not found at {wavcaps_dir}, skipping")
        return {}

    _log(f"Processing WavCaps from {wavcaps_dir} ...")

    audio_files = _find_audio_files(wavcaps_dir)
    _log(f"Found {len(audio_files)} audio files")

    annotation_files = list(wavcaps_dir.rglob("*.json")) + list(wavcaps_dir.rglob("*.csv"))
    _log(f"Found {len(annotation_files)} annotation files: {[f.name for f in annotation_files[:5]]}")

    if annotation_files:
        for af in annotation_files[:2]:
            _log(f"  Peeking {af.name} ({af.stat().st_size / 1024:.1f} KB) ...")
            try:
                with open(af, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    _log(f"    Top-level list, {len(data)} items")
                    if data:
                        _log(f"    First item keys: {list(data[0].keys())}")
                        _log(f"    Sample: {json.dumps(data[0], ensure_ascii=False)[:200]}")
                elif isinstance(data, dict):
                    _log(f"    Top-level dict, keys: {list(data.keys())}")
                    for k, v in data.items():
                        if isinstance(v, list):
                            _log(f"    '{k}': list of {len(v)}")
                            if v and isinstance(v[0], dict):
                                _log(f"      First item keys: {list(v[0].keys())}")
            except Exception as e:
                _log(f"    Failed to parse: {e}")

    records = []
    matched = 0
    unmatched = 0
    if annotation_files:
        for ann_file in annotation_files:
            try:
                with open(ann_file, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                _log(f"  Skipping unparseable {ann_file.name}")
                continue

            items = data if isinstance(data, list) else data.get("data", [])
            _log(f"  Processing {ann_file.name}: {len(items)} annotation items")
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
                    matched += 1
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
                    unmatched += 1
        _log(f"  Matched: {matched}  |  Unmatched: {unmatched}")
    else:
        _log("  No annotation files found, generating from filenames ...")
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
        _log("WARNING: No WavCaps records produced (no matching annotations)")
        return {}

    out_path = processed_dir / "pt" / "wavcaps.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(f"Wrote {len(records)} records ({size_mb:.1f} MB) to {out_path}")
    if records:
        _log(f"  Sample: {json.dumps(records[0], ensure_ascii=False)[:200]}")
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
        _log(f"ClothoAQA not found at {clotho_dir}, skipping")
        return {}

    _log(f"Processing ClothoAQA from {clotho_dir} ...")

    audio_files = _find_audio_files(clotho_dir)
    _log(f"Found {len(audio_files)} audio files")

    annotation_files = list(clotho_dir.rglob("*.json")) + list(clotho_dir.rglob("*.csv"))
    _log(f"Found {len(annotation_files)} annotation files: {[f.name for f in annotation_files[:5]]}")

    if annotation_files:
        for af in annotation_files[:2]:
            _log(f"  Peeking {af.name} ({af.stat().st_size / 1024:.1f} KB) ...")
            try:
                with open(af, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    _log(f"    List of {len(data)}, first keys: {list(data[0].keys()) if data else 'empty'}")
                    if data:
                        _log(f"    Sample: {json.dumps(data[0], ensure_ascii=False)[:200]}")
                elif isinstance(data, dict):
                    _log(f"    Dict keys: {list(data.keys())}")
            except Exception as e:
                _log(f"    Failed: {e}")

    records = []
    for ann_file in annotation_files:
        try:
            with open(ann_file, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _log(f"  Skipping unparseable {ann_file.name}")
            continue

        items = data if isinstance(data, list) else data.get("data", [])
        _log(f"  Processing {ann_file.name}: {len(items)} items")
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
        _log("WARNING: No ClothoAQA records produced")
        return {}

    out_path = processed_dir / "sft" / "clothoaqa.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    size_mb = out_path.stat().st_size / 1024 / 1024
    _log(f"Wrote {len(records)} records ({size_mb:.1f} MB) to {out_path}")
    if records:
        _log(f"  Sample: {json.dumps(records[0], ensure_ascii=False)[:200]}")
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

    _log(f"Manifest before: {len(manifest)} datasets")
    manifest.update(new_entries)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _log(f"Manifest after:  {len(manifest)} datasets → {manifest_path}")
    for k in new_entries:
        _log(f"  + {k}: {new_entries[k]['samples']:,} samples, role={new_entries[k]['role']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    _log(f"Raw dir: {args.raw_dir.resolve()}")
    _log(f"Processed dir: {args.processed_dir.resolve()}")

    if not args.raw_dir.is_dir():
        _log(f"ERROR: raw dir not found: {args.raw_dir}")
        sys.exit(1)

    _peek_raw_structure(args.raw_dir)

    _log("Starting audio dataset processing ...")
    all_new = {}
    all_new.update(process_wavcaps(args.raw_dir, args.processed_dir, args.data_root))
    all_new.update(process_clothoaqa(args.raw_dir, args.processed_dir, args.data_root))

    if all_new:
        update_manifest(args.processed_dir, all_new)
        _log(f"Done. Added {len(all_new)} audio datasets.")
    else:
        _log("No audio data processed. Check raw directory structure.")
    _log(f"Total elapsed: {time.time() - _LOG_T0:.1f}s")


if __name__ == "__main__":
    main()
