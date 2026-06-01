#!/usr/bin/env python3
"""Data inventory scanner — run on compute platform, export summary to dev machine.

Usage:
    python scripts/scan_data.py /path/to/DATA_ROOT [-o data_inventory.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _safe_read_jsonl(path: Path, max_lines: int = 3) -> list[dict]:
    items = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    items.append({"_raw": line[:500]})
    except Exception as e:
        items.append({"_error": str(e)})
    return items


def _scan_annotation_file(path: Path) -> dict:
    info = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "size_human": _human_size(path.stat().st_size),
        "exists": True,
    }
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        info["samples"] = _safe_read_jsonl(path, max_lines=3)
        total_lines = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        total_lines += 1
        except Exception:
            pass
        info["total_lines"] = total_lines

        if info["samples"] and "_error" not in info["samples"][0]:
            sample = info["samples"][0]
            info["fields"] = sorted(sample.keys())
            info["has_conversations"] = "conversations" in sample
            info["has_image"] = any(k in sample for k in ("image", "images", "image_path"))
            info["has_audio"] = any(k in sample for k in ("audio", "audios", "audio_path"))
            info["has_video"] = any(k in sample for k in ("video", "videos", "video_path"))
    elif suffix in (".csv", ".tsv"):
        info["type"] = "csv/tsv"
    elif suffix in (".parquet",):
        info["type"] = "parquet"
    return info


def _scan_directory(data_root: Path) -> dict:
    result = {
        "scanned_at": datetime.now().isoformat(),
        "data_root": str(data_root),
        "directories": {},
        "summary": {
            "total_dirs": 0,
            "total_annotation_files": 0,
            "total_annotation_lines": 0,
            "total_size_bytes": 0,
            "total_size_human": "0 B",
            "modality_breakdown": defaultdict(int),
        },
    }

    if not data_root.exists():
        result["error"] = f"DATA_ROOT does not exist: {data_root}"
        result["summary"]["modality_breakdown"] = dict(result["summary"]["modality_breakdown"])
        return result

    annotation_exts = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}

    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue

        dir_info = {
            "path": str(entry.relative_to(data_root)),
            "annotation_files": [],
            "subdirs": [],
            "total_size_bytes": 0,
        }

        for root, dirs, files in os.walk(entry):
            rel = Path(root)
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in annotation_exts:
                    fpath = rel / fname
                    af = _scan_annotation_file(fpath)
                    af["rel_path"] = str(fpath.relative_to(entry))
                    dir_info["annotation_files"].append(af)
                    result["summary"]["total_annotation_lines"] += af.get("total_lines", 0)

            sub_depth = len(Path(root).relative_to(entry).parts)
            if sub_depth < 2:
                for d in dirs:
                    dir_info["subdirs"].append(
                        str((Path(root) / d).relative_to(entry))
                    )

        dir_info["total_size_bytes"] = sum(
            af.get("size_bytes", 0) for af in dir_info["annotation_files"]
        )
        dir_info["total_size_human"] = _human_size(dir_info["total_size_bytes"])

        modalities = []
        if any(af.get("has_image") for af in dir_info["annotation_files"]):
            modalities.append("image")
        if any(af.get("has_audio") for af in dir_info["annotation_files"]):
            modalities.append("audio")
        if any(af.get("has_video") for af in dir_info["annotation_files"]):
            modalities.append("video")
        if dir_info["annotation_files"] and not modalities:
            modalities.append("text")
        dir_info["modalities"] = modalities

        for m in modalities:
            result["summary"]["modality_breakdown"][m] += 1

        result["summary"]["total_dirs"] += 1
        result["summary"]["total_size_bytes"] += dir_info["total_size_bytes"]
        result["summary"]["total_annotation_files"] += len(dir_info["annotation_files"])

        result["directories"][entry.name] = dir_info

    result["summary"]["total_size_human"] = _human_size(result["summary"]["total_size_bytes"])
    result["summary"]["modality_breakdown"] = dict(result["summary"]["modality_breakdown"])

    return result


def _cross_reference(result: dict) -> dict:
    if not _HAS_YAML:
        return {"note": "PyYAML not installed, skipping cross-reference with recipe/pools"}

    recipe_root = Path(__file__).resolve().parents[1] / "recipe" / "pools"
    if not recipe_root.is_dir():
        return {"note": f"recipe/pools not found at {recipe_root}"}

    expected_datasets = {}
    for pool_file in sorted(recipe_root.glob("*.yaml")):
        try:
            data = yaml.safe_load(pool_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        pool_id = data.get("pool_id", pool_file.stem)
        datasets = data.get("datasets", [])
        if not datasets and "sub_pools" in data:
            for sp_name, sp in data["sub_pools"].items():
                datasets.extend(sp.get("datasets", []))
        for ds in datasets:
            source = ds.get("source", "")
            source = source.replace("$DATA_ROOT/", "").replace("$DATA_ROOT", "")
            if source:
                expected_datasets[source] = {
                    "pool": pool_id,
                    "name": ds.get("name", ""),
                    "key": ds.get("key", ""),
                    "expected_format": ds.get("format", ""),
                    "expected_annotation": ds.get("annotation_file", ""),
                }

    available = set(result.get("directories", {}).keys())
    matched = {}
    missing = {}
    for source, info in expected_datasets.items():
        if source in available:
            dir_info = result["directories"][source]
            annotation_names = [
                af["rel_path"] for af in dir_info.get("annotation_files", [])
            ]
            expected_ann = info["expected_annotation"]
            has_expected_annotation = (
                not expected_ann
                or any(expected_ann in a for a in annotation_names)
                or any(a.endswith(".jsonl") for a in annotation_names)
            )
            matched[source] = {
                **info,
                "found": True,
                "annotation_match": has_expected_annotation,
                "actual_lines": sum(
                    af.get("total_lines", 0)
                    for af in dir_info.get("annotation_files", [])
                ),
            }
        else:
            missing[source] = {**info, "found": False}

    unexpected = available - set(expected_datasets.keys())

    return {
        "recipe_pool_match": {
            "total_expected": len(expected_datasets),
            "matched": len(matched),
            "missing": len(missing),
            "unexpected_dirs": len(unexpected),
        },
        "matched_datasets": matched,
        "missing_datasets": missing,
        "unexpected_dirs": {d: {"note": "not in any recipe/pool"} for d in sorted(unexpected)},
    }


def main():
    parser = argparse.ArgumentParser(description="Scan data directory and export inventory")
    parser.add_argument("data_root", type=Path, help="Path to DATA_ROOT")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data_inventory.json"),
        help="Output file (default: data_inventory.json)",
    )
    args = parser.parse_args()

    print(f"Scanning {args.data_root} ...")
    result = _scan_directory(args.data_root)

    print(f"Cross-referencing with recipe/pools ...")
    result["cross_reference"] = _cross_reference(result)

    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"  Dirs: {s['total_dirs']}  |  Annotation files: {s['total_annotation_files']}")
    print(f"  Total annotation lines: {s['total_annotation_lines']:,}")
    print(f"  Total size: {s['total_size_human']}")
    print(f"  Modalities: {s['modality_breakdown']}")
    xr = result.get("cross_reference", {})
    if "recipe_pool_match" in xr:
        m = xr["recipe_pool_match"]
        print(f"  Recipe match: {m['matched']}/{m['total_expected']} datasets found")
        print(f"  Missing: {m['missing']}  |  Unexpected: {m['unexpected_dirs']}")
    print(f"{'='*60}")

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nExported to {args.output}")
    print(f"Copy to dev machine:  scp {args.output} dev:/path/to/train_tower/")


if __name__ == "__main__":
    main()
