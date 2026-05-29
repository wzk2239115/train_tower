from __future__ import annotations

import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from tower.config import PROCESSED_DIR, DatasetSpec
from tower.convert.base import BaseConverter
from tower.convert.parallel import merge_jsonl_shards, merge_skip_counts, merge_stage_counts, shard_path
from tower.io.images import bytes_to_jpeg_path, image_size
from tower.io.writer import ConvertReport, StageWriter
from tower.schema import UnifiedSample, caption_conversation, validate_sample

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _tar_extractall(tar_fp: tarfile.TarFile, dest: Path) -> None:
    if sys.version_info >= (3, 12):
        tar_fp.extractall(path=dest, filter="data")
    else:
        tar_fp.extractall(path=dest)


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _dir_has_files(path: Path) -> bool:
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def _extract_with_system_tar(tar_p: Path, dest: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["tar", "-xf", str(tar_p), "-C", str(dest)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "tar_not_found"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return err or f"tar exit {proc.returncode}"
    return None


def _extract_one_tar(tar_path: str, images_dir: str) -> tuple[str, int, str | None]:
    """Return (tar_stem, status, error). status: 1=new extract, 0=skipped existing."""
    tar_p = Path(tar_path)
    dest = Path(images_dir) / tar_p.stem
    if dest.is_dir() and _dir_has_files(dest):
        return tar_p.stem, 0, None
    dest.mkdir(parents=True, exist_ok=True)
    err = _extract_with_system_tar(tar_p, dest)
    if err == "tar_not_found":
        try:
            with tarfile.open(tar_p, "r") as tar_fp:
                _tar_extractall(tar_fp, dest)
        except tarfile.TarError as exc:
            return tar_p.stem, 0, str(exc)
        return tar_p.stem, 1, None
    if err:
        return tar_p.stem, 0, err
    return tar_p.stem, 1, None


def _iter_image_files(root: Path) -> list[Path]:
    direct = sorted(p for p in root.iterdir() if p.is_file() and _is_image(p))
    if direct:
        return direct
    return sorted(p for p in root.rglob("*") if p.is_file() and _is_image(p))


def _count_tar_samples(tar_path: str) -> tuple[int, int]:
    skipped = 0
    count = 0
    with tarfile.open(tar_path, "r") as tar_fp:
        members = {m.name: m for m in tar_fp.getmembers() if m.isfile()}
        stems = sorted({Path(n).stem for n in members if _is_image(Path(n))})
        for stem in stems:
            txt_name = next((n for n in members if Path(n).stem == stem and Path(n).suffix.lower() == ".txt"), None)
            if not txt_name:
                skipped += 1
                continue
            count += 1
    return count, skipped


def _jsonl_one_extract_dir(
    extract_dir: str,
    spec_key: str,
    role: str,
    stages: tuple[str, ...],
    shard_paths: dict[str, str],
    dry_run: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    root = Path(extract_dir)
    stage_counts: dict[str, int] = {stage: 0 for stage in stages}
    skipped: dict[str, int] = defaultdict(int)

    writers: dict[str, StageWriter] = {}
    if not dry_run:
        for stage in stages:
            writer = StageWriter(stage=stage, dataset_key=spec_key, output_path=Path(shard_paths[stage]))
            writer.open()
            writers[stage] = writer

    tar_stem = root.name
    for img_path in _iter_image_files(root):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.is_file():
            skipped["missing_caption"] += 1
            continue

        caption = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not caption:
            skipped["missing_caption"] += 1
            continue

        sample_id = f"{spec_key}_{tar_stem}_{img_path.stem}"
        sample = UnifiedSample(
            id=sample_id,
            image=str(img_path.resolve()),
            conversations=caption_conversation(caption),
            meta={"dataset": spec_key, "role": role, "source_id": f"{tar_stem}/{img_path.stem}"},
        )
        err = validate_sample(sample)
        if err:
            skipped[err] += 1
            continue

        if dry_run:
            for stage in stages:
                stage_counts[stage] += 1
            continue

        record = sample.to_dict()
        for stage in stages:
            writers[stage].write(record)
            stage_counts[stage] += 1

    for writer in writers.values():
        writer.close()

    return stage_counts, dict(skipped)


class Blip3oTarConverter(BaseConverter):
    def _tar_files(self, spec: DatasetSpec) -> list[Path]:
        return sorted(spec.raw_dir.glob("*.tar"))

    def _extract_subdirs(self, spec: DatasetSpec) -> list[Path]:
        images_dir = spec.images_dir
        if not images_dir.is_dir():
            return []
        return sorted(p for p in images_dir.iterdir() if p.is_dir())

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        """Legacy slow path: per-sample PIL re-encode (used with --limit / --legacy-convert)."""
        tar_paths = self._tar_files(spec)
        if not tar_paths:
            raise FileNotFoundError(f"No .tar files under {spec.raw_dir}")

        count = 0
        images_dir = spec.images_dir

        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                stems = sorted({Path(n).stem for n in members if _is_image(Path(n))})

                for stem in stems:
                    img_name = next(
                        (n for n in members if Path(n).stem == stem and _is_image(Path(n))),
                        None,
                    )
                    txt_name = next((n for n in members if Path(n).stem == stem and Path(n).suffix.lower() == ".txt"), None)
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

    def _extract_fast(self, spec: DatasetSpec, *, dry_run: bool, workers: int, verbose: bool = False) -> ConvertReport:
        tar_paths = self._tar_files(spec)
        if not tar_paths:
            raise FileNotFoundError(f"No .tar files under {spec.raw_dir}")

        report = ConvertReport(dataset=spec.key, role=spec.role, dry_run=dry_run)
        spec.images_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            total = 0
            for tar_path in tar_paths:
                n, skip = _count_tar_samples(str(tar_path))
                total += n
                report.skipped["missing_caption"] += skip
            for stage in spec.stages:
                report.stages[stage] = total
            return report

        extracted = 0
        skipped_tars = 0
        pending = len(tar_paths)
        with ProcessPoolExecutor(max_workers=min(workers, len(tar_paths))) as pool:
            futures = {
                pool.submit(_extract_one_tar, str(tar_path), str(spec.images_dir)): tar_path.name
                for tar_path in tar_paths
            }
            pbar = tqdm(total=len(futures), desc=f"{spec.key} extract", unit="tar")
            for fut in as_completed(futures):
                tar_name = futures[fut]
                stem, status, err = fut.result()
                pending -= 1
                pbar.update(1)
                if err:
                    report.skipped[f"extract_error:{err}"] += 1
                    if verbose:
                        print(f"  extract FAIL {tar_name}: {err}", flush=True)
                    continue
                if status == 0:
                    skipped_tars += 1
                else:
                    extracted += 1
                if verbose and (extracted + skipped_tars) % 50 == 0:
                    print(
                        f"  extract progress: done={extracted + skipped_tars} "
                        f"new={extracted} skipped={skipped_tars} pending={pending}",
                        flush=True,
                    )
            pbar.close()

        report.skipped["extract_skipped_existing"] = skipped_tars
        report.skipped["extracted_tars"] = extracted
        return report

    def _jsonl_fast(self, spec: DatasetSpec, *, dry_run: bool, workers: int, verbose: bool = False) -> ConvertReport:
        subdirs = self._extract_subdirs(spec)
        if not subdirs:
            raise FileNotFoundError(
                f"No extracted subdirs under {spec.images_dir}. Run extract first (--extract-only)."
            )

        report = ConvertReport(dataset=spec.key, role=spec.role, dry_run=dry_run)
        output_files = {stage: PROCESSED_DIR / stage / f"{spec.key}.jsonl" for stage in spec.stages}
        if not dry_run:
            report.output_files = output_files

        shard_groups: dict[str, list[Path]] = {stage: [] for stage in spec.stages}
        stage_parts: list[dict[str, int]] = []
        skip_parts: list[dict[str, int]] = []

        with ProcessPoolExecutor(max_workers=min(workers, len(subdirs))) as pool:
            futures = {}
            for shard_id, subdir in enumerate(subdirs):
                shard_paths = {stage: str(shard_path(output_files[stage], shard_id)) for stage in spec.stages}
                if not dry_run:
                    for stage in spec.stages:
                        shard_groups[stage].append(Path(shard_paths[stage]))
                fut = pool.submit(
                    _jsonl_one_extract_dir,
                    str(subdir),
                    spec.key,
                    spec.role,
                    spec.stages,
                    shard_paths,
                    dry_run,
                )
                futures[fut] = subdir.name

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{spec.key} jsonl", unit="dir"):
                stage_counts, skipped = fut.result()
                stage_parts.append(stage_counts)
                skip_parts.append(skipped)

        report.stages = merge_stage_counts(stage_parts)
        report.skipped = merge_skip_counts(skip_parts)

        if not dry_run:
            for stage, out_path in output_files.items():
                merge_jsonl_shards(shard_groups[stage], out_path)

        return report

    def convert(
        self,
        spec: DatasetSpec,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        workers: int = 1,
        extract_only: bool = False,
        jsonl_only: bool = False,
        legacy_convert: bool = False,
        verbose: bool = False,
    ) -> ConvertReport:
        use_legacy = legacy_convert or limit is not None
        if use_legacy:
            return super().convert(spec, limit=limit, dry_run=dry_run, workers=workers)

        if extract_only and jsonl_only:
            raise ValueError("Use only one of extract_only or jsonl_only")

        if extract_only:
            return self._extract_fast(spec, dry_run=dry_run, workers=max(workers, 1), verbose=verbose)

        if jsonl_only:
            return self._jsonl_fast(spec, dry_run=dry_run, workers=max(workers, 1), verbose=verbose)

        extract_report = self._extract_fast(spec, dry_run=dry_run, workers=max(workers, 1), verbose=verbose)
        if dry_run:
            return extract_report

        jsonl_report = self._jsonl_fast(spec, dry_run=False, workers=max(workers, 1), verbose=verbose)
        jsonl_report.skipped = {**extract_report.skipped, **jsonl_report.skipped}
        return jsonl_report
