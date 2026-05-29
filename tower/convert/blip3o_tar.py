from __future__ import annotations

import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from tower.config import DatasetSpec
from tower.convert.base import BaseConverter
from tower.convert.parallel import merge_jsonl_shards, merge_skip_counts, merge_stage_counts, shard_path
from tower.io.images import bytes_to_jpeg_path, image_size
from tower.io.writer import ConvertReport, StageWriter
from tower.schema import UnifiedSample, caption_conversation, validate_sample



def _process_blip3o_tar(
    tar_path: str,
    spec_key: str,
    role: str,
    images_dir: str,
    stages: tuple[str, ...],
    shard_paths: dict[str, str],
    dry_run: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    from tower.io.writer import StageWriter

    tar = Path(tar_path)
    images_root = Path(images_dir)
    stage_counts: dict[str, int] = {stage: 0 for stage in stages}
    skipped: dict[str, int] = defaultdict(int)

    writers: dict[str, StageWriter] = {}
    if not dry_run:
        for stage in stages:
            writer = StageWriter(stage=stage, dataset_key=spec_key, output_path=Path(shard_paths[stage]))
            writer.open()
            writers[stage] = writer

    with tarfile.open(tar, "r") as tar_fp:
        members = {m.name: m for m in tar_fp.getmembers() if m.isfile()}
        stems = sorted({Path(n).stem for n in members if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))})
        for stem in stems:
            img_name = next(
                (n for n in members if Path(n).stem == stem and n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))),
                None,
            )
            txt_name = next((n for n in members if Path(n).stem == stem and n.lower().endswith(".txt")), None)
            if not img_name:
                continue

            img_member = members[img_name]
            img_data = tar_fp.extractfile(img_member)
            if img_data is None:
                continue
            img_bytes = img_data.read()

            caption = ""
            if txt_name:
                txt_file = tar_fp.extractfile(members[txt_name])
                if txt_file:
                    caption = txt_file.read().decode("utf-8", errors="replace").strip()

            if not caption:
                skipped["missing_caption"] += 1
                continue

            dest = bytes_to_jpeg_path(img_bytes, images_root / f"{stem}.jpg")
            w, h = image_size(dest)
            sample = UnifiedSample(
                id=f"{spec_key}_{stem}",
                image=str(dest.resolve()),
                width=w,
                height=h,
                conversations=caption_conversation(caption),
                meta={"dataset": spec_key, "role": role, "source_id": stem},
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

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        tar_paths = self._tar_files(spec)
        if not tar_paths:
            raise FileNotFoundError(f"No .tar files under {spec.raw_dir}")

        count = 0
        images_dir = spec.images_dir

        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                stems = sorted({Path(n).stem for n in members if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))})

                for stem in stems:
                    img_name = next(
                        (n for n in members if Path(n).stem == stem and n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))),
                        None,
                    )
                    txt_name = next((n for n in members if Path(n).stem == stem and n.lower().endswith(".txt")), None)
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

    def convert(
        self,
        spec: DatasetSpec,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        workers: int = 1,
    ) -> ConvertReport:
        if workers <= 1 or limit is not None:
            return super().convert(spec, limit=limit, dry_run=dry_run, workers=workers)

        tar_paths = self._tar_files(spec)
        if not tar_paths:
            raise FileNotFoundError(f"No .tar files under {spec.raw_dir}")

        report = ConvertReport(dataset=spec.key, role=spec.role, dry_run=dry_run)
        from tower.config import PROCESSED_DIR

        output_files = {stage: PROCESSED_DIR / stage / f"{spec.key}.jsonl" for stage in spec.stages}
        if not dry_run:
            report.output_files = output_files

        shard_groups: dict[str, list[Path]] = {stage: [] for stage in spec.stages}
        stage_parts: list[dict[str, int]] = []
        skip_parts: list[dict[str, int]] = []

        with ProcessPoolExecutor(max_workers=min(workers, len(tar_paths))) as pool:
            futures = {}
            for shard_id, tar_path in enumerate(tar_paths):
                shard_paths = {
                    stage: str(shard_path(output_files[stage], shard_id))
                    for stage in spec.stages
                }
                if not dry_run:
                    for stage in spec.stages:
                        shard_groups[stage].append(Path(shard_paths[stage]))
                fut = pool.submit(
                    _process_blip3o_tar,
                    str(tar_path),
                    spec.key,
                    spec.role,
                    str(spec.images_dir),
                    spec.stages,
                    shard_paths,
                    dry_run,
                )
                futures[fut] = tar_path.name

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{spec.key} tars", unit="tar"):
                stage_counts, skipped = fut.result()
                stage_parts.append(stage_counts)
                skip_parts.append(skipped)

        report.stages = merge_stage_counts(stage_parts)
        report.skipped = merge_skip_counts(skip_parts)

        if not dry_run:
            for stage, out_path in output_files.items():
                merge_jsonl_shards(shard_groups[stage], out_path)

        return report
