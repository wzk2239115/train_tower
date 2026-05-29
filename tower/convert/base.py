from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator

from tqdm import tqdm

from tower.config import DatasetSpec, PROCESSED_DIR
from tower.io.writer import ConvertReport, StageWriter
from tower.schema import UnifiedSample, validate_sample


class BaseConverter(ABC):
    @abstractmethod
    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        ...

    def convert(
        self,
        spec: DatasetSpec,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        workers: int = 1,
    ) -> ConvertReport:
        report = ConvertReport(dataset=spec.key, role=spec.role, dry_run=dry_run)
        writers: dict[str, StageWriter] = {}

        if not dry_run:
            for stage in spec.stages:
                out = PROCESSED_DIR / stage / f"{spec.key}.jsonl"
                writer = StageWriter(stage=stage, dataset_key=spec.key, output_path=out)
                writer.open()
                writers[stage] = writer
                report.output_files[stage] = out

        written = 0
        iterator = self.iter_samples(spec, limit=limit)
        if limit is None:
            iterator = tqdm(iterator, desc=spec.key, unit="sample")

        for sample in iterator:
            err = validate_sample(sample)
            if err:
                report.skipped[err] += 1
                continue

            if dry_run:
                written += 1
                for stage in spec.stages:
                    report.stages[stage] = report.stages.get(stage, 0) + 1
                continue

            record = sample.to_dict()
            for stage in spec.stages:
                writers[stage].write(record)
                report.stages[stage] = report.stages.get(stage, 0) + 1
            written += 1

        for writer in writers.values():
            writer.close()

        return report


class UrlOnlyConverter(BaseConverter):
    """Placeholder for datasets that only have URLs in raw parquet."""

    def iter_samples(self, spec: DatasetSpec, *, limit: int | None = None) -> Iterator[UnifiedSample]:
        if False:  # pragma: no cover
            yield UnifiedSample(id="", image="", conversations=[])
        raise RuntimeError(
            f"Dataset '{spec.key}' is URL-only (no local images). "
            "Download images first or skip in phase 1."
        )

    def convert(
        self,
        spec: DatasetSpec,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        workers: int = 1,
    ) -> ConvertReport:
        report = ConvertReport(dataset=spec.key, role=spec.role, dry_run=dry_run)
        report.skipped["url_only_dataset"] = 1
        return report
