from __future__ import annotations

from tower.io.writer import ConvertReport


def run_convert_job(
    key: str,
    *,
    limit: int | None,
    dry_run: bool,
    workers: int,
) -> tuple[str, ConvertReport | None, str | None]:
    from tower.config import load_dataset_specs
    from tower.registry import get_converter

    try:
        spec = load_dataset_specs()[key]
        converter = get_converter(key)
        report = converter.convert(spec, limit=limit, dry_run=dry_run, workers=workers)
        return key, report, None
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        return key, None, str(exc)
