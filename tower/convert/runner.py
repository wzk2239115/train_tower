from __future__ import annotations

from tower.io.writer import ConvertReport


def run_convert_job(
    key: str,
    *,
    limit: int | None,
    dry_run: bool,
    workers: int,
    extract_only: bool = False,
    jsonl_only: bool = False,
    legacy_convert: bool = False,
    verbose: bool = False,
) -> tuple[str, ConvertReport | None, str | None]:
    from tower.config import load_dataset_specs
    from tower.registry import get_converter

    try:
        spec = load_dataset_specs()[key]
        converter = get_converter(key)
        report = converter.convert(
            spec,
            limit=limit,
            dry_run=dry_run,
            workers=workers,
            extract_only=extract_only,
            jsonl_only=jsonl_only,
            legacy_convert=legacy_convert,
            verbose=verbose,
        )
        return key, report, None
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as exc:
        return key, None, str(exc)
