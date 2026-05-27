from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream items from a top-level JSON array without loading the full file."""
    try:
        import ijson

        with path.open("rb") as f:
            yield from ijson.items(f, "item")
        return
    except ImportError:
        pass

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    yield from data
