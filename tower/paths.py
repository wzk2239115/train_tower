"""Setup PYTHONPATH for third_party NEO and SenseNova-U1."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = PROJECT_ROOT / "third_party"
NEO_ROOT = THIRD_PARTY / "NEO"
SNU_ROOT = THIRD_PARTY / "SenseNova-U1" / "src"


def ensure_train_paths() -> None:
    for path in (NEO_ROOT, SNU_ROOT):
        if path.is_dir():
            s = str(path.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)


def neo_root() -> Path:
    return NEO_ROOT


def snu_root() -> Path:
    return SNU_ROOT
