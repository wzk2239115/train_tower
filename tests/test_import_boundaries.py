"""Enforce import boundaries: tower.neo and tower.models.neo_unify only via backends."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOWER_ROOT = PROJECT_ROOT / "tower"

# Modules allowed to import migrated upstream code (plus TYPE_CHECKING-only stubs).
_ALLOWED_IMPORT_PREFIXES = (
    "tower.unify.backends.neo",
    "tower.unify.backends.sensenova",
    "tower.unify.backends",
)

# Internal packages — may import themselves.
_INTERNAL_PACKAGES = (
    "tower.neo",
    "tower.models.neo_unify",
)

_FORBIDDEN_PREFIXES = ("tower.neo", "tower.models.neo_unify")


def _module_name(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _is_allowed_file(path: Path) -> bool:
    mod = _module_name(path)
    if any(mod == p or mod.startswith(p + ".") for p in _ALLOWED_IMPORT_PREFIXES):
        return True
    if any(mod == p or mod.startswith(p + ".") for p in _INTERNAL_PACKAGES):
        return True
    return False


def _forbidden_imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _is_forbidden(node.module):
                hits.append(node.module)
    return hits


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".") for prefix in _FORBIDDEN_PREFIXES
    )


def test_no_direct_migrated_imports_outside_backends():
    violations: list[str] = []
    for path in TOWER_ROOT.rglob("*.py"):
        if _is_allowed_file(path):
            continue
        for imp in _forbidden_imports_in_file(path):
            violations.append(f"{path.relative_to(PROJECT_ROOT)}: {imp}")
    assert not violations, "Direct migrated imports outside backends:\n" + "\n".join(violations)
