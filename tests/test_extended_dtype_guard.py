"""Production numerics must not name a platform-dependent extended dtype.

``np.longdouble`` is float64 on Windows and macOS ARM and 80-bit elsewhere, so
code that relies on it behaves differently per platform (see AGENTS.md,
"Numerical policy"). Tests may still use extended types as inputs.
"""

import ast
from itertools import chain
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src" / "superglm"
EXTENDED = frozenset(
    {
        "longdouble",
        "float128",
        "float96",
        "clongdouble",
        "complex256",
        "complex192",
        "longfloat",
        "longcomplex",
    }
)


def _names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.ImportFrom):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return []


def _extended_references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(SOURCE)
    hits = ((node, EXTENDED.intersection(_names(node))) for node in ast.walk(tree))
    return [f"{relative}:{node.lineno} {sorted(names)}" for node, names in hits if names]


def test_source_references_no_extended_dtype() -> None:
    paths = sorted(SOURCE.rglob("*.py"))
    assert paths
    offenders = list(chain.from_iterable(map(_extended_references, paths)))
    assert not offenders, f"extended dtypes are platform-dependent: {offenders}"
