"""Production numerics must not name a platform-dependent extended dtype.

``np.longdouble`` is float64 on Windows and macOS ARM and 80-bit elsewhere, so
code that relies on it behaves differently per platform (see AGENTS.md,
"Numerical policy"). Tests may still use extended types as inputs.
"""

import ast
from itertools import chain
from pathlib import Path

import pytest

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


# NumPy's character codes for the same types. One-letter strings are common
# elsewhere (plot colours), so codes count only in dtype= keywords and in the
# positional arguments of NumPy calls and .astype(). A code-like argument
# that is not a dtype, as in np.where(mask, "g", "r"), fails loudly.
CODES = frozenset({"g", "G", "f12", "f16", "c24", "c32"})


def _dtype_codes(node: ast.Call) -> list[str]:
    """String dtype codes passed to a NumPy call, .astype(...) or dtype=.

    Positional codes count too, as in np.asarray(x, "g") or np.finfo("g").
    """
    arguments = [keyword.value for keyword in node.keywords if keyword.arg == "dtype"]
    func = node.func
    numpy_call = isinstance(func, ast.Attribute) and (
        func.attr == "astype"
        or (isinstance(func.value, ast.Name) and func.value.id in {"np", "numpy"})
    )
    if numpy_call:
        arguments.extend(node.args)
    strings = (arg.value for arg in arguments if isinstance(arg, ast.Constant))
    return [
        code.lstrip("<>=|")
        for code in strings
        if isinstance(code, str) and code.lstrip("<>=|") in CODES
    ]


def _names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Call):
        return _dtype_codes(node)
    if isinstance(node, ast.Attribute):
        names = [node.attr]
    elif isinstance(node, ast.ImportFrom):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        names = [node.value]
    else:
        return []
    return [name for name in names if name in EXTENDED]


def _extended_references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(SOURCE)
    hits = ((node, _names(node)) for node in ast.walk(tree))
    return [f"{relative}:{node.lineno} {sorted(names)}" for node, names in hits if names]


def test_source_references_no_extended_dtype() -> None:
    paths = sorted(SOURCE.rglob("*.py"))
    assert paths
    offenders = list(chain.from_iterable(map(_extended_references, paths)))
    assert not offenders, f"extended dtypes are platform-dependent: {offenders}"


@pytest.mark.parametrize(
    ("source", "codes"),
    [
        ('np.asarray(x, "g")', ["g"]),
        ('np.zeros(n, "f16")', ["f16"]),
        ('np.finfo("g")', ["g"]),
        ('np.dtype("G")', ["G"]),
        ('x.astype("<g")', ["g"]),
        ('np.empty(3, dtype="c32")', ["c32"]),
        ('ax.plot(x, "g")', []),
        ('dict(color="g")', []),
    ],
)
def test_dtype_codes_are_read_where_numpy_expects_a_dtype(source, codes) -> None:
    assert _names(ast.parse(source, mode="eval").body) == codes
