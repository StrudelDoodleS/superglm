"""Architecture checks for the structured-solver compatibility facade."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import superglm.solvers.structured as structured

_SOLVER_DIR = Path(structured.__file__).resolve().parent


def _assert_owned(module_name: str, symbols: tuple[str, ...]) -> None:
    path = _SOLVER_DIR / "_structured" / f"{module_name}.py"
    assert path.is_file(), f"missing structured owner: {path}"
    owner = import_module(f"superglm.solvers._structured.{module_name}")
    for symbol in symbols:
        assert getattr(structured, symbol) is getattr(owner, symbol)


def test_compact_operators_have_internal_owner() -> None:
    _assert_owned(
        "operators",
        (
            "SymmetricBlockOperator",
            "BlockSymmetricOperator",
            "SumToZeroBlockOperator",
            "CenteredBlockOperator",
            "LowRankSymmetricOperator",
            "SumBlockOperator",
            "CompactSymmetricOperator",
            "_BlockDiagonalLowRank",
            "_operator_bdlr",
            "_trace_symmetric_bdlr",
            "materialize_compact_operator",
            "compact_operator_diagonal",
        ),
    )


def test_estimability_geometry_has_internal_owner() -> None:
    _assert_owned(
        "geometry",
        (
            "_bounded_centered_estimability",
            "_orthonormal_column_span",
            "_sum_to_zero_public_null_geometry",
            "_certified_ritz_discarded",
            "centered_operator_coefficient_estimable",
        ),
    )


def test_schur_factors_have_internal_owner() -> None:
    _assert_owned(
        "factors",
        (
            "ScalarSchurFactor",
            "BlockSchurFactor",
            "ProfiledBlockSchurFactor",
            "ProfiledScalarSchurFactor",
        ),
    )
