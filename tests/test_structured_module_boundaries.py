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


def test_backend_selection_has_internal_owner() -> None:
    _assert_owned(
        "selection",
        (
            "StructuredGroupSelection",
            "StructuredBackendDecision",
            "select_structured_group",
            "resolve_structured_backend",
        ),
    )


def test_structured_layouts_have_internal_owner() -> None:
    _assert_owned(
        "layout",
        (
            "ScalarStructuredLayout",
            "BlockStructuredLayout",
            "get_structured_layout",
            "structured_design_matvec",
            "structured_design_rmatvec",
        ),
    )


def test_structured_moments_have_internal_owner() -> None:
    _assert_owned(
        "moments",
        (
            "ScalarStructuredSystem",
            "BlockStructuredSystem",
            "SumToZeroBlockStructuredSystem",
            "build_scalar_structured_system",
            "build_block_structured_system",
            "build_structured_system",
        ),
    )


def test_penalized_assembly_has_internal_owner() -> None:
    _assert_owned(
        "assembly",
        (
            "CachedScalarStructuredSolution",
            "CachedBlockStructuredSolution",
            "CachedSumToZeroStructuredSolution",
            "build_penalized_structured_operator",
            "build_augmented_structured_factor",
            "solve_cached_structured",
        ),
    )


def test_retained_structured_state_has_internal_owner() -> None:
    _assert_owned(
        "state",
        (
            "StructuredLevelSupport",
            "FactorSmoothLevelSupport",
            "StructuredLinearSystemState",
        ),
    )


def test_structured_module_is_implementation_free_facade() -> None:
    import ast

    tree = ast.parse(Path(structured.__file__).read_text())
    implementations = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert implementations == []
    assert structured.__all__
