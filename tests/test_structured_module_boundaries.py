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
            "centered_operator_coefficient_estimable",
        ),
    )


def test_structured_factors_have_internal_owners() -> None:
    _assert_owned("nested", ("NestedSchurFactor", "ProfiledNestedSchurFactor"))
    _assert_owned("block_leaves", ("FactorSmoothLeafFactor", "ProfiledFactorSmoothLeafFactor"))
    _assert_owned("balance_tree", ("SumToZeroTreeFactor", "ProfiledSumToZeroTreeFactor"))


def test_retired_factor_names_are_not_on_the_facade() -> None:
    """The retired factor families stay importable only at their pickled paths (§3.12)."""
    from superglm.solvers._structured import factors, retired

    for name in ("ScalarSchurFactor", "ProfiledScalarSchurFactor", "BlockSchurFactor"):
        assert not hasattr(structured, name)
        assert issubclass(getattr(factors, name), retired.RetiredStructuredState)


def test_the_explicit_row_leaf_builder_is_a_test_fixture() -> None:
    """Nothing in the engine builds a leaf system from explicit rows, so the
    builder the factor tests use lives with them (``tests/_leaf_systems.py``)."""
    from superglm.solvers._structured import block_leaves

    assert not hasattr(block_leaves, "leaf_system_from_rows")


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
            "FactorSmoothLeafLayout",
            "get_structured_layout",
            "structured_design_matvec",
            "structured_design_rmatvec",
        ),
    )


def test_structured_moments_have_internal_owner() -> None:
    _assert_owned(
        "moments",
        (
            "NestedStructuredSystem",
            "FactorSmoothMomentSystem",
            "SumToZeroMomentSystem",
            "build_block_structured_system",
            "build_structured_system",
        ),
    )


def test_penalized_assembly_has_internal_owner() -> None:
    _assert_owned(
        "assembly",
        (
            "CachedNestedStructuredSolution",
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
