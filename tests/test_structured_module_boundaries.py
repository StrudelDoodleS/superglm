"""Architecture checks for the structured-solver package."""

from __future__ import annotations


def test_the_explicit_row_leaf_builder_is_a_test_fixture() -> None:
    """Nothing in the engine builds a leaf system from explicit rows, so the
    builder the factor tests use lives with them (``tests/_leaf_systems.py``)."""
    from superglm.solvers._structured import block_leaves

    assert not hasattr(block_leaves, "leaf_system_from_rows")


def test_retired_factor_names_stay_importable_at_their_pickled_paths() -> None:
    """The retired factor families stay importable only at their pickled paths (§3.12)."""
    from superglm.solvers._structured import factors, retired

    for name in ("ScalarSchurFactor", "ProfiledScalarSchurFactor", "BlockSchurFactor"):
        assert issubclass(getattr(factors, name), retired.RetiredStructuredState)
