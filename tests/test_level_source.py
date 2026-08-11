"""tests/test_level_source.py"""

import numpy as np
import pandas as pd
import pytest

from superglm.features._level_source import resolve_level_source


def test_list_preserves_order():
    assert resolve_level_source(["b", "a", "c"]) == ["b", "a", "c"]


def test_tuple_preserves_order():
    assert resolve_level_source(("z", "y")) == ["z", "y"]


def test_object_series_sorted_uniques():
    s = pd.Series(["b", "a", "b", "c"])
    assert resolve_level_source(s) == ["a", "b", "c"]


def test_numpy_array_sorted_uniques():
    assert resolve_level_source(np.array(["b", "a", "b"])) == ["a", "b"]


def test_numeric_series_sorts_numerically():
    # `key=str` puts "10" before "2", so a numeric column read through `levels=`
    # came out in an order no numeric reader would predict -- and, worse, in a
    # DIFFERENT order from `pd.factorize(sort=True)`, which is what the inferred
    # and full-frame-binding paths use on the same column. base="first" then
    # names a different level depending on which path bound the universe.
    assert resolve_level_source(pd.Series([10, 2, 2])) == [2, 10]


def test_float_array_sorts_numerically():
    assert resolve_level_source(np.array([10.5, 2.25, 10.5])) == [2.25, 10.5]


def test_mixed_type_array_sorts_by_str():
    # int beside str has no natural order (`1 < "MISSING"` is a TypeError), so
    # the str fallback is the only thing that can order it -- and ordering it at
    # all is the requirement: this is the shape the sort must not crash on.
    out = resolve_level_source(np.array([2, "MISSING", 1], dtype=object))
    assert out == [1, 2, "MISSING"]


def test_mixed_type_with_a_two_digit_number_still_resolves():
    out = resolve_level_source(np.array([10, 2, "MISSING"], dtype=object))
    assert set(out) == {10, 2, "MISSING"} and len(out) == 3


def test_categorical_series_uses_dtype_categories_and_order():
    s = pd.Series(pd.Categorical(["a", "b"], categories=["c", "b", "a"]))
    assert resolve_level_source(s) == ["c", "b", "a"]  # declared-but-unobserved 'c' kept


def test_categorical_dtype_direct():
    dt = pd.CategoricalDtype(["x", "y", "z"])
    assert resolve_level_source(dt) == ["x", "y", "z"]


def test_nan_in_source_raises():
    with pytest.raises(ValueError, match="missing value"):
        resolve_level_source(pd.Series(["a", None, "b"]))


def test_duplicate_labels_raise():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_level_source(["a", "a", "b"])


def test_singleton_raises():
    with pytest.raises(ValueError, match=">= 2"):
        resolve_level_source(["only"])


def test_fitted_encoder_rejected_with_guidance():
    class FakeEncoder:
        categories_ = [np.array(["a", "b"])]

    with pytest.raises(TypeError, match=r"categories_\[0\]"):
        resolve_level_source(FakeEncoder())


def test_unsupported_type_rejected():
    with pytest.raises(TypeError, match="levels"):
        resolve_level_source(42)


def test_context_prefixes_errors():
    with pytest.raises(ValueError, match=r"\[vehicle_group\]"):
        resolve_level_source(["a", "a"], context="vehicle_group")


def test_polars_enum_series_uses_declared_categories():
    pl = pytest.importorskip("polars")
    s = pl.Series("g", ["a", "b"], dtype=pl.Enum(["c", "b", "a"]))
    assert resolve_level_source(s) == ["c", "b", "a"]


def test_polars_plain_categorical_uses_observed_uniques():
    # A plain polars Categorical declares nothing: its dtype's `categories` is
    # the process-wide registry shared by every such column, so reading it
    # would import labels this column never held. Only an Enum declares.
    pl = pytest.importorskip("polars")
    foreign = pl.Series("other", ["zzz_foreign"], dtype=pl.Categorical)
    s = pl.Series("g", ["b", "a", "b"], dtype=pl.Categorical)
    assert "zzz_foreign" in list(foreign.dtype.categories)  # registry is shared
    assert resolve_level_source(s) == ["a", "b"]
