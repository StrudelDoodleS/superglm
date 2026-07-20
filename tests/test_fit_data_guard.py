"""Focused tests for retained caller-data mutation guards."""

import numpy as np
import pandas as pd
import polars as pl

from superglm import Numeric, SuperGLM
from superglm.model.fit_data_guard import FitDataGuard


def test_capture_hashes_frame_without_making_a_shallow_copy(monkeypatch):
    X = pd.DataFrame({"x": [1.0, 2.0], "group": ["a", "b"]})
    y = np.array([3.0, 4.0])

    def fail_copy(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("FitDataGuard.capture copied the retained frame")

    monkeypatch.setattr(pd.DataFrame, "copy", fail_copy)

    guard = FitDataGuard.capture(X, y)

    assert guard.matches(X, y, None, None, fit_weights=None, fit_offset=None)


def test_explicit_fit_guard_ignores_unhashable_values_in_unused_columns():
    X = pd.DataFrame(
        {
            "used": np.linspace(-1.0, 1.0, 12),
            "unused": [[index] if index % 2 else {"index": index} for index in range(12)],
        }
    )
    y = 0.5 + 1.2 * X["used"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"used": Numeric()},
    ).fit(X, y)

    first = model.metrics(X, y)
    X.at[0, "unused"] = {"changed": True}
    second = model.metrics(X, y)

    assert second is first


def test_polars_guard_accepts_equal_native_frames_and_rejects_value_or_backend_changes():
    X = pl.DataFrame({"x": [1.0, 2.0], "group": ["a", "b"]})
    y = np.array([3.0, 4.0])

    guard = FitDataGuard.capture(X, y)

    assert guard.x_backend == "polars"
    assert guard.matches(X, y, None, None, fit_weights=None, fit_offset=None)
    assert guard.matches(X.clone(), y.copy(), None, None, fit_weights=None, fit_offset=None)
    assert not guard.matches(
        X.with_columns(pl.col("x") + 1.0),
        y,
        None,
        None,
        fit_weights=None,
        fit_offset=None,
    )
    assert not guard.matches(
        pd.DataFrame({"x": [1.0, 2.0], "group": ["a", "b"]}),
        y,
        None,
        None,
        fit_weights=None,
        fit_offset=None,
    )


def test_polars_guard_with_no_fitted_columns_preserves_row_count_only():
    X = pl.DataFrame({"unused": [1, 2, 3]})
    y = np.array([3.0, 4.0, 5.0])

    guard = FitDataGuard.capture(X, y, columns=())

    assert guard.matches_retained_values(X.with_columns(pl.col("unused") + 10), y)
    assert not guard.matches_retained_values(pl.DataFrame({"unused": [1, 2]}), y)
