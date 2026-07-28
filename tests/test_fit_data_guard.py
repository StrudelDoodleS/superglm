"""Focused tests for retained caller-data mutation guards."""

import copy
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from superglm import Numeric, SuperGLM
from superglm.model.fit_data_guard import (
    FitDataGuard,
    FitGeometryGuard,
    require_unchanged_fit_data,
)


def test_capture_hashes_frame_without_making_a_shallow_copy(monkeypatch):
    X = pd.DataFrame({"x": [1.0, 2.0], "group": ["a", "b"]})
    y = np.array([3.0, 4.0])

    def fail_copy(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("FitDataGuard.capture copied the retained frame")

    monkeypatch.setattr(pd.DataFrame, "copy", fail_copy)

    guard = FitDataGuard.capture(X, y)

    assert guard.matches(X, y, None, None, fit_weights=None, fit_offset=None)


def test_geometry_guard_can_require_the_training_response():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 12)})
    y = 0.5 + 0.2 * X["x"].to_numpy()
    weights = np.linspace(0.5, 1.5, len(X))
    offset = np.linspace(-0.1, 0.1, len(X))
    guard = FitGeometryGuard.capture(
        X,
        y,
        weights,
        offset,
        columns=("x",),
    )

    assert guard.matches(X.copy(), weights.copy(), offset.copy())
    assert guard.matches_training(
        X.copy(),
        y.copy(),
        weights.copy(),
        offset.copy(),
    )
    changed = y.copy()
    changed[0] += 1.0
    assert not guard.matches_training(X, changed, weights, offset)


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


@pytest.mark.parametrize(
    "round_trip",
    [
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda value: pickle.loads(pickle.dumps(value)), id="pickle"),
    ],
)
def test_polars_retained_guard_survives_model_round_trip(round_trip) -> None:
    X = pl.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = 0.5 + 0.3 * X["x"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(X, y)

    restored = round_trip(model)

    assert isinstance(restored._fit_X_ref, pl.DataFrame)
    assert restored._fit_data_guard.matches_retained_values(
        restored._fit_X_ref,
        restored._fit_y_ref,
    )
    first = restored.metrics(restored._fit_X_ref, restored._fit_y_ref)
    second = restored.metrics(restored._fit_X_ref, restored._fit_y_ref)
    assert second is first


def test_compact_polars_fit_releases_native_reference_and_guard() -> None:
    X = pl.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = 0.5 + 0.3 * X["x"].to_numpy()

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
        retain_fit_state=False,
    ).fit(X, y)

    assert model._fit_state.retained is False
    assert model._fit_X_ref is None
    assert model._fit_data_guard is None


def test_failed_polars_guard_verification_does_not_change_published_revision() -> None:
    X = pl.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = 0.5 + 0.3 * X["x"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(X, y)
    state_before = model._fit_state
    result_before = model.result
    revision_before = model._fit_revision

    with pytest.raises(RuntimeError, match="retained fit data were mutated"):
        require_unchanged_fit_data(
            model,
            X.with_columns(pl.col("x") + 1.0),
            y,
        )

    assert model._fit_state is state_before
    assert model.result is result_before
    assert model._fit_revision == revision_before
