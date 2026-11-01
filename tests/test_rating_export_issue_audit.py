"""Workbook-level regressions for the joint rating-table impact disclosure."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, Polynomial, SuperGLM
from superglm.export.rating_tables import build_rating_table_payload

_INTERVAL = re.compile(r"^\[([^,]+), ([^\]\)]+)[\]\)]$")


def _interval_factor(table, values):
    """Read published interval keys and factors, without producer helpers."""
    bounds = [_INTERVAL.fullmatch(str(key)) for key in table.iloc[:, 0]]
    assert all(bound is not None for bound in bounds)
    factors = table.Relativity.to_numpy(dtype=np.float64)
    # A consumer advances to the next interval at its printed lower boundary;
    # the first and last rows also carry the out-of-range values.
    lower = np.array([float(bound.group(1)) for bound in bounds])
    index = np.searchsorted(lower, values, side="right") - 1
    return factors[np.clip(index, 0, len(factors) - 1)]


def _workbook_predictions(payload, X, *, offset=None):
    rated = np.full(len(X), payload.base_relativity, dtype=np.float64)
    for block in payload.main_effects:
        table = block.table
        if block.kind == "continuous":
            factor = _interval_factor(table, X[block.name].to_numpy())
        elif block.kind == "numeric":
            factor = float(table.Relativity.iloc[0]) ** X[block.name].to_numpy()
        elif block.kind == "categorical":
            lookup = dict(zip(table.iloc[:, 0].astype(str), table.Relativity))
            factor = np.array([lookup[str(value)] for value in X[block.name]])
        elif block.kind == "offset_per_unit":
            factor = float(table.Relativity.iloc[0]) * np.exp(offset)
        elif block.kind == "offset":
            factor = _interval_factor(table, np.exp(offset))
        else:
            raise AssertionError(f"unexpected workbook block: {block.kind}")
        rated *= factor
    assert not payload.interactions
    return rated


def _assert_joint_impact(payload, fitted, rated, y, weights):
    sheet = payload.discretization_impact
    rows = sheet.loc[sheet.exported]
    assert not rows.empty
    moved = np.abs(rated - fitted) / fitted * 100.0
    # Independent Poisson deviance, including its y=0 limit.
    logarithmic = np.zeros(len(y))
    positive = y > 0
    logarithmic[positive] = y[positive] * np.log(y[positive] / rated[positive])
    deviance = float(np.sum(2 * weights * (logarithmic - y + rated)))
    for _, row in rows.iterrows():
        assert row.max_abs_prediction_change_pct == pytest.approx(moved.max(), rel=2e-11)
        assert row.mean_abs_prediction_change_pct == pytest.approx(
            np.average(moved, weights=weights), rel=2e-11
        )
        assert row.deviance_discretized == pytest.approx(deviance, rel=2e-11, abs=1e-10)


def _cancelling_fit(*, slope=400.0, span=0.5, shift=0.0):
    rng = np.random.default_rng(313)
    x = rng.uniform(-span, span, 400) + shift
    z = x + rng.uniform(-0.01, 0.01, len(x))
    X = pd.DataFrame({"x": x, "z": z})
    y = np.exp(0.2 + slope * (x - z))
    model = SuperGLM(
        family="poisson",
        features={"x": Polynomial(degree=1), "z": Numeric()},
        selection_penalty=0,
        tol=1e-10,
    ).fit(X, y)
    assert model.result.converged
    return model, X, y


@pytest.mark.parametrize("slope", [400.0, 480.0])
def test_impact_measures_unbounded_workbook_means(slope):
    """Clipping either eta or positive-family means understates the workbook."""
    model, X, y = _cancelling_fit(slope=slope)
    payload = build_rating_table_payload(model, X, y, n_bins=2, impact_bins=(2,))
    rated = _workbook_predictions(payload, X)
    fitted = model.predict(X)
    assert np.max(np.log(rated)) > 80
    if slope == 480.0:
        assert rated.max() > 1e50, "also exercises the family mean clamp"
    _assert_joint_impact(payload, fitted, rated, y, np.ones(len(X)))
    diagnostic = model.discretization_impact(X, y, n_bins=2, features=["x"])
    np.testing.assert_allclose(diagnostic.predictions, rated, rtol=2e-11)


def test_impact_correlation_handles_large_finite_unbounded_predictions():
    model, X, y = _cancelling_fit(span=2.0)
    with np.errstate(over="raise", invalid="raise"):
        result = model.discretization_impact(X, y, n_bins=2, features=["x"])
    original = result.original_predictions
    approximated = result.predictions
    assert approximated.max() > np.sqrt(np.finfo(float).max)
    expected = np.corrcoef(original / original.max(), approximated / approximated.max())[0, 1]
    assert result.metrics["prediction_correlation"] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("shift", [-1.3, 1.3])
def test_numeric_factor_range_loss_on_export_frame_is_refused(shift):
    """A finite per-unit factor can overflow or become zero when exponentiated."""
    model, X, y = _cancelling_fit(span=0.9, shift=shift)
    assert np.all(np.isfinite(model.predict(X)))
    with pytest.raises(OverflowError, match="Numeric.*non-finite or zero"):
        build_rating_table_payload(model, X, y, n_bins=2, impact_bins=())


@pytest.mark.parametrize("impact_bins", [(), (2,)])
def test_export_refuses_nonfinite_selected_approximation_even_without_sweep(impact_bins):
    model, X, y = _cancelling_fit(span=4.0)
    with pytest.raises(OverflowError, match="discretized.*non-finite"):
        build_rating_table_payload(model, X, y, n_bins=1, impact_bins=impact_bins)


class _OverflowGrid:
    """Finite emitted factors whose joint approximation is unrepresentable."""

    def __init__(self, parent_names):
        self.parent_names = parent_names

    def score(self, x1, x2, beta):
        return np.zeros(len(x1))

    def reconstruct(self, beta, n_points=50):
        return {
            "x1": np.array([0.0, 1.0]),
            "x2": np.array([0.0, 1.0]),
            "relativity": np.full((2, 2), np.exp(400.0)),
            "interaction": True,
        }


@pytest.mark.parametrize("impact_bins", [(), (2,)])
def test_export_refuses_nonfinite_joint_grid_approximation_without_sweep(impact_bins):
    rng = np.random.default_rng(1313)
    X = pd.DataFrame(rng.uniform(0, 1, (200, 3)), columns=["x", "z", "w"])
    y = np.exp(0.2 + 0.05 * X.x.to_numpy() + 0.03 * X.z.to_numpy())
    model = SuperGLM(
        family="poisson",
        features={name: Numeric() for name in X},
        interactions=[("x", "z"), ("x", "w")],
        selection_penalty=0,
    ).fit(X, y)
    # The scoring side remains benign. The independent reconstruction contract
    # supplies two usable grid factors, whose product overflows on every row.
    for name in model._interaction_order:
        model._interaction_specs[name] = _OverflowGrid(tuple(name.split(":")))
    model._prediction_plan = None
    with pytest.raises(OverflowError, match="discretized.*non-finite"):
        build_rating_table_payload(model, X, y, n_bins=2, impact_bins=impact_bins)


def _offset_fit(*, continuous):
    rng = np.random.default_rng(314)
    n = 240
    X = pd.DataFrame({"region": rng.choice(["A", "B"], n)})
    features = {"region": Categorical(base="first")}
    offset = np.log(np.linspace(1.0, 4.0, n))
    eta = -0.4 + 0.25 * (X.region.to_numpy() == "B") + offset
    if continuous:
        X["x"] = rng.uniform(-1.0, 1.0, n)
        features["x"] = Polynomial(degree=2)
        eta += 0.8 * X.x.to_numpy() ** 2 - 0.5 * X.x.to_numpy()
    weights = rng.integers(1, 5, n).astype(float)
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(
        family="poisson",
        features=features,
        selection_penalty=0,
        weight_semantics="frequency",
        tol=1e-10,
    ).fit(X, y, sample_weight=weights, offset=offset)
    assert model.result.converged
    return model, X, y, weights, offset


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("strategy", ["exposure_quantile", "uniform", "winsorized"])
def test_binned_offset_is_included_in_joint_workbook_impact(continuous, strategy):
    """Omitting the offset delta breaks both offset-only and mixed workbooks."""
    model, X, y, weights, offset = _offset_fit(continuous=continuous)
    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=weights,
        n_bins=5,
        impact_bins=(3,),
        bin_strategy=strategy,
        offset_kind="binned",
    )
    sheet = payload.discretization_impact
    expected_features = {"Offset Multiplier", "x"} if continuous else {"Offset Multiplier"}
    assert set(sheet.feature) == expected_features
    assert set(sheet.n_bins) == {3, 5}
    offset_rows = sheet.loc[sheet.feature == "Offset Multiplier"]
    assert offset_rows.actual_bins.tolist() == [3, 5]
    rated = _workbook_predictions(payload, X, offset=offset)
    fitted = model.predict(X, offset=offset)
    assert np.max(np.abs(rated / fitted - 1)) > 0.05
    _assert_joint_impact(payload, fitted, rated, y, weights)


def test_default_continuous_offset_stays_exact_and_is_not_reported_as_binned():
    model, X, y, weights, offset = _offset_fit(continuous=False)
    payload = build_rating_table_payload(
        model, X, y, sample_weight=weights, n_bins=5, impact_bins=(5,)
    )
    rated = _workbook_predictions(payload, X, offset=offset)
    np.testing.assert_allclose(rated, model.predict(X, offset=offset), rtol=5e-14)
    assert payload.discretization_impact.empty
    assert next(block for block in payload.main_effects if "offset" in block.kind).kind == (
        "offset_per_unit"
    )
