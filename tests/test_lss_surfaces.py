"""Contract tests for LSS surfaces, spread, portfolio and the ratio-of-sums helper."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.checks._aggregate import grouped_ratio
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.posterior import posterior_bounds, posterior_draws
from superglm.distributional.surfaces import (
    DensityFan,
    Histogram,
    Portfolio,
    RiskCurves,
    Spread,
    _clipped_density,
    _json_scalar,
    _Segmentation,
    _SegmentTotals,
    _WeightedTotal,
    density_fan,
    parameter_spread,
    portfolio,
    risk_curves,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _gaussian_sample(n: int = 1000, seed: int = 20260903) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    location = 0.6 * np.sin(2.4 * x) + np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    return pd.DataFrame({"x": x, "g": g}), location + scale * rng.standard_normal(n)


def _gamma_sample(n: int = 1000, seed: int = 20260904) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(["a", "b"], n)
    mean = np.exp(1.0 + 0.8 * np.sin(3.0 * x) + np.where(g == "a", 0.2, -0.2))
    cv = np.exp(-0.5 + 0.3 * x)
    shape = 1.0 / (cv * cv)
    return pd.DataFrame({"x": x, "g": g}), rng.gamma(shape, mean / shape)


@pytest.fixture(scope="module")
def gaussian_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    X, y = _gaussian_sample()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8), "g": Categorical()}),
            Predictor("scale", {"x": Spline("cr", k=6)}),
        ],
    ).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def gamma_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    X, y = _gamma_sample()
    model = SuperLSS(
        family=GammaLS(),
        predictors=[
            Predictor("mean", {"x": Spline("cr", k=8), "g": Categorical()}),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ],
    ).fit_reml(X, y)
    return model._require_fitted(), X, y


def _assert_json_leaves(payload: object) -> None:
    """Every leaf of a payload's JSON is a float, int, bool, str or None."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert isinstance(key, str)
            _assert_json_leaves(value)
        return
    if isinstance(payload, list):
        for value in payload:
            _assert_json_leaves(value)
        return
    assert payload is None or type(payload) in (float, int, bool, str), repr(payload)
    assert not (isinstance(payload, float) and math.isnan(payload))


def test_json_scalars_are_plain_values_and_nothing_unrepresentable() -> None:
    assert _json_scalar(None) is None
    assert _json_scalar(np.nan) is None
    assert _json_scalar(np.inf) is None
    assert _json_scalar(np.True_) is True
    assert _json_scalar(np.int64(3)) == 3
    assert _json_scalar(np.float64(2.5)) == 2.5
    assert _json_scalar(np.str_("a")) == "a"
    for value in (None, np.nan, np.True_, np.int64(3), np.float64(2.5), np.str_("a")):
        _assert_json_leaves(_json_scalar(value))


class _MeanlessFamily:
    """A distribution-function family that names no default prediction."""

    def __init__(self, family) -> None:
        self._family = family

    @property
    def parameters(self):
        return self._family.parameters

    def cdf(self, y, theta):
        return self._family.cdf(y, theta)

    def quantile(self, p, theta):
        return self._family.quantile(p, theta)


class _CdflessFamily:
    """A family with a default prediction but no distribution functions."""

    def __init__(self, family) -> None:
        self._family = family

    @property
    def parameters(self):
        return self._family.parameters

    @property
    def default_prediction_name(self) -> str:
        return self._family.default_prediction_name

    def default_prediction(self, theta):
        return self._family.default_prediction(theta)


# --------------------------------------------------------------------------- #
# grouped_ratio aggregation contract
# --------------------------------------------------------------------------- #


def test_grouped_ratio_is_a_ratio_of_sums_not_a_mean_of_ratios() -> None:
    # y = [2, 4] with weights [1, 3] is 14 / 4 = 3.5, not (2 + 4) / 2 = 3.0.
    weights = np.array([1.0, 3.0])
    response = np.array([2.0, 4.0])
    total, exposure, ratio = grouped_ratio(weights * response, weights, np.zeros(2, dtype=int))

    assert total.shape == exposure.shape == ratio.shape == (1,)
    assert total[0] == pytest.approx(14.0)
    assert exposure[0] == pytest.approx(4.0)
    assert ratio[0] == pytest.approx(3.5)
    assert ratio[0] != pytest.approx(float(np.mean(response)))


def test_grouped_ratio_separates_groups_and_pads_to_the_declared_count() -> None:
    weights = np.array([1.0, 3.0, 2.0, 2.0])
    response = np.array([2.0, 4.0, 10.0, 20.0])
    groups = np.array([0, 0, 1, 1])

    total, exposure, ratio = grouped_ratio(weights * response, weights, groups)
    np.testing.assert_allclose(total, [14.0, 60.0])
    np.testing.assert_allclose(exposure, [4.0, 4.0])
    np.testing.assert_allclose(ratio, [3.5, 15.0])

    padded = grouped_ratio(weights * response, weights, groups, n_groups=4)
    assert padded[0].shape == (4,)
    np.testing.assert_allclose(padded[0][:2], total)
    assert np.array_equal(padded[1][2:], np.zeros(2))
    assert np.all(np.isnan(padded[2][2:]))


def test_grouped_ratio_reports_an_empty_denominator_as_not_a_number() -> None:
    total, exposure, ratio = grouped_ratio(
        np.array([0.0, 5.0]), np.array([0.0, 2.0]), np.array([0, 1])
    )
    assert exposure[0] == 0.0
    assert math.isnan(float(ratio[0]))
    assert ratio[1] == pytest.approx(2.5)

    empty = grouped_ratio(np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))
    assert all(part.shape == (0,) for part in empty)


def test_grouped_ratio_refuses_groups_it_cannot_index() -> None:
    numerator = np.array([1.0, 2.0])
    denominator = np.array([1.0, 1.0])
    groups = np.array([0, 1])

    with pytest.raises(ValueError, match="same number of rows"):
        grouped_ratio(numerator, np.array([1.0]), groups)
    with pytest.raises(ValueError, match="same number of rows"):
        grouped_ratio(numerator, denominator, np.array([0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        grouped_ratio(numerator.reshape(2, 1), denominator, groups)
    with pytest.raises(ValueError, match="non-negative integer"):
        grouped_ratio(numerator, denominator, np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="non-negative integer"):
        grouped_ratio(numerator, denominator, np.array([0, -1]))
    with pytest.raises(ValueError, match="one-dimensional array of group codes"):
        grouped_ratio(numerator, denominator, groups.reshape(2, 1))
    with pytest.raises(ValueError, match="n_groups"):
        grouped_ratio(numerator, denominator, groups, n_groups=1)
    with pytest.raises(ValueError, match="finite"):
        grouped_ratio(np.array([1.0, np.inf]), denominator, groups)


# --------------------------------------------------------------------------- #
# Risk curves
# --------------------------------------------------------------------------- #


def test_risk_curves_sweep_the_training_range_with_ordered_quantiles(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    curves = risk_curves(fitted, X, {"g": "a"}, "x", n_points=40, n_draws=200, seed=3)

    assert isinstance(curves, RiskCurves)
    assert curves.kind == "risk_curves"
    assert curves.schema_version >= 1
    assert curves.covariate == "x"
    assert curves.levels is None
    assert curves.quantiles == (0.5, 0.9, 0.99)
    assert curves.values.shape == curves.lower.shape == curves.upper.shape == (3, 40)
    assert curves.x.shape == (40,)
    assert curves.x[0] == pytest.approx(float(X["x"].min()))
    assert curves.x[-1] == pytest.approx(float(X["x"].max()))
    assert np.all(np.diff(curves.x) > 0.0)
    assert curves.reference == {"g": "a"}
    assert curves.level == 0.9
    assert curves.n_draws == 200
    assert curves.seed == 3
    assert curves.covariance == "fixed"

    assert np.all(curves.values[0] < curves.values[1])
    assert np.all(curves.values[1] < curves.values[2])
    assert np.all(curves.lower <= curves.values)
    assert np.all(curves.values <= curves.upper)
    # One draw set for every quantile keeps the bands ordered as the curves are.
    assert np.all(np.diff(curves.lower, axis=0) > 0.0)
    assert np.all(np.diff(curves.upper, axis=0) > 0.0)


def test_risk_curves_share_one_posterior_draw_set_across_quantiles(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    curves = risk_curves(fitted, X, {"g": "b"}, "x", n_points=12, n_draws=150, seed=11, level=0.8)

    sweep = pd.DataFrame({"x": curves.x, "g": np.array(["b"] * 12, dtype=object)})
    shared = posterior_draws(fitted, 150, seed=11)
    for index, level in enumerate(curves.quantiles):
        bounds = posterior_bounds(fitted, sweep, ("quantile", level), level=0.8, draws=shared)
        assert np.array_equal(curves.values[index], bounds["estimate"].to_numpy())
        assert np.array_equal(curves.lower[index], bounds["lower"].to_numpy())
        assert np.array_equal(curves.upper[index], bounds["upper"].to_numpy())


def test_risk_curves_sweep_the_levels_of_a_non_numeric_covariate(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    curves = risk_curves(
        fitted, X, {"x": 0.4}, "g", quantiles=(0.5, 0.95), n_points=99, n_draws=120, seed=5
    )

    assert curves.levels == ("a", "b", "c")
    assert np.array_equal(curves.x, np.arange(3.0))
    assert curves.values.shape == (2, 3)
    assert curves.reference == {"x": 0.4}
    # The fitted level ordering is the simulated one: a above c above b.
    assert curves.values[0, 0] > curves.values[0, 2] > curves.values[0, 1]

    # The sweep really holds x at the reference value, not at zero or a median.
    held = pd.DataFrame({"x": np.full(3, 0.4), "g": np.array(["a", "b", "c"], dtype=object)})
    theta = fitted.predict_parameters(held)
    np.testing.assert_allclose(curves.values[1], fitted.family.quantile(np.full(3, 0.95), theta))


def test_risk_curves_default_the_columns_the_reference_leaves_out(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    mode = str(X["g"].mode().iat[0])
    defaulted = risk_curves(fitted, X, {}, "x", n_points=6, n_draws=60, seed=1)
    assert defaulted.reference == {"g": mode}

    named = risk_curves(fitted, X, {"g": mode}, "x", n_points=6, n_draws=60, seed=1)
    assert np.array_equal(defaulted.values, named.values)

    # A reference row of the training frame carries the swept covariate too; the
    # sweep overrides it rather than refusing the row.
    row = X.iloc[0]
    from_row = risk_curves(fitted, X, row, "x", n_points=6, n_draws=60, seed=1)
    assert "x" not in from_row.reference
    assert from_row.reference == {"g": row["g"]}

    numeric = risk_curves(fitted, X, {"g": "a"}, "g", n_points=3, n_draws=60, seed=1)
    assert numeric.reference["x"] == pytest.approx(float(np.median(X["x"].to_numpy())))


def test_risk_curves_refuse_a_reference_and_a_covariate_they_cannot_place(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    with pytest.raises(ValueError, match="not a column of the training frame"):
        risk_curves(fitted, X, {"nonesuch": 1.0}, "x", n_points=4, n_draws=60)
    with pytest.raises(ValueError, match="does not use"):
        risk_curves(fitted, X.assign(unused=1.0), {}, "unused", n_points=4, n_draws=60)
    with pytest.raises(ValueError, match="at least one quantile"):
        risk_curves(fitted, X, {}, "x", quantiles=(), n_points=4, n_draws=60)
    with pytest.raises(ValueError, match="distinct"):
        risk_curves(fitted, X, {}, "x", quantiles=(0.5, 0.5), n_points=4, n_draws=60)
    with pytest.raises(ValueError, match="at least two"):
        risk_curves(fitted, X, {}, "x", n_points=1, n_draws=60)
    with pytest.raises(ValueError, match="strictly inside"):
        risk_curves(fitted, X, {}, "x", quantiles=(1.5,), n_points=4, n_draws=60)
    with pytest.raises(TypeError, match="mapping or a pandas Series"):
        risk_curves(fitted, X, 0.5, "x", n_points=4, n_draws=60)


def test_risk_curves_to_json_emits_plain_values(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    payload = risk_curves(fitted, X, {"g": "a"}, "x", n_points=5, n_draws=60, seed=2).to_json()

    _assert_json_leaves(payload)
    assert payload["kind"] == "risk_curves"
    assert payload["levels"] is None
    assert len(payload["x"]) == 5
    assert len(payload["values"]) == len(payload["lower"]) == len(payload["upper"]) == 3
    assert all(len(row) == 5 for row in payload["values"])
    assert payload["reference"] == {"g": "a"}

    levels = risk_curves(fitted, X, {"x": 0.0}, "g", n_points=5, n_draws=60, seed=2).to_json()
    _assert_json_leaves(levels)
    assert levels["levels"] == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Density fan
# --------------------------------------------------------------------------- #


def test_density_fan_integrates_to_one_over_the_swept_range(gaussian_case) -> None:
    fitted, X, _ = gaussian_case
    fan = density_fan(fitted, X, {"g": "a"}, "x", n_points=20, n_y=200)

    assert isinstance(fan, DensityFan)
    assert fan.kind == "density_fan"
    assert fan.covariate == "x"
    assert fan.levels is None
    assert fan.x.shape == (20,)
    assert fan.y_grid.shape == (200,)
    assert np.all(np.diff(fan.y_grid) > 0.0)
    assert fan.density.shape == (20, 200)
    assert np.all(fan.density >= 0.0)
    assert fan.quantile_levels == (0.5, 0.9, 0.99)
    assert fan.quantiles is not None and fan.quantiles.shape == (3, 20)
    assert np.all(np.diff(fan.quantiles, axis=0) > 0.0)
    assert fan.to_json()["quantiles"] is not None
    assert (
        density_fan(fitted, X, {"g": "a"}, "x", n_points=4, n_y=24, quantiles=None).quantiles
        is None
    )

    theta = fitted.predict_parameters(pd.DataFrame({"x": fan.x, "g": ["a"] * 20}))
    lower = fitted.family.quantile(np.full(20, 0.001), theta)
    upper = fitted.family.quantile(np.full(20, 0.999), theta)
    assert fan.y_grid[0] == pytest.approx(float(lower.min()))
    assert fan.y_grid[-1] == pytest.approx(float(upper.max()))

    mass = integrate.trapezoid(fan.density, fan.y_grid, axis=1)
    assert np.all(np.abs(mass - 1.0) < 0.02)

    payload = fan.to_json()
    _assert_json_leaves(payload)
    assert len(payload["density"]) == 20
    assert all(len(row) == 200 for row in payload["density"])


def test_density_fan_sweeps_levels_and_refuses_a_short_grid(gamma_case) -> None:
    fitted, X, _ = gamma_case
    fan = density_fan(fitted, X, {"x": 0.5}, "g", n_points=7, n_y=150)

    assert fan.levels == ("a", "b")
    assert fan.density.shape == (2, 150)
    assert np.all(fan.y_grid > 0.0)
    mass = integrate.trapezoid(fan.density, fan.y_grid, axis=1)
    assert np.all(np.abs(mass - 1.0) < 0.02)

    with pytest.raises(ValueError, match="at least three"):
        density_fan(fitted, X, {}, "x", n_points=5, n_y=2)

    cdfless = SimpleNamespace(family=_CdflessFamily(fitted.family))
    with pytest.raises(NotImplementedError, match="cdf and a quantile"):
        density_fan(cdfless, X, {}, "x", n_points=5, n_y=50)


def test_clipped_density_clips_round_off_and_refuses_a_material_negative() -> None:
    values = np.array([[1.0, -1.0e-15], [0.5, 2.0]])
    clipped = _clipped_density(values)
    assert np.all(clipped >= 0.0)
    assert clipped[0, 1] == 0.0
    assert clipped[1, 1] == 2.0

    with pytest.raises(ValueError, match="negative density"):
        _clipped_density(np.array([[1.0, -0.25]]))


# --------------------------------------------------------------------------- #
# Parameter spread
# --------------------------------------------------------------------------- #


def test_parameter_spread_histograms_count_every_row(gamma_case) -> None:
    fitted, X, y = gamma_case
    threshold = float(np.quantile(y, 0.9))
    spread = parameter_spread(fitted, X, threshold=threshold, n_bins=10)

    assert isinstance(spread, Spread)
    assert spread.kind == "spread"
    assert spread.threshold == pytest.approx(threshold)
    assert spread.tail_p == 0.99
    assert spread.by == "mean"
    assert tuple(spread.parameters) == ("mean", "scale")
    for name, histogram in spread.parameters.items():
        assert isinstance(histogram, Histogram)
        assert histogram.counts.shape == (30,)
        assert histogram.counts.sum() == len(X)
        assert histogram.edges.shape == (31,)
        assert np.all(np.diff(histogram.edges) > 0.0), name
    assert spread.tail_quantile.counts.shape == (30,)
    assert spread.tail_quantile.counts.sum() == len(X)

    theta = fitted.predict_parameters(X)
    tail = fitted.family.quantile(np.full(len(X), 0.99), theta)
    assert spread.tail_quantile.edges[0] == pytest.approx(float(tail.min()))
    assert spread.tail_quantile.edges[-1] == pytest.approx(float(tail.max()))

    payload = spread.to_json()
    _assert_json_leaves(payload)
    assert set(payload["parameters"]) == {"mean", "scale"}
    assert len(payload["identically_priced"]["bin"]) == 10


def test_identically_priced_bins_are_equal_count_ratios_of_sums(gamma_case) -> None:
    fitted, X, y = gamma_case
    threshold = float(np.quantile(y, 0.9))
    rng = np.random.default_rng(4)
    weights = rng.gamma(2.0, 0.5, len(X))
    spread = parameter_spread(fitted, X, threshold=threshold, n_bins=20, sample_weight=weights)

    table = spread.identically_priced
    assert list(table.columns) == [
        "bin",
        "n",
        "weight",
        "mean_lo",
        "mean_hi",
        "mean",
        "p_lo",
        "p_hi",
        "ratio",
    ]
    assert len(table) == 20
    assert table["bin"].tolist() == list(range(20))
    assert table["n"].sum() == len(X)
    assert table["n"].max() - table["n"].min() <= 1
    assert table["weight"].sum() == pytest.approx(float(weights.sum()))

    theta = fitted.predict_parameters(X)
    mean = np.asarray(fitted.family.default_prediction(theta))
    order = np.argsort(mean, kind="stable")
    codes = np.empty(len(mean), dtype=np.intp)
    unweighted = np.empty(20)
    for index, rows in enumerate(np.array_split(order, 20)):
        codes[rows] = index
        unweighted[index] = mean[rows].mean()
    expected = np.bincount(codes, weights=weights * mean, minlength=20) / np.bincount(
        codes, weights=weights, minlength=20
    )
    np.testing.assert_allclose(table["mean"].to_numpy(), expected)
    # The weighted ratio of sums is not the plain per-row mean it replaces.
    assert np.any(np.abs(table["mean"].to_numpy() - unweighted) > 1.0e-6)

    assert np.all(table["mean_lo"].to_numpy() <= table["mean"].to_numpy())
    assert np.all(table["mean"].to_numpy() <= table["mean_hi"].to_numpy())
    assert np.all(np.diff(table["mean_lo"].to_numpy()) > 0.0)

    exceedance = 1.0 - np.asarray(fitted.family.cdf(np.full(len(X), threshold), theta))
    np.testing.assert_allclose(
        table["p_lo"].to_numpy(), [np.quantile(exceedance[codes == b], 0.05) for b in range(20)]
    )
    assert np.all(table["p_lo"].to_numpy() <= table["p_hi"].to_numpy())
    assert np.all(table["ratio"].to_numpy() >= 1.0)
    assert np.isfinite(table["ratio"].to_numpy()).all()
    # Identically priced policies still differ many-fold in tail probability.
    assert table["ratio"].max() > 2.0


def test_parameter_spread_refuses_what_it_cannot_price(gamma_case) -> None:
    fitted, X, y = gamma_case
    threshold = float(np.quantile(y, 0.9))

    with pytest.raises(NotImplementedError, match="by='mean'"):
        parameter_spread(fitted, X, threshold=threshold, by="scale")
    with pytest.raises(ValueError, match="fewer rows than bins"):
        parameter_spread(fitted, X.head(5), threshold=threshold, n_bins=20)
    with pytest.raises(ValueError, match="n_bins"):
        parameter_spread(fitted, X, threshold=threshold, n_bins=0)
    with pytest.raises(ValueError, match="one weight per row"):
        parameter_spread(fitted, X, threshold=threshold, sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="finite and non-negative"):
        parameter_spread(fitted, X, threshold=threshold, sample_weight=-np.ones(len(X)))
    with pytest.raises(ValueError, match="strictly inside"):
        parameter_spread(fitted, X, threshold=threshold, tail_p=1.0)

    meanless = SimpleNamespace(
        family=_MeanlessFamily(fitted.family),
        predict_parameters=fitted.predict_parameters,
    )
    with pytest.raises(NotImplementedError, match="default prediction"):
        parameter_spread(meanless, X, threshold=threshold)

    cdfless = SimpleNamespace(
        family=_CdflessFamily(fitted.family), predict_parameters=fitted.predict_parameters
    )
    with pytest.raises(NotImplementedError, match="needs a family with a cdf"):
        parameter_spread(cdfless, X, threshold=threshold)


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #


def test_portfolio_totals_are_ordered_and_segments_sum_to_the_total(gamma_case) -> None:
    fitted, X, _ = gamma_case
    book = portfolio(fitted, X, n_draws=120, by="g", seed=9, return_draws=True)

    assert isinstance(book, Portfolio)
    assert book.kind == "portfolio"
    assert book.by == "g"
    assert book.n_draws == 120
    assert book.seed == 9
    assert book.parameter_uncertainty is True
    assert book.total_draws is not None
    assert book.total_draws.shape == (120,)
    assert not book.total_draws.flags.writeable

    levels = list(book.total_quantiles)
    assert levels == [0.5, 0.9, 0.99]
    values = [book.total_quantiles[level] for level in levels]
    assert values == sorted(values)
    assert book.total_mean == pytest.approx(float(book.total_draws.mean()))
    # Posterior parameter uncertainty is common to every row, so it widens the
    # book total well beyond the plug-in simulation of the same draws.
    plug_in = portfolio(fitted, X, n_draws=120, by="g", seed=9, parameter_uncertainty=False)
    assert plug_in.parameter_uncertainty is False
    assert book.total_sd > 1.2 * plug_in.total_sd

    table = book.by_segment
    assert list(table.columns) == ["segment", "n", "mean_total", "q0.5", "q0.9", "q0.99"]
    assert table["segment"].tolist() == ["a", "b"]
    assert table["n"].sum() == len(X)
    assert float(table["mean_total"].sum()) == pytest.approx(book.total_mean, rel=1.0e-6)
    assert np.all(table["q0.5"].to_numpy() <= table["q0.99"].to_numpy())

    payload = book.to_json()
    _assert_json_leaves(payload)
    assert payload["quantiles"] == [0.5, 0.9, 0.99]
    assert payload["total_quantiles"] == values
    assert payload["by_segment"]["segment"] == ["a", "b"]
    assert len(payload["total_draws"]) == 120


def test_portfolio_segment_totals_hold_their_own_rows_under_chunking(gamma_case) -> None:
    # A predictive total is not chunk-invariant -- the uniforms are drawn per
    # chunk -- so each segment is checked against the plug-in expectation of the
    # rows it owns, which a mis-attributed chunk would miss by many draws' worth.
    fitted, X, _ = gamma_case
    theta = np.asarray(fitted.predict_parameters(X))
    mean = np.asarray(fitted.family.default_prediction(theta))
    variance = (mean * theta[:, 1]) ** 2
    labels = X["g"].to_numpy()

    for chunk_rows in (None, 137):
        book = portfolio(
            fitted,
            X,
            n_draws=200,
            by="g",
            seed=2,
            parameter_uncertainty=False,
            chunk_rows=chunk_rows,
        )
        table = book.by_segment
        assert table["segment"].tolist() == ["a", "b"]
        for index, segment in enumerate(table["segment"]):
            rows = labels == segment
            assert int(table["n"].iat[index]) == int(rows.sum())
            expected = float(mean[rows].sum())
            deviation = math.sqrt(float(variance[rows].sum()))
            assert (
                abs(float(table["mean_total"].iat[index]) - expected) < 4.0 * deviation / 200**0.5
            )
            # Each segment's quantiles are quantiles of its own draws: the two
            # segments here sit eight of these deviations apart.
            assert abs(float(table["q0.5"].iat[index]) - expected) < 0.5 * deviation
            assert expected < float(table["q0.9"].iat[index]) < expected + 2.5 * deviation
            assert (
                float(table["q0.9"].iat[index])
                < float(table["q0.99"].iat[index])
                < expected + 4.0 * deviation
            )
        assert float(table["mean_total"].sum()) == pytest.approx(book.total_mean, rel=1.0e-9)


def test_portfolio_without_segments_keeps_only_the_total(gamma_case) -> None:
    fitted, X, _ = gamma_case
    # "south" comes first in the frame and second in the table: segments are
    # reported in label order, not in order of appearance.
    labels = np.where(np.arange(len(X)) % 3 == 0, "south", "north")
    total_only = portfolio(fitted, X, n_draws=30, seed=4, quantiles=(0.25, 0.75))

    assert total_only.by is None
    assert total_only.by_segment is None
    assert total_only.total_draws is None
    assert list(total_only.total_quantiles) == [0.25, 0.75]
    assert total_only.to_json()["by_segment"] is None

    labelled = portfolio(fitted, X, n_draws=30, seed=4, by=labels, quantiles=(0.25, 0.75))
    assert labelled.by is None
    assert labelled.by_segment["segment"].tolist() == ["north", "south"]
    assert labelled.by_segment["n"].tolist() == [
        int((labels == "north").sum()),
        int((labels == "south").sum()),
    ]
    assert total_only.total_mean == pytest.approx(labelled.total_mean, rel=1.0e-12)


def test_portfolio_refuses_a_segmentation_it_cannot_index(gamma_case) -> None:
    fitted, X, _ = gamma_case
    with pytest.raises(ValueError, match="one segment label per row"):
        portfolio(fitted, X, n_draws=20, by=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="missing segment labels"):
        portfolio(fitted, X, n_draws=20, by=np.where(np.arange(len(X)) == 0, None, "a"))
    with pytest.raises(ValueError, match="not a column"):
        portfolio(fitted, X, n_draws=20, by="nonesuch")
    with pytest.raises(ValueError, match="distinct"):
        portfolio(fitted, X, n_draws=20, quantiles=(0.5, 0.5))
    with pytest.raises(ValueError, match="distinct"):
        # Distinct floats that share a column name are as ambiguous as duplicates.
        portfolio(fitted, X, n_draws=20, quantiles=(0.5, 0.5000000001))
    with pytest.raises(ValueError, match="at least one quantile"):
        portfolio(fitted, X, n_draws=20, quantiles=())
    with pytest.raises(ValueError, match="at least two"):
        portfolio(fitted, X, n_draws=1)


def test_portfolio_pays_each_row_on_its_own_prior_weighted_law(gamma_case) -> None:
    fitted, X, _ = gamma_case
    rng = np.random.default_rng(17)
    weights = rng.uniform(0.2, 1.0, len(X))
    theta = np.asarray(fitted.predict_parameters(X))
    mean = np.asarray(fitted.family.default_prediction(theta))
    paid = weights * mean
    # Shape ``w / cv^2`` and scale ``mean cv^2 / w`` keep the row's mean and
    # give ``Var(w Y) = w mean^2 cv^2``: the paid total is a sum of those.
    deviation = math.sqrt(float((weights * (mean * theta[:, 1]) ** 2).sum()))
    draws = 300

    book = portfolio(
        fitted, X, n_draws=draws, by="g", seed=5, parameter_uncertainty=False, weights=weights
    )
    assert abs(book.total_mean - float(paid.sum())) < 4.0 * deviation / math.sqrt(draws)

    unweighted = portfolio(fitted, X, n_draws=draws, by="g", seed=5, parameter_uncertainty=False)
    assert abs(unweighted.total_mean - float(mean.sum())) < 4.0 * math.sqrt(
        float(((mean * theta[:, 1]) ** 2).sum())
    ) / math.sqrt(draws)
    # Exposures under a year buy less cover, so the paid book is far below the
    # book the unit law simulates.
    assert book.total_mean < 0.9 * unweighted.total_mean

    labels = X["g"].to_numpy()
    table = book.by_segment
    for index, segment in enumerate(table["segment"]):
        rows = labels == segment
        segment_deviation = math.sqrt(
            float((weights[rows] * (mean[rows] * theta[rows, 1]) ** 2).sum())
        )
        assert abs(float(table["mean_total"].iat[index]) - float(paid[rows].sum())) < (
            4.0 * segment_deviation / math.sqrt(draws)
        )
    assert float(table["mean_total"].sum()) == pytest.approx(book.total_mean, rel=1.0e-9)

    # A book at full exposure is the book the primitive already simulated.
    ones = portfolio(
        fitted,
        X,
        n_draws=draws,
        by="g",
        seed=5,
        parameter_uncertainty=False,
        weights=np.ones(len(X)),
    )
    assert ones.total_mean == pytest.approx(unweighted.total_mean, rel=1.0e-12)

    # An unsegmented book pays the same total through the primitive's own
    # reduce, on the same simulated responses.
    total_only = portfolio(
        fitted, X, n_draws=draws, seed=5, parameter_uncertainty=False, weights=weights
    )
    assert total_only.by_segment is None
    assert total_only.total_mean == pytest.approx(book.total_mean, rel=1.0e-12)
    with pytest.raises(ValueError, match="one weight per row"):
        portfolio(fitted, X, n_draws=20, weights=np.ones(3))


def test_surfaces_pass_the_row_law_and_the_sweep_offsets_through(gamma_case) -> None:
    fitted, X, y = gamma_case
    family = fitted.family
    reference = {"x": 0.5, "g": "a"}
    points = 12

    # A risk curve is a reference policy at full exposure unless it is told
    # otherwise; an exposure below one widens the gamma at an unchanged mean.
    unit = risk_curves(
        fitted, X, reference, "x", quantiles=(0.9,), n_points=points, n_draws=64, seed=3
    )
    weighted = risk_curves(
        fitted,
        X,
        reference,
        "x",
        quantiles=(0.9,),
        n_points=points,
        n_draws=64,
        seed=3,
        weights=np.full(points, 0.4),
    )
    assert np.all(weighted.values[0] > unit.values[0])
    full = risk_curves(
        fitted,
        X,
        reference,
        "x",
        quantiles=(0.9,),
        n_points=points,
        n_draws=64,
        seed=3,
        weights=np.ones(points),
    )
    assert np.array_equal(full.values, unit.values)

    threshold = float(np.quantile(y, 0.9))
    exposure = np.full(len(X), 0.4)
    theta = np.asarray(fitted.predict_parameters(X))
    plain = parameter_spread(fitted, X, threshold=threshold, n_bins=10)
    spread = parameter_spread(fitted, X, threshold=threshold, n_bins=10, weights=exposure)
    tail = np.asarray(
        family.quantile_prior_weighted(np.full(len(X), 0.99), theta, exposure), dtype=float
    )
    assert spread.tail_quantile.edges[0] == pytest.approx(float(tail.min()))
    assert spread.tail_quantile.edges[-1] == pytest.approx(float(tail.max()))
    # The prior weight leaves the mean alone, so the prices the table bins by
    # do not move; the tail probability it reports beside them does.
    np.testing.assert_allclose(
        spread.identically_priced["mean"].to_numpy(),
        plain.identically_priced["mean"].to_numpy(),
    )
    assert not np.allclose(
        spread.identically_priced["p_hi"].to_numpy(),
        plain.identically_priced["p_hi"].to_numpy(),
    )

    # The density fan reads sweep offsets like every other prediction path.
    fan = density_fan(fitted, X, reference, "x", n_points=8, n_y=40)
    shifted = density_fan(
        fitted, X, reference, "x", n_points=8, n_y=40, offsets={"mean": np.full(8, 0.5)}
    )
    assert float(shifted.y_grid[-1]) > float(fan.y_grid[-1])
    assert not np.array_equal(shifted.density, fan.density)


def test_segment_totals_refuse_a_block_wider_than_the_rows_they_hold() -> None:
    accumulator = _SegmentTotals(_Segmentation(None, np.zeros(3, dtype=np.intp), ("a",)), 2)
    np.testing.assert_allclose(accumulator(np.ones((2, 3))), [3.0, 3.0])
    assert accumulator.cursor == 3
    np.testing.assert_allclose(accumulator.table([], (), rows=3)["mean_total"], [3.0])
    with pytest.raises(RuntimeError, match="did not see every row"):
        accumulator.table([], (), rows=4)
    with pytest.raises(RuntimeError, match="more rows than X has"):
        accumulator(np.ones((2, 1)))

    payer = _WeightedTotal(np.array([2.0, 3.0]))
    np.testing.assert_allclose(payer(np.ones((2, 2))), [5.0, 5.0])
    with pytest.raises(RuntimeError, match="more rows than X has"):
        payer(np.ones((2, 1)))
