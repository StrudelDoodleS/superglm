"""Contract tests for binned residual checks, actual-versus-expected and calibration.

The fits here are small simulated ones; the assertions are about the statistical
contract each payload states -- a band that contains the truth on a correct
model and excludes it on a wrong one, a grouped mean that is a ratio of sums,
and a table that agrees with the literally replicated rows under frequency
weights.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import stats

from superglm import Categorical, Spline, SuperLSS
from superglm._frame import as_eager_frame
from superglm.distributional import Predictor
from superglm.distributional.checks.binned import (
    BinnedCheck,
    BinnedCheck2D,
    _covariate_bins,
    binned_check,
    binned_check_2d,
)
from superglm.distributional.checks.calibration import (
    ActualExpected,
    CalibrationPayload,
    ReliabilityCurve,
    VarianceFamily,
    actual_expected_check,
    calibration_payload,
    reliability_curve,
)
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.model import (
    DenseDistributionalModel,
    fit_dense_distributional,
)
from superglm.distributional.residuals import ResidualSet, compute_residuals
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.weights import WeightContract

# --------------------------------------------------------------------------- #
# Simulated data and fits
# --------------------------------------------------------------------------- #


def _gaussian_sample(n: int = 1500, seed: int = 20260903) -> tuple[pd.DataFrame, NDArray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    location = 0.6 * np.sin(2.4 * x) + np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    scale = np.exp(-1.0 + 0.9 * np.cos(1.8 * x))
    return pd.DataFrame({"x": x, "g": g}), location + scale * rng.standard_normal(n)


def _gamma_sample(n: int = 1200, seed: int = 20260904) -> tuple[pd.DataFrame, NDArray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    mean = np.exp(1.0 + 0.8 * np.sin(3.0 * x))
    cv = np.exp(-0.5 + 0.3 * x)
    shape = 1.0 / (cv * cv)
    return pd.DataFrame({"x": x}), rng.gamma(shape, mean / shape)


def _burn_cost_sample(n: int = 1500, seed: int = 20260905) -> tuple[pd.DataFrame, NDArray, NDArray]:
    """Compound Poisson-gamma rows with exposure weights, drawn without the engine."""
    rng = np.random.default_rng(seed)
    first = rng.uniform(-1.0, 1.0, n)
    second = rng.uniform(-1.0, 1.0, n)
    weights = rng.uniform(0.2, 1.0, n)
    mean = np.exp(0.4 + 0.5 * np.sin(np.pi * first))
    dispersion = np.exp(-0.2 + 0.3 * second)
    power = np.full(n, 1.5)
    tail_index, jump_index = 2.0 - power, power - 1.0
    rate = weights * mean**tail_index / (dispersion * tail_index)
    jump_scale = dispersion * jump_index * mean**jump_index / weights
    counts = rng.poisson(rate)
    response = np.zeros(n, dtype=np.float64)
    claimed = counts > 0
    response[claimed] = rng.gamma(
        (tail_index / jump_index)[claimed] * counts[claimed], jump_scale[claimed]
    )
    return pd.DataFrame({"first": first, "second": second}), response, weights


@pytest.fixture(scope="module")
def gaussian_case() -> tuple[DenseDistributionalModel, pd.DataFrame, NDArray]:
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
def missing_effect_case() -> tuple[DenseDistributionalModel, pd.DataFrame, NDArray]:
    """The same data with ``x`` left out of the location predictor."""
    X, y = _gaussian_sample()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"g": Categorical()}),
            Predictor("scale", {"x": Spline("cr", k=6)}),
        ],
    ).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def constant_scale_case() -> tuple[DenseDistributionalModel, pd.DataFrame, NDArray]:
    """The same data with a constant scale, so the second moment is wrong by region."""
    X, y = _gaussian_sample()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8), "g": Categorical()}),
            Predictor("scale", {}),
        ],
    ).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def gamma_case() -> tuple[DenseDistributionalModel, pd.DataFrame, NDArray]:
    X, y = _gamma_sample()
    model = SuperLSS(
        family=GammaLS(),
        predictors=[
            Predictor("mean", {"x": Spline("cr", k=8)}),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ],
    ).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def burn_cost_case() -> tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray]:
    frame, response, weights = _burn_cost_sample()
    model = fit_dense_distributional(
        frame,
        response,
        family=TweedieLSS(),
        predictors=(
            Predictor("mean", {"first": Spline(kind="cr", n_knots=6)}),
            Predictor("dispersion", {"second": Spline(kind="cr", n_knots=5)}),
            Predictor("power", {}),
        ),
        weight_contract=WeightContract("prior"),
        sample_weight=weights,
        config=DenseSolverConfig(coefficient_curvature="observed", tolerance=1.0e-8),
        lambdas={"mean:first#wiggle": 1.0, "dispersion:second#wiggle": 1.0},
        discrete=False,
        chunk_size=None,
    )
    return model, frame, response, weights


@pytest.fixture(scope="module")
def gaussian_residuals(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> ResidualSet:
    fitted, X, y = gaussian_case
    return compute_residuals(fitted, X, y)


def _residual_set(
    values: NDArray,
    *,
    weights: NDArray | None = None,
    semantics: str = "prior",
) -> ResidualSet:
    """A residual payload built straight from quantile residuals, for the edge cases."""
    quantile = np.asarray(values, dtype=np.float64)
    rows = len(quantile)
    return ResidualSet(
        pit=stats.norm.cdf(quantile),
        quantile=quantile,
        theta=np.column_stack([np.zeros(rows), np.ones(rows)]),
        eta=np.column_stack([np.zeros(rows), np.zeros(rows)]),
        y=quantile,
        weights=np.ones(rows) if weights is None else np.asarray(weights, dtype=np.float64),
        prior_weights=np.ones(rows),
        clipped_rows=0,
        randomised_rows=0,
        weight_semantics=semantics,
    )


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


# --------------------------------------------------------------------------- #
# Binned residual checks
# --------------------------------------------------------------------------- #


def test_binned_bands_contain_the_null_moments_on_the_true_model(
    gaussian_residuals: ResidualSet,
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    _fitted, X, _y = gaussian_case
    check = binned_check(gaussian_residuals, X["x"].to_numpy(), name="x", n_bins=20)

    assert isinstance(check, BinnedCheck)
    assert check.kind == "binned"
    assert check.schema_version == 1
    assert check.levels is None
    assert check.edges is not None
    assert len(check.edges) == 21
    assert len(check.centers) == 20
    assert int(check.n.sum()) == gaussian_residuals.n_rows
    assert np.all(check.mean_lower <= check.mean) and np.all(check.mean <= check.mean_upper)

    # The bands are pointwise, so a correct model still puts a bin or three of
    # twenty outside one of them; what a broken moment would give is far worse.
    contains_zero = (check.mean_lower <= 0.0) & (check.mean_upper >= 0.0)
    contains_one = (check.sd_lower <= 1.0) & (check.sd_upper >= 1.0)
    assert contains_zero.mean() >= 0.9
    assert contains_one.mean() >= 0.8


def test_binned_mean_flags_a_missing_covariate_effect(
    missing_effect_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = missing_effect_case
    residuals = compute_residuals(fitted, X, y)
    check = binned_check(residuals, X["x"].to_numpy(), name="x", n_bins=20)

    excludes_zero = (check.mean_lower > 0.0) | (check.mean_upper < 0.0)
    assert excludes_zero.mean() >= 0.3


def test_binned_sd_flags_a_constant_scale_misfit(
    constant_scale_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = constant_scale_case
    residuals = compute_residuals(fitted, X, y)
    check = binned_check(residuals, X["x"].to_numpy(), name="x", n_bins=20)

    excludes_one = (check.sd_lower > 1.0) | (check.sd_upper < 1.0)
    assert excludes_one.mean() >= 0.3
    assert np.all(np.isfinite(check.skew))


def test_binned_check_on_a_categorical_covariate_uses_one_bin_per_level(
    gaussian_residuals: ResidualSet,
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    _fitted, X, _y = gaussian_case
    check = binned_check(gaussian_residuals, X["g"].to_numpy(), name="g", n_bins=20)

    assert check.edges is None
    assert check.levels == ("a", "b", "c")
    assert np.array_equal(check.centers, np.arange(3.0))
    assert int(check.n.sum()) == gaussian_residuals.n_rows


def test_binned_check_reports_nan_moments_in_a_bin_below_three_rows() -> None:
    residuals = _residual_set(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    check = binned_check(residuals, np.arange(5.0), name="index", n_bins=2, n_boot=20)

    assert np.array_equal(check.n, np.array([2, 3]))
    assert np.isnan(check.mean[0]) and np.isnan(check.sd[0]) and np.isnan(check.skew[0])
    assert np.isnan(check.mean_lower[0]) and np.isnan(check.skew_upper[0])
    assert np.isfinite(check.mean[1]) and np.isfinite(check.sd[1]) and np.isfinite(check.skew[1])


def test_binned_check_expands_frequency_weights() -> None:
    values = np.linspace(-2.0, 2.0, 12)
    covariate = np.arange(12.0)
    counts = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.float64)
    weighted = binned_check(
        _residual_set(values, weights=counts, semantics="frequency"),
        covariate,
        name="index",
        n_bins=3,
        n_boot=25,
    )
    index = np.repeat(np.arange(12), counts.astype(int))
    replicated = binned_check(
        _residual_set(values[index]),
        covariate[index],
        name="index",
        n_bins=3,
        n_boot=25,
    )

    assert int(weighted.n.sum()) == int(counts.sum())
    assert np.array_equal(weighted.n, replicated.n)
    assert np.array_equal(weighted.mean, replicated.mean)
    assert np.array_equal(weighted.sd_lower, replicated.sd_lower)
    assert np.array_equal(weighted.skew_upper, replicated.skew_upper)


def test_binned_check_handles_a_constant_covariate(gaussian_residuals: ResidualSet) -> None:
    check = binned_check(
        gaussian_residuals,
        np.full(gaussian_residuals.n_rows, 3.0),
        name="constant",
        n_bins=8,
        n_boot=20,
    )

    assert len(check.centers) == 1
    assert int(check.n[0]) == gaussian_residuals.n_rows
    assert check.edges is not None and np.array_equal(check.edges, np.array([3.0, 3.0]))


def test_binned_check_2d_counts_sum_to_the_rows_and_empty_cells_are_nan(
    gaussian_residuals: ResidualSet,
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    _fitted, X, _y = gaussian_case
    x = X["x"].to_numpy()
    check = binned_check_2d(
        gaussian_residuals, x, gaussian_residuals.theta[:, 1], names=("x", "scale")
    )

    assert isinstance(check, BinnedCheck2D)
    assert check.kind == "binned2d"
    assert check.mean.shape == (12, 12)
    assert check.count.shape == (12, 12)
    assert int(check.count.sum()) == gaussian_residuals.n_rows
    assert np.all(np.isnan(check.mean[check.count == 0]))
    assert np.all(np.isfinite(check.mean[check.count > 0]))
    assert check.count.sum() > 0 and np.any(check.count == 0)


def test_binned_check_2d_refuses_a_non_numeric_axis(
    gaussian_residuals: ResidualSet,
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    _fitted, X, _y = gaussian_case
    with pytest.raises(ValueError, match="numeric y_cov"):
        binned_check_2d(gaussian_residuals, X["x"].to_numpy(), X["g"].to_numpy(), names=("x", "g"))
    with pytest.raises(ValueError, match="numeric x"):
        binned_check_2d(gaussian_residuals, X["g"].to_numpy(), X["x"].to_numpy(), names=("g", "x"))


def test_covariate_bins_refuses_a_shapeless_covariate() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        _covariate_bins(np.zeros((2, 2)), 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        _covariate_bins(np.zeros(0), 2)


def test_binned_check_validates_its_inputs(gaussian_residuals: ResidualSet) -> None:
    covariate = np.zeros(gaussian_residuals.n_rows)
    with pytest.raises(TypeError, match="ResidualSet"):
        binned_check(object(), covariate, name="x")
    with pytest.raises(ValueError, match="one value per residual row"):
        binned_check(gaussian_residuals, covariate[:-1], name="x")
    with pytest.raises(ValueError, match="at least one bin"):
        binned_check(gaussian_residuals, covariate, name="x", n_bins=0)
    with pytest.raises(ValueError, match="at least one resample"):
        binned_check(gaussian_residuals, covariate, name="x", n_boot=0)
    with pytest.raises(ValueError, match="finite"):
        binned_check(
            gaussian_residuals,
            np.where(np.arange(gaussian_residuals.n_rows) == 0, np.nan, 1.0),
            name="x",
        )
    with pytest.raises(ValueError, match="missing"):
        binned_check(
            _residual_set(np.array([0.1, 0.2, 0.3])),
            np.array(["a", "b", None], dtype=object),
            name="g",
        )


def test_binned_payloads_round_trip_through_json(
    gaussian_residuals: ResidualSet,
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    _fitted, X, _y = gaussian_case
    numeric = binned_check(gaussian_residuals, X["x"].to_numpy(), name="x", n_boot=20)
    levels = binned_check(gaussian_residuals, X["g"].to_numpy(), name="g", n_boot=20)
    two_d = binned_check_2d(
        gaussian_residuals, X["x"].to_numpy(), gaussian_residuals.theta[:, 0], names=("x", "loc")
    )

    for payload in (numeric, levels, two_d):
        encoded = payload.to_json()
        _assert_json_leaves(encoded)
        assert json.loads(json.dumps(encoded)) == encoded
    assert numeric.to_json()["levels"] is None
    assert levels.to_json()["edges"] is None
    assert levels.to_json()["levels"] == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# Stub fits for the exact actual-versus-expected arithmetic
# --------------------------------------------------------------------------- #


class _MeanOnlyFamily:
    """A family with a mean and nothing else: the minimum the table needs."""

    @property
    def default_prediction_name(self) -> str:
        return "mean"

    def default_prediction(self, theta: NDArray) -> NDArray:
        return np.asarray(theta, dtype=np.float64)[:, 0]


class _MeanVarianceFamily(_MeanOnlyFamily):
    """The same family with a closed-form predictive variance."""

    def variance(self, theta: NDArray) -> NDArray:
        return np.asarray(theta, dtype=np.float64)[:, 1]


class _NormalLikeFamily(_MeanOnlyFamily):
    """A family with a distribution function but no weighted law and no variance."""

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray:
        values = np.asarray(theta, dtype=np.float64)
        return stats.norm.cdf(y, values[:, 0], np.sqrt(values[:, 1]))

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray:
        values = np.asarray(theta, dtype=np.float64)
        return stats.norm.ppf(p, values[:, 0], np.sqrt(values[:, 1]))


class _WeightedNormalLikeFamily(_NormalLikeFamily):
    """A family with a prior-weighted law but no closed-form weighted variance."""

    def cdf_prior_weighted(self, y: NDArray, theta: NDArray, weights: NDArray) -> NDArray:
        values = np.asarray(theta, dtype=np.float64)
        return stats.norm.cdf(y, values[:, 0], np.sqrt(values[:, 1] / weights))

    def quantile_prior_weighted(self, p: NDArray, theta: NDArray, weights: NDArray) -> NDArray:
        values = np.asarray(theta, dtype=np.float64)
        return stats.norm.ppf(p, values[:, 0], np.sqrt(values[:, 1] / weights))


class _NoPredictionFamily:
    """A family that names no default prediction."""


class _StubFitted:
    """The three attributes ``actual_expected_check`` reads off a fitted model."""

    def __init__(self, family: Any, semantics: str = "prior") -> None:
        self.family = family
        self.fit_state = SimpleNamespace(weight_contract=WeightContract(semantics))
        self.layout = SimpleNamespace(
            predictors=(SimpleNamespace(name="mu"), SimpleNamespace(name="var"))
        )

    def predict_parameters(self, X: Any, offsets: Any = None) -> NDArray:
        frame = as_eager_frame(X)
        return np.column_stack(
            [
                frame.column_array("mu", dtype=np.float64),
                frame.column_array("var", dtype=np.float64),
            ]
        )


def test_actual_expected_is_the_ratio_of_weighted_totals() -> None:
    """Two rows in one bin: 14 over 4, not the mean of 2 and 4/3."""
    frame = pd.DataFrame({"mu": [1.0, 1.0], "var": [1.0, 1.0], "band": ["one", "one"]})
    response = np.array([2.0, 4.0])
    check = actual_expected_check(
        _StubFitted(_MeanVarianceFamily()),
        frame,
        response,
        frame["band"].to_numpy(),
        name="band",
        sample_weight=np.array([1.0, 3.0]),
    )

    assert isinstance(check, ActualExpected)
    assert check.kind == "actual_expected"
    assert check.levels == ("one",)
    assert check.weight == pytest.approx([4.0])
    assert check.actual == pytest.approx([14.0])
    assert check.expected == pytest.approx([4.0])
    assert check.ratio == pytest.approx([3.5])
    assert check.ratio_se == pytest.approx([math.sqrt(1.0 + 9.0) / 4.0])
    # A prior weight belongs inside the row's law, and this family cannot say
    # what its weighted variance is; the payload says which law it read.
    assert check.variance_law == "family_unit_law"


def test_actual_expected_frequency_weights_match_the_replicated_rows() -> None:
    rng = np.random.default_rng(11)
    rows = 24
    frame = pd.DataFrame(
        {
            "mu": rng.uniform(1.0, 3.0, rows),
            "var": rng.uniform(0.5, 2.0, rows),
            "band": rng.choice(["low", "high"], rows),
        }
    )
    response = rng.uniform(0.5, 5.0, rows)
    counts = rng.integers(1, 4, rows).astype(np.float64)
    fitted = _StubFitted(_MeanVarianceFamily(), semantics="frequency")

    weighted = actual_expected_check(
        fitted, frame, response, frame["band"].to_numpy(), name="band", sample_weight=counts
    )
    index = np.repeat(np.arange(rows), counts.astype(int))
    replicated = actual_expected_check(
        fitted,
        frame.iloc[index].reset_index(drop=True),
        response[index],
        frame["band"].to_numpy()[index],
        name="band",
    )

    assert weighted.levels == replicated.levels
    assert weighted.weight == pytest.approx(replicated.weight)
    assert weighted.actual == pytest.approx(replicated.actual)
    assert weighted.expected == pytest.approx(replicated.expected)
    assert weighted.ratio == pytest.approx(replicated.ratio)
    assert weighted.ratio_se == pytest.approx(replicated.ratio_se)
    assert weighted.variance_law == replicated.variance_law == "family"


def test_actual_expected_refuses_a_family_without_a_default_prediction() -> None:
    frame = pd.DataFrame({"mu": [1.0], "var": [1.0]})
    with pytest.raises(NotImplementedError, match="default prediction"):
        actual_expected_check(
            _StubFitted(_NoPredictionFamily()),
            frame,
            np.array([1.0]),
            np.array([0.0]),
            name="z",
        )


def test_actual_expected_validates_the_covariate_length() -> None:
    frame = pd.DataFrame({"mu": [1.0, 2.0], "var": [1.0, 1.0]})
    with pytest.raises(ValueError, match="one value per row"):
        actual_expected_check(
            _StubFitted(_MeanVarianceFamily()),
            frame,
            np.array([1.0, 2.0]),
            np.array([0.0]),
            name="z",
        )


def test_actual_expected_says_when_it_read_the_unit_law_under_prior_weights() -> None:
    """No weighted law and no closed-form variance: the payload names the fallback."""
    rng = np.random.default_rng(4)
    rows = 40
    frame = pd.DataFrame({"mu": rng.uniform(2.0, 4.0, rows), "var": rng.uniform(0.5, 1.5, rows)})
    response = frame["mu"].to_numpy() + 0.2 * rng.standard_normal(rows)
    check = actual_expected_check(
        _StubFitted(_NormalLikeFamily()),
        frame,
        response,
        np.arange(float(rows)),
        name="index",
        sample_weight=rng.uniform(0.5, 2.0, rows),
        n_bins=2,
        n_draws=64,
        seed=3,
    )

    assert check.variance_law == "unit_law_draws"
    assert np.all(check.ratio_se > 0.0)
    assert int(check.n.sum()) == rows


def test_actual_expected_simulates_when_the_family_has_no_closed_form_variance() -> None:
    """Both simulated laws: the unit one at unit weight, the weighted one under weights."""
    rng = np.random.default_rng(9)
    rows = 40
    frame = pd.DataFrame({"mu": rng.uniform(2.0, 4.0, rows), "var": rng.uniform(0.5, 1.5, rows)})
    response = frame["mu"].to_numpy() + 0.2 * rng.standard_normal(rows)
    covariate = np.arange(float(rows))

    unit = actual_expected_check(
        _StubFitted(_NormalLikeFamily()),
        frame,
        response,
        covariate,
        name="index",
        n_bins=2,
        n_draws=64,
        seed=3,
    )
    weighted = actual_expected_check(
        _StubFitted(_WeightedNormalLikeFamily()),
        frame,
        response,
        covariate,
        name="index",
        sample_weight=rng.uniform(0.5, 2.0, rows),
        n_bins=2,
        n_draws=64,
        seed=3,
    )

    assert unit.variance_law == "draws"
    assert weighted.variance_law == "prior_weighted_draws"
    assert np.all(unit.ratio_se > 0.0) and np.all(weighted.ratio_se > 0.0)


def test_variance_family_protocol_recognises_only_a_family_with_variance() -> None:
    assert isinstance(_MeanVarianceFamily(), VarianceFamily)
    assert not isinstance(_MeanOnlyFamily(), VarianceFamily)


# --------------------------------------------------------------------------- #
# Actual versus expected on real fits
# --------------------------------------------------------------------------- #


def test_actual_expected_on_the_true_gaussian_model_is_within_its_error(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gaussian_case
    check = actual_expected_check(fitted, X, y, X["x"].to_numpy(), name="x", n_bins=10, n_draws=200)

    mu = fitted.family.default_prediction(fitted.predict_parameters(X))
    assert check.variance_law == "family"
    assert float(check.weight.sum()) == pytest.approx(float(len(y)))
    assert float(check.actual.sum()) == pytest.approx(float(y.sum()))
    assert float(check.expected.sum()) == pytest.approx(float(mu.sum()))
    assert np.all(check.ratio_se > 0.0)
    assert np.all(np.abs(check.ratio - 1.0) <= 3.0 * check.ratio_se)


def test_actual_expected_on_a_categorical_covariate(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gaussian_case
    check = actual_expected_check(fitted, X, y, X["g"].to_numpy(), name="g", n_draws=50)

    assert check.levels == ("a", "b", "c")
    assert check.edges is None
    assert int(check.n.sum()) == len(y)
    encoded = check.to_json()
    _assert_json_leaves(encoded)
    assert json.loads(json.dumps(encoded)) == encoded


def test_actual_expected_reads_the_prior_weighted_law_on_a_tweedie_fit(
    burn_cost_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray],
) -> None:
    fitted, X, y, weights = burn_cost_case
    rows = slice(0, 200)
    check = actual_expected_check(
        fitted,
        X.iloc[rows].reset_index(drop=True),
        y[rows],
        X["first"].to_numpy()[rows],
        name="first",
        sample_weight=weights[rows],
        n_bins=4,
        n_draws=40,
    )

    assert check.variance_law == "family_prior_weighted"
    assert float(check.weight.sum()) == pytest.approx(float(weights[rows].sum()))
    assert float(check.actual.sum()) == pytest.approx(float((weights[rows] * y[rows]).sum()))
    assert np.all(np.isfinite(check.ratio)) and np.all(check.ratio_se > 0.0)


# --------------------------------------------------------------------------- #
# Calibration tables
# --------------------------------------------------------------------------- #


def test_coverage_matches_the_nominal_levels_on_the_true_model(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
    gaussian_residuals: ResidualSet,
) -> None:
    fitted, X, y = gaussian_case
    payload = calibration_payload(fitted, X, y, residuals=gaussian_residuals, thresholds=())

    assert isinstance(payload, CalibrationPayload)
    assert payload.kind == "calibration"
    coverage = payload.coverage
    assert list(coverage.columns) == ["level", "group", "n", "weight", "realised", "se"]

    overall = coverage[coverage["group"] == "all"].set_index("level")["realised"]
    assert abs(float(overall.loc[0.9]) - 0.9) <= 0.03
    assert abs(float(overall.loc[0.5]) - 0.5) <= 0.05
    assert float(overall.loc[0.5]) < float(overall.loc[0.99])

    for name in ("location", "scale"):
        rows = coverage[(coverage["level"] == 0.9) & (coverage["group"].str.startswith(name))]
        assert len(rows) == 10
        assert int(rows["n"].sum()) == len(y)
        assert set(rows["group"]) == {f"{name}:decile {k}" for k in range(1, 11)}


def test_tail_expectations_are_within_the_poisson_binomial_error(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gaussian_case
    thresholds = (float(np.quantile(y, 0.9)), float(np.quantile(y, 0.99)))
    payload = calibration_payload(fitted, X, y, levels=(0.9,), thresholds=thresholds)

    tails = payload.tails
    assert list(tails.columns) == [
        "threshold",
        "group",
        "n",
        "weight",
        "expected",
        "realised",
        "se",
        "log_score",
    ]
    overall = tails[tails["group"] == "all"]
    assert len(overall) == 2
    assert np.all(np.abs(overall["expected"] - overall["realised"]) <= 3.0 * overall["se"])
    assert np.all(overall["log_score"] > 0.0)

    deciles = tails[(tails["threshold"] == thresholds[0]) & (tails["group"] != "all")]
    assert len(deciles) == 10
    assert int(deciles["n"].sum()) == len(y)
    assert float(deciles["expected"].sum()) == pytest.approx(float(overall["expected"].iloc[0]))
    assert set(payload.reliability) == set(thresholds)


def test_quantile_calibration_is_within_the_binomial_error(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gaussian_case
    payload = calibration_payload(fitted, X, y, levels=(0.9,))

    table = payload.quantiles
    assert list(table.columns) == ["p", "n", "weight", "expected", "realised_exceedance", "se"]
    assert len(table) == 9
    assert table["expected"].to_numpy() == pytest.approx(1.0 - table["p"].to_numpy())
    deviation = np.abs(table["realised_exceedance"] - table["expected"]).to_numpy()
    assert np.all(deviation <= 3.0 * table["se"].to_numpy() + 0.005)


def test_calibration_payload_without_deciles_reports_only_the_overall_group(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gaussian_case
    payload = calibration_payload(
        fitted,
        X,
        y,
        levels=(0.9,),
        thresholds=(float(np.quantile(y, 0.9)),),
        quantile_grid=(0.5,),
        by_parameter_deciles=False,
    )

    assert set(payload.coverage["group"]) == {"all"}
    assert set(payload.tails["group"]) == {"all"}
    assert len(payload.coverage) == 1
    assert payload.n_rows == len(y)
    assert payload.weight_semantics == "prior"


def test_calibration_payload_on_a_gamma_fit_exercises_the_log_link(
    gamma_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gamma_case
    payload = calibration_payload(fitted, X, y, levels=(0.8, 0.95), quantile_grid=(0.1, 0.5, 0.9))

    overall = payload.coverage[payload.coverage["group"] == "all"].set_index("level")["realised"]
    assert abs(float(overall.loc[0.8]) - 0.8) <= 0.04
    assert abs(float(overall.loc[0.95]) - 0.95) <= 0.03
    assert np.all(payload.quantiles["realised_exceedance"] > 0.0)

    check = actual_expected_check(fitted, X, y, X["x"].to_numpy(), name="x", n_bins=8, n_draws=100)
    assert np.all(np.abs(check.ratio - 1.0) <= 3.0 * check.ratio_se)


def test_calibration_payload_on_the_tweedie_fit_covers_the_zero_atom(
    burn_cost_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray],
) -> None:
    fitted, X, y, weights = burn_cost_case
    payload = calibration_payload(
        fitted,
        X,
        y,
        sample_weight=weights,
        levels=(0.9,),
        thresholds=(0.0,),
        quantile_grid=(0.5, 0.9),
        by_parameter_deciles=False,
    )

    assert payload.weight_semantics == "prior"
    # The atom is read through the randomised PIT, so the central interval holds
    # its nominal level rather than counting every zero as inside it.
    assert payload.calibration_law == "randomised_pit"
    realised = float(payload.coverage["realised"].iloc[0])
    assert abs(realised - 0.9) <= 3.0 * math.sqrt(0.9 * 0.1 / len(y))

    tail = payload.tails.iloc[0]
    assert float(tail["realised"]) == pytest.approx(float((y > 0.0).sum()))
    assert abs(float(tail["expected"]) - float(tail["realised"])) <= 3.0 * float(tail["se"])

    table = payload.quantiles.set_index("p")
    assert np.all(table["realised_exceedance"] <= table["expected"] + 3.0 * table["se"])


def test_atom_family_calibration_reads_the_randomised_pit(
    burn_cost_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray],
) -> None:
    """The point mass makes the response reading wide; the randomised PIT does not.

    A zero response sits inside every central interval whose lower quantile has
    landed on the atom, so the response-based coverage of the central 50 % band
    reads high by the mass of those rows.  The randomised PIT is uniform under
    the model whatever the atom does, which is what the table now reads.
    """
    fitted, X, y, weights = burn_cost_case
    residuals = compute_residuals(fitted, X, y, sample_weight=weights)
    payload = calibration_payload(
        fitted,
        X,
        y,
        sample_weight=weights,
        residuals=residuals,
        levels=(0.5,),
        quantile_grid=(0.5,),
        by_parameter_deciles=False,
    )

    assert payload.calibration_law == "randomised_pit"
    assert residuals.randomised_rows == int((y == 0.0).sum()) > 0

    n_rows = len(y)
    error = math.sqrt(0.5 * 0.5 / n_rows)
    realised = float(payload.coverage["realised"].iloc[0])
    assert abs(realised - 0.5) <= 3.0 * error

    lower = fitted.family.quantile_prior_weighted(np.full(n_rows, 0.25), residuals.theta, weights)
    upper = fitted.family.quantile_prior_weighted(np.full(n_rows, 0.75), residuals.theta, weights)
    on_the_response = float(((y >= lower) & (y <= upper)).mean())
    assert on_the_response - 0.5 > 3.0 * error


def test_quantile_calibration_on_an_atom_family_tracks_one_minus_p(
    burn_cost_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray, NDArray],
) -> None:
    """Below the atom every predicted quantile is zero, so the response is flat."""
    fitted, X, y, weights = burn_cost_case
    payload = calibration_payload(
        fitted,
        X,
        y,
        sample_weight=weights,
        levels=(0.9,),
        by_parameter_deciles=False,
    )

    assert payload.calibration_law == "randomised_pit"
    table = payload.quantiles
    deviation = np.abs(table["realised_exceedance"] - table["expected"]).to_numpy()
    assert np.all(deviation <= 3.0 * table["se"].to_numpy())

    theta = compute_residuals(fitted, X, y, sample_weight=weights).theta
    n_rows = len(y)
    on_the_response = [
        float(
            (
                y
                > fitted.family.quantile_prior_weighted(
                    np.full(n_rows, probability), theta, weights
                )
            ).mean()
        )
        for probability in (0.01, 0.05)
    ]
    claim_rate = float((y > 0.0).mean())
    assert on_the_response == pytest.approx([claim_rate, claim_rate], abs=0.02)


def test_the_response_law_reads_the_prior_weighted_quantile(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    """Without an atom the tables stay on the response -- of the weighted law."""
    fitted, X, y = gaussian_case
    weights = np.linspace(0.5, 2.0, len(y))
    payload = calibration_payload(
        fitted,
        X,
        y,
        sample_weight=weights,
        levels=(0.9,),
        quantile_grid=(0.5,),
        by_parameter_deciles=False,
    )

    assert payload.calibration_law == "response"
    theta = compute_residuals(fitted, X, y, sample_weight=weights).theta
    median = fitted.family.quantile_prior_weighted(np.full(len(y), 0.5), theta, weights)
    assert float(payload.quantiles["realised_exceedance"].iloc[0]) == float((y > median).mean())


def test_a_family_without_an_atom_keeps_the_response_law(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
    gaussian_residuals: ResidualSet,
) -> None:
    """No atom, no randomisation: both tables stay the response-based ones."""
    fitted, X, y = gaussian_case
    levels, grid = (0.5, 0.9), (0.1, 0.9)
    payload = calibration_payload(
        fitted,
        X,
        y,
        residuals=gaussian_residuals,
        levels=levels,
        quantile_grid=grid,
        by_parameter_deciles=False,
    )

    assert payload.calibration_law == "response"
    assert gaussian_residuals.randomised_rows == 0

    theta = gaussian_residuals.theta
    n_rows = len(y)

    def quantile(probability: float) -> NDArray:
        return fitted.family.quantile(np.full(n_rows, probability), theta)

    coverage = [
        float(((y >= quantile(0.5 * (1.0 - level))) & (y <= quantile(0.5 * (1.0 + level)))).mean())
        for level in levels
    ]
    exceedance = [float((y > quantile(probability)).mean()) for probability in grid]

    assert np.array_equal(payload.coverage["realised"].to_numpy(), np.array(coverage))
    assert np.array_equal(payload.quantiles["realised_exceedance"].to_numpy(), np.array(exceedance))


def test_calibration_payload_validates_its_arguments(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
    gaussian_residuals: ResidualSet,
) -> None:
    fitted, X, y = gaussian_case
    with pytest.raises(ValueError, match="strictly inside"):
        calibration_payload(fitted, X, y, levels=(1.0,))
    with pytest.raises(ValueError, match="strictly inside"):
        calibration_payload(fitted, X, y, quantile_grid=(0.0,))
    with pytest.raises(TypeError, match="ResidualSet"):
        calibration_payload(fitted, X, y, residuals=object())
    with pytest.raises(ValueError, match="same rows"):
        calibration_payload(fitted, X.iloc[:-1], y[:-1], residuals=gaussian_residuals)
    with pytest.raises(ValueError, match="same rows"):
        calibration_payload(fitted, X, y + 1.0, residuals=gaussian_residuals)


def test_calibration_payload_drops_zero_weight_rows(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    """A zero-weight row leaves the tables the way it leaves the likelihood."""
    fitted, X, y = gaussian_case
    weights = np.ones(len(y))
    weights[:20] = 0.0
    payload = calibration_payload(
        fitted,
        X,
        y,
        sample_weight=weights,
        levels=(0.9,),
        quantile_grid=(0.5,),
        by_parameter_deciles=False,
    )

    assert payload.n_rows == len(y) - 20
    assert int(payload.coverage["n"].iloc[0]) == len(y) - 20


def test_calibration_refuses_a_family_without_a_distribution_function() -> None:
    rows = 6
    frame = pd.DataFrame({"mu": np.full(rows, 2.0), "var": np.ones(rows)})
    response = np.linspace(1.0, 3.0, rows)
    with pytest.raises(NotImplementedError, match="distribution function"):
        calibration_payload(
            _StubFitted(_MeanVarianceFamily()),
            frame,
            response,
            residuals=_residual_set(response),
            levels=(0.9,),
            quantile_grid=(0.5,),
            by_parameter_deciles=False,
        )


def test_calibration_refuses_a_family_without_its_prior_weighted_law() -> None:
    """Non-unit prior weights change the row law; the unit law is not a fallback."""
    rows = 6
    frame = pd.DataFrame({"mu": np.full(rows, 2.0), "var": np.ones(rows)})
    response = np.linspace(1.0, 3.0, rows)
    with pytest.raises(NotImplementedError, match="prior-weighted distribution function"):
        calibration_payload(
            _StubFitted(_NormalLikeFamily()),
            frame,
            response,
            residuals=_residual_set(response),
            sample_weight=np.linspace(0.5, 1.5, rows),
            levels=(0.9,),
            quantile_grid=(0.5,),
            by_parameter_deciles=False,
        )


def test_calibration_payload_round_trips_through_json(
    gaussian_case: tuple[DenseDistributionalModel, pd.DataFrame, NDArray],
) -> None:
    fitted, X, y = gaussian_case
    payload = calibration_payload(
        fitted,
        X,
        y,
        levels=(0.9,),
        thresholds=(float(np.quantile(y, 0.9)),),
        quantile_grid=(0.5,),
        by_parameter_deciles=False,
    )

    encoded = payload.to_json()
    _assert_json_leaves(encoded)
    assert json.loads(json.dumps(encoded)) == encoded
    assert encoded["kind"] == "calibration"
    assert set(encoded["reliability"]) == {str(float(np.quantile(y, 0.9)))}


# --------------------------------------------------------------------------- #
# CORP reliability curve
# --------------------------------------------------------------------------- #


def _bernoulli_forecast(n: int, *, shift: float = 0.0, seed: int = 7) -> tuple[NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    probability = rng.uniform(0.05, 0.95, n)
    event = rng.uniform(size=n) < np.clip(probability + shift, 0.0, 1.0)
    return probability, event


def test_reliability_curve_is_monotone_and_spans_the_forecast_values() -> None:
    probability, event = _bernoulli_forecast(600)
    curve = reliability_curve(probability, event, n_boot=50, seed=3)

    assert isinstance(curve, ReliabilityCurve)
    assert curve.kind == "reliability"
    assert np.all(np.diff(curve.calibrated) >= -1.0e-12)
    assert np.all(np.diff(curve.x) > 0.0)
    assert float(curve.x[0]) == pytest.approx(float(probability.min()))
    assert float(curve.x[-1]) == pytest.approx(float(probability.max()))
    assert int(curve.count.sum()) == len(probability)
    assert np.all((curve.calibrated >= 0.0) & (curve.calibrated <= 1.0))
    assert np.all(curve.lower <= curve.upper)
    encoded = curve.to_json()
    _assert_json_leaves(encoded)
    assert json.loads(json.dumps(encoded)) == encoded


def test_reliability_band_brackets_a_calibrated_forecast_and_flags_a_biased_one() -> None:
    probability, event = _bernoulli_forecast(800, seed=5)
    calibrated = reliability_curve(probability, event, n_boot=200, seed=5)
    inside = (calibrated.calibrated >= calibrated.lower) & (
        calibrated.calibrated <= calibrated.upper
    )
    assert inside.mean() >= 0.8

    biased_probability, biased_event = _bernoulli_forecast(800, shift=0.25, seed=5)
    biased = reliability_curve(biased_probability, biased_event, n_boot=200, seed=5)
    outside = biased.calibrated > biased.upper
    assert outside.mean() >= 0.3


def test_reliability_curve_pools_ties_and_recalibrates_a_discrete_forecast() -> None:
    probability = np.repeat(np.array([0.2, 0.4, 0.6, 0.8]), 50)
    event = np.repeat(np.array([1.0, 0.0, 1.0, 1.0]), 50)
    curve = reliability_curve(probability, event, n_boot=20, seed=1)

    assert np.array_equal(curve.x, np.array([0.2, 0.4, 0.6, 0.8]))
    assert np.array_equal(curve.count, np.array([50, 50, 50, 50]))
    # The violating pair 1, 0 is pooled to their common mean, the rest stand.
    assert curve.calibrated == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_reliability_curve_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="one event per forecast"):
        reliability_curve(np.array([0.1, 0.2]), np.array([1.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        reliability_curve(np.array([1.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="zero or one"):
        reliability_curve(np.array([0.5]), np.array([0.5]))
    with pytest.raises(ValueError, match="at least one resample"):
        reliability_curve(np.array([0.5]), np.array([1.0]), n_boot=0)
    with pytest.raises(ValueError, match="at least one row"):
        reliability_curve(np.zeros(0), np.zeros(0))
