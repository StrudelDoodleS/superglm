"""Contract tests for LSS proper scores, paired comparison and the Murphy diagram.

The closed forms are checked against the quantile-score integral and, where a
formula is load bearing, against an independent brute-force integral of
``(F(z) - 1{y <= z})^2 dz`` -- the two routes are derived from different
representations of the CRPS, so agreement is evidence and not a tautology.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.checks.compare import (
    Comparison,
    MurphyPayload,
    _replicated_quantiles,
    compare_models,
    murphy_diagram,
)
from superglm.distributional.checks.scores import (
    crps,
    crps_closed_form,
    crps_numeric,
    has_closed_form_crps,
    log_score,
    score_table,
    threshold_weighted_crps,
)
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    resolve_likelihood_weights,
)

_GRID = 96


def _simulated(n: int = 1200, seed: int = 20260903) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    location = 0.6 * np.sin(2.4 * x) + np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    scale = np.exp(-1.0 + 0.7 * np.cos(1.8 * x))
    return pd.DataFrame({"x": x, "g": g}), location + scale * rng.standard_normal(n)


def _fit(scale_features: dict, *, weight_semantics: str = "prior", sample_weight=None):
    X, y = _simulated()
    model = SuperLSS(
        family=GaussianLS(),
        weight_semantics=weight_semantics,
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8), "g": Categorical()}),
            Predictor("scale", scale_features),
        ],
    ).fit_reml(X, y, sample_weight=sample_weight)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def fit_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    return _fit({"x": Spline("cr", k=6)})


@pytest.fixture(scope="module")
def misfit_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    """The same location model with a constant scale: heteroscedasticity ignored."""
    return _fit({})


@pytest.fixture(scope="module")
def frequency_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    return _fit({"x": Spline("cr", k=6)}, weight_semantics="frequency")


@pytest.fixture(scope="module")
def frequency_misfit_case() -> tuple[DenseDistributionalModel, pd.DataFrame, np.ndarray]:
    return _fit({}, weight_semantics="frequency")


class _QuantileFreeFamily:
    """A family with neither a cdf nor a quantile function."""

    @property
    def parameters(self):  # pragma: no cover - the guard never reaches it
        return ()


def _brute_force_crps(cdf, y: float, lower: float, upper: float) -> float:
    """``int (F(z) - 1{y <= z})^2 dz`` by adaptive quadrature, split at the kink."""

    def integrand(z: float) -> float:
        return (cdf(z) - (1.0 if y <= z else 0.0)) ** 2

    left, _ = integrate.quad(integrand, lower, y, limit=300, epsabs=1e-13, epsrel=1e-13)
    right, _ = integrate.quad(integrand, y, upper, limit=300, epsabs=1e-13, epsrel=1e-13)
    return left + right


# --------------------------------------------------------------------------
# Numeric quadrature against the closed forms
# --------------------------------------------------------------------------


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected) / np.abs(expected)))


def test_numeric_crps_matches_the_gaussian_closed_form() -> None:
    rng = np.random.default_rng(11)
    mean = rng.uniform(-2.0, 2.0, _GRID)
    sd = rng.uniform(0.2, 2.0, _GRID)
    theta = np.column_stack((mean, sd))
    y = mean + sd * rng.standard_normal(_GRID)
    family = GaussianLS()

    closed = crps_closed_form(family, y, theta)
    z = (y - mean) / sd
    assert (
        _relative(
            closed,
            sd * (z * (2 * special.ndtr(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)),
        )
        < 1e-14
    )
    assert _relative(crps_numeric(family, y, theta, n_nodes=64), closed) < 1e-6
    assert has_closed_form_crps(family)

    row = 0
    brute = _brute_force_crps(
        lambda z: float(special.ndtr((z - mean[row]) / sd[row])),
        float(y[row]),
        float(mean[row] - 60.0 * sd[row]),
        float(mean[row] + 60.0 * sd[row]),
    )
    assert abs(closed[row] - brute) / brute < 1e-9


def test_numeric_crps_matches_the_gamma_closed_form() -> None:
    rng = np.random.default_rng(12)
    mean = rng.uniform(0.5, 20.0, _GRID)
    cv = rng.uniform(0.3, 1.2, _GRID)
    theta = np.column_stack((mean, cv))
    shape = 1.0 / (cv * cv)
    scale = mean * cv * cv
    y = stats.gamma.rvs(shape, scale=scale, random_state=13)
    family = GammaLS()

    closed = crps_closed_form(family, y, theta)
    expected = (
        y * (2 * special.gammainc(shape, y / scale) - 1)
        - shape * scale * (2 * special.gammainc(shape + 1, y / scale) - 1)
        - scale / special.beta(0.5, shape)
    )
    # The restatement re-associates shape and scale from cv, so agreement is
    # bounded by float re-association rather than by the formula.
    assert _relative(closed, expected) < 1e-13
    assert _relative(crps_numeric(family, y, theta, n_nodes=64), closed) < 1e-6

    row = 0
    brute = _brute_force_crps(
        lambda z: float(stats.gamma.cdf(z, shape[row], scale=scale[row])),
        float(y[row]),
        0.0,
        float(stats.gamma.ppf(1 - 1e-15, shape[row], scale=scale[row])),
    )
    assert abs(closed[row] - brute) / brute < 1e-9


@pytest.mark.parametrize("parametrisation", ["mean", "location"])
def test_numeric_crps_matches_the_log_normal_closed_form(parametrisation: str) -> None:
    rng = np.random.default_rng(14)
    sigma = rng.uniform(0.15, 1.1, _GRID)
    mu = rng.uniform(-1.0, 2.0, _GRID)
    first = np.exp(mu + 0.5 * sigma * sigma) if parametrisation == "mean" else mu
    theta = np.column_stack((first, sigma))
    y = np.exp(mu + sigma * rng.standard_normal(_GRID))
    family = LogNormalLS(parametrisation=parametrisation)

    closed = crps_closed_form(family, y, theta)
    z = (np.log(y) - mu) / sigma
    expected = y * (2 * special.ndtr(z) - 1) - 2 * np.exp(mu + 0.5 * sigma * sigma) * (
        special.ndtr(z - sigma) + special.ndtr(sigma / np.sqrt(2.0)) - 1.0
    )
    assert _relative(closed, expected) < 1e-12
    assert _relative(crps_numeric(family, y, theta, n_nodes=64), closed) < 1e-6

    row = 0
    brute = _brute_force_crps(
        lambda z: float(stats.lognorm.cdf(z, sigma[row], scale=np.exp(mu[row]))),
        float(y[row]),
        0.0,
        float(stats.lognorm.ppf(1 - 1e-16, sigma[row], scale=np.exp(mu[row]))),
    )
    assert abs(closed[row] - brute) / brute < 1e-9


def test_closed_form_and_numeric_routes_are_selectable_on_a_fit(fit_case) -> None:
    fitted, X, y = fit_case
    automatic = crps(fitted, X, y)
    assert np.array_equal(automatic, crps(fitted, X, y, method="closed"))
    assert _relative(crps(fitted, X, y, method="numeric", n_nodes=64), automatic) < 1e-6
    assert automatic.shape == (len(y),)
    assert np.all(automatic > 0.0)


def test_prior_weighted_gaussian_crps_uses_sigma_over_sqrt_weight(fit_case) -> None:
    """Dropping the prior-law scale adjustment would restore the wider unit law."""
    fitted, X, y = fit_case
    frame = X.iloc[:8]
    response = y[:8]
    weights = np.array([0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
    theta = np.asarray(fitted.predict_parameters(frame), dtype=np.float64)
    prior_theta = np.array(theta, copy=True)
    prior_theta[:, 1] /= np.sqrt(weights)
    expected = crps_closed_form(GaussianLS(), response, prior_theta)

    assert np.allclose(
        crps(fitted, frame, response, method="closed", sample_weight=weights),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.allclose(
        crps(fitted, frame, response, method="numeric", sample_weight=weights, n_nodes=64),
        expected,
        rtol=1.0e-6,
        atol=1.0e-10,
    )

    threshold = float(np.median(response))
    expected_tail = crps_numeric(
        GaussianLS(), response, prior_theta, threshold=threshold, n_nodes=64
    )
    assert np.allclose(
        threshold_weighted_crps(
            fitted,
            frame,
            response,
            threshold,
            sample_weight=weights,
            n_nodes=64,
        ),
        expected_tail,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_non_unit_prior_crps_propagates_the_log_normal_contract_error(fit_case) -> None:
    """Using LogNormalLS's unit law here would silently invent a weighted law."""
    fitted, X, y = fit_case
    log_normal = DenseDistributionalModel(LogNormalLS(parametrisation="location"), fitted.fit_state)
    frame = X.iloc[:5]
    response = np.exp(y[:5])
    weights = np.array([0.5, 0.75, 1.0, 1.5, 2.0])

    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior weights"):
        crps(log_normal, frame, response, sample_weight=weights)
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior weights"):
        threshold_weighted_crps(
            log_normal,
            frame,
            response,
            float(np.median(response)),
            sample_weight=weights,
        )


def test_a_family_without_a_catalogued_closed_form_falls_back_and_refuses(fit_case) -> None:
    fitted, X, y = fit_case
    theta = np.asarray(fitted.predict_parameters(X))

    class _UncataloguedGaussian(GaussianLS):
        """Same distribution, a class name the catalogue does not know."""

    family = _UncataloguedGaussian()
    assert not has_closed_form_crps(family)
    with pytest.raises(NotImplementedError, match="no catalogued closed-form CRPS"):
        crps_closed_form(family, y, theta)
    assert (
        _relative(crps_numeric(family, y, theta), crps_closed_form(GaussianLS(), y, theta)) < 1e-6
    )


def test_numeric_crps_validates_its_arguments(fit_case) -> None:
    fitted, X, y = fit_case
    theta = np.asarray(fitted.predict_parameters(X))
    with pytest.raises(ValueError, match="n_nodes"):
        crps_numeric(fitted.family, y, theta, n_nodes=0)
    with pytest.raises(NotImplementedError, match="quantile"):
        crps_numeric(_QuantileFreeFamily(), y, theta)
    with pytest.raises(ValueError, match="one response value per row"):
        crps_numeric(fitted.family, y[:3], theta)
    with pytest.raises(ValueError, match=r"\(rows, parameters\) matrix"):
        crps_closed_form(fitted.family, y, theta[:, 0])
    with pytest.raises(ValueError, match="method"):
        crps(fitted, X, y, method="bogus")
    with pytest.raises(NotImplementedError, match="no catalogued closed-form CRPS"):
        crps_closed_form(_QuantileFreeFamily(), y, theta)


def test_a_scalar_response_broadcasts_across_rows(fit_case) -> None:
    fitted, X, y = fit_case
    theta = np.asarray(fitted.predict_parameters(X.iloc[:20]))
    broadcast = crps_closed_form(fitted.family, 0.25, theta)
    assert np.array_equal(broadcast, crps_closed_form(fitted.family, np.full(20, 0.25), theta))
    assert np.allclose(crps_numeric(fitted.family, 0.25, theta, n_nodes=48), broadcast, rtol=1e-6)


# --------------------------------------------------------------------------
# Threshold-weighted CRPS
# --------------------------------------------------------------------------


def test_threshold_weighted_crps_reduces_to_crps_at_minus_infinity(fit_case) -> None:
    fitted, X, y = fit_case
    theta = np.asarray(fitted.predict_parameters(X))
    unweighted = crps_numeric(fitted.family, y, theta, n_nodes=48)

    assert np.array_equal(
        crps_numeric(fitted.family, y, theta, n_nodes=48, threshold=-np.inf), unweighted
    )
    assert np.array_equal(threshold_weighted_crps(fitted, X, y, -np.inf, n_nodes=48), unweighted)


def test_threshold_weighted_crps_shrinks_as_the_threshold_rises(fit_case) -> None:
    fitted, X, y = fit_case
    reference = threshold_weighted_crps(fitted, X, y, -np.inf, n_nodes=48)
    low = threshold_weighted_crps(fitted, X, y, float(np.quantile(y, 0.10)), n_nodes=48)
    high = threshold_weighted_crps(fitted, X, y, float(np.quantile(y, 0.90)), n_nodes=48)

    assert np.all(low <= reference + 1e-12)
    assert np.all(high <= low + 1e-12)
    assert np.all(high >= -1e-12)
    assert high.mean() < 0.5 * low.mean()
    assert np.all(threshold_weighted_crps(fitted, X, y, np.inf, n_nodes=48) == 0.0)


# --------------------------------------------------------------------------
# Log score
# --------------------------------------------------------------------------


def test_log_score_is_the_negated_reported_row_likelihood(fit_case) -> None:
    fitted, X, y = fit_case
    weights = resolve_likelihood_weights(
        None, n_observations=len(y), contract=fitted.fit_state.weight_contract
    )
    plan = fitted.family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    evaluation = fitted.family.evaluate_natural(
        y, np.asarray(fitted.predict_parameters(X)), plan, derivative_order=0
    )

    scored = log_score(fitted, X, y)
    assert scored.shape == (len(y),)
    assert np.allclose(scored, -np.asarray(evaluation.reported_log_likelihood), rtol=0, atol=1e-12)


def test_log_score_drops_zero_weight_rows_and_names_them(fit_case) -> None:
    fitted, X, y = fit_case
    weights = np.ones(len(y))
    weights[[3, 17, 900]] = 0.0

    scored = log_score(fitted, X, y, sample_weight=weights)
    assert np.all(np.isnan(scored[[3, 17, 900]]))
    retained = np.setdiff1d(np.arange(len(y)), [3, 17, 900])
    assert np.allclose(scored[retained], log_score(fitted, X, y)[retained])


def test_log_score_replicates_rows_under_frequency_semantics(frequency_case) -> None:
    fitted, X, y = frequency_case
    assert fitted.fit_state.weight_contract.semantics == "frequency"
    counts = np.full(len(y), 1.0)
    counts[::4] = 3.0

    weighted = log_score(fitted, X, y, sample_weight=counts)
    assert np.allclose(weighted, counts * log_score(fitted, X, y))


def test_log_score_validates_shapes(fit_case) -> None:
    fitted, X, y = fit_case
    with pytest.raises(ValueError, match="one response value per row"):
        log_score(fitted, X, y[:-1])
    with pytest.raises(ValueError, match="unknown offset predictor name"):
        log_score(fitted, X, y, offsets={"nowhere": np.zeros(len(y))})


def test_log_score_carries_the_offset_through(fit_case) -> None:
    fitted, X, y = fit_case
    shifted = log_score(fitted, X, y, offsets={"location": np.full(len(y), 0.4)})
    assert not np.allclose(shifted, log_score(fitted, X, y))


# --------------------------------------------------------------------------
# Score table
# --------------------------------------------------------------------------


def test_score_table_carries_one_column_per_requested_score(fit_case) -> None:
    fitted, X, y = fit_case
    threshold = float(np.quantile(y, 0.9))
    table = score_table(fitted, X, y, thresholds=(threshold,), n_nodes=48)

    assert list(table.columns) == ["log", "crps", f"twcrps_{threshold:g}"]
    assert table.index.equals(X.index)
    assert np.allclose(table["log"].to_numpy(), log_score(fitted, X, y))
    assert np.allclose(table["crps"].to_numpy(), crps(fitted, X, y))
    assert np.allclose(
        table[f"twcrps_{threshold:g}"].to_numpy(),
        threshold_weighted_crps(fitted, X, y, threshold, n_nodes=48),
    )

    only_log = score_table(fitted, X, y, which=("log",))
    assert list(only_log.columns) == ["log"]
    only_crps = score_table(fitted, X, y, which=("crps",), thresholds=(threshold,), n_nodes=48)
    assert list(only_crps.columns) == ["crps", f"twcrps_{threshold:g}"]


def test_score_table_refuses_an_unknown_score(fit_case) -> None:
    fitted, X, y = fit_case
    with pytest.raises(ValueError, match="unknown score"):
        score_table(fitted, X, y, which=("brier",))
    with pytest.raises(ValueError, match="at least one score"):
        score_table(fitted, X, y, which=())


def test_frequency_score_table_is_the_compressed_explicit_replication(frequency_case) -> None:
    """Losing the aggregation multiplier would understate every non-unit row."""
    fitted, X, y = frequency_case
    frame = X.iloc[:8].reset_index(drop=True)
    response = np.array(y[:8], copy=True)
    counts = np.array([2, 0, 3, 1, 4, 2, 1, 3])
    response[1] = np.inf  # a dropped row must never reach any scoring primitive
    threshold = float(np.median(response[counts > 0]))
    expanded_positions = np.repeat(np.arange(len(frame)), counts)
    expanded = score_table(
        fitted,
        frame.iloc[expanded_positions].reset_index(drop=True),
        response[expanded_positions],
        thresholds=(threshold,),
        n_nodes=48,
    )

    expected = np.full((len(frame), expanded.shape[1]), np.nan)
    cursor = 0
    for position, count in enumerate(counts):
        if count:
            expected[position] = expanded.iloc[cursor : cursor + count].sum(axis=0)
            cursor += count

    compressed = score_table(
        fitted,
        frame,
        response,
        sample_weight=counts,
        thresholds=(threshold,),
        n_nodes=48,
    )
    assert list(compressed.columns) == ["log", "crps", f"twcrps_{threshold:g}"]
    assert np.all(np.isnan(compressed.iloc[1].to_numpy()))
    assert np.allclose(compressed.to_numpy(), expected, rtol=1.0e-12, atol=1.0e-12, equal_nan=True)


def test_zero_weight_rows_are_removed_before_offset_value_validation(frequency_case) -> None:
    fitted, X, y = frequency_case
    frame = X.iloc[:3].reset_index(drop=True)
    response = np.array([y[0], np.inf, y[2]])
    counts = np.array([1, 0, 2])
    location_offset = np.array([0.1, np.nan, -0.2])
    retained = np.array([0, 2])
    threshold = float(np.median(response[retained]))

    expected = score_table(
        fitted,
        frame.iloc[retained].reset_index(drop=True),
        response[retained],
        sample_weight=counts[retained],
        offsets={"location": location_offset[retained]},
        thresholds=(threshold,),
        n_nodes=48,
    )
    actual = score_table(
        fitted,
        frame,
        response,
        sample_weight=counts,
        offsets={"location": location_offset},
        thresholds=(threshold,),
        n_nodes=48,
    )

    assert np.all(np.isnan(actual.iloc[1].to_numpy()))
    assert np.allclose(
        actual.iloc[retained].to_numpy(),
        expected.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# --------------------------------------------------------------------------
# Comparison and Murphy diagram
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["log", "crps"])
def test_the_true_model_beats_the_constant_scale_misfit(fit_case, misfit_case, which: str) -> None:
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case

    comparison = compare_models(fitted, misfitted, X, y, which=which)
    assert isinstance(comparison, Comparison)
    assert comparison.kind == "comparison"
    assert comparison.score == which
    assert comparison.overall["n"] == len(y)
    assert comparison.overall["mean_diff"] < 0.0
    assert comparison.overall["t"] < -3.0
    assert comparison.by_segment is None
    assert comparison.murphy is None


def test_comparison_segments_the_difference(fit_case, misfit_case) -> None:
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case

    by_name = compare_models(fitted, misfitted, X, y, which="crps", by="g")
    assert list(by_name.by_segment.index) == ["a", "b", "c"]
    assert list(by_name.by_segment.columns) == ["n", "mean_diff", "se", "t"]
    assert int(by_name.by_segment["n"].sum()) == len(y)

    labels = np.where(X["x"].to_numpy() < 0.0, "low", "high")
    by_array = compare_models(fitted, misfitted, X, y, which="crps", by=labels)
    assert list(by_array.by_segment.index) == ["high", "low"]

    singleton = np.array(["only"] + ["rest"] * (len(y) - 1))
    thin = compare_models(fitted, misfitted, X, y, which="crps", by=singleton)
    assert np.isnan(thin.by_segment.loc["only", "se"])
    assert np.isnan(thin.by_segment.loc["only", "t"])


def test_prior_weighted_comparison_uses_the_weighted_gaussian_row_laws(
    fit_case, misfit_case
) -> None:
    """Using unit-law quantiles would move both the CRPS gap and Murphy curves."""
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case
    frame = X.iloc[:8].reset_index(drop=True)
    response = y[:8]
    weights = np.array([0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
    level = 0.8
    thresholds = np.linspace(float(response.min()), float(response.max()), 31)

    theta_a = np.array(fitted.predict_parameters(frame), dtype=np.float64, copy=True)
    theta_b = np.array(misfitted.predict_parameters(frame), dtype=np.float64, copy=True)
    theta_a[:, 1] /= np.sqrt(weights)
    theta_b[:, 1] /= np.sqrt(weights)
    differences = crps_closed_form(GaussianLS(), response, theta_a) - crps_closed_form(
        GaussianLS(), response, theta_b
    )
    expected_mean = float(differences.mean())
    expected_se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    expected_quantiles = murphy_diagram(
        GaussianLS().quantile(np.full(len(frame), level), theta_a),
        GaussianLS().quantile(np.full(len(frame), level), theta_b),
        response,
        level=level,
        thresholds=thresholds,
    )

    comparison = compare_models(
        fitted,
        misfitted,
        frame,
        response,
        which="crps",
        sample_weight=weights,
        murphy_quantile=level,
        thresholds=thresholds,
    )
    assert comparison.overall["n"] == len(frame)
    assert np.isclose(comparison.overall["mean_diff"], expected_mean, rtol=1.0e-12)
    assert np.isclose(comparison.overall["se"], expected_se, rtol=1.0e-12)
    assert comparison.murphy is not None
    for name in ("a", "b", "difference", "difference_se"):
        assert np.allclose(
            getattr(comparison.murphy, name),
            getattr(expected_quantiles, name),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


@pytest.mark.parametrize("which", ["log", "crps"])
def test_frequency_comparison_matches_explicit_row_replication(
    frequency_case, frequency_misfit_case, which: str
) -> None:
    """Treating compressed log scores as rows again would square the counts."""
    fitted, X, y = frequency_case
    misfitted, _, _ = frequency_misfit_case
    frame = X.iloc[:9].reset_index(drop=True)
    response = np.array(y[:9], copy=True)
    counts = np.array([2, 0, 3, 1, 4, 2, 1, 3, 2])
    response[1] = np.inf
    labels = np.array(["a", "dropped", "a", "b", "b", "a", "b", "a", "b"])
    level = 0.8
    expanded_positions = np.repeat(np.arange(len(frame)), counts)

    weighted = compare_models(
        fitted,
        misfitted,
        frame,
        response,
        which=which,
        sample_weight=counts,
        by=labels,
        murphy_quantile=level,
        n_nodes=48,
    )
    expanded = compare_models(
        fitted,
        misfitted,
        frame.iloc[expanded_positions].reset_index(drop=True),
        response[expanded_positions],
        which=which,
        by=labels[expanded_positions],
        murphy_quantile=level,
        n_nodes=48,
    )

    assert weighted.overall["n"] == expanded.overall["n"] == int(counts.sum())
    for name in ("mean_diff", "se", "t"):
        assert np.isclose(
            weighted.overall[name], expanded.overall[name], rtol=1.0e-12, atol=1.0e-12
        )
    assert list(weighted.by_segment.index) == list(expanded.by_segment.index) == ["a", "b"]
    assert np.array_equal(weighted.by_segment["n"].to_numpy(), expanded.by_segment["n"].to_numpy())
    assert np.allclose(
        weighted.by_segment[["mean_diff", "se", "t"]],
        expanded.by_segment[["mean_diff", "se", "t"]],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert weighted.murphy is not None and expanded.murphy is not None
    assert np.array_equal(weighted.murphy.thresholds, expanded.murphy.thresholds)
    for name in ("a", "b", "difference", "difference_se"):
        assert np.allclose(
            getattr(weighted.murphy, name),
            getattr(expanded.murphy, name),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    assert weighted.murphy.n_observations == expanded.murphy.n_observations == int(counts.sum())


def test_compressed_replication_quantiles_match_numpy_upper_interpolation() -> None:
    """Interpolating only from below changes some default grids by several ulps."""
    values = np.array([-2.0, -1.0, 0.1])
    counts = np.array([4, 4, 1])
    probability = np.array([0.99])

    expected = np.quantile(np.repeat(values, counts), probability)
    actual = _replicated_quantiles(values, counts.astype(np.float64), probability)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("compressed", [True, False])
def test_default_murphy_grid_refuses_unrepresentable_extreme_gaussian_range(
    frequency_case, compressed: bool
) -> None:
    fitted, X, _ = frequency_case
    frame = X.iloc[:2].reset_index(drop=True)
    counts = np.array([1, 66])
    location_offset = np.array([-np.finfo(np.float64).max, np.finfo(np.float64).max])
    response = np.asarray(
        fitted.predict_parameters(frame, offsets={"location": location_offset})[:, 0],
        dtype=np.float64,
    )
    assert np.all(np.isfinite(response))

    if compressed:
        scoring_frame = frame
        scoring_response = response
        scoring_offsets = location_offset
        sample_weight = counts
    else:
        positions = np.repeat(np.arange(len(frame)), counts)
        scoring_frame = frame.iloc[positions].reset_index(drop=True)
        scoring_response = response[positions]
        scoring_offsets = location_offset[positions]
        sample_weight = None

    with pytest.raises(ValueError, match="finite and representable"):
        compare_models(
            fitted,
            fitted,
            scoring_frame,
            scoring_response,
            which="crps",
            sample_weight=sample_weight,
            offsets={"location": scoring_offsets},
            murphy_quantile=0.5,
        )


def test_non_unit_comparison_refuses_candidates_with_different_weight_semantics(
    fit_case, frequency_case
) -> None:
    fitted, X, y = fit_case
    frequency_fitted, _, _ = frequency_case
    frame = X.iloc[:8]
    response = y[:8]

    unit = compare_models(
        fitted,
        frequency_fitted,
        frame,
        response,
        which="crps",
        sample_weight=np.ones(len(frame)),
    )
    assert unit.overall["n"] == len(frame)
    with pytest.raises(UnsupportedLikelihoodContractError, match="weight semantics"):
        compare_models(
            fitted,
            frequency_fitted,
            frame,
            response,
            which="crps",
            sample_weight=np.full(len(frame), 2),
        )
    with pytest.raises(UnsupportedLikelihoodContractError, match="weight semantics"):
        compare_models(
            fitted,
            frequency_fitted,
            frame.iloc[:2],
            response[:2],
            which="crps",
            sample_weight=np.array([0.5, 1.5]),
        )


def test_comparison_refuses_mismatched_rows_and_unknown_scores(fit_case, misfit_case) -> None:
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case
    with pytest.raises(ValueError, match="unknown score"):
        compare_models(fitted, misfitted, X, y, which="brier")
    with pytest.raises(ValueError, match="one segment label per row"):
        compare_models(fitted, misfitted, X, y, by=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="unknown segment column"):
        compare_models(fitted, misfitted, X, y, by="absent")


def test_murphy_scores_are_non_negative_and_integrate_to_the_pinball_loss() -> None:
    rng = np.random.default_rng(21)
    n = 500
    y = rng.normal(0.0, 1.0, n)
    a_quantiles = y + rng.normal(0.0, 0.3, n)
    b_quantiles = rng.normal(0.0, 1.0, n)
    level = 0.75
    thresholds = np.linspace(-6.0, 6.0, 4001)

    payload = murphy_diagram(a_quantiles, b_quantiles, y, level=level, thresholds=thresholds)
    assert isinstance(payload, MurphyPayload)
    assert payload.kind == "murphy"
    assert payload.a.shape == payload.b.shape == thresholds.shape
    assert np.all(payload.a >= 0.0) and np.all(payload.b >= 0.0)
    assert np.allclose(payload.difference, payload.a - payload.b)
    assert payload.n_observations == n
    assert payload.a.mean() < payload.b.mean()

    for curve, forecast in ((payload.a, a_quantiles), (payload.b, b_quantiles)):
        pinball = ((y <= forecast).astype(float) - level) * (forecast - y)
        assert abs(integrate.trapezoid(curve, thresholds) - pinball.mean()) / pinball.mean() < 5e-3


def test_murphy_payload_rides_along_with_a_comparison(fit_case, misfit_case) -> None:
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case
    thresholds = np.linspace(float(y.min()), float(y.max()), 41)

    comparison = compare_models(
        fitted, misfitted, X, y, which="crps", murphy_quantile=0.9, thresholds=thresholds
    )
    assert comparison.murphy is not None
    assert comparison.murphy.level == 0.9
    assert np.array_equal(comparison.murphy.thresholds, thresholds)
    assert np.all(comparison.murphy.a >= 0.0)

    default_grid = compare_models(fitted, misfitted, X, y, murphy_quantile=0.9)
    assert default_grid.murphy.thresholds.shape == (101,)
    assert default_grid.murphy.thresholds[0] < default_grid.murphy.thresholds[-1]


def test_murphy_diagram_validates_its_inputs() -> None:
    y = np.zeros(5)
    with pytest.raises(ValueError, match="one forecast per row"):
        murphy_diagram(np.zeros(4), np.zeros(5), y, level=0.5, thresholds=np.zeros(3))
    with pytest.raises(ValueError, match="level"):
        murphy_diagram(np.zeros(5), np.zeros(5), y, level=1.0, thresholds=np.zeros(3))
    with pytest.raises(ValueError, match="at least one threshold"):
        murphy_diagram(np.zeros(5), np.zeros(5), y, level=0.5, thresholds=np.zeros(0))
    with pytest.raises(ValueError, match="at least two rows"):
        murphy_diagram(np.zeros(1), np.zeros(1), np.zeros(1), level=0.5, thresholds=np.zeros(3))
    with pytest.raises(ValueError, match="one value per threshold"):
        MurphyPayload(
            level=0.5,
            thresholds=np.zeros(3),
            a=np.zeros(2),
            b=np.zeros(3),
            difference=np.zeros(3),
            difference_se=np.zeros(3),
            n_observations=5,
        )


def test_a_family_without_a_quantile_refuses_a_murphy_diagram(fit_case) -> None:
    fitted, X, y = fit_case
    stub = SimpleNamespace(family=_QuantileFreeFamily())
    with pytest.raises(NotImplementedError, match="quantile"):
        compare_models(fitted, stub, X, y, murphy_quantile=0.5)


def test_payloads_survive_a_json_round_trip(fit_case, misfit_case) -> None:
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case
    comparison = compare_models(fitted, misfitted, X, y, which="log", by="g", murphy_quantile=0.8)

    encoded = comparison.to_json()
    decoded = json.loads(json.dumps(encoded))
    assert decoded == encoded
    assert decoded["kind"] == "comparison"
    assert decoded["schema_version"] == 1
    assert decoded["score"] == "log"
    assert set(decoded["overall"]) == {"mean_diff", "se", "t", "n"}
    assert [row["segment"] for row in decoded["by_segment"]] == ["a", "b", "c"]
    assert len(decoded["murphy"]["thresholds"]) == len(decoded["murphy"]["a"])

    bare = compare_models(fitted, misfitted, X, y, which="log")
    bare_json = bare.to_json()
    assert bare_json["by_segment"] is None and bare_json["murphy"] is None
    assert json.loads(json.dumps(bare_json)) == bare_json


def test_a_non_finite_score_is_named_rather_than_averaged(fit_case, misfit_case) -> None:
    fitted, X, y = fit_case
    misfitted, _, _ = misfit_case
    broken = np.array(y, dtype=float)
    broken[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        compare_models(fitted, misfitted, X, broken, which="crps")
