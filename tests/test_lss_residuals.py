"""Contract tests for LSS PIT and randomised quantile residuals."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from superglm import Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.residuals import (
    ResidualSet,
    _sample_residuals,
    compute_residuals,
    replication_sample,
    residual_values,
)
from superglm.distributional.weights import UnsupportedLikelihoodContractError


def _simulated(n: int = 1500, seed: int = 20260903) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    location = 0.6 * np.sin(2.4 * x)
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    return pd.DataFrame({"x": x}), location + scale * rng.standard_normal(n)


def _predictors(scale_features: dict | None = None) -> list[Predictor]:
    return [
        Predictor("location", {"x": Spline("cr", k=8)}),
        Predictor("scale", {} if scale_features is None else scale_features),
    ]


@pytest.fixture(scope="module")
def fit_case():
    """The true data-generating process: a smooth location and a smooth scale."""
    X, y = _simulated()
    model = SuperLSS(
        family=GaussianLS(), predictors=_predictors({"x": Spline("cr", k=6)})
    ).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def misspecified_case():
    """A Gaussian with a constant scale fitted to log-normal responses."""
    rng = np.random.default_rng(4242)
    n = 1500
    x = rng.uniform(-1.0, 1.0, n)
    y = np.exp(0.5 * x + 0.8 * rng.standard_normal(n))
    X = pd.DataFrame({"x": x})
    model = SuperLSS(family=GaussianLS(), predictors=_predictors()).fit_reml(X, y)
    return model._require_fitted(), X, y


@pytest.fixture(scope="module")
def frequency_case():
    """A small fit declaring replication weights rather than prior weights."""
    rng = np.random.default_rng(7)
    n = 80
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.4 * x + 0.5 * rng.standard_normal(n)
    X = pd.DataFrame({"x": x})
    counts = rng.integers(1, 4, n).astype(np.float64)
    model = SuperLSS(
        family=GaussianLS(),
        weight_semantics="frequency",
        predictors=[Predictor("location", {"x": Spline("cr", k=4)}), Predictor("scale", {})],
    ).fit_reml(X, y, sample_weight=counts)
    return model._require_fitted(), X, y, counts


class _StubFamily:
    """A Gaussian location-scale stand-in with a switchable weight refusal."""

    def __init__(self, parameters, *, refuse: bool = False) -> None:
        self.parameters = parameters
        self._refuse = refuse

    def bind_likelihood(self, y, weights, observation):
        if self._refuse:
            raise UnsupportedLikelihoodContractError("stub refuses prior likelihood weights")
        return None

    def cdf(self, y, theta):
        return special.ndtr((np.asarray(y, dtype=float) - theta[:, 0]) / theta[:, 1])

    def quantile(self, p, theta):
        return theta[:, 0] + theta[:, 1] * special.ndtri(p)


class _PriorWeightedStubFamily(_StubFamily):
    """The same law with the prior weight scaling the variance, as ``sigma^2 / w``."""

    def cdf_prior_weighted(self, y, theta, weights):
        deviation = np.asarray(y, dtype=float) - theta[:, 0]
        return special.ndtr(deviation * np.sqrt(weights) / theta[:, 1])

    def quantile_prior_weighted(self, p, theta, weights):
        return theta[:, 0] + theta[:, 1] * special.ndtri(p) / np.sqrt(weights)


class _ZeroAtomFamily(_StubFamily):
    """A point mass ``mass`` at zero above an exponential continuous part."""

    mass = 0.3

    def _continuous(self, y, theta):
        return self.mass + (1.0 - self.mass) * -np.expm1(-np.asarray(y, dtype=float) / theta[:, 1])

    def cdf(self, y, theta):
        values = np.asarray(y, dtype=float)
        return np.where(values > 0.0, self._continuous(values, theta), self.mass)

    def cdf_left_limit(self, y, theta, weights=None):
        values = np.asarray(y, dtype=float)
        return np.where(values > 0.0, self._continuous(values, theta), 0.0)


class _NoDistributionFunctionFamily:
    """A family exposing neither ``cdf`` nor ``quantile``."""

    def __init__(self, parameters) -> None:
        self.parameters = parameters


def _shim(fitted, family) -> SimpleNamespace:
    """A fitted-model stand-in whose family is the stub under test."""
    return SimpleNamespace(
        family=family,
        fit_state=fitted.fit_state,
        predict_eta=fitted.predict_eta,
        predict_parameters=fitted.predict_parameters,
    )


def _residual_kwargs(n: int = 4, k: int = 2) -> dict:
    pit = np.linspace(0.1, 0.9, n)
    return {
        "pit": pit,
        "quantile": special.ndtri(pit),
        "theta": np.ones((n, k)),
        "eta": np.zeros((n, k)),
        "y": np.arange(float(n)),
        "weights": np.ones(n),
        "prior_weights": np.ones(n),
        "clipped_rows": 0,
        "randomised_rows": 0,
        "weight_semantics": "prior",
    }


@pytest.mark.parametrize("case", ["frequency_case", "fit_case"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_retained_nonfinite_response_refuses_after_selection(request, case, invalid):
    fitted, X, y, *weights = request.getfixturevalue(case)
    counts = weights[0] if weights else np.ones(len(y))
    bad = y.copy()
    bad[0] = invalid
    with pytest.raises(ValueError, match="finite"):
        compute_residuals(fitted, X, bad, sample_weight=counts)
    counts = counts.copy()
    counts[0] = 0
    assert np.all(np.isfinite(compute_residuals(fitted, X, bad, sample_weight=counts).pit))


def test_frequency_atom_occurrences_are_independent(frequency_case):
    from superglm.distributional import residuals as module

    fitted, X, _, _ = frequency_case
    atom = _shim(fitted, _ZeroAtomFamily(fitted.family.parameters))
    residuals = compute_residuals(atom, X.iloc[:1], np.zeros(1), sample_weight=np.array([5000.0]))
    sample = module._sample_residuals(residuals, seed=7)
    assert len(np.unique(sample.pit)) == 5000
    assert np.all(sample.rows == 0)
    assert np.array_equal(sample.quantile, special.ndtri(sample.pit))
    assert np.all((sample.pit >= 0) & (sample.pit <= 0.3))
    # Hoeffding bound at failure probability 1e-8 for the bounded mean.
    bound = np.sqrt(np.log(2e8) / (2 * len(sample.pit)))
    assert abs(np.mean(sample.pit / 0.3) - 0.5) < bound
    assert np.array_equal(sample.pit, module._sample_residuals(residuals, seed=7).pit)
    assert not np.array_equal(sample.pit, module._sample_residuals(residuals, seed=8).pit)


def _interval_payload(counts):
    kwargs = _residual_kwargs(n=2)
    kwargs.update(
        weights=np.asarray(counts, dtype=float),
        weight_semantics="frequency",
        pit=np.array([0.1, 0.7]),
        quantile=special.ndtri([0.1, 0.7]),
        pit_lower=np.array([0.0, 0.4]),
        pit_upper=np.array([0.2, 1.0]),
        randomised_rows=2,
        eta=np.array([[10.0], [20.0]]),
        theta=np.array([[100.0], [200.0]]),
    )
    return ResidualSet(**kwargs)


def test_capped_atom_mixture_has_conditional_uniform_draws_and_alignment():
    residuals = _interval_payload([200000, 800000])
    sample = _sample_residuals(residuals, max_rows=20000, seed=19)
    assert len(sample.rows) == 20000
    np.testing.assert_array_equal(
        sample.rows, replication_sample(residuals, max_rows=20000, seed=19)
    )
    np.testing.assert_array_equal(residuals.eta[sample.rows, 0], 10 * (1 + sample.rows))
    np.testing.assert_array_equal(residuals.theta[sample.rows, 0], 100 * (1 + sample.rows))
    normalized = (sample.pit - residuals.pit_lower[sample.rows]) / (
        residuals.pit_upper[sample.rows] - residuals.pit_lower[sample.rows]
    )
    # Ten conditional bin indicators plus two bounded conditional means;
    # Hoeffding with a union bound gives failure probability at most 1e-8.
    for row in range(2):
        values = normalized[sample.rows == row]
        bound = np.sqrt(np.log(24e8) / (2 * len(values)))
        assert abs(values.mean() - 0.5) < bound
        assert np.all(
            np.abs(np.histogram(values, bins=5, range=(0, 1))[0] / len(values) - 0.2) < bound
        )
    # Changing the stored physical random draws cannot alter occurrence draws.
    changed = replace(residuals, pit=np.array([0.15, 0.8]), quantile=special.ndtri([0.15, 0.8]))
    np.testing.assert_array_equal(
        _sample_residuals(changed, max_rows=20000, seed=19).pit, sample.pit
    )


def test_conditional_uniformity_regression_detects_restarted_selection_stream(monkeypatch):
    from superglm.distributional import residuals as module

    monkeypatch.setattr(module, "_residual_rng", lambda seed, domain: np.random.default_rng(seed))
    with pytest.raises(AssertionError):
        test_capped_atom_mixture_has_conditional_uniform_draws_and_alignment()


def test_integer_atom_sampling_matches_literal_expansion_in_distribution(frequency_case):
    fitted, X, _, _ = frequency_case
    atom = _shim(fitted, _ZeroAtomFamily(fitted.family.parameters))
    X = X.iloc[:2]
    counts = np.array([30.0, 70.0])
    index = np.repeat(np.arange(2), counts.astype(int))
    compressed = compute_residuals(atom, X, np.zeros(2), sample_weight=counts)
    rates = [[], []]
    for seed in range(100):
        expanded = compute_residuals(atom, X.iloc[index], np.zeros(100), seed=seed)
        sample = _sample_residuals(compressed, seed=seed)
        np.testing.assert_array_equal(sample.rows, index)
        rates[0].append(np.mean(sample.pit < 0.15))
        rates[1].append(np.mean(expanded.pit < 0.15))
    # The count is Binomial(100, .5): variance of the rate is .0025.
    # The variance estimator bound uses its exact fourth central moment.
    expected_variance = 0.25 / 100
    fourth_moment = 3 * expected_variance**2 - 0.125 / 100**3
    variance_se = np.sqrt((fourth_moment - 97 / 99 * expected_variance**2) / 100)
    for values in rates:
        assert abs(np.mean(values) - 0.5) < 6 * np.sqrt(expected_variance / 100)
        assert abs(np.var(values, ddof=1) - expected_variance) < 6 * variance_se


def test_sampler_preserves_continuous_unit_and_prior_values(frequency_case, fit_case):
    fitted, X, y, counts = frequency_case
    continuous = compute_residuals(fitted, X, y, sample_weight=counts)
    sample = _sample_residuals(continuous)
    np.testing.assert_array_equal(sample.pit, continuous.pit[sample.rows])
    np.testing.assert_array_equal(sample.quantile, continuous.quantile[sample.rows])
    for case in (frequency_case, fit_case):
        fitted, X, *_ = case
        atom = _shim(fitted, _ZeroAtomFamily(fitted.family.parameters))
        unit = compute_residuals(atom, X, np.zeros(len(X)), seed=9)
        sample = _sample_residuals(unit, max_rows=20, seed=7)
        np.testing.assert_array_equal(sample.pit, unit.pit[sample.rows])
    manual = ResidualSet(**_residual_kwargs())
    assert len(_sample_residuals(manual).rows) == 4
    missing = replace(_interval_payload([2, 3]), pit_lower=None, pit_upper=None)
    with pytest.raises(ValueError, match="CDF intervals"):
        _sample_residuals(missing)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (None, [1, 1]),
        ([0], [1, 1]),
        ([0, np.nan], [1, 1]),
        ([0, 0], [1, np.inf]),
        ([-0.1, 0], [1, 1]),
        ([0, 0], [1.1, 1]),
        ([0.5, 0], [0.1, 1]),
    ],
)
def test_residual_intervals_validate_paired_finite_ordered_shapes(lower, upper):
    with pytest.raises(ValueError, match="together|CDF intervals"):
        replace(_interval_payload([2, 3]), pit_lower=lower, pit_upper=upper)


def test_intervals_are_owned_physical_arrays():
    lower, upper = np.zeros(2), np.ones(2)
    residuals = replace(_interval_payload([2, 3]), pit_lower=lower, pit_upper=upper)
    lower[:] = 0.25
    upper[:] = 0.75
    np.testing.assert_array_equal(residuals.pit_lower, [0, 0])
    np.testing.assert_array_equal(residuals.pit_upper, [1, 1])
    assert not residuals.pit_lower.flags.writeable
    assert not residuals.pit_upper.flags.writeable
    assert len(residuals.to_json()["pit_lower"]) == 2


def test_frequency_below_support_response_remains_a_scoring_input(frequency_case):
    fitted, X, *_ = frequency_case
    stub = _shim(fitted, GammaLS())
    stub.predict_eta = lambda frame, offsets=None: np.zeros((len(frame), 2))
    stub.predict_parameters = lambda frame, offsets=None: np.ones((len(frame), 2))
    residuals = compute_residuals(stub, X.iloc[:1], np.array([-1.0]))
    assert residuals.pit[0] == 1e-12
    assert residuals.clipped_rows == 1


def test_the_true_model_gives_a_uniform_unclipped_pit(fit_case) -> None:
    fitted, X, y = fit_case
    residuals = compute_residuals(fitted, X, y)

    assert isinstance(residuals, ResidualSet)
    assert residuals.pit.shape == (len(y),)
    assert residuals.theta.shape == (len(y), 2)
    assert residuals.eta.shape == (len(y), 2)
    assert not residuals.pit.flags.writeable
    assert not residuals.theta.flags.writeable

    assert stats.kstest(residuals.pit, "uniform").pvalue > 0.01
    assert abs(float(residuals.pit.mean()) - 0.5) < 0.05
    assert np.array_equal(residuals.quantile, stats.norm.ppf(residuals.pit))
    assert residuals.clipped_rows == 0
    assert residuals.randomised_rows == 0
    assert residuals.weight_semantics == "prior"
    assert np.array_equal(residuals.weights, np.ones(len(y)))
    assert np.array_equal(residuals.theta, fitted.predict_parameters(X))
    assert np.array_equal(residuals.eta, fitted.predict_eta(X))
    assert np.array_equal(compute_residuals(fitted, X, y).pit, residuals.pit)


def test_a_misspecified_model_shows_in_the_pit(misspecified_case) -> None:
    fitted, X, y = misspecified_case
    residuals = compute_residuals(fitted, X, y)

    assert stats.kstest(residuals.pit, "uniform").pvalue < 1.0e-3
    assert residuals.clipped_rows > 0
    assert np.all(residuals.pit >= 1.0e-12)
    assert np.all(residuals.pit <= 1.0 - 1.0e-12)
    assert np.all(np.isfinite(residuals.quantile))


def test_residual_values_returns_the_named_kind(fit_case) -> None:
    fitted, X, y = fit_case
    residuals = compute_residuals(fitted, X, y)

    assert np.array_equal(residual_values(fitted, X, y, kind="pit"), residuals.pit)
    assert np.array_equal(residual_values(fitted, X, y), residuals.quantile)
    with pytest.raises(ValueError, match="kind must be"):
        residual_values(fitted, X, y, kind="deviance")
    with pytest.raises(ValueError, match="kind must be"):
        compute_residuals(fitted, X, y, kind="deviance")


def test_zero_prior_weights_leave_the_diagnostics(fit_case) -> None:
    fitted, X, y = fit_case
    weights = np.ones(len(y))
    weights[:5] = 0.0
    offsets = {"location": np.full(len(y), 0.25)}

    residuals = compute_residuals(fitted, X, y, sample_weight=weights, offsets=offsets)

    assert residuals.pit.shape == (len(y) - 5,)
    assert residuals.y.shape == (len(y) - 5,)
    assert np.array_equal(residuals.y, y[5:])
    retained_offsets = {"location": offsets["location"][5:]}
    expected = fitted.predict_parameters(X.iloc[5:], offsets=retained_offsets)
    assert np.array_equal(residuals.theta, expected)


def test_frequency_weights_replicate_rows_literally(frequency_case) -> None:
    fitted, X, y, counts = frequency_case
    residuals = compute_residuals(fitted, X, y, sample_weight=counts)

    assert residuals.weight_semantics == "frequency"
    assert np.array_equal(residuals.weights, counts)

    rows = replication_sample(residuals)
    assert rows.dtype == np.intp
    multiplicity = np.bincount(rows, minlength=len(counts))
    assert np.array_equal(multiplicity, counts.astype(np.intp))

    capped = replication_sample(residuals, max_rows=50)
    assert capped.shape == (50,)
    assert np.array_equal(capped, replication_sample(residuals, max_rows=50))
    assert not np.array_equal(capped, replication_sample(residuals, max_rows=50, seed=7))
    with pytest.raises(ValueError, match="max_rows"):
        replication_sample(residuals, max_rows=0)


def test_prior_weights_replicate_no_rows(fit_case) -> None:
    fitted, X, y = fit_case
    residuals = compute_residuals(fitted, X, y)
    assert np.array_equal(replication_sample(residuals), np.arange(len(y)))


def test_fractional_replication_weights_are_resampled() -> None:
    kwargs = _residual_kwargs()
    kwargs["weights"] = np.array([0.5, 1.5, 2.5, 3.5])
    kwargs["weight_semantics"] = "frequency"
    rows = replication_sample(ResidualSet(**kwargs), seed=3)

    assert rows.shape == (4,)
    assert np.all(rows >= 0) and np.all(rows < 4)
    with pytest.raises(TypeError, match="ResidualSet"):
        replication_sample(kwargs)


def test_a_family_that_refuses_prior_weights_propagates_its_error(fit_case) -> None:
    fitted, X, y = fit_case
    stub = _shim(fitted, _StubFamily(fitted.family.parameters, refuse=True))
    weights = np.full(len(y), 2.0)

    with pytest.raises(UnsupportedLikelihoodContractError, match="stub refuses"):
        compute_residuals(stub, X, y, sample_weight=weights)


def test_prior_weights_enter_the_row_law_when_the_family_owns_one(fit_case) -> None:
    fitted, X, y = fit_case
    rng = np.random.default_rng(11)
    weights = rng.uniform(0.5, 3.0, len(y))
    theta = np.asarray(fitted.predict_parameters(X))
    expected = special.ndtr((y - theta[:, 0]) * np.sqrt(weights) / theta[:, 1])

    weighted = _shim(fitted, _PriorWeightedStubFamily(fitted.family.parameters))
    residuals = compute_residuals(weighted, X, y, sample_weight=weights)
    assert np.array_equal(residuals.pit, expected)
    assert np.array_equal(residuals.weights, np.ones(len(y)))

    unit = compute_residuals(weighted, X, y)
    assert np.array_equal(unit.pit, special.ndtr((y - theta[:, 0]) / theta[:, 1]))

    plain = _shim(fitted, _StubFamily(fitted.family.parameters))
    with pytest.raises(NotImplementedError, match="_StubFamily"):
        compute_residuals(plain, X, y, sample_weight=weights)


def test_prior_weights_travel_with_the_residuals(fit_case, frequency_case) -> None:
    """The row law's weights ride along so a builder can hand them to the primitive."""
    fitted, X, y = fit_case
    rng = np.random.default_rng(11)
    weights = rng.uniform(0.5, 3.0, len(y))
    weighted = _shim(fitted, _PriorWeightedStubFamily(fitted.family.parameters))

    residuals = compute_residuals(weighted, X, y, sample_weight=weights)
    assert np.array_equal(residuals.prior_weights, weights)
    assert np.array_equal(residuals.weights, np.ones(len(y)))
    assert not residuals.prior_weights.flags.writeable

    # Zero-weight rows leave the diagnostics, so what travels is the retained
    # rows' own weights and not the vector the caller passed.
    dropped = weights.copy()
    dropped[:5] = 0.0
    retained = compute_residuals(weighted, X, y, sample_weight=dropped)
    assert np.array_equal(retained.prior_weights, dropped[5:])

    unit = compute_residuals(fitted, X, y)
    assert np.array_equal(unit.prior_weights, np.ones(len(y)))

    # Under the frequency contract the weight is replication, never a law, so
    # every row's prior weight is one and ``weights`` carries the counts.
    counted, count_X, count_y, counts = frequency_case
    replicated = compute_residuals(counted, count_X, count_y, sample_weight=counts)
    assert np.array_equal(replicated.prior_weights, np.ones(len(counts)))
    assert np.array_equal(replicated.weights, counts)

    payload = residuals.to_json()
    assert payload["prior_weights"][:4] == [float(value) for value in weights[:4]]
    assert json.loads(json.dumps(payload)) == payload


def test_atom_rows_get_a_randomised_pit(fit_case) -> None:
    fitted, X, _ = fit_case
    rng = np.random.default_rng(5)
    n = len(X)
    atoms = rng.random(n) < 0.25
    y = np.where(atoms, 0.0, rng.exponential(0.5, n) + 1.0e-6)
    family = _ZeroAtomFamily(fitted.family.parameters)
    stub = _shim(fitted, family)

    residuals = compute_residuals(stub, X, y, seed=3)
    theta = np.asarray(fitted.predict_parameters(X))

    assert residuals.randomised_rows == int(np.count_nonzero(atoms))
    assert np.all(residuals.pit[atoms] > 0.0)
    assert np.all(residuals.pit[atoms] < family.mass)
    assert np.array_equal(residuals.pit[~atoms], family.cdf(y, theta)[~atoms])
    assert np.array_equal(compute_residuals(stub, X, y, seed=3).pit, residuals.pit)
    assert not np.array_equal(compute_residuals(stub, X, y, seed=4).pit, residuals.pit)


def test_a_family_without_a_distribution_function_refuses(fit_case) -> None:
    fitted, X, y = fit_case
    stub = _shim(fitted, _NoDistributionFunctionFamily(fitted.family.parameters))
    with pytest.raises(NotImplementedError, match="distribution function"):
        compute_residuals(stub, X, y)


def test_to_json_is_plain_data(fit_case) -> None:
    fitted, X, y = fit_case
    residuals = compute_residuals(fitted, X.iloc[:6], y[:6])
    payload = residuals.to_json()

    assert payload["n_rows"] == 6
    assert payload["weight_semantics"] == "prior"
    assert payload["clipped_rows"] == 0
    assert isinstance(payload["theta"][0], list)
    assert json.loads(json.dumps(payload)) == payload

    kwargs = _residual_kwargs()
    kwargs["pit"] = np.array([np.nan, 0.2, 0.4, 0.6])
    kwargs["quantile"] = special.ndtri(kwargs["pit"])
    nan_payload = ResidualSet(**kwargs).to_json()
    assert nan_payload["pit"][0] is None
    assert nan_payload["quantile"][0] is None
    assert json.loads(json.dumps(nan_payload)) == nan_payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pit", np.ones((2, 2)), "one value per row"),
        ("y", np.ones(3), "one value per row"),
        ("weights", np.zeros(4), "finite and positive"),
        ("prior_weights", np.zeros(4), "finite and positive"),
        ("prior_weights", np.ones(3), "one value per row"),
        ("theta", np.ones(4), "parameter matrix"),
        ("eta", np.ones((3, 2)), "parameter matrix"),
        ("clipped_rows", -1, "row counts"),
        ("randomised_rows", 9, "row counts"),
        ("weight_semantics", "bogus", "weight_semantics must be"),
    ],
)
def test_residual_set_validates_its_payload(field, value, message) -> None:
    kwargs = _residual_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        ResidualSet(**kwargs)
