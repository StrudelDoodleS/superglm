"""Tests for superglm.stats.model_tests — model adequacy tests (T13-T26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, Spline, SuperGLM
from superglm._roundoff import UNIT_ROUNDOFF, gamma
from superglm.distributions import Gamma, NegativeBinomial
from superglm.stats.model_tests import (
    DispersionTestResult,
    ScoreTestZIResult,
    VuongTestResult,
    ZeroInflationResult,
    dispersion_test,
    score_test_zi,
    vuong_test,
    zero_inflation_index,
)

# ── Helpers ──────────────────────────────────────────────────────


def _fit_poisson(X, y, **kwargs):
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=5)},
        selection_penalty=0.0,
        **kwargs,
    )
    model.fit(X, y)
    return model


def _fit_poisson_numeric(X, y, features=None, **kwargs):
    if features is None:
        features = {col: Numeric() for col in X.columns}
    model = SuperGLM(
        family="poisson",
        features=features,
        selection_penalty=0.0,
        **kwargs,
    )
    model.fit(X, y)
    return model


# ── T13: zero_inflation_index — Poisson ──────────────────────────


class TestZeroInflationIndexPoisson:
    """T13: ZI index for Poisson."""

    def test_expected_zeros_analytical(self):
        rng = np.random.default_rng(42)
        n = 5000
        mu = rng.uniform(0.5, 3.0, n)
        y = rng.poisson(mu)
        result = zero_inflation_index(y, mu, family="poisson")
        assert isinstance(result, ZeroInflationResult)
        # Expected zeros should match analytical value
        analytical = np.sum(np.exp(-mu))
        assert abs(result.expected_zeros - analytical) < 1e-6

    def test_no_zero_inflation_ratio_near_one(self):
        rng = np.random.default_rng(42)
        n = 10000
        mu = np.full(n, 2.0)
        y = rng.poisson(mu)
        result = zero_inflation_index(y, mu, family="poisson")
        # Ratio should be near 1.0 for non-zero-inflated data
        assert abs(result.ratio - 1.0) < 0.15


# ── T14: zero_inflation_index — NB2 ─────────────────────────────


class TestZeroInflationIndexNB2:
    """T14: ZI index for NB2."""

    def test_expected_zeros_nb2(self):
        rng = np.random.default_rng(42)
        n = 5000
        theta = 2.0
        mu = rng.uniform(0.5, 3.0, n)
        y = rng.negative_binomial(theta, theta / (theta + mu))
        result = zero_inflation_index(y, mu, family="nb2", theta=theta)
        assert isinstance(result, ZeroInflationResult)
        # Expected zeros should match NB2 formula
        analytical = np.sum((theta / (theta + mu)) ** theta)
        assert abs(result.expected_zeros - analytical) < 1e-6


# ── T15: zero_inflation_index — unsupported family ───────────────


class TestZeroInflationIndexUnsupported:
    """T15: ValueError for unsupported families."""

    def test_gaussian_raises(self):
        with pytest.raises(ValueError, match="'poisson' or 'nb2'"):
            zero_inflation_index(np.array([1, 2, 3]), np.array([1, 2, 3]), family="gaussian")

    def test_gamma_raises(self):
        with pytest.raises(ValueError, match="'poisson' or 'nb2'"):
            zero_inflation_index(np.array([1, 2, 3]), np.array([1, 2, 3]), family="gamma")


# ── T16: score_test_zi — calibration under H0 ────────────────────


class TestScoreTestZIH0:
    """T16: Under H0 (pure Poisson), p-value should be large."""

    def test_pure_poisson_not_rejected(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        result = score_test_zi(model, X, y)
        assert isinstance(result, ScoreTestZIResult)
        assert result.p_value > 0.01

    def test_direction_is_valid(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        result = score_test_zi(model, X, y)
        assert result.direction in ("inflated", "deflated")


# ── T17: score_test_zi — power under H1 ──────────────────────────


class TestScoreTestZIH1:
    """T17: Under H1 (zero-inflated Poisson), test should reject."""

    def test_zero_inflated_detected(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        # Inject 30% structural zeros
        zero_mask = rng.random(n) < 0.3
        y[zero_mask] = 0.0
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        result = score_test_zi(model, X, y)
        assert result.p_value < 0.05
        assert result.direction == "inflated"


# ── T18: score_test_zi — non-Poisson family ──────────────────────


class TestScoreTestZINonPoisson:
    """T18: ValueError for non-Poisson family."""

    def test_gamma_raises(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 5, n)
        mu = np.exp(1.0 + 0.2 * x)
        y = rng.gamma(5.0, mu / 5.0, n)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(family=Gamma(), features={"x": Numeric()}, selection_penalty=0.0)
        model.fit(X, y)
        with pytest.raises(ValueError, match="requires family"):
            score_test_zi(model, X, y)


# ── T18b: score_test_zi — sample_weight raises ────────────────────


class TestScoreTestZIWeights:
    """sample_weight should raise NotImplementedError."""

    def test_weights_raise(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        w = np.ones(n)
        with pytest.raises(NotImplementedError, match="sample_weight"):
            score_test_zi(model, X, y, sample_weight=w)


# ── T19: dispersion_test — equidispersed Poisson ─────────────────


class TestDispersionTestEquidispersed:
    """T19: Equidispersed Poisson data should not be rejected."""

    def test_equidispersed_not_rejected(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        result = dispersion_test(model, X, y)
        assert isinstance(result, DispersionTestResult)
        assert result.p_value > 0.01
        assert abs(result.alpha_hat) < 0.5


# ── T20: dispersion_test — overdispersed data ────────────────────


class TestDispersionTestOverdispersed:
    """T20: Overdispersed NB2 data fitted with Poisson should be detected."""

    def test_overdispersion_detected(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x)
        theta = 1.0  # moderate overdispersion
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        result = dispersion_test(model, X, y)
        assert result.p_value < 0.05
        assert result.alpha_hat > 0


# ── T21: dispersion_test — alternative parameter ─────────────────


class TestDispersionTestAlternative:
    """T21: alternative parameter produces valid results."""

    @pytest.fixture
    def fitted(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        return model, X, y

    @pytest.mark.parametrize("alt", ["greater", "less", "two-sided"])
    def test_alternatives(self, fitted, alt):
        model, X, y = fitted
        result = dispersion_test(model, X, y, alternative=alt)
        assert isinstance(result, DispersionTestResult)
        assert 0 <= result.p_value <= 1
        assert result.alternative == alt


# ── T21b: dispersion_test — sample_weight raises ──────────────────


class TestDispersionTestWeights:
    """sample_weight should raise NotImplementedError."""

    def test_weights_raise(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        w = np.ones(n)
        with pytest.raises(NotImplementedError, match="sample_weight"):
            dispersion_test(model, X, y, sample_weight=w)


# ── T22: vuong_test — nested models ──────────────────────────────


class TestVuongTestNested:
    """T22: Vuong test with full vs reduced model."""

    def test_nested_models_valid(self):
        rng = np.random.default_rng(42)
        n = 1000
        x1 = rng.uniform(0, 5, n)
        x2 = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x1 + 0.1 * x2)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        # Full model
        model_full = _fit_poisson_numeric(X, y, features={"x1": Numeric(), "x2": Numeric()})
        # Reduced model
        model_reduced = _fit_poisson_numeric(X, y, features={"x1": Numeric()})
        result = vuong_test(model_full, model_reduced, X, y)
        assert isinstance(result, VuongTestResult)
        assert np.isfinite(result.statistic)
        assert 0 <= result.p_value <= 1
        assert result.preferred in ("model_a", "model_b", "indistinguishable")


# ── T23: vuong_test — identical models ───────────────────────────


class TestVuongTestIdentical:
    """T23: Same model vs itself → statistic ≈ 0."""

    def test_self_comparison(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = _fit_poisson(X, y)
        result = vuong_test(model, model, X, y, correction="none")
        assert abs(result.statistic) < 1e-6
        assert result.preferred == "indistinguishable"


# ── T24: vuong_test — corrections ────────────────────────────────


class TestVuongTestCorrections:
    """T24: All three corrections produce valid results."""

    @pytest.fixture
    def two_models(self):
        rng = np.random.default_rng(42)
        n = 1000
        x1 = rng.uniform(0, 5, n)
        x2 = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x1 + 0.1 * x2)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        model_full = _fit_poisson_numeric(X, y, features={"x1": Numeric(), "x2": Numeric()})
        model_reduced = _fit_poisson_numeric(X, y, features={"x1": Numeric()})
        return model_full, model_reduced, X, y

    @pytest.mark.parametrize("corr", ["none", "aic", "bic"])
    def test_corrections_valid(self, two_models, corr):
        model_full, model_reduced, X, y = two_models
        result = vuong_test(model_full, model_reduced, X, y, correction=corr)
        assert isinstance(result, VuongTestResult)
        assert np.isfinite(result.statistic)
        assert 0 <= result.p_value <= 1
        assert result.correction == corr

    def test_corrections_differ_for_different_complexity(self, two_models):
        model_full, model_reduced, X, y = two_models
        r_none = vuong_test(model_full, model_reduced, X, y, correction="none")
        r_aic = vuong_test(model_full, model_reduced, X, y, correction="aic")
        r_bic = vuong_test(model_full, model_reduced, X, y, correction="bic")
        # The corrections should shift the statistic when models differ in complexity
        stats_set = {
            round(r_none.statistic, 6),
            round(r_aic.statistic, 6),
            round(r_bic.statistic, 6),
        }
        assert len(stats_set) >= 2  # at least some differ


# ── T25: vuong_test — different families ─────────────────────────


class TestVuongTestDifferentFamilies:
    """T25: Poisson vs NB2 on overdispersed data → NB2 preferred."""

    def test_nb2_preferred(self):
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x)
        theta = 1.0
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
        X = pd.DataFrame({"x": x})
        model_pois = SuperGLM(
            family="poisson", features={"x": Spline(n_knots=5)}, selection_penalty=0.0
        )
        model_pois.fit(X, y)
        model_nb2 = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            features={"x": Spline(n_knots=5)},
            selection_penalty=0.0,
        )
        model_nb2.fit(X, y)
        result = vuong_test(model_pois, model_nb2, X, y, correction="aic")
        # NB2 should be preferred (model_b) since data is overdispersed
        assert result.preferred == "model_b"


# ── T26: vuong_test — mismatched data ────────────────────────────


class TestVuongTestMismatchedData:
    """T26: Models fitted on different data but evaluated on the same test data."""

    def test_mismatched_data_works(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        y_train = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})

        # Fit two models on different random samples
        model_a = _fit_poisson_numeric(X, y_train, features={"x": Numeric()})

        y_train_b = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        model_b = _fit_poisson_numeric(X, y_train_b, features={"x": Numeric()})

        # Evaluate on new test data
        y_test = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        result = vuong_test(model_a, model_b, X, y_test, correction="none")
        assert isinstance(result, VuongTestResult)
        assert np.isfinite(result.statistic)


# ── T27: vuong_test — sample_weight ──────────────────────────────


class TestVuongTestWeighted:
    """Vuong test with sample_weight produces valid, different results."""

    def test_weighted_vs_unweighted(self):
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 5, n)
        x2 = rng.uniform(0, 5, n)
        mu = np.exp(0.5 + 0.3 * x1 + 0.1 * x2)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        w = rng.uniform(0.1, 5.0, n)

        # Different feature sets so LL differs per observation
        model_a = _fit_poisson_numeric(X, y, features={"x1": Numeric(), "x2": Numeric()})
        model_b = _fit_poisson_numeric(X[["x1"]], y, features={"x1": Numeric()})

        r_uw = vuong_test(model_a, model_b, X, y, correction="none")
        r_w = vuong_test(model_a, model_b, X, y, sample_weight=w, correction="none")

        assert isinstance(r_w, VuongTestResult)
        assert np.isfinite(r_w.statistic)
        assert 0 <= r_w.p_value <= 1
        # Weighted reweights the LL diffs, so statistic should differ
        assert abs(r_w.statistic - r_uw.statistic) > 1e-6

    def test_uniform_weight_matches_unweighted(self):
        rng = np.random.default_rng(99)
        n = 300
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})

        m1 = _fit_poisson_numeric(X, y, features={"x": Numeric()})
        m2 = _fit_poisson_numeric(X, y, features={"x": Numeric()})

        r_uw = vuong_test(m1, m2, X, y, correction="none")
        r_w = vuong_test(m1, m2, X, y, sample_weight=np.ones(n), correction="none")

        # Uniform weights should give identical results
        assert abs(r_w.statistic - r_uw.statistic) < 1e-10
        assert abs(r_w.omega - r_uw.omega) < 1e-10


class TestVuongTestNB2PoissonLimit:
    """Review of #614: the unweighted NB2 rows carried ``theta u`` near the Poisson limit.

    ``gammaln(y + theta) - gammaln(theta)`` subtracts two values of about
    ``theta log theta`` and ``theta log(theta / (mu + theta))`` takes the log of a
    quotient near one. On Poisson data, NB2 at theta = 1e8 and Poisson are the same
    model to about 1e-8 per row, yet the unweighted test said "Poisson preferred"
    with p = 0 (statistic -31.9), and at theta = 1e12 "NB preferred" (+58.3), while
    ``sample_weight=np.ones(n)`` gave p = 0.32 through the family's stable density.
    """

    def test_per_row_log_density_matches_an_80_digit_reference(self):
        from superglm.stats.model_tests import _per_obs_ll_nb2
        from tests._nb2_oracle import log_density

        y = np.array([0.0, 1.0, 2.0, 7.0, 30.0, 0.5])
        mu = np.array([0.3, 1.2, 2.0, 5.5, 25.0, 0.8])
        for theta in (1.7, 1e8, 1e12):
            actual = _per_obs_ll_nb2(y, mu, theta)
            expected = np.array([log_density(a, b, theta) for a, b in zip(y, mu, strict=True)])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("theta", [1e8, 1e12])
    def test_unweighted_and_unit_weighted_tests_agree(self, theta):
        rng = np.random.default_rng(42)
        n = 4000
        x1 = rng.uniform(0, 2, n)
        x2 = rng.uniform(0, 2, n)
        y = rng.poisson(np.exp(-0.5 + 0.3 * x1 + 0.2 * x2)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        features = {"x1": Numeric(), "x2": Numeric()}
        poisson = _fit_poisson_numeric(X, y, features=dict(features))
        nb = SuperGLM(
            family=NegativeBinomial(theta=theta), features=dict(features), selection_penalty=0.0
        )
        nb.fit(X, y)

        unweighted = vuong_test(nb, poisson, X, y, correction="none")
        unit = vuong_test(nb, poisson, X, y, sample_weight=np.ones(n), correction="none")
        assert unweighted.p_value > 0.05
        assert unweighted.preferred == unit.preferred
        np.testing.assert_allclose(unweighted.statistic, unit.statistic, rtol=1e-6, atol=1e-6)


class TestVuongTestUnits:
    @pytest.mark.parametrize("unit", [1e-12, 1e9])
    def test_statistic_ignores_the_units_of_the_log_likelihood(self, monkeypatch, unit):
        """Rescaling the per-row log-likelihoods (by weights) leaves the test unchanged."""
        from superglm.stats import model_tests

        rng = np.random.default_rng(7)
        n = 1000
        x1 = rng.uniform(0, 5, n)
        x2 = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x1 + 0.1 * x2)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        full = _fit_poisson_numeric(X, y, features={"x1": Numeric(), "x2": Numeric()})
        reduced = _fit_poisson_numeric(X, y, features={"x1": Numeric()})
        reference = vuong_test(full, reduced, X, y, correction="none")
        real_rows = model_tests._per_obs_log_likelihood
        monkeypatch.setattr(
            model_tests, "_per_obs_log_likelihood", lambda *a, **k: unit * real_rows(*a, **k)
        )

        rescaled = vuong_test(full, reduced, X, y, correction="none")

        assert rescaled.statistic == pytest.approx(reference.statistic, rel=1e-9)
        assert rescaled.preferred == reference.preferred

    @pytest.mark.parametrize("units", [1e-30, 1e30])
    def test_identical_models_threshold_ignores_a_change_of_response_units(
        self, monkeypatch, units
    ):
        """A change of units adds -log(units) to every row's log-likelihood of both models."""
        from superglm.stats import model_tests

        rng = np.random.default_rng(7)
        n = 1000
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model_a = _fit_poisson_numeric(X, y)
        model_b = _fit_poisson_numeric(X, y)
        real_rows = model_tests._per_obs_log_likelihood
        # model_b differs from model_a by a per-row amount far below the rows' spread.
        nudge = 1e-9 * (1.0 + rng.normal(size=n))
        # Shifting a row rounds its difference by at most 2u(|ll| + |shift|), a
        # fraction of the nudge that bounds the change in its mean and spread.
        largest = float(np.max(np.abs(real_rows(model_a, X, y, None)))) + abs(np.log(units))
        rel = 4.0 * 2.0 * UNIT_ROUNDOFF * largest / 1e-9

        def rows(shift):
            def per_obs(model, *args, **kwargs):
                ll = real_rows(model, *args, **kwargs) + shift
                return ll + nudge if model is model_b else ll

            return per_obs

        monkeypatch.setattr(model_tests, "_per_obs_log_likelihood", rows(0.0))
        reference = vuong_test(model_a, model_b, X, y, correction="none")
        monkeypatch.setattr(model_tests, "_per_obs_log_likelihood", rows(-np.log(units)))
        shifted = vuong_test(model_a, model_b, X, y, correction="none")

        assert reference.preferred == "model_b"
        assert shifted.statistic == pytest.approx(reference.statistic, rel=rel)
        assert shifted.preferred == reference.preferred

    def test_a_row_without_frequency_weight_does_not_set_the_threshold(self):
        """A zero frequency weight is no replication: the row enters no moment."""
        rng = np.random.default_rng(7)
        n = 500
        x, z, noise = rng.normal(size=(3, n))
        y = 2.0 + 3.0 * x + noise
        X = pd.DataFrame({"x": x, "z": z})
        common = dict(family="gaussian", selection_penalty=0.0, weight_semantics="frequency")
        model_x = SuperGLM(features={"x": Numeric()}, **common).fit(X[["x"]], y)
        model_z = SuperGLM(features={"z": Numeric()}, **common).fit(X[["z"]], y)
        reference = vuong_test(model_x, model_z, X, y, sample_weight=np.ones(n))

        padded = vuong_test(
            model_x,
            model_z,
            pd.concat([X, X.iloc[[0]]], ignore_index=True),
            np.append(y, 1e8),
            sample_weight=np.append(np.ones(n), 0.0),
        )

        assert reference.preferred == "model_a"
        # The zero-weight row adds exact zeros to each sum: a reordering's gamma_n at most.
        assert padded.statistic == pytest.approx(reference.statistic, rel=4 * gamma(n + 1))
        assert padded.preferred == reference.preferred


# ── Statistics against independent references ─────────────────


def _cancellation(terms):
    """sum |t| / |sum t|: how much a sum's relative rounding error is magnified."""
    terms = np.asarray(terms, dtype=np.float64)
    return float(np.sum(np.abs(terms)) / abs(np.sum(terms)))


class TestStatisticsMatchReferences:
    """The statistics a user reads before switching to a zero-inflated or negative
    binomial model, each against a reference the test computes itself.

    Sums of n terms carry relative error about n u times their cancellation factor
    (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., section 4.2),
    so each tolerance is built from those factors. A p-value is compared absolutely:
    1 - F(t) in float64 is accurate to about u, not relative to a tiny p.
    """

    @pytest.fixture
    def poisson_case(self):
        rng = np.random.default_rng(20261125)
        n = 400
        x = rng.uniform(0.0, 2.0, n)
        y = rng.poisson(np.exp(-0.3 + 0.6 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        return _fit_poisson_numeric(X, y), X, y

    def test_dispersion_statistic_is_the_cameron_trivedi_regression_t(self, poisson_case):
        """The t statistic of Cameron and Trivedi's auxiliary regression of
        ((y - mu)^2 - y) / mu on mu without intercept (Regression Analysis of Count Data,
        2nd ed., 2013, section 3.4), from statsmodels OLS; a t half its size would hide
        overdispersion."""
        sm = pytest.importorskip("statsmodels.api")
        from scipy import stats

        model, X, y = poisson_case
        n = len(y)
        mu = np.asarray(model.predict(X), dtype=np.float64)
        dep = ((y - mu) ** 2 - y) / mu
        ols = sm.OLS(dep, mu).fit()
        alpha_hat, t = float(ols.params[0]), float(ols.tvalues[0])
        rtol = 8.0 * n * UNIT_ROUNDOFF * (_cancellation(mu * dep) + 1.0)
        p_atol = 64.0 * UNIT_ROUNDOFF + stats.t.pdf(t, n - 1) * abs(t) * rtol

        expected_p = {
            "greater": stats.t.sf(t, n - 1),
            "less": stats.t.cdf(t, n - 1),
            "two-sided": 2.0 * stats.t.sf(abs(t), n - 1),
        }
        for alternative, p in expected_p.items():
            result = dispersion_test(model, X, y, alternative=alternative)
            assert result.alpha_hat == pytest.approx(alpha_hat, rel=rtol, abs=0)
            assert result.statistic == pytest.approx(t, rel=rtol, abs=0)
            assert result.p_value == pytest.approx(p, rel=0, abs=p_atol)

    def test_zero_inflation_statistic_is_van_den_broeks_score_at_a_constant_mean(self):
        """Van den Broek's score statistic [sum (1{y=0} - p0) / p0]^2 /
        [sum (1 - p0) / p0 - sum mu] (Biometrics 51, 1995, 738-743) for a model whose
        fitted mean is constant; a statistic missing the variance correction for the
        estimated mean would be 40% too small."""
        from scipy import stats

        rng = np.random.default_rng(20261126)
        n = 600
        # Zeros added to a Poisson(1.2) sample, so the statistic is far from zero.
        y = np.where(rng.uniform(size=n) < 0.08, 0.0, rng.poisson(1.2, n)).astype(float)
        X = pd.DataFrame({"x": rng.normal(size=n)})
        # A group-lasso penalty far above the gradient's norm sets the feature to
        # exactly zero, so the fitted mean is the same on every row.
        model = SuperGLM(
            family="poisson",
            penalty="group_lasso",
            selection_penalty=1e3,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        mu = np.asarray(model.predict(X), dtype=np.float64)
        assert np.ptp(mu) == 0.0

        p0 = np.exp(-mu)
        score_terms = ((y == 0) - p0) / p0
        variance = np.sum((1.0 - p0) / p0) - np.sum(mu)
        expected = np.sum(score_terms) ** 2 / variance
        variance_cancellation = (np.sum((1.0 - p0) / p0) + np.sum(mu)) / variance
        rtol = 8.0 * n * UNIT_ROUNDOFF * (2.0 * _cancellation(score_terms) + variance_cancellation)

        result = score_test_zi(model, X, y)
        assert result.statistic == pytest.approx(expected, rel=rtol, abs=0)
        p_atol = 64.0 * UNIT_ROUNDOFF + stats.chi2.pdf(expected, 1) * expected * rtol
        assert result.p_value == pytest.approx(stats.chi2.sf(expected, 1), rel=0, abs=p_atol)
        assert result.direction == "inflated"

    @pytest.mark.parametrize("correction", ["none", "aic", "bic"])
    def test_vuong_statistic_matches_its_definition_on_poisson_against_nb2(self, correction):
        """Vuong's sqrt(n) mean(m) / sd(m) with m_i the per-row log-likelihood ratio
        (Econometrica 57, 1989, section 5; sd with n - 1 as in pscl's vuong), and the
        Akaike or Schwarz shift of the mean by (k_a - k_b) / n or (k_a - k_b) log(n) / (2n),
        with densities from scipy; a halved statistic would flip the preferred model."""
        from scipy import stats

        rng = np.random.default_rng(20261127)
        n = 500
        x = rng.uniform(0.0, 2.0, n)
        mu_true = np.exp(0.2 + 0.5 * x)
        theta = 1.5
        y = rng.negative_binomial(theta, theta / (theta + mu_true)).astype(float)
        X = pd.DataFrame({"x": x, "x2": x**2})
        model_a = _fit_poisson_numeric(X, y)
        model_b = SuperGLM(
            family=NegativeBinomial(theta=theta),
            features={"x": Numeric()},
            selection_penalty=0.0,
        )
        model_b.fit(X[["x"]], y)

        mu_a = np.asarray(model_a.predict(X), dtype=np.float64)
        mu_b = np.asarray(model_b.predict(X[["x"]]), dtype=np.float64)
        m = stats.poisson.logpmf(y, mu_a) - stats.nbinom.logpmf(y, theta, theta / (theta + mu_b))
        k_a, k_b = 3, 2  # unpenalised coefficients, intercepts included
        shift = {
            "none": 0.0,
            "aic": (k_a - k_b) / n,
            "bic": (k_a - k_b) * np.log(n) / (2.0 * n),
        }[correction]
        mean = np.mean(m) - shift
        expected = np.sqrt(n) * mean / np.std(m, ddof=1)
        mean_cancellation = (np.mean(np.abs(m)) + shift) / abs(mean)
        rtol = 8.0 * n * UNIT_ROUNDOFF * (mean_cancellation + 2.0)
        expected_p = 2.0 * stats.norm.sf(abs(expected))

        result = vuong_test(model_a, model_b, X, y, correction=correction)
        assert result.statistic == pytest.approx(expected, rel=rtol, abs=0)
        p_atol = 64.0 * UNIT_ROUNDOFF + 2.0 * stats.norm.pdf(expected) * abs(expected) * rtol
        assert result.p_value == pytest.approx(expected_p, rel=0, abs=p_atol)
        assert expected < 0 and expected_p < 0.05, "fixture needs NB2 clearly preferred"
        assert result.preferred == "model_b"


# ── score_test_zi — the statistic, its null distribution and its refusals ──


def _poisson_frame(rng, n, low, high):
    """A numeric and a three-level factor whose Poisson mean spans ``low`` to ``high``."""
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    shift = np.where(g == "b", 0.25, np.where(g == "c", -0.25, 0.0))
    return pd.DataFrame({"x": x, "g": g}), low * (high / low) ** x * np.exp(shift)


def _fit_poisson_frame(X, y):
    from superglm import Categorical

    features = {"x": Numeric(), "g": Categorical(base="first")}
    return SuperGLM(family="poisson", features=features, selection_penalty=0.0).fit(X, y)


def _zi_score_statistic(y, mu):
    """[sum (1{y=0} - p0)/p0 + (y - mu)]^2 / sum((1 - p0)/p0 - mu) at means ``mu``."""
    p0 = np.exp(-mu)
    return np.sum(((y == 0) - p0) / p0 + (y - mu)) ** 2 / np.sum((1.0 - p0) / p0 - mu)


def _null_statistics(rng, replicates, mu, fit, X):
    """``score_test_zi`` on ``replicates`` Poisson samples at ``mu``, each refitted."""
    results = []
    for _ in range(replicates):
        y = rng.poisson(mu).astype(float)
        results.append(score_test_zi(fit(X, y), X, y))
    return np.array([r.statistic for r in results]), np.array([r.p_value for r in results])


def _assert_chi_square_one(statistics, p_values):
    """A 5% rejection rate and a mean of 1, each within three standard errors."""
    replicates = len(statistics)
    assert abs(np.mean(p_values < 0.05) - 0.05) <= 3.0 * np.sqrt(0.05 * 0.95 / replicates)
    assert abs(np.mean(statistics) - 1.0) <= 3.0 * np.sqrt(2.0 / replicates)


@pytest.fixture(scope="module")
def covariate_fit():
    """An unpenalised fit on a numeric and a three-level factor, with its dense design."""
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0, 2, n)
    g = rng.choice(["a", "b", "c"], n)
    eta = -0.3 + 0.6 * x + np.where(g == "b", 0.4, np.where(g == "c", -0.5, 0.0))
    y = rng.poisson(np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x, "g": g})
    design = np.column_stack([np.ones(n), x, g == "b", g == "c"]).astype(float)
    return _fit_poisson_frame(X, y), X, y, design


class TestScoreTestZIStatistic:
    """The score statistic for a constant inflation probability, on any fitted mean."""

    def test_statistic_with_covariates_is_the_score_statistic_of_the_fitted_design(
        self, covariate_fit
    ):
        """With covariates in a log-link mean the statistic is [sum (1{y=0} - p0)/p0]^2
        / [sum (1 - p0)/p0 - mu' D (D' diag(mu) D)^-1 D' mu], D the design of the mean
        (Deng and Paul, Statistica Sinica 15, 2005, Theorem 3.3's V11 at a log link); a
        variance that allows for an intercept only gives 0.657 for this 0.874."""
        from scipy import stats

        model, X, y, design = covariate_fit
        mu = np.asarray(model.predict(X), dtype=np.float64)
        p0 = np.exp(-mu)
        projected = design.T @ mu
        variance = np.sum((1.0 - p0) / p0) - projected @ np.linalg.solve(
            design.T @ (mu[:, None] * design), projected
        )
        expected = np.sum(((y == 0) - p0) / p0) ** 2 / variance

        result = score_test_zi(model, X, y)
        # The fit stops at its tolerance, and the statistic moves with the fitted mean.
        assert result.statistic == pytest.approx(expected, rel=1e-6)
        assert result.p_value == pytest.approx(stats.chi2.sf(expected, 1), rel=1e-6)
        assert result.direction == "deflated"

    def test_statistic_with_covariates_matches_statsmodels(self, covariate_fit):
        """statsmodels' Jansakul-Hinde test with a constant inflation probability, on
        its own maximum-likelihood fit of the same design."""
        sm = pytest.importorskip("statsmodels.api")
        model, X, y, design = covariate_fit
        reference = (
            sm.Poisson(y, design)
            .fit(disp=0, tol=1e-12)
            .get_diagnostic()
            .test_poisson_zeroinflation(method="broek", exog_infl=np.ones(len(y)))
        )

        result = score_test_zi(model, X, y)
        assert result.statistic == pytest.approx(reference.statistic, rel=1e-6)
        assert result.p_value == pytest.approx(reference.pvalue, rel=1e-6)

    @pytest.mark.pins_behaviour
    def test_constant_mean_statistic_is_van_den_broeks(self):
        """Pins the statistic at a fitted mean that is the same on every row, where it
        is van den Broek's (n0 - n p0)^2 / (n p0 (1 - p0) - n ybar p0^2) with
        p0 = exp(-ybar), as it was before the variance allowed for covariates."""
        rng = np.random.default_rng(11)
        n = 500
        y = np.where(rng.uniform(size=n) < 0.1, 0.0, rng.poisson(1.4, n)).astype(float)
        X = pd.DataFrame({"x": rng.normal(size=n)})
        # A group-lasso penalty far above the gradient's norm zeroes the feature.
        model = SuperGLM(
            family="poisson",
            penalty="group_lasso",
            selection_penalty=1e3,
            features={"x": Numeric()},
        ).fit(X, y)
        assert np.ptp(model.predict(X)) == 0.0
        p0 = np.exp(-np.mean(y))
        expected = (np.sum(y == 0) - n * p0) ** 2 / (n * p0 * (1 - p0) - np.sum(y) * p0**2)

        result = score_test_zi(model, X, y)
        assert result.statistic == pytest.approx(expected, rel=1e-6)  # fit tolerance
        assert result.direction == "inflated"

    def test_reml_smooth_fit_keeps_the_statistic(self):
        """A penalty that leaves the intercept free keeps sum(y - mu) at zero, so the
        score and its variance are the unpenalised fit's functions of the fitted mean."""
        rng = np.random.default_rng(5)
        n = 400
        X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
        y = rng.poisson(np.exp(-0.8 + 1.5 * np.sin(2 * np.pi * X["x"]))).astype(float)
        model = SuperGLM(family="poisson", features={"x": Spline(n_knots=8)}, selection_penalty=0.0)
        model.fit_reml(X, y)
        mu = np.asarray(model.predict(X), dtype=np.float64)
        p0 = np.exp(-mu)
        classical = np.sum(((y == 0) - p0) / p0) ** 2 / (np.sum((1.0 - p0) / p0) - np.sum(mu))

        assert model.result.effective_df < len(model.result.beta)  # the penalty acted
        assert abs(np.sum(y - mu)) <= 1e-6 * np.sum(y)
        assert score_test_zi(model, X, y).statistic == pytest.approx(classical, rel=1e-6)

    def test_rows_the_model_was_not_fitted_to_carry_their_mean_error(self):
        """On held-out rows sum(y - mu) is not zero, and the score includes it: the
        term that removes the first-order effect of the error in the fitted mean."""
        rng = np.random.default_rng(3)
        X, mu_true = _poisson_frame(rng, 800, 0.05, 2.5)
        y = rng.poisson(mu_true).astype(float)
        fitted, held_out = slice(0, 400), slice(400, 800)
        model = _fit_poisson_frame(X[fitted], y[fitted])
        mu = np.asarray(model.predict(X[held_out]), dtype=np.float64)

        assert abs(np.sum(y[held_out] - mu)) > 1.0
        result = score_test_zi(model, X[held_out], y[held_out])
        assert result.statistic == pytest.approx(_zi_score_statistic(y[held_out], mu), rel=1e-12)

    @pytest.mark.slow
    def test_null_rejection_rate_with_covariates_is_five_percent(self):
        """Over 500 seeded Poisson samples whose mean runs from 0.02 to 3 across the
        covariates, the test rejects at its 5% level in 5% of them and the statistic
        averages chi-square's 1; a variance that allows for an intercept only rejects
        in under 1% and averages 0.5."""
        rng = np.random.default_rng(20261126)
        X, mu = _poisson_frame(rng, 600, 0.02, 3.0)
        _assert_chi_square_one(*_null_statistics(rng, 500, mu, _fit_poisson_frame, X))

    @pytest.mark.slow
    def test_null_rejection_rate_of_reml_smooth_fits_is_five_percent(self):
        """The same over 300 seeded samples from a smooth mean between 0.06 and 2.2,
        each fitted by REML with a penalised spline."""

        def fit(X, y):
            model = SuperGLM(
                family="poisson", features={"x": Spline(n_knots=8)}, selection_penalty=0.0
            )
            model.fit_reml(X, y)
            return model

        rng = np.random.default_rng(20261127)
        X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 400)})
        mu = np.exp(-1.0 + 1.8 * np.sin(2 * np.pi * X["x"].to_numpy()))
        _assert_chi_square_one(*_null_statistics(rng, 300, mu, fit, X))

    @pytest.mark.parametrize(
        ("offset", "message"),
        [(-60.0, "every predicted mean is 0 to rounding"), (9.0, "overflows float64")],
    )
    def test_a_score_without_a_finite_positive_variance_is_refused(
        self, covariate_fit, offset, message
    ):
        """Forced through the offset: at -60 exp(mu) - 1 - mu rounds to 0 on every row,
        and at 9 the means pass 709, where exp(mu) leaves float64."""
        model, X, y, _ = covariate_fit
        with pytest.raises(ValueError, match=message):
            score_test_zi(model, X, y, offset=np.full(len(y), offset))


class TestUpperTailPValues:
    """A statistic far into the tail reports its tail probability, not zero."""

    def test_zero_inflation(self):
        from scipy import stats

        rng = np.random.default_rng(42)
        n = 2000
        X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
        y = rng.poisson(np.exp(0.2 + 0.5 * X["x"])).astype(float)
        y[rng.random(n) < 0.3] = 0.0
        result = score_test_zi(_fit_poisson(X, y), X, y)
        assert result.statistic > 100.0
        assert 0.0 < result.p_value == stats.chi2.sf(result.statistic, 1)

    def test_dispersion(self):
        from scipy import stats

        rng = np.random.default_rng(42)
        n = 2000
        X = pd.DataFrame({"x": rng.uniform(0, 5, n)})
        mu = np.exp(0.5 + 0.3 * X["x"].to_numpy())
        y = rng.negative_binomial(1.0, 1.0 / (1.0 + mu)).astype(float)
        result = dispersion_test(_fit_poisson(X, y), X, y)
        assert result.statistic > 10.0
        assert 0.0 < result.p_value == stats.t.sf(result.statistic, n - 1)

    def test_vuong(self):
        from scipy import stats

        rng = np.random.default_rng(42)
        n = 2000
        X = pd.DataFrame({"x": rng.uniform(0, 5, n), "z": rng.normal(size=n)})
        y = rng.poisson(np.exp(0.2 + 0.5 * X["x"])).astype(float)
        model_x = _fit_poisson_numeric(X[["x"]], y)
        model_z = _fit_poisson_numeric(X[["z"]], y)
        result = vuong_test(model_x, model_z, X, y, correction="none")
        assert result.statistic > 10.0
        assert 0.0 < result.p_value == 2.0 * stats.norm.sf(result.statistic)
