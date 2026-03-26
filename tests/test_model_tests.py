"""Tests for superglm.model_tests — model adequacy tests (T13-T26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, Spline, SuperGLM
from superglm.distributions import Gamma, NegativeBinomial
from superglm.model_tests import (
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
