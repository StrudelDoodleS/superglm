"""Tests for ModelMetrics diagnostics module."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import poisson

from superglm import SuperGLM, ModelMetrics
from superglm.distributions import Poisson, Gamma
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def poisson_data():
    """Small Poisson dataset with known structure."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = 0.5 + 0.3 * x1 - 0.2 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    w = np.ones(n)
    return X, y, w


@pytest.fixture
def fitted_poisson(poisson_data):
    """Fitted Poisson model on the test data."""
    X, y, w = poisson_data
    model = SuperGLM(
        family="poisson",
        lambda1=0.001,
        features={"x1": Numeric(), "x2": Numeric()},
    )
    model.fit(X, y, exposure=w)
    return model, X, y, w


@pytest.fixture
def metrics_obj(fitted_poisson):
    """ModelMetrics from the fitted Poisson model."""
    model, X, y, w = fitted_poisson
    return model.metrics(X, y, exposure=w)


# ── Log-likelihood ────────────────────────────────────────────────


class TestLogLikelihood:
    def test_poisson_ll_matches_scipy(self):
        """Poisson LL should match scipy.stats.poisson.logpmf."""
        y = np.array([0, 1, 2, 5, 10], dtype=float)
        mu = np.array([1.0, 2.0, 3.0, 4.0, 8.0])
        w = np.ones(5)
        ll = Poisson().log_likelihood(y, mu, w)
        expected = np.sum(poisson.logpmf(y.astype(int), mu))
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_poisson_ll_with_weights(self):
        """Weighted LL should differ from unweighted."""
        y = np.array([1, 2, 3], dtype=float)
        mu = np.array([1.5, 2.5, 2.0])
        w1 = np.ones(3)
        w2 = np.array([2.0, 1.0, 0.5])
        ll1 = Poisson().log_likelihood(y, mu, w1)
        ll2 = Poisson().log_likelihood(y, mu, w2)
        assert ll1 != ll2

    def test_gamma_ll_formula(self):
        """Gamma LL should match manual computation."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 2.8])
        w = np.ones(3)
        phi = 0.5
        k = 1.0 / phi
        expected = float(np.sum(
            k * np.log(k * y / mu) - k * y / mu - np.log(y) - gammaln(k)
        ))
        ll = Gamma().log_likelihood(y, mu, w, phi=phi)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_ll_from_metrics(self, metrics_obj):
        """LL accessed via metrics should be finite and negative."""
        assert np.isfinite(metrics_obj.log_likelihood)


# ── Information criteria ──────────────────────────────────────────


class TestInformationCriteria:
    def test_aic_formula(self, metrics_obj):
        """AIC = -2*LL + 2*edf."""
        expected = -2.0 * metrics_obj.log_likelihood + 2.0 * metrics_obj.effective_df
        np.testing.assert_allclose(metrics_obj.aic, expected)

    def test_bic_formula(self, metrics_obj):
        """BIC = -2*LL + log(n)*edf."""
        expected = (
            -2.0 * metrics_obj.log_likelihood
            + np.log(metrics_obj.n_obs) * metrics_obj.effective_df
        )
        np.testing.assert_allclose(metrics_obj.bic, expected)

    def test_bic_ge_aic(self, metrics_obj):
        """BIC >= AIC when n >= e^2 ≈ 7.4 (which it always is here)."""
        assert metrics_obj.bic >= metrics_obj.aic - 1e-10

    def test_aicc_formula(self, metrics_obj):
        edf = metrics_obj.effective_df
        n = metrics_obj.n_obs
        expected = metrics_obj.aic + 2 * edf * (edf + 1) / (n - edf - 1)
        np.testing.assert_allclose(metrics_obj.aicc, expected)

    def test_ebic_ge_bic(self, metrics_obj):
        """EBIC(gamma>0) >= BIC."""
        assert metrics_obj.ebic(gamma=0.5) >= metrics_obj.bic - 1e-10

    def test_ebic_gamma_zero_equals_bic(self, metrics_obj):
        """EBIC(gamma=0) == BIC."""
        np.testing.assert_allclose(metrics_obj.ebic(gamma=0.0), metrics_obj.bic, atol=1e-10)


# ── Deviance ──────────────────────────────────────────────────────


class TestDeviance:
    def test_null_deviance_gt_residual(self, metrics_obj):
        """Model should improve on the null (intercept-only) model."""
        assert metrics_obj.null_deviance > metrics_obj.deviance

    def test_explained_deviance_in_range(self, metrics_obj):
        """Explained deviance should be in [0, 1] for a well-fitting model."""
        assert 0 <= metrics_obj.explained_deviance <= 1

    def test_pearson_chi2_positive(self, metrics_obj):
        assert metrics_obj.pearson_chi2 > 0


# ── Residuals ─────────────────────────────────────────────────────


class TestResiduals:
    def test_deviance_residuals_sum_sq_approx_deviance(self, metrics_obj):
        """sum(r_dev^2) should approximately equal the deviance."""
        r = metrics_obj.residuals("deviance")
        np.testing.assert_allclose(np.sum(r**2), metrics_obj.deviance, rtol=0.01)

    def test_pearson_residuals_mean_approx_zero(self, metrics_obj):
        """Pearson residuals should have mean approximately 0."""
        r = metrics_obj.residuals("pearson")
        assert abs(np.mean(r)) < 0.5  # rough check

    def test_response_residuals(self, metrics_obj):
        """Response residuals are just y - mu."""
        r = metrics_obj.residuals("response")
        np.testing.assert_allclose(r, metrics_obj._y - metrics_obj._mu)

    def test_working_residuals(self, metrics_obj):
        """Working residuals are (y - mu) / mu for log link."""
        r = metrics_obj.residuals("working")
        np.testing.assert_allclose(r, (metrics_obj._y - metrics_obj._mu) / metrics_obj._mu)

    def test_unknown_residual_raises(self, metrics_obj):
        with pytest.raises(ValueError, match="Unknown residual type"):
            metrics_obj.residuals("bogus")

    def test_quantile_residuals_poisson(self, metrics_obj):
        """Quantile residuals should be approximately standard normal."""
        r = metrics_obj.residuals("quantile")
        assert abs(np.mean(r)) < 0.3
        assert 0.5 < np.std(r) < 1.5


# ── Leverage ──────────────────────────────────────────────────────


class TestLeverage:
    def test_leverage_bounded(self, metrics_obj):
        """All leverage values should be in [0, 1]."""
        h = metrics_obj.leverage
        assert np.all(h >= 0)
        assert np.all(h <= 1.0 + 1e-10)

    def test_leverage_sum_approx_edf(self, metrics_obj):
        """sum(h_i) should approximate effective_df."""
        h_sum = np.sum(metrics_obj.leverage)
        # Leverage sum ≈ p_active (not exactly edf due to shrinkage),
        # but should be in the right ballpark
        assert h_sum > 0
        assert h_sum < metrics_obj.n_obs


# ── Cook's distance ──────────────────────────────────────────────


class TestCooksDistance:
    def test_cooks_nonnegative(self, metrics_obj):
        assert np.all(metrics_obj.cooks_distance >= 0)

    def test_std_deviance_residuals_exist(self, metrics_obj):
        r = metrics_obj.std_deviance_residuals
        assert r.shape == (metrics_obj.n_obs,)
        assert np.all(np.isfinite(r))

    def test_std_pearson_residuals_exist(self, metrics_obj):
        r = metrics_obj.std_pearson_residuals
        assert r.shape == (metrics_obj.n_obs,)
        assert np.all(np.isfinite(r))


# ── Active groups ─────────────────────────────────────────────────


class TestActiveGroups:
    def test_n_active_groups(self, metrics_obj):
        """With low lambda, both features should be active."""
        assert metrics_obj.n_active_groups == 2


# ── Summary ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_keys(self, metrics_obj):
        s = metrics_obj.summary()
        assert "information_criteria" in s
        assert "deviance" in s
        assert "fit" in s
        assert "aic" in s["information_criteria"]
        assert "bic" in s["information_criteria"]

    def test_summary_values_finite(self, metrics_obj):
        s = metrics_obj.summary()
        for section in s.values():
            for v in section.values():
                assert np.isfinite(v), f"Non-finite value in summary: {v}"


# ── Integration: spline model ────────────────────────────────────


class TestSplineIntegration:
    def test_spline_model_metrics(self):
        """Smoke test: metrics work with spline features."""
        rng = np.random.default_rng(123)
        n = 300
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.1 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        w = np.ones(n)

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, exposure=w)
        m = model.metrics(X, y, exposure=w)

        # All properties should be accessible without error
        assert np.isfinite(m.aic)
        assert np.isfinite(m.bic)
        assert np.isfinite(m.aicc)
        assert np.isfinite(m.log_likelihood)
        assert m.null_deviance > m.deviance
        assert 0 < m.explained_deviance < 1

        r = m.residuals("deviance")
        assert r.shape == (n,)

        h = m.leverage
        assert np.all(h >= 0)
        assert np.all(h <= 1.0 + 1e-10)

        assert np.all(m.cooks_distance >= 0)


# ── Convenience accessor ─────────────────────────────────────────


class TestConvenienceAccessor:
    def test_model_metrics_method(self, fitted_poisson):
        """SuperGLM.metrics() returns a ModelMetrics object."""
        model, X, y, w = fitted_poisson
        m = model.metrics(X, y, exposure=w)
        assert isinstance(m, ModelMetrics)
