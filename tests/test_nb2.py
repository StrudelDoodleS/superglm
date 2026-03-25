"""Tests for Negative Binomial (NB2) distribution."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from superglm import NegativeBinomial, SuperGLM, SuperGLMRegressor
from superglm.distributions import resolve_distribution
from superglm.features.numeric import Numeric
from superglm.nb_profile import NBProfileResult, estimate_nb_theta
from superglm.penalties.group_lasso import GroupLasso

# =====================================================================
# Helpers
# =====================================================================


def _generate_nb2(n, mu, theta, rng=None):
    """Simulate NB2(mu, theta) using scipy's nbinom."""
    if rng is None:
        rng = np.random.default_rng()
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()
    p = theta / (mu + theta)
    return rng.negative_binomial(theta, p).astype(np.float64)


# =====================================================================
# TestNB2Distribution
# =====================================================================


class TestNB2VarianceFunction:
    def test_basic(self):
        nb = NegativeBinomial(theta=5.0)
        mu = np.array([1.0, 2.0, 5.0, 10.0])
        expected = mu + mu**2 / 5.0
        np.testing.assert_allclose(nb.variance(mu), expected)

    def test_large_theta_approaches_poisson(self):
        """V(mu) = mu + mu^2/theta -> mu as theta -> inf."""
        nb = NegativeBinomial(theta=1e8)
        mu = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(nb.variance(mu), mu, rtol=1e-6)

    def test_small_theta_large_variance(self):
        nb = NegativeBinomial(theta=0.5)
        mu = np.array([5.0])
        expected = 5.0 + 25.0 / 0.5  # 55
        np.testing.assert_allclose(nb.variance(mu), expected)


class TestNB2DevianceUnit:
    def test_positive_y(self):
        nb = NegativeBinomial(theta=5.0)
        y = np.array([3.0, 7.0, 1.0])
        mu = np.array([2.0, 5.0, 3.0])
        d = nb.deviance_unit(y, mu)
        # All unit deviances should be non-negative
        assert np.all(d >= 0)

    def test_y_equals_mu(self):
        """Unit deviance at y=mu should be zero."""
        nb = NegativeBinomial(theta=5.0)
        mu = np.array([2.0, 5.0, 10.0])
        d = nb.deviance_unit(mu, mu)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_y_zero(self):
        """y=0 case uses special formula."""
        nb = NegativeBinomial(theta=5.0)
        y = np.array([0.0, 0.0])
        mu = np.array([2.0, 5.0])
        d = nb.deviance_unit(y, mu)
        expected = 2 * 5.0 * np.log(5.0 / (mu + 5.0))
        np.testing.assert_allclose(d, expected)

    def test_total_deviance_positive(self):
        nb = NegativeBinomial(theta=3.0)
        rng = np.random.default_rng(42)
        y = _generate_nb2(1000, mu=5.0, theta=3.0, rng=rng)
        mu = np.full_like(y, 5.0)
        d = nb.deviance_unit(y, mu)
        assert np.sum(d) > 0


class TestNB2LogLikelihood:
    def test_matches_scipy(self):
        """Log-likelihood should match scipy.stats.nbinom.logpmf."""
        nb = NegativeBinomial(theta=5.0)
        y = np.array([0, 1, 2, 5, 10], dtype=float)
        mu = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        weights = np.ones_like(y)

        ll_ours = nb.log_likelihood(y, mu, weights)

        # scipy nbinom: n=theta, p=theta/(mu+theta)
        p_nb = 5.0 / (3.0 + 5.0)
        ll_scipy = np.sum(nbinom.logpmf(y.astype(int), n=5.0, p=p_nb))

        np.testing.assert_allclose(ll_ours, ll_scipy, rtol=1e-10)

    def test_weighted(self):
        nb = NegativeBinomial(theta=5.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([2.0, 2.0, 2.0])
        w1 = np.ones(3)
        w2 = np.array([2.0, 2.0, 2.0])
        ll1 = nb.log_likelihood(y, mu, w1)
        ll2 = nb.log_likelihood(y, mu, w2)
        np.testing.assert_allclose(ll2, 2.0 * ll1)


class TestNB2PoissonLimit:
    def test_large_theta_coefficients(self):
        """With very large theta, NB2 fit should give similar results to Poisson."""
        rng = np.random.default_rng(42)
        n = 5000
        x = rng.normal(0, 1, n)
        log_mu = 1.0 + 0.5 * x
        mu = np.exp(log_mu)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        # Poisson fit
        m_pois = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        m_pois.fit(X, y)

        # NB2 with large theta
        m_nb = SuperGLM(
            family=NegativeBinomial(theta=1e6),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        m_nb.fit(X, y)

        np.testing.assert_allclose(m_nb.result.intercept, m_pois.result.intercept, atol=0.05)
        np.testing.assert_allclose(m_nb.result.beta, m_pois.result.beta, atol=0.05)


# =====================================================================
# TestNB2Fitting
# =====================================================================


class TestNB2FixedThetaFit:
    def test_convergence(self):
        """NB2 model with fixed theta converges on synthetic data."""
        rng = np.random.default_rng(42)
        n = 3000
        theta = 5.0
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=theta, rng=rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(X, y)

        assert model.result.converged
        # Check intercept near 1.0 and coef near 0.3
        np.testing.assert_allclose(model.result.intercept, 1.0, atol=0.15)

    def test_prediction_reasonable(self):
        rng = np.random.default_rng(42)
        n = 2000
        theta = 3.0
        mu_true = 5.0
        y = _generate_nb2(n, mu=mu_true, theta=theta, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        pred = model.predict(X)
        np.testing.assert_allclose(pred.mean(), mu_true, rtol=0.1)


# =====================================================================
# TestNB2Profile
# =====================================================================


class TestNB2ProfileTheta:
    def test_recovers_theta(self):
        """Profile estimation recovers theta from synthetic data."""
        rng = np.random.default_rng(42)
        n = 3000
        theta_true = 5.0
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=theta_true, rng=rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),  # initial guess
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )

        result = estimate_nb_theta(
            model,
            X,
            y,
            theta_bounds=(0.5, 20.0),
        )
        assert isinstance(result, NBProfileResult)
        np.testing.assert_allclose(result.theta_hat, theta_true, atol=2.0)

    def test_result_has_cache(self):
        rng = np.random.default_rng(42)
        n = 2000
        y = _generate_nb2(n, mu=5.0, theta=3.0, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )

        result = estimate_nb_theta(model, X, y, theta_bounds=(0.5, 15.0))
        assert len(result.cache) >= 1  # alternating alg converges in few iters
        assert result.n_evaluations >= 1

    def test_family_must_be_nb(self):
        model = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="NegativeBinomial"):
            estimate_nb_theta(model, X, y)


class TestNB2AutoTheta:
    def test_auto_theta_flow(self):
        """nb_theta='auto' triggers profile estimation in fit()."""
        rng = np.random.default_rng(42)
        n = 2000
        theta_true = 5.0
        y = _generate_nb2(n, mu=5.0, theta=theta_true, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        # After fit, family.theta should be a float (estimated)
        assert isinstance(model.family.theta, float)
        assert model.family.theta > 0
        assert model.result.converged


# =====================================================================
# TestNB2QuantileResiduals
# =====================================================================


class TestNB2QuantileResiduals:
    def test_approx_normal(self):
        """Quantile residuals should be ~N(0,1) for well-specified NB2."""
        rng = np.random.default_rng(42)
        n = 5000
        theta = 5.0
        mu_true = 5.0
        y = _generate_nb2(n, mu=mu_true, theta=theta, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        qr = metrics.residuals("quantile")

        # Should be approximately N(0,1)
        assert abs(qr.mean()) < 0.15
        assert abs(qr.std() - 1.0) < 0.15


# =====================================================================
# TestNB2MetricsSummary
# =====================================================================


class TestNB2MetricsSummary:
    def test_summary_works(self):
        rng = np.random.default_rng(42)
        n = 1000
        y = _generate_nb2(n, mu=5.0, theta=3.0, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=3.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        summary = metrics.summary()
        text = str(summary)
        assert "NegativeBinomial" in text or "Neg. Binomial" in text


# =====================================================================
# TestNB2Sklearn
# =====================================================================


class TestNB2Sklearn:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=5.0, rng=rng)
        X = pd.DataFrame({"x": x})

        reg = SuperGLMRegressor(
            family=NegativeBinomial(theta=5.0),
            selection_penalty=0.0,
        )
        reg.fit(X, y)
        pred = reg.predict(X)

        assert pred.shape == (n,)
        assert np.all(pred > 0)


# =====================================================================
# TestNB2Validation
# =====================================================================


class TestNB2InvalidTheta:
    def test_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            NegativeBinomial(theta=0.0)

    def test_negative(self):
        with pytest.raises(ValueError, match="must be > 0"):
            NegativeBinomial(theta=-1.0)


class TestNB2ResolveDistribution:
    def test_resolve_object(self):
        dist = resolve_distribution(NegativeBinomial(theta=5.0))
        assert isinstance(dist, NegativeBinomial)
        assert dist.theta == 5.0

    def test_resolve_missing_theta(self):
        with pytest.raises(ValueError, match="requires parameters"):
            resolve_distribution("negative_binomial")

    def test_resolve_passthrough(self):
        nb = NegativeBinomial(theta=3.0)
        assert resolve_distribution(nb) is nb
