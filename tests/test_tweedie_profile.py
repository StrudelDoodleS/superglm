"""Tests for Tweedie profile likelihood — p estimation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie as TweedieDistribution
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.penalties.group_lasso import GroupLasso
from superglm.tweedie_profile import (
    TweedieProfileResult,
    _profile_phi,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)


def _generate_weighted_tweedie(mu, phi, p, weights, rng):
    """Simulate Tweedie responses under the prior-weight convention phi / w."""
    mu = np.asarray(mu, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    y = np.empty(len(mu), dtype=np.float64)
    for i in range(len(mu)):
        y[i] = generate_tweedie_cpg(1, mu=mu[i], phi=phi / weights[i], p=p, rng=rng)[0]
    return y


# =====================================================================
# TestGenerateTweedieCPG
# =====================================================================


class TestGenerateTweedieCPG:
    def test_mean_matches_mu(self):
        rng = np.random.default_rng(42)
        y = generate_tweedie_cpg(50_000, mu=10.0, phi=3.0, p=1.6, rng=rng)
        np.testing.assert_allclose(y.mean(), 10.0, rtol=0.05)

    def test_variance_matches(self):
        rng = np.random.default_rng(42)
        mu, phi, p = 10.0, 3.0, 1.6
        y = generate_tweedie_cpg(100_000, mu=mu, phi=phi, p=p, rng=rng)
        expected_var = phi * mu**p
        np.testing.assert_allclose(y.var(), expected_var, rtol=0.15)

    def test_zero_probability(self):
        rng = np.random.default_rng(42)
        mu, phi, p = 10.0, 3.0, 1.6
        y = generate_tweedie_cpg(100_000, mu=mu, phi=phi, p=p, rng=rng)
        lam = mu ** (2 - p) / ((2 - p) * phi)
        expected_zero_prob = np.exp(-lam)
        actual_zero_prob = np.mean(y == 0)
        np.testing.assert_allclose(actual_zero_prob, expected_zero_prob, atol=0.02)

    def test_heterogeneous_mu(self):
        rng = np.random.default_rng(42)
        mu = rng.uniform(5, 50, size=10_000)
        y = generate_tweedie_cpg(10_000, mu=mu, phi=3.0, p=1.6, rng=rng)
        assert y.shape == (10_000,)
        assert np.all(y >= 0)

    def test_insurance_like(self):
        """High zero-rate typical of motor insurance claims."""
        rng = np.random.default_rng(42)
        mu, phi, p = 341.0, 30_000.0, 1.89
        y = generate_tweedie_cpg(100_000, mu=mu, phi=phi, p=p, rng=rng)
        lam = mu ** (2 - p) / ((2 - p) * phi)
        expected_zero = np.exp(-lam)
        actual_zero = np.mean(y == 0)
        np.testing.assert_allclose(actual_zero, expected_zero, atol=0.01)


# =====================================================================
# TestTweedieLogpdf
# =====================================================================


class TestTweedieLogpdf:
    def test_zero_obs_point_mass(self):
        """y=0 formula: logpdf = -mu^(2-p) / ((2-p) * phi)."""
        y = np.array([0.0, 0.0])
        mu = np.array([5.0, 10.0])
        phi, p = 2.0, 1.5
        result = tweedie_logpdf(y, mu, phi, p)
        expected = -np.power(mu, 2 - p) / ((2 - p) * phi)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_logpdf_finite_positive(self):
        """All logpdf values should be finite for y > 0 from CPG."""
        rng = np.random.default_rng(42)
        mu_val, phi, p = 10.0, 3.0, 1.6
        y = generate_tweedie_cpg(5_000, mu=mu_val, phi=phi, p=p, rng=rng)
        pos = y > 0
        mu = np.full_like(y, mu_val)
        lp = tweedie_logpdf(y[pos], mu[pos], phi, p)
        assert np.all(np.isfinite(lp))

    def test_nll_minimized_at_true_mu(self):
        """NLL should be lower at the true mu than at a wrong mu."""
        rng = np.random.default_rng(42)
        mu_true, phi, p = 10.0, 3.0, 1.6
        y = generate_tweedie_cpg(10_000, mu=mu_true, phi=phi, p=p, rng=rng)
        mu_arr_true = np.full_like(y, mu_true)
        mu_arr_wrong = np.full_like(y, 20.0)
        nll_true = -np.mean(tweedie_logpdf(y, mu_arr_true, phi, p))
        nll_wrong = -np.mean(tweedie_logpdf(y, mu_arr_wrong, phi, p))
        assert nll_true < nll_wrong

    def test_saddlepoint_fallback(self):
        """Extreme t_arg values should still produce finite results."""
        # Force saddlepoint by using a very low t_arg_limit
        y = np.array([100.0, 200.0, 500.0])
        mu = np.array([50.0, 100.0, 250.0])
        phi, p = 5.0, 1.5
        lp = tweedie_logpdf(y, mu, phi, p, t_arg_limit=0.0)  # forces saddlepoint
        assert np.all(np.isfinite(lp))

    def test_weights_scale_phi(self):
        """logpdf(y, mu, phi, p, weights=2) == logpdf(y, mu, phi/2, p)."""
        rng = np.random.default_rng(42)
        y = generate_tweedie_cpg(1_000, mu=10.0, phi=3.0, p=1.6, rng=rng)
        mu = np.full_like(y, 10.0)
        phi, p = 3.0, 1.6

        lp_weighted = tweedie_logpdf(y, mu, phi, p, weights=np.full_like(y, 2.0))
        lp_half_phi = tweedie_logpdf(y, mu, phi / 2.0, p)
        np.testing.assert_allclose(lp_weighted, lp_half_phi, rtol=1e-10)

    def test_distribution_log_likelihood_matches_weighted_logpdf(self):
        """Tweedie.log_likelihood should sum weighted logpdf once, not twice."""
        rng = np.random.default_rng(123)
        n = 2_000
        mu = np.full(n, 10.0)
        weights = rng.uniform(0.5, 2.0, n)
        phi, p = 3.0, 1.6
        y = _generate_weighted_tweedie(mu, phi, p, weights, rng)

        dist = TweedieDistribution(p)
        ll_direct = float(np.sum(tweedie_logpdf(y, mu, phi, p, weights=weights)))
        ll_dist = dist.log_likelihood(y, mu, weights, phi=phi)
        np.testing.assert_allclose(ll_dist, ll_direct, rtol=1e-10)


# =====================================================================
# TestEstimatePhi
# =====================================================================


class TestEstimatePhi:
    def test_phi_recovery(self):
        rng = np.random.default_rng(42)
        mu, phi_true, p = 10.0, 3.0, 1.6
        y = generate_tweedie_cpg(50_000, mu=mu, phi=phi_true, p=p, rng=rng)
        mu_arr = np.full_like(y, mu)
        phi_hat = estimate_phi(y, mu_arr, p)
        np.testing.assert_allclose(phi_hat, phi_true, rtol=0.1)

    def test_phi_positive(self):
        rng = np.random.default_rng(42)
        y = generate_tweedie_cpg(1_000, mu=10.0, phi=3.0, p=1.6, rng=rng)
        mu_arr = np.full_like(y, 10.0)
        assert estimate_phi(y, mu_arr, 1.6) > 0

    def test_weighted_phi_recovery(self):
        rng = np.random.default_rng(123)
        n = 12_000
        mu = np.full(n, 10.0)
        phi_true, p = 3.0, 1.6
        weights = rng.uniform(0.5, 2.0, n)
        y = _generate_weighted_tweedie(mu, phi_true, p, weights, rng)

        phi_hat = estimate_phi(y, mu, p, weights=weights)
        np.testing.assert_allclose(phi_hat, phi_true, rtol=0.12)

    def test_mle_phi_recovery(self):
        rng = np.random.default_rng(456)
        n = 20_000
        mu = np.full(n, 10.0)
        phi_true, p = 3.0, 1.6
        y = generate_tweedie_cpg(n, mu=mu, phi=phi_true, p=p, rng=rng)

        phi_hat, _ = _profile_phi(y, mu, p, phi_method="mle")
        np.testing.assert_allclose(phi_hat, phi_true, rtol=0.12)


# =====================================================================
# TestProfileLikelihood
# =====================================================================


def _make_intercept_model(p=1.6, lambda1=0.0):
    """Create a minimal intercept-only Tweedie model."""
    m = SuperGLM(family=TweedieDistribution(p=1.5), penalty=GroupLasso(lambda1=lambda1))
    return m


def _make_model_with_covariates(lambda1=0.0):
    """Create a Tweedie model with numeric covariates."""
    return SuperGLM(
        family=TweedieDistribution(p=1.5),
        penalty=GroupLasso(lambda1=lambda1),
        features={"x1": Numeric(), "x2": Numeric()},
    )


class TestProfileLikelihood:
    def test_recovers_p_simple(self):
        """Intercept-only model recovers p from simulated data."""
        import pandas as pd

        rng = np.random.default_rng(42)
        p_true = 1.6
        n = 5_000
        y = generate_tweedie_cpg(n, mu=10.0, phi=3.0, p=p_true, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )

        result = estimate_tweedie_p(model, X, y, p_bounds=(1.1, 1.9))
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)

    def test_recovers_p_covariates(self):
        """Model with covariates recovers p."""
        import pandas as pd

        rng = np.random.default_rng(123)
        p_true = 1.7
        n = 3_000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        log_mu = 2.0 + 0.3 * x1 - 0.2 * x2
        mu = np.exp(log_mu)
        y = generate_tweedie_cpg(n, mu=mu, phi=3.0, p=p_true, rng=rng)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = _make_model_with_covariates(lambda1=0.0)
        result = estimate_tweedie_p(model, X, y, p_bounds=(1.1, 1.9))
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    @pytest.mark.parametrize("phi_method", ["pearson", "mle"])
    def test_recovers_p_with_prior_weights(self, phi_method):
        """Profile likelihood should recover p when sample_weight acts through phi / w."""
        rng = np.random.default_rng(321)
        p_true = 1.6
        phi_true = 3.0
        n = 4_000
        x1 = rng.normal(0, 1, n)
        sample_weight = rng.uniform(0.5, 2.0, n)
        mu = np.exp(1.5 + 0.25 * x1)
        y = _generate_weighted_tweedie(mu, phi_true, p_true, sample_weight, rng)
        X = pd.DataFrame({"x1": x1})

        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric()},
        )

        result = estimate_tweedie_p(
            model, X, y, sample_weight=sample_weight, p_bounds=(1.1, 1.9), phi_method=phi_method
        )
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)
        np.testing.assert_allclose(result.phi_hat, phi_true, rtol=0.2)

    def test_notebook_style_profile_recovers_true_p_under_prior_weights(self):
        """Notebook-style exposure weights should not bias Pearson profiling downward."""
        rng = np.random.default_rng(42)
        p_true = 1.6
        phi_true = 2.0
        n = 12_000

        x = rng.uniform(0.0, 1.0, n)
        sample_weight = rng.uniform(0.5, 2.0, n)
        mu_rate = np.exp(np.log(5.0) + 0.5 * np.sin(2.0 * np.pi * x))
        mu_total = mu_rate * sample_weight
        y = _generate_weighted_tweedie(mu_total, phi_true, p_true, sample_weight, rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=TweedieDistribution(p=1.2),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Spline(n_knots=10)},
        )

        result = estimate_tweedie_p(
            model,
            X,
            y,
            sample_weight=sample_weight,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.06)
        np.testing.assert_allclose(result.phi_hat, phi_true, rtol=0.12)

    @pytest.mark.slow
    def test_insurance_like(self):
        """Insurance-like data with sample_weight and high zero rate."""
        import pandas as pd

        rng = np.random.default_rng(77)
        p_true = 1.85
        n = 20_000
        sample_weight = rng.uniform(0.5, 1.5, n)
        x1 = rng.normal(0, 1, n)
        log_mu = np.log(300) + 0.1 * x1
        mu = np.exp(log_mu) * sample_weight
        y = generate_tweedie_cpg(n, mu=mu, phi=20_000.0, p=p_true, rng=rng)

        # Scale down for numerical stability
        scale = 1000.0
        y_scaled = y / scale
        exposure_scaled = sample_weight  # sample_weight is unitless

        X = pd.DataFrame({"x1": x1})
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric()},
        )

        result = estimate_tweedie_p(
            model,
            X,
            y_scaled,
            sample_weight=exposure_scaled,
            offset=np.log(sample_weight),
            p_bounds=(1.2, 1.95),
        )
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)

    def test_family_must_be_tweedie(self):
        """Raises ValueError if family is not tweedie."""
        import pandas as pd

        model = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="tweedie"):
            estimate_tweedie_p(model, X, y)

    def test_result_has_search_trace(self):
        """search_trace should be populated with >= 3 entries from Brent."""
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 2_000
        y = generate_tweedie_cpg(n, mu=10.0, phi=3.0, p=1.6, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )

        result = estimate_tweedie_p(model, X, y, p_bounds=(1.1, 1.9))
        assert len(result.search_trace) >= 3
        assert result.method == "brent"
        assert result.phi_method == "pearson"


class TestWeightedPhiConvention:
    @staticmethod
    def _make_weighted_dataset(seed: int = 2026, n: int = 4_000):
        rng = np.random.default_rng(seed)
        p_true = 1.6
        phi_true = 2.0
        x = rng.uniform(0.0, 1.0, n)
        sample_weight = rng.uniform(0.5, 2.0, n)
        mu = np.exp(1.2 + 0.7 * x) * sample_weight
        y = _generate_weighted_tweedie(mu, phi_true, p_true, sample_weight, rng)
        X = pd.DataFrame({"x": x})
        return X, y, sample_weight

    @staticmethod
    def _assert_prior_weight_phi(model, X, y, sample_weight):
        mu = np.asarray(model.predict(X), dtype=np.float64)
        edf = float(model.result.effective_df)
        pearson_chi2 = float(np.sum(sample_weight * (y - mu) ** 2 / np.maximum(mu, 1e-10) ** 1.6))
        expected_phi = pearson_chi2 / max(len(y) - edf, 1.0)
        wrong_phi = pearson_chi2 / max(float(np.sum(sample_weight)) - edf, 1.0)
        np.testing.assert_allclose(model.result.phi, expected_phi, rtol=0.02)
        assert abs(model.result.phi - wrong_phi) / expected_phi > 0.10

    def test_direct_irls_uses_observation_count_df_for_weighted_phi(self):
        X, y, sample_weight = self._make_weighted_dataset()
        model = SuperGLM(
            family=TweedieDistribution(p=1.6),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        self._assert_prior_weight_phi(model, X, y, sample_weight)

    def test_pirls_uses_observation_count_df_for_weighted_phi(self):
        X, y, sample_weight = self._make_weighted_dataset()
        model = SuperGLM(
            family=TweedieDistribution(p=1.6),
            penalty=GroupLasso(lambda1=0.05),
            features={"x": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        self._assert_prior_weight_phi(model, X, y, sample_weight)


# =====================================================================
# TestNumericalStability
# =====================================================================


class TestNumericalStability:
    def test_all_zero_response(self):
        """logpdf should handle all-zero y without NaN/Inf."""
        y = np.zeros(100)
        mu = np.full(100, 5.0)
        lp = tweedie_logpdf(y, mu, phi=2.0, p=1.5)
        assert np.all(np.isfinite(lp))
        assert np.all(lp < 0)  # log-probabilities are negative

    def test_very_small_mu(self):
        """Small mu should not cause overflow/NaN."""
        y = np.array([0.0, 0.001, 0.0, 0.0005])
        mu = np.array([0.001, 0.001, 0.002, 0.001])
        lp = tweedie_logpdf(y, mu, phi=1.0, p=1.5)
        assert np.all(np.isfinite(lp))

    def test_p_near_lower_bound(self):
        """p close to 1 (Poisson-like)."""
        rng = np.random.default_rng(42)
        y = generate_tweedie_cpg(5_000, mu=10.0, phi=3.0, p=1.02, rng=rng)
        mu = np.full_like(y, 10.0)
        lp = tweedie_logpdf(y, mu, phi=3.0, p=1.02)
        assert np.all(np.isfinite(lp))

    def test_p_near_upper_bound(self):
        """p close to 2 (Gamma-like)."""
        rng = np.random.default_rng(42)
        y = generate_tweedie_cpg(5_000, mu=10.0, phi=3.0, p=1.98, rng=rng)
        # Filter to positive only since p~2 has very few zeros
        pos = y > 0
        mu = np.full(pos.sum(), 10.0)
        lp = tweedie_logpdf(y[pos], mu, phi=3.0, p=1.98)
        assert np.all(np.isfinite(lp))


# =====================================================================
# Fit metadata tracking
# =====================================================================


class TestFitMetadata:
    def test_fit_records_metadata(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x": rng.uniform(0, 1, 200)})
        y = rng.poisson(1.0, 200).astype(float)
        model = SuperGLM(family="poisson", selection_penalty=0.01, features={"x": Numeric()})
        model.fit(X, y)
        assert model._last_fit_meta is not None
        assert model._last_fit_meta["method"] == "fit"
        assert model._last_fit_meta["discrete"] is False

    def test_fit_reml_records_metadata(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x": rng.uniform(0, 1, 200)})
        y = rng.poisson(1.0, 200).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X, y)
        assert model._last_fit_meta is not None
        assert model._last_fit_meta["method"] == "fit_reml"

    def test_fit_reml_discrete_records_metadata(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"x": rng.uniform(0, 1, 500)})
        y = rng.poisson(1.0, 500).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X, y)
        assert model._last_fit_meta["method"] == "fit_reml"
        assert model._last_fit_meta["discrete"] is True


# =====================================================================
# Tweedie p profiling with fit_mode
# =====================================================================


def _tweedie_data(n=3000, p_true=1.6, seed=42):
    """Synthetic Tweedie data with one covariate."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    log_mu = 2.0 + 0.3 * x1
    mu = np.exp(log_mu)
    y = generate_tweedie_cpg(n, mu=mu, phi=3.0, p=p_true, rng=rng)
    X = pd.DataFrame({"x1": x1})
    return X, y, p_true


class TestEstimatePFitMode:
    def test_fit_mode_fit_recovers_p(self):
        """fit_mode='fit' (default) should recover p."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = model.estimate_p(X, y, fit_mode="fit")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)
        # Model should be refitted with estimated p
        assert model.family.p == result.p_hat
        assert model._result is not None
        assert model._last_fit_meta["method"] == "fit"

    def test_fit_mode_fit_recovers_p_mle_phi(self):
        """fit_mode='fit' should also recover p with phi_method='mle'."""
        X, y, p_true = _tweedie_data(n=2_000, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = model.estimate_p(X, y, fit_mode="fit", phi_method="mle")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)
        assert model._last_fit_meta["method"] == "fit"

    @pytest.mark.slow
    def test_fit_mode_reml_recovers_p(self):
        """fit_mode='reml' should recover p using REML fits."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        result = model.estimate_p(X, y, fit_mode="reml")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)
        # Model should be refitted with REML
        assert model.family.p == result.p_hat
        assert model._last_fit_meta["method"] == "fit_reml"
        assert hasattr(model, "_reml_result")

    @pytest.mark.slow
    def test_fit_mode_reml_recovers_p_mle_phi(self):
        """fit_mode='reml' should support phi_method='mle'."""
        X, y, p_true = _tweedie_data(n=1_500, seed=11)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        result = model.estimate_p(X, y, fit_mode="reml", phi_method="mle")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.25)
        assert model._last_fit_meta["method"] == "fit_reml"

    def test_fit_mode_inherit_from_fit(self):
        """After fit(), inherit should use the fit path."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        model.fit(X, y)
        assert model._last_fit_meta["method"] == "fit"

        result = model.estimate_p(X, y, fit_mode="inherit")
        assert model._last_fit_meta["method"] == "fit"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    @pytest.mark.slow
    def test_fit_mode_inherit_from_reml(self):
        """After fit_reml(), inherit should use the REML path."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X, y)
        assert model._last_fit_meta["method"] == "fit_reml"

        result = model.estimate_p(X, y, fit_mode="inherit")
        assert model._last_fit_meta["method"] == "fit_reml"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    def test_fit_mode_inherit_no_prior_fit_falls_back(self):
        """inherit with no prior fit falls back to 'fit'."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        assert model._last_fit_meta is None
        model.estimate_p(X, y, fit_mode="inherit")
        assert model._last_fit_meta["method"] == "fit"

    def test_invalid_fit_mode_raises(self):
        """Invalid fit_mode should raise immediately."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(ValueError, match="fit_mode"):
            model.estimate_p(X, y, fit_mode="bogus")

    def test_invalid_phi_method_raises(self):
        """Invalid phi_method should raise immediately."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(ValueError, match="phi_method"):
            model.estimate_p(X, y, phi_method="bogus")

    def test_wrong_family_raises(self):
        """Non-Tweedie model should raise immediately."""
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        model = SuperGLM(family="poisson", selection_penalty=0, features={"x": Numeric()})
        with pytest.raises(ValueError, match="tweedie"):
            model.estimate_p(X, y)

    @pytest.mark.slow
    def test_reml_and_fit_agree_on_p(self):
        """REML and fit paths should agree on p estimate for the same data."""
        X, y, p_true = _tweedie_data()
        model_fit = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result_fit = model_fit.estimate_p(X, y, fit_mode="fit")

        model_reml = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        result_reml = model_reml.estimate_p(X, y, fit_mode="reml")

        # Both should land near p_true; allow wider tolerance since
        # different model flexibility may shift the estimate slightly
        np.testing.assert_allclose(result_fit.p_hat, result_reml.p_hat, atol=0.3)


# =====================================================================
# Search methods
# =====================================================================


class TestSearchMethods:
    """Tests for grid, grid_refine, and profile_opt search methods."""

    def test_grid_recovers_p(self):
        """method='grid' should recover p from synthetic data."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, method="grid", n_grid=20, p_bounds=(1.1, 1.9))
        assert isinstance(result, TweedieProfileResult)
        assert result.method == "grid"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)

    def test_grid_explicit_grid(self):
        """User-supplied grid array should be used."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        grid = np.array([1.3, 1.5, 1.6, 1.7, 1.9])
        result = estimate_tweedie_p(model, X, y, method="grid", grid=grid)
        assert len(result.search_trace) == len(grid)
        assert result.p_hat in grid

    def test_grid_refine_recovers_p(self):
        """method='grid_refine' should recover p."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(
            model, X, y, method="grid_refine", n_grid_coarse=10, p_bounds=(1.1, 1.9)
        )
        assert result.method == "grid_refine"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)

    def test_profile_opt_recovers_p(self):
        """method='profile_opt' should recover p far from init grid.

        Regression test: with 6-decimal cache rounding, L-BFGS-B finite-
        difference probes aliased to the same key, making the gradient
        appear zero. The optimizer would stop at the best init point (1.5)
        instead of actually searching. Using p_true=1.35 (far from 1.5)
        and checking for optimizer trace rows catches this.
        """
        X, y, p_true = _tweedie_data(p_true=1.35, seed=99)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, method="profile_opt", p_bounds=(1.1, 1.9))
        assert result.method == "profile_opt"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)
        # Must have optimizer evals beyond init — proves the cache didn't
        # flatten the objective for L-BFGS-B
        sources = set(result.search_trace["source"].unique())
        assert "optimizer" in sources, f"Only sources: {sources}; optimizer never explored"
        assert result.n_evaluations > 3, f"Only {result.n_evaluations} evals — stopped at init"

    def test_profile_opt_powell(self):
        """optimizer='Powell' should also recover p."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(
            model, X, y, method="profile_opt", optimizer="Powell", p_bounds=(1.1, 1.9)
        )
        assert result.method == "profile_opt"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    @pytest.mark.slow
    def test_low_p_boundary_regression(self):
        """Low-p profiles should not spuriously prefer the lower bound."""
        X, y, _ = _tweedie_data(n=2_200, p_true=1.25, seed=7)
        kwargs = {"p_bounds": (1.1, 1.9), "phi_method": "mle"}

        grid_model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        grid = np.linspace(1.1, 1.9, 81)
        r_grid = estimate_tweedie_p(grid_model, X, y, method="grid", grid=grid, **kwargs)

        lbfgsb_model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        r_lbfgsb = estimate_tweedie_p(
            lbfgsb_model, X, y, method="profile_opt", optimizer="L-BFGS-B", **kwargs
        )

        powell_model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        r_powell = estimate_tweedie_p(
            powell_model, X, y, method="profile_opt", optimizer="Powell", **kwargs
        )

        assert r_grid.p_hat > 1.15
        np.testing.assert_allclose(r_grid.p_hat, r_lbfgsb.p_hat, atol=0.02)
        np.testing.assert_allclose(r_grid.p_hat, r_powell.p_hat, atol=0.02)

    def test_low_p_saddlepoint_warning(self):
        """Warn when saddlepoint dominates the final low-p profile fit."""
        X, y, _ = _tweedie_data(n=2_500, p_true=1.08, seed=4)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        with pytest.warns(UserWarning, match="Saddlepoint approximation used"):
            result = estimate_tweedie_p(
                model,
                X,
                y,
                method="profile_opt",
                optimizer="Powell",
                p_bounds=(1.05, 1.9),
                phi_method="mle",
            )

        assert result.saddlepoint_fraction >= 0.25
        assert result.n_saddlepoint > 0
        assert result.n_positive > 0
        assert len(result.warnings) == 1

    def test_regular_profile_has_no_saddlepoint_warning(self):
        """Typical interior fits should not warn about saddlepoint usage."""
        X, y, _ = _tweedie_data(n=2_500, p_true=1.25, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimate_tweedie_p(
                model,
                X,
                y,
                method="profile_opt",
                optimizer="Powell",
                p_bounds=(1.05, 1.9),
                phi_method="mle",
            )

        assert not caught
        assert result.saddlepoint_fraction < 0.10
        assert result.warnings == []

    def test_grid_with_weights(self):
        """Grid search should forward sample_weight correctly."""
        rng = np.random.default_rng(321)
        p_true, phi_true = 1.6, 3.0
        n = 3_000
        x1 = rng.normal(0, 1, n)
        sample_weight = rng.uniform(0.5, 2.0, n)
        mu = np.exp(1.5 + 0.25 * x1)
        y = _generate_weighted_tweedie(mu, phi_true, p_true, sample_weight, rng)
        X = pd.DataFrame({"x1": x1})

        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(
            model, X, y, sample_weight=sample_weight, method="grid", n_grid=15, p_bounds=(1.1, 1.9)
        )
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    @pytest.mark.slow
    def test_grid_with_reml(self):
        """method='grid' should work with fit_mode='fit_reml'."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        result = estimate_tweedie_p(
            model, X, y, method="grid", n_grid=10, fit_mode="fit_reml", p_bounds=(1.1, 1.9)
        )
        assert result.method == "grid"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.25)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(ValueError, match="method"):
            estimate_tweedie_p(model, X, y, method="bogus")

    def test_invalid_optimizer_raises(self):
        """Invalid optimizer should raise ValueError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(ValueError, match="optimizer"):
            estimate_tweedie_p(model, X, y, method="profile_opt", optimizer="bogus")

    def test_joint_ml_not_implemented(self):
        """method='joint_ml' should raise NotImplementedError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(NotImplementedError, match="joint_ml"):
            estimate_tweedie_p(model, X, y, method="joint_ml")

    def test_integrated_not_implemented(self):
        """method='integrated' should raise NotImplementedError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(NotImplementedError, match="integrated"):
            estimate_tweedie_p(model, X, y, method="integrated")


# =====================================================================
# Search trace
# =====================================================================


class TestSearchTrace:
    """Tests for search_trace output across methods."""

    def test_brent_has_trace(self):
        """Brent should produce a trace with expected columns."""
        X, y, _ = _tweedie_data(n=2000, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, p_bounds=(1.1, 1.9))
        trace = result.search_trace
        assert isinstance(trace, pd.DataFrame)
        expected_cols = {"step", "p", "phi", "nll", "n_iter", "fit_converged", "source"}
        assert expected_cols.issubset(set(trace.columns))
        assert len(trace) >= 3
        assert (trace["source"] == "brent").all()

    def test_grid_trace_len_matches_n_grid(self):
        """Grid trace should have exactly n_grid rows."""
        X, y, _ = _tweedie_data(n=2000, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        n_grid = 12
        result = estimate_tweedie_p(model, X, y, method="grid", n_grid=n_grid, p_bounds=(1.1, 1.9))
        assert len(result.search_trace) == n_grid
        assert (result.search_trace["source"] == "grid").all()

    def test_grid_refine_trace_has_both_sources(self):
        """Grid-refine trace should have coarse and refine sources."""
        X, y, _ = _tweedie_data(n=2000, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(
            model, X, y, method="grid_refine", n_grid_coarse=8, p_bounds=(1.1, 1.9)
        )
        sources = set(result.search_trace["source"].unique())
        assert "grid_coarse" in sources
        assert "brent_refine" in sources

    def test_profile_opt_trace_has_init(self):
        """Profile-opt trace should have init source (optimizer evals may be cached)."""
        X, y, _ = _tweedie_data(n=2000, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, method="profile_opt", p_bounds=(1.1, 1.9))
        sources = set(result.search_trace["source"].unique())
        assert "init" in sources
        # Optimizer evals may hit cached init points, so "optimizer" source
        # is not guaranteed but trace should have >= 3 init rows
        assert len(result.search_trace) >= 3

    def test_result_has_method_field(self):
        """All results should have method and phi_method set."""
        X, y, _ = _tweedie_data(n=2000, seed=7)
        for method in ("brent", "grid", "grid_refine", "profile_opt"):
            m = SuperGLM(
                family=TweedieDistribution(p=1.5),
                selection_penalty=0,
                features={"x1": Numeric()},
            )
            result = estimate_tweedie_p(m, X, y, method=method, p_bounds=(1.1, 1.9))
            assert result.method == method
            assert result.phi_method == "pearson"


# =====================================================================
# Method agreement
# =====================================================================


class TestMethodAgreement:
    """Cross-method agreement tests on clean synthetic data."""

    def test_brent_vs_grid_agree(self):
        """Brent and grid should agree within tolerance."""
        X, y, _ = _tweedie_data(n=3000)

        m1 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r1 = estimate_tweedie_p(m1, X, y, method="brent", p_bounds=(1.1, 1.9))

        m2 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r2 = estimate_tweedie_p(m2, X, y, method="grid", n_grid=30, p_bounds=(1.1, 1.9))

        np.testing.assert_allclose(r1.p_hat, r2.p_hat, atol=0.1)

    def test_grid_refine_vs_brent_agree(self):
        """Grid-refine and Brent should agree within tolerance."""
        X, y, _ = _tweedie_data(n=3000)

        m1 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r1 = estimate_tweedie_p(m1, X, y, method="brent", p_bounds=(1.1, 1.9))

        m2 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r2 = estimate_tweedie_p(
            m2, X, y, method="grid_refine", n_grid_coarse=10, p_bounds=(1.1, 1.9)
        )

        np.testing.assert_allclose(r1.p_hat, r2.p_hat, atol=0.1)

    @pytest.mark.parametrize("method", ["brent", "grid", "grid_refine", "profile_opt"])
    def test_all_methods_recover_p(self, method):
        """All profile methods should recover p from clean synthetic data."""
        X, y, p_true = _tweedie_data(n=3000)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        result = estimate_tweedie_p(model, X, y, method=method, p_bounds=(1.1, 1.9))
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)


# =====================================================================
# Deprecated .cache shim
# =====================================================================


class TestDeprecatedCache:
    def test_cache_property_returns_dict(self):
        """Deprecated .cache should return p→nll dict from search_trace."""
        X, y, _ = _tweedie_data(n=2000, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = estimate_tweedie_p(model, X, y, method="grid", n_grid=5, p_bounds=(1.1, 1.9))

        with pytest.warns(DeprecationWarning, match="cache.*deprecated"):
            cache = result.cache

        assert isinstance(cache, dict)
        assert len(cache) == 5
        for p_val, nll_val in cache.items():
            assert isinstance(p_val, float)
            assert isinstance(nll_val, float)
