"""Tweedie profile likelihood: density checks, p estimation and publication."""

import inspect
import math
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM, generate_tweedie_cpg, tweedie_logpdf
from superglm.distributions import Tweedie as TweedieDistribution
from superglm.distributions import clip_mu
from superglm.features.interaction import TensorInteraction
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.links import LogLink, stabilize_eta
from superglm.model import fit_ops as fit_ops_module
from superglm.model import profile_ops as profile_ops_module
from superglm.penalties.base import penalty_has_targets
from superglm.penalties.group_lasso import GroupLasso
from superglm.profiling._scalar import Interval, RecordedObjective
from superglm.profiling.tweedie import TweedieProfileResult, profile_phi_at, search_power
from superglm.solvers.pirls import PIRLSResult


def _generate_weighted_tweedie(mu, phi, p, weights, rng):
    """Simulate Tweedie responses under the prior-weight convention phi / w."""
    mu = np.asarray(mu, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return generate_tweedie_cpg(len(mu), mu=mu, phi=phi / weights, p=p, rng=rng)


def _profile_solver_result(dm, *, effective_df=1.0):
    """Minimal converged solver result for one-point profile dispatch spies."""
    return SimpleNamespace(
        beta=np.zeros(dm.shape[1], dtype=np.float64),
        intercept=0.0,
        effective_df=effective_df,
        n_iter=1,
        converged=True,
        iteration_log=[],
    )


def _snapshot_fitted_model(model, X, *, offset=None):
    """Capture exact caller state plus important top-level object identities."""
    prediction = model.predict(X, offset=offset).copy()
    return {
        "prediction": prediction,
        "identity": {name: id(value) for name, value in model.__dict__.items()},
        "state": pickle.dumps(model.__dict__, protocol=5),
    }


def _assert_fitted_model_unchanged(model, X, snapshot, *, offset=None):
    """Assert profiling preserved values, aliases, caches, and predictions."""
    np.testing.assert_allclose(model.predict(X, offset=offset), snapshot["prediction"])
    assert {name: id(value) for name, value in model.__dict__.items()} == snapshot["identity"]
    assert pickle.dumps(model.__dict__, protocol=5) == snapshot["state"]


# =====================================================================
# TestGenerateTweedieCPG
# =====================================================================


class TestGenerateTweedieCPG:
    def test_heterogeneous_mu(self):
        rng = np.random.default_rng(42)
        mu = rng.uniform(5, 50, size=10_000)
        y = generate_tweedie_cpg(10_000, mu=mu, phi=3.0, p=1.6, rng=rng)
        assert y.shape == (10_000,)
        assert np.all(y >= 0)

    @pytest.mark.slow
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

    @pytest.mark.parametrize(
        "invalid_weight",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
        ],
    )
    def test_invalid_weight_value_is_rejected(self, invalid_weight):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])
        weights = np.array([1.0, invalid_weight, 1.0])

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            tweedie_logpdf(y, mu, 2.0, 1.5, weights=weights)

    @pytest.mark.parametrize(
        "invalid_weights",
        [
            pytest.param(np.ones((3, 1)), id="two-dimensional"),
            pytest.param(np.ones(2), id="mismatched-length"),
        ],
    )
    def test_invalid_weight_shape_is_rejected(self, invalid_weights):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="one-dimensional with the same shape"):
            tweedie_logpdf(y, mu, 2.0, 1.5, weights=invalid_weights)

    @pytest.mark.parametrize(
        "invalid_y",
        [
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
        ],
    )
    def test_invalid_input_y_is_rejected(self, invalid_y):
        y = np.array([0.0, invalid_y, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="y must be finite and non-negative"):
            tweedie_logpdf(y, mu, 2.0, 1.5)

    @pytest.mark.parametrize(
        "invalid_mu",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
        ],
    )
    def test_invalid_input_mu_is_rejected(self, invalid_mu):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, invalid_mu, 2.5])

        with pytest.raises(ValueError, match="mu must be finite and strictly positive"):
            tweedie_logpdf(y, mu, 2.0, 1.5)

    @pytest.mark.parametrize(
        "invalid_p",
        [
            pytest.param(1.0, id="lower-bound"),
            pytest.param(2.0, id="upper-bound"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
        ],
    )
    def test_invalid_input_p_is_rejected(self, invalid_p):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="p must be in the open interval"):
            tweedie_logpdf(y, mu, 2.0, invalid_p)

    @pytest.mark.parametrize(
        ("y", "mu"),
        [
            pytest.param(np.array([0.0, 1.0, 2.0]), np.ones(2), id="different-lengths"),
            pytest.param(np.array([[0.0], [1.0]]), np.ones((2, 1)), id="two-dimensional"),
        ],
    )
    def test_invalid_input_y_mu_shape_is_rejected(self, y, mu):
        with pytest.raises(ValueError, match="one-dimensional with the same shape"):
            tweedie_logpdf(y, mu, 2.0, 1.5)

    @pytest.mark.parametrize(
        "invalid_phi",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
        ],
    )
    def test_invalid_input_phi_is_rejected(self, invalid_phi):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="phi must be finite and strictly positive"):
            tweedie_logpdf(y, mu, invalid_phi, 1.5)


# =====================================================================
# TestMaximumLikelihoodPhi
# =====================================================================


class TestMaximumLikelihoodPhi:
    """The dispersion at a known mean is the maximiser of the density's likelihood."""

    def test_weighted_phi_recovery(self):
        rng = np.random.default_rng(123)
        n = 12_000
        mu = np.full(n, 10.0)
        phi_true, p = 3.0, 1.6
        weights = rng.uniform(0.5, 2.0, n)
        y = _generate_weighted_tweedie(mu, phi_true, p, weights, rng)

        phi_hat = profile_phi_at(y, mu, weights, p).phi
        np.testing.assert_allclose(phi_hat, phi_true, rtol=0.12)

    def test_mle_phi_recovery(self):
        rng = np.random.default_rng(456)
        n = 20_000
        mu = np.full(n, 10.0)
        phi_true, p = 3.0, 1.6
        y = generate_tweedie_cpg(n, mu=mu, phi=phi_true, p=p, rng=rng)

        phi_hat = profile_phi_at(y, mu, np.ones(n), p).phi
        np.testing.assert_allclose(phi_hat, phi_true, rtol=0.12)

    def test_near_one_multimodal_dispersion_reaches_the_global_minimum(self):
        """The Newton root from the saddlepoint start is the local minimum at phi = 35.94."""
        p = 1.0181533410437358
        y = np.array([1.81787899, 11275.9262, 0.0, 0.00306563885, 0.0000232882792, 1.18207511])
        mu = np.array(
            [0.0000253947806, 44091.7359, 198.869667, 0.000051937831, 331.859132, 0.0054422757]
        )
        weights = np.array(
            [83.2444169, 0.17590785, 2.31976211, 463.433307, 2.50852264, 0.416322332]
        )

        solved = profile_phi_at(y, mu, weights, p)

        # Master's pinned global optimum of the exact profile.
        np.testing.assert_allclose(solved.phi, 31.731271940671984, rtol=2e-7)
        np.testing.assert_allclose(solved.criterion / y.size, 185.18683913586867, atol=1e-9)

    def test_a_power_too_close_to_one_is_skipped_as_infeasible(self):
        """The power search routes around a power whose phi profile it cannot search."""
        import pandas as pd

        from superglm.profiling.tweedie import _PowerProfile

        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        y = rng.poisson(np.exp(0.3 + 0.2 * x)).astype(float)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x": Numeric()}
        )
        profile = _PowerProfile(model, pd.DataFrame({"x": x}), y, np.ones(y.size), None, "fit")
        assert profile(1.0 + 1e-8) == math.inf
        assert "too close to 1" in profile.infeasible[1.0 + 1e-8]

    def test_rows_sharing_a_peak_index_bend_the_profile_together(self):
        """Tiled 600 times, Q is 600 times the fixture's: the same minima, the same answer.

        Its 3,000 positive rows are past the 0.878 / (p - 1)^2 a single row's lattice
        term needs, but each row shares its phase with 599 others, and together
        they bend Q as the fixture's single rows do.
        """
        p = 1.0181533410437358
        y = np.array([1.81787899, 11275.9262, 0.0, 0.00306563885, 0.0000232882792, 1.18207511])
        mu = np.array(
            [0.0000253947806, 44091.7359, 198.869667, 0.000051937831, 331.859132, 0.0054422757]
        )
        weights = np.array(
            [83.2444169, 0.17590785, 2.31976211, 463.433307, 2.50852264, 0.416322332]
        )

        solved = profile_phi_at(np.tile(y, 600), np.tile(mu, 600), np.tile(weights, 600), p)

        np.testing.assert_allclose(solved.phi, 31.731271940671984, rtol=2e-7)
        np.testing.assert_allclose(solved.criterion / (600 * y.size), 185.18683913586867, atol=1e-9)


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

        result = model.estimate_p(X, y, p_bounds=(1.1, 1.9))
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
        result = model.estimate_p(X, y, p_bounds=(1.1, 1.9))
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    def test_prior_weight_mle_p_phi_recovery(self):
        """Exact-MLE profiling should recover p when prior weights act through phi / w."""
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

        result = model.estimate_p(X, y, sample_weight=sample_weight, p_bounds=(1.1, 1.9))
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)
        np.testing.assert_allclose(result.phi_hat, phi_true, rtol=0.2)
        assert result.converged

    def test_notebook_style_profile_recovers_true_p_under_prior_weights(self):
        """Notebook-style exposure weights should not bias the profile downward."""
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

        result = model.estimate_p(X, y, sample_weight=sample_weight, p_bounds=(1.1, 1.9))
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

        result = model.estimate_p(
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
            model.estimate_p(X, y)


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


def _offset_spline_tweedie_data(n=72, seed=20260720):
    """Small offset-aware Tweedie sample for final-refit state tests."""
    rng = np.random.default_rng(seed)
    x1 = np.linspace(-1.0, 1.0, n)
    offset = 0.25 * np.sin(np.pi * x1)
    mu = np.exp(0.6 + 0.35 * x1 + offset)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.47, rng=rng)
    sample_weight = rng.uniform(0.75, 1.25, n)
    return pd.DataFrame({"x1": x1}), y, sample_weight, offset


def _deterministic_profile_result(objective=None):
    return TweedieProfileResult(
        p_hat=1.47,
        phi_hat=7.25,
        nll=0.0,
        converged=True,
        fit_mode="fit",
        evaluations=pd.DataFrame({"p": [1.47], "nll": [0.0], "phi": [7.25]}),
        warnings=[],
        search_nll=0.0,
        _objective=objective,
        _ll_scale=1.0,
        _ci_bounds=(1.02, 1.98),
    )


def _reference_offset_tweedie_null_mu(y, sample_weight, offset, distribution):
    """Evaluate the closed-form Tweedie log-link intercept-score root."""
    y_arr = np.asarray(y, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    offset_arr = np.asarray(offset, dtype=np.float64)
    p = float(distribution.p)
    numerator = np.sum(weights * y_arr * np.exp((1.0 - p) * offset_arr))
    denominator = np.sum(weights * np.exp((2.0 - p) * offset_arr))
    intercept = float(np.log(numerator / denominator))
    link = LogLink()
    eta = stabilize_eta(intercept + offset_arr, link)
    return clip_mu(link.inverse(eta), distribution)


@pytest.mark.parametrize("function", [SuperGLM.estimate_p, profile_ops_module.estimate_p])
def test_public_tweedie_profile_entry_points_default_to_lazy_ci(function):
    signature = inspect.signature(function)

    assert signature.parameters["ci_alpha"].default is None


class TestEstimatePFitMode:
    def test_reml_mode_rejects_selection_before_profile_work(self, monkeypatch):
        X, y, _ = _tweedie_data(n=24, seed=20260721)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty="auto",
            features={"x1": Numeric()},
        )
        profile_calls = []

        def unexpected_profile(*args, **kwargs):
            profile_calls.append(True)
            raise AssertionError("profile work must not start")

        monkeypatch.setattr(tweedie_module, "search_power", unexpected_profile)

        with pytest.raises(ValueError, match="does not support selection penalties"):
            model.estimate_p(X, y, fit_mode="reml")

        assert profile_calls == []

    @pytest.mark.parametrize("already_fitted", [False, True])
    def test_profile_publication_preserves_subclass_configuration_aliases(
        self, monkeypatch, already_fitted
    ):
        class ConfigAliasedSuperGLM(SuperGLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.config_alias = self._config
                self.family_alias = self._family_config

        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = ConfigAliasedSuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = _deterministic_profile_result()
        monkeypatch.setattr(tweedie_module, "search_power", lambda *args, **kwargs: result)
        if already_fitted:
            model.fit(X, y, sample_weight=sample_weight, offset=offset)
        config_revision = model._config_revision

        model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode="fit",
        )

        assert model.config_alias is model._config
        assert model.family_alias is model._family_config
        assert model._config_revision == config_revision + 1
        assert model.family.p == pytest.approx(result.p_hat)

    @pytest.mark.parametrize("fit_mode", ["fit", "reml"])
    @pytest.mark.parametrize("retain_fit_state", [True, False])
    def test_final_profile_refit_atomically_synchronizes_model_state(
        self, monkeypatch, fit_mode, retain_fit_state
    ):
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            retain_fit_state=retain_fit_state,
            features={"x1": Spline(n_knots=5, penalty="ssp")},
        )
        result = _deterministic_profile_result()
        monkeypatch.setattr(tweedie_module, "search_power", lambda *args, **kwargs: result)

        fit_name = "_fit_reml_in_workspace" if fit_mode == "reml" else "_fit_in_workspace"
        real_final_fit = getattr(fit_ops_module, fit_name)
        captured = {}

        def final_fit_with_primed_caches(candidate, *args, **kwargs):
            captured["retain_during_fit"] = candidate._retain_fit_state
            if fit_mode == "reml":
                kwargs["max_reml_iter"] = 3
            fitted = real_final_fit(candidate, *args, **kwargs)
            captured["public_result"] = candidate.result
            captured["solver_result"] = candidate._solver_pirls_result()
            captured["reml_result"] = candidate._reml_result
            captured["reml_lambdas"] = candidate._reml_lambdas
            captured["reml_penalties"] = candidate._reml_penalties
            captured["fit_meta"] = candidate._last_fit_meta
            captured["runtime_state"] = candidate._runtime_canonical_state
            captured["prediction_plan"] = candidate._prediction_plan
            captured["fast_prediction_state"] = candidate._fast_prediction_state
            # Before the fix, retain=False has already released these rows.  Skip
            # cache priming so the regression fails at the production access.
            if candidate._dm is None:
                return fitted

            solver = captured["solver_result"]
            # Keep the metadata comparison independent of publication's
            # intentional ownership copy; the unit test below checks that the
            # phi-only replacement itself preserves dynamic values by identity.
            captured["public_dynamic_metadata"] = ("public-profile-metadata",)
            captured["solver_dynamic_metadata"] = (1.0, 2.0)
            candidate.result.profile_sync_metadata = captured["public_dynamic_metadata"]
            solver.profile_sync_metadata = captured["solver_dynamic_metadata"]
            if candidate._reml_result is not None:
                captured["reml_dynamic_metadata"] = ("reml-profile-metadata",)
                candidate._reml_result.profile_sync_metadata = captured["reml_dynamic_metadata"]
            eta = candidate._dm.matvec(solver.beta) + solver.intercept + candidate._fit_offset
            eta = stabilize_eta(eta, candidate._link)
            captured["solver_mu"] = clip_mu(candidate._link.inverse(eta), candidate._distribution)
            captured["old_covariance"] = candidate._coef_covariance
            captured["old_active_info"] = candidate._fit_active_info
            captured["old_inference_info"] = candidate._fit_inference_info
            captured["old_group_edf"] = candidate._group_edf
            captured["old_metrics"] = candidate.metrics(
                X, y, sample_weight=sample_weight, offset=offset
            )
            captured["old_summary"] = candidate.summary()
            return fitted

        monkeypatch.setattr(fit_ops_module, fit_name, final_fit_with_primed_caches)
        real_release = fit_ops_module._maybe_release_fit_state
        release_events = []

        def release_spy(candidate):
            release_events.append(candidate._retain_fit_state)
            if not candidate._retain_fit_state:
                captured["pre_release_null_mu"] = candidate._fit_null_mu.copy()
                captured["pre_release_fit_stats"] = candidate._fit_stats
                captured["pre_release_summary"] = candidate.summary()
            return real_release(candidate)

        monkeypatch.setattr(fit_ops_module, "_maybe_release_fit_state", release_spy)

        returned = model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode=fit_mode,
        )

        assert returned is result
        assert captured["retain_during_fit"] is True
        assert model._retain_fit_state is retain_fit_state
        assert model._tweedie_profile_result is not result
        assert model._tweedie_profile_result._ci_cache is not result._ci_cache
        # The model's compaction runs after the published dispersion is
        # restated, so the released covariance is scaled at that phi.
        expected_release_flags = [True] if retain_fit_state else [True, False]
        assert release_events == expected_release_flags

        assert model.family.p == pytest.approx(result.p_hat)
        assert model.distribution_.p == pytest.approx(result.p_hat)
        assert model._fit_state.distribution is model._distribution
        assert model.result.phi == pytest.approx(result.phi_hat)
        assert model._solver_pirls_result().phi == pytest.approx(result.phi_hat)
        assert model.result is not captured["public_result"]
        assert model._solver_pirls_result() is not captured["solver_result"]
        np.testing.assert_array_equal(model.result.beta, captured["public_result"].beta)
        np.testing.assert_array_equal(
            model._solver_pirls_result().beta, captured["solver_result"].beta
        )
        assert not np.shares_memory(model.result.beta, captured["public_result"].beta)
        assert not np.shares_memory(
            model._solver_pirls_result().beta, captured["solver_result"].beta
        )
        assert model.result.profile_sync_metadata == captured["public_dynamic_metadata"]
        assert (
            model._solver_pirls_result().profile_sync_metadata
            == captured["solver_dynamic_metadata"]
        )
        assert captured["public_result"].phi != pytest.approx(result.phi_hat)
        assert captured["solver_result"].phi != pytest.approx(result.phi_hat)

        assert model._last_fit_meta is captured["fit_meta"]
        assert model._runtime_canonical_state is captured["runtime_state"]
        assert model._prediction_plan is captured["prediction_plan"]
        assert model._fast_prediction_state is captured["fast_prediction_state"]
        if fit_mode == "reml":
            assert model._reml_result is not captured["reml_result"]
            assert model._reml_result.pirls_result is model._solver_pirls_result()
            assert model._reml_result.pirls_result.phi == pytest.approx(result.phi_hat)
            assert model._reml_result.lambdas == captured["reml_result"].lambdas
            assert model._reml_result.lambda_history is captured["reml_result"].lambda_history
            assert model._reml_lambdas is captured["reml_lambdas"]
            assert model._reml_penalties is captured["reml_penalties"]
            assert model._reml_result.profile_sync_metadata == captured["reml_dynamic_metadata"]
        else:
            assert model._reml_result is None

        np.testing.assert_allclose(
            model.predict(X, offset=offset), captured["solver_mu"], rtol=1e-10, atol=1e-10
        )
        expected_ll = model._distribution.log_likelihood(
            y, captured["solver_mu"], sample_weight, result.phi_hat
        )
        assert model._fit_stats.log_likelihood == pytest.approx(expected_ll)
        reference_null_mu = _reference_offset_tweedie_null_mu(
            y, sample_weight, offset, model._distribution
        )
        expected_null_ll = model._distribution.log_likelihood(
            y, reference_null_mu, sample_weight, result.phi_hat
        )
        expected_null_deviance = float(
            np.sum(sample_weight * model._distribution.deviance_unit(y, reference_null_mu))
        )
        expected_deviance = float(
            np.sum(sample_weight * model._distribution.deviance_unit(y, captured["solver_mu"]))
        )
        expected_explained_deviance = 1.0 - expected_deviance / expected_null_deviance
        assert model._fit_stats.null_log_likelihood == pytest.approx(expected_null_ll)
        assert model._fit_stats.null_deviance == pytest.approx(expected_null_deviance)
        assert model._fit_stats.explained_deviance == pytest.approx(expected_explained_deviance)

        if retain_fit_state:
            np.testing.assert_allclose(model._fit_mu, captured["solver_mu"])
            np.testing.assert_allclose(
                model._fit_null_mu, reference_null_mu, rtol=1e-10, atol=1e-10
            )
            for cache_name in (
                "_coef_covariance",
                "_fit_active_info",
                "_fit_inference_info",
                "_group_edf",
            ):
                assert cache_name not in model.__dict__
            expected_covariance = (
                result.phi_hat / captured["solver_result"].phi * captured["old_covariance"][0]
            )
            np.testing.assert_allclose(model._coef_covariance[0], expected_covariance)
        else:
            np.testing.assert_allclose(
                captured["pre_release_null_mu"], reference_null_mu, rtol=1e-10, atol=1e-10
            )
            assert captured["pre_release_fit_stats"] is model._fit_stats
            pre_release_summary = captured["pre_release_summary"]
            assert pre_release_summary["information_criteria"][
                "null_log_likelihood"
            ] == pytest.approx(expected_null_ll)
            assert pre_release_summary["deviance"]["null_deviance"] == pytest.approx(
                expected_null_deviance
            )
            assert pre_release_summary["deviance"]["explained_deviance"] == pytest.approx(
                expected_explained_deviance
            )
            for released_name in (
                "_dm",
                "_fit_weights",
                "_fit_offset",
                "_fit_mu",
                "_fit_null_mu",
                "_fit_X_ref",
                "_fit_y_ref",
                "_fit_sample_weight_ref",
                "_fit_offset_ref",
            ):
                assert getattr(model, released_name) is None
            assert model.__dict__["_fit_inference_info"] is not captured["old_inference_info"]
            expected_covariance = (
                result.phi_hat * captured["old_inference_info"]["XtWX_inv_aug"][1:, 1:]
            )
            np.testing.assert_allclose(model._coef_covariance[0], expected_covariance)

        assert model._fit_metrics_cache is None
        assert model._fit_metrics_cache_signature is None
        assert model._summary_cache is None
        fresh_metrics = model.metrics(X, y, sample_weight=sample_weight, offset=offset)
        assert fresh_metrics is not captured["old_metrics"]
        assert np.isfinite(fresh_metrics.log_likelihood)
        fresh_summary = model.summary()
        assert fresh_summary is not captured["old_summary"]
        assert fresh_summary["information_criteria"]["null_log_likelihood"] == pytest.approx(
            expected_null_ll
        )
        assert fresh_summary["deviance"]["null_deviance"] == pytest.approx(expected_null_deviance)
        assert fresh_summary["deviance"]["explained_deviance"] == pytest.approx(
            expected_explained_deviance
        )

        # The returned handle may remain convenient and cache-compatible, but
        # mutating its public estimate fields must not rewrite installed fit
        # provenance or the model's reporting state. The installed phi is the
        # dispersion re-profiled against the published refit, not the stub's
        # 7.25 -- capture it symbolically rather than pinning the stub value.
        installed_phi = float(model._tweedie_profile_result.phi_hat)
        returned.p_hat = 1.91
        returned.phi_hat = 91.0
        returned._ci_cache[0.05] = (1.85, 1.95)
        assert model._tweedie_profile_result.p_hat == pytest.approx(1.47)
        assert model._tweedie_profile_result.phi_hat == pytest.approx(installed_phi)
        assert model._tweedie_profile_result._ci_cache == {}
        immutable_summary = model.summary()
        assert immutable_summary._info["tweedie_p"] == pytest.approx(1.47)
        assert immutable_summary._info["tweedie_phi"] == pytest.approx(installed_phi)
        assert immutable_summary._info["tweedie_p_ci_status"] == "not computed"

    @pytest.mark.parametrize("fit_mode", ["fit", "reml"])
    @pytest.mark.parametrize("retain_fit_state", [True, False])
    def test_final_profile_refit_failure_restores_retention_without_installing_result(
        self, monkeypatch, fit_mode, retain_fit_state
    ):
        X, y, sample_weight, offset = _offset_spline_tweedie_data(n=24)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            retain_fit_state=retain_fit_state,
            features={"x1": Spline(n_knots=5, penalty="ssp")},
        )
        result = _deterministic_profile_result()
        monkeypatch.setattr(tweedie_module, "search_power", lambda *args, **kwargs: result)
        seen_retain_flags = []
        config_before = model._config
        family_before = model._family_config
        config_revision_before = model._config_revision
        fit_revision_before = model._fit_revision

        def failing_final_fit(candidate, *args, **kwargs):
            seen_retain_flags.append(candidate._retain_fit_state)
            raise RuntimeError("final refit failed")

        fit_name = "_fit_reml_in_workspace" if fit_mode == "reml" else "_fit_in_workspace"
        monkeypatch.setattr(fit_ops_module, fit_name, failing_final_fit)

        with pytest.raises(RuntimeError, match="final refit failed"):
            model.estimate_p(
                X,
                y,
                sample_weight=sample_weight,
                offset=offset,
                fit_mode=fit_mode,
            )

        assert seen_retain_flags == [True]
        assert model._retain_fit_state is retain_fit_state
        assert model._tweedie_profile_result is None
        assert model._config is config_before
        assert model._family_config is family_before
        assert model._config_revision == config_revision_before
        assert model._fit_revision == fit_revision_before

    def test_pirls_phi_replacement_preserves_declared_and_dynamic_state(self):
        beta = np.array([0.25, -0.5])
        original = PIRLSResult(
            beta=beta,
            intercept=1.25,
            n_iter=4,
            deviance=3.5,
            converged=True,
            phi=0.75,
            effective_df=2.0,
            iteration_log=[],
        )
        original.scop_states = {"smooth": np.array([1.0, 2.0])}
        original.future_metadata = np.array([3.0, 4.0])

        replacement = profile_ops_module._replace_pirls_phi(original, 7.25)

        assert replacement is not original
        assert replacement.phi == pytest.approx(7.25)
        assert original.phi == pytest.approx(0.75)
        assert replacement.beta is beta
        assert replacement.iteration_log is original.iteration_log
        assert replacement.scop_states is original.scop_states
        assert replacement.future_metadata is original.future_metadata

    def test_public_estimate_is_lazy_about_ci_and_profile_evaluations(self, monkeypatch):
        X, y, _ = _tweedie_data(n=48, seed=20260804)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        objective_calls = []

        def objective(p):
            objective_calls.append(float(p))
            return (float(p) - 1.5) ** 2

        recorded = RecordedObjective(objective)
        recorded(1.4)
        recorded(1.5)
        objective_calls.clear()
        result = TweedieProfileResult(
            p_hat=1.5,
            phi_hat=1.0,
            nll=0.0,
            converged=True,
            fit_mode="fit",
            evaluations=pd.DataFrame({"p": [1.4, 1.5], "nll": [0.01, 0.0]}),
            warnings=[],
            search_nll=0.0,
            _objective=recorded,
            _ll_scale=float(len(y)),
            _ci_bounds=(1.02, 1.98),
        )
        profiler_kwargs = {}

        def fake_search_power(*args, **kwargs):
            profiler_kwargs.update(kwargs)
            return result

        def unexpected_interval(*args, **kwargs):
            raise AssertionError("public estimate_p must not compute a profile CI eagerly")

        monkeypatch.setattr(tweedie_module, "search_power", fake_search_power)
        monkeypatch.setattr(result, "interval", unexpected_interval)
        progress_events = []

        returned = model.estimate_p(
            X,
            y,
            progress_callback=lambda phase, payload: progress_events.append((phase, payload)),
        )

        assert returned is result
        assert profiler_kwargs["fit_mode"] == "fit"
        assert [phase for phase, _ in progress_events] == ["best_found", "final_refit"]
        assert all(
            payload["profile_estimate"]["ci_status"] == "not computed"
            for _, payload in progress_events
        )
        assert result._ci_cache == {}
        assert objective_calls == []

        interval = TweedieProfileResult.interval(result, alpha=0.05)
        assert result._ci_cache[0.05] is interval

        summary = model.summary(alpha=0.05)
        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"

        installed_interval = TweedieProfileResult.interval(
            model._tweedie_profile_result,
            alpha=0.05,
        )
        # The two share the recorded objective, so the second root search
        # starts from more evaluations; both roots are located to 1e-4.
        assert (installed_interval.lower, installed_interval.upper) == pytest.approx(
            (interval.lower, interval.upper), abs=2e-4
        )
        assert installed_interval is not interval
        summary = model.summary(alpha=0.05)
        assert summary._info["tweedie_p_ci"] == (installed_interval.lower, installed_interval.upper)
        assert summary._info["tweedie_p_ci_status"] == "available"

    def test_explicit_ci_alpha_populates_returned_and_installed_summary_caches(
        self,
        monkeypatch,
    ):
        X, y, _ = _tweedie_data(n=48, seed=20260805)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = _deterministic_profile_result()
        ci_calls = []
        expected_interval = (1.31, 1.62)

        def compute_interval(alpha=0.05):
            alpha_value = float(alpha)
            ci_calls.append(alpha_value)
            result._ci_cache[alpha_value] = Interval(*expected_interval, False, False)
            return result._ci_cache[alpha_value]

        monkeypatch.setattr(tweedie_module, "search_power", lambda *args, **kwargs: result)
        monkeypatch.setattr(result, "interval", compute_interval)

        returned = model.estimate_p(X, y, ci_alpha=0.05)
        installed = model._tweedie_profile_result

        assert returned is result
        assert ci_calls == [0.05]
        assert installed._ci_cache[0.05] == returned._ci_cache[0.05]
        assert returned._ci_cache is not installed._ci_cache

        returned._ci_cache[0.05] = Interval(1.8, 1.9, False, False)
        summary = model.summary(alpha=0.05)
        assert summary._info["tweedie_p_ci"] == pytest.approx(expected_interval)
        assert summary._info["tweedie_p_ci_status"] == "available"
        other_alpha = model.summary(alpha=0.10)
        assert other_alpha._info["tweedie_p_ci"] is None
        assert other_alpha._info["tweedie_p_ci_status"] == "not computed"

    @pytest.mark.parametrize(
        "ci_alpha",
        [0.0, 1.0, np.nan, np.inf, True, np.array([0.05, 0.10])],
    )
    def test_invalid_ci_alpha_is_rejected_before_profile_work(
        self,
        monkeypatch,
        ci_alpha,
    ):
        X, y, _ = _tweedie_data(n=24, seed=20260806)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        profile_calls = []

        def unexpected_profile(*args, **kwargs):
            profile_calls.append(True)
            raise AssertionError("profile work must not start")

        monkeypatch.setattr(tweedie_module, "search_power", unexpected_profile)

        with pytest.raises(ValueError):
            model.estimate_p(X, y, ci_alpha=ci_alpha)

        assert profile_calls == []

    @pytest.mark.parametrize("xatol", [0.0, -1e-3, np.nan, np.inf])
    def test_invalid_xatol_is_rejected_before_profile_work(self, monkeypatch, xatol):
        X, y, _ = _tweedie_data(n=24, seed=20260806)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        def unexpected_profile(*args, **kwargs):
            raise AssertionError("profile work must not start")

        monkeypatch.setattr(tweedie_module, "search_power", unexpected_profile)

        with pytest.raises(ValueError, match="xatol must be finite and strictly positive"):
            model.estimate_p(X, y, xatol=xatol)

    def test_ci_failure_preserves_previously_fitted_revision(self, monkeypatch):
        X, y, _ = _tweedie_data(n=48, seed=20260808)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        model.fit(X, y)
        snapshot = _snapshot_fitted_model(model, X)
        result = _deterministic_profile_result()

        def failing_interval(alpha=0.05):
            raise RuntimeError(f"CI failed at alpha={alpha}")

        monkeypatch.setattr(tweedie_module, "search_power", lambda *args, **kwargs: result)
        monkeypatch.setattr(result, "interval", failing_interval)

        with pytest.raises(RuntimeError, match="CI failed"):
            model.estimate_p(X, y, ci_alpha=0.05)

        _assert_fitted_model_unchanged(model, X, snapshot)

    def test_invalid_complex_weight_is_rejected_before_feature_auto_detection(self):
        X, y, _ = _tweedie_data(n=24, seed=20260719)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            splines=[],
        )
        invalid_weights = np.ones(len(y), dtype=np.complex128)
        invalid_weights[3] = 1.0 + 1.0j
        family_before = model._family_config
        config_before = model._config

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            model.estimate_p(
                X,
                y,
                sample_weight=invalid_weights,
                fit_mode="fit",
            )

        assert model._family_config is family_before
        assert model._config is config_before
        assert model._specs == {}
        assert model._feature_order == []

    @pytest.mark.parametrize("fit_mode", ["fit", "reml"])
    def test_invalid_weight_is_rejected_before_feature_auto_detection(self, fit_mode):
        X, y, _ = _tweedie_data(n=24, seed=20260716)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            splines=[],
        )
        invalid_weights = np.ones(len(y) - 1)
        family_before = model._family_config
        config_before = model._config
        result_before = model._result
        distribution_before = model._distribution
        specs_before = dict(model._specs)
        feature_order_before = list(model._feature_order)

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            model.estimate_p(
                X,
                y,
                sample_weight=invalid_weights,
                fit_mode=fit_mode,
            )

        assert model._family_config is family_before
        assert model._config is config_before
        assert model._result is result_before
        assert model._distribution is distribution_before
        assert model._specs == specs_before
        assert model._feature_order == feature_order_before

    @pytest.mark.parametrize("fit_mode", ["fit", "reml"])
    def test_invalid_weight_preserves_existing_profile_model_state(self, fit_mode):
        X, y, _ = _tweedie_data(n=80, seed=20260717)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        model.fit(X, y)
        invalid_weights = np.ones(len(y) - 1)
        family_before = model._family_config
        config_before = model._config
        result_before = model._result
        distribution_before = model._distribution
        profile_result_before = model._tweedie_profile_result
        prediction_before = model.predict(X)

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            model.estimate_p(
                X,
                y,
                sample_weight=invalid_weights,
                fit_mode=fit_mode,
            )

        assert model._family_config is family_before
        assert model._config is config_before
        assert model._result is result_before
        assert model._distribution is distribution_before
        assert model._tweedie_profile_result is profile_result_before
        np.testing.assert_allclose(model.predict(X), prediction_before)

    def test_invalid_zero_weight_is_rejected_by_ordinary_tweedie_fit(self):
        X, y, _ = _tweedie_data(n=40, seed=20260718)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        weights = np.ones(len(y))
        weights[5] = 0.0

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            model.fit(X, y, sample_weight=weights)

    def test_invalid_tweedie_weight_rule_does_not_reject_poisson_zero_weight(self):
        X = pd.DataFrame({"x1": np.linspace(-1.0, 1.0, 12)})
        y = np.array([0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0])
        weights = np.ones(len(y))
        weights[5] = 0.0
        model = SuperGLM(family="poisson", selection_penalty=0, features={"x1": Numeric()})

        model.fit(X, y, sample_weight=weights)

        assert np.all(np.isfinite(model.predict(X)))

    def test_unweighted_cpg_default_mle_p_phi_recovery(self):
        """The real public default-MLE fit path should recover p."""
        X, y, p_true = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = model.estimate_p(X, y, fit_mode="fit")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)
        np.testing.assert_allclose(result.phi_hat, 3.0, rtol=0.15)
        assert result.converged
        assert result.warnings == []
        # Model should be refitted with estimated p
        assert model.family.p == result.p_hat
        assert model._result is not None
        assert model._last_fit_meta["method"] == "fit"

    @pytest.mark.slow
    def test_fit_mode_reml_recovers_p(self):
        """fit_mode='reml' should recover p using REML fits, agreeing with fit."""
        X, y, p_true = _tweedie_data(n=600)
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

        # The truth is log-linear, inside the spline penalty's null space, so
        # REML shrinks the spline to the Numeric fit and the two profiles are one
        # curve: measured gap 3.9e-6.  Each Brent search resolves its optimum to
        # xatol=1e-3, so 5e-3 bounds the pair.  A REML side that profiles phi
        # differently, or at a power 0.02 off, opens a ~0.02 gap while p still
        # recovers within 0.2; the old 0.3 bound let both through.
        result_fit = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        ).estimate_p(X, y, fit_mode="fit")
        np.testing.assert_allclose(result_fit.p_hat, result.p_hat, atol=5e-3)

    @pytest.mark.slow
    def test_flexible_spline_reml_mle_p_phi_recovery(self):
        """fit_mode='reml' recovers p and the maximum-likelihood phi."""
        X, y, p_true = _tweedie_data(n=1_500, seed=11)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        result = model.estimate_p(X, y, fit_mode="reml")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.25)
        np.testing.assert_allclose(result.phi_hat, 3.0, rtol=0.20)
        assert result.converged
        winning_row = result.evaluations.iloc[result.evaluations["nll"].to_numpy().argmin()]
        assert winning_row["p"] == pytest.approx(result.p_hat)
        assert model.result.phi == pytest.approx(result.phi_hat, rel=1e-12, abs=1e-12)
        assert model._last_fit_meta["method"] == "fit_reml"

    @pytest.mark.slow
    def test_fit_mode_reml_profile_ci_leaves_final_fit_state(self):
        """An explicit later CI should not leave the fitted model at a CI probe p."""
        X, y, _ = _tweedie_data(n=600, seed=17)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=5, penalty="ssp")},
        )

        result = model.estimate_p(
            X,
            y,
            fit_mode="reml",
            p_bounds=(1.3, 1.75),
            xatol=1e-4,
        )

        assert result._ci_cache == {}
        result.ci(alpha=0.05)
        assert 0.05 in result._ci_cache
        assert model.family.p == pytest.approx(result.p_hat)
        assert model._distribution.p == pytest.approx(result.p_hat)

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

    def test_fit_mode_inherit_from_fit_path_falls_back_to_fit(self, monkeypatch):
        """After fit_path(), inherit should profile with ordinary ML fits."""
        X, y, _ = _tweedie_data(n=80, seed=20260803)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        model._last_fit_meta = {"method": "fit_path", "discrete": False}
        calls: list[str] = []

        def fake_search_power(*args, **kwargs):
            calls.append(kwargs["fit_mode"])
            result = _deterministic_profile_result()
            result.p_hat = 1.45
            return result

        monkeypatch.setattr("superglm.profiling.tweedie.search_power", fake_search_power)

        result = model.estimate_p(X, y, fit_mode="inherit")

        assert result.p_hat == 1.45
        assert calls == ["fit"]
        assert model._last_fit_meta["method"] == "fit"

    def test_reverse_coupling_preflights_the_publication_regime(self, monkeypatch):
        """fit_mode='fit' with search_fit_mode='reml' on a model whose
        features require REML (a lambda_policy here) is a guaranteed
        publication failure: the ordinary final fit rejects them. That must
        be refused before the multi-candidate REML search burns its whole
        budget, not discovered after it."""
        from superglm import LambdaPolicy, Spline

        def search_must_not_run(*args, **kwargs):
            raise AssertionError("the REML search ran before the publication preflight")

        monkeypatch.setattr(tweedie_module, "search_power", search_must_not_run)
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"x1": rng.uniform(0.0, 1.0, 200)})
        y = rng.gamma(1.2, 2.0, 200)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            features={
                "x1": Spline(
                    kind="cr", n_knots=5, lambda_policy=LambdaPolicy(mode="fixed", value=1.0)
                )
            },
        )

        with pytest.raises(NotImplementedError, match="lambda_policy"):
            model.estimate_p(X, y, fit_mode="fit", search_fit_mode="reml")

    @pytest.mark.slow
    def test_fit_mode_inherit_from_reml(self):
        """After fit_reml(), inherit should use the REML path."""
        X, y, p_true = _tweedie_data(n=600)
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

    def test_wrong_family_raises(self):
        """Non-Tweedie model should raise immediately."""
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        model = SuperGLM(family="poisson", selection_penalty=0, features={"x": Numeric()})
        with pytest.raises(ValueError, match="tweedie"):
            model.estimate_p(X, y)


class TestDecoupledSearchFitMode:
    """The power search and the published fit can use different regimes."""

    @staticmethod
    def _reml_model():
        return SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=4)},
        )

    def test_ml_search_publishes_a_reml_fit_without_searching_under_reml(self):
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = self._reml_model()

        result = model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="fit",
        )

        # The published convergence describes the publication refit -- a REML
        # fit -- even though the search itself was ML.
        assert result.converged is True
        assert result.fit_mode == "fit_reml"
        # The publication is a REML fit, and the model agrees.
        assert model.reml_diagnostics()["converged"] is True
        assert model.reml_diagnostics()["lambdas"]

    def test_published_dispersion_matches_the_returned_estimate(self):
        """The result must describe the model it published, not the search."""
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = self._reml_model()

        result = model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="fit",
        )

        assert model.result.phi == pytest.approx(result.phi_hat)
        assert model.family.p == pytest.approx(result.p_hat)

    def test_decoupled_dispersion_is_reprofiled_off_the_published_fit(self):
        """Carrying the ML search's phi across the mode switch would be wrong."""
        X, y, sample_weight, offset = _offset_spline_tweedie_data()

        searched = self._reml_model().estimate_p(
            X, y, sample_weight=sample_weight, offset=offset, fit_mode="fit"
        )
        decoupled = self._reml_model().estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="fit",
        )

        assert decoupled.p_hat == pytest.approx(searched.p_hat)
        assert decoupled.phi_hat != pytest.approx(searched.phi_hat, rel=1e-12)

    def test_a_coupled_run_defaults_reprofiles_and_still_inverts(self, subtests):
        """Three properties of a coupled REML run, checked on one search.

        The search is the expensive part, and each of these used to pay for
        its own copy of the same one.
        """
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        result = self._reml_model().estimate_p(
            X, y, sample_weight=sample_weight, offset=offset, fit_mode="reml"
        )

        with subtests.test("the default leaves the coupled REML path unchanged"):
            # An omitted search_fit_mode resolves to the publication mode.
            assert result.fit_mode == "fit_reml"

        with subtests.test("a coupled search also reprofiles against its publication"):
            # The publication refit runs at the tight publication tolerance
            # while candidates ran at the search bar, so the published
            # dispersion is re-profiled in coupled mode too; the searched value
            # stays behind as the reference the CI and plots measure against.
            assert result.search_nll == result._objective(result.p_hat)
            assert result.nll != result.search_nll

        with subtests.test("the lazy CI works when search and publication agree"):
            lower, upper = result.ci(alpha=0.05)
            assert lower < result.p_hat < upper

    def test_invalid_search_fit_mode_is_rejected(self):
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = self._reml_model()

        with pytest.raises(ValueError, match="search_fit_mode"):
            model.estimate_p(
                X, y, sample_weight=sample_weight, offset=offset, search_fit_mode="nonsense"
            )


class TestDecoupledSearchConfidenceInterval:
    """A decoupled run can still be inverted -- against the profile it searched.

    ``p_hat`` comes from the search, so inverting the search's own objective
    around its own value at ``p_hat`` is the standard construction and is
    internally consistent. The publication moves ``nll`` to the published
    fit's value; the interval and the plot measure against ``search_nll``.
    """

    @staticmethod
    def _reml_model():
        return SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=4)},
        )

    def _decoupled(self):
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = self._reml_model()
        result = model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="fit",
        )
        return result

    def test_a_decoupled_search_still_yields_an_interval(self):
        result = self._decoupled()

        lower, upper = result.ci(alpha=0.05)

        assert lower < result.p_hat < upper

    def test_the_interval_inverts_the_profile_that_produced_p_hat(self):
        """The reference must be the searched objective at ``p_hat``, not the
        published dispersion's likelihood.

        Subtracting the published value is what produced the negative
        likelihood ratio that motivated the original refusal: the search's own
        minimum then appears to sit *above* the reference, and the interval
        code reports "found a better profile value" against a search that did
        nothing wrong.
        """
        result = self._decoupled()

        reference = result.search_nll
        at_optimum = float(result._objective(float(result.p_hat)))

        # The reference IS the searched objective's value at p_hat, so the
        # likelihood ratio there is zero rather than negative.
        assert at_optimum == pytest.approx(reference, rel=1e-12)
        assert 2.0 * result._ll_scale * (at_optimum - reference) >= -1e-9

    def test_the_published_dispersion_is_still_reported_separately(self):
        """Fixing the CI reference must not silently restore the search's phi."""
        result = self._decoupled()

        assert result.search_nll != result.nll

    def test_eager_ci_alpha_matches_the_lazy_interval_on_a_decoupled_run(self):
        """`ci_alpha=` and `result.ci()` are one computation and must agree.

        7d2f745 withdrew the lazy refusal but the eager gate in
        `profile_ops.estimate_p` predates it; left in place it refuses at the
        call site the very interval the returned object hands out. The eager
        interval is computed after publication records `search_nll`, so it has
        the same searched reference the lazy path uses.
        """
        X, y, sample_weight, offset = _offset_spline_tweedie_data()
        model = self._reml_model()

        result = model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode="reml",
            search_fit_mode="fit",
            ci_alpha=0.05,
        )

        lower, upper = result.ci(alpha=0.05)
        assert lower < result.p_hat < upper
        assert 0.05 in result._ci_cache

    def test_the_profile_plot_measures_the_searched_curve_against_its_own_optimum(self):
        """The plotted statistic is the same subtraction the CI makes.

        `profile_plot` draws searched values minus a reference. Against the
        published `nll` the whole curve is displaced by the gap between the two
        regimes, which puts the search's own optimum below zero -- a likelihood
        ratio that cannot happen.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._decoupled()

        fig, ax = plt.subplots()
        try:
            result.profile_plot(ax=ax)
            plotted = np.concatenate(
                [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()]
            )
        finally:
            plt.close(fig)

        finite = plotted[np.isfinite(plotted)]
        assert finite.size > 0
        assert finite.min() >= -1e-6


# =====================================================================
# Profile inputs, candidate-fit parity and low powers
# =====================================================================


class TestProfileContextInputOwnership:
    """Lazy profile probes must use the data supplied when estimation began."""

    @staticmethod
    def _problem():
        x = np.linspace(-1.0, 1.0, 12)
        X = pd.DataFrame({"x": x})
        y = np.exp(0.4 + 0.2 * x)
        sample_weight = np.linspace(0.5, 1.5, len(x))
        offset = 0.1 * x
        return X, y, sample_weight, offset

    @staticmethod
    def _model():
        return SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x": Numeric()},
        )

    def test_fit_profile_owns_arrays_used_by_lazy_profile_probes(self):
        X, y, sample_weight, offset = self._problem()
        profile = tweedie_module._PowerProfile(self._model(), X, y, sample_weight, offset, "fit")
        expected_design = profile.clone._dm.toarray().copy()
        expected_y = profile.y.copy()
        expected_weight = profile.w.copy()
        expected_offset = profile.offset.copy()

        assert not np.shares_memory(profile.y, y)
        assert not np.shares_memory(profile.w, sample_weight)
        assert not np.shares_memory(profile.offset, offset)

        X.iloc[:, 0] = 99.0
        y[:] = 101.0
        sample_weight[:] = 103.0
        offset[:] = 107.0

        np.testing.assert_array_equal(profile.clone._dm.toarray(), expected_design)
        np.testing.assert_array_equal(profile.y, expected_y)
        np.testing.assert_array_equal(profile.w, expected_weight)
        np.testing.assert_array_equal(profile.offset, expected_offset)

    def test_reml_profile_owns_all_inputs_used_by_lazy_profile_probes(self):
        X, y, sample_weight, offset = self._problem()
        profile = tweedie_module._PowerProfile(
            self._model(), X, y, sample_weight, offset, "fit_reml"
        )
        expected_X = profile.X.copy(deep=True)
        expected_y = profile.y.copy()
        expected_weight = profile.w.copy()
        expected_offset = profile.offset.copy()

        assert profile.X is not X
        assert not np.shares_memory(profile.X["x"].to_numpy(), X["x"].to_numpy())
        assert not np.shares_memory(profile.y, y)
        assert not np.shares_memory(profile.w, sample_weight)
        assert not np.shares_memory(profile.offset, offset)

        X.iloc[:, 0] = 99.0
        y[:] = 101.0
        sample_weight[:] = 103.0
        offset[:] = 107.0

        pd.testing.assert_frame_equal(profile.X, expected_X)
        np.testing.assert_array_equal(profile.y, expected_y)
        np.testing.assert_array_equal(profile.w, expected_weight)
        np.testing.assert_array_equal(profile.offset, expected_offset)


class TestProfileFitParity:
    """Fixed-p profile fits must be identical to the ordinary fit regimes."""

    @staticmethod
    def _custom_tensor_problem():
        rng = np.random.default_rng(20260717)
        n = 40
        t = np.linspace(0.0, 1.0, n)
        x1 = np.linspace(-1.0, 1.0, n)
        x2 = np.sin(4.0 * np.pi * t) + 0.35 * np.cos(9.0 * np.pi * t) + 0.1 * t
        X = pd.DataFrame({"x1": x1, "x2": x2})
        mu = np.exp(0.6 + 0.25 * x1 - 0.2 * x2 + 0.3 * x1 * x2)
        y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.5, rng=rng)
        return X, y

    @staticmethod
    def _custom_tensor_model():
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            spline_penalty=2.0,
            features={
                "x1": Spline(n_knots=5, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model._add_interaction(
            "x1",
            "x2",
            name="custom_surface",
            n_knots=(3, 4),
            decompose=True,
        )
        return model

    @staticmethod
    def _group_signature(groups):
        return [(group.name, group.end - group.start) for group in groups]

    @staticmethod
    def _fixed_p_metrics(model, y):
        """The profile's quantities at p = 1.5 computed from an ordinary fit."""
        assert model._fit_mu is not None
        mu = np.asarray(model._fit_mu, dtype=np.float64)
        solved = profile_phi_at(y, mu, np.ones_like(y), 1.5)
        return solved.phi, solved.criterion / y.size

    @staticmethod
    def _profile_at_one_point(model, X, y, fit_mode, offset=None):
        """One candidate evaluation at p = 1.5 of the search's own profile."""
        profile = tweedie_module._PowerProfile(model, X, y, np.ones(len(y)), offset, fit_mode)
        nll = profile(1.5)
        return profile, profile.candidates[1.5].phi, nll

    @classmethod
    def _resolved_custom_tensor_model(cls):
        X, y = cls._custom_tensor_problem()
        model = cls._custom_tensor_model()
        model._build_design_matrix(X, y, None, None)
        return model, X

    def test_profile_clone_preserves_resolved_custom_tensor_interaction(self):
        model, X = self._resolved_custom_tensor_model()
        original = model._interaction_specs["custom_surface"]

        clone = tweedie_module._clone_profile_model(model, X, None)

        assert clone._interaction_order == ["custom_surface"]
        assert clone._pending_interactions == ()
        cloned = clone._interaction_specs["custom_surface"]
        assert isinstance(cloned, TensorInteraction)
        assert cloned.parent_names == ("x1", "x2")
        assert cloned._n_knots == (3, 4)
        assert cloned._decompose is True
        assert (cloned._p1, cloned._p2) == (original._p1, original._p2)
        assert cloned._marginal1 is not None
        assert cloned._marginal2 is not None
        assert cloned._R_inv is not None

        rematerialized = clone._config.materialize(type(clone))
        assert rematerialized._interaction_order == ["custom_surface"]
        assert rematerialized._pending_interactions == ()
        configured = rematerialized._interaction_specs["custom_surface"]
        assert configured.parent_names == ("x1", "x2")
        assert configured._n_knots == (3, 4)
        assert configured._decompose is True

    def test_profile_clone_deep_copies_resolved_custom_tensor_state(self):
        model, X = self._resolved_custom_tensor_model()
        original = model._interaction_specs["custom_surface"]
        assert original._marginal1 is not None
        assert original._marginal2 is not None
        assert original._R_inv is not None

        clone = tweedie_module._clone_profile_model(model, X, None)

        cloned = clone._interaction_specs["custom_surface"]
        assert cloned is not original
        assert cloned._marginal1 is not original._marginal1
        assert cloned._marginal2 is not original._marginal2
        assert cloned._R_inv is not original._R_inv
        np.testing.assert_allclose(cloned._marginal1.basis, original._marginal1.basis)
        np.testing.assert_allclose(cloned._marginal2.basis, original._marginal2.basis)
        np.testing.assert_allclose(cloned._R_inv, original._R_inv)

        original_basis = original._marginal1.basis.copy()
        original_R_inv = original._R_inv.copy()
        cloned._marginal1.basis[0, 0] += 1.0
        cloned._R_inv[0, 0] += 1.0
        cloned._n_knots = (8, 9)
        cloned._decompose = False

        np.testing.assert_array_equal(original._marginal1.basis, original_basis)
        np.testing.assert_array_equal(original._R_inv, original_R_inv)
        assert original._n_knots == (3, 4)
        assert original._decompose is True

    def test_fit_profile_custom_tensor_matches_independent_fixed_p_fit(self):
        X, y = self._custom_tensor_problem()
        independent = self._custom_tensor_model()
        independent.fit(X, y)
        independent_phi, independent_nll = self._fixed_p_metrics(independent, y)
        expected_groups = [
            ("x1", 8),
            ("x2", 9),
            ("custom_surface:bilinear", 1),
            ("custom_surface:wiggly", 41),
        ]
        assert self._group_signature(independent._groups) == expected_groups

        profile, phi, nll = self._profile_at_one_point(self._custom_tensor_model(), X, y, "fit")

        assert self._group_signature(profile.clone._groups) == expected_groups
        assert phi == pytest.approx(independent_phi, rel=1e-10, abs=1e-10)
        assert nll == pytest.approx(independent_nll, rel=1e-10, abs=1e-10)

    @pytest.mark.slow
    def test_reml_profile_custom_tensor_matches_independent_fixed_p_fit(self):
        X, y = self._custom_tensor_problem()
        independent = self._custom_tensor_model()
        # Candidate evaluations run at the search tolerance; a 1e-8 parity
        # claim against them requires the independent fit at the same bar.
        independent.fit_reml(X, y, reml_tol=tweedie_module._SEARCH_REML_TOL)
        independent_phi, independent_nll = self._fixed_p_metrics(independent, y)
        expected_groups = [
            ("x1", 8),
            ("x2", 9),
            ("custom_surface:bilinear", 1),
            ("custom_surface:wiggly", 41),
        ]
        assert self._group_signature(independent._groups) == expected_groups

        profile, phi, nll = self._profile_at_one_point(
            self._custom_tensor_model(), X, y, "fit_reml"
        )

        assert profile.clone._interaction_order == ["custom_surface"]
        assert self._group_signature(profile.clone._groups) == expected_groups
        assert phi == pytest.approx(independent_phi, rel=1e-8, abs=1e-8)
        assert nll == pytest.approx(independent_nll, rel=1e-8, abs=1e-8)

    @pytest.mark.parametrize(
        ("fit_mode", "fit_method"),
        [
            pytest.param("fit", "fit", id="fit"),
            pytest.param(
                "fit_reml",
                "fit_reml",
                id="fit_reml",
                marks=pytest.mark.slow,
            ),
        ],
    )
    def test_custom_tensor_profile_and_later_probe_leave_caller_unchanged(
        self,
        fit_mode,
        fit_method,
    ):
        X, y = self._custom_tensor_problem()
        model = self._custom_tensor_model()
        getattr(model, fit_method)(X, y)
        snapshot = _snapshot_fitted_model(model, X)

        result = search_power(
            model, X, y, np.ones(len(y)), None, fit_mode=fit_mode, p_bounds=(1.45, 1.55), xatol=0.05
        )
        result._objective(1.6)

        _assert_fitted_model_unchanged(model, X, snapshot)

    def test_a_search_out_of_iterations_warns_and_is_not_converged(self):
        X, y, _ = _tweedie_data(n=200, seed=20260928)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        with pytest.warns(UserWarning, match="iteration limit"):
            result = search_power(model, X, y, np.ones(len(y)), None, fit_mode="fit", maxiter=1)

        assert not result.converged
        assert any("iteration limit" in message for message in result.warnings)

    def test_profile_clone_keeps_shorthand_interaction_pending_until_build(self):
        X, y = self._custom_tensor_problem()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            spline_penalty=2.0,
            splines=["x1", "x2"],
            n_knots=[5, 6],
            interactions=[("x1", "x2")],
        )
        caller_state = pickle.dumps(model.__dict__, protocol=5)
        assert model._specs == {}
        assert model._interaction_specs == {}
        assert model._pending_interactions == (("x1", "x2"),)

        clone = tweedie_module._clone_profile_model(model, X, None)

        assert clone._interaction_specs == {}
        assert clone._interaction_order == []
        assert clone._pending_interactions == (("x1", "x2"),)
        assert list(clone._specs) == ["x1", "x2"]
        clone._build_design_matrix(X, y, None, None)
        assert clone._pending_interactions == ()
        assert clone._interaction_order == ["x1:x2"]
        assert isinstance(clone._interaction_specs["x1:x2"], TensorInteraction)

        profiled = search_power(
            model, X, y, np.ones(len(y)), None, fit_mode="fit", p_bounds=(1.45, 1.55), xatol=0.05
        )

        assert np.isfinite(profiled.nll)
        assert pickle.dumps(model.__dict__, protocol=5) == caller_state

    def test_pirls_profile_forwards_model_controls_and_lambda2(self, monkeypatch):
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 12)})
        y = np.linspace(0.5, 2.0, len(X))
        captured = {}

        def fake_fit_pirls(**kwargs):
            captured.update(kwargs)
            return _profile_solver_result(kwargs["X"])

        monkeypatch.setattr(fit_ops_module, "fit_pirls", fake_fit_pirls)
        with pytest.warns(UserWarning, match="convergence='coefficients' is experimental"):
            model = SuperGLM(
                family=TweedieDistribution(p=1.5),
                selection_penalty=0.2,
                spline_penalty=7.5,
                active_set=True,
                tol=2e-5,
                max_iter=17,
                convergence="coefficients",
                features={"x": Numeric()},
            )
            profile = tweedie_module._PowerProfile(model, X, y, np.ones(len(y)), None, "fit")
        profile(1.5)

        assert captured["lambda2"] == 7.5
        assert captured["max_iter_outer"] == 17
        assert captured["tol"] == pytest.approx(2e-5)
        assert captured["active_set"] is True
        assert captured["convergence"] == "coefficients"
        assert captured["penalty"] is profile.penalty

    @pytest.mark.parametrize(
        ("selection_penalty", "expects_auto"),
        [
            pytest.param(None, False, id="none-disabled"),
            pytest.param(0.0, False, id="zero-disabled"),
            pytest.param("auto", True, id="explicit-auto"),
        ],
    )
    def test_profile_context_resolves_selection_intent_numerically(
        self,
        selection_penalty,
        expects_auto,
    ):
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 24)})
        y = np.linspace(0.5, 2.0, len(X))
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=selection_penalty,
            features={"x": Numeric()},
        )

        profile = tweedie_module._PowerProfile(model, X, y, np.ones(len(y)), None, "fit")

        assert isinstance(profile.penalty.lambda1, float)
        if expects_auto:
            assert profile.penalty.lambda1 > 0.0
            assert profile.has_lambda1_targets is True
        else:
            assert profile.penalty.lambda1 == pytest.approx(0.0)

    def test_direct_profile_forwards_model_controls_and_lambda2(self, monkeypatch):
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 12)})
        y = np.linspace(0.5, 2.0, len(X))
        captured = {}

        def fake_fit_irls_direct(**kwargs):
            captured.update(kwargs)
            return _profile_solver_result(kwargs["X"]), None

        monkeypatch.setattr(fit_ops_module, "fit_irls_direct", fake_fit_irls_direct)
        with pytest.warns(UserWarning, match="convergence='coefficients' is experimental"):
            model = SuperGLM(
                family=TweedieDistribution(p=1.5),
                selection_penalty=0,
                spline_penalty=8.5,
                direct_solve="qr",
                tol=3e-5,
                max_iter=19,
                convergence="coefficients",
                features={"x": Numeric()},
            )
            profile = tweedie_module._PowerProfile(model, X, y, np.ones(len(y)), None, "fit")
        profile(1.5)

        assert captured["lambda2"] == 8.5
        assert captured["max_iter"] == 19
        assert captured["tol"] == pytest.approx(3e-5)
        assert captured["direct_solve"] == "qr"
        assert captured["convergence"] == "coefficients"

    def test_positive_lambda1_without_targets_dispatches_to_direct(self, monkeypatch):
        X = pd.DataFrame(index=np.arange(12))
        y = np.linspace(0.5, 2.0, len(X))
        direct_calls = []

        def fake_fit_irls_direct(**kwargs):
            direct_calls.append(kwargs)
            return _profile_solver_result(kwargs["X"], effective_df=0.0), None

        def fail_pirls(**kwargs):
            raise AssertionError("no-target profile incorrectly dispatched to PIRLS")

        monkeypatch.setattr(fit_ops_module, "fit_irls_direct", fake_fit_irls_direct)
        monkeypatch.setattr(fit_ops_module, "fit_pirls", fail_pirls)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0.25,
            features={},
        )

        profile = tweedie_module._PowerProfile(model, X, y, np.ones(len(y)), None, "fit")
        assert profile.penalty.lambda1 > 0.0
        assert not penalty_has_targets(profile.penalty, profile.clone._groups)

        profile(1.5)

        assert len(direct_calls) == 1

    def test_flexible_spline_profile_matches_independent_fixed_p_fit(self):
        rng = np.random.default_rng(20260716)
        n = 120
        x = np.linspace(0.0, 1.0, n)
        X = pd.DataFrame({"x": x})
        mu = np.exp(1.2 + 0.8 * np.sin(2.0 * np.pi * x))
        y = generate_tweedie_cpg(n, mu=mu, phi=1.5, p=1.5, rng=rng)
        model_kwargs = dict(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0.01,
            spline_penalty=1000.0,
            features={"x": Spline(n_knots=20)},
        )

        independent = SuperGLM(**model_kwargs)
        independent.fit(X, y)
        independent_phi = profile_phi_at(y, independent.predict(X), np.ones(n), 1.5).phi

        _, phi, _ = self._profile_at_one_point(SuperGLM(**model_kwargs), X, y, "fit")

        assert phi == pytest.approx(independent_phi, rel=1e-10, abs=1e-10)

    def test_reml_profile_uses_offset_aware_fitted_mean(self):
        rng = np.random.default_rng(20260716)
        n = 120
        x = np.linspace(-1.0, 1.0, n)
        X = pd.DataFrame({"x": x})
        offset = 1.2 * np.sin(np.pi * x) + 0.3 * x
        mu = np.exp(0.7 + 0.25 * x + offset)
        y = generate_tweedie_cpg(n, mu=mu, phi=1.2, p=1.5, rng=rng)
        model_kwargs = dict(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x": Spline(n_knots=6, penalty="ssp")},
        )

        independent = SuperGLM(**model_kwargs)
        # Candidate fits run at the search tolerance; parity at 1e-9
        # requires the independent fit at the same bar.
        independent.fit_reml(X, y, offset=offset, reml_tol=tweedie_module._SEARCH_REML_TOL)
        independent_mu = independent._fit_mu
        assert independent_mu is not None
        independent_phi, independent_nll = self._fixed_p_metrics(independent, y)
        offset_free = profile_phi_at(y, independent.predict(X), np.ones(n), 1.5)

        _, phi, nll = self._profile_at_one_point(
            SuperGLM(**model_kwargs), X, y, "fit_reml", offset=offset
        )

        assert phi == pytest.approx(independent_phi, rel=1e-9, abs=1e-9)
        assert nll == pytest.approx(independent_nll, rel=1e-9, abs=1e-9)
        # Dropping the offset from the candidate mean is a gross error, not round-off.
        assert abs(phi - offset_free.phi) > 1.0
        assert abs(nll - offset_free.criterion / n) > 0.5

    def test_low_level_ordinary_profile_and_later_probe_leave_caller_unchanged(self):
        rng = np.random.default_rng(20260716)
        n = 48
        x = np.linspace(-1.0, 1.0, n)
        X = pd.DataFrame({"x": x})
        offset = 0.3 * x
        mu = np.exp(0.5 + 0.2 * x + offset)
        y = generate_tweedie_cpg(n, mu=mu, phi=1.0, p=1.5, rng=rng)
        model = SuperGLM(
            family=TweedieDistribution(p=1.4),
            selection_penalty=0,
            features={"x": Numeric()},
        )
        model.fit(X, y, offset=offset)
        snapshot = _snapshot_fitted_model(model, X, offset=offset)

        result = search_power(
            model, X, y, np.ones(n), offset, fit_mode="fit", p_bounds=(1.4, 1.6), xatol=0.05
        )
        result._objective(1.7)

        _assert_fitted_model_unchanged(model, X, snapshot, offset=offset)

    def test_low_level_profile_keeps_unfitted_shorthand_configuration_immutable(self):
        rng = np.random.default_rng(20260716)
        n = 40
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, n)})
        y = generate_tweedie_cpg(
            n,
            mu=np.exp(0.5 + 0.2 * X["x"].to_numpy()),
            phi=1.0,
            p=1.5,
            rng=rng,
        )
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            spline_penalty=2.0,
            splines=["x"],
            n_knots=7,
            degree=2,
        )
        assert model._specs == {}
        snapshot = pickle.dumps(model.__dict__, protocol=5)

        result = search_power(
            model, X, y, np.ones(n), None, fit_mode="fit", p_bounds=(1.45, 1.55), xatol=0.05
        )

        assert np.isfinite(result.nll)
        assert pickle.dumps(model.__dict__, protocol=5) == snapshot

    def test_low_level_reml_profile_and_later_probe_leave_caller_unchanged(self):
        rng = np.random.default_rng(20260716)
        n = 48
        x = np.linspace(-1.0, 1.0, n)
        X = pd.DataFrame({"x": x})
        offset = 0.4 * np.sin(np.pi * x)
        mu = np.exp(0.4 + 0.3 * x + offset)
        y = generate_tweedie_cpg(n, mu=mu, phi=1.0, p=1.5, rng=rng)
        model = SuperGLM(
            family=TweedieDistribution(p=1.4),
            selection_penalty=0,
            spline_penalty=0.25,
            features={"x": Spline(n_knots=5, penalty="ssp")},
        )
        model.fit_reml(X, y, offset=offset, max_reml_iter=5)
        snapshot = _snapshot_fitted_model(model, X, offset=offset)

        result = search_power(
            model, X, y, np.ones(n), offset, fit_mode="fit_reml", p_bounds=(1.4, 1.6), xatol=0.05
        )
        result._objective(1.7)

        _assert_fitted_model_unchanged(model, X, snapshot, offset=offset)

    def test_reml_profile_clone_uses_configured_lambda2_not_previous_reml_lambdas(self):
        rng = np.random.default_rng(20260716)
        n = 40
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, n)})
        y = generate_tweedie_cpg(
            n,
            mu=np.exp(0.4 + 0.2 * X["x"].to_numpy()),
            phi=1.0,
            p=1.5,
            rng=rng,
        )
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            spline_penalty=0.25,
            features={"x": Spline(n_knots=5, penalty="ssp")},
        )
        model.fit_reml(X, y, max_reml_iter=5)
        assert model._reml_lambdas is not None

        profile = tweedie_module._PowerProfile(model, X, y, np.ones(n), None, "fit_reml")

        assert profile.clone.lambda2 == model.lambda2 == 0.25


class TestLowPowerProfiles:
    """The Brent search near p = 1, where rounded data can fake an edge maximum."""

    @pytest.mark.slow
    def test_low_p_boundary_regression(self):
        """Low-p profiles should not spuriously prefer the lower bound."""
        # With every row forced onto the saddlepoint -- the leak this pinned on
        # master -- the search landed on the 1.10 bound at this size.
        X, y, _ = _tweedie_data(n=1_100, p_true=1.25, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        result = model.estimate_p(X, y, p_bounds=(1.1, 1.9))

        assert result.p_hat > 1.15
        assert not any("search bound" in warning for warning in result.warnings)

    def test_low_p_profile_completes_without_warning(self):
        """A power near 1 is estimated from the exact series with nothing to disclose."""
        X, y, _ = _tweedie_data(n=2_500, p_true=1.08, seed=4)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.estimate_p(X, y, p_bounds=(1.05, 1.9))

        messages = [str(item.message) for item in caught]
        assert not messages
        assert result.warnings == []
        assert result.converged

    def test_regular_profile_has_no_warning(self):
        """Typical interior fits should not warn."""
        X, y, _ = _tweedie_data(n=2_500, p_true=1.25, seed=7)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.estimate_p(X, y, p_bounds=(1.05, 1.9))

        assert not caught
        assert result.warnings == []
