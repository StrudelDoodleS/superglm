"""Tests for Tweedie profile likelihood — p estimation."""

import inspect
import pickle
import warnings
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize_scalar as scipy_minimize_scalar

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm.distributions import Tweedie as TweedieDistribution
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.links import LogLink
from superglm.model import profile_ops as profile_ops_module
from superglm.penalties.base import penalty_has_targets
from superglm.penalties.group_lasso import GroupLasso
from superglm.profiling.tweedie import (
    TweedieProfileResult,
    _profile_phi,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)

_COUNT_DIAGNOSTICS = object()


def _legacy_pickle_profile_objective(p):
    """Pickleable profile objective for pre-density-field result state."""
    return 100.0 * (float(p) - 1.5) ** 2


def _legacy_pickle_evaluation_record(p):
    """Pickleable authoritative record lookup for legacy-result CI regression."""
    key = float(p)
    n_saddlepoint = 2 if np.isclose(key, 1.55, rtol=0.0, atol=1e-14) else 0
    return SimpleNamespace(
        nll=_legacy_pickle_profile_objective(key),
        source="legacy_pickle",
        fit_converged=True,
        phi_result=SimpleNamespace(
            objective_finite=True,
            converged=True,
            diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                n_positive=10,
                n_saddlepoint=n_saddlepoint,
            ),
        ),
    )


def _finalized_density_result(
    *,
    p: float = 1.5,
    n_positive: int = 100,
    n_saddlepoint: int = 0,
    phi_method: str = "mle",
    diagnostics_override=_COUNT_DIAGNOSTICS,
):
    """Finalize one immutable record for density-provenance regressions."""
    diagnostics = (
        tweedie_module._TweedieLogpdfDiagnostics(
            n_positive=n_positive,
            n_saddlepoint=n_saddlepoint,
        )
        if diagnostics_override is _COUNT_DIAGNOSTICS
        else diagnostics_override
    )
    phi_result = tweedie_module._PhiProfileResult(
        phi=1.0,
        nll=1.0,
        converged=True,
        objective_finite=True,
        n_evaluations=1,
        n_score_evaluations=1,
        n_value_only_evaluations=0,
        n_fallback_evaluations=0,
        optimizer="brentq",
        score=0.0,
        used_fallback=False,
        fallback_reason=None,
        branch_switch_detected=False,
        lower_boundary=False,
        upper_boundary=False,
        diagnostics=diagnostics,
        message="",
    )
    record = tweedie_module._ProfileEvaluation(
        step=1,
        p=p,
        mu=np.ones(3),
        edf=1.0,
        n_iter=1,
        fit_converged=True,
        source="fixture",
        fit_trace=(),
        fit_trace_kind="solver",
        phi_result=phi_result,
    )
    cache = {p: record}
    ctx = SimpleNamespace(
        phi_method=phi_method,
        ll_scale=3.0,
        _evaluation_cache=cache,
        evaluate=lambda value, source="": cache[float(value)].nll,
        evaluation_count=lambda: len(cache),
        evaluation_record=lambda value: cache.get(float(value)),
    )
    return tweedie_module._finalize_profile_record(
        ctx,
        record,
        method="grid",
        outer_converged=True,
    )


def _generate_weighted_tweedie(mu, phi, p, weights, rng):
    """Simulate Tweedie responses under the prior-weight convention phi / w."""
    mu = np.asarray(mu, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    y = np.empty(len(mu), dtype=np.float64)
    for i in range(len(mu)):
        y[i] = generate_tweedie_cpg(1, mu=mu[i], phi=phi / weights[i], p=p, rng=rng)[0]
    return y


def _call_tweedie_low_level(function_name, y, mu, *, phi=2.0, p=1.5, weights=None):
    """Call one of the public low-level Tweedie helpers with shared inputs."""
    if function_name == "tweedie_logpdf":
        return tweedie_logpdf(y, mu, phi, p, weights=weights)
    return estimate_phi(y, mu, p, weights=weights)


def _stable_p_one_half_deviance(y, mu):
    """Closed-form p=1.5 unit deviance without subtracting close square roots."""
    delta = (y - mu) / mu
    root_difference = delta / (np.sqrt(1.0 + delta) + 1.0)
    return 4.0 * np.sqrt(mu) * root_difference**2


def _stable_p_one_half_extreme_deviance(y, mu):
    """Closed-form p=1.5 deviance for ratios too large to form directly."""
    with np.errstate(over="ignore"):
        return 4.0 * (np.sqrt(y) - np.sqrt(mu)) ** 2 / np.sqrt(mu)


def _tight_value_only_phi_reference(y, mu, p, *, weights=None):
    """Independent tight derivative-free reference over the hard log-phi range."""
    calls = 0

    def objective(log_phi):
        nonlocal calls
        calls += 1
        logpdf = tweedie_logpdf(
            y,
            mu,
            float(np.exp(log_phi)),
            p,
            weights=weights,
        )
        return -float(np.mean(logpdf))

    result = scipy_minimize_scalar(
        objective,
        bounds=(np.log(1e-12), np.log(1e12)),
        method="bounded",
        options={"xatol": 1e-11, "maxiter": 500},
    )
    assert result.success
    return result, calls


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

    def test_saddlepoint_close_y_mu_is_stable_at_large_scale(self):
        """The compatibility helper must not lose close-y/mu unit deviance."""
        mu = np.array([1e12])
        y = mu * (1.0 + 1e-12)
        phi = np.array([1e-12])
        p = 1.5
        deviance = _stable_p_one_half_deviance(y, mu)
        expected = -0.5 * (np.log(2.0 * np.pi) + np.log(phi) + p * np.log(y)) - deviance / (
            2.0 * phi
        )

        actual = tweedie_module._saddlepoint(y, mu, phi, p)

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)

    def test_saddlepoint_extreme_positive_ratio_is_stable(self):
        """The compatibility helper must avoid inf-inf for finite y >> mu."""
        y = np.array([1.0])
        mu = np.array([1e-320])
        phi = np.array([1e-8])
        p = 1.5
        deviance = _stable_p_one_half_extreme_deviance(y, mu)
        expected = -0.5 * (np.log(2.0 * np.pi) + np.log(phi) + p * np.log(y)) - deviance / (
            2.0 * phi
        )

        actual = tweedie_module._saddlepoint(y, mu, phi, p)

        assert np.all(np.isfinite(deviance))
        np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_saddlepoint_overflowing_positive_ratio_returns_negative_infinity(self):
        """A truly overflowing deviance is extended-real, not NaN."""
        y = np.array([1e308])
        mu = np.array([1e-320])
        phi = np.array([1e-8])

        actual = tweedie_module._saddlepoint(y, mu, phi, 1.5)

        assert np.all(np.isneginf(actual))

    def test_all_invalid_wright_terms_use_saddlepoint(self, monkeypatch):
        """Every invalid Wright term must be populated by the fallback."""
        y = np.array([0.5, 2.0, 8.0])
        mu = np.array([0.8, 1.5, 6.0])
        phi, p = 2.0, 1.5

        def all_nan_wright(a, b, t):
            return np.full_like(t, np.nan, dtype=np.float64)

        monkeypatch.setattr(tweedie_module, "wright_bessel", all_nan_wright)

        logpdf, diagnostics = tweedie_module._tweedie_logpdf_impl(y, mu, phi, p)
        expected = tweedie_module._saddlepoint(y, mu, np.full_like(y, phi), p)

        np.testing.assert_allclose(logpdf, expected, rtol=1e-14, atol=1e-14)
        assert diagnostics.n_positive == len(y)
        assert diagnostics.n_saddlepoint == diagnostics.n_positive

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

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    @pytest.mark.parametrize(
        "invalid_weight",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
            pytest.param(1.0 + 1.0j, id="complex"),
        ],
    )
    def test_invalid_weight_value_is_rejected(self, function_name, invalid_weight):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])
        weights = np.array([1.0, invalid_weight, 1.0])

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            _call_tweedie_low_level(function_name, y, mu, weights=weights)

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    @pytest.mark.parametrize(
        "invalid_weights",
        [
            pytest.param(np.ones((3, 1)), id="two-dimensional"),
            pytest.param(np.ones(2), id="mismatched-length"),
            pytest.param([[1.0], [1.0, 2.0]], id="ragged"),
        ],
    )
    def test_invalid_weight_shape_is_rejected(self, function_name, invalid_weights):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            _call_tweedie_low_level(function_name, y, mu, weights=invalid_weights)

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    @pytest.mark.parametrize(
        "invalid_y",
        [
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
            pytest.param(1.0 + 1.0j, id="complex"),
        ],
    )
    def test_invalid_input_y_is_rejected(self, function_name, invalid_y):
        y = np.array([0.0, invalid_y, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="y must be finite and non-negative"):
            _call_tweedie_low_level(function_name, y, mu)

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    @pytest.mark.parametrize(
        "invalid_mu",
        [
            pytest.param(0.0, id="zero"),
            pytest.param(-1.0, id="negative"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
            pytest.param(1.0 + 1.0j, id="complex"),
        ],
    )
    def test_invalid_input_mu_is_rejected(self, function_name, invalid_mu):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, invalid_mu, 2.5])

        with pytest.raises(ValueError, match="mu must be finite and strictly positive"):
            _call_tweedie_low_level(function_name, y, mu)

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    @pytest.mark.parametrize(
        "invalid_p",
        [
            pytest.param(1.0, id="lower-bound"),
            pytest.param(2.0, id="upper-bound"),
            pytest.param(np.nan, id="nan"),
            pytest.param(np.inf, id="inf"),
        ],
    )
    def test_invalid_input_p_is_rejected(self, function_name, invalid_p):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([1.0, 1.5, 2.5])

        with pytest.raises(ValueError, match="p must be finite and in the open interval"):
            _call_tweedie_low_level(function_name, y, mu, p=invalid_p)

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    @pytest.mark.parametrize(
        ("y", "mu"),
        [
            pytest.param(np.array([0.0, 1.0, 2.0]), np.ones(2), id="different-lengths"),
            pytest.param(np.array([[0.0], [1.0]]), np.ones((2, 1)), id="two-dimensional"),
        ],
    )
    def test_invalid_input_y_mu_shape_is_rejected(self, function_name, y, mu):
        with pytest.raises(ValueError, match="y and mu must be one-dimensional arrays"):
            _call_tweedie_low_level(function_name, y, mu)

    @pytest.mark.parametrize("function_name", ["tweedie_logpdf", "estimate_phi"])
    def test_invalid_input_empty_y_mu_is_rejected(self, function_name):
        with pytest.raises(ValueError, match="non-empty"):
            _call_tweedie_low_level(function_name, np.array([]), np.array([]))

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


class TestTweedieLogPhiScore:
    """Analytic mean-NLL derivatives with respect to ``log(phi)``."""

    @staticmethod
    def _finite_difference_score(prepared, phi):
        h = 1e-5
        u = np.log(phi)

        def mean_nll(log_phi):
            evaluation = tweedie_module._evaluate_tweedie_density(
                prepared,
                float(np.exp(log_phi)),
            )
            return -float(np.mean(evaluation.logpdf))

        return (mean_nll(u + h) - mean_nll(u - h)) / (2.0 * h)

    def test_saddlepoint_branch_mask_distinguishes_equal_counts(self, monkeypatch):
        """Branch identity is the full positive-observation mask, not its count."""
        y = np.array([1.0, 4.0])
        mu = np.array([1.3, 3.2])
        prepared = tweedie_module._prepare_tweedie_density(y, mu, 1.5)
        real_wright_bessel = tweedie_module.wright_bessel

        def fail_density_recurrence_in_middle(a, b, t):
            values = np.asarray(real_wright_bessel(a, b, t), dtype=np.float64)
            if np.isclose(b, a + 1.0):
                values = values.copy()
                values[(t >= 2.0) & (t <= 8.0)] = np.nan
            return values

        monkeypatch.setattr(
            tweedie_module,
            "wright_bessel",
            fail_density_recurrence_in_middle,
        )

        phi_one = tweedie_module._evaluate_tweedie_density(
            prepared,
            1.0,
            compute_score=True,
        )
        phi_two = tweedie_module._evaluate_tweedie_density(
            prepared,
            2.0,
            compute_score=True,
        )

        assert phi_one.diagnostics.n_saddlepoint == phi_two.diagnostics.n_saddlepoint == 1
        np.testing.assert_array_equal(phi_one.positive_saddlepoint_mask, [True, False])
        np.testing.assert_array_equal(phi_two.positive_saddlepoint_mask, [False, True])
        assert phi_one.score_valid
        assert phi_two.score_valid
        assert not phi_one.positive_saddlepoint_mask.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            phi_one.positive_saddlepoint_mask[0] = False

    @pytest.mark.parametrize(
        ("y", "mu", "phi", "p", "weights", "t_arg_limit"),
        [
            pytest.param(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.7, 2.0, 8.0]),
                1.7,
                1.5,
                None,
                1e14,
                id="all-zeros",
            ),
            pytest.param(
                np.array([0.3, 1.2, 4.5]),
                np.array([0.5, 1.5, 3.7]),
                1.3,
                1.5,
                None,
                1e14,
                id="exact-positives",
            ),
            pytest.param(
                np.array([0.0, 0.2, 0.0, 3.0]),
                np.array([0.4, 0.3, 2.0, 2.5]),
                2.1,
                1.6,
                None,
                1e14,
                id="mixed",
            ),
            pytest.param(
                np.array([0.2, 1.0, 5.0, 9.0]),
                np.array([0.3, 1.4, 4.0, 7.0]),
                2.4,
                1.55,
                np.array([0.25, 0.8, 1.7, 4.0]),
                1e14,
                id="unequal-prior-weights",
            ),
            pytest.param(
                np.array([0.3, 2.0, 7.0]),
                np.array([0.6, 1.5, 5.5]),
                1.8,
                1.5,
                None,
                0.0,
                id="forced-saddlepoint",
            ),
            pytest.param(
                np.array([0.0, 0.4, 3.0, 8.0]),
                np.array([0.5, 0.7, 2.5, 6.0]),
                2.2,
                1.65,
                np.array([0.3, 0.9, 2.0, 3.5]),
                0.0,
                id="weighted-forced-saddlepoint",
            ),
            pytest.param(
                np.array([0.04, 0.05, 0.06]),
                np.array([0.035, 0.055, 0.08]),
                1.0,
                1.05,
                None,
                1e14,
                id="p-near-one",
            ),
            pytest.param(
                np.array([0.2, 1.0, 5.0]),
                np.array([0.3, 1.4, 4.0]),
                1.2,
                1.95,
                None,
                1e14,
                id="p-near-two",
            ),
        ],
    )
    def test_log_phi_score_matches_centered_finite_difference(
        self,
        y,
        mu,
        phi,
        p,
        weights,
        t_arg_limit,
    ):
        prepared = tweedie_module._prepare_tweedie_density(
            y,
            mu,
            p,
            weights=weights,
            t_arg_limit=t_arg_limit,
        )
        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )

        assert evaluation.score_valid
        assert evaluation.log_phi_score is not None
        analytic = float(np.mean(evaluation.log_phi_score))
        finite_difference = self._finite_difference_score(prepared, phi)
        np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-9)

    def test_log_phi_score_constant_weight_matches_rescaled_phi(self):
        y = np.array([0.0, 0.4, 1.5, 5.0])
        mu = np.array([0.2, 0.7, 1.2, 4.5])
        phi, p, weight = 2.3, 1.6, 2.75
        weighted = tweedie_module._prepare_tweedie_density(
            y,
            mu,
            p,
            weights=np.full_like(y, weight),
        )
        unweighted = tweedie_module._prepare_tweedie_density(y, mu, p)

        weighted_eval = tweedie_module._evaluate_tweedie_density(
            weighted,
            phi,
            compute_score=True,
        )
        rescaled_eval = tweedie_module._evaluate_tweedie_density(
            unweighted,
            phi / weight,
            compute_score=True,
        )

        assert weighted_eval.score_valid
        assert rescaled_eval.score_valid
        np.testing.assert_allclose(weighted_eval.logpdf, rescaled_eval.logpdf, rtol=1e-13)
        np.testing.assert_allclose(
            weighted_eval.log_phi_score,
            rescaled_eval.log_phi_score,
            rtol=1e-13,
        )

    def test_log_phi_score_t_underflow_preserves_exact_branch(self):
        """Finite log(t) must keep an exact density even when exp(log(t)) is zero."""
        y = np.array([1.0, 2.0])
        mu = np.array([0.8, 2.5])
        phi, p = 1e300, 1.5
        prepared = tweedie_module._prepare_tweedie_density(y, mu, p)

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )

        assert evaluation.diagnostics.n_saddlepoint == 0
        assert np.all(np.isfinite(evaluation.logpdf))
        assert evaluation.score_valid
        assert evaluation.log_phi_score is not None
        analytic = float(np.mean(evaluation.log_phi_score))
        finite_difference = self._finite_difference_score(prepared, phi)
        np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-9)

    def test_log_phi_score_forced_saddlepoint_close_y_mu_is_stable(self):
        """Close large responses need a non-negative deviance, value, and score."""
        mu = np.array([1e12])
        y = mu * (1.0 + 1e-12)
        phi, p = 1e-12, 1.5
        expected_deviance = _stable_p_one_half_deviance(y, mu)
        prepared = tweedie_module._prepare_tweedie_density(
            y,
            mu,
            p,
            t_arg_limit=0.0,
        )

        assert np.all(np.isfinite(prepared.positive_saddlepoint_deviance))
        assert np.all(prepared.positive_saddlepoint_deviance >= 0.0)
        np.testing.assert_allclose(
            prepared.positive_saddlepoint_deviance,
            expected_deviance,
            rtol=2e-12,
            atol=0.0,
        )

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )
        expected_logpdf = -0.5 * (
            np.log(2.0 * np.pi) + np.log(phi) + p * np.log(y)
        ) - expected_deviance / (2.0 * phi)
        expected_score = 0.5 - expected_deviance / (2.0 * phi)

        assert evaluation.score_valid
        np.testing.assert_allclose(evaluation.logpdf, expected_logpdf, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(
            evaluation.log_phi_score,
            expected_score,
            rtol=1e-12,
            atol=1e-12,
        )
        finite_difference = self._finite_difference_score(prepared, phi)
        np.testing.assert_allclose(
            float(np.mean(evaluation.log_phi_score)),
            finite_difference,
            rtol=1e-8,
            atol=1e-9,
        )

    def test_log_phi_score_forced_saddlepoint_extreme_positive_ratio_is_stable(self):
        """A representable y >> mu deviance must produce a finite value and score."""
        y = np.array([1.0])
        mu = np.array([1e-320])
        phi, p = 1e-8, 1.5
        expected_deviance = _stable_p_one_half_extreme_deviance(y, mu)
        prepared = tweedie_module._prepare_tweedie_density(
            y,
            mu,
            p,
            t_arg_limit=0.0,
        )

        assert np.all(np.isfinite(prepared.positive_saddlepoint_deviance))
        np.testing.assert_allclose(
            prepared.positive_saddlepoint_deviance,
            expected_deviance,
            rtol=1e-13,
        )

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )
        expected_logpdf = -0.5 * (
            np.log(2.0 * np.pi) + np.log(phi) + p * np.log(y)
        ) - expected_deviance / (2.0 * phi)
        expected_score = 0.5 - expected_deviance / (2.0 * phi)

        assert evaluation.score_valid
        assert not np.any(np.isnan(evaluation.logpdf))
        assert not np.any(np.isnan(evaluation.log_phi_score))
        np.testing.assert_allclose(evaluation.logpdf, expected_logpdf, rtol=1e-13)
        np.testing.assert_allclose(evaluation.log_phi_score, expected_score, rtol=1e-13)
        finite_difference = self._finite_difference_score(prepared, phi)
        np.testing.assert_allclose(
            float(np.mean(evaluation.log_phi_score)),
            finite_difference,
            rtol=1e-8,
        )

    def test_log_phi_score_overflowing_positive_ratio_uses_extended_real_values(self):
        """An overflowing deviance must yield +inf deviance and -inf value/score."""
        y = np.array([1e308])
        mu = np.array([1e-320])
        prepared = tweedie_module._prepare_tweedie_density(
            y,
            mu,
            1.5,
            t_arg_limit=0.0,
        )

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            1e-8,
            compute_score=True,
        )

        assert np.all(np.isposinf(prepared.positive_saddlepoint_deviance))
        assert np.all(np.isneginf(evaluation.logpdf))
        assert evaluation.log_phi_score is not None
        assert np.all(np.isneginf(evaluation.log_phi_score))
        assert not evaluation.score_valid

    def test_log_phi_score_accepts_scaled_ratio_roundoff_near_one(self):
        """An O(a*eps) ratio deficit is numerical noise, not score failure."""
        y = np.array([1.0])
        mu = np.array([1.0])
        phi, p = 94.32317335438263, 102.0 / 101.0
        prepared = tweedie_module._prepare_tweedie_density(y, mu, p)

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )

        assert evaluation.diagnostics.n_saddlepoint == 0
        assert evaluation.score_valid
        assert evaluation.log_phi_score is not None
        finite_difference = self._finite_difference_score(prepared, phi)
        np.testing.assert_allclose(
            float(np.mean(evaluation.log_phi_score)),
            finite_difference,
            rtol=1e-8,
            atol=1e-9,
        )

    def test_log_phi_score_rejects_materially_subunit_ratio(self, monkeypatch):
        """The scaled tolerance must not turn a materially invalid ratio into a score."""
        y = np.array([1.0])
        mu = np.array([1.0])
        phi, p = 94.32317335438263, 102.0 / 101.0
        prepared = tweedie_module._prepare_tweedie_density(y, mu, p)
        exact_value = tweedie_module._evaluate_tweedie_density(prepared, phi)
        real_wright_bessel = tweedie_module.wright_bessel

        def materially_low_ratio(a, b, t):
            if b == a:
                return 0.99 * a * real_wright_bessel(a, a + 1.0, t)
            return real_wright_bessel(a, b, t)

        monkeypatch.setattr(tweedie_module, "wright_bessel", materially_low_ratio)

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )

        assert evaluation.diagnostics.n_saddlepoint == 0
        np.testing.assert_array_equal(evaluation.logpdf, exact_value.logpdf)
        assert not evaluation.score_valid
        assert evaluation.log_phi_score is not None
        assert np.all(np.isnan(evaluation.log_phi_score))

    def test_log_phi_score_derivative_failure_keeps_exact_density(self, monkeypatch):
        y = np.array([0.4, 1.5, 5.0])
        mu = np.array([0.7, 1.2, 4.5])
        phi, p = 1.9, 1.6
        prepared = tweedie_module._prepare_tweedie_density(y, mu, p)
        exact_value = tweedie_module._evaluate_tweedie_density(prepared, phi)
        real_wright_bessel = tweedie_module.wright_bessel

        def fail_derivative_wright(a, b, t):
            result = real_wright_bessel(a, b, t)
            if np.isclose(b, a):
                return np.full_like(t, np.nan, dtype=np.float64)
            return result

        monkeypatch.setattr(tweedie_module, "wright_bessel", fail_derivative_wright)

        evaluation = tweedie_module._evaluate_tweedie_density(
            prepared,
            phi,
            compute_score=True,
        )
        saddlepoint = tweedie_module._saddlepoint(y, mu, np.full_like(y, phi), p)

        assert exact_value.diagnostics.n_saddlepoint == 0
        assert evaluation.diagnostics.n_saddlepoint == 0
        np.testing.assert_array_equal(evaluation.logpdf, exact_value.logpdf)
        assert not np.allclose(evaluation.logpdf, saddlepoint, rtol=1e-8, atol=1e-10)
        assert not evaluation.score_valid
        assert evaluation.log_phi_score is not None
        assert np.all(np.isnan(evaluation.log_phi_score))


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


class TestDetailedPhiProfile:
    @staticmethod
    def _shifted_realized_branch_fixture():
        base = tweedie_module._prepare_tweedie_density(
            np.ones(3),
            np.ones(3),
            1.5,
        )
        nominal = np.array([-10.0, -0.6, -0.4])
        log_independent = nominal * (base.a + 1.0) + base.log_t_arg_limit
        prepared = replace(
            base,
            positive_log_t_phi_independent=tweedie_module._readonly_copy(
                log_independent,
                dtype=np.float64,
            ),
        )
        calibrated = nominal.copy()
        realized = calibrated + 10.0

        def fake_evaluate(prepared, phi, *, compute_score=False):
            u = float(np.log(phi))
            before_realized_edge = u < 0.0
            nll = (u + 0.5) ** 2 if before_realized_edge else (u - 0.2) ** 2 - 2.0
            score_value = 2.0 * (u + 0.5) if before_realized_edge else 2.0 * (u - 0.2)
            score = np.full(3, score_value) if compute_score else None
            mask = u < realized
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(3, -nll),
                log_phi_score=score,
                positive_saddlepoint_mask=mask,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=3,
                    n_saddlepoint=int(np.count_nonzero(mask)),
                ),
                score_valid=compute_score,
            )

        def fake_localize(prepared, threshold, positive_indices, remaining_probes):
            del prepared, positive_indices
            return (
                (threshold + 10.0 - 1e-6, threshold + 10.0 + 1e-6),
                remaining_probes - 1,
                True,
            )

        return prepared, calibrated, realized, fake_evaluate, fake_localize

    @pytest.mark.parametrize(
        ("y", "mu", "p", "weights", "force_saddlepoint"),
        [
            pytest.param(
                np.array([0.3, 1.2, 4.5]),
                np.array([0.5, 1.5, 3.7]),
                1.5,
                None,
                False,
                id="regular-exact",
            ),
            pytest.param(
                np.array([0.0, 0.0, 0.3, 2.5]),
                np.array([0.4, 1.2, 0.5, 2.0]),
                1.6,
                None,
                False,
                id="zero-heavy",
            ),
            pytest.param(
                np.array([0.0, 0.2, 1.0, 5.0]),
                np.array([0.3, 0.4, 1.4, 4.0]),
                1.55,
                np.array([0.25, 0.8, 1.7, 4.0]),
                False,
                id="unequal-prior-weights",
            ),
            pytest.param(
                np.array([0.3, 2.0, 7.0]),
                np.array([0.6, 1.5, 5.5]),
                1.5,
                None,
                True,
                id="forced-saddlepoint",
            ),
        ],
    )
    def test_mle_matches_tight_value_only_reference(
        self,
        monkeypatch,
        y,
        mu,
        p,
        weights,
        force_saddlepoint,
    ):
        if force_saddlepoint:
            real_wright_bessel = tweedie_module.wright_bessel

            def fail_density_recurrence(a, b, t):
                values = np.asarray(real_wright_bessel(a, b, t), dtype=np.float64)
                if np.isclose(b, a + 1.0):
                    return np.full_like(values, np.nan)
                return values

            monkeypatch.setattr(tweedie_module, "wright_bessel", fail_density_recurrence)

        reference, _ = _tight_value_only_phi_reference(
            y,
            mu,
            p,
            weights=weights,
        )
        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            weights=weights,
            phi_method="mle",
        )

        assert result.converged is (not force_saddlepoint)
        assert result.used_fallback is force_saddlepoint
        assert result.objective_finite
        assert result.optimizer == "brentq"
        assert not result.lower_boundary
        assert not result.upper_boundary
        assert result.score is not None
        assert abs(result.score) <= 1e-6
        np.testing.assert_allclose(result.nll, reference.fun, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.log(result.phi), reference.x, rtol=0.0, atol=5e-6)

    @pytest.mark.parametrize(
        ("y", "mu", "expected_phi", "boundary_name"),
        [
            pytest.param(
                np.zeros(3),
                np.array([0.7, 2.0, 8.0]),
                1e12,
                "upper_boundary",
                id="all-zero-upper",
            ),
            pytest.param(
                np.array([0.7, 2.0, 8.0]),
                np.array([0.7, 2.0, 8.0]),
                1e-12,
                "lower_boundary",
                id="positive-at-mean-lower",
            ),
        ],
    )
    def test_legitimate_hard_boundary_optima(self, y, mu, expected_phi, boundary_name):
        p = 1.5
        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")
        expected_nll = -float(np.mean(tweedie_logpdf(y, mu, expected_phi, p)))

        assert result.phi == expected_phi
        assert result.nll == expected_nll
        assert not result.converged
        assert result.used_fallback
        assert result.objective_finite
        assert getattr(result, boundary_name)
        assert result.lower_boundary is (expected_phi == 1e-12)
        assert result.upper_boundary is (expected_phi == 1e12)

    def test_boundary_without_valid_score_uses_exact_inward_objective_probe(
        self,
        monkeypatch,
    ):
        y = np.array([0.7, 2.0, 8.0])
        mu = y.copy()
        p = 1.5
        calls = []
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def invalidate_lower_boundary_score(prepared, phi, *, compute_score=False):
            evaluation = real_evaluate(prepared, phi, compute_score=compute_score)
            calls.append((float(phi), compute_score))
            if not compute_score or phi != 1e-12:
                return evaluation
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=evaluation.logpdf,
                log_phi_score=np.full_like(evaluation.logpdf, np.nan),
                positive_saddlepoint_mask=evaluation.positive_saddlepoint_mask,
                diagnostics=evaluation.diagnostics,
                score_valid=False,
            )

        monkeypatch.setattr(
            tweedie_module,
            "_evaluate_tweedie_density",
            invalidate_lower_boundary_score,
        )

        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")
        inward_u = np.log(1e-12) + 1e-5

        assert result.phi == 1e-12
        assert result.lower_boundary
        assert not result.converged
        assert result.used_fallback
        assert result.score is None
        assert any(
            not compute_score and abs(np.log(phi) - inward_u) <= 1e-12
            for phi, compute_score in calls
        )

    def test_derivative_only_failure_preserves_exact_objective_and_uses_fallback(
        self,
        monkeypatch,
    ):
        y = np.array([0.4, 1.5, 5.0])
        mu = np.array([0.7, 1.2, 4.5])
        p = 1.6
        real_wright_bessel = tweedie_module.wright_bessel

        def fail_derivative_recurrence(a, b, t):
            values = np.asarray(real_wright_bessel(a, b, t), dtype=np.float64)
            if np.isclose(b, a):
                return np.full_like(values, np.nan)
            return values

        monkeypatch.setattr(tweedie_module, "wright_bessel", fail_derivative_recurrence)
        reference, _ = _tight_value_only_phi_reference(y, mu, p)

        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")

        assert not result.converged
        assert result.objective_finite
        assert result.optimizer == "bounded"
        assert result.used_fallback
        assert result.fallback_reason is not None
        assert "derivative" in result.fallback_reason.lower()
        assert 0 < result.n_fallback_evaluations <= result.n_value_only_evaluations
        np.testing.assert_allclose(result.nll, reference.fun, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.log(result.phi), reference.x, rtol=0.0, atol=5e-6)

    def test_branch_jump_is_not_accepted_as_a_score_root(self):
        p = 1.0181533410437358
        y = np.array([1.81787899, 11275.9262, 0.0, 0.00306563885, 0.0000232882792, 1.18207511])
        mu = np.array(
            [
                0.0000253947806,
                44091.7359,
                198.869667,
                0.000051937831,
                331.859132,
                0.0054422757,
            ]
        )
        weights = np.array(
            [83.2444169, 0.17590785, 2.31976211, 463.433307, 2.50852264, 0.416322332]
        )

        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            weights=weights,
            phi_method="mle",
        )
        exact_nll = -float(np.mean(tweedie_logpdf(y, mu, result.phi, p, weights=weights)))

        assert result.used_fallback
        assert result.branch_switch_detected
        assert result.optimizer == "bounded"
        assert result.objective_finite
        assert result.nll == exact_nll
        assert result.score is None or abs(result.score) <= 1e-6
        assert not (
            abs(np.log(result.phi) - 3.807489) < 1e-4
            and result.score is not None
            and abs(result.score) > 1e-6
        )

    def test_nontoggling_nominal_threshold_does_not_explain_realized_transition(self):
        p = 1.2
        y = np.array([1.0])
        prepared = tweedie_module._prepare_tweedie_density(y, y, p)
        cache = tweedie_module._PhiEvaluationCache(prepared)
        thresholds = tweedie_module._phi_analytic_branch_thresholds(prepared)
        threshold = float(thresholds[0])
        nominal_left = cache.evaluate(threshold - 1e-12, compute_score=False)
        nominal_right = cache.evaluate(threshold + 1e-12, compute_score=False)
        realized_right = cache.evaluate(threshold + 0.49, compute_score=False)

        np.testing.assert_array_equal(
            nominal_left.positive_saddlepoint_mask,
            nominal_right.positive_saddlepoint_mask,
        )
        assert nominal_right.positive_saddlepoint_mask.tolist() == [True]
        assert realized_right.positive_saddlepoint_mask.tolist() == [False]
        assert tweedie_module._phi_branch_change_is_unexplained(
            nominal_right,
            realized_right,
            thresholds,
        )

    def test_global_fallback_calibrates_realized_wright_validity_transition(
        self,
        monkeypatch,
    ):
        calibrated_ceilings = []
        real_calibrate = tweedie_module._calibrate_wright_log_t_ceiling

        def spy_calibrate(prepared):
            ceiling = real_calibrate(prepared)
            calibrated_ceilings.append(ceiling)
            return ceiling

        monkeypatch.setattr(
            tweedie_module,
            "_calibrate_wright_log_t_ceiling",
            spy_calibrate,
        )

        prepared = tweedie_module._prepare_tweedie_density(
            np.array([1.0]),
            np.array([1.0]),
            1.2,
        )
        result = tweedie_module._profile_phi_detailed(
            np.array([1.0]),
            np.array([1.0]),
            1.2,
            phi_method="mle",
        )

        assert result.used_fallback
        assert result.branch_switch_detected
        assert len(calibrated_ceilings) == 1
        assert calibrated_ceilings[0] is not None
        realized_thresholds, calibrated = tweedie_module._phi_realized_wright_thresholds(prepared)
        assert calibrated
        realized_threshold = float(realized_thresholds[0])
        cache = tweedie_module._PhiEvaluationCache(prepared)
        left = cache.evaluate(realized_threshold - 1e-12, compute_score=False)
        right = cache.evaluate(realized_threshold + 1e-12, compute_score=False)
        assert left.branch_signature != right.branch_signature

    def test_realized_transition_probe_cap_exhaustion_marks_fallback_incomplete(
        self,
        monkeypatch,
    ):
        prepared = tweedie_module._prepare_tweedie_density(
            np.array([1.0]),
            np.array([1.0]),
            1.2,
        )
        cache = tweedie_module._PhiEvaluationCache(prepared)
        monkeypatch.setattr(
            tweedie_module,
            "_calibrate_wright_log_t_ceiling",
            lambda prepared: None,
        )
        monkeypatch.setattr(tweedie_module, "_PHI_MAX_NUMERIC_BRANCH_PROBES", 0)

        bounded = tweedie_module._run_phi_bounded_fallback(cache, required=True)

        assert not bounded.success
        assert bounded.branch_switch_detected

    def test_clean_root_cannot_bypass_later_realized_branch_basin(
        self,
        monkeypatch,
    ):
        realized_transition = -4.7

        def synthetic_double_basin(prepared, phi, *, compute_score=False):
            u = float(np.log(phi))
            saddlepoint = u < realized_transition
            if saddlepoint:
                nll = (u + 6.0) ** 2 + 1.0
                score_value = 2.0 * (u + 6.0)
            else:
                nll = (u + 4.0) ** 2 - 1.0
                score_value = 2.0 * (u + 4.0)
            logpdf = np.full(len(prepared.y), -nll, dtype=np.float64)
            score = (
                np.full(len(prepared.y), score_value, dtype=np.float64) if compute_score else None
            )
            mask = np.full(
                prepared.positive_indices.size,
                saddlepoint,
                dtype=np.bool_,
            )
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=logpdf,
                log_phi_score=score,
                positive_saddlepoint_mask=mask,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=int(np.count_nonzero(mask)),
                ),
                score_valid=compute_score,
            )

        monkeypatch.setattr(
            tweedie_module,
            "_evaluate_tweedie_density",
            synthetic_double_basin,
        )

        result = tweedie_module._profile_phi_detailed(
            np.array([1.0]),
            np.array([1.0]),
            1.2,
            phi_method="mle",
            phi_start=float(np.exp(-6.0)),
        )

        assert result.used_fallback
        assert result.branch_switch_detected
        assert result.optimizer == "bounded"
        assert not result.converged
        assert result.nll < 0.0
        np.testing.assert_allclose(np.log(result.phi), -4.0, rtol=0.0, atol=1e-5)

    def test_clean_root_probes_nearest_realized_not_nominal_branch_edge(
        self,
        monkeypatch,
    ):
        prepared, calibrated, realized, fake_evaluate, fake_localize = (
            self._shifted_realized_branch_fixture()
        )
        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", fake_evaluate)
        monkeypatch.setattr(
            tweedie_module,
            "_locate_first_realized_phi_branch_transition",
            fake_localize,
        )
        monkeypatch.setattr(
            tweedie_module,
            "_phi_realized_wright_thresholds",
            lambda prepared: (calibrated.copy(), True),
        )
        cache = tweedie_module._PhiEvaluationCache(prepared)
        root = cache.evaluate(-0.5, compute_score=True)

        candidates, branch_switch, requires_fallback = (
            tweedie_module._better_phi_branch_edge_probes(
                cache,
                [tweedie_module._PhiCandidate(root, "brentq", validated=True)],
            )
        )

        assert cache.evaluate(1e-6, compute_score=False).nll < root.nll
        assert not candidates
        assert not branch_switch
        assert requires_fallback

    def test_clean_root_rejects_delayed_edge_calibrated_below_lower_bound(
        self,
        monkeypatch,
    ):
        prepared = tweedie_module._prepare_tweedie_density(
            np.ones(1),
            np.ones(1),
            1.5,
        )

        def fake_evaluate(prepared, phi, *, compute_score=False):
            u = float(np.log(phi))
            before_realized_edge = u < 0.0
            nll = (u + 0.5) ** 2 if before_realized_edge else (u - 0.2) ** 2 - 2.0
            score_value = 2.0 * (u + 0.5) if before_realized_edge else 2.0 * (u - 0.2)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.array([-nll]),
                log_phi_score=np.array([score_value]) if compute_score else None,
                positive_saddlepoint_mask=np.array([before_realized_edge]),
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=1,
                    n_saddlepoint=int(before_realized_edge),
                ),
                score_valid=compute_score,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", fake_evaluate)
        monkeypatch.setattr(
            tweedie_module,
            "_phi_realized_wright_thresholds",
            lambda prepared: (np.array([-30.0]), True),
        )
        monkeypatch.setattr(
            tweedie_module,
            "_positive_saddlepoint_mask_subset",
            lambda prepared, u, positive_indices: np.ones(
                len(positive_indices),
                dtype=np.bool_,
            ),
        )
        cache = tweedie_module._PhiEvaluationCache(prepared)
        root = cache.evaluate(-0.5, compute_score=True)

        candidates, branch_switch, requires_fallback = (
            tweedie_module._better_phi_branch_edge_probes(
                cache,
                [tweedie_module._PhiCandidate(root, "brentq", validated=True)],
            )
        )

        assert not candidates
        assert not branch_switch
        assert requires_fallback

    def test_profile_clean_root_cannot_skip_shifted_realized_global_basin(
        self,
        monkeypatch,
    ):
        prepared, calibrated, realized, fake_evaluate, fake_localize = (
            self._shifted_realized_branch_fixture()
        )
        monkeypatch.setattr(
            tweedie_module,
            "_prepare_tweedie_density",
            lambda *args, **kwargs: prepared,
        )
        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", fake_evaluate)
        monkeypatch.setattr(
            tweedie_module,
            "_locate_first_realized_phi_branch_transition",
            fake_localize,
        )
        monkeypatch.setattr(
            tweedie_module,
            "_phi_realized_wright_thresholds",
            lambda prepared: (calibrated.copy(), True),
        )

        result = tweedie_module._profile_phi_detailed(
            np.ones(3),
            np.ones(3),
            1.5,
            phi_method="mle",
            phi_start=float(np.exp(-0.5)),
        )

        assert result.used_fallback
        assert result.branch_switch_detected
        assert not result.converged
        np.testing.assert_allclose(np.log(result.phi), 0.2, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(result.nll, -2.0, rtol=0.0, atol=1e-10)

    def test_branch_switched_bounded_basin_is_not_reported_globally_converged(self):
        p = 1.0076499464775093
        y = np.array(
            [
                3.72526820233238e-06,
                1.613555626301106e-06,
                1.3843014725309668e-07,
            ]
        )
        mu = np.array(
            [
                5.302077533862348e-06,
                0.0006695682845387034,
                0.27629554894504144,
            ]
        )
        better_phi = 0.0004034278248560756

        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")
        better_nll = -float(np.mean(tweedie_logpdf(y, mu, better_phi, p)))

        assert result.objective_finite
        assert result.used_fallback
        assert result.branch_switch_detected
        assert result.optimizer == "bounded"
        assert result.nll <= better_nll + 1e-8
        assert not result.converged

    @pytest.mark.parametrize(
        ("p", "y", "mu", "better_phi"),
        [
            pytest.param(
                1.02841933650712,
                np.array([0.013700521389325804, 14.578262417603804]),
                np.array([4.151274914315888e-05, 39050.6378678546]),
                172.0116844400835,
                id="two-positive-responses",
            ),
            pytest.param(
                1.0418278153435188,
                np.array([0.0, 0.0010178807460830947, 0.0, 18.141399305020826]),
                np.array(
                    [
                        0.0001233220265692182,
                        0.0033903221015031517,
                        0.18914375881791862,
                        17.613781883771676,
                    ]
                ),
                0.007394671397250919,
                id="zero-heavy",
            ),
        ],
    )
    def test_better_analytic_branch_edge_promotes_local_root_to_global_fallback(
        self,
        p,
        y,
        mu,
        better_phi,
    ):
        better_nll = -float(np.mean(tweedie_logpdf(y, mu, better_phi, p)))

        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")

        assert result.objective_finite
        assert result.used_fallback
        assert result.branch_switch_detected
        assert result.optimizer == "bounded"
        assert result.nll <= better_nll + 1e-8
        assert not result.converged

    @pytest.mark.parametrize(
        "phi_start",
        [
            pytest.param(None, id="no-warm-start"),
            pytest.param(float(np.exp(5.03825)), id="warm-second-minimum"),
        ],
    )
    def test_multiple_score_roots_are_compared_by_exact_objective(self, phi_start):
        p = 1.1023681265404395
        y = np.array([0.0, 3.30259948, 0.0, 0.0])
        mu = np.array([4.94001718, 5.87112367, 0.20868529, 14.91399245])
        weights = np.array([0.00679757427, 46.1238526, 8.73493384, 3.03253224])
        worse_local_nll = 0.6801646102439636

        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            weights=weights,
            phi_method="mle",
            phi_start=phi_start,
        )
        exact_nll = -float(np.mean(tweedie_logpdf(y, mu, result.phi, p, weights=weights)))

        assert not result.converged
        assert result.objective_finite
        assert result.used_fallback
        assert result.nll == exact_nll
        assert result.nll < worse_local_nll - 0.05
        np.testing.assert_allclose(result.nll, 0.614000943088867, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(np.log(result.phi), 5.038249984662266, rtol=0.0, atol=5e-6)

    def test_unsuccessful_fallback_cannot_be_overwritten_by_a_finite_start(
        self,
        monkeypatch,
    ):
        y = np.array([0.4, 1.5, 5.0])
        mu = np.array([0.7, 1.2, 4.5])
        p = 1.6
        real_wright_bessel = tweedie_module.wright_bessel

        def fail_derivative_recurrence(a, b, t):
            values = np.asarray(real_wright_bessel(a, b, t), dtype=np.float64)
            if np.isclose(b, a):
                return np.full_like(values, np.nan)
            return values

        def unsuccessful_bounded(objective, *, bounds, method, options):
            x = float(np.log(1.0))
            return OptimizeResult(
                x=x,
                fun=objective(x),
                success=False,
                message="forced bounded failure",
            )

        monkeypatch.setattr(tweedie_module, "wright_bessel", fail_derivative_recurrence)
        monkeypatch.setattr(tweedie_module, "minimize_scalar", unsuccessful_bounded)

        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            phi_method="mle",
            phi_start=1.0,
        )

        assert result.objective_finite
        assert np.isfinite(result.nll)
        assert result.used_fallback
        assert result.optimizer == "bounded"
        assert not result.converged
        assert "forced bounded failure" in result.message

    def test_evaluation_accounting_matches_actual_density_passes_and_beats_reference(
        self,
        monkeypatch,
    ):
        y = np.array([0.3, 1.2, 4.5])
        mu = np.array([0.5, 1.5, 3.7])
        p = 1.5
        actual_calls = []
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def spy_evaluate(prepared, phi, *, compute_score=False):
            actual_calls.append((float(phi), compute_score))
            return real_evaluate(prepared, phi, compute_score=compute_score)

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", spy_evaluate)
        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")
        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", real_evaluate)
        reference, reference_calls = _tight_value_only_phi_reference(y, mu, p)

        assert result.n_evaluations == len(actual_calls)
        assert result.n_score_evaluations == sum(score for _, score in actual_calls)
        assert result.n_value_only_evaluations == sum(not score for _, score in actual_calls)
        assert result.n_evaluations == (
            result.n_score_evaluations + result.n_value_only_evaluations
        )
        score_phis = {phi for phi, compute_score in actual_calls if compute_score}
        assert not any(
            phi in score_phis and not compute_score for phi, compute_score in actual_calls
        )
        assert result.n_evaluations < reference_calls
        np.testing.assert_allclose(result.nll, reference.fun, rtol=1e-10, atol=1e-10)

    def test_fallback_branch_localization_has_a_bounded_density_pass_budget(self):
        rng = np.random.default_rng(11)
        n = 100
        x = rng.normal(size=n)
        mu = np.exp(2.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu=mu, phi=3.0, p=1.1, rng=rng)

        result = tweedie_module._profile_phi_detailed(y, mu, 1.1, phi_method="mle")

        assert result.objective_finite
        assert result.used_fallback
        assert result.n_fallback_evaluations <= 350

    def test_large_profile_keeps_enough_branch_edges_for_the_global_basin(self):
        rng = np.random.default_rng(71616)
        for _ in range(68):
            n = 100
            p = float(rng.uniform(1.005, 1.22))
            mu = np.exp(
                rng.normal(
                    rng.uniform(-1.0, 3.0),
                    rng.uniform(0.2, 1.5),
                    n,
                )
            )
            phi = float(10 ** rng.uniform(-1.0, 1.0))
            y = generate_tweedie_cpg(n, mu=mu, phi=phi, p=p, rng=rng)
            weights = 10 ** rng.uniform(-1.0, 1.0, n)

        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            weights=weights,
            phi_method="mle",
        )

        assert p == 1.0444295667392194
        assert result.used_fallback
        assert not result.converged
        np.testing.assert_allclose(result.phi, 0.6387398791114645, rtol=1e-8)
        np.testing.assert_allclose(result.nll, 2.5154680680526336, rtol=0.0, atol=1e-9)
        assert result.n_evaluations <= 350

    def test_zero_heavy_large_profile_uses_large_profile_refinement_cap(
        self,
        monkeypatch,
    ):
        n = 1_000
        y = np.zeros(n)
        y[0] = 1.0
        prepared = tweedie_module._prepare_tweedie_density(y, np.ones(n), 1.5)
        cache = tweedie_module._PhiEvaluationCache(prepared)
        interval_calls = []
        real_interval = tweedie_module._run_phi_bounded_interval

        def force_eight_segments(points):
            finite = [point for point in points if point.objective_finite]
            return [finite[index : index + 2] for index in range(0, 16, 2)]

        def spy_interval(cache, bounds):
            interval_calls.append(bounds)
            return real_interval(cache, bounds)

        monkeypatch.setattr(
            tweedie_module,
            "_finite_phi_fallback_segments",
            force_eight_segments,
        )
        monkeypatch.setattr(
            tweedie_module,
            "_run_phi_bounded_interval",
            spy_interval,
        )

        tweedie_module._run_phi_bounded_fallback(cache, required=True)

        assert len(interval_calls) == 1 + tweedie_module._PHI_MAX_LARGE_FALLBACK_REFINEMENTS

    def test_threshold_selection_does_not_build_observation_by_anchor_matrix(
        self,
        monkeypatch,
    ):
        thresholds = np.linspace(-20.0, 20.0, 10_000)
        anchors = np.linspace(-18.0, 18.0, 114).tolist()
        real_min = tweedie_module.np.min

        def reject_dense_min(values, *args, **kwargs):
            assert np.asarray(values).ndim <= 1
            return real_min(values, *args, **kwargs)

        monkeypatch.setattr(tweedie_module.np, "min", reject_dense_min)

        selected, completed, n_unique = tweedie_module._select_phi_branch_thresholds(
            thresholds,
            anchors,
        )

        assert selected.size == tweedie_module._PHI_MAX_ANALYTIC_BRANCH_EDGES
        assert not completed
        assert n_unique == thresholds.size

    def test_cached_profile_point_retains_one_packed_branch_mask(self):
        n = 1_000
        prepared = tweedie_module._prepare_tweedie_density(
            np.ones(n),
            np.ones(n),
            1.5,
        )
        point = tweedie_module._PhiEvaluationCache(prepared).evaluate(
            0.0,
            compute_score=False,
        )

        assert "positive_saddlepoint_mask" not in vars(point)
        assert "branch_signature" not in vars(point)
        assert len(point.branch_mask.packed) == (n + 7) // 8
        assert point.branch_signature[1] is point.branch_mask.packed
        np.testing.assert_array_equal(
            point.positive_saddlepoint_mask,
            np.zeros(n, dtype=np.bool_),
        )

    def test_fallback_boundary_without_global_certificate_is_not_converged(
        self,
        monkeypatch,
    ):
        hidden_center = 7.123456789
        hidden_half_width = 1e-12

        def hidden_switchback_basin(prepared, phi, *, compute_score=False):
            u = float(np.log(phi))
            hidden = abs(u - hidden_center) <= hidden_half_width
            nll = -100.0 if hidden else u - tweedie_module._LOG_PHI_LOWER_BOUND
            score = np.full(len(prepared.y), np.nan, dtype=np.float64) if compute_score else None
            mask = np.full(prepared.positive_indices.size, hidden, dtype=np.bool_)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(len(prepared.y), -nll, dtype=np.float64),
                log_phi_score=score,
                positive_saddlepoint_mask=mask,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=int(np.count_nonzero(mask)),
                ),
                score_valid=False,
            )

        monkeypatch.setattr(
            tweedie_module,
            "_evaluate_tweedie_density",
            hidden_switchback_basin,
        )

        result = tweedie_module._profile_phi_detailed(
            np.array([1.0]),
            np.array([1.0]),
            1.5,
            phi_method="mle",
            phi_start=1.0,
        )

        assert result.used_fallback
        assert result.lower_boundary
        assert result.phi == tweedie_module._PHI_LOWER_BOUND
        assert result.objective_finite
        assert not result.converged

    def test_fallback_brentq_winner_without_global_certificate_is_not_converged(
        self,
        monkeypatch,
    ):
        hidden_center = 7.123456789
        hidden_half_width = 1e-12
        invalid_seed_u = -6.0

        def hidden_switchback_basin(prepared, phi, *, compute_score=False):
            u = float(np.log(phi))
            hidden = abs(u - hidden_center) <= hidden_half_width
            nll = -100.0 if hidden else u**2
            score_valid = bool(compute_score and abs(u - invalid_seed_u) > 1e-12)
            score = None
            if compute_score:
                score_value = 2.0 * u if score_valid else np.nan
                score = np.full(len(prepared.y), score_value, dtype=np.float64)
            mask = np.full(prepared.positive_indices.size, hidden, dtype=np.bool_)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(len(prepared.y), -nll, dtype=np.float64),
                log_phi_score=score,
                positive_saddlepoint_mask=mask,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=int(np.count_nonzero(mask)),
                ),
                score_valid=score_valid,
            )

        monkeypatch.setattr(
            tweedie_module,
            "_evaluate_tweedie_density",
            hidden_switchback_basin,
        )

        result = tweedie_module._profile_phi_detailed(
            np.array([2.0]),
            np.array([1.0]),
            1.5,
            phi_method="mle",
            phi_start=float(np.exp(invalid_seed_u)),
        )

        assert result.used_fallback
        assert result.optimizer == "brentq"
        assert result.objective_finite
        np.testing.assert_allclose(np.log(result.phi), 0.0, rtol=0.0, atol=1e-8)
        assert not result.converged

    def test_pearson_detailed_result_is_frozen_and_uses_exact_objective(self):
        y = np.array([0.0, 0.4, 1.5, 5.0])
        mu = np.array([0.2, 0.7, 1.2, 4.5])
        weights = np.array([0.3, 0.9, 2.0, 3.5])
        p = 1.6
        expected_phi = max(estimate_phi(y, mu, p, weights=weights), 1e-10)
        expected_nll = -float(np.mean(tweedie_logpdf(y, mu, expected_phi, p, weights=weights)))

        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            weights=weights,
            phi_method="pearson",
        )

        assert result.phi == expected_phi
        assert result.nll == expected_nll
        assert result.optimizer == "pearson"
        assert result.converged
        assert result.objective_finite
        assert result.n_evaluations == result.n_value_only_evaluations == 1
        assert result.n_score_evaluations == result.n_fallback_evaluations == 0
        assert result.score is None
        assert not result.used_fallback
        with pytest.raises(FrozenInstanceError):
            result.phi = 2.0

    def test_tuple_wrapper_delegates_once_and_exactly_matches_detailed_result(
        self,
        monkeypatch,
    ):
        y = np.array([0.3, 1.2, 4.5])
        mu = np.array([0.5, 1.5, 3.7])
        p = 1.5
        expected = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")
        calls = 0

        def detailed_spy(*args, **kwargs):
            nonlocal calls
            calls += 1
            return expected

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", detailed_spy)

        actual = _profile_phi(y, mu, p, phi_method="mle")

        assert actual == (expected.phi, expected.nll)
        assert calls == 1

    def test_nonfinite_exact_objectives_are_never_reported_converged(self, monkeypatch):
        y = np.array([0.0, 0.4, 1.5])
        mu = np.array([0.2, 0.7, 1.2])
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def invalidate_objective(prepared, phi, *, compute_score=False):
            evaluation = real_evaluate(prepared, phi, compute_score=compute_score)
            score = (
                np.full_like(evaluation.logpdf, np.nan, dtype=np.float64) if compute_score else None
            )
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full_like(evaluation.logpdf, np.nan, dtype=np.float64),
                log_phi_score=score,
                positive_saddlepoint_mask=evaluation.positive_saddlepoint_mask,
                diagnostics=evaluation.diagnostics,
                score_valid=False,
            )

        monkeypatch.setattr(
            tweedie_module,
            "_evaluate_tweedie_density",
            invalidate_objective,
        )

        result = tweedie_module._profile_phi_detailed(y, mu, 1.6, phi_method="mle")

        assert np.isinf(result.nll)
        assert not result.objective_finite
        assert not result.converged


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

    def test_design_matrix_error_restores_temporary_family(self, monkeypatch):
        """Temporary p used for design-matrix setup should be exception-safe."""
        model = SuperGLM(
            family=TweedieDistribution(p=1.7),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        caller_state = pickle.dumps(model.__dict__, protocol=5)

        def fail_build(*args, **kwargs):
            raise RuntimeError("build failed")

        monkeypatch.setattr(SuperGLM, "_build_design_matrix", fail_build)

        with pytest.raises(RuntimeError, match="build failed"):
            estimate_tweedie_p(model, X, y)

        assert model.family.p == pytest.approx(1.7)
        assert pickle.dumps(model.__dict__, protocol=5) == caller_state

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

        result = estimate_tweedie_p(
            model,
            X,
            y,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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


@pytest.mark.parametrize(
    "function",
    [SuperGLM.estimate_p, profile_ops_module.estimate_p, estimate_tweedie_p],
)
def test_public_tweedie_profile_entry_points_default_to_mle_and_brent(function):
    signature = inspect.signature(function)

    assert signature.parameters["phi_method"].default == "mle"
    assert signature.parameters["method"].default == "brent"


class TestEstimatePFitMode:
    def test_public_estimate_is_lazy_about_ci_and_profile_evaluations(self, monkeypatch):
        X, y, _ = _tweedie_data(n=48, seed=20260804)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        objective_calls = []
        total_evaluations = 2

        def objective(p):
            objective_calls.append(float(p))
            return 0.0

        result = TweedieProfileResult(
            p_hat=1.5,
            phi_hat=1.0,
            nll=0.0,
            n_evaluations=total_evaluations,
            converged=True,
            method="brent",
            phi_method="mle",
            search_trace=pd.DataFrame({"p": [1.4, 1.5], "nll": [0.1, 0.0]}),
            _objective=objective,
            _ll_scale=float(len(y)),
            _evaluation_count=lambda: total_evaluations,
        )
        profiler_kwargs = {}

        def fake_estimate_tweedie_p(*args, **kwargs):
            profiler_kwargs.update(kwargs)
            return result

        def unexpected_ci(*args, **kwargs):
            raise AssertionError("public estimate_p must not compute a profile CI eagerly")

        monkeypatch.setattr(tweedie_module, "estimate_tweedie_p", fake_estimate_tweedie_p)
        monkeypatch.setattr(result, "ci", unexpected_ci)
        progress_phases = []

        returned = model.estimate_p(
            X,
            y,
            progress_callback=lambda phase, payload: progress_phases.append(phase),
        )

        assert returned is result
        assert profiler_kwargs["phi_method"] == "mle"
        assert profiler_kwargs["method"] == "brent"
        assert progress_phases == ["best_found", "final_refit"]
        assert result._ci_cache == {}
        assert result._ci_details_cache == {}
        assert objective_calls == []
        assert result.n_total_evaluations == total_evaluations

    def test_invalid_complex_weight_is_rejected_before_feature_auto_detection(self):
        X, y, _ = _tweedie_data(n=24, seed=20260719)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            splines=[],
        )
        invalid_weights = np.ones(len(y), dtype=np.complex128)
        invalid_weights[3] = 1.0 + 1.0j
        family_before = model.family

        with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
            model.estimate_p(X, y, sample_weight=invalid_weights, fit_mode="fit")

        assert model.family is family_before
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
        family_before = model.family
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
                method="grid",
                grid=np.array([1.5]),
            )

        assert model.family is family_before
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
        family_before = model.family
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
                method="grid",
                grid=np.array([1.5]),
            )

        assert model.family is family_before
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
            phi_method="mle",
            p_bounds=(1.3, 1.75),
            xatol=1e-4,
        )

        assert result._ci_cache == {}
        assert result._ci_details_cache == {}
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

        class FakeResult:
            p_hat = 1.45
            phi_hat = 1.0
            _objective = None

        def fake_estimate_tweedie_p(*args, **kwargs):
            calls.append(kwargs["fit_mode"])
            return FakeResult()

        monkeypatch.setattr(
            "superglm.profiling.tweedie.estimate_tweedie_p",
            fake_estimate_tweedie_p,
        )

        result = model.estimate_p(X, y, fit_mode="inherit")

        assert result.p_hat == 1.45
        assert calls == ["fit"]
        assert model._last_fit_meta["method"] == "fit"

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


class TestProfileFitParity:
    """Fixed-p profile fits must be identical to the ordinary fit regimes."""

    def test_pirls_profile_forwards_model_controls_and_lambda2(self, monkeypatch):
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 12)})
        y = np.linspace(0.5, 2.0, len(X))
        captured = {}

        def fake_fit_pirls(**kwargs):
            captured.update(kwargs)
            return _profile_solver_result(kwargs["X"])

        monkeypatch.setattr(tweedie_module, "fit_pirls", fake_fit_pirls)
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
            ctx = tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)
        ctx.evaluate(1.5, source="one_point")

        assert captured["lambda2"] == 7.5
        assert captured["max_iter_outer"] == 17
        assert captured["tol"] == pytest.approx(2e-5)
        assert captured["active_set"] is True
        assert captured["convergence"] == "coefficients"
        assert captured["penalty"] is ctx.penalty

    def test_direct_profile_forwards_model_controls_and_lambda2(self, monkeypatch):
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 12)})
        y = np.linspace(0.5, 2.0, len(X))
        captured = {}

        def fake_fit_irls_direct(**kwargs):
            captured.update(kwargs)
            return _profile_solver_result(kwargs["X"]), None

        monkeypatch.setattr(tweedie_module, "fit_irls_direct", fake_fit_irls_direct)
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
            ctx = tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)
        ctx.evaluate(1.5, source="one_point")

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

        monkeypatch.setattr(tweedie_module, "fit_irls_direct", fake_fit_irls_direct)
        monkeypatch.setattr(tweedie_module, "fit_pirls", fail_pirls)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0.25,
            features={},
        )

        ctx = tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)
        assert ctx.penalty.lambda1 > 0.0
        assert not penalty_has_targets(ctx.penalty, ctx.groups)

        ctx.evaluate(1.5, source="one_point")

        assert ctx.use_direct is True
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
        independent_mu = independent.predict(X)
        independent_edf = float(independent.result.effective_df)
        independent_phi = estimate_phi(
            y,
            independent_mu,
            1.5,
            df_resid=float(n) - independent_edf,
        )

        profiled = estimate_tweedie_p(
            SuperGLM(**model_kwargs),
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )
        row = profiled.search_trace.iloc[0]

        assert row["edf"] == pytest.approx(independent_edf, rel=1e-10, abs=1e-10)
        assert profiled.phi_hat == pytest.approx(independent_phi, rel=1e-10, abs=1e-10)

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
        independent.fit_reml(X, y, offset=offset)
        independent_mu = independent._fit_mu
        assert independent_mu is not None
        independent_edf = float(independent.result.effective_df)
        independent_phi = estimate_phi(
            y,
            independent_mu,
            1.5,
            df_resid=float(n) - independent_edf,
        )
        independent_nll = -float(np.mean(tweedie_logpdf(y, independent_mu, independent_phi, 1.5)))

        offset_free_mu = independent.predict(X)
        offset_free_phi = estimate_phi(
            y,
            offset_free_mu,
            1.5,
            df_resid=float(n) - independent_edf,
        )
        offset_free_nll = -float(np.mean(tweedie_logpdf(y, offset_free_mu, offset_free_phi, 1.5)))

        profiled = estimate_tweedie_p(
            SuperGLM(**model_kwargs),
            X,
            y,
            offset=offset,
            fit_mode="fit_reml",
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )

        assert profiled.phi_hat == pytest.approx(independent_phi, rel=1e-9, abs=1e-9)
        assert profiled.nll == pytest.approx(independent_nll, rel=1e-9, abs=1e-9)
        assert abs(profiled.phi_hat - offset_free_phi) > 1.0
        assert abs(profiled.nll - offset_free_nll) > 0.5

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

        result = estimate_tweedie_p(
            model,
            X,
            y,
            offset=offset,
            method="grid",
            grid=np.array([1.4, 1.6]),
            phi_method="pearson",
        )
        result._objective(1.7, source="ci_probe")

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

        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
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

        result = estimate_tweedie_p(
            model,
            X,
            y,
            offset=offset,
            fit_mode="fit_reml",
            method="grid",
            grid=np.array([1.4, 1.6]),
            phi_method="pearson",
        )
        result._objective(1.7, source="ci_probe")

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

        ctx = tweedie_module._build_profile_context_reml(model, X, y, None, None, "pearson", False)

        assert ctx.model.lambda2 == model.lambda2 == 0.25


@pytest.mark.filterwarnings("ignore:Saddlepoint approximation used")
class TestImmutableProfileEvaluations:
    """Finalization must use the complete cached winning candidate."""

    _CANDIDATES = {
        1.2: {
            "mu": 2.0,
            "edf": 0.2,
            "n_iter": 12,
            "fit_converged": True,
            "phi": 12.0,
            "nll": 2.0,
            "phi_converged": True,
            "phi_evaluations": 21,
            "phi_optimizer": "brentq",
            "phi_boundary": "",
            "phi_used_fallback": False,
            "n_positive": 10,
            "n_saddlepoint": 1,
        },
        1.5: {
            "mu": 5.0,
            "edf": 1.5,
            "n_iter": 15,
            "fit_converged": False,
            "phi": 15.0,
            "nll": 0.5,
            "phi_converged": False,
            "phi_evaluations": 51,
            "phi_optimizer": "bounded",
            "phi_boundary": "upper",
            "phi_used_fallback": True,
            "n_positive": 10,
            "n_saddlepoint": 2,
        },
        1.8: {
            "mu": 8.0,
            "edf": 1.8,
            "n_iter": 18,
            "fit_converged": True,
            "phi": 18.0,
            "nll": 1.8,
            "phi_converged": True,
            "phi_evaluations": 81,
            "phi_optimizer": "brentq",
            "phi_boundary": "lower",
            "phi_used_fallback": False,
            "n_positive": 10,
            "n_saddlepoint": 8,
        },
    }

    @classmethod
    def _phi_result(cls, p):
        values = cls._CANDIDATES[round(float(p), 1)]
        return tweedie_module._PhiProfileResult(
            phi=values["phi"],
            nll=values["nll"],
            converged=values["phi_converged"],
            objective_finite=True,
            n_evaluations=values["phi_evaluations"],
            n_score_evaluations=values["phi_evaluations"] - 3,
            n_value_only_evaluations=3,
            n_fallback_evaluations=3 if values["phi_used_fallback"] else 0,
            optimizer=values["phi_optimizer"],
            score=0.0,
            used_fallback=values["phi_used_fallback"],
            fallback_reason="forced fixture" if values["phi_used_fallback"] else None,
            branch_switch_detected=values["phi_used_fallback"],
            lower_boundary=values["phi_boundary"] == "lower",
            upper_boundary=values["phi_boundary"] == "upper",
            diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                n_positive=values["n_positive"],
                n_saddlepoint=values["n_saddlepoint"],
            ),
            message=f"phi profile at p={p}",
        )

    @staticmethod
    def _mutating_callback(rows):
        def callback(row):
            rows.append(row)
            row["phi"] = -999.0
            if row["fit_trace"]:
                row["fit_trace"][0]["loss"] = -999.0

        return callback

    @classmethod
    def _assert_winning_result(cls, ctx, result, callback_rows, fit_calls, phi_calls):
        winner = cls._CANDIDATES[1.5]
        assert result.p_hat == pytest.approx(1.5)
        assert result.phi_hat == pytest.approx(winner["phi"])
        assert result.nll == pytest.approx(winner["nll"])
        assert result.n_evaluations == len(result.search_trace) == 3
        assert fit_calls == [1.2, 1.5, 1.8]
        assert [call[0] for call in phi_calls] == [1.2, 1.5, 1.8]
        assert [call[1] for call in phi_calls] == [None, 12.0, 15.0]

        winning_row = result.search_trace.loc[result.search_trace["p"] == 1.5].iloc[0]
        assert winning_row["edf"] == pytest.approx(winner["edf"])
        assert not bool(winning_row["fit_converged"])
        assert not bool(winning_row["phi_converged"])
        assert winning_row["phi_n_evaluations"] == winner["phi_evaluations"]
        assert winning_row["phi_boundary"] == winner["phi_boundary"]
        assert winning_row["phi_optimizer"] == winner["phi_optimizer"]
        assert bool(winning_row["objective_finite"])
        assert bool(winning_row["phi_used_fallback"])
        assert winning_row["n_positive"] == winner["n_positive"]
        assert winning_row["n_saddlepoint"] == winner["n_saddlepoint"]
        assert winning_row["saddlepoint_fraction"] == pytest.approx(0.2)
        assert winning_row["fit_trace"][0]["loss"] != -999.0
        assert callback_rows[1]["phi"] == -999.0

        assert result.n_positive == winner["n_positive"]
        assert result.n_saddlepoint == winner["n_saddlepoint"]
        assert result.saddlepoint_fraction == pytest.approx(0.2)

        record = ctx._evaluation_cache[1.5]
        assert record.p == pytest.approx(1.5)
        assert isinstance(record.phi_result, tweedie_module._PhiProfileResult)
        assert record.phi == pytest.approx(winner["phi"])
        assert record.nll == pytest.approx(winner["nll"])
        assert record.mu.flags.owndata
        assert not record.mu.flags.writeable
        np.testing.assert_allclose(record.mu, winner["mu"])
        with pytest.raises(FrozenInstanceError):
            record.edf = -1.0

    def test_fit_context_finalizes_cached_winner_without_extra_work(self, monkeypatch):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        fit_calls = []
        phi_calls = []
        callback_rows = []

        def fake_fit_irls_direct(**kwargs):
            p = round(float(kwargs["family"].p), 1)
            fit_calls.append(p)
            values = self._CANDIDATES[p]
            mu = np.full(len(y), values["mu"])
            result = SimpleNamespace(
                beta=np.log(mu),
                intercept=0.0,
                effective_df=values["edf"],
                n_iter=values["n_iter"],
                converged=values["fit_converged"],
                iteration_log=[SimpleNamespace(iteration=values["n_iter"], deviance=100.0 * p)],
            )
            return result, None

        def fake_profile_phi(y_arg, mu, p, **kwargs):
            p = round(float(p), 1)
            values = self._CANDIDATES[p]
            np.testing.assert_allclose(mu, values["mu"])
            phi_calls.append((p, kwargs.get("phi_start", "missing")))
            return self._phi_result(p)

        monkeypatch.setattr(tweedie_module, "fit_irls_direct", fake_fit_irls_direct)
        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", fake_profile_phi)

        ctx = tweedie_module._ProfileContext(
            y_arr=y,
            w_arr=np.ones(len(y)),
            offset_arr=np.zeros(len(y)),
            dm=SimpleNamespace(matvec=lambda beta: beta),
            groups=[],
            link=LogLink(),
            penalty=None,
            use_direct=True,
            lambda2=None,
            direct_solve="auto",
            phi_method="mle",
            verbose=False,
            ll_scale=float(len(y)),
            trace_callback=self._mutating_callback(callback_rows),
            trace_iterations=True,
        )
        for p in (1.2, 1.5, 1.8):
            ctx.evaluate(p, source="grid")

        def fail_extra_work(*args, **kwargs):
            raise AssertionError("finalization performed extra fit/profile/density work")

        ctx.evaluate = fail_extra_work
        monkeypatch.setattr(tweedie_module, "_tweedie_logpdf_impl", fail_extra_work)
        result = ctx.finalize(1.5, method="grid", converged=True)

        self._assert_winning_result(ctx, result, callback_rows, fit_calls, phi_calls)

    def test_reml_context_finalizes_cached_winner_without_extra_work(self, monkeypatch):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        fit_calls = []
        phi_calls = []
        callback_rows = []
        case = self

        class FakeREMLModel:
            family = None
            result = None
            _reml_result = None

            def fit_reml(self, X, y_arg, *, sample_weight=None, offset=None):
                p = round(float(self.family.p), 1)
                fit_calls.append(p)
                values = case._CANDIDATES[p]
                self._mu = np.full(len(y), values["mu"])
                self.result = SimpleNamespace(
                    effective_df=values["edf"],
                    converged=values["fit_converged"],
                )
                self._reml_result = SimpleNamespace(
                    n_reml_iter=values["n_iter"],
                    objective_history=[100.0 * p, 10.0 * p],
                )

            def predict(self, X, offset=None):
                return self._mu

        def fake_profile_phi(y_arg, mu, p, **kwargs):
            p = round(float(p), 1)
            values = self._CANDIDATES[p]
            np.testing.assert_allclose(mu, values["mu"])
            phi_calls.append((p, kwargs.get("phi_start", "missing")))
            return self._phi_result(p)

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", fake_profile_phi)
        ctx = tweedie_module._ProfileContextREML(
            model=FakeREMLModel(),
            X=np.ones((len(y), 1)),
            y=y,
            sample_weight=None,
            offset=None,
            w_arr=np.ones(len(y)),
            phi_method="mle",
            verbose=False,
            ll_scale=float(len(y)),
            trace_callback=self._mutating_callback(callback_rows),
            trace_iterations=True,
        )
        for p in (1.2, 1.5, 1.8):
            ctx.evaluate(p, source="grid")

        def fail_extra_work(*args, **kwargs):
            raise AssertionError("finalization performed extra fit/profile/density work")

        ctx.evaluate = fail_extra_work
        monkeypatch.setattr(tweedie_module, "_tweedie_logpdf_impl", fail_extra_work)
        result = ctx.finalize(1.5, method="grid", converged=True)

        self._assert_winning_result(ctx, result, callback_rows, fit_calls, phi_calls)

    @staticmethod
    def _assert_exact_float_candidates(ctx, first, second, fit_calls, phi_calls, callbacks):
        first_result = ctx.finalize(first, method="grid", converged=True)
        second_result = ctx.finalize(second, method="grid", converged=True)

        assert fit_calls == [first, second]
        assert phi_calls == [first, second]
        assert len(ctx._evaluation_cache) == len(callbacks) == 2
        assert list(ctx._evaluation_cache) == [first, second]
        assert [row["source"] for row in callbacks] == ["near_first", "near_second"]
        assert first_result.p_hat == first
        assert first_result.phi_hat == pytest.approx(101.0)
        assert first_result.nll == pytest.approx(1.01)
        assert second_result.p_hat == second
        assert second_result.phi_hat == pytest.approx(202.0)
        assert second_result.nll == pytest.approx(2.02)
        assert list(second_result.search_trace["p"]) == [first, second]
        assert list(second_result.search_trace["source"]) == ["near_first", "near_second"]

    def test_fit_context_preserves_distinct_exact_float_cache_keys(self, monkeypatch):
        first = 1.5000000000001
        second = 1.5000000000004
        y = np.array([0.0, 1.0])
        fit_calls = []
        phi_calls = []
        callbacks = []

        def candidate_index(p):
            return 1 if p == first else 2

        def fake_fit_irls_direct(**kwargs):
            p = float(kwargs["family"].p)
            fit_calls.append(p)
            index = candidate_index(p)
            mu = np.full(len(y), float(index))
            result = SimpleNamespace(
                beta=np.log(mu),
                intercept=0.0,
                effective_df=0.1 * index,
                n_iter=index,
                converged=True,
                iteration_log=[],
            )
            return result, None

        def fake_profile_phi(y_arg, mu, p, **kwargs):
            p = float(p)
            phi_calls.append(p)
            index = candidate_index(p)
            return replace(
                self._phi_result(1.5),
                phi=101.0 if index == 1 else 202.0,
                nll=1.01 if index == 1 else 2.02,
            )

        monkeypatch.setattr(tweedie_module, "fit_irls_direct", fake_fit_irls_direct)
        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", fake_profile_phi)
        ctx = tweedie_module._ProfileContext(
            y_arr=y,
            w_arr=np.ones(len(y)),
            offset_arr=np.zeros(len(y)),
            dm=SimpleNamespace(matvec=lambda beta: beta),
            groups=[],
            link=LogLink(),
            penalty=None,
            use_direct=True,
            lambda2=None,
            direct_solve="auto",
            phi_method="mle",
            verbose=False,
            ll_scale=float(len(y)),
            trace_callback=callbacks.append,
        )

        assert ctx.evaluate(first, source="near_first") == pytest.approx(1.01)
        assert ctx.evaluate(second, source="near_second") == pytest.approx(2.02)
        self._assert_exact_float_candidates(ctx, first, second, fit_calls, phi_calls, callbacks)

    def test_reml_context_preserves_distinct_exact_float_cache_keys(self, monkeypatch):
        first = 1.5000000000001
        second = 1.5000000000004
        y = np.array([0.0, 1.0])
        fit_calls = []
        phi_calls = []
        callbacks = []
        case = self

        def candidate_index(p):
            return 1 if p == first else 2

        class FakeREMLModel:
            family = None
            result = None
            _reml_result = None

            def fit_reml(self, X, y_arg, *, sample_weight=None, offset=None):
                p = float(self.family.p)
                fit_calls.append(p)
                index = candidate_index(p)
                self._mu = np.full(len(y), float(index))
                self.result = SimpleNamespace(effective_df=0.1 * index, converged=True)
                self._reml_result = SimpleNamespace(
                    n_reml_iter=index,
                    objective_history=[],
                )

            def predict(self, X, offset=None):
                return self._mu

        def fake_profile_phi(y_arg, mu, p, **kwargs):
            p = float(p)
            phi_calls.append(p)
            index = candidate_index(p)
            return replace(
                case._phi_result(1.5),
                phi=101.0 if index == 1 else 202.0,
                nll=1.01 if index == 1 else 2.02,
            )

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", fake_profile_phi)
        ctx = tweedie_module._ProfileContextREML(
            model=FakeREMLModel(),
            X=np.ones((len(y), 1)),
            y=y,
            sample_weight=None,
            offset=None,
            w_arr=np.ones(len(y)),
            phi_method="mle",
            verbose=False,
            ll_scale=float(len(y)),
            trace_callback=callbacks.append,
        )

        assert ctx.evaluate(first, source="near_first") == pytest.approx(1.01)
        assert ctx.evaluate(second, source="near_second") == pytest.approx(2.02)
        self._assert_exact_float_candidates(ctx, first, second, fit_calls, phi_calls, callbacks)


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

        messages = [str(item.message) for item in caught]
        assert sum("Saddlepoint approximation used" in message for message in messages) == 1
        assert sum("near-power boundary instability" in message for message in messages) == 1
        assert result.saddlepoint_fraction >= 0.25
        assert result.n_saddlepoint > 0
        assert result.n_positive > 0
        assert any("Saddlepoint approximation used" in message for message in result.warnings)
        assert any("inner phi profile did not converge" in message for message in result.warnings)
        assert not result.converged

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


def _fake_search_context(objective):
    """Build a cheap real profile context whose evaluations are synthetic."""
    ctx = tweedie_module._ProfileContext(
        y_arr=np.ones(1),
        w_arr=np.ones(1),
        offset_arr=np.zeros(1),
        dm=SimpleNamespace(),
        groups=[],
        link=LogLink(),
        penalty=None,
        use_direct=True,
        lambda2=None,
        direct_solve="auto",
        phi_method="mle",
        verbose=False,
        ll_scale=1.0,
    )

    def evaluate(p, source=""):
        key = float(p)
        if key in ctx._evaluation_cache:
            return ctx._evaluation_cache[key].nll
        spec = objective(key)
        phi_result = tweedie_module._PhiProfileResult(
            phi=float(spec.get("phi", 1.0)),
            nll=float(spec.get("nll", np.inf)),
            converged=bool(spec.get("phi_converged", True)),
            objective_finite=bool(spec.get("objective_finite", True)),
            n_evaluations=7,
            n_score_evaluations=5,
            n_value_only_evaluations=2,
            n_fallback_evaluations=1,
            optimizer="brentq",
            score=0.0,
            used_fallback=True,
            fallback_reason="synthetic",
            branch_switch_detected=True,
            lower_boundary=bool(spec.get("phi_lower_boundary", False)),
            upper_boundary=bool(spec.get("phi_upper_boundary", False)),
            diagnostics=tweedie_module._TweedieLogpdfDiagnostics(),
            message="synthetic phi profile",
        )
        record = tweedie_module._ProfileEvaluation(
            step=len(ctx._evaluation_cache),
            p=float(spec.get("p", key)),
            mu=np.ones(1),
            edf=1.0,
            n_iter=1,
            fit_converged=bool(spec.get("fit_converged", True)),
            source=source,
            fit_trace=(),
            fit_trace_kind="",
            phi_result=phi_result,
        )
        ctx._evaluation_cache[key] = record
        return record.nll

    ctx.evaluate = evaluate
    return ctx


class TestOuterSearchHonesty:
    @pytest.mark.parametrize("method", ["brent", "grid_refine", "profile_opt"])
    @pytest.mark.parametrize(
        ("case", "diagnostic"),
        [
            ("missing_success", "success"),
            ("non_boolean_success", "boolean"),
            ("missing_x", "result.x"),
            ("malformed_x", "one scalar"),
            ("nonfinite_x", "finite"),
            ("infinite_x", "finite"),
        ],
    )
    def test_malformed_optimizer_result_preserves_best_cached_record(
        self, monkeypatch, method, case, diagnostic
    ):
        ctx = _fake_search_context(lambda p: {"nll": (p - 1.5) ** 2})
        valid_x = 0.0 if method == "profile_opt" else 1.5
        attributes = {"message": "synthetic malformed result"}
        if case != "missing_success":
            attributes["success"] = 1 if case == "non_boolean_success" else True
        if case != "missing_x":
            if case == "malformed_x":
                attributes["x"] = np.array([valid_x, valid_x])
            elif case == "nonfinite_x":
                attributes["x"] = np.nan
            elif case == "infinite_x":
                attributes["x"] = np.inf
            else:
                attributes["x"] = valid_x
        optimizer_result = SimpleNamespace(**attributes)

        def scalar_optimizer(objective, **kwargs):
            objective(1.5)
            return optimizer_result

        def vector_optimizer(objective, **kwargs):
            objective(np.array([0.0]))
            return optimizer_result

        monkeypatch.setattr(tweedie_module, "minimize_scalar", scalar_optimizer)
        monkeypatch.setattr(tweedie_module, "minimize", vector_optimizer)

        if method == "brent":
            result = tweedie_module._search_brent(ctx, (1.1, 1.9), 1e-3, 30)
        elif method == "grid_refine":
            result = tweedie_module._search_grid_refine(ctx, (1.2, 1.8), 3, 1e-3, 30)
        else:
            result = tweedie_module._search_profile_opt(ctx, (1.1, 1.9), "L-BFGS-B", 1e-3, 30)

        assert result.p_hat == pytest.approx(1.5)
        assert not result.outer_converged
        assert not result.converged
        assert diagnostic in result.outer_message.lower()

    @pytest.mark.parametrize("method", ["brent", "grid_refine"])
    def test_out_of_range_scalar_optimizer_candidate_is_not_converged(self, monkeypatch, method):
        ctx = _fake_search_context(lambda p: {"nll": (p - 1.5) ** 2})

        def out_of_range_optimizer(objective, **kwargs):
            objective(1.5)
            return OptimizeResult(x=2.5, fun=0.0, success=True, message="synthetic")

        monkeypatch.setattr(tweedie_module, "minimize_scalar", out_of_range_optimizer)
        if method == "brent":
            result = tweedie_module._search_brent(ctx, (1.1, 1.9), 1e-3, 30)
        else:
            result = tweedie_module._search_grid_refine(ctx, (1.2, 1.8), 3, 1e-3, 30)

        assert result.p_hat == pytest.approx(1.5)
        assert not result.outer_converged
        assert "outside" in result.outer_message.lower()

    @pytest.mark.parametrize(
        ("winner", "boundary"),
        [(1.1, "lower"), (1.9, "upper")],
    )
    def test_brent_evaluates_and_can_select_endpoint(self, monkeypatch, winner, boundary):
        ctx = _fake_search_context(lambda p: {"nll": (p - winner) ** 2})

        def bounded(objective, **kwargs):
            objective(1.5)
            return OptimizeResult(x=1.5, fun=objective(1.5), success=True, message="ok")

        monkeypatch.setattr(tweedie_module, "minimize_scalar", bounded)
        result = tweedie_module._search_brent(ctx, (1.1, 1.9), 1e-3, 30)

        assert result.p_hat == pytest.approx(winner)
        assert result.outer_boundary == boundary
        assert result.outer_converged
        assert result.converged
        assert set(result.search_trace["p"]) == {1.1, 1.5, 1.9}

    def test_brent_evaluates_optimizer_reported_candidate(self, monkeypatch):
        ctx = _fake_search_context(lambda p: {"nll": (p - 1.4) ** 2})

        def bounded_without_objective_call(objective, **kwargs):
            return OptimizeResult(x=1.4, fun=0.0, success=True, message="ok")

        monkeypatch.setattr(tweedie_module, "minimize_scalar", bounded_without_objective_call)
        result = tweedie_module._search_brent(ctx, (1.1, 1.9), 1e-3, 30)

        assert result.p_hat == pytest.approx(1.4)
        assert 1.4 in set(result.search_trace["p"])

    @pytest.mark.parametrize(
        ("fit_converged", "phi_converged"),
        [(False, True), (True, False)],
    )
    def test_aggregate_convergence_includes_winning_fit_and_phi(
        self, monkeypatch, fit_converged, phi_converged
    ):
        def objective(p):
            return {
                "nll": (p - 1.5) ** 2,
                "fit_converged": fit_converged if p == 1.5 else True,
                "phi_converged": phi_converged if p == 1.5 else True,
            }

        ctx = _fake_search_context(objective)

        def bounded(fn, **kwargs):
            return OptimizeResult(x=1.5, fun=fn(1.5), success=True, message="ok")

        monkeypatch.setattr(tweedie_module, "minimize_scalar", bounded)
        result = tweedie_module._search_brent(ctx, (1.1, 1.9), 1e-3, 30)

        assert result.p_hat == pytest.approx(1.5)
        assert result.outer_converged
        assert result.fit_converged is fit_converged
        assert result.phi_converged is phi_converged
        assert not result.converged
        assert result.objective_finite
        assert result.phi_n_evaluations == 7
        assert result.phi_n_score_evaluations == 5
        assert result.phi_n_value_only_evaluations == 2
        assert result.phi_n_fallback_evaluations == 1
        assert result.phi_optimizer == "brentq"
        assert result.phi_score == pytest.approx(0.0)
        assert result.phi_used_fallback
        assert result.phi_fallback_reason == "synthetic"
        assert result.phi_branch_switch_detected
        assert result.phi_message == "synthetic phi profile"
        winning_row = result.search_trace.loc[result.search_trace["p"] == 1.5].iloc[0]
        assert winning_row["phi_n_score_evaluations"] == 5
        assert winning_row["phi_n_value_only_evaluations"] == 2
        assert winning_row["phi_n_fallback_evaluations"] == 1
        assert winning_row["phi_fallback_reason"] == "synthetic"
        assert bool(winning_row["phi_branch_switch_detected"])
        failure_text = " ".join(result.warnings).lower()
        if not fit_converged:
            assert "fit" in failure_text and "converge" in failure_text
        if not phi_converged:
            assert "phi" in failure_text and "converge" in failure_text

    def test_grid_refine_keeps_better_coarse_record(self, monkeypatch):
        ctx = _fake_search_context(lambda p: {"nll": 0.0 if p == 1.5 else 1.0})

        def worse_refinement(objective, **kwargs):
            return OptimizeResult(x=1.6, fun=objective(1.6), success=True, message="ok")

        monkeypatch.setattr(tweedie_module, "minimize_scalar", worse_refinement)
        result = tweedie_module._search_grid_refine(ctx, (1.2, 1.8), 3, 1e-3, 30)

        assert result.p_hat == pytest.approx(1.5)
        assert result.nll == pytest.approx(0.0)

    def test_failed_profile_optimizer_returns_best_valid_cached_record(self, monkeypatch):
        ctx = _fake_search_context(lambda p: {"nll": 0.0 if p == 1.5 else 1.0})

        def failed_optimizer(objective, **kwargs):
            q = (1.6 - 1.1) / (1.9 - 1.1)
            t = np.log(q / (1.0 - q))
            objective(np.array([t]))
            return OptimizeResult(
                x=np.array([t]), fun=1.0, success=False, message="forced optimizer failure"
            )

        monkeypatch.setattr(tweedie_module, "minimize", failed_optimizer)
        result = tweedie_module._search_profile_opt(ctx, (1.1, 1.9), "L-BFGS-B", 1e-3, 30)

        assert result.p_hat == pytest.approx(1.5)
        assert np.isfinite(result.nll)
        assert not result.outer_converged
        assert "forced optimizer failure" in result.outer_message
        assert not result.converged
        assert any("outer" in warning.lower() for warning in result.warnings)

    def test_explicit_grid_defines_effective_boundary_and_ties_keep_first(self):
        ctx = _fake_search_context(lambda p: {"nll": 0.0})
        grid = np.array([1.6, 1.4, 1.2])

        result = tweedie_module._search_grid(ctx, (1.05, 1.95), len(grid), grid)

        assert result.p_hat == pytest.approx(1.6)
        assert result.outer_boundary == "upper"

    def test_one_point_grid_has_no_directional_outer_boundary(self):
        ctx = _fake_search_context(lambda p: {"nll": 0.0})

        result = tweedie_module._search_grid(ctx, (1.05, 1.95), 1, np.array([1.5]))

        assert result.outer_boundary is None
        assert not any("outside" in warning.lower() for warning in result.warnings)

    def test_winning_phi_boundary_is_reported(self):
        ctx = _fake_search_context(
            lambda p: {"nll": (p - 1.5) ** 2, "phi_lower_boundary": p == 1.5}
        )

        result = tweedie_module._search_grid(ctx, (1.1, 1.9), 3, np.array([1.3, 1.5, 1.7]))

        assert result.phi_boundary == "lower"
        assert any(
            "phi" in warning.lower() and "boundary" in warning.lower()
            for warning in result.warnings
        )

    def test_context_finalize_rejects_an_invalid_cached_record(self):
        ctx = _fake_search_context(lambda p: {"nll": 0.0, "objective_finite": False})
        ctx.evaluate(1.5, source="grid")

        with pytest.raises(RuntimeError, match=r"grid.*invalid.*1\.5"):
            ctx.finalize(1.5, method="grid", converged=True)

    def test_all_invalid_records_raise_descriptive_error(self):
        def invalid(p):
            if p == 1.2:
                return {"nll": np.nan}
            if p == 1.3:
                return {"nll": 0.0, "phi": np.nan}
            if p == 1.4:
                return {"nll": 0.0, "phi": 0.0}
            if p == 1.5:
                return {"nll": 0.0, "objective_finite": False}
            return {"p": np.inf, "nll": 0.0}

        ctx = _fake_search_context(invalid)
        grid = np.array([1.2, 1.3, 1.4, 1.5, 1.6])

        with pytest.raises(RuntimeError, match=r"grid.*5.*1\.2.*1\.6"):
            tweedie_module._search_grid(ctx, (1.05, 1.95), len(grid), grid)

    @pytest.mark.parametrize("method", ["brent", "grid_refine", "profile_opt"])
    def test_each_optimizer_search_rejects_all_invalid_records(self, monkeypatch, method):
        ctx = _fake_search_context(lambda p: {"nll": 0.0, "phi": 0.0})

        def bounded(objective, **kwargs):
            return OptimizeResult(x=1.5, fun=objective(1.5), success=False, message="failed")

        def general(objective, **kwargs):
            return OptimizeResult(
                x=np.array([0.0]), fun=objective(np.array([0.0])), success=False, message="failed"
            )

        monkeypatch.setattr(tweedie_module, "minimize_scalar", bounded)
        monkeypatch.setattr(tweedie_module, "minimize", general)

        with pytest.raises(RuntimeError, match=rf"{method}.*evaluations.*1\.1.*1\.9"):
            if method == "brent":
                tweedie_module._search_brent(ctx, (1.1, 1.9), 1e-3, 30)
            elif method == "grid_refine":
                tweedie_module._search_grid_refine(ctx, (1.1, 1.9), 3, 1e-3, 30)
            else:
                tweedie_module._search_profile_opt(ctx, (1.1, 1.9), "L-BFGS-B", 1e-3, 30)

    @pytest.mark.parametrize(
        ("reml_converged", "solver_converged", "expected_fit"),
        [(False, True, False), (True, False, False), (None, True, True)],
    )
    def test_reml_fit_convergence_combines_outer_reml_and_final_solver(
        self, monkeypatch, reml_converged, solver_converged, expected_fit
    ):
        class FakeREMLModel:
            family = None
            result = None
            _reml_result = None

            def fit_reml(self, X, y, *, sample_weight=None, offset=None):
                self._fit_mu = np.ones(len(y))
                self.result = SimpleNamespace(effective_df=1.0, converged=solver_converged)
                self._reml_result = (
                    None
                    if reml_converged is None
                    else SimpleNamespace(
                        n_reml_iter=2,
                        objective_history=[],
                        converged=reml_converged,
                    )
                )

        phi_result = tweedie_module._PhiProfileResult(
            phi=1.0,
            nll=0.0,
            converged=True,
            objective_finite=True,
            n_evaluations=1,
            n_score_evaluations=1,
            n_value_only_evaluations=0,
            n_fallback_evaluations=0,
            optimizer="brentq",
            score=0.0,
            used_fallback=False,
            fallback_reason=None,
            branch_switch_detected=False,
            lower_boundary=False,
            upper_boundary=False,
            diagnostics=tweedie_module._TweedieLogpdfDiagnostics(),
            message="ok",
        )
        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", lambda *a, **k: phi_result)
        ctx = tweedie_module._ProfileContextREML(
            model=FakeREMLModel(),
            X=np.ones((3, 1)),
            y=np.ones(3),
            sample_weight=None,
            offset=None,
            w_arr=np.ones(3),
            phi_method="mle",
            verbose=False,
            ll_scale=3.0,
        )

        result = tweedie_module._search_grid(ctx, (1.1, 1.9), 1, np.array([1.5]))

        assert result.solver_converged is solver_converged
        assert result.reml_converged is reml_converged
        assert result.fit_converged is expected_fit
        row = result.search_trace.iloc[0]
        assert bool(row["solver_converged"]) is solver_converged
        if reml_converged is None:
            assert row["reml_converged"] is None
        else:
            assert bool(row["reml_converged"]) is reml_converged

    def test_partial_reml_results_are_not_certified_converged(self, monkeypatch):
        template = _fake_search_context(lambda p: {"nll": 0.0})
        template.evaluate(1.5)
        phi_result = template._evaluation_cache[1.5].phi_result

        class PartialREMLModel:
            family = None
            result = None
            _reml_result = None

            def fit_reml(self, X, y, *, sample_weight=None, offset=None):
                self._fit_mu = np.ones(len(y))
                self.result = SimpleNamespace(effective_df=1.0)
                self._reml_result = SimpleNamespace(n_reml_iter=1, objective_history=[])

        monkeypatch.setattr(tweedie_module, "_profile_phi_detailed", lambda *a, **k: phi_result)
        ctx = tweedie_module._ProfileContextREML(
            model=PartialREMLModel(),
            X=np.ones((3, 1)),
            y=np.ones(3),
            sample_weight=None,
            offset=None,
            w_arr=np.ones(3),
            phi_method="mle",
            verbose=False,
            ll_scale=3.0,
        )

        result = tweedie_module._search_grid(ctx, (1.1, 1.9), 1, np.array([1.5]))

        assert not result.solver_converged
        assert result.reml_converged is False
        assert not result.fit_converged
        assert not result.converged


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
            result = estimate_tweedie_p(
                m,
                X,
                y,
                method=method,
                p_bounds=(1.1, 1.9),
                phi_method="pearson",
            )
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


class TestDensityProvenance:
    @pytest.mark.parametrize(
        ("n_positive", "n_saddlepoint", "method", "exact"),
        [
            (0, 0, "exact", True),
            (10, 0, "exact", True),
            (10, 1, "hybrid_exact_saddlepoint", False),
            (10, 10, "saddlepoint", False),
        ],
    )
    def test_final_density_classification(self, n_positive, n_saddlepoint, method, exact):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = _finalized_density_result(
                n_positive=n_positive,
                n_saddlepoint=n_saddlepoint,
            )

        assert result.density_method == method
        assert result.density_exact is exact
        assert result.converged
        row = result.search_trace.iloc[0]
        assert row["density_method"] == method
        assert bool(row["density_exact"]) is exact
        assert list(result.search_trace.columns)[-2:] == ["density_method", "density_exact"]

    @pytest.mark.parametrize(
        ("n_saddlepoint", "severity", "warning_pattern"),
        [
            (9, "label", None),
            (10, "warning", "Saddlepoint approximation used"),
            (49, "warning", "Saddlepoint approximation used"),
            (50, "high", "High-severity.*Saddlepoint approximation used"),
        ],
    )
    def test_saddlepoint_warning_thresholds_are_inclusive(
        self, n_saddlepoint, severity, warning_pattern
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _finalized_density_result(
                n_positive=100,
                n_saddlepoint=n_saddlepoint,
            )

        assert result.density_warning_severity == severity
        if warning_pattern is None:
            assert caught == []
            assert result.warnings == []
        else:
            assert len(caught) == 1
            assert len(result.warnings) == 1
            assert __import__("re").search(warning_pattern, str(caught[0].message))

    @pytest.mark.parametrize("p", [1.08, 1.98])
    def test_near_power_boundary_is_separate_from_density_approximation(self, p):
        with pytest.warns(UserWarning, match="near-power boundary instability") as caught:
            result = _finalized_density_result(p=p, n_positive=100, n_saddlepoint=1)

        assert len(caught) == 1
        assert result.near_power_boundary
        assert result.density_method == "hybrid_exact_saddlepoint"
        assert not result.density_exact
        assert result.density_warning_severity == "high"
        assert result.converged
        assert "inherent boundary instability" in result.warnings[0]

    @pytest.mark.parametrize("p", [1.08, 1.98])
    def test_exact_density_near_power_boundary_is_not_flagged(self, p):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _finalized_density_result(p=p, n_positive=10, n_saddlepoint=0)

        assert caught == []
        assert not result.near_power_boundary
        assert result.density_method == "exact"
        assert result.density_exact

    @pytest.mark.parametrize(
        "counts",
        [(-1, 0), (10, -1), (10, 11), (0, 1), (True, 0), ("10", 0), ([10], 0)],
    )
    def test_inconsistent_density_counts_are_not_certified_exact(self, counts):
        with pytest.warns(UserWarning, match="inconsistent density diagnostics"):
            result = _finalized_density_result(
                n_positive=counts[0],
                n_saddlepoint=counts[1],
            )

        assert not result.density_exact
        assert result.density_warning_severity == "high"

    @pytest.mark.parametrize(
        "diagnostics",
        [
            None,
            SimpleNamespace(n_positive="10", n_saddlepoint=[1]),
        ],
        ids=["missing", "malformed"],
    )
    def test_invalid_winning_diagnostics_finalize_with_normalized_trace_counts(self, diagnostics):
        with pytest.warns(UserWarning, match="inconsistent density diagnostics"):
            result = _finalized_density_result(diagnostics_override=diagnostics)

        row = result.search_trace.iloc[0]
        assert result.n_positive == row["n_positive"] == -1
        assert result.n_saddlepoint == row["n_saddlepoint"] == -1
        assert isinstance(row["n_positive"], int | np.integer)
        assert isinstance(row["n_saddlepoint"], int | np.integer)
        assert row["density_method"] == result.density_method == "hybrid_exact_saddlepoint"
        assert bool(row["density_exact"]) is result.density_exact is False

    def test_legacy_positional_result_derives_density_fields_without_shifting_warnings(self):
        trace = pd.DataFrame({"p": [1.5], "nll": [0.0]})
        result = tweedie_module.TweedieProfileResult(
            1.5,
            1.0,
            0.0,
            1,
            True,
            "brent",
            "mle",
            trace,
            0.2,
            2,
            10,
            ["legacy warning"],
        )

        assert result.warnings == ["legacy warning"]
        assert result.density_method == "hybrid_exact_saddlepoint"
        assert result.density_exact is False

    def test_full_legacy_positional_result_preserves_private_callback_slots(self):
        trace = pd.DataFrame({"p": [1.5], "nll": [0.0]})

        def objective(p):
            return float(p)

        def evaluation_count():
            return 7

        def evaluation_record(p):
            return "record", float(p)

        ci_cache = {0.05: (1.4, 1.6)}
        ci_details_cache = {}

        result = tweedie_module.TweedieProfileResult(
            1.5,
            1.0,
            0.0,
            1,
            True,
            "brent",
            "mle",
            trace,
            0.2,
            2,
            10,
            ["legacy warning"],
            True,
            "",
            None,
            True,
            True,
            None,
            True,
            True,
            4,
            3,
            1,
            0,
            "brentq",
            0.0,
            False,
            None,
            False,
            "",
            "",
            objective,
            100.0,
            ci_cache,
            ci_details_cache,
            (1.1, 1.9),
            (1.3, 1.5, 1.7),
            evaluation_count,
            evaluation_record,
        )

        assert result._objective is objective
        assert result._ll_scale == 100.0
        assert result._ci_cache is ci_cache
        assert result._ci_details_cache is ci_details_cache
        assert result._ci_p_range == (1.1, 1.9)
        assert result._ci_seed_points == (1.3, 1.5, 1.7)
        assert result._evaluation_count is evaluation_count
        assert result._evaluation_record is evaluation_record
        assert result.density_method == "hybrid_exact_saddlepoint"
        assert result.density_exact is False

    def test_pre_density_field_pickle_restores_compatibility_state_for_ci(self):
        result = tweedie_module.TweedieProfileResult(
            p_hat=1.5,
            phi_hat=1.0,
            nll=0.0,
            n_evaluations=1,
            converged=True,
            method="brent",
            phi_method="mle",
            search_trace=pd.DataFrame({"p": [1.5], "nll": [0.0]}),
            n_positive=10,
            n_saddlepoint=0,
            _objective=_legacy_pickle_profile_objective,
            _ll_scale=1.0,
            _ci_p_range=(1.1, 1.9),
            _ci_seed_points=(1.5, 1.55, 1.85),
            _evaluation_record=_legacy_pickle_evaluation_record,
        )
        for name in (
            "density_method",
            "density_exact",
            "density_warning_severity",
            "near_power_boundary",
            "_emitted_ci_density_warning_signatures",
        ):
            delattr(result, name)

        restored = pickle.loads(pickle.dumps(result))

        assert restored.density_method == "exact"
        assert restored.density_exact is True
        assert restored.density_warning_severity == "none"
        assert restored.near_power_boundary is False
        assert restored._emitted_ci_density_warning_signatures == set()
        with pytest.warns(UserWarning, match="evaluated LR region"):
            interval = restored.ci()
        assert interval[0] < restored.p_hat < interval[1]
        assert restored._emitted_ci_density_warning_signatures == {"saddle:warning"}
