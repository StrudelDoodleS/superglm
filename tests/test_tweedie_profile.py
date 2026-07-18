"""Tests for Tweedie profile likelihood — p estimation."""

import gc
import inspect
import pickle
import warnings
import weakref
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, tzinfo
from enum import Enum, IntEnum
from fractions import Fraction
from types import SimpleNamespace
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize_scalar as scipy_minimize_scalar

import superglm.profiling.tweedie as tweedie_module
from superglm import SuperGLM
from superglm._tweedie_density import TweedieDensityError, approximate_tweedie_logpdf
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
from superglm.profiling.tweedie import (
    TweedieProfileResult,
    _profile_phi,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)
from superglm.solvers.pirls import PIRLSResult

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

    def test_saddlepoint_approximation_is_explicit(self):
        """The diagnostic approximation is available only by an explicit call."""
        y = np.array([100.0, 200.0, 500.0])
        mu = np.array([50.0, 100.0, 250.0])
        phi, p = 5.0, 1.5
        evaluation = approximate_tweedie_logpdf(y, mu, phi, p)

        assert np.all(np.isfinite(evaluation.logpdf))
        assert evaluation.diagnostics.method == "saddlepoint"
        assert not evaluation.diagnostics.exact
        assert not evaluation.diagnostics.certified

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
    """Certified log-density derivatives with respect to ``log(phi)``."""

    @staticmethod
    def _finite_difference_score(prepared, phi):
        h = 1e-5
        u = np.log(phi)

        def mean_logpdf(log_phi):
            evaluation = tweedie_module._evaluate_tweedie_density(
                prepared,
                float(np.exp(log_phi)),
            )
            return float(np.mean(evaluation.logpdf))

        return (mean_logpdf(u + h) - mean_logpdf(u - h)) / (2.0 * h)

    @pytest.mark.parametrize(
        ("y", "mu", "phi", "p", "weights"),
        [
            pytest.param(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.7, 2.0, 8.0]),
                1.7,
                1.5,
                None,
                id="all-zeros",
            ),
            pytest.param(
                np.array([0.3, 1.2, 4.5]),
                np.array([0.5, 1.5, 3.7]),
                1.3,
                1.5,
                None,
                id="positive",
            ),
            pytest.param(
                np.array([0.0, 0.2, 0.0, 3.0]),
                np.array([0.4, 0.3, 2.0, 2.5]),
                2.1,
                1.6,
                None,
                id="mixed",
            ),
            pytest.param(
                np.array([0.2, 1.0, 5.0, 9.0]),
                np.array([0.3, 1.4, 4.0, 7.0]),
                2.4,
                1.55,
                np.array([0.25, 0.8, 1.7, 4.0]),
                id="prior-weights",
            ),
            pytest.param(
                np.array([0.04, 0.05, 0.06]),
                np.array([0.035, 0.055, 0.08]),
                1.0,
                1.05,
                None,
                id="low-power",
            ),
            pytest.param(
                np.array([0.2, 1.0, 5.0]),
                np.array([0.3, 1.4, 4.0]),
                1.2,
                1.95,
                None,
                id="high-power",
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
    ):
        prepared = tweedie_module._prepare_tweedie_density(
            y,
            mu,
            p,
            weights=weights,
        )
        evaluation = tweedie_module._evaluate_tweedie_density(prepared, phi)

        assert evaluation.score_valid
        assert evaluation.diagnostics.n_saddlepoint == 0
        analytic = float(np.mean(evaluation.log_phi_score))
        finite_difference = self._finite_difference_score(prepared, phi)
        np.testing.assert_allclose(analytic, finite_difference, rtol=3e-7, atol=3e-8)

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

        weighted_eval = tweedie_module._evaluate_tweedie_density(weighted, phi)
        rescaled_eval = tweedie_module._evaluate_tweedie_density(
            unweighted,
            phi / weight,
        )

        np.testing.assert_allclose(weighted_eval.logpdf, rescaled_eval.logpdf, rtol=1e-13)
        np.testing.assert_allclose(
            weighted_eval.log_phi_score,
            rescaled_eval.log_phi_score,
            rtol=1e-13,
        )


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
        n = 500
        mu = np.full(n, 10.0)
        phi_true, p = 3.0, 1.6
        y = generate_tweedie_cpg(n, mu=mu, phi=phi_true, p=p, rng=rng)

        phi_hat, _ = _profile_phi(y, mu, p, phi_method="mle")
        np.testing.assert_allclose(phi_hat, phi_true, rtol=0.12)


class TestDetailedPhiProfile:
    @pytest.mark.parametrize(
        ("y", "mu", "p", "weights"),
        [
            pytest.param(
                np.array([0.3, 1.2, 4.5]),
                np.array([0.5, 1.5, 3.7]),
                1.5,
                None,
                id="regular",
            ),
            pytest.param(
                np.array([0.0, 0.0, 0.3, 2.5]),
                np.array([0.4, 1.2, 0.5, 2.0]),
                1.6,
                None,
                id="zero-heavy",
            ),
            pytest.param(
                np.array([0.0, 0.2, 1.0, 5.0]),
                np.array([0.3, 0.4, 1.4, 4.0]),
                1.55,
                np.array([0.25, 0.8, 1.7, 4.0]),
                id="prior-weights",
            ),
        ],
    )
    def test_mle_matches_tight_value_only_reference(self, y, mu, p, weights):
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

        assert result.converged
        assert not result.used_fallback
        assert not result.branch_switch_detected
        assert result.objective_finite
        assert result.optimizer == "brentq"
        assert not result.lower_boundary
        assert not result.upper_boundary
        assert result.score is not None
        assert abs(result.score) <= 1e-6
        assert result.diagnostics.n_saddlepoint == 0
        np.testing.assert_allclose(result.nll, reference.fun, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.log(result.phi), reference.x, rtol=0.0, atol=5e-6)

    def test_certified_interior_root_does_not_probe_hard_phi_bounds(
        self,
        monkeypatch,
    ):
        center = float(np.log(2.0))
        evaluated_phi = []

        def interior_quadratic(prepared, phi, *, compute_score=False):
            del compute_score
            evaluated_phi.append(float(phi))
            u = float(np.log(phi))
            nll = (u - center) ** 2
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(len(prepared.y), -nll),
                log_phi_score=np.full(len(prepared.y), -2.0 * (u - center)),
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=True,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", interior_quadratic)
        result = tweedie_module._profile_phi_detailed(
            np.array([0.5, 1.5]),
            np.array([0.8, 1.2]),
            1.5,
            phi_method="mle",
            phi_start=2.0,
        )

        assert result.converged
        assert result.phi == pytest.approx(2.0)
        assert tweedie_module._PHI_LOWER_BOUND not in evaluated_phi
        assert tweedie_module._PHI_UPPER_BOUND not in evaluated_phi

    def test_all_zero_profile_probes_only_the_indicated_upper_boundary(
        self,
        monkeypatch,
    ):
        evaluated_phi = []
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def spy(prepared, phi, *, compute_score=False):
            evaluated_phi.append(float(phi))
            return real_evaluate(prepared, phi, compute_score=compute_score)

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", spy)
        result = tweedie_module._profile_phi_detailed(
            np.zeros(3),
            np.array([0.7, 2.0, 8.0]),
            1.5,
            phi_method="mle",
        )

        assert result.upper_boundary
        assert result.phi == tweedie_module._PHI_UPPER_BOUND
        assert result.score is not None and result.score < 0.0
        assert tweedie_module._PHI_UPPER_BOUND in evaluated_phi
        assert tweedie_module._PHI_LOWER_BOUND not in evaluated_phi

    def test_required_boundary_certification_failure_propagates_identically(
        self,
        monkeypatch,
    ):
        error = TweedieDensityError(
            observation_index=0,
            power=1.5,
            dispersion=tweedie_module._PHI_UPPER_BOUND,
            term_count=11,
            requested_rtol=1e-12,
            reason="required boundary not certified",
        )

        def monotone_to_upper(prepared, phi, *, compute_score=False):
            del compute_score
            if phi == tweedie_module._PHI_UPPER_BOUND:
                raise error
            u = float(np.log(phi))
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(len(prepared.y), u),
                log_phi_score=np.ones(len(prepared.y)),
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=True,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", monotone_to_upper)

        with pytest.raises(TweedieDensityError) as caught:
            tweedie_module._profile_phi_detailed(
                np.array([0.5]),
                np.array([0.8]),
                1.5,
                phi_method="mle",
                phi_start=1.0,
            )

        assert caught.value is error

    def test_bounded_rescue_preserves_required_boundary_failure_identity(
        self,
        monkeypatch,
    ):
        error = TweedieDensityError(
            observation_index=0,
            power=1.5,
            dispersion=tweedie_module._PHI_UPPER_BOUND,
            term_count=13,
            requested_rtol=1e-12,
            reason="bounded boundary not certified",
        )

        def invalid_score_to_upper(prepared, phi, *, compute_score=False):
            del compute_score
            if phi == tweedie_module._PHI_UPPER_BOUND:
                raise error
            u = float(np.log(phi))
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(len(prepared.y), u),
                log_phi_score=np.full(len(prepared.y), np.nan),
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=False,
            )

        monkeypatch.setattr(
            tweedie_module,
            "_evaluate_tweedie_density",
            invalid_score_to_upper,
        )

        with pytest.raises(TweedieDensityError) as caught:
            tweedie_module._profile_phi_detailed(
                np.array([0.5]),
                np.array([0.8]),
                1.5,
                phi_method="mle",
            )

        assert caught.value is error

    @pytest.mark.parametrize(
        ("nll_slope", "expected_phi", "boundary_name"),
        [
            pytest.param(1.0, 1e-12, "lower_boundary", id="lower"),
            pytest.param(-1.0, 1e12, "upper_boundary", id="upper"),
        ],
    )
    def test_hard_boundary_kkt_orientation(
        self,
        monkeypatch,
        nll_slope,
        expected_phi,
        boundary_name,
    ):
        def monotone_exact(prepared, phi, *, compute_score=False):
            del compute_score
            u = float(np.log(phi))
            logpdf = np.full(len(prepared.y), -nll_slope * u)
            log_score = np.full(len(prepared.y), -nll_slope)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=logpdf,
                log_phi_score=log_score,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=True,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", monotone_exact)
        result = tweedie_module._profile_phi_detailed(
            np.array([0.5, 1.5]),
            np.array([0.8, 1.2]),
            1.5,
            phi_method="mle",
            phi_start=1.0,
        )

        assert result.phi == expected_phi
        assert getattr(result, boundary_name)
        assert result.lower_boundary is (expected_phi == 1e-12)
        assert result.upper_boundary is (expected_phi == 1e12)
        assert result.objective_finite
        assert not result.branch_switch_detected

    def test_derivative_failure_uses_value_only_certified_rescue(
        self,
        monkeypatch,
    ):
        y = np.array([0.4, 1.5, 5.0])
        mu = np.array([0.7, 1.2, 4.5])
        p = 1.6
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def invalidate_score(prepared, phi, *, compute_score=False):
            del compute_score
            evaluation = real_evaluate(prepared, phi)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=evaluation.logpdf,
                log_phi_score=np.full_like(evaluation.log_phi_score, np.nan),
                diagnostics=evaluation.diagnostics,
                score_valid=False,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", invalidate_score)
        reference, _ = _tight_value_only_phi_reference(y, mu, p)
        result = tweedie_module._profile_phi_detailed(y, mu, p, phi_method="mle")

        assert not result.converged
        assert result.objective_finite
        assert result.optimizer == "bounded"
        assert result.used_fallback
        assert result.fallback_reason is not None
        assert "derivative" in result.fallback_reason.lower()
        assert result.n_fallback_evaluations > 0
        assert not result.branch_switch_detected
        np.testing.assert_allclose(result.nll, reference.fun, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.log(result.phi), reference.x, rtol=0.0, atol=5e-6)

    def test_uncertifiable_candidate_enters_fail_closed_rescue(
        self,
        monkeypatch,
    ):
        y = np.array([0.4, 1.5, 5.0])
        mu = np.array([0.7, 1.2, 4.5])
        p = 1.6
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def fail_at_unit(prepared, phi, *, compute_score=False):
            del compute_score
            if abs(float(np.log(phi))) <= 1e-14:
                raise TweedieDensityError(
                    observation_index=0,
                    power=p,
                    dispersion=float(phi),
                    term_count=9,
                    requested_rtol=1e-12,
                    reason="tail not certified",
                )
            return real_evaluate(prepared, phi)

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", fail_at_unit)
        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            phi_method="mle",
            phi_start=1.0,
        )

        failed_point = tweedie_module._PhiEvaluationCache(
            tweedie_module._prepare_tweedie_density(y, mu, p)
        )
        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", fail_at_unit)
        point = failed_point.evaluate(0.0, compute_score=True)

        assert not point.objective_finite
        assert not point.score_valid
        assert result.objective_finite
        assert result.used_fallback
        assert not result.converged
        assert not result.branch_switch_detected

    def test_maximum_score_orientation_uses_value_only_rescue(self, monkeypatch):
        def maximum_at_zero(prepared, phi, *, compute_score=False):
            del compute_score
            u = float(np.log(phi))
            logpdf = np.full(len(prepared.y), u**2)
            log_score = np.full(len(prepared.y), 2.0 * u)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=logpdf,
                log_phi_score=log_score,
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=True,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", maximum_at_zero)
        result = tweedie_module._profile_phi_detailed(
            np.array([0.5]),
            np.array([0.8]),
            1.5,
            phi_method="mle",
            phi_start=1.0,
        )

        assert result.used_fallback
        assert result.fallback_reason is not None
        assert "orientation" in result.fallback_reason
        assert result.objective_finite
        assert not result.converged

    def test_multiple_smooth_score_minima_are_compared_without_branch_logic(
        self,
        monkeypatch,
    ):
        def double_well(prepared, phi, *, compute_score=False):
            del compute_score
            u = float(np.log(phi))
            nll = (u * u - 1.0) ** 2
            nll_score = 4.0 * u * (u * u - 1.0)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=np.full(len(prepared.y), -nll),
                log_phi_score=np.full(len(prepared.y), -nll_score),
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=True,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", double_well)
        prepared = tweedie_module._prepare_tweedie_density(
            np.array([0.5]),
            np.array([0.8]),
            1.5,
        )
        cache = tweedie_module._PhiEvaluationCache(prepared)
        score_search = tweedie_module._search_phi_score_candidates(
            cache,
            [(-0.5, "left seed"), (0.5, "right seed")],
        )
        bounded = tweedie_module._run_phi_bounded_fallback(cache, required=True)
        result = tweedie_module._finalize_phi_mle_result(
            cache,
            score_search,
            bounded,
            tweedie_module._TweedieLogpdfDiagnostics(n_positive=1, n_saddlepoint=0),
        )

        roots = sorted(candidate.point.u for candidate in score_search.root_candidates)
        np.testing.assert_allclose(roots, [-1.0, 1.0], atol=1e-9)
        assert "multiple distinct score minima" in " ".join(score_search.fallback_reasons)
        assert result.objective_finite
        assert result.nll == pytest.approx(0.0, abs=1e-12)
        assert result.used_fallback
        assert not result.converged
        assert not result.branch_switch_detected

    def test_unsuccessful_rescue_cannot_be_overwritten_by_a_seed(
        self,
        monkeypatch,
    ):
        y = np.array([0.4, 1.5, 5.0])
        mu = np.array([0.7, 1.2, 4.5])
        p = 1.6
        real_evaluate = tweedie_module._evaluate_tweedie_density

        def invalidate_score(prepared, phi, *, compute_score=False):
            del compute_score
            evaluation = real_evaluate(prepared, phi)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=evaluation.logpdf,
                log_phi_score=np.full_like(evaluation.log_phi_score, np.nan),
                diagnostics=evaluation.diagnostics,
                score_valid=False,
            )

        def unsuccessful_bounded(objective, *, bounds, method, options):
            x = 0.0
            return OptimizeResult(
                x=x,
                fun=objective(x),
                success=False,
                message="forced bounded failure",
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", invalidate_score)
        monkeypatch.setattr(tweedie_module, "minimize_scalar", unsuccessful_bounded)
        result = tweedie_module._profile_phi_detailed(
            y,
            mu,
            p,
            phi_method="mle",
            phi_start=1.0,
        )

        assert result.objective_finite
        assert result.used_fallback
        assert result.optimizer == "bounded"
        assert not result.converged
        assert "forced bounded failure" in result.message

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
        assert result.n_evaluations == result.n_score_evaluations == 1
        assert result.n_value_only_evaluations == result.n_fallback_evaluations == 0
        assert result.score is None
        assert not result.used_fallback
        assert result.diagnostics.n_saddlepoint == 0
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
        def invalidate_objective(prepared, phi, *, compute_score=False):
            del phi, compute_score
            values = np.full(len(prepared.y), np.nan, dtype=np.float64)
            return tweedie_module._TweedieDensityEvaluation(
                logpdf=values,
                log_phi_score=values.copy(),
                diagnostics=tweedie_module._TweedieLogpdfDiagnostics(
                    n_positive=prepared.positive_indices.size,
                    n_saddlepoint=0,
                ),
                score_valid=False,
            )

        monkeypatch.setattr(tweedie_module, "_evaluate_tweedie_density", invalidate_objective)
        result = tweedie_module._profile_phi_detailed(
            np.array([0.0, 0.4, 1.5]),
            np.array([0.2, 0.7, 1.2]),
            1.6,
            phi_method="mle",
        )

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

        result = estimate_tweedie_p(
            model,
            X,
            y,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)

    def test_prior_weight_mle_p_phi_recovery(self):
        """Exact-MLE profiling should recover p when prior weights act through phi / w."""
        rng = np.random.default_rng(321)
        p_true = 1.6
        phi_true = 3.0
        n = 150
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
            model,
            X,
            y,
            sample_weight=sample_weight,
            p_bounds=(1.1, 1.9),
            method="grid",
            grid=[1.45, 1.6, 1.75],
            phi_method="mle",
        )
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.15)
        np.testing.assert_allclose(result.phi_hat, phi_true, rtol=0.2)
        assert result.converged
        assert not result.phi_used_fallback
        assert result.density_exact
        assert result.n_saddlepoint == 0

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
            phi_method="pearson",
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
            estimate_tweedie_p(model, X, y, phi_method="pearson")

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
            estimate_tweedie_p(model, X, y, phi_method="pearson")

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


def _offset_spline_tweedie_data(n=72, seed=20260720):
    """Small offset-aware Tweedie sample for final-refit state tests."""
    rng = np.random.default_rng(seed)
    x1 = np.linspace(-1.0, 1.0, n)
    offset = 0.25 * np.sin(np.pi * x1)
    mu = np.exp(0.6 + 0.35 * x1 + offset)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.47, rng=rng)
    sample_weight = rng.uniform(0.75, 1.25, n)
    return pd.DataFrame({"x1": x1}), y, sample_weight, offset


def _deterministic_profile_result():
    return TweedieProfileResult(
        p_hat=1.47,
        phi_hat=7.25,
        nll=0.0,
        n_evaluations=1,
        converged=True,
        method="brent",
        phi_method="mle",
        search_trace=pd.DataFrame({"p": [1.47], "phi": [7.25], "nll": [0.0]}),
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


@pytest.mark.parametrize(
    "function",
    [SuperGLM.estimate_p, profile_ops_module.estimate_p, estimate_tweedie_p],
)
def test_public_tweedie_profile_entry_points_default_to_mle_and_brent(function):
    signature = inspect.signature(function)

    assert signature.parameters["phi_method"].default == "mle"
    assert signature.parameters["method"].default == "brent"
    assert "_prepared_inputs" not in signature.parameters


class TestEstimatePFitMode:
    @pytest.mark.parametrize(
        ("fit_mode", "final_fit_name"),
        [("fit", "fit"), ("reml", "fit_reml")],
    )
    def test_final_refit_uses_profile_input_snapshot_after_callback_mutation(
        self, monkeypatch, fit_mode, final_fit_name
    ):
        X = pd.DataFrame({"x1": np.array([-1.0, 0.0, 1.0])})
        y = np.array([0.0, 1.0, 2.0])
        sample_weight = np.array([0.75, 1.0, 1.25])
        offset = np.array([-0.2, 0.0, 0.2])
        caller_objects = (X, y, sample_weight, offset)
        baseline = (
            X.copy(deep=True),
            y.copy(),
            sample_weight.copy(),
            offset.copy(),
        )
        result = _deterministic_profile_result()
        observed = {}

        def mutate_caller_inputs(_row):
            X.iloc[:, :] = 101.0
            y[:] = 102.0
            sample_weight[:] = 103.0
            offset[:] = 104.0

        def fake_profile(candidate, prepared):
            observed["profile_objects"] = (
                prepared.X,
                prepared.y,
                prepared.sample_weight,
                prepared.offset,
            )
            observed["profile_values"] = (
                prepared.X.copy(deep=True),
                np.array(prepared.y, copy=True),
                np.array(prepared.sample_weight, copy=True),
                np.array(prepared.offset, copy=True),
            )
            prepared.trace_callback({})
            return result

        monkeypatch.setattr(tweedie_module, "_estimate_tweedie_p_prepared", fake_profile)

        def final_fit(fit_X, fit_y, *, sample_weight, offset):
            observed["final_objects"] = (fit_X, fit_y, sample_weight, offset)
            observed["final_values"] = (
                fit_X.copy(deep=True),
                np.array(fit_y, copy=True),
                np.array(sample_weight, copy=True),
                np.array(offset, copy=True),
            )

        def unexpected_fit(*args, **kwargs):
            raise AssertionError("estimate_p selected the wrong final refit method")

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
            fit=unexpected_fit,
            fit_reml=unexpected_fit,
        )
        setattr(model, final_fit_name, final_fit)

        def synchronize(candidate, sync_y, profile_result):
            assert candidate is model
            assert profile_result is result
            observed["sync_y"] = sync_y
            observed["sync_y_value"] = np.array(sync_y, copy=True)

        monkeypatch.setattr(
            profile_ops_module,
            "_synchronize_tweedie_profile_refit",
            synchronize,
        )

        returned = profile_ops_module.estimate_p(
            model,
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode=fit_mode,
            trace_callback=mutate_caller_inputs,
        )

        assert returned is result
        for caller, profiled, refitted in zip(
            caller_objects,
            observed["profile_objects"],
            observed["final_objects"],
            strict=True,
        ):
            assert profiled is refitted
            assert profiled is not caller
        assert observed["sync_y"] is observed["profile_objects"][1]

        pd.testing.assert_frame_equal(
            observed["profile_values"][0], baseline[0], check_column_type=False
        )
        pd.testing.assert_frame_equal(
            observed["final_values"][0], baseline[0], check_column_type=False
        )
        for actual, expected in zip(
            observed["profile_values"][1:],
            baseline[1:],
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
            observed["final_values"][1:],
            baseline[1:],
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(observed["sync_y_value"], baseline[1])
        assert np.all(X.to_numpy() == 101.0)
        assert np.all(y == 102.0)
        assert np.all(sample_weight == 103.0)
        assert np.all(offset == 104.0)

    def test_invalid_profile_controls_fail_before_input_snapshot(self, monkeypatch):
        model = SimpleNamespace(family=TweedieDistribution(p=1.5))
        X = pd.DataFrame({"x1": [0.0]})

        def unexpected_snapshot(*args, **kwargs):
            raise AssertionError("invalid controls must fail before snapshotting row inputs")

        monkeypatch.setattr(
            profile_ops_module,
            "_snapshot_tweedie_profile_refit_inputs",
            unexpected_snapshot,
        )

        with pytest.raises(ValueError, match="method"):
            profile_ops_module.estimate_p(model, X, np.array([1.0]), method=[])

        with pytest.raises(TypeError, match="estimate_tweedie_p.*unexpected keyword.*bogus"):
            profile_ops_module.estimate_p(model, X, np.array([1.0]), bogus=True)

    @pytest.mark.parametrize("row_kind", ["memoryview", "no-deepcopy-array"])
    def test_valid_numeric_row_inputs_do_not_require_deepcopy(self, monkeypatch, row_kind):
        class NoDeepcopyArray(np.ndarray):
            def __deepcopy__(self, memo):
                raise AssertionError("numeric row normalization must not call deepcopy")

        sources = (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.75, 1.0, 1.25]),
            np.array([-0.2, 0.0, 0.2]),
        )
        if row_kind == "memoryview":
            y, sample_weight, offset = (memoryview(value) for value in sources)
        else:
            y, sample_weight, offset = (value.view(NoDeepcopyArray) for value in sources)

        result = _deterministic_profile_result()
        observed = {}

        def fake_profile(candidate, prepared):
            observed["profile"] = (
                prepared.y,
                prepared.sample_weight,
                prepared.offset,
            )
            return result

        monkeypatch.setattr(tweedie_module, "_estimate_tweedie_p_prepared", fake_profile)

        def final_fit(fit_X, fit_y, *, sample_weight, offset):
            observed["final"] = (fit_y, sample_weight, offset)

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
            fit=final_fit,
        )
        monkeypatch.setattr(
            profile_ops_module, "_synchronize_tweedie_profile_refit", lambda *a: None
        )

        profile_ops_module.estimate_p(
            model,
            pd.DataFrame({"x1": [-1.0, 0.0, 1.0]}),
            y,
            sample_weight=sample_weight,
            offset=offset,
        )

        for expected, profiled, refitted in zip(
            sources,
            observed["profile"],
            observed["final"],
            strict=True,
        ):
            assert type(profiled) is np.ndarray
            assert profiled.flags.owndata
            assert profiled is refitted
            np.testing.assert_array_equal(profiled, expected)

    def test_object_dataframe_scalar_values_use_owned_frame_snapshot(self, monkeypatch):
        caller_values = ["low", "low", "high"]
        X = pd.DataFrame({"x1": pd.Series(caller_values, dtype=object)})
        result = _deterministic_profile_result()
        observed = {}

        def mutate_caller_frame(_row):
            X.iloc[:, 0] = ["changed", "changed", "changed"]

        def fake_profile(candidate, prepared):
            observed["profile_X"] = prepared.X
            observed["profile_values"] = prepared.X["x1"].tolist()
            prepared.trace_callback({})
            return result

        monkeypatch.setattr(tweedie_module, "_estimate_tweedie_p_prepared", fake_profile)

        def final_fit(fit_X, fit_y, **kwargs):
            observed["final_X"] = fit_X
            observed["final_values"] = fit_X["x1"].tolist()

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
            fit=final_fit,
        )
        monkeypatch.setattr(
            profile_ops_module, "_synchronize_tweedie_profile_refit", lambda *a: None
        )

        profile_ops_module.estimate_p(
            model,
            X,
            np.array([0.0, 1.0, 2.0]),
            trace_callback=mutate_caller_frame,
        )

        assert observed["profile_X"] is observed["final_X"]
        assert observed["profile_X"] is not X
        assert observed["profile_values"] == caller_values
        assert observed["final_values"] == caller_values
        assert X["x1"].tolist() == ["changed", "changed", "changed"]

    def test_categorical_category_buffer_is_detached_from_caller(self, monkeypatch):
        caller_categories = ["low", "middle", "high"]
        X = pd.DataFrame(
            {
                "x1": pd.Categorical(
                    caller_categories,
                    categories=caller_categories,
                    ordered=True,
                )
            }
        )
        result = _deterministic_profile_result()
        observed = {}

        def mutate_caller_categories(_row):
            category_values = X["x1"].cat.categories.to_numpy(copy=False)
            category_values.setflags(write=True)
            category_values[:] = ["changed-low", "changed-middle", "changed-high"]

        def values(frame):
            return frame["x1"].tolist()

        def fake_profile(candidate, prepared):
            observed["profile_X"] = prepared.X
            observed["profile_values"] = values(prepared.X)
            prepared.trace_callback({})
            return result

        monkeypatch.setattr(tweedie_module, "_estimate_tweedie_p_prepared", fake_profile)

        def final_fit(fit_X, fit_y, **kwargs):
            observed["final_X"] = fit_X
            observed["final_values"] = values(fit_X)

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
            fit=final_fit,
        )
        monkeypatch.setattr(
            profile_ops_module, "_synchronize_tweedie_profile_refit", lambda *a: None
        )

        profile_ops_module.estimate_p(
            model,
            X,
            np.array([0.0, 1.0, 2.0]),
            trace_callback=mutate_caller_categories,
        )

        assert observed["profile_X"] is observed["final_X"]
        assert observed["profile_values"] == caller_categories
        assert observed["final_values"] == caller_categories
        assert observed["profile_X"]["x1"].cat.categories.tolist() == caller_categories
        assert X["x1"].cat.categories.tolist() == [
            "changed-low",
            "changed-middle",
            "changed-high",
        ]

    @pytest.mark.parametrize(
        "categories",
        [
            pd.Index(
                pd.array(["a", "b"], dtype=pd.StringDtype(storage="python")),
                name="levels",
            ),
            pd.Index(pd.array([True, False], dtype="boolean"), name="levels"),
            pd.Index(
                [("a", 1), ("b", 2)],
                dtype=object,
                name="pair",
                tupleize_cols=False,
            ),
        ],
        ids=["string-extension", "nullable-boolean", "tuple-object"],
    )
    def test_profile_snapshot_preserves_categorical_category_dtype(self, categories):
        X = pd.DataFrame(
            {
                "x1": pd.Categorical.from_codes(
                    [0, 1, 0],
                    categories=categories,
                    ordered=True,
                )
            }
        )

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([0.0, 1.0, 2.0]),
        )

        original_categories = X["x1"].cat.categories
        snapshot_categories = prepared.X["x1"].cat.categories
        assert snapshot_categories.dtype == original_categories.dtype
        assert snapshot_categories.name == original_categories.name
        assert snapshot_categories.equals(original_categories)
        assert prepared.X["x1"].dtype == X["x1"].dtype

    def test_profile_snapshot_normalizes_nullable_numeric_category_dtype(self):
        X = pd.DataFrame(
            {
                "x1": pd.Categorical.from_codes(
                    [0, 1, 0],
                    categories=pd.Index(pd.array([1, 2], dtype="Int64")),
                )
            }
        )

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([0.0, 1.0, 2.0]),
        )

        assert prepared.X["x1"].cat.categories.tolist() == [1, 2]
        assert prepared.X["x1"].cat.categories.dtype == np.dtype("int64")
        assert prepared.X["x1"].tolist() == [1, 2, 1]

    @pytest.mark.parametrize(
        "categories",
        [
            pd.date_range("2026-01-01", periods=2, freq="D", name="levels"),
            pd.timedelta_range("1D", periods=2, freq="2D", name="levels"),
            pd.IntervalIndex.from_breaks([0, 1, 2], name="levels"),
        ],
        ids=["datetime", "timedelta", "interval"],
    )
    def test_profile_snapshot_preserves_typed_categorical_categories(self, categories):
        X = pd.DataFrame(
            {
                "x1": pd.Categorical.from_codes(
                    [0, 1, 0],
                    categories=categories,
                    ordered=True,
                )
            }
        )

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([0.0, 1.0, 2.0]),
        )

        pd.testing.assert_index_equal(
            prepared.X["x1"].cat.categories,
            categories,
            exact=True,
        )
        assert prepared.X["x1"].cat.ordered

    @pytest.mark.parametrize("axis_name", ["index", "columns"])
    def test_profile_snapshot_detaches_plain_axis_buffers(self, axis_name):
        X = pd.DataFrame(
            {"x1": pd.Series([1.0, 2.0, 3.0], dtype=np.float64)},
            index=pd.Index(["row-a", "row-b", "row-c"], dtype=object),
        )
        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0, 3.0]),
        )

        source_axis = getattr(X, axis_name)
        source_values = source_axis.to_numpy(copy=False)
        source_values.setflags(write=True)
        source_values[0] = "changed"

        expected = ["row-a", "row-b", "row-c"] if axis_name == "index" else ["x1"]
        assert getattr(prepared.X, axis_name).tolist() == expected

    def test_profile_snapshot_detaches_categorical_index_categories(self):
        X = pd.DataFrame(
            {"x1": [1.0, 2.0, 3.0]},
            index=pd.CategoricalIndex(
                ["row-a", "row-b", "row-a"],
                categories=["row-a", "row-b"],
                ordered=True,
                name="rows",
            ),
        )
        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0, 3.0]),
        )

        source_values = X.index.categories.to_numpy(copy=False)
        source_values.setflags(write=True)
        source_values[:] = ["changed-a", "changed-b"]

        assert prepared.X.index.tolist() == ["row-a", "row-b", "row-a"]
        assert prepared.X.index.categories.tolist() == ["row-a", "row-b"]
        assert prepared.X.index.name == "rows"

    def test_profile_snapshot_rejects_dataframe_subclasses_and_attrs(self):
        class FrameSubclass(pd.DataFrame):
            pass

        class MutableMetadata:
            pass

        cases = [
            FrameSubclass({"x1": [1.0]}),
            pd.DataFrame({"x1": [1.0]}),
        ]
        cases[1].attrs["metadata"] = MutableMetadata()

        for X in cases:
            with pytest.raises(TypeError, match="snapshot.*X|plain pandas DataFrame"):
                tweedie_module._prepare_tweedie_profile_inputs(
                    SimpleNamespace(family=TweedieDistribution(p=1.5)),
                    X,
                    np.array([1.0]),
                )

    def test_profile_snapshot_rejects_custom_column_labels_before_hashing(self):
        class CustomLabel:
            def __hash__(self):
                raise AssertionError("unsupported labels must not be hashed")

        X = pd.DataFrame([[1.0]])
        X.columns = pd.Index([CustomLabel()], dtype=object)

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    def test_profile_snapshot_contextualizes_excessively_nested_object_values(self):
        value = "leaf"
        for _ in range(2_000):
            value = (value,)
        X = pd.DataFrame({"x1": pd.Series([value], dtype=object)})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    def test_profile_snapshot_rejects_mutable_sparse_object_values(self):
        class MutableValue:
            pass

        sparse = pd.arrays.SparseArray(
            [MutableValue()],
            dtype=pd.SparseDtype(object, fill_value=None),
        )
        X = pd.DataFrame({"x1": sparse})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    def test_profile_snapshot_rejects_metadata_bearing_column_dtype(self):
        dtype = np.dtype("f8", metadata={"state": [1.0]})
        X = pd.DataFrame({"x1": np.array([1.0], dtype=dtype)})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    @pytest.mark.parametrize("storage_kind", ["native", "integer-ea", "integer-buffer"])
    def test_profile_snapshot_rejects_untrusted_backing_arrays(self, storage_kind):
        class RetainedArray(np.ndarray):
            def copy(self, *args, **kwargs):
                return self

        values = np.array([1, 2], dtype=np.int64).view(RetainedArray)
        if storage_kind == "native":
            column = values
        else:
            mask = np.zeros(2, dtype=bool)
            integer = pd.arrays.IntegerArray(values, mask)
            if storage_kind == "integer-ea":

                class RetainedIntegerArray(pd.arrays.IntegerArray):
                    def copy(self):
                        return self

                integer = RetainedIntegerArray(values.view(np.ndarray), mask)
            column = integer
        X = pd.DataFrame({"x1": column}, copy=False)

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0, 2.0]),
            )

    def test_profile_snapshot_rejects_untrusted_axis_storage_and_subclasses(self):
        class RetainedArray(np.ndarray):
            def copy(self, *args, **kwargs):
                return self

        class IndexSubclass(pd.Index):
            pass

        retained = np.array([1.0, 2.0]).view(RetainedArray)
        cases = [
            pd.Index(retained, copy=False),
            IndexSubclass._simple_new(
                np.array(["row-a", "row-b"], dtype=object),
                name=None,
            ),
        ]

        for index in cases:
            X = pd.DataFrame({"x1": [1.0, 2.0]}, index=index)
            with pytest.raises(TypeError, match="snapshot.*X"):
                tweedie_module._prepare_tweedie_profile_inputs(
                    SimpleNamespace(family=TweedieDistribution(p=1.5)),
                    X,
                    np.array([1.0, 2.0]),
                )

    @pytest.mark.parametrize(
        "column",
        [
            pd.arrays.SparseArray([0.0, 1.0, 0.0], fill_value=0.0),
            pd.array(["2026-01", "2026-02", "2026-03"], dtype="period[M]"),
        ],
        ids=["sparse", "period"],
    )
    def test_profile_snapshot_rejects_storage_with_shared_internal_state(self, column):
        X = pd.DataFrame({"x1": column})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0, 2.0, 3.0]),
            )

    def test_profile_snapshot_rejects_string_extension_scalar_subclasses(self):
        class MutableStr(str):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.state = [1.0]
                return instance

        X = pd.DataFrame(
            {
                "x1": pd.array(
                    [MutableStr("level")],
                    dtype=pd.StringDtype(storage="python"),
                )
            }
        )

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            (pd.StringDtype(storage="python"), ["a", None]),
            (pd.BooleanDtype(), [1, None]),
        ],
        ids=["string", "boolean"],
    )
    def test_profile_snapshot_reconstructs_extension_dtype_identity(self, dtype, values):
        X = pd.DataFrame({"x1": pd.array(values, dtype=dtype)})

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0]),
        )

        assert prepared.X["x1"].dtype == X["x1"].dtype
        assert prepared.X["x1"].dtype is not X["x1"].dtype

    @pytest.mark.parametrize(
        "column",
        [
            pd.Series(["a", "b"], dtype=object),
            pd.array(["a", None], dtype=pd.StringDtype(storage="python")),
            pd.array([True, None], dtype="boolean"),
            pd.date_range("2026-01-01", periods=2),
            pd.date_range("2026-01-01", periods=2, tz="UTC"),
            pd.to_timedelta([1, 2], unit="D"),
            pd.arrays.IntervalArray.from_breaks([0, 1, 2]),
            pd.Categorical.from_codes(
                [0, 1],
                categories=pd.Index(np.array(["a", "b"], dtype=object), dtype=object),
                ordered=True,
            ),
        ],
        ids=[
            "object-string",
            "string-extension",
            "nullable-boolean",
            "datetime",
            "datetime-tz",
            "timedelta",
            "interval",
            "categorical",
        ],
    )
    def test_profile_snapshot_preserves_supported_storage_values_and_dtype(self, column):
        X = pd.DataFrame({"x1": column})
        expected = X["x1"].copy(deep=True)

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0]),
        )

        pd.testing.assert_series_equal(prepared.X["x1"], expected)
        assert prepared.X["x1"]._values is not X["x1"]._values

    def test_profile_snapshot_canonicalizes_builtin_arrow_strings(self):
        pytest.importorskip("pyarrow")
        arrow_dtype = pd.StringDtype(storage="pyarrow")
        X = pd.DataFrame(
            {"x1": pd.array(["a", None], dtype=arrow_dtype)},
            index=pd.Index(pd.array(["row-a", "row-b"], dtype=arrow_dtype)),
        )
        X.columns = pd.Index(pd.array(["x1"], dtype=arrow_dtype))

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0]),
        )

        assert prepared.X["x1"].tolist() == ["a", pd.NA]
        assert prepared.X.index.tolist() == ["row-a", "row-b"]
        assert prepared.X.columns.tolist() == ["x1"]
        assert prepared.X["x1"].dtype.storage == "python"
        assert prepared.X.index.dtype.storage == "python"
        assert prepared.X.columns.dtype.storage == "python"
        assert type(prepared.X["x1"]._values) is pd.arrays.StringArray
        assert type(prepared.X.index._values) is pd.arrays.StringArray
        assert type(prepared.X.columns._values) is pd.arrays.StringArray

        plain = pd.DataFrame({"x1": [1.0, 2.0]})
        plain_prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            plain,
            np.array([1.0, 2.0]),
        )
        pd.testing.assert_frame_equal(
            plain_prepared.X,
            plain,
            check_dtype=False,
            check_column_type=False,
        )

    def test_profile_snapshot_canonicalizes_named_pytz_storage(self):
        pytz = pytest.importorskip("pytz")
        X = pd.DataFrame(
            {
                "x1": pd.date_range(
                    "2026-01-01",
                    periods=2,
                    tz=pytz.timezone("Europe/London"),
                )
            }
        )

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0]),
        )

        pd.testing.assert_series_equal(
            prepared.X["x1"].dt.tz_convert("UTC"),
            X["x1"].dt.tz_convert("UTC"),
            check_dtype=False,
        )
        assert type(prepared.X["x1"].dtype.tz) is ZoneInfo

    @pytest.mark.parametrize("dtype", ["Int64", "UInt64", "Float64"])
    def test_profile_snapshot_normalizes_nullable_numeric_dtypes(self, dtype):
        X = pd.DataFrame({"x1": pd.array([1, None], dtype=dtype)})

        prepared = tweedie_module._prepare_tweedie_profile_inputs(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([1.0, 2.0]),
        )

        assert prepared.X["x1"].dtype == np.dtype("float64")
        np.testing.assert_array_equal(
            prepared.X["x1"].to_numpy(),
            np.array([1.0, np.nan]),
        )
        assert prepared.X["x1"]._values is not X["x1"]._values

    @pytest.mark.parametrize(
        ("value", "dtype"),
        [
            (2**63 + 1, "UInt64"),
            (2**64 - 1, "UInt64"),
            (2**63 - 1, "Int64"),
        ],
    )
    def test_profile_snapshot_rejects_inexact_nullable_integer_normalization(self, value, dtype):
        X = pd.DataFrame({"x1": pd.array([value, None], dtype=dtype)})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0, 2.0]),
            )

    def test_profile_snapshot_rejects_object_cell_back_reference(self):
        class FrameLinkedFloat:
            def __init__(self, value):
                self.value = value
                self.owner = None

            def __float__(self):
                return float(self.owner.iloc[0, 0].value)

        caller_cell = FrameLinkedFloat(1.0)
        X = pd.DataFrame({"x1": pd.Series([caller_cell], dtype=object)})
        caller_cell.owner = X

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    @pytest.mark.parametrize("container_kind", ["series", "dataframe"])
    def test_profile_snapshot_rejects_nested_pandas_payloads(self, container_kind):
        class MutableFloat:
            def __init__(self, value):
                self.value = value

        class ContainerLinkedFloat:
            def __init__(self, payload):
                self.payload = payload

            def __float__(self):
                if isinstance(self.payload, pd.DataFrame):
                    return float(self.payload.iloc[0, 0].value)
                return float(self.payload.iloc[0].value)

        mutable = MutableFloat(1.0)
        if container_kind == "series":
            payload = pd.Series([mutable], dtype=object)
        else:
            payload = pd.DataFrame({"nested": pd.Series([mutable], dtype=object)})
        X = pd.DataFrame({"x1": pd.Series([ContainerLinkedFloat(payload)], dtype=object)})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    def test_profile_snapshot_checks_callable_instance_payloads(self):
        class MutableFloat:
            def __init__(self, value):
                self.value = value

        class CallableFloat:
            def __init__(self, payload):
                self.payload = payload

            def __call__(self):
                return float(self)

            def __float__(self):
                return float(self.payload.iloc[0].value)

        X = pd.DataFrame(
            {
                "x1": pd.Series(
                    [CallableFloat(pd.Series([MutableFloat(1.0)], dtype=object))],
                    dtype=object,
                )
            }
        )

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    @pytest.mark.parametrize("column_kind", ["object", "categorical"])
    def test_profile_snapshot_rejects_enum_members_with_mutable_values(self, column_kind):
        class MutableEnum(Enum):
            LEVEL = [1.0]

        if column_kind == "object":
            column = pd.Series([MutableEnum.LEVEL], dtype=object)
        else:
            column = pd.Categorical(
                [MutableEnum.LEVEL],
                categories=[MutableEnum.LEVEL],
            )
        X = pd.DataFrame({"x1": column})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    def test_profile_snapshot_rejects_enum_singletons_even_with_immutable_values(self):
        class ImmutableEnum(Enum):
            LOW = "low"
            HIGH = "high"

        X = pd.DataFrame(
            {
                "object": pd.Series([ImmutableEnum.LOW], dtype=object),
                "categorical": pd.Categorical(
                    [ImmutableEnum.HIGH],
                    categories=[ImmutableEnum.LOW, ImmutableEnum.HIGH],
                ),
            }
        )

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    def test_profile_snapshot_rejects_weak_references(self):
        class MutableFloat:
            def __init__(self, value):
                self.value = value

        class WeakLinkedFloat:
            def __init__(self, target):
                self.reference = weakref.ref(target)

            def __float__(self):
                return float(self.reference().value)

        target = MutableFloat(1.0)
        X = pd.DataFrame({"x1": pd.Series([WeakLinkedFloat(target)], dtype=object)})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    @pytest.mark.parametrize(
        "payload_kind",
        [
            "custom-object",
            "float-subclass",
            "enum",
            "int-enum",
            "list",
            "object-array",
            "timezone-datetime",
            "dtype-metadata",
            "slice",
            "fraction",
            "pandas-timestamp",
            "pandas-timedelta",
            "pandas-period",
            "uuid-state",
        ],
    )
    def test_profile_snapshot_rejects_nonimmutable_object_payloads(self, payload_kind):
        class MutableValue:
            def __init__(self, value=1.0):
                self.value = value

            def __float__(self):
                return float(self.value)

        class MutableFloat(float):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.state = [float(value)]
                return instance

            def __float__(self):
                return self.state[0]

        class MutableEnum(Enum):
            LEVEL = [1.0]

        class MutableIntEnum(IntEnum):
            LEVEL = 1

            def __float__(self):
                return self.state[0]

        MutableIntEnum.LEVEL.state = [1.0]

        class MutableTimezone(tzinfo):
            def __init__(self):
                self.state = [1.0]

            def utcoffset(self, value):
                return timedelta(hours=self.state[0])

            def dst(self, value):
                return timedelta(0)

        mutable = MutableValue()
        payloads = {
            "custom-object": mutable,
            "float-subclass": MutableFloat(1.0),
            "enum": MutableEnum.LEVEL,
            "int-enum": MutableIntEnum.LEVEL,
            "list": [mutable],
            "object-array": np.array([mutable], dtype=object),
            "timezone-datetime": datetime(2026, 1, 1, tzinfo=MutableTimezone()),
            "dtype-metadata": np.dtype("f8", metadata={"state": mutable}),
            "slice": slice(mutable),
            "fraction": Fraction(1, 2),
            "pandas-timestamp": pd.Timestamp("2026-01-01"),
            "pandas-timedelta": pd.Timedelta(days=1),
            "pandas-period": pd.Period("2026-01", freq="M"),
            "uuid-state": UUID(int=0, is_safe=mutable),
        }
        X = pd.DataFrame({"x1": pd.Series([payloads[payload_kind]], dtype=object)})

        with pytest.raises(TypeError, match="snapshot.*X"):
            tweedie_module._prepare_tweedie_profile_inputs(
                SimpleNamespace(family=TweedieDistribution(p=1.5)),
                X,
                np.array([1.0]),
            )

    @pytest.mark.parametrize(
        ("fit_mode", "builder_name"),
        [("fit", "_build_profile_context"), ("fit_reml", "_build_profile_context_reml")],
    )
    def test_direct_profile_owns_object_dataframe_before_trace_callback(
        self, monkeypatch, fit_mode, builder_name
    ):
        caller_values = ["low", "middle", "high"]
        X = pd.DataFrame({"x1": pd.Series(caller_values, dtype=object)})
        result = _deterministic_profile_result()
        observed = {}

        def mutate_caller_frame(_row):
            X.iloc[:, 0] = ["changed", "changed", "changed"]

        def fake_builder(
            candidate,
            profile_X,
            profile_y,
            sample_weight,
            offset,
            phi_method,
            verbose,
            trace_callback,
            trace_iterations,
            *,
            _inputs_owned=False,
        ):
            assert _inputs_owned
            observed["profile_X"] = profile_X
            observed["before"] = profile_X["x1"].tolist()
            trace_callback({})
            observed["after"] = profile_X["x1"].tolist()
            return SimpleNamespace()

        monkeypatch.setattr(tweedie_module, builder_name, fake_builder)
        monkeypatch.setattr(tweedie_module, "_search_brent", lambda *args: result)

        returned = estimate_tweedie_p(
            SimpleNamespace(family=TweedieDistribution(p=1.5)),
            X,
            np.array([0.0, 1.0, 2.0]),
            fit_mode=fit_mode,
            trace_callback=mutate_caller_frame,
        )

        assert returned is result
        assert observed["before"] == caller_values
        assert observed["after"] == caller_values
        assert observed["profile_X"] is not X
        assert X["x1"].tolist() == ["changed", "changed", "changed"]

    def test_completed_profile_releases_trace_callback_before_lazy_probes(self):
        class Holder:
            pass

        X, y, _ = _tweedie_data(n=40, seed=20260731)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        holder = Holder()
        holder_ref = weakref.ref(holder)
        callback_events = []

        def trace_callback(row, retained=holder):
            callback_events.append(row)

        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
            trace_callback=trace_callback,
        )
        events_after_search = len(callback_events)
        evaluations_after_search = result.n_total_evaluations

        del trace_callback
        del holder
        gc.collect()

        assert holder_ref() is None
        assert np.isfinite(result._objective(1.6, source="later_probe"))
        assert result.n_total_evaluations == evaluations_after_search + 1
        assert len(callback_events) == events_after_search

    @pytest.mark.parametrize(
        ("fit_mode", "builder_name", "final_fit_name"),
        [
            ("fit", "_build_profile_context", "fit"),
            ("reml", "_build_profile_context_reml", "fit_reml"),
        ],
    )
    def test_wrapper_prepares_one_input_graph_for_profile_and_final_refit(
        self, monkeypatch, fit_mode, builder_name, final_fit_name
    ):
        X = pd.DataFrame({"x1": pd.Series(["level"], dtype=object)})
        result = _deterministic_profile_result()
        observed = {}
        snapshot_calls = []
        real_snapshot = tweedie_module._snapshot_tweedie_profile_dataframe

        def counted_snapshot(frame):
            snapshot_calls.append(frame)
            return real_snapshot(frame)

        monkeypatch.setattr(
            tweedie_module,
            "_snapshot_tweedie_profile_dataframe",
            counted_snapshot,
        )

        def fake_builder(
            candidate,
            profile_X,
            profile_y,
            sample_weight,
            offset,
            phi_method,
            verbose,
            trace_callback,
            trace_iterations,
            *,
            _inputs_owned=False,
        ):
            assert _inputs_owned
            observed["profile_X"] = profile_X
            return SimpleNamespace()

        monkeypatch.setattr(tweedie_module, builder_name, fake_builder)
        monkeypatch.setattr(tweedie_module, "_search_brent", lambda *args: result)

        def final_fit(fit_X, fit_y, **kwargs):
            observed["final_X"] = fit_X

        def unexpected_fit(*args, **kwargs):
            raise AssertionError("estimate_p selected the wrong final refit method")

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
            fit=unexpected_fit,
            fit_reml=unexpected_fit,
        )
        setattr(model, final_fit_name, final_fit)
        monkeypatch.setattr(
            profile_ops_module, "_synchronize_tweedie_profile_refit", lambda *a: None
        )

        returned = profile_ops_module.estimate_p(
            model,
            X,
            np.array([1.0]),
            fit_mode=fit_mode,
        )

        assert returned is result
        assert len(snapshot_calls) == 1
        assert snapshot_calls[0] is X
        assert observed["profile_X"] is observed["final_X"]
        assert observed["profile_X"] is not X
        assert observed["profile_X"].iloc[0, 0] == "level"

    def test_wrapper_preserves_public_estimator_signature_for_instrumentation(self, monkeypatch):
        X = pd.DataFrame({"x1": [1.0]})
        y = np.array([1.0])
        result = _deterministic_profile_result()
        observed = {}

        def strict_public_estimator(
            candidate,
            profile_X,
            profile_y,
            sample_weight=None,
            offset=None,
            *,
            p_bounds=(1.05, 1.95),
            xatol=1e-3,
            maxiter=30,
            verbose=False,
            fit_mode="fit",
            phi_method="mle",
            method="brent",
            n_grid=20,
            grid=None,
            n_grid_coarse=10,
            optimizer="L-BFGS-B",
            trace_callback=None,
            trace_iterations=False,
        ):
            observed["profile_X"] = profile_X
            observed["profile_y"] = profile_y
            return result

        monkeypatch.setattr(tweedie_module, "estimate_tweedie_p", strict_public_estimator)

        def final_fit(fit_X, fit_y, **kwargs):
            observed["final_X"] = fit_X
            observed["final_y"] = fit_y

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
            fit=final_fit,
        )
        monkeypatch.setattr(
            profile_ops_module, "_synchronize_tweedie_profile_refit", lambda *args: None
        )

        returned = profile_ops_module.estimate_p(model, X, y)

        assert returned is result
        assert observed["profile_X"] is observed["final_X"]
        assert observed["profile_y"] is observed["final_y"]
        assert observed["profile_X"] is not X
        assert observed["profile_y"] is not y

    def test_uncopyable_object_dataframe_cell_fails_before_profile_or_mutation(self, monkeypatch):
        class UncopyableFloat:
            def __float__(self):
                return 1.0

            def __deepcopy__(self, memo):
                raise RuntimeError("object cell refuses copying")

        model = SimpleNamespace(
            family=TweedieDistribution(p=1.5),
            _retain_fit_state=True,
        )
        original_family = model.family
        calls = []
        monkeypatch.setattr(
            tweedie_module,
            "_estimate_tweedie_p_prepared",
            lambda *args, **kwargs: calls.append("profile"),
        )

        with pytest.raises(TypeError, match="snapshot.*X"):
            profile_ops_module.estimate_p(
                model,
                pd.DataFrame({"x1": pd.Series([UncopyableFloat()], dtype=object)}),
                np.array([1.0]),
                trace_callback=lambda row: calls.append("callback"),
            )

        assert calls == []
        assert model.family is original_family

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
        monkeypatch.setattr(
            tweedie_module,
            "_estimate_tweedie_p_prepared",
            lambda *args, **kwargs: result,
        )

        fit_name = "fit_reml" if fit_mode == "reml" else "fit"
        real_final_fit = getattr(model, fit_name)
        captured = {}

        def final_fit_with_primed_caches(*args, **kwargs):
            captured["retain_during_fit"] = model._retain_fit_state
            if fit_mode == "reml":
                kwargs["max_reml_iter"] = 3
            fitted = real_final_fit(*args, **kwargs)
            captured["public_result"] = model.result
            captured["solver_result"] = model._solver_pirls_result()
            captured["reml_result"] = model._reml_result
            captured["reml_lambdas"] = model._reml_lambdas
            captured["reml_penalties"] = model._reml_penalties
            captured["fit_meta"] = model._last_fit_meta
            captured["runtime_state"] = model._runtime_canonical_state
            captured["prediction_plan"] = model._prediction_plan
            captured["fast_prediction_state"] = model._fast_prediction_state
            # Before the fix, retain=False has already released these rows.  Skip
            # cache priming so the regression fails at the production access.
            if model._dm is None:
                return fitted

            solver = captured["solver_result"]
            captured["public_dynamic_metadata"] = object()
            captured["solver_dynamic_metadata"] = np.array([1.0, 2.0])
            model.result.profile_sync_metadata = captured["public_dynamic_metadata"]
            solver.profile_sync_metadata = captured["solver_dynamic_metadata"]
            if model._reml_result is not None:
                captured["reml_dynamic_metadata"] = {"future": np.array([3.0, 4.0])}
                model._reml_result.profile_sync_metadata = captured["reml_dynamic_metadata"]
            eta = model._dm.matvec(solver.beta) + solver.intercept + model._fit_offset
            eta = stabilize_eta(eta, model._link)
            captured["solver_mu"] = clip_mu(model._link.inverse(eta), model._distribution)
            captured["old_covariance"] = model._coef_covariance
            captured["old_active_info"] = model._fit_active_info
            captured["old_inference_info"] = model._fit_inference_info
            captured["old_group_edf"] = model._group_edf
            captured["old_metrics"] = model.metrics(
                X, y, sample_weight=sample_weight, offset=offset
            )
            captured["old_summary"] = model.summary()
            return fitted

        monkeypatch.setattr(model, fit_name, final_fit_with_primed_caches)
        real_release = fit_ops_module._maybe_release_fit_state
        release_events = []

        def release_spy(candidate):
            release_events.append((candidate._retain_fit_state, candidate._tweedie_profile_result))
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
        assert model._tweedie_profile_result is result
        assert all(profile_result is None for _, profile_result in release_events)
        expected_release_flags = [True] if retain_fit_state else [True, False]
        assert [flag for flag, _ in release_events] == expected_release_flags

        assert model.family is model._distribution
        assert model.family.p == pytest.approx(result.p_hat)
        assert model.result.phi == pytest.approx(result.phi_hat)
        assert model._solver_pirls_result().phi == pytest.approx(result.phi_hat)
        assert model.result is not captured["public_result"]
        assert model._solver_pirls_result() is not captured["solver_result"]
        assert model.result.beta is captured["public_result"].beta
        assert model._solver_pirls_result().beta is captured["solver_result"].beta
        assert model.result.profile_sync_metadata is captured["public_dynamic_metadata"]
        assert (
            model._solver_pirls_result().profile_sync_metadata
            is captured["solver_dynamic_metadata"]
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
            assert model._reml_result.profile_sync_metadata is captured["reml_dynamic_metadata"]
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
        monkeypatch.setattr(
            tweedie_module,
            "_estimate_tweedie_p_prepared",
            lambda *args, **kwargs: result,
        )
        seen_retain_flags = []

        def failing_final_fit(*args, **kwargs):
            seen_retain_flags.append(model._retain_fit_state)
            raise RuntimeError("final refit failed")

        fit_name = "fit_reml" if fit_mode == "reml" else "fit"
        monkeypatch.setattr(model, fit_name, failing_final_fit)

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
        total_evaluations = 2

        def objective(p):
            objective_calls.append(float(p))
            return (float(p) - 1.5) ** 2

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

        def fake_estimate_tweedie_p(candidate, prepared):
            profiler_kwargs.update(
                phi_method=prepared.phi_method,
                method=prepared.method,
            )
            return result

        def unexpected_ci(*args, **kwargs):
            raise AssertionError("public estimate_p must not compute a profile CI eagerly")

        monkeypatch.setattr(
            tweedie_module,
            "_estimate_tweedie_p_prepared",
            fake_estimate_tweedie_p,
        )
        monkeypatch.setattr(result, "ci", unexpected_ci)
        progress_events = []

        returned = model.estimate_p(
            X,
            y,
            progress_callback=lambda phase, payload: progress_events.append((phase, payload)),
        )

        assert returned is result
        assert profiler_kwargs["phi_method"] == "mle"
        assert profiler_kwargs["method"] == "brent"
        assert [phase for phase, _ in progress_events] == ["best_found", "final_refit"]
        assert all(
            payload["profile_estimate"]["ci_status"] == "not computed"
            for _, payload in progress_events
        )
        assert result._ci_cache == {}
        assert result._ci_details_cache == {}
        assert objective_calls == []
        assert result.n_total_evaluations == total_evaluations

        interval = TweedieProfileResult.ci(result, alpha=0.05)
        assert result._ci_cache[0.05] is interval

        summary = model.summary(alpha=0.05)
        assert summary._info["tweedie_p_ci"] is interval
        assert summary._info["tweedie_p_ci_status"] == "available"

    def test_progress_payload_ignores_stale_pearson_lr_cache(self):
        result = SimpleNamespace(
            p_hat=1.5,
            nll=0.0,
            phi_method="pearson",
            _ci_cache={0.05: (1.4, 1.6)},
        )

        payload = profile_ops_module._tweedie_estimate_payload(result)

        assert payload["ci_low"] is None
        assert payload["ci_high"] is None
        assert payload["ci_status"] == "unavailable for Pearson plug-in"

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
            model.estimate_p(
                X,
                y,
                sample_weight=invalid_weights,
                fit_mode="fit",
                phi_method="pearson",
            )

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
                phi_method="pearson",
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
                phi_method="pearson",
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

    def test_unweighted_cpg_default_mle_p_phi_recovery(self):
        """The real public default-MLE fit path should recover p."""
        X, y, p_true = _tweedie_data(n=150)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result = model.estimate_p(
            X,
            y,
            fit_mode="fit",
            method="grid",
            grid=[1.45, 1.6, 1.75],
        )
        assert isinstance(result, TweedieProfileResult)
        assert result.phi_method == "mle"
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)
        np.testing.assert_allclose(result.phi_hat, 3.0, rtol=0.15)
        assert result.converged
        assert result.density_exact
        assert result.n_saddlepoint == 0
        # Model should be refitted with estimated p
        assert model.family.p == result.p_hat
        assert model._result is not None
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
        result = model.estimate_p(X, y, fit_mode="reml", phi_method="pearson")
        assert isinstance(result, TweedieProfileResult)
        np.testing.assert_allclose(result.p_hat, p_true, atol=0.2)
        # Model should be refitted with REML
        assert model.family.p == result.p_hat
        assert model._last_fit_meta["method"] == "fit_reml"
        assert hasattr(model, "_reml_result")

    @pytest.mark.slow
    def test_flexible_spline_reml_mle_p_phi_recovery(self):
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
        np.testing.assert_allclose(result.phi_hat, 3.0, rtol=0.20)
        assert result.converged
        assert not result.phi_used_fallback
        winning_row = result.search_trace.iloc[result.search_trace["nll"].to_numpy().argmin()]
        assert winning_row["p"] == pytest.approx(result.p_hat)
        assert winning_row["edf"] == pytest.approx(model.result.effective_df, rel=1e-8, abs=1e-8)
        assert model.result.phi == pytest.approx(result.phi_hat, rel=1e-12, abs=1e-12)
        assert result.density_exact
        assert result.n_saddlepoint == 0
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

        result = model.estimate_p(X, y, fit_mode="inherit", phi_method="pearson")
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

        result = model.estimate_p(X, y, fit_mode="inherit", phi_method="pearson")

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

        result = model.estimate_p(X, y, fit_mode="inherit", phi_method="pearson")
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
        model.estimate_p(X, y, fit_mode="inherit", phi_method="pearson")
        assert model._last_fit_meta["method"] == "fit"

    def test_invalid_fit_mode_raises(self):
        """Invalid fit_mode should raise immediately."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(ValueError, match="fit_mode"):
            model.estimate_p(X, y, fit_mode="bogus", phi_method="pearson")

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
            model.estimate_p(X, y, phi_method="pearson")

    @pytest.mark.slow
    def test_reml_and_fit_agree_on_p(self):
        """REML and fit paths should agree on p estimate for the same data."""
        X, y, p_true = _tweedie_data()
        model_fit = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Numeric()},
        )
        result_fit = model_fit.estimate_p(X, y, fit_mode="fit", phi_method="pearson")

        model_reml = SuperGLM(
            family=TweedieDistribution(p=1.5),
            selection_penalty=0,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        result_reml = model_reml.estimate_p(X, y, fit_mode="reml", phi_method="pearson")

        # Both should land near p_true; allow wider tolerance since
        # different model flexibility may shift the estimate slightly
        np.testing.assert_allclose(result_fit.p_hat, result_reml.p_hat, atol=0.3)


# =====================================================================
# Search methods
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

    def test_fit_context_owns_arrays_used_by_lazy_profile_probes(self):
        X, y, sample_weight, offset = self._problem()
        ctx = tweedie_module._build_profile_context(
            self._model(),
            X,
            y,
            sample_weight,
            offset,
            "pearson",
            False,
        )
        expected_design = ctx.dm.toarray().copy()
        expected_y = ctx.y_arr.copy()
        expected_weight = ctx.w_arr.copy()
        expected_offset = ctx.offset_arr.copy()

        assert not np.shares_memory(ctx.y_arr, y)
        assert not np.shares_memory(ctx.w_arr, sample_weight)
        assert not np.shares_memory(ctx.offset_arr, offset)

        X.iloc[:, 0] = 99.0
        y[:] = 101.0
        sample_weight[:] = 103.0
        offset[:] = 107.0

        np.testing.assert_array_equal(ctx.dm.toarray(), expected_design)
        np.testing.assert_array_equal(ctx.y_arr, expected_y)
        np.testing.assert_array_equal(ctx.w_arr, expected_weight)
        np.testing.assert_array_equal(ctx.offset_arr, expected_offset)

    def test_reml_context_owns_all_inputs_used_by_lazy_profile_probes(self):
        X, y, sample_weight, offset = self._problem()
        ctx = tweedie_module._build_profile_context_reml(
            self._model(),
            X,
            y,
            sample_weight,
            offset,
            "pearson",
            False,
        )
        expected_X = ctx.X.copy(deep=True)
        expected_y = ctx.y.copy()
        expected_weight = np.asarray(ctx.sample_weight).copy()
        expected_offset = np.asarray(ctx.offset).copy()

        assert ctx.X is not X
        assert not np.shares_memory(ctx.X["x"].to_numpy(), X["x"].to_numpy())
        assert not np.shares_memory(ctx.y, y)
        assert not np.shares_memory(ctx.sample_weight, sample_weight)
        assert not np.shares_memory(ctx.offset, offset)

        X.iloc[:, 0] = 99.0
        y[:] = 101.0
        sample_weight[:] = 103.0
        offset[:] = 107.0

        pd.testing.assert_frame_equal(ctx.X, expected_X)
        np.testing.assert_array_equal(ctx.y, expected_y)
        np.testing.assert_array_equal(ctx.sample_weight, expected_weight)
        np.testing.assert_array_equal(ctx.offset, expected_offset)


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
        assert model._fit_mu is not None
        mu = np.asarray(model._fit_mu, dtype=np.float64)
        edf = float(model.result.effective_df)
        phi = estimate_phi(y, mu, 1.5, df_resid=float(len(y)) - edf)
        nll = -float(np.mean(tweedie_logpdf(y, mu, phi, 1.5)))
        return edf, phi, nll

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
        assert clone._pending_interactions == []
        cloned = clone._interaction_specs["custom_surface"]
        assert isinstance(cloned, TensorInteraction)
        assert cloned.parent_names == ("x1", "x2")
        assert cloned._n_knots == (3, 4)
        assert cloned._decompose is True
        assert (cloned._p1, cloned._p2) == (original._p1, original._p2)
        assert cloned._marginal1 is not None
        assert cloned._marginal2 is not None
        assert cloned._R_inv is not None

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
        independent_edf, independent_phi, independent_nll = self._fixed_p_metrics(
            independent,
            y,
        )
        expected_groups = [
            ("x1", 8),
            ("x2", 9),
            ("custom_surface:bilinear", 1),
            ("custom_surface:wiggly", 41),
        ]
        assert self._group_signature(independent._groups) == expected_groups

        ctx = tweedie_module._build_profile_context(
            self._custom_tensor_model(),
            X,
            y,
            None,
            None,
            "pearson",
            False,
        )

        assert self._group_signature(ctx.groups) == expected_groups
        ctx.evaluate(1.5, source="one_point")
        profiled = ctx.finalize(1.5, method="grid", converged=True)

        assert profiled.search_trace.iloc[0]["edf"] == pytest.approx(
            independent_edf,
            rel=1e-10,
            abs=1e-10,
        )
        assert profiled.phi_hat == pytest.approx(independent_phi, rel=1e-10, abs=1e-10)
        assert profiled.nll == pytest.approx(independent_nll, rel=1e-10, abs=1e-10)

    @pytest.mark.slow
    def test_reml_profile_custom_tensor_matches_independent_fixed_p_fit(self):
        X, y = self._custom_tensor_problem()
        independent = self._custom_tensor_model()
        independent.fit_reml(X, y)
        independent_edf, independent_phi, independent_nll = self._fixed_p_metrics(
            independent,
            y,
        )
        expected_groups = [
            ("x1", 8),
            ("x2", 9),
            ("custom_surface:bilinear", 1),
            ("custom_surface:wiggly", 41),
        ]
        assert self._group_signature(independent._groups) == expected_groups

        ctx = tweedie_module._build_profile_context_reml(
            self._custom_tensor_model(),
            X,
            y,
            None,
            None,
            "pearson",
            False,
        )

        assert ctx.model._interaction_order == ["custom_surface"]
        ctx.evaluate(1.5, source="one_point")
        profiled = ctx.finalize(1.5, method="grid", converged=True)

        assert self._group_signature(ctx.model._groups) == expected_groups
        assert profiled.search_trace.iloc[0]["edf"] == pytest.approx(
            independent_edf,
            rel=1e-8,
            abs=1e-8,
        )
        assert profiled.phi_hat == pytest.approx(independent_phi, rel=1e-8, abs=1e-8)
        assert profiled.nll == pytest.approx(independent_nll, rel=1e-8, abs=1e-8)

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

        result = estimate_tweedie_p(
            model,
            X,
            y,
            fit_mode=fit_mode,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )
        result._objective(1.6, source="later_probe")

        _assert_fitted_model_unchanged(model, X, snapshot)

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
        assert model._pending_interactions == [("x1", "x2")]

        clone = tweedie_module._clone_profile_model(model, X, None)

        assert clone._interaction_specs == {}
        assert clone._interaction_order == []
        assert clone._pending_interactions == [("x1", "x2")]
        assert list(clone._specs) == ["x1", "x2"]
        clone._build_design_matrix(X, y, None, None)
        assert clone._pending_interactions == []
        assert clone._interaction_order == ["x1:x2"]
        assert isinstance(clone._interaction_specs["x1:x2"], TensorInteraction)

        profiled = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            n_grid=20,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            grid=grid,
            phi_method="pearson",
        )
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
            model,
            X,
            y,
            method="grid_refine",
            n_grid_coarse=10,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="profile_opt",
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
            model,
            X,
            y,
            method="profile_opt",
            optimizer="Powell",
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
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

    def test_exact_profile_has_no_saddlepoint_warning(self):
        """Certified profile likelihood never reports implicit approximation use."""
        X, y, _ = _tweedie_data(n=50, p_true=1.25, seed=7)
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
                method="grid",
                grid=[1.25],
                phi_method="pearson",
            )

        messages = [str(item.message) for item in caught]
        assert not any("Saddlepoint approximation used" in message for message in messages)
        assert result.saddlepoint_fraction == 0.0
        assert result.n_saddlepoint == 0
        assert result.density_exact

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
            model,
            X,
            y,
            sample_weight=sample_weight,
            method="grid",
            n_grid=15,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
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
            model,
            X,
            y,
            method="grid",
            n_grid=10,
            fit_mode="fit_reml",
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
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
            estimate_tweedie_p(model, X, y, method="bogus", phi_method="pearson")

    def test_invalid_optimizer_raises(self):
        """Invalid optimizer should raise ValueError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(ValueError, match="optimizer"):
            estimate_tweedie_p(
                model,
                X,
                y,
                method="profile_opt",
                optimizer="bogus",
                phi_method="pearson",
            )

    def test_joint_ml_not_implemented(self):
        """method='joint_ml' should raise NotImplementedError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(NotImplementedError, match="joint_ml"):
            estimate_tweedie_p(model, X, y, method="joint_ml", phi_method="pearson")

    def test_integrated_not_implemented(self):
        """method='integrated' should raise NotImplementedError."""
        X, y, _ = _tweedie_data()
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        with pytest.raises(NotImplementedError, match="integrated"):
            estimate_tweedie_p(model, X, y, method="integrated", phi_method="pearson")


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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            n_grid=n_grid,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
            model,
            X,
            y,
            method="grid_refine",
            n_grid_coarse=8,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="profile_opt",
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
        r1 = estimate_tweedie_p(
            m1,
            X,
            y,
            method="brent",
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )

        m2 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r2 = estimate_tweedie_p(
            m2,
            X,
            y,
            method="grid",
            n_grid=30,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )

        np.testing.assert_allclose(r1.p_hat, r2.p_hat, atol=0.1)

    def test_grid_refine_vs_brent_agree(self):
        """Grid-refine and Brent should agree within tolerance."""
        X, y, _ = _tweedie_data(n=3000)

        m1 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r1 = estimate_tweedie_p(
            m1,
            X,
            y,
            method="brent",
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )

        m2 = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        r2 = estimate_tweedie_p(
            m2,
            X,
            y,
            method="grid_refine",
            n_grid_coarse=10,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )

        np.testing.assert_allclose(r1.p_hat, r2.p_hat, atol=0.1)

    @pytest.mark.parametrize("method", ["brent", "grid", "grid_refine", "profile_opt"])
    def test_all_methods_recover_p(self, method):
        """All profile methods should recover p from clean synthetic data."""
        X, y, p_true = _tweedie_data(n=3000)
        model = SuperGLM(
            family=TweedieDistribution(p=1.5), selection_penalty=0, features={"x1": Numeric()}
        )
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method=method,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )
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
        result = estimate_tweedie_p(
            model,
            X,
            y,
            method="grid",
            n_grid=5,
            p_bounds=(1.1, 1.9),
            phi_method="pearson",
        )

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

    def test_origin_master_pickle_restores_missing_ci_details_cache(self):
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
        )
        origin_master_fields = {
            "p_hat",
            "phi_hat",
            "nll",
            "n_evaluations",
            "converged",
            "method",
            "phi_method",
            "search_trace",
            "saddlepoint_fraction",
            "n_saddlepoint",
            "n_positive",
            "warnings",
            "_objective",
            "_ll_scale",
            "_ci_cache",
        }
        for name in set(vars(result)) - origin_master_fields:
            del result.__dict__[name]

        restored = pickle.loads(pickle.dumps(result))

        interval = restored.ci()
        assert restored._ci_cache[0.05] is interval
        assert restored._ci_details_cache[0.05].interval is interval
