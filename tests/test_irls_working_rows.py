"""Exact coefficient-working-row regressions for direct GLM fitting."""

from __future__ import annotations

import numpy as np
import pytest

from superglm._tweedie import tweedie_logpdf
from superglm.distributions import Gamma, Gaussian, Poisson, Tweedie
from superglm.links import IdentityLink, LogLink, SqrtLink
from superglm.solvers.working_rows import (
    _gamma_log_observed_rows,
    _tweedie_log_observed_rows,
    coefficient_initial_intercept,
    coefficient_working_rows,
)


def test_gaussian_identity_preserves_exact_constant_working_rows() -> None:
    y = np.array([1.0, -3.0, 7.5])
    eta = np.array([1.0e16, -1.0e16, 3.0])
    sample_weight = np.array([0.5, 2.0, 4.0])

    rows = coefficient_working_rows(
        distribution=Gaussian(),
        link=IdentityLink(),
        y=y,
        mu=eta,
        eta=eta,
        sample_weight=sample_weight,
        prefer_observed=False,
    )

    np.testing.assert_array_equal(rows.response, y)
    np.testing.assert_array_equal(rows.weights, sample_weight)
    assert not np.shares_memory(rows.response, y)
    assert not np.shares_memory(rows.weights, sample_weight)


def test_gamma_log_uses_exact_observed_newton_rows() -> None:
    y = np.array([0.25, 1.5, 8.0, 3.0])
    mu = np.array([0.5, 1.0, 4.0, 6.0])
    eta = np.log(mu)
    sample_weight = np.array([2.0, 0.5, 3.0, 0.0])

    rows = coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=eta,
        sample_weight=sample_weight,
        prefer_observed=True,
    )

    assert rows.curvature_source == "observed"
    np.testing.assert_allclose(
        rows.weights,
        sample_weight * y / mu,
        rtol=0.0,
        atol=0.0,
    )
    expected_z = eta.copy()
    active = sample_weight > 0.0
    expected_z[active] += (y[active] - mu[active]) / y[active]
    np.testing.assert_allclose(rows.response, expected_z, rtol=2e-16, atol=2e-16)


EPS = np.finfo(np.float64).eps
# Zero rows included: Tweedie/log curvature is positive at y = 0.
TWEEDIE_Y = np.array([0.0, 0.0, 1.0e-9, 0.3, 1.0, 4.0, 25.0])
TWEEDIE_MU = np.array([0.05, 3.0, 0.9, 1.0, 1.0, 2.0, 0.5])
TWEEDIE_W = np.array([1.0, 2.5, 1.0, 0.5, 1.0, 3.0, 1.0])


def _tweedie_log_rows(p: float):
    return coefficient_working_rows(
        distribution=Tweedie(p),
        link=LogLink(),
        y=TWEEDIE_Y,
        mu=TWEEDIE_MU,
        eta=np.log(TWEEDIE_MU),
        sample_weight=TWEEDIE_W,
        prefer_observed=True,
    )


@pytest.mark.parametrize("p", [1.1, 1.5, 1.8, 1.95])
def test_tweedie_log_observed_rows_are_the_exact_newton_model(p: float) -> None:
    """W = -d2l/deta2 and z = eta + (dl/deta) / W for l = log f at phi = 1."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    rows = _tweedie_log_rows(p)
    assert rows.curvature_source == "observed"
    eta = np.log(TWEEDIE_MU)
    for i, (y, mu, w) in enumerate(zip(TWEEDIE_Y, TWEEDIE_MU, TWEEDIE_W, strict=True)):
        # tweedie_logpdf depends on the mean only through -w d(y, mu) / 2, whose
        # eta-dependent part is this closed form; mpmath differentiates it.
        P, Y, W = mp.mpf(p), mp.mpf(y), mp.mpf(w)

        def loglik(t, P=P, Y=Y, W=W):
            return W * (Y * mp.exp((1 - P) * t) / (1 - P) - mp.exp((2 - P) * t) / (2 - P))

        at = mp.log(mp.mpf(mu))
        score, curvature = mp.diff(loglik, at, 1), -mp.diff(loglik, at, 2)
        # W: c = (2-p)mu + (p-1)y adds two non-negative products (3u), one pow
        # within an ulp (2u) and two products (2u): 7u.
        assert abs(float(mp.mpf(rows.weights[i]) / curvature - 1)) <= 7 * EPS / 2
        # z: y - mu (u), the quotient by c (3u + u), then the sum with eta (u);
        # 1e-40 covers the 50-digit differentiation itself where z is zero.
        step = float(score / curvature)
        expected = mp.mpf(eta[i]) + score / curvature
        bound = (EPS / 2) * (6 * abs(step) + abs(eta[i])) + 1e-40
        assert abs(float(mp.mpf(rows.response[i]) - expected)) <= bound


@pytest.mark.parametrize("p", [1.1, 1.5, 1.8, 1.95])
def test_tweedie_log_observed_rows_differentiate_tweedie_logpdf(p: float) -> None:
    """Central differences of the package's own density recover W and W (z - eta)."""
    rows = _tweedie_log_rows(p)
    eta = np.log(TWEEDIE_MU)

    def logpdf(shift):
        return tweedie_logpdf(TWEEDIE_Y, np.exp(eta + shift), 1.0, p, weights=TWEEDIE_W)

    centre = logpdf(0.0)
    # |d^k l / deta^k| <= M e^h for every k (|1-p|, |2-p| < 1), and each density
    # value is within delta = 64 eps max(1, |l|). Steps minimise truncation plus
    # round-off: h^2 M / 6 + delta / h for the score, h^2 M / 12 + 4 delta / h^2
    # for the curvature.
    bound_m = TWEEDIE_W * (TWEEDIE_Y * TWEEDIE_MU ** (1 - p) + TWEEDIE_MU ** (2 - p)) * np.e
    delta = 64 * EPS * np.maximum(1.0, np.abs(centre))
    h1 = np.cbrt(3 * delta / bound_m)
    h2 = (48 * delta / bound_m) ** 0.25
    score = (logpdf(h1) - logpdf(-h1)) / (2 * h1)
    curvature = -(logpdf(h2) - 2 * centre + logpdf(-h2)) / h2**2
    np.testing.assert_array_less(
        np.abs(rows.weights * (rows.response - eta) - score),
        h1**2 * bound_m / 6 + delta / h1,
    )
    np.testing.assert_array_less(
        np.abs(rows.weights - curvature),
        h2**2 * bound_m / 12 + 4 * delta / h2**2,
    )


def test_tweedie_log_observed_rows_reduce_to_gamma_at_p_two() -> None:
    """At p = 2, c = y: W = w y / mu and z = eta + (y - mu) / y, the Gamma/log rows."""
    y = TWEEDIE_Y[2:]
    mu = TWEEDIE_MU[2:]
    w = TWEEDIE_W[2:]
    eta = np.log(mu)
    tweedie_w, tweedie_z = _tweedie_log_observed_rows(2.0, y=y, mu=mu, eta=eta, sample_weight=w)
    gamma_w, gamma_z = _gamma_log_observed_rows(
        y=y, mu=mu, eta=eta, sample_weight=w, active=w > 0.0
    )
    # 0 * mu + 1 * y is exactly y, so the responses take identical operations;
    # the weights differ by one pow (2u) and the order of two products (2u each).
    np.testing.assert_array_equal(tweedie_z, gamma_z)
    np.testing.assert_allclose(tweedie_w, gamma_w, rtol=3 * EPS, atol=0.0)


def test_observed_newton_can_be_disabled_for_fisher_controller() -> None:
    y = np.array([0.5, 2.0, 4.0])
    mu = np.array([1.0, 1.5, 3.0])
    eta = np.log(mu)
    sample_weight = np.array([1.0, 2.0, 0.5])

    rows = coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=eta,
        sample_weight=sample_weight,
        prefer_observed=False,
    )

    assert rows.curvature_source == "fisher"
    np.testing.assert_allclose(rows.weights, sample_weight, rtol=2e-16, atol=2e-16)
    np.testing.assert_allclose(rows.response, eta + (y - mu) / mu)


def test_unapproved_family_link_pairs_retain_fisher_scoring() -> None:
    y = np.array([0.0, 1.0, 3.0])
    mu = np.array([0.5, 1.5, 2.5])
    sample_weight = np.array([1.0, 2.0, 0.5])

    poisson = coefficient_working_rows(
        distribution=Poisson(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=np.log(mu),
        sample_weight=sample_weight,
        prefer_observed=True,
    )
    gamma_identity = coefficient_working_rows(
        distribution=Gamma(),
        link=IdentityLink(),
        y=np.maximum(y, 0.25),
        mu=mu,
        eta=mu,
        sample_weight=sample_weight,
        prefer_observed=True,
    )

    assert poisson.curvature_source == "fisher"
    assert gamma_identity.curvature_source == "fisher"


def test_invalid_observed_rows_fall_back_atomically_to_fisher() -> None:
    y = np.array([2.0, 1.0])
    mu = np.ones(2)
    eta = np.log(mu)
    sample_weight = np.array([1.0e308, 1.0])

    rows = coefficient_working_rows(
        distribution=Gamma(),
        link=LogLink(),
        y=y,
        mu=mu,
        eta=eta,
        sample_weight=sample_weight,
        prefer_observed=True,
    )

    assert rows.curvature_source == "fisher"
    assert rows.fallback_reason == "invalid_observed_rows"
    assert np.all(np.isfinite(rows.weights))
    assert np.all(np.isfinite(rows.response))


def test_poisson_sqrt_exact_zero_uses_structural_fisher_limit() -> None:
    y = np.array([0.0, 1.0e-30, 1.0e-16, 100.0, 1.0e12])
    eta = np.array([0.0, -0.0, 0.0, -0.0, 0.0])
    sample_weight = np.array([0.5, 1.0, 2.0, 3.0, 4.0])

    rows = coefficient_working_rows(
        distribution=Poisson(),
        link=SqrtLink(),
        y=y,
        mu=np.full_like(y, 1.0e-50),
        eta=eta,
        sample_weight=sample_weight,
        prefer_observed=False,
    )

    np.testing.assert_allclose(
        rows.weights,
        4.0 * sample_weight,
        rtol=2.0e-16,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        rows.response,
        np.copysign(np.sqrt(y), eta),
    )


def test_poisson_sqrt_tiny_nonzero_predictor_keeps_fisher_row() -> None:
    eta = np.array([1.0e-6, -1.0e-4, 1.0e-60, -1.0e-150])
    y = eta**2
    numerically_floored_mu = np.maximum(y, 1.0e-50)

    rows = coefficient_working_rows(
        distribution=Poisson(),
        link=SqrtLink(),
        y=y,
        mu=numerically_floored_mu,
        eta=eta,
        sample_weight=np.ones_like(y),
        prefer_observed=False,
    )

    np.testing.assert_array_equal(rows.weights, np.full_like(y, 4.0))
    np.testing.assert_allclose(rows.response, eta, rtol=2.0e-16, atol=0.0)


@pytest.mark.parametrize(
    "eta",
    [
        np.array([np.nextafter(0.0, 1.0), -1.0e-307]),
        np.array([1.0e-305] * 20),
    ],
)
def test_poisson_sqrt_unrepresentable_fisher_system_uses_finite_trust_response(
    eta: np.ndarray,
) -> None:
    y = np.full_like(eta, 100.0)

    rows = coefficient_working_rows(
        distribution=Poisson(),
        link=SqrtLink(),
        y=y,
        mu=eta**2,
        eta=eta,
        sample_weight=np.ones_like(y),
        prefer_observed=False,
    )

    np.testing.assert_array_equal(rows.weights, np.full_like(y, 4.0))
    np.testing.assert_array_equal(
        rows.response,
        np.copysign(np.sqrt(y), eta),
    )
    assert np.all(np.isfinite(rows.response))


def test_poisson_sqrt_representable_large_fisher_system_is_unchanged() -> None:
    eta = np.array([1.0e-300, -1.0e-300])
    y = np.full_like(eta, 100.0)

    rows = coefficient_working_rows(
        distribution=Poisson(),
        link=SqrtLink(),
        y=y,
        mu=eta**2,
        eta=eta,
        sample_weight=np.ones_like(y),
        prefer_observed=False,
    )

    expected = 0.5 * (eta + y / eta)
    np.testing.assert_array_equal(rows.response, expected)


def test_poisson_sqrt_initial_intercept_preserves_tiny_response_mean() -> None:
    y = np.array([1.0e-30, 4.0e-30])
    weights = np.array([1.0, 3.0])

    intercept = coefficient_initial_intercept(
        distribution=Poisson(),
        link=SqrtLink(),
        y=y,
        sample_weight=weights,
    )

    assert intercept**2 == pytest.approx(
        np.average(y, weights=weights),
        rel=2.0e-16,
    )
