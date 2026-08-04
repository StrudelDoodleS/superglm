"""Exact coefficient-working-row regressions for direct GLM fitting."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Gamma, Gaussian, Poisson
from superglm.links import IdentityLink, LogLink, SqrtLink
from superglm.solvers.working_rows import (
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
