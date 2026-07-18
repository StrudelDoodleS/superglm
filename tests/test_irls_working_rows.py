"""Exact coefficient-working-row regressions for direct GLM fitting."""

from __future__ import annotations

import numpy as np

from superglm.distributions import Gamma, Poisson
from superglm.links import IdentityLink, LogLink
from superglm.solvers.working_rows import coefficient_working_rows


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
