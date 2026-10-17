"""Discrete REML integration checks for family-correct scale profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features import Spline
from superglm.reml.scale import (
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
)


def _gamma_data(n: int = 240) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4821)
    x = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.35 + 0.7 * np.sin(2.2 * x))
    y = rng.gamma(shape=4.5, scale=mu / 4.5)
    weights = rng.integers(1, 4, size=n).astype(np.float64)
    return pd.DataFrame({"x": x}), y, weights


def _penalized_deviance(result, group_matrices, lambdas, penalties) -> float:
    penalty_quad = 0.0
    for component in penalties:
        omega = component.omega_ssp
        if omega is None:
            gm = group_matrices[component.group_index]
            omega = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta = result.beta[component.group_sl]
        penalty_quad += float(lambdas[component.name]) * float(beta @ omega @ beta)
    return float(result.deviance + penalty_quad)


def test_discrete_gamma_prepares_once_and_forwards_fd_profile_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.reml.discrete as discrete

    X, y, weights = _gamma_data()
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")
    prepare_calls = 0
    hessian_checks = 0
    original_prepare = discrete.prepare_reml_scale_data
    original_hessian = discrete.reml_direct_hessian

    def counted_prepare(distribution, actual_y, actual_weights, *, weight_semantics):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(
            distribution,
            actual_y,
            actual_weights,
            weight_semantics=weight_semantics,
        )

    def checked_hessian(*args, **kwargs):
        nonlocal hessian_checks
        group_matrices = args[0]
        lambdas = args[3]
        result = kwargs["pirls_result"]
        penalties = kwargs["reml_penalties"]
        penalty_nullity = kwargs["penalty_nullity"]
        expected = profile_gamma_reml_scale(
            profile_data,
            _penalized_deviance(result, group_matrices, lambdas, penalties),
            penalty_nullity,
        )
        assert kwargs["inverse_phi"] == pytest.approx(expected.inverse_phi, rel=2.0e-13)
        assert kwargs["d_inverse_phi_d_penalized_deviance"] == pytest.approx(
            expected.d_inverse_phi_d_penalized_deviance,
            rel=2.0e-13,
        )

        step = max(1.0e-6, 1.0e-6 * _penalized_deviance(result, group_matrices, lambdas, penalties))
        lower = profile_gamma_reml_scale(
            profile_data,
            _penalized_deviance(result, group_matrices, lambdas, penalties) - step,
            penalty_nullity,
        )
        upper = profile_gamma_reml_scale(
            profile_data,
            _penalized_deviance(result, group_matrices, lambdas, penalties) + step,
            penalty_nullity,
        )
        finite_difference = (upper.inverse_phi - lower.inverse_phi) / (2.0 * step)
        assert kwargs["d_inverse_phi_d_penalized_deviance"] == pytest.approx(
            finite_difference,
            rel=3.0e-7,
            abs=2.0e-12,
        )
        hessian_checks += 1
        return original_hessian(*args, **kwargs)

    monkeypatch.setattr(discrete, "prepare_reml_scale_data", counted_prepare)
    monkeypatch.setattr(discrete, "reml_direct_hessian", checked_hessian)

    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        discrete=True,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )
    model.fit_reml(
        X,
        y,
        sample_weight=weights,
        max_reml_iter=6,
        runtime_validation="skip",
    )

    assert prepare_calls == 1
    assert hessian_checks >= 2
