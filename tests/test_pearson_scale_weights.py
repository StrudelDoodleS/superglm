"""Frequency-weight invariants for Pearson dispersion estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, SuperGLM
from superglm.distributions import Gamma, Gaussian, Tweedie
from superglm.features.spline import PSpline
from superglm.links import IdentityLink, LogLink
from superglm.penalties.ridge import Ridge
from superglm.reml import scop_efs as scop_efs_module
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls
from superglm.types import GroupSlice


def _frequency_problem(
    distribution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    x = np.array([-1.4, -1.0, -0.55, -0.1, 0.35, 0.8, 1.25, 1.65])
    residual_pattern = np.array([0.18, -0.24, 0.08, -0.15, 0.22, -0.12, 0.17, -0.09])
    weights = np.array([1, 3, 2, 1, 4, 2, 3, 1], dtype=np.float64)
    if isinstance(distribution, Gaussian):
        y = 0.7 + 0.55 * x + residual_pattern
        link = IdentityLink()
    else:
        y = np.exp(0.4 + 0.3 * x) * np.exp(residual_pattern)
        link = LogLink()
    return x[:, None], y, weights, link


def _replicate_rows(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.repeat(np.arange(len(y)), weights.astype(int))
    return X[indices], y[indices], np.ones(indices.size, dtype=np.float64)


@pytest.mark.parametrize("distribution", [Gaussian(), Gamma()])
def test_direct_pearson_phi_frequency_weights_match_row_replication(distribution) -> None:
    X, y, weights, link = _frequency_problem(distribution)
    X_repeated, y_repeated, repeated_weights = _replicate_rows(X, y, weights)
    groups = [GroupSlice("x", 0, 1)]

    weighted, _ = fit_irls_direct(
        X,
        y,
        weights,
        distribution,
        link,
        groups,
        lambda2=0.4,
        _use_observed_newton=False,
    )
    repeated, _ = fit_irls_direct(
        X_repeated,
        y_repeated,
        repeated_weights,
        distribution,
        link,
        groups,
        lambda2=0.4,
        _use_observed_newton=False,
    )

    np.testing.assert_allclose(weighted.beta, repeated.beta, rtol=2e-10, atol=2e-10)
    assert weighted.intercept == pytest.approx(repeated.intercept, rel=2e-10, abs=2e-10)
    assert weighted.effective_df == pytest.approx(repeated.effective_df, rel=2e-10)
    assert weighted.phi == pytest.approx(repeated.phi, rel=2e-10)


@pytest.mark.parametrize("distribution", [Gaussian(), Gamma()])
def test_composite_pirls_pearson_phi_frequency_weights_match_row_replication(
    distribution,
) -> None:
    X, y, weights, link = _frequency_problem(distribution)
    X_repeated, y_repeated, repeated_weights = _replicate_rows(X, y, weights)
    groups = [GroupSlice("x", 0, 1)]

    weighted = fit_pirls(
        X,
        y,
        weights,
        distribution,
        link,
        groups,
        Ridge(lambda1=0.4),
    )
    repeated = fit_pirls(
        X_repeated,
        y_repeated,
        repeated_weights,
        distribution,
        link,
        groups,
        Ridge(lambda1=0.4),
    )

    np.testing.assert_allclose(weighted.beta, repeated.beta, rtol=2e-9, atol=2e-9)
    assert weighted.intercept == pytest.approx(repeated.intercept, rel=2e-9, abs=2e-9)
    assert weighted.effective_df == pytest.approx(repeated.effective_df, rel=2e-9)
    assert weighted.phi == pytest.approx(repeated.phi, rel=2e-9)


def test_scop_pearson_phi_frequency_weights_match_row_replication() -> None:
    x = np.linspace(0.0, 1.0, 28)
    y = 0.6 + 1.4 * x + 0.06 * np.sin(6.0 * x)
    weights = np.resize(np.array([1, 3, 2, 4], dtype=np.float64), x.size)
    frame = pd.DataFrame({"x": x})
    repeated_indices = np.repeat(np.arange(x.size), weights.astype(int))
    repeated_frame = frame.iloc[repeated_indices].reset_index(drop=True)

    def model() -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.8,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=6,
                    knot_strategy="uniform",
                    constraint=Constraint.fit.increasing,
                )
            },
        )

    weighted = model().fit(frame, y, sample_weight=weights)
    repeated = model().fit(repeated_frame, y[repeated_indices])

    np.testing.assert_allclose(
        weighted.predict(frame),
        repeated.predict(frame),
        rtol=5e-6,
        atol=2e-6,
    )
    assert weighted.result.effective_df == pytest.approx(
        repeated.result.effective_df,
        rel=5e-6,
    )
    assert weighted.result.phi == pytest.approx(repeated.result.phi, rel=2e-6)


def test_postfit_shape_repair_uses_frequency_weight_likelihood_size() -> None:
    x = np.linspace(0.0, 1.0, 28)
    y = 1.5 - 1.1 * x + 0.08 * np.sin(7.0 * x)
    weights = np.resize(np.array([1, 3, 2, 4], dtype=np.float64), x.size)
    frame = pd.DataFrame({"x": x})
    weighted = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.8,
        features={
            "x": PSpline(
                n_knots=6,
                knot_strategy="uniform",
                constraint=Constraint.postfit.increasing,
            )
        },
    ).fit(frame, y, sample_weight=weights)
    weighted.apply_shape_postfit(frame, sample_weight=weights, n_grid=80)

    variance = weighted._distribution.variance(weighted._fit_mu)
    pearson = float(np.sum(weights * (y - weighted._fit_mu) ** 2 / variance))
    expected = pearson / max(float(np.sum(weights)) - weighted.result.effective_df, 1.0)
    assert weighted.result.phi == pytest.approx(expected, rel=2e-13, abs=2e-13)


def test_postfit_shape_repair_frequency_weights_match_row_replication() -> None:
    x = np.linspace(0.0, 1.0, 28)
    y = 1.5 - 1.1 * x + 0.08 * np.sin(7.0 * x)
    weights = np.resize(np.array([1, 3, 2, 4], dtype=np.float64), x.size)
    frame = pd.DataFrame({"x": x})
    repeated_indices = np.repeat(np.arange(x.size), weights.astype(int))
    repeated_frame = frame.iloc[repeated_indices].reset_index(drop=True)

    def model() -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=0.8,
            features={
                "x": PSpline(
                    n_knots=6,
                    knot_strategy="uniform",
                    constraint=Constraint.postfit.increasing,
                )
            },
        )

    weighted = model().fit(frame, y, sample_weight=weights)
    repeated = model().fit(repeated_frame, y[repeated_indices])
    np.testing.assert_allclose(
        weighted.predict(frame),
        repeated.predict(frame),
        rtol=2e-12,
        atol=2e-12,
    )

    weighted.apply_shape_postfit(frame, sample_weight=weights, n_grid=80)
    repeated.apply_shape_postfit(repeated_frame, n_grid=80)

    np.testing.assert_allclose(
        weighted.predict(frame),
        repeated.predict(frame),
        rtol=2e-12,
        atol=2e-12,
    )
    assert weighted.result.phi == pytest.approx(repeated.result.phi, rel=2e-12, abs=2e-12)


def test_weighted_tweedie_scop_reml_uses_observation_count_for_terminal_phi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 36
    x = np.linspace(0.0, 1.0, n)
    y = np.exp(0.3 + 0.5 * x) * (1.0 + 0.06 * np.sin(5.0 * x))
    weights = np.resize(np.array([1, 3, 2, 4], dtype=np.float64), n)
    frame = pd.DataFrame({"x": x})
    fallback_sizes: list[float] = []
    original = scop_efs_module._reml_evaluation_phi

    def recording_phi(evaluation, *, scale_known, fallback_likelihood_size):
        fallback_sizes.append(float(fallback_likelihood_size))
        return original(
            evaluation,
            scale_known=scale_known,
            fallback_likelihood_size=fallback_likelihood_size,
        )

    monkeypatch.setattr(scop_efs_module, "_reml_evaluation_phi", recording_phi)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        spline_penalty=0.8,
        features={
            "x": PSpline(
                n_knots=6,
                knot_strategy="uniform",
                constraint=Constraint.fit.increasing,
            )
        },
    )
    model.fit_reml(frame, y, sample_weight=weights, max_reml_iter=2)

    assert fallback_sizes
    assert fallback_sizes == pytest.approx([float(n)] * len(fallback_sizes))
