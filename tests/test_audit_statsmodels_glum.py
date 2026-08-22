"""Calibration oracles for the replication weight contract.

statsmodels' ``freq_weights`` and glum's ``sum(w)`` dispersion denominator are
the oracles, so every fit here declares ``weight_semantics="frequency"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from superglm import Numeric, SuperGLM

_TRUE_BETA = np.array([0.2, 0.3], dtype=np.float64)
_TRUE_PHI = 0.16


def _fit_model(
    family: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[SuperGLM, object]:
    model = SuperGLM(
        family=family,
        link="identity" if family == "gaussian" else "log",
        selection_penalty=0.0,
        features={"x": Numeric()},
        tol=1e-12,
        max_iter=500,
        weight_semantics="frequency",
    ).fit(frame, y, sample_weight=weights)
    return model, model.metrics(frame, y, sample_weight=weights)


def _parameters(model: SuperGLM) -> np.ndarray:
    return np.r_[model.result.intercept, model.result.beta]


def _full_covariance(metrics) -> np.ndarray:
    augmented_inverse = np.asarray(metrics._active_info[3], dtype=np.float64)
    return float(metrics.phi) * augmented_inverse


def _standard_errors(model: SuperGLM, metrics) -> np.ndarray:
    return np.r_[
        metrics.intercept_se,
        *(metrics.coefficient_se[name] for name in model._feature_order),
    ]


def _statsmodels_family(sm, family: str):
    if family == "gaussian":
        return sm.families.Gaussian(link=sm.families.links.Identity())
    if family == "gamma":
        return sm.families.Gamma(link=sm.families.links.Log())
    raise AssertionError(f"unsupported family {family!r}")


def _known_dispersion_problem(
    family: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build a full-rank score-balanced fixture with exactly known Pearson phi."""
    n_rows = 48
    x = np.linspace(-1.7, 2.1, n_rows)
    weights = (1 + (7 * np.arange(n_rows)) % 5).astype(np.float64)
    design = np.column_stack([np.ones(n_rows), x])

    # Project a deterministic residual shape off the weighted model space.
    # Consequently X' W r == 0, so _TRUE_BETA is the exact Gaussian/identity
    # and Gamma/log score root.
    base = np.sin(1.7 * x) + 0.35 * np.cos(3.1 * x) + 0.2 * np.sin(0.9 * np.arange(n_rows))
    sqrt_weights = np.sqrt(weights)
    weighted_design = sqrt_weights[:, None] * design
    weighted_residual = sqrt_weights * base
    weighted_residual -= (
        weighted_design
        @ np.linalg.lstsq(
            weighted_design,
            weighted_residual,
            rcond=None,
        )[0]
    )
    residual = weighted_residual / sqrt_weights

    residual_df = float(weights.sum() - design.shape[1])
    residual *= np.sqrt(_TRUE_PHI * residual_df / np.dot(weights, np.square(residual)))
    np.testing.assert_allclose(design.T @ (weights * residual), 0.0, atol=3e-14)
    assert np.dot(weights, np.square(residual)) / residual_df == pytest.approx(
        _TRUE_PHI,
        rel=3e-15,
    )

    eta = design @ _TRUE_BETA
    if family == "gaussian":
        y = eta + residual
    elif family == "gamma":
        y = np.exp(eta) * (1.0 + residual)
        assert np.all(y > 0.0)
    else:  # pragma: no cover - helper is only called by the parametrization
        raise AssertionError(f"unsupported family {family!r}")

    return pd.DataFrame({"x": x}), y, weights, design


@pytest.mark.parametrize("family", ["gaussian", "gamma"])
def test_known_dispersion_frequency_rows_match_statsmodels_and_replication(
    family: str,
) -> None:
    """Known phi and Fisher covariance use sum(frequency)-p, not n-p."""
    sm = pytest.importorskip("statsmodels.api")
    frame, y, weights, design = _known_dispersion_problem(family)
    expanded_rows = np.repeat(np.arange(len(y)), weights.astype(np.int64))
    expanded_frame = frame.iloc[expanded_rows].reset_index(drop=True)
    expanded_y = y[expanded_rows]

    weighted_model, weighted_metrics = _fit_model(family, frame, y, weights)
    expanded_model, expanded_metrics = _fit_model(
        family,
        expanded_frame,
        expanded_y,
        None,
    )

    statsmodels_weighted = sm.GLM(
        y,
        design,
        family=_statsmodels_family(sm, family),
        freq_weights=weights,
    ).fit(
        maxiter=500,
        tol=1e-13,
        atol=1e-13,
        rtol=0.0,
        tol_criterion="params",
    )
    statsmodels_expanded = sm.GLM(
        expanded_y,
        design[expanded_rows],
        family=_statsmodels_family(sm, family),
    ).fit(
        maxiter=500,
        tol=1e-13,
        atol=1e-13,
        rtol=0.0,
        tol_criterion="params",
    )

    expected_covariance = _TRUE_PHI * np.linalg.inv(design.T @ (weights[:, None] * design))
    weighted_covariance = _full_covariance(weighted_metrics)
    expanded_covariance = _full_covariance(expanded_metrics)

    np.testing.assert_allclose(
        _parameters(weighted_model),
        _TRUE_BETA,
        rtol=0.0,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        _parameters(weighted_model),
        _parameters(expanded_model),
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        _parameters(weighted_model),
        statsmodels_weighted.params,
        rtol=2e-7,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        statsmodels_weighted.params,
        statsmodels_expanded.params,
        rtol=0.0,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        weighted_model.predict(frame),
        statsmodels_weighted.fittedvalues,
        rtol=8e-8,
        atol=5e-8,
    )

    assert weighted_metrics.deviance == pytest.approx(
        expanded_metrics.deviance,
        rel=0.0,
        abs=2e-13,
    )
    assert weighted_metrics.deviance == pytest.approx(
        statsmodels_weighted.deviance,
        rel=2e-12,
        abs=2e-12,
    )
    assert statsmodels_weighted.deviance == pytest.approx(
        statsmodels_expanded.deviance,
        rel=0.0,
        abs=2e-13,
    )

    assert weighted_model.result.phi == pytest.approx(
        _TRUE_PHI,
        rel=2e-7,
        abs=2e-10,
    )
    assert weighted_model.result.phi == pytest.approx(
        expanded_model.result.phi,
        rel=0.0,
        abs=2e-13,
    )
    assert weighted_model.result.phi == pytest.approx(
        statsmodels_weighted.scale,
        rel=2e-7,
        abs=2e-10,
    )
    assert statsmodels_weighted.scale == pytest.approx(
        statsmodels_expanded.scale,
        rel=0.0,
        abs=3e-15,
    )

    np.testing.assert_allclose(
        weighted_covariance,
        expected_covariance,
        rtol=3e-7,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        weighted_covariance,
        expanded_covariance,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        weighted_covariance,
        statsmodels_weighted.cov_params(),
        rtol=3e-7,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        statsmodels_weighted.cov_params(),
        statsmodels_expanded.cov_params(),
        rtol=3e-13,
        atol=3e-15,
    )


@pytest.mark.parametrize(
    ("family", "seed"),
    [
        pytest.param("gaussian", 21901, id="gaussian"),
        pytest.param("gamma", 21902, id="gamma"),
    ],
)
def test_literal_frequency_weight_wald_coverage_is_calibrated(
    family: str,
    seed: int,
) -> None:
    """A bounded seeded ensemble checks phi recovery and 95% Wald coverage.

    Each trial draws 640 iid rows from eight fixed ``(x, y)`` support points
    and stores their multinomial cell counts as frequency weights. Expanding
    those counts gives the realized iid sample exactly. This is deliberately
    not an EDM prior-weight fixture with ``Var(y_i) = phi * V(mu_i) / w_i``.

    The support has exactly the GLM conditional mean and variance, which are
    sufficient for Fisher/Wald calibration of these Gaussian and Gamma fits.
    """
    sm = pytest.importorskip("statsmodels.api")
    true_phi = 0.25
    true_beta = np.array([0.15, 0.3], dtype=np.float64)
    x = np.repeat(np.array([-1.5, -0.5, 0.5, 1.5]), 2)
    residual_sign = np.tile(np.array([-1.0, 1.0]), 4)
    eta = true_beta[0] + true_beta[1] * x
    if family == "gaussian":
        y = eta + np.sqrt(true_phi) * residual_sign
    else:
        y = np.exp(eta) * (1.0 + np.sqrt(true_phi) * residual_sign)
    frame = pd.DataFrame({"x": x})
    design = np.column_stack([np.ones(len(x)), x])

    rng = np.random.default_rng(seed)
    n_trials = 160
    covered = 0
    phi_estimates = np.empty(n_trials, dtype=np.float64)
    standardized_errors = np.empty(n_trials, dtype=np.float64)

    for trial in range(n_trials):
        weights = rng.multinomial(640, np.full(len(x), 1.0 / len(x))).astype(np.float64)
        model, metrics = _fit_model(family, frame, y, weights)
        estimate = float(model.result.beta[0])
        standard_error = float(metrics.coefficient_se["x"][0])
        z_score = (estimate - true_beta[1]) / standard_error

        standardized_errors[trial] = z_score
        phi_estimates[trial] = model.result.phi
        covered += int(abs(z_score) <= norm.ppf(0.975))

        if trial == 0:
            reference = sm.GLM(
                y,
                design,
                family=_statsmodels_family(sm, family),
                freq_weights=weights,
            ).fit(maxiter=500, tol=1e-12)
            np.testing.assert_allclose(
                _parameters(model),
                reference.params,
                rtol=2e-6,
                atol=2e-8,
            )
            assert model.result.phi == pytest.approx(
                reference.scale,
                rel=2e-6,
                abs=2e-8,
            )
            np.testing.assert_allclose(
                _standard_errors(model, metrics),
                reference.bse,
                rtol=2e-6,
                atol=2e-8,
            )

    coverage = covered / n_trials
    assert np.mean(phi_estimates) == pytest.approx(true_phi, rel=0.0, abs=0.015)
    assert abs(float(np.mean(standardized_errors))) < 0.25
    assert 0.8 < float(np.std(standardized_errors)) < 1.2
    assert 0.90 <= coverage <= 0.99
