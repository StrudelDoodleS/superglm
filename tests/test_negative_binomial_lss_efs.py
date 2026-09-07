"""Public automatic-smoothing and Poisson-boundary behavior for NB2 LSS."""

from __future__ import annotations

import math
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

import superglm.distributional as distributional_api
import superglm.distributional.api as distributional_api_module
import superglm.distributional.families.negative_binomial as negative_binomial_family
from superglm import SuperLSS
from superglm.distributional import NegativeBinomialLS, Predictor
from superglm.distributional.curvature import (
    CurvaturePolicyError,
    RepeatedCurvatureIndefinitenessError,
)
from superglm.distributional.weights import WeightContract
from superglm.features import Spline
from superglm.types import LambdaPolicy

_INITIAL_LAMBDAS = {
    "mean:x_mean#wiggle": 0.8,
    "theta:x_theta#wiggle": 1.1,
}


def _overdispersed_fixture(n_rows: int = 480) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2026083103)
    axis = np.linspace(-1.0, 1.0, n_rows)
    x_mean = rng.permutation(axis)
    x_theta = rng.permutation(axis)
    mean = np.exp(0.55 + 0.50 * np.sin(np.pi * x_mean) + 0.12 * x_mean)
    theta = np.exp(0.05 + 0.40 * np.cos(np.pi * x_theta) - 0.12 * x_theta)
    response = nbinom.rvs(
        n=theta,
        p=theta / (mean + theta),
        random_state=rng,
    ).astype(np.float64)
    return pd.DataFrame({"x_mean": x_mean, "x_theta": x_theta}), response


def _poisson_like_fixture(n_rows: int = 256) -> tuple[pd.DataFrame, np.ndarray]:
    probabilities = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows
    response = poisson.ppf(probabilities, mu=2.0).astype(np.float64)
    rng = np.random.default_rng(2026083106)
    axis = np.linspace(-1.0, 1.0, n_rows)
    frame = pd.DataFrame(
        {
            "x_mean": rng.permutation(axis),
            "x_theta": rng.permutation(axis),
        }
    )
    return frame, response[rng.permutation(n_rows)]


def _model() -> SuperLSS:
    return SuperLSS(
        family=NegativeBinomialLS(),
        predictors=tuple(
            Predictor(
                name,
                {
                    f"x_{name}": Spline(
                        kind="cr",
                        n_knots=5,
                        knot_strategy="quantile_rows",
                        lambda_policy=LambdaPolicy.estimate(),
                    )
                },
            )
            for name in ("mean", "theta")
        ),
    )


def _fit_reml(model: SuperLSS, frame: pd.DataFrame, response: np.ndarray) -> SuperLSS:
    return model.fit_reml(
        frame,
        response,
        lambdas=_INITIAL_LAMBDAS,
        max_reml_iter=100,
        reml_tol=1.0e-6,
        max_log_step=1.0,
        max_inner_iter=100,
        inner_tol=1.0e-8,
        practical_reml=False,
    )


def test_public_fit_reml_accepts_updates_and_publishes_observed_inference() -> None:
    frame, response = _overdispersed_fixture()
    model = _model()

    assert _fit_reml(model, frame, response) is model

    result = model.result_
    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None
    assert any(iteration.accepted for iteration in smoothing.history)
    assert result.smoothing_converged is True
    assert smoothing.matched_certified is True
    assert result.n_smoothing_iter > 0
    assert np.all(np.isfinite(model.covariance_))
    assert np.isfinite(result.total_effective_df)
    assert all(np.isfinite(value) for value in result.predictor_edf.values())

    terminal = fitted.result
    assert terminal.terminal_curvature.requested_source == "observed"
    assert terminal.terminal_curvature.actual_source == "observed"
    assert terminal.terminal_curvature.fallback_count == 0
    assert np.all(np.isfinite(terminal.terminal_data_curvature))
    assert np.all(np.isfinite(terminal.terminal_penalized_curvature))
    reconstructed = terminal.terminal_data_curvature + terminal.penalty
    residual_norm = float(
        np.linalg.norm(terminal.terminal_penalized_curvature - reconstructed, ord=np.inf)
    )
    dimension = terminal.terminal_penalized_curvature.shape[0]
    scale = max(
        float(np.linalg.norm(terminal.terminal_penalized_curvature, ord=np.inf)),
        float(np.linalg.norm(terminal.terminal_data_curvature, ord=np.inf))
        + float(np.linalg.norm(terminal.penalty, ord=np.inf)),
        np.finfo(np.float64).tiny,
    )
    bound = 8.0 * dimension * np.finfo(np.float64).eps * scale
    assert residual_norm <= bound
    assert result.rank == terminal.terminal_rank.rank == len(result.coefficients)


def test_poisson_like_fit_has_a_typed_public_boundary_diagnostic() -> None:
    """Poisson-like theta non-identification is not EFS-lambda convergence."""

    frame, response = _poisson_like_fixture()
    assert float(np.var(response)) <= float(np.mean(response))

    boundary_error = getattr(distributional_api, "NegativeBinomialPoissonBoundaryError")
    with pytest.raises(boundary_error) as caught:
        _fit_reml(_model(), frame, response)

    assert caught.type is boundary_error
    assert getattr(negative_binomial_family, "NegativeBinomialPoissonBoundaryError") is caught.type
    assert type(caught.value.__cause__) is RepeatedCurvatureIndefinitenessError
    message = str(caught.value).lower()
    assert "could not establish a stable finite theta" in message
    assert "poisson-like at the diagnostic boundary" in message
    assert "smoothing parameter" not in message


def test_finite_overdispersion_does_not_classify_a_curvature_failure_as_poisson_like() -> None:
    response = np.array([0.0, 0.0, 1.0, 2.0, 4.0, 8.0])
    assert float(np.var(response)) > float(np.mean(response))

    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        sample_weight=None,
        weight_contract=WeightContract("prior"),
        failure=RepeatedCurvatureIndefinitenessError("synthetic repeated failure"),
    )

    assert result is None


def test_nb2_diagnosis_drops_zero_weight_rows_before_response_validation() -> None:
    response = np.array([0.5, 1.0, 1.0, 1.0])
    sample_weight = np.array([0.0, 1.0, 1.0, 1.0])

    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        sample_weight=sample_weight,
        weight_contract=WeightContract("prior"),
        failure=RepeatedCurvatureIndefinitenessError("synthetic repeated failure"),
    )

    assert type(result) is negative_binomial_family.NegativeBinomialPoissonBoundaryError


def test_nb2_diagnosis_keeps_the_exact_failure_guard() -> None:
    response = np.array([0.5, 1.0, 1.0, 1.0])
    sample_weight = np.array([0.0, 1.0, 1.0, 1.0])
    arguments = {
        "sample_weight": sample_weight,
        "weight_contract": WeightContract("prior"),
    }

    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        failure=CurvaturePolicyError("synthetic base-class curvature failure"),
        **arguments,
    )
    assert result is None

    class DerivedRepeatedCurvatureIndefinitenessError(RepeatedCurvatureIndefinitenessError):
        pass

    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        failure=DerivedRepeatedCurvatureIndefinitenessError("synthetic derived failure"),
        **arguments,
    )
    assert result is None


def test_nb2_diagnosis_keeps_the_exact_family_guard() -> None:
    class DerivedNegativeBinomialLS(NegativeBinomialLS):
        pass

    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        DerivedNegativeBinomialLS(),
        np.array([0.5, 1.0, 1.0, 1.0]),
        sample_weight=np.array([0.0, 1.0, 1.0, 1.0]),
        weight_contract=WeightContract("prior"),
        failure=RepeatedCurvatureIndefinitenessError("synthetic repeated failure"),
    )

    assert result is None


def test_prior_moment_direction_does_not_override_positive_poisson_face_direction() -> None:
    response = np.array([5.5, 2.0])
    sample_weight = np.array([2.0, 1.0])
    initial_mean = 13.0 / 3.0
    moment_direction = math.fsum(
        float(weight * (value - initial_mean) ** 2 - initial_mean)
        for value, weight in zip(response, sample_weight, strict=True)
    )
    boundary_direction = math.fsum(
        float(weight * (value - initial_mean) ** 2 - value)
        for value, weight in zip(response, sample_weight, strict=True)
    )

    np.testing.assert_array_equal(sample_weight * response, [11.0, 2.0])
    assert moment_direction == pytest.approx(-0.5)
    assert boundary_direction == pytest.approx(2.0 / 3.0)
    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        sample_weight=sample_weight,
        weight_contract=WeightContract("prior"),
        failure=RepeatedCurvatureIndefinitenessError("synthetic repeated failure"),
    )

    assert result is None


def test_cancellation_limited_frequency_moment_direction_fails_closed() -> None:
    large_weight = 2**52
    response = np.array([0.0, 2.0])
    sample_weight = np.array([large_weight, large_weight - 1], dtype=np.float64)
    exact_mean = Fraction(2 * (large_weight - 1), 2 * large_weight - 1)
    exact_direction = large_weight * ((Fraction(0) - exact_mean) ** 2 - exact_mean) + (
        large_weight - 1
    ) * ((Fraction(2) - exact_mean) ** 2 - exact_mean)
    exact_numerator = (2 * large_weight - 1) * exact_mean**2

    assert exact_direction > 0
    assert exact_numerator / exact_direction == 2 * large_weight - 2
    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        sample_weight=sample_weight,
        weight_contract=WeightContract("frequency"),
        failure=RepeatedCurvatureIndefinitenessError("synthetic repeated failure"),
    )
    assert result is None

    result = distributional_api_module._diagnosed_repeated_curvature_failure(
        NegativeBinomialLS(),
        response,
        sample_weight=np.ones(2),
        weight_contract=WeightContract("frequency"),
        failure=RepeatedCurvatureIndefinitenessError("synthetic repeated failure"),
    )
    assert result is None


class _RaisingDiagnosticNegativeBinomial(NegativeBinomialLS):
    def diagnose_repeated_curvature_failure(self, y, weights):
        raise RuntimeError("sentinel diagnostic failure")


def test_diagnostic_failure_preserves_the_original_curvature_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RepeatedCurvatureIndefinitenessError("original repeated failure")
    monkeypatch.setattr(
        distributional_api_module,
        "fit_dense_distributional",
        Mock(side_effect=original),
    )
    model = SuperLSS(
        family=_RaisingDiagnosticNegativeBinomial(),
        predictors=(Predictor("mean", {}), Predictor("theta", {})),
    )

    with pytest.raises(RepeatedCurvatureIndefinitenessError) as caught:
        model.fit(pd.DataFrame(index=range(3)), np.ones(3), lambdas={})

    assert caught.value is original
