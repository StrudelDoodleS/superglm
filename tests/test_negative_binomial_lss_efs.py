"""Public automatic-smoothing and Poisson-boundary behavior for NB2 LSS."""

from __future__ import annotations

import math
import warnings
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

import superglm.distributional.api as distributional_api_module
import superglm.distributional.families.negative_binomial as negative_binomial_family
from superglm import SuperLSS, cat, s
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import NegativeBinomialLS, Predictor
from superglm.distributional.curvature import (
    CurvaturePolicyError,
    RepeatedCurvatureIndefinitenessError,
)
from superglm.distributional.weights import WeightContract
from superglm.features import Spline
from superglm.types import LambdaPolicy
from tests.bound_predictor_fixtures import model_from_templates

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
    return model_from_templates(
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


def _fit_reml(
    model: SuperLSS,
    frame: pd.DataFrame,
    response: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> SuperLSS:
    return model.fit_reml(
        frame,
        response,
        sample_weight=sample_weight,
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


@pytest.mark.parametrize("dropped_rows", [0, 3])
def test_poisson_like_fit_finishes_and_discloses_the_boundary(dropped_rows: int) -> None:
    """Poisson-like theta non-identification is disclosed, not refused (it used to raise).

    The fit returns, its final curvature records the fallback it took, and the
    family's Poisson-boundary diagnosis is warned and kept in result_.warnings.
    Zero-weight rows are dropped before fitting; the diagnosis still reads the
    retained rows at their input positions.
    """
    frame, response = _poisson_like_fixture()
    assert float(np.var(response)) <= float(np.mean(response))
    weights = None
    if dropped_rows:
        frame = pd.concat([frame.iloc[:dropped_rows], frame], ignore_index=True)
        response = np.r_[np.full(dropped_rows, 40.0), response]
        weights = np.r_[np.zeros(dropped_rows), np.ones(len(response) - dropped_rows)]

    with pytest.warns(ConvergenceWarning, match="Poisson-like at the diagnostic boundary"):
        model = _fit_reml(_model(), frame, response, weights)

    telemetry = model._require_fitted().fit_state.solver_result.terminal_curvature
    assert telemetry.requested_source == "observed"
    assert telemetry.fallback_count >= 1
    assert telemetry.actual_source in ("fisher", "nearest_semidefinite")
    assert any(
        "could not establish a stable finite theta" in message for message in model.result_.warnings
    )
    assert np.all(np.isfinite(model.covariance_))


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
    model = model_from_templates(
        family=_RaisingDiagnosticNegativeBinomial(),
        predictors=(Predictor("mean", {}), Predictor("theta", {})),
    )

    with pytest.raises(RepeatedCurvatureIndefinitenessError) as caught:
        model.fit(pd.DataFrame(index=range(3)), np.ones(3), lambdas={})

    assert caught.value is original


def _claims_book_fixture(seed: int = 7, n_rows: int = 1500):
    """A small synthetic motor book shaped like freMTPL2: exposure offsets and sparse tails.

    Vehicle age has a geometric tail, exposure is short for new vehicles, and
    claims are rare, so on this seed master's terminal observed curvature stayed
    indefinite after its retry and the fit raised
    RepeatedCurvatureIndefinitenessError.
    """
    rng = np.random.default_rng(seed)
    vehicle_age = np.minimum(rng.geometric(0.12, n_rows) - 1, 99).astype(np.float64)
    driver_age = rng.uniform(18.0, 90.0, n_rows)
    bonus = np.minimum(50 + rng.geometric(0.15, n_rows) - 1, 230).astype(np.float64)
    exposure = np.clip(rng.beta(0.9, 0.7, n_rows), 0.0027, 1.0)
    exposure = np.where(vehicle_age == 0, exposure * 0.4, exposure).clip(0.0027, 1.0)
    brand = rng.choice([f"B{k}" for k in range(11)], n_rows, p=np.r_[np.full(10, 0.0995), 0.005])
    gas = rng.choice(["Diesel", "Regular"], n_rows)
    area = rng.choice(list("ABCDEF"), n_rows)
    eta = (
        -2.3
        + 0.2 * np.sin(driver_age / 15.0)
        - 0.02 * vehicle_age
        + 0.006 * (bonus - 50.0)
        + np.where(gas == "Diesel", 0.1, 0.0)
    )
    mean = exposure * np.exp(eta)
    size = exposure * np.exp(0.3)
    counts = rng.negative_binomial(size, size / (size + mean)).astype(np.float64)
    frame = pd.DataFrame(
        {
            "DrivAge": driver_age,
            "VehAge": vehicle_age,
            "BonusMalus": bonus,
            "VehBrand": brand,
            "VehGas": gas,
            "Area": area,
        }
    )
    return frame, counts, np.log(exposure)


def test_claims_book_shaped_fit_finishes_instead_of_refusing() -> None:
    """Kills the refusal on ordinary-looking data: the fit returns and discloses its fallback."""
    frame, counts, log_exposure = _claims_book_fixture()
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(
            s("DrivAge", kind="cr", k=10),
            s("VehAge", kind="cr", k=10),
            s("BonusMalus", kind="cr", k=10),
            cat("VehBrand"),
            cat("VehGas"),
            cat("Area"),
        ),
        family.theta(s("DrivAge", kind="cr", k=5), cat("VehGas"), cat("Area")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(frame, counts, offsets={"mean": log_exposure, "theta": log_exposure})

    fits = model._require_fitted().fit_state.smoothing.coefficient_fits
    assert any(fit.terminal_curvature.fallback_count > 0 for fit in fits)
    assert np.all(np.isfinite(model.covariance_))
    if not model.result_.converged:
        assert any(issubclass(item.category, ConvergenceWarning) for item in caught)
    for message in model.result_.warnings:
        assert any(str(item.message) == message for item in caught)
