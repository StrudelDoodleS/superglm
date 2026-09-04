from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, dataclass, replace

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special

import superglm.distributional.efs as efs_module
import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.loop as smoothing_loop
import superglm.distributional.solver.chunks as chunking
from superglm._frame import as_eager_frame
from superglm.distributional.derivatives import (
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.families.gamma import (
    GammaInitializationError,
    GammaLikelihoodPlan,
    GammaLS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    ExpectedInformationFamily,
    LikelihoodPlanValidatingFamily,
    validate_family,
)
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
)
from superglm.distributional.weights import (
    LikelihoodWeightError,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import RandomEffect, Spline
from superglm.links import LogLink
from tests._distributional_family_kernels import gamma as gamma_kernel
from tests._gamma_lss_oracles import (
    finite_difference_error_bounds,
    finite_difference_row_derivatives,
    gamma_asymptotic_oracle,
    gamma_row_reference,
    gamma_scaled_ratio_oracle,
)

_gamma_initial_target = gamma_kernel._gamma_initial_target
_gamma_log_normalizer = gamma_kernel._gamma_log_normalizer
_scaled_digamma_residual = gamma_kernel._scaled_digamma_residual
_scaled_ratio_terms = gamma_kernel._scaled_ratio_terms
_scaled_trigamma_residual = gamma_kernel._scaled_trigamma_residual
_shape_from_scale = gamma_kernel._shape_from_scale

Y = np.array([0.17, 0.9, 2.4, 8.0])
MEAN = np.array([0.3, 1.2, 1.7, 5.0])
SCALE = np.array([0.18, 0.55, 1.1, 2.0])
PRIOR_WEIGHTS = np.array([0.35, 1.0, 2.75, 7.0])
FREQUENCY_WEIGHTS = np.array([1.0, 2.0, 3.0, 5.0])


def _resolved(weights: np.ndarray, semantics: str):
    return resolve_likelihood_weights(
        weights,
        n_observations=len(weights),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )


def _plan(family: GammaLS, y: np.ndarray, weights: np.ndarray, semantics: str):
    return family.bind_likelihood(
        y,
        _resolved(weights, semantics),
        COMPLETE_OBSERVATION,
    )


def _gamma_chunk_problem():
    family = GammaLS()
    resolved = _resolved(PRIOR_WEIGHTS, "prior")
    frame = as_eager_frame(pd.DataFrame({"row": np.arange(len(Y), dtype=np.float64)}))
    compiled = compile_predictors(
        frame,
        resolved,
        family.parameters,
        (Predictor("mean", {}), Predictor("scale", {})),
    )
    layout = build_stacked_layout(compiled)
    plan = family.bind_likelihood(Y, resolved, COMPLETE_OBSERVATION)
    return family, layout, plan, np.zeros(layout.n_coefficients, dtype=np.float64)


@pytest.mark.parametrize(
    ("semantics", "weights"),
    [("prior", PRIOR_WEIGHTS), ("frequency", FREQUENCY_WEIGHTS)],
)
def test_normalized_rows_and_natural_derivatives_match_independent_oracles(
    semantics: str, weights: np.ndarray
) -> None:
    """Kills wrong normalization, shape, multiplier, or natural derivative formulas."""

    family = GammaLS()
    theta = np.column_stack((MEAN, SCALE))
    actual = family.evaluate_natural(Y, theta, _plan(family, Y, weights, semantics))
    expected_rows = gamma_row_reference(Y, MEAN, SCALE, weights, semantics)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        actual.reported_log_likelihood, expected_rows, rtol=2e-13, atol=2e-13
    )

    for row in range(len(Y)):
        expected_score, expected_hessian = finite_difference_row_derivatives(
            Y[row],
            MEAN[row],
            SCALE[row],
            weights[row],
            semantics,  # type: ignore[arg-type]
            step_scale=0.5,
        )
        score_bound, hessian_bound = finite_difference_error_bounds(
            Y[row],
            MEAN[row],
            SCALE[row],
            weights[row],
            semantics,  # type: ignore[arg-type]
        )
        np.testing.assert_array_less(np.abs(actual.score[row] - expected_score), score_bound)
        np.testing.assert_array_less(
            np.abs(actual.hessian_packed[row] - expected_hessian), hessian_bound
        )


@pytest.mark.parametrize(
    ("semantics", "weights"),
    [("prior", PRIOR_WEIGHTS), ("frequency", FREQUENCY_WEIGHTS)],
)
def test_gamma_returns_the_exact_requested_derivative_order(
    semantics: str,
    weights: np.ndarray,
) -> None:
    """Kills ignored orders and dummy unrequested Gamma derivative arrays."""

    family = GammaLS()
    theta = np.column_stack((MEAN, SCALE))
    plan = _plan(family, Y, weights, semantics)
    full = family.evaluate_natural(Y, theta, plan, derivative_order=2)
    score_only = family.evaluate_natural(Y, theta, plan, derivative_order=1)
    value_only = family.evaluate_natural(Y, theta, plan, derivative_order=0)

    assert (value_only.derivative_order, score_only.derivative_order, full.derivative_order) == (
        0,
        1,
        2,
    )
    assert value_only.score is None and value_only.hessian_packed is None
    assert score_only.score is not None and score_only.hessian_packed is None
    np.testing.assert_array_equal(
        value_only.optimizing_log_likelihood, full.optimizing_log_likelihood
    )
    np.testing.assert_array_equal(
        value_only.parameter_independent_carrier, full.parameter_independent_carrier
    )
    np.testing.assert_array_equal(value_only.valid, full.valid)
    np.testing.assert_array_equal(
        score_only.optimizing_log_likelihood, full.optimizing_log_likelihood
    )
    np.testing.assert_array_equal(
        score_only.parameter_independent_carrier, full.parameter_independent_carrier
    )
    np.testing.assert_array_equal(score_only.score, full.score)
    np.testing.assert_array_equal(score_only.valid, full.valid)


def test_gamma_primitive_kernel_repeats_and_matches_every_public_entry_point() -> None:
    family = GammaLS()
    response = np.array([0.5, 2.0])
    theta = np.array([[0.8, 0.7], [1.5, 1.2]])
    plan = _plan(family, response, np.ones(2), "prior")
    expected_initial = family.initialize(response, plan).theta
    expected_evaluation = family.evaluate_natural(response, theta, plan)
    expected_information = family.expected_information_natural(theta, plan)

    for _ in range(2):
        actual_initial = gamma_kernel.initialize_gamma(
            response,
            plan.weights.values,
            "prior",
        )
        actual_evaluation = gamma_kernel.evaluate_gamma_rows(
            response,
            theta[:, 0],
            theta[:, 1],
            plan.weights.values,
            "prior",
            derivative_order=2,
        )
        actual_information = gamma_kernel.gamma_expected_information(
            theta[:, 0],
            theta[:, 1],
            plan.weights.values,
            "prior",
        )

        np.testing.assert_equal(actual_initial, expected_initial)
        np.testing.assert_equal(
            actual_evaluation.optimizing_log_likelihood,
            expected_evaluation.optimizing_log_likelihood,
        )
        np.testing.assert_equal(actual_evaluation.score, expected_evaluation.score)
        np.testing.assert_equal(
            actual_evaluation.hessian_packed,
            expected_evaluation.hessian_packed,
        )
        np.testing.assert_equal(actual_evaluation.valid, expected_evaluation.valid)
        np.testing.assert_equal(actual_information, expected_information)


def _call_gamma_primitive_entry_point(entry_point: str, vector: object) -> None:
    mean = np.array([0.8, 1.5], dtype=np.float64)
    scale = np.array([0.7, 1.2], dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    if entry_point == "initialize":
        gamma_kernel.initialize_gamma(vector, weights, "prior")
    elif entry_point == "evaluate":
        gamma_kernel.evaluate_gamma_rows(
            vector,
            mean,
            scale,
            weights,
            "prior",
            derivative_order=2,
        )
    elif entry_point == "information":
        gamma_kernel.gamma_expected_information(vector, scale, weights, "prior")
    else:
        gamma_kernel.gamma_predictor_curvature_directional(
            vector,
            np.log(np.column_stack((mean, scale))),
            np.ones((2, 2), dtype=np.float64),
            weights,
            "prior",
        )


@pytest.mark.parametrize(
    ("mutation", "vector"),
    [
        ("non-array", [0.5, 1.5]),
        ("string", np.array(["0.5", "1.5"])),
        ("integer", np.array([1, 2], dtype=np.int64)),
        ("float32", np.array([0.5, 1.5], dtype=np.float32)),
        ("boolean", np.array([True, True])),
        ("non-vector", np.array([[0.5, 1.5]], dtype=np.float64)),
        ("nonfinite", np.array([0.5, np.nan], dtype=np.float64)),
    ],
)
@pytest.mark.parametrize("entry_point", ["initialize", "evaluate", "information", "directional"])
def test_gamma_primitive_entry_points_refuse_vector_coercion(
    entry_point: str,
    mutation: str,
    vector: object,
) -> None:
    """Kills coercion of non-literal float64 vectors at every primitive boundary."""

    del mutation
    with pytest.raises(ValueError, match="float64"):
        _call_gamma_primitive_entry_point(entry_point, vector)


def test_gamma_kernel_evaluation_owns_and_freezes_every_array_field() -> None:
    """Kills writable or caller-aliased arrays in the Gamma kernel record."""

    sources = {
        "optimizing_log_likelihood": np.array([-0.5, -1.5], dtype=np.float64),
        "score": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "hessian_packed": np.array([[-1.0, 0.25, -2.0], [-3.0, 0.5, -4.0]]),
        "valid": np.ones(2, dtype=bool),
    }
    expected = {name: values.copy() for name, values in sources.items()}
    evaluation = gamma_kernel.GammaKernelEvaluation(**sources)

    for values in sources.values():
        values[...] = 0

    for name, expected_values in expected.items():
        actual = getattr(evaluation, name)
        np.testing.assert_array_equal(actual, expected_values)
        assert actual.flags.owndata
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.flat[0] = 0


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("optimizing_log_likelihood", np.ones((2, 1), dtype=np.float64)),
        ("optimizing_log_likelihood", np.array([0.0, np.nan], dtype=np.float64)),
        ("score", np.ones((2, 1), dtype=np.float64)),
        ("score", np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float64)),
        ("hessian_packed", np.ones((2, 2), dtype=np.float64)),
        (
            "hessian_packed",
            np.array([[0.0, 0.0, np.inf], [0.0, 0.0, 0.0]], dtype=np.float64),
        ),
        ("valid", np.ones((2, 1), dtype=bool)),
        ("valid", np.ones(2, dtype=np.float64)),
    ],
)
def test_gamma_kernel_evaluation_rejects_malformed_array_fields(
    field: str,
    malformed: np.ndarray,
) -> None:
    """Kills Gamma kernel records that do not certify result shape and finiteness."""

    fields = {
        "optimizing_log_likelihood": np.zeros(2, dtype=np.float64),
        "score": np.zeros((2, 2), dtype=np.float64),
        "hessian_packed": np.zeros((2, 3), dtype=np.float64),
        "valid": np.ones(2, dtype=bool),
    }
    fields[field] = malformed

    with pytest.raises(ValueError, match=field):
        gamma_kernel.GammaKernelEvaluation(**fields)


def test_gamma_lower_orders_skip_unrequested_special_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills lower-order routes that compute and discard A(a) or J(a)."""

    family = GammaLS()
    theta = np.column_stack((MEAN, SCALE))
    plan = _plan(family, Y, PRIOR_WEIGHTS, "prior")

    def explode(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unrequested Gamma special-function work executed")

    monkeypatch.setattr(gamma_kernel, "_scaled_digamma_residual", explode)
    monkeypatch.setattr(gamma_kernel, "_scaled_trigamma_residual", explode)
    assert family.evaluate_natural(Y, theta, plan, derivative_order=0).derivative_order == 0


def test_gamma_order_one_skips_trigamma(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills score-only evaluation that still computes J(a) and a Hessian."""

    family = GammaLS()
    theta = np.column_stack((MEAN, SCALE))
    plan = _plan(family, Y, PRIOR_WEIGHTS, "prior")

    def explode(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unrequested Gamma trigamma work executed")

    monkeypatch.setattr(gamma_kernel, "_scaled_trigamma_residual", explode)
    assert family.evaluate_natural(Y, theta, plan, derivative_order=1).derivative_order == 1


def test_all_unit_prior_and_frequency_rows_are_the_same_law() -> None:
    """Kills a semantic branch that changes unit-weight Gamma rows."""

    family = GammaLS()
    weights = np.ones(len(Y))
    theta = np.column_stack((MEAN, SCALE))
    prior = family.evaluate_natural(Y, theta, _plan(family, Y, weights, "prior"))
    frequency = family.evaluate_natural(Y, theta, _plan(family, Y, weights, "frequency"))
    prior_information = family.expected_information_natural(
        theta, _plan(family, Y, weights, "prior")
    )
    frequency_information = family.expected_information_natural(
        theta, _plan(family, Y, weights, "frequency")
    )
    for left, right in (
        (prior.reported_log_likelihood, frequency.reported_log_likelihood),
        (prior.score, frequency.score),
        (prior.hessian_packed, frequency.hessian_packed),
        (prior_information, frequency_information),
    ):
        np.testing.assert_array_equal(left, right)


def test_integer_frequency_rows_equal_literal_replication() -> None:
    """Kills frequency handling that changes parameters instead of replicating rows."""

    family = GammaLS()
    weights = np.array([2.0, 3.0, 4.0, 5.0])
    theta = np.column_stack((MEAN, SCALE))
    compressed_plan = _plan(family, Y, weights, "frequency")
    compressed = family.evaluate_natural(Y, theta, compressed_plan)
    compressed_fisher = family.expected_information_natural(theta, compressed_plan)

    expanded_y = np.repeat(Y, weights.astype(int))
    expanded_theta = np.repeat(theta, weights.astype(int), axis=0)
    expanded_plan = _plan(family, expanded_y, np.ones(len(expanded_y)), "frequency")
    expanded = family.evaluate_natural(expanded_y, expanded_theta, expanded_plan)
    expanded_fisher = family.expected_information_natural(expanded_theta, expanded_plan)
    starts = np.concatenate(([0], np.cumsum(weights.astype(int))[:-1]))
    for compressed_rows, expanded_rows in (
        (compressed.reported_log_likelihood, expanded.reported_log_likelihood),
        (compressed.score, expanded.score),
        (compressed.hessian_packed, expanded.hessian_packed),
        (compressed_fisher, expanded_fisher),
    ):
        reduced = np.add.reduceat(expanded_rows, starts, axis=0)
        np.testing.assert_allclose(compressed_rows, reduced, rtol=2e-15, atol=2e-15)


def test_unequal_prior_weights_are_not_literal_replication() -> None:
    """Kills multiplying the complete unit Gamma law by prior weight."""

    family = GammaLS()
    theta = np.column_stack((MEAN, SCALE))
    prior = family.evaluate_natural(Y, theta, _plan(family, Y, PRIOR_WEIGHTS, "prior"))
    replicated = gamma_row_reference(Y, MEAN, SCALE, PRIOR_WEIGHTS, "frequency")
    assert not np.allclose(prior.reported_log_likelihood, replicated)


def test_prior_and_frequency_scale_scores_differ_at_unit_state() -> None:
    """Kills the complete-unit-density-times-w prior mutation in the scale score."""

    family = GammaLS()
    y = np.array([1.0])
    theta = np.array([[1.0, 1.0]])
    weight = np.array([4.0])
    prior = family.evaluate_natural(y, theta, _plan(family, y, weight, "prior"))
    frequency = family.evaluate_natural(y, theta, _plan(family, y, weight, "frequency"))
    assert prior.score[0, 1] != frequency.score[0, 1]


def test_fisher_cross_is_structural_zero_and_prior_scale_information_is_nonlinear() -> None:
    """Kills non-orthogonal Fisher and w-times-unit prior scale information."""

    family = GammaLS()
    theta = np.column_stack((MEAN, SCALE))
    prior_plan = _plan(family, Y, PRIOR_WEIGHTS, "prior")
    information = family.expected_information_natural(theta, prior_plan)
    shape = PRIOR_WEIGHTS / SCALE**2
    expected_scale = 4.0 * shape * (shape * special.polygamma(1, shape) - 1.0) / SCALE**2
    unit_j = (1.0 / SCALE**2) * (special.polygamma(1, 1.0 / SCALE**2) / SCALE**2 - 1.0)
    wrong_scale = 4.0 * PRIOR_WEIGHTS * unit_j / SCALE**2
    np.testing.assert_array_equal(information[:, 1], np.zeros(len(Y)))
    np.testing.assert_allclose(information[:, 2], expected_scale, rtol=3e-14)
    assert not np.allclose(information[:, 2], wrong_scale)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_gamma_predictor_curvature_directional_derivative_matches_independent_secants(
    semantics: str,
) -> None:
    """The endpoint adapter differentiates the observed predictor curvature."""

    family = GammaLS()
    weights = PRIOR_WEIGHTS if semantics == "prior" else FREQUENCY_WEIGHTS
    plan = _plan(family, Y, weights, semantics)
    eta = np.log(np.column_stack((MEAN, SCALE)))
    direction = np.array(
        [
            [0.25, -0.4],
            [-0.3, 0.15],
            [0.5, 0.2],
            [-0.1, -0.35],
        ],
        dtype=np.float64,
    )
    links = (LogLink(), LogLink())

    actual = family.predictor_curvature_directional_derivative(
        Y,
        eta,
        direction,
        links,
        plan,
    )

    def curvature(step: float) -> np.ndarray:
        candidate_eta = eta + step * direction
        candidate_theta = np.exp(candidate_eta)
        natural = family.evaluate_natural(Y, candidate_theta, plan, derivative_order=2)
        return transform_natural_derivatives(
            natural,
            candidate_eta,
            links,
        ).curvature_packed

    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        secant = (curvature(step) - curvature(-step)) / (2.0 * step)
        errors.append(float(np.max(np.abs(actual - secant))))
    assert errors[2] < errors[1] < errors[0]
    scale = max(float(np.max(np.abs(actual))), 1.0)
    assert errors[-1] <= 2.0e-7 * scale


def test_gamma_scale_curvature_direction_uses_shape_curvature_derivative() -> None:
    """Kills treating the trigamma residual as constant along the scale direction."""

    family = GammaLS()
    y = np.array([0.7, 1.3, 3.1])
    mean = np.array([0.9, 1.1, 2.4])
    scale = np.array([0.2, 0.8, 1.7])
    weights = np.array([0.4, 1.5, 6.0])
    eta = np.log(np.column_stack((mean, scale)))
    direction = np.column_stack((np.zeros(3), np.ones(3)))
    plan = _plan(family, y, weights, "prior")

    derivative = family.predictor_curvature_directional_derivative(
        y,
        eta,
        direction,
        (LogLink(), LogLink()),
        plan,
    )
    shape = weights / scale**2
    multiplier = np.ones(3)
    _, at, ad = _scaled_ratio_terms(y, mean, shape, derivative_order=2)
    assert at is not None
    b = _scaled_digamma_residual(shape) + ad
    j = _scaled_trigamma_residual(shape)
    wrong_without_shape_derivative = 4.0 * multiplier * (-2.0 * (b + j))

    assert not np.allclose(derivative[:, 2], wrong_without_shape_derivative)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_initialization_scale_refinement_improves_semantic_moment_seed(
    semantics: str,
) -> None:
    """Kills skipping the cheap refinement on an ordinary conditioned start."""

    family = GammaLS()
    y = np.array([0.4, 0.9, 2.3, 7.0])
    weights = np.array([0.35, 1.0, 2.75, 7.0]) if semantics == "prior" else np.array([1, 2, 3, 5])
    plan = _plan(family, y, weights, semantics)
    initialized = family.initialize(y, plan)
    mu0 = float(np.dot(weights / np.sum(weights), y))
    denominator = len(y) if semantics == "prior" else float(np.sum(weights))
    variance = float(np.dot(weights, ((y - mu0) / np.max(abs(y - mu0))) ** 2))
    variance *= float(np.max(abs(y - mu0))) ** 2 / denominator
    seed = np.sqrt(variance) / mu0
    sigma0 = float(initialized.theta[0, 1])

    def intercept_target(rho: float) -> float:
        k = np.exp(rho)
        shape = weights * k if semantics == "prior" else np.full(len(y), k)
        multiplier = np.ones(len(y)) if semantics == "prior" else weights
        return float(
            np.sum(
                multiplier
                * shape
                * (special.digamma(shape) - np.log(shape) + y / mu0 - 1.0 - np.log(y / mu0))
            )
        )

    optimum_rho = optimize.brentq(intercept_target, -20.0, 20.0)
    initialized_rho = -2.0 * np.log(sigma0)
    seed_rho = -2.0 * np.log(seed)
    rho_tolerance = 8.0 * np.sqrt(np.finfo(np.float64).eps) * max(1.0, abs(float(optimum_rho)))
    np.testing.assert_allclose(initialized.theta[:, 0], mu0, rtol=2e-15)
    np.testing.assert_allclose(initialized.theta[:, 1], sigma0, rtol=0.0, atol=0.0)
    assert abs(initialized_rho - optimum_rho) <= rho_tolerance
    assert abs(initialized_rho - optimum_rho) < abs(seed_rho - optimum_rho)


@pytest.mark.parametrize(
    ("y", "weights", "semantics"),
    [
        (np.array([0.4, 0.9, 2.3, 7.0]), np.array([0.35, 1.0, 2.75, 7.0]), "prior"),
        (
            np.array([2.7875185304300627e-282, 1.1871015142550953e-199]),
            np.array([9.4454689048755826e-104, 1.3364568025851005e38]),
            "prior",
        ),
        (
            np.array([5.6854377200949004e-21, 1676910765542.104]),
            np.array([3.0, 2174.0]),
            "frequency",
        ),
    ],
    ids=("ordinary-prior", "extreme-prior", "extreme-frequency"),
)
def test_initial_target_requests_only_the_ratio_value_channel(
    monkeypatch: pytest.MonkeyPatch,
    y: np.ndarray,
    weights: np.ndarray,
    semantics: str,
) -> None:
    """Kills allocating unused ratio derivatives or touching J in initialization."""

    mean = float(np.dot(weights / np.sum(weights), y))
    legacy_target = _gamma_initial_target(y, mean, weights, semantics, 0.0)
    assert legacy_target is not None
    original_ratio = gamma_kernel._scaled_ratio_terms
    requested_orders: list[int] = []

    def audited_ratio(y, mean, shape, *, derivative_order=2):
        requested_orders.append(derivative_order)
        if derivative_order != 0:
            raise AssertionError("Gamma initialization requested unused ratio derivatives")
        value_only = original_ratio(y, mean, shape, derivative_order=0)
        full = original_ratio(y, mean, shape, derivative_order=2)
        assert value_only[0] is None and value_only[1] is None
        np.testing.assert_array_equal(value_only[2], full[2])
        return value_only

    def unexpected_trigamma(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Gamma initialization touched J")

    monkeypatch.setattr(gamma_kernel, "_scaled_ratio_terms", audited_ratio)
    monkeypatch.setattr(gamma_kernel, "_scaled_trigamma_residual", unexpected_trigamma)
    target = _gamma_initial_target(y, mean, weights, semantics, 0.0)

    assert requested_orders == [0]
    assert target == legacy_target


def _assert_executable_initial_state(
    y: np.ndarray,
    weights: np.ndarray,
    semantics: str,
) -> np.ndarray:
    family = GammaLS()
    plan = _plan(family, y, weights, semantics)
    initialized = family.initialize(y, plan)
    evaluation = family.evaluate_natural(y, initialized.theta, plan)
    information = family.expected_information_natural(initialized.theta, plan)
    assert np.all(np.isfinite(initialized.theta))
    assert np.all(initialized.theta > 0.0)
    assert np.all(np.isfinite(evaluation.optimizing_log_likelihood))
    assert np.all(np.isfinite(evaluation.score))
    assert np.all(np.isfinite(evaluation.hessian_packed))
    assert np.all(np.isfinite(information))
    assert np.all(information[:, (0, 2)] > 0.0)
    return initialized.theta


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_tiny_constant_initialization_searches_toward_smaller_shape(semantics: str) -> None:
    """Kills searching only toward larger shapes after the initial guard fails."""

    response = np.array([1e-300])
    theta = _assert_executable_initial_state(response, np.ones(1), semantics)
    np.testing.assert_array_equal(theta[:, 0], response)


def test_initialization_searches_beyond_nonexecutable_refined_and_moment_states() -> None:
    """Kills refusing after only the refined and moment Gamma states fail."""

    family = GammaLS()
    y = np.array([2.7875185304300627e-282, 1.1871015142550953e-199])
    weights = np.array([9.4454689048755826e-104, 1.3364568025851005e38])
    plan = _plan(family, y, weights, "prior")
    mean = float(np.dot(weights / np.sum(weights), y))
    residual = y - mean
    residual_scale = float(np.max(np.abs(residual)))
    scaled_ss = float(np.dot(weights, (residual / residual_scale) ** 2))
    log_seed = (
        math.log(residual_scale) + 0.5 * (math.log(scaled_ss) - math.log(len(y))) - math.log(mean)
    )
    moment_rho = -2.0 * log_seed

    def independent_target(rho: float) -> float:
        shape = weights * math.exp(rho)
        ratio = y / mean
        large_inverse = 1.0 / shape[1]
        a_residual = np.array(
            [
                shape[0] * (special.digamma(shape[0]) - math.log(shape[0])),
                -0.5 - large_inverse / 12.0 + large_inverse**3 / 120.0,
            ]
        )
        return math.fsum(a_residual + shape * (ratio - 1.0 - np.log(ratio)))

    refined_rho = optimize.brentq(independent_target, moment_rho - 64.0, moment_rho)
    for rho in (refined_rho, moment_rho):
        theta = np.column_stack((np.full(len(y), mean), np.full(len(y), math.exp(-0.5 * rho))))
        with pytest.raises(ValueError):
            family.evaluate_natural(y, theta, plan, derivative_order=2)

    initialized = family.initialize(y, plan)
    evaluation = family.evaluate_natural(y, initialized.theta, plan, derivative_order=2)
    information = family.expected_information_natural(initialized.theta, plan)
    assert np.all(np.isfinite(initialized.theta))
    assert np.all(initialized.theta > 0.0)
    assert np.all(np.isfinite(evaluation.optimizing_log_likelihood))
    assert np.all(np.isfinite(evaluation.score))
    assert np.all(np.isfinite(evaluation.hessian_packed))
    assert np.all(np.isfinite(information))
    assert np.all(information[:, (0, 2)] > 0.0)


def test_initialization_accepts_state_when_float_target_sign_is_ambiguous() -> None:
    """Kills treating a rounded heuristic-target sign as initialization authority."""

    _assert_executable_initial_state(
        np.array([3.0737851905211083e-08, 275.4184541132178]),
        np.array([0.16196822028308927, 22.394820206868346]),
        "prior",
    )


def test_initialization_accepts_refined_state_for_extreme_frequency_counts() -> None:
    """Kills refusing an executable refined state for extreme frequency counts."""

    # The review labeled fractional masses as frequency weights, which the
    # public contract forbids. These exact counts preserve its adversarial
    # response and nearby weight ratio on the real likelihood path.
    _assert_executable_initial_state(
        np.array([5.6854377200949004e-21, 1676910765542.104]),
        np.array([3, 2174]),
        "frequency",
    )


def test_exact_constant_response_preserves_mean_and_uses_only_an_initial_guard() -> None:
    """Kills weighted-mean rounding of constants and a zero/fitted scale floor."""

    family = GammaLS()
    constant = np.float64(0.1)
    y = np.full(4, constant)
    weights = np.array([0.35, 1.0, 2.75, 7.0])
    initialized = family.initialize(y, _plan(family, y, weights, "prior"))
    np.testing.assert_array_equal(initialized.theta[:, 0], y)
    assert np.all(initialized.theta[:, 1] > 0.0)
    tiny_theta = np.column_stack((y, np.full(len(y), np.nextafter(0.0, 1.0))))
    # There is no configured family floor; refusal, if any, is only numerical
    # representability of the derived shape.
    with pytest.raises(ValueError, match="representable|shape"):
        family.evaluate_natural(y, tiny_theta, _plan(family, y, weights, "prior"))


def test_constant_response_guard_certifies_channels_or_raises_typed_refusal() -> None:
    """Kills an arbitrary constant-response guard that underflows required channels."""

    family = GammaLS()
    response = np.array([1e300])
    ordinary_plan = _plan(family, response, np.ones(1), "prior")
    initialized = family.initialize(response, ordinary_plan)
    evaluation = family.evaluate_natural(response, initialized.theta, ordinary_plan)
    information = family.expected_information_natural(initialized.theta, ordinary_plan)
    assert np.all(np.isfinite(evaluation.score))
    assert np.all(information[:, (0, 2)] > 0.0)

    impossible_plan = _plan(
        family,
        response,
        np.array([np.nextafter(0.0, 1.0)]),
        "prior",
    )
    with pytest.raises(GammaInitializationError, match="representable"):
        family.initialize(response, impossible_plan)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_plan_binds_exact_response_separately_from_colliding_log_carrier(
    semantics: str,
) -> None:
    """Kills identifying responses only by their non-injective -m*log(y) carrier."""

    family = GammaLS()
    for base in (np.float64(1e-300), np.float64(1e300)):
        adjacent = np.nextafter(base, np.inf)
        assert base != adjacent
        assert np.log(base) == np.log(adjacent)
        weights = np.ones(1)
        first = _plan(family, np.array([base]), weights, semantics)
        second = _plan(family, np.array([adjacent]), weights, semantics)
        np.testing.assert_array_equal(
            first.parameter_independent_carrier,
            second.parameter_independent_carrier,
        )
        assert first.response_digest != second.response_digest
        assert first.plan_identifier != second.plan_identifier


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_plan_is_byte_immutable_and_take_rebinds_ordered_exact_children(
    semantics: str,
) -> None:
    """Kills mutable carriers, response bindings, and reordered children."""

    family = GammaLS()
    weights = PRIOR_WEIGHTS if semantics == "prior" else FREQUENCY_WEIGHTS
    root = _plan(family, Y, weights, semantics)
    indices = np.array([3, 1], dtype=np.intp)
    child = root.take(indices)
    for values in (
        root.exact_response,
        root.parameter_independent_carrier,
        child.exact_response,
        child.parameter_independent_carrier,
    ):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    np.testing.assert_array_equal(child.exact_response, Y[indices])
    expected_carrier = -np.log(Y[indices])
    if semantics == "frequency":
        expected_carrier *= weights[indices]
    np.testing.assert_array_equal(child.parameter_independent_carrier, expected_carrier)


def test_gamma_plan_subclass_is_validated_structurally_by_fields() -> None:
    class GammaPlanSubclass(GammaLikelihoodPlan):
        pass

    family = GammaLS()
    response = np.array([0.5, 2.0])
    theta = np.array([[0.8, 0.7], [1.5, 1.2]])
    base = _plan(family, response, np.ones(2), "prior")
    plan = GammaPlanSubclass(**vars(base))

    initialized = family.initialize(response, plan)
    evaluated = family.evaluate_natural(response, theta, plan)

    np.testing.assert_equal(initialized.theta, family.initialize(response, base).theta)
    np.testing.assert_equal(
        evaluated.reported_log_likelihood,
        family.evaluate_natural(response, theta, base).reported_log_likelihood,
    )
    malformed = replace(plan, exact_response=response.copy())
    with pytest.raises(UnsupportedLikelihoodContractError, match="invalid bound response"):
        family.validate_likelihood_plan(response, malformed)


def test_gamma_response_carrier_plan_uses_public_one_shot_protocol() -> None:
    family = GammaLS()
    response = np.array([0.5, 2.0])
    plan = _plan(family, response, np.ones(2), "prior")

    assert isinstance(family, LikelihoodPlanValidatingFamily)
    canonical = family.validate_likelihood_plan(response, plan)

    assert canonical is plan.exact_response
    assert canonical.dtype == np.float64
    assert not canonical.flags.writeable


def test_gamma_bound_likelihood_refuses_a_same_shape_different_response() -> None:
    family = GammaLS()
    response = np.array([0.5, 1.0, 2.0])
    plan = family.bind_likelihood(
        response,
        resolve_likelihood_weights(
            np.ones(3),
            n_observations=3,
            contract=WeightContract("prior"),
        ),
        COMPLETE_OBSERVATION,
    )
    changed = response.copy()
    changed[1] = np.nextafter(changed[1], np.inf)

    with pytest.raises(UnsupportedLikelihoodContractError, match="Gamma|response"):
        chunking._validate_bound_likelihood(family, plan, changed)

    accepted_plan, canonical = chunking._validate_bound_likelihood(family, plan, response)
    assert accepted_plan is plan
    assert canonical is plan.exact_response
    assert not canonical.flags.writeable


@pytest.mark.parametrize("bad_y", [np.array([0.0]), np.array([-1.0]), np.array([np.nan])])
def test_bind_rejects_nonpositive_or_nonfinite_retained_response(bad_y: np.ndarray) -> None:
    """Kills relaxing Gamma's strict positive finite response support."""

    family = GammaLS()
    with pytest.raises(ValueError, match="strictly positive|finite"):
        family.bind_likelihood(bad_y, _resolved(np.ones(len(bad_y)), "prior"), COMPLETE_OBSERVATION)


def test_family_rejects_theta_shape_support_order_and_plan_contract_errors() -> None:
    """Kills unchecked theta shape/support/order and unsupported observation plans."""

    family = GammaLS()
    plan = _plan(family, Y, PRIOR_WEIGHTS, "prior")
    theta = np.column_stack((MEAN, SCALE))
    with pytest.raises(ValueError, match="theta.*shape"):
        family.evaluate_natural(Y, theta[:, :1], plan)
    with pytest.raises(ValueError, match="outside its finite support"):
        family.evaluate_natural(Y, np.column_stack((MEAN, np.zeros(len(Y)))), plan)
    with pytest.raises(ValueError, match="derivative_order"):
        family.evaluate_natural(Y, theta, plan, derivative_order=3)
    with pytest.raises(LikelihoodWeightError, match="likelihood"):
        family.initialize(Y, np.ones(len(Y)))
    with pytest.raises(LikelihoodWeightError, match="likelihood"):
        family.expected_information_natural(theta, np.ones(len(Y)))


def test_tiny_and_large_shape_helpers_have_their_bounded_limits() -> None:
    """Kills raw special-function cancellation/overflow at either shape extreme."""

    shapes = np.array([1e-300, 1e-100, 1e-16, 1e-8, 1e8, 1e32, 1e100, 1e300])
    a_residual = _scaled_digamma_residual(shapes)
    j_residual = _scaled_trigamma_residual(shapes)
    normalizer = _gamma_log_normalizer(shapes)
    assert np.all(np.isfinite(a_residual))
    assert np.all(np.isfinite(j_residual))
    assert np.all(np.isfinite(normalizer))
    np.testing.assert_allclose(a_residual[:3], -np.ones(3), rtol=0.0, atol=8e-14)
    np.testing.assert_allclose(j_residual[:3], np.ones(3), rtol=0.0, atol=8e-14)
    np.testing.assert_allclose(a_residual[-3:], -0.5 * np.ones(3), rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(j_residual[-3:], 0.5 * np.ones(3), rtol=0.0, atol=2e-14)
    expected_large_normalizer = 0.5 * np.log(shapes[-3:] / (2.0 * np.pi))
    np.testing.assert_allclose(normalizer[-3:], expected_large_normalizer, rtol=3e-15)


def test_large_trigamma_series_keeps_the_last_resolved_bernoulli_terms() -> None:
    """Kills gating J on a later omitted term while dropping resolvable a^-9/a^-11 terms."""

    a = 32.0
    inverse = 1.0 / a
    expected = (
        0.5
        + inverse / 6.0
        - inverse**3 / 30.0
        + inverse**5 / 42.0
        - inverse**7 / 30.0
        + 5.0 * inverse**9 / 66.0
        - 691.0 * inverse**11 / 2730.0
    )
    actual = _scaled_trigamma_residual(np.array([a]))[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=np.spacing(expected))


def _large_shape_route_gate(combination: str) -> float:
    eps = np.finfo(np.float64).eps
    if combination == "A":
        return ((691.0 / 32760.0) / (eps / 8.0)) ** (1.0 / 11.0)
    if combination == "J":
        return ((7.0 / 6.0) / (eps / 8.0)) ** (1.0 / 13.0)

    def balance(log_shape: float) -> float:
        return (
            math.log(1.0 / 1188.0)
            - 9.0 * log_shape
            - math.log(eps)
            - math.log(max(1.0, log_shape))
            + math.log(8.0)
        )

    log_gate = optimize.brentq(balance, math.log(16.0), math.log(1e4))
    gate = math.exp(log_gate)

    def asymptotic_route(shape: float) -> bool:
        omitted = (1.0 / 1188.0) * shape**-9
        threshold = eps * max(1.0, math.log(shape)) / 8.0
        return omitted <= threshold

    while not asymptotic_route(gate):
        gate = float(np.nextafter(gate, np.inf))
    while asymptotic_route(float(np.nextafter(gate, 0.0))):
        gate = float(np.nextafter(gate, 0.0))
    return gate


@pytest.mark.parametrize(
    ("combination", "helper"),
    [
        ("A", _scaled_digamma_residual),
        ("J", _scaled_trigamma_residual),
        ("C", _gamma_log_normalizer),
    ],
)
def test_large_shape_routes_obey_signed_bernoulli_remainders(
    combination: str,
    helper,
) -> None:
    """Kills a missing, sign-flipped, or mis-scaled retained Bernoulli term."""

    gate = _large_shape_route_gate(combination)
    shapes = np.unique(np.array([np.nextafter(gate, np.inf), 2.0 * gate, 128.0, 1e4, 1e32]))
    actual = helper(shapes)
    for shape, value in zip(shapes, actual, strict=True):
        oracle = gamma_asymptotic_oracle(combination, float(shape))
        coarse_lower, coarse_upper = oracle.coarse_interval
        tight_lower, tight_upper = oracle.tight_interval
        assert coarse_lower <= tight_lower <= tight_upper <= coarse_upper
        rounding = 8.0 * abs(np.spacing(np.float64(oracle.retained_float)))
        assert abs(float(value) - oracle.retained_float) <= rounding


def test_special_function_routes_are_continuous_at_error_gates() -> None:
    """Kills discontinuous small-series/direct or direct/asymptotic route changes."""

    small_boundary = 0.25
    small_points = np.array([np.nextafter(small_boundary, 0.0), small_boundary])
    for helper in (
        _scaled_digamma_residual,
        _scaled_trigamma_residual,
        _gamma_log_normalizer,
    ):
        values = helper(small_points)
        assert abs(values[1] - values[0]) <= 16.0 * np.spacing(max(abs(values)))

    eps = np.finfo(np.float64).eps
    gates = (
        ((691.0 / 32760.0) / (eps / 8.0)) ** (1.0 / 11.0),
        ((7.0 / 6.0) / (eps / 8.0)) ** (1.0 / 13.0),
    )
    for helper, gate in zip(
        (_scaled_digamma_residual, _scaled_trigamma_residual), gates, strict=True
    ):
        below = np.nextafter(gate, 0.0)
        above = np.nextafter(gate, np.inf)
        values = helper(np.array([below, above]))
        direct_roundoff = (
            64.0 * eps * gate * (abs(float(special.digamma(gate))) + abs(np.log(gate)) + 1.0)
        )
        assert abs(values[1] - values[0]) <= direct_roundoff

    c_gate = _large_shape_route_gate("C")
    below = np.nextafter(c_gate, 0.0)
    above = c_gate
    values = _gamma_log_normalizer(np.array([below, above]))
    direct_magnitude = (
        abs(c_gate * math.log(c_gate)) + c_gate + abs(float(special.gammaln(c_gate))) + 1.0
    )
    first_omitted = (1.0 / 1188.0) * c_gate**-9
    assert abs(values[1] - values[0]) <= 64.0 * eps * direct_magnitude + first_omitted


def test_ordinary_gamma_rows_batch_each_scipy_special_function_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills Python-row dispatch to SciPy on the ordinary direct mask."""

    n_rows = 4096
    family = GammaLS()
    y = np.full(n_rows, 1.5)
    theta = np.column_stack((np.ones(n_rows), np.full(n_rows, 0.5)))
    plan = _plan(family, y, np.ones(n_rows), "frequency")
    seen: dict[str, list[tuple[int, ...]]] = {"gammaln": [], "digamma": [], "polygamma": []}
    originals = {
        "gammaln": special.gammaln,
        "digamma": special.digamma,
        "polygamma": special.polygamma,
    }

    def counted_gammaln(values):
        seen["gammaln"].append(np.asarray(values).shape)
        return originals["gammaln"](values)

    def counted_digamma(values):
        seen["digamma"].append(np.asarray(values).shape)
        return originals["digamma"](values)

    def counted_polygamma(order, values):
        seen["polygamma"].append(np.asarray(values).shape)
        return originals["polygamma"](order, values)

    monkeypatch.setattr(gamma_kernel.special, "gammaln", counted_gammaln)
    monkeypatch.setattr(gamma_kernel.special, "digamma", counted_digamma)
    monkeypatch.setattr(gamma_kernel.special, "polygamma", counted_polygamma)
    result = family.evaluate_natural(y, theta, plan, derivative_order=2)

    assert result.derivative_order == 2
    assert seen == {
        "gammaln": [(n_rows,)],
        "digamma": [(n_rows,)],
        "polygamma": [(n_rows,)],
    }


def test_vector_shape_preserves_the_certified_composed_rounding() -> None:
    """Kills direct w/(σ²) rounding that destabilized an exact-discrete fit."""

    weight = float.fromhex("0x1.48baaa5d0087cp-1")
    scale = float.fromhex("0x1.f4ed29caee4c9p+2")
    expected = float.fromhex("0x1.576c7883533b3p-7")
    direct = weight / (scale * scale)

    shape, multiplier = _shape_from_scale(
        np.array([scale]),
        np.array([weight]),
        "prior",
    )

    assert direct.hex() == "0x1.576c7883533b4p-7"
    assert shape[0] == expected
    assert multiplier[0] == 1.0


def test_ordinary_gamma_rows_do_not_enter_scalar_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills scalar fallback work proportional to the ordinary row count."""

    names = (
        "_binary_product_divide",
        "_channel",
        "_small_a_residual",
        "_small_j_residual",
        "_small_log_normalizer",
        "_log_ratio",
        "_deviance_from_t",
    )
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(gamma_kernel, name)

        def counted(*args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(gamma_kernel, name, counted)

    n_rows = 4096
    family = GammaLS()
    y = np.full(n_rows, 1.5)
    theta = np.column_stack((np.ones(n_rows), np.full(n_rows, 0.5)))
    family.evaluate_natural(y, theta, _plan(family, y, np.ones(n_rows), "frequency"))

    assert calls == dict.fromkeys(names, 0)


def test_gamma_scalar_fallback_work_depends_only_on_extreme_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills fallback dispatch that grows when only ordinary rows are duplicated."""

    names = (
        "_binary_product_divide",
        "_channel",
        "_small_a_residual",
        "_small_j_residual",
        "_small_log_normalizer",
        "_log_ratio",
        "_deviance_from_t",
    )
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(gamma_kernel, name)

        def counted(*args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(gamma_kernel, name, counted)

    extreme_y = np.array([np.nextafter(1.0, np.inf), 1.234e300, 1e-170, 1e300])
    extreme_mean = np.array([1.0, 1.0, 1e160, 1e-100])
    extreme_shape = np.array([1e32, 9.876e-301, 1.0, 1e-300])

    def evaluate(n_ordinary: int) -> dict[str, int]:
        calls.update(dict.fromkeys(names, 0))
        y = np.concatenate((np.full(n_ordinary, 1.5), extreme_y))
        mean = np.concatenate((np.ones(n_ordinary), extreme_mean))
        scale = np.concatenate((np.full(n_ordinary, 0.5), 1.0 / np.sqrt(extreme_shape)))
        family = GammaLS()
        family.evaluate_natural(
            y,
            np.column_stack((mean, scale)),
            _plan(family, y, np.ones(len(y)), "frequency"),
        )
        return calls.copy()

    first = evaluate(32)
    second = evaluate(64)
    assert any(first.values())
    assert second == first


def test_stable_sum_fallback_work_depends_only_on_cancellation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills sending conditioned ordinary rows through scalar ``math.fsum``."""

    calls = 0
    original_fsum = gamma_kernel.math.fsum

    def counted_fsum(values):
        nonlocal calls
        calls += 1
        return original_fsum(values)

    monkeypatch.setattr(gamma_kernel.math, "fsum", counted_fsum)
    extreme_y = np.array([np.nextafter(1.0, np.inf), 1.234e300, 1e-170, 1e300])
    extreme_mean = np.array([1.0, 1.0, 1e160, 1e-100])
    extreme_shape = np.array([1e32, 9.876e-301, 1.0, 1e-300])

    def evaluate(n_ordinary: int, *, append_extremes: bool) -> int:
        nonlocal calls
        calls = 0
        y = np.full(n_ordinary, 1.5)
        mean = np.ones(n_ordinary)
        scale = np.full(n_ordinary, 0.5)
        if append_extremes:
            y = np.concatenate((y, extreme_y))
            mean = np.concatenate((mean, extreme_mean))
            scale = np.concatenate((scale, 1.0 / np.sqrt(extreme_shape)))
        family = GammaLS()
        family.evaluate_natural(
            y,
            np.column_stack((mean, scale)),
            _plan(family, y, np.ones(len(y)), "frequency"),
        )
        return calls

    assert evaluate(4096, append_extremes=False) == 0
    assert evaluate(8192, append_extremes=False) == 0
    fixed_mask_calls = evaluate(4096, append_extremes=True)
    assert fixed_mask_calls > 0
    assert evaluate(8192, append_extremes=True) == fixed_mask_calls


def test_ratio_helper_preserves_coupled_extremes_and_near_equality() -> None:
    """Kills direct ratio under/overflow, log-exp reconstruction, and t-log1p cancellation."""

    tiny = np.nextafter(0.0, 1.0)
    y = np.array([np.nextafter(1.0, np.inf), 1.234e300, 1e-170, 1e300, 7.5e299])
    mean = np.array([1.0, 1.0, 1e160, 1e-100, 5e299])
    shape = np.array([1e32, 9.876e-301, 1.0, 1e-300, 3.0])
    assert y[2] / mean[2] == 0.0
    az, at, ad = _scaled_ratio_terms(y, mean, shape)
    assert np.all(np.isfinite(az))
    assert np.all(np.isfinite(at))
    assert np.all(np.isfinite(ad))
    expected_near = 0.5 * shape[0] * np.spacing(1.0) ** 2
    np.testing.assert_allclose(ad[0], expected_near, rtol=5e-16)
    np.testing.assert_allclose(az[1], 1.2186983999999998, rtol=0.0, atol=np.spacing(1.0))
    assert az[2] == 0.0 and at[2] == -1.0 and ad[2] > 700.0
    np.testing.assert_allclose(az[3], 1e100, rtol=3e-16)
    np.testing.assert_allclose(az[4], 4.5, rtol=3e-16)
    assert tiny > 0.0


def _assert_scaled_ratio_matches_decimal_oracle(y: np.ndarray, mean: np.ndarray) -> None:
    shape = np.full(len(y), 3.75)
    az, at, ad = _scaled_ratio_terms(y, mean, shape)
    actual = np.column_stack((az, at, ad))
    expected = gamma_scaled_ratio_oracle(y, mean, shape)
    bound = 64.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(expected))
    np.testing.assert_array_less(np.abs(actual - expected), bound)


def test_ratio_helper_matches_decimal_oracle_across_exponent_gap() -> None:
    """Kills a discontinuity where the exponent-gap route changes."""

    y = np.array([np.nextafter(4.0, 0.0), 4.0])
    _assert_scaled_ratio_matches_decimal_oracle(y, np.ones(len(y)))


def test_ratio_helper_matches_decimal_oracle_across_series_boundary() -> None:
    """Kills a discontinuity at either side of ``|t| = 0.125``."""

    y = np.array(
        [
            np.nextafter(0.875, 0.0),
            0.875,
            1.125,
            np.nextafter(1.125, np.inf),
        ]
    )
    _assert_scaled_ratio_matches_decimal_oracle(y, np.ones(len(y)))


def test_complete_extreme_rows_and_link_transform_remain_finite() -> None:
    """Kills helpers that work alone but lose information in complete family channels."""

    family = GammaLS()
    y = np.array([np.nextafter(1.0, np.inf), 1.234e300, 1e-170, 1e300])
    mean = np.array([1.0, 1.0, 1e160, 1e-100])
    shape = np.array([1e32, 9.876e-301, 1.0, 1e-300])
    scale = 1.0 / np.sqrt(shape)
    theta = np.column_stack((mean, scale))
    plan = _plan(family, y, np.ones(len(y)), "frequency")
    evaluation = family.evaluate_natural(y, theta, plan)
    fisher = family.expected_information_natural(theta, plan)
    links = (LogLink(), LogLink())
    eta = np.log(theta)
    transformed = transform_natural_derivatives(
        evaluation,
        eta,
        links,
    )
    transformed_fisher = transform_natural_information(fisher, eta, links)
    for values in (
        evaluation.optimizing_log_likelihood,
        evaluation.score,
        evaluation.hessian_packed,
        fisher,
        transformed.score_eta,
        transformed.curvature_packed,
        transformed_fisher,
    ):
        assert np.all(np.isfinite(values))


def test_underflow_rescue_composes_before_rounding_intermediate() -> None:
    """Kills rounding a*t to zero before the final Umean and Imean channels."""

    family = GammaLS()
    delta = np.nextafter(0.0, 1.0)
    y = np.array([0.25])
    theta = np.array([[0.5, 1.0]])
    plan = _plan(family, y, np.array([delta]), "prior")
    evaluation = family.evaluate_natural(y, theta, plan)
    information = family.expected_information_natural(theta, plan)
    assert evaluation.score[0, 0] == -delta
    assert evaluation.hessian_packed[0, 0] == 0.0
    assert information[0, 0] == 4.0 * delta


def test_nonzero_mean_hessian_composes_before_subnormal_numerator_rounding() -> None:
    """Kills rounding nonzero a*(1+2t) to zero before the final mean divisions."""

    family = GammaLS()
    delta = np.nextafter(0.0, 1.0)
    mean = 1e-150
    response = np.array([0.75 * mean])
    theta = np.array([[mean, 1.0]])
    plan = _plan(family, response, np.array([delta]), "prior")
    evaluation = family.evaluate_natural(response, theta, plan)
    expected = -2.4703282292062325e-24
    assert evaluation.hessian_packed[0, 0] == expected


@pytest.mark.parametrize(
    ("operation", "y", "mean", "scale", "weight", "message"),
    [
        ("fisher", 1.0, 1.0, 1e300, 1e300, "scale.*information|representable"),
        ("fisher", 1e300, 1e300, 1.0, 1.0, "mean.*information|representable"),
        ("evaluate", 1e300, np.nextafter(0.0, 1.0), 1.0, 1.0, "representable"),
    ],
)
def test_unrepresentable_final_natural_channels_are_refused(
    operation: str,
    y: float,
    mean: float,
    scale: float,
    weight: float,
    message: str,
) -> None:
    """Kills clipping or accepting genuine final-channel overflow/underflow."""

    response = np.array([y])
    theta = np.array([[mean, scale]])
    weights = np.array([weight])
    with pytest.raises(ValueError, match=message):
        if operation == "fisher":
            gamma_kernel.gamma_expected_information(
                theta[:, 0],
                theta[:, 1],
                weights,
                "prior",
            )
        else:
            gamma_kernel.evaluate_gamma_rows(
                response,
                theta[:, 0],
                theta[:, 1],
                weights,
                "prior",
                derivative_order=2,
            )


def test_genuine_zero_score_and_hessian_loci_are_accepted() -> None:
    """Kills indiscriminate rejection of exact zero natural channels."""

    family = GammaLS()
    y = np.array([1.0, 1.0])
    mean = np.array([1.0, 2.0])
    theta = np.column_stack((mean, np.ones(2)))
    evaluation = family.evaluate_natural(y, theta, _plan(family, y, np.ones(2), "prior"))
    assert evaluation.score[0, 0] == 0.0
    assert evaluation.hessian_packed[1, 0] == 0.0


def test_gamma_metadata_configuration_prediction_and_ownership() -> None:
    """Kills incorrect public family metadata, prediction column, or mutable results."""

    family = GammaLS()
    parameters = validate_family(family)
    assert tuple(parameter.name for parameter in parameters) == ("mean", "scale")
    assert tuple(parameter.role for parameter in parameters) == ("mean", "scale")
    assert tuple(parameter.curvature for parameter in parameters) == ("fisher", "fisher")
    assert all(isinstance(parameter.default_link, LogLink) for parameter in parameters)
    assert all(parameter.support.lower == 0.0 for parameter in parameters)
    assert all(not parameter.support.lower_inclusive for parameter in parameters)
    assert family.to_config() == {"type": "GammaLS", "parameterization": "mean_cv"}
    assert json.loads(json.dumps(family.to_config())) == family.to_config()
    assert family.default_prediction_name == "conditional_mean"
    assert family.capabilities.max_derivative_order == 2
    assert family.capabilities.expected_information is True
    assert family.capabilities.response_mean is True
    assert family.capabilities.cdf is True
    assert family.capabilities.quantile is True
    assert family.capabilities.random is False
    assert family.capabilities.censored_response is False
    assert isinstance(family, ExpectedInformationFamily)
    theta = np.array([[1.0, 0.4], [2.0, 0.8]])
    prediction = family.default_prediction(theta)
    theta[0, 0] = 99.0
    np.testing.assert_array_equal(prediction, np.array([1.0, 2.0]))
    assert not prediction.flags.writeable
    with pytest.raises(FrozenInstanceError):
        family.extra = 1  # type: ignore[misc]


@dataclass(frozen=True)
class _EndpointPolishFixture:
    family: GammaLS
    layout: StackedLayout
    plan: GammaLikelihoodPlan
    initial: np.ndarray
    config: DenseSolverConfig
    chunk_size: int | None
    source: DenseSolverResult
    polished: DenseSolverResult
    correction: np.ndarray
    candidate: np.ndarray


def _endpoint_polish_fixture(
    *,
    chunk_size: int | None = None,
    retained_rows: int | None = None,
    source_score_relative: float = 1.0,
    polished_score_relative: float = 0.0,
) -> _EndpointPolishFixture:
    family, layout, plan, initial = _gamma_chunk_problem()
    config = DenseSolverConfig(
        max_iterations=150,
        tolerance=1.0e-12,
        coefficient_curvature="observed",
        newton_decrement_tolerance=None,
    )
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    base = efs_module.fit_dense_fixed_lambda(
        family,
        layout,
        Y,
        plan,
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
    )
    if retained_rows is not None:
        repeats = math.ceil(retained_rows / len(base.eta))
        eta = np.tile(base.eta, (repeats, 1))[:retained_rows]
        theta = np.tile(base.theta, (repeats, 1))[:retained_rows]
        base = replace(base, eta=eta, theta=theta)
    score = np.array([1.0e-6, -5.0e-7], dtype=np.float64)
    source = replace(
        base,
        terminal_score=score,
        score_relative=source_score_relative,
        convergence_reason="objective_and_step",
    )
    correction = source.solve_terminal(source.terminal_score)
    candidate = source.coefficients + correction
    polished = replace(
        base,
        coefficients=candidate,
        terminal_score=np.zeros_like(score),
        score_relative=polished_score_relative,
        convergence_reason="score",
    )
    return _EndpointPolishFixture(
        family=family,
        layout=layout,
        plan=plan,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
        source=source,
        polished=polished,
        correction=correction,
        candidate=candidate,
    )


def _invoke_endpoint_polish(
    monkeypatch: pytest.MonkeyPatch,
    fixture: _EndpointPolishFixture,
    *,
    polished: DenseSolverResult | None = None,
    correction: np.ndarray | None = None,
) -> tuple[DenseSolverResult, tuple[np.ndarray, ...]]:
    outputs = (fixture.source, fixture.polished if polished is None else polished)
    initials: list[np.ndarray] = []

    def fixed_state(*args, **kwargs):
        del args
        initials.append(np.array(kwargs["initial"], copy=True))
        return outputs[len(initials) - 1]

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", fixed_state)
    monkeypatch.setattr(smoothing_loop, "_fit_fixed_state", fixed_state)
    monkeypatch.setattr(efs_module, "_fit_fixed_state", fixed_state)
    if correction is not None:
        real_solve = DenseSolverResult.solve_terminal

        def terminal_solve(result: DenseSolverResult, rhs: np.ndarray) -> np.ndarray:
            if result is fixture.source:
                return np.array(correction, copy=True)
            return real_solve(result, rhs)

        monkeypatch.setattr(DenseSolverResult, "solve_terminal", terminal_solve)
    result = efs_module._fit_endpoint_authority_stationary(
        fixture.family,
        fixture.layout,
        Y,
        fixture.plan,
        lambdas={},
        face=None,
        initial=fixture.initial,
        config=fixture.config,
        chunk_size=fixture.chunk_size,
        phase_recorder=None,
    )
    return result, tuple(initials)


def _replace_penalized_optimizing_objective(
    result: DenseSolverResult,
    value: float,
) -> DenseSolverResult:
    optimizing = value + result.penalty_value
    reported = optimizing + result.parameter_independent_carrier
    return replace(
        result,
        optimizing_log_likelihood=optimizing,
        log_likelihood=reported,
        penalized_optimizing_log_likelihood=value,
        penalized_log_likelihood=reported - result.penalty_value,
    )


def _audited_polish_objective_bound(
    source: DenseSolverResult,
    polished: DenseSolverResult,
) -> float:
    n_rows = source.eta.shape[0]
    chunks = (
        1 if source.resolved_chunk_size is None else math.ceil(n_rows / source.resolved_chunk_size)
    )
    dimension = max(n_rows + chunks, source.coefficients.size, 1)
    dtype = np.result_type(source.coefficients.dtype, polished.coefficients.dtype)
    epsilon = np.finfo(dtype).eps
    operations = 64 * dimension
    operation_error = operations * epsilon
    assert operation_error < 1.0
    gamma = operation_error / (1.0 - operation_error)

    def scale(result: DenseSolverResult) -> float:
        penalty_product = 0.5 * float(
            np.abs(result.coefficients) @ np.abs(result.penalty) @ np.abs(result.coefficients)
        )
        assert result.penalized_optimizing_log_likelihood is not None
        assert result.optimizing_log_likelihood is not None
        return max(
            1.0,
            abs(result.penalized_optimizing_log_likelihood),
            abs(result.optimizing_log_likelihood) + penalty_product,
        )

    return float(np.nextafter(gamma * max(scale(source), scale(polished)), math.inf))


def test_endpoint_authority_polish_recomputes_retained_kkt_instead_of_scalar_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills trusting either source or polished ``score_relative`` metadata."""

    fixture = _endpoint_polish_fixture(
        source_score_relative=0.0,
        polished_score_relative=1.0,
    )
    result, initials = _invoke_endpoint_polish(monkeypatch, fixture)

    assert result is fixture.polished
    assert len(initials) == 2
    np.testing.assert_array_equal(initials[1], fixture.candidate)


@pytest.mark.parametrize(
    "correction_kind", ["zero", "nonfinite", "wrong_shape", "wrong_sign", "scaled"]
)
def test_endpoint_authority_polish_refuses_uncertified_terminal_corrections(
    monkeypatch: pytest.MonkeyPatch,
    correction_kind: str,
) -> None:
    """Kills accepting a no-op, malformed, descent, or bad-residual Newton correction."""

    fixture = _endpoint_polish_fixture()
    corrections = {
        "zero": np.zeros_like(fixture.correction),
        "nonfinite": np.full_like(fixture.correction, np.inf),
        "wrong_shape": fixture.correction[:1],
        "wrong_sign": -fixture.correction,
        "scaled": 2.0 * fixture.correction,
    }
    result, initials = _invoke_endpoint_polish(
        monkeypatch,
        fixture,
        correction=corrections[correction_kind],
    )

    assert result is fixture.source
    assert len(initials) == 1


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_endpoint_authority_polish_uses_row_and_chunk_aware_objective_accumulation(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int | None,
) -> None:
    """Kills reverting to a coefficient-width-only objective allowance."""

    fixture = _endpoint_polish_fixture(chunk_size=chunk_size, retained_rows=4096)
    assert fixture.source.penalized_optimizing_log_likelihood is not None
    polished = _replace_penalized_optimizing_objective(
        fixture.polished,
        fixture.source.penalized_optimizing_log_likelihood - 1.0e-11,
    )
    bound = _audited_polish_objective_bound(fixture.source, polished)
    assert 1.0e-11 < bound

    result, _ = _invoke_endpoint_polish(monkeypatch, fixture, polished=polished)

    assert result is polished


def test_endpoint_authority_polish_refuses_objective_loss_outside_audited_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _endpoint_polish_fixture(retained_rows=4096)
    bound = _audited_polish_objective_bound(fixture.source, fixture.polished)
    assert fixture.source.penalized_optimizing_log_likelihood is not None
    polished = _replace_penalized_optimizing_objective(
        fixture.polished,
        fixture.source.penalized_optimizing_log_likelihood - 2.0 * bound,
    )

    result, _ = _invoke_endpoint_polish(monkeypatch, fixture, polished=polished)

    assert result is fixture.source


def test_endpoint_authority_polish_refuses_worsened_retained_kkt_with_forged_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _endpoint_polish_fixture()
    polished = replace(
        fixture.polished,
        terminal_score=2.0 * fixture.source.terminal_score,
        score_relative=0.0,
        convergence_reason="objective_and_step",
    )

    result, _ = _invoke_endpoint_polish(monkeypatch, fixture, polished=polished)

    assert result is fixture.source


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "plan",
        "resolved_chunk",
        "backend",
        "fallback",
        "face",
        "rank_policy",
        "rank_method",
        "rank",
        "rank_resolution",
        "active_columns",
        "penalty",
    ],
)
def test_endpoint_authority_polish_refuses_changed_same_fit_provenance(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _endpoint_polish_fixture(chunk_size=2)
    polished = fixture.polished
    if mutation == "config":
        polished = replace(
            polished,
            config=replace(polished.config, max_iterations=polished.config.max_iterations + 1),
        )
    elif mutation == "plan":
        polished = replace(polished, family_likelihood_plan_identifier="forged-plan")
    elif mutation == "resolved_chunk":
        polished = replace(polished, resolved_chunk_size=1)
    elif mutation == "backend":
        object.__setattr__(
            polished,
            "execution_backend_identifier",
            "distributional-dense-v1",
        )
    elif mutation == "fallback":
        polished = replace(
            polished,
            terminal_curvature=replace(polished.terminal_curvature, fallback_count=1),
        )
    elif mutation == "face":

        class ForgedIdentityFace:
            @staticmethod
            def reduce_vector(values: np.ndarray) -> np.ndarray:
                return np.asarray(values)

        object.__setattr__(polished, "coefficient_face", ForgedIdentityFace())
        object.__setattr__(polished, "terminal_reduced_rank", polished.terminal_rank)
    elif mutation in {
        "rank_policy",
        "rank_method",
        "rank",
        "rank_resolution",
        "active_columns",
    }:
        rank_changes: dict[str, object] = {
            "rank_policy": {"policy_version": polished.terminal_rank.policy_version + 1},
            "rank_method": {"method": "gram_eigh"},
            "rank": {"rank": polished.terminal_rank.rank - 1},
            "rank_resolution": {
                "resolution_limited": not polished.terminal_rank.resolution_limited
            },
            "active_columns": {"active_columns": polished.terminal_rank.active_columns[::-1]},
        }[mutation]
        polished = replace(
            polished,
            terminal_rank=replace(polished.terminal_rank, **rank_changes),
        )
    else:
        forged_penalty = np.array(polished.penalty, copy=True)
        forged_penalty[0, 0] = 1.0
        object.__setattr__(polished, "penalty", forged_penalty)

    result, _ = _invoke_endpoint_polish(monkeypatch, fixture, polished=polished)

    assert result is fixture.source


def test_endpoint_authority_polish_refuses_a_refit_far_from_the_newton_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _endpoint_polish_fixture()
    moved = np.array(fixture.polished.coefficients, copy=True)
    moved[0] += 1.0e-6
    polished = replace(fixture.polished, coefficients=moved)

    result, _ = _invoke_endpoint_polish(monkeypatch, fixture, polished=polished)

    assert result is fixture.source


def _gamma_exact_face_problem() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x_unique = np.linspace(-1.0, 1.0, 24)
    groups = np.array(["a", "b", "c"])
    residual_factors = np.array([0.62, 0.84, 1.16, 1.38])
    x = np.repeat(x_unique, len(groups) * len(residual_factors))
    group = np.tile(np.repeat(groups, len(residual_factors)), len(x_unique))
    factors = np.tile(np.tile(residual_factors, len(groups)), len(x_unique))
    mean = np.exp(0.45 + 0.48 * np.sin(np.pi * x) + 0.18 * x)
    return (
        pd.DataFrame({"x": x, "group": group}),
        mean * factors,
        0.65 + 0.7 * (x + 1.0) / 2.0,
    )


@pytest.mark.parametrize("chunk_size", [None, 37])
def test_gamma_efs_selects_an_exact_irrelevant_face_and_keeps_a_finite_smooth(
    chunk_size: int | None,
) -> None:
    frame, response, weights = _gamma_exact_face_problem()
    maximum_lambda = 10.0

    model = fit_dense_distributional(
        frame,
        response,
        family=GammaLS(),
        weight_contract=WeightContract("prior"),
        predictors=(
            Predictor("mean", {"x": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {"group": RandomEffect()}),
        ),
        sample_weight=weights,
        lambdas={"mean:x#wiggle": 0.5, "scale:group#wiggle": maximum_lambda},
        config=DenseSolverConfig(tolerance=1.0e-9, max_iterations=150),
        efs_config=DistributionalEFSConfig(
            max_iterations=120,
            tolerance=1.0e-3,
            maximum_lambda=maximum_lambda,
            # The exact-face selection and the Fellner--Schall residual it asserts are
            # the Fellner--Schall path's contract.
            outer="efs",
        ),
        chunk_size=chunk_size,
        retain_rows=True,
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.converged is True
    assert smoothing.matched_certified is False
    with pytest.raises(
        RuntimeError,
        match="exact coefficient face is numerically supported but not certified",
    ):
        smoothing.assert_matched_certified()
    assert smoothing.lambdas["mean:x#wiggle"] < maximum_lambda / 2.0
    assert smoothing.terminal_raw_max_log_step <= smoothing.config.tolerance
    assert smoothing.unresolved_upper_bound == ()

    terminal = smoothing.terminal_fit
    assert efs_module._endpoint_retained_kkt_relative(terminal) <= terminal.config.tolerance
    assert terminal.resolved_chunk_size == chunk_size
    face = terminal.coefficient_face
    assert face is not None
    assert face.component_names == ("scale:group#wiggle",)
    evidence = smoothing.terminal_endpoint_directions["scale:group#wiggle"]
    assert evidence.decision == "endpoint"
    assert evidence.endpoint_objective == smoothing.objective
    for item in smoothing.history:
        if item.activated_face_components or item.revalidated_face_components:
            for fit_index in item.coefficient_fit_indices:
                authority_config = smoothing.coefficient_fits[fit_index].config
                assert authority_config.coefficient_curvature == "observed"
                assert authority_config.tolerance == 1.0e-12
                assert authority_config.newton_decrement_tolerance is None
    null_residual = np.linalg.norm(face.constraint_matrix @ terminal.coefficients, ord=2)
    coefficient_scale = max(1.0, float(np.linalg.norm(terminal.coefficients, ord=2)))
    assert null_residual <= face.null_residual_bound * coefficient_scale

    predictions = model.predict(frame)
    parameters = model.predict_parameters(frame)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0.0)
    assert np.all(np.isfinite(parameters))
    assert np.all(parameters > 0.0)
    constrained_covariance = face.constraint_basis.T @ model.inference.covariance
    covariance_scale = max(1.0, float(np.linalg.norm(model.inference.covariance, ord=2)))
    covariance_bound = 256.0 * face.width * np.finfo(np.float64).eps * covariance_scale
    assert np.linalg.norm(constrained_covariance, ord=2) <= covariance_bound


def test_gamma_endpoint_polish_refusal_publishes_honest_nonstationarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused endpoint polish must publish honest nonstationarity."""

    real_solve_terminal = DenseSolverResult.solve_terminal
    reversed_faces: list[tuple[str, ...]] = []

    def reverse_endpoint_newton_direction(
        result: DenseSolverResult,
        rhs: np.ndarray,
    ) -> np.ndarray:
        correction = real_solve_terminal(result, rhs)
        if (
            result.config.coefficient_curvature == "observed"
            and result.config.tolerance == 1.0e-12
            and np.array_equal(rhs, result.terminal_score)
            and efs_module._endpoint_retained_kkt_relative(result) > result.config.tolerance
        ):
            reversed_faces.append(
                () if result.coefficient_face is None else result.coefficient_face.component_names
            )
            return -correction
        return correction

    monkeypatch.setattr(
        DenseSolverResult,
        "solve_terminal",
        reverse_endpoint_newton_direction,
    )
    frame, response, weights = _gamma_exact_face_problem()
    maximum_lambda = 10.0

    model = fit_dense_distributional(
        frame,
        response,
        family=GammaLS(),
        weight_contract=WeightContract("prior"),
        predictors=(
            Predictor("mean", {"x": Spline(kind="cr", n_knots=5)}),
            Predictor("scale", {"group": RandomEffect()}),
        ),
        sample_weight=weights,
        lambdas={"mean:x#wiggle": 0.5, "scale:group#wiggle": maximum_lambda},
        # Fisher scoring leaves the accepted fit less tight than Newton, so the
        # observed endpoint polish has a correction to make and the reversal bites.
        config=DenseSolverConfig(
            tolerance=1.0e-9, max_iterations=150, coefficient_curvature="fisher"
        ),
        efs_config=DistributionalEFSConfig(
            max_iterations=120,
            tolerance=1.0e-3,
            maximum_lambda=maximum_lambda,
            # The endpoint-polish refusal is the Fellner--Schall path's contract.
            outer="efs",
        ),
        retain_rows=True,
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.matched_certified is False
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_endpoint_directions == {}
    assert all(not item.activated_face_components for item in smoothing.history)
    assert reversed_faces == [(), ("scale:group#wiggle",)]
    refusals = tuple(
        item
        for item in smoothing.history
        if item.endpoint_assessment_failure_reason == "analytic_unavailable"
    )
    assert refusals
    for refusal in refusals:
        assert refusal.refused_face_components
        assert len(refusal.coefficient_fit_indices) == 2
        for fit_index, tolerance in zip(
            refusal.coefficient_fit_indices,
            refusal.coefficient_tolerances,
            strict=True,
        ):
            assessment_fit = smoothing.coefficient_fits[fit_index]
            assert assessment_fit.converged
            assert efs_module._endpoint_retained_kkt_relative(assessment_fit) > tolerance
            assert not efs_module._assessment_is_numerically_stationary(
                assessment_fit,
                tolerance,
            )
