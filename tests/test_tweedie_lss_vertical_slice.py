from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import superglm.distributional.efs as efs_module
import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.evidence as smoothing_evidence
import superglm.distributional.smoothing.loop as smoothing_loop
import superglm.distributional.smoothing.objective as smoothing_objective
from superglm.distributional.assembly import dense_predictor_matrices
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.model import DenseDistributionalModel, fit_dense_distributional
from superglm.distributional.prediction_design import build_joint_prediction_design
from superglm.distributional.predictor import Predictor
from superglm.distributional.result import (
    DenseSolverConfig,
    DistributionalEFSConfig,
    DistributionalEFSResult,
)
from superglm.distributional.weights import WeightContract
from superglm.features import Spline
from superglm.reml.efs_update import wood_fasiolo_update
from superglm.solvers.rank import decompose_gram
from superglm.types import LambdaPolicy
from tests._distributional_family_kernels import tweedie as tweedie_kernel
from tests.test_gamma_ls_vertical_slice import (
    _CROSS_FAMILY_OUTER_TOLERANCE,
    _assert_cross_route_fit_provenance,
    _assert_default_none_identity,
    _assert_genuine_multisecant_trials,
    _assert_multisecant_laml_not_worse,
    _assert_same_basin_then_observables,
    _capture_multisecant_decisions,
    _implemented_laml_receipt,
    _ImplementedLAMLReceipt,
)

evaluate_tweedie_rows = tweedie_kernel.evaluate_tweedie_rows

_EPS = np.finfo(np.float64).eps
_FIT_TOLERANCE = 1.0e-7
_PARAMETER_NAMES = ("mean", "dispersion", "power")
_PENALTY_NAMES = (
    "mean:x_mean#wiggle",
    "dispersion:x_dispersion#wiggle",
    "power:x_power#wiggle",
)
_NATURAL_HESSIAN_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (2, 2),
)


def _roundoff_factor(operations: int) -> float:
    if isinstance(operations, bool) or not isinstance(operations, int) or operations < 1:
        raise ValueError("operations must be a positive integer")
    product = operations * _EPS
    if product >= 1.0:
        raise ValueError("operation count lies outside the float64 gamma regime")
    return float(np.nextafter(product / (1.0 - product), np.inf))


def _fsum(values: NDArray[np.float64]) -> float:
    return math.fsum(float(value) for value in np.ravel(values))


@dataclass(frozen=True)
class _TweedieFixture:
    frame: pd.DataFrame
    response: NDArray[np.float64]
    true_mean: NDArray[np.float64]
    true_dispersion: NDArray[np.float64]
    true_power: NDArray[np.float64]
    counts: NDArray[np.int64]


def _compound_poisson_gamma_fixture(*, n_rows: int = 720) -> _TweedieFixture:
    """Generate the mixed law without importing a production Tweedie evaluator."""
    rng = np.random.default_rng(2026082804)
    axis = np.linspace(-1.0, 1.0, n_rows)
    x_mean = rng.permutation(axis)
    x_dispersion = rng.permutation(axis)
    x_power = rng.permutation(axis)

    mean = np.exp(0.55 + 0.45 * np.sin(np.pi * x_mean) + 0.15 * x_mean)
    dispersion = np.exp(-0.05 + 0.28 * np.cos(np.pi * x_dispersion) - 0.12 * x_dispersion)
    power = 1.5 + 0.16 * np.sin(np.pi * x_power) + 0.04 * x_power

    r = power - 1.0
    s = 2.0 - power
    rate = mean**s / (dispersion * s)
    jump_shape = s / r
    jump_scale = dispersion * r * mean**r
    counts = rng.poisson(rate)
    response = np.zeros(n_rows, dtype=np.float64)
    positive = counts > 0
    response[positive] = rng.gamma(
        jump_shape[positive] * counts[positive],
        jump_scale[positive],
    )
    return _TweedieFixture(
        frame=pd.DataFrame(
            {
                "x_mean": x_mean,
                "x_dispersion": x_dispersion,
                "x_power": x_power,
            }
        ),
        response=response,
        true_mean=mean,
        true_dispersion=dispersion,
        true_power=power,
        counts=counts,
    )


def _predictors(
    *,
    n_knots: int = 5,
    estimate_smoothing: bool = False,
) -> tuple[Predictor, Predictor, Predictor]:
    lambda_policy = LambdaPolicy.estimate() if estimate_smoothing else None
    return (
        Predictor(
            "mean",
            {
                "x_mean": Spline(
                    kind="cr",
                    n_knots=n_knots,
                    knot_strategy="quantile_rows",
                    lambda_policy=lambda_policy,
                )
            },
        ),
        Predictor(
            "dispersion",
            {
                "x_dispersion": Spline(
                    kind="cr",
                    n_knots=n_knots,
                    knot_strategy="quantile_rows",
                    lambda_policy=lambda_policy,
                )
            },
        ),
        Predictor(
            "power",
            {
                "x_power": Spline(
                    kind="cr",
                    n_knots=n_knots,
                    knot_strategy="quantile_rows",
                    lambda_policy=lambda_policy,
                )
            },
        ),
    )


def _fixed_lambdas() -> dict[str, float]:
    return dict.fromkeys(_PENALTY_NAMES, 1.0)


def _fit_tweedie(
    fixture: _TweedieFixture,
    *,
    semantics: str,
    sample_weight: NDArray[np.float64] | None = None,
    initial: NDArray[np.float64] | None = None,
    n_knots: int = 5,
) -> DenseDistributionalModel:
    return fit_dense_distributional(
        fixture.frame,
        fixture.response,
        family=TweedieLSS(),
        predictors=_predictors(n_knots=n_knots),
        weight_contract=WeightContract(semantics),
        sample_weight=sample_weight,
        config=DenseSolverConfig(
            coefficient_curvature="observed",
            tolerance=_FIT_TOLERANCE,
        ),
        lambdas=_fixed_lambdas(),
        initial=initial,
        discrete=False,
        chunk_size=None,
    )


def _fit_prior(
    fixture: _TweedieFixture,
    *,
    initial: NDArray[np.float64] | None = None,
) -> DenseDistributionalModel:
    return _fit_tweedie(fixture, semantics="prior", initial=initial)


def _fit_automatic_prior(fixture: _TweedieFixture) -> DenseDistributionalModel:
    return fit_dense_distributional(
        fixture.frame,
        fixture.response,
        family=TweedieLSS(),
        predictors=_predictors(estimate_smoothing=True),
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(
            coefficient_curvature="observed",
            tolerance=_FIT_TOLERANCE,
        ),
        lambdas=_fixed_lambdas(),
        efs_config=DistributionalEFSConfig(
            max_iterations=120,
            tolerance=1.0e-4,
            max_log_step=1.0,
            max_backtracks=6,
            objective_tolerance=1.0e-9,
            # This slice pins the Fellner--Schall stop and its receipts.
            outer="efs",
        ),
        discrete=False,
        chunk_size=None,
    )


@dataclass(frozen=True)
class _FitDiagnostics:
    objective: float
    objective_error: float
    penalized_score: NDArray[np.float64]
    score_error: NDArray[np.float64]
    normalized_kkt: float
    kkt_envelope: float
    expected_penalty: NDArray[np.float64]
    expected_data_curvature: NDArray[np.float64]
    data_curvature_error: NDArray[np.float64]
    expected_penalized_curvature: NDArray[np.float64]
    penalized_curvature_error: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvalue_error: float
    rank_cutoff: float
    rank: int
    expected_covariance: NDArray[np.float64]
    inverse_norm: float
    condition: float
    expected_log_pdet: float
    log_pdet_error: float
    telemetry_condition: float
    telemetry_condition_error: float
    inverse_agreement_error: float
    full_rank_residual: float
    moore_penrose_residual: float
    inverse_residual_envelope: float
    covariance_error_envelope: float
    solve_residual: float


@dataclass(frozen=True)
class _PenaltySpectrum:
    rank: int
    log_pdet: float
    log_pdet_error: float


def _independent_penalty(model: DenseDistributionalModel) -> NDArray[np.float64]:
    """Embed the named component blocks with their published fitted lambdas."""
    layout = model.layout
    assert tuple(layout.penalty_names) == _PENALTY_NAMES
    assert tuple(model.lambdas) == _PENALTY_NAMES
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    for expected_name, component in zip(_PENALTY_NAMES, layout.penalties, strict=True):
        assert component.name == expected_name
        assert component.penalty_kind == "dense"
        assert component.repeat_count == 1
        assert component.block_width is None
        assert component.omega_ssp is not None
        block = np.asarray(component.omega_ssp, dtype=np.float64)
        component_slice = component.group_sl
        block_width = component_slice.stop - component_slice.start
        assert block.shape == (block_width, block_width)
        lambda_value = float(model.lambdas[expected_name])
        assert math.isfinite(lambda_value) and lambda_value > 0.0
        penalty[component_slice, component_slice] += lambda_value * block
    assert np.array_equal(penalty, penalty.T)
    return penalty


def _independent_penalty_spectrum(penalty: NDArray[np.float64]) -> _PenaltySpectrum:
    """Certify the positive spectrum without production rank/logdet helpers."""
    symmetric = 0.5 * (penalty + penalty.T)
    assert np.array_equal(symmetric, penalty)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    width = len(eigenvalues)
    spectral_scale = max(1.0, float(np.linalg.norm(symmetric, ord=2)))
    eigenvalue_error = float(
        np.nextafter(
            _roundoff_factor(128 * width**2 + 1) * spectral_scale,
            np.inf,
        )
    )
    assert float(np.min(eigenvalues, initial=0.0)) >= -eigenvalue_error
    ranks = {
        int(np.count_nonzero(eigenvalues > multiplier * eigenvalue_error))
        for multiplier in (1.0, 8.0, 64.0, 512.0)
    }
    assert len(ranks) == 1, f"penalty rank is not separated from numerical zero: {ranks}"
    rank = ranks.pop()
    assert 0 < rank < width
    positive = eigenvalues[-rank:]
    assert positive[0] > 512.0 * eigenvalue_error
    log_values = np.log(positive)
    log_pdet = _fsum(log_values)
    log_pdet_error = float(
        np.nextafter(
            _fsum(-np.log1p(-eigenvalue_error / positive))
            + _roundoff_factor(64 * width**2 + 32 * rank + 1) * (1.0 + _fsum(np.abs(log_values))),
            np.inf,
        )
    )
    return _PenaltySpectrum(
        rank=rank,
        log_pdet=log_pdet,
        log_pdet_error=log_pdet_error,
    )


def _independent_eta_derivatives(
    model: DenseDistributionalModel,
    natural_score: NDArray[np.float64],
    natural_hessian: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply the inverse-link chain rule with a locally declared channel order."""
    theta = model.result.theta
    power_span = model.family.power_upper - model.family.power_lower
    power_probability = (theta[:, 2] - model.family.power_lower) / power_span
    first = np.column_stack(
        (
            theta[:, 0],
            theta[:, 1],
            power_span * power_probability * (1.0 - power_probability),
        )
    )
    second = np.column_stack(
        (
            theta[:, 0],
            theta[:, 1],
            first[:, 2] * (1.0 - 2.0 * power_probability),
        )
    )
    score_eta = natural_score * first
    curvature_eta = np.empty_like(natural_hessian)
    curvature_scale = np.empty_like(natural_hessian)
    for packed_index, (left, right) in enumerate(_NATURAL_HESSIAN_PAIRS):
        transformed = natural_hessian[:, packed_index] * first[:, left] * first[:, right]
        scale = np.abs(natural_hessian[:, packed_index]) * np.abs(first[:, left] * first[:, right])
        if left == right:
            diagonal_chain = natural_score[:, left] * second[:, left]
            transformed = transformed + diagonal_chain
            scale = scale + np.abs(diagonal_chain)
        curvature_eta[:, packed_index] = -transformed
        curvature_scale[:, packed_index] = scale
    return score_eta, curvature_eta, curvature_scale


def _independent_data_curvature(
    model: DenseDistributionalModel,
    curvature_eta: NDArray[np.float64],
    curvature_scale: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble every dense signed block without generic transform/assembly code."""
    layout = model.layout
    designs = dense_predictor_matrices(layout)
    n_rows = len(curvature_eta)
    width = layout.n_coefficients
    expected = np.zeros((width, width), dtype=np.float64)
    error = np.zeros_like(expected)
    for packed_index, (left_index, right_index) in enumerate(_NATURAL_HESSIAN_PAIRS):
        left_state, right_state = (
            layout.predictors[left_index],
            layout.predictors[right_index],
        )
        left_design, right_design = designs[left_index], designs[right_index]
        block = np.empty((left_design.shape[1], right_design.shape[1]), dtype=np.float64)
        block_error = np.empty_like(block)
        operations = 64 * n_rows + 32 * (left_design.shape[1] + right_design.shape[1]) + 64
        gamma = _roundoff_factor(operations)
        for left_column in range(left_design.shape[1]):
            for right_column in range(right_design.shape[1]):
                design_product = left_design[:, left_column] * right_design[:, right_column]
                terms = design_product * curvature_eta[:, packed_index]
                scale_terms = np.abs(design_product) * curvature_scale[:, packed_index]
                block[left_column, right_column] = _fsum(terms)
                block_error[left_column, right_column] = np.nextafter(
                    gamma * _fsum(scale_terms),
                    np.inf,
                )
        expected[left_state.coefficient_slice, right_state.coefficient_slice] = block
        error[left_state.coefficient_slice, right_state.coefficient_slice] = block_error
        if left_index != right_index:
            expected[right_state.coefficient_slice, left_state.coefficient_slice] = block.T
            error[right_state.coefficient_slice, left_state.coefficient_slice] = block_error.T
    assert np.array_equal(expected, expected.T)
    return expected, error


def _independent_fit_diagnostics(
    model: DenseDistributionalModel,
    response: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    semantics: str,
) -> _FitDiagnostics:
    """Reconstruct value, score, curvature, penalty, rank, and covariance."""
    result = model.result
    layout = model.layout
    n_rows = len(response)
    width = layout.n_coefficients
    row = evaluate_tweedie_rows(
        response,
        result.theta[:, 0],
        result.theta[:, 1],
        result.theta[:, 2],
        weights,
        semantics,
        derivative_order=2,
    )
    assert row.score is not None
    assert row.hessian_packed is not None
    score_eta, curvature_eta, curvature_scale = _independent_eta_derivatives(
        model,
        row.score,
        row.hessian_packed,
    )
    expected_data_curvature, data_curvature_error = _independent_data_curvature(
        model,
        curvature_eta,
        curvature_scale,
    )
    expected_penalty = _independent_penalty(model)
    expected_penalized_curvature = expected_data_curvature + expected_penalty
    assert np.array_equal(expected_penalized_curvature, expected_penalized_curvature.T)
    penalized_curvature_error = np.nextafter(
        data_curvature_error
        + _roundoff_factor(4) * (np.abs(expected_data_curvature) + np.abs(expected_penalty)),
        np.inf,
    )

    designs = dense_predictor_matrices(layout)
    data_score = np.empty(width, dtype=np.float64)
    data_scale = np.empty(width, dtype=np.float64)
    for parameter_index, (state, design) in enumerate(zip(layout.predictors, designs, strict=True)):
        for local_index, global_index in enumerate(
            range(state.coefficient_slice.start, state.coefficient_slice.stop)
        ):
            terms = design[:, local_index] * score_eta[:, parameter_index]
            data_score[global_index] = _fsum(terms)
            data_scale[global_index] = _fsum(np.abs(terms))

    penalty_terms = expected_penalty * result.coefficients[None, :]
    penalty_score = np.array([_fsum(row_terms) for row_terms in penalty_terms])
    penalized_score = data_score - penalty_score
    score_error = np.nextafter(
        _roundoff_factor(4 * n_rows + 4 * width + 16)
        * (data_scale + np.array([_fsum(np.abs(row_terms)) for row_terms in penalty_terms])),
        np.inf,
    )

    likelihood = _fsum(row.log_likelihood)
    quadratic_terms = result.coefficients[:, None] * penalty_terms
    penalty_value = 0.5 * _fsum(quadratic_terms)
    objective = likelihood - penalty_value
    objective_scale = _fsum(np.abs(row.log_likelihood)) + 0.5 * _fsum(np.abs(quadratic_terms))
    objective_error = float(
        np.nextafter(
            _roundoff_factor(4 * n_rows + 4 * width**2 + 32) * objective_scale,
            np.inf,
        )
    )

    eigenvalues, eigenvectors = np.linalg.eigh(expected_penalized_curvature)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    rank_cutoff = float(np.nextafter(width * _EPS * spectral_scale, np.inf))
    retained = eigenvalues > rank_cutoff
    rank = int(np.count_nonzero(retained))
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    expected_covariance = (retained_vectors / retained_values) @ retained_vectors.T
    expected_covariance = 0.5 * (expected_covariance + expected_covariance.T)
    solved_covariance = np.linalg.solve(expected_penalized_curvature, np.eye(width))
    inverse_norm = float(np.linalg.norm(solved_covariance, ord=np.inf))
    condition = float(retained_values[-1] / retained_values[0])
    inverse_roundoff = (
        _roundoff_factor(64 * width**3 + 64 * width**2 + 1)
        * (1.0 + condition)
        * max(1.0, inverse_norm)
    )
    inverse_agreement_error = float(
        np.linalg.norm(expected_covariance - solved_covariance, ord=np.inf)
    )
    identity = np.eye(width)
    full_rank_residual = max(
        float(
            np.linalg.norm(
                expected_penalized_curvature @ solved_covariance - identity,
                ord=np.inf,
            )
        ),
        float(
            np.linalg.norm(
                solved_covariance @ expected_penalized_curvature - identity,
                ord=np.inf,
            )
        ),
    )
    left_projection = expected_penalized_curvature @ expected_covariance
    right_projection = expected_covariance @ expected_penalized_curvature
    moore_penrose_residual = max(
        float(
            np.linalg.norm(
                expected_penalized_curvature @ expected_covariance @ expected_penalized_curvature
                - expected_penalized_curvature,
                ord=np.inf,
            )
            / (1.0 + np.linalg.norm(expected_penalized_curvature, ord=np.inf))
        ),
        float(
            np.linalg.norm(
                expected_covariance @ expected_penalized_curvature @ expected_covariance
                - expected_covariance,
                ord=np.inf,
            )
            / (1.0 + inverse_norm)
        ),
        float(np.linalg.norm(left_projection - left_projection.T, ord=np.inf)),
        float(np.linalg.norm(right_projection - right_projection.T, ord=np.inf)),
    )
    inverse_residual_envelope = (
        _roundoff_factor(128 * width**3 + 128 * width**2 + 1) * (1.0 + condition) ** 2 * width
    )
    h_error_norm = float(np.linalg.norm(penalized_curvature_error, ord=np.inf))
    perturbation_ratio = inverse_norm * h_error_norm
    assert perturbation_ratio < 0.5
    covariance_error_envelope = float(
        np.nextafter(
            inverse_norm * perturbation_ratio / (1.0 - perturbation_ratio) + 2.0 * inverse_roundoff,
            np.inf,
        )
    )
    log_pdet_perturbation = h_error_norm / retained_values[0]
    assert log_pdet_perturbation < 0.5
    log_eigenvalues = np.log(retained_values)
    expected_log_pdet = _fsum(log_eigenvalues)
    log_pdet_error = float(
        np.nextafter(
            rank * -math.log1p(-log_pdet_perturbation)
            + _roundoff_factor(128 * width**3 + 32 * width + 1)
            * (1.0 + _fsum(np.abs(log_eigenvalues))),
            np.inf,
        )
    )
    diagonal = np.diag(expected_penalized_curvature)
    diagonal_error = np.diag(penalized_curvature_error)
    assert np.all(diagonal > diagonal_error)
    diagonal_scale = np.sqrt(diagonal)
    equilibrated = expected_penalized_curvature / np.outer(diagonal_scale, diagonal_scale)
    equilibrated_inverse = np.linalg.solve(equilibrated, np.eye(width))
    equilibrated_condition_one = float(
        np.linalg.norm(equilibrated, ord=1) * np.linalg.norm(equilibrated_inverse, ord=1)
    )
    telemetry_condition = math.sqrt(equilibrated_condition_one)
    diagonal_relative_error = float(np.max(diagonal_error / diagonal))
    equilibrated_error = penalized_curvature_error / np.outer(
        diagonal_scale, diagonal_scale
    ) + np.abs(equilibrated) * diagonal_relative_error / (1.0 - diagonal_relative_error)
    equilibrated_perturbation = float(
        np.linalg.norm(equilibrated_inverse, ord=1) * np.linalg.norm(equilibrated_error, ord=1)
    )
    assert equilibrated_perturbation < 0.5
    telemetry_condition_error = float(
        np.nextafter(
            telemetry_condition
            * (
                equilibrated_perturbation
                + np.linalg.norm(equilibrated_error, ord=1) / np.linalg.norm(equilibrated, ord=1)
            )
            / (1.0 - equilibrated_perturbation)
            + _roundoff_factor(128 * width**3 + 1) * (1.0 + telemetry_condition) ** 2,
            np.inf,
        )
    )
    eigenvalue_error = float(
        np.nextafter(
            h_error_norm
            + _roundoff_factor(32 * width**2 + 1)
            * float(np.linalg.norm(expected_penalized_curvature, ord=2)),
            np.inf,
        )
    )

    solve_residual = 0.0 if not result.history else result.history[-1].solve_residual
    arithmetic_scale = max(
        1.0,
        float(np.linalg.norm(data_scale, ord=np.inf)),
        float(np.linalg.norm(expected_penalty, ord=np.inf))
        * max(1.0, float(np.linalg.norm(result.coefficients, ord=np.inf))),
    )
    denominator = 1.0 + abs(objective)
    kkt_roundoff = float(np.linalg.norm(score_error, ord=np.inf)) / denominator
    solve_envelope = (
        (width + 1.0)
        * (1.0 + condition)
        * max(solve_residual, _roundoff_factor(8 * width**3 + 1))
        * arithmetic_scale
        / denominator
    )
    kkt_envelope = float(
        np.nextafter(
            kkt_roundoff
            + solve_envelope
            + _roundoff_factor(64 * (n_rows * width + width**2 + width**3)),
            np.inf,
        )
    )
    normalized_kkt = float(np.linalg.norm(penalized_score, ord=np.inf) / denominator)
    return _FitDiagnostics(
        objective=objective,
        objective_error=objective_error,
        penalized_score=penalized_score,
        score_error=score_error,
        normalized_kkt=normalized_kkt,
        kkt_envelope=kkt_envelope,
        expected_penalty=expected_penalty,
        expected_data_curvature=expected_data_curvature,
        data_curvature_error=data_curvature_error,
        expected_penalized_curvature=expected_penalized_curvature,
        penalized_curvature_error=penalized_curvature_error,
        eigenvalues=eigenvalues,
        eigenvalue_error=eigenvalue_error,
        rank_cutoff=rank_cutoff,
        rank=rank,
        expected_covariance=expected_covariance,
        inverse_norm=inverse_norm,
        condition=condition,
        expected_log_pdet=expected_log_pdet,
        log_pdet_error=log_pdet_error,
        telemetry_condition=telemetry_condition,
        telemetry_condition_error=telemetry_condition_error,
        inverse_agreement_error=inverse_agreement_error,
        full_rank_residual=full_rank_residual,
        moore_penrose_residual=moore_penrose_residual,
        inverse_residual_envelope=inverse_residual_envelope,
        covariance_error_envelope=covariance_error_envelope,
        solve_residual=solve_residual,
    )


def _assert_independent_fit_geometry(
    model: DenseDistributionalModel,
    diagnostics: _FitDiagnostics,
    *,
    local_width: int,
    require_unit_lambdas: bool = True,
) -> None:
    result = model.result
    assert tuple(state.name for state in model.layout.predictors) == _PARAMETER_NAMES
    assert tuple(
        state.coefficient_slice.stop - state.coefficient_slice.start
        for state in model.layout.predictors
    ) == (local_width, local_width, local_width)
    assert tuple(model.layout.penalty_names) == _PENALTY_NAMES
    if require_unit_lambdas:
        assert dict(model.lambdas) == _fixed_lambdas()
    assert np.array_equal(result.penalty, diagnostics.expected_penalty), (
        "published aggregate penalty differs from the named lambda-scaled component blocks"
    )
    assert np.all(
        np.abs(result.terminal_data_curvature - diagnostics.expected_data_curvature)
        <= diagnostics.data_curvature_error
    ), "terminal data curvature differs from the independent row/block reconstruction"
    assert np.all(
        np.abs(result.terminal_penalized_curvature - diagnostics.expected_penalized_curvature)
        <= diagnostics.penalized_curvature_error
    ), "terminal penalized curvature differs from independent Kobs + S"

    assert diagnostics.rank == model.layout.n_coefficients
    assert diagnostics.eigenvalues[0] > diagnostics.rank_cutoff
    terminal = result.terminal_curvature
    published_spectrum = np.linalg.eigvalsh(result.terminal_penalized_curvature)
    assert np.all(
        np.abs(published_spectrum - diagnostics.eigenvalues) <= diagnostics.eigenvalue_error
    )
    assert result.terminal_rank.rank == model.inference.rank == model.fitted_result.rank
    assert result.terminal_rank.rank == terminal.rank == diagnostics.rank
    assert abs(terminal.minimum_eigenvalue - diagnostics.eigenvalues[0]) <= (
        diagnostics.eigenvalue_error
    )
    assert terminal.condition_estimate is not None
    assert terminal.condition_estimate == result.terminal_rank.pre_truncation_condition
    # Telemetry is a LAPACK norm estimate, not NumPy's exact one-norm condition.
    # The sqrt(width) floor is the norm-equivalence bracket for the independently
    # equilibrated matrix; the additive term propagates H and arithmetic error.
    condition_floor = max(
        1.0,
        diagnostics.telemetry_condition / math.sqrt(model.layout.n_coefficients)
        - diagnostics.telemetry_condition_error,
    )
    condition_ceiling = diagnostics.telemetry_condition + diagnostics.telemetry_condition_error
    assert condition_floor <= terminal.condition_estimate <= condition_ceiling

    rank_covariance = result.terminal_rank.pseudo_inverse()
    rank_covariance_gap = float(
        np.linalg.norm(rank_covariance - diagnostics.expected_covariance, ord=np.inf)
    )
    assert rank_covariance_gap <= diagnostics.covariance_error_envelope, (
        "terminal rank pseudo-inverse differs from the independent inverse"
    )
    rank_model_covariance_gap = float(
        np.linalg.norm(rank_covariance - model.covariance, ord=np.inf)
    )
    assert rank_model_covariance_gap <= 2.0 * diagnostics.covariance_error_envelope, (
        "terminal rank pseudo-inverse differs from the published covariance"
    )

    width = model.layout.n_coefficients
    expected_hessian = diagnostics.expected_penalized_curvature
    hessian_norm = float(np.linalg.norm(expected_hessian, ord=np.inf))
    rank_covariance_norm = float(np.linalg.norm(rank_covariance, ord=np.inf))
    product_roundoff = _roundoff_factor(8 * width + 1) * max(
        1.0,
        hessian_norm * rank_covariance_norm,
    )
    rank_residual_envelope = float(
        np.nextafter(
            hessian_norm * diagnostics.covariance_error_envelope + product_roundoff,
            np.inf,
        )
    )
    identity = np.eye(width)
    left_projection = expected_hessian @ rank_covariance
    right_projection = rank_covariance @ expected_hessian
    rank_full_residual = max(
        float(np.linalg.norm(left_projection - identity, ord=np.inf)),
        float(np.linalg.norm(right_projection - identity, ord=np.inf)),
    )
    assert rank_full_residual <= rank_residual_envelope, (
        "terminal rank pseudo-inverse fails the independent full-rank residual"
    )
    rank_moore_penrose_residual = max(
        float(
            np.linalg.norm(
                expected_hessian @ rank_covariance @ expected_hessian - expected_hessian,
                ord=np.inf,
            )
            / (1.0 + hessian_norm)
        ),
        float(
            np.linalg.norm(
                rank_covariance @ expected_hessian @ rank_covariance - rank_covariance,
                ord=np.inf,
            )
            / (1.0 + rank_covariance_norm)
        ),
        float(np.linalg.norm(left_projection - left_projection.T, ord=np.inf)),
        float(np.linalg.norm(right_projection - right_projection.T, ord=np.inf)),
    )
    assert rank_moore_penrose_residual <= 4.0 * rank_residual_envelope, (
        "terminal rank pseudo-inverse fails the independent Moore-Penrose residual"
    )

    probe = np.linspace(-1.0, 1.0, width, dtype=np.float64)
    expected_solution = np.linalg.solve(expected_hessian, probe)
    rank_solution = result.terminal_rank.solve(probe)
    solve_error_envelope = float(
        np.nextafter(
            diagnostics.covariance_error_envelope * float(np.linalg.norm(probe, ord=np.inf))
            + _roundoff_factor(64 * width**3 + 1)
            * (1.0 + diagnostics.condition)
            * max(1.0, float(np.linalg.norm(expected_solution, ord=np.inf))),
            np.inf,
        )
    )
    assert float(np.linalg.norm(rank_solution - expected_solution, ord=np.inf)) <= (
        solve_error_envelope
    ), "terminal rank solve differs from the independent NumPy solve"
    solve_residual_envelope = float(
        np.nextafter(
            hessian_norm * solve_error_envelope
            + _roundoff_factor(8 * width + 1)
            * max(
                1.0,
                hessian_norm * float(np.linalg.norm(rank_solution, ord=np.inf)),
                float(np.linalg.norm(probe, ord=np.inf)),
            ),
            np.inf,
        )
    )
    assert float(np.linalg.norm(expected_hessian @ rank_solution - probe, ord=np.inf)) <= (
        solve_residual_envelope
    ), "terminal rank solve fails the independent Hessian residual"
    assert abs(result.terminal_rank.log_pdet - diagnostics.expected_log_pdet) <= (
        diagnostics.log_pdet_error
    ), "terminal rank log-pdet differs from the independent positive eigenspectrum"

    assert diagnostics.inverse_agreement_error <= diagnostics.inverse_residual_envelope
    assert diagnostics.full_rank_residual <= diagnostics.inverse_residual_envelope
    assert diagnostics.moore_penrose_residual <= diagnostics.inverse_residual_envelope
    np.testing.assert_array_equal(model.covariance, model.inference.covariance)
    assert (
        float(np.linalg.norm(model.covariance - diagnostics.expected_covariance, ord=np.inf))
        <= diagnostics.covariance_error_envelope
    ), "published covariance differs from the independent NumPy inverse"

    assert result.converged is True
    assert diagnostics.normalized_kkt <= result.config.tolerance + diagnostics.kkt_envelope
    assert abs(result.penalized_optimizing_log_likelihood - diagnostics.objective) <= (
        diagnostics.objective_error
    )
    assert np.all(
        np.abs(result.terminal_score - diagnostics.penalized_score) <= diagnostics.score_error
    )
    assert terminal.requested_source == "observed"
    assert terminal.actual_source == "observed"
    assert terminal.fallback_count == 0


@dataclass(frozen=True)
class _ForwardEnvelope:
    coefficient_radius: float
    parameter_radius: NDArray[np.float64]
    objective_radius: float


def _forward_envelope(
    model: DenseDistributionalModel,
    frame: pd.DataFrame,
    diagnostics: _FitDiagnostics,
) -> _ForwardEnvelope:
    width = model.layout.n_coefficients
    score_upper = float(
        np.linalg.norm(diagnostics.penalized_score, ord=np.inf)
        + np.linalg.norm(diagnostics.score_error, ord=np.inf)
    )
    curvature_uncertainty = diagnostics.inverse_norm * float(
        np.linalg.norm(diagnostics.penalized_curvature_error, ord=np.inf)
    )
    assert curvature_uncertainty < 0.5
    coefficient_radius = float(
        np.nextafter(
            diagnostics.inverse_norm * score_upper / (1.0 - curvature_uncertainty)
            + _roundoff_factor(64 * width**3 + 1)
            * (1.0 + diagnostics.condition)
            * max(1.0, float(np.linalg.norm(model.result.coefficients, ord=np.inf))),
            np.inf,
        )
    )

    prediction_design = build_joint_prediction_design(
        frame,
        model.compiled_predictors,
        model.layout,
    )
    parameters = model.predict_parameters(frame)
    parameter_radius = np.empty(3, dtype=np.float64)
    prediction_roundoff = _roundoff_factor(64 * len(frame) * width + 1)
    for parameter_index, name in enumerate(_PARAMETER_NAMES):
        design_scale = float(
            np.max(
                np.sum(np.abs(prediction_design.local[name]), axis=1),
                initial=0.0,
            )
        )
        eta_radius = design_scale * coefficient_radius
        if parameter_index < 2:
            link_forward_radius = float(
                np.max(parameters[:, parameter_index]) * np.expm1(eta_radius)
            )
        else:
            link_forward_radius = (
                0.25 * (model.family.power_upper - model.family.power_lower) * eta_radius
            )
        parameter_radius[parameter_index] = np.nextafter(
            link_forward_radius
            + prediction_roundoff * max(1.0, float(np.max(np.abs(parameters[:, parameter_index])))),
            np.inf,
        )

    objective_radius = float(
        np.nextafter(
            diagnostics.objective_error
            + width * score_upper * coefficient_radius
            + 0.5
            * width
            * float(np.linalg.norm(diagnostics.expected_penalized_curvature, ord=np.inf))
            * coefficient_radius**2
            + _roundoff_factor(64 * width**3 + 1) * (1.0 + abs(diagnostics.objective)),
            np.inf,
        )
    )
    return _ForwardEnvelope(
        coefficient_radius=coefficient_radius,
        parameter_radius=parameter_radius,
        objective_radius=objective_radius,
    )


def _assert_stationary_fit_pair(
    left: DenseDistributionalModel,
    right: DenseDistributionalModel,
    frame: pd.DataFrame,
    left_diagnostics: _FitDiagnostics,
    right_diagnostics: _FitDiagnostics,
    *,
    local_width: int,
) -> None:
    _assert_independent_fit_geometry(left, left_diagnostics, local_width=local_width)
    _assert_independent_fit_geometry(right, right_diagnostics, local_width=local_width)
    left_forward = _forward_envelope(left, frame, left_diagnostics)
    right_forward = _forward_envelope(right, frame, right_diagnostics)
    left_parameters = left.predict_parameters(frame)
    right_parameters = right.predict_parameters(frame)
    pair_roundoff = _roundoff_factor(64 * len(frame) * left.layout.n_coefficients + 1) * np.maximum(
        1.0,
        np.maximum(
            np.max(np.abs(left_parameters), axis=0), np.max(np.abs(right_parameters), axis=0)
        ),
    )
    parameter_envelope = (
        left_forward.parameter_radius + right_forward.parameter_radius + pair_roundoff
    )
    assert np.all(np.max(np.abs(left_parameters - right_parameters), axis=0) <= parameter_envelope)
    assert abs(left_diagnostics.objective - right_diagnostics.objective) <= (
        left_forward.objective_radius + right_forward.objective_radius
    )


def _assert_complete_efs_history(
    smoothing: DistributionalEFSResult,
) -> tuple[int, int, int]:
    """Bind every recorded trial to one outer step and every accepted state."""
    converged_reasons = {"score", "objective_and_step", "objective_and_score"}
    assert smoothing.coefficient_fits
    for fit_index, fit in enumerate(smoothing.coefficient_fits):
        assert fit.converged, f"coefficient fit {fit_index} did not converge"
        assert fit.convergence_reason in converged_reasons, (
            f"coefficient fit {fit_index} has an illegitimate converged reason"
        )
    assert smoothing.iterations == len(smoothing.history)
    assert 0 < smoothing.iterations <= smoothing.config.max_iterations
    attempted_indices: list[int] = []
    accepted_indices = [0]
    accepted_attempts = 0
    total_backtracks = 0
    current_fit_index = 0
    for iteration_number, iteration in enumerate(smoothing.history, start=1):
        assert iteration.iteration == iteration_number
        assert iteration.source_fit_index == current_fit_index
        assert iteration.coefficient_fit_indices
        assert iteration.backtracks == len(iteration.coefficient_fit_indices) - 1
        total_backtracks += iteration.backtracks
        attempted_indices.extend(iteration.coefficient_fit_indices)
        objective_ceiling = iteration.objective_before + (
            smoothing.config.objective_tolerance * (1.0 + abs(iteration.objective_before))
        )
        if iteration.accepted:
            assert iteration.accepted_fit_index == iteration.coefficient_fit_indices[-1]
            assert iteration.accepted_fit_index is not None
            assert smoothing.coefficient_fits[iteration.accepted_fit_index].converged
            assert iteration.objective_after <= objective_ceiling
            current_fit_index = iteration.accepted_fit_index
            accepted_indices.append(iteration.accepted_fit_index)
            accepted_attempts += 1
        else:
            assert iteration.accepted_fit_index is None
            assert iteration.objective_after == iteration.objective_before
            assert dict(iteration.lambdas_after) == dict(iteration.lambdas_before)

    assert tuple(attempted_indices) == tuple(range(1, len(smoothing.coefficient_fits)))
    assert len(smoothing.coefficient_fits) == len(attempted_indices) + 1
    assert smoothing.terminal_fit_index == current_fit_index == accepted_indices[-1]
    assert smoothing.terminal_fit is smoothing.coefficient_fits[accepted_indices[-1]]
    return len(attempted_indices), accepted_attempts, total_backtracks


def _assert_objective_plateau_evidence(smoothing: DistributionalEFSResult) -> None:
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "objective_plateau"
    assert smoothing.terminal_raw_max_log_step <= smoothing.config.tolerance
    assert smoothing.unresolved_upper_bound == ()
    plateau_iterations = smoothing.config.plateau_iterations
    assert len(smoothing.history) >= max(2, plateau_iterations)
    plateau_tail = smoothing.history[-plateau_iterations:]
    assert len(plateau_tail) == plateau_iterations
    for iteration in plateau_tail:
        assert iteration.accepted, "declared plateau tail contains a rejected iteration"
        assert iteration.objective_relative_change <= smoothing.config.plateau_tolerance, (
            "declared plateau objective tail exceeds the configured tolerance"
        )
    assert plateau_tail[-1].max_accepted_log_step <= plateau_tail[-2].max_accepted_log_step, (
        "terminal accepted log step does not satisfy the live contraction predicate"
    )


def _assert_estimated_lambda_movement(
    smoothing: DistributionalEFSResult,
    estimated_names: tuple[str, ...],
) -> None:
    assert estimated_names == _PENALTY_NAMES, "estimated smoothing name coverage is incomplete"
    for name in estimated_names:
        initial = smoothing.initial_lambdas[name]
        terminal = smoothing.lambdas[name]
        comparison_scale = max(1.0, abs(initial), abs(terminal))
        resolution = _roundoff_factor(8) * comparison_scale
        assert resolution > 0.0
        assert abs(terminal - initial) > resolution, (
            f"estimated lambda {name!r} did not move beyond comparison resolution"
        )


def _iteration_cap_status(
    model: DenseDistributionalModel,
    *,
    max_iterations: int,
) -> DistributionalEFSResult:
    smoothing = model.smoothing
    assert smoothing is not None
    assert 0 < max_iterations < smoothing.iterations
    history = smoothing.history[:max_iterations]
    terminal_fit_index = history[-1].accepted_fit_index
    assert terminal_fit_index is not None
    terminal_lambdas = history[-1].lambdas_after
    terminal_fit = smoothing.coefficient_fits[terminal_fit_index]
    components = efs_module._component_states(model.layout, terminal_lambdas)
    estimated_names = efs_module._estimated_names(components)
    update = wood_fasiolo_update(
        components,
        terminal_fit.coefficients,
        terminal_fit.terminal_rank.pseudo_inverse(),
        inverse_scale=1.0,
        max_log_step=smoothing.config.max_log_step,
        minimum_lambda=smoothing.config.minimum_lambda,
        maximum_lambda=smoothing.config.maximum_lambda,
    )
    terminal_raw_max_log_step = max(
        abs(float(update.stationarity_log_residuals[name])) for name in estimated_names
    )
    unresolved_upper_bound = tuple(
        name
        for name in estimated_names
        if terminal_lambdas[name] == smoothing.config.maximum_lambda
        and update.raw_log_steps[name] > 0.0
    )
    return replace(
        smoothing,
        config=replace(smoothing.config, max_iterations=max_iterations),
        lambdas=terminal_lambdas,
        objective=history[-1].objective_after,
        converged=False,
        convergence_reason="max_iterations",
        terminal_raw_max_log_step=terminal_raw_max_log_step,
        unresolved_upper_bound=unresolved_upper_bound,
        iterations=len(history),
        history=history,
        coefficient_fits=smoothing.coefficient_fits[: terminal_fit_index + 1],
        terminal_fit_index=terminal_fit_index,
    )


def _inject_first_outer_nonconverged_backtrack(
    smoothing: DistributionalEFSResult,
) -> DistributionalEFSResult:
    first_accepted_fit = smoothing.coefficient_fits[1]
    failed_trial = replace(
        first_accepted_fit,
        converged=False,
        convergence_reason="max_iterations",
    )
    coefficient_fits = (
        smoothing.coefficient_fits[0],
        failed_trial,
        *smoothing.coefficient_fits[1:],
    )
    history = []
    for iteration in smoothing.history:
        if iteration.iteration == 1:
            assert iteration.coefficient_fit_indices == (1,)
            history.append(
                replace(
                    iteration,
                    backtracks=1,
                    raw_backtracks=1,
                    coefficient_fit_indices=(1, 2),
                    accepted_fit_index=2,
                    coefficient_tolerances=(
                        iteration.coefficient_tolerances[0],
                        iteration.coefficient_tolerances[0],
                    ),
                )
            )
            continue
        assert iteration.accepted_fit_index is not None
        history.append(
            replace(
                iteration,
                source_fit_index=iteration.source_fit_index + 1,
                coefficient_fit_indices=tuple(
                    fit_index + 1 for fit_index in iteration.coefficient_fit_indices
                ),
                accepted_fit_index=iteration.accepted_fit_index + 1,
            )
        )
    return replace(
        smoothing,
        history=tuple(history),
        coefficient_fits=coefficient_fits,
        terminal_fit_index=smoothing.terminal_fit_index + 1,
    )


def test_nonconverged_efs_trial_is_never_compared_as_laml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = np.linspace(0.0, 1.0, 72)
    frame = pd.DataFrame({"x": axis})
    response = 0.3 + 0.7 * np.sin(2.0 * np.pi * axis)
    predictors = (
        Predictor(
            "location",
            {
                "x": Spline(
                    kind="cr",
                    n_knots=5,
                    lambda_policy={"wiggle": LambdaPolicy.estimate()},
                )
            },
        ),
        Predictor("scale", {}),
    )
    original_solver = efs_module.fit_dense_fixed_lambda
    solver_calls = 0

    def nonconverged_first_trial(*args, **kwargs):
        nonlocal solver_calls
        result = original_solver(*args, **kwargs)
        solver_calls += 1
        if solver_calls == 2:
            return replace(result, converged=False, convergence_reason="max_iterations")
        return result

    original_objective = efs_module.joint_laplace_objective
    objective_fits = []

    def guarded_objective(fit, *args, **kwargs):
        objective_fits.append(fit)
        if not fit.converged:
            raise AssertionError("a nonconverged outer trial reached the LAML objective")
        return original_objective(fit, *args, **kwargs)

    monkeypatch.setattr(smoothing_authority, "fit_dense_fixed_lambda", nonconverged_first_trial)
    monkeypatch.setattr(smoothing_loop, "fit_dense_fixed_lambda", nonconverged_first_trial)
    monkeypatch.setattr(efs_module, "fit_dense_fixed_lambda", nonconverged_first_trial)
    monkeypatch.setattr(smoothing_objective, "joint_laplace_objective", guarded_objective)
    monkeypatch.setattr(efs_module, "joint_laplace_objective", guarded_objective)

    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(),
        weight_contract=WeightContract(semantics="prior"),
        predictors=predictors,
        lambdas={"location:x#wiggle": 0.5},
        config=DenseSolverConfig(
            tolerance=1.0e-8,
            coefficient_curvature="observed",
        ),
        efs_config=DistributionalEFSConfig(
            max_iterations=1,
            tolerance=1.0e-12,
            max_log_step=0.5,
            max_backtracks=0,
        ),
    )

    smoothing = model.smoothing
    assert smoothing is not None
    assert solver_calls == 2
    assert len(objective_fits) == 1
    assert objective_fits[0] is smoothing.coefficient_fits[0]
    assert len(smoothing.coefficient_fits) == 2
    assert smoothing.coefficient_fits[0].converged is True
    assert smoothing.coefficient_fits[1].converged is False
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "coefficient_not_converged"
    assert len(smoothing.history) == 1
    assert smoothing.history[0].accepted is False
    assert smoothing.history[0].coefficient_fit_indices == (1,)


def test_fixed_smoothing_tweedie_lss_fit() -> None:
    fixture = _compound_poisson_gamma_fixture()
    assert len(fixture.response) >= 600
    assert 0 < np.count_nonzero(fixture.response == 0.0) < len(fixture.response)
    assert all(
        float(np.ptp(signal)) > 0.1
        for signal in (fixture.true_mean, fixture.true_dispersion, fixture.true_power)
    )

    model = _fit_prior(fixture)
    result = model.result
    fitted = model.fitted_result
    theta = result.theta
    diagnostics = _independent_fit_diagnostics(
        model,
        fixture.response,
        np.ones(len(fixture.response), dtype=np.float64),
        semantics="prior",
    )

    assert result.execution_backend_identifier == "distributional-dense-v1"
    assert result.resolved_chunk_size is None
    assert result.config.coefficient_curvature == "observed"
    assert np.all(np.isfinite(theta))
    assert np.all(theta[:, :2] > 0.0)
    assert np.all(
        (theta[:, 2] > model.family.power_lower) & (theta[:, 2] < model.family.power_upper)
    )
    assert result.converged is True
    assert fitted.coefficient_converged is True
    assert result.penalized_optimizing_log_likelihood is not None
    assert result.initial_penalized_optimizing_log_likelihood is not None
    normalized_improvement = (
        result.penalized_optimizing_log_likelihood
        - result.initial_penalized_optimizing_log_likelihood
    ) / (1.0 + abs(result.initial_penalized_optimizing_log_likelihood))
    assert normalized_improvement > result.config.tolerance

    _assert_independent_fit_geometry(model, diagnostics, local_width=7)
    assert result.terminal_rank.rank == model.inference.rank == fitted.rank == 21
    assert model.inference.curvature_source == "observed"
    assert np.all(np.isfinite(model.covariance))
    forward = _forward_envelope(model, fixture.frame, diagnostics)
    fitted_variation = np.ptp(theta, axis=0)
    assert fitted_variation[1] > 32.0 * forward.parameter_radius[1]
    assert fitted_variation[2] > 32.0 * forward.parameter_radius[2]
    assert math.isfinite(diagnostics.condition)
    published_kkt = float(
        np.linalg.norm(result.terminal_score, ord=np.inf)
        / (1.0 + abs(result.penalized_optimizing_log_likelihood))
    )
    assert published_kkt <= result.config.tolerance + diagnostics.kkt_envelope

    identity_rank = decompose_gram(np.eye(model.layout.n_coefficients))
    identity_telemetry = replace(
        result.terminal_curvature,
        rank=model.layout.n_coefficients,
        condition_estimate=1.0,
    )
    mutant_solver = replace(
        result,
        terminal_rank=identity_rank,
        terminal_curvature=identity_telemetry,
    )
    mutant_inference = replace(model.inference, curvature_telemetry=identity_telemetry)
    mutant_result = replace(model.fitted_result, curvature_telemetry=identity_telemetry)
    mutant_state = replace(
        model.fit_state,
        solver_result=mutant_solver,
        inference=mutant_inference,
        result=mutant_result,
    )
    mutant_model = replace(model, _fit_state=mutant_state)
    with pytest.raises(
        AssertionError,
        match="terminal rank pseudo-inverse differs from the independent inverse",
    ):
        _assert_independent_fit_geometry(mutant_model, diagnostics, local_width=7)


def test_automatic_smoothing_tweedie_lss_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _compound_poisson_gamma_fixture()
    model = _fit_automatic_prior(fixture)
    smoothing = model.smoothing
    assert smoothing is not None
    result = model.result
    fitted = model.fitted_result
    unit_weights = np.ones(len(fixture.response), dtype=np.float64)
    diagnostics = _independent_fit_diagnostics(
        model,
        fixture.response,
        unit_weights,
        semantics="prior",
    )

    assert tuple(model.layout.penalty_names) == _PENALTY_NAMES
    for parameter_name in _PARAMETER_NAMES:
        components = tuple(
            component
            for component in model.layout.penalties
            if component.name.startswith(f"{parameter_name}:")
        )
        assert components
        assert any(
            component.lambda_policy is not None and component.lambda_policy.mode == "estimate"
            for component in components
        )
    assert dict(smoothing.initial_lambdas) == _fixed_lambdas()
    assert dict(model.lambdas) == dict(smoothing.lambdas)
    assert all(
        math.isfinite(value)
        and smoothing.config.minimum_lambda <= value <= smoothing.config.maximum_lambda
        for value in smoothing.lambdas.values()
    )
    component_states = efs_module._component_states(
        model.layout,
        smoothing.initial_lambdas,
    )
    estimated_names = efs_module._estimated_names(component_states)
    _assert_estimated_lambda_movement(smoothing, estimated_names)
    original_estimated_names = efs_module._estimated_names
    with monkeypatch.context() as mutation:
        mutation.setattr(
            smoothing_evidence,
            "_estimated_names",
            lambda components: original_estimated_names(components)[:1],
        )
        mutation.setattr(
            efs_module,
            "_estimated_names",
            lambda components: original_estimated_names(components)[:1],
        )
        with pytest.raises(AssertionError, match="estimated smoothing name coverage"):
            _assert_estimated_lambda_movement(
                smoothing,
                efs_module._estimated_names(component_states),
            )

    attempted, accepted, backtracks = _assert_complete_efs_history(smoothing)
    assert attempted == accepted == smoothing.iterations
    assert backtracks == 0
    nonconverged_backtrack = _inject_first_outer_nonconverged_backtrack(smoothing)
    with pytest.raises(AssertionError, match="coefficient fit 1 did not converge"):
        _assert_complete_efs_history(nonconverged_backtrack)
    assert smoothing.terminal_fit is result
    assert result is model.fit_state.solver_result
    assert result.converged is True
    assert smoothing.coefficient_converged is True
    assert smoothing.converged is True
    assert smoothing.convergence_reason == "objective_plateau"
    assert smoothing.config.max_iterations == 120
    assert smoothing.config.tolerance == 1.0e-4
    assert smoothing.config.objective_tolerance == 1.0e-9
    assert smoothing.config.plateau_tolerance == 1.0e-7
    assert smoothing.config.plateau_iterations == 3
    assert smoothing.iterations < smoothing.config.max_iterations
    _assert_objective_plateau_evidence(smoothing)
    genuine_early_cap = _iteration_cap_status(model, max_iterations=10)
    assert genuine_early_cap.config.max_iterations == 10
    assert genuine_early_cap.converged is False
    assert genuine_early_cap.convergence_reason == "max_iterations"
    assert genuine_early_cap.terminal_raw_max_log_step > (genuine_early_cap.config.tolerance)
    with pytest.raises(ValueError, match="fresh convergence"):
        replace(
            genuine_early_cap,
            config=smoothing.config,
            converged=True,
            convergence_reason="objective_plateau",
        )
    assert smoothing.matched_certified is True
    smoothing.assert_matched_certified()

    terminal_components = efs_module._component_states(model.layout, smoothing.lambdas)
    terminal_estimated_names = efs_module._estimated_names(terminal_components)
    terminal_update = wood_fasiolo_update(
        terminal_components,
        smoothing.terminal_fit.coefficients,
        smoothing.terminal_fit.terminal_rank.pseudo_inverse(),
        inverse_scale=1.0,
        max_log_step=smoothing.config.max_log_step,
        minimum_lambda=smoothing.config.minimum_lambda,
        maximum_lambda=smoothing.config.maximum_lambda,
    )
    reconstructed_raw_maximum = max(
        abs(float(terminal_update.stationarity_log_residuals[name]))
        for name in terminal_estimated_names
    )
    reconstruction_envelope = _roundoff_factor(
        64 * (model.layout.n_coefficients**2 + len(terminal_estimated_names) + 1)
    ) * max(
        1.0,
        reconstructed_raw_maximum,
        smoothing.terminal_raw_max_log_step,
    )
    assert abs(smoothing.terminal_raw_max_log_step - reconstructed_raw_maximum) <= (
        reconstruction_envelope
    )
    assert smoothing.unresolved_upper_bound == ()
    if smoothing.convergence_reason == "objective_plateau":
        assert reconstructed_raw_maximum <= (smoothing.config.tolerance + reconstruction_envelope)
    assert fitted.coefficient_converged is True
    assert fitted.smoothing_converged is True
    assert fitted.converged is True
    assert fitted.n_smoothing_iter == smoothing.iterations
    assert fitted.n_inner_iter == sum(fit.iterations for fit in smoothing.coefficient_fits)

    assert all(
        fit.execution_backend_identifier == "distributional-dense-v1"
        and fit.resolved_chunk_size is None
        and fit.config.coefficient_curvature == "observed"
        and fit.terminal_curvature.requested_source == "observed"
        and fit.terminal_curvature.actual_source == "observed"
        and fit.terminal_curvature.reason is None
        and fit.terminal_curvature.fallback_count == 0
        for fit in smoothing.coefficient_fits
    )
    assert np.all(np.isfinite(result.theta))
    assert np.all(result.theta[:, :2] > 0.0)
    assert np.all(
        (result.theta[:, 2] > model.family.power_lower)
        & (result.theta[:, 2] < model.family.power_upper)
    )
    _assert_independent_fit_geometry(
        model,
        diagnostics,
        local_width=7,
        require_unit_lambdas=False,
    )
    assert result.terminal_rank.rank == model.inference.rank == fitted.rank
    assert model.inference.curvature_source == "observed"
    assert np.all(np.isfinite(model.covariance))

    penalty_spectrum = _independent_penalty_spectrum(diagnostics.expected_penalty)
    declared_penalty_rank = 0
    for component in model.layout.penalties:
        integer_rank = int(component.rank)
        assert float(integer_rank) == component.rank
        declared_penalty_rank += integer_rank
    assert penalty_spectrum.rank == declared_penalty_rank
    assert result.parameter_independent_carrier == 0.0
    assert result.penalized_optimizing_log_likelihood is not None
    assert abs(result.penalized_optimizing_log_likelihood - diagnostics.objective) <= (
        diagnostics.objective_error
    )
    independent_laml = -diagnostics.objective + 0.5 * (
        result.terminal_rank.log_pdet - penalty_spectrum.log_pdet
    )
    laml_error = float(
        np.nextafter(
            diagnostics.objective_error
            + 0.5 * penalty_spectrum.log_pdet_error
            + _roundoff_factor(64 * model.layout.n_coefficients + 1)
            * max(
                1.0,
                abs(independent_laml),
                abs(diagnostics.objective),
                abs(result.terminal_rank.log_pdet),
                abs(penalty_spectrum.log_pdet),
            ),
            np.inf,
        )
    )
    assert math.isfinite(smoothing.initial_objective)
    assert math.isfinite(smoothing.objective)
    assert abs(smoothing.objective - independent_laml) <= laml_error
    assert smoothing.objective <= (
        smoothing.initial_objective
        + smoothing.config.objective_tolerance * (1.0 + abs(smoothing.initial_objective))
    )


def _fit_tweedie_cross_family_route(
    fixture: _TweedieFixture,
    *,
    acceleration: str | None,
    lambdas: dict[str, float] | None = None,
    initial: NDArray[np.float64] | None = None,
    outer_tolerance: float = _CROSS_FAMILY_OUTER_TOLERANCE,
    inner_tolerance: float = _FIT_TOLERANCE,
) -> DenseDistributionalModel:
    config_options: dict[str, object] = {
        "max_iterations": 120,
        "tolerance": outer_tolerance,
        "max_log_step": 1.0,
        "max_backtracks": 6,
        "objective_tolerance": _FIT_TOLERANCE,
        # The cross-family receipts describe the Fellner--Schall route.
        "outer": "efs",
    }
    if acceleration is not None:
        config_options["acceleration"] = acceleration
    return fit_dense_distributional(
        fixture.frame,
        fixture.response,
        family=TweedieLSS(),
        predictors=_predictors(n_knots=4, estimate_smoothing=True),
        weight_contract=WeightContract("prior"),
        config=DenseSolverConfig(
            coefficient_curvature="observed",
            tolerance=inner_tolerance,
        ),
        lambdas=_fixed_lambdas() if lambdas is None else lambdas,
        initial=initial,
        efs_config=DistributionalEFSConfig(**config_options),
        retain_rows=False,
        discrete=False,
        chunk_size=None,
    )


def _tweedie_laml_receipt(
    diagnostics: _FitDiagnostics,
) -> _ImplementedLAMLReceipt:
    penalty = _independent_penalty_spectrum(diagnostics.expected_penalty)
    return _implemented_laml_receipt(
        coefficient_objective=diagnostics.objective,
        coefficient_objective_error=diagnostics.objective_error,
        hessian=(diagnostics.expected_log_pdet, diagnostics.log_pdet_error),
        penalty=(penalty.log_pdet, penalty.log_pdet_error),
    )


def test_tweedie_multisecant_cross_family_correctness_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independently assemble KKT over the separately validated production row law.

    The Tweedie series diagnostics remain row-law evidence, not an exact tail
    enclosure. This test reconstructs the joint assembly around that validated law.
    """
    fixture = _compound_poisson_gamma_fixture(n_rows=240)
    default = _fit_tweedie_cross_family_route(fixture, acceleration=None)
    raw = _fit_tweedie_cross_family_route(fixture, acceleration="none")
    decisions = _capture_multisecant_decisions(monkeypatch)
    multisecant = _fit_tweedie_cross_family_route(fixture, acceleration="multisecant")

    _assert_genuine_multisecant_trials(multisecant, decisions)
    _assert_default_none_identity(default, raw, fixture.frame)
    _assert_cross_route_fit_provenance(
        raw,
        multisecant,
        require_rejected_candidate=False,
    )
    unit_weights = np.ones(len(fixture.response), dtype=np.float64)
    diagnostics = tuple(
        _independent_fit_diagnostics(
            model,
            fixture.response,
            unit_weights,
            semantics="prior",
        )
        for model in (raw, multisecant)
    )
    for model, item in zip((raw, multisecant), diagnostics, strict=True):
        _assert_independent_fit_geometry(
            model,
            item,
            local_width=6,
            require_unit_lambdas=False,
        )
    _assert_multisecant_laml_not_worse(
        raw,
        multisecant,
        _tweedie_laml_receipt(diagnostics[0]),
        _tweedie_laml_receipt(diagnostics[1]),
    )

    continuation_tolerance = _CROSS_FAMILY_OUTER_TOLERANCE / 4.0
    continuation_inner_tolerance = _FIT_TOLERANCE / 4.0
    continuations = tuple(
        _fit_tweedie_cross_family_route(
            fixture,
            acceleration="none",
            lambdas=dict(endpoint.lambdas),
            initial=endpoint.coefficients,
            outer_tolerance=continuation_tolerance,
            inner_tolerance=continuation_inner_tolerance,
        )
        for endpoint in (raw, multisecant)
    )
    _assert_same_basin_then_observables(
        raw,
        multisecant,
        continuations[0],
        continuations[1],
        fixture.frame,
    )


def test_fixed_smoothing_tweedie_lss_fit_is_stable_from_an_alternate_start() -> None:
    fixture = _compound_poisson_gamma_fixture()
    reference = _fit_prior(fixture)
    initial = np.zeros(reference.layout.n_coefficients, dtype=np.float64)
    initial[reference.layout.predictor("mean").intercept_index] = math.log(1.25)
    initial[reference.layout.predictor("dispersion").intercept_index] = math.log(1.30)
    power_state = reference.layout.predictor("power")
    initial[power_state.intercept_index] = float(
        power_state.link.link(np.array([1.68], dtype=np.float64))[0]
    )
    alternate = _fit_prior(fixture, initial=initial)
    unit_weights = np.ones(len(fixture.response), dtype=np.float64)
    reference_diagnostics = _independent_fit_diagnostics(
        reference,
        fixture.response,
        unit_weights,
        semantics="prior",
    )
    alternate_diagnostics = _independent_fit_diagnostics(
        alternate,
        fixture.response,
        unit_weights,
        semantics="prior",
    )

    assert reference.result.initial_penalized_optimizing_log_likelihood is not None
    assert alternate.result.initial_penalized_optimizing_log_likelihood is not None
    initial_separation = abs(
        reference.result.initial_penalized_optimizing_log_likelihood
        - alternate.result.initial_penalized_optimizing_log_likelihood
    ) / (
        1.0
        + max(
            abs(reference.result.initial_penalized_optimizing_log_likelihood),
            abs(alternate.result.initial_penalized_optimizing_log_likelihood),
        )
    )
    assert initial_separation > 100.0 * _FIT_TOLERANCE
    _assert_stationary_fit_pair(
        reference,
        alternate,
        fixture.frame,
        reference_diagnostics,
        alternate_diagnostics,
        local_width=7,
    )


def test_frequency_fixed_fit_matches_literal_integer_replication() -> None:
    fixture = _compound_poisson_gamma_fixture(n_rows=144)
    replication_counts = np.tile(
        np.array([1.0, 3.0, 2.0, 4.0, 1.0, 2.0], dtype=np.float64),
        24,
    )
    take = np.repeat(np.arange(len(fixture.response)), replication_counts.astype(np.intp))
    expanded = _TweedieFixture(
        frame=fixture.frame.iloc[take].reset_index(drop=True),
        response=fixture.response[take],
        true_mean=fixture.true_mean[take],
        true_dispersion=fixture.true_dispersion[take],
        true_power=fixture.true_power[take],
        counts=fixture.counts[take],
    )

    frequency = _fit_tweedie(
        fixture,
        semantics="frequency",
        sample_weight=replication_counts,
        n_knots=4,
    )
    literal = _fit_tweedie(
        expanded,
        semantics="frequency",
        sample_weight=np.ones(len(take), dtype=np.float64),
        n_knots=4,
    )
    frequency_diagnostics = _independent_fit_diagnostics(
        frequency,
        fixture.response,
        replication_counts,
        semantics="frequency",
    )
    literal_diagnostics = _independent_fit_diagnostics(
        literal,
        expanded.response,
        np.ones(len(take), dtype=np.float64),
        semantics="frequency",
    )

    assert frequency.fit_state.weight_provenance.likelihood_count == len(take)
    assert literal.fit_state.weight_provenance.likelihood_count == len(take)
    _assert_stationary_fit_pair(
        frequency,
        literal,
        fixture.frame,
        frequency_diagnostics,
        literal_diagnostics,
        local_width=6,
    )
