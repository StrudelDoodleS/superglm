"""Fixed-state fitting and the endpoint authority that certifies a face."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    _assessment_is_numerically_stationary,
)
from superglm.distributional.smoothing.objective import _penalty_lambdas
from superglm.distributional.smoothing.penalty_face import PenaltyFace, build_penalty_face
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import (
    DenseSolverError,
    _DenseObservedReuseSession,
    _fit_dense_fixed_lambda_score_only,
    fit_dense_fixed_lambda,
)
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.solvers.rank import RankDecomposition


def _fit_fixed_state(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
    initial: NDArray,
    config: DenseSolverConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    score_only: bool = False,
    _reuse_session: _DenseObservedReuseSession | None = None,
    _reuse_source: DenseSolverResult | None = None,
) -> DenseSolverResult:
    with measure_phase(phase_recorder, "layout_penalty_assembly"):
        penalty = layout.penalty_matrix(_penalty_lambdas(lambdas, face))
    fit_solver = _fit_dense_fixed_lambda_score_only if score_only else fit_dense_fixed_lambda
    # The session's memoised dense predictor matrices serve every dense fit;
    # result reuse still needs an ordinary, face-free fit with a source.
    reuse_kwargs: dict[str, object] = {"_reuse_session": _reuse_session}
    if not score_only and face is None:
        reuse_kwargs["_reuse_source"] = _reuse_source
    return fit_solver(
        family,
        layout,
        y,
        likelihood_plan,
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        coefficient_face=face,
        **reuse_kwargs,  # ty: ignore[invalid-argument-type] -- correlated dispatch kwargs
    )


def _endpoint_shared_provenance(result: DenseSolverResult) -> tuple[object, ...]:
    return (
        result.family_likelihood_plan_identifier,
        result.execution_backend_identifier,
        result.terminal_curvature.requested_source,
        result.terminal_curvature.actual_source,
        result.terminal_rank.policy_version,
    )


def _optional_penalty_face(
    layout: StackedLayout,
    component_names: tuple[str, ...],
) -> PenaltyFace | None:
    if not component_names:
        return None
    selected = frozenset(component_names)
    ordered = tuple(name for name in layout.penalty_names if name in selected)
    return build_penalty_face(layout, ordered)


def _is_sole_cap_outside_face(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    *,
    component_name: str,
    finite_face: PenaltyFace | None,
    maximum_lambda: float,
) -> bool:
    finite_names = frozenset(() if finite_face is None else finite_face.component_names)
    capped_outside = tuple(
        name
        for name in layout.penalty_names
        if name not in finite_names and lambdas[name] == maximum_lambda
    )
    return capped_outside == (component_name,)


def _face_authority_config(config: DenseSolverConfig) -> DenseSolverConfig:
    """Use one coefficient policy for an active face and its terminal check."""

    return replace(
        config,
        max_iterations=max(config.max_iterations, 150),
        tolerance=min(config.tolerance, 1.0e-12),
        newton_decrement_tolerance=None,
        coefficient_curvature="observed",
    )


def _endpoint_retained_rank(result: DenseSolverResult) -> RankDecomposition | None:
    if result.coefficient_face is None:
        return result.terminal_rank
    return result.terminal_reduced_rank


def _endpoint_retained_score(result: DenseSolverResult) -> NDArray[np.float64]:
    if result.coefficient_face is None:
        return result.terminal_score
    return result.coefficient_face.reduce_vector(result.terminal_score)


def _endpoint_retained_curvature(result: DenseSolverResult) -> NDArray[np.float64]:
    if result.coefficient_face is None:
        return result.terminal_penalized_curvature
    return result.coefficient_face.reduce_matrix(result.terminal_penalized_curvature)


def _endpoint_retained_kkt_relative(result: DenseSolverResult) -> float:
    objective = result.penalized_optimizing_log_likelihood
    assert objective is not None
    retained_score = _endpoint_retained_score(result)
    score_norm = float(np.max(np.abs(retained_score), initial=0.0))
    return score_norm / (1.0 + abs(objective))


def _endpoint_retained_rank_provenance(
    result: DenseSolverResult,
) -> tuple[object, ...] | None:
    rank = _endpoint_retained_rank(result)
    if rank is None:
        return None
    return (
        rank.policy_version,
        rank.method,
        rank.rank,
        rank.rank_truncated,
        rank.used_svd_fallback,
        rank.resolution_limited,
        tuple(int(index) for index in rank.active_columns),
    )


def _endpoint_polish_provenance_matches(
    source: DenseSolverResult,
    polished: DenseSolverResult,
    *,
    config: DenseSolverConfig,
    face: PenaltyFace | None,
) -> bool:
    source_curvature = source.terminal_curvature
    polished_curvature = polished.terminal_curvature
    curvature_provenance = (
        source_curvature.requested_source,
        source_curvature.actual_source,
        source_curvature.fallback_count,
    )
    retained_width = source.coefficients.size if face is None else face.reduced_width
    source_rank = _endpoint_retained_rank(source)
    polished_rank = _endpoint_retained_rank(polished)
    return bool(
        source.config == config
        and polished.config == config
        and source.family_likelihood_plan_identifier == polished.family_likelihood_plan_identifier
        and source.resolved_chunk_size == polished.resolved_chunk_size
        and source.execution_backend_identifier == polished.execution_backend_identifier
        and curvature_provenance == ("observed", "observed", 0)
        and (
            polished_curvature.requested_source,
            polished_curvature.actual_source,
            polished_curvature.fallback_count,
        )
        == curvature_provenance
        and source.coefficient_face is face
        and polished.coefficient_face is face
        and source.coefficients.shape == polished.coefficients.shape
        and source.eta.shape == polished.eta.shape
        and np.array_equal(source.penalty, polished.penalty)
        and source_rank is not None
        and polished_rank is not None
        and source_rank.rank == retained_width
        and polished_rank.rank == retained_width
        and _endpoint_retained_rank_provenance(source)
        == _endpoint_retained_rank_provenance(polished)
    )


def _endpoint_positive_dot(
    score: NDArray,
    correction: NDArray,
    *,
    epsilon: float,
) -> bool:
    width = score.size
    operation_error = 2.0 * width * epsilon
    if operation_error >= 1.0:
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        dot = float(score @ correction)
        product_sum = float(np.abs(score) @ np.abs(correction))
        dot_error = float(
            np.nextafter(
                operation_error / (1.0 - operation_error) * product_sum,
                math.inf,
            )
        )
    return math.isfinite(dot) and math.isfinite(dot_error) and dot > dot_error


def _endpoint_candidate_refit_bound(
    candidate: NDArray,
    polished: DenseSolverResult,
    *,
    tolerance: float,
) -> float | None:
    coefficient_dimension = candidate.size
    dtype = np.result_type(candidate.dtype, polished.coefficients.dtype)
    epsilon = float(np.finfo(dtype).eps)
    operation_error = 64.0 * coefficient_dimension * epsilon
    if operation_error >= 1.0:
        return None
    gamma = operation_error / (1.0 - operation_error)
    candidate_norm = float(np.max(np.abs(candidate), initial=0.0))
    polished_norm = float(np.max(np.abs(polished.coefficients), initial=0.0))
    scale = max(1.0, candidate_norm, polished_norm)
    bound = float(np.nextafter((tolerance + gamma) * scale, math.inf))
    return bound if math.isfinite(bound) else None


def _endpoint_objective_accumulation_bound(
    source: DenseSolverResult,
    polished: DenseSolverResult,
) -> float | None:
    row_count = source.eta.shape[0]
    chunk_count = (
        1
        if source.resolved_chunk_size is None
        else math.ceil(row_count / source.resolved_chunk_size)
    )
    coefficient_dimension = source.coefficients.size
    dimension = max(row_count + chunk_count, coefficient_dimension, 1)
    dtype = np.result_type(source.coefficients.dtype, polished.coefficients.dtype)
    epsilon = float(np.finfo(dtype).eps)
    operation_error = 64.0 * dimension * epsilon
    if operation_error >= 1.0:
        return None
    gamma = operation_error / (1.0 - operation_error)

    scales: list[float] = []
    for result in (source, polished):
        objective = result.penalized_optimizing_log_likelihood
        optimizing = result.optimizing_log_likelihood
        assert objective is not None and optimizing is not None
        with np.errstate(over="ignore", invalid="ignore"):
            penalty_product = 0.5 * float(
                np.abs(result.coefficients) @ np.abs(result.penalty) @ np.abs(result.coefficients)
            )
        scale = max(1.0, abs(objective), abs(optimizing) + penalty_product)
        if not math.isfinite(scale):
            return None
        scales.append(scale)
    bound = float(np.nextafter(gamma * max(scales), math.inf))
    return bound if math.isfinite(bound) else None


def _fit_endpoint_authority_stationary(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
    initial: NDArray,
    config: DenseSolverConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> DenseSolverResult:
    """Require retained-score stationarity from a strict endpoint-authority fit.

    A line search can accept a sub-ULP coefficient step or exhaust its ordinary
    comparisons while its retained score remains material.  Endpoint authority
    has a stricter contract: try one full Newton correction from the fit's
    published terminal system, then retain the refitted state only when it
    independently satisfies that contract.
    """

    fit = _fit_fixed_state(
        family,
        layout,
        y,
        likelihood_plan,
        lambdas=lambdas,
        face=face,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=_reuse_session,
    )
    source_kkt = _endpoint_retained_kkt_relative(fit)
    score_only_refit = bool(
        (
            not fit.converged
            and fit.convergence_reason == "line_search_failed"
            and fit.iterations == 0
        )
        or (fit.converged and fit.convergence_reason == "objective_and_step")
    )
    polishable_source = fit.converged or score_only_refit
    if not polishable_source or source_kkt <= config.tolerance:
        return fit

    retained_rank = _endpoint_retained_rank(fit)
    retained_width = fit.coefficients.size if face is None else face.reduced_width
    curvature = fit.terminal_curvature
    if (
        fit.config != config
        or config.coefficient_curvature != "observed"
        or fit.coefficient_face is not face
        or curvature.requested_source != "observed"
        or curvature.actual_source != "observed"
        or curvature.fallback_count != 0
        or retained_rank is None
        or retained_rank.rank != retained_width
    ):
        return fit
    try:
        raw_terminal_correction = np.asarray(fit.solve_terminal(fit.terminal_score))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return fit
    if (
        raw_terminal_correction.shape != fit.coefficients.shape
        or raw_terminal_correction.dtype.kind != "f"
        or not np.all(np.isfinite(raw_terminal_correction))
    ):
        return fit
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        terminal_correction = np.asarray(
            raw_terminal_correction,
            dtype=fit.coefficients.dtype,
        )
    if terminal_correction.shape != fit.coefficients.shape or not np.all(
        np.isfinite(terminal_correction)
    ):
        return fit

    retained_score = _endpoint_retained_score(fit)
    retained_curvature = _endpoint_retained_curvature(fit)
    retained_correction = (
        terminal_correction if face is None else face.reduce_vector(terminal_correction)
    )
    if (
        retained_score.shape != (retained_width,)
        or retained_correction.shape != (retained_width,)
        or retained_curvature.shape != (retained_width, retained_width)
        or not np.all(np.isfinite(retained_correction))
    ):
        return fit
    with np.errstate(over="ignore", invalid="ignore"):
        system_residual = retained_curvature @ retained_correction - retained_score
        relative_residual = float(
            np.linalg.norm(system_residual, ord=2)
            / max(1.0, float(np.linalg.norm(retained_score, ord=2)))
        )
    epsilon = float(np.finfo(fit.coefficients.dtype).eps)
    if (
        not math.isfinite(relative_residual)
        or relative_residual > config.residual_tolerance
        or not _endpoint_positive_dot(
            retained_score,
            retained_correction,
            epsilon=epsilon,
        )
    ):
        return fit

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        candidate = np.asarray(
            fit.coefficients + terminal_correction,
            dtype=fit.coefficients.dtype,
        )
    if candidate.shape != fit.coefficients.shape or not np.all(np.isfinite(candidate)):
        return fit
    if face is not None:
        try:
            candidate = np.asarray(face.project(candidate), dtype=fit.coefficients.dtype)
        except ValueError:
            return fit
    if candidate.shape != fit.coefficients.shape or not np.all(np.isfinite(candidate)):
        return fit
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_correction = candidate - fit.coefficients
    if (
        candidate_correction.shape != fit.coefficients.shape
        or not np.all(np.isfinite(candidate_correction))
        or np.array_equal(candidate, fit.coefficients)
    ):
        return fit
    applied_retained_correction = (
        candidate_correction if face is None else face.reduce_vector(candidate_correction)
    )
    if not _endpoint_positive_dot(
        retained_score,
        applied_retained_correction,
        epsilon=epsilon,
    ):
        return fit

    try:
        polished = _fit_fixed_state(
            family,
            layout,
            y,
            likelihood_plan,
            lambdas=lambdas,
            face=face,
            initial=candidate,
            config=config,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
            _reuse_session=_reuse_session,
            score_only=score_only_refit,
        )
    except (DenseSolverError, ValueError, np.linalg.LinAlgError):
        return fit
    if not polished.converged or not _endpoint_polish_provenance_matches(
        fit,
        polished,
        config=config,
        face=face,
    ):
        return fit
    polished_kkt = _endpoint_retained_kkt_relative(polished)
    if (
        not _assessment_is_numerically_stationary(polished, config.tolerance)
        or polished_kkt >= source_kkt
    ):
        return fit

    coefficient_bound = _endpoint_candidate_refit_bound(
        candidate,
        polished,
        tolerance=config.tolerance,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        coefficient_movement = float(np.max(np.abs(polished.coefficients - candidate), initial=0.0))
    if (
        coefficient_bound is None
        or not math.isfinite(coefficient_movement)
        or coefficient_movement > coefficient_bound
    ):
        return fit

    source_objective = fit.penalized_optimizing_log_likelihood
    polished_objective = polished.penalized_optimizing_log_likelihood
    assert source_objective is not None and polished_objective is not None
    accumulation_bound = _endpoint_objective_accumulation_bound(fit, polished)
    if accumulation_bound is None or polished_objective < source_objective - accumulation_bound:
        return fit
    return polished
