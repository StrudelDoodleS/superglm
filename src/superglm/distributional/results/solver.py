"""The coefficient solver's configuration, iteration record and certified result."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.results.endpoint_evidence import (
    CHUNKED_EXECUTION_BACKEND_IDENTIFIER,
    DENSE_EXECUTION_BACKEND_IDENTIFIER,
    CoefficientCurvature,
    ConvergenceReason,
    EndpointDirectionEvidence,
    ExecutionBackendIdentifier,
    _finite_positive,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.solvers.rank import RankDecomposition, decompose_gram


def _dense_penalty_fingerprint(values: NDArray) -> str:
    """Return a canonical compact record of one assembled dense penalty."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _assessment_unpenalized_logdet_term(result: DenseSolverResult) -> float:
    """Return −ℓₚ + ½ log|H|⁺ without the grouped penalty determinant."""

    rank = result.terminal_rank
    if result.coefficient_face is not None:
        rank = result.terminal_reduced_rank
        if rank is None:
            raise ValueError("endpoint assessment fit has no reduced-rank result")
    optimizing = result.penalized_optimizing_log_likelihood
    if optimizing is None:
        raise ValueError("endpoint assessment fit has no optimizing likelihood")
    value = -float(optimizing) + 0.5 * float(rank.log_pdet)
    if not math.isfinite(value):
        raise ValueError("endpoint assessment objective terms could not be verified")
    return value


def _assessment_exact_face_objective(result: DenseSolverResult) -> tuple[float, float]:
    """Rebuild one fitted face objective from its retained penalty matrix."""

    face = result.coefficient_face
    if face is None:
        raise ValueError("joint endpoint assessment requires an exact coefficient face")
    try:
        penalty_rank = decompose_gram(face.reduce_matrix(result.penalty))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError("joint endpoint assessment has invalid retained penalty geometry") from exc
    value = _assessment_unpenalized_logdet_term(result) - 0.5 * penalty_rank.log_pdet
    if not math.isfinite(value):
        raise ValueError("joint endpoint assessment objective could not be verified")
    return float(value), float(penalty_rank.log_pdet)


def _assessment_finite_objective(result: DenseSolverResult) -> tuple[float, float]:
    """Rebuild one finite fitted objective from its complete penalty matrix."""

    if result.coefficient_face is not None:
        raise ValueError("finite rollback objective requires a finite coefficient fit")
    try:
        penalty_rank = decompose_gram(result.penalty)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError("finite rollback has invalid penalty geometry") from exc
    value = _assessment_unpenalized_logdet_term(result) - 0.5 * penalty_rank.log_pdet
    if not math.isfinite(value):
        raise ValueError("finite rollback objective could not be verified")
    return float(value), float(penalty_rank.log_pdet)


def _assessment_face_geometry_matches(left: PenaltyFace, right: PenaltyFace) -> bool:
    return bool(
        left.component_names == right.component_names
        and left.coefficient_names == right.coefficient_names
        and left.constraint_rank == right.constraint_rank
        and np.array_equal(left.constraint_matrix, right.constraint_matrix)
        and np.array_equal(left.projector, right.projector)
    )


def _assessment_complete_face_penalty_matches(
    *,
    face: PenaltyFace,
    endpoint_penalty: NDArray,
    finite_penalty: NDArray,
) -> bool:
    """Authenticate the finite penalty added when a complete face is removed."""

    penalty_delta = np.asarray(finite_penalty) - np.asarray(endpoint_penalty)
    if penalty_delta.shape != (face.width, face.width) or not np.all(np.isfinite(penalty_delta)):
        return False
    penalty_delta = 0.5 * penalty_delta + 0.5 * penalty_delta.T
    try:
        delta_rank = decompose_gram(penalty_delta)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return False
    if delta_rank.rank != face.constraint_rank:
        return False
    scale = float(np.linalg.norm(penalty_delta, ord=2))
    if not math.isfinite(scale) or scale <= 0.0:
        return False
    projector = face.constraint_basis @ face.constraint_basis.T
    residual = max(
        float(np.linalg.norm(penalty_delta @ face.null_basis, ord=2)),
        float(np.linalg.norm(penalty_delta - projector @ penalty_delta @ projector, ord=2)),
    )
    operations = 1024 * max(face.width, 1)
    epsilon = np.finfo(np.float64).eps
    bound = operations * epsilon * scale
    return bool(math.isfinite(residual) and residual <= bound)


def _assessment_retained_kkt_ratio(result: DenseSolverResult) -> float:
    """Recompute endpoint-authority stationarity from retained fit state."""

    retained_score = result.terminal_score
    if result.coefficient_face is not None:
        retained_score = result.coefficient_face.reduce_vector(retained_score)
    score_norm = float(np.max(np.abs(retained_score), initial=0.0))
    optimizing = result.penalized_optimizing_log_likelihood
    if optimizing is None:
        raise ValueError("endpoint assessment fit has no optimizing likelihood")
    return score_norm / (1.0 + abs(float(optimizing)))


def _assessment_is_numerically_stationary(
    result: DenseSolverResult,
    tolerance: float,
) -> bool:
    """Interpret strict score or validated resolution at an exact-face seam."""

    return bool(
        _assessment_retained_kkt_ratio(result) <= tolerance
        or result.convergence_reason == "resolution_limited_stationarity"
    )


def _assessment_scalar_error_bound(
    left: float,
    right: float,
    *,
    width: int,
    calculation_scale: float = 1.0,
) -> float:
    scale = max(abs(left), abs(right), calculation_scale, 1.0)
    # These values are already-reduced scalar fit terms.  Reconstructing their
    # difference takes a fixed number of scalar operations; coefficient width
    # affects the stored log-determinants, but not this authentication step.
    # Allow a small logarithmic margin for the reduction tree that originally
    # produced those stored totals without letting a large common offset hide
    # a material change in their difference.
    operations = 32 + 4 * max(int(math.ceil(math.log2(max(width, 1)))), 0)
    eps = np.finfo(np.float64).eps
    gamma = operations * eps / (1.0 - operations * eps)
    return gamma * scale


def _assessment_coefficient_refit_bound(
    source: NDArray,
    refit: NDArray,
    *,
    tolerance: float,
) -> float | None:
    if source.shape != refit.shape:
        return None
    width = source.size
    epsilon = float(np.finfo(np.result_type(source.dtype, refit.dtype)).eps)
    operation_error = 64.0 * width * epsilon
    if operation_error >= 1.0:
        return None
    gamma = operation_error / (1.0 - operation_error)
    scale = max(
        1.0,
        float(np.max(np.abs(source), initial=0.0)),
        float(np.max(np.abs(refit), initial=0.0)),
    )
    bound = float(np.nextafter((tolerance + gamma) * scale, math.inf))
    return bound if math.isfinite(bound) else None


def _assessment_penalty_direction_matches(
    *,
    cap_fit: DenseSolverResult,
    endpoint_fit: DenseSolverResult,
    cap_lambda: float,
    selected_rank: int,
) -> RankDecomposition | None:
    """Authenticate the selected penalty from the cap and endpoint fits."""

    if cap_lambda <= 0.0 or not math.isfinite(cap_lambda):
        return None
    endpoint_penalty = endpoint_fit.penalty
    cap_direction = (cap_fit.penalty - endpoint_penalty) / cap_lambda
    if not np.all(np.isfinite(cap_direction)):
        return None
    entry_scale = float(np.max(np.abs(cap_direction), initial=0.0))
    if entry_scale == 0.0 or not math.isfinite(entry_scale):
        return None
    try:
        decomposition = decompose_gram(0.5 * cap_direction + 0.5 * cap_direction.T)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    if decomposition.rank != selected_rank:
        return None
    return decomposition


def _frozen_float_mapping(
    values: Mapping[str, float],
    *,
    name: str,
    nonnegative: bool = False,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool):
            raise ValueError(f"{name} values must be numeric, not bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} values must be finite numeric values") from exc
        if not math.isfinite(numeric) or (nonnegative and numeric < 0.0):
            qualification = " finite non-negative" if nonnegative else " finite"
            raise ValueError(f"{name} values must be{qualification}")
        result[key] = numeric
    return MappingProxyType(result)


def _frozen_endpoint_mapping(
    values: Mapping[str, EndpointDirectionEvidence],
) -> Mapping[str, EndpointDirectionEvidence]:
    if not isinstance(values, Mapping):
        raise TypeError("terminal_endpoint_directions must be a mapping")
    result: dict[str, EndpointDirectionEvidence] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("terminal endpoint direction names must be non-empty strings")
        if not isinstance(value, EndpointDirectionEvidence):
            raise TypeError("terminal endpoint directions must contain direction evidence")
        result[key] = value
    return MappingProxyType(result)


@dataclass(frozen=True)
class DenseSolverConfig:
    """Safeguards and convergence policy for the dense reference solver."""

    max_iterations: int = 100
    tolerance: float = 1.0e-8
    coefficient_curvature: CoefficientCurvature = "observed"
    max_backtracks: int = 30
    backtrack_factor: float = 0.5
    armijo_constant: float = 1.0e-4
    max_predictor_step: float = 3.0
    residual_tolerance: float = 1.0e-7
    initial_levenberg_shift: float = 1.0e-10
    levenberg_growth: float = 10.0
    max_levenberg_attempts: int = 20
    terminal_retry_iterations: int = 25
    terminal_retry_tolerance_factor: float = 0.1
    newton_decrement_tolerance: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_iterations",
            "max_backtracks",
            "max_levenberg_attempts",
            "terminal_retry_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "tolerance",
            "max_predictor_step",
            "residual_tolerance",
            "initial_levenberg_shift",
            "levenberg_growth",
            "terminal_retry_tolerance_factor",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        if self.newton_decrement_tolerance is not None:
            object.__setattr__(
                self,
                "newton_decrement_tolerance",
                _finite_positive(
                    self.newton_decrement_tolerance,
                    name="newton_decrement_tolerance",
                ),
            )
        if self.coefficient_curvature not in ("fisher", "observed"):
            raise ValueError("coefficient_curvature must be 'fisher' or 'observed'")
        if not math.isfinite(self.backtrack_factor) or not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie strictly between zero and one")
        if not math.isfinite(self.armijo_constant) or not 0.0 < self.armijo_constant < 1.0:
            raise ValueError("armijo_constant must lie strictly between zero and one")
        if self.levenberg_growth <= 1.0:
            raise ValueError("levenberg_growth must be greater than one")
        if self.terminal_retry_tolerance_factor >= 1.0:
            raise ValueError("terminal_retry_tolerance_factor must be less than one")


@dataclass(frozen=True)
class SolverIteration:
    """One accepted coefficient update and its complete safeguards."""

    iteration: int
    objective_before: float
    objective_after: float
    objective_relative_change: float
    step_relative: float
    score_relative: float
    step_scale: float
    backtracks: int
    levenberg_shift: float
    rank: int
    condition_estimate: float | None
    solve_residual: float


def _readonly_finite(values: NDArray, *, name: str) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _validate_newton_decrement_certificate(
    *,
    config: DenseSolverConfig,
    converged: bool,
    terminal_curvature: CurvatureTelemetry,
    score: NDArray[np.float64],
    penalized_curvature: NDArray[np.float64],
    penalized_objective: float,
    face: PenaltyFace | None,
) -> None:
    if not converged:
        raise ValueError("a Newton decrement certificate requires converged=True")
    if config.coefficient_curvature != "observed":
        raise ValueError("a Newton decrement certificate requires observed curvature")
    if config.newton_decrement_tolerance is None:
        raise ValueError("a Newton decrement certificate was not enabled for this fit")
    if (
        not isinstance(terminal_curvature, CurvatureTelemetry)
        or terminal_curvature.requested_source != "observed"
        or terminal_curvature.actual_source != "observed"
        or terminal_curvature.fallback_count != 0
    ):
        raise ValueError(
            "a Newton decrement certificate requires unfallbacked observed terminal curvature"
        )
    if face is None:
        optimization_score = score
        optimization_curvature = penalized_curvature
    else:
        optimization_score = face.reduce_vector(score)
        optimization_curvature = face.reduce_matrix(penalized_curvature)
    decomposition = decompose_gram(
        optimization_curvature,
        residual_tol=config.residual_tolerance,
    )
    if decomposition.rank != len(optimization_score):
        raise ValueError("a Newton decrement certificate requires full retained rank")
    step = decomposition.solve(optimization_score)
    decrement = float(optimization_score @ step)
    limit = config.newton_decrement_tolerance * (1.0 + abs(penalized_objective))
    product_sum = float(np.abs(optimization_score) @ np.abs(step))
    operations = max(2 * len(optimization_score), 1)
    operation_error = operations * np.finfo(np.float64).eps
    rounding = (
        operation_error
        / (1.0 - operation_error)
        * max(product_sum, abs(decrement), limit, np.finfo(np.float64).tiny)
    )
    rounding = float(np.nextafter(16.0 * rounding, math.inf))
    if decrement < -rounding or decrement > limit + rounding:
        raise ValueError(
            "terminal score and curvature do not satisfy the Newton decrement certificate"
        )


def _resolution_limited_decrement_is_within_objective_ulp(
    score: NDArray[np.float64],
    correction: NDArray[np.float64],
    objective: float,
) -> bool:
    score = np.asarray(score, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if (
        score.ndim != 1
        or correction.shape != score.shape
        or score.size == 0
        or not np.all(np.isfinite(score))
        or not np.all(np.isfinite(correction))
        or not math.isfinite(objective)
    ):
        return False
    operation_error = 2.0 * score.size * np.finfo(np.float64).eps
    if operation_error >= 1.0:
        return False
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        directional_dot = float(score @ correction)
        product_sum = float(np.abs(score) @ np.abs(correction))
        relative_error = operation_error / (1.0 - operation_error) * product_sum
        # The n products and n - 1 additions can each lose at most half the
        # smallest subnormal under round-to-nearest, so n subnormals enclose
        # the absolute underflow channel in addition to the relative bound.
        underflow_error = score.size * np.nextafter(0.0, math.inf)
        dot_error = float(
            np.nextafter(
                relative_error + underflow_error,
                math.inf,
            )
        )
        enclosed_dot = float(np.nextafter(directional_dot + dot_error, math.inf))
        predicted_gain = float(np.nextafter(0.5 * enclosed_dot, math.inf))
        objective_ulp = float(np.nextafter(objective, math.inf) - objective)
    return bool(
        math.isfinite(directional_dot)
        and math.isfinite(dot_error)
        and directional_dot > dot_error
        and math.isfinite(predicted_gain)
        and math.isfinite(objective_ulp)
        and objective_ulp > 0.0
        and predicted_gain <= objective_ulp
    )


def _validate_resolution_limited_stationarity(
    *,
    config: DenseSolverConfig,
    converged: bool,
    terminal_curvature: CurvatureTelemetry,
    score: NDArray[np.float64],
    penalized_curvature: NDArray[np.float64],
    penalized_objective: float,
    face: PenaltyFace | None,
    retained_rank: RankDecomposition | None,
) -> None:
    if not converged:
        raise ValueError("resolution-limited stationarity requires converged=True")
    if face is None:
        raise ValueError("resolution-limited stationarity requires a coefficient face")
    if config.coefficient_curvature != "observed":
        raise ValueError("resolution-limited stationarity requires observed curvature")
    if (
        not isinstance(terminal_curvature, CurvatureTelemetry)
        or terminal_curvature.requested_source != "observed"
        or terminal_curvature.actual_source != "observed"
        or terminal_curvature.fallback_count != 0
    ):
        raise ValueError("resolution-limited stationarity requires unfallbacked observed curvature")
    retained_score = face.reduce_vector(score)
    retained_curvature = face.reduce_matrix(penalized_curvature)
    retained_width = face.reduced_width
    if (
        retained_rank is None
        or retained_rank.width != retained_width
        or retained_rank.rank != retained_width
        or retained_rank.rank_truncated
        or retained_rank.used_svd_fallback
        or retained_rank.resolution_limited
    ):
        raise ValueError("resolution-limited stationarity requires full retained rank")
    raw_kkt = float(np.max(np.abs(retained_score), initial=0.0)) / (1.0 + abs(penalized_objective))
    if not math.isfinite(raw_kkt) or raw_kkt <= config.tolerance:
        raise ValueError(
            "resolution-limited stationarity requires raw retained KKT above tolerance"
        )
    try:
        reconstructed_rank = decompose_gram(
            retained_curvature,
            residual_tol=config.residual_tolerance,
        )
        stored_correction = retained_rank.solve(retained_score)
        correction = reconstructed_rank.solve(retained_score)
    except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "resolution-limited stationarity could not reconstruct its retained solve"
        ) from exc
    if reconstructed_rank.rank != retained_width:
        raise ValueError("resolution-limited stationarity requires full reconstructed rank")
    for candidate in (stored_correction, correction):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = float(
                np.linalg.norm(retained_curvature @ candidate - retained_score, ord=2)
                / max(1.0, float(np.linalg.norm(retained_score, ord=2)))
            )
        if not math.isfinite(residual) or residual > config.residual_tolerance:
            raise ValueError("resolution-limited stationarity violates the retained solve residual")
    if not _resolution_limited_decrement_is_within_objective_ulp(
        retained_score,
        correction,
        penalized_objective,
    ):
        raise ValueError("resolution-limited stationarity exceeds one optimizing-objective ULP")


@dataclass(frozen=True)
class DenseSolverResult:
    """Defensive publication of one accepted dense fixed-lambda fit."""

    config: DenseSolverConfig
    family_likelihood_plan_identifier: str
    resolved_chunk_size: int | None
    execution_backend_identifier: ExecutionBackendIdentifier
    coefficients: NDArray[np.float64]
    eta: NDArray[np.float64]
    theta: NDArray[np.float64]
    penalty: NDArray[np.float64]
    initial_penalized_log_likelihood: float
    log_likelihood: float
    penalty_value: float
    penalized_log_likelihood: float
    terminal_score: NDArray[np.float64]
    score_relative: float
    objective_relative_change: float
    step_relative: float
    converged: bool
    convergence_reason: ConvergenceReason
    iterations: int
    history: tuple[SolverIteration, ...]
    backtracking_steps: int
    terminal_data_curvature: NDArray[np.float64]
    terminal_penalized_curvature: NDArray[np.float64]
    terminal_rank: RankDecomposition
    terminal_curvature: CurvatureTelemetry
    optimizing_log_likelihood: float | None = None
    parameter_independent_carrier: float = 0.0
    penalized_optimizing_log_likelihood: float | None = None
    initial_penalized_optimizing_log_likelihood: float | None = None
    coefficient_face: PenaltyFace | None = None
    terminal_reduced_rank: RankDecomposition | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.family_likelihood_plan_identifier, str)
            or not self.family_likelihood_plan_identifier
        ):
            raise ValueError("family likelihood plan identifier must be a non-empty string")
        chunk_size = self.resolved_chunk_size
        if chunk_size is not None and (
            isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1
        ):
            raise ValueError("resolved_chunk_size must be None or a positive integer")
        expected_backend = (
            DENSE_EXECUTION_BACKEND_IDENTIFIER
            if chunk_size is None
            else CHUNKED_EXECUTION_BACKEND_IDENTIFIER
        )
        if self.execution_backend_identifier != expected_backend:
            raise ValueError(
                "execution backend identifier must agree with the resolved chunk route"
            )
        coefficients = _readonly_finite(self.coefficients, name="coefficients")
        eta = _readonly_finite(self.eta, name="eta")
        theta = _readonly_finite(self.theta, name="theta")
        penalty = _readonly_finite(self.penalty, name="penalty")
        score = _readonly_finite(self.terminal_score, name="terminal_score")
        data_curvature = _readonly_finite(
            self.terminal_data_curvature,
            name="terminal_data_curvature",
        )
        penalized_curvature = _readonly_finite(
            self.terminal_penalized_curvature,
            name="terminal_penalized_curvature",
        )
        width = len(coefficients)
        if coefficients.shape != (width,) or score.shape != (width,):
            raise ValueError("coefficient and terminal score shapes must agree")
        if eta.ndim != 2 or theta.shape != eta.shape:
            raise ValueError("eta and theta must be matching two-dimensional arrays")
        if penalty.shape != (width, width):
            raise ValueError("penalty shape must match coefficients")
        if data_curvature.shape != (width, width) or penalized_curvature.shape != (width, width):
            raise ValueError("terminal curvature shapes must match coefficients")
        if not np.array_equal(penalty, penalty.T):
            raise ValueError("penalty must be symmetric")
        if not np.array_equal(data_curvature, data_curvature.T):
            raise ValueError("terminal data curvature must be symmetric")
        if not np.array_equal(penalized_curvature, data_curvature + penalty):
            raise ValueError("terminal penalized curvature must equal data curvature plus penalty")
        if self.coefficient_face is None:
            if self.terminal_reduced_rank is not None:
                raise ValueError("a reduced terminal rank requires a coefficient face")
        else:
            if not isinstance(self.coefficient_face, PenaltyFace):
                raise TypeError("coefficient_face must be a PenaltyFace")
            if self.coefficient_face.width != width:
                raise ValueError("coefficient face width must match coefficients")
            if self.terminal_reduced_rank is None:
                raise ValueError("a coefficient face requires its reduced terminal rank")
            if self.terminal_reduced_rank.width != self.coefficient_face.reduced_width:
                raise ValueError("reduced terminal rank width must match the coefficient face")
            coefficient_scale = max(1.0, float(np.linalg.norm(coefficients, ord=2)))
            constraint_residual = float(
                np.linalg.norm(
                    self.coefficient_face.constraint_matrix @ coefficients,
                    ord=2,
                )
            )
            if constraint_residual > self.coefficient_face.null_residual_bound * coefficient_scale:
                raise ValueError("coefficients must lie in their certified coefficient face")
        for name in (
            "initial_penalized_log_likelihood",
            "log_likelihood",
            "penalty_value",
            "penalized_log_likelihood",
            "score_relative",
            "objective_relative_change",
            "step_relative",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        reconstructed_penalty_value = 0.5 * float(coefficients @ penalty @ coefficients)
        penalty_sum = 0.5 * float(np.abs(coefficients) @ np.abs(penalty) @ np.abs(coefficients))
        penalty_value_bound = (
            64.0
            * max(width, 1)
            * np.finfo(np.float64).eps
            * max(
                1.0,
                abs(float(self.penalty_value)),
                abs(reconstructed_penalty_value),
                penalty_sum,
            )
        )
        if abs(float(self.penalty_value) - reconstructed_penalty_value) > penalty_value_bound:
            raise ValueError(
                "penalty value must equal one half beta.T @ penalty @ beta within "
                "floating-point error"
            )
        optimizing = (
            float(self.log_likelihood)
            if self.optimizing_log_likelihood is None
            else float(self.optimizing_log_likelihood)
        )
        carrier = float(self.parameter_independent_carrier)
        penalized_optimizing = (
            float(self.penalized_log_likelihood)
            if self.penalized_optimizing_log_likelihood is None
            else float(self.penalized_optimizing_log_likelihood)
        )
        initial_penalized_optimizing = (
            float(self.initial_penalized_log_likelihood) - carrier
            if self.initial_penalized_optimizing_log_likelihood is None
            else float(self.initial_penalized_optimizing_log_likelihood)
        )
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        if self.iterations < 0 or self.backtracking_steps < 0:
            raise ValueError("iteration counts must be non-negative")
        retained_score = (
            score if self.coefficient_face is None else self.coefficient_face.reduce_vector(score)
        )
        retained_kkt = float(np.max(np.abs(retained_score), initial=0.0)) / (
            1.0 + abs(penalized_optimizing)
        )
        if self.convergence_reason == "score" and (
            not self.converged or retained_kkt > self.config.tolerance
        ):
            raise ValueError("score convergence requires retained KKT within tolerance")
        if self.convergence_reason == "newton_decrement":
            _validate_newton_decrement_certificate(
                config=self.config,
                converged=self.converged,
                terminal_curvature=self.terminal_curvature,
                score=score,
                penalized_curvature=penalized_curvature,
                penalized_objective=penalized_optimizing,
                face=self.coefficient_face,
            )
        if self.convergence_reason == "resolution_limited_stationarity":
            _validate_resolution_limited_stationarity(
                config=self.config,
                converged=self.converged,
                terminal_curvature=self.terminal_curvature,
                score=score,
                penalized_curvature=penalized_curvature,
                penalized_objective=penalized_optimizing,
                face=self.coefficient_face,
                retained_rank=self.terminal_reduced_rank,
            )
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "terminal_score", score)
        object.__setattr__(self, "terminal_data_curvature", data_curvature)
        object.__setattr__(self, "terminal_penalized_curvature", penalized_curvature)
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "optimizing_log_likelihood", optimizing)
        object.__setattr__(self, "parameter_independent_carrier", carrier)
        object.__setattr__(
            self,
            "penalized_optimizing_log_likelihood",
            penalized_optimizing,
        )
        object.__setattr__(
            self,
            "initial_penalized_optimizing_log_likelihood",
            initial_penalized_optimizing,
        )
        validate_solver_likelihood_decomposition(self)

    @property
    def objective(self) -> float:
        """Minimization-form objective used by independent optimizers."""
        assert self.penalized_optimizing_log_likelihood is not None
        return -self.penalized_optimizing_log_likelihood

    def solve_terminal(self, rhs: NDArray) -> NDArray[np.float64]:
        """Solve terminal curvature in the fitted coefficient subspace."""
        values = np.asarray(rhs, dtype=np.float64)
        if values.shape != self.coefficients.shape or not np.all(np.isfinite(values)):
            raise ValueError("rhs must match the full coefficient space")
        if self.coefficient_face is None:
            return self.terminal_rank.solve(values)
        assert self.terminal_reduced_rank is not None
        reduced = self.coefficient_face.reduce_vector(values)
        return self.coefficient_face.lift_vector(self.terminal_reduced_rank.solve(reduced))

    def terminal_pseudo_inverse(self) -> NDArray[np.float64]:
        """Return covariance on the fitted face, lifted to full coordinates."""
        if self.coefficient_face is None:
            return self.terminal_rank.pseudo_inverse()
        assert self.terminal_reduced_rank is not None
        reduced = self.terminal_reduced_rank.pseudo_inverse()
        basis = self.coefficient_face.null_basis
        inverse = basis @ reduced @ basis.T
        return 0.5 * (inverse + inverse.T)


def validate_solver_likelihood_decomposition(result: DenseSolverResult) -> None:
    """Revalidate terminal and initial optimizing/carrier/reporting identities."""

    optimizing = result.optimizing_log_likelihood
    penalized_optimizing = result.penalized_optimizing_log_likelihood
    initial_optimizing = result.initial_penalized_optimizing_log_likelihood
    components = (
        optimizing,
        result.parameter_independent_carrier,
        result.log_likelihood,
        result.penalty_value,
        penalized_optimizing,
        result.penalized_log_likelihood,
        initial_optimizing,
        result.initial_penalized_log_likelihood,
    )
    if any(
        value is None or isinstance(value, bool) or not math.isfinite(float(value))
        for value in components
    ):
        raise ValueError("solver likelihood decomposition must contain finite values")
    assert optimizing is not None
    assert penalized_optimizing is not None
    assert initial_optimizing is not None
    carrier = result.parameter_independent_carrier
    if result.log_likelihood != optimizing + carrier:
        raise ValueError(
            "solver likelihood decomposition requires reported likelihood = "
            "optimizing likelihood + carrier"
        )
    if penalized_optimizing != optimizing - result.penalty_value:
        raise ValueError(
            "solver likelihood decomposition requires penalized optimizing likelihood = "
            "optimizing likelihood - penalty"
        )
    if result.penalized_log_likelihood != result.log_likelihood - result.penalty_value:
        raise ValueError(
            "solver likelihood decomposition requires reported penalized likelihood = "
            "reported likelihood - penalty"
        )
    initial_reconstructed = initial_optimizing + carrier
    initial_error = abs(result.initial_penalized_log_likelihood - initial_reconstructed)
    initial_bound = (
        8.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(result.initial_penalized_log_likelihood),
            abs(initial_optimizing),
            abs(carrier),
        )
    )
    if initial_error > initial_bound:
        raise ValueError(
            "solver likelihood decomposition requires initial reported penalized likelihood = "
            "initial penalized optimizing likelihood + carrier within floating-point error"
        )
