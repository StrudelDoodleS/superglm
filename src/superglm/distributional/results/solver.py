"""The coefficient solver's configuration, iteration record and certified result."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.results.endpoint_evidence import (
    CoefficientCurvature,
    ConvergenceReason,
    EndpointDirectionEvidence,
    ExecutionBackendIdentifier,
    _finite_positive,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.solvers._structured.retired import module_getattr
from superglm.solvers.rank import RankDecomposition, decompose_gram


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


def _assessment_unchanged_failed_cap(
    cap: DenseSolverResult, source: DenseSolverResult | None
) -> bool:
    """An unchanged stalled refit can seed a face, not certify a finite cap."""
    return bool(
        source is not None
        and source.converged
        and not cap.converged
        and cap.convergence_reason in ("line_search_failed", "rank_reduced_at_stop")
        and cap.iterations == 0
        and not cap.history
        and np.array_equal(cap.coefficients, source.coefficients)
    )


@dataclass(frozen=True)
class RetainedCoordinates:
    """The coefficients ``K`` a rank-deficient representative keeps; the dropped ``J`` are zero.

    The representative inverts the curvature's ``K`` block and gives ``J``
    zeros, so its fit is the reduced problem on ``K`` (Wood, Pya and Saefken
    2016, sec. 3.1.2), whose score and curvature are the ``K`` entries of the
    full ones. The basis ``I[:, K]`` makes every reduction and lift exact.
    """

    dropped: NDArray[np.bool_]

    @classmethod
    def of(cls, rank: RankDecomposition) -> RetainedCoordinates | None:
        """``rank``'s retained coordinates; ``None`` unless it is a rank-deficient representative."""
        aliases = rank.structural_aliases
        if aliases is None or rank.rank == rank.width:
            return None
        if np.count_nonzero(aliases) != rank.width - rank.rank:
            return None
        return cls(np.array(aliases, dtype=bool))

    @property
    def null_basis(self) -> NDArray[np.float64]:
        return np.eye(self.dropped.size)[:, ~self.dropped]

    def reduce_vector(self, values: NDArray) -> NDArray[np.float64]:
        return np.asarray(values, dtype=np.float64)[~self.dropped]

    def reduce_matrix(self, values: NDArray) -> NDArray[np.float64]:
        kept = ~self.dropped
        return np.asarray(values, dtype=np.float64)[np.ix_(kept, kept)]

    def lift_vector(self, values: NDArray) -> NDArray[np.float64]:
        lifted = np.zeros(self.dropped.size)
        lifted[~self.dropped] = values
        return lifted


def _score_within_rounding_floor(
    *,
    config: DenseSolverConfig,
    penalty: NDArray,
    coefficients: NDArray,
    score: NDArray,
    penalized_curvature: NDArray,
    face: PenaltyFace | RetainedCoordinates | None,
    objective: float,
    tolerance: float,
) -> bool:
    """Return whether the retained score is within its rounding floor and moves nothing.

    The solver forms ``g = s - P b`` in float64. Row ``i`` of ``P b`` errs by at
    most ``gamma_{k_i} (|P| |b|)_i``, ``k_i`` the nonzeros in that row, and the
    subtraction by ``u |g_i| / (1 - u)`` (Higham 2002, eqs. 3.5 and 3.12);
    reducing to a face adds ``gamma_w |Z|' |g|`` and carries the row bound
    through ``|Z|'``. At a large smoothing parameter this floor exceeds any
    fixed relative tolerance. The data score's own evaluation error is left
    out, so a score above this floor can still be rounding.

    The floor bounds the rounding error; it is not evidence that the score is
    rounding. Along a direction the penalty does not act on, a representable
    move changes the likelihood and leaves ``P b`` unchanged, so a genuine
    score there can sit inside a floor taken from penalized rows. A score
    within its floor is therefore stationary only when the Newton correction
    it would drive is certified not to change the objective by more than
    ``tolerance (1 + |f|)``. That is the objective condition of Gill, Murray
    and Wright's termination test (1981, sec. 8.2.3), which must hold beside
    the gradient condition's escape ``||g|| < eps_A`` that the floor states.
    ``penalized_curvature`` is the observed penalized curvature at ``b``.

    The certificate holds for every score within the floor of the computed
    one, not only for the computed score: rounding of ``P b`` can cancel a
    genuine score along a direction the curvature barely resists, where the
    Newton correction is largest. Newton's method in floating point is
    accurate only to the residual's evaluation error times ``||H^-1||``
    (Tisseur 2001, Cor. 2.3), so the decrement certified is the computed
    score's plus that error's, ``(sqrt(r' H^-1 r) + sqrt(||H^-1||) ||floor||)^2``
    in the equilibrated coordinates the certificate works in.
    """
    unit_roundoff = 0.5 * np.finfo(np.float64).eps
    penalty = np.asarray(penalty, dtype=np.float64)
    absolute_score = np.abs(np.asarray(score, dtype=np.float64))
    nonzeros = np.count_nonzero(penalty, axis=1)
    width = absolute_score.size
    retained_score = absolute_score if face is None else np.abs(face.reduce_vector(score))
    with np.errstate(over="ignore", invalid="ignore"):
        row_gamma = nonzeros * unit_roundoff / (1.0 - nonzeros * unit_roundoff)
        floor = row_gamma * (np.abs(penalty) @ np.abs(np.asarray(coefficients, dtype=np.float64)))
        floor += unit_roundoff / (1.0 - unit_roundoff) * absolute_score
        roundings = int(np.max(nonzeros, initial=0)) + 4
        if face is not None:
            reduction = width * unit_roundoff / (1.0 - width * unit_roundoff)
            floor = np.abs(face.null_basis).T @ (floor + reduction * absolute_score)
            roundings += width + 2
        # Every term is nonnegative, so the computed floor is at most gamma_m
        # low for the m roundings that formed it; restore it outward, then
        # add the relative tolerance outward too.
        restore = 2 * roundings * unit_roundoff / (1.0 - 2 * roundings * unit_roundoff)
        floor = np.nextafter(floor * (1.0 + restore), np.inf)
        scale = np.nextafter(1.0 + abs(float(objective)), np.inf)
        limit = np.nextafter(tolerance * scale, np.inf)
        allowed = np.nextafter(limit + floor, np.inf)
    if not (np.all(np.isfinite(allowed)) and np.all(retained_score <= allowed)):
        return False
    return _newton_decrement_is_certified(
        config=config,
        score=score,
        penalized_curvature=penalized_curvature,
        penalized_objective=objective,
        face=face,
        tolerance=tolerance,
        score_error=floor,
    )


def _is_unfallbacked_observed(telemetry: CurvatureTelemetry) -> bool:
    return bool(
        isinstance(telemetry, CurvatureTelemetry)
        and telemetry.requested_source == "observed"
        and telemetry.actual_source == "observed"
        and telemetry.fallback_count == 0
    )


def _assessment_is_strictly_stationary(result: DenseSolverResult, tolerance: float) -> bool:
    """The stationarity rule before the rounding floor: retained score or validated resolution."""

    return bool(
        _assessment_retained_kkt_ratio(result) <= tolerance
        or result.convergence_reason == "resolution_limited_stationarity"
    )


def _assessment_is_numerically_stationary(
    result: DenseSolverResult,
    tolerance: float,
) -> bool:
    """Interpret retained score, its rounding floor or validated resolution at an exact-face seam."""

    optimizing = result.penalized_optimizing_log_likelihood
    assert optimizing is not None
    return bool(
        _assessment_is_strictly_stationary(result, tolerance)
        or (
            _is_unfallbacked_observed(result.terminal_curvature)
            and _score_within_rounding_floor(
                config=result.config,
                penalty=result.penalty,
                coefficients=result.coefficients,
                score=result.terminal_score,
                penalized_curvature=result.terminal_penalized_curvature,
                face=result.coefficient_face,
                objective=optimizing,
                tolerance=tolerance,
            )
        )
    )


def _endpoint_revalidation_projection_bound(
    face: PenaltyFace,
    coefficients: NDArray,
) -> float | None:
    """Bound one floating-point replay of the exact-face projection."""
    values = np.asarray(coefficients)
    if values.shape != (face.width,) or values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        return None
    basis = np.asarray(face.null_basis)
    width, retained_width = basis.shape
    if width != face.width:
        return None
    epsilon = float(np.finfo(np.result_type(values.dtype, basis.dtype)).eps)
    projection_error = float(face.orthogonality_bound)
    if not math.isfinite(projection_error) or projection_error < 0.0 or projection_error >= 1.0:
        return None
    projection_norm = float(np.linalg.norm(basis, ord="fro"))
    coefficient_norm = float(np.linalg.norm(values, ord=2))
    if (
        not math.isfinite(projection_norm)
        or not math.isfinite(coefficient_norm)
        or projection_norm < 0.0
    ):
        return None
    projection_operations = width * epsilon
    reduction_operations = retained_width * epsilon
    if projection_operations >= 1.0 or reduction_operations >= 1.0:
        return None
    gamma_projection = projection_operations / (1.0 - projection_operations)
    gamma_reduction = reduction_operations / (1.0 - reduction_operations)
    basis_scale = math.sqrt(1.0 + projection_error)
    denominator = math.sqrt(1.0 - projection_error) - gamma_reduction * projection_norm
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    lift_error = gamma_reduction * projection_norm / denominator
    factor = (
        projection_error
        + (1.0 + projection_error) * lift_error
        + (gamma_projection + gamma_reduction) * projection_norm * basis_scale
        + gamma_projection * gamma_reduction * projection_norm**2
    )
    if not math.isfinite(factor) or factor < 0.0:
        return None
    bound = float(np.nextafter(factor * coefficient_norm, math.inf))
    return bound if math.isfinite(bound) else None


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
    # ``||H s - g|| / max(1, ||g||)``, or, for an untruncated solve that misses the
    # residual tolerance on it, the normwise backward error ``||H s - g|| /
    # (||H|| ||s|| + ||g||)`` that accepted it (``solver._solve_residual``).
    solve_residual: float


def _readonly_finite(values: NDArray, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(result.tobytes(order="C"), dtype=result.dtype).reshape(result.shape)


def _newton_decrement_is_certified(
    *,
    config: DenseSolverConfig,
    score: NDArray[np.float64],
    penalized_curvature: NDArray[np.float64],
    penalized_objective: float,
    face: PenaltyFace | RetainedCoordinates | None,
    tolerance: float,
    score_error: NDArray[np.float64] | None = None,
) -> bool:
    """Return whether retained observed curvature certifies a scaled decrement.

    ``score_error`` bounds, componentwise, the error of the retained score.
    When it is given, the decrement is certified for every score within it.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0 or not math.isfinite(penalized_objective):
        return False
    if face is None:
        optimization_score = score
        optimization_curvature = penalized_curvature
    else:
        optimization_score = face.reduce_vector(score)
        optimization_curvature = face.reduce_matrix(penalized_curvature)
    try:
        decomposition = decompose_gram(
            optimization_curvature,
            residual_tol=config.residual_tolerance,
        )
    except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
        return False
    if (
        decomposition.rank != len(optimization_score)
        or decomposition.rank_truncated
        or decomposition.used_svd_fallback
        or decomposition.resolution_limited
    ):
        return False
    try:
        step = decomposition.solve(optimization_score)
    except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
        return False
    if not np.all(np.isfinite(step)):
        return False
    dimension = len(optimization_score)
    epsilon = float(np.finfo(np.float64).eps)
    scales = np.asarray(decomposition.column_scale, dtype=np.float64)
    if scales.shape != (dimension,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale_outer = np.outer(scales, scales)
        equilibrated_curvature = optimization_curvature / scale_outer
        equilibrated_score = optimization_score / scales
        equilibrated_step = step * scales
    if not (
        np.all(np.isfinite(equilibrated_curvature))
        and np.all(np.isfinite(equilibrated_score))
        and np.all(np.isfinite(equilibrated_step))
    ):
        return False

    def operation_gamma(count: int) -> float | None:
        count = max(int(count), 1)
        denominator = 1.0 - count * epsilon
        if denominator <= 0.0:
            return None
        return count * epsilon / denominator

    def upper_frobenius_norm(values: NDArray[np.float64]) -> float | None:
        """Return a one-sided floating-point enclosure of a Frobenius norm."""
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(values, ord="fro"))
        if not math.isfinite(norm):
            return None
        # The reduction has one rounded accumulation per entry.  Scale the
        # margin with the actual number of terms rather than using a width-
        # independent ulp allowance for matrices.
        reduction_gamma = operation_gamma(values.size)
        if reduction_gamma is None:
            return None
        margin = reduction_gamma * max(norm, np.finfo(np.float64).tiny)
        bound = float(np.nextafter(norm + margin, math.inf))
        return bound if math.isfinite(bound) else None

    curvature_norm = upper_frobenius_norm(equilibrated_curvature)
    score_norm = upper_frobenius_norm(equilibrated_score.reshape(1, -1))
    step_norm = upper_frobenius_norm(equilibrated_step.reshape(1, -1))
    if curvature_norm is None or score_norm is None or step_norm is None:
        return False
    arithmetic_gamma = operation_gamma(4 * (dimension + 1))
    if arithmetic_gamma is None:
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        computed_residual = equilibrated_curvature @ equilibrated_step - equilibrated_score
    computed_residual_norm = upper_frobenius_norm(computed_residual.reshape(1, -1))
    if computed_residual_norm is None:
        return False
    # Enclose the exact residual of the represented matrix/vector.  Checking
    # only ``fl(H @ step - score)`` can report zero after cancellation even
    # when the matvec and subtraction each incurred material error.
    residual_rounding = arithmetic_gamma * (curvature_norm * step_norm + score_norm)
    residual_bound = float(np.nextafter(computed_residual_norm + residual_rounding, math.inf))
    score_scale = max(1.0, score_norm)
    residual = residual_bound / score_scale
    if (
        not math.isfinite(residual_bound)
        or not math.isfinite(residual)
        or residual > config.residual_tolerance
    ):
        return False

    try:
        inverse_approximation = np.asarray(decomposition.pseudo_inverse(), dtype=np.float64)
    except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
        return False
    if inverse_approximation.shape != optimization_curvature.shape or not np.all(
        np.isfinite(inverse_approximation)
    ):
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        equilibrated_inverse = inverse_approximation * scale_outer
    if not np.all(np.isfinite(equilibrated_inverse)):
        return False
    inverse_norm = upper_frobenius_norm(equilibrated_inverse)
    if inverse_norm is None:
        return False
    # Bound ||H^-1|| from the computed inverse approximation itself.  This
    # avoids treating a condition estimate or an unverified pseudo-inverse
    # norm as an upper bound: if ||I - H B|| < 1, Banach's lemma gives
    # ||H^-1|| <= ||B|| / (1 - ||I - H B||).
    identity = np.eye(dimension, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        computed_inverse_defect = identity - equilibrated_curvature @ equilibrated_inverse
    inverse_defect = upper_frobenius_norm(computed_inverse_defect)
    if inverse_defect is None:
        return False
    inverse_product_rounding = arithmetic_gamma * (
        curvature_norm * inverse_norm + math.sqrt(float(dimension))
    )
    inverse_defect = float(np.nextafter(inverse_defect + inverse_product_rounding, math.inf))
    if not math.isfinite(inverse_defect) or inverse_defect >= 1.0:
        return False
    inverse_norm_bound = float(np.nextafter(inverse_norm / (1.0 - inverse_defect), math.inf))
    if not math.isfinite(inverse_norm_bound):
        return False
    solve_error = score_norm * inverse_norm_bound * residual_bound
    if not math.isfinite(solve_error):
        return False
    decrement = float(equilibrated_score @ equilibrated_step)
    limit = tolerance * (1.0 + abs(penalized_objective))
    product_sum = float(np.nextafter(score_norm * step_norm, math.inf))
    dot_gamma = operation_gamma(2 * (dimension + 1))
    if dot_gamma is None:
        return False
    rounding = dot_gamma * max(product_sum, abs(decrement), limit, np.finfo(np.float64).tiny)
    rounding = float(np.nextafter(16.0 * (rounding + solve_error), math.inf))
    if not (math.isfinite(decrement) and math.isfinite(limit) and decrement >= -rounding):
        return False
    if score_error is None:
        return decrement + rounding <= limit
    error = np.asarray(score_error, dtype=np.float64)
    if error.shape != (dimension,) or not np.all(np.isfinite(error)) or np.any(error < 0.0):
        return False
    # The exact decrement is ||H^-1/2 (r + e)||^2 with |e| <= error, so its root
    # is at most the computed one plus ||H^-1||_2^(1/2) ||D^-1 error||_2. Each
    # operation below is nondecreasing in nonnegative operands and is rounded
    # outward once.
    with np.errstate(over="ignore", invalid="ignore"):
        equilibrated_error = np.nextafter(error / scales, np.inf)
    error_norm = upper_frobenius_norm(equilibrated_error.reshape(1, -1))
    if error_norm is None:
        return False
    spread = float(np.nextafter(math.sqrt(inverse_norm_bound), math.inf))
    spread = float(np.nextafter(spread * error_norm, math.inf))
    certified = float(np.nextafter(max(decrement + rounding, 0.0), math.inf))
    root = float(np.nextafter(math.sqrt(certified), math.inf))
    root = float(np.nextafter(root + spread, math.inf))
    bound = float(np.nextafter(root * root, math.inf))
    return bool(math.isfinite(bound) and bound <= limit)


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
    # Each solve is checked against the matrix it inverts.  A terminal rank
    # decided from the rows inverts ``E diag(r) E'`` (``certificate.py``), not
    # the formed curvature, in the directions it read.
    with np.errstate(over="ignore", invalid="ignore"):
        stored_product = retained_curvature @ stored_correction
        if retained_rank.row_decided:
            basis = np.asarray(retained_rank.estimable_functional_basis, dtype=np.float64)
            retained = np.asarray(retained_rank.retained_values, dtype=np.float64)
            stored_product = basis @ (retained * (basis.T @ stored_correction))
        products = (stored_product, retained_curvature @ correction)
    for product in products:
        with np.errstate(over="ignore", invalid="ignore"):
            residual = float(
                np.linalg.norm(product - retained_score, ord=2)
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
    eta: NDArray[np.float64] | None
    theta: NDArray[np.float64] | None
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
    row_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        eta = None if self.eta is None else _readonly_finite(self.eta, name="eta")
        theta = None if self.theta is None else _readonly_finite(self.theta, name="theta")
        carrier = float(self.parameter_independent_carrier)
        object.__setattr__(
            self, "coefficients", _readonly_finite(self.coefficients, name="coefficients")
        )
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "row_shape", self.row_shape if eta is None else eta.shape)
        object.__setattr__(self, "penalty", _readonly_finite(self.penalty, name="penalty"))
        object.__setattr__(
            self, "terminal_score", _readonly_finite(self.terminal_score, name="terminal_score")
        )
        object.__setattr__(
            self,
            "terminal_data_curvature",
            _readonly_finite(self.terminal_data_curvature, name="terminal_data_curvature"),
        )
        object.__setattr__(
            self,
            "terminal_penalized_curvature",
            _readonly_finite(
                self.terminal_penalized_curvature, name="terminal_penalized_curvature"
            ),
        )
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(
            self,
            "optimizing_log_likelihood",
            float(self.log_likelihood)
            if self.optimizing_log_likelihood is None
            else float(self.optimizing_log_likelihood),
        )
        object.__setattr__(self, "parameter_independent_carrier", carrier)
        object.__setattr__(
            self,
            "penalized_optimizing_log_likelihood",
            float(self.penalized_log_likelihood)
            if self.penalized_optimizing_log_likelihood is None
            else float(self.penalized_optimizing_log_likelihood),
        )
        object.__setattr__(
            self,
            "initial_penalized_optimizing_log_likelihood",
            float(self.initial_penalized_log_likelihood) - carrier
            if self.initial_penalized_optimizing_log_likelihood is None
            else float(self.initial_penalized_optimizing_log_likelihood),
        )

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


# A v0.41.0 model pickles its smoothing result's assessment context and its
# layout's objective receipts.  The names stay importable as inert stand-ins
# (PEP 562) so such a model loads; nothing reads either.
__getattr__ = module_getattr(
    __name__, frozenset({"_PenaltyAssessmentContext", "_PenaltyObjectiveReceipt"})
)
