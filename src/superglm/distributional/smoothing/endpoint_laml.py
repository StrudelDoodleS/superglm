"""Penalty determinant primitives for distributional infinity-face LAML."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from superglm.distributional._row_design import bounded_predictor_matrices, matrix_row_chunk
from superglm.distributional.family import (
    DistributionalFamily,
    FamilyLikelihoodPlan,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    ANALYTIC_DIRECTION_AUTHORITY,
    DIRECTION_AUTHORITIES,
    FINITE_DIFFERENCE_DIRECTION_AUTHORITY,
    DenseSolverResult,
    EndpointDirectionDecision,
    EndpointDirectionEvidence,
)
from superglm.distributional.results.solver import _record_penalty_objective
from superglm.distributional.smoothing.endpoint_direction import (
    FiniteDifferenceDirection,
    analytic_curvature_direction,
    finite_difference_curvature_direction,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.smoothing.penalty_geometry import (
    EndpointLaplaceError,
    _finite_penalty_evaluation_from_components,
)
from superglm.distributional.smoothing.penalty_geometry import (
    _projected_penalty_group_indices as _projected_penalty_group_indices,
)
from superglm.distributional.solver.assembly import assemble_grouped_geometry
from superglm.distributional.solver.packing import packed_pairs
from superglm.reml.penalty_algebra import _PenaltyLogdetEvaluation, penalty_component_dense_matrix
from superglm.solvers.rank import RankDecomposition, decompose_gram


@dataclass(frozen=True)
class EndpointLaplaceDerivative:
    """Analytic one-sided derivative of negative LAML at τ = 1/λ = 0+."""

    authority_identifier: str
    decision: EndpointDirectionDecision
    derivative: float
    profile_score_term: float
    curvature_schur_term: float
    curvature_drift_term: float
    numerical_error: float
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if self.authority_identifier not in DIRECTION_AUTHORITIES:
            raise ValueError("unknown endpoint derivative authority")
        if self.decision not in {"endpoint", "finite", "unresolved"}:
            raise ValueError("invalid endpoint derivative decision")
        for name in (
            "derivative",
            "profile_score_term",
            "curvature_schur_term",
            "curvature_drift_term",
            "numerical_error",
            "lower_bound",
            "upper_bound",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar")
            object.__setattr__(self, name, float(value))
        if self.numerical_error < 0.0:
            raise ValueError("endpoint derivative error must be nonnegative")
        if self.lower_bound != self.derivative - self.numerical_error:
            raise ValueError("endpoint derivative lower bound is inconsistent")
        if self.upper_bound != self.derivative + self.numerical_error:
            raise ValueError("endpoint derivative upper bound is inconsistent")
        expected: EndpointDirectionDecision
        if self.lower_bound > 0.0:
            expected = "endpoint"
        elif self.upper_bound < 0.0:
            expected = "finite"
        else:
            expected = "unresolved"
        if self.decision != expected:
            raise ValueError("endpoint derivative decision is inconsistent with its bounds")


def resolve_endpoint_direction(
    endpoint_objective: float,
    *,
    analytic: EndpointLaplaceDerivative,
) -> EndpointDirectionEvidence:
    """Publish analytic endpoint evidence for one fitted exact face."""
    if isinstance(endpoint_objective, bool) or not isinstance(endpoint_objective, Real):
        raise TypeError("endpoint_objective must be a finite real scalar")
    endpoint = float(endpoint_objective)
    if not np.isfinite(endpoint):
        raise ValueError("endpoint_objective must be finite")
    if not isinstance(analytic, EndpointLaplaceDerivative):
        raise TypeError("analytic must be EndpointLaplaceDerivative")
    return EndpointDirectionEvidence(
        authority_identifier=analytic.authority_identifier,
        decision=analytic.decision,
        endpoint_objective=endpoint,
        analytic_derivative=analytic.derivative,
        profile_score_term=analytic.profile_score_term,
        curvature_schur_term=analytic.curvature_schur_term,
        curvature_drift_term=analytic.curvature_drift_term,
        numerical_error=analytic.numerical_error,
        lower_bound=analytic.lower_bound,
        upper_bound=analytic.upper_bound,
    )


@dataclass(frozen=True)
class ProjectedPenaltyLogDet:
    """Rank and log pseudo-determinant of retained projected penalties."""

    component_names: tuple[str, ...]
    rank: int
    log_pdet: float

    def __post_init__(self) -> None:
        if isinstance(self.component_names, (str, bytes)):
            raise TypeError("component_names must be a sequence of names")
        component_names = tuple(self.component_names)
        if any(not isinstance(name, str) or not name for name in component_names):
            raise ValueError("component_names must contain nonempty strings")
        if len(set(component_names)) != len(component_names):
            raise ValueError("component_names must be unique")
        if isinstance(self.rank, bool) or not isinstance(self.rank, Integral):
            raise TypeError("rank must be a nonnegative integer")
        rank = int(self.rank)
        if rank < 0 or (rank > 0 and not component_names):
            raise ValueError("rank must be compatible with retained components")
        if isinstance(self.log_pdet, bool) or not isinstance(self.log_pdet, Real):
            raise TypeError("log_pdet must be a finite real value")
        log_pdet = float(self.log_pdet)
        if not np.isfinite(log_pdet) or (rank == 0 and log_pdet != 0.0):
            raise ValueError("log_pdet must be finite and zero when rank is zero")
        object.__setattr__(self, "component_names", component_names)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "log_pdet", log_pdet)


@dataclass(frozen=True)
class EndpointLaplaceEvaluation:
    """Exact negative-LAML terms on one already-fitted penalty face."""

    objective: float
    face_component_names: tuple[str, ...]
    finite_component_names: tuple[str, ...]
    reduced_width: int
    hessian_rank: int
    penalty_rank: int
    hessian_log_pdet: float
    penalty_log_pdet: float

    def __post_init__(self) -> None:
        names: dict[str, tuple[str, ...]] = {}
        for field_name in ("face_component_names", "finite_component_names"):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{field_name} must be a sequence of names")
            normalized = tuple(value)
            if any(not isinstance(name, str) or not name for name in normalized):
                raise ValueError(f"{field_name} must contain nonempty strings")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{field_name} must contain unique names")
            names[field_name] = normalized
        if set(names["face_component_names"]).intersection(names["finite_component_names"]):
            raise ValueError("face and finite component names must be disjoint")

        integers: dict[str, int] = {}
        for field_name in ("reduced_width", "hessian_rank", "penalty_rank"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be a nonnegative integer")
            normalized = int(value)
            if normalized < 0:
                raise ValueError(f"{field_name} must be nonnegative")
            integers[field_name] = normalized
        for rank_name in ("hessian_rank", "penalty_rank"):
            if integers[rank_name] > integers["reduced_width"]:
                raise ValueError(f"{rank_name} cannot exceed reduced_width")

        numerics: dict[str, float] = {}
        for field_name in ("objective", "hessian_log_pdet", "penalty_log_pdet"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field_name} must be a finite real value")
            normalized = float(value)
            if not np.isfinite(normalized):
                raise ValueError(f"{field_name} must be finite")
            numerics[field_name] = normalized
        for rank_name, log_name in (
            ("hessian_rank", "hessian_log_pdet"),
            ("penalty_rank", "penalty_log_pdet"),
        ):
            if integers[rank_name] == 0 and numerics[log_name] != 0.0:
                raise ValueError(f"{log_name} must be zero when {rank_name} is zero")

        for field_name, value in names.items():
            object.__setattr__(self, field_name, value)
        for field_name, value in integers.items():
            object.__setattr__(self, field_name, value)
        for field_name, value in numerics.items():
            object.__setattr__(self, field_name, value)


def _validated_complete_lambdas(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(lambdas, Mapping):
        raise TypeError("lambdas must be a mapping")
    expected = set(layout.penalty_names)
    supplied = set(lambdas)
    unknown = supplied - expected
    missing = expected - supplied
    if unknown:
        raise ValueError(f"unknown penalty lambda names: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing penalty lambda names: {sorted(missing)}")

    resolved: dict[str, float] = {}
    for name in layout.penalty_names:
        value = lambdas[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"lambda for {name!r} must be a finite nonnegative real")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"lambda for {name!r} must be finite and nonnegative")
        resolved[name] = numeric
    return resolved


def _projected_finite_penalty_evaluation(
    *, layout: StackedLayout, lambdas: Mapping[str, float], face: PenaltyFace
) -> _PenaltyLogdetEvaluation:
    """Project frozen component roots, retaining their arithmetic evidence.

    The face basis is validated independently. A finite component is selected
    as a PSD representative before the projection F_i Q. Forming Q' P_i Q
    and selecting its rank again would change the represented penalty.
    """
    face.validate_layout(layout)
    resolved = _validated_complete_lambdas(layout, lambdas)
    return _finite_penalty_evaluation_from_components(
        components=layout.penalties, resolved=resolved, face=face
    )


def _spectral_norm(matrix: NDArray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, ord=2))


def _validate_reduced_terminal_provenance(
    result: DenseSolverResult,
    *,
    face: PenaltyFace,
    stored: RankDecomposition,
) -> None:
    reduced_curvature = face.reduce_matrix(result.terminal_penalized_curvature)
    try:
        fresh = decompose_gram(reduced_curvature)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise EndpointLaplaceError("result has an invalid reduced terminal decomposition") from exc

    try:
        if isinstance(stored.rank, bool) or not isinstance(stored.rank, Integral):
            raise TypeError("stored reduced terminal rank must be an integer")
        stored_rank = int(stored.rank)
        if stored_rank < 0 or stored_rank > stored.width:
            raise ValueError("stored reduced terminal rank is outside its width")
        if isinstance(stored.log_pdet, bool) or not isinstance(stored.log_pdet, Real):
            raise TypeError("stored reduced terminal log_pdet must be a real scalar")
        stored_log_pdet = float(stored.log_pdet)
        if not np.isfinite(stored_log_pdet):
            raise ValueError("stored reduced terminal log_pdet must be finite")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EndpointLaplaceError("result has an invalid reduced terminal decomposition") from exc

    if (
        stored.policy_version != fresh.policy_version
        or stored.width != fresh.width
        or stored_rank != fresh.rank
        or stored.resolution_limited != fresh.resolution_limited
    ):
        raise EndpointLaplaceError("result has an inconsistent reduced terminal decomposition")

    width = fresh.width
    eps = np.finfo(np.float64).eps
    curvature_scale = _spectral_norm(reduced_curvature)
    try:
        fresh_inverse = fresh.pseudo_inverse()
        stored_inverse = np.asarray(stored.pseudo_inverse(), dtype=np.float64)
    except (
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        np.linalg.LinAlgError,
    ) as exc:
        raise EndpointLaplaceError("result has an invalid reduced terminal decomposition") from exc
    if stored_inverse.shape != (width, width) or not np.all(np.isfinite(stored_inverse)):
        raise EndpointLaplaceError("result has an invalid reduced terminal decomposition")
    log_scale = max(abs(float(fresh.log_pdet)), abs(stored_log_pdet), 1.0)
    log_bound = 512.0 * max(width, 1) * eps * log_scale
    if abs(stored_log_pdet - float(fresh.log_pdet)) > log_bound:
        raise EndpointLaplaceError("result has an inconsistent reduced terminal decomposition")
    if width == 0:
        return

    inverse_scale = _spectral_norm(stored_inverse)
    fresh_inverse_scale = _spectral_norm(fresh_inverse)
    reconstruction = reduced_curvature @ stored_inverse
    reverse_reconstruction = stored_inverse @ reduced_curvature
    fresh_reconstruction = reduced_curvature @ fresh_inverse
    fresh_reverse_reconstruction = fresh_inverse @ reduced_curvature
    backward_factor = 4096.0 * max(width, 1) * eps
    checks = (
        (
            _spectral_norm(
                reduced_curvature @ stored_inverse @ reduced_curvature - reduced_curvature
            ),
            _spectral_norm(
                reduced_curvature @ fresh_inverse @ reduced_curvature - reduced_curvature
            )
            + backward_factor * max(curvature_scale, np.finfo(np.float64).tiny),
        ),
        (
            _spectral_norm(stored_inverse @ reduced_curvature @ stored_inverse - stored_inverse),
            _spectral_norm(fresh_inverse @ reduced_curvature @ fresh_inverse - fresh_inverse)
            + backward_factor * max(inverse_scale, fresh_inverse_scale, np.finfo(np.float64).tiny),
        ),
        (
            _spectral_norm(stored_inverse - stored_inverse.T),
            _spectral_norm(fresh_inverse - fresh_inverse.T)
            + backward_factor * max(inverse_scale, fresh_inverse_scale, 1.0),
        ),
        (
            _spectral_norm(reconstruction - reconstruction.T),
            _spectral_norm(fresh_reconstruction - fresh_reconstruction.T)
            + backward_factor
            * max(_spectral_norm(reconstruction), _spectral_norm(fresh_reconstruction), 1.0),
        ),
        (
            _spectral_norm(reverse_reconstruction - reverse_reconstruction.T),
            _spectral_norm(fresh_reverse_reconstruction - fresh_reverse_reconstruction.T)
            + backward_factor
            * max(
                _spectral_norm(reverse_reconstruction),
                _spectral_norm(fresh_reverse_reconstruction),
                1.0,
            ),
        ),
    )
    if any(error > bound for error, bound in checks):
        raise EndpointLaplaceError("result has an inconsistent reduced terminal decomposition")


def _selected_whitened_basis(
    layout: StackedLayout,
    *,
    component_name: str,
    finite_face: PenaltyFace | None,
    endpoint_face: PenaltyFace,
) -> NDArray[np.float64]:
    """Return a selected-range basis R with RᵀSⱼR = I."""

    previous_names = () if finite_face is None else finite_face.component_names
    expected_names = tuple(
        name for name in layout.penalty_names if name in {*previous_names, component_name}
    )
    if endpoint_face.component_names != expected_names or component_name in previous_names:
        raise EndpointLaplaceError("endpoint face does not add exactly the selected component")
    components = {component.name: component for component in layout.penalties}
    component = components.get(component_name)
    if component is None:
        raise EndpointLaplaceError("endpoint component is absent from the fitted layout")
    local = np.asarray(penalty_component_dense_matrix(component), dtype=np.float64)
    full = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    full[component.group_sl, component.group_sl] = local
    base = np.eye(layout.n_coefficients) if finite_face is None else finite_face.null_basis
    projected = base.T @ full @ base
    projected = 0.5 * (projected + projected.T)
    try:
        values, vectors = np.linalg.eigh(projected)
    except np.linalg.LinAlgError as exc:
        raise EndpointLaplaceError("endpoint penalty range could not be decomposed") from exc
    selected_rank = endpoint_face.constraint_rank - (
        0 if finite_face is None else finite_face.constraint_rank
    )
    if selected_rank < 1 or selected_rank > len(values):
        raise EndpointLaplaceError("endpoint penalty has an inconsistent constrained rank")
    first = len(values) - selected_rank
    scale = float(np.max(np.abs(values), initial=0.0))
    resolution = 64.0 * max(len(values), 1) * np.finfo(np.float64).eps * scale
    if (
        scale == 0.0
        or float(np.max(np.abs(values[:first]), initial=0.0)) > resolution
        or float(values[first]) <= 8.0 * resolution
    ):
        raise EndpointLaplaceError("endpoint penalty range is not numerically resolved")
    selected = base @ vectors[:, first:]
    basis = selected * (1.0 / np.sqrt(values[first:]))[None, :]
    identity_error = _spectral_norm(basis.T @ full @ basis - np.eye(selected_rank))
    identity_bound = 512.0 * max(layout.n_coefficients, 1) * np.finfo(np.float64).eps
    if identity_error > identity_bound:
        raise EndpointLaplaceError("endpoint penalty whitening failed its residual check")
    return basis


def _predictor_direction(
    layout: StackedLayout,
    coefficient_direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.asarray(coefficient_direction, dtype=np.float64)
    if values.shape != (layout.n_coefficients,) or not np.all(np.isfinite(values)):
        raise EndpointLaplaceError("endpoint coefficient direction is invalid")
    n_observations = layout.predictors[0].design.n
    result = np.empty((n_observations, len(layout.predictors)), dtype=np.float64)
    for state in layout.predictors:
        local = values[state.coefficient_slice]
        intercept = int(state.intercept_index is not None)
        channel = np.full(
            n_observations,
            local[0] if intercept else 0.0,
            dtype=np.float64,
        )
        if state.design.p:
            channel += state.design.matvec(local[intercept:])
        result[:, state.parameter_index] = channel
    if not np.all(np.isfinite(result)):
        raise EndpointLaplaceError("endpoint predictor direction is not finite")
    return result


def _row_certificate_error(
    layout: StackedLayout,
    certificate: NDArray[np.float64],
    basis: NDArray[np.float64],
    inverse: NDArray[np.float64],
) -> float:
    """Bound on half the drift term's error, contracted row by row before any sum.

    The drift term is ``tr(R^-1 B' (sum_i X_i' D_i X_i) B) = sum_i <D_i, G_i>_w``
    with ``G_i = X_i Q X_i'``, ``Q = B R^-1 B'`` and ``w`` two on an off-diagonal
    packed channel, so an entrywise error ``|dD_i| <= C_i`` moves it by at most
    ``sum_i <C_i, |G_i|>_w``.  Each row is contracted before rows are summed,
    so mixed-sign design rows cannot cancel each other's error.  ``G_i`` is
    formed in float64 from ``p``-term dot products, so ``gamma_{2p + 4}`` of
    ``|X_i| |Q| |X_i|'`` is added to ``|G_i|``; the nonnegative chunk sums are
    within ``gamma_{chunk + 4}`` and are inflated by it, and the chunks are
    summed by ``math.fsum``.  A channel whose certificate is zero on every
    row contributes nothing and is skipped (NB2 certifies only ``l_bbb``, so
    two of its three channels are zero), and an all-zero certificate is zero.
    """
    matrices = bounded_predictor_matrices(layout)
    pairs = packed_pairs(len(matrices))
    live = [channel for channel in range(len(pairs)) if np.any(certificate[:, channel])]
    if not live:
        return 0.0
    slices = [state.coefficient_slice for state in layout.predictors]
    projector = basis @ np.asarray(inverse, dtype=np.float64) @ basis.T
    absolute_projector = np.abs(projector)
    unit_roundoff = 0.5 * np.finfo(np.float64).eps

    def gamma(count: int) -> float:
        return count * unit_roundoff / (1.0 - count * unit_roundoff)

    leverage_rounding = gamma(2 * layout.n_coefficients + 4)
    n_observations = certificate.shape[0]
    chunk = matrix_row_chunk(matrices, 4096)
    partials = []
    for start in range(0, n_observations, chunk):
        rows = slice(start, min(start + chunk, n_observations))
        local = [matrix[rows] for matrix in matrices]
        absolute = [np.abs(values) for values in local]
        total = np.zeros(rows.stop - rows.start, dtype=np.float64)
        for channel in live:
            left, right = pairs[channel]
            block = (slices[left], slices[right])
            leverage = np.einsum("ij,ij->i", local[left] @ projector[block], local[right])
            scale = np.einsum(
                "ij,ij->i", absolute[left] @ absolute_projector[block], absolute[right]
            )
            weight = 1.0 if left == right else 2.0
            total += (
                weight * certificate[rows, channel] * (np.abs(leverage) + leverage_rounding * scale)
            )
        partials.append(float(np.sum(total, dtype=np.float64)))
    inflation = 1.0 + gamma(chunk + 4)
    return 0.5 * inflation * math.fsum(partials)


def evaluate_endpoint_laplace_derivative(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    component_name: str,
    finite_face: PenaltyFace | None,
    endpoint_face: PenaltyFace,
    endpoint_fit: DenseSolverResult,
) -> EndpointLaplaceDerivative:
    """Evaluate the analytic local negative-LAML derivative at one exact face."""

    eta = endpoint_fit.eta
    if eta is None:
        raise EndpointLaplaceError("endpoint derivative requires retained predictor rows")
    if endpoint_fit.coefficient_face is not endpoint_face:
        raise EndpointLaplaceError("endpoint derivative requires the fitted endpoint face")
    endpoint_face.validate_layout(layout)
    if finite_face is not None:
        finite_face.validate_layout(layout)
    if (
        endpoint_fit.terminal_curvature.actual_source != "observed"
        or endpoint_fit.terminal_curvature.fallback_count != 0
    ):
        raise EndpointLaplaceError("endpoint derivative requires unfallbacked observed curvature")
    _validated_complete_lambdas(layout, lambdas)
    selected_basis = _selected_whitened_basis(
        layout,
        component_name=component_name,
        finite_face=finite_face,
        endpoint_face=endpoint_face,
    )
    retained_basis = endpoint_face.null_basis
    retained_width = endpoint_face.reduced_width
    selected_rank = selected_basis.shape[1]
    retained_rank = endpoint_fit.terminal_reduced_rank
    if retained_rank is None or retained_rank.rank != retained_width:
        raise EndpointLaplaceError("endpoint derivative requires full retained curvature rank")
    _validate_reduced_terminal_provenance(
        endpoint_fit,
        face=endpoint_face,
        stored=retained_rank,
    )
    penalty_cross = np.asarray(endpoint_fit.penalty, dtype=np.float64) @ selected_basis
    penalty_cross_scale = _spectral_norm(endpoint_fit.penalty) * _spectral_norm(selected_basis)
    penalty_cross_bound = (
        512.0
        * max(layout.n_coefficients, 1)
        * np.finfo(np.float64).eps
        * max(penalty_cross_scale, np.finfo(np.float64).tiny)
    )
    if _spectral_norm(penalty_cross) > penalty_cross_bound:
        raise EndpointLaplaceError(
            "endpoint derivative currently requires an isolated penalty component"
        )

    curvature = np.asarray(endpoint_fit.terminal_penalized_curvature, dtype=np.float64)
    cross_curvature = retained_basis.T @ curvature @ selected_basis
    selected_curvature = selected_basis.T @ curvature @ selected_basis
    retained_inverse = retained_rank.pseudo_inverse()
    selected_score = selected_basis.T @ endpoint_fit.terminal_score
    profiled_cross = retained_inverse @ (cross_curvature @ selected_score)
    coefficient_direction = selected_basis @ selected_score - retained_basis @ profiled_cross
    eta_direction = _predictor_direction(layout, coefficient_direction)
    links = tuple(state.link for state in layout.predictors)
    certificate: NDArray[np.float64] | None = None
    try:
        # A family that refuses these links (UnsupportedLikelihoodContractError)
        # is differenced, as v0.40.0 differenced every family without the protocol.
        analytic = analytic_curvature_direction(
            family,
            y,
            eta,
            eta_direction,
            links,
            likelihood_plan,
        )
        if analytic is not None:
            curvature_direction, certificate = analytic
            authority = ANALYTIC_DIRECTION_AUTHORITY
        else:
            numeric: FiniteDifferenceDirection = finite_difference_curvature_direction(
                family,
                y,
                eta,
                eta_direction,
                links,
                likelihood_plan,
            )
            curvature_direction = numeric.values
            certificate = numeric.certificate
            authority = FINITE_DIFFERENCE_DIRECTION_AUTHORITY
    except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise EndpointLaplaceError(
            "family could not evaluate the endpoint curvature direction"
        ) from exc
    n_observations = eta.shape[0]
    n_parameters = eta.shape[1]
    expected_channels = n_parameters * (n_parameters + 1) // 2
    if curvature_direction.shape != (n_observations, expected_channels) or not np.all(
        np.isfinite(curvature_direction)
    ):
        raise EndpointLaplaceError("family returned an invalid endpoint curvature direction")
    if certificate is not None and (
        certificate.shape != curvature_direction.shape
        or not np.all(np.isfinite(certificate))
        or np.any(certificate < 0.0)
    ):
        raise EndpointLaplaceError("family returned an invalid curvature-direction certificate")
    zeros_score = np.zeros((n_observations, n_parameters), dtype=np.float64)
    zeros_penalty = np.zeros_like(curvature)
    zeros_coefficients = np.zeros(layout.n_coefficients, dtype=np.float64)
    curvature_dot = assemble_grouped_geometry(
        layout,
        zeros_score,
        curvature_direction,
        penalty=zeros_penalty,
        coefficients=zeros_coefficients,
    ).data_curvature
    reduced_curvature_dot = retained_basis.T @ curvature_dot @ retained_basis

    profile_score_term = -float(selected_score @ selected_score)
    schur = selected_curvature - cross_curvature.T @ retained_inverse @ cross_curvature
    curvature_schur_term = float(np.trace(schur))
    curvature_drift_term = float(np.trace(retained_inverse @ reduced_curvature_dot))
    certificate_error = 0.0
    if certificate is not None:
        certificate_error = _row_certificate_error(
            layout, certificate, retained_basis, retained_inverse
        )
    derivative = 0.5 * (profile_score_term + curvature_schur_term + curvature_drift_term)
    magnitude = max(
        abs(profile_score_term) + abs(curvature_schur_term) + abs(curvature_drift_term),
        float(np.sum(np.abs(schur), dtype=np.float64)),
        float(np.sum(np.abs(retained_inverse @ reduced_curvature_dot), dtype=np.float64)),
    )
    operations = max(n_observations, layout.n_coefficients, selected_rank, 1)
    error_scale = 65536.0 * operations * np.finfo(np.float64).eps * magnitude
    if magnitude == 0.0:
        numerical_error = certificate_error
    elif error_scale < np.finfo(np.float64).tiny:
        # ``tiny`` only identifies underflow; the fallback retains derivative units.
        numerical_error = magnitude + certificate_error
    else:
        numerical_error = float(np.nextafter(error_scale + certificate_error, np.inf))
    lower = derivative - numerical_error
    upper = derivative + numerical_error
    decision: EndpointDirectionDecision
    if lower > 0.0:
        decision = "endpoint"
    elif upper < 0.0:
        decision = "finite"
    else:
        decision = "unresolved"
    return EndpointLaplaceDerivative(
        authority_identifier=authority,
        decision=decision,
        derivative=derivative,
        profile_score_term=profile_score_term,
        curvature_schur_term=curvature_schur_term,
        curvature_drift_term=curvature_drift_term,
        numerical_error=numerical_error,
        lower_bound=lower,
        upper_bound=upper,
    )


def evaluate_endpoint_laplace(
    result: DenseSolverResult,
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace,
) -> EndpointLaplaceEvaluation:
    """Evaluate negative LAML on an exact, already-fitted coefficient face."""
    if not isinstance(result, DenseSolverResult):
        raise TypeError("result must be a DenseSolverResult")
    if result.coefficient_face is not face:
        raise EndpointLaplaceError("result does not carry the supplied coefficient face")
    face.validate_layout(layout)

    penalty_evaluation = _projected_finite_penalty_evaluation(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )
    projected_penalty = ProjectedPenaltyLogDet(
        component_names=tuple(penalty_evaluation.gradient),
        rank=penalty_evaluation.rank,
        log_pdet=penalty_evaluation.logdet,
    )
    endpoint_lambdas = dict(_validated_complete_lambdas(layout, lambdas))
    for name in face.component_names:
        endpoint_lambdas[name] = 0.0
    if not np.array_equal(result.penalty, layout.penalty_matrix(endpoint_lambdas)):
        raise EndpointLaplaceError("result penalty does not match the endpoint lambdas")

    reduced_rank = result.terminal_reduced_rank
    if reduced_rank is None or reduced_rank.width != face.reduced_width:
        raise EndpointLaplaceError("result is missing the face's reduced terminal rank")
    _validate_reduced_terminal_provenance(result, face=face, stored=reduced_rank)
    hessian_log_pdet = float(reduced_rank.log_pdet)
    penalized_optimizing = result.penalized_optimizing_log_likelihood
    assert penalized_optimizing is not None
    objective = -float(penalized_optimizing) + 0.5 * (hessian_log_pdet - projected_penalty.log_pdet)
    evaluation = EndpointLaplaceEvaluation(
        objective=objective,
        face_component_names=face.component_names,
        finite_component_names=projected_penalty.component_names,
        reduced_width=face.reduced_width,
        hessian_rank=reduced_rank.rank,
        penalty_rank=projected_penalty.rank,
        hessian_log_pdet=hessian_log_pdet,
        penalty_log_pdet=projected_penalty.log_pdet,
    )
    _record_penalty_objective(
        layout,
        result,
        lambdas=endpoint_lambdas,
        evaluation=penalty_evaluation,
        objective=evaluation.objective,
    )
    return evaluation
