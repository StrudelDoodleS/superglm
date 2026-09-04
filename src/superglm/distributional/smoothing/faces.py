"""Exact infinity-face assessment: direction checks, joint faces, revalidation, retraction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    ANALYTIC_DIRECTION_AUTHORITY,
    JOINT_ANALYTIC_DIRECTION_AUTHORITY,
    JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY,
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    EndpointAssessmentFailureReason,
    JointEndpointDirectionEvidence,
    _assessment_is_numerically_stationary,
)
from superglm.distributional.smoothing.authority import (
    _endpoint_candidate_refit_bound,
    _endpoint_retained_rank,
    _endpoint_shared_provenance,
    _face_authority_config,
    _fit_endpoint_authority_stationary,
    _is_sole_cap_outside_face,
    _optional_penalty_face,
)
from superglm.distributional.smoothing.endpoint_laml import (
    EndpointDirectionEvidence,
    EndpointLaplaceError,
    evaluate_endpoint_laplace_derivative,
    resolve_endpoint_direction,
)
from superglm.distributional.smoothing.evidence import _FreshRawEvidence
from superglm.distributional.smoothing.objective import _complete_mapping, _laplace_objective
from superglm.distributional.smoothing.penalty_face import (
    PenaltyFace,
    PenaltyFaceError,
    build_penalty_face,
)
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import _DenseObservedReuseSession
from superglm.distributional.timing import FitPhaseRecorder, measure_phase


@dataclass(frozen=True)
class _FacePromotion:
    """One accepted exact-face transition at the current finite state."""

    component_names: tuple[str, ...]
    face: PenaltyFace
    fit: DenseSolverResult
    objective: float
    direction: EndpointDirectionEvidence | JointEndpointDirectionEvidence
    assessment_fits: tuple[DenseSolverResult, ...]


@dataclass(frozen=True)
class _FaceDirectionCheck:
    """One endpoint direction together with its reusable finite-cap fit."""

    cap_fit: DenseSolverResult
    cap_objective: float
    endpoint_fit: DenseSolverResult
    endpoint_objective: float
    direction: EndpointDirectionEvidence
    coefficient_tolerance: float

    @property
    def assessment_fits(self) -> tuple[DenseSolverResult, ...]:
        """Tight finite-cap and exact-endpoint fits."""

        return (self.cap_fit, self.endpoint_fit)


@dataclass(frozen=True)
class _FaceDirectionAttempt:
    """All coefficient fits executed while trying one endpoint direction."""

    check: _FaceDirectionCheck | None
    assessment_fits: tuple[DenseSolverResult, ...]
    coefficient_tolerance: float
    failure_reason: EndpointAssessmentFailureReason | Literal["endpoint_state_changed"] | None


@dataclass(frozen=True)
class _JointFaceDirectionAttempt:
    """The sole fitted joint face and its all-or-nothing direction receipt."""

    component_names: tuple[str, ...]
    direction: JointEndpointDirectionEvidence | None
    assessment_fits: tuple[DenseSolverResult, ...]
    coefficient_tolerance: float
    failure_reason: EndpointAssessmentFailureReason | None


@dataclass(frozen=True)
class _FaceRetraction:
    """A terminal face component whose endpoint direction no longer certifies."""

    component_names: tuple[str, ...]
    fit: DenseSolverResult
    objective: float
    direction: EndpointDirectionEvidence | JointEndpointDirectionEvidence | None
    failure_reason: EndpointAssessmentFailureReason | None
    coefficient_tolerance: float
    assessment_fits: tuple[DenseSolverResult, ...]
    joint_rollback_penalty_fingerprint: str | None = None


@dataclass(frozen=True)
class _FaceRecheck:
    """Successful component checks followed by an optional whole-face rollback."""

    checks: tuple[tuple[str, _FaceDirectionCheck], ...]
    retraction: _FaceRetraction | None


def _check_face_direction(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    component_name: str,
    finite_face: PenaltyFace | None,
    endpoint_face: PenaltyFace,
    initial: NDArray,
    endpoint_initial: NDArray | None = None,
    allow_nonstationary_cap: bool = False,
    solver_config: DenseSolverConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> _FaceDirectionAttempt:
    """Evaluate one τ = 0+ direction, refusing unstable fit provenance."""
    authority_config = _face_authority_config(solver_config)

    def refused(
        reason: EndpointAssessmentFailureReason | Literal["endpoint_state_changed"] | None,
        *fits: DenseSolverResult,
    ) -> _FaceDirectionAttempt:
        return _FaceDirectionAttempt(
            check=None,
            assessment_fits=tuple(fits),
            coefficient_tolerance=authority_config.tolerance,
            failure_reason=reason,
        )

    cap = float(lambdas[component_name])
    if cap <= 0.0 or not math.isfinite(cap):
        return refused(None)
    cap_fit = _fit_endpoint_authority_stationary(
        family,
        layout,
        y,
        likelihood_plan,
        lambdas=lambdas,
        face=finite_face,
        initial=initial,
        config=authority_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=_reuse_session,
    )
    if not cap_fit.converged:
        return refused("cap_not_converged", cap_fit)
    cap_stationary = _assessment_is_numerically_stationary(
        cap_fit,
        authority_config.tolerance,
    )
    if not cap_stationary and not allow_nonstationary_cap:
        return refused("cap_not_stationary", cap_fit)
    with measure_phase(phase_recorder, "efs_update_backtracking"):
        cap_objective = _laplace_objective(
            cap_fit,
            layout=layout,
            lambdas=lambdas,
            face=finite_face,
        )
    endpoint_fit = _fit_endpoint_authority_stationary(
        family,
        layout,
        y,
        likelihood_plan,
        lambdas=lambdas,
        face=endpoint_face,
        initial=cap_fit.coefficients if endpoint_initial is None else endpoint_initial,
        config=authority_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=_reuse_session,
    )
    if not endpoint_fit.converged:
        return refused("endpoint_not_converged", cap_fit, endpoint_fit)
    if not _assessment_is_numerically_stationary(
        endpoint_fit,
        authority_config.tolerance,
    ):
        return refused(
            "endpoint_not_stationary" if cap_stationary else "analytic_unavailable",
            cap_fit,
            endpoint_fit,
        )
    if not cap_stationary:
        endpoint_rank = _endpoint_retained_rank(endpoint_fit)
        endpoint_curvature = endpoint_fit.terminal_curvature
        if (
            endpoint_rank is None
            or endpoint_rank.rank != endpoint_face.reduced_width
            or endpoint_curvature.requested_source != "observed"
            or endpoint_curvature.actual_source != "observed"
            or endpoint_curvature.fallback_count != 0
        ):
            return refused("analytic_unavailable", cap_fit, endpoint_fit)
    if _endpoint_shared_provenance(endpoint_fit) != _endpoint_shared_provenance(cap_fit):
        return refused("provenance_changed", cap_fit, endpoint_fit)
    if endpoint_initial is not None and not np.array_equal(
        endpoint_fit.coefficients,
        endpoint_initial,
    ):
        if cap_stationary or not allow_nonstationary_cap:
            return refused("endpoint_state_changed", cap_fit, endpoint_fit)
        endpoint_bound = _endpoint_candidate_refit_bound(
            np.asarray(endpoint_initial),
            endpoint_fit,
            tolerance=authority_config.tolerance,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            endpoint_movement = float(
                np.max(
                    np.abs(endpoint_fit.coefficients - endpoint_initial),
                    initial=0.0,
                )
            )
        if (
            endpoint_bound is None
            or not math.isfinite(endpoint_movement)
            or endpoint_movement > endpoint_bound
        ):
            return refused("endpoint_state_changed", cap_fit, endpoint_fit)
    with measure_phase(phase_recorder, "efs_update_backtracking"):
        resolved_endpoint_objective = _laplace_objective(
            endpoint_fit,
            layout=layout,
            lambdas=lambdas,
            face=endpoint_face,
        )
        try:
            analytic = evaluate_endpoint_laplace_derivative(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=lambdas,
                component_name=component_name,
                finite_face=finite_face,
                endpoint_face=endpoint_face,
                endpoint_fit=endpoint_fit,
            )
        except EndpointLaplaceError:
            return refused("analytic_unavailable", cap_fit, endpoint_fit)
    with measure_phase(phase_recorder, "efs_update_backtracking"):
        direction = resolve_endpoint_direction(
            resolved_endpoint_objective,
            analytic=analytic,
        )
    check = _FaceDirectionCheck(
        cap_fit=cap_fit,
        cap_objective=cap_objective,
        endpoint_fit=endpoint_fit,
        endpoint_objective=resolved_endpoint_objective,
        direction=direction,
        coefficient_tolerance=authority_config.tolerance,
    )
    return _FaceDirectionAttempt(
        check=check,
        assessment_fits=check.assessment_fits,
        coefficient_tolerance=authority_config.tolerance,
        failure_reason=None,
    )


def _assess_joint_face_directions(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    component_names: tuple[str, ...],
    joint_face: PenaltyFace,
    finite_faces: tuple[tuple[str, PenaltyFace | None], ...],
    source_fit: DenseSolverResult,
    source_objective: float,
    solver_config: DenseSolverConfig,
    efs_config: DistributionalEFSConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> _JointFaceDirectionAttempt:
    """Evaluate named directions from one strict common exact-face fit."""
    authority_config = _face_authority_config(solver_config)

    def refused(
        reason: EndpointAssessmentFailureReason | None,
        fit: DenseSolverResult,
        *,
        direction: JointEndpointDirectionEvidence | None = None,
    ) -> _JointFaceDirectionAttempt:
        return _JointFaceDirectionAttempt(
            component_names=component_names,
            direction=direction,
            assessment_fits=(fit,),
            coefficient_tolerance=authority_config.tolerance,
            failure_reason=reason,
        )

    endpoint_fit = _fit_endpoint_authority_stationary(
        family,
        layout,
        y,
        likelihood_plan,
        lambdas=lambdas,
        face=joint_face,
        initial=source_fit.coefficients,
        config=authority_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=_reuse_session,
    )
    if not endpoint_fit.converged:
        return refused("joint_endpoint_not_converged", endpoint_fit)
    if not _assessment_is_numerically_stationary(
        endpoint_fit,
        authority_config.tolerance,
    ):
        return refused("joint_endpoint_not_stationary", endpoint_fit)
    source_rank = _endpoint_retained_rank(source_fit)
    endpoint_rank = _endpoint_retained_rank(endpoint_fit)
    endpoint_curvature = endpoint_fit.terminal_curvature
    if (
        source_rank is None
        or endpoint_rank is None
        or endpoint_fit.family_likelihood_plan_identifier
        != source_fit.family_likelihood_plan_identifier
        or endpoint_fit.execution_backend_identifier != source_fit.execution_backend_identifier
        or endpoint_rank.policy_version != source_rank.policy_version
        or endpoint_curvature.requested_source != "observed"
        or endpoint_curvature.actual_source != "observed"
        or endpoint_curvature.fallback_count != 0
    ):
        return refused("joint_analytic_unavailable", endpoint_fit)

    with measure_phase(phase_recorder, "efs_update_backtracking"):
        try:
            endpoint_objective = _laplace_objective(
                endpoint_fit,
                layout=layout,
                lambdas=lambdas,
                face=joint_face,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return refused("joint_analytic_unavailable", endpoint_fit)
    current_ceiling = source_objective + efs_config.objective_tolerance * (
        1.0 + abs(source_objective)
    )
    if endpoint_objective > current_ceiling:
        return refused("joint_objective_rejected", endpoint_fit)

    component_directions: list[tuple[str, EndpointDirectionEvidence]] = []
    with measure_phase(phase_recorder, "efs_update_backtracking"):
        for name, finite_face in finite_faces:
            try:
                analytic = evaluate_endpoint_laplace_derivative(
                    family,
                    layout,
                    y,
                    likelihood_plan,
                    lambdas=lambdas,
                    component_name=name,
                    finite_face=finite_face,
                    endpoint_face=joint_face,
                    endpoint_fit=endpoint_fit,
                )
            except EndpointLaplaceError:
                return refused("joint_analytic_unavailable", endpoint_fit)
            component_directions.append(
                (
                    name,
                    resolve_endpoint_direction(endpoint_objective, analytic=analytic),
                )
            )
    direction = JointEndpointDirectionEvidence(
        authority_identifier=(
            JOINT_ANALYTIC_DIRECTION_AUTHORITY
            if all(
                evidence.authority_identifier == ANALYTIC_DIRECTION_AUTHORITY
                for _name, evidence in component_directions
            )
            else JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY
        ),
        component_directions=tuple(component_directions),
    )
    return _JointFaceDirectionAttempt(
        component_names=component_names,
        direction=direction,
        assessment_fits=(endpoint_fit,),
        coefficient_tolerance=authority_config.tolerance,
        failure_reason=None,
    )


def _joint_direction_is_endpoint(attempt: _JointFaceDirectionAttempt) -> bool:
    direction = attempt.direction
    return bool(
        direction is not None
        and all(
            component_direction.decision == "endpoint" and component_direction.lower_bound > 0.0
            for _name, component_direction in direction.component_directions
        )
    )


def _try_joint_exact_face(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    current_face: PenaltyFace | None,
    current_fit: DenseSolverResult,
    current_objective: float,
    terminal_evidence: _FreshRawEvidence,
    solver_config: DenseSolverConfig,
    efs_config: DistributionalEFSConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> tuple[_FacePromotion | None, _JointFaceDirectionAttempt | None]:
    """Assess all capped non-active components from one common exact-face fit."""

    active_names = () if current_face is None else current_face.component_names
    active = frozenset(active_names)
    estimated = frozenset(terminal_evidence.estimated_names)
    positive_pressure = frozenset(terminal_evidence.unresolved_upper_bound)
    candidate_names = tuple(
        name
        for name in layout.penalty_names
        if name in estimated and name not in active and lambdas[name] == efs_config.maximum_lambda
    )
    nominated_names = tuple(name for name in candidate_names if name in positive_pressure)
    if not nominated_names or len(candidate_names) < 2:
        return None, None

    selected = frozenset((*active_names, *candidate_names))
    joint_names = tuple(name for name in layout.penalty_names if name in selected)
    try:
        joint_face = build_penalty_face(layout, joint_names)
        finite_faces = tuple(
            (
                name,
                _optional_penalty_face(
                    layout,
                    tuple(item for item in joint_face.component_names if item != name),
                ),
            )
            for name in candidate_names
        )
    except PenaltyFaceError:
        return None, None

    attempt = _assess_joint_face_directions(
        family,
        layout,
        y,
        likelihood_plan,
        lambdas=lambdas,
        component_names=candidate_names,
        joint_face=joint_face,
        finite_faces=finite_faces,
        source_fit=current_fit,
        source_objective=current_objective,
        solver_config=solver_config,
        efs_config=efs_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=_reuse_session,
    )
    if not _joint_direction_is_endpoint(attempt):
        return None, attempt
    direction = attempt.direction
    assert direction is not None
    endpoint_fit = attempt.assessment_fits[0]
    return (
        _FacePromotion(
            component_names=candidate_names,
            face=joint_face,
            fit=endpoint_fit,
            objective=direction.endpoint_objective,
            direction=direction,
            assessment_fits=attempt.assessment_fits,
        ),
        attempt,
    )


def _try_exact_face(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    current_face: PenaltyFace | None,
    current_fit: DenseSolverResult,
    current_objective: float,
    terminal_evidence: _FreshRawEvidence,
    solver_config: DenseSolverConfig,
    efs_config: DistributionalEFSConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> tuple[_FacePromotion | None, _FaceDirectionAttempt | None]:
    """Promote one capped coordinate only after a resolved τ = 0+ test."""
    if not terminal_evidence.unresolved_upper_bound:
        return None, None
    component_name = terminal_evidence.unresolved_upper_bound[0]
    active_names = () if current_face is None else current_face.component_names
    selected_names = frozenset((*active_names, component_name))
    ordered_names = tuple(name for name in layout.penalty_names if name in selected_names)
    try:
        face = build_penalty_face(layout, ordered_names)
    except PenaltyFaceError:
        return None, None
    sole_capped_component = _is_sole_cap_outside_face(
        layout,
        lambdas,
        component_name=component_name,
        finite_face=current_face,
        maximum_lambda=efs_config.maximum_lambda,
    )
    attempt = _check_face_direction(
        family,
        layout,
        y,
        likelihood_plan,
        lambdas=lambdas,
        component_name=component_name,
        finite_face=current_face,
        endpoint_face=face,
        initial=current_fit.coefficients,
        allow_nonstationary_cap=sole_capped_component,
        solver_config=solver_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=_reuse_session,
    )
    check = attempt.check
    if check is None:
        return None, attempt
    if not sole_capped_component:
        return None, attempt
    current_ceiling = current_objective + efs_config.objective_tolerance * (
        1.0 + abs(current_objective)
    )
    cap_ceiling = check.cap_objective + efs_config.objective_tolerance * (
        1.0 + abs(check.cap_objective)
    )
    cap_stationary = _assessment_is_numerically_stationary(
        check.cap_fit,
        check.coefficient_tolerance,
    )
    if (
        check.direction.decision != "endpoint"
        or check.direction.lower_bound <= 0.0
        or check.endpoint_objective > current_ceiling
        or (cap_stationary and check.endpoint_objective > cap_ceiling)
    ):
        return None, attempt
    return (
        _FacePromotion(
            component_names=(component_name,),
            face=face,
            fit=check.endpoint_fit,
            objective=check.endpoint_objective,
            direction=check.direction,
            assessment_fits=check.assessment_fits,
        ),
        attempt,
    )


def _recheck_exact_face(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    face: PenaltyFace,
    current_fit: DenseSolverResult,
    current_objective: float,
    solver_config: DenseSolverConfig,
    efs_config: DistributionalEFSConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> _FaceRecheck:
    """Revalidate every active component at the terminal finite smoothing state."""
    checks: list[tuple[str, _FaceDirectionCheck]] = []
    canonical_fit = current_fit
    accepted_objective = current_objective
    for component_name in face.component_names:
        remaining_names = tuple(name for name in face.component_names if name != component_name)
        sole_capped_component = False
        try:
            finite_face = _optional_penalty_face(layout, remaining_names)
        except PenaltyFaceError:
            attempt = None
        else:
            sole_capped_component = _is_sole_cap_outside_face(
                layout,
                lambdas,
                component_name=component_name,
                finite_face=finite_face,
                maximum_lambda=efs_config.maximum_lambda,
            )
            attempt = _check_face_direction(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=lambdas,
                component_name=component_name,
                finite_face=finite_face,
                endpoint_face=face,
                initial=canonical_fit.coefficients,
                endpoint_initial=canonical_fit.coefficients,
                allow_nonstationary_cap=sole_capped_component,
                solver_config=solver_config,
                chunk_size=chunk_size,
                phase_recorder=phase_recorder,
                _reuse_session=_reuse_session,
            )
        check = None if attempt is None else attempt.check
        direction_resolved = (
            sole_capped_component and check is not None and check.direction.decision == "endpoint"
        )
        objective_resolved = False
        if check is not None:
            current_ceiling = accepted_objective + efs_config.objective_tolerance * (
                1.0 + abs(accepted_objective)
            )
            cap_ceiling = check.cap_objective + efs_config.objective_tolerance * (
                1.0 + abs(check.cap_objective)
            )
            cap_stationary = _assessment_is_numerically_stationary(
                check.cap_fit,
                check.coefficient_tolerance,
            )
            objective_resolved = check.endpoint_objective <= current_ceiling and (
                not cap_stationary or check.endpoint_objective <= cap_ceiling
            )
        if not direction_resolved or not objective_resolved:
            authority_config = _face_authority_config(solver_config)
            cap_fit = _fit_endpoint_authority_stationary(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=lambdas,
                face=None,
                initial=canonical_fit.coefficients,
                config=authority_config,
                chunk_size=chunk_size,
                phase_recorder=phase_recorder,
                _reuse_session=_reuse_session,
            )
            cap_objective = _laplace_objective(
                cap_fit,
                layout=layout,
                lambdas=lambdas,
                face=None,
            )
            failed_assessments = () if attempt is None else attempt.assessment_fits
            return _FaceRecheck(
                checks=tuple(checks),
                retraction=_FaceRetraction(
                    component_names=face.component_names,
                    fit=cap_fit,
                    objective=cap_objective,
                    direction=(
                        check.direction
                        if check is not None and check.direction.decision != "endpoint"
                        else None
                    ),
                    failure_reason=None,
                    coefficient_tolerance=authority_config.tolerance,
                    assessment_fits=(*failed_assessments, cap_fit),
                ),
            )
        checks.append((component_name, check))
        canonical_fit = check.endpoint_fit
        accepted_objective = check.endpoint_objective
    return _FaceRecheck(checks=tuple(checks), retraction=None)


def _face_assessment_refusal_iteration(
    *,
    iteration: int,
    source_fit_index: int,
    coefficient_fit_indices: tuple[int, ...],
    coefficient_tolerance: float,
    component_names: tuple[str, ...],
    direction: EndpointDirectionEvidence | JointEndpointDirectionEvidence | None,
    failure_reason: EndpointAssessmentFailureReason | None,
    lambdas: Mapping[str, float],
    objective: float,
    evidence: _FreshRawEvidence,
    source_fit: DenseSolverResult,
) -> DistributionalEFSIteration:
    """Record an endpoint assessment that left the accepted state unchanged."""

    update = evidence.update
    if update is None:
        raise RuntimeError("a face assessment requires a fresh EFS update")
    zeros = {name: 0.0 for name in lambdas}
    return DistributionalEFSIteration(
        iteration=iteration,
        source_fit_index=source_fit_index,
        lambdas_before=lambdas,
        proposed_lambdas=lambdas,
        lambdas_after=lambdas,
        proposed_log_steps=zeros,
        accepted_log_steps=zeros,
        quadratic_forms=_complete_mapping(lambdas, update.quadratic_forms, missing=0.0),
        trace_terms=_complete_mapping(lambdas, update.trace_terms, missing=0.0),
        objective_before=objective,
        objective_after=objective,
        objective_relative_change=0.0,
        max_proposed_log_step=0.0,
        max_accepted_log_step=0.0,
        accepted=False,
        acceleration_outcome="disabled",
        acceleration_refusal_reason=None,
        accelerated_fit_index=None,
        backtracks=0,
        raw_backtracks=0,
        coefficient_fit_indices=coefficient_fit_indices,
        accepted_fit_index=None,
        coefficient_tolerances=(coefficient_tolerance,) * len(coefficient_fit_indices),
        boundary_nominations=(),
        update_curvature=source_fit.terminal_curvature,
        accepted_curvature=None,
        refused_face_components=component_names,
        endpoint_direction_evidence=direction,
        endpoint_assessment_failure_reason=failure_reason,
    )


def _face_transition_iteration(
    *,
    iteration: int,
    source_fit_index: int,
    accepted_fit_index: int,
    coefficient_fit_indices: tuple[int, ...],
    lambdas: Mapping[str, float],
    objective_before: float,
    promotion: _FacePromotion,
    evidence: _FreshRawEvidence,
    coefficient_tolerance: float,
    source_fit: DenseSolverResult,
) -> DistributionalEFSIteration:
    update = evidence.update
    if update is None:
        raise RuntimeError("a face transition requires a fresh EFS update")
    zeros = {name: 0.0 for name in lambdas}
    quadratic_forms = _complete_mapping(lambdas, update.quadratic_forms, missing=0.0)
    trace_terms = _complete_mapping(lambdas, update.trace_terms, missing=0.0)
    relative_change = abs(promotion.objective - objective_before) / (1.0 + abs(objective_before))
    return DistributionalEFSIteration(
        iteration=iteration,
        source_fit_index=source_fit_index,
        lambdas_before=lambdas,
        proposed_lambdas=lambdas,
        lambdas_after=lambdas,
        proposed_log_steps=zeros,
        accepted_log_steps=zeros,
        quadratic_forms=quadratic_forms,
        trace_terms=trace_terms,
        objective_before=objective_before,
        objective_after=promotion.objective,
        objective_relative_change=relative_change,
        max_proposed_log_step=0.0,
        max_accepted_log_step=0.0,
        accepted=True,
        acceleration_outcome="disabled",
        acceleration_refusal_reason=None,
        accelerated_fit_index=None,
        backtracks=0,
        raw_backtracks=0,
        coefficient_fit_indices=coefficient_fit_indices,
        accepted_fit_index=accepted_fit_index,
        coefficient_tolerances=(coefficient_tolerance,) * len(coefficient_fit_indices),
        boundary_nominations=(),
        update_curvature=source_fit.terminal_curvature,
        accepted_curvature=promotion.fit.terminal_curvature,
        activated_face_components=promotion.component_names,
        endpoint_direction_evidence=promotion.direction,
    )


def _indexed_direction(
    direction: EndpointDirectionEvidence,
    fit_indices: tuple[int, ...],
    coefficient_tolerance: float,
) -> EndpointDirectionEvidence:
    if len(fit_indices) != 2:
        raise RuntimeError("endpoint assessment omitted coefficient fits")
    return replace(
        direction,
        fit_indices=fit_indices,
        coefficient_tolerance=coefficient_tolerance,
    )


def _indexed_joint_direction(
    direction: JointEndpointDirectionEvidence,
    fit_indices: tuple[int, ...],
    coefficient_tolerance: float,
) -> JointEndpointDirectionEvidence:
    if len(fit_indices) != 1:
        raise RuntimeError("joint endpoint assessment omitted its common coefficient fit")
    return replace(
        direction,
        endpoint_fit_index=fit_indices[0],
        coefficient_tolerance=coefficient_tolerance,
    )


def _face_revalidation_iteration(
    *,
    iteration: int,
    source_fit_index: int,
    accepted_fit_index: int,
    coefficient_fit_indices: tuple[int, ...],
    component_name: str,
    lambdas: Mapping[str, float],
    objective_before: float,
    check: _FaceDirectionCheck,
    direction: EndpointDirectionEvidence,
    evidence: _FreshRawEvidence,
    source_fit: DenseSolverResult,
) -> DistributionalEFSIteration:
    update = evidence.update
    zeros = {name: 0.0 for name in lambdas}
    quadratic_forms = (
        zeros if update is None else _complete_mapping(lambdas, update.quadratic_forms, missing=0.0)
    )
    trace_terms = (
        zeros if update is None else _complete_mapping(lambdas, update.trace_terms, missing=0.0)
    )
    relative_change = abs(check.endpoint_objective - objective_before) / (
        1.0 + abs(objective_before)
    )
    return DistributionalEFSIteration(
        iteration=iteration,
        source_fit_index=source_fit_index,
        lambdas_before=lambdas,
        proposed_lambdas=lambdas,
        lambdas_after=lambdas,
        proposed_log_steps=zeros,
        accepted_log_steps=zeros,
        quadratic_forms=quadratic_forms,
        trace_terms=trace_terms,
        objective_before=objective_before,
        objective_after=check.endpoint_objective,
        objective_relative_change=relative_change,
        max_proposed_log_step=0.0,
        max_accepted_log_step=0.0,
        accepted=True,
        acceleration_outcome="disabled",
        acceleration_refusal_reason=None,
        accelerated_fit_index=None,
        backtracks=0,
        raw_backtracks=0,
        coefficient_fit_indices=coefficient_fit_indices,
        accepted_fit_index=accepted_fit_index,
        coefficient_tolerances=(check.coefficient_tolerance,) * len(coefficient_fit_indices),
        boundary_nominations=(),
        update_curvature=source_fit.terminal_curvature,
        accepted_curvature=check.endpoint_fit.terminal_curvature,
        revalidated_face_components=(component_name,),
        endpoint_direction_evidence=direction,
    )


def _joint_face_revalidation_iteration(
    *,
    iteration: int,
    source_fit_index: int,
    accepted_fit_index: int,
    coefficient_fit_indices: tuple[int, ...],
    component_names: tuple[str, ...],
    lambdas: Mapping[str, float],
    objective_before: float,
    endpoint_fit: DenseSolverResult,
    direction: JointEndpointDirectionEvidence,
    coefficient_tolerance: float,
    evidence: _FreshRawEvidence,
    source_fit: DenseSolverResult,
) -> DistributionalEFSIteration:
    update = evidence.update
    zeros = {name: 0.0 for name in lambdas}
    quadratic_forms = (
        zeros if update is None else _complete_mapping(lambdas, update.quadratic_forms, missing=0.0)
    )
    trace_terms = (
        zeros if update is None else _complete_mapping(lambdas, update.trace_terms, missing=0.0)
    )
    return DistributionalEFSIteration(
        iteration=iteration,
        source_fit_index=source_fit_index,
        lambdas_before=lambdas,
        proposed_lambdas=lambdas,
        lambdas_after=lambdas,
        proposed_log_steps=zeros,
        accepted_log_steps=zeros,
        quadratic_forms=quadratic_forms,
        trace_terms=trace_terms,
        objective_before=objective_before,
        objective_after=direction.endpoint_objective,
        objective_relative_change=abs(direction.endpoint_objective - objective_before)
        / (1.0 + abs(objective_before)),
        max_proposed_log_step=0.0,
        max_accepted_log_step=0.0,
        accepted=True,
        acceleration_outcome="disabled",
        acceleration_refusal_reason=None,
        accelerated_fit_index=None,
        backtracks=0,
        raw_backtracks=0,
        coefficient_fit_indices=coefficient_fit_indices,
        accepted_fit_index=accepted_fit_index,
        coefficient_tolerances=(coefficient_tolerance,),
        boundary_nominations=(),
        update_curvature=source_fit.terminal_curvature,
        accepted_curvature=endpoint_fit.terminal_curvature,
        revalidated_face_components=component_names,
        endpoint_direction_evidence=direction,
    )


def _face_retraction_iteration(
    *,
    iteration: int,
    source_fit_index: int,
    accepted_fit_index: int,
    coefficient_fit_indices: tuple[int, ...],
    lambdas: Mapping[str, float],
    objective_before: float,
    retraction: _FaceRetraction,
    evidence: _FreshRawEvidence,
    source_fit: DenseSolverResult,
) -> DistributionalEFSIteration:
    update = evidence.update
    zeros = {name: 0.0 for name in lambdas}
    quadratic_forms = (
        zeros if update is None else _complete_mapping(lambdas, update.quadratic_forms, missing=0.0)
    )
    trace_terms = (
        zeros if update is None else _complete_mapping(lambdas, update.trace_terms, missing=0.0)
    )
    return DistributionalEFSIteration(
        iteration=iteration,
        source_fit_index=source_fit_index,
        lambdas_before=lambdas,
        proposed_lambdas=lambdas,
        lambdas_after=lambdas,
        proposed_log_steps=zeros,
        accepted_log_steps=zeros,
        quadratic_forms=quadratic_forms,
        trace_terms=trace_terms,
        objective_before=objective_before,
        objective_after=retraction.objective,
        objective_relative_change=abs(retraction.objective - objective_before)
        / (1.0 + abs(objective_before)),
        max_proposed_log_step=0.0,
        max_accepted_log_step=0.0,
        accepted=True,
        acceleration_outcome="disabled",
        acceleration_refusal_reason=None,
        accelerated_fit_index=None,
        backtracks=0,
        raw_backtracks=0,
        coefficient_fit_indices=coefficient_fit_indices,
        accepted_fit_index=accepted_fit_index,
        coefficient_tolerances=(retraction.coefficient_tolerance,) * len(coefficient_fit_indices),
        boundary_nominations=(),
        update_curvature=source_fit.terminal_curvature,
        accepted_curvature=retraction.fit.terminal_curvature,
        deactivated_face_components=retraction.component_names,
        endpoint_direction_evidence=retraction.direction,
        endpoint_assessment_failure_reason=retraction.failure_reason,
        joint_rollback_penalty_fingerprint=(retraction.joint_rollback_penalty_fingerprint),
    )
