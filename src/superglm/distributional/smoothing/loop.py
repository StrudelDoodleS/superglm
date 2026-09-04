"""Safeguarded EFS orchestration for dense distributional models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    DistributionalEFSResult,
    EFSConvergenceReason,
    JointEndpointDirectionEvidence,
    _assessment_is_numerically_stationary,
    _dense_penalty_fingerprint,
    _maximum_relative_natural_parameter_change,
)
from superglm.distributional.smoothing.acceleration import WindowedTypeIIAnderson
from superglm.distributional.smoothing.authority import (
    _face_authority_config,
    _fit_endpoint_authority_stationary,
    _fit_fixed_state,
    _optional_penalty_face,
)
from superglm.distributional.smoothing.endpoint_laml import EndpointDirectionEvidence
from superglm.distributional.smoothing.evidence import (
    _fresh_raw_evidence,
    _FreshRawEvidence,
    _lower_bound_pressure,
    _saturated_names,
)
from superglm.distributional.smoothing.faces import (
    _assess_joint_face_directions,
    _face_assessment_refusal_iteration,
    _face_retraction_iteration,
    _face_revalidation_iteration,
    _face_transition_iteration,
    _FaceDirectionAttempt,
    _FaceRetraction,
    _indexed_direction,
    _indexed_joint_direction,
    _joint_direction_is_endpoint,
    _joint_face_revalidation_iteration,
    _JointFaceDirectionAttempt,
    _recheck_exact_face,
    _try_exact_face,
    _try_joint_exact_face,
)
from superglm.distributional.smoothing.newton import (
    BracketAttempt,
    BracketRelease,
    EndgameState,
    NewtonEndgameOutcome,
    bracket_refused_component,
    newton_iteration_record,
    run_newton_endgame,
    should_hand_off,
)
from superglm.distributional.smoothing.objective import (
    _complete_mapping,
    _laplace_objective,
    _maximum_step,
    initialize_distributional_lambdas,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace, PenaltyFaceError
from superglm.distributional.smoothing.proposals import (
    _accelerated_proposal,
    _acceleration_provenance,
    _ordered_log_lambdas,
    _ordered_steps,
    _scaled_proposal,
)
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import _DenseObservedReuseSession, fit_dense_fixed_lambda
from superglm.distributional.timing import FitPhaseRecorder, measure_phase


def _efs_result(
    *,
    config: DistributionalEFSConfig,
    initial_lambdas: Mapping[str, float],
    lambdas: Mapping[str, float],
    initial_objective: float,
    objective: float,
    converged: bool,
    reason: EFSConvergenceReason,
    terminal_evidence: _FreshRawEvidence,
    history: list[DistributionalEFSIteration],
    coefficient_fits: list[DenseSolverResult],
    terminal_fit_index: int,
    terminal_endpoint_directions: Mapping[str, EndpointDirectionEvidence] | None = None,
    endgame: NewtonEndgameOutcome | None = None,
    beyond_cap_components: tuple[str, ...] = (),
) -> DistributionalEFSResult:
    derivatives = None if endgame is None else endgame.derivatives
    gradient = certificate = hessian = hessian_certificate = None
    if derivatives is not None:
        gradient = dict(zip(derivatives.names, derivatives.gradient.tolist(), strict=True))
        certificate = dict(
            zip(derivatives.names, derivatives.gradient_certificate.tolist(), strict=True)
        )
        hessian = derivatives.hessian
        hessian_certificate = derivatives.hessian_certificate
    return DistributionalEFSResult(
        config=config,
        initial_lambdas=initial_lambdas,
        lambdas=lambdas,
        initial_objective=initial_objective,
        objective=objective,
        converged=converged,
        convergence_reason=reason,
        terminal_raw_max_log_step=terminal_evidence.maximum,
        unresolved_upper_bound=terminal_evidence.unresolved_upper_bound,
        iterations=len(history),
        history=tuple(history),
        coefficient_fits=tuple(coefficient_fits),
        terminal_fit_index=terminal_fit_index,
        terminal_endpoint_directions=(
            {} if terminal_endpoint_directions is None else terminal_endpoint_directions
        ),
        terminal_gradient=gradient,
        terminal_gradient_certificate=certificate,
        terminal_projected_gradient_norm=(
            None if endgame is None else endgame.projected_gradient_norm
        ),
        smoothing_hessian=hessian,
        smoothing_hessian_certificate=hessian_certificate,
        newton_iterations=sum(item.stage == "newton" for item in history),
        bfgs_fallback_iterations=sum(item.step_source == "bfgs" for item in history),
        beyond_cap_components=tuple(beyond_cap_components),
    )


def fit_distributional_efs(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float] | None = None,
    solver_config: DenseSolverConfig | None = None,
    efs_config: DistributionalEFSConfig | None = None,
    initial: NDArray | None = None,
    chunk_size: ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
) -> DistributionalEFSResult:
    """Fit smoothing parameters with fresh joint coefficient geometry each iteration."""
    inner_config = DenseSolverConfig() if solver_config is None else solver_config
    outer_config = DistributionalEFSConfig() if efs_config is None else efs_config
    if not isinstance(inner_config, DenseSolverConfig):
        raise TypeError("solver_config must be DenseSolverConfig")
    if not isinstance(outer_config, DistributionalEFSConfig):
        raise TypeError("efs_config must be DistributionalEFSConfig")
    reuse_session = _DenseObservedReuseSession()

    with measure_phase(phase_recorder, "layout_penalty_assembly"):
        current_lambdas = initialize_distributional_lambdas(layout, lambdas, outer_config)
        initial_penalty = layout.penalty_matrix(current_lambdas)
    initial_lambdas = dict(current_lambdas)
    initial_fit = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        likelihood_plan,
        initial_penalty,
        initial=initial,
        config=inner_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        _reuse_session=reuse_session,
    )
    coefficient_fits = [initial_fit]
    history: list[DistributionalEFSIteration] = []
    terminal_fit_index = 0
    current_fit = initial_fit
    current_face: PenaltyFace | None = None
    with measure_phase(phase_recorder, "efs_update_backtracking"):
        current_objective = _laplace_objective(
            initial_fit,
            layout=layout,
            lambdas=current_lambdas,
            face=current_face,
        )
    initial_objective = current_objective
    with measure_phase(phase_recorder, "efs_update_backtracking"):
        terminal_evidence = _fresh_raw_evidence(
            layout,
            current_lambdas,
            current_fit,
            outer_config,
            face=current_face,
        )
    estimated_names = terminal_evidence.estimated_names

    if not estimated_names:
        return _efs_result(
            config=outer_config,
            initial_lambdas=initial_lambdas,
            lambdas=current_lambdas,
            initial_objective=initial_objective,
            objective=current_objective,
            converged=current_fit.converged,
            reason="fixed_only" if current_fit.converged else "coefficient_not_converged",
            terminal_evidence=terminal_evidence,
            history=history,
            coefficient_fits=coefficient_fits,
            terminal_fit_index=terminal_fit_index,
        )
    if not current_fit.converged:
        return _efs_result(
            config=outer_config,
            initial_lambdas=initial_lambdas,
            lambdas=current_lambdas,
            initial_objective=initial_objective,
            objective=current_objective,
            converged=False,
            reason="coefficient_not_converged",
            terminal_evidence=terminal_evidence,
            history=history,
            coefficient_fits=coefficient_fits,
            terminal_fit_index=terminal_fit_index,
        )
    # The Newton endgame's bookkeeping.  ``endgame_outcome`` is the last run's
    # outcome (its derivatives are the result's terminal gradient and Hessian);
    # ``endgame_stationary`` routes a stationary outcome through the strict-stop
    # block below so an active exact face is revalidated exactly as an EFS stop
    # would; ``pending_cap`` is the cap-pressure outcome the face block is
    # deciding; ``beyond_cap`` holds the raised upper bound of every component
    # the bracketed search released above ``maximum_lambda``.
    endgame_outcome: NewtonEndgameOutcome | None = None
    endgame_stationary = False
    pending_cap: NewtonEndgameOutcome | None = None
    newton_budget = outer_config.max_newton_iterations
    beyond_cap: dict[str, float] = {}

    def _result(
        *,
        converged: bool,
        reason: EFSConvergenceReason,
        terminal_endpoint_directions: Mapping[str, EndpointDirectionEvidence] | None = None,
    ) -> DistributionalEFSResult:
        return _efs_result(
            config=outer_config,
            initial_lambdas=initial_lambdas,
            lambdas=current_lambdas,
            initial_objective=initial_objective,
            objective=current_objective,
            converged=converged,
            reason=reason,
            terminal_evidence=terminal_evidence,
            history=history,
            coefficient_fits=coefficient_fits,
            terminal_fit_index=terminal_fit_index,
            terminal_endpoint_directions=terminal_endpoint_directions,
            endgame=endgame_outcome,
            beyond_cap_components=tuple(name for name in current_lambdas if name in beyond_cap),
        )

    def _strict_reason() -> EFSConvergenceReason:
        if endgame_stationary:
            return "stationary"
        return "objective_plateau" if plateau_qualified else "lambda_change"

    def _endgame_state() -> EndgameState:
        return EndgameState(
            lambdas=current_lambdas,
            fit=current_fit,
            objective=current_objective,
            face=current_face,
            evidence=terminal_evidence,
            terminal_fit_index=terminal_fit_index,
        )

    def _run_endgame() -> DistributionalEFSResult | None:
        """Run the endgame from the accepted state; a result ends the fit, ``None`` continues."""
        nonlocal current_lambdas, current_fit, current_objective, terminal_fit_index
        nonlocal terminal_evidence, endgame_outcome, endgame_stationary, pending_cap, newton_budget
        if newton_budget <= 0:
            return _result(converged=False, reason="max_iterations")
        outcome = run_newton_endgame(
            family,
            layout,
            y,
            likelihood_plan,
            state=_endgame_state(),
            solver_config=inner_config,
            efs_config=outer_config,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
            reuse_session=reuse_session,
            history=history,
            coefficient_fits=coefficient_fits,
            upper_bounds=beyond_cap,
            budget=newton_budget,
        )
        newton_budget -= outcome.newton_iterations
        endgame_outcome = outcome
        state = outcome.state
        current_lambdas = dict(state.lambdas)
        current_fit = state.fit
        current_objective = state.objective
        terminal_fit_index = state.terminal_fit_index
        terminal_evidence = state.evidence
        if outcome.kind == "stationary":
            # The exact gradient is the authority now: no component is at the cap
            # with an outward gradient, so a Fellner--Schall nomination is void.
            endgame_stationary = True
            pending_cap = None
            terminal_evidence = replace(terminal_evidence, unresolved_upper_bound=())
            return None
        if outcome.kind == "cap_pressure":
            pending_cap = outcome
            nominated = set(terminal_evidence.unresolved_upper_bound) | set(outcome.cap_pressure)
            terminal_evidence = replace(
                terminal_evidence,
                unresolved_upper_bound=tuple(name for name in current_lambdas if name in nominated),
            )
            return None
        return _result(converged=False, reason=outcome.kind)

    def _bracket_after_refusal(
        attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None,
    ) -> BracketRelease | BracketAttempt | None:
        """Search beyond the cap for the first refused component with a finite verdict."""
        if attempt is None or pending_cap is None or pending_cap.derivatives is None:
            return None
        directions: dict[str, EndpointDirectionEvidence] = {}
        if isinstance(attempt, _JointFaceDirectionAttempt):
            if attempt.direction is not None:
                directions = dict(attempt.direction.component_directions)
        elif attempt.check is not None:
            directions = {terminal_evidence.unresolved_upper_bound[0]: attempt.check.direction}
        for name, direction in directions.items():
            if direction.decision != "finite":
                continue
            found = bracket_refused_component(
                family,
                layout,
                y,
                likelihood_plan,
                name=name,
                endpoint_derivative=float(direction.analytic_derivative),
                pending=pending_cap,
                state=_endgame_state(),
                solver_config=inner_config,
                efs_config=outer_config,
                chunk_size=chunk_size,
                phase_recorder=phase_recorder,
                reuse_session=reuse_session,
            )
            if found is not None:
                return found
        return None

    def _accept_bracket_release(
        release: BracketRelease,
        attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None,
    ) -> None:
        """Record the released component as one accepted bracket iteration."""
        nonlocal current_lambdas, current_fit, current_objective, terminal_fit_index
        nonlocal terminal_evidence
        assert pending_cap is not None and pending_cap.derivatives is not None
        state = _endgame_state()
        assessment_fits = () if attempt is None else tuple(attempt.assessment_fits)
        first = len(coefficient_fits)
        coefficient_fits.extend(assessment_fits)
        coefficient_fits.extend(release.fits)
        indices = tuple(range(first, len(coefficient_fits)))
        tolerances = (
            *(() if attempt is None else (attempt.coefficient_tolerance,) * len(assessment_fits)),
            *((release.tolerance,) * len(release.fits)),
        )
        evidence = _fresh_raw_evidence(
            layout, release.lambdas, release.fit, outer_config, face=current_face
        )
        history.append(
            newton_iteration_record(
                iteration=len(history) + 1,
                state=state,
                proposed_lambdas=release.lambdas,
                lambdas_after=release.lambdas,
                objective_after=release.objective,
                evidence=evidence,
                fit_indices=indices,
                tolerances=tolerances,
                accepted_fit=release.fit,
                step_source="bracket",
                derivatives=pending_cap.derivatives,
                projected_gradient_norm=pending_cap.projected_gradient_norm or 0.0,
                hessian_certificate=None,
                ridge=None,
                estimated_names=estimated_names,
            )
        )
        current_lambdas = dict(release.lambdas)
        current_fit = release.fit
        current_objective = release.objective
        terminal_fit_index = indices[-1]
        terminal_evidence = evidence
        maximum_lambda_conditioning = outer_config.maximum_lambda_conditioning
        assert maximum_lambda_conditioning is not None
        beyond_cap[release.name] = maximum_lambda_conditioning

    def _record_bracket_attempt(attempt: BracketAttempt) -> None:
        """Record a search that found no root as one rejected bracket iteration."""
        assert pending_cap is not None and pending_cap.derivatives is not None
        first = len(coefficient_fits)
        coefficient_fits.extend(attempt.fits)
        indices = tuple(range(first, len(coefficient_fits)))
        history.append(
            newton_iteration_record(
                iteration=len(history) + 1,
                state=_endgame_state(),
                proposed_lambdas=current_lambdas,
                lambdas_after=current_lambdas,
                objective_after=current_objective,
                evidence=terminal_evidence,
                fit_indices=indices,
                tolerances=(attempt.tolerance,) * len(indices),
                accepted_fit=None,
                step_source="bracket",
                derivatives=pending_cap.derivatives,
                projected_gradient_norm=pending_cap.projected_gradient_norm or 0.0,
                hessian_certificate=None,
                ridge=None,
                estimated_names=estimated_names,
            )
        )

    plateau_run = 0
    practical_run = 0
    plateau_qualified = False
    previous_accepted_step = math.inf
    saturated_run: dict[str, int] = dict.fromkeys(estimated_names, 0)
    accelerator = (
        WindowedTypeIIAnderson(
            history=outer_config.acceleration_history,
            max_amplification=outer_config.acceleration_max_amplification,
        )
        if outer_config.acceleration == "multisecant"
        else None
    )
    minimum_log_lambda = math.log(outer_config.minimum_lambda)
    maximum_log_lambda = math.log(outer_config.maximum_lambda)
    while True:
        if terminal_evidence.unresolved_upper_bound:
            if len(history) >= outer_config.max_iterations:
                return _result(converged=False, reason="lambda_cap_unresolved")
            promotion, assessment_attempt = _try_joint_exact_face(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=current_lambdas,
                current_face=current_face,
                current_fit=current_fit,
                current_objective=current_objective,
                terminal_evidence=terminal_evidence,
                solver_config=inner_config,
                efs_config=outer_config,
                chunk_size=chunk_size,
                phase_recorder=phase_recorder,
                _reuse_session=reuse_session,
            )
            if promotion is None and assessment_attempt is None:
                promotion, assessment_attempt = _try_exact_face(
                    family,
                    layout,
                    y,
                    likelihood_plan,
                    lambdas=current_lambdas,
                    current_face=current_face,
                    current_fit=current_fit,
                    current_objective=current_objective,
                    terminal_evidence=terminal_evidence,
                    solver_config=inner_config,
                    efs_config=outer_config,
                    chunk_size=chunk_size,
                    phase_recorder=phase_recorder,
                    _reuse_session=reuse_session,
                )
            if promotion is None:
                if outer_config.outer == "efs+newton" and pending_cap is None:
                    if newton_budget > 0 and estimated_names:
                        # The FS proposal nominated this cap; the exact gradient decides
                        # before a refusal is recorded (these assessment fits are dropped).
                        endgame_result = _run_endgame()
                        if endgame_result is not None:
                            return endgame_result
                        continue
                elif outer_config.outer == "efs+newton" and pending_cap is not None:
                    bracket = _bracket_after_refusal(assessment_attempt)
                    if isinstance(bracket, BracketRelease):
                        _accept_bracket_release(bracket, assessment_attempt)
                        pending_cap = None
                        endgame_result = _run_endgame()
                        if endgame_result is not None:
                            return endgame_result
                        continue
                    if isinstance(bracket, BracketAttempt):
                        _record_bracket_attempt(bracket)
                if assessment_attempt is not None and assessment_attempt.assessment_fits:
                    if assessment_attempt.failure_reason == "endpoint_state_changed":
                        raise RuntimeError(
                            "endpoint state changes are valid only during face revalidation"
                        )
                    first_assessment_index = len(coefficient_fits)
                    coefficient_fits.extend(assessment_attempt.assessment_fits)
                    assessment_indices = tuple(range(first_assessment_index, len(coefficient_fits)))
                    if isinstance(assessment_attempt, _JointFaceDirectionAttempt):
                        component_names = assessment_attempt.component_names
                        refusal_direction = (
                            None
                            if assessment_attempt.direction is None
                            else _indexed_joint_direction(
                                assessment_attempt.direction,
                                assessment_indices,
                                assessment_attempt.coefficient_tolerance,
                            )
                        )
                    else:
                        component_names = (terminal_evidence.unresolved_upper_bound[0],)
                        refusal_direction = (
                            None
                            if assessment_attempt.check is None
                            else _indexed_direction(
                                assessment_attempt.check.direction,
                                assessment_indices,
                                assessment_attempt.coefficient_tolerance,
                            )
                        )
                    history.append(
                        _face_assessment_refusal_iteration(
                            iteration=len(history) + 1,
                            source_fit_index=terminal_fit_index,
                            coefficient_fit_indices=assessment_indices,
                            coefficient_tolerance=assessment_attempt.coefficient_tolerance,
                            component_names=component_names,
                            direction=refusal_direction,
                            failure_reason=assessment_attempt.failure_reason,
                            lambdas=current_lambdas,
                            objective=current_objective,
                            evidence=terminal_evidence,
                            source_fit=current_fit,
                        )
                    )
                return _result(converged=False, reason="lambda_cap_unresolved")
            source_fit_index = terminal_fit_index
            first_assessment_index = len(coefficient_fits)
            coefficient_fits.extend(promotion.assessment_fits)
            assessment_indices = tuple(range(first_assessment_index, len(coefficient_fits)))
            terminal_fit_index = assessment_indices[-1]
            if assessment_attempt is None:
                raise RuntimeError("an accepted exact face requires an authority assessment")
            coefficient_tolerance = assessment_attempt.coefficient_tolerance
            indexed_direction = (
                _indexed_joint_direction(
                    promotion.direction,
                    assessment_indices,
                    coefficient_tolerance,
                )
                if isinstance(promotion.direction, JointEndpointDirectionEvidence)
                else _indexed_direction(
                    promotion.direction,
                    assessment_indices,
                    coefficient_tolerance,
                )
            )
            promotion = replace(
                promotion,
                direction=indexed_direction,
            )
            history.append(
                _face_transition_iteration(
                    iteration=len(history) + 1,
                    source_fit_index=source_fit_index,
                    accepted_fit_index=terminal_fit_index,
                    coefficient_fit_indices=assessment_indices,
                    lambdas=current_lambdas,
                    objective_before=current_objective,
                    promotion=promotion,
                    evidence=terminal_evidence,
                    coefficient_tolerance=coefficient_tolerance,
                    source_fit=current_fit,
                )
            )
            current_face = promotion.face
            current_fit = promotion.fit
            current_objective = promotion.objective
            pending_cap = None
            endgame_outcome = None  # its derivatives named the promoted component
            with measure_phase(phase_recorder, "efs_update_backtracking"):
                terminal_evidence = _fresh_raw_evidence(
                    layout,
                    current_lambdas,
                    current_fit,
                    outer_config,
                    face=current_face,
                )
            estimated_names = terminal_evidence.estimated_names
            plateau_run = 0
            practical_run = 0
            plateau_qualified = False
            previous_accepted_step = math.inf
            saturated_run = dict.fromkeys(estimated_names, 0)
            if accelerator is not None:
                accelerator.reset()
            continue
        if endgame_stationary or (
            not terminal_evidence.working_infinity
            and terminal_evidence.maximum <= outer_config.tolerance
        ):
            if (
                not endgame_stationary
                and estimated_names
                and newton_budget > 0
                and should_hand_off(
                    "objective_plateau" if plateau_qualified else "lambda_change",
                    max_accepted_step=0.0,
                    iterations=len(history),
                    config=outer_config,
                )
            ):
                endgame_result = _run_endgame()
                if endgame_result is not None:
                    return endgame_result
                continue
            terminal_endpoint_directions: Mapping[str, EndpointDirectionEvidence] = {}
            if current_face is not None:
                revalidation_events = (
                    1
                    if len(current_face.component_names) > 1
                    else len(current_face.component_names)
                )
                if len(history) + revalidation_events > outer_config.max_iterations:
                    return _result(converged=False, reason="max_iterations")
                if len(current_face.component_names) > 1:
                    authority_config = _face_authority_config(inner_config)
                    try:
                        finite_faces = tuple(
                            (
                                name,
                                _optional_penalty_face(
                                    layout,
                                    tuple(
                                        item
                                        for item in current_face.component_names
                                        if item != name
                                    ),
                                ),
                            )
                            for name in current_face.component_names
                        )
                    except PenaltyFaceError:
                        joint_attempt = _JointFaceDirectionAttempt(
                            component_names=current_face.component_names,
                            direction=None,
                            assessment_fits=(),
                            coefficient_tolerance=authority_config.tolerance,
                            failure_reason="joint_analytic_unavailable",
                        )
                    else:
                        joint_attempt = _assess_joint_face_directions(
                            family,
                            layout,
                            y,
                            likelihood_plan,
                            lambdas=current_lambdas,
                            component_names=current_face.component_names,
                            joint_face=current_face,
                            finite_faces=finite_faces,
                            source_fit=current_fit,
                            source_objective=current_objective,
                            solver_config=inner_config,
                            efs_config=outer_config,
                            chunk_size=chunk_size,
                            phase_recorder=phase_recorder,
                            _reuse_session=reuse_session,
                        )
                    if _joint_direction_is_endpoint(joint_attempt):
                        source_fit_index = terminal_fit_index
                        source_fit = current_fit
                        objective_before = current_objective
                        first_assessment_index = len(coefficient_fits)
                        coefficient_fits.extend(joint_attempt.assessment_fits)
                        assessment_indices = tuple(
                            range(first_assessment_index, len(coefficient_fits))
                        )
                        terminal_fit_index = assessment_indices[0]
                        endpoint_fit = joint_attempt.assessment_fits[0]
                        direction = joint_attempt.direction
                        assert direction is not None
                        indexed_direction = _indexed_joint_direction(
                            direction,
                            assessment_indices,
                            joint_attempt.coefficient_tolerance,
                        )
                        history.append(
                            _joint_face_revalidation_iteration(
                                iteration=len(history) + 1,
                                source_fit_index=source_fit_index,
                                accepted_fit_index=terminal_fit_index,
                                coefficient_fit_indices=assessment_indices,
                                component_names=current_face.component_names,
                                lambdas=current_lambdas,
                                objective_before=objective_before,
                                endpoint_fit=endpoint_fit,
                                direction=indexed_direction,
                                coefficient_tolerance=joint_attempt.coefficient_tolerance,
                                evidence=terminal_evidence,
                                source_fit=source_fit,
                            )
                        )
                        current_fit = endpoint_fit
                        current_objective = indexed_direction.endpoint_objective
                        terminal_endpoint_directions = dict(indexed_direction.component_directions)
                        with measure_phase(phase_recorder, "efs_update_backtracking"):
                            terminal_evidence = _fresh_raw_evidence(
                                layout,
                                current_lambdas,
                                current_fit,
                                outer_config,
                                face=current_face,
                            )
                        if not endgame_stationary and (
                            terminal_evidence.working_infinity
                            or terminal_evidence.maximum > outer_config.tolerance
                        ):
                            continue
                        return _result(
                            converged=True,
                            reason=_strict_reason(),
                            terminal_endpoint_directions=terminal_endpoint_directions,
                        )
                    else:
                        rollback_penalty_fingerprint = _dense_penalty_fingerprint(
                            layout.penalty_matrix(current_lambdas)
                        )
                        rollback_initial = (
                            current_fit.coefficients
                            if not joint_attempt.assessment_fits
                            else joint_attempt.assessment_fits[-1].coefficients
                        )
                        rollback_fit = _fit_endpoint_authority_stationary(
                            family,
                            layout,
                            y,
                            likelihood_plan,
                            lambdas=current_lambdas,
                            face=None,
                            initial=rollback_initial,
                            config=authority_config,
                            chunk_size=chunk_size,
                            phase_recorder=phase_recorder,
                            _reuse_session=reuse_session,
                        )
                        rollback_objective = _laplace_objective(
                            rollback_fit,
                            layout=layout,
                            lambdas=current_lambdas,
                            face=None,
                        )
                        source_fit_index = terminal_fit_index
                        source_fit = current_fit
                        objective_before = current_objective
                        first_assessment_index = len(coefficient_fits)
                        rollback_fits = (*joint_attempt.assessment_fits, rollback_fit)
                        coefficient_fits.extend(rollback_fits)
                        assessment_indices = tuple(
                            range(first_assessment_index, len(coefficient_fits))
                        )
                        terminal_fit_index = assessment_indices[-1]
                        retraction_direction = joint_attempt.direction
                        if retraction_direction is not None:
                            retraction_direction = _indexed_joint_direction(
                                retraction_direction,
                                assessment_indices[:1],
                                joint_attempt.coefficient_tolerance,
                            )
                        retraction = _FaceRetraction(
                            component_names=current_face.component_names,
                            fit=rollback_fit,
                            objective=rollback_objective,
                            direction=retraction_direction,
                            failure_reason=(
                                joint_attempt.failure_reason
                                if joint_attempt.assessment_fits
                                else None
                            ),
                            coefficient_tolerance=joint_attempt.coefficient_tolerance,
                            assessment_fits=rollback_fits,
                            joint_rollback_penalty_fingerprint=(rollback_penalty_fingerprint),
                        )
                        history.append(
                            _face_retraction_iteration(
                                iteration=len(history) + 1,
                                source_fit_index=source_fit_index,
                                accepted_fit_index=terminal_fit_index,
                                coefficient_fit_indices=assessment_indices,
                                lambdas=current_lambdas,
                                objective_before=objective_before,
                                retraction=retraction,
                                evidence=terminal_evidence,
                                source_fit=source_fit,
                            )
                        )
                        current_face = None
                        current_fit = rollback_fit
                        current_objective = rollback_objective
                        with measure_phase(phase_recorder, "efs_update_backtracking"):
                            terminal_evidence = _fresh_raw_evidence(
                                layout,
                                current_lambdas,
                                current_fit,
                                outer_config,
                                face=current_face,
                            )
                        return _result(converged=False, reason="endpoint_revalidation_failed")
                recheck = _recheck_exact_face(
                    family,
                    layout,
                    y,
                    likelihood_plan,
                    lambdas=current_lambdas,
                    face=current_face,
                    current_fit=current_fit,
                    current_objective=current_objective,
                    solver_config=inner_config,
                    efs_config=outer_config,
                    chunk_size=chunk_size,
                    phase_recorder=phase_recorder,
                    _reuse_session=reuse_session,
                )
                resolved_directions: dict[str, EndpointDirectionEvidence] = {}
                for component_name, check in recheck.checks:
                    source_fit_index = terminal_fit_index
                    source_fit = current_fit
                    objective_before = current_objective
                    first_assessment_index = len(coefficient_fits)
                    coefficient_fits.extend(check.assessment_fits)
                    assessment_indices = tuple(range(first_assessment_index, len(coefficient_fits)))
                    terminal_fit_index = assessment_indices[-1]
                    direction = _indexed_direction(
                        check.direction,
                        assessment_indices,
                        check.coefficient_tolerance,
                    )
                    history.append(
                        _face_revalidation_iteration(
                            iteration=len(history) + 1,
                            source_fit_index=source_fit_index,
                            accepted_fit_index=terminal_fit_index,
                            coefficient_fit_indices=assessment_indices,
                            component_name=component_name,
                            lambdas=current_lambdas,
                            objective_before=objective_before,
                            check=check,
                            direction=direction,
                            evidence=terminal_evidence,
                            source_fit=source_fit,
                        )
                    )
                    current_fit = check.endpoint_fit
                    current_objective = check.endpoint_objective
                    resolved_directions[component_name] = direction
                    with measure_phase(phase_recorder, "efs_update_backtracking"):
                        terminal_evidence = _fresh_raw_evidence(
                            layout,
                            current_lambdas,
                            current_fit,
                            outer_config,
                            face=current_face,
                        )
                if recheck.retraction is not None:
                    retraction = recheck.retraction
                    source_fit_index = terminal_fit_index
                    source_fit = current_fit
                    objective_before = current_objective
                    first_assessment_index = len(coefficient_fits)
                    coefficient_fits.extend(retraction.assessment_fits)
                    assessment_indices = tuple(range(first_assessment_index, len(coefficient_fits)))
                    terminal_fit_index = assessment_indices[-1]
                    if retraction.direction is not None:
                        if not isinstance(retraction.direction, EndpointDirectionEvidence):
                            raise RuntimeError("scalar revalidation produced joint evidence")
                        retraction = replace(
                            retraction,
                            direction=_indexed_direction(
                                retraction.direction,
                                assessment_indices[:2],
                                retraction.coefficient_tolerance,
                            ),
                        )
                    history.append(
                        _face_retraction_iteration(
                            iteration=len(history) + 1,
                            source_fit_index=source_fit_index,
                            accepted_fit_index=terminal_fit_index,
                            coefficient_fit_indices=assessment_indices,
                            lambdas=current_lambdas,
                            objective_before=objective_before,
                            retraction=retraction,
                            evidence=terminal_evidence,
                            source_fit=source_fit,
                        )
                    )
                    current_face = None
                    current_fit = retraction.fit
                    current_objective = retraction.objective
                    with measure_phase(phase_recorder, "efs_update_backtracking"):
                        terminal_evidence = _fresh_raw_evidence(
                            layout,
                            current_lambdas,
                            current_fit,
                            outer_config,
                            face=current_face,
                        )
                    return _result(converged=False, reason="endpoint_revalidation_failed")
                terminal_endpoint_directions = resolved_directions
                if not endgame_stationary and (
                    terminal_evidence.working_infinity
                    or terminal_evidence.maximum > outer_config.tolerance
                ):
                    continue
            return _result(
                converged=True,
                reason=_strict_reason(),
                terminal_endpoint_directions=terminal_endpoint_directions,
            )
        if len(history) >= outer_config.max_iterations:
            if newton_budget > 0 and should_hand_off(
                "max_iterations",
                max_accepted_step=0.0,
                iterations=len(history),
                config=outer_config,
            ):
                endgame_result = _run_endgame()
                if endgame_result is not None:
                    return endgame_result
                continue
            return _result(converged=False, reason="max_iterations")
        if beyond_cap and outer_config.outer == "efs+newton":
            # A component released beyond the cap never meets the FS proposal's box.
            endgame_result = _run_endgame()
            if endgame_result is not None:
                return endgame_result
            continue

        iteration_number = len(history) + 1
        source_fit_index = terminal_fit_index
        with measure_phase(phase_recorder, "efs_update_backtracking"):
            components = terminal_evidence.components
            plain_update = terminal_evidence.update
            if plain_update is None:
                raise RuntimeError("estimated smoothing components require a raw EFS update")
            max_proposed_step = _maximum_step(plain_update, estimated_names)
            saturated = _saturated_names(
                components,
                plain_update,
                estimated_names,
                outer_config.boundary_saturation,
            )
            for name in estimated_names:
                saturated_run[name] = saturated_run[name] + 1 if name in saturated else 0
            nominations = frozenset(
                name
                for name in estimated_names
                if saturated_run[name] >= outer_config.boundary_iterations
            )

        acceleration_outcome = "disabled"
        acceleration_refusal_reason = None
        accelerated_proposal = None
        if accelerator is not None:
            if not all(plain_update.proposal_kinds[name] == "gfs" for name in estimated_names):
                accelerator.reset()
                acceleration_outcome = "refused"
                acceleration_refusal_reason = "non_gfs_proposal"
            else:
                with measure_phase(phase_recorder, "efs_update_backtracking"):
                    provenance = _acceleration_provenance(estimated_names, current_fit)
                    accelerator.record_accepted(
                        log_lambdas=_ordered_log_lambdas(current_lambdas, estimated_names),
                        raw_residual=_ordered_steps(
                            plain_update.raw_log_steps,
                            estimated_names,
                        ),
                        provenance=provenance,
                    )
                    decision = accelerator.propose(
                        max_log_step=outer_config.max_log_step,
                        minimum_log_lambda=minimum_log_lambda,
                        maximum_log_lambda=maximum_log_lambda,
                    )
                accelerated_proposal = decision.proposal
                if accelerated_proposal is None:
                    if decision.refusal_reason == "warming":
                        acceleration_outcome = "warming"
                    elif decision.refusal_reason is None:
                        raise RuntimeError("multisecant refusal must identify its reason")
                    else:
                        acceleration_outcome = "refused"
                        acceleration_refusal_reason = decision.refusal_reason

        attempted_fit_indices: list[int] = []
        attempted_tolerances: list[float] = []
        accepted_fit: DenseSolverResult | None = None
        accepted_fit_index: int | None = None
        accepted_lambdas: dict[str, float] | None = None
        accepted_log_steps: dict[str, float] | None = None
        accepted_objective: float | None = None
        any_converged_trial = False
        accelerated_fit_index: int | None = None
        raw_backtracks = 0
        proposed_lambdas: Mapping[str, float] = _complete_mapping(
            current_lambdas,
            plain_update.lambdas,
            missing=current_lambdas,
        )
        proposed_log_steps: Mapping[str, float] = _complete_mapping(
            current_lambdas,
            plain_update.log_steps,
            missing=0.0,
        )
        quadratic_forms = _complete_mapping(
            current_lambdas,
            plain_update.quadratic_forms,
            missing=0.0,
        )
        trace_terms = _complete_mapping(
            current_lambdas,
            plain_update.trace_terms,
            missing=0.0,
        )
        boundary_nominations = tuple(name for name in estimated_names if name in nominations)
        objective_ceiling = current_objective + outer_config.objective_tolerance * (
            1.0 + abs(current_objective)
        )

        accelerated_values = None
        trial_config = (
            inner_config if current_face is None else _face_authority_config(inner_config)
        )
        fit_trial_state = (
            _fit_fixed_state if current_face is None else _fit_endpoint_authority_stationary
        )
        reuse_kwargs = (
            {
                "_reuse_session": reuse_session,
                "_reuse_source": current_fit,
            }
            if current_face is None
            else {"_reuse_session": reuse_session}
        )
        if accelerated_proposal is not None:
            with measure_phase(phase_recorder, "efs_update_backtracking"):
                accelerated_values = _accelerated_proposal(
                    current_lambdas,
                    estimated_names,
                    accelerated_proposal.log_lambdas,
                    accelerated_proposal.log_step,
                    outer_config,
                )
            if accelerated_values is None:
                acceleration_outcome = "refused"
                acceleration_refusal_reason = "box_blocked"

        if accelerated_values is not None:
            acceleration_outcome = "rejected"
            trial_lambdas, trial_log_steps = accelerated_values
            trial_fit = fit_trial_state(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=trial_lambdas,
                face=current_face,
                initial=current_fit.coefficients,
                config=trial_config,
                chunk_size=chunk_size,
                phase_recorder=phase_recorder,
                **reuse_kwargs,  # ty: ignore[invalid-argument-type] -- correlated dispatch kwargs
            )
            coefficient_fits.append(trial_fit)
            fit_index = len(coefficient_fits) - 1
            attempted_fit_indices.append(fit_index)
            attempted_tolerances.append(trial_config.tolerance)
            accelerated_fit_index = fit_index
            trial_stationary = trial_fit.converged and (
                current_face is None
                or _assessment_is_numerically_stationary(
                    trial_fit,
                    trial_config.tolerance,
                )
            )
            any_converged_trial = any_converged_trial or trial_stationary
            if trial_stationary:
                with measure_phase(phase_recorder, "efs_update_backtracking"):
                    trial_objective = _laplace_objective(
                        trial_fit,
                        layout=layout,
                        lambdas=trial_lambdas,
                        face=current_face,
                    )
                if (
                    _acceleration_provenance(estimated_names, trial_fit) == provenance
                    and trial_objective <= objective_ceiling
                ):
                    accepted_fit = trial_fit
                    accepted_fit_index = fit_index
                    accepted_lambdas = trial_lambdas
                    accepted_log_steps = trial_log_steps
                    accepted_objective = trial_objective
                    acceleration_outcome = "accepted"
                    proposed_lambdas = trial_lambdas
                    proposed_log_steps = trial_log_steps
                    max_proposed_step = max(
                        (abs(trial_log_steps[name]) for name in estimated_names),
                        default=0.0,
                    )
            if accepted_fit is None:
                assert accelerator is not None
                accelerator.reject()

        if accepted_fit is None:
            for raw_backtrack in range(outer_config.max_backtracks + 1):
                step_scale = outer_config.backtrack_factor**raw_backtrack
                with measure_phase(phase_recorder, "efs_update_backtracking"):
                    trial_lambdas, trial_log_steps = _scaled_proposal(
                        current_lambdas,
                        plain_update.log_steps,
                        estimated_names,
                        step_scale,
                        outer_config,
                    )
                trial_fit = fit_trial_state(
                    family,
                    layout,
                    y,
                    likelihood_plan,
                    lambdas=trial_lambdas,
                    face=current_face,
                    initial=current_fit.coefficients,
                    config=trial_config,
                    chunk_size=chunk_size,
                    phase_recorder=phase_recorder,
                    **reuse_kwargs,  # ty: ignore[invalid-argument-type] -- correlated dispatch kwargs
                )
                coefficient_fits.append(trial_fit)
                fit_index = len(coefficient_fits) - 1
                attempted_fit_indices.append(fit_index)
                attempted_tolerances.append(trial_config.tolerance)
                raw_backtracks = raw_backtrack
                trial_stationary = trial_fit.converged and (
                    current_face is None
                    or _assessment_is_numerically_stationary(
                        trial_fit,
                        trial_config.tolerance,
                    )
                )
                any_converged_trial = any_converged_trial or trial_stationary
                if not trial_stationary:
                    continue
                with measure_phase(phase_recorder, "efs_update_backtracking"):
                    trial_objective = _laplace_objective(
                        trial_fit,
                        layout=layout,
                        lambdas=trial_lambdas,
                        face=current_face,
                    )
                if trial_objective <= objective_ceiling:
                    accepted_fit = trial_fit
                    accepted_fit_index = fit_index
                    accepted_lambdas = trial_lambdas
                    accepted_log_steps = trial_log_steps
                    accepted_objective = trial_objective
                    break

        if accepted_fit is None:
            zero_steps = {name: 0.0 for name in current_lambdas}
            history.append(
                DistributionalEFSIteration(
                    iteration=iteration_number,
                    source_fit_index=source_fit_index,
                    lambdas_before=current_lambdas,
                    proposed_lambdas=proposed_lambdas,
                    lambdas_after=current_lambdas,
                    proposed_log_steps=proposed_log_steps,
                    accepted_log_steps=zero_steps,
                    quadratic_forms=quadratic_forms,
                    trace_terms=trace_terms,
                    objective_before=current_objective,
                    objective_after=current_objective,
                    objective_relative_change=0.0,
                    max_proposed_log_step=max_proposed_step,
                    max_accepted_log_step=0.0,
                    accepted=False,
                    acceleration_outcome=acceleration_outcome,
                    acceleration_refusal_reason=acceleration_refusal_reason,
                    accelerated_fit_index=accelerated_fit_index,
                    backtracks=len(attempted_fit_indices) - 1,
                    raw_backtracks=raw_backtracks,
                    coefficient_fit_indices=tuple(attempted_fit_indices),
                    accepted_fit_index=None,
                    coefficient_tolerances=tuple(attempted_tolerances),
                    boundary_nominations=boundary_nominations,
                    update_curvature=current_fit.terminal_curvature,
                    accepted_curvature=None,
                )
            )
            reason: EFSConvergenceReason = (
                "objective_rejected" if any_converged_trial else "coefficient_not_converged"
            )
            if newton_budget > 0 and should_hand_off(
                reason, max_accepted_step=0.0, iterations=len(history), config=outer_config
            ):
                endgame_result = _run_endgame()
                if endgame_result is not None:
                    return endgame_result
                continue
            return _result(converged=False, reason=reason)

        assert accepted_fit_index is not None
        assert accepted_lambdas is not None
        assert accepted_log_steps is not None
        assert accepted_objective is not None
        objective_change = abs(accepted_objective - current_objective) / (
            1.0 + abs(current_objective)
        )
        max_accepted_step = max(
            (abs(accepted_log_steps[name]) for name in estimated_names),
            default=0.0,
        )
        practical_parameter_change = (
            _maximum_relative_natural_parameter_change(
                current_fit.theta,
                accepted_fit.theta,
            )
            if outer_config.practical_convergence
            else math.inf
        )
        history.append(
            DistributionalEFSIteration(
                iteration=iteration_number,
                source_fit_index=source_fit_index,
                lambdas_before=current_lambdas,
                proposed_lambdas=proposed_lambdas,
                lambdas_after=accepted_lambdas,
                proposed_log_steps=proposed_log_steps,
                accepted_log_steps=accepted_log_steps,
                quadratic_forms=quadratic_forms,
                trace_terms=trace_terms,
                objective_before=current_objective,
                objective_after=accepted_objective,
                objective_relative_change=objective_change,
                max_proposed_log_step=max_proposed_step,
                max_accepted_log_step=max_accepted_step,
                accepted=True,
                acceleration_outcome=acceleration_outcome,
                acceleration_refusal_reason=acceleration_refusal_reason,
                accelerated_fit_index=accelerated_fit_index,
                backtracks=len(attempted_fit_indices) - 1,
                raw_backtracks=raw_backtracks,
                coefficient_fit_indices=tuple(attempted_fit_indices),
                accepted_fit_index=accepted_fit_index,
                coefficient_tolerances=tuple(attempted_tolerances),
                boundary_nominations=boundary_nominations,
                update_curvature=current_fit.terminal_curvature,
                accepted_curvature=accepted_fit.terminal_curvature,
            )
        )
        current_lambdas = accepted_lambdas
        current_fit = accepted_fit
        current_objective = accepted_objective
        terminal_fit_index = accepted_fit_index
        with measure_phase(phase_recorder, "efs_update_backtracking"):
            terminal_evidence = _fresh_raw_evidence(
                layout,
                current_lambdas,
                current_fit,
                outer_config,
                face=current_face,
            )
        if terminal_evidence.estimated_names != estimated_names:
            raise RuntimeError("estimated smoothing names changed inside one EFS fit")
        # The step test alone is a first-order quantity on a harmonic tail; the
        # objective it exists to settle can be flat for many iterations before
        # the step is. Require both a flat objective and a contracting step, so
        # a stalled line search cannot nominate a plateau reason. Fresh
        # evidence above remains the strict stopping authority.  The practical
        # policy below intentionally accepts its separate sustained plateau
        # while preserving the larger strict residual in the result.
        plateau_run = plateau_run + 1 if objective_change <= outer_config.plateau_tolerance else 0
        plateau_qualified = (
            plateau_run >= outer_config.plateau_iterations
            and max_accepted_step <= previous_accepted_step
        )
        if outer_config.practical_convergence:
            assert terminal_evidence.update is not None
            lower_pressure = _lower_bound_pressure(
                terminal_evidence,
                current_lambdas,
                terminal_evidence.update.raw_log_steps,
                outer_config,
            )
            practical_run = (
                practical_run + 1
                if objective_change <= outer_config.plateau_tolerance
                and practical_parameter_change <= outer_config.practical_parameter_tolerance
                and max_accepted_step <= previous_accepted_step
                and not lower_pressure
                else 0
            )
        previous_accepted_step = max_accepted_step
        # A practical plateau is an interior stop.  While any estimated
        # component sits at the cap with outward pressure, or is a working
        # infinity, the top of the loop owns the next decision: it either
        # promotes an exact face or refuses with a named reason.  Returning
        # here would publish a worse point one iteration early.
        if (
            outer_config.practical_convergence
            and current_face is None
            and practical_run >= outer_config.plateau_iterations
            and terminal_evidence.maximum > outer_config.tolerance
            and not terminal_evidence.unresolved_upper_bound
            and not terminal_evidence.working_infinity
        ):
            if newton_budget > 0 and should_hand_off(
                "practical_plateau",
                max_accepted_step=max_accepted_step,
                iterations=len(history),
                config=outer_config,
            ):
                endgame_result = _run_endgame()
                if endgame_result is not None:
                    return endgame_result
                continue
            return _result(converged=True, reason="practical_plateau")
        if newton_budget > 0 and should_hand_off(
            None,
            max_accepted_step=max_accepted_step,
            iterations=len(history),
            config=outer_config,
        ):
            endgame_result = _run_endgame()
            if endgame_result is not None:
                return endgame_result
