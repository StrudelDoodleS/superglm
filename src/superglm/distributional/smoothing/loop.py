"""Safeguarded EFS orchestration for dense distributional models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.results.endpoint_evidence import (
    EFSConvergenceReason,
    JointEndpointDirectionEvidence,
)
from superglm.distributional.results.iteration import (
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    _maximum_relative_natural_parameter_change,
)
from superglm.distributional.results.smoothing import (
    DistributionalEFSResult,
    _practical_outward_window,
)
from superglm.distributional.results.solver import (
    DenseSolverConfig,
    DenseSolverResult,
    _assessment_is_numerically_stationary,
)
from superglm.distributional.smoothing.acceleration import WindowedTypeIIAnderson
from superglm.distributional.smoothing.authority import (
    _face_authority_config,
    _fit_endpoint_authority_stationary,
    _fit_fixed_state,
    _optional_penalty_face,
)
from superglm.distributional.smoothing.endpoint_laml import (
    EndpointDirectionEvidence,
    EndpointLaplaceError,
)
from superglm.distributional.smoothing.evidence import (
    _fresh_raw_evidence,
    _FreshRawEvidence,
    _lower_bound_pressure,
)
from superglm.distributional.smoothing.faces import (
    _assess_joint_face_directions,
    _direction_leaves_face,
    _EndpointDerivativeMemo,
    _face_assessment_refusal_iteration,
    _face_retraction_iteration,
    _face_revalidation_iteration,
    _face_transition_iteration,
    _FaceDirectionAttempt,
    _FaceDirectionCheck,
    _FacePromotion,
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
from superglm.distributional.smoothing.history import compact_coefficient_history
from superglm.distributional.smoothing.initialization import prepare_distributional_initialization
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
    no_higher_within_rounding,
    rank_matched_objectives,
    remaining_fall,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace, PenaltyFaceError
from superglm.distributional.smoothing.penalty_geometry import _PenaltyEvaluationMemo
from superglm.distributional.smoothing.proposals import (
    _accelerated_proposal,
    _acceleration_provenance,
    _endpoint_proposal,
    _ordered_log_lambdas,
    _ordered_steps,
    _scaled_proposal,
    endpoint_trial_names,
    resolved_flat_tails,
)
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import _DenseObservedReuseSession, fit_dense_fixed_lambda
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.reml.penalty_support import PenaltyNumericalError


def _within_ceiling(
    current: DenseSolverResult,
    current_objective: float,
    trial: DenseSolverResult,
    trial_objective: float,
    config: DistributionalEFSConfig,
) -> bool:
    """Accept a trial no worse than the current objective, on a common subspace.

    A rise inside the rounding of the two objectives is none the objective
    resolves (``no_higher_within_rounding``), and beside a penalty held at its
    bound in the data's coordinates that rounding is several times the
    tolerance. The tie goes to the proposal: halving a step answers a rise the
    objective shows (Wood and Fasiolo, Biometrics 73(4), 2017, sec. 2), so no
    step length is chosen inside it.
    """
    reference, candidate = rank_matched_objectives(
        current, current_objective, trial, trial_objective
    )
    return no_higher_within_rounding(
        candidate, trial, reference, current, config.objective_tolerance
    )


def _trial_laplace_objective(
    result: DenseSolverResult,
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
    objective_tolerance: float,
    penalty_memo: _PenaltyEvaluationMemo | None = None,
) -> float | None:
    """Reject an uncertifiable penalty trial without changing accepted state."""
    try:
        return _laplace_objective(
            result,
            layout=layout,
            lambdas=lambdas,
            face=face,
            objective_tolerance=objective_tolerance,
            penalty_memo=penalty_memo,
        )
    except PenaltyNumericalError:
        return None
    except EndpointLaplaceError as exc:
        if isinstance(exc.__cause__, PenaltyNumericalError):
            return None
        raise


def _efs_result(
    *,
    layout: StackedLayout,
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
    compact_coefficient_history(
        coefficient_fits, history, terminal_fit_index=terminal_fit_index, config=config
    )
    derivatives = None if endgame is None else endgame.derivatives
    gradient = certificate = hessian = hessian_certificate = None
    projected_norm = None if endgame is None else endgame.projected_gradient_norm
    if derivatives is not None:
        gradient = dict(zip(derivatives.names, derivatives.gradient.tolist(), strict=True))
        certificate = dict(
            zip(derivatives.names, derivatives.gradient_certificate.tolist(), strict=True)
        )
        hessian = derivatives.hessian
        hessian_certificate = derivatives.hessian_certificate
    elif reason == "stationary" and not terminal_evidence.estimated_names:
        # Every estimated smoothing parameter is on a validated exact face.
        gradient, certificate, projected_norm = {}, {}, 0.0
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
        terminal_raw_log_steps=(
            {
                name: (
                    terminal_evidence.update.raw_log_steps[name]
                    if name in terminal_evidence.estimated_names
                    else 0.0
                )
                for name in lambdas
            }
            if terminal_evidence.update is not None and terminal_evidence.estimated_names
            else None
        ),
        terminal_projected_gradient_norm=projected_norm,
        smoothing_hessian=hessian,
        smoothing_hessian_certificate=hessian_certificate,
        newton_iterations=sum(item.stage == "newton" for item in history),
        bfgs_fallback_iterations=sum(item.step_source == "bfgs" for item in history),
        beyond_cap_components=tuple(beyond_cap_components),
    )


# The plateau finish's bar as a fraction of the plateau's stationarity tolerance.
_FINISH_TOLERANCE = 0.1
# The Newton iterations a plateau finish may spend. A finish past this count is on a
# fit whose exact gradient gives it no quick way down, which keeps its plateau.
_FINISH_ITERATIONS = 16


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
    search = _SmoothingSearch(
        family, layout, y, likelihood_plan, inner_config, outer_config, chunk_size, phase_recorder
    )
    search.fit_start(lambdas, initial)
    return search.run()


class _SmoothingSearch:
    """One smoothing-parameter search, for one call of ``fit_distributional_efs``.

    Its Fellner--Schall rounds, cap and face decisions and Newton endgame share
    the accepted state and the bookkeeping held here.
    """

    def __init__(
        self,
        family: DistributionalFamily,
        layout: StackedLayout,
        y: NDArray,
        likelihood_plan: FamilyLikelihoodPlan,
        inner_config: DenseSolverConfig,
        outer_config: DistributionalEFSConfig,
        chunk_size: ChunkSize | None,
        phase_recorder: FitPhaseRecorder | None,
    ) -> None:
        self.family = family
        self.layout = layout
        self.y = y
        self.likelihood_plan = likelihood_plan
        self.inner_config = inner_config
        self.outer_config = outer_config
        self.chunk_size = chunk_size
        self.phase_recorder = phase_recorder
        self.reuse_session = _DenseObservedReuseSession()
        self.penalty_memo = _PenaltyEvaluationMemo()
        self.endpoint_derivatives = _EndpointDerivativeMemo()

    def fit_start(self, lambdas: Mapping[str, float] | None, initial: NDArray | None) -> None:
        """Fit the start: its smoothing parameters, coefficients, objective and evidence."""
        with measure_phase(self.phase_recorder, "layout_penalty_assembly"):
            current_lambdas, initial, lambda_units = prepare_distributional_initialization(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                supplied=lambdas,
                config=self.outer_config,
                solver_config=self.inner_config,
                initial=initial,
                chunk_size=self.chunk_size,
                reuse_session=self.reuse_session,
                phase_recorder=self.phase_recorder,
            )
            if lambda_units:
                self.outer_config = replace(self.outer_config, lambda_units=lambda_units)
            initial_penalty = self.layout.penalty_matrix(current_lambdas)
        self.initial_lambdas = dict(current_lambdas)
        initial_fit = fit_dense_fixed_lambda(
            self.family,
            self.layout,
            self.y,
            self.likelihood_plan,
            initial_penalty,
            initial=initial,
            config=self.inner_config,
            chunk_size=self.chunk_size,
            phase_recorder=self.phase_recorder,
            _reuse_session=self.reuse_session,
        )
        self.coefficient_fits = [initial_fit]
        self.history: list[DistributionalEFSIteration] = []
        terminal_fit_index = 0
        current_fit = initial_fit
        current_face: PenaltyFace | None = None
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            current_objective = _laplace_objective(
                initial_fit,
                layout=self.layout,
                lambdas=current_lambdas,
                face=current_face,
                objective_tolerance=self.outer_config.objective_tolerance,
            )
        self.initial_objective = current_objective
        del initial_fit  # current_fit and history own it until their live window ends.
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            terminal_evidence = _fresh_raw_evidence(
                self.layout,
                current_lambdas,
                current_fit,
                self.outer_config,
                face=current_face,
            )
        self.state = EndgameState(
            lambdas=current_lambdas,
            fit=current_fit,
            objective=current_objective,
            face=current_face,
            evidence=terminal_evidence,
            terminal_fit_index=terminal_fit_index,
        )
        self.estimated_names = terminal_evidence.estimated_names

    def run(self) -> DistributionalEFSResult:
        """Search from the start fit until a phase returns the result."""
        if not self.estimated_names:
            return _efs_result(
                layout=self.layout,
                config=self.outer_config,
                initial_lambdas=self.initial_lambdas,
                lambdas=self.state.lambdas,
                initial_objective=self.initial_objective,
                objective=self.state.objective,
                converged=self.state.fit.converged,
                reason="fixed_only" if self.state.fit.converged else "coefficient_not_converged",
                terminal_evidence=self.state.evidence,
                history=self.history,
                coefficient_fits=self.coefficient_fits,
                terminal_fit_index=self.state.terminal_fit_index,
            )
        if not self.state.fit.converged:
            return _efs_result(
                layout=self.layout,
                config=self.outer_config,
                initial_lambdas=self.initial_lambdas,
                lambdas=self.state.lambdas,
                initial_objective=self.initial_objective,
                objective=self.state.objective,
                converged=False,
                reason="coefficient_not_converged",
                terminal_evidence=self.state.evidence,
                history=self.history,
                coefficient_fits=self.coefficient_fits,
                terminal_fit_index=self.state.terminal_fit_index,
            )
        # The Newton endgame's bookkeeping.  ``endgame_outcome`` is the last run's
        # outcome (its derivatives are the result's terminal gradient and Hessian);
        # ``endgame_stationary`` routes a stationary outcome through the strict-stop
        # block below so an active exact face is revalidated exactly as an EFS stop
        # would; ``pending_cap`` is the cap-pressure outcome the face block is
        # deciding; ``beyond_cap`` holds the raised upper bound of every component
        # the bracketed search released above ``maximum_lambda``.
        self.endgame_outcome: NewtonEndgameOutcome | None = None
        self.stationary_search = self.outer_config.outer == "efs+stationary"
        self.endgame_stationary = False
        self.pending_cap: NewtonEndgameOutcome | None = None
        self.newton_budget = (
            (
                _FINISH_ITERATIONS
                if self.outer_config.outer == "efs+finish"
                else self.outer_config.max_iterations
            )
            if self.outer_config.max_newton_iterations is None
            else self.outer_config.max_newton_iterations
        )
        # ``"efs+finish"`` finishes a plateau on the exact gradient, an order tighter than
        # the plateau's own bar.
        self.endgame_config = (
            replace(
                self.outer_config,
                outer="efs+newton",
                tolerance=_FINISH_TOLERANCE * self.outer_config.tolerance,
            )
            if self.outer_config.outer == "efs+finish"
            else self.outer_config
        )
        self.beyond_cap: dict[str, float] = {}
        # Under plain Fellner--Schall a cap whose promotion is refused is held as an
        # active bound while the other components move (Burke and Moré 1988), so a cap
        # does not stop them. ``cap_held`` names those caps and ``refused_fit`` is the fit
        # the last refusal judged. Every stop returns the held caps to the evidence, so
        # they are judged again where the other components end; ``caps_final`` records
        # that, and from then on a refusal is the fit's answer.
        self.cap_held: frozenset[str] = frozenset()
        self.refused_fit: DenseSolverResult | None = None
        self.caps_final = False
        # A multiplicative update keeps every iterate interior, as an interior-point method
        # does (Zhou, Hu, Zhou and Lange, arXiv 1509.07426), so a component heading for its
        # upper bound never arrives. One whose working problem falls all the way there is
        # tried at the bound once (``endpoint_tried``), and the objective and the evidence
        # at the bound decide. An accepted trial (``endpoint_caps``) holds its components at
        # their caps like a refused promotion's, so the other components' fits stay off the
        # face and it is assessed once, where they end. That stop is theirs: ``endgame_due``
        # returns a search whose endgame waits on the assessment straight to it, and
        # ``plateau_earned`` keeps a practical plateau through it. The exact gradient in log
        # lambda vanishes at a cap whatever the multiplier of the bound tau = 1/lambda >= 0,
        # which is what decides an optimum on it (Mishchenko, Holmgren and Ronnegard, arXiv
        # 0711.2619, section 4.3), so a stationary endgame hands the caps a trial reached to
        # the assessor (``handed_over``), whose refusal leaves its verdict standing.
        # The trial takes at once the fall that component's walk spreads over the rounds
        # to come. On that fall the walk's rounds are accepted, whatever the other
        # components' steps do to the objective, up to the plateau the default search
        # finishes. With the fall taken the same steps can be rejected at every length,
        # so beside a cap a trial holds a rejected round goes to the finish as that
        # plateau does. It has earned no plateau (``unearned``), so a finish that stalls
        # leaves the round's verdict.
        self.endpoint_tried: set[str] = set()
        self.endpoint_caps: set[str] = set()
        self.endgame_due = False
        self.handed_over = False
        self.plateau_earned = False
        self.unearned = False
        # The objective's fall over the last two accepted Fellner--Schall rounds.
        self.fall = self.previous_fall = math.inf

        self.plateau_run = 0
        self.practical_run = 0
        self.plateau_qualified = False
        self.previous_accepted_step = math.inf
        self.accelerator = (
            WindowedTypeIIAnderson(
                history=self.outer_config.acceleration_history,
                max_amplification=self.outer_config.acceleration_max_amplification,
            )
            if self.outer_config.acceleration == "multisecant"
            else None
        )
        while True:
            compact_coefficient_history(
                self.coefficient_fits,
                self.history,
                terminal_fit_index=self.state.terminal_fit_index,
                config=self.outer_config,
            )
            evidence = self.state.evidence
            if evidence.unresolved_upper_bound:
                result = self._decide_caps()
            elif self._stop_earned() or (
                not evidence.working_infinity and evidence.maximum <= self.outer_config.tolerance
            ):
                result = self._strict_stop()
            elif len(self.history) >= self.outer_config.max_iterations:
                result = self._out_of_rounds()
            elif self.beyond_cap and self.outer_config.outer in ("efs+newton", "efs+stationary"):
                # A component released beyond the cap never meets the FS proposal's box.
                result = self._run_endgame()
            else:
                result = self._fellner_schall_round()
            if result is not None:
                return result

    def _decide_caps(self) -> DistributionalEFSResult | None:
        """A cap is pressed outward: promote it to an exact face, or answer for the refusal."""
        if (len(self.history) >= self.outer_config.max_iterations and not self.plateau_earned) or (
            self.caps_final and self.state.fit is self.refused_fit
        ):
            # No round is left, or these caps were refused at this very fit. Held caps
            # no assessment has refused were never judged: the search ran out of rounds.
            # Judging the caps of a plateau already earned is its terminal check, not a
            # search step, so the round limit does not refuse it.
            unjudged = self.caps_final and self.refused_fit is None
            return self._result(
                converged=False,
                reason="max_iterations" if unjudged else "lambda_cap_unresolved",
            )
        promotion, attempt = self._assess_caps()
        if promotion is None:
            return self._caps_refused(attempt)
        return self._promote(promotion, attempt)

    def _assess_caps(
        self,
    ) -> tuple[_FacePromotion | None, _FaceDirectionAttempt | _JointFaceDirectionAttempt | None]:
        """Assess the pressed caps jointly, or one by one when the joint assessor has no answer."""
        for assess in (_try_joint_exact_face, _try_exact_face):
            promotion, attempt = assess(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                lambdas=self.state.lambdas,
                current_face=self.state.face,
                current_fit=self.state.fit,
                current_objective=self.state.objective,
                terminal_evidence=self.state.evidence,
                solver_config=self.inner_config,
                efs_config=self.outer_config,
                chunk_size=self.chunk_size,
                phase_recorder=self.phase_recorder,
                _reuse_session=self.reuse_session,
                derivatives=self.endpoint_derivatives,
            )
            if promotion is not None or attempt is not None:
                return promotion, attempt
        return None, None

    def _caps_refused(
        self, attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None
    ) -> DistributionalEFSResult | None:
        """After a refusal: the exact gradient decides first, the caps are held, or the fit ends."""
        if self.handed_over:
            pass  # the endgame's verdict stands, with the refusal recorded
        elif self.outer_config.outer != "efs" and self.pending_cap is None:
            if self._endgame_budget() > 0 and self.estimated_names:
                # The FS proposal nominated this cap; the exact gradient decides
                # before a refusal is recorded (these assessment fits are dropped).
                return self._run_endgame()
        elif (
            self.outer_config.outer in ("efs+newton", "efs+stationary")
            and self.pending_cap is not None
        ):
            bracket = self._bracket_after_refusal(attempt)
            if isinstance(bracket, BracketRelease):
                self._accept_bracket_release(bracket, attempt)
                self.pending_cap = None
                return self._run_endgame()
            if isinstance(bracket, BracketAttempt):
                self._record_bracket_attempt(bracket)
        self._record_refusal(attempt)
        if self.handed_over:
            self.endpoint_caps.difference_update(self.state.evidence.unresolved_upper_bound)
            self.endgame_due = self.handed_over = False
            self.state = replace(
                self.state,
                evidence=replace(self.state.evidence, unresolved_upper_bound=()),
            )
            return None
        if self.outer_config.outer == "efs" and not self.caps_final:
            self.refused_fit = self.state.fit
            self.cap_held |= frozenset(self.state.evidence.unresolved_upper_bound)
            self.state = replace(self.state, evidence=self._evidence())
            return None
        return self._result(converged=False, reason="lambda_cap_unresolved")

    def _record_refusal(
        self, attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None
    ) -> None:
        """Record a refused assessment that made fits as one history row."""
        if attempt is None or not attempt.assessment_fits:
            return
        assessment_indices = self._append_fits(attempt.assessment_fits)
        if isinstance(attempt, _JointFaceDirectionAttempt):
            component_names = attempt.component_names
            refusal_direction = (
                None
                if attempt.direction is None
                else _indexed_joint_direction(
                    attempt.direction,
                    assessment_indices,
                    attempt.coefficient_tolerance,
                )
            )
        else:
            component_names = (self.state.evidence.unresolved_upper_bound[0],)
            refusal_direction = (
                None
                if attempt.check is None
                else _indexed_direction(
                    attempt.check.direction,
                    assessment_indices,
                    attempt.coefficient_tolerance,
                )
            )
        self.history.append(
            _face_assessment_refusal_iteration(
                iteration=len(self.history) + 1,
                source_fit_index=self.state.terminal_fit_index,
                coefficient_fit_indices=assessment_indices,
                coefficient_tolerance=attempt.coefficient_tolerance,
                component_names=component_names,
                direction=refusal_direction,
                failure_reason=attempt.failure_reason,
                lambdas=self.state.lambdas,
                objective=self.state.objective,
                evidence=self.state.evidence,
                source_fit=self.state.fit,
            )
        )

    def _promote(
        self,
        promotion: _FacePromotion,
        attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None,
    ) -> DistributionalEFSResult | None:
        """Take the assessed exact face: record the promotion and adopt its fit."""
        source_fit_index = self.state.terminal_fit_index
        assessment_indices = self._append_fits(promotion.assessment_fits)
        self.state = replace(self.state, terminal_fit_index=assessment_indices[-1])
        if attempt is None:
            raise RuntimeError("an accepted exact face requires an authority assessment")
        coefficient_tolerance = attempt.coefficient_tolerance
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
        self.history.append(
            _face_transition_iteration(
                iteration=len(self.history) + 1,
                source_fit_index=source_fit_index,
                accepted_fit_index=self.state.terminal_fit_index,
                coefficient_fit_indices=assessment_indices,
                lambdas=self.state.lambdas,
                objective_before=self.state.objective,
                promotion=promotion,
                evidence=self.state.evidence,
                coefficient_tolerance=coefficient_tolerance,
                source_fit=self.state.fit,
            )
        )
        self.state = replace(
            self.state,
            face=promotion.face,
            fit=promotion.fit,
            objective=promotion.objective,
        )
        self._active_set_changed(held_promoted=True)
        resume = self.endgame_due and bool(self.estimated_names)
        # A stationary verdict whose handed caps were all that was left to estimate
        # stands: there is no gradient to take again.
        self.endgame_stationary = self.handed_over and not self.estimated_names
        self.endgame_due = self.handed_over = False
        if resume:
            return self._run_endgame()
        return None

    def _append_fits(self, fits: Sequence[DenseSolverResult]) -> tuple[int, ...]:
        """Append ``fits`` to the search's coefficient fits and return their indices."""
        first = len(self.coefficient_fits)
        self.coefficient_fits.extend(fits)
        return tuple(range(first, len(self.coefficient_fits)))

    def _stop_earned(self) -> bool:
        """Whether a stationary endgame or an earned plateau has stopped the search."""
        return self.endgame_stationary or self.plateau_earned

    def _strict_stop(self) -> DistributionalEFSResult | None:
        """The free components have stopped: judge held caps, hand off, revalidate the face."""
        if self.cap_held:
            # The free terms have stopped: judge the stop again with the caps in it.
            self._release_held_caps()
            return None
        if self._stop_hands_off():
            return self._run_endgame()
        face = self.state.face
        if face is None:
            return self._converged({})
        revalidation_events = 1 if len(face.component_names) > 1 else len(face.component_names)
        # Revalidating the face of a stationary endgame or an earned plateau is
        # its terminal check, not a search step, so the iteration budget does not
        # refuse it.
        if (
            not self._stop_earned()
            and len(self.history) + revalidation_events > self.outer_config.max_iterations
        ):
            return self._result(converged=False, reason="max_iterations")
        if len(face.component_names) > 1:
            return self._revalidate_joint_face(face)
        return self._revalidate_scalar_face(face)

    def _stop_hands_off(self) -> bool:
        """Whether the stop goes to the endgame, whose exact gradient has not judged it."""
        return bool(
            not self.endgame_stationary
            and self.estimated_names
            and (self.stationary_search or self._endgame_budget() > 0)
            and should_hand_off(
                "objective_plateau" if self.plateau_qualified else "lambda_change",
                max_accepted_step=0.0,
                iterations=len(self.history),
                config=self.outer_config,
            )
        )

    def _revalidate_joint_face(self, face: PenaltyFace) -> DistributionalEFSResult | None:
        """Keep a joint face whose direction is still its endpoint's, or release it."""
        authority_config = _face_authority_config(self.inner_config)
        joint_attempt = self._assess_joint_face(face, authority_config)
        if _joint_direction_is_endpoint(joint_attempt):
            return self._keep_joint_face(face, joint_attempt)
        return self._retract(*self._joint_retraction(face, joint_attempt, authority_config))

    def _assess_joint_face(
        self, face: PenaltyFace, authority_config: DenseSolverConfig
    ) -> _JointFaceDirectionAttempt:
        """The joint face's direction at the accepted fit, or why it cannot be had."""
        try:
            finite_faces = tuple(
                (
                    name,
                    _optional_penalty_face(
                        self.layout,
                        tuple(item for item in face.component_names if item != name),
                    ),
                )
                for name in face.component_names
            )
        except PenaltyFaceError:
            return _JointFaceDirectionAttempt(
                component_names=face.component_names,
                direction=None,
                assessment_fits=(),
                coefficient_tolerance=authority_config.tolerance,
                failure_reason="joint_analytic_unavailable",
            )
        return _assess_joint_face_directions(
            self.family,
            self.layout,
            self.y,
            self.likelihood_plan,
            lambdas=self.state.lambdas,
            component_names=face.component_names,
            joint_face=face,
            finite_faces=finite_faces,
            source_fit=self.state.fit,
            source_objective=self.state.objective,
            endpoint_fit=self.state.fit,
            solver_config=self.inner_config,
            efs_config=self.outer_config,
            chunk_size=self.chunk_size,
            phase_recorder=self.phase_recorder,
            _reuse_session=self.reuse_session,
            derivatives=self.endpoint_derivatives,
        )

    def _keep_joint_face(
        self, face: PenaltyFace, joint_attempt: _JointFaceDirectionAttempt
    ) -> DistributionalEFSResult | None:
        """Keep the joint face at its endpoint fit and record the revalidation."""
        source_fit_index = self.state.terminal_fit_index
        source_fit = self.state.fit
        objective_before = self.state.objective
        assessment_indices = self._append_fits(joint_attempt.assessment_fits)
        self.state = replace(self.state, terminal_fit_index=assessment_indices[0])
        endpoint_fit = joint_attempt.assessment_fits[0]
        direction = joint_attempt.direction
        assert direction is not None
        indexed_direction = _indexed_joint_direction(
            direction,
            assessment_indices,
            joint_attempt.coefficient_tolerance,
        )
        self.history.append(
            _joint_face_revalidation_iteration(
                iteration=len(self.history) + 1,
                source_fit_index=source_fit_index,
                accepted_fit_index=self.state.terminal_fit_index,
                coefficient_fit_indices=assessment_indices,
                component_names=face.component_names,
                lambdas=self.state.lambdas,
                objective_before=objective_before,
                endpoint_fit=endpoint_fit,
                direction=indexed_direction,
                coefficient_tolerance=joint_attempt.coefficient_tolerance,
                evidence=self.state.evidence,
                source_fit=source_fit,
            )
        )
        self.state = replace(
            self.state,
            fit=endpoint_fit,
            objective=indexed_direction.endpoint_objective,
        )
        terminal_endpoint_directions = dict(indexed_direction.component_directions)
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            self.state = replace(
                self.state,
                evidence=_fresh_raw_evidence(
                    self.layout,
                    self.state.lambdas,
                    self.state.fit,
                    self.outer_config,
                    face=self.state.face,
                ),
            )
        return self._stop_after_revalidation(terminal_endpoint_directions)

    def _joint_retraction(
        self,
        face: PenaltyFace,
        joint_attempt: _JointFaceDirectionAttempt,
        authority_config: DenseSolverConfig,
    ) -> tuple[_FaceRetraction, tuple[int, ...]]:
        """Refit off a joint face that did not hold: the retraction and its fits' indices."""
        rollback_initial = (
            self.state.fit.coefficients
            if not joint_attempt.assessment_fits
            else joint_attempt.assessment_fits[-1].coefficients
        )
        rollback_fit = _fit_endpoint_authority_stationary(
            self.family,
            self.layout,
            self.y,
            self.likelihood_plan,
            lambdas=self.state.lambdas,
            face=None,
            initial=rollback_initial,
            config=authority_config,
            chunk_size=self.chunk_size,
            phase_recorder=self.phase_recorder,
            _reuse_session=self.reuse_session,
        )
        rollback_objective = _laplace_objective(
            rollback_fit,
            layout=self.layout,
            lambdas=self.state.lambdas,
            face=None,
            objective_tolerance=self.outer_config.objective_tolerance,
        )
        rollback_fits = (*joint_attempt.assessment_fits, rollback_fit)
        assessment_indices = self._append_fits(rollback_fits)
        retraction_direction = joint_attempt.direction
        if retraction_direction is not None:
            retraction_direction = _indexed_joint_direction(
                retraction_direction,
                assessment_indices[:1],
                joint_attempt.coefficient_tolerance,
            )
        retraction = _FaceRetraction(
            component_names=face.component_names,
            fit=rollback_fit,
            objective=rollback_objective,
            direction=retraction_direction,
            failure_reason=(
                joint_attempt.failure_reason if joint_attempt.assessment_fits else None
            ),
            coefficient_tolerance=joint_attempt.coefficient_tolerance,
            assessment_fits=rollback_fits,
        )
        return retraction, assessment_indices

    def _revalidate_scalar_face(self, face: PenaltyFace) -> DistributionalEFSResult | None:
        """Recheck the face one component at a time; release it where one turns finite."""
        recheck = _recheck_exact_face(
            self.family,
            self.layout,
            self.y,
            self.likelihood_plan,
            lambdas=self.state.lambdas,
            face=face,
            current_fit=self.state.fit,
            current_objective=self.state.objective,
            solver_config=self.inner_config,
            efs_config=self.outer_config,
            chunk_size=self.chunk_size,
            phase_recorder=self.phase_recorder,
            _reuse_session=self.reuse_session,
            derivatives=self.endpoint_derivatives,
            other_caps_stationary=self.endgame_stationary,
        )
        resolved_directions: dict[str, EndpointDirectionEvidence] = {}
        for component_name, check in recheck.checks:
            resolved_directions[component_name] = self._keep_face_component(component_name, check)
        if recheck.retraction is not None:
            return self._retract(*self._scalar_retraction(recheck.retraction))
        return self._stop_after_revalidation(resolved_directions)

    def _keep_face_component(
        self, component_name: str, check: _FaceDirectionCheck
    ) -> EndpointDirectionEvidence:
        """Keep one component on the face at its endpoint fit and record the revalidation."""
        source_fit_index = self.state.terminal_fit_index
        source_fit = self.state.fit
        objective_before = self.state.objective
        assessment_indices = self._append_fits(check.assessment_fits)
        self.state = replace(self.state, terminal_fit_index=assessment_indices[-1])
        direction = _indexed_direction(
            check.direction,
            assessment_indices,
            check.coefficient_tolerance,
        )
        self.history.append(
            _face_revalidation_iteration(
                iteration=len(self.history) + 1,
                source_fit_index=source_fit_index,
                accepted_fit_index=self.state.terminal_fit_index,
                coefficient_fit_indices=assessment_indices,
                component_name=component_name,
                lambdas=self.state.lambdas,
                objective_before=objective_before,
                check=check,
                direction=direction,
                evidence=self.state.evidence,
                source_fit=source_fit,
            )
        )
        self.state = replace(self.state, fit=check.endpoint_fit, objective=check.endpoint_objective)
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            self.state = replace(
                self.state,
                evidence=_fresh_raw_evidence(
                    self.layout,
                    self.state.lambdas,
                    self.state.fit,
                    self.outer_config,
                    face=self.state.face,
                ),
            )
        return direction

    def _scalar_retraction(
        self, retraction: _FaceRetraction
    ) -> tuple[_FaceRetraction, tuple[int, ...]]:
        """The scalar recheck's retraction, with its fits appended and its direction indexed."""
        assessment_indices = self._append_fits(retraction.assessment_fits)
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
        return retraction, assessment_indices

    def _retract(
        self, retraction: _FaceRetraction, assessment_indices: tuple[int, ...]
    ) -> DistributionalEFSResult | None:
        """Leave a face that did not hold; the search goes on if its direction leaves the face."""
        self.history.append(
            _face_retraction_iteration(
                iteration=len(self.history) + 1,
                source_fit_index=self.state.terminal_fit_index,
                accepted_fit_index=assessment_indices[-1],
                coefficient_fit_indices=assessment_indices,
                lambdas=self.state.lambdas,
                objective_before=self.state.objective,
                retraction=retraction,
                evidence=self.state.evidence,
                source_fit=self.state.fit,
            )
        )
        self.state = replace(
            self.state,
            terminal_fit_index=assessment_indices[-1],
            face=None,
            fit=retraction.fit,
            objective=retraction.objective,
        )
        if _direction_leaves_face(retraction.direction):
            self._active_set_changed()
            return None
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            self.state = replace(
                self.state,
                evidence=_fresh_raw_evidence(
                    self.layout,
                    self.state.lambdas,
                    self.state.fit,
                    self.outer_config,
                    face=self.state.face,
                ),
            )
        return self._result(converged=False, reason="endpoint_revalidation_failed")

    def _stop_after_revalidation(
        self, directions: Mapping[str, EndpointDirectionEvidence]
    ) -> DistributionalEFSResult | None:
        """End on the revalidated face, unless its fresh evidence sends the search on."""
        if not self._stop_earned() and (
            self.state.evidence.working_infinity
            or self.state.evidence.maximum > self.outer_config.tolerance
        ):
            return None
        return self._converged(directions)

    def _converged(
        self, directions: Mapping[str, EndpointDirectionEvidence]
    ) -> DistributionalEFSResult:
        """The strict stop's converged result, with the directions that kept its face."""
        return self._result(
            converged=True,
            reason=self._strict_reason(),
            terminal_endpoint_directions=directions,
        )

    def _out_of_rounds(self) -> DistributionalEFSResult | None:
        """No round is left: the held caps go back to the evidence and the fit ends."""
        if self._release_held_caps() and self.state.fit is self.refused_fit:
            return None
        # The other components ran out of rounds; the result names the held caps.
        return self._result(converged=False, reason="max_iterations")

    def _fellner_schall_round(self) -> DistributionalEFSResult | None:
        """One round: propose, try the bound, the accelerated point and shorter steps."""
        iteration_number = len(self.history) + 1
        source_fit_index = self.state.terminal_fit_index
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            plain_update = self.state.evidence.update
            if plain_update is None:
                raise RuntimeError("estimated smoothing components require a raw EFS update")
            max_proposed_step = _maximum_step(plain_update, self.estimated_names)
        acceleration_outcome = "disabled"
        acceleration_refusal_reason = None
        accelerated_proposal = None
        if self.accelerator is not None:
            if not all(plain_update.proposal_kinds[name] == "gfs" for name in self.estimated_names):
                self.accelerator.reset()
                acceleration_outcome = "refused"
                acceleration_refusal_reason = "non_gfs_proposal"
            else:
                with measure_phase(self.phase_recorder, "efs_update_backtracking"):
                    provenance = _acceleration_provenance(self.estimated_names, self.state.fit)
                    self.accelerator.record_accepted(
                        log_lambdas=_ordered_log_lambdas(self.state.lambdas, self.estimated_names),
                        raw_residual=_ordered_steps(
                            plain_update.raw_log_steps,
                            self.estimated_names,
                        ),
                        provenance=provenance,
                    )
                    decision = self.accelerator.propose(
                        max_log_step=self.outer_config.max_log_step,
                        minimum_log_lambda=np.array(
                            [
                                math.log(self.outer_config.lower(name))
                                for name in self.estimated_names
                            ]
                        ),
                        maximum_log_lambda=np.array(
                            [
                                math.log(self.outer_config.upper(name))
                                for name in self.estimated_names
                            ]
                        ),
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

        self.attempted_fit_indices: list[int] = []
        self.attempted_tolerances: list[float] = []
        accepted_fit: DenseSolverResult | None = None
        accepted_fit_index: int | None = None
        accepted_lambdas: dict[str, float] | None = None
        accepted_log_steps: dict[str, float] | None = None
        accepted_objective: float | None = None
        self.any_converged_trial = False
        plateau_eligible = True
        accelerated_fit_index: int | None = None
        raw_backtracks = 0
        proposed_lambdas: Mapping[str, float] = _complete_mapping(
            self.state.lambdas,
            plain_update.lambdas,
            missing=self.state.lambdas,
        )
        proposed_log_steps: Mapping[str, float] = _complete_mapping(
            self.state.lambdas,
            plain_update.log_steps,
            missing=0.0,
        )
        quadratic_forms = _complete_mapping(
            self.state.lambdas,
            plain_update.quadratic_forms,
            missing=0.0,
        )
        trace_terms = _complete_mapping(
            self.state.lambdas,
            plain_update.trace_terms,
            missing=0.0,
        )

        endpoint_names = self._endpoint_names(
            self.state, plain_update.log_steps, self.estimated_names
        )
        accepted_evidence: _FreshRawEvidence | None = None
        if endpoint_names:
            with measure_phase(self.phase_recorder, "efs_update_backtracking"):
                trial_lambdas, trial_log_steps = _endpoint_proposal(
                    self.state.lambdas,
                    plain_update.log_steps,
                    self.estimated_names,
                    endpoint_names,
                    self.outer_config,
                )
            trial_fit, fit_index, trial_objective = self._trial(trial_lambdas)
            if trial_objective is not None and _within_ceiling(
                self.state.fit,
                self.state.objective,
                trial_fit,
                trial_objective,
                self.outer_config,
            ):
                accepted_evidence = self._evidence_at(
                    trial_lambdas,
                    trial_fit,
                    self.penalty_memo,
                    self.cap_held | frozenset(endpoint_names),
                )
                if not frozenset(endpoint_names) <= frozenset(accepted_evidence.cap_held):
                    # A bound overshoots: the evidence there asks for a finite value.
                    accepted_evidence = None
            if accepted_evidence is not None:
                self.endpoint_caps.update(endpoint_names)
                accepted_fit = trial_fit
                accepted_fit_index = fit_index
                accepted_lambdas = trial_lambdas
                accepted_log_steps = trial_log_steps
                accepted_objective = trial_objective
                proposed_lambdas = trial_lambdas
                proposed_log_steps = trial_log_steps
                max_proposed_step = max(abs(trial_log_steps[name]) for name in self.estimated_names)

        accelerated_values = None
        if accelerated_proposal is not None:
            with measure_phase(self.phase_recorder, "efs_update_backtracking"):
                accelerated_values = _accelerated_proposal(
                    self.state.lambdas,
                    self.estimated_names,
                    accelerated_proposal.log_lambdas,
                    accelerated_proposal.log_step,
                    self.outer_config,
                )
            if accelerated_values is None:
                acceleration_outcome = "refused"
                acceleration_refusal_reason = "box_blocked"

        if accelerated_values is not None:
            acceleration_outcome = "rejected"
            trial_lambdas, trial_log_steps = accelerated_values
            trial_fit, fit_index, trial_objective = self._trial(trial_lambdas)
            accelerated_fit_index = fit_index
            if (
                trial_objective is None
                or _acceleration_provenance(self.estimated_names, trial_fit) != provenance
            ):
                plateau_eligible = False
            elif _within_ceiling(
                self.state.fit,
                self.state.objective,
                trial_fit,
                trial_objective,
                self.outer_config,
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
                    (abs(trial_log_steps[name]) for name in self.estimated_names),
                    default=0.0,
                )
            if accepted_fit is None:
                assert self.accelerator is not None
                self.accelerator.reject()

        if accepted_fit is None:
            for raw_backtrack in range(self.outer_config.max_backtracks + 1):
                step_scale = self.outer_config.backtrack_factor**raw_backtrack
                with measure_phase(self.phase_recorder, "efs_update_backtracking"):
                    trial_lambdas, trial_log_steps = _scaled_proposal(
                        self.state.lambdas,
                        plain_update.log_steps,
                        self.estimated_names,
                        step_scale,
                        self.outer_config,
                    )
                trial_fit, fit_index, trial_objective = self._trial(trial_lambdas)
                raw_backtracks = raw_backtrack
                if trial_objective is None:
                    plateau_eligible = False
                    continue
                if _within_ceiling(
                    self.state.fit,
                    self.state.objective,
                    trial_fit,
                    trial_objective,
                    self.outer_config,
                ):
                    accepted_fit = trial_fit
                    accepted_fit_index = fit_index
                    accepted_lambdas = trial_lambdas
                    accepted_log_steps = trial_log_steps
                    accepted_objective = trial_objective
                    break

        if accepted_fit is None:
            zero_steps = {name: 0.0 for name in self.state.lambdas}
            self.history.append(
                DistributionalEFSIteration(
                    iteration=iteration_number,
                    source_fit_index=source_fit_index,
                    lambdas_before=self.state.lambdas,
                    proposed_lambdas=proposed_lambdas,
                    lambdas_after=self.state.lambdas,
                    proposed_log_steps=proposed_log_steps,
                    accepted_log_steps=zero_steps,
                    quadratic_forms=quadratic_forms,
                    trace_terms=trace_terms,
                    objective_before=self.state.objective,
                    objective_after=self.state.objective,
                    objective_relative_change=0.0,
                    max_proposed_log_step=max_proposed_step,
                    max_accepted_log_step=0.0,
                    accepted=False,
                    acceleration_outcome=acceleration_outcome,
                    acceleration_refusal_reason=acceleration_refusal_reason,
                    accelerated_fit_index=accelerated_fit_index,
                    backtracks=len(self.attempted_fit_indices) - 1,
                    raw_backtracks=raw_backtracks,
                    coefficient_fit_indices=tuple(self.attempted_fit_indices),
                    accepted_fit_index=None,
                    coefficient_tolerances=tuple(self.attempted_tolerances),
                    update_curvature=self.state.fit.terminal_curvature,
                    accepted_curvature=None,
                    endpoint_components=endpoint_names,
                )
            )
            reason: EFSConvergenceReason = (
                "objective_rejected" if self.any_converged_trial else "coefficient_not_converged"
            )
            # A step rejected at every length is the default search's stop, but beside
            # a cap an endpoint trial holds, where the finish decides.
            self.unearned = (
                reason == "objective_rejected"
                and self._plateau_finishes()
                and not self.endpoint_caps.isdisjoint(self.cap_held)
            )
            if self._endgame_budget() > 0 and (
                self.unearned
                or should_hand_off(
                    reason,
                    max_accepted_step=0.0,
                    iterations=len(self.history),
                    config=self.outer_config,
                )
            ):
                return self._run_endgame()
            if self._release_held_caps():
                return None
            return self._result(converged=False, reason=reason)

        assert accepted_fit_index is not None
        assert accepted_lambdas is not None
        assert accepted_log_steps is not None
        assert accepted_objective is not None
        objective_change = abs(accepted_objective - self.state.objective) / (
            1.0 + abs(self.state.objective)
        )
        max_accepted_step = max(
            (abs(accepted_log_steps[name]) for name in self.estimated_names),
            default=0.0,
        )
        # A component taken to its bound has no next step to judge the search by: the
        # hand-off reads the ordinary steps of the others.
        ordinary_step = (
            max_accepted_step
            if accepted_evidence is None
            else max(
                (
                    abs(accepted_log_steps[name])
                    for name in self.estimated_names
                    if name not in endpoint_names
                ),
                default=0.0,
            )
        )
        practical_parameter_change = (
            _maximum_relative_natural_parameter_change(
                self.state.fit.theta,
                accepted_fit.theta,
            )
            if self.outer_config.practical_convergence
            else math.inf
        )
        self.history.append(
            DistributionalEFSIteration(
                iteration=iteration_number,
                source_fit_index=source_fit_index,
                lambdas_before=self.state.lambdas,
                proposed_lambdas=proposed_lambdas,
                lambdas_after=accepted_lambdas,
                proposed_log_steps=proposed_log_steps,
                accepted_log_steps=accepted_log_steps,
                quadratic_forms=quadratic_forms,
                trace_terms=trace_terms,
                objective_before=self.state.objective,
                objective_after=accepted_objective,
                objective_relative_change=objective_change,
                max_proposed_log_step=max_proposed_step,
                max_accepted_log_step=max_accepted_step,
                accepted=True,
                acceleration_outcome=acceleration_outcome,
                acceleration_refusal_reason=acceleration_refusal_reason,
                accelerated_fit_index=accelerated_fit_index,
                backtracks=len(self.attempted_fit_indices) - 1,
                raw_backtracks=raw_backtracks,
                coefficient_fit_indices=tuple(self.attempted_fit_indices),
                accepted_fit_index=accepted_fit_index,
                coefficient_tolerances=tuple(self.attempted_tolerances),
                update_curvature=self.state.fit.terminal_curvature,
                accepted_curvature=accepted_fit.terminal_curvature,
                step_source="efs" if accepted_evidence is None else "endpoint",
                endpoint_components=endpoint_names,
            )
        )
        self.previous_fall, self.fall = self.fall, self.state.objective - accepted_objective
        self.state = replace(
            self.state,
            lambdas=accepted_lambdas,
            fit=accepted_fit,
            objective=accepted_objective,
            terminal_fit_index=accepted_fit_index,
        )
        self.state = replace(
            self.state,
            evidence=self._evidence(self.penalty_memo)
            if accepted_evidence is None
            else self._adopt(accepted_evidence),
        )
        if self.state.evidence.estimated_names != self.estimated_names:
            raise RuntimeError("estimated smoothing names changed inside one EFS fit")
        # The step test alone is first-order on a harmonic tail, so require both
        # a flat objective and a contracting step before nominating a plateau.
        # The practical policy below accepts its own sustained plateau and keeps
        # the larger strict residual in the result.
        # An unusable trial can force a tiny accepted step while leaving the
        # objective at the longer step unknown. It breaks both windows.
        comparable_flat_step = (
            plateau_eligible and objective_change <= self.outer_config.plateau_tolerance
        )
        self.plateau_run = self.plateau_run + 1 if comparable_flat_step else 0
        self.plateau_qualified = (
            self.plateau_run >= self.outer_config.plateau_iterations
            and max_accepted_step <= self.previous_accepted_step
        )
        if self.outer_config.practical_convergence:
            assert self.state.evidence.update is not None
            lower_pressure = _lower_bound_pressure(
                self.state.evidence,
                self.state.lambdas,
                self.state.evidence.update.raw_log_steps,
                self.outer_config,
            )
            self.practical_run = (
                self.practical_run + 1
                if comparable_flat_step
                and practical_parameter_change <= self.outer_config.practical_parameter_tolerance
                and max_accepted_step <= self.previous_accepted_step
                and not lower_pressure
                else 0
            )
        self.previous_accepted_step = max_accepted_step if plateau_eligible else math.inf

        # A finite profile can also settle while a penalty continues moving
        # outward. Require substantial accepted probes before this policy may
        # preempt the next cap or endpoint decision; retain the raw pressure.
        outward_plateau = (
            self.outer_config.practical_convergence
            and self.plateau_run >= self.outer_config.plateau_iterations
            and self.state.face is None
            and self.state.evidence.maximum > self.outer_config.tolerance
            and self.state.evidence.update is not None
            and _practical_outward_window(
                history=self.history,
                coefficient_fits=self.coefficient_fits,
                terminal_raw_log_steps={
                    name: (
                        self.state.evidence.update.raw_log_steps[name]
                        if name in self.state.evidence.estimated_names
                        else 0.0
                    )
                    for name in self.state.lambdas
                },
                config=self.outer_config,
            )
        )
        # A practical plateau is an interior stop of the components off the active
        # face: while one sits at the cap with outward pressure or is a working
        # infinity, an outward plateau stops here and otherwise the top of the loop
        # decides (promote an exact face, or refuse with a named reason). A promotion
        # restarts the plateau's count. Off the cap, either plateau hands to a finish
        # where one is available, so which of the two a fit meets first does not
        # decide its outcome.
        interior = (
            not self.state.evidence.unresolved_upper_bound
            and not self.state.evidence.working_infinity
        )
        finishes = self._plateau_finishes() and self._endgame_budget() > 0
        if self.stationary_search and (
            outward_plateau or self.practical_run >= self.outer_config.plateau_iterations
        ):
            return self._run_endgame()
        if outward_plateau and not (finishes and interior):
            return self._plateau_result()
        if (
            self.outer_config.practical_convergence
            # Plain Fellner--Schall has no other exit from a face. A spent finish leaves
            # a face to the strict stop, and a Newton endgame takes it at its hand-off.
            and (self.state.face is None or finishes or self.outer_config.outer == "efs")
            and (outward_plateau or self.practical_run >= self.outer_config.plateau_iterations)
            and self.state.evidence.maximum > self.outer_config.tolerance
            and interior
        ):
            if self._endgame_budget() > 0 and (
                finishes
                or should_hand_off(
                    "practical_plateau",
                    max_accepted_step=ordinary_step,
                    iterations=len(self.history),
                    config=self.outer_config,
                )
            ):
                return self._run_endgame()
            return self._plateau_result()
        if self._endgame_budget() > 0 and should_hand_off(
            None,
            max_accepted_step=ordinary_step,
            iterations=len(self.history),
            config=self.outer_config,
        ):
            return self._run_endgame()
        return None

    def _evidence_at(
        self,
        lambdas: Mapping[str, float],
        fit: DenseSolverResult,
        memo: _PenaltyEvaluationMemo | None = None,
        held: frozenset[str] | None = None,
    ) -> _FreshRawEvidence:
        """Fresh evidence at one state on the current face, less the caps ``held`` there."""
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            return _fresh_raw_evidence(
                self.layout,
                lambdas,
                fit,
                self.outer_config,
                face=self.state.face,
                penalty_memo=memo,
                held=self.cap_held if held is None else held,
            )

    def _adopt(self, evidence: _FreshRawEvidence) -> _FreshRawEvidence:
        """Take ``evidence`` as the accepted state's: the caps it holds are the ones still held."""
        self.cap_held = frozenset(evidence.cap_held)
        return evidence

    def _evidence(self, memo: _PenaltyEvaluationMemo | None = None) -> _FreshRawEvidence:
        return self._adopt(self._evidence_at(self.state.lambdas, self.state.fit, memo))

    def _trial(
        self,
        trial_lambdas: Mapping[str, float],
    ) -> tuple[DenseSolverResult, int, float | None]:
        """Fit one trial state from the accepted fit and record it.

        The objective is ``None`` when the fit is not stationary or its objective
        cannot be certified.
        """
        if self.state.face is None:
            tolerance = self.inner_config.tolerance
            trial_fit = _fit_fixed_state(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                lambdas=trial_lambdas,
                face=None,
                initial=self.state.fit.coefficients,
                config=self.inner_config,
                chunk_size=self.chunk_size,
                phase_recorder=self.phase_recorder,
                _reuse_session=self.reuse_session,
                _reuse_source=self.state.fit,
            )
            stationary = trial_fit.converged
        else:
            authority_config = _face_authority_config(self.inner_config)
            tolerance = authority_config.tolerance
            trial_fit = _fit_endpoint_authority_stationary(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                lambdas=trial_lambdas,
                face=self.state.face,
                initial=self.state.fit.coefficients,
                config=authority_config,
                chunk_size=self.chunk_size,
                phase_recorder=self.phase_recorder,
                _reuse_session=self.reuse_session,
            )
            stationary = trial_fit.converged and _assessment_is_numerically_stationary(
                trial_fit, tolerance
            )
        self.coefficient_fits.append(trial_fit)
        fit_index = len(self.coefficient_fits) - 1
        self.attempted_fit_indices.append(fit_index)
        self.attempted_tolerances.append(tolerance)
        self.any_converged_trial = self.any_converged_trial or stationary
        if not stationary:
            return trial_fit, fit_index, None
        with measure_phase(self.phase_recorder, "efs_update_backtracking"):
            trial_objective = _trial_laplace_objective(
                trial_fit,
                layout=self.layout,
                lambdas=trial_lambdas,
                face=self.state.face,
                objective_tolerance=self.outer_config.objective_tolerance,
                penalty_memo=self.penalty_memo,
            )
        return trial_fit, fit_index, trial_objective

    def _release_held_caps(self) -> bool:
        """Return the held caps to the evidence; true when one is still pressed outward."""
        if not self.cap_held:
            return False
        self.cap_held = frozenset()
        self.state = replace(self.state, evidence=self._evidence())
        self.caps_final = bool(self.state.evidence.unresolved_upper_bound)
        return self.caps_final

    def _foreign_cap(self) -> bool:
        """Whether a component the search does not hold stands at its cap off the face.

        A face is promoted with every cap in it, so beside such a component the held
        ones cannot be.
        """
        return any(
            self.state.lambdas[name] == self.outer_config.upper(name)
            for name in self.estimated_names
            if name not in self.cap_held
        )

    def _endpoint_names(
        self, state: EndgameState, log_steps: Mapping[str, float], names: tuple[str, ...]
    ) -> tuple[str, ...]:
        """The components to try at their bounds beside the step ``log_steps`` from ``state``.

        Each is tried once in a search. The accelerator extrapolates the same
        fixed-point map, so an accelerated search makes no endpoint trial. A
        practical plateau reads the objective, so ``"efs"`` and ``"efs+finish"``
        with one make no trial for a fall within the score bar, unless the step
        itself puts a tail on its bound (``resolved_flat_tails``). A strict stop,
        and the hand-off of the other searches, read the step, which a flat tail
        never shrinks: there every tail is tried.
        """
        if self.accelerator is not None:
            return ()
        flat_tail = state.evidence.flat_tail
        plateau = self.outer_config.practical_convergence and self.outer_config.outer in (
            "efs",
            "efs+finish",
        )
        resolution = self.outer_config.score_bar(state.objective) if plateau else -math.inf
        # The steps ``_scaled_proposal`` assigns the bound to.
        arriving = [
            name
            for name in flat_tail
            if log_steps[name]
            >= math.log(self.outer_config.upper(name)) - math.log(state.lambdas[name])
        ]
        tails = resolved_flat_tails(flat_tail, self.endpoint_tried, resolution, arriving)
        chosen = endpoint_trial_names(tails, log_steps, names, self.endpoint_tried)
        self.endpoint_tried.update(chosen)
        return chosen

    def _handed_caps(self) -> tuple[str, ...]:
        """The caps a stationary endgame hands to the face assessor.

        They are the pressed caps an endpoint trial reached, where a round is left
        to promote them in and to take the gradient on their face, and no other
        component stands at its cap, beside which none could be promoted.
        """
        pressed = self.state.evidence.unresolved_upper_bound
        capped = [
            name
            for name in self.estimated_names
            if self.state.lambdas[name] == self.outer_config.upper(name)
        ]
        if (
            not pressed
            or any(name not in self.endpoint_caps or name not in pressed for name in capped)
            or min(self.newton_budget, self.outer_config.max_iterations - len(self.history) - 1)
            <= 0
        ):
            return ()
        return pressed

    def _plateau_result(self) -> DistributionalEFSResult | None:
        """End at a practical plateau, or go on (``None``).

        A plateau reads one round's change, which a slow steady drift keeps small, so
        a search with an active bound, an exact face or a cap a trial holds for one,
        goes on while its objective is on course to fall further than the plateau's
        tolerance (``remaining_fall``). Either bound asks the same question, so it
        does not matter to the stop when a held cap is promoted. The held caps are
        judged before the plateau stands, but for caps no assessment has refused
        beside another component's cap, where none could be promoted: the result
        names those. A face is revalidated by the strict stop the earned plateau
        goes to.
        """
        ceiling = self.outer_config.plateau_tolerance * (1.0 + abs(self.state.objective))
        resolution = self.outer_config.objective_tolerance * (1.0 + abs(self.state.objective))
        bound = self.state.face is not None or not self.endpoint_caps.isdisjoint(self.cap_held)
        if bound and remaining_fall(self.fall, self.previous_fall, resolution) > ceiling:
            return None
        if (self.refused_fit is not None or not self._foreign_cap()) and self._release_held_caps():
            self.plateau_earned = True
            return None
        if self.state.face is None:
            return self._result(converged=True, reason="practical_plateau")
        self.plateau_earned = not self.cap_held
        return None

    def _active_set_changed(self, *, held_promoted: bool = False) -> None:
        """A face was promoted or released: the plateau and the endgame's verdict were the old set's.

        ``held_promoted`` is the promotion of caps held while the other components
        reached a practical plateau, which that plateau survives.
        """
        self.plateau_earned = self.plateau_earned and held_promoted
        self.fall = self.previous_fall = math.inf
        self.pending_cap = None
        self.endgame_outcome = None  # its derivatives named the old set's components
        self.endgame_stationary = False
        self.state = replace(self.state, evidence=self._evidence())
        self.estimated_names = self.state.evidence.estimated_names
        self.plateau_run = 0
        self.practical_run = 0
        self.plateau_qualified = False
        self.previous_accepted_step = math.inf
        if self.accelerator is not None:
            self.accelerator.reset()

    def _plateau_finishes(self) -> bool:
        """Whether a practical plateau hands to the endgame rather than stopping.

        Fellner--Schall has no exact-gradient test, and its plateau can leave V several
        stationarity bars short, so ``"efs+finish"`` ends with Wood, Pya and Saefken's
        (2016) Newton iteration on the components off the active face. A finish that
        cannot reach stationarity leaves the fit at its plateau.
        """
        return self.outer_config.outer == "efs+finish"

    def _endgame_budget(self) -> int:
        """The endgame iterations left: ``max_iterations`` bounds warm-up and endgame together."""
        return min(self.newton_budget, self.outer_config.max_iterations - len(self.history))

    def _endgame_at(self, entry: EndgameState, budget: int) -> NewtonEndgameOutcome:
        return run_newton_endgame(
            self.family,
            self.layout,
            self.y,
            self.likelihood_plan,
            state=entry,
            solver_config=self.inner_config,
            efs_config=self.endgame_config,
            chunk_size=self.chunk_size,
            phase_recorder=self.phase_recorder,
            reuse_session=self.reuse_session,
            history=self.history,
            coefficient_fits=self.coefficient_fits,
            upper_bounds=self.beyond_cap,
            budget=budget,
            cap_tolerance=self.outer_config.tolerance,
            endpoint_trial=self._endpoint_names,
        )

    def _result(
        self,
        *,
        converged: bool,
        reason: EFSConvergenceReason,
        terminal_endpoint_directions: Mapping[str, EndpointDirectionEvidence] | None = None,
    ) -> DistributionalEFSResult:
        self._release_held_caps()  # a result names every cap still pressed
        evidence = self.state.evidence
        endgame = self.endgame_outcome
        if converged and self.endgame_stationary:
            # The exact gradient voids a Fellner--Schall cap nomination, including
            # one on evidence refreshed after an exact-face revalidation. That
            # revalidation judged the endgame's own fit, so the derivatives the
            # endgame published are already those of the published fit.
            evidence = replace(evidence, unresolved_upper_bound=())
        return _efs_result(
            layout=self.layout,
            config=self.outer_config,
            initial_lambdas=self.initial_lambdas,
            lambdas=self.state.lambdas,
            initial_objective=self.initial_objective,
            objective=self.state.objective,
            converged=converged,
            reason=reason,
            terminal_evidence=evidence,
            history=self.history,
            coefficient_fits=self.coefficient_fits,
            terminal_fit_index=self.state.terminal_fit_index,
            terminal_endpoint_directions=terminal_endpoint_directions,
            endgame=endgame,
            beyond_cap_components=tuple(
                name for name in self.state.lambdas if name in self.beyond_cap
            ),
        )

    def _strict_reason(self) -> EFSConvergenceReason:
        if self.plateau_earned:
            return "practical_plateau"
        if self.endgame_stationary or (
            self.stationary_search and not self.state.evidence.estimated_names
        ):
            return "stationary"
        return "objective_plateau" if self.plateau_qualified else "lambda_change"

    def _resume_fellner_schall(self) -> None:
        """Keep the accepted fit; drop what cannot describe the next EFS step.

        Disabling handoff bounds the recovery by the ordinary outer budget.
        """
        self.endgame_outcome = None
        self.endgame_stationary = False
        self.pending_cap = None
        self.newton_budget = 0
        self.plateau_run = 0
        self.practical_run = 0
        self.plateau_qualified = False
        self.previous_accepted_step = math.inf
        if self.accelerator is not None:
            self.accelerator.reset()

    def _run_endgame(self) -> DistributionalEFSResult | None:
        """Run the endgame from the accepted state; a result ends the fit, ``None`` continues.

        A finish that stalls leaves a plateau standing. One a rejected round came to
        (``unearned``), at once or after the caps it held were judged, has none and
        leaves that round's verdict.
        """
        budget = self._endgame_budget()
        if budget <= 0:
            return self._result(converged=False, reason="max_iterations")
        # A plateau's finish is a few steps, so the held caps are judged first and its
        # exact gradient is taken on the face they belong to. A search handed to the
        # endgame runs it off the face, as it does beside another component's cap, and
        # the gradient decides the caps.
        pause = self._plateau_finishes() and not self._foreign_cap()
        self.endgame_due = self._release_held_caps() and pause
        if self.endgame_due:
            return None
        entry = self.state
        entry_fits = list(self.coefficient_fits) if self._plateau_finishes() else []
        entry_iterations = len(self.history)
        outcome = self._endgame_at(entry, budget)
        stalled = self._plateau_finishes() and outcome.kind not in ("stationary", "cap_pressure")
        if stalled:
            # The plateau held where the finish started, not where it stalled, so a
            # finish that cannot reach stationarity leaves the search where it was.
            self.coefficient_fits[:] = entry_fits
            del self.history[entry_iterations:]
            outcome = replace(outcome, state=entry, derivatives=None)
        self.newton_budget -= outcome.newton_iterations
        self.endpoint_caps.update(outcome.endpoint_components)
        self.endgame_outcome = outcome
        state = outcome.state
        self.state = replace(
            self.state,
            lambdas=dict(state.lambdas),
            fit=state.fit,
            objective=state.objective,
            terminal_fit_index=state.terminal_fit_index,
            evidence=state.evidence,
        )
        if stalled:
            self.endgame_outcome = None
            if self.unearned:
                return self._result(converged=False, reason="objective_rejected")
            if self.state.face is None and not self.state.evidence.unresolved_upper_bound:
                return self._result(converged=True, reason="practical_plateau")
            # Only the strict stop revalidates a face, and a stall decides nothing about
            # a nominated cap, so both carry on with the finish spent.
            self._resume_fellner_schall()
            return None
        if outcome.kind == "derivative_unavailable":
            if self.beyond_cap or self.stationary_search:
                # EFS proposals use the original finite box and cannot safely
                # continue from a component released above that box.
                self.endgame_outcome = None
                return self._result(converged=False, reason="derivative_unavailable")
            self._resume_fellner_schall()
            return None
        if outcome.kind == "stationary":
            # The exact gradient is the authority now: no component is at the cap
            # with an outward gradient, so a Fellner--Schall nomination is void,
            # but for the caps handed to the assessor.
            self.endgame_stationary = True
            self.pending_cap = None
            handed = self._handed_caps()
            self.endgame_due = self.handed_over = bool(handed)
            self.state = replace(
                self.state, evidence=replace(self.state.evidence, unresolved_upper_bound=handed)
            )
            return None
        if outcome.kind == "cap_pressure":
            self.pending_cap = outcome
            self.endgame_due = True  # the endgame goes on from a face promoted for this pressure
            nominated = set(self.state.evidence.unresolved_upper_bound) | set(outcome.cap_pressure)
            self.state = replace(
                self.state,
                evidence=replace(
                    self.state.evidence,
                    unresolved_upper_bound=tuple(
                        name for name in self.state.lambdas if name in nominated
                    ),
                ),
            )
            return None
        return self._result(converged=False, reason=outcome.kind)

    def _bracket_after_refusal(
        self,
        attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None,
    ) -> BracketRelease | BracketAttempt | None:
        """Search beyond the cap for the first refused component with a finite verdict."""
        if attempt is None or self.pending_cap is None or self.pending_cap.derivatives is None:
            return None
        directions: dict[str, EndpointDirectionEvidence] = {}
        if isinstance(attempt, _JointFaceDirectionAttempt):
            if attempt.direction is not None:
                directions = dict(attempt.direction.component_directions)
        elif attempt.check is not None:
            directions = {self.state.evidence.unresolved_upper_bound[0]: attempt.check.direction}
        for name, direction in directions.items():
            if direction.decision != "finite":
                continue
            found = bracket_refused_component(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                name=name,
                endpoint_derivative=float(direction.analytic_derivative),
                pending=self.pending_cap,
                state=self.state,
                solver_config=self.inner_config,
                efs_config=self.outer_config,
                chunk_size=self.chunk_size,
                phase_recorder=self.phase_recorder,
                reuse_session=self.reuse_session,
            )
            if found is not None:
                return found
        return None

    def _accept_bracket_release(
        self,
        release: BracketRelease,
        attempt: _FaceDirectionAttempt | _JointFaceDirectionAttempt | None,
    ) -> None:
        """Record the released component as one accepted bracket iteration."""
        assert self.pending_cap is not None and self.pending_cap.derivatives is not None
        assessment_fits = () if attempt is None else tuple(attempt.assessment_fits)
        indices = self._append_fits((*assessment_fits, *release.fits))
        tolerances = (
            *(() if attempt is None else (attempt.coefficient_tolerance,) * len(assessment_fits)),
            *((release.tolerance,) * len(release.fits)),
        )
        evidence = _fresh_raw_evidence(
            self.layout, release.lambdas, release.fit, self.outer_config, face=self.state.face
        )
        self.history.append(
            newton_iteration_record(
                iteration=len(self.history) + 1,
                state=self.state,
                proposed_lambdas=release.lambdas,
                lambdas_after=release.lambdas,
                objective_after=release.objective,
                evidence=evidence,
                fit_indices=indices,
                tolerances=tolerances,
                accepted_fit=release.fit,
                step_source="bracket",
                derivatives=self.pending_cap.derivatives,
                projected_gradient_norm=self.pending_cap.projected_gradient_norm or 0.0,
                hessian_certificate=None,
                ridge=None,
                estimated_names=self.estimated_names,
            )
        )
        self.state = replace(
            self.state,
            lambdas=dict(release.lambdas),
            fit=release.fit,
            objective=release.objective,
            terminal_fit_index=indices[-1],
            evidence=evidence,
        )
        self.beyond_cap[release.name] = self.outer_config.conditioning_limit(release.name)

    def _record_bracket_attempt(self, attempt: BracketAttempt) -> None:
        """Record a search that found no root as one rejected bracket iteration."""
        assert self.pending_cap is not None and self.pending_cap.derivatives is not None
        indices = self._append_fits(attempt.fits)
        self.history.append(
            newton_iteration_record(
                iteration=len(self.history) + 1,
                state=self.state,
                proposed_lambdas=self.state.lambdas,
                lambdas_after=self.state.lambdas,
                objective_after=self.state.objective,
                evidence=self.state.evidence,
                fit_indices=indices,
                tolerances=(attempt.tolerance,) * len(indices),
                accepted_fit=None,
                step_source="bracket",
                derivatives=self.pending_cap.derivatives,
                projected_gradient_norm=self.pending_cap.projected_gradient_norm or 0.0,
                hessian_certificate=None,
                ridge=None,
                estimated_names=self.estimated_names,
            )
        )
