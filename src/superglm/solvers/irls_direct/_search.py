"""The direct IRLS fit's shared line search (``_IRLSDirectTrialSearch``)."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from superglm.distributions import Binomial
from superglm.links import IdentityLink
from superglm.solvers.constrained_qp import _homogeneous_feasible_start
from superglm.solvers.irls_direct._qp import _QP_FEASIBILITY_TOL
from superglm.solvers.irls_state import (
    StateInvalid,
    _IRLSState,
    _IRLSStepDecision,
    _poisson_sqrt_halving_budget,
    _select_irls_trial,
    _stable_penalized_deviance_delta,
    _state_is_finite,
    interior_start_intercept,
    link_domain_violation,
    mean_space_violation,
)
from superglm.solvers.working_rows import coefficient_initial_intercept

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.distributions import Distribution
    from superglm.links import Link
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


def trial_refusal(
    family: Distribution,
    link: Link,
    committed: _IRLSState,
    weights: NDArray,
    other: StateInvalid | None,
) -> StateInvalid | None:
    """The line search's refusal of a trial that leaves what the committed state is inside, else ``other``.

    The family's mean space (``mean_space_violation``) and the link's own
    domain (``link_domain_violation``) are each convex, so halving from a state
    inside one finds a trial inside it (Nocedal & Wright 2006, section 3.1,
    backtracking with a feasibility test).  A committed state outside one has
    no such guarantee, and that domain is not tested.  ``other`` is the
    route's own refusal, the feasible set or finite observed rows.
    """
    refusal = other
    for outside in (mean_space_violation(family, link), link_domain_violation(link, family)):
        if outside is not None and not outside(committed.eta_unclipped, weights):
            refusal = _refuses(outside, weights, refusal)
    if isinstance(family, Binomial) and type(link) is IdentityLink:
        # A model may have no strictly interior point: e.g. an offset spans
        # exactly one on rows with identical predictors.  Its boundary start
        # still constrains every subsequent trial to valid probabilities.
        def outside_closed(eta: NDArray, prior: NDArray) -> bool:
            carried = eta[prior > 0.0]
            return not bool(np.all((carried >= 0.0) & (carried <= 1.0)))

        refusal = _refuses(outside_closed, weights, refusal)
    return refusal


def extended_halving_budget(
    fit: _IRLSDirectFit, committed: _IRLSState, proposal: _IRLSState, lowered: Callable | None
) -> Callable[[], int]:
    """The line search's extended halving budget: the Poisson/sqrt rule's, or a lowered start's if larger."""
    max_halving = fit.max_halving
    return lambda: max(
        _poisson_sqrt_halving_budget(
            committed=committed,
            proposal=proposal,
            y=fit.y,
            weights=fit.weights,
            family=fit.family,
            link=fit.link,
            default=max_halving,
        ),
        max_halving if lowered is None else lowered(),
    )


def retain_trial(
    fit: _IRLSDirectFit, retained: _IRLSState, decision: _IRLSStepDecision, t0: float
) -> None:
    """Write the retained state's predictor, mean, coefficients and deviance to the fit.

    The trial evaluations, timed from ``t0``, count as deviance time.
    """
    evaluation_elapsed = time.perf_counter() - t0
    fit.recorder.t_deviance += evaluation_elapsed
    fit.recorder.t_deviance_eval += evaluation_elapsed
    fit.eta_unclipped = retained.eta_unclipped
    fit.eta = retained.eta
    fit.mu = retained.mu
    fit.beta = retained.beta.copy()
    fit.intercept, fit.dev = retained.intercept, retained.deviance
    fit.n_halvings, fit.decision, fit.retained = decision.step_halvings, decision, retained


def _refuses(
    outside: Callable[[NDArray, NDArray], bool], weights: NDArray, other: StateInvalid | None
) -> StateInvalid:
    def invalid(candidate: _IRLSState) -> bool:
        return outside(candidate.eta_unclipped, weights) or (other is not None and other(candidate))

    return invalid


class _IRLSDirectTrialSearch:
    """The line search of every step without its own: QR, dense, the QP and structured.

    Each iteration evaluates the step's proposal and halves along the segment
    from the committed state (``irls_state._select_irls_trial``) until the
    penalised objective decreases.  A trial is refused, without its deviance
    counting, when it leaves what the committed state keeps:

    - the family's mean space and the link's own domain, while the committed
      state is inside them (#425's guard, #498): this unit's predicates;
    - a constrained fit's feasible set ``A beta >= b``, the feasibility
      contract's predicate (``_IRLSDirectQPStep.feasible``);
    - finite observed rows, on a fit taking Newton steps
      (``_IRLSDirectCurvaturePolicy.observed_rows_invalid``).

    The binomial/log overlay adds its first halving, a lowered start's
    budget and, under its Newton steps, its merit
    (``_IRLSDirectMeanSpaceOverlay.line_search``).  The feasibility contract
    also accepts a finite feasible QP proposal that an infeasible committed
    state would otherwise reject (raised into the link's domain when it left
    it), restarts from the phase-one point when the QP gave no such proposal,
    certifies the retained state's KKT conditions only on a full step
    (``_IRLSDirectQPStep.retained_converged``) and, on a rejected step,
    restores the active set the iteration started from.

    ``search(fit, it)`` writes what the SCOP step's own search writes: the
    retained state and its ``eta``, ``mu`` and deviance, ``beta`` and
    ``intercept``, the decision, ``n_halvings``, ``step_rejected`` and
    ``proposal_state``.  The unit keeps nothing: no reference to the fit.
    """

    __slots__ = ()

    def search(self, fit: _IRLSDirectFit, it: int) -> None:
        """Iteration ``it``'s line search from the committed state to the step's proposal."""
        beta, intercept, committed = fit.beta, fit.intercept, fit.committed
        qp, newton_score, mean_space = fit.qp, fit.newton_score, fit.mean_space
        y, weights = fit.y, fit.weights
        family, link = fit.family, fit.link
        max_halving, penalty_matvec = fit.max_halving, fit.penalty_matvec
        evaluate_state, recorder = fit.evaluate_state, fit.recorder
        _t0 = time.perf_counter()
        proposal = evaluate_state(
            beta,
            intercept,
            phase="proposal",
            iteration=it + 1,
            alpha=1.0,
            centred_intercept=fit.proposal_centred_intercept,
        )
        trial_cache: dict[float, _IRLSState] = {1.0: proposal}
        trial_directions: tuple[NDArray, float, NDArray] | None = None

        def evaluate_trial(alpha: float) -> _IRLSState:
            nonlocal trial_directions
            if trial_directions is None:
                trial_directions = (
                    proposal.beta - committed.beta,
                    proposal.intercept - committed.intercept,
                    proposal.eta_unclipped - committed.eta_unclipped,
                )
            beta_direction, intercept_direction, eta_direction = trial_directions
            beta_trial = committed.beta + alpha * beta_direction
            intercept_trial = committed.intercept + alpha * intercept_direction
            started = time.perf_counter()
            eta_trial = committed.eta_unclipped + alpha * eta_direction
            recorder.eta_formed(time.perf_counter() - started)
            centred_trial = None
            if committed.centred_intercept is not None and proposal.centred_intercept is not None:
                centred_trial = committed.centred_intercept + alpha * (
                    proposal.centred_intercept - committed.centred_intercept
                )
            candidate = evaluate_state(
                beta_trial,
                intercept_trial,
                phase="line_search_trial",
                iteration=it + 1,
                alpha=alpha,
                eta_unclipped=eta_trial,
                enclosing_proposal_state_id=proposal.state_id,
                centred_intercept=centred_trial,
            )
            trial_cache[alpha] = candidate
            return candidate

        committed_constraints_feasible = True
        proposal_constraints_feasible = True
        curvature = fit.curvature
        constraint_trial_is_invalid = (
            curvature.observed_rows_invalid if curvature.observed else None
        )
        if qp is not None:
            # the feasibility contract, on the QP step's constraint system
            committed_constraints_feasible = qp.feasible(committed.beta)
            proposal_constraints_feasible = qp.feasible(proposal.beta)

            def constraint_trial_is_invalid(candidate: _IRLSState) -> bool:
                return not qp.feasible(candidate.beta)

        trial_is_invalid = trial_refusal(
            family, link, committed, weights, constraint_trial_is_invalid
        )
        # the mean space's hooks (``_IRLSDirectMeanSpaceOverlay.line_search``):
        # its first halving, a lowered start's budget and, under Newton steps,
        # the merit on the model's own deviance
        first_halving = lowered_budget = newton_delta = newton_roundoff = None
        if mean_space is not None:
            first_halving, lowered_budget, newton_delta, newton_roundoff = mean_space.line_search(
                fit, committed, proposal, newton_score
            )
        decision = _select_irls_trial(
            committed=committed,
            proposal=proposal,
            evaluate_state=evaluate_trial,
            invalid_state=trial_is_invalid,
            max_halving=max_halving,
            extended_max_halving=extended_halving_budget(fit, committed, proposal, lowered_budget),
            merit_scale=fit.objective_merit_scale,
            merit_delta=lambda candidate, base: _stable_penalized_deviance_delta(
                candidate,
                base,
                penalty_matvec,
                deviance_delta=None if newton_delta is None else newton_delta(candidate, base),
            ),
            merit_roundoff=newton_roundoff,
            first_halving=first_halving,
        )
        if decision.skipped:
            recorder.count("irls_halvings_skipped", decision.skipped)
        if (
            qp is not None
            and decision.step_rejected
            and not committed_constraints_feasible
            and not (proposal_constraints_feasible and _state_is_finite(proposal))
        ):
            # The QP formed at an infeasible start gave no feasible
            # proposal: the fit restarts from the phase-one point, where
            # each violated block sits at its feasible zero.
            feasible_beta = _homogeneous_feasible_start(
                committed.beta, qp.A_all, qp.b_all, _QP_FEASIBILITY_TOL
            )
            if feasible_beta is not None:
                proposal = evaluate_state(
                    feasible_beta,
                    committed.intercept,
                    phase="proposal",
                    iteration=it + 1,
                    alpha=1.0,
                    centred_intercept=(
                        None
                        if committed.centred_intercept is None
                        else committed.centred_intercept
                        + math.fsum(fit._state_center * (feasible_beta - committed.beta))
                    ),
                )
                trial_cache[1.0] = proposal
                qp.proposal_converged = False
                proposal_constraints_feasible = True
        if (
            qp is not None
            and decision.step_rejected
            and not committed_constraints_feasible
            and proposal_constraints_feasible
            and _state_is_finite(proposal)
            and not (
                isinstance(family, Binomial)
                and type(link) is IdentityLink
                and trial_is_invalid is not None
                and trial_is_invalid(proposal)
            )
        ):
            # The objective comparison is not allowed to prefer an
            # inadmissible state over a finite feasible QP proposal. Once
            # this proposal is committed, convexity of A beta >= b and the
            # invalid-state predicate above preserve feasibility on every
            # subsequent accepted line-search segment.
            outside_link_domain = link_domain_violation(link, family)
            if outside_link_domain is not None and outside_link_domain(
                proposal.eta_unclipped, weights
            ):
                # Outside the link's domain the proposal is no state of
                # the model. No constraint row holds the intercept, so it
                # is raised as a start is; the state is then not the QP's
                # solution.
                raise_by = interior_start_intercept(
                    family,
                    link,
                    proposal.eta_unclipped,
                    weights,
                    0.0,
                    level=coefficient_initial_intercept(
                        distribution=family, link=link, y=y, sample_weight=weights
                    ),
                )
                proposal = evaluate_state(
                    proposal.beta,
                    proposal.intercept + raise_by,
                    phase="proposal",
                    iteration=it + 1,
                    alpha=1.0,
                    eta_unclipped=proposal.eta_unclipped + raise_by,
                    centred_intercept=(
                        None
                        if proposal.centred_intercept is None
                        else proposal.centred_intercept + raise_by
                    ),
                )
                trial_cache[1.0] = proposal
                qp.proposal_converged = False
            decision = _IRLSStepDecision(
                alpha=1.0,
                step_halvings=0,
                step_rejected=False,
                trials_attempted=decision.trials_attempted,
            )
            logger.info(
                "  irls_direct iter=%d: accepted finite feasible constrained "
                "proposal after objective line search found no admissible trial",
                it + 1,
            )
        retained = committed if decision.step_rejected else trial_cache[decision.alpha]
        if qp is not None:
            # Rejection retains coefficients from the preceding working
            # problem, while damping retains an interpolation.  Neither
            # state is the current H/g solution whose stationarity and
            # dual feasibility the inner QP certified, and a certificate
            # retained from the preceding iteration belongs to other
            # working weights: only a certified full proposal of this
            # iteration's working problem sets the flag.
            qp.retained_converged = bool(
                not decision.step_rejected and decision.alpha == 1.0 and qp.proposal_converged
            )
        retain_trial(fit, retained, decision, _t0)
        step_rejected = decision.step_rejected
        if step_rejected and qp is not None:
            # the rejected proposal's active set belongs to no retained state;
            # the finaliser reads the committed state's binding rows
            qp.active_set = qp.committed_active_set
        if decision.step_halvings and not step_rejected:
            logger.info(
                "  irls_direct iter=%d: accepted step fraction %.5g after %d halvings, dev=%.2e",
                it + 1,
                decision.alpha,
                decision.step_halvings,
                fit.dev,
            )
        fit.proposal_state = proposal
        fit.step_rejected = step_rejected
