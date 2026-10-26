"""The direct IRLS fit's shared line search (``_IRLSDirectTrialSearch``)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from superglm.solvers.irls_state import (
    _IRLSState,
    _IRLSStepDecision,
    _poisson_sqrt_halving_budget,
    _select_irls_trial,
    _stable_penalized_deviance_delta,
    _state_is_finite,
    link_domain_violation,
    mean_space_violation,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.distributions import Distribution
    from superglm.links import Link
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


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
    state would otherwise reject, and certifies the retained state's KKT
    conditions only on a full step (``_IRLSDirectQPStep.retained_converged``).

    ``search(fit, it)`` writes what the SCOP step's own search writes: the
    retained state and its ``eta``, ``mu`` and deviance, ``beta`` and
    ``intercept``, the decision, ``n_halvings``, ``step_rejected`` and
    ``proposal_state``.  The unit keeps the two domain predicates, fixed for
    the fit, and no reference to the fit.
    """

    __slots__ = ("_mean_space_invalid", "_link_domain_invalid")

    def __init__(self, family: Distribution, link: Link) -> None:
        # The family's mean space, when the link's inverse can leave it (declared
        # by the family and link, ``irls_state.mean_space_violation``).
        self._mean_space_invalid = mean_space_violation(family, link)
        # The link inverse's own domain (``irls_state.link_domain_violation``).
        self._link_domain_invalid = link_domain_violation(link)

    def search(self, fit: _IRLSDirectFit, it: int) -> None:
        """Iteration ``it``'s line search from the committed state to the step's proposal."""
        beta, intercept, committed = fit.beta, fit.intercept, fit.committed
        qp, newton_score, mean_space = fit.qp, fit.newton_score, fit.mean_space
        y, weights = fit.y, fit.weights
        family, link, _mean_space_invalid = fit.family, fit.link, self._mean_space_invalid
        _link_domain_invalid = self._link_domain_invalid
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
            eta_trial = committed.eta_unclipped + alpha * eta_direction
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

        trial_is_invalid = constraint_trial_is_invalid
        if _mean_space_invalid is not None and not _mean_space_invalid(
            committed.eta_unclipped, weights
        ):
            # A feasible committed state keeps every accepted trial in the
            # family's mean space (``mean_space_violation``).
            def trial_is_invalid(candidate: _IRLSState, other=constraint_trial_is_invalid) -> bool:
                assert _mean_space_invalid is not None
                if _mean_space_invalid(candidate.eta_unclipped, weights):
                    return True
                return other is not None and other(candidate)

        if _link_domain_invalid is not None and not _link_domain_invalid(
            committed.eta_unclipped, weights
        ):
            # A committed state inside the link's domain keeps every
            # accepted trial inside it (``link_domain_violation``).
            def trial_is_invalid(candidate: _IRLSState, other=trial_is_invalid) -> bool:
                assert _link_domain_invalid is not None
                if _link_domain_invalid(candidate.eta_unclipped, weights):
                    return True
                return other is not None and other(candidate)

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
            extended_max_halving=lambda: max(
                _poisson_sqrt_halving_budget(
                    committed=committed,
                    proposal=proposal,
                    y=y,
                    weights=weights,
                    family=family,
                    link=link,
                    default=max_halving,
                ),
                max_halving if lowered_budget is None else lowered_budget(),
            ),
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
            and proposal_constraints_feasible
            and _state_is_finite(proposal)
        ):
            # The objective comparison is not allowed to prefer an
            # inadmissible state over a finite feasible QP proposal. Once
            # this proposal is committed, convexity of A beta >= b and the
            # invalid-state predicate above preserve feasibility on every
            # subsequent accepted line-search segment.
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
        evaluation_elapsed = time.perf_counter() - _t0
        recorder.t_deviance += evaluation_elapsed
        recorder.t_deviance_eval += evaluation_elapsed
        beta = retained.beta.copy()
        intercept = retained.intercept
        fit.eta_unclipped = retained.eta_unclipped
        fit.eta = retained.eta
        fit.mu = retained.mu
        dev = retained.deviance
        n_halvings = decision.step_halvings
        step_rejected = decision.step_rejected
        if n_halvings and not step_rejected:
            logger.info(
                "  irls_direct iter=%d: accepted step fraction %.5g after %d halvings, dev=%.2e",
                it + 1,
                decision.alpha,
                n_halvings,
                dev,
            )
        fit.proposal_state, fit.decision, fit.retained = proposal, decision, retained
        fit.beta = beta
        fit.intercept, fit.dev, fit.n_halvings = intercept, dev, n_halvings
        fit.step_rejected = step_rejected
