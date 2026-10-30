"""The direct IRLS fit's stop rule (``_IRLSDirectConvergence``): one verdict per iteration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from superglm.solvers.irls_state import _irls_objective_relative_change
from superglm.solvers.mode_score import rate_exhausts_budget, stagnation_window

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.mode_score import ModeResidual
    from superglm.solvers.pirls import TerminationReason


class _IRLSDirectConvergence:
    """The fit's stop rule: whether an iteration's retained state ends the solve, and why.

    ``judge`` gives one verdict per iteration, in this order, each part able
    to overrule the parts before it:

    1. the criterion the fit's ``convergence`` names, on the retained state,
       then the step-rejected veto (``_criterion``);
    2. the binomial/log mean space's judge, on the model's own score
       (``_IRLSDirectMeanSpaceOverlay.judge``), when the fit holds the overlay;
    3. the inner QP's KKT veto on a constrained fit's retained state, then the
       termination reason (``_terminate``).

    The verdict is the fit's: ``converged_this_iter``, ``score_stagnated``,
    ``termination_reason``, the rule's numbers ``convergence_label``,
    ``convergence_value`` and ``convergence_tolerance`` (the judge's when it
    decides), and ``coef_change`` and ``dev_rel_change``.  The loop, the
    recorder and the finalisation read them there.  The rule keeps what
    lasts across iterations: the mode score's ratios, its last residual
    (``last_mode_residual``, which the recorder publishes), and the
    stagnation window, which the judge's refused stops use too.

    It is not pure: the criterion switches the fit's curvature to Newton
    steps (``_IRLSDirectCurvaturePolicy.switch_to_newton``), and the judge
    may engage the mean space's Newton steps; the next iteration's working
    rows and step read both.  It keeps no reference to the fit, whose state
    each method is handed.
    """

    __slots__ = ("stagnation_window", "_mode_ratios", "last_mode_residual")

    def __init__(self, max_iter: int, mode_bar: float) -> None:
        # the iterations over which a resolved score must at least halve
        # (``mode_score.stagnation_window``)
        self.stagnation_window = stagnation_window(max_iter, mode_bar)
        # the certificate ratio at every iterate of a mode_score solve, for the
        # stagnation stop
        self._mode_ratios: list[float] = []
        self.last_mode_residual: ModeResidual | None = None

    def judge(self, fit: _IRLSDirectFit, it: int) -> None:
        """Iteration ``it``'s verdict on the retained state, in the order above."""
        self._criterion(fit, it)
        mean_space = fit.mean_space
        if mean_space is not None:
            mean_space.judge(fit, self.stagnation_window)
        self._terminate(fit, it)

    def _criterion(self, fit: _IRLSDirectFit, it: int) -> None:
        """The criterion the fit's ``convergence`` names, then the step-rejected veto (L16).

        - ``"mode_score"``, once a step has set the score's centre: the
          certificate's relative score against the bar, the stagnation stop,
          the Gamma/log switch to Newton steps and the float64 fixed point.
        - ``"coefficients"``, and ``"mode_score"`` on a route that sets no
          centre: the largest relative change in the coefficients, the
          intercept and the step's latent coordinates (``latent_change``).
        - ``"deviance"``: the relative change in the penalised objective
          from the committed state.

        Under the last two a damped step whose change bounds nothing (the
        step's ``damped``) never stops the fit.
        """
        convergence, tol, beta, step = fit.convergence, fit.tol, fit.beta, fit.step
        intercept, retained, committed = fit.intercept, fit.retained, fit.committed
        beta_prev, intercept_prev = committed.beta, committed.intercept
        step_rejected = fit.step_rejected
        _mode_ratios, _stagnation_window = self._mode_ratios, self.stagnation_window
        _last_mode_residual = self.last_mode_residual
        # a stagnated score ends the loop, so every iteration starts unstagnated
        score_stagnated = False
        dev_rel_change = None
        coef_change = None
        fit.convergence_label = convergence
        fit.convergence_tolerance = tol
        if np.isfinite(fit.dev):
            if convergence == "mode_score" and fit._score_centre is not None:
                # One-engine design §3.8: stop on the certificate's own score,
                # in centred coordinates over the identified coefficients,
                # once it meets the bar the certificate reads (no margin: the
                # certificate is this residual, ``mode_score``).  The step
                # length, with its raw intercept, is only reported.
                coef_change = float(
                    np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta)), initial=0.0)
                )
                residual = fit.mode_residual(
                    beta, intercept, fit.mu, fit.eta, centred_intercept=retained.centred_intercept
                )
                convergence_value = fit.mode_bar * residual.ratio()
                converged_this_iter = residual.ratio() <= 1.0
                _last_mode_residual = residual
                _mode_ratios.append(residual.ratio())
                # The score stopped contracting (``stagnation_window``):
                # the iterate is at its limiting accuracy, so the solve ends
                # uncertified and the caller discloses it.
                score_stagnated = (
                    not converged_this_iter
                    and residual.resolved
                    and len(_mode_ratios) > _stagnation_window
                    and min(_mode_ratios[-_stagnation_window:])
                    > 0.5 * min(_mode_ratios[:-_stagnation_window])
                )
                switched_curvature, curvature = False, fit.curvature
                if (
                    curvature.stagnation_newton_available
                    and not curvature.observed
                    and not converged_this_iter
                    and (
                        score_stagnated
                        or (
                            residual.resolved
                            and rate_exhausts_budget(
                                _mode_ratios, _stagnation_window, it + 1, fit.max_iter
                            )
                        )
                    )
                ):
                    # Fisher scoring near the mode contracts too slowly to
                    # meet the bar in the budget, or not at all (the comment
                    # at ``_IRLSDirectCurvaturePolicy``): the iterations left
                    # take Newton steps on the observed information.
                    curvature.switch_to_newton(fit.recorder)
                    score_stagnated = False
                    switched_curvature = True
                    _mode_ratios.clear()
                if (
                    not converged_this_iter
                    and not switched_curvature
                    and not step_rejected
                    and intercept == intercept_prev
                    and retained.centred_intercept == committed.centred_intercept
                    and np.array_equal(beta, beta_prev)
                ):
                    # The full step left the state bitwise where it was: the
                    # iteration is a deterministic map at a float64 fixed
                    # point, and every further iteration repeats this one
                    # (the step-tolerance stop of Dennis & Schnabel 1996,
                    # section 7.2, at its strictest).  A centred state is
                    # ``(alpha, beta)``, and the raw intercept only its
                    # reading, which rounds ``alpha - c' beta`` at ``|c' beta|``
                    # and can hold still while ``alpha`` moves, so ``alpha``
                    # is compared too (``None`` on both sides otherwise).  An
                    # iteration that has just changed the map (the Newton
                    # switch above) is not one: the next iteration takes the
                    # other step.  A fixed point ends the solve, with the
                    # floors resolved whatever the relative score: where
                    # the residuals are at the rounding of the fitted values
                    # (a model that fits its response exactly, as a linear
                    # model can) the score's own scale ``sum |s|`` is
                    # rounding too, the relative score sits near 1 and the
                    # resolve cap would never let the floor be read.  A score
                    # within its floor is certified; one above it is the
                    # limiting accuracy, published as not converged.
                    residual = fit.mode_residual(
                        beta,
                        intercept,
                        fit.mu,
                        fit.eta,
                        centred_intercept=retained.centred_intercept,
                        resolve_cap=math.inf,
                    )
                    convergence_value = fit.mode_bar * residual.ratio()
                    converged_this_iter = residual.ratio() <= 1.0
                    score_stagnated = not converged_this_iter
                    _last_mode_residual = residual
                    _mode_ratios[-1] = residual.ratio()
            elif convergence in ("coefficients", "mode_score"):
                coef_change = float(
                    np.max(np.abs(beta - beta_prev) / np.maximum(1.0, np.abs(beta)), initial=0.0)
                )
                coef_change = max(
                    coef_change,
                    abs(intercept - intercept_prev) / max(1.0, abs(intercept)),
                )
                latent_change = step.latent_change
                if latent_change is not None:
                    coef_change = max(coef_change, latent_change())
                convergence_value = coef_change
                converged_this_iter = coef_change < tol
                damped = step.damped
                if damped is not None and damped(fit):
                    # a damped step's change is a fraction of the full
                    # step's and bounds nothing either (below)
                    converged_this_iter = False
            else:
                objective = (
                    retained.deviance
                    if retained.penalized_deviance is None
                    else retained.penalized_deviance
                )
                # the committed state's objective: the iteration's base
                objective_prev = (
                    committed.deviance
                    if committed.penalized_deviance is None
                    else committed.penalized_deviance
                )
                if np.isfinite(objective_prev):
                    dev_rel_change = _irls_objective_relative_change(
                        objective=objective,
                        previous=objective_prev,
                        objective_scale=fit.objective_merit_scale,
                    )
                converged_this_iter = dev_rel_change is not None and dev_rel_change < tol
                damped = step.damped
                if damped is not None and damped(fit):
                    # A step whose decrease bounds nothing never stops the fit
                    # (the SCOP step's: a damped joint Newton step).
                    converged_this_iter = False
                convergence_value = dev_rel_change
        else:
            converged_this_iter = False
            convergence_value = None
        if step_rejected:
            # a rejected step never stops the fit
            converged_this_iter = False
        fit.dev_rel_change, fit.coef_change = dev_rel_change, coef_change
        fit.convergence_value, fit.converged_this_iter = convergence_value, converged_this_iter
        fit.score_stagnated, self.last_mode_residual = score_stagnated, _last_mode_residual

    def _terminate(self, fit: _IRLSDirectFit, it: int) -> None:
        """The inner QP's KKT veto on a constrained fit, then the termination reason (L18).

        The certificate stands for primal feasibility too: the inner QP
        certifies a solve only where its own feasibility test passes at the
        contract's tolerance, and the line search hands the certificate only to
        the full proposal it solved (``_IRLSDirectQPStep.retained_converged``),
        so a retained state outside ``A beta >= b`` never stops the fit.  The
        state's feasibility still names the reason a fit ends on.
        """
        converged_this_iter, step_rejected = fit.converged_this_iter, fit.step_rejected
        dev, max_iter, qp = fit.dev, fit.max_iter, fit.qp
        constraints_feasible_this_iter = True
        if qp is not None:
            constraints_feasible_this_iter = qp.feasible(fit.beta)
            # Outer coefficient/deviance stagnation cannot replace the inner
            # QP's certificate of stationarity, dual and primal feasibility.
            # Keep iterating while budget remains: a later full QP proposal may
            # still obtain the missing certificate, or repair an infeasible
            # warm start.
            if not qp.retained_converged:
                converged_this_iter = False

        terminal_constraint_infeasible = bool(
            not constraints_feasible_this_iter
            and (step_rejected or not np.isfinite(dev) or it + 1 == max_iter)
        )
        terminal_constraint_kkt_incomplete = bool(
            qp is not None
            and not qp.retained_converged
            and (step_rejected or not np.isfinite(dev) or it + 1 == max_iter)
        )

        termination_reason: TerminationReason
        if terminal_constraint_infeasible:
            termination_reason = "constraint_infeasible"
        elif terminal_constraint_kkt_incomplete:
            termination_reason = "constraint_kkt_incomplete"
        elif step_rejected:
            termination_reason = "step_rejected"
        elif not np.isfinite(dev):
            termination_reason = "nonfinite_deviance"
        elif converged_this_iter:
            termination_reason = "converged"
        elif fit.score_stagnated:
            termination_reason = "score_stagnated"
        elif it + 1 == max_iter:
            termination_reason = "max_iter"
        else:
            termination_reason = "continue"
        fit.converged_this_iter, fit.termination_reason = converged_this_iter, termination_reason
