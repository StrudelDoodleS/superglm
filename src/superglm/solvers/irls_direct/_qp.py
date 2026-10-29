"""The QP step: an active-set QP on the centred normal equations, for a fit with linear constraints."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm import _held_warnings as held_warnings
from superglm._roundoff import gamma as _gamma
from superglm.solvers.constrained_qp import (
    QPResult,
    _feasibility_slack,
    _homogeneous_feasible_start,
    _is_feasible,
    solve_constrained_qp,
)
from superglm.solvers.irls_direct._centred import _system_offset_mean
from superglm.solvers.irls_direct._finalise import _finalise_centred
from superglm.solvers.irls_direct._step import _IRLSDirectStep, _note_svd_fallback
from superglm.solvers.irls_state import link_domain_violation

if TYPE_CHECKING:
    from superglm.distributions import Distribution
    from superglm.links import Link
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.types import GroupSlice

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)

# The feasibility contract's tolerance on the scaled slack of ``A beta >= b``
# (``constrained_qp._is_feasible``).
_QP_FEASIBILITY_TOL = 1e-12


def _solve_constrained_qp_with_cold_retry(
    H: NDArray,
    g: NDArray,
    A: NDArray,
    b: NDArray,
    active_set_init: list[int] | None,
    *,
    degeneracy_retry: bool = True,
) -> QPResult:
    """Retry a failed warm active set without weakening the KKT contract.

    A warm active set is an acceleration hint, not part of the mathematical
    problem. Near an active boundary, platform-dependent roundoff can make the
    warm solve project a stationary candidate and correctly withhold its KKT
    certificate even though the identical problem converges from a cold active
    set. Retry only that failed warm case, and replace its best iterate only
    when the ordinary cold solve returns a complete certificate.

    If the ordinary solves end uncertified, one more cold solve runs with the
    degenerate-vertex safeguards (issue #472). That covers a failed warm
    solve followed by a failed cold one. It also covers a single failed cold
    solve, made when ``active_set_init`` is ``None`` or empty. The
    safeguards are for a flat monotone term at zero with more active rows
    than coefficients, and for a term whose coefficients are orders of
    magnitude below the rest. The safeguarded result replaces the first one
    only if it passes the same complete certificate, so a solve that
    certifies without it is unchanged.

    ``degeneracy_retry=False`` skips that attempt. ``fit_irls_direct`` passes
    it while the previous solve in the fit was uncertified and re-arms it
    after any certified solve: a QP that keeps failing pays for the attempt
    once per run of failures, not once per IRLS iteration. With the default,
    an uncertified result means the attempt ran and failed.
    """
    result = solve_constrained_qp(
        H,
        g,
        A,
        b,
        active_set_init=active_set_init,
    )
    if active_set_init and not result.converged:
        cold_result = solve_constrained_qp(
            H,
            g,
            A,
            b,
            active_set_init=None,
        )
        if cold_result.converged:
            return cold_result
    if degeneracy_retry and not result.converged:
        safeguarded = solve_constrained_qp(
            H,
            g,
            A,
            b,
            active_set_init=None,
            _degeneracy_safeguards=True,
        )
        if safeguarded.converged:
            return safeguarded
    return result


def _binding_rows(A: NDArray, b: NDArray, beta: NDArray, column_scale: NDArray) -> NDArray:
    """The rows of ``A beta >= b`` that bind at ``beta``, each scaled by a power of two.

    ``column_scale`` is the working columns' norms.  A slack below
    ``_QP_FEASIBILITY_TOL`` of the fit's largest column contribution reads as
    zero, whatever the coefficient's own size.
    """
    # Each row and its bound scaled by the one power of two that brings the
    # row's largest entry into [1/2, 1) (exact), so the fit's reading and the
    # row norms do not depend on a row's magnitude; a bound far larger than its
    # row leaves the row's norm in range.
    exponent = np.frexp(np.max(np.abs(A), axis=1))[1]
    held = np.ldexp(A, -exponent[:, None])
    bound = np.ldexp(b, -exponent)
    # A row binds when its slack is zero to rounding on its own scale
    # (overflow-safe) or on the whole fit's, the largest column contribution
    # |beta_j| ||x_j||_W read back in each column's units: a term the
    # constraint holds flat has coefficients at round-off, which its own scale
    # reads as free slopes.
    bar = _QP_FEASIBILITY_TOL + _gamma(A.shape[1] + 2)
    reach = float(np.max(np.abs(beta) * column_scale, initial=0.0))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        per_unit = np.divide(
            reach, column_scale, out=np.zeros_like(column_scale), where=column_scale > 0.0
        )
        # summed over the columns each row touches: a column it does not touch
        # cannot carry another column's overflow into it as 0 * inf
        reached = np.sum(np.abs(held) * per_unit, axis=1, where=held != 0.0)
        fit_scale = np.maximum(np.abs(bound), reached)
        on_fit = (held @ beta - bound) / fit_scale
    # Beyond float64's range (or at a zero scale) the fit's reading abstains;
    # the row's own reading is on the unscaled row, where a bound that
    # overflows when scaled stays readable.
    on_fit = np.isfinite(fit_scale) & (fit_scale > 0.0) & (on_fit <= bar)
    return held[(_feasibility_slack(A, beta, b) <= bar) | on_fit]


class _IRLSDirectQPStep(_IRLSDirectStep):
    """The fit's step when a group has linear constraints ``A beta >= b``: an active-set QP.

    Chosen where the route is decided (``_IRLSDirectFit._setup_routes``): a
    constrained fit has no SCOP group (``fit_irls_direct`` refuses the mix) and
    takes neither the QR route nor the structured one, whose selector refuses
    constrained groups; it has the dense step's state centre and no observed
    curvature.  Each iteration solves the centred normal equations, the
    intercept profiled out, subject to the stacked constraints, warm-started
    from the last active set (``_solve_constrained_qp_with_cold_retry``).

    It keeps the constraint system, fixed for the whole fit, the active set
    of the last solve and the one the iteration started from, the proposal's
    and the retained state's KKT certificates, the #472 retry and report
    latches, and whether the link's domain is bounded at zero; no reference
    to the fit, whose state each method is handed.  The feasibility contract
    the fit applies whatever the step (the shared line search, the stop
    rule's veto and ``_IRLSDirectFit._finish``) tests ``feasible`` and reads
    the certificates; the line search writes ``retained_converged`` and, on a
    rejected step, restores the iteration's starting active set, whose
    binding rows the finaliser reads (``binding_rows``).
    """

    __slots__ = (
        "A_all b_all abs_A_all active_set committed_active_set proposal_converged"
        " retained_converged _degeneracy_retry_failed _reported_nonconvergence _zero_bounded"
    ).split()
    backend = "gram"
    finalise = _finalise_centred

    def __init__(self, groups: list[GroupSlice], p: int, family: Distribution, link: Link) -> None:
        A_blocks: list[np.ndarray] = []
        b_blocks: list[np.ndarray] = []
        for g in groups:
            if g.constraints is not None:
                A_model = np.zeros((g.constraints.n_constraints, p))
                A_model[:, g.sl] = g.constraints.A
                A_blocks.append(A_model)
                b_blocks.append(g.constraints.b)
        self.A_all = A_all = np.vstack(A_blocks)
        self.b_all = np.concatenate(b_blocks)
        # The feasibility scale needs ``|A_all|`` on every test, and the aggregate
        # system is fixed for the whole fit -- so it is built once here rather than
        # rebuilt two to three times per IRLS iteration plus once per line-search
        # halving.  See ``constrained_qp._feasibility_slack``.
        self.abs_A_all = np.abs(A_all)
        self.active_set: list[int] | None = None
        self.committed_active_set: list[int] | None = None
        # A constrained fit-entry state has not been certified by the inner QP.
        self.proposal_converged = False
        self.retained_converged = False
        # the #472 retry is off after a failure, until a solve certifies
        self._degeneracy_retry_failed = False
        # transient note once; terminal authority is separate
        self._reported_nonconvergence = False
        # Under a link whose domain is bounded at zero (``link_domain_violation``)
        # rows at that edge can leave the working QP unbounded.
        self._zero_bounded = link_domain_violation(link, family) is not None

    def propose(self, fit: _IRLSDirectFit, it: int) -> bool | None:
        """Solve this iteration's working QP; ``True`` when the fit ends at its committed state.

        A ``ValueError`` from the working QP under a link bounded at zero
        (``_zero_bounded``) is not raised: an infeasible committed state
        restarts from its phase-one point, as one whose QP proposal stays
        infeasible does (``_IRLSDirectTrialSearch``), and any other committed
        state is returned unconverged and disclosed.
        """
        active_set = self.active_set
        self.committed_active_set = None if active_set is None else list(active_set)
        # Gram path: form X'WX via per-group gram, solve (p+1)×(p+1).
        _t0 = time.perf_counter()
        z_off = fit.z - fit.offset
        newton_score = fit.newton_score
        centered = fit.get_centered_system(fit.W, z_off)
        if newton_score is not None:
            # the mean space's Newton score in the right-hand side
            centered, _ = fit.mean_space.centred_rhs(centered, newton_score, fit.dm)
        fit._last_working_centered = centered
        fit.recorder.t_gram += time.perf_counter() - _t0

        # Solve the constrained system in its existing coordinate space.
        _t0 = time.perf_counter()
        try:
            qp_result = _solve_constrained_qp_with_cold_retry(
                centered.hessian,
                centered.rhs,
                self.A_all,
                self.b_all,
                active_set,
                degeneracy_retry=not self._degeneracy_retry_failed,
            )
        except ValueError as error:
            if not self._zero_bounded:
                raise
            if self._unbounded(fit, error):
                return True
            _note_svd_fallback(fit)
            return None
        beta = qp_result.beta
        fit.intercept = centered.mean_z - float(centered.mean_x @ beta)
        _state_center = fit._state_center
        if _state_center is not None:
            # the intercept about the state's centre, as the unconstrained
            # solve forms it: shape rows act on the coefficients only, so the
            # centred state is the same
            proposal_centred_intercept = centered.mean_z - math.fsum(
                _system_offset_mean(fit.dm, fit.W, centered, _state_center) * beta
            )
            fit.proposal_centred_intercept = proposal_centred_intercept
            fit.intercept = proposal_centred_intercept - math.fsum(_state_center * beta)
        self.proposal_converged = bool(qp_result.converged)
        self.active_set = qp_result.active_set
        # An uncertified result means the degenerate-vertex retry ran
        # and failed (or was already latched off). It costs up to a
        # full active-set solve, so it is not repeated while the QP
        # keeps failing. A certified result re-arms it: while latched,
        # that can only be an ordinary solve, so the failure was
        # transient and a later #472 vertex keeps its rescue.
        self._degeneracy_retry_failed = not qp_result.converged
        # Non-convergence usually persists for the rest of the fit, so
        # latch the report to the first occurrence rather than emitting
        # one identical line per IRLS iteration. A later solve may
        # recover; the retained state's terminal authority is checked
        # independently (``_IRLSDirectFit._finish``).
        if not qp_result.converged and not self._reported_nonconvergence:
            self._reported_nonconvergence = True
            logger.info(
                "fit_irls_direct: constrained QP did not converge at "
                "iteration %d; its KKT certificate is incomplete. "
                "A later IRLS iteration may recover; subsequent inner "
                "failures in this fit are not reported here.",
                it + 1,
            )
        # Report what the QP actually inverted.  These were hardcoded
        # to False/0.0, so the constrained branch published "perfectly
        # conditioned" to the consumers below while the unconstrained
        # branch read the real numbers off its decomposition.
        fit._used_svd = qp_result.used_svd_fallback
        fit._cond_est = qp_result.condition
        fit.recorder.t_solve += time.perf_counter() - _t0
        fit.beta = beta
        _note_svd_fallback(fit)
        return None

    def _unbounded(self, fit: _IRLSDirectFit, error: ValueError) -> bool:
        """After the working QP raised: a phase-one restart (``False``), or the fit's end (``True``).

        An infeasible committed state restarts from its phase-one point, where
        each violated block sits at its feasible zero, its intercept held.  A
        feasible one, or one with no phase-one point, ends the fit there,
        unconverged, with a ``ConvergenceWarning``: the iteration's line search
        and stop rule do not run, and the retained state is the committed one.
        """
        committed = fit.committed
        restart_beta = (
            None
            if self.feasible(committed.beta)
            else _homogeneous_feasible_start(
                committed.beta, self.A_all, self.b_all, _QP_FEASIBILITY_TOL
            )
        )
        if restart_beta is None:
            from superglm.diagnostics.convergence import ConvergenceWarning

            link, family = fit.link, fit.family
            # fit_irls_direct's frame: _unbounded, propose, run, _fit_irls_direct_once, it
            held_warnings.warn(
                "The coefficient fit stopped early and returns its last "
                "accepted coefficients, not converged: some rows' linear "
                "predictor sits at zero, the edge of the "
                f"range the {type(link).__name__} allows a "
                f"{type(family).__name__} mean, where the next working problem "
                "is unbounded. This link may not suit these data; a log link "
                f"cannot reach that edge. The working problem reported: {error}",
                ConvergenceWarning,
                stacklevel=5,
            )
            # the committed state, uncertified by this iteration's working problem
            self.proposal_converged = self.retained_converged = False
            fit.beta, fit.intercept = committed.beta.copy(), committed.intercept
            fit.retained, fit.dev, fit.step_rejected = committed, committed.deviance, True
            # the final refusal (``_IRLSDirectFit._finish``) names the reason
            fit.termination_reason = "constraint_kkt_incomplete"
            return True
        # the intercept held, as the phase-one restart of the line search holds it
        fit.beta = restart_beta
        fit.intercept = committed.intercept
        if committed.centred_intercept is not None:
            fit.proposal_centred_intercept = committed.centred_intercept + math.fsum(
                fit._state_center * (restart_beta - committed.beta)
            )
        self.proposal_converged = False
        self.active_set = None
        # no system was inverted this iteration
        fit._used_svd, fit._cond_est = False, math.nan
        return False

    def binding_rows(self, beta: NDArray, column_scale: NDArray) -> NDArray:
        """The rows of the last solve's active set that bind at ``beta`` (``_binding_rows``)."""
        active_rows = np.asarray([] if self.active_set is None else self.active_set, dtype=np.intp)
        return _binding_rows(self.A_all[active_rows], self.b_all[active_rows], beta, column_scale)

    def feasible(self, beta: NDArray) -> bool:
        """Whether ``beta`` satisfies ``A beta >= b`` within the feasibility contract's tolerance."""
        return _is_feasible(self.A_all, beta, self.b_all, _QP_FEASIBILITY_TOL, abs_A=self.abs_A_all)

    def certificate_inputs(self) -> tuple:
        """The constraint rows active at the proposal: the certificate's KKT residual reads them."""
        active_set = self.active_set
        return None, self.A_all[list(active_set)] if active_set else None, None, None
