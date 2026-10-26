"""The QP step: an active-set QP on the centred normal equations, for a fit with linear constraints."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.solvers.constrained_qp import QPResult, solve_constrained_qp
from superglm.solvers.irls_direct._step import _IRLSDirectStep, _note_svd_fallback
from superglm.solvers.mode_score import centred_data_score

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.types import GroupSlice

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


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
    than coefficients. The safeguarded result replaces the first one only if
    it passes the same complete certificate, so a solve that certifies
    without it is unchanged.

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


class _IRLSDirectQPStep(_IRLSDirectStep):
    """The fit's step when a group has linear constraints ``A beta >= b``: an active-set QP.

    Chosen where the route is decided (``_IRLSDirectFit._setup_routes``): a
    constrained fit has no SCOP group (``fit_irls_direct`` refuses the mix) and
    takes neither the QR route nor the structured one, whose selector refuses
    constrained groups; it has no state centre and no observed curvature.
    Each iteration solves the centred normal equations, the intercept profiled
    out, subject to the stacked constraints, warm-started from the last active
    set (``_solve_constrained_qp_with_cold_retry``).

    It keeps the constraint system, fixed for the whole fit, the active set
    with the committed state's copy, the proposal's and the retained state's
    KKT certificates, and the #472 retry and report latches; no reference to
    the fit, whose state each method is handed.  The feasibility contract the
    fit applies whatever the step (``_IRLSDirectFit._search``, ``_terminate``
    and ``_finish``) reads the constraint system and the certificates: it
    writes ``retained_converged``, and on a rejected step restores
    ``active_set`` to the committed state's.
    """

    __slots__ = (
        "A_all b_all abs_A_all active_set committed_active_set proposal_converged"
        " retained_converged _degeneracy_retry_failed _reported_nonconvergence"
    ).split()
    backend = "gram"

    def __init__(self, groups: list[GroupSlice], p: int) -> None:
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

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        active_set = self.active_set
        # the committed state's active set, which a rejected step restores
        self.committed_active_set = None if active_set is None else list(active_set)
        # Gram path: form X'WX via per-group gram, solve (p+1)×(p+1).
        _t0 = time.perf_counter()
        z_off = fit.z - fit.offset
        newton_score = fit.newton_score
        centered = fit.get_centered_system(fit.W, z_off)
        if newton_score is not None:
            centered = replace(
                centered,
                mean_z=centered.mean_z + float(np.sum(newton_score)) / centered.sum_w,
                rhs=centered.rhs + centred_data_score(fit.dm, newton_score, centered.mean_x),
            )
        fit._last_working_centered = centered
        fit.recorder.t_gram += time.perf_counter() - _t0

        # Solve the constrained system in its existing coordinate space.
        _t0 = time.perf_counter()
        qp_result = _solve_constrained_qp_with_cold_retry(
            centered.hessian,
            centered.rhs,
            self.A_all,
            self.b_all,
            active_set,
            degeneracy_retry=not self._degeneracy_retry_failed,
        )
        beta = qp_result.beta
        fit.intercept = centered.mean_z - float(centered.mean_x @ beta)
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
