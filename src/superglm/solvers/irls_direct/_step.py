"""What the direct IRLS fit asks of its step method (``_IRLSDirectStep``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.solvers.centered_system import CenteredSystem
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.irls_state import _IRLSState
    from superglm.solvers.pirls import PIRLSResult
    from superglm.solvers.rank import RankDecomposition

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


class _IRLSDirectStep(Protocol):
    """One fit's step method, chosen once at setup (``_IRLSDirectFit._setup_systems``,
    or ``_setup_design`` for the structured step, whose layout fixes the state's centre,
    or ``_setup_routes`` for the QP and SCOP steps).

    ``_IRLSDirectQRStep``, ``_IRLSDirectDenseStep``, ``_IRLSDirectQPStep``,
    ``_IRLSDirectStructuredStep`` and ``_IRLSDirectSCOPStep`` are the five.  Each
    subclasses this interface and takes ``None`` for the hooks it leaves out.  A
    step holds what it builds once per fit and no reference to the fit: each
    method is handed the fit, so the two are freed by reference counting, never
    by the collector.  The proposal and the working rows are the fit's own
    fields, not objects passed between them, so a step allocates nothing per
    iteration that the fit did not already.

    ``backend`` is the backend the fit publishes (``PIRLSResult.direct_backend``).
    ``finalise(fit)`` is the step's own finalisation of the retained model, or
    ``None`` for the centred one (``_IRLSDirectFit._final_system`` and
    ``_final_geometry``); a finaliser writes what the fit publishes from its
    geometry: ``XtWX_beta``, ``XtWX_S_inv_beta``, ``log_det_H``,
    ``reml_hessian_rank``, ``reml_geometry_summary``,
    ``retained_reml_decomposition``, ``p_eff``, ``rank_info`` and
    ``structured_factor``, and fills ``cache_out``.  The centred finaliser reads
    ``data_rank``, which factors the retained fit's data rank from the step's own
    design, as ``data_rank(centered_final, W)``; ``None`` takes the module route
    (``_retained_rank_info``).  A step's own data rank holds the step, so the fit
    computes it at once and never defers it.

    The hooks below are the SCOP step's; ``None`` on every other step, where the
    fit does what it does for all of them.

    - ``initial_state(fit, committed)``: the fit-entry state the loop starts
      from, in place of ``committed``, whose evaluation the fit then does not
      trace (the step traces its own).
    - ``search(fit, it)``: the step's own line search, in place of the shared
      one (``_IRLSDirectFit._search``).  It writes what that one writes: the
      retained state and its ``eta``, ``mu`` and deviance, ``beta`` and
      ``intercept``, the decision, ``n_halvings``, ``step_rejected`` and
      ``proposal_state``.
    - ``latent_change()``: the iteration's largest relative change in the
      step's own coordinates, which the coefficient criterion takes with the
      coefficients'.
    - ``damped(fit)``: whether the retained step was damped in a way that
      bars a stop this iteration.
    - ``latent_groups()``: the retained iterate's latent groups, each with its
      spec, which the true-score certificate tests in their own coordinates
      (``true_mode_residual``).
    - ``terminal(fit)``: the step's own state at the retained model, before the
      export rows are formed.
    - ``postfit(fit, result)``: the step's part of the publication; what it
      returns is the fit's ``scop_states`` return value.
    """

    __slots__ = ()
    backend: str
    data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None = None
    finalise: Callable[[_IRLSDirectFit], None] | None = None
    initial_state: Callable[[_IRLSDirectFit, _IRLSState], _IRLSState] | None = None
    search: Callable[[_IRLSDirectFit, int], None] | None = None
    latent_change: Callable[[], float] | None = None
    damped: Callable[[_IRLSDirectFit], bool] | None = None
    latent_groups: Callable[[], tuple] | None = None
    terminal: Callable[[_IRLSDirectFit], None] | None = None
    postfit: Callable[[_IRLSDirectFit, PIRLSResult], dict | None] | None = None

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        """Iteration ``it``'s full step from the fit's working rows (``W``, ``z``, ``newton_score``).

        It writes the proposal to the fit: ``beta``, ``intercept`` and
        ``proposal_centred_intercept``; the solve's ``_cond_est``, ``_used_svd``
        and ``rank_truncated``, and ``_held_rank[0]`` where it holds a
        decomposition; the system it solved (``_last_working_centered``, or the
        step's own) and, under ``mode_score``, ``_score_centre``; and its time to
        the recorder's phase timers.
        """


def _note_svd_fallback(fit: _IRLSDirectFit) -> None:
    """Count a Gram-path step's uncertified SVD fallback, and warn at the third in a row.

    A bounded factor pass can intentionally replace an uncertain Gram rank.
    That is kept in the iteration diagnostics, but it is neither reported as a
    failed solve nor a reason to recommend switching the whole fit to dense QR.
    """
    _used_svd, used_rank_certification = fit._used_svd, fit.used_rank_certification
    _consecutive_svd = fit._consecutive_svd
    warnable_svd_fallback = _used_svd and not used_rank_certification
    if warnable_svd_fallback:
        _consecutive_svd += 1
    else:
        _consecutive_svd = 0
    fit._consecutive_svd = _consecutive_svd
    if fit.direct_solve == "auto" and _consecutive_svd == 3:
        logger.warning(
            "fit_irls_direct: %d consecutive SVD fallbacks (cond ~%.1e). "
            "Consider direct_solve='qr' for near-collinear data.",
            _consecutive_svd,
            fit._cond_est,
        )
