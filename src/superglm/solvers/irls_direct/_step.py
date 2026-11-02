"""What the direct IRLS fit asks of its step method (``_IRLSDirectStep``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.solvers.centered_system import CenteredSystem
    from superglm.solvers.irls_direct._finalise import _IRLSDirectFinalGeometry
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.rank import RankDecomposition

# the package's logger: its records keep the name they had when the solver was one module
logger = logging.getLogger(__package__)


class _IRLSDirectStep(Protocol):
    """One fit's step method, chosen once at setup (``_IRLSDirectFit._setup_systems``,
    or ``_setup_design`` for the structured step, whose layout fixes the state's centre,
    or ``_setup_routes`` for the QP step).

    ``_IRLSDirectQRStep``, ``_IRLSDirectDenseStep``, ``_IRLSDirectQPStep`` and
    ``_IRLSDirectStructuredStep`` are the four.  Each subclasses this interface.  A
    step holds what it builds once per fit and no reference to the fit: each
    method is handed the fit, so the two are freed by reference counting, never
    by the collector.  The proposal and the working rows are the fit's own
    fields, not objects passed between them, so a step allocates nothing per
    iteration that the fit did not already.

    ``backend`` is the backend the fit publishes (``PIRLSResult.direct_backend``).
    ``finalise(fit)`` is the finaliser the step names, which every step must:
    after the loop it fills ``cache_out`` and returns the retained model's
    geometry, which the fit publishes (``_finalise._IRLSDirectFinalGeometry``).
    The structured step has its own; the other three name the centred one
    (``_finalise._finalise_centred``).  The centred finaliser reads
    ``data_rank``, which factors the retained fit's data rank from the step's own
    design, as ``data_rank(centered_final, W)``; ``None`` takes the module route
    (``_retained_rank_info``).  A step's own data rank holds the step, so the fit
    computes it at once and never defers it.
    """

    __slots__ = ()
    backend: str
    finalise: Callable[[_IRLSDirectFit], _IRLSDirectFinalGeometry]
    data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None = None

    def propose(self, fit: _IRLSDirectFit, it: int) -> bool | None:
        """Iteration ``it``'s full step from the fit's working rows (``W``, ``z``, ``newton_score``).

        It writes the proposal to the fit: ``beta``, ``intercept`` and
        ``proposal_centred_intercept``; the solve's ``_cond_est``, ``_used_svd``
        and ``rank_truncated``; the system it solved (``_last_working_centered``,
        or the step's own) and, under ``mode_score``, ``_score_centre``; and its
        time to the recorder's phase timers.  ``True`` ends the fit at its
        committed state before the iteration's search: the step has written the
        retained state, ``dev`` and ``step_rejected`` itself (the QP step, when
        its working problem is unbounded at a link's zero edge).
        """

    def certificate_inputs(self) -> tuple:
        """What the binomial/log true-score certificate reads of this iteration's step.

        ``(held_rank, active_rows)`` (``_IRLSDirectMeanSpaceOverlay.certify``):
        the decomposition the solve holds, whose truncated directions it judges
        (QR and dense), and the constraint rows active at the proposal (QP).
        ``None`` for each a step does not hold.
        """
        return _NO_CERTIFICATE_INPUTS


_NO_CERTIFICATE_INPUTS = (None, None)


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
