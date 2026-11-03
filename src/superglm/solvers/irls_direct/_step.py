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

# One logger for the whole package, so every submodule's records share its name.
logger = logging.getLogger(__package__)


class _IRLSDirectStep(Protocol):
    """One fit's step method (QR, dense, QP or structured), chosen once at setup.

    A step holds what it builds once per fit and no reference to the fit: each
    method is handed the fit, so the two are freed by reference counting, not
    by the collector.  ``backend`` is published as
    ``PIRLSResult.direct_backend``.  ``finalise(fit)`` fills ``cache_out`` after
    the loop and returns the retained geometry.  ``data_rank(centered_final, W)``
    factors the retained data rank from the step's own design; ``None`` takes
    the module route (``_retained_rank_info``).
    """

    __slots__ = ()
    backend: str
    finalise: Callable[[_IRLSDirectFit], _IRLSDirectFinalGeometry]
    data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None = None

    def propose(self, fit: _IRLSDirectFit, it: int) -> bool | None:
        """Iteration ``it``'s full step from the fit's working rows (``W``, ``z``, ``newton_score``).

        Writes the proposal (``beta``, ``intercept``,
        ``proposal_centred_intercept``), the solve's diagnostics and the system
        it solved to the fit.  ``True`` ends the fit at its committed state
        before the search; the step has then written the retained state itself
        (the QP step, when its working problem is unbounded at a link's zero
        edge).
        """

    def certificate_inputs(self) -> tuple:
        """What the binomial/log true-score certificate reads of this iteration's step.

        ``(held_rank, active_rows)``: the decomposition whose truncated
        directions it judges (QR and dense) and the constraint rows active at
        the proposal (QP); ``None`` for each a step does not hold.
        """
        return _NO_CERTIFICATE_INPUTS


_NO_CERTIFICATE_INPUTS = (None, None)


def _note_svd_fallback(fit: _IRLSDirectFit) -> None:
    """Count a Gram-path step's uncertified SVD fallback, and warn at the third in a row.

    A certified factor pass that replaces an uncertain Gram rank is not counted:
    it is not a failed solve and no reason to recommend dense QR.
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
