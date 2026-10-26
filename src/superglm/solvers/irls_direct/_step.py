"""What the direct IRLS fit asks of its step method (``_IRLSDirectStep``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from superglm.solvers.centered_system import CenteredSystem
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit
    from superglm.solvers.rank import RankDecomposition


class _IRLSDirectStep(Protocol):
    """One fit's step method, chosen once at setup (``_IRLSDirectFit._setup_systems``).

    ``_IRLSDirectQRStep`` is one; ``_IRLSDirectGramStep`` covers the routes that
    are not yet steps of their own (SCOP, structured, dense and the QP).  A step
    holds what it builds once per fit and no reference to the fit: each method
    is handed the fit, so the two are freed by reference counting, never by the
    collector.  The proposal and the working rows are the fit's own fields, not
    objects passed between them, so a step allocates nothing per iteration that
    the fit did not already.

    ``backend`` is the backend the fit publishes (``PIRLSResult.direct_backend``).
    ``data_rank`` factors the retained fit's data rank from the step's own design,
    as ``data_rank(centered_final, W)``; ``None`` takes the module route
    (``_retained_rank_info``).  A step's own data rank holds the step, so the fit
    computes it at once and never defers it.
    """

    backend: str
    data_rank: Callable[[CenteredSystem, NDArray], RankDecomposition] | None

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        """Iteration ``it``'s full step from the fit's working rows (``W``, ``z``, ``newton_score``).

        It writes the proposal to the fit: ``beta``, ``intercept`` and
        ``proposal_centred_intercept``; the solve's ``_cond_est``, ``_used_svd``
        and ``rank_truncated``, and ``_held_rank[0]`` where it holds a
        decomposition; the system it solved (``_last_working_centered`` or
        ``_last_working_structured``) and, under ``mode_score``,
        ``_score_centre``; and its time to the recorder's phase timers.
        """
