"""What a neighbourhood cross-validation search returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import NDArray

from superglm.reml.result import REMLResult

StopReason = Literal[
    "gradient",
    "no_smoothing_parameters",
    "criterion_settled",
    "max_iter",
    "line_search",
    "no_finite_criterion",
]
"""Why a search ended; see :data:`STOP_REASONS`."""

STOP_REASONS: dict[StopReason, str] = {
    "gradient": (
        "the first-order test was met: no slope of the criterion in a log smoothing parameter "
        "exceeds the tolerance times the criterion, a slope that points out of the box at a "
        "bound aside"
    ),
    "no_smoothing_parameters": "the model has no smoothing parameter to estimate",
    "criterion_settled": (
        "a step lowered the criterion by less than the square of the tolerance, relative to "
        "it, before the first-order test was met"
    ),
    "max_iter": "the limit on steps was reached",
    "line_search": "no step that lowers the criterion was found from the smoothing returned",
    "no_finite_criterion": "the criterion was not finite at any smoothing tried",
}
"""Every way a search ends, with what it means.  Only the first two are a converged search."""


@dataclass
class NCVResult(REMLResult):
    """The smoothing a neighbourhood cross-validation search chose, and how the search went.

    Read as a :class:`~superglm.reml.result.REMLResult`: ``lambdas`` holds every penalty
    component (the fixed ones at their values), ``n_reml_iter`` the accepted steps,
    ``objective`` the criterion (the summed held-out deviance) at ``lambdas``,
    ``lambda_history`` and ``objective_history`` each start and each accepted step,
    ``termination_reason`` one of :data:`STOP_REASONS`, and ``bound_pressure`` the estimated
    components at a bound whose slope still points out of the box by the stopping bar or more.

    ``converged`` is a first-order stop of a local numerical search, not a proof of a minimum.
    When no finite criterion was found, ``objective`` and ``gradient`` are None and ``lambdas``
    is the smoothing the search started from.

    The result holds no design-sized or coefficient-sized array of the search and nothing that
    depends on the basis it ran in: ``pirls_result`` and ``reml_penalties`` are None until a fit
    is published at ``lambdas``.
    """

    criterion: str = "ncv"
    #: The criterion over the likelihood size of the scored rows.
    criterion_mean: float | None = None
    #: Criterion evaluations, the infeasible ones included.
    n_evaluations: int = 0
    #: Evaluations with no certified inner mode or no finite, differentiable criterion.
    infeasible_trials: int = 0
    #: The criterion's slope in each estimated log smoothing parameter at ``lambdas``.
    gradient: NDArray | None = None
    #: The estimated components' smoothing at the search's first start.
    start_lambdas: dict[str, float] = field(default_factory=dict)
    #: ``(component, "floor" | "ceiling")`` for each estimated component on a bound of the box.
    at_bound: tuple[tuple[str, str], ...] = ()
    #: The neighbourhoods' form, count and sizes.
    neighbourhoods: dict[str, Any] = field(default_factory=dict)
    #: Work counts of the whole search (``engine.WorkCounts``, summed over its evaluations).
    work: dict[str, float] = field(default_factory=dict)
    #: The held-out results at ``lambdas``.
    heldout: Any | None = None


__all__ = ["STOP_REASONS", "NCVResult", "StopReason"]
