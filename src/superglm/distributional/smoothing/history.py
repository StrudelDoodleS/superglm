"""Release obsolete smoothing rows while preserving complete fit evidence."""

from collections.abc import Sequence
from dataclasses import replace

from superglm.distributional.results.iteration import (
    DistributionalEFSConfig,
    DistributionalEFSIteration,
)
from superglm.distributional.results.solver import DenseSolverResult


def compact_coefficient_history(
    coefficient_fits: list[DenseSolverResult],
    history: Sequence[DistributionalEFSIteration],
    *,
    terminal_fit_index: int,
    config: DistributionalEFSConfig,
) -> None:
    """Keep real rows for terminal/plateau replay, replacing only old entries.

    Live solver and endpoint/retry references remain full results. This changes
    only the history's references, never an object a caller might still use.
    """
    if config.retain_history_rows:
        return
    retained = {terminal_fit_index}
    for item in history[-config.plateau_iterations :]:
        retained.add(item.source_fit_index)
        if item.accepted_fit_index is not None:
            retained.add(item.accepted_fit_index)
    for index, fit in enumerate(coefficient_fits):
        if index not in retained and fit.eta is not None:
            coefficient_fits[index] = replace(fit, eta=None, theta=None)
