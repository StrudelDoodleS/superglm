"""Metadata-only scope for automatic streamed support moments."""

from __future__ import annotations

from superglm.distributional._panel_policy import automatic_small_group_panel_budget
from superglm.distributional.family import _likelihood_reuse_contract
from superglm.distributional.layout import StackedLayout
from superglm.distributional.weights import ResolvedLikelihoodWeights
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
)
from superglm.links import IdentityLink, LogLink

AUTO_GLOBAL_MOMENT_MIN_ROWS = 262144
_SUPPORT_TYPES = (DiscretizedSSPGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)
_ORDINARY_TYPES = (DenseGroupMatrix, CategoricalGroupMatrix)


def automatic_global_moment_budget(
    family: object, likelihood_plan: object, layout: StackedLayout
) -> int | None:
    """Admit the measured large mixed-layout scope without scanning arrays.

    The initial scope requires at least 262144 rows and at most one histogram
    cell to initialize per four potential row updates. These limits describe
    the measured workload; they are not a general performance crossover.
    Only adapters declaring deterministic replay, their exact bound-plan and
    resolved-weight types, and audited exact link types may retry the ordinary
    chunk stream after numerical refusal. Custom links or weights keep one
    ordinary stream because replay could change their state and thus the
    likelihood being evaluated.

    The builder independently validates storage, support authority, numerical
    domain and the additional 64 MiB allowance. Lossless CSR/support subclasses
    retain automatic panels until their global representation is supported.
    """
    contract = _likelihood_reuse_contract(family)
    if (
        contract is None
        or not contract.deterministic_chunk_replay
        or type(likelihood_plan) is not contract.plan_type
        or type(likelihood_plan.weights) is not ResolvedLikelihoodWeights
    ):
        return None
    budget = automatic_small_group_panel_budget(layout)
    if budget is None or not 1 <= len(layout.predictors) <= 2:
        return None
    n = layout.predictors[0].design.n
    if n < AUTO_GLOBAL_MOMENT_MIN_ROWS:
        return None
    sizes = []
    for state in layout.predictors:
        if type(state.link) not in (IdentityLink, LogLink, *contract.link_types):
            return None
        if state.design.n != n or len(state.design.group_matrices) > 64:
            return None
        for group in state.design.group_matrices:
            if type(group) in _SUPPORT_TYPES:
                # Dimensions only: no row gathers, equality scans or plans.
                if len(group.B_unique.shape) != 2:
                    return None
                bins, raw_width = group.B_unique.shape
                if not 1 <= bins <= 4096 or not 1 <= raw_width <= 64:
                    return None
                sizes.append(bins)
            elif type(group) not in _ORDINARY_TYPES:
                return None
    count = len(sizes) * (len(sizes) - 1) // 2
    cells = (sum(sizes) ** 2 - sum(size * size for size in sizes)) // 2
    if not count or 4 * cells > n * count:
        return None
    return budget
