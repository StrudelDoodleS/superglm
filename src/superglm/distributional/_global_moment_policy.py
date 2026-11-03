"""Metadata-only scope for automatic streamed support moments."""

from __future__ import annotations

from typing import cast

from superglm.distributional.family import FamilyLikelihoodPlan, _likelihood_reuse_contract
from superglm.distributional.layout import StackedLayout
from superglm.distributional.weights import ResolvedLikelihoodWeights
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import IdentityLink, LogLink

AUTO_GLOBAL_MOMENT_MIN_ROWS = 262144

# The additional global-moment workspace allowance, independent of the row
# chunk selector. Neither budget is a cap on complete-fit/process memory.
AUTO_GLOBAL_MOMENT_BYTES = 64 * 1024 * 1024
AUTO_SMALL_GROUP_MAX_WIDTH = 32

_SPLINES = frozenset(
    (SparseSSPGroupMatrix, DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix)
)
_GROUPED_CURVES = frozenset(
    (
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    )
)
_MIXED_GROUP_TYPES = _SPLINES | _GROUPED_CURVES | {DenseGroupMatrix, CategoricalGroupMatrix}
_SUPPORT_TYPES = (DiscretizedSSPGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)
_ORDINARY_TYPES = (DenseGroupMatrix, CategoricalGroupMatrix)


def automatic_global_moment_budget(
    family: object, likelihood_plan: object, layout: StackedLayout
) -> int | None:
    """Admit the large mixed-layout scope without scanning arrays.

    The scope requires at least 262144 rows and at most one histogram cell to
    initialize per four potential row updates; these bound the workload the
    route was validated on, not a general crossover. Only adapters declaring
    deterministic replay, with their exact bound-plan, resolved-weight and
    link types, may retry the ordinary chunk stream after numerical refusal;
    custom links or weights keep one stream, since replay could change their
    state. The builder separately validates storage, support, domain and the
    64 MiB allowance.
    """
    contract = _likelihood_reuse_contract(family)
    if (
        contract is None
        or not contract.deterministic_chunk_replay
        or type(likelihood_plan) is not contract.plan_type
        or type(cast(FamilyLikelihoodPlan, likelihood_plan).weights)
        is not ResolvedLikelihoodWeights
    ):
        return None
    if not _mixed_small_group_layout(layout) or not 1 <= len(layout.predictors) <= 2:
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
                support = cast(
                    DiscretizedSSPGroupMatrix | DiscretizedSplineCategoricalGroupMatrix, group
                )
                if len(support.B_unique.shape) != 2:
                    return None
                bins, raw_width = support.B_unique.shape
                if not 1 <= bins <= 4096 or not 1 <= raw_width <= 64:
                    return None
                sizes.append(bins)
            elif type(group) not in _ORDINARY_TYPES:
                return None
    count = len(sizes) * (len(sizes) - 1) // 2
    cells = (sum(sizes) ** 2 - sum(size * size for size in sizes)) // 2
    if not count or 4 * cells > n * count:
        return None
    return AUTO_GLOBAL_MOMENT_BYTES


def _mixed_small_group_layout(layout: StackedLayout) -> bool:
    """Whether every predictor holds the mixed small-group layout.

    Every predictor with slopes must contain dense numeric, categorical,
    stored spline and spline-by-category groups, each at most 32 columns;
    intercept-only predictors may accompany it. Tensor, FactorSmooth, other
    sparse formats and custom subclasses keep the per-pass contraction. Reads
    only types and dimensions; the builder remains the authority for
    workspace size, storage and numerical-domain refusal.
    """
    if type(layout) is not StackedLayout:
        return False
    mixed_predictors = 0
    for state in layout.predictors:
        design = state.design
        if type(design) is not DesignMatrix:
            return False
        kinds = set()
        width = 0
        for group in design.group_matrices:
            kind = type(group)
            if kind not in _MIXED_GROUP_TYPES:
                return False
            rows, columns = group.shape
            if rows != design.n or not 0 <= columns <= AUTO_SMALL_GROUP_MAX_WIDTH:
                return False
            width += columns
            if columns:
                kinds.add(kind)
        if width != design.p:
            return False
        if not width:
            continue
        if (
            DenseGroupMatrix not in kinds
            or CategoricalGroupMatrix not in kinds
            or not kinds & _SPLINES
            or not kinds & _GROUPED_CURVES
        ):
            return False
        mixed_predictors += 1
    return mixed_predictors > 0
