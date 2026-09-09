"""Metadata-only admission to bounded ordinary-group curvature panels."""

from __future__ import annotations

from superglm.distributional.layout import StackedLayout
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

# An additional panel workspace allowance, independent of the existing row
# chunk selector. Neither budget is a cap on complete-fit/process memory.
AUTO_SMALL_GROUP_PANEL_BYTES = 64 * 1024 * 1024
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
_ORDINARY = _SPLINES | _GROUPED_CURVES | {DenseGroupMatrix, CategoricalGroupMatrix}


def automatic_small_group_panel_budget(layout: StackedLayout) -> int | None:
    """Admit the initial mixed-layout scope without inspecting row values.

    Every predictor with slopes must contain dense numeric, categorical,
    stored spline and spline-by-category groups, each at most 32 columns.
    Intercept-only predictors may accompany that mix. These are conservative
    scope limits for the measured small-group workload, not speed crossovers.
    Tensor, SCOP, FactorSmooth, other sparse formats and custom subclasses keep
    their existing routes. Explicit panel budgets bypass this admission rule.

    This selector reads only types and dimensions; the builder remains the
    authority for actual workspace size, storage and numerical-domain refusal.
    It constructs no plans, scans no arrays and populates no design caches.
    """
    if type(layout) is not StackedLayout:
        return None
    mixed_predictors = 0
    for state in layout.predictors:
        design = state.design
        if type(design) is not DesignMatrix:
            return None
        kinds = set()
        width = 0
        for group in design.group_matrices:
            kind = type(group)
            if kind not in _ORDINARY:
                return None
            rows, columns = group.shape
            if rows != design.n or not 0 <= columns <= AUTO_SMALL_GROUP_MAX_WIDTH:
                return None
            width += columns
            if columns:
                kinds.add(kind)
        if width != design.p:
            return None
        if not width:
            continue
        if (
            DenseGroupMatrix not in kinds
            or CategoricalGroupMatrix not in kinds
            or not kinds & _SPLINES
            or not kinds & _GROUPED_CURVES
        ):
            return None
        mixed_predictors += 1
    return AUTO_SMALL_GROUP_PANEL_BYTES if mixed_predictors else None
