"""Row-block panels: a predictor's intercept, dense and small categorical groups packed together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numba import njit, types

from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    OrdinaryPanelGroupMatrix,
    RandomEffectGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm._numba_compile import compile_closed
from superglm.group_matrix import DesignMatrix


@njit(cache=True, fastmath=False)
def _write_categorical_block(panel, codes, start, width):
    """Write the complete zero/one block, including every sink/base row.

    The caller checks intp code bounds [0, width], panel bounds and row counts
    before this native writer. Columns outside the block and unused trailing
    rows remain untouched. Every active block entry is written exactly once.
    """
    for row in range(codes.shape[0]):
        code = codes[row]
        for column in range(width):
            panel[row, start + column] = 1.0 if code == column else 0.0


# One signature serves every operand form: categorical codes a row span of
# stored codes, the destination a row block's panel.
_CATEGORICAL_WRITER = (
    _write_categorical_block,
    (
        types.Array(types.float64, 2, "A"),
        types.Array(types.intp, 1, "A", readonly=True),
        types.intp,
        types.intp,
    ),
)


def _warmup_ordinary_packing() -> None:
    """Compile the panel's categorical writer."""
    compile_closed(_CATEGORICAL_WRITER)


# A categorical of at most this many levels joins its predictor's panel as
# one-hot columns; a wider one keeps its codes, whose block with the panel is
# one pass by level.
PANEL_MAX_LEVELS = 32
# Groups whose block with a panel is one pass over the rows: a weighted
# product, or a sum of the panel's rows by bin or level (``_cross_gram_panel``
# and the by-bin route of ``_cross_gram``).
_PANEL_PARTNERS = frozenset(
    (
        DenseGroupMatrix,
        CategoricalGroupMatrix,
        RandomEffectGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SupportCompressedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
    )
)


@dataclass(frozen=True)
class PredictorPanel:
    """How one predictor packs its intercept, dense and small categorical groups into a panel.

    The packed design is the panel, then the predictor's other groups in
    order, with no implicit intercept. ``sources`` are ``(group index, panel
    column)``, index -1 the intercept; ``order[c]`` is the coefficient, within
    the predictor's slice, of packed column ``c``.
    """

    sources: tuple[tuple[int, int], ...]
    width: int
    rest: tuple[int, ...]
    order: np.ndarray

    def panel(self, design: DesignMatrix, start: int, stop: int) -> OrdinaryPanelGroupMatrix:
        """The panel of rows ``[start, stop)``, written once."""
        values = np.empty((stop - start, self.width), dtype=np.float64)
        for index, column in self.sources:
            if index < 0:
                values[:, column] = 1.0
                continue
            group = design.group_matrices[index]
            if type(group) is DenseGroupMatrix:
                values[:, column : column + group.shape[1]] = group.M[start:stop]
            else:
                group = cast(CategoricalGroupMatrix, group)
                _write_categorical_block(values, group.codes[start:stop], column, group.n_levels)
        return OrdinaryPanelGroupMatrix(values)

    def packed_design(self, panel, rest_groups, n_rows: int) -> DesignMatrix:
        width = self.width + sum(group.shape[1] for group in rest_groups)
        return DesignMatrix([panel, *rest_groups], n_rows, width)


def predictor_panels(layout) -> tuple[PredictorPanel | None, ...]:
    """Each predictor's panel: two or more dense or small categorical groups pack.

    None for every predictor when any group of the layout has no one-pass
    block with a panel, or a design is not a plain ``DesignMatrix``: those
    keep their group pairs.
    """
    designs = [state.design for state in layout.predictors]
    if any(
        type(design) is not DesignMatrix
        or (design._tabmat_vector_candidate and design._tabmat_holder.split is not None)
        or any(type(group) not in _PANEL_PARTNERS for group in design.group_matrices)
        for design in designs
    ):
        return (None,) * len(designs)
    panels: list[PredictorPanel | None] = []
    for state, design in zip(layout.predictors, designs, strict=True):
        packed = [
            index
            for index, group in enumerate(design.group_matrices)
            if type(group) is DenseGroupMatrix
            or (type(group) is CategoricalGroupMatrix and group.n_levels <= PANEL_MAX_LEVELS)
        ]
        if len(packed) < 2:
            panels.append(None)
            continue
        intercept = int(state.intercept_index is not None)
        columns = []
        start = intercept
        for group in design.group_matrices:
            columns.append(range(start, start + group.shape[1]))
            start += group.shape[1]
        sources = [(-1, 0)] if intercept else []
        order = [0] if intercept else []
        for index in packed:
            sources.append((index, len(order)))
            order.extend(columns[index])
        width = len(order)
        rest = tuple(index for index in range(len(columns)) if index not in packed)
        for index in rest:
            order.extend(columns[index])
        panels.append(PredictorPanel(tuple(sources), width, rest, np.asarray(order, dtype=np.intp)))
    return tuple(panels)
