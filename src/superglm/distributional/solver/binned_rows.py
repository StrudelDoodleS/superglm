"""A binned layout's design rows, gathered from bin tables formed once per pass.

A binned layout's group rows are lookups (Wood, Li, Shaddick and Augustin 2017,
*JASA* 112:1199): row ``i`` of a support, tensor or spline-by-category level is
row ``k(i)`` of its table ``S = B R`` (a level reads a zero row off its rows), a
categorical or random effect reads the one-hot row of its code, and the
intercept reads ``[1]``; dense groups keep their values. :class:`BinnedRows`
forms each table once, so a pass gathers its row chunks from the tables instead
of re-forming ``B R`` for every chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES, DenseGroupMatrix
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_range import _category_rows
from superglm.distributional.layout import PredictorState, StackedLayout
from superglm.group_matrix import DesignMatrix

_SUPPORTS = (
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
_LEVELS = (DiscretizedSplineCategoricalGroupMatrix, SupportCompressedSplineCategoricalGroupMatrix)
_GROUPS = (*_SUPPORTS, *_LEVELS, *ONE_HOT_CODE_TYPES, DenseGroupMatrix)


def binned_layout(layout: StackedLayout) -> bool:
    """Whether every group reads its rows as a lookup or dense values, and one is binned.

    The tables are the product ``B R`` the dense rows form, on the same operands,
    so any range or dtype a support has reads the same rows either way; any
    other group or design type leaves the layout to its ordinary row chunks.
    """
    if any(type(state.design) is not DesignMatrix for state in layout.predictors):
        return False
    groups = [group for state in layout.predictors for group in state.design.group_matrices]
    return all(type(group) in _GROUPS for group in groups) and any(
        type(group) in (*_SUPPORTS, *_LEVELS) for group in groups
    )


@dataclass(frozen=True, eq=False)
class _Term:
    """One group's rows (``group`` None: the intercept) on its predictor ``columns``.

    ``table`` holds the rows the codes read (``S = B R``, a level's with a zero
    row last); a one-hot term reads its code's column and keeps no table, a
    dense term reads its values.
    """

    columns: slice
    group: object
    table: NDArray | None


def _terms(state: PredictorState) -> tuple[_Term, ...]:
    """A predictor's intercept and groups, each on its columns of the predictor's block."""
    column = 0
    terms = []
    if state.intercept_index is not None:
        terms.append(_Term(slice(0, 1), None, np.ones((1, 1))))
        column = 1
    for group in state.design.group_matrices:
        width = group.shape[1]
        table = None
        if type(group) in (*_SUPPORTS, *_LEVELS):
            table = group.B_unique @ group.R_inv
            if type(group) in _LEVELS:
                table = np.vstack((table, np.zeros((1, width))))
        if width:
            terms.append(_Term(slice(column, column + width), group, table))
        column += width
    return tuple(terms)


def _codes(term: _Term, start: int, stop: int) -> NDArray[np.intp]:
    """The table rows (one-hot: columns) that rows ``[start, stop)`` read."""
    group = term.group
    if group is None:
        return np.zeros(stop - start, dtype=np.intp)
    if type(group) in _SUPPORTS:
        return group.bin_idx[start:stop]
    if type(group) in ONE_HOT_CODE_TYPES:
        return group.codes[start:stop]
    # A level's rows off its level read the table's last, zero row.
    codes = np.full(stop - start, group.n_bins, dtype=np.intp)
    located = _category_rows(group, start, stop)
    if located is None:
        sub = group.row_subset(np.arange(start, stop, dtype=np.intp))
        located = sub.row_idx, sub.bin_idx_level
    codes[located[0]] = located[1]
    return codes


class BinnedRows:
    """A binned layout's design rows, read from each term's table formed once.

    ``rows(q, start, stop)`` is ``X_q[start:stop]`` as dense C-order float64.
    Owner: the pass that builds it, for that pass; the tables depend only on the
    layout's fixed bases, so nothing invalidates them while it lives. Build it
    with :func:`binned_rows`, which declines a layout that is not binned. A
    non-finite table or dense group raises as the dense rows' design check does.
    """

    __slots__ = ("_terms", "_widths", "n")

    def __init__(self, layout: StackedLayout) -> None:
        if not binned_layout(layout):
            raise ValueError("BinnedRows needs a binned layout; see binned_rows")
        self._terms = tuple(_terms(state) for state in layout.predictors)
        for state, terms in zip(layout.predictors, self._terms, strict=True):
            values = (
                term.group.M if type(term.group) is DenseGroupMatrix else term.table
                for term in terms
                if term.table is not None or type(term.group) is DenseGroupMatrix
            )
            if not all(np.all(np.isfinite(value)) for value in values):
                raise ValueError(f"design for predictor {state.name!r} must be finite")
        self._widths = tuple(
            state.coefficient_slice.stop - state.coefficient_slice.start
            for state in layout.predictors
        )
        self.n = layout.predictors[0].design.n

    def rows(self, q: int, start: int, stop: int) -> NDArray[np.float64]:
        """Predictor ``q``'s design rows ``[start, stop)``."""
        if not 0 <= start <= stop <= self.n:
            raise ValueError("rows must lie inside the design")
        out = np.zeros((stop - start, self._widths[q]), dtype=np.float64)
        for term in self._terms[q]:
            if type(term.group) is DenseGroupMatrix:
                out[:, term.columns] = term.group.M[start:stop]
            elif type(term.group) in ONE_HOT_CODE_TYPES:
                # A categorical's base level reads its code past the last column: a zero row.
                codes = _codes(term, start, stop)
                hit = np.flatnonzero(codes < term.columns.stop - term.columns.start)
                out[hit, term.columns.start + codes[hit]] = 1.0
            else:
                out[:, term.columns] = term.table[_codes(term, start, stop)]
        return out


def binned_rows(layout: StackedLayout) -> BinnedRows | None:
    """The layout's :class:`BinnedRows`, or None where it is not binned."""
    return BinnedRows(layout) if binned_layout(layout) else None
