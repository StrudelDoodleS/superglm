"""A binned layout's design rows, gathered from bin tables formed once per pass.

A binned layout's group rows are lookups (Wood, Li, Shaddick and Augustin 2017,
*JASA* 112:1199): row ``i`` of a support, tensor or spline-by-category level is
row ``k(i)`` of its table ``S = B R`` (a level reads a zero row off its rows), a
categorical or random effect reads the one-hot row of its code, and the
intercept reads ``[1]``; dense groups keep their values. :class:`BinnedRows`
forms each table once, so a pass gathers its row chunks from the tables instead
of re-forming ``B R`` for every chunk.

A pass that needs only products of the rows reads them without forming a row:
``X C`` is a sum of gathers from each term's ``T C``, and the leverage ``x_i' V
y_i`` is a sum of gathers from each term pair's ``T_t V_tu T_u'``, the scattered
form of GLAM's ``diag(X S X')`` (Currie, Durban and Eilers, *JRSS-B* 68:259, 2006,
eq. 2.7), so its cost per row does not depend on the number of columns.
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
from superglm._group_matrix._lookup_tables import gather_channels
from superglm.distributional.layout import PredictorState, StackedLayout
from superglm.distributional.solver.packing import packed_pairs
from superglm.group_matrix import DesignMatrix

_SUPPORTS = (
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
_LEVELS = (DiscretizedSplineCategoricalGroupMatrix, SupportCompressedSplineCategoricalGroupMatrix)
_GROUPS = (*_SUPPORTS, *_LEVELS, *ONE_HOT_CODE_TYPES, DenseGroupMatrix)
# Cells the leverage's term-pair tables may hold together; a layout past this
# (a wide random effect) keeps the row products.
_LEVERAGE_MAX_CELLS = 1 << 26


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


def _table_codes(term: _Term, start: int, stop: int) -> NDArray[np.intp]:
    """The rows of the term's lookup table that rows ``[start, stop)`` read."""
    codes = _codes(term, start, stop)
    if type(term.group) in ONE_HOT_CODE_TYPES:
        return np.minimum(codes, term.columns.stop - term.columns.start)
    return codes


def _lookup_table(term: _Term) -> NDArray[np.float64]:
    """The table the term's rows read: its ``S = B R``, or a categorical's identity over a zero row."""
    if type(term.group) in ONE_HOT_CODE_TYPES:
        width = term.columns.stop - term.columns.start
        return np.vstack((np.eye(width), np.zeros((1, width))))
    return term.table


class RowProducts:
    """``X_q[rows] @ coefficients`` as a call on a row slice, each term's product formed once."""

    __slots__ = ("_dense", "_gathers", "_width", "n")

    def __init__(self, terms: tuple[_Term, ...], coefficients: NDArray, n: int) -> None:
        self._gathers, self._dense = [], []
        for term in terms:
            local = coefficients[term.columns]
            if type(term.group) is DenseGroupMatrix:
                self._dense.append((term.group.M, local))
            elif type(term.group) in ONE_HOT_CODE_TYPES:
                # A categorical's base level reads its code past the last column: a zero row.
                self._gathers.append((term, np.vstack((local, np.zeros((1, local.shape[1]))))))
            else:
                self._gathers.append((term, term.table @ local))
        self._width = coefficients.shape[1]
        self.n = n

    def __call__(self, rows: slice) -> NDArray[np.float64]:
        start, stop, _ = rows.indices(self.n)
        out = np.zeros((stop - start, self._width), dtype=np.float64)
        for term, values in self._gathers:
            out += values[_table_codes(term, start, stop)]
        for matrix, local in self._dense:
            out += matrix[start:stop] @ local
        return out


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

    def row_products(self, q: int, coefficients: NDArray) -> RowProducts:
        """``X_q[rows] @ coefficients`` for ``coefficients`` of shape ``(P_q, m)``."""
        return RowProducts(self._terms[q], coefficients, self.n)

    def leverages(self, inverse: NDArray, slices: tuple[slice, ...]) -> NDArray | None:
        """``G_i = X_i V X_i'`` for every row, packed ``(n, K(K+1)/2)``; None without a lookup.

        ``G_i[q, r] = sum_t sum_u M_tu[c_t(i), c_u(i)]`` with ``M_tu = T_t V_tu
        T_u'``, ``t`` over predictor ``q``'s terms and ``u`` over ``r``'s; a
        diagonal channel reads each unordered pair once at twice its table. A
        dense term or more than ``_LEVERAGE_MAX_CELLS`` table cells return None,
        and the caller forms the rows.
        """
        terms = [(q, term) for q, predictor in enumerate(self._terms) for term in predictor]
        if any(type(term.group) is DenseGroupMatrix for _, term in terms):
            return None
        tables = [_lookup_table(term) for _, term in terms]
        sizes = [table.shape[0] for table in tables]
        if (sum(sizes) ** 2 + sum(size**2 for size in sizes)) // 2 > _LEVERAGE_MAX_CELLS:
            return None
        channels = {pair: index for index, pair in enumerate(packed_pairs(len(self._terms)))}
        flat = np.empty((sum(sizes) ** 2 + sum(size**2 for size in sizes)) // 2, dtype=np.float64)
        pairs, offset = [], 0
        for t, ((q, term_t), table_t) in enumerate(zip(terms, tables, strict=True)):
            columns_t = _shifted(term_t.columns, slices[q])
            for u in range(t, len(terms)):
                r, term_u = terms[u]
                block = table_t @ inverse[columns_t, _shifted(term_u.columns, slices[r])]
                block = block @ tables[u].T
                flat[offset : offset + block.size] = (
                    2.0 * block if q == r and t != u else block
                ).ravel()
                pairs.append((channels[(q, r)], t, u, offset, sizes[u]))
                offset += block.size
        codes = np.empty((len(terms), self.n), dtype=np.int32)
        for t, (_, term) in enumerate(terms):
            codes[t] = _table_codes(term, 0, self.n)
        return gather_channels(codes, np.array(pairs, dtype=np.int64), flat, len(channels))


def _shifted(columns: slice, block: slice) -> slice:
    """A term's columns of its predictor's block, in the coefficients of the whole layout."""
    return slice(block.start + columns.start, block.start + columns.stop)


def binned_rows(layout: StackedLayout) -> BinnedRows | None:
    """The layout's :class:`BinnedRows`, or None where it is not binned."""
    return BinnedRows(layout) if binned_layout(layout) else None
