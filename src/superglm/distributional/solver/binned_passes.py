"""Row passes of a binned layout read through coefficient-space tables: ``X D``, ``x' V x``, ``X'WX``.

A binned layout's group rows are lookups: row ``i`` of a support, tensor or
spline-by-category level is row ``k(i)`` of its table ``S = B R`` (a level
reads a zero row off its rows), a categorical or random effect reads the
one-hot row of its code, and the intercept reads the one-row table ``[1]``;
dense groups keep their values. Each pass works in coefficient space and
reads rows only through their codes (Wood, Li, Shaddick and Augustin 2017,
*JASA* 112:1199; Li and Wood 2020, *Stat. Comput.* 30:19):

- :func:`design_dot_rows`, ``X_q[rows] D``: each term's ``S D`` read at its codes.
- :func:`binned_leverage_blocks`, ``G_i = x_l,i' V_lr x_r,i``: each term pair's
  table ``S_t V_tu S_u'`` read at the pair's codes, the scattered-row form of
  ``diag(X V X')`` (``_lookup_tables``); a pair whose table is too large or
  dearer than its rows reads ``Q = S_t V_tu`` at ``k_t`` against ``S_u``'s row,
  as Li and Wood's Algorithms 2/3 stand to their Algorithm 1.
- :func:`binned_cross_grams`, ``X' diag-blocks(w_k) X`` for each smoothing
  component ``k``: the binned SuperLSS assembly (``GroupedGeometryAccumulator``
  with its pass tables and range rules) once per component over shared blocks.

Each returns ``None`` for a layout that is not binned (:func:`binned_layout`):
its callers keep their dense row chunks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import PassTables
from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES, DenseGroupMatrix
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_kernels import _ssp_gram_needs_exact
from superglm._group_matrix._group_matrix_range import _category_rows
from superglm._group_matrix._lookup_tables import _CELL_CAP, _P_SPACE_WEIGHT
from superglm.distributional.layout import PredictorState, StackedLayout
from superglm.distributional.solver._ordinary_packing import predictor_panels
from superglm.distributional.solver.assembly import GroupedGeometryAccumulator
from superglm.distributional.solver.chunks import (
    _BLOCK_SUM_LIMIT,
    AUTO_CONTRACTION_MAX_ROWS,
    AUTO_CONTRACTION_MEMORY_BYTES,
    _channel_peak,
    block_plans,
    contraction_block_rows,
    contraction_row_bytes,
)
from superglm.distributional.solver.packing import packed_pairs
from superglm.group_matrix import DesignMatrix

_SUPPORTS = (
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
_LEVELS = (DiscretizedSplineCategoricalGroupMatrix, SupportCompressedSplineCategoricalGroupMatrix)
_GROUPS = (*_SUPPORTS, *_LEVELS, *ONE_HOT_CODE_TYPES, DenseGroupMatrix)
# Rows per block of the leverage gather; its tables take at most the contraction allowance.
_LEVERAGE_ROWS = 1 << 14


def binned_layout(layout: StackedLayout) -> bool:
    """Whether every group reads its rows as a lookup or dense values, and one is binned.

    A support or level must also keep ``B R`` in the range the binned routes
    reassociate in (``_ssp_gram_needs_exact``); any other group, design type
    or factor leaves the layout to the dense row chunks.
    """
    if any(type(state.design) is not DesignMatrix for state in layout.predictors):
        return False
    groups = [group for state in layout.predictors for group in state.design.group_matrices]
    if any(type(group) not in _GROUPS for group in groups):
        return False
    binned = [group for group in groups if type(group) in (*_SUPPORTS, *_LEVELS)]
    return bool(binned) and not any(
        group.B_unique.dtype != np.float64
        or group.R_inv.dtype != np.float64
        or _ssp_gram_needs_exact(group.B_unique, group.R_inv)
        for group in binned
    )


@dataclass(frozen=True, eq=False)
class _Term:
    """One group's rows (``group`` None: the intercept) on its coefficient ``columns``.

    ``table`` holds the rows the codes read (``S = B R``, a level's with a zero
    row last); a one-hot term reads the identity over a zero row and keeps no
    table, a dense term reads its values.
    """

    columns: slice
    group: object
    table: NDArray | None

    @property
    def one_hot(self) -> bool:
        return type(self.group) in ONE_HOT_CODE_TYPES

    @property
    def dense(self) -> bool:
        return type(self.group) is DenseGroupMatrix


def _terms(state: PredictorState, start: int) -> list[_Term]:
    """A predictor's intercept and groups, each on its columns offset by ``start``."""
    column = start
    terms = []
    if state.intercept_index is not None:
        terms.append(_Term(slice(column, column + 1), None, np.ones((1, 1))))
        column += 1
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
    return terms


def _codes(term: _Term, start: int, stop: int) -> NDArray[np.intp]:
    """The table rows that rows ``[start, stop)`` read (a level's last row off its rows)."""
    group = term.group
    if group is None:
        return np.zeros(stop - start, dtype=np.intp)
    if type(group) in _SUPPORTS:
        return group.bin_idx[start:stop]
    if term.one_hot:
        return group.codes[start:stop]
    codes = np.full(stop - start, group.n_bins, dtype=np.intp)
    located = _category_rows(group, start, stop)
    if located is None:
        sub = group.row_subset(np.arange(start, stop, dtype=np.intp))
        located = sub.row_idx, sub.bin_idx_level
    codes[located[0]] = located[1]
    return codes


def _left(term: _Term, M: NDArray) -> NDArray:
    """``S M`` for the term's table; a one-hot term's is ``M`` over a zero row."""
    if term.one_hot:
        return np.vstack((M, np.zeros((1, M.shape[1]))))
    return term.table @ M


def design_dot_rows(
    layout: StackedLayout, q: int, D: NDArray, start: int, stop: int
) -> NDArray[np.float64] | None:
    """``X_q[start:stop] @ D`` for ``D`` of shape ``(p_q, m)``, or None off a binned layout."""
    if not binned_layout(layout):
        return None
    state = layout.predictors[q]
    block = state.coefficient_slice
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != block.stop - block.start:
        raise ValueError("D must have one row per coefficient of the predictor")
    if not 0 <= start <= stop <= state.design.n:
        raise ValueError("rows must lie inside the design")
    out = np.zeros((stop - start, D.shape[1]), dtype=np.float64)
    for term in _terms(state, 0):
        part = D[term.columns]
        if term.dense:
            out += np.asarray(term.group.M[start:stop], dtype=np.float64) @ part
        else:
            out += _left(term, part)[_codes(term, start, stop)]
    return out


def _pair_reader(t: _Term, u: _Term, V: NDArray, n: int, budget: list[int]) -> Callable:
    """``(codes, values) -> x_t' V x_u`` over a block, for one term pair.

    A lookup pair reads ``S_t V S_u'`` at its codes where the table fits
    ``_CELL_CAP`` and the remaining ``budget`` and is cheaper than reading
    ``S_t V`` against the narrower side's rows (``_choose_pairs``' cost
    model); a one-hot side always takes the table that fits, and is never read
    as a row. A dense side reads its values against the other side's ``S V``.
    """
    if t.dense and u.dense:
        return lambda codes, values: np.einsum("ij,ij->i", values[t] @ V, values[u])
    if t.dense:
        t, u, V = u, t, V.T
    if u.dense:
        F = _left(t, V)
        return lambda codes, values: np.einsum("ij,ij->i", F[codes[t]], values[u])
    (p_t, p_u), hot = V.shape, t.one_hot or u.one_hot
    m_t = p_t + 1 if t.one_hot else t.table.shape[0]
    m_u = p_u + 1 if u.one_hot else u.table.shape[0]
    cells = m_t * m_u
    left_first = m_t * p_t * p_u + m_t * p_u * m_u
    right_first = m_u * p_u * p_t + m_u * p_t * m_t
    table_cost = n + _P_SPACE_WEIGHT * min(left_first, right_first)
    rows_cost = n * min(p_t, p_u) + _P_SPACE_WEIGHT * max(m_t, m_u) * p_t * p_u
    fits = cells <= _CELL_CAP and 8 * cells <= budget[0]
    if (t.one_hot and u.one_hot) or (fits and (hot or table_cost <= rows_cost)):
        budget[0] -= 8 * cells
        if left_first <= right_first:
            G = _left(u, _left(t, V).T).T
        else:
            G = _left(t, _left(u, V.T).T)
        flat = np.ascontiguousarray(G).ravel()
        return lambda codes, values: flat[codes[t] * m_u + codes[u]]
    if u.one_hot or (not t.one_hot and p_t < p_u):
        t, u, V = u, t, V.T
    Q = _left(t, V)
    return lambda codes, values: np.einsum("ij,ij->i", Q[codes[t]], u.table[codes[u]])


def _diagonal_reader(t: _Term, V: NDArray) -> Callable:
    """``rows -> x_t' V x_t`` for one term: its table's diagonal read at its codes."""
    if t.dense:
        return lambda codes, values: np.einsum("ij,ij->i", values[t] @ V, values[t])
    if t.one_hot:
        e = np.append(np.diag(V), 0.0)
    else:
        e = np.einsum("ij,ij->i", t.table @ V, t.table)
    return lambda codes, values: e[codes[t]]


def binned_leverage_blocks(layout: StackedLayout, inverse: NDArray) -> NDArray[np.float64] | None:
    """``G_i = X_i H^-1 X_i'`` packed ``(n, K(K+1)/2)``, or None off a binned layout.

    The same output as ``_leverage_blocks`` on the layout's predictor matrices:
    channel ``(l, r)`` holds ``x_l,i' inverse[l, r] x_r,i``.
    """
    if not binned_layout(layout):
        return None
    inverse = np.asarray(inverse, dtype=np.float64)
    width = layout.n_coefficients
    if inverse.shape != (width, width):
        raise ValueError("inverse must match the layout width")
    n = layout.predictors[0].design.n
    terms = [_terms(state, state.coefficient_slice.start) for state in layout.predictors]
    budget = [AUTO_CONTRACTION_MEMORY_BYTES]
    channels = []
    for left, right in packed_pairs(len(terms)):
        readers = []
        for i, t in enumerate(terms[left]):
            for j, u in enumerate(terms[right]):
                V = inverse[t.columns, u.columns]
                if left != right:
                    readers.append(_pair_reader(t, u, V, n, budget))
                elif i == j:
                    readers.append(_diagonal_reader(t, V))
                elif i < j:
                    readers.append(_pair_reader(t, u, 2.0 * V, n, budget))
        channels.append(readers)
    every = [term for predictor in terms for term in predictor]
    out = np.zeros((n, len(channels)), dtype=np.float64)
    for start in range(0, n, _LEVERAGE_ROWS):
        stop = min(start + _LEVERAGE_ROWS, n)
        codes = {t: _codes(t, start, stop) for t in every if not t.dense}
        values = {t: np.asarray(t.group.M[start:stop], dtype=np.float64) for t in every if t.dense}
        for channel, readers in enumerate(channels):
            for reader in readers:
                out[start:stop, channel] += reader(codes, values)
    return out if np.all(np.isfinite(out)) else None


def binned_cross_grams(
    layout: StackedLayout,
    packed_rows: Callable[[int, int], NDArray],
    width: int,
) -> NDArray[np.float64] | None:
    """``X' diag-blocks(packed[:, k, :]) X`` for every ``k``, ``(m, P, P)``; None off a binned layout.

    The same output as ``_cross_all``. ``packed_rows(start, stop)`` gives one
    row block's ``(stop - start, m, K(K+1)/2)`` weights and is called once per
    block, in row order. Blocks hold the assembly's rows and the producer's
    weights within the contraction allowance; each component runs the binned
    assembly, its pass tables taking an equal share of what the block leaves.
    """
    if not binned_layout(layout):
        return None
    if width != layout.n_coefficients:
        raise ValueError("width must match the layout")
    n = layout.predictors[0].design.n
    k_parameters = len(layout.predictors)
    channels = k_parameters * (k_parameters + 1) // 2
    panels = predictor_panels(layout)
    row_bytes = contraction_row_bytes(layout, panels) or 8 * (width + channels)
    rows = contraction_block_rows(n, row_bytes, 1)
    if rows >= n:
        panels = (None,) * k_parameters
    score = np.zeros((rows, k_parameters), dtype=np.float64)
    accumulators: list[GroupedGeometryAccumulator] | None = None
    summed: list[list[float]] = []
    start = 0
    while start < n:
        stop = min(start + rows, n)
        packed = np.asarray(packed_rows(start, stop), dtype=np.float64)
        count = packed.shape[1] if packed.ndim == 3 else -1
        if packed.shape != (stop - start, count, channels) or (
            accumulators is not None and count != len(accumulators)
        ):
            raise ValueError("packed_rows must give (rows, m, K(K+1)/2) weights for its rows")
        if accumulators is None:
            block_bytes = row_bytes + 8 * count * channels
            rows = max(
                1,
                min(rows, AUTO_CONTRACTION_MAX_ROWS, AUTO_CONTRACTION_MEMORY_BYTES // block_bytes),
            )
            share = (AUTO_CONTRACTION_MEMORY_BYTES - block_bytes * rows) // max(count, 1)
            several = stop < n and share > 0
            zero = np.zeros(width, dtype=np.float64)
            accumulators = [
                GroupedGeometryAccumulator(
                    layout,
                    penalty=np.zeros((width, width)),
                    coefficients=zero,
                    tables=PassTables(share) if several else None,
                    panels=panels,
                )
                for _ in range(count)
            ]
            summed = [[0, 0.0] for _ in range(count)]
        plans = block_plans(layout, panels, start, stop)
        for k, accumulator in enumerate(accumulators):
            weights = np.ascontiguousarray(packed[:, k, :].T)
            # A table entry sums at most the rows seen at their peak: release before it could overflow.
            peak = max(summed[k][1], _channel_peak(weights))
            if (summed[k][0] + stop - start) * peak > _BLOCK_SUM_LIMIT:
                accumulator.release_tables()
            summed[k] = [summed[k][0] + stop - start, peak]
            accumulator.add_block(plans, score[: stop - start], weights)
        start = stop
    if not accumulators:
        return np.zeros((0, width, width), dtype=np.float64)
    return np.stack([accumulator.finish().data_curvature for accumulator in accumulators])
