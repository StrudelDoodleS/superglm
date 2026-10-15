"""Leverages ``h_r = x_r' V x_r`` read back from per-term-pair tables, and ``X_c' v``.

The exact REML gradient's weight-derivative term needs, for every row of the
centred design, the leverage ``h_r = x_r' H^-1 x_r`` (Wood 2011, sec. 3.5.1).
When every group of the design is a *lookup* -- its row ``r`` is row
``k_t(r)`` of a small table ``A_t`` (a categorical's identity, a binned or
support-compressed spline's ``B_unique R_inv``, a tensor's occupied cells, a
spline-by-category level) -- the leverage is a sum over term pairs::

    h_r = sum_{t,u} A_t[k_t(r)] V_tu A_u[k_u(r)]' = sum_{t<=u} G_tu[k_t(r), k_u(r)],
    G_tu = A_t V_tu A_u'    (twice that for t < u).

Each ``G_tu`` is ``m_t x m_u`` and is formed once in coefficient space, so a
row costs one gather per term pair instead of a ``p``-wide solved row.  This
is the transpose of the weighted cross product ``A_t' Wbar_tu A_u`` that Li &
Wood (*Stat. Comput.* 30:19, 2020, Algorithm 1) form from an ``m_t x m_u``
weight table, and the scattered-row form of GLAM's ``diag(X S X')``
(Currie, Durban & Eilers, *JRSS-B* 68:259, 2006, eq. 2.7).  Wood, Li,
Shaddick & Augustin (*JASA* 112:1199, 2017, App. A) compute ``diag(X V X')``
column by column, ``sum_i (X V_{:,i}) * X_{:,i}``, at ``O(n)`` gathers per
coefficient.  As in Li & Wood (Algorithms 2/3), a pair whose table would be
too large is *accumulated* instead: ``Q_tu = A_t V_tu`` (``m_t x p_u``) and
the row adds ``Q_tu[k_t(r)] . A_u[k_u(r)]`` over the nonzero window of
``A_u``'s row.

**Centring.**  The REML rows are centred, ``x_r = (raw_r - c) - c_lo``.  As
on the support route (``_leverage_rows``), a group every row of which reads a
real table row (a spline, a tensor) has its table centred before any product,
``A_t = (values - c_t) - c_lo,t``; a group with a zero row (a categorical's
base level, the rows outside a spline-by-category level) keeps its raw table
and its centre ``z_t`` enters as constants: with ``a_r`` the concatenated
table rows, ``x_r = a_r - z + d_r`` (``d_r`` the centred dense columns), and
``-2 a_r' V z + z' V z`` folds into each term's diagonal table ``e_t`` and
one constant ``h0``.  The levels of one spline-by-category term have disjoint
rows, so consecutive level groups with disjoint rows merge into one term
whose table stacks the levels' rows (``k = `` the level's block plus the bin,
or the shared zero row).

**Dense columns** (``DenseGroupMatrix``, not lookups) are read per row as on
the support route: the block's centred values ``d_r`` (formed exactly as
``_leverage_rows`` forms them) times per-term tables ``F_t = 2 A_t V_{t,D}``
and the dense block ``V_DD``.

**Error.**  Write ``|.|`` entrywise and ``eta_r = (|a_r| + |z| + |d_r|)'
|V| (|a_r| + |z| + |d_r|)``.  Every table entry is a bilinear form in rows of
length at most ``p`` read through at most ``3p + 1`` roundings (two inner
products, the ``V z`` product and one difference; Higham 2002, secs. 3.1,
3.5), and a row then sums ``S_r <= 1 + T + P + n_d (T + n_d + 3)`` computed
terms (``T`` terms, ``P`` pairs, ``n_d`` dense columns), so
``|h_r(computed) - h_r| <= gamma_K eta_r`` with ``K = 3p + S + 1`` (Higham
Lemma 3.3; :meth:`TablePlan.count_roundings`).  ``gamma_K = K u / (1 - K
u)``, ``u = 2**-53``.

**Determinism.**  A row's sum runs in one fixed order (terms, table pairs,
accumulated pairs, dense columns) inside one task, the rows are split into
fixed blocks whatever the worker count, and each block writes only its rows,
so ``h`` is bitwise identical for any number of workers.  The scatter
``X_c' v`` runs serially in row order with a compensated (Neumaier) sum per
bin (Higham 2002, sec. 4.3): within ``2u |s| + O(n u^2) sum |terms|``.

**Ownership.**  A :class:`TablePlan` (codes, tables, windows) is built for
one REML iteration's centre by :func:`table_plan`, read with that
iteration's inverse, and dropped when its weight correction returns; nothing
is cached.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.group_matrix import DesignMatrix

# Fixed row blocks: the partition never depends on the worker count.
_BLOCK_ROWS = 8192
# A pair table larger than this many cells is accumulated instead.
_CELL_CAP = 1 << 22
# Relative cost of one coefficient-space multiply-add (BLAS, in cache) against
# one random row gather (prototype measurement, solver-scaling report sec. 7).
_P_SPACE_WEIGHT = 0.1


@dataclass(frozen=True)
class _Term:
    """One lookup term: rows ``table[codes]`` on ``columns``; ``zero`` its centre if raw."""

    columns: NDArray  # (p_t,) global coefficient columns
    table: NDArray  # (m_t, p_t) float64
    zero: NDArray | None  # (p_t,) the centre on these columns when the table is raw


@dataclass(frozen=True)
class TablePlan:
    """The read-back's index and tables (owner: one weight-correction call)."""

    n: int
    p: int
    terms: tuple
    codes: NDArray  # (n, T) int32
    dense: tuple  # (column, width, values) per DenseGroupMatrix
    dense_columns: NDArray  # (n_d,) intp
    tab_pairs: NDArray  # (P1, 2) term indices of pairs read from a full table
    acc_pairs: NDArray  # (P2, 2) (row term, window term) of accumulated pairs

    @property
    def n_pairs(self) -> int:
        return int(self.tab_pairs.shape[0] + self.acc_pairs.shape[0])

    def count_roundings(self) -> int:
        """``K = 3p + S + 1``, the roundings along one leverage's longest chain (module docstring)."""
        t, n_d = len(self.terms), int(self.dense_columns.size)
        return 3 * self.p + 1 + (1 + t + self.n_pairs + n_d * (t + n_d + 3)) + 1


def _lookup_parts(gm):
    """``(values, transform, codes, level)`` for a lookup group, else ``None``."""
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
        return gm.B_unique, gm.R_inv, None, (gm.row_idx, gm.bin_idx_level)
    if isinstance(gm, DiscretizedSSPGroupMatrix):
        return gm.B_unique, gm.R_inv, gm.bin_idx, None
    if isinstance(gm, DiscretizedSCOPGroupMatrix):
        return gm.B_scop_unique, None, gm.bin_idx, None
    if type(gm) is CategoricalGroupMatrix:
        return None, None, gm.codes, None
    return None


def _windows(table: NDArray) -> tuple[NDArray, NDArray]:
    """Each table row's nonzero column span ``[lo, hi)``."""
    nonzero = table != 0.0
    any_ = nonzero.any(axis=1)
    lo = np.where(any_, nonzero.argmax(axis=1), 0)
    hi = np.where(any_, table.shape[1] - nonzero[:, ::-1].argmax(axis=1), 0)
    return lo.astype(np.int32), hi.astype(np.int32)


def table_plan(
    dm: DesignMatrix,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
) -> TablePlan | None:
    """The read-back's terms for a design of lookups and dense columns, else ``None``.

    ``None`` (the solved-row route) when a group is neither a lookup nor a
    ``DenseGroupMatrix``, when no group is a lookup, when a table does not
    read as float64, or when the index and tables would not fit a quarter of
    ``max_memory``.
    """
    from superglm._group_matrix._group_matrix_centered import _reads_as_float64
    from superglm._parallel import resolve_max_memory
    from superglm.group_matrix import DenseGroupMatrix

    if type(dm) is not DesignMatrix:
        return None
    parts = []
    for gm in dm.group_matrices:
        if type(gm) is DenseGroupMatrix:
            parts.append(None)
            continue
        found = _lookup_parts(gm)
        if found is None:
            return None
        if any(
            operand is not None and not _reads_as_float64(np.asarray(operand).dtype)
            for operand in found[:2]
        ):
            return None
        parts.append(found)
    if all(part is None for part in parts):
        return None
    n = dm.n
    table_rows = sum(
        (np.asarray(part[0]).shape[0] if part[0] is not None else gm.shape[1]) + 1
        for part, gm in zip(parts, dm.group_matrices, strict=True)
        if part is not None
    )
    # The index, and the term tables with their dense-column tables.
    if 4 * n * len(parts) + 8 * table_rows * dm.p > resolve_max_memory() // 4:
        return None

    centre = np.asarray(centre_hi, dtype=np.float64)
    low = None if centre_lo is None else np.asarray(centre_lo, dtype=np.float64)

    def zero_of(columns: NDArray) -> NDArray:
        zero = centre[columns].copy()
        if low is not None:
            zero += low[columns]
        return zero

    terms: list[_Term] = []
    term_codes: list[NDArray] = []
    dense: list[tuple[int, int, NDArray]] = []
    run: dict | None = None  # an open spline-by-category run of disjoint levels

    def close_run() -> None:
        nonlocal run
        if run is None:
            return
        tables = run["tables"]
        stacked = np.zeros(
            (sum(t.shape[0] for t in tables) + 1, sum(t.shape[1] for t in tables)),
            dtype=np.float64,
        )
        r = c = 0
        for table in tables:
            stacked[r : r + table.shape[0], c : c + table.shape[1]] = table
            r += table.shape[0]
            c += table.shape[1]
        codes = run["codes"]
        codes[codes < 0] = stacked.shape[0] - 1
        columns = np.concatenate(run["columns"])
        terms.append(_Term(columns, stacked, zero_of(columns)))
        term_codes.append(codes)
        run = None

    column = 0
    for gm, part in zip(dm.group_matrices, parts, strict=True):
        width = int(gm.shape[1])
        columns = np.arange(column, column + width, dtype=np.intp)
        column += width
        if part is None:
            close_run()
            dense.append((int(columns[0]), width, gm.M))
            continue
        values, transform, codes, level = part
        if values is None:  # categorical: the identity, its base level the zero row
            raw = np.vstack((np.eye(width, dtype=np.float64), np.zeros((1, width))))
        else:
            raw = np.asarray(values, dtype=np.float64)
            if transform is not None:
                raw = raw @ np.asarray(transform, dtype=np.float64)
        if level is not None:
            level_rows = np.asarray(level[0], dtype=np.intp)
            if run is not None and np.any(run["codes"][level_rows] >= 0):
                close_run()  # rows shared with the open run: a new term
            if run is None:
                run = {"tables": [], "columns": [], "codes": np.full(n, -1, np.int32), "next": 0}
            run["codes"][level_rows] = run["next"] + np.asarray(level[1], dtype=np.int32)
            run["next"] += raw.shape[0]
            run["tables"].append(raw)
            run["columns"].append(columns)
            continue
        close_run()
        if values is None:
            terms.append(_Term(columns, raw, zero_of(columns)))
        else:
            centred = raw - centre[columns]
            if low is not None:
                centred -= low[columns]
            terms.append(_Term(columns, centred, None))
        term_codes.append(np.asarray(codes, dtype=np.int32))
    close_run()

    index = np.empty((n, len(terms)), dtype=np.int32)
    for t, term_code in enumerate(term_codes):
        index[:, t] = term_code
    tab_pairs, acc_pairs = _choose_pairs(terms, index)
    return TablePlan(
        n=n,
        p=dm.p,
        terms=tuple(terms),
        codes=index,
        dense=tuple(dense),
        dense_columns=np.array(
            [start + j for start, width, _ in dense for j in range(width)], dtype=np.intp
        ),
        tab_pairs=tab_pairs,
        acc_pairs=acc_pairs,
    )


def _choose_pairs(terms: list[_Term], codes: NDArray) -> tuple[NDArray, NDArray]:
    """Per term pair, a full table or an accumulation (Li & Wood 2020's choice, as a cost model).

    A table costs one gather a row plus ``m_t p_t p_u + m_t p_u m_u`` to form;
    an accumulation the window of the other term's row a row plus ``m_t p_t
    p_u``.  Tables are taken while their cells stay under ``_CELL_CAP`` each
    and a quarter of ``max_memory`` in all.  The choice moves cost only: both
    forms are inside the module's bound.
    """
    from superglm._parallel import resolve_max_memory

    n = codes.shape[0]
    avg_window = []
    for t, term in enumerate(terms):
        lo, hi = _windows(term.table)
        counts = np.bincount(codes[:, t], minlength=term.table.shape[0]).astype(np.float64)
        avg_window.append(float(counts @ (hi - lo)) / max(n, 1))
    budget = resolve_max_memory() // 4
    tab, acc = [], []
    for a in range(len(terms)):
        for b in range(a + 1, len(terms)):
            (ma, pa), (mb, pb) = terms[a].table.shape, terms[b].table.shape
            cells = ma * mb
            build = min(ma * pa * pb + ma * pb * mb, mb * pb * pa + mb * pa * ma)
            c_tab = n + _P_SPACE_WEIGHT * build
            c_ab = n * avg_window[b] + _P_SPACE_WEIGHT * ma * pa * pb
            c_ba = n * avg_window[a] + _P_SPACE_WEIGHT * mb * pa * pb
            if cells <= _CELL_CAP and 8 * cells <= budget and c_tab <= min(c_ab, c_ba):
                tab.append((a, b))
                budget -= 8 * cells
            elif c_ab <= c_ba:
                acc.append((a, b))
            else:
                acc.append((b, a))
    return (
        np.array(tab, dtype=np.int64).reshape(-1, 2),
        np.array(acc, dtype=np.int64).reshape(-1, 2),
    )


@dataclass(frozen=True)
class _Tables:
    """One inverse's flat tables, in the kernel's layout."""

    h0: float
    e: NDArray
    e_off: NDArray
    tab: NDArray  # (P1, 4): t, u, offset, m_u
    G: NDArray
    acc: NDArray  # (P2, 4): x, y, offset, p_y
    Q: NDArray
    A: NDArray
    A_off: NDArray
    lo: NDArray
    hi: NDArray
    w_off: NDArray
    F: NDArray
    f_off: NDArray
    g: NDArray
    Vdd: NDArray  # diagonal V_jj, off-diagonal 2 V_ij


def _flat(parts: list) -> NDArray:
    return np.ascontiguousarray(np.concatenate(parts)) if parts else np.zeros(1)


def _tables(plan: TablePlan, V: NDArray) -> _Tables:
    """Every table of the read-back for the inverse ``V`` (coefficient space only)."""
    terms = plan.terms
    zero = np.zeros(plan.p, dtype=np.float64)
    for term in terms:
        if term.zero is not None:
            zero[term.columns] = term.zero
    Vz = V @ zero
    e_parts = []
    for term in terms:
        A = term.table
        # Every term meets the raw terms' centres, its own or another's.
        e = np.einsum("ij,ij->i", A @ V[np.ix_(term.columns, term.columns)], A)
        e -= 2.0 * (A @ Vz[term.columns])
        e_parts.append(e)
    tab_rows, G_parts, offset = [], [], 0
    for a, b in plan.tab_pairs:
        A, B = terms[a].table, terms[b].table
        Vab = V[np.ix_(terms[a].columns, terms[b].columns)]
        G = (2.0 * (A @ Vab)) @ B.T if A.shape[0] <= B.shape[0] else (2.0 * A) @ (Vab @ B.T)
        tab_rows.append((a, b, offset, B.shape[0]))
        G_parts.append(G.ravel())
        offset += G.size
    acc_rows, Q_parts, offset = [], [], 0
    for x, y in plan.acc_pairs:
        Q = 2.0 * (terms[x].table @ V[np.ix_(terms[x].columns, terms[y].columns)])
        acc_rows.append((x, y, offset, terms[y].table.shape[1]))
        Q_parts.append(Q.ravel())
        offset += Q.size
    windows = [_windows(term.table) for term in terms]
    dense = plan.dense_columns
    n_d = dense.size
    F_parts = [
        np.ascontiguousarray(2.0 * (term.table @ V[np.ix_(term.columns, dense)])).ravel()
        for term in terms
    ]
    Vdd = 2.0 * V[np.ix_(dense, dense)]
    if n_d:
        Vdd[np.diag_indices(n_d)] = np.diag(V)[dense]
    rows = np.array([term.table.shape[0] for term in terms], dtype=np.int64)
    sizes = np.array([term.table.size for term in terms], dtype=np.int64)
    return _Tables(
        h0=float(zero @ Vz),
        e=_flat(e_parts),
        e_off=np.cumsum(np.r_[0, rows])[:-1],
        tab=np.array(tab_rows, dtype=np.int64).reshape(-1, 4),
        G=_flat(G_parts),
        acc=np.array(acc_rows, dtype=np.int64).reshape(-1, 4),
        Q=_flat(Q_parts),
        A=_flat([term.table.ravel() for term in terms]),
        A_off=np.cumsum(np.r_[0, sizes])[:-1],
        lo=_flat([w[0] for w in windows]).astype(np.int32),
        hi=_flat([w[1] for w in windows]).astype(np.int32),
        w_off=np.cumsum(np.r_[0, rows])[:-1],
        F=_flat(F_parts),
        f_off=np.cumsum(np.r_[0, rows * n_d])[:-1],
        g=np.ascontiguousarray(-2.0 * Vz[dense]),
        Vdd=np.ascontiguousarray(Vdd),
    )


@njit(cache=True, nogil=True)
def _gather_leverages(
    start,
    stop,
    codes,
    active,
    h0,
    e,
    e_off,
    tab,
    G,
    acc,
    Q,
    A,
    A_off,
    lo,
    hi,
    w_off,
    dense_values,
    F,
    f_off,
    g,
    Vdd,
    out,
):  # pragma: no cover - compiled
    """``out[r] = h_r`` for the active rows of ``[start, stop)``, in the fixed term order."""
    n_terms = codes.shape[1]
    n_dense = g.shape[0]
    for r in range(start, stop):
        if not active[r]:
            out[r] = 0.0
            continue
        h = h0
        for t in range(n_terms):
            h += e[e_off[t] + codes[r, t]]
        for q in range(tab.shape[0]):
            h += G[tab[q, 2] + codes[r, tab[q, 0]] * tab[q, 3] + codes[r, tab[q, 1]]]
        for q in range(acc.shape[0]):
            y = acc[q, 1]
            width = acc[q, 3]
            k = codes[r, y]
            source = A_off[y] + k * width
            target = acc[q, 2] + codes[r, acc[q, 0]] * width
            s = 0.0
            for c in range(lo[w_off[y] + k], hi[w_off[y] + k]):
                s += Q[target + c] * A[source + c]
            h += s
        if n_dense:
            row = dense_values[r - start]
            for j in range(n_dense):
                s = g[j]
                for t in range(n_terms):
                    s += F[f_off[t] + codes[r, t] * n_dense + j]
                for i in range(j):
                    s += row[i] * Vdd[i, j]
                s += row[j] * Vdd[j, j]
                h += row[j] * s
        out[r] = h


@njit(cache=True)
def _scatter_margins(
    start, stop, codes, values, m_off, total, compensation, dense_values, dense_total, dense_comp
):  # pragma: no cover - compiled
    """Add ``values[r]`` into each term's bin and ``d_r values[r]`` into each dense sum (Neumaier)."""
    n_terms = codes.shape[1]
    n_dense = dense_total.shape[0]
    for r in range(start, stop):
        v = values[r]
        if v == 0.0:
            continue
        for t in range(n_terms):
            k = m_off[t] + codes[r, t]
            s = total[k]
            u = s + v
            if abs(s) >= abs(v):
                compensation[k] += (s - u) + v
            else:
                compensation[k] += (v - u) + s
            total[k] = u
        for j in range(n_dense):
            x = dense_values[r - start, j] * v
            s = dense_total[j]
            u = s + x
            if abs(s) >= abs(x):
                dense_comp[j] += (s - u) + x
            else:
                dense_comp[j] += (x - u) + s
            dense_total[j] = u


_POOLED_ROW_KERNELS = (_gather_leverages,)
"""The kernels a leverage block runs on its pool worker: ``nogil``, never ``prange``, each
writing only its own rows of ``out``."""


def _dense_block(plan: TablePlan, start: int, stop: int, centre_hi, centre_lo) -> NDArray:
    """The block's centred dense values ``(x - c) - c_lo``, as ``_leverage_rows`` forms them."""
    out = np.empty((stop - start, plan.dense_columns.size), dtype=np.float64)
    filled = 0
    for column, width, matrix in plan.dense:
        cols = slice(column, column + width)
        values = np.asarray(matrix[start:stop], dtype=np.float64) - centre_hi[cols]
        if centre_lo is not None:
            values -= centre_lo[cols]
        out[:, filled : filled + width] = values
        filled += width
    return out


def table_leverages(
    plan: TablePlan,
    inverse: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
    active: NDArray,
) -> NDArray:
    """``h_r = x_r' inverse x_r`` for the active rows (``0`` elsewhere), read back from the tables.

    Rows run in fixed blocks of ``_BLOCK_ROWS`` on ``pool_workers`` threads
    (BLAS held at one thread meanwhile); the result is bitwise the serial one.
    """
    from superglm._blas_threads import pooled_blas_threads
    from superglm._parallel import pool_workers

    tables = _tables(plan, np.ascontiguousarray(inverse, dtype=np.float64))
    active = np.ascontiguousarray(active, dtype=np.bool_)
    out = np.empty(plan.n, dtype=np.float64)
    starts = range(0, plan.n, _BLOCK_ROWS)

    def run(start: int) -> None:
        stop = min(start + _BLOCK_ROWS, plan.n)
        _gather_leverages(
            start,
            stop,
            plan.codes,
            active,
            tables.h0,
            tables.e,
            tables.e_off,
            tables.tab,
            tables.G,
            tables.acc,
            tables.Q,
            tables.A,
            tables.A_off,
            tables.lo,
            tables.hi,
            tables.w_off,
            _dense_block(plan, start, stop, centre_hi, centre_lo),
            tables.F,
            tables.f_off,
            tables.g,
            tables.Vdd,
            out,
        )

    block_bytes = 8 * _BLOCK_ROWS * max(int(plan.dense_columns.size), 1)
    workers = pool_workers(len(starts), block_bytes)
    if workers <= 1:
        for start in starts:
            run(start)
    else:
        with pooled_blas_threads(), ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, starts))
    return out


def table_rmatvec(
    plan: TablePlan,
    values: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
) -> NDArray:
    """``X_c' values`` for the centred design the plan reads, serially in row order.

    A term's share is ``A_t' M_t - z_t sum(values)``, ``M_t`` its bins'
    compensated sums; a dense column's is the compensated sum of ``d_r
    values_r``.
    """
    sizes = np.array([term.table.shape[0] for term in plan.terms], dtype=np.int64)
    m_off = np.cumsum(np.r_[0, sizes])[:-1]
    total = np.zeros(int(sizes.sum()), dtype=np.float64)
    compensation = np.zeros_like(total)
    n_d = int(plan.dense_columns.size)
    dense_total = np.zeros(n_d, dtype=np.float64)
    dense_comp = np.zeros(n_d, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    for start in range(0, plan.n, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, plan.n)
        _scatter_margins(
            start,
            stop,
            plan.codes,
            values,
            m_off,
            total,
            compensation,
            _dense_block(plan, start, stop, centre_hi, centre_lo),
            dense_total,
            dense_comp,
        )
    margins = total + compensation
    value_sum = math.fsum(values)
    out = np.zeros(plan.p, dtype=np.float64)
    for t, term in enumerate(plan.terms):
        share = term.table.T @ margins[m_off[t] : m_off[t] + sizes[t]]
        if term.zero is not None:
            share = share - term.zero * value_sum
        out[term.columns] = share
    if n_d:
        out[plan.dense_columns] = dense_total + dense_comp
    return out
