"""Row passes of a design of lookups through per-term-pair tables: Grams, leverages, ``X_c' v``.

A *lookup* group's row ``r`` is row ``k_t(r)`` of a small table ``A_t``: a
categorical's identity, a binned or support-compressed spline's ``B_unique
R_inv``, a tensor's occupied cells, a spline-by-category level.  When every
group of a design is a lookup or a dense column (``DenseGroupMatrix``), each
row pass reduces to sweeps over the term index columns:

- **Gram** ``X_c' W X_c`` (Li & Wood, *Stat. Comput.* 30:19, 2020, Algorithm
  1): per term pair the weighted 2-D histogram ``T_tu[k_t, k_u] = sum W_r``,
  then ``A_t' T_tu A_u`` in coefficient space; a pair whose histogram would be
  too large is *accumulated* (their Algorithms 2/3), ``M[k_t] += W_r
  A_u[k_u(r)]`` over the nonzero window of ``A_u``'s row, then ``A_t' M``.
- **Leverages** ``h_r = x_r' V x_r``, the transpose of the same tables:
  ``G_tu = A_t V_tu A_u'`` gathered per row (the scattered-row form of GLAM's
  ``diag(X S X')``, Currie, Durban & Eilers, *JRSS-B* 68:259, 2006, eq.
  2.7; Wood, Li, Shaddick & Augustin, *JASA* 112:1199, 2017, App. A, form it
  column by column), or ``Q_tu = A_t V_tu`` dotted with the window.
- ``X_c' v``: one 1-D histogram per term.

Consecutive spline-by-category levels with disjoint rows merge into one term
(their table stacks the levels' rows; a row outside every level reads a zero
row).  The structural plan (:func:`lookup_plan`: index columns, raw tables,
pair forms) depends on the design only; it is held in the design's
``_structured_layout_cache`` (owner: the ``DesignMatrix``; lifetime: one
fit, released when the fit publishes its state, :func:`release_lookup_plan`; invalidated when any group matrix, its basis, transform or index
array, or a dense group's values are replaced, which the plan checks by
identity).  Weights, centre and
inverse enter per call.

**Centring.**  Rows are the profiled-intercept rows ``x_r = (raw_r - c) -
c_lo``.  The Gram centres every table before its product (``A_t - c_t``; a
zero row centres to ``-c_t``), which is exact for a histogram pair whatever
the term; an accumulated pair reads a raw window of a term with a zero row,
whose centre needs no term of its own because the row side is centred about
the weighted mean (``A_t' m_t = 0``).  The leverage reads a term with a zero row raw and its centre ``z`` as
constants, as the support route does (``reml/_leverage_rows.py``).  Dense
columns are centred about the exact pair superglm uses everywhere: the Gram
about the rounded anchor with Bjorck's two-pass correction
(``centered_system.two_pass_centred_gram``), the leverage and ``X_c' v`` as
``(x - c) - c_lo``.

**Error** (``u = 2**-53``, ``gamma_k = k u / (1 - k u)``, Higham 2002, secs.
3.1, 3.5, Lemma 3.3; ``|.|`` entrywise).
- Gram: a histogram cell sums nonnegative weights, so it is within
  ``gamma_n`` of its exact value relatively, and ``A_t' T A_u`` adds
  ``gamma_(m_t + m_u)``; an entry is within ``gamma_(n + 2m + 2) sum_r W_r
  (|x_r| + |c|)(|x_r| + |c|)'`` (``m`` the largest table, ``c`` the centre,
  on every lookup column), the bound of a direct centred Gram plus the
  centres: a raw window reads its centre, and an accumulated pair's row side
  is centred about the computed mean, whose weighted residual ``e_x``
  multiplies the window's centre (as it multiplies a dense remainder, whose
  correction reads ``e`` as zero on the lookup columns).  ``e_x`` is within
  ``gamma_(n + m + 2) sum_r W_r |a_x,r|`` of zero, ``a`` the table rows as
  read: a whole support's table is anchored before ``R`` applies, ``(B -
  B[a]) R`` with the shared row ``B[a] R`` kept apart (the established
  support route's anchoring, ``_anchor_center_support``), so in the Gram
  neither ``e_x`` nor any row of such a support carries a common component of
  ``B``, whether ``R`` cancels it or not; a categorical's rows are
  indicators.  Two reads keep ``u |B| |R|``-sized errors.  The leverage and
  ``X_c' v`` read a whole support as ``raw - (c - B[a] R)``, where ``c -
  B[a] R`` differs from the Gram's shift by up to ``u |c|`` (v0.40.0's
  leverage read ``fl(B R) - c``, no better; a design whose offset dwarfs its
  rows past ``_SEPARATION_LIMIT`` is declined).  A spline-by-category level
  is projected unanchored, its zero row kept zero, in the leverage as in
  v0.40.0 but in the Gram a narrowing against v0.40.0: its rung ladder
  certified the raw moments and fell back to anchoring a level's ``B``
  before ``R``, which the table Gram replaces, so a level whose ``B``
  carries a common component that ``R`` cancels keeps a ``u C |R|`` error
  there (at rounding level for B-spline levels, ``|B| <= 1``).  Applying Bjorck's correction for
  ``e_x`` (Chan, Golub & LeVeque 1983) moved no entry beyond that rounding
  on the common-component fixtures, so it is not applied.
- Leverage: with ``eta_r = (|a_r| + |z| + |d_r|)' |V| (|a_r| + |z| + |d_r|)``
  (``a_r`` the table rows read, ``d_r`` the centred dense values) every table
  entry is a bilinear form read through at most ``3p + 1`` roundings and a
  row sums ``S <= 1 + T + P + n_d (T + n_d + 3)`` terms, so ``|h_r(computed)
  - h_r| <= gamma_K eta_r`` with ``K = 3p + S + 2``
  (:meth:`LookupPlan.count_roundings`).
- ``X_c' v``: compensated (Neumaier) sums per bin (Higham 2002, sec. 4.3),
  within ``2u |s| + O(n u^2) sum |terms|``, then ``gamma_(m+1)`` in
  coefficient space.

**Determinism.**  Every Gram table is filled by one owner, in row order; a
leverage row is summed in one fixed order (terms, table pairs, accumulated
pairs, dense columns) inside a fixed row block.  Results are bitwise
identical for any worker count.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Fixed row blocks of the leverage gather: the partition never depends on the worker count.
_BLOCK_ROWS = 8192
# A pair histogram larger than this many cells is accumulated instead.
_CELL_CAP = 1 << 22
# Declines (to the chunked route) where float64 cannot keep what the tables
# keep apart: a quantity rounded at ``u`` times a magnitude this many times
# the one it must resolve has lost more than ``sqrt(u)`` of it.
_SEPARATION_LIMIT = 2.0**26
# Relative cost of one coefficient-space multiply-add (BLAS, in cache) against
# one random row access (prototype measurement, solver-scaling report sec. 7).
_P_SPACE_WEIGHT = 0.1
_CACHE_KEY = "lookup_tables"
# Tests only: every pair read through a full table ("table") or accumulated
# ("accumulate") instead of the cost model's choice.
_FORCE_PAIR_FORM: str | None = None
# The Gram takes the tables from this many rows: below it a Gram pass is
# milliseconds on every route, and the established rungs keep small fits.
_MIN_TABLE_GRAM_ROWS = 20_000
# ...and only while the design has at least this many columns per row sweep of
# the tables (one per term, term pair, term and dense block, and dense pair).
# The tables cost a sweep each, the chunked rungs about one pass per column:
# per call on freMTPL2 (542,406 rows, one thread) 3.2-3.8 ns per row and
# sweep against 0.56-0.77 ns per row and column at up to 340 columns.  Below
# the crossover a main-effects design (67 columns, 21 sweeps) is 1.65 times
# faster on the rungs; a spline-by-factor pair (197 columns, 28 sweeps) is
# 1.65 times faster on the tables, a tensor pair (184, 28) the same on both.
_TABLE_GRAM_COLUMNS_PER_SWEEP = 6

_gram_override: ContextVar[bool | None] = ContextVar("superglm_table_gram", default=None)


@contextmanager
def table_gram_config(enabled: bool | None) -> Iterator[None]:
    """Hidden override for tests and measurement: force the table Gram on or off (``None``: the rule)."""
    token = _gram_override.set(enabled)
    try:
        yield
    finally:
        _gram_override.reset(token)


@dataclass(frozen=True)
class _Term:
    """One lookup term: rows ``raw[codes]`` on ``columns``; ``raw_read`` when it has a zero row."""

    columns: NDArray  # (p_t,) global coefficient columns
    raw: NDArray  # (m_t, p_t) float64, uncentred
    raw_read: bool  # a categorical or a spline-by-category run: read raw, centre as a constant
    lo: NDArray  # (m_t,) int32 window start the kernels read (raw rows if raw_read, else 0)
    hi: NDArray  # (m_t,) int32 window stop (raw rows if raw_read, else p_t)
    offset: NDArray  # (p_t,) the rows' common part: a design row is raw[k] + offset
    identity: bool = False  # a categorical: raw is the identity over a zero row


@dataclass
class _LevelRun:
    """An open spline-by-category run: disjoint levels stacked into one term."""

    codes: NDArray  # (n,) int32 stacked-table row per observation, -1 outside the run
    tables: list[NDArray] = field(default_factory=list)
    columns: list[NDArray] = field(default_factory=list)
    next: int = 0  # first stacked-table row of the next level


@dataclass(frozen=True)
class LookupPlan:
    """The design-only part of the table passes (owner: the design's layout cache)."""

    n: int
    p: int
    terms: tuple
    codes: NDArray  # (T, n) int32, one contiguous index column per term
    dense: tuple  # (column, width, values) per DenseGroupMatrix
    dense_columns: NDArray  # (n_d,) intp
    tab_pairs: NDArray  # (P1, 2) term indices of pairs read through a full table
    acc_pairs: NDArray  # (P2, 2) (row term, window term) of accumulated pairs

    @property
    def n_pairs(self) -> int:
        return int(self.tab_pairs.shape[0] + self.acc_pairs.shape[0])

    @property
    def n_sweeps(self) -> int:
        """Row sweeps of one table Gram: margins, pairs, term-by-dense blocks, dense pairs."""
        t, d = len(self.terms), len(self.dense)
        return t + self.n_pairs + t * d + d * (d + 1) // 2

    def count_roundings(self) -> int:
        """``K = 3p + S + 2``, the roundings along one leverage's longest chain (module docstring).

        ``3p + 1`` for a table entry's bilinear form, ``S`` for the row's sum,
        and one for a dense column's last product ``d_j (partial + d_j V_jj)``."""
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


def _owners(dm) -> tuple:
    """Every object the plan reads, compared by identity: the group tuple, each group matrix
    and each array it reads (a dense group's ``M``; a level's rows and codes, flattened)."""
    from superglm.group_matrix import DenseGroupMatrix

    owners: list = [dm.group_matrices]
    for gm in dm.group_matrices:
        owners.append(gm)
        parts = (gm.M,) if type(gm) is DenseGroupMatrix else _lookup_parts(gm) or ()
        for part in parts:
            if isinstance(part, tuple):
                owners.extend(part)
            elif part is not None:
                owners.append(part)
    return tuple(owners)


def lookup_plan(dm) -> LookupPlan | None:
    """The design's table plan, built once and held in its layout cache; ``None`` if it has none.

    ``None`` (the callers' other routes) when a group is neither a lookup nor a
    ``DenseGroupMatrix``, when no group is a lookup, when a table does not
    read as float64, when a support's factors lie outside the range in which
    ``B R`` may be reassociated (``_ssp_gram_needs_exact``, which sends the
    established routes to exact moments) or its product is not finite, or
    when the index and tables would not fit a quarter of ``max_memory``.  A
    zero-width group (an empty categorical) reads no column and is no term.
    """
    from superglm.group_matrix import DesignMatrix

    if type(dm) is not DesignMatrix:
        return None
    cache = dm._structured_layout_cache
    entry = cache.get(_CACHE_KEY)
    if entry is not None:
        owners, plan = entry
        current = _owners(dm)
        if len(owners) == len(current) and all(a is b for a, b in zip(owners, current)):
            return plan
    plan = _build_plan(dm)
    cache[_CACHE_KEY] = (_owners(dm), plan)
    return plan


def release_lookup_plan(layout_cache: dict | None) -> None:
    """Drop the design's plan once a fit has published its state.

    The plan (an ``(T, n)`` index and the raw tables, about ``4 T n`` bytes)
    serves the fit's repeated passes; a later pass on the design rebuilds it.
    """
    if layout_cache is not None:
        layout_cache.pop(_CACHE_KEY, None)


def table_gram_plan(dm) -> LookupPlan | None:
    """The plan when the centred Gram should come from the tables, else ``None``.

    The exact path's designs only: every lookup group a categorical or a
    lossless support (``is_lossless_support``).  A binned design (the
    discrete path) keeps its own tensor routes, which were measured there.
    A design with fewer than ``_TABLE_GRAM_COLUMNS_PER_SWEEP`` columns per row
    sweep (:meth:`LookupPlan.n_sweeps`) keeps the chunked rungs, cheaper there,
    and builds no plan where the group types settle that (:func:`_sweep_bounds`).
    """
    from superglm.group_matrix import CategoricalGroupMatrix, DenseGroupMatrix

    enabled = _gram_override.get()
    if enabled is False or (enabled is None and dm.n < _MIN_TABLE_GRAM_ROWS):
        return None
    if enabled is None and not all(
        type(gm) in (CategoricalGroupMatrix, DenseGroupMatrix)
        or getattr(gm, "is_lossless_support", False)
        for gm in getattr(dm, "group_matrices", ())
    ):
        return None
    if enabled is None:
        # Decided from the groups before any index is allocated where the
        # bounds settle it; only a design between them builds the plan to count.
        fewest, most = _sweep_bounds(dm)
        if dm.p < _TABLE_GRAM_COLUMNS_PER_SWEEP * fewest:
            return None
        if dm.p < _TABLE_GRAM_COLUMNS_PER_SWEEP * most:
            plan = lookup_plan(dm)
            if plan is None or dm.p < _TABLE_GRAM_COLUMNS_PER_SWEEP * plan.n_sweeps:
                return None
            return plan
    return lookup_plan(dm)


def _sweeps(terms: int, dense: int) -> int:
    """:attr:`LookupPlan.n_sweeps` for ``terms`` terms (every pair one sweep) and ``dense`` blocks."""
    return terms + terms * (terms - 1) // 2 + terms * dense + dense * (dense + 1) // 2


def _sweep_bounds(dm) -> tuple[int, int]:
    """Bounds on the plan's row sweeps from the group types alone.

    Spline-by-category levels merge into runs (:func:`_build_plan`): at least
    one run per stretch of consecutive levels, at most one per level.
    """
    from superglm.group_matrix import DenseGroupMatrix, DiscretizedSplineCategoricalGroupMatrix

    lookups = runs = levels = dense = 0
    in_run = False
    for gm in dm.group_matrices:
        if not gm.shape[1]:
            continue
        if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
            levels += 1
            runs += not in_run
            in_run = True
            continue
        in_run = False
        if type(gm) is DenseGroupMatrix:
            dense += 1
        else:
            lookups += 1
    return _sweeps(lookups + runs, dense), _sweeps(lookups + levels, dense)


def _build_plan(dm) -> LookupPlan | None:
    from superglm._group_matrix._group_matrix_centered import _reads_as_float64
    from superglm._group_matrix._group_matrix_kernels import _ssp_gram_needs_exact
    from superglm._parallel import resolve_max_memory
    from superglm.group_matrix import DenseGroupMatrix

    # A zero-width group reads nothing: no term, no table, no window.
    groups = [gm for gm in dm.group_matrices if gm.shape[1]]
    parts = []
    for gm in groups:
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
        if found[1] is not None and _ssp_gram_needs_exact(found[0], found[1]):
            return None
        parts.append(found)
    if all(part is None for part in parts):
        return None
    n = dm.n
    table_rows = sum(
        (np.asarray(part[0]).shape[0] if part[0] is not None else gm.shape[1]) + 1
        for part, gm in zip(parts, groups, strict=True)
        if part is not None
    )
    if 4 * n * len(parts) + 8 * table_rows * dm.p > resolve_max_memory() // 4:
        return None

    terms: list[_Term] = []
    term_codes: list[NDArray] = []
    dense: list[tuple[int, int, NDArray]] = []
    run: _LevelRun | None = None

    def add_term(
        columns: NDArray,
        raw: NDArray,
        raw_read: bool,
        offset: NDArray | None = None,
        identity: bool = False,
    ) -> None:
        if raw_read:
            lo, hi = _windows(raw)
        else:
            lo = np.zeros(raw.shape[0], dtype=np.int32)
            hi = np.full(raw.shape[0], raw.shape[1], dtype=np.int32)
        if offset is None:
            offset = np.zeros(raw.shape[1], dtype=np.float64)
        terms.append(_Term(columns, np.ascontiguousarray(raw), raw_read, lo, hi, offset, identity))

    def close_run() -> None:
        nonlocal run
        if run is None:
            return
        tables = run.tables
        stacked = np.zeros(
            (sum(t.shape[0] for t in tables) + 1, sum(t.shape[1] for t in tables)),
            dtype=np.float64,
        )
        r = c = 0
        for table in tables:
            stacked[r : r + table.shape[0], c : c + table.shape[1]] = table
            r += table.shape[0]
            c += table.shape[1]
        codes = run.codes
        codes[codes < 0] = stacked.shape[0] - 1
        add_term(np.concatenate(run.columns), stacked, True)
        term_codes.append(codes)
        run = None

    column = 0
    for gm, part in zip(groups, parts, strict=True):
        width = int(gm.shape[1])
        columns = np.arange(column, column + width, dtype=np.intp)
        column += width
        if part is None:
            close_run()
            dense.append((int(columns[0]), width, gm.M))
            continue
        values, transform, codes, level = part
        offset = None
        if values is None:  # categorical: the identity, its base level the zero row
            raw = np.vstack((np.eye(width, dtype=np.float64), np.zeros((1, width))))
        else:
            raw = np.asarray(values, dtype=np.float64)
            if transform is not None and level is None:
                # Anchored before projecting, as the established support route
                # (``_anchor_center_support``): a common component of ``B``
                # that ``R`` cancels never enters a product, and the anchor's
                # row ``B[a] R`` is the offset every row shares.
                transform = np.asarray(transform, dtype=np.float64)
                anchor = raw[int(np.argmax(np.bincount(codes, minlength=raw.shape[0])))]
                offset = anchor @ transform
                raw = (raw - anchor) @ transform
            elif transform is not None:
                raw = raw @ np.asarray(transform, dtype=np.float64)
            if not (np.all(np.isfinite(raw)) and (offset is None or np.all(np.isfinite(offset)))):
                return None
            if offset is not None and np.any(
                np.abs(offset) > _SEPARATION_LIMIT * np.abs(raw).max(axis=0, initial=0.0)
            ):
                # The leverage and ``X_c' v`` read a centre ``c = fl(offset +
                # shift)`` back as ``c - offset``, within ``u |offset|`` of the
                # Gram's shift; past the limit that erases the rows' scale.
                return None
        if level is not None:
            level_rows = np.asarray(level[0], dtype=np.intp)
            if run is not None and np.any(run.codes[level_rows] >= 0):
                close_run()  # rows shared with the open run: a new term
            open_run = run if run is not None else _LevelRun(np.full(n, -1, np.int32))
            run = open_run
            open_run.codes[level_rows] = open_run.next + np.asarray(level[1], dtype=np.int32)
            open_run.next += raw.shape[0]
            open_run.tables.append(raw)
            open_run.columns.append(columns)
            continue
        close_run()
        add_term(columns, raw, values is None, offset, values is None)
        term_codes.append(np.asarray(codes, dtype=np.int32))
    close_run()

    index = np.empty((len(terms), n), dtype=np.int32)
    for t, term_code in enumerate(term_codes):
        index[t] = term_code
    tab_pairs, acc_pairs = _choose_pairs(terms, index)
    return LookupPlan(
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

    A table costs one access a row plus ``m_t p_t p_u + m_t p_u m_u`` to form;
    an accumulation the window of the other term's row a row plus ``m_t p_t
    p_u``.  Tables are taken while their cells stay under ``_CELL_CAP`` each
    and a quarter of ``max_memory`` in all.  The choice moves cost only: both
    forms are inside the module's bounds.
    """
    from superglm._parallel import resolve_max_memory

    n = codes.shape[1]
    avg_window = []
    for t, term in enumerate(terms):
        counts = np.bincount(codes[t], minlength=term.raw.shape[0]).astype(np.float64)
        avg_window.append(float(counts @ (term.hi - term.lo)) / max(n, 1))
    budget = resolve_max_memory() // 4
    tab, acc = [], []
    for a in range(len(terms)):
        for b in range(a + 1, len(terms)):
            (ma, pa), (mb, pb) = terms[a].raw.shape, terms[b].raw.shape
            cells = ma * mb
            build = min(ma * pa * pb + ma * pb * mb, mb * pb * pa + mb * pa * ma)
            c_tab = n + _P_SPACE_WEIGHT * build
            c_ab = n * avg_window[b] + _P_SPACE_WEIGHT * ma * pa * pb
            c_ba = n * avg_window[a] + _P_SPACE_WEIGHT * mb * pa * pb
            take_table = cells <= _CELL_CAP and 8 * cells <= budget and c_tab <= min(c_ab, c_ba)
            if _FORCE_PAIR_FORM is not None:
                take_table = _FORCE_PAIR_FORM == "table"
            if take_table:
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


def _read_tables(plan: LookupPlan, centre: NDArray) -> tuple[list[NDArray], NDArray]:
    """Each term's table as the passes read it, and the centre ``z`` of the raw-read terms.

    A raw-read term keeps its raw table (its centre goes to ``z``); every other
    term is centred, ``raw - (c - offset)``.
    """
    zero = np.zeros(plan.p, dtype=np.float64)
    tables = []
    for term in plan.terms:
        if term.raw_read:
            zero[term.columns] = centre[term.columns]
            tables.append(term.raw)
        else:
            tables.append(term.raw - (centre[term.columns] - term.offset))
    return tables, zero


def _pool_run(jobs: list, job_bytes: int) -> None:
    """Run ``jobs`` (callables writing disjoint outputs) on ``pool_workers`` threads, else serially.

    BLAS runs on one thread for the whole run, serial or pooled, under every
    policy, so a job's arithmetic does not depend on the worker count; each
    pooled job runs in a copy of the caller's context, which carries its
    ``np.errstate`` (as ``_parallel.pooled_map`` does).
    """
    import contextvars

    from superglm._blas_threads import pooled_blas_threads
    from superglm._parallel import pool_workers

    workers = pool_workers(len(jobs), job_bytes)
    with pooled_blas_threads():
        if workers <= 1:
            for job in jobs:
                job()
            return
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="superglm-tables") as pool:
            futures = [pool.submit(contextvars.copy_context().run, job) for job in jobs]
            for future in futures:
                future.result()


# --------------------------------------------------------------------------- Gram


@njit(cache=True, nogil=True)
def _margins_sweep(codes, W, Wz, wm, zm):  # pragma: no cover - compiled
    """``wm[k] += W_r``, ``zm[k] += W_r z_r`` over one term's index column, in row order."""
    for r in range(codes.shape[0]):
        k = codes[r]
        wm[k] += W[r]
        zm[k] += Wz[r]


@njit(cache=True, nogil=True)
def _hist_sweep(ca, cb, W, mb, out):  # pragma: no cover - compiled
    """The weighted 2-D histogram of one term pair, in row order."""
    for r in range(ca.shape[0]):
        out[ca[r] * mb + cb[r]] += W[r]


@njit(cache=True, nogil=True)
def _acc_sweep(cx, cy, W, Y, lo, hi, width, out):  # pragma: no cover - compiled
    """``out[k_x] += W_r Y[k_y]`` over the window of ``Y``'s row, in row order."""
    for r in range(cx.shape[0]):
        k = cy[r]
        source = k * width
        target = cx[r] * width
        w = W[r]
        for c in range(lo[k], hi[k]):
            out[target + c] += w * Y[source + c]


@njit(cache=True, nogil=True)
def _dense_term_sweep(ct, W, X, anchor, out):  # pragma: no cover - compiled
    """``out[k_t, j] += W_r (x_rj - a_j)`` for one term and one dense block, in row order."""
    width = X.shape[1]
    for r in range(ct.shape[0]):
        target = ct[r] * width
        w = W[r]
        for j in range(width):
            out[target + j] += w * (X[r, j] - anchor[j])


@njit(cache=True, nogil=True)
def _dense_moments_sweep(W, Wz, X1, a1, X2, a2, gram, first, rhs, own):  # pragma: no cover
    """``sum W d1 d2'`` for two dense blocks about their anchors; with ``own``, ``d1``'s moments too."""
    w1 = X1.shape[1]
    w2 = X2.shape[1]
    d2 = np.empty(w2)
    for r in range(X1.shape[0]):
        w = W[r]
        for j in range(w2):
            d2[j] = X2[r, j] - a2[j]
        for i in range(w1):
            d1 = X1[r, i] - a1[i]
            wd = w * d1
            for j in range(w2):
                gram[i, j] += wd * d2[j]
            if own:
                first[i] += wd
                rhs[i] += Wz[r] * d1


def _left(term: _Term, A: NDArray, d: NDArray, X: NDArray) -> NDArray:
    """``A' X`` for the term's centred table ``A = raw - d``.

    A categorical's raw table is the identity over a zero row, so ``A' X =
    X[:K] - d (1' X)``: ``O(m p)`` where the product is ``O(m p K)``.
    """
    if term.identity:
        return X[: term.raw.shape[1]] - np.outer(d, X.sum(axis=0))
    return A.T @ X


def _right(term: _Term, A: NDArray, d: NDArray, X: NDArray) -> NDArray:
    """``X A`` for the term's centred table ``A = raw - d`` (``_left``'s transpose)."""
    if term.identity:
        return X[:, : term.raw.shape[1]] - np.outer(X.sum(axis=1), d)
    return X @ A


def _left_cost(term: _Term, rows: int, cols: int) -> int:
    """Multiply-adds of ``_left`` (or ``_right``) on a ``rows x cols`` operand."""
    return rows * cols * (1 if term.identity else term.raw.shape[1])


def _rows_times(term: _Term, table: NDArray, M: NDArray) -> NDArray:
    """``table @ M`` for a term's table as the leverage reads it (a categorical's raw identity)."""
    if term.identity and table is term.raw:
        return np.vstack((M, np.zeros((1, M.shape[1]), dtype=M.dtype)))
    return table @ M


def _pair_block(
    ta: _Term, A: NDArray, da: NDArray, tb: _Term, B: NDArray, db: NDArray, T: NDArray
) -> NDArray:
    """``A' T B`` for two centred tables, in the cheaper association by its full cost."""
    (ma, pa), (mb, pb) = A.shape, B.shape
    if _left_cost(tb, ma, mb) + _left_cost(ta, ma, pb) <= _left_cost(ta, ma, mb) + _left_cost(
        tb, pa, mb
    ):
        return _left(ta, A, da, _right(tb, B, db, T))
    return _right(tb, B, db, _left(ta, A, da, T))


def _dominated(wm: NDArray) -> bool:
    """One bin outweighs the rest of a categorical past ``_SEPARATION_LIMIT``.

    ``_identity_gram``'s diagonal ``w_k - 2 d_k w_k + S d_k^2`` is ``w_k R_k / S``
    (``R_k`` the other bins' mass) computed with an error of ``u w_k``: past the
    limit it has lost more than ``sqrt(u)`` of the curvature, all of it as the
    rest falls under the dominant bin's spacing.
    """
    mass = np.abs(wm)
    k = int(np.argmax(mass))
    rest = float(np.sum(mass[:k]) + np.sum(mass[k + 1 :]))
    return bool(mass[k] > _SEPARATION_LIMIT * rest)


def _identity_gram(wm: NDArray, d: NDArray) -> NDArray:
    """A categorical's own block ``sum_k wm_k (e_k - d)(e_k - d)'`` (``e_K = 0``, the base level).

    ``diag(w) - d w' - w d' + (1' wm) d d'`` with ``w`` the levels' weights:
    ``O(K^2)`` where the centred table's GEMM is ``O(K^3)``, and exactly
    symmetric.  This is a raw-moment subtraction: level ``k`` with weight
    share ``d_k`` loses about ``u * S / R_k`` of its curvature to
    cancellation (``R_k`` the rest of the weight), where the chunked route
    centres each row before multiplying.  ``_dominated`` keeps that factor
    below ``2**26`` (about ``1e-8`` of the column's curvature), within
    float64 accuracy for every fitted value and standard error; a more
    dominant level sends the call to the chunked route.
    """
    w = wm[: d.shape[0]]
    cross = np.outer(d, w)
    return np.diag(w) - cross - cross.T + float(np.sum(wm)) * np.outer(d, d)


def _rows_cost(term: _Term, table: NDArray, cols: int) -> int:
    """Multiply-adds of ``_rows_times(term, table, M)`` for an ``M`` of ``cols`` columns."""
    rows, inner = table.shape
    return rows * cols * (1 if term.identity and table is term.raw else inner)


def _pair_table(ta: _Term, A: NDArray, tb: _Term, B: NDArray, X: NDArray) -> NDArray:
    """``A X B'`` in the cheaper association by its full cost (both products)."""
    if _rows_cost(ta, A, X.shape[1]) + _rows_cost(tb, B, A.shape[0]) <= _rows_cost(
        tb, B, X.shape[0]
    ) + _rows_cost(ta, A, B.shape[0]):
        return _rows_times(tb, B, _rows_times(ta, A, X).T).T
    return _rows_times(ta, A, _rows_times(tb, B, X.T).T)


# Rows per range of the Gram's sweeps.  The ranges depend on the row count
# alone, never on the worker count, so the summed tables are the same at
# every worker count; a design of at most this many rows sweeps in one range,
# exactly as a single pass.
_SWEEP_ROWS = 1 << 17
# At most this many ranges: past 16 * _SWEEP_ROWS rows a range grows with the
# design instead, so a split sweep holds at most 16 copies of its outputs at any
# row count (still a function of the design alone).
_MAX_SWEEP_RANGES = 16
# A sweep splits only while its outputs are this small: each range then holds
# and adds one copy of them, which a table of many cells (a tensor's
# occupied cells against another's) would make cost more than the split buys.
_SPLIT_MAX_ENTRIES = 1 << 14


class _Sweeps:
    """Row sweeps over fixed row ranges, each range into its own partial outputs.

    ``add(cost, outputs, run)`` registers one sweep: ``run(lo, hi, parts)``
    sweeps rows ``lo:hi`` into ``parts``, arrays shaped as ``outputs``.  A
    design of one range sweeps straight into ``outputs``; otherwise every
    range sweeps into its own zeroed partials, and :meth:`run` sums them into
    ``outputs`` in range order once the pool has run them (at most
    ``_MAX_SWEEP_RANGES`` ranges, so at most that many copies), largest estimated
    cost (``cost`` per row) first.  A sweep whose outputs exceed
    ``_SPLIT_MAX_ENTRIES`` runs whole.  Each partial has one owner, and
    whether a sweep splits depends on the design alone, so the sums are
    bitwise the same at every worker count.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        rows = max(_SWEEP_ROWS, -(-n // _MAX_SWEEP_RANGES))
        self.ranges = [(lo, min(lo + rows, n)) for lo in range(0, n, rows)]
        self.jobs: list[tuple[float, Callable[[], None]]] = []
        self.reductions: list[tuple[list[NDArray], list[list[NDArray]]]] = []

    def add(self, cost: float, outputs: list[NDArray], run: Callable) -> None:
        if len(self.ranges) <= 1 or sum(out.size for out in outputs) > _SPLIT_MAX_ENTRIES:
            self.jobs.append((cost * self.n, lambda: run(0, self.n, outputs)))
            return
        parts = [[np.zeros_like(out) for out in outputs] for _ in self.ranges]
        for (lo, hi), part in zip(self.ranges, parts, strict=True):
            self.jobs.append((cost * (hi - lo), lambda lo=lo, hi=hi, part=part: run(lo, hi, part)))
        self.reductions.append((outputs, parts))

    def run(self, job_bytes: int) -> None:
        order = sorted(range(len(self.jobs)), key=lambda i: (-self.jobs[i][0], i))
        _pool_run([self.jobs[i][1] for i in order], job_bytes)
        for outputs, parts in self.reductions:
            for k, out in enumerate(outputs):
                out[...] = parts[0][k]
                for part in parts[1:]:
                    out += part[k]


def table_centred_system(plan: LookupPlan, dm, W: NDArray, z_centered: NDArray, sum_w: float):
    """``(mean_x, data_gram, rhs, mean_hi, mean_lo)`` of the centred system, from the tables.

    ``None`` (the caller's chunked route) when a categorical's weights are
    dominated by one bin (``_dominated``).

    The same quantities ``centered_system.build_centered_system`` returns:
    ``data_gram = X_c' W X_c`` and ``rhs = X_c' W z_centered`` about the
    weighted column means, a dense column about the exact pair ``(hi, lo)``
    by the corrected two-pass algorithm.  Every table has one owner and is
    filled in row order (bitwise identical for any worker count).
    """
    from superglm.solvers.centered_system import dense_anchor, two_pass_centred_gram

    W = np.ascontiguousarray(W, dtype=np.float64)
    Wz = np.ascontiguousarray(W * z_centered, dtype=np.float64)
    terms, codes, p = plan.terms, plan.codes, plan.p
    margins = [(np.zeros(term.raw.shape[0]), np.zeros(term.raw.shape[0])) for term in plan.terms]
    sweeps = _Sweeps(plan.n)
    for t in range(len(terms)):
        sweeps.add(
            2.0,
            list(margins[t]),
            lambda lo, hi, out, t=t: _margins_sweep(
                codes[t][lo:hi], W[lo:hi], Wz[lo:hi], out[0], out[1]
            ),
        )
    sweeps.run(16 * max(term.raw.shape[0] for term in terms))
    for term, (wm, _) in zip(terms, margins, strict=True):
        if term.identity and _dominated(wm):
            return None

    mean_x = np.zeros(p, dtype=np.float64)
    shifts = []
    for term, (wm, _) in zip(terms, margins, strict=True):
        shifts.append((term.raw.T @ wm) / sum_w)
        mean_x[term.columns] = term.offset + shifts[-1]
    dense_mask = np.zeros(p, dtype=bool)
    dense_mask[plan.dense_columns] = True
    anchor = dense_anchor(dm, W, sum_w) if plan.dense else np.zeros(p)
    mean_hi = np.where(dense_mask, anchor, mean_x)
    centred = [term.raw - shift for term, shift in zip(terms, shifts, strict=True)]

    tab_out = [np.zeros(terms[a].raw.shape[0] * terms[b].raw.shape[0]) for a, b in plan.tab_pairs]
    acc_out = [np.zeros(terms[x].raw.shape[0] * terms[y].raw.shape[1]) for x, y in plan.acc_pairs]
    acc_read = [
        np.ascontiguousarray(terms[y].raw if terms[y].raw_read else centred[y]).ravel()
        for _, y in plan.acc_pairs
    ]
    dense_values = [np.asarray(values, dtype=np.float64) for (_, _, values) in plan.dense]
    dense_out = [
        [np.zeros(term.raw.shape[0] * width) for (_, width, _) in plan.dense] for term in terms
    ]
    n_blocks = len(plan.dense)
    dd_out = {
        (g, h): np.zeros((plan.dense[g][1], plan.dense[h][1]))
        for g in range(n_blocks)
        for h in range(g, n_blocks)
    }
    first = [np.zeros(width) for (_, width, _) in plan.dense]
    dense_rhs = [np.zeros(width) for (_, width, _) in plan.dense]

    sweeps = _Sweeps(plan.n)
    for q, (a, b) in enumerate(plan.tab_pairs):
        sweeps.add(
            1.0,
            [tab_out[q]],
            lambda lo, hi, out, a=a, b=b: _hist_sweep(
                codes[a][lo:hi], codes[b][lo:hi], W[lo:hi], terms[b].raw.shape[0], out[0]
            ),
        )
    for q, (x, y) in enumerate(plan.acc_pairs):
        sweeps.add(
            1.0 + float(np.mean(terms[y].hi - terms[y].lo)),
            [acc_out[q]],
            lambda lo, hi, out, x=x, y=y, q=q: _acc_sweep(
                codes[x][lo:hi],
                codes[y][lo:hi],
                W[lo:hi],
                acc_read[q],
                terms[y].lo,
                terms[y].hi,
                terms[y].raw.shape[1],
                out[0],
            ),
        )
    for t in range(len(terms)):
        for g, (column, width, _) in enumerate(plan.dense):
            sweeps.add(
                1.0 + width,
                [dense_out[t][g]],
                lambda lo, hi, out, t=t, g=g, column=column, width=width: _dense_term_sweep(
                    codes[t][lo:hi],
                    W[lo:hi],
                    dense_values[g][lo:hi],
                    anchor[column : column + width],
                    out[0],
                ),
            )
    for (g, h), out in dd_out.items():
        cg, wg, _ = plan.dense[g]
        ch, wh, _ = plan.dense[h]
        # Only the diagonal pair accumulates ``first`` and ``rhs``, so only it owns them.
        outputs = [out, first[g], dense_rhs[g]] if g == h else [out]
        sweeps.add(
            1.0 + wg * wh,
            outputs,
            lambda lo, hi, parts, g=g, h=h, cg=cg, wg=wg, ch=ch, wh=wh: _dense_moments_sweep(
                W[lo:hi],
                Wz[lo:hi],
                dense_values[g][lo:hi],
                anchor[cg : cg + wg],
                dense_values[h][lo:hi],
                anchor[ch : ch + wh],
                parts[0],
                parts[1] if g == h else np.zeros(wg),
                parts[2] if g == h else np.zeros(wg),
                g == h,
            ),
        )
    largest = max([o.size for o in tab_out] + [o.size for o in acc_out] + [1])
    sweeps.run(8 * largest)

    # Every coefficient block is one product, written once to its own part of
    # the Gram (and its transpose), so the products run on the pool too.
    gram = np.zeros((p, p), dtype=np.float64)
    rhs = np.zeros(p, dtype=np.float64)
    blocks: list[tuple[float, Callable[[], None]]] = []

    def own_block(t: int) -> None:
        term, (wm, zm) = terms[t], margins[t]
        A, d = centred[t], shifts[t]
        own = _identity_gram(wm, d) if term.identity else (A * wm[:, None]).T @ A
        gram[np.ix_(term.columns, term.columns)] = own
        rhs[term.columns] = _left(term, A, d, zm[:, None])[:, 0]

    def pair_block(q: int, a: int, b: int) -> None:
        A, B = centred[a], centred[b]
        T = tab_out[q].reshape(A.shape[0], B.shape[0])
        block = _pair_block(terms[a], A, shifts[a], terms[b], B, shifts[b], T)
        gram[np.ix_(terms[a].columns, terms[b].columns)] = block
        gram[np.ix_(terms[b].columns, terms[a].columns)] = block.T

    def acc_block(q: int, x: int, y: int) -> None:
        # A raw window's centre enters through the row side's residual alone,
        # ``A_x' (m_x c_y') = e_x c_y'`` with ``e_x = A_x' W 1`` (module docstring).
        M = acc_out[q].reshape(terms[x].raw.shape[0], terms[y].raw.shape[1])
        block = _left(terms[x], centred[x], shifts[x], M)
        gram[np.ix_(terms[x].columns, terms[y].columns)] = block
        gram[np.ix_(terms[y].columns, terms[x].columns)] = block.T

    def dense_block(t: int, g: int) -> None:
        term = terms[t]
        column, width, _ = plan.dense[g]
        block = _left(
            term, centred[t], shifts[t], dense_out[t][g].reshape(term.raw.shape[0], width)
        )
        cols = np.arange(column, column + width)
        gram[np.ix_(term.columns, cols)] = block
        gram[np.ix_(cols, term.columns)] = block.T

    for t, term in enumerate(terms):
        rows, width = term.raw.shape
        blocks.append((float(rows * width * width), lambda t=t: own_block(t)))
    for q, (a, b) in enumerate(plan.tab_pairs):
        (ma, pa), (mb, pb) = terms[a].raw.shape, terms[b].raw.shape
        blocks.append(
            (
                float(ma * mb * min(pa, pb) + pa * pb * max(ma, mb)),
                lambda q=q, a=a, b=b: pair_block(q, a, b),
            )
        )
    for q, (x, y) in enumerate(plan.acc_pairs):
        (mx, px), py = terms[x].raw.shape, terms[y].raw.shape[1]
        blocks.append((float(mx * px * py), lambda q=q, x=x, y=y: acc_block(q, x, y)))
    for t, term in enumerate(terms):
        for g, (_, width, _) in enumerate(plan.dense):
            rows, pt = term.raw.shape
            blocks.append((float(rows * pt * width), lambda t=t, g=g: dense_block(t, g)))
    blocks.sort(key=lambda entry: -entry[0])
    _pool_run([job for _, job in blocks], 8 * max(term.raw.size for term in terms))
    first_all = np.zeros(p, dtype=np.float64)
    for (g, h), out in dd_out.items():
        cg, wg, _ = plan.dense[g]
        ch, wh, _ = plan.dense[h]
        gram[cg : cg + wg, ch : ch + wh] = out
        gram[ch : ch + wh, cg : cg + wg] = out.T
    for g, (column, width, _) in enumerate(plan.dense):
        first_all[column : column + width] = first[g]
        rhs[column : column + width] = dense_rhs[g]
    if not plan.dense:
        # An own-term block is a GEMM, ``(A W)' A``, whose triangles round apart.
        return mean_x, 0.5 * (gram + gram.T), rhs, mean_x.copy(), None
    mean_lo = np.where(dense_mask, first_all / sum_w, 0.0)
    gram = two_pass_centred_gram(first_all, gram, mean_lo, sum_w)
    rhs = rhs - mean_lo * float(np.dot(W, z_centered))
    return np.where(dense_mask, mean_hi + mean_lo, mean_hi), gram, rhs, mean_hi, mean_lo


# --------------------------------------------------------------------------- leverages


@dataclass(frozen=True)
class _LeverageTables:
    """One inverse's flat tables, in the gather kernel's layout."""

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


def _leverage_tables(plan: LookupPlan, V: NDArray, centre: NDArray) -> _LeverageTables:
    """Every table of the leverage gather for the inverse ``V`` (coefficient space only)."""
    terms = plan.terms
    tables, zero = _read_tables(plan, centre)
    Vz = V @ zero
    e_parts = []
    for term, A in zip(terms, tables, strict=True):
        # Every term meets the raw terms' centres, its own or another's.
        Vtt = V[np.ix_(term.columns, term.columns)]
        if term.identity and A is term.raw:
            e = np.append(np.diag(Vtt), 0.0)
        else:
            e = np.einsum("ij,ij->i", A @ Vtt, A)
        e -= 2.0 * _rows_times(term, A, Vz[term.columns][:, None])[:, 0]
        e_parts.append(e)
    tab_rows, G_parts, offset = [], [], 0
    for a, b in plan.tab_pairs:
        A, B = tables[a], tables[b]
        Vab = V[np.ix_(terms[a].columns, terms[b].columns)]
        G = _pair_table(terms[a], A, terms[b], B, 2.0 * Vab)
        tab_rows.append((a, b, offset, B.shape[0]))
        G_parts.append(G.ravel())
        offset += G.size
    acc_rows, Q_parts, offset = [], [], 0
    for x, y in plan.acc_pairs:
        Q = _rows_times(terms[x], tables[x], 2.0 * V[np.ix_(terms[x].columns, terms[y].columns)])
        acc_rows.append((x, y, offset, tables[y].shape[1]))
        Q_parts.append(Q.ravel())
        offset += Q.size
    dense = plan.dense_columns
    n_d = dense.size
    F_parts = [
        np.ascontiguousarray(_rows_times(term, A, 2.0 * V[np.ix_(term.columns, dense)])).ravel()
        for term, A in zip(terms, tables, strict=True)
    ]
    Vdd = 2.0 * V[np.ix_(dense, dense)]
    if n_d:
        Vdd[np.diag_indices(n_d)] = np.diag(V)[dense]
    rows = np.array([A.shape[0] for A in tables], dtype=np.int64)
    sizes = np.array([A.size for A in tables], dtype=np.int64)
    return _LeverageTables(
        h0=float(zero @ Vz),
        e=_flat(e_parts),
        e_off=np.cumsum(np.r_[0, rows])[:-1],
        tab=np.array(tab_rows, dtype=np.int64).reshape(-1, 4),
        G=_flat(G_parts),
        acc=np.array(acc_rows, dtype=np.int64).reshape(-1, 4),
        Q=_flat(Q_parts),
        A=_flat([np.ascontiguousarray(A).ravel() for A in tables]),
        A_off=np.cumsum(np.r_[0, sizes])[:-1],
        lo=_flat([term.lo for term in terms]).astype(np.int32),
        hi=_flat([term.hi for term in terms]).astype(np.int32),
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
    """``out[r] = h_r`` on ``[start, stop)``: one table at a time, each row in the fixed term order.

    A row adds ``h0``, its terms' diagonal entries, its table pairs, its
    accumulated pairs and its dense columns, in that order, whatever the
    block; rows without weight are set to zero at the end.
    """
    n_terms = codes.shape[0]
    n_dense = g.shape[0]
    for r in range(start, stop):
        out[r] = h0
    for t in range(n_terms):
        column = codes[t]
        base = e_off[t]
        for r in range(start, stop):
            out[r] += e[base + column[r]]
    for q in range(tab.shape[0]):
        ca = codes[tab[q, 0]]
        cb = codes[tab[q, 1]]
        base = tab[q, 2]
        mb = tab[q, 3]
        for r in range(start, stop):
            out[r] += G[base + ca[r] * mb + cb[r]]
    for q in range(acc.shape[0]):
        cx = codes[acc[q, 0]]
        y = acc[q, 1]
        cy = codes[y]
        width = acc[q, 3]
        base = acc[q, 2]
        a0 = A_off[y]
        w0 = w_off[y]
        for r in range(start, stop):
            k = cy[r]
            source = a0 + k * width
            target = base + cx[r] * width
            s = 0.0
            for c in range(lo[w0 + k], hi[w0 + k]):
                s += Q[target + c] * A[source + c]
            out[r] += s
    if n_dense:
        partial = np.empty(stop - start)
        for j in range(n_dense):
            for r in range(stop - start):
                partial[r] = g[j]
            for t in range(n_terms):
                column = codes[t]
                base = f_off[t]
                for r in range(start, stop):
                    partial[r - start] += F[base + column[r] * n_dense + j]
            for i in range(j):
                v = Vdd[i, j]
                for r in range(stop - start):
                    partial[r] += dense_values[r, i] * v
            v = Vdd[j, j]
            for r in range(stop - start):
                d = dense_values[r, j]
                out[start + r] += d * (partial[r] + d * v)
    for r in range(start, stop):
        if not active[r]:
            out[r] = 0.0


_POOLED_ROW_KERNELS = (
    _gather_leverages,
    _margins_sweep,
    _hist_sweep,
    _acc_sweep,
    _dense_term_sweep,
    _dense_moments_sweep,
)
"""The kernels a pooled job runs on its worker: ``nogil``, never ``prange``, each writing only
the output its job owns."""


def _dense_block(plan: LookupPlan, start: int, stop: int, centre_hi, centre_lo) -> NDArray:
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
    plan: LookupPlan,
    inverse: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
    active: NDArray,
) -> NDArray:
    """``h_r = x_r' inverse x_r`` for the active rows (``0`` elsewhere), read back from the tables.

    Rows run in fixed blocks of ``_BLOCK_ROWS`` on ``pool_workers`` threads
    (BLAS held at one thread meanwhile); the result is bitwise the serial one.
    """
    # ``centre_lo`` is nonzero on dense columns only, which no table reads.
    centre = np.asarray(centre_hi, dtype=np.float64)
    tables = _leverage_tables(plan, np.ascontiguousarray(inverse, dtype=np.float64), centre)
    active = np.ascontiguousarray(active, dtype=np.bool_)
    out = np.empty(plan.n, dtype=np.float64)

    def job(start: int):
        stop = min(start + _BLOCK_ROWS, plan.n)
        return lambda: _gather_leverages(
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

    _pool_run(
        [job(start) for start in range(0, plan.n, _BLOCK_ROWS)],
        8 * _BLOCK_ROWS * (2 + int(plan.dense_columns.size)),
    )
    return out


@njit(cache=True)
def _scatter_margins(
    start, stop, codes, values, m_off, total, compensation, dense_values, dense_total, dense_comp
):  # pragma: no cover - compiled
    """Add ``values[r]`` into each term's bin and ``d_r values[r]`` into each dense sum (Neumaier)."""
    n_terms = codes.shape[0]
    n_dense = dense_total.shape[0]
    for r in range(start, stop):
        v = values[r]
        if v == 0.0:
            continue
        for t in range(n_terms):
            k = m_off[t] + codes[t, r]
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


def table_rmatvec(
    plan: LookupPlan,
    values: NDArray,
    centre_hi: NDArray,
    centre_lo: NDArray | None,
) -> NDArray:
    """``X_c' values`` for the centred design, serially in row order.

    A term's share is ``A_t' M_t - z_t sum(values)`` with ``M_t`` its bins'
    compensated sums (a centred term's table is centred, ``z_t = 0``); a
    dense column's is the compensated sum of ``d_r values_r``.
    """
    # ``centre_lo`` is nonzero on dense columns only, which no table reads.
    centre = np.asarray(centre_hi, dtype=np.float64)
    tables, zero = _read_tables(plan, centre)
    sizes = np.array([A.shape[0] for A in tables], dtype=np.int64)
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
    for t, (term, A) in enumerate(zip(plan.terms, tables, strict=True)):
        share = A.T @ margins[m_off[t] : m_off[t] + sizes[t]]
        if term.raw_read:
            share = share - zero[term.columns] * value_sum
        out[term.columns] = share
    if n_d:
        out[plan.dense_columns] = dense_total + dense_comp
    return out


def _warmup_lookup_tables() -> None:
    """Compile the table sweeps and gathers for the operands an exact-path fit passes.

    Empty operands of the fit's dtypes, layouts and flags: the kernels compile
    without reading a row.  Term codes are int32.
    """
    i32, i64 = np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    f64 = np.zeros(0, dtype=np.float64)
    codes, f64_2d = np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float64)
    i64_2d = np.zeros((0, 0), dtype=np.int64)
    frozen = np.zeros((0, 0), dtype=np.float64)
    frozen.setflags(write=False)
    _margins_sweep(i32, f64, f64, f64, f64)
    _hist_sweep(i32, i32, f64, 0, f64)
    _acc_sweep(i32, i32, f64, f64, i32, i32, 0, f64)
    # Dense blocks arrive read-only from the design, writable from estimate_p's refits.
    for block in (frozen, f64_2d):
        _dense_term_sweep(i32, f64, block, f64, f64)
        _dense_moments_sweep(f64, f64, block, f64, block, f64, f64_2d, f64, f64, True)
    _gather_leverages(
        0, 0, codes, np.zeros(0, dtype=np.bool_), 0.0, f64, i64, i64_2d, f64, i64_2d, f64,
        f64, i64, i32, i32, i64, f64_2d, f64, i64, f64, f64_2d, f64,
    )  # fmt: skip
    _scatter_margins(0, 0, codes, f64, i64, f64, f64, f64_2d, f64, f64)
