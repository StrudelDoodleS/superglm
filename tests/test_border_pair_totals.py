"""A random effect in the structured solver's border, summed by (leaf, level) pair.

Within a leaf of the eliminated effect, the rows that take one level of a
border random effect (a pair) share their centred indicator row, so the row
pass adds each row once to its pair and forms every block from the pair sums
(``moments._indicator_pairs``, ``leaf_kernels``).  These tests hold that form
to the row sums it regroups: exact rational sums of the same rows, the
previous row expansion (``tests._row_expanded_indicators``) and the exact
zeros and ones the leaf means carry.  Section numbers refer to
``notes/research/2026-09-26-nested-random-effect-elimination.md``.
"""

from __future__ import annotations

import tracemalloc
import warnings
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from superglm import RandomEffect, Spline, SuperGLM
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix, RandomEffectGroupMatrix
from superglm.inference.covariance import covariance_selected_diagonal
from superglm.solvers._structured import moments
from superglm.solvers._structured.layout import get_structured_layout
from superglm.solvers._structured.nested import ProfiledNestedSchurFactor
from superglm.types import GroupSlice
from tests import _row_expanded_indicators as row_expanded
from tests.test_nested_structured_plumbing import (
    _border_center,
    _border_rows,
    _indicator_case,
    _layout,
    _nested_case,
    build_nested_leaf_statistics,
)

EPS = np.finfo(np.float64).eps
LEAVES, FIRST, SECOND = 12, 9, 6
# Border columns of ``_edge_case``: two dense columns, then the two crossed effects.
FIRST_COLUMNS, SECOND_COLUMNS = np.arange(2, 2 + FIRST), np.arange(2 + FIRST, 2 + FIRST + SECOND)
DOMINATED = 1e-13


def _edge_case(seed: int = 20261124):
    """A lone random effect of 12 leaves beside two dense columns and two crossed random effects.

    By leaf, on the first crossed effect: 0 has no weighted row; 1 has one
    row; every row of 2 takes level 3; the weighted rows of 3 take level 5
    and two zero-weight rows take levels 1 and 2; in 4 level 7, which no
    other leaf takes, carries all but ``DOMINATED`` of the weight; 5 to 11
    mix levels 0 to 6.  Level 8 is never taken.  The second effect mixes
    freely, and the second dense column is constant within leaves.
    """
    rng = np.random.default_rng(seed)
    sizes = [6, 1, 12, 14, 20] + [40] * (LEAVES - 5)
    leaf = np.repeat(np.arange(LEAVES), sizes)
    n = len(leaf)
    first = rng.integers(0, FIRST - 2, n)
    weights = np.exp(rng.normal(size=n))
    weights[leaf == 0] = 0.0
    first[leaf == 2] = 3
    rows = np.flatnonzero(leaf == 3)
    first[rows], first[rows[[4, 9]]] = 5, (1, 2)
    weights[rows[[4, 9]]] = 0.0
    rows = np.flatnonzero(leaf == 4)
    first[rows[::2]], first[rows[1::2]] = FIRST - 2, np.arange(10) % 6
    weights[rows[1::2]] *= DOMINATED
    weights[rng.choice(np.flatnonzero(leaf >= 5), 12, replace=False)] = 0.0
    order = rng.permutation(n)
    leaf, first, weights = leaf[order], first[order], weights[order]
    dense = np.column_stack([rng.normal(size=n), rng.normal(size=LEAVES)[leaf]])
    matrices = [
        DenseGroupMatrix(dense),
        RandomEffectGroupMatrix(first, FIRST),
        RandomEffectGroupMatrix(rng.integers(0, SECOND, n), SECOND),
        RandomEffectGroupMatrix(leaf, LEAVES),
    ]
    groups, start = [], 0
    for index, matrix in enumerate(matrices):
        groups.append(GroupSlice(name=f"g{index}", start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n=n, p=start)
    layout = get_structured_layout(dm, groups, dominant_group_index=3, chain_group_indices=(3,))
    assert [cells.block for cells in layout.sparse_indicators] == [1, 2]
    return layout, matrices, leaf, weights


def _design(leaf, indicators, dense=None):
    """A lone random effect ``leaf`` beside dense columns and crossed effects ``(codes, levels)``."""
    matrices = [] if dense is None else [DenseGroupMatrix(dense)]
    matrices += [RandomEffectGroupMatrix(codes, levels) for codes, levels in indicators]
    matrices.append(RandomEffectGroupMatrix(leaf, int(leaf.max()) + 1))
    groups, start = [], 0
    for index, matrix in enumerate(matrices):
        groups.append(GroupSlice(name=f"g{index}", start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n=len(leaf), p=start)
    last = len(matrices) - 1
    layout = get_structured_layout(
        dm, groups, dominant_group_index=last, chain_group_indices=(last,)
    )
    assert len(layout.sparse_indicators) == len(indicators)
    return layout, matrices


def _passes(layout, matrices, weights, chunk_size):
    """The data pass, a data pass with negative rows and a signed pass about the data means."""
    rng = np.random.default_rng(7)
    data = build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk_size)
    observed = np.where(rng.random(len(weights)) < 0.15, -0.3, 1.0) * weights
    signed = rng.normal(size=len(weights)) * weights
    return data, ((weights, None), (observed, None), (signed, data.mean))


def _exact_scatter(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``sum_r a_r d_r d_r'`` of the float64 rows in exact rational arithmetic."""
    n, q = d.shape
    rows = [[Fraction(float(v)) for v in row] for row in d]
    weights = [Fraction(float(v)) for v in a]
    exact = np.empty((q, q))
    for i in range(q):
        for j in range(i, q):
            total = sum((weights[r] * rows[r][i] * rows[r][j] for r in range(n)), Fraction(0))
            exact[i, j] = exact[j, i] = float(total)
    return exact


def _exact_leaf_sums(d: np.ndarray, a: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    """``sum_{r in l} a_r d_r`` ``(LEAVES, q)`` in exact rational arithmetic."""
    exact = [[Fraction(0)] * d.shape[1] for _ in range(LEAVES)]
    for row, weight, cell in zip(d, a, leaf, strict=True):
        for j, value in enumerate(row):
            exact[cell][j] += Fraction(float(weight)) * Fraction(float(value))
    return np.array([[float(value) for value in row] for row in exact])


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("chunk_size", [5, 64, 10_000])
def test_pair_sums_are_within_the_row_sums_bound_of_the_exact_scatter(chunk_size) -> None:
    """Every entry is within ``gamma sum_r |a_r| |d_ri| |d_rj|`` of the exact sum over rows.

    ``d = x - c - m`` with the stored means, as
    ``test_signed_pass_scatter_matches_exact_rational_sums`` holds the dense
    columns: within ``(n + 5 + 2 chunks) eps`` of the scatter of the rounded
    ``d``, whose entries are within ``delta = 2 eps (|x - c| + |d|)`` of the
    exact ones; an indicator column has ``c = 0``, so its ``d = x - m`` takes
    one rounding, ``delta = eps |d|``.  The bound is the absolute mass the
    factor charges the pass with (§8), entry by entry, so it holds where a pair
    dominates its leaf: in leaf 4 the entries of level 7 are 1e-13 of its
    weight, and a form that subtracts leaf totals (``diag(c) - c m' - m c' + A
    m m'``, or the other rows' sums as the total less the pair's) misses them
    by ``eps`` times the weight.  A data pass is held uncut: a cut leaf's pieces
    are centred on their own means, which this bound does not describe.  The
    absolute mass takes a caller's error scale that charges the rows of zero
    weight and the leaf with none, and the deviation ``sum_r a_r d_r`` (a
    signed pass) or ``-2 sum_{a_r < 0} |a_r| d_r`` (a data pass) is held to
    ``test_signed_pass_deviation_matches_exact_rational_sums``'s bound.
    """
    layout, matrices, leaf, weights = _edge_case()
    n = len(leaf)
    data, passes = _passes(layout, matrices, weights, chunk_size)
    X, c = _border_rows(layout), _border_center(layout)
    chunks = -(-n // chunk_size)
    for a, mean in passes:
        if mean is None and chunk_size < n:
            continue
        error = np.abs(a) + 0.5
        stats = build_nested_leaf_statistics(
            layout,
            matrices,
            a,
            mean=mean,
            center=None if mean is None else data.center,
            chunk_size=chunk_size,
            error=error,
        )
        d = X - c - stats.mean[leaf]
        exact = _exact_scatter(d, a)
        delta = 2 * EPS * (np.abs(X - c) + np.abs(d))
        delta[:, 2:] = EPS * np.abs(d[:, 2:])
        weight = np.abs(a)[:, None]
        rounded = (np.abs(d) + delta).T @ (weight * (np.abs(d) + delta))
        moved = np.abs(d).T @ (weight * delta)
        moved = moved + moved.T + delta.T @ (weight * delta)
        bound = (n + 5 + 2 * chunks) * EPS * rounded + moved
        assert np.all(np.abs(stats.within - exact) <= bound)
        np.testing.assert_array_equal(stats.within, stats.within.T)
        # the dominated pair's own entry is far below one rounding of its leaf's weight
        level, total = FIRST_COLUMNS[FIRST - 2], np.sum(np.abs(a)[leaf == 4])
        assert bound[level, level] < 1e-6 * EPS * total
        squares = error @ d**2
        slack = error @ (2.0 * np.abs(d) * delta + delta**2)
        assert np.all(np.abs(stats.absolute - squares) <= (n + 4) * EPS * squares + slack)
        if stats.deviation is None:
            assert mean is None and not np.any(a < 0.0)
            continue
        carried = a if mean is not None else 2.0 * np.minimum(a, 0.0)
        scale = np.zeros((LEAVES, X.shape[1]))
        np.add.at(scale, leaf, np.abs(carried)[:, None] * (np.abs(d) + np.abs(X - c)))
        difference = np.abs(stats.deviation - _exact_leaf_sums(d, carried, leaf))
        assert np.all(difference <= (np.bincount(leaf).max() + 4) * EPS * scale)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("chunk_size", [5, 64, 10_000])
def test_a_leaf_that_meets_one_border_level_keeps_exact_zeros_and_ones(chunk_size) -> None:
    """Leaf 2 (every row on level 3) and leaf 3 (every weighted row on level 5).

    The leaf mean of that level is exactly 1 and of every other level exactly
    0 (the shifted form, §3.4), so each weighted row's centred indicator row
    is exactly zero: the leaf adds exact zeros to the scatter, the deviation
    and, charged with ``|a|``, the absolute mass.  With the two leaves alone
    carrying weight every entry of the first effect's block is an exact zero;
    read from the pairs' own weights (``c / W``) the means would be 1 only to
    rounding.  A leaf with one row, or none of any weight, adds exact zeros too.
    """
    layout, matrices, leaf, weights = _edge_case()
    only = np.where(np.isin(leaf, (1, 2, 3)), weights, 0.0)
    data, passes = _passes(layout, matrices, only, chunk_size)
    expected = np.zeros((LEAVES, FIRST))
    expected[2, 3] = expected[3, 5] = 1.0
    expected[1, matrices[1].codes[leaf == 1][0]] = 1.0
    for a, mean in passes:
        stats = build_nested_leaf_statistics(
            layout,
            matrices,
            a,
            mean=mean,
            center=None if mean is None else data.center,
            chunk_size=chunk_size,
        )
        np.testing.assert_array_equal(stats.mean[:, FIRST_COLUMNS], expected)
        np.testing.assert_array_equal(stats.within[FIRST_COLUMNS], 0.0)
        np.testing.assert_array_equal(stats.absolute[FIRST_COLUMNS], 0.0)
        if stats.deviation is not None:
            np.testing.assert_array_equal(stats.deviation[:, FIRST_COLUMNS], 0.0)
        # the second effect mixes within those leaves: its own block is not zero
        assert np.any(stats.absolute[SECOND_COLUMNS] > 0.0)


@pytest.mark.parametrize("case", [pytest.param("data", marks=pytest.mark.pins_behaviour), "signed"])
def test_a_weight_meets_one_mean_before_the_other(case) -> None:
    """Weights of ``2**600`` and ``1e180`` beside leaf means near their reciprocals.

    A row off levels ``j`` and ``k`` adds ``a m_j m_k``.  The product of the
    two means underflows where the entry is an ordinary number, so the weight
    meets one mean first, as the row sum's ``(a d_j) d_k`` does.  Data pass:
    one leaf with weights ``2**600, 1, 2`` on three levels, whose (1, 2) entry
    is ``2**-599 (1 - 1 - 1)``; with the means multiplied first it is
    ``-2**-598``.  Signed pass: a row of weight ``-1e180`` about stored means
    ``1e-180`` and ``2e-180``, whose block ``-1e180 m m'`` would be four
    zeros.  Both are held to the exact rational sums over rows within the row
    sum's bound, and the absolute mass, formed the same way, to its own exact
    sum, so it still bounds the diagonal.  The data pass is a control: the row
    expansion multiplied the weight first too.  Its absolute mass squared the
    mean first and read zero on the signed pass.
    """
    leaf = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3])
    layout, matrices = _design(leaf, [(np.array([0, 1, 2, 0, 2, 1, 2, 0, 1]), 3)])
    n, a, options = len(leaf), np.zeros(len(leaf)), {}
    if case == "data":
        a[:3] = 2.0**600, 1.0, 2.0
    else:
        stored = np.array([1.0, 1e-180, 2e-180] + [1.0] * 6)
        data = build_nested_leaf_statistics(layout, matrices, stored)
        np.testing.assert_array_equal(data.mean[0], stored[:3])
        a[0], options = -1e180, dict(mean=data.mean, center=data.center)
    stats = build_nested_leaf_statistics(layout, matrices, a, **options)
    d = _border_rows(layout) - stats.mean[leaf]
    exact = _exact_scatter(d, a)
    assert np.all(exact[1:, 1:] != 0.0)
    bound = (n + 7) * EPS * (np.abs(d).T @ (np.abs(a)[:, None] * np.abs(d)))
    assert np.all(np.abs(stats.within - exact) <= bound)
    squares = np.array(
        [
            float(sum(Fraction(abs(w)) * Fraction(v) ** 2 for w, v in zip(a, column, strict=True)))
            for column in d.T
        ]
    )
    assert np.all(np.abs(stats.absolute - squares) <= (n + 4) * EPS * squares)
    assert np.all(np.abs(np.diag(stats.within)) <= (1 + 2 * n * EPS) * stats.absolute)


@pytest.mark.parametrize("case", ["edge", "indicator", "chain"])
@pytest.mark.parametrize("chunk_size", [7, 64, 3000])
def test_pair_sums_give_the_expanded_rows_statistics(case, chunk_size, monkeypatch) -> None:
    """The pass from pair sums against the row expansion it replaced, on the same rows.

    Three designs: the edge leaves above with two crossed effects (their
    cross block is summed from the joint pairs), four indicator blocks beside
    categoricals (``_indicator_case``), and a three-level nested chain with a
    crossed effect in its border (``_nested_case``).  Both sum the same
    terms, grouped differently, so they differ by twice
    ``test_sparse_indicator_blocks_give_the_dense_pass_statistics``'s
    ``gamma`` bound; the means come from the same code and are equal.  A
    level absent from a chunk, a cut leaf, zero-weight rows and negative
    rows are all here, under a caller's error scale that charges every row:
    a cut leaf's pieces then carry an error residual that is not zero.  The
    absolute mass is also held to the dense pass's, which forms every
    indicator column as dense rows and shares none of the pair bookkeeping.
    """
    if case == "edge":
        layout, matrices, leaf, weights = _edge_case()
    elif case == "indicator":
        dm, groups = _indicator_case()
        matrices = list(dm.group_matrices)
        layout = get_structured_layout(dm, groups, dominant_group_index=6, chain_group_indices=(6,))
        leaf = matrices[6].codes
        weights = np.exp(np.random.default_rng(3).normal(size=dm.n))
        weights[np.isin(leaf, (3, 7))] = 0.0
    else:
        nested = _nested_case()
        layout, matrices, leaf, weights = (
            _layout(nested),
            nested.matrices,
            nested.codes[2],
            nested.weights,
        )
    n, K = len(leaf), layout.tree.sizes[-1]
    X = _border_rows(layout) - _border_center(layout)
    data, passes = _passes(layout, matrices, weights, chunk_size)
    gamma = (6 * n + 4 * -(-n // chunk_size) + 32) * EPS
    dense = replace(layout)
    dense.__dict__["sparse_indicators"] = ()
    for a, mean in passes:
        error = np.abs(a) + 0.5
        options = dict(mean=mean, center=None if mean is None else data.center, error=error)
        pairs = build_nested_leaf_statistics(layout, matrices, a, chunk_size=chunk_size, **options)
        with monkeypatch.context() as patch:
            written = row_expanded.install(patch)
            rows = build_nested_leaf_statistics(
                layout, matrices, a, chunk_size=chunk_size, **options
            )
        assert sum(written) > n
        deviation = np.zeros((K, X.shape[1]))
        np.maximum.at(deviation, leaf, np.abs(X - rows.mean[leaf]))
        M = 2.0 * deviation + np.abs(rows.mean)
        A = np.bincount(leaf, weights=np.abs(a), minlength=K)
        np.testing.assert_array_equal(pairs.mean, rows.mean)
        assert np.all(np.abs(pairs.within - rows.within) <= 2 * gamma * (M.T @ (A[:, None] * M)))
        E = np.bincount(leaf, weights=error, minlength=K)
        assert np.all(np.abs(pairs.absolute - rows.absolute) <= 2 * gamma * (E @ M**2))
        reference = build_nested_leaf_statistics(
            dense, matrices, a, chunk_size=chunk_size, **options
        )
        assert np.all(np.abs(pairs.absolute - reference.absolute) <= 2 * gamma * (E @ M**2))
        assert (pairs.deviation is None) == (rows.deviation is None)
        if rows.deviation is not None:
            bound = 2 * gamma * A[:, None] * M
            assert np.all(np.abs(pairs.deviation - rows.deviation) <= bound)


def test_the_pass_adds_each_row_once_and_makes_no_sparse_product(monkeypatch) -> None:
    """Work of the indicator blocks: one entry a row, whatever the levels a leaf meets.

    40 leaves of 150 rows each meet all 60 levels of one crossed effect and
    all 50 of another.  The pair kernel is handed every row once per block,
    and the pass builds no sparse matrix.  The row expansion wrote each row
    once for each level its leaf meets: 110 entries a row here.
    """
    rng = np.random.default_rng(11)
    leaves, per_leaf, levels = 40, 150, (60, 50)
    n = leaves * per_leaf
    leaf = np.repeat(np.arange(leaves), per_leaf)
    crossed = [np.tile(rng.permutation(per_leaf) % k, leaves) for k in levels]
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        *(RandomEffectGroupMatrix(codes, k) for codes, k in zip(crossed, levels, strict=True)),
        RandomEffectGroupMatrix(leaf, leaves),
    ]
    groups, start = [], 0
    for index, matrix in enumerate(matrices):
        groups.append(GroupSlice(name=f"g{index}", start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n=n, p=start)
    layout = get_structured_layout(dm, groups, dominant_group_index=3, chain_group_indices=(3,))
    weights = rng.exponential(size=n)
    build_nested_leaf_statistics(layout, matrices, weights, chunk_size=1024)  # the cached cells
    handed: list[int] = []
    kernel = moments._pair_sums

    def counted(own, *rest):
        handed.append(len(own))
        return kernel(own, *rest)

    def refused(*args, **kwargs):
        raise AssertionError("the row pass built a sparse matrix")

    with monkeypatch.context() as patch:
        patch.setattr(moments, "_pair_sums", counted)
        for name in ("csr_array", "csc_array", "coo_array", "csr_matrix", "csc_matrix"):
            patch.setattr(scipy.sparse, name, refused)
        build_nested_leaf_statistics(layout, matrices, weights, chunk_size=1024)
    assert sum(handed) == len(levels) * n
    written = row_expanded.install(monkeypatch)
    build_nested_leaf_statistics(layout, matrices, weights, chunk_size=1024)
    assert sum(written) == sum(levels) * n


@pytest.mark.parametrize("design, chunk_size", [("cut", 16), ("cut", 8192), ("short", 8192)])
def test_the_pair_scratch_is_sized_by_the_chunk_not_by_a_leafs_levels(
    design, chunk_size, monkeypatch
) -> None:
    """Traced bytes of the indicator step beyond the scatter it returns, chunk by chunk.

    ``cut``: one leaf of 768 rows, each on its own level of a 768-level
    effect, beside a 2-level effect and 112 dense columns.  A 16-row chunk
    takes 16 of that leaf's 768 pairs and sums those alone (a signed pass
    reads the other 752 from the stored mean), and with the leaf in one chunk
    the two effects' joint table is 769 by 3.  ``short``: 4,096 leaves of two
    rows on three levels beside 90 dense columns.  The step holds the pair
    sums (a row for each pair the chunk's rows take, at most ``rows + 1``) and
    the scratch that centres them (the longest segment's pairs), ``width + 3``
    wide; three joint tables of the longest segment; and vectors of the
    chunk's rows and the border's width, allowed 32 of each.  Fails with a
    dense row per cell of a touched leaf (769 by 115 for the 16-row chunk), a
    spare row for every segment (``short``), or joint tables as squares of
    the larger side (three of 769 by 769).  Byte counts, never wall time.
    """
    rng = np.random.default_rng(90)
    if design == "cut":
        levels, width = (768, 2), 112
        leaf = np.r_[np.zeros(768, dtype=np.intp), 1 + np.arange(12)]
        crossed = [np.r_[np.arange(768), np.arange(12)], np.arange(len(leaf)) % 2]
    else:
        levels, width = (3,), 90
        leaf = np.repeat(np.arange(4096), 2)
        crossed = [rng.integers(0, 3, len(leaf))]
    n, q = len(leaf), width + sum(levels)
    layout, matrices = _design(
        leaf, list(zip(crossed, levels, strict=True)), rng.normal(size=(n, width))
    )
    weights = rng.uniform(0.3, 2.0, n)
    data = build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk_size)
    passes = (
        (weights, {}),
        (rng.normal(size=n) * weights, dict(mean=data.mean, center=data.center)),
    )
    step, excess = moments._indicator_pairs, []

    def traced(*args):
        held = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        scatter, *pairs = step(*args)
        rows, longest = len(args[6]), int(np.max(np.diff(args[5])))
        table = np.prod([min(longest, k) + 1 for k in levels]) if len(levels) > 1 else 0
        allowed = (rows + 1 + min(longest, max(levels))) * (width + 3) + 3 * table + 32 * (rows + q)
        used = tracemalloc.get_traced_memory()[1] - held - scatter.nbytes
        excess.append(used - 8 * allowed)
        return scatter, *pairs

    monkeypatch.setattr(moments, "_indicator_pairs", traced)
    for a, options in passes:  # the kernels compile, or load, outside the trace
        build_nested_leaf_statistics(layout, matrices, a, chunk_size=chunk_size, **options)
    excess.clear()
    tracemalloc.start()
    try:
        for a, options in passes:
            build_nested_leaf_statistics(layout, matrices, a, chunk_size=chunk_size, **options)
    finally:
        tracemalloc.stop()
    assert excess and max(excess) <= 0


@pytest.mark.parametrize("chain", ["lone", "nested"])
def test_a_border_without_a_random_effect_never_enters_the_pair_kernels(chain, monkeypatch) -> None:
    """One random effect, or a nested chain alone, beside splines and categoricals.

    No border block is a random effect (a chain's parents leave the border),
    so no pair kernel runs, and with them refused the statistics are the
    same bits.
    """
    case = _nested_case()
    keep = [i for i, m in enumerate(case.matrices) if not isinstance(m, RandomEffectGroupMatrix)]
    members = (1, 3, 7) if chain == "nested" else (7,)
    matrices = [case.matrices[i] for i in (*keep, *members)]
    groups, start = [], 0
    for index, matrix in enumerate(matrices):
        groups.append(GroupSlice(name=f"g{index}", start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n=case.dm.n, p=start)
    chain_indices = tuple(range(len(keep), len(matrices)))
    layout = get_structured_layout(
        dm, groups, dominant_group_index=chain_indices[-1], chain_group_indices=chain_indices
    )
    assert not layout.sparse_indicators
    signed = np.random.default_rng(2).normal(size=case.dm.n) * case.weights

    def statistics():
        data = build_nested_leaf_statistics(layout, matrices, case.weights, chunk_size=97)
        return data, build_nested_leaf_statistics(
            layout, matrices, signed, mean=data.mean, center=data.center, chunk_size=97
        )

    free = statistics()

    def refused(*args):
        raise AssertionError("a pair kernel ran without a random effect in the border")

    kernels = ("_pair_sums", "_center_pairs", "_pair_block", "_crossed_blocks", "_absent_levels")
    for name in kernels:
        monkeypatch.setattr(moments, name, refused)
    for left, right in zip(free, statistics(), strict=True):
        for name in ("mean", "within", "absolute", "deviation"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    # the same chain with its crossed effect in the border does enter them
    with pytest.raises(AssertionError, match="pair kernel"):
        build_nested_leaf_statistics(_layout(case), case.matrices, case.weights)


def _crossed_frame(seed: int = 5, n: int = 2400):
    """A Poisson response on one spline, a 60-level effect nested in a 10-level one, and
    two effects crossed with both, one of which most leaves meet on a single level."""
    rng = np.random.default_rng(seed)
    leaf = rng.integers(0, 60, n)
    partner = np.where(rng.random(n) < 0.9, leaf % 7, rng.integers(0, 7, n))
    free = rng.integers(0, 9, n)
    x = rng.uniform(0.0, 1.0, n)
    exposure = rng.uniform(0.2, 1.0, n)
    eta = (
        np.sin(3.0 * x)
        + rng.normal(0.0, 0.3, 60)[leaf]
        + rng.normal(0.0, 0.2, 7)[partner]
        + rng.normal(0.0, 0.2, 9)[free]
    )
    frame = pd.DataFrame(
        {
            "x": x,
            "leaf": np.array([f"l{v:02d}" for v in leaf], dtype=object),
            "group": np.array([f"g{v}" for v in leaf // 6], dtype=object),
            "partner": np.array([f"p{v}" for v in partner], dtype=object),
            "free": np.array([f"f{v}" for v in free], dtype=object),
        }
    )
    return frame, rng.poisson(exposure * np.exp(eta)) / exposure, exposure


def _crossed_fit(discrete: bool):
    frame, y, exposure = _crossed_frame()
    model = SuperGLM(
        family="poisson",
        features={
            "x": Spline(kind="cr", k=8),
            "leaf": RandomEffect(),
            "group": RandomEffect(),
            "partner": RandomEffect(),
            "free": RandomEffect(),
        },
        discrete=discrete,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=exposure)
    factor = model._linear_system_state.profiled_factor
    assert isinstance(factor, ProfiledNestedSchurFactor)
    assert factor.chain_group_names == ("group", "leaf")
    return model


@pytest.mark.parametrize("discrete", [True, False])
def test_a_fit_with_crossed_border_effects_is_the_expanded_rows_fit(discrete, monkeypatch) -> None:
    """``fit_reml`` on the pair sums against the same fit on the expanded rows.

    A nested chain of two levels with two crossed effects in its border,
    binned and exact (whose weight-derivative passes are signed, about the
    factor's means).  The two passes differ by rounding in the border Schur
    complement ``Q``: ``gamma_Q = (n + k + q + 10) eps`` componentwise (§8),
    and one pass through ``Q^-1`` multiplies it by ``q kappa_s(Q)``.  Both
    fits take the same steps, so what a user reads moves by that
    ``gamma_border`` per inner solve: the bound charges every solve of the
    fit and is still below 1e-7.
    """
    pairs = _crossed_fit(discrete)
    with monkeypatch.context() as patch:
        written = row_expanded.install(patch)
        rows = _crossed_fit(discrete)
    assert written
    profile = pairs._reml_profile
    n, p = pairs._dm.n, pairs._dm.p
    nodes = 60 + 10
    q = p + 1 - nodes
    kappa = profile["structured_schur_condition"]
    solves = profile["irls_iters"] + profile["n_reml_iter"]
    gamma_border = solves * q * kappa * (n + nodes + q + 10) * EPS
    assert np.isfinite(kappa) and gamma_border < 1e-7
    assert pairs.result.converged and rows.result.converged
    for name in ("n_reml_iter", "irls_iters", "structured_border_rank"):
        assert pairs._reml_profile[name] == rows._reml_profile[name]
    scale = np.max(np.abs(rows.result.beta))
    assert np.max(np.abs(pairs.result.beta - rows.result.beta)) <= gamma_border * scale
    for name, value in rows._reml_lambdas.items():
        assert abs(pairs._reml_lambdas[name] - value) <= gamma_border * value
    assert abs(pairs.result.deviance - rows.result.deviance) <= gamma_border * rows.result.deviance
    edf = rows.result.effective_df
    assert abs(pairs.result.effective_df - edf) <= gamma_border * edf
    index = np.arange(p)
    variances = [
        covariance_selected_diagonal(model._coef_covariance[0], index) for model in (pairs, rows)
    ]
    assert np.all(np.abs(variances[0] - variances[1]) <= gamma_border * variances[1])
