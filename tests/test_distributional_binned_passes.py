"""Binned derivative-pass kernels equal the dense row chunks they replace, reading no dense rows."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.distributional import GaussianLS, TweedieLSS
from superglm.distributional._row_design import BoundedPredictorMatrix
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.smoothing.derivatives import _cross_all, _leverage_blocks
from superglm.distributional.solver import binned_passes as bp
from superglm.distributional.solver._ordinary_packing import predictor_panels
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.features import Categorical, Numeric, RandomEffect, Spline, SplineCategorical
from superglm.features.interaction import TensorInteraction
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior

U = 2.0**-53
N = 640


def _layout(k_parameters: int, discrete: bool = True):
    rng = np.random.default_rng(8810 + k_parameters)
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": rng.uniform(-1, 1, N),
                "z": rng.uniform(-1, 1, N),
                "d": rng.normal(size=N),
                "g": rng.choice(list("abc"), N),
                "h": rng.choice(list("pqrs"), N),
                "r": rng.choice([f"l{i}" for i in range(23)], N),
            }
        )
    )
    family = GaussianLS() if k_parameters == 2 else TweedieLSS()
    names = [parameter.name for parameter in family.parameters]
    predictors = [
        Predictor(
            names[0],
            {
                "x": Spline(kind="cr", n_knots=6),
                "z": Spline(kind="cr", n_knots=5),
                "g": Categorical(base="a"),
                "d": Numeric(),
                "r": RandomEffect(),
            },
            interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        Predictor(
            names[1],
            {"x": Spline(kind="cr", n_knots=5), "h": Categorical(base="p"), "d": Numeric()},
            interaction_specs={"x:h": SplineCategorical("x", "h")},
        ),
    ]
    if k_parameters == 3:
        predictors.append(Predictor(names[2], {"z": Spline(kind="cr", n_knots=4)}))
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(N)),
        family.parameters,
        predictors,
        model_discrete=discrete,
        n_bins_config=24,
    )
    layout = build_stacked_layout(compiled)
    matrices = dense_predictor_matrices(layout)
    return layout, matrices, tuple(state.coefficient_slice for state in layout.predictors), rng


def _excess(got, reference, magnitude, width):
    """How far ``got`` lies outside ``gamma``-style bounds ``8 P u`` times the magnitude."""
    return float(np.max(np.abs(got - reference) - 8 * width * U * magnitude, initial=-np.inf))


@pytest.fixture(params=[2, 3], ids=["K2", "K3"])
def problem(request):
    return _layout(request.param)


@pytest.fixture
def split_blocks(monkeypatch):
    """Blocks of 97 and 37 rows: every bin's rows split across blocks."""
    monkeypatch.setattr(bp, "_LEVERAGE_ROWS", 37)
    monkeypatch.setattr(bp, "contraction_block_rows", lambda n, row_bytes, chunk: 97)


def test_fixture_covers_every_group_kind_and_packs_panels(problem):
    layout = problem[0]
    kinds = {type(g).__name__ for state in layout.predictors for g in state.design.group_matrices}
    assert kinds == {
        "DiscretizedSSPGroupMatrix",
        "DiscretizedTensorGroupMatrix",
        "DiscretizedSplineCategoricalGroupMatrix",
        "CategoricalGroupMatrix",
        "RandomEffectGroupMatrix",
        "DenseGroupMatrix",
    }
    assert bp.binned_layout(layout)
    # Several blocks pack each predictor's intercept, dense and small categorical
    # groups into an OrdinaryPanelGroupMatrix.
    assert sum(panel is not None for panel in predictor_panels(layout)) >= 2


@pytest.mark.parametrize("cell_cap", [None, 0], ids=["tables", "rows"])
def test_leverage_blocks_equal_the_dense_rows(problem, split_blocks, monkeypatch, cell_cap):
    layout, matrices, slices, rng = problem
    if cell_cap is not None:
        monkeypatch.setattr(bp, "_CELL_CAP", cell_cap)
    width = layout.n_coefficients
    factor = rng.normal(size=(width, width))
    inverse = factor @ factor.T / width
    reference = _leverage_blocks(matrices, slices, inverse)
    magnitude = _leverage_blocks(tuple(map(np.abs, matrices)), slices, np.abs(inverse))
    got = bp.binned_leverage_blocks(layout, inverse)
    assert got.shape == reference.shape
    assert _excess(got, reference, magnitude, width) <= 0.0


@pytest.mark.parametrize("block", [97, N], ids=["blocks", "one-block"])
def test_cross_grams_equal_the_dense_rows_with_one_producer_call_per_block(
    problem, monkeypatch, block
):
    monkeypatch.setattr(bp, "contraction_block_rows", lambda n, row_bytes, chunk: block)
    layout, matrices, slices, rng = problem
    k_parameters = len(matrices)
    width = layout.n_coefficients
    packed = rng.normal(size=(N, 4, k_parameters * (k_parameters + 1) // 2))
    calls = []

    def packed_rows(start, stop):
        calls.append((start, stop))
        return packed[start:stop]

    got = bp.binned_cross_grams(layout, packed_rows, width)
    reference = _cross_all(matrices, slices, packed, width)
    magnitude = _cross_all(tuple(map(np.abs, matrices)), slices, np.abs(packed), width)
    assert _excess(got, reference, magnitude, N) <= 0.0
    np.testing.assert_array_equal(got, np.transpose(got, (0, 2, 1)))
    # An empty first call gives m, then one call per block in row order.
    assert calls == [(0, 0)] + [(start, min(start + block, N)) for start in range(0, N, block)]


def _component_excess(layout, matrices, slices, packed, got):
    """Each component's excess over its own ``8 N u`` magnitude bound."""
    width = layout.n_coefficients
    reference = _cross_all(matrices, slices, packed, width)
    magnitude = _cross_all(tuple(map(np.abs, matrices)), slices, np.abs(packed), width)
    return [_excess(g, r, m, N) for g, r, m in zip(got, reference, magnitude, strict=True)]


def _scaled_components(problem):
    """Weights of three components at ``2**-600``, 1 and ``2**600``: one stacked pass."""
    layout, matrices, slices, rng = problem
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 3, k_parameters * (k_parameters + 1) // 2))
    return packed * np.array([2.0**-600, 1.0, 2.0**600])[None, :, None]


def test_cross_grams_of_components_far_apart_in_magnitude(problem, split_blocks):
    """Each component's stacked slice contracts alone: none takes another's rows or range."""
    layout, matrices, slices, _rng = problem
    packed = _scaled_components(problem)
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], layout.n_coefficients)
    assert max(_component_excess(layout, matrices, slices, packed, got)) <= 0.0


def test_mutation_cross_grams_contracting_one_slice_for_every_component_fail(
    problem, split_blocks, monkeypatch
):
    layout, matrices, slices, _rng = problem
    packed = _scaled_components(problem)
    original = algebra._StackedComponent.contracted

    def first_slice(self, factors):
        k, self._k = self._k, 0
        yield from original(self, factors)
        self._k = k

    monkeypatch.setattr(algebra._StackedComponent, "contracted", first_slice)
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], layout.n_coefficients)
    assert max(_component_excess(layout, matrices, slices, packed, got)) > 0.0


@pytest.mark.parametrize("mutated", [False, True], ids=["released", "mutation-dropped"])
def test_cross_grams_release_the_stacked_tables_mid_pass(
    problem, split_blocks, monkeypatch, mutated
):
    """A release before the third block contracts every component's slice of the first two."""
    layout, matrices, slices, rng = problem
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 3, k_parameters * (k_parameters + 1) // 2))
    peak = float(np.max(np.abs(packed[: 3 * 97])))
    monkeypatch.setattr(bp, "_BLOCK_SUM_LIMIT", 2.5 * 97 * peak)
    released = []
    contracted = algebra._StackedComponent.contracted

    def recorded(self, factors):
        released.append(self._k)
        if not mutated:
            yield from contracted(self, factors)

    monkeypatch.setattr(algebra._StackedComponent, "contracted", recorded)
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], layout.n_coefficients)
    assert released[:3] == [0, 1, 2]
    excess = max(_component_excess(layout, matrices, slices, packed, got))
    assert (excess > 0.0) if mutated else (excess <= 0.0)


@pytest.mark.parametrize("mutated", [False, True], ids=["rows", "mutation-misaligned-rows"])
def test_cross_grams_sum_support_rows_where_no_joint_table_is_kept(
    problem, split_blocks, monkeypatch, mutated
):
    """Supports with no joint table keep the right accumulation, ``S_i' (sum w S_j[k_j])``."""
    layout, matrices, slices, _rng = problem
    joint = algebra._joint_table_kind
    monkeypatch.setattr(
        algebra,
        "_joint_table_kind",
        lambda i, j: None if joint(i, j) == "supports" else joint(i, j),
    )
    added = []
    original = algebra._add_rows_stack

    def recorded(support, transposed, table, gm_i, gm_j, W):
        added.append(transposed)
        if mutated:
            # The block's weights read against the wrong rows.
            W = W[::-1]
        return original(support, transposed, table, gm_i, gm_j, W)

    monkeypatch.setattr(algebra, "_add_rows_stack", recorded)
    packed = _scaled_components(problem)
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], layout.n_coefficients)
    assert added
    excess = max(_component_excess(layout, matrices, slices, packed, got))
    assert (excess > 0.0) if mutated else (excess <= 0.0)


def test_every_kernel_repeats_its_bits(problem, split_blocks):
    """A second call on the same inputs gives the same bits: nothing persists across calls."""
    layout, matrices, _slices, rng = problem
    width = layout.n_coefficients
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 3, k_parameters * (k_parameters + 1) // 2))
    factor = rng.normal(size=(width, width))
    D = rng.normal(size=(matrices[0].shape[1], 2))
    for run in (
        lambda: bp.binned_cross_grams(layout, lambda a, b: packed[a:b], width),
        lambda: bp.binned_leverage_blocks(layout, factor @ factor.T),
        lambda: bp.design_dot_rows(layout, 0, D, 11, N),
    ):
        np.testing.assert_array_equal(run(), run())


@pytest.mark.parametrize("rows", [(0, N), (13, 13), (31, 500)])
def test_design_dot_rows_equal_the_dense_rows(problem, rows):
    layout, matrices, _slices, rng = problem
    start, stop = rows
    for q, matrix in enumerate(matrices):
        D = rng.normal(size=(matrix.shape[1], 7))
        got = bp.design_dot_rows(layout, q, D, start, stop)
        magnitude = np.abs(matrix[start:stop]) @ np.abs(D)
        assert got.shape == (stop - start, 7)
        assert _excess(got, matrix[start:stop] @ D, magnitude, matrix.shape[1]) <= 0.0


def test_exact_route_layouts_have_no_kernel():
    layout = _layout(2, discrete=False)[0]
    assert not bp.binned_layout(layout)
    assert bp.binned_leverage_blocks(layout, np.eye(layout.n_coefficients)) is None
    assert bp.binned_cross_grams(layout, pytest.fail, layout.n_coefficients) is None
    assert bp.design_dot_rows(layout, 0, np.zeros((1, 1)), 0, 1) is None


def test_no_dense_design_rows_are_formed(problem, split_blocks, monkeypatch):
    layout, matrices, _slices, rng = problem
    width = layout.n_coefficients

    def forbidden(*args, **kwargs):
        pytest.fail("a binned kernel formed dense design rows")

    for state in layout.predictors:
        for group in state.design.group_matrices:
            if type(group).__name__ != "DenseGroupMatrix":
                monkeypatch.setattr(type(group), "toarray", forbidden)
    monkeypatch.setattr(DesignMatrix, "toarray", forbidden)
    monkeypatch.setattr(BoundedPredictorMatrix, "__getitem__", forbidden)
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 2, k_parameters * (k_parameters + 1) // 2))
    assert bp.binned_leverage_blocks(layout, np.eye(width)) is not None
    assert bp.binned_cross_grams(layout, lambda a, b: packed[a:b], width) is not None
    assert bp.design_dot_rows(layout, 1, np.ones((matrices[1].shape[1], 2)), 5, 90) is not None


# Mutation checks: each kernel's comparison fails against a deliberately broken variant.


def _level_rows_read_bin_zero(original):
    def codes(term, start, stop):
        result = original(term, start, stop)
        if type(term.group).__name__ == "DiscretizedSplineCategoricalGroupMatrix":
            result = np.where(result == term.group.n_bins, 0, result)
        return result

    return codes


def test_mutation_leverage_off_level_rows_reading_a_bin_fail(problem, split_blocks, monkeypatch):
    layout, matrices, slices, rng = problem
    width = layout.n_coefficients
    inverse = np.eye(width) + 0.01
    reference = _leverage_blocks(matrices, slices, inverse)
    magnitude = _leverage_blocks(tuple(map(np.abs, matrices)), slices, np.abs(inverse))
    monkeypatch.setattr(bp, "_codes", _level_rows_read_bin_zero(bp._codes))
    got = bp.binned_leverage_blocks(layout, inverse)
    assert _excess(got, reference, magnitude, width) > 0.0


def test_mutation_cross_grams_reusing_the_first_block_plans_fail(
    problem, split_blocks, monkeypatch
):
    layout, matrices, slices, rng = problem
    width = layout.n_coefficients
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 2, k_parameters * (k_parameters + 1) // 2))
    reference = _cross_all(matrices, slices, packed, width)
    magnitude = _cross_all(tuple(map(np.abs, matrices)), slices, np.abs(packed), width)
    original = bp.block_plans
    first = {}

    def stale(layout, panels, start, stop):
        plans = first.setdefault("plans", original(layout, panels, start, stop))
        return plans if stop - start == 97 else original(layout, panels, start, stop)

    monkeypatch.setattr(bp, "block_plans", stale)
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], width)
    assert _excess(got, reference, magnitude, N) > 0.0


def test_mutation_design_dot_rows_reading_the_base_level_fails(problem, monkeypatch):
    layout, matrices, _slices, rng = problem
    original = bp._left

    def base_reads_level_zero(term, M):
        result = original(term, M)
        if term.one_hot:
            result[-1] = M[0]
        return result

    monkeypatch.setattr(bp, "_left", base_reads_level_zero)
    D = rng.normal(size=(matrices[0].shape[1], 3))
    got = bp.design_dot_rows(layout, 0, D, 0, N)
    magnitude = np.abs(matrices[0]) @ np.abs(D)
    assert _excess(got, matrices[0] @ D, magnitude, matrices[0].shape[1]) > 0.0


# Rounding counts: each bound holds against an exact reference on a small layout.


def _counted_layout():
    """200 rows in 6 bins and 3 levels: many rows per bin, as the counts are for."""
    rng = np.random.default_rng(4410)
    n = 200
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": rng.uniform(-1, 1, n),
                "z": rng.uniform(-1, 1, n),
                "d": rng.normal(size=n),
                "g": rng.choice(list("abc"), n),
            }
        )
    )
    family = GaussianLS()
    names = [parameter.name for parameter in family.parameters]
    predictors = [
        Predictor(names[0], {"x": Spline(kind="cr", n_knots=5), "g": Categorical(base="a")}),
        Predictor(names[1], {"z": Spline(kind="cr", n_knots=4), "d": Numeric()}),
    ]
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        family.parameters,
        predictors,
        model_discrete=True,
        n_bins_config=6,
    )
    return build_stacked_layout(compiled), rng


def _rows(layout, q, exact):
    """Predictor ``q``'s rows: exact Fractions of ``S = B R``, or ``|B| |R|`` in float."""
    state = layout.predictors[q]
    n = state.design.n
    out = np.zeros((n, state.coefficient_slice.stop - state.coefficient_slice.start), object)
    for term in bp._terms(state, 0):
        columns = range(term.columns.start, term.columns.stop)
        if term.dense:
            values = np.asarray(term.group.M, dtype=np.float64)
            out[:, term.columns] = np.vectorize(Fraction)(values) if exact else np.abs(values)
            continue
        if term.one_hot:
            table = np.vstack((np.eye(len(columns)), np.zeros((1, len(columns)))))
        elif term.group is None:
            table = np.ones((1, 1))
        else:
            B, R = term.group.B_unique, term.group.R_inv
            if exact:
                B, R = np.vectorize(Fraction)(B), np.vectorize(Fraction)(R)
            else:
                B, R = np.abs(B), np.abs(R)
            table = B @ R
            if table.shape[0] < term.table.shape[0]:
                table = np.vstack((table, np.zeros((1, len(columns)), table.dtype)))
        if exact and table.dtype != object:
            table = np.vectorize(Fraction)(table)
        out[:, term.columns] = table[bp._codes(term, 0, n)]
    return out if exact else out.astype(np.float64)


def _within(got, exact, magnitude, k):
    gamma = Fraction(k) * Fraction(U) / (1 - Fraction(k) * Fraction(U))
    errors = [
        abs(Fraction(float(g)) - e) - gamma * Fraction(float(m))
        for g, e, m in zip(np.ravel(got), np.ravel(exact), np.ravel(magnitude), strict=True)
    ]
    return max(errors) <= 0


def test_rounding_counts_bound_each_kernel_against_exact_sums(monkeypatch):
    layout, rng = _counted_layout()
    monkeypatch.setattr(bp, "contraction_block_rows", lambda n, row_bytes, chunk: 40)
    n, width = layout.predictors[0].design.n, layout.n_coefficients
    counts = bp.binned_rounding_counts(layout, 1)
    assert counts.cross < n
    exact = [_rows(layout, q, True) for q in range(2)]
    magnitude = [_rows(layout, q, False) for q in range(2)]
    slices = tuple(state.coefficient_slice for state in layout.predictors)
    packed = rng.normal(size=(n, 1, 3))
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], width)[0]
    weights = np.vectorize(Fraction)(packed[:, 0, :])
    for channel, (left, right) in enumerate(((0, 0), (0, 1), (1, 1))):
        block = (slices[left], slices[right])
        w = weights[:, channel]
        reference = exact[left].T @ (w[:, None] * exact[right])
        bound = magnitude[left].T @ (np.abs(packed[:, 0, channel])[:, None] * magnitude[right])
        assert _within(got[block], reference, bound, counts.cross)
    factor = rng.normal(size=(width, width))
    inverse = factor @ factor.T / width
    got = bp.binned_leverage_blocks(layout, inverse)
    V = np.vectorize(Fraction)(inverse)
    for channel, (left, right) in enumerate(((0, 0), (0, 1), (1, 1))):
        block = V[slices[left], slices[right]]
        reference = np.sum((exact[left] @ block) * exact[right], axis=1)
        bound = np.sum(
            (magnitude[left] @ np.abs(inverse[slices[left], slices[right]])) * magnitude[right],
            axis=1,
        )
        assert _within(got[:, channel], reference, bound, counts.leverage)
    D = rng.normal(size=(exact[1].shape[1], 2))
    got = bp.design_dot_rows(layout, 1, D, 0, n)
    reference = exact[1] @ np.vectorize(Fraction)(D)
    assert _within(got, reference, magnitude[1] @ np.abs(D), counts.dot)


def test_rounding_counts_are_none_off_a_binned_layout():
    assert bp.binned_rounding_counts(_layout(2, discrete=False)[0], 1) is None


# Range guards on the associations the kernels choose for themselves.


@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "mutation-unguarded"])
def test_leverage_table_keeps_the_rows_order_where_right_first_would_underflow(
    guarded, monkeypatch
):
    """``S_t = 2**128 [[1,0],[0,1],[1,1]]``, ``S_u = 2**-128 I`` and ``V = 2**-1000 I``:
    the cost model picks the right-first table, whose ``V S_u'`` underflows to 0.

    The rows' own order, ``(S_t V) S_u'``, keeps every partial product normal, and
    the first row's cross leverage is exactly 2**-1000. Mutation check: dropping
    the range check takes right-first and reads 0.
    """
    if not guarded:
        monkeypatch.setattr(bp, "_cross_factors_in_range", lambda *operands: True)
    t = bp._Term(slice(0, 2), None, 2.0**128 * np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    u = bp._Term(slice(2, 4), None, 2.0**-128 * np.eye(2))
    reader = bp._pair_reader(t, u, 2.0**-1000 * np.eye(2), 6, [1 << 20])
    codes = {t: np.array([0, 1, 2, 0, 1, 2]), u: np.array([0, 1, 0, 1, 0, 1])}
    got = reader(codes, {})
    assert bool(abs(got[0] - 2.0**-1000) <= 4 * U * 2.0**-1000) is guarded


def test_support_rows_beyond_the_release_margin_keep_no_rows_table():
    """A right accumulation sums ``w S_j`` with the weights bounded only by the release
    limit, 2**1000, so a support at 2**128 could overflow it: that pair keeps no rows
    table, and its blocks take the joint or per-block routes with their own range rules.
    """
    from superglm._group_matrix._group_matrix_discretized import DiscretizedSSPGroupMatrix

    bins = np.arange(6)
    large = DiscretizedSSPGroupMatrix(np.full((6, 1), 2.0**64), np.array([[2.0**64]]), bins)
    small = DiscretizedSSPGroupMatrix(np.full((6, 1), 2.0**-128), np.array([[1.0]]), bins)
    # The summing side's partner is the large support in this order.
    assert algebra._rows_table(small, large) is None
    assert algebra._rows_table(large, small) is not None
    unit = DiscretizedSSPGroupMatrix(np.full((6, 1), 0.5), np.array([[1.0]]), bins)
    assert algebra._rows_table(unit, unit) is not None


def _compressed_layout():
    """An exact-mode layout on integer covariates, which compresses each spline to its supports."""
    rng = np.random.default_rng(3)
    n = 1500
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "a": rng.integers(18, 60, n).astype(float),
                "b": rng.integers(0, 12, n).astype(float),
                "d": rng.normal(size=n),
                "g": rng.choice(list("abc"), n),
                "h": rng.choice(list("pqrs"), n),
            }
        )
    )
    family = GaussianLS()
    names = [parameter.name for parameter in family.parameters]
    predictors = [
        Predictor(
            names[0],
            {
                "a": Spline(kind="cr", n_knots=6),
                "g": Categorical(base="a"),
                "h": Categorical(base="p"),
                "d": Numeric(),
            },
            interaction_specs={"a:h": SplineCategorical("a", "h")},
        ),
        Predictor(names[1], {"b": Spline(kind="cr", n_knots=5), "h": Categorical(base="p")}),
    ]
    compiled = compile_predictors(
        frame, resolved_prior(np.ones(n)), family.parameters, predictors, model_discrete=False
    )
    layout = build_stacked_layout(compiled)
    matrices = dense_predictor_matrices(layout)
    return layout, matrices, tuple(state.coefficient_slice for state in layout.predictors), rng


def test_support_compressed_layouts_equal_the_dense_rows(split_blocks):
    """Exact fits on few-valued covariates keep support-compressed splines and levels,
    which ``binned_layout`` admits: every kernel matches the dense rows on them."""
    layout, matrices, slices, rng = _compressed_layout()
    kinds = {type(g).__name__ for state in layout.predictors for g in state.design.group_matrices}
    assert {
        "SupportCompressedSSPGroupMatrix",
        "SupportCompressedSplineCategoricalGroupMatrix",
    } <= kinds
    assert bp.binned_layout(layout)
    n, width = matrices[0].shape[0], layout.n_coefficients
    factor = rng.normal(size=(width, width))
    inverse = factor @ factor.T / width
    reference = _leverage_blocks(matrices, slices, inverse)
    magnitude = _leverage_blocks(tuple(map(np.abs, matrices)), slices, np.abs(inverse))
    assert _excess(bp.binned_leverage_blocks(layout, inverse), reference, magnitude, width) <= 0.0
    packed = rng.normal(size=(n, 4, 3))
    got = bp.binned_cross_grams(layout, lambda start, stop: packed[start:stop], width)
    reference = _cross_all(matrices, slices, packed, width)
    magnitude = _cross_all(tuple(map(np.abs, matrices)), slices, np.abs(packed), width)
    assert _excess(got, reference, magnitude, n) <= 0.0
    for q, matrix in enumerate(matrices):
        D = rng.normal(size=(matrix.shape[1], 5))
        dot = bp.design_dot_rows(layout, q, D, 31, 900)
        dot_magnitude = np.abs(matrix[31:900]) @ np.abs(D)
        assert _excess(dot, matrix[31:900] @ D, dot_magnitude, matrix.shape[1]) <= 0.0


@pytest.mark.parametrize("dense", [False, True], ids=["rows-swap", "dense-swap"])
@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "mutation-unguarded"])
def test_leverage_rows_keep_their_order_where_reading_v_x_u_first_would_underflow(
    dense, guarded, monkeypatch
):
    """``x_t = 2**128 e_0`` and ``x_u = 2**-128 e_0`` with ``V = 2**-1000 [I 0]``: the
    narrower or dense side would read ``V x_u`` first, which underflows to 0.

    The rows' order ``(x_t V) x_u`` gives exactly 2**-1000. Mutation check:
    dropping the range check takes the swap and reads 0.
    """
    from superglm._group_matrix._group_matrix_core import DenseGroupMatrix

    if not guarded:
        monkeypatch.setattr(bp, "_cross_factors_in_range", lambda *operands: True)
    monkeypatch.setattr(bp, "_CELL_CAP", 0)
    left = 2.0**128 * np.eye(2)
    if dense:
        t = bp._Term(slice(0, 2), DenseGroupMatrix(np.tile(left[0], (6, 1))), None)
    else:
        t = bp._Term(slice(0, 2), None, left)
    u = bp._Term(slice(2, 5), None, 2.0**-128 * np.eye(3))
    reader = bp._pair_reader(t, u, 2.0**-1000 * np.eye(2, 3), 6, [1 << 20])
    codes = (
        {u: np.zeros(6, np.intp)} if dense else {t: np.zeros(6, np.intp), u: np.zeros(6, np.intp)}
    )
    values = {t: t.group.M} if dense else {}
    got = reader(codes, values)
    assert bool(abs(got[0] - 2.0**-1000) <= 4 * U * 2.0**-1000) is guarded


@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "mutation-unguarded"])
def test_cross_grams_whose_sum_is_not_finite_are_none(problem, split_blocks, monkeypatch, guarded):
    """A nan in the summed curvature (an overflow meeting signed weights) gives None,
    so a caller falls back to its dense rows. The event is forced: one block's
    curvature entry is set to nan. Mutation check: without the finiteness check,
    ``finish()``'s symmetry check raises instead.
    """
    from superglm.distributional.solver.assembly import GroupedGeometryAccumulator

    original = GroupedGeometryAccumulator.add_block

    def poisoned(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._curvature[0, 0] = np.nan

    monkeypatch.setattr(GroupedGeometryAccumulator, "add_block", poisoned)
    if not guarded:
        monkeypatch.setattr(GroupedGeometryAccumulator, "curvature_is_finite", lambda self: True)
    layout, matrices, _slices, rng = problem
    k = len(matrices)
    packed = rng.normal(size=(N, 2, k * (k + 1) // 2))
    run = lambda: bp.binned_cross_grams(layout, lambda a, b: packed[a:b], layout.n_coefficients)  # noqa: E731
    if guarded:
        assert run() is None
    else:
        with pytest.raises(RuntimeError, match="symmetry"):
            run()


def test_out_of_range_inputs_refuse_or_fall_back(problem, split_blocks):
    """Weights at 1e308 meet the assembly's refusal of a moment it cannot represent, as
    every chunked SuperLSS fit does; an inverse at 1e308 overflows the leverage
    blocks, which then give None."""
    layout, matrices, _slices, _rng = problem
    k = len(matrices)
    packed = np.full((N, 2, k * (k + 1) // 2), 1e308)
    with pytest.raises(np.linalg.LinAlgError, match="not representable"):
        bp.binned_cross_grams(layout, lambda a, b: packed[a:b], layout.n_coefficients)
    inverse = np.full((layout.n_coefficients,) * 2, 1e308)
    assert bp.binned_leverage_blocks(layout, inverse) is None
