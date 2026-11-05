"""A binned layout's rows and row products, read from its bin tables, equal the dense design's."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm._group_matrix import _lookup_tables as lookup_tables
from superglm._parallel import parallel_config
from superglm.distributional import GaussianLS, TweedieLSS
from superglm.distributional._row_design import BoundedPredictorMatrix
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import binned_rows as br
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.features import Categorical, Numeric, RandomEffect, Spline, SplineCategorical
from superglm.features.interaction import TensorInteraction
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior

U = 2.0**-53
N = 640


def _layout(k_parameters: int, discrete: bool = True, dense: bool = True):
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
                **({"d": Numeric()} if dense else {}),
                "r": RandomEffect(),
            },
            interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        Predictor(
            names[1],
            {
                "x": Spline(kind="cr", n_knots=5),
                "h": Categorical(base="p"),
                **({"d": Numeric()} if dense else {}),
            },
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


def _within(got, reference, width):
    """got within 8 p u of the dense rows, normwise: the tables' B R against the rows'."""
    return bool(
        np.max(np.abs(got - reference), initial=0.0)
        <= 8 * width * U * np.max(np.abs(reference), initial=0.0)
    )


@pytest.fixture(params=[2, 3], ids=["K2", "K3"])
def problem(request):
    return _layout(request.param)


def test_fixture_covers_every_group_kind(problem):
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
    assert br.binned_layout(layout)


@pytest.mark.parametrize("rows", [(0, N), (13, 13), (31, 500)])
def test_rows_equal_the_dense_design_rows(problem, rows):
    layout, matrices, _slices, _rng = problem
    reader = br.binned_rows(layout)
    start, stop = rows
    for q, matrix in enumerate(matrices):
        got = reader.rows(q, start, stop)
        assert got.shape == (stop - start, matrix.shape[1]) and got.flags.c_contiguous
        assert _within(got, matrix[start:stop], matrix.shape[1])


def _compressed_layout(dense: bool = True):
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
                **({"d": Numeric()} if dense else {}),
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


def test_support_compressed_layouts_equal_the_dense_rows():
    """Exact fits on few-valued covariates keep support-compressed splines and levels,
    which binned_layout admits: their gathered rows match the dense rows."""
    layout, matrices, _slices, _rng = _compressed_layout()
    kinds = {type(g).__name__ for state in layout.predictors for g in state.design.group_matrices}
    assert {
        "SupportCompressedSSPGroupMatrix",
        "SupportCompressedSplineCategoricalGroupMatrix",
    } <= kinds
    reader = br.binned_rows(layout)
    for q, matrix in enumerate(matrices):
        assert _within(reader.rows(q, 31, 900), matrix[31:900], matrix.shape[1])


@pytest.mark.parametrize("compressed", [False, True], ids=["binned", "support-compressed"])
def test_level_rows_without_a_certified_lookup_equal_the_dense_rows(
    problem, monkeypatch, compressed
):
    """A level whose row lookup has no certificate reads its rows through ``row_subset``;
    the event is forced, and rows from an interior start must still match."""
    layout, matrices, _slices, _rng = _compressed_layout() if compressed else problem
    monkeypatch.setattr(br, "_category_rows", lambda group, start, stop: None)
    reader = br.binned_rows(layout)
    for q, matrix in enumerate(matrices):
        assert _within(reader.rows(q, 31, 500), matrix[31:500], matrix.shape[1])


def test_exact_route_layouts_have_no_reader():
    layout = _layout(2, discrete=False)[0]
    assert not br.binned_layout(layout)
    assert br.binned_rows(layout) is None
    with pytest.raises(ValueError, match="binned layout"):
        br.BinnedRows(layout)


@pytest.mark.parametrize("kind", ["table", "dense"])
def test_a_non_finite_design_is_refused_as_the_dense_rows_refuse_it(problem, kind, monkeypatch):
    """A non-finite support table or dense column raises the dense rows' ``ValueError``."""
    from superglm._group_matrix._group_matrix_core import DenseGroupMatrix

    original = br._terms

    def poisoned(state):
        terms = list(original(state))
        for i, term in enumerate(terms):
            if kind == "table" and type(term.group).__name__ == "DiscretizedSSPGroupMatrix":
                table = term.table.copy()
                table[0, 0] = np.inf
                terms[i] = br._Term(term.columns, term.group, table)
                break
            if kind == "dense" and type(term.group) is DenseGroupMatrix:
                M = np.array(term.group.M, dtype=np.float64)
                M[0, 0] = np.nan
                terms[i] = br._Term(term.columns, DenseGroupMatrix(M), None)
                break
        return tuple(terms)

    monkeypatch.setattr(br, "_terms", poisoned)
    with pytest.raises(ValueError, match="must be finite"):
        br.BinnedRows(problem[0])


def test_rows_outside_the_design_are_refused(problem):
    reader = br.binned_rows(problem[0])
    for start, stop in ((-1, 3), (5, 4), (0, N + 1)):
        with pytest.raises(ValueError, match="inside the design"):
            reader.rows(0, start, stop)


def test_tables_are_formed_once_and_no_dense_rows_are_read(problem, monkeypatch):
    """The reader forms each predictor's tables at construction; its row reads form no
    dense design rows and re-form no table."""
    layout, matrices, _slices, _rng = problem
    built = []
    original = br._terms
    monkeypatch.setattr(br, "_terms", lambda state: built.append(state) or original(state))

    def forbidden(*args, **kwargs):
        pytest.fail("the reader formed dense design rows")

    for state in layout.predictors:
        for group in state.design.group_matrices:
            if type(group).__name__ != "DenseGroupMatrix":
                monkeypatch.setattr(type(group), "toarray", forbidden)
    monkeypatch.setattr(DesignMatrix, "toarray", forbidden)
    monkeypatch.setattr(BoundedPredictorMatrix, "__getitem__", forbidden)
    reader = br.binned_rows(layout)
    # Re-forming a table needs ``R``: take it away once the reader holds its tables.
    for state in layout.predictors:
        for group in state.design.group_matrices:
            if hasattr(group, "R_inv"):
                monkeypatch.setattr(group, "R_inv", None)
    for q in range(len(matrices)):
        for start in range(0, N, 97):
            reader.rows(q, start, min(start + 97, N))
    assert len(built) == len(layout.predictors)


def test_rows_repeat_their_bits(problem):
    layout = problem[0]
    first, second = br.binned_rows(layout), br.binned_rows(layout)
    for q in range(len(layout.predictors)):
        np.testing.assert_array_equal(first.rows(q, 0, N), first.rows(q, 0, N))
        np.testing.assert_array_equal(first.rows(q, 7, 300), second.rows(q, 7, 300))


def test_the_score_task_costs_its_row_passes_not_its_columns(monkeypatch):
    """``X' v`` reads each row once per group, then its bins through its basis. Mutation check:
    a cost of rows times columns makes this the queue's largest task and caps the pool."""
    from superglm.distributional.solver import assembly

    layout, _matrices, _slices, _rng = _layout(2, dense=False)
    queued = []
    run = assembly.run_block_tasks

    def spy(tasks, cache, profile, *, dispatch_bound=False, row_visits=False):
        assert dispatch_bound and row_visits
        queued.extend(tasks)
        run(tasks, cache, profile, dispatch_bound=dispatch_bound, row_visits=row_visits)

    monkeypatch.setattr(assembly, "run_block_tasks", spy)
    width = layout.n_coefficients
    accumulator = assembly.GroupedGeometryAccumulator(
        layout, penalty=np.zeros((width, width)), coefficients=np.zeros(width)
    )
    plans = assembly._grouped_predictor_plans(layout)
    accumulator.add_block(plans, np.zeros((N, 2)), np.zeros((3, N)))
    score = queued[-1]
    assert score.cost < N * width


def test_row_products_equal_the_dense_rows_times_the_coefficients(problem):
    layout, matrices, _slices, rng = problem
    reader = br.binned_rows(layout)
    for q, matrix in enumerate(matrices):
        coefficients = rng.normal(size=(matrix.shape[1], 3))
        products = reader.row_products(q, coefficients)
        for rows in (slice(0, N), slice(13, 13), slice(31, 500)):
            reference = matrix[rows] @ coefficients
            got = products(rows)
            assert got.shape == reference.shape
            assert _within(got, reference, matrix.shape[1])


def _inverse(layout, rng):
    """A symmetric positive definite stand-in for ``H^-1`` on the whole layout."""
    root = rng.normal(size=(layout.n_coefficients, layout.n_coefficients))
    return root @ root.T / layout.n_coefficients + np.eye(layout.n_coefficients)


def _dense_leverages(matrices, slices, inverse):
    """``G_i`` from the dense rows, in packed order: the reference the tables must reproduce."""
    pairs = br.packed_pairs(len(matrices))
    blocks = np.empty((len(matrices[0]), len(pairs)))
    for channel, (left, right) in enumerate(pairs):
        product = matrices[left] @ inverse[slices[left], slices[right]]
        blocks[:, channel] = np.einsum("ij,ij->i", product, matrices[right])
    return blocks


@pytest.mark.parametrize("case", ["K2", "K3", "compressed"])
def test_leverages_equal_the_dense_row_products(case, monkeypatch):
    monkeypatch.setattr(lookup_tables, "_BLOCK_ROWS", 64)
    if case == "compressed":
        layout, matrices, slices, rng = _compressed_layout(dense=False)
    else:
        layout, matrices, slices, rng = _layout(int(case[1]), dense=False)
    inverse = _inverse(layout, rng)
    got = br.binned_rows(layout).leverages(inverse, slices)
    reference = _dense_leverages(matrices, slices, inverse)
    assert got.shape == reference.shape
    assert _within(got, reference, layout.n_coefficients)


def test_a_dense_term_or_too_many_table_bytes_keep_the_row_products(monkeypatch):
    layout, _matrices, slices, rng = _layout(2)
    inverse = _inverse(layout, rng)
    assert br.binned_rows(layout).leverages(inverse, slices) is None
    layout, _matrices, slices, rng = _layout(2, dense=False)
    inverse = _inverse(layout, rng)
    reader = br.binned_rows(layout)
    assert reader.leverages(inverse, slices) is not None
    with parallel_config(max_memory=1 << 16):
        assert reader.leverages(inverse, slices) is None
    monkeypatch.setattr(br, "_LEVERAGE_MAX_CELLS", 1)
    assert reader.leverages(inverse, slices) is None


def test_a_wide_random_effect_is_declined_before_any_table_is_formed():
    """30,000 levels hold 4.5e8 cells past the budget, so the leverage declines holding next to
    nothing. Mutation check: forming the categorical's identity first holds 7.2 GB here."""
    n = 30_000
    frame = as_eager_frame(
        pd.DataFrame({"r": [f"l{i}" for i in range(n)], "x": np.linspace(-1, 1, n)})
    )
    family = GaussianLS()
    names = [parameter.name for parameter in family.parameters]
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        family.parameters,
        [
            Predictor(names[0], {"x": Spline(kind="cr", n_knots=5), "r": RandomEffect()}),
            Predictor(names[1], {}),
        ],
        model_discrete=True,
        n_bins_config=24,
    )
    layout = build_stacked_layout(compiled)
    reader = br.binned_rows(layout)
    slices = tuple(state.coefficient_slice for state in layout.predictors)
    tracemalloc.start()
    try:
        assert reader.leverages(None, slices) is None
        assert tracemalloc.get_traced_memory()[1] < 1 << 20
    finally:
        tracemalloc.stop()


def test_leverages_keep_their_bits_at_every_worker_count(monkeypatch):
    monkeypatch.setattr(lookup_tables, "_BLOCK_ROWS", 64)
    started = []
    pool = lookup_tables.ThreadPoolExecutor

    def spy(*args, **kwargs):
        started.append(kwargs["max_workers"])
        return pool(*args, **kwargs)

    monkeypatch.setattr(lookup_tables, "ThreadPoolExecutor", spy)
    layout, _matrices, slices, rng = _layout(3, dense=False)
    inverse = _inverse(layout, rng)
    reader = br.binned_rows(layout)
    with parallel_config(n_jobs=1):
        serial = reader.leverages(inverse, slices)
    with parallel_config(n_jobs=4):
        pooled = reader.leverages(inverse, slices)
    assert started
    np.testing.assert_array_equal(serial, pooled)


# Mutation checks: the comparison fails against a deliberately broken reader.


def test_mutation_leverage_reading_each_pair_at_its_table_once_fails(monkeypatch):
    """A diagonal channel holds both orders of a pair: its table at twice its value."""
    layout, matrices, slices, rng = _layout(2, dense=False)
    inverse = _inverse(layout, rng)
    original = br.gather_channels

    def halved(codes, pairs, flat, channels):
        out = original(codes, pairs, flat, channels)
        one_read = pairs[pairs[:, 1] != pairs[:, 2]]
        return out - 0.5 * original(codes, one_read, flat, channels)

    monkeypatch.setattr(br, "gather_channels", halved)
    got = br.binned_rows(layout).leverages(inverse, slices)
    assert not _within(got, _dense_leverages(matrices, slices, inverse), layout.n_coefficients)


def test_mutation_off_level_rows_reading_a_bin_fail(problem, monkeypatch):
    layout, matrices, _slices, _rng = problem
    original = br._codes

    def level_rows_read_bin_zero(term, start, stop):
        result = original(term, start, stop)
        if type(term.group).__name__ == "DiscretizedSplineCategoricalGroupMatrix":
            result = np.where(result == term.group.n_bins, 0, result)
        return result

    monkeypatch.setattr(br, "_codes", level_rows_read_bin_zero)
    reader = br.binned_rows(layout)
    assert not all(_within(reader.rows(q, 0, N), m, m.shape[1]) for q, m in enumerate(matrices))


def test_mutation_base_level_reading_the_first_column_fails(problem, monkeypatch):
    layout, matrices, _slices, _rng = problem
    original = br._codes

    def base_reads_level_zero(term, start, stop):
        result = original(term, start, stop)
        if type(term.group).__name__ == "CategoricalGroupMatrix":
            width = term.columns.stop - term.columns.start
            result = np.where(result >= width, 0, result)
        return result

    monkeypatch.setattr(br, "_codes", base_reads_level_zero)
    reader = br.binned_rows(layout)
    assert not _within(reader.rows(0, 0, N), matrices[0], matrices[0].shape[1])
