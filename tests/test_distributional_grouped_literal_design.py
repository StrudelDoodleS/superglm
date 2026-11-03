"""Grouped chunk assembly matches an independent literal design."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm._group_matrix._block_queue as block_queue
import superglm._group_matrix._group_matrix_algebra as algebra
import superglm._parallel as parallel
from superglm._frame import as_eager_frame
from superglm._group_matrix import _group_matrix_core
from superglm._group_matrix._block_queue import block_queue_config
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, PredictorExecutionPlan, compile_predictors
from superglm.distributional.solver import chunks
from superglm.features import Categorical, Numeric, RandomEffect, Spline
from superglm.group_matrix import DiscretizedSSPGroupMatrix

from ._distributional_weights import resolved_prior
from ._traced_memory import own_traced_peak
from .test_distributional_fused_trial import _literal_design


def _problem():
    n = 29
    x = np.linspace(-1.1, 0.8, n)
    z = np.cos(np.arange(n) * 0.3)
    category = np.resize(["a", "b", "c"], n)
    family = GaussianLS()
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": z, "g": category}))
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        family.parameters,
        (
            Predictor("location", {"x": Numeric(), "g": Categorical(base="c")}),
            Predictor("scale", {"z": Numeric()}, intercept=False),
        ),
    )
    layout = build_stacked_layout(compiled)
    # Literal model columns are independent of any stored-design renderer.
    literal = (
        np.column_stack((np.ones(n), x, category == "a", category == "b")),
        z[:, None],
    )
    score = np.column_stack((np.sin(np.arange(n)), np.linspace(-0.3, 0.8, n)))
    curvature = np.column_stack(
        (np.linspace(-0.7, 1.1, n), np.cos(np.arange(n)), np.linspace(1.3, -0.2, n))
    )
    curvature[::4] = 0
    return layout, literal, score, curvature


def _stream(layout, score, curvature):
    for start in range(0, len(score), 8):
        stop = min(start + 8, len(score))
        rows = np.arange(start, stop, dtype=np.intp)
        yield SimpleNamespace(
            plans=tuple(
                PredictorExecutionPlan(
                    state.design.row_subset(rows), state.intercept_index is not None
                )
                for state in layout.predictors
            ),
            score_eta=score[start:stop],
            curvature_packed=curvature[start:stop],
        )


def _assemble(monkeypatch, *, tiny_cross=False, problem=None):
    layout, literal, score, curvature = _problem() if problem is None else problem
    if tiny_cross:
        curvature[:, 1] *= 2.0**-200
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **kwargs: _stream(layout, score, curvature),
    )
    coefficients = np.linspace(-0.2, 0.1, layout.n_coefficients)
    penalty = np.diag(np.linspace(0, 0.3, layout.n_coefficients))
    result = chunks.assemble_chunked_geometry(
        None,
        layout,
        None,
        None,
        coefficients,
        penalty=penalty,
        chunk_size=8,
        curvature_source="observed",
    )
    return result, layout, literal, score, curvature, penalty, coefficients


def _assert_geometry(result, layout, literal, score, curvature, penalty, coefficients):
    expected_score = np.concatenate([matrix.T @ score[:, j] for j, matrix in enumerate(literal)])
    expected = np.zeros_like(penalty)
    for channel, (left, right) in enumerate(((0, 0), (0, 1), (1, 1))):
        block = literal[left].T @ (curvature[:, channel, None] * literal[right])
        ls = layout.predictors[left].coefficient_slice
        rs = layout.predictors[right].coefficient_slice
        expected[ls, rs] = block
        expected[rs, ls] = block.T
    absolute_design = np.column_stack([np.abs(matrix) for matrix in literal])
    scale = np.linalg.norm(absolute_design, "fro") ** 2 * np.max(np.abs(curvature))
    tolerance = 32 * len(score) * np.finfo(float).eps * max(1, scale)
    np.testing.assert_allclose(result.score_data, expected_score, rtol=0, atol=tolerance)
    np.testing.assert_allclose(result.data_curvature, expected, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        result.score_penalized, expected_score - penalty @ coefficients, rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        result.penalized_curvature, expected + penalty, rtol=0, atol=tolerance
    )
    np.testing.assert_array_equal(result.data_curvature, result.data_curvature.T)


def test_default_geometry_matches_independent_literal_design(monkeypatch):
    _assert_geometry(*_assemble(monkeypatch))


def test_tiny_signed_cross_channel_matches_the_literal_design(monkeypatch):
    cross = PredictorExecutionPlan.cross_tasks
    contracted_rows = []

    def counted_cross(self, right, weights, **kwargs):
        contracted_rows.append(len(weights))
        return cross(self, right, weights, **kwargs)

    monkeypatch.setattr(PredictorExecutionPlan, "cross_tasks", counted_cross)
    result, layout, literal, _, curvature, _, _ = _assemble(monkeypatch, tiny_cross=True)
    expected = literal[0].T @ (curvature[:, 1, None] * literal[1])
    absolute_sum = np.abs(literal[0]).T @ (np.abs(curvature[:, 1, None]) * np.abs(literal[1]))
    tolerance = 32 * len(curvature) * np.finfo(float).eps * np.max(absolute_sum)
    np.testing.assert_allclose(
        result.data_curvature[
            layout.predictors[0].coefficient_slice, layout.predictors[1].coefficient_slice
        ],
        expected,
        rtol=0,
        atol=tolerance,
    )
    # Four 8-row chunks stream in; the cross channel is contracted once over all rows.
    assert contracted_rows == [len(curvature)]


def _mixed_problem(*, extended=False):
    n = 29
    rows = np.arange(n)
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.1, 0.8, n),
            "s": np.cos(rows * 0.7),
            "t": np.sin(rows * 0.4),
            "g": np.resize(list("abc"), n),
            "h": np.resize(list("pqrst"), n),
            "u": np.cos(rows * 0.3),
        }
    )
    location = {"x": Numeric(), "s": Spline(kind="cr", k=4), "g": Categorical()}
    interactions = (("s", "g"),)
    if extended:
        # A random effect, an exact spline and a second spline term on the
        # same categorical.
        location |= {
            "h": RandomEffect(),
            "t": Spline(kind="cr", k=4),
            "u": Spline(kind="cr", k=5, discrete=False),
        }
        interactions += (("t", "g"), ("u", "g"))
    family = GaussianLS()
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(np.ones(n)),
        family.parameters,
        (
            Predictor("location", location, interactions=interactions),
            Predictor("scale", {"s": Spline(kind="cr", k=4)}),
        ),
        model_discrete=True,
        n_bins_config=9,
    )
    layout = build_stacked_layout(compiled)
    score = np.column_stack((np.sin(rows), np.linspace(-0.3, 0.8, n)))
    curvature = np.column_stack(
        (np.linspace(-0.7, 1.1, n), np.cos(rows), np.linspace(1.3, -0.2, n))
    )
    literal = None if extended else _literal_design(layout)
    return layout, literal, score, curvature


def test_two_chunk_blocks_match_the_literal_mixed_design(monkeypatch):
    problem = _mixed_problem()
    kinds = {type(g).__name__ for g in problem[0].predictors[0].design.group_matrices}
    assert "DiscretizedSplineCategoricalGroupMatrix" in kinds
    # The location predictor packs its intercept, numeric and two categorical columns.
    assert chunks.predictor_panels(problem[0])[0].width == 1 + 1 + 2
    monkeypatch.setattr(chunks, "contraction_block_rows", lambda n, row_bytes, rows: 2 * rows)
    moments = []
    for name in ("diagonal_tasks", "cross_tasks"):
        original = getattr(PredictorExecutionPlan, name)

        def counted(self, *args, _original=original, **kwargs):
            moments.append(self.design.n)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(PredictorExecutionPlan, name, counted)
    _assert_geometry(*_assemble(monkeypatch, problem=problem))
    # Four 8-row chunks: blocks of 16 and 13 rows, three channels each.
    assert moments == [16] * 3 + [13] * 3


def test_a_pass_of_one_block_keeps_each_group_pairs_product(monkeypatch):
    problem = _mixed_problem()
    k = len(problem[0].predictors)
    assert chunks.predictor_panels(problem[0])[0] is not None
    # The 29 rows are one block, which packs no panel: its sums are the group pairs' own.
    packed, *_ = _assemble(monkeypatch, problem=problem)
    monkeypatch.setattr(chunks, "predictor_panels", lambda layout: (None,) * k)
    unpacked, *_ = _assemble(monkeypatch, problem=problem)
    for name in ("score_data", "data_curvature"):
        actual, expected = getattr(packed, name), getattr(unpacked, name)
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64)), name


def test_blocks_whose_sums_could_leave_the_range_contract_chunk_by_chunk(monkeypatch):
    layout, literal, score, curvature = _problem()
    # A sum over the 29-row block could pass 2**1000; each 8-row chunk's cannot.
    score *= 2.0**996
    curvature *= 2.0**996
    moments = []
    for name in ("diagonal_tasks", "cross_tasks"):
        original = getattr(PredictorExecutionPlan, name)

        def counted(self, *args, _original=original, **kwargs):
            moments.append(self.design.n)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(PredictorExecutionPlan, name, counted)
    _assert_geometry(*_assemble(monkeypatch, problem=(layout, literal, score, curvature)))
    assert moments == [8] * 9 + [5] * 3


def _by_bin_contractions(monkeypatch):
    contracted = []
    original = algebra._by_bin_contraction

    def counted(gm, cache, table, **kwargs):
        contracted.append(table.shape)
        return original(gm, cache, table, **kwargs)

    monkeypatch.setattr(algebra, "_by_bin_contraction", counted)
    return contracted


@pytest.mark.parametrize("chunks_per_block", [1, 2])
@pytest.mark.parametrize("kept", [True, False], ids=["kept", "declined"])
def test_by_bin_tables_are_contracted_once_per_pass(monkeypatch, chunks_per_block, kept):
    problem = _mixed_problem()
    monkeypatch.setattr(
        chunks, "contraction_block_rows", lambda n, row_bytes, rows: chunks_per_block * rows
    )
    if not kept:
        monkeypatch.setattr(chunks, "PassTables", lambda budget: algebra.PassTables(0))
    contracted = _by_bin_contractions(monkeypatch)
    _assert_geometry(*_assemble(monkeypatch, problem=problem))
    # Four by-bin pairs: the numeric column and the categorical against the
    # location spline, and both against the scale spline. Four 8-row chunks
    # form 4 or 2 blocks; kept tables contract once per pass.
    blocks = -(-4 // chunks_per_block)
    assert len(contracted) == (4 if kept else 4 * blocks)


def _joint_contractions(monkeypatch):
    contracted, histograms = [], []
    original_block, original_hist = algebra._joint_block, algebra._disc_disc_2d_hist

    def block(kind, gm_i, gm_j, H, cache):
        contracted.append(kind)
        return original_block(kind, gm_i, gm_j, H, cache)

    def hist(*args):
        histograms.append(len(args[2]))
        return original_hist(*args)

    monkeypatch.setattr(algebra, "_joint_block", block)
    monkeypatch.setattr(algebra, "_disc_disc_2d_hist", hist)
    return contracted, histograms


@pytest.mark.parametrize("chunks_per_block", [1, 2])
@pytest.mark.parametrize("kept", [True, False], ids=["kept", "declined"])
def test_joint_histograms_are_contracted_once_per_pass(monkeypatch, chunks_per_block, kept):
    problem = _mixed_problem()
    monkeypatch.setattr(
        chunks, "contraction_block_rows", lambda n, row_bytes, rows: chunks_per_block * rows
    )
    if not kept:
        monkeypatch.setattr(chunks, "PassTables", lambda budget: algebra.PassTables(0))
    contracted, histograms = _joint_contractions(monkeypatch)
    _assert_geometry(*_assemble(monkeypatch, problem=problem))
    # Five binned pairs: the location spline with each of the two levels of its
    # by-category term, and the scale spline with the location spline and with
    # each level. Kept, each histogram sums every block and is contracted once;
    # declined, every block forms and contracts its own.
    blocks = -(-4 // chunks_per_block)
    if kept:
        assert sorted(contracted) == ["level"] * 2 + ["level_t"] * 2 + ["supports"]
        assert histograms == []
    else:
        assert contracted == []
        assert len(histograms) == 5 * blocks


def test_pass_tables_are_contracted_before_their_sums_could_leave_the_range(monkeypatch):
    layout, literal, score, curvature = _mixed_problem()
    # One 8-row block's sums stay below 2**1000; two blocks' could pass it.
    score *= 2.0**996
    curvature *= 2.0**996
    monkeypatch.setattr(chunks, "contraction_block_rows", lambda n, row_bytes, rows: rows)
    contracted = _by_bin_contractions(monkeypatch)
    _assert_geometry(*_assemble(monkeypatch, problem=(layout, literal, score, curvature)))
    # The tables hold the first block and are contracted before the second;
    # the other three blocks contract their own four pairs.
    assert len(contracted) == 4 + 3 * 4


def test_a_block_of_several_chunks_contracts_the_tables_before_the_next_could_pass_the_range(
    monkeypatch,
):
    """Blocks of two chunks release the pass tables once the next block's sums could pass 2**1000.

    The first 16-row block's sums stay below the limit, so it adds into the
    tables; the next 13 rows could take them past it, so ``contract()``
    contracts the tables before that block and the block contracts its own
    pairs. Mutation check: without the range check in ``contract()`` the
    tables sum both blocks and contract once, four contractions instead of
    eight, and none before the second block.
    """
    layout, literal, score, curvature = _mixed_problem()
    score *= 2.0**995
    curvature *= 2.0**995
    peak = max(np.max(np.abs(score)), np.max(np.abs(curvature)))
    assert 16 * peak <= chunks._BLOCK_SUM_LIMIT < 29 * peak
    monkeypatch.setattr(chunks, "contraction_block_rows", lambda n, row_bytes, rows: 2 * rows)
    contracted = _by_bin_contractions(monkeypatch)
    joint, histograms = _joint_contractions(monkeypatch)
    # By-bin contractions made before each block adds its rows.
    before = []
    original = chunks.GroupedGeometryAccumulator.add_block

    def block(self, plans, *args, **kwargs):
        before.append((plans[0].design.n, len(contracted)))
        return original(self, plans, *args, **kwargs)

    monkeypatch.setattr(chunks.GroupedGeometryAccumulator, "add_block", block)
    _assert_geometry(*_assemble(monkeypatch, problem=(layout, literal, score, curvature)))
    # Each block is contracted whole; the first block's four by-bin and five
    # joint tables are contracted before the second, which forms its own.
    assert before == [(16, 0), (13, 4)]
    assert len(contracted) == 4 + 4
    assert sorted(joint) == ["level"] * 2 + ["level_t"] * 2 + ["supports"]
    assert len(histograms) == 5


def _table_adds(monkeypatch):
    """Each block's add into a kept pass table, recorded as whether a pool worker made it."""
    adds = []
    original = algebra.PassTables.add

    def counted(self, key, gm_i, gm_j, W, cache):
        kept = original(self, key, gm_i, gm_j, W, cache)
        if kept:
            adds.append(cache._shared is not None)
        return kept

    monkeypatch.setattr(algebra.PassTables, "add", counted)
    return adds


@pytest.mark.parametrize("chunks_per_block", [1, 2])
@pytest.mark.parametrize("extended", [False, True], ids=["mixed", "extended"])
def test_pass_tables_filled_on_a_pool_keep_the_serial_bits(monkeypatch, chunks_per_block, extended):
    """Pass tables filled by four pool workers give the serial score and curvature to the bit.

    Each block's queue (level and panel pairs, the by-bin and joint table
    adds, the score and formation tasks) runs on four workers, each adding
    into the tables through its own view of the cache; every key is written by
    its own task, so the pooled pass is the serial one. The extended problem
    puts random-effect and exact-level groups in the same queue. Mutation
    check: a worker view that drops its table adds, or a worker add that
    overwrites the earlier blocks' sums as a lost update would, changes the
    curvature's bits.
    """
    monkeypatch.setattr(
        chunks, "contraction_block_rows", lambda n, row_bytes, rows: chunks_per_block * rows
    )
    # Every task its own unit, and no task's overhead serial, so blocks this
    # small still reach the pool.
    monkeypatch.setattr(block_queue, "_TINY_COST", 0.0)
    monkeypatch.setattr(block_queue, "_TASK_OVERHEAD_COST", 1)
    adds = _table_adds(monkeypatch)
    workers = []
    original_section = parallel.section_executor

    def recorded(count, name):
        workers.append(count)
        return original_section(count, name)

    monkeypatch.setattr(parallel, "section_executor", recorded)
    runs = []
    for config in ({"workers": 1}, {"workers": 4, "min_cost": 0}):
        adds.clear()
        workers.clear()
        with block_queue_config(**config):
            result, *_ = _assemble(monkeypatch, problem=_mixed_problem(extended=extended))
        runs.append((result, list(adds), list(workers)))
    (serial, serial_adds, serial_workers), (pooled, pooled_adds, pooled_workers) = runs
    # Every block adds into each kept table: the mixed problem's four by-bin
    # and five joint pairs, or the extended problem's twenty-four.
    blocks = -(-4 // chunks_per_block)
    assert len(serial_adds) == (24 if extended else 9) * blocks
    assert not any(serial_adds) and not serial_workers
    assert pooled_adds == [True] * len(serial_adds) and max(pooled_workers) > 1
    for name in ("score_data", "data_curvature"):
        actual, expected = getattr(pooled, name), getattr(serial, name)
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64)), name


def test_an_assembly_projects_each_support_once(monkeypatch):
    problem = _mixed_problem()
    monkeypatch.setattr(chunks, "contraction_block_rows", lambda n, row_bytes, rows: rows)
    formed, calls = [], []
    original = algebra.PassFactors.projection

    def counted(self, B_unique, R_inv):
        key = (id(B_unique), id(R_inv))
        calls.append(key)
        if key not in self.projections:
            formed.append(key)
        return original(self, B_unique, R_inv)

    monkeypatch.setattr(algebra.PassFactors, "projection", counted)
    _assert_geometry(*_assemble(monkeypatch, problem=problem))
    groups = [group for state in problem[0].predictors for group in state.design.group_matrices]
    supports = {
        (id(group.B_unique), id(group.R_inv))
        for group in groups
        if type(group) is DiscretizedSSPGroupMatrix
    }
    # Binned levels project their support too, where their tables contract.
    levels = {
        (id(group.B_unique), id(group.R_inv))
        for group in groups
        if type(group).__name__ == "DiscretizedSplineCategoricalGroupMatrix"
    }
    # Each block and channel asks again; each support is formed once.
    assert len(supports) == 2
    assert len(formed) == len(set(formed)) and supports <= set(formed) <= supports | levels
    assert all(calls.count(key) >= 4 for key in supports)


def test_assemblies_sharing_factors_project_each_support_once(monkeypatch):
    layout, _literal, score, curvature = _mixed_problem()
    monkeypatch.setattr(chunks, "contraction_block_rows", lambda n, row_bytes, rows: rows)
    monkeypatch.setattr(
        chunks,
        "iter_likelihood_chunks",
        lambda *args, **kwargs: _stream(layout, score, curvature),
    )
    formed = []
    original = algebra.PassFactors.projection

    def counted(self, B_unique, R_inv):
        if (id(B_unique), id(R_inv)) not in self.projections:
            formed.append(1)
        return original(self, B_unique, R_inv)

    monkeypatch.setattr(algebra.PassFactors, "projection", counted)

    def assemble(factors):
        return chunks.assemble_chunked_geometry(
            None,
            layout,
            None,
            None,
            np.linspace(-0.2, 0.1, layout.n_coefficients),
            penalty=np.diag(np.linspace(0, 0.3, layout.n_coefficients)),
            chunk_size=8,
            curvature_source="observed",
            pass_factors=factors,
        )

    alone = assemble(None)
    per_assembly = len(formed)
    del formed[:]
    shared = algebra.PassFactors()
    first, second = assemble(shared), assemble(shared)
    # The second assembly finds every support projected; the geometry is the same bits.
    assert per_assembly > 0 and len(formed) == per_assembly
    for name in ("score_data", "data_curvature"):
        for result in (first, second):
            actual, expected = getattr(result, name), getattr(alone, name)
            assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64)), name


def _arrays(group):
    names = {name for kind in type(group).__mro__ for name in getattr(kind, "__slots__", ())}
    names |= set(getattr(group, "__dict__", {}))
    return [value for name in names if isinstance(value := getattr(group, name, None), np.ndarray)]


def _sparse_arrays(group):
    """The value, index and pointer arrays of a group's sparse attributes."""
    names = {name for kind in type(group).__mro__ for name in getattr(kind, "__slots__", ())}
    names |= set(getattr(group, "__dict__", {}))
    return [
        array
        for name in names
        if sp.issparse(value := getattr(group, name, None))
        for array in (value.data, value.indices, value.indptr)
    ]


def _owned_bytes(parent, child, block_rows):
    """Bytes a block child owns that share no memory with its parent, and its rows."""
    parents = _arrays(parent) + _sparse_arrays(parent)
    # A level group's row arrays cover only its level's rows; an exact level
    # group's value copy has one entry per stored value.
    child_rows = len(child.row_idx) if hasattr(child, "row_idx") else block_rows
    level = getattr(child, "B_level", None)
    shapes = {(child_rows,)} | ({(level.nnz,)} if sp.issparse(level) else set())
    # A sparse basis owns its slice whatever the arrays' lengths.
    candidates = [a for a in _arrays(child) if a.shape[:1] in shapes]
    candidates += _sparse_arrays(child)
    distinct = {a.__array_interface__["data"][0]: a for a in candidates}
    owned = sum(
        array.nbytes
        for array in distinct.values()
        if not any(np.shares_memory(array, source) for source in parents)
    )
    return owned, child_rows


def test_block_design_rows_own_no_more_than_the_budget_counts():
    layout = _mixed_problem(extended=True)[0]
    rows = chunks.RowChunk(start=5, stop=21, indices=np.arange(5, 21, dtype=np.intp))
    length = len(rows.indices)
    kinds, level_groups = set(), 0
    for state in layout.predictors:
        block = chunks._row_design(state, rows, copy=False)
        # Contract once, so arrays a group builds lazily are counted too.
        PredictorExecutionPlan(block, state.intercept_index is not None).diagonal_moment(
            np.linspace(0.5, 1.5, length)
        )
        # Owned spline-by-category bytes, spread over the block rows each level owns.
        per_row = np.zeros(length)
        for parent, child in zip(state.design.group_matrices, block.group_matrices, strict=True):
            owned, child_rows = _owned_bytes(parent, child, length)
            kinds.add(type(child).__name__)
            if type(parent) in chunks._SPLINE_CATEGORY_GROUPS:
                level_groups += 1
                if child_rows:
                    assert np.max(child.row_idx) < length
                    per_row[child.row_idx] += owned / child_rows
                continue
            counted = chunks._group_row_floats(parent)
            assert owned <= 8 * length * counted, (type(child).__name__, owned, counted)
        charged = chunks._spline_category_row_floats(state.design)
        assert np.max(per_row) <= 8 * charged, (np.max(per_row), charged)
    assert {"CategoricalGroupMatrix", "RandomEffectGroupMatrix"} <= kinds
    assert {"DiscretizedSSPGroupMatrix", "DiscretizedSplineCategoricalGroupMatrix"} <= kinds
    assert {"SparseSSPGroupMatrix", "SplineCategoricalGroupMatrix"} <= kinds
    # Two binned spline terms on one categorical, two non-base levels each.
    assert level_groups == 4


def test_chunk_design_rows_own_no_more_than_the_chunk_selector_counts():
    layout = _mixed_problem(extended=True)[0]
    rows = chunks.RowChunk(start=5, stop=21, indices=np.arange(5, 21, dtype=np.intp))
    length = len(rows.indices)
    # Owned bytes per chunk row: a level's own arrays over its rows, the rest
    # (an exact level group's basis slice included) over every row.
    per_row = np.zeros(length)
    kinds = set()
    for state in layout.predictors:
        chunk = chunks._row_design(state, rows)
        for parent, child in zip(state.design.group_matrices, chunk.group_matrices, strict=True):
            kinds.add(type(child).__name__)
            owned, child_rows = _owned_bytes(parent, child, length)
            basis = getattr(child, "B", None) if hasattr(child, "row_idx") else None
            spread = 0
            if sp.issparse(basis):
                spread = sum(a.nbytes for a in (basis.data, basis.indices, basis.indptr))
            per_row += spread / length
            if hasattr(child, "row_idx"):
                if child_rows:
                    per_row[child.row_idx] += (owned - spread) / child_rows
            else:
                per_row += owned / length
    assert np.max(per_row) <= 8 * chunks._chunk_design_floats(layout)
    assert {"DenseGroupMatrix", "CategoricalGroupMatrix", "RandomEffectGroupMatrix"} <= kinds
    assert {"DiscretizedSSPGroupMatrix", "DiscretizedSplineCategoricalGroupMatrix"} <= kinds
    assert {"SparseSSPGroupMatrix", "SplineCategoricalGroupMatrix"} <= kinds


def test_an_exact_level_group_filling_its_block_owns_no_more_than_its_charge():
    n = 40
    rows = np.arange(n)
    frame = pd.DataFrame(
        {
            "u": np.cos(rows * 0.3),
            # Sorted levels, so the block below lies inside level b.
            "g": np.where(rows < 20, "a", "b"),
        }
    )
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(np.ones(n)),
        GaussianLS().parameters,
        (
            Predictor(
                "location",
                {"u": Spline(kind="cr", k=6, discrete=False), "g": Categorical()},
                interactions=(("u", "g"),),
            ),
            Predictor("scale", {}),
        ),
        model_discrete=True,
        n_bins_config=9,
    )
    state = build_stacked_layout(compiled).predictors[0]
    block_rows = chunks.RowChunk(start=22, stop=38, indices=np.arange(22, 38, dtype=np.intp))
    length = len(block_rows.indices)
    block = chunks._row_design(state, block_rows, copy=False)
    # Contract once, so a saturated level's dense rows are built.
    PredictorExecutionPlan(block, state.intercept_index is not None).diagonal_moment(
        np.linspace(0.5, 1.5, length)
    )
    filled = []
    for parent, child in zip(state.design.group_matrices, block.group_matrices, strict=True):
        if type(child).__name__ != "SplineCategoricalGroupMatrix":
            continue
        owned, child_rows = _owned_bytes(parent, child, length)
        if child_rows == length:
            filled.append(type(child).__name__)
            counted = chunks._group_row_floats(parent)
            assert owned <= 8 * length * counted, (owned, counted)
    assert filled == ["SplineCategoricalGroupMatrix"]


def test_a_saturated_exact_spline_block_holds_one_dense_copy_of_its_rows():
    """A cubic basis with a knot row's zeros dropped is no dense view; its gram scratch is charged.

    The charge's weighted-copy term is 8 * width bytes per row. Mutation check:
    weighting every row at once beside the dense rows holds two copies (16 * width;
    the traced peak is 2.6 MB against 1.6 MB and the 1.9 MB bound).
    """
    n = 20_000
    x = np.random.default_rng(5).uniform(0.0, 1.0, n)
    x[0], x[-1] = 0.0, 1.0  # knot rows: both cubic terms vanish, so stored zeros drop
    compiled = compile_predictors(
        as_eager_frame(pd.DataFrame({"x": x})),
        resolved_prior(np.ones(n)),
        GaussianLS().parameters,
        (
            Predictor("location", {"x": Spline(kind="cr_cardinal", k=8, discrete=False)}),
            Predictor("scale", {}),
        ),
        model_discrete=True,
        n_bins_config=9,
    )
    state = build_stacked_layout(compiled).predictors[0]
    parent = next(
        g for g in state.design.group_matrices if type(g).__name__ == "SparseSSPGroupMatrix"
    )
    width = parent.B.shape[1]
    assert parent.B.data.size < n * width  # stored zeros dropped: not a dense view
    assert parent.B.data.size >= 0.9 * n * width  # saturated
    block_rows = chunks.RowChunk(start=0, stop=n, indices=np.arange(n, dtype=np.intp))
    block = chunks._row_design(state, block_rows, copy=False)
    plan = PredictorExecutionPlan(block, state.intercept_index is not None)
    weights = np.linspace(0.5, 1.5, n)
    peak, _ = own_traced_peak(plan.diagonal_moment, weights)
    assert peak < 1.5 * 8 * n * width, (peak, 8 * n * width)
    # The sub-blocks pair each row with its own weight: the gram matches one product.
    gram = _group_matrix_core._saturated_ssp_gram(parent.B, weights)
    dense = parent.B.toarray()
    reference = (dense * weights[:, None]).T @ dense
    scale = (np.abs(dense) * weights[:, None]).T @ np.abs(dense)
    gamma = n * 2.0**-53 / (1.0 - n * 2.0**-53)
    assert np.all(np.abs(gram - reference) <= 2.0 * gamma * scale)
