"""The Gram's block queue: blocks on worker threads, bitwise the serial Gram at any worker count."""

from __future__ import annotations

import sys
import threading
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm._group_matrix._block_queue as queue
import superglm._group_matrix._group_matrix_algebra as algebra
import superglm._group_matrix._group_matrix_execution as execution
import superglm._group_matrix._group_matrix_kernels as kernels
import superglm._parallel as parallel
from superglm import Categorical, Spline, SuperGLM
from superglm._group_matrix._block_queue import BlockTask, block_queue_config
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm._parallel import parallel_config
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
)

WORKERS = (1, 2, 8)


def _assert_bitwise(actual, expected) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # Compare encodings, so a -0.0 against 0.0 counts as a difference too.
    assert np.array_equal(
        np.ascontiguousarray(actual).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8)
    )


def _frame(n: int, seed: int = 5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.uniform(-1.0, 1.0, n) for c in "abcd"})
    # A level carrying more than half the weight, as the real book's VehGas does.
    X["g"] = pd.Categorical(rng.choice(list("pqrs"), size=n, p=[0.55, 0.2, 0.15, 0.1]))
    eta = -1.0 + 0.5 * np.sin(2.0 * X["a"]) + 0.3 * X["b"] * X["c"] + 0.2 * X["d"]
    return X, rng.poisson(np.exp(eta)).astype(float), rng


@pytest.fixture(scope="module")
def tensor_plan():
    """A discrete design: four spline mains, a categorical, four tensors (two share margins)."""
    X, y, rng = _frame(20_000)
    model = SuperGLM(
        family="poisson",
        discrete=True,
        n_bins=64,
        features={**{c: Spline(kind="ps", k=8) for c in "abcd"}, "g": Categorical()},
        interactions=[("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model._build_design_matrix(X, y, None, None)
    plan = model._dm.execution_plan
    W = rng.uniform(0.05, 3.0, plan.n)
    return plan, W, rng.normal(size=plan.n), rng.normal(size=plan.n)


def _mixed_plan(n: int = 6_000):
    """Every other block kind: dense, sparse, categorical, binned spline, SSP, factor smooths."""
    rng = np.random.default_rng(11)
    sparse_values = rng.normal(size=(n, 4))
    sparse_values[rng.random(sparse_values.shape) < 0.7] = 0.0
    bins = rng.integers(0, 24, n)
    support = rng.uniform(0.0, 1.0, (24, 5))
    codes = rng.integers(0, 6, n)
    natural_map = rng.normal(size=(5, 3))
    smooths = [
        FactorSmoothGroupMatrix(
            support if discrete else sp.csr_matrix(support[bins]),
            codes,
            6,
            natural_map=natural_map,
            levels=tuple("uvwxyz"),
            repeated_penalty_components=(("wiggle", np.eye(3)),),
            factor_basis=basis,
            bin_idx=bins if discrete else None,
        )
        for discrete, basis in ((True, "fs"), (False, "sz"))
    ]
    csr = sp.csr_matrix(rng.normal(size=(n, 6)) * (rng.random((n, 6)) < 0.5))
    groups = [
        DenseGroupMatrix(rng.normal(size=(n, 3))),
        SparseGroupMatrix(sp.csr_matrix(sparse_values)),
        CategoricalGroupMatrix(rng.integers(0, 9, n), n_levels=9),
        DiscretizedSSPGroupMatrix(support, rng.normal(size=(5, 4)), bins),
        SparseSSPGroupMatrix(csr, rng.normal(size=(6, 4))),
        *smooths,
    ]
    plan = MatrixExecutionPlan(groups, n=n, ordinary_tabmat=False)
    return plan, rng.uniform(0.1, 2.0, n), rng.normal(size=n)


def _moments(plan, W, z, **config):
    profile: dict = {}
    with block_queue_config(**config):
        moments = plan.moments(W, rhs=(z,), include_xtw=True, profile=profile)
    return moments, profile


@pytest.mark.parametrize("split", [None, 3])
@pytest.mark.parametrize("workers", WORKERS)
def test_pooled_gram_is_bitwise_the_serial_gram(tensor_plan, workers, split):
    """Whole blocks, and a tensor pair split by cells, are bitwise the serial blocks.

    Each worker owns its block, its scratch and its profile; the shared
    weighted state is formed once and handed out in the serial layout.
    """
    plan, W, z, signed_W = tensor_plan
    expected, _ = _moments(plan, W, z, workers=1)
    with block_queue_config(workers=1):
        signed_expected = plan.moments(signed_W, signed=True).gram
    actual, profile = _moments(plan, W, z, workers=workers, min_cost=0, split=split)
    _assert_bitwise(actual.gram, expected.gram)
    _assert_bitwise(actual.xtw, expected.xtw)
    _assert_bitwise(actual.xt_rhs[0], expected.xt_rhs[0])
    if workers > 1:
        assert profile["block_pool_workers"] == workers
    if split is not None:
        # The forced split reached the raw-band route it splits.
        assert profile["block_cross_tensor_tensor_channel_split"] > 0
    signed_profile: dict = {}
    with block_queue_config(workers=workers, min_cost=0, split=split):
        signed = plan._moments_prevalidated(signed_W, signed=True, profile=signed_profile)
    _assert_bitwise(signed.gram, signed_expected)
    assert signed_profile.get("block_pool_workers", 1) == workers


def test_pooled_gram_of_every_other_block_kind_is_bitwise_serial():
    plan, W, z = _mixed_plan()
    expected, serial_profile = _moments(plan, W, z, workers=1)
    for workers in WORKERS[1:]:
        actual, profile = _moments(plan, W, z, workers=workers, min_cost=0)
        _assert_bitwise(actual.gram, expected.gram)
        _assert_bitwise(actual.xtw, expected.xtw)
        _assert_bitwise(actual.xt_rhs[0], expected.xt_rhs[0])
        assert profile["block_pool_workers"] == workers
        # Shared entries are formed once, as the serial assembly forms them.
        assert profile.get("block_solver_support_builds") == serial_profile.get(
            "block_solver_support_builds"
        )


def test_an_assembly_without_blocks_queues_nothing():
    """A plan with no group (an intercept-only predictor) has no task, at any setting."""
    W, z = np.linspace(0.5, 1.5, 40), np.linspace(-1.0, 1.0, 40)
    plan = MatrixExecutionPlan([], n=40)
    for config in ({}, {"workers": 8, "min_cost": 0}, {"split": 3}):
        moments, profile = _moments(plan, W, z, **config)
        assert moments.gram.shape == (0, 0) and moments.xtw.shape == (0,)
        assert moments.xt_rhs[0].shape == (0,)
        assert "block_pool_calls" not in profile


def _kernel_calls():
    """One call of every pooled kernel on inputs large enough to overlap across threads."""
    rng = np.random.default_rng(29)
    n, n1, n2, levels = 60_000, 23, 17, 7
    idx1, idx2 = rng.integers(0, n1, n), rng.integers(0, n2, n)
    W, Wz = rng.uniform(0.0, 2.0, n), rng.normal(size=n)
    codes, other_codes = rng.integers(0, levels, n), rng.integers(0, levels + 1, n)
    csr = sp.csr_matrix(rng.normal(size=(n, 6)) * (rng.random((n, 6)) < 0.4))
    data, indices, indptr = csr.data, csr.indices, csr.indptr
    other = sp.csr_matrix(rng.normal(size=(n, 4)) * (rng.random((n, 4)) < 0.5))
    dense = rng.normal(size=(n, 3))
    table1, table2 = rng.normal(size=(n1, 5)), rng.normal(size=(n2, 5))
    ptr, order = kernels._cell_csr(idx1, idx2, n1, n2)
    bin1, bin2 = kernels._gather_cell_order(order, idx2, idx1)
    k1_raw, k2_raw = 9, 8
    offsets1, offsets2 = rng.integers(0, k1_raw - 2, n2), rng.integers(0, k2_raw - 3, n1)
    values1, values2 = rng.normal(size=(n2, 3)), rng.normal(size=(n1, 4))
    support = rng.uniform(size=(n1, 4))
    raw_coefficients = rng.normal(size=(levels, 6))
    return {
        "_tensor_operand_in_reassociation_range": lambda: (
            kernels._tensor_operand_in_reassociation_range(dense)
        ),
        "_float64_operand_exponent_bounds": lambda: kernels._float64_operand_exponent_bounds(dense),
        "_operand_exponent_bounds": lambda: kernels._operand_exponent_bounds(dense),
        "_indexed_row_dot": lambda: kernels._indexed_row_dot(table1, table2, idx1, idx2),
        "_csr_weighted_gram": lambda: kernels._csr_weighted_gram(data, indices, indptr, W, 6),
        "_csr_weighted_cross": lambda: kernels._csr_weighted_cross(
            data, indices, indptr, other.data, other.indices, other.indptr, W, 6, 4
        ),
        "_weighted_bincount_2d": lambda: kernels._weighted_bincount_2d(idx1, W, dense, n1),
        "_support_weighted_bincount_2d": lambda: (
            kernels._support_weighted_bincount_2d(
                out := np.zeros((n2, 3)), idx2, W, support, idx1, 1
            )
            or out
        ),
        "_csr_weighted_bincount": lambda: kernels._csr_weighted_bincount(
            data, indices, indptr, 6, idx1, W, n1
        ),
        "_disc_disc_2d_hist": lambda: kernels._disc_disc_2d_hist(idx1, idx2, W, n1, n2),
        "_disc_disc_2d_hist_channels": lambda: kernels._disc_disc_2d_hist_channels(
            idx1, idx2, idx2, W, table2, n1, n2
        ),
        "_gather_cell_order": lambda: kernels._gather_cell_order(order, idx2, idx1),
        "_cell_hist_raw_kron": lambda: (
            kernels._cell_hist_raw_kron(
                ptr,
                bin1,
                bin2,
                W[order],
                offsets1,
                values1,
                offsets2,
                values2,
                k2_raw,
                out := np.empty((n1 * n2, k1_raw * k2_raw)),
            )
            or out
        ),
        "_cell_csr_matches": lambda: kernels._cell_csr_matches(ptr, order, idx1, idx2, n1, n2),
        "_cell_csr": lambda: kernels._cell_csr(idx1, idx2, n1, n2),
        "_fused_bincount_2": lambda: kernels._fused_bincount_2(idx1, W, Wz, n1),
        "_factor_smooth_csr_matvec": lambda: kernels._factor_smooth_csr_matvec(
            data, indices, indptr, codes, raw_coefficients
        ),
        "_factor_smooth_support_matvec": lambda: kernels._factor_smooth_support_matvec(
            support, idx1, codes, raw_coefficients[:, :4]
        ),
        "_factor_smooth_csr_rmatvec": lambda: kernels._factor_smooth_csr_rmatvec(
            data, indices, indptr, codes, Wz, levels, 6
        ),
        "_factor_smooth_support_rmatvec": lambda: kernels._factor_smooth_support_rmatvec(
            support, idx1, codes, Wz, levels
        ),
        "_factor_smooth_csr_sufficient_stats": lambda: kernels._factor_smooth_csr_sufficient_stats(
            data, indices, indptr, codes, W, Wz, levels, 6
        ),
        "_factor_smooth_support_cell_aggregates": lambda: (
            kernels._factor_smooth_support_cell_aggregates(idx1, codes, W, Wz, levels, n1)
        ),
        "_factor_smooth_csr_dense_cross": lambda: kernels._factor_smooth_csr_dense_cross(
            data, indices, indptr, codes, W, dense, levels, 6
        ),
        "_factor_smooth_support_dense_cross": lambda: kernels._factor_smooth_support_dense_cross(
            support, idx1, codes, W, dense, levels
        ),
        "_fused_2d_bincount_2": lambda: kernels._fused_2d_bincount_2(idx1, idx2, W, Wz, n1, n2),
        "_cat_weighted_bincount": lambda: kernels._cat_weighted_bincount(
            other_codes, idx1, W, n1, levels
        ),
        "_cat_cat_weighted_crosstab": lambda: kernels._cat_cat_weighted_crosstab(
            codes, other_codes, W, levels, levels
        ),
        "_level_sums": lambda: kernels._level_sums(other_codes, Wz, levels + 1)[0],
    }


def _flatten(value) -> list[np.ndarray]:
    if isinstance(value, tuple):
        return [array for item in value for array in _flatten(item)]
    return [np.atleast_1d(np.asarray(value))]


def test_pooled_kernels_run_without_the_gil_from_many_threads_bitwise():
    """Every pooled kernel releases the GIL, and eight threads at once give the serial bits.

    Each call allocates its own output (or is handed its own), so concurrent
    calls share inputs only; a kernel that wrote anything shared would
    disagree with its serial result here.
    """
    calls = _kernel_calls()
    assert set(calls) == {kernel.py_func.__name__ for kernel in kernels._POOLED_BLOCK_KERNELS}
    for name in calls:
        assert getattr(kernels, name).targetoptions.get("nogil") is True, name
    expected = {name: _flatten(call()) for name, call in calls.items()}
    barrier = threading.Barrier(8)
    mismatches: list[str] = []

    def worker(offset: int) -> None:
        names = list(calls)
        names = names[offset:] + names[:offset]
        barrier.wait()
        for _ in range(3):
            for name in names:
                for got, want in zip(_flatten(calls[name]()), expected[name], strict=True):
                    if not np.array_equal(got.view(np.uint8), want.view(np.uint8)):
                        mismatches.append(name)

    threads = [threading.Thread(target=worker, args=(3 * k,)) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not mismatches


def test_level_sums_are_np_bincount_bitwise():
    """``_level_sums`` adds each row into its level in row order, as ``np.bincount`` does.

    Weights spanning 24 decades of both signs make the sum order-sensitive,
    so any other order (pairwise, by level, reversed: the mutation checks)
    moves bits.  A code outside ``[0, length)`` and a length mismatch are
    refused (``ok`` False) for ``np.bincount`` to handle.
    """
    rng = np.random.default_rng(31)
    n, levels = 50_000, 11
    codes = rng.integers(0, levels + 1, n)
    weights = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-12.0, 12.0, n)
    sums, ok = kernels._level_sums(codes, weights, levels + 1)
    expected = np.bincount(codes, weights=weights, minlength=levels + 1)
    assert ok and np.array_equal(sums.view(np.uint64), expected.view(np.uint64))
    assert not kernels._level_sums(codes, weights, levels)[1]
    assert not kernels._level_sums(codes - 1, weights, levels + 1)[1]
    assert not kernels._level_sums(codes, weights[:-1], levels + 1)[1]


def test_categorical_blocks_sum_their_levels_without_np_bincount(monkeypatch):
    """A categorical Gram diagonal and ``X'w`` no longer go through ``np.bincount``.

    ``np.bincount`` holds the GIL for its whole pass, so the categorical
    diagonals the block queue runs on its workers ran one at a time; against
    the former ``np.bincount`` calls both lines below raise.  The values are
    ``np.bincount``'s to the bit, and an input the kernel refuses still
    reaches ``np.bincount``.
    """
    rng = np.random.default_rng(37)
    n, levels = 20_000, 9
    codes = rng.integers(-1, levels, n)
    W = rng.uniform(0.0, 3.0, n)
    gm = CategoricalGroupMatrix(codes, levels)
    expected_gram = np.diag(np.bincount(gm.codes, weights=W, minlength=levels + 1)[:levels])
    expected_rmatvec = np.bincount(gm.codes, weights=-W, minlength=levels + 1)[:levels]
    bincount = np.bincount

    def refused(*args, **kwargs):
        raise AssertionError("np.bincount on a categorical block")

    monkeypatch.setattr(np, "bincount", refused)
    assert np.array_equal(gm.gram(W).view(np.uint64), expected_gram.view(np.uint64))
    assert np.array_equal(gm.rmatvec(-W).view(np.uint64), expected_rmatvec.view(np.uint64))
    monkeypatch.setattr(np, "bincount", bincount)
    wide = CategoricalGroupMatrix(np.full(4, levels + 2), levels)
    assert np.array_equal(wide.rmatvec(np.ones(4)), np.zeros(levels))


def test_only_the_pooled_block_kernels_release_the_gil_and_none_is_parallel():
    pooled_kernels = kernels._POOLED_BLOCK_KERNELS + kernels._POOLED_LEAF_KERNELS
    pooled = {id(kernel) for kernel in pooled_kernels}
    for kernel in pooled_kernels:
        assert kernel.targetoptions.get("nogil") is True
        assert not kernel.targetoptions.get("parallel")
    released = {
        f"{module.__name__}.{name}"
        for module in list(sys.modules.values())
        if module is not None and module.__name__.startswith("superglm")
        for name, value in list(vars(module).items())
        if type(value).__name__ == "CPUDispatcher"
        and value.targetoptions.get("nogil")
        and id(value) not in pooled
    }
    assert not released


def test_worker_count_is_capped_by_the_memory_budget(tensor_plan):
    """Workers are ``min(n_jobs, units, max_memory // largest task)``; one worker runs serially."""
    plan, W, z, _signed = tensor_plan
    groups, n = plan.group_matrices, plan.n
    largest = max(
        [queue.diagonal_bytes(group, n) for group in groups]
        + [
            queue.block_bytes(left, right, n)
            for i, left in enumerate(groups)
            for right in groups[i + 1 :]
        ]
    )
    expected, _ = _moments(plan, W, z, workers=1)
    with parallel_config(n_jobs=8, max_memory=3 * largest + 1):
        capped, profile = _moments(plan, W, z, min_cost=0)
    assert profile["block_pool_workers"] == 3
    _assert_bitwise(capped.gram, expected.gram)
    with parallel_config(n_jobs=8, max_memory=largest - 1):
        _serial, profile = _moments(plan, W, z, min_cost=0)
    assert "block_pool_calls" not in profile
    with parallel_config(n_jobs=1, max_memory="1T"):
        _serial, profile = _moments(plan, W, z, min_cost=0)
    assert "block_pool_calls" not in profile


def test_a_failing_block_stops_the_pool_and_raises(monkeypatch, tensor_plan):
    plan, W, z, _signed = tensor_plan
    original = execution._cross_gram
    failing = (plan.group_matrices[2], plan.group_matrices[6])

    def cross_gram(left, right, *args, **kwargs):
        if (left, right) == failing:
            raise FloatingPointError("block (2, 6)")
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(execution, "_cross_gram", cross_gram)
    with pytest.raises(FloatingPointError, match=r"block \(2, 6\)"):
        _moments(plan, W, z, workers=4, min_cost=0)
    monkeypatch.setattr(execution, "_cross_gram", original)
    expected, _ = _moments(plan, W, z, workers=1)
    again, _ = _moments(plan, W, z, workers=4, min_cost=0)
    _assert_bitwise(again.gram, expected.gram)


def test_pooled_blocks_keep_the_callers_errstate(monkeypatch, tensor_plan):
    """A block that overflows under the caller's ``errstate(over="ignore")`` stays quiet when pooled.

    Worker threads start with an empty context, so without the caller's
    context they ran NumPy's default ``over="warn"``, which ``simplefilter("error")``
    raised as a task failure that serial assembly never saw.
    """
    plan, W, z, _signed = tensor_plan
    original = execution._cross_gram

    def cross_gram(*args, **kwargs):
        np.multiply(np.array([1e308]), 10.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_cross_gram", cross_gram)
    with warnings.catch_warnings(), np.errstate(over="ignore"):
        warnings.simplefilter("error")
        expected, _ = _moments(plan, W, z, workers=1)
        actual, profile = _moments(plan, W, z, workers=4, min_cost=0)
    assert profile["block_pool_workers"] == 4
    _assert_bitwise(actual.gram, expected.gram)


def test_a_shared_grid_is_formed_once_across_orientations(monkeypatch):
    """A worker whose lookup missed a rival's claim on the transpose waits for it, not re-forms it."""
    shared = algebra._SharedEntries()
    store: dict = {}
    rival = shared.claim(store, ("R",))
    real_lookup = algebra._SharedEntries.lookup
    first = [True]

    def stale_lookup(self, *args):
        # The interleaving: this lookup ran before the rival's claim.
        if first[0]:
            first[0] = False
            return None, None
        return real_lookup(self, *args)

    monkeypatch.setattr(algebra._SharedEntries, "lookup", stale_lookup)
    formed: list[str] = []
    found: list = []
    worker = threading.Thread(
        target=lambda: found.append(
            shared.once(store, "K", lambda: formed.append("K") or np.ones(1), alternate="R")
        )
    )
    worker.start()
    shared.publish(store, rival, ("R",), (np.zeros(1),))
    worker.join(timeout=30)
    assert formed == []
    assert found[0][0] == "R" and found[0][2] is False


def test_a_pooled_worker_holds_no_finished_block():
    """Each block is placed as it is formed, so finished blocks never outnumber the workers."""
    lock = threading.Lock()
    formed, placed, held = [0], [0], []
    values: dict[int, int] = {}

    def run(index):
        def form(cache, profile):
            with lock:
                held.append(formed[0] - placed[0])
                formed[0] += 1
            return index

        return form

    def place(index):
        def write(value):
            with lock:
                placed[0] += 1
                values[index] = value

        return write

    tasks = [BlockTask(i, queue._TINY_COST, 1, run(i), place(i)) for i in range(12)]

    class Cache:
        _batch = None
        _profile = None

        def worker_view(self, shared, profile, work_queue):
            return self

    with block_queue_config(workers=2, min_cost=0):
        queue.run_block_tasks(tasks, Cache(), None)
    assert values == {i: i for i in range(12)}
    assert max(held) <= 2


def test_largest_first_with_tiny_tasks_batched_and_the_oversized_pair_split(monkeypatch):
    def tasks() -> list[BlockTask]:
        small = queue._TINY_COST / 4
        costs = (1, 10, 1, 80, 1, 1, 9)
        return [
            BlockTask(
                index,
                cost * small,
                1,
                lambda cache, profile: None,
                lambda value: None,
                split_bytes=100 if index == 3 else 0,
            )
            for index, cost in enumerate(costs)
        ]

    units = queue._units(tasks())
    assert [[t.index for t in unit] for unit in units] == [[3], [1], [6], [0, 2, 4, 5]]

    seen: list[int] = []

    def workers(n_units, task_bytes):
        seen.append(n_units)
        return 4

    monkeypatch.setattr(parallel, "pool_workers", workers)

    class Cache:
        _batch = None
        _profile = None

        def worker_view(self, shared, profile, work_queue):
            return self

    parts = {}
    for budget in (4 * 1 + 100, 4 * 1 + 99):
        planned = tasks()
        with block_queue_config(min_cost=0), parallel_config(max_memory=budget):
            queue.run_block_tasks(planned, Cache(), None)
        parts[budget] = [t.parts for t in planned]
    assert seen == [4, 4]
    # 80 of 103 small units: above the four-worker bound of 25.75, so ceil(320 / 103) parts,
    # while the budget holds four one-byte tasks and the split's 100 more; not one byte less.
    assert parts == {104: [1, 1, 1, 4, 1, 1, 1], 103: [1] * 7}


def test_blocks_split_at_once_share_one_memory_budget(monkeypatch):
    """Each split adds one histogram, and every split block's parts can run at once.

    Two pairs of 40 small units in 86, each above the eight-worker bound of
    10.75, each adding 100 bytes when split.  A budget of eight one-byte
    tasks and one split splits only the first pair in LPT order (ties in
    serial order); one that covers both splits both.  Checked one split at a
    time, both split under the smaller budget and held 200 bytes against
    its 100.
    """

    def tasks() -> list[BlockTask]:
        small = queue._TINY_COST / 4
        costs = (1, 40, 1, 40, 1, 1, 1, 1)
        return [
            BlockTask(
                index,
                cost * small,
                1,
                lambda cache, profile: None,
                lambda value: None,
                split_bytes=100 if cost == 40 else 0,
            )
            for index, cost in enumerate(costs)
        ]

    monkeypatch.setattr(parallel, "pool_workers", lambda n_units, task_bytes: 8)

    class Cache:
        _batch = None
        _profile = None

        def worker_view(self, shared, profile, work_queue):
            return self

    parts = {}
    for budget in (8 * 1 + 100, 8 * 1 + 199, 8 * 1 + 200):
        planned = tasks()
        with block_queue_config(min_cost=0), parallel_config(max_memory=budget):
            queue.run_block_tasks(planned, Cache(), None)
        parts[budget] = [t.parts for t in planned]
    # ceil(40 * 8 / 86) = 4 parts a split block.
    assert parts == {
        108: [1, 4, 1, 1, 1, 1, 1, 1],
        207: [1, 4, 1, 1, 1, 1, 1, 1],
        208: [1, 4, 1, 4, 1, 1, 1, 1],
    }


def test_estimator_threads_never_change_the_fit(monkeypatch):
    """``n_jobs`` and ``max_memory`` are constructor intent, applied inside the fit only."""
    for bad in (0, -1, True, 1.5, "many"):
        with pytest.raises(ValueError, match="n_jobs"):
            SuperGLM(n_jobs=bad)
    for bad in (0, "lots", True, 2.5, "inf", "1e400", "nan", "", " ", "B", "iB"):
        with pytest.raises(ValueError, match="max_memory"):
            SuperGLM(max_memory=bad)
    # A malformed environment default warns and falls back; it never fails a fit.
    with monkeypatch.context() as patched:
        patched.setenv("SUPERGLM_MAX_MEMORY", "inf")
        with pytest.warns(UserWarning, match="SUPERGLM_MAX_MEMORY"):
            assert parallel.resolve_max_memory() == parallel.default_max_memory()
    X, y, _rng = _frame(4_000, seed=9)

    def model(**threads):
        return SuperGLM(
            family="poisson",
            discrete=True,
            n_bins=32,
            features={**{c: Spline(kind="ps", k=6) for c in "abc"}, "g": Categorical()},
            interactions=[("a", "b"), ("b", "c")],
            **threads,
        )

    seen: list[tuple[int, int]] = []
    original = execution.run_block_tasks

    def recording(tasks, cache, profile):
        seen.append((parallel.resolve_n_jobs(), parallel.resolve_max_memory()))
        return original(tasks, cache, profile)

    monkeypatch.setattr(execution, "run_block_tasks", recording)
    fits = {}
    with warnings.catch_warnings(), block_queue_config(min_cost=0):
        warnings.simplefilter("ignore")
        for n_jobs in (1, 4):
            fits[n_jobs] = model(n_jobs=n_jobs, max_memory="2G").fit_reml(X, y, max_reml_iter=3)
            assert set(seen) == {(n_jobs, 2 << 30)}
            seen.clear()
    _assert_bitwise(fits[4].result.beta, fits[1].result.beta)
    assert fits[4].result.deviance == fits[1].result.deviance
    clone = fits[4].clone_unfitted()
    assert (clone._n_jobs, clone._max_memory) == (4, "2G")
    kwargs = fits[4]._config.constructor_kwargs()
    assert (kwargs["n_jobs"], kwargs["max_memory"]) == (4, "2G")
    assert (model()._n_jobs, model()._max_memory) == ("auto", "auto")


def test_estimator_limits_reach_refits_and_post_fit_inference(monkeypatch):
    """Every Gram build and data-rank factor a fitted estimator starts runs under its own limits.

    ``n_jobs`` and ``max_memory`` used to apply inside ``fit``, ``fit_path``
    and ``fit_reml`` only: the refits of ``drop1``, ``refit_unpenalised`` and
    ``estimate_p``, and the post-fit inference of ``summary``, ``metrics``,
    ``term_inference`` and ``relativities``, ran under the process default
    (on a 16-core machine, 16 workers and a quarter of RAM against an
    estimator's 1 worker and 64 MiB).  The estimator's ``(3, 64 MiB)``
    differs from the process default in both fields.
    """
    from superglm import families
    from superglm.solvers import rank

    X, y, rng = _frame(3_000, seed=12)
    seen: dict[str, set[tuple[int, int]]] = {}
    label = ["fit_reml"]
    original_tasks = execution.run_block_tasks
    original_factor = rank._tsqr_weighted_factor

    def record():
        limits = (parallel.resolve_n_jobs(), parallel.resolve_max_memory())
        seen.setdefault(label[0], set()).add(limits)

    def recording_tasks(tasks, cache, profile):
        record()
        return original_tasks(tasks, cache, profile)

    def recording_factor(*args, **kwargs):
        record()
        return original_factor(*args, **kwargs)

    monkeypatch.setattr(execution, "run_block_tasks", recording_tasks)
    monkeypatch.setattr(rank, "_tsqr_weighted_factor", recording_factor)

    def model(family="poisson"):
        return SuperGLM(
            family=family,
            discrete=True,
            n_bins=32,
            features={**{c: Spline(kind="ps", k=6) for c in "abc"}, "g": Categorical()},
            interactions=[("a", "b"), ("b", "c")],
            n_jobs=3,
            max_memory="64M",
        )

    fitted = model()
    severity = np.where(rng.random(len(y)) < 0.6, 0.0, rng.gamma(2.0, 0.5, len(y)))
    calls = {
        "fit_reml": lambda: fitted.fit_reml(X, y, max_reml_iter=3),
        "summary": fitted.summary,
        "metrics": lambda: fitted.metrics(X, y),
        "term_inference": lambda: fitted.term_inference("a"),
        "relativities": lambda: fitted.relativities(with_se=True),
        "drop1": lambda: fitted.drop1(X, y),
        "refit_unpenalised": lambda: fitted.refit_unpenalised(X, y),
        "estimate_p": lambda: model(families.Tweedie(p=1.5)).estimate_p(
            X, severity, p_bounds=(1.3, 1.7), xatol=5e-2
        ),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, call in calls.items():
            label[0] = name
            call()
    expected = (3, 64 << 20)
    assert (parallel.resolve_n_jobs(), parallel.resolve_max_memory()) != expected
    assert seen == {name: {expected} for name in seen}
    assert {"fit_reml", "drop1", "refit_unpenalised", "estimate_p"} <= seen.keys()
    assert seen.keys() & {"summary", "metrics", "term_inference", "relativities"}


@pytest.mark.threads
def test_pooled_gram_under_default_thread_pools_holds_blas_at_one_thread(monkeypatch, tensor_plan):
    """With every pool at its default, the workers' BLAS runs on one thread and the bits hold."""
    from threadpoolctl import threadpool_info

    plan, W, z, _signed = tensor_plan
    expected, _ = _moments(plan, W, z, workers=1)
    blas_threads: set[int] = set()
    original = execution._cross_gram

    def cross_gram(*args, **kwargs):
        blas_threads.update(
            pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_cross_gram", cross_gram)
    actual, profile = _moments(plan, W, z, workers=4, min_cost=0)
    assert profile["block_pool_workers"] == 4
    _assert_bitwise(actual.gram, expected.gram)
    if not any(pool["user_api"] == "blas" for pool in threadpool_info()):
        # Accelerate (the macOS ARM64 wheels' BLAS) has no threadpoolctl pool.
        pytest.skip("threadpoolctl exposes no BLAS pools; thread counts cannot be verified")
    assert blas_threads == {1}
