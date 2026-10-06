"""The data-rank factor as a TSQR: fixed leaves, a fixed tree, any worker count, bounded memory."""

from __future__ import annotations

import math
import threading
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm._parallel as parallel
import superglm.solvers.rank as rank
from superglm import Categorical, Spline, SuperGLM
from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import SupportCompressedSSPGroupMatrix
from superglm._parallel import parallel_config, pool_workers
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.solvers.centered_system import (
    grouped_weighted_factor,
    grouped_weighted_factor_rhs,
)
from tests._exact_reference import exact_matmul

_U = 2.0**-53


def _gamma(k: int) -> float:
    """``gamma~_k`` of Higham Thm 19.4 with its unspecified constant taken as ``c = 1``,
    the strictest reading: any ``c >= 1`` bound contains this one."""
    return k * _U / (1.0 - k * _U)


def _near_rank_rows(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows with an exact alias, a 1e-9 near alias and a column at an offset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, 1] = 2.0 * X[:, 0]
    X[:, 3] = X[:, 2] + 1e-9 * rng.standard_normal(n)
    X[:, 4] += 1e3
    return X, rng.uniform(0.2, 3.0, n)


def _leaf_rows_for(monkeypatch, width: int, rows: int) -> None:
    """Make ``tsqr_leaf_rows(width) == rows`` (``rows >= 2 * width``)."""
    monkeypatch.setattr(rank, "_TSQR_LEAF_BYTES", rows * rank._TSQR_LEAF_COPIES * 8 * width)
    assert rank.tsqr_leaf_rows(width) == rows


def _record_calls(monkeypatch, name: str) -> list[str]:
    """Record the thread that runs each ``rank.<name>`` call."""
    calls: list[str] = []
    original = getattr(rank, name)

    def recorded(*args, **kwargs):
        calls.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(rank, name, recorded)
    return calls


def test_tsqr_factor_is_bitwise_identical_across_worker_counts(monkeypatch):
    """The leaves and the tree depend on (n, p) alone, so the worker count cannot move a bit.

    1,000 rows in leaves of 24 give 42 leaves and 41 merges at every worker
    count, the pool really runs at 2 and 8 workers, and the factor is
    Householder-accurate: ``R'R = (A + dA)'(A + dA)`` with ``||da_j|| <=
    gamma_k ||a_j||`` (Higham 2002, Thm 19.4), where a column passes through
    one leaf of ``L`` rows and ``ceil(log2 m)`` merges of ``2p`` rows, so ``k =
    (L + 2p ceil(log2 m)) p`` and every entry of ``R'R - A'A`` is within
    ``(2 gamma_k + gamma_k^2) ||a_i|| ||a_j||``.  Both Grams are formed
    exactly, so that is the factor's error alone.
    """
    n, p, leaf = 1_000, 12, 24
    X, weights = _near_rank_rows(n, p, seed=20261006)
    centre = weights @ X / np.sum(weights)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)
    _leaf_rows_for(monkeypatch, p, leaf)
    leaves = _record_calls(monkeypatch, "_tsqr_leaf")
    merges = _record_calls(monkeypatch, "_tsqr_merge")
    used: list[int] = []
    original_workers = parallel.pool_workers

    def recorded_workers(*args):
        used.append(original_workers(*args))
        return used[-1]

    monkeypatch.setattr(parallel, "pool_workers", recorded_workers)
    factors = {}
    for jobs in (1, 2, 8):
        leaves.clear()
        merges.clear()
        with parallel_config(n_jobs=jobs):
            factors[jobs] = grouped_weighted_factor(dm, weights, center=centre)
        n_leaves = math.ceil(n / leaf)
        assert len(leaves) == n_leaves == 42
        assert len(merges) == n_leaves - 1
        assert used[-1] == jobs
        if jobs > 1:
            assert all(name.startswith("superglm-tsqr") for name in leaves)
    assert np.array_equal(factors[1], factors[2])
    assert np.array_equal(factors[1], factors[8])

    rows = np.sqrt(weights)[:, None] * (X - centre)
    factor = factors[1]
    assert factor.shape == (p, p)
    reference = exact_matmul((rows.T, rows))
    computed = exact_matmul((factor.T, factor))
    norms = np.sqrt(np.diag(reference))
    gamma = _gamma((leaf + 2 * p * math.ceil(math.log2(n / leaf))) * p)
    bound = (2.0 * gamma + gamma * gamma) * np.outer(norms, norms)
    assert np.all(np.abs(computed - reference) <= bound)


@pytest.mark.parametrize("short", ["last_leaf", "every_leaf"])
def test_tsqr_short_leaves_and_the_response_keep_the_joint_gram(monkeypatch, short):
    """Leaves shorter than the width, and the appended response, at 1, 2 and 8 workers.

    ``last_leaf``: ``grouped_weighted_factor_rhs`` (the QR-route solves) over
    223 rows in leaves of 24 at width 12, ten leaves of which the last holds
    7 rows, fewer than the width, so its merge stacks a trapezoid.
    ``every_leaf``: ``streamed_weighted_factor`` over chunks of 5 rows, as
    ``metrics``' ``iter_dense_chunks`` yields them for a wide design, so
    every leaf and merge is a trapezoid.  A joint factor ``F`` of the
    weighted rows ``[A, b]`` (``b`` left out without a response) has ``F'F
    = [A, b]'[A, b]`` within ``(2 gamma_k + gamma_k^2) ||a_i|| ||a_j||``,
    ``k = (L + 2 q ceil(log2 m)) q`` for width ``q`` (Higham 2002, Thm 19.4,
    as in the bitwise test above), bitwise the same at every worker count.
    Mutation checks: slicing a leaf's response from the wrong rows, or
    dropping a lower factor shorter than the width in the merge, puts the
    Gram orders of magnitude outside the bound (both passed every focused
    test before this one).
    """
    n, p = 223, 12
    rng = np.random.default_rng(20261010)
    X = rng.standard_normal((n, p))
    weights = rng.uniform(0.2, 3.0, n)
    response = rng.standard_normal(n)
    leaves = _record_calls(monkeypatch, "_tsqr_leaf")
    if short == "last_leaf":
        leaf = 2 * p
        _leaf_rows_for(monkeypatch, p, leaf)
        dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)

        def joint():
            factor, transformed = grouped_weighted_factor_rhs(dm, weights, response)
            return np.column_stack((factor, transformed))

        rows = np.column_stack((np.sqrt(weights)[:, None] * X, np.sqrt(weights) * response))
    else:
        leaf = 5

        def joint():
            chunks = (
                (start, min(start + leaf, n), X[start : start + leaf])
                for start in range(0, n, leaf)
            )
            return rank.streamed_weighted_factor(chunks, weights)

        rows = np.sqrt(weights)[:, None] * X
    factors = {}
    for jobs in (1, 2, 8):
        leaves.clear()
        with parallel_config(n_jobs=jobs):
            factors[jobs] = joint()
        assert len(leaves) == math.ceil(n / leaf)
        if jobs > 1:
            assert all(name.startswith("superglm-tsqr") for name in leaves)
    assert np.array_equal(factors[1], factors[2])
    assert np.array_equal(factors[1], factors[8])
    width = rows.shape[1]
    assert factors[1].shape == (width, width)
    reference = exact_matmul((rows.T, rows))
    computed = exact_matmul((factors[1].T, factors[1]))
    norms = np.sqrt(np.diag(reference))
    gamma = _gamma((leaf + 2 * width * math.ceil(math.log2(n / leaf))) * width)
    bound = (2.0 * gamma + gamma * gamma) * np.outer(norms, norms)
    assert np.all(np.abs(computed - reference) <= bound)


def test_tsqr_rank_decision_matches_one_householder_qr(monkeypatch):
    """At 1, 2 and 8 workers the certificate keeps the same directions as one QR of all rows.

    The exact alias is dropped, and so is the 1e-9 near alias, whose singular
    value sits far below the ``sqrt(eps)`` factor cut: rank ``p - 2``.  The
    dropped subspace moves by Wedin's ``sin(theta) <= ||E|| / gap`` (Golub &
    Van Loan, 4th ed., Thm 8.6.5), in the equilibrated coordinates the cut is
    taken in: both factors are columnwise backward stable (Higham 2002, Thm
    19.4), ``||E D^-1||_2 <= sqrt(p) (gamma_tsqr + gamma_np)``, plus each
    SVD's ``gamma_{p^2}``, against the gap between the smallest kept and the
    largest dropped equilibrated singular values.
    """
    n, p = 600, 8
    X, weights = _near_rank_rows(n, p, seed=11)
    centre = weights @ X / np.sum(weights)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)
    leaf = 2 * p
    _leaf_rows_for(monkeypatch, p, leaf)
    rows = np.sqrt(weights)[:, None] * (X - centre)
    reference = rank.decompose_factor(np.linalg.qr(rows, mode="r"))
    assert reference.rank == p - 2
    scale = np.linalg.norm(rows, axis=0)
    singular = np.linalg.svd(rows / scale, compute_uv=False)
    gap = singular[reference.rank - 1] - singular[reference.rank]
    depth = math.ceil(math.log2(n / leaf))
    error = math.sqrt(p) * (
        _gamma((leaf + 2 * p * depth) * p) + _gamma(n * p) + 2.0 * _gamma(p * p)
    )

    def dropped(decomposition):
        basis, _ = np.linalg.qr(scale[:, None] * decomposition.parameter_null_basis)
        return basis

    reference_null = dropped(reference)
    for jobs in (1, 2, 8):
        with parallel_config(n_jobs=jobs):
            certified = rank.decompose_factor(grouped_weighted_factor(dm, weights, center=centre))
        assert certified.rank == reference.rank
        null = dropped(certified)
        sin_theta = np.linalg.norm(null - reference_null @ (reference_null.T @ null), 2)
        assert sin_theta <= error / gap


def test_tsqr_leaves_keep_the_callers_errstate(monkeypatch):
    """A leaf that overflows under the caller's ``errstate(over="ignore")`` stays quiet when pooled."""
    n, p = 400, 6
    X, weights = _near_rank_rows(n, p, seed=7)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)
    _leaf_rows_for(monkeypatch, p, 24)
    original = rank._tsqr_leaf
    threads: list[str] = []

    def leaf(*args, **kwargs):
        threads.append(threading.current_thread().name)
        np.multiply(np.array([1e308]), 10.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(rank, "_tsqr_leaf", leaf)
    factors = {}
    with warnings.catch_warnings(), np.errstate(over="ignore"):
        warnings.simplefilter("error")
        for jobs in (1, 4):
            with parallel_config(n_jobs=jobs, max_memory="1G"):
                factors[jobs] = grouped_weighted_factor(dm, weights)
    assert any(name.startswith("superglm-tsqr") for name in threads)
    assert np.array_equal(factors[1], factors[4])


def _mixed_design(n: int, seed: int) -> DesignMatrix:
    """One block of each kind a deferred leaf writes.

    One-hot codes with base-level rows (the sink code), all-level codes, a
    support table gathered by row index, and two blocks that materialise
    their own rows: dense columns at an offset and sparse columns.
    """
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n, 2))
    dense[:, 1] += 1e3
    groups = [
        CategoricalGroupMatrix(rng.integers(-1, 4, n), 4),
        RandomEffectGroupMatrix(rng.integers(0, 3, n), 3),
        SupportCompressedSSPGroupMatrix(
            rng.uniform(0.0, 1.0, (7, 4)), rng.standard_normal((4, 3)), rng.integers(0, 7, n)
        ),
        DenseGroupMatrix(dense),
        SparseGroupMatrix(sp.random(n, 2, density=0.3, random_state=seed, format="csr")),
    ]
    return DesignMatrix(groups, n=n, p=14)


@pytest.mark.parametrize("centring", ["none", "centre", "split"])
def test_deferred_leaves_factor_the_materialised_rows_bitwise(monkeypatch, centring):
    """Leaves written on the workers from the compact forms give the materialised rows' bits.

    ``iter_grouped_design_leaves`` defers each leaf to the worker that
    factors it, which writes ``sqrt(w) * ((x - c) - c_lo)`` from the codes,
    the support table or the block's own rows; the same leaves materialised
    by ``row_subset(...).toarray()`` take the rows branch of
    ``rank._tsqr_leaf``.  Every kind of block, with and without the split
    centre and the appended response, at 1, 2 and 8 workers: equal to the
    bit.  Mutation checks: subtracting ``c + c_lo`` at once, reading the
    sink code as a level, or weighting a materialising block's rows out of
    order each break the equality.
    """
    n, leaf = 300, 40
    dm = _mixed_design(n, seed=20261006)
    _leaf_rows_for(monkeypatch, dm.p, leaf)
    rng = np.random.default_rng(5)
    weights = rng.uniform(0.2, 3.0, n)
    response = rng.standard_normal(n)
    dense = dm.toarray()
    centre = None if centring == "none" else weights @ dense / np.sum(weights)
    centre_lo = None
    if centring == "split":
        centre_lo = 1e-13 * np.abs(centre) * rng.uniform(-1.0, 1.0, dm.p)
    kwargs = {"center": centre, "center_lo": centre_lo}

    def materialised():
        for start in range(0, n, leaf):
            rows = np.arange(start, min(start + leaf, n))
            yield start, rows[-1] + 1, np.asarray(dm.row_subset(rows).toarray(), dtype=float)

    with parallel_config(n_jobs=1):
        expected = rank.streamed_weighted_factor(materialised(), weights, **kwargs)
        expected_rhs = np.column_stack(
            rank.streamed_weighted_factor_rhs(materialised(), weights, response, **kwargs)
        )
    for jobs in (1, 2, 8):
        with parallel_config(n_jobs=jobs, max_memory="1G"):
            factor = grouped_weighted_factor(dm, weights, **kwargs)
            joint = np.column_stack(grouped_weighted_factor_rhs(dm, weights, response, **kwargs))
        assert np.array_equal(factor.view(np.uint64), expected.view(np.uint64)), jobs
        assert np.array_equal(joint.view(np.uint64), expected_rhs.view(np.uint64)), jobs


def test_tsqr_leaves_and_merges_run_on_the_pool(monkeypatch):
    """At four workers the calling thread only schedules the TSQR.

    Eight leaves make a complete tree (seven merges, no leftover to fold),
    so every leaf's rows are written, every leaf is factored and every merge
    runs on a pool worker, and no design rows are materialised on the
    calling thread.  The former producer materialised each leaf there
    (``DesignMatrix.row_subset(...).toarray()``) and merged there: against
    it the materialising and merging threads below are the main thread.
    """
    n, leaf = 320, 40
    dm = _mixed_design(n, seed=11)
    _leaf_rows_for(monkeypatch, dm.p, leaf)
    weights = np.random.default_rng(2).uniform(0.2, 3.0, n)
    leaves = _record_calls(monkeypatch, "_tsqr_leaf")
    merges = _record_calls(monkeypatch, "_tsqr_merge")
    materialising: list[str] = []
    for owner, name in (
        (DesignMatrix, "row_subset"),
        (DesignMatrix, "toarray"),
        (DenseGroupMatrix, "toarray"),
        (SparseGroupMatrix, "toarray"),
    ):
        original = getattr(owner, name)

        def recorded(self, *args, _original=original):
            materialising.append(threading.current_thread().name)
            return _original(self, *args)

        monkeypatch.setattr(owner, name, recorded)
    with parallel_config(n_jobs=4, max_memory="1G"):
        grouped_weighted_factor(dm, weights)
    assert len(leaves) == 8 and len(merges) == 7
    on_pool = [name.startswith("superglm-tsqr") for name in leaves + merges + materialising]
    assert materialising and all(on_pool)


@pytest.mark.parametrize("width", [1, 5, 31, 32, 33, 70])
def test_structured_merge_is_scipys_dtpqrt_bitwise(width):
    """The merge reaches SciPy's ``dtpqrt`` through its Cython LAPACK pointer, without the GIL.

    The f2py wrapper calls the same routine with the same block size ``nb =
    min(width, 32)``, so the merged triangle is the same to the bit on both
    sides of the block size, and the merge leaves its operands as they were.
    """
    from scipy.linalg.lapack import dtpqrt

    rng = np.random.default_rng(width)
    upper = np.linalg.qr(rng.standard_normal((2 * width, width)), mode="r")
    lower = np.linalg.qr(rng.standard_normal((3 * width, width)), mode="r")
    kept = upper.copy(), lower.copy()
    assert rank._dtpqrt_nogil() is not None
    merged = rank._tsqr_merge(upper, lower)
    reference, _, _, info = dtpqrt(
        width, min(width, 32), np.asfortranarray(kept[0]), np.asfortranarray(kept[1])
    )
    assert info == 0
    assert np.array_equal(merged.view(np.uint64), np.triu(reference).view(np.uint64))
    assert np.array_equal(upper, kept[0]) and np.array_equal(lower, kept[1])


def test_worker_count_is_capped_by_memory_not_cores():
    """``min(n_jobs, tasks, max_memory // task_bytes)``, at least one."""
    with parallel_config(n_jobs=8, max_memory=3 * 1024):
        assert pool_workers(100, 1024) == 3
        assert pool_workers(2, 1024) == 2
        assert pool_workers(100, 10 * 1024) == 1
    with parallel_config(n_jobs=4, max_memory="1G"):
        assert pool_workers(100, 1024) == 4
        with parallel_config(n_jobs=1):
            assert pool_workers(100, 1024) == 1
            assert parallel.resolve_max_memory() == 1 << 30


def test_default_memory_budget_honours_the_cgroup_limit(monkeypatch, tmp_path):
    """A quarter of the tighter of physical memory and the process's cgroup limit.

    Inside a memory-limited container or systemd scope a quarter of the
    host's RAM does not bind: a 900 MiB scope on a 64 GiB host let the TSQR
    start 16 leaves of 80 MiB and was killed for memory where one worker
    completed.  cgroup v2 ``memory.max`` (``max`` is no limit) and v1
    ``memory.limit_in_bytes`` are read along the process's path up to the
    mount root, where a container sees its own limit; the tightest wins.
    """
    proc, root = tmp_path / "cgroup", tmp_path / "fs"
    scope = root / "user.slice" / "app.scope"
    scope.mkdir(parents=True)
    (root / "user.slice" / "memory.max").write_text("max\n")
    (scope / "memory.max").write_text("943718400\n")
    v1 = root / "memory" / "docker" / "c1"
    v1.mkdir(parents=True)
    (v1 / "memory.limit_in_bytes").write_text("2147483648\n")
    (root / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    monkeypatch.setattr(parallel, "_PROC_CGROUP", str(proc), raising=False)
    monkeypatch.setattr(parallel, "_CGROUP_ROOT", str(root), raising=False)
    monkeypatch.setattr(parallel, "_physical_memory", lambda: 64 << 30)
    cases = [
        ("0::/user.slice/app.scope\n", 943718400 // 4),
        ("12:memory:/docker/c1\n0::/\n", (2 << 30) // 4),
        ("0::/user.slice\n", (64 << 30) // 4),
        (None, (64 << 30) // 4),
    ]
    try:
        for content, expected in cases:
            if content is None:
                proc.unlink()
            else:
                proc.write_text(content)
            parallel.default_max_memory.cache_clear()
            assert parallel.default_max_memory() == expected, content
            with parallel_config(n_jobs=16):
                assert pool_workers(100, 80 << 20) == min(16, expected // (80 << 20))
    finally:
        parallel.default_max_memory.cache_clear()


def test_tsqr_holds_at_most_one_leaf_beyond_its_workers(monkeypatch):
    """Peak allocation is set by the workers, not by the leaf count.

    Each leaf in flight holds at most its materialised rows, the weighted
    copy and ``numpy.linalg.qr``'s working copy (three copies of ``L x p``
    doubles); the producer runs at most one leaf ahead of the workers and
    materialises one more (two copies while ``hstack`` joins the groups).
    64 leaves on 4 workers must therefore peak below ``(4 + 2) * 3`` leaf
    copies plus the tree's ``log2(64) + 1`` triangles and the per-leaf
    weights, under a third of what holding every leaf would take.  The
    workers are slowed (a sleep in each leaf, not a timing assertion) so a
    producer without backpressure would queue every leaf.  ``tracemalloc``
    sees NumPy's allocations but not LAPACK's ``malloc``-ed working buffer,
    so this checks the backpressure, not the whole three-copy working set.
    """
    n, p, leaf, workers = 64 * 512, 16, 512, 4
    X, weights = _near_rank_rows(n, p, seed=3)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=p)
    _leaf_rows_for(monkeypatch, p, leaf)
    original = rank._tsqr_leaf

    def slow_leaf(*args):
        time.sleep(0.01)
        return original(*args)

    monkeypatch.setattr(rank, "_tsqr_leaf", slow_leaf)
    copy = leaf * p * 8
    bound = (workers + 2) * 3 * copy + (int(math.log2(64)) + 2) * 3 * p * p * 8 + 64 * leaf * 8
    with parallel_config(n_jobs=workers):
        grouped_weighted_factor(dm, weights)  # warm the pool's imports
        tracemalloc.start()
        try:
            grouped_weighted_factor(dm, weights)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak <= bound
    assert bound < 64 * copy


def test_reml_fit_factors_each_set_of_rows_once(monkeypatch):
    """One ``fit_reml`` runs the TSQR at most once per set of weights, centre and response.

    Two aliased factors keep every PIRLS step and every observed geometry of
    this Tweedie fit on the factor certificate.  The REML loop asks again for
    factors it has formed -- the observed geometry of an accepted trial, a
    PIRLS start shared by successive smoothing states -- and the factor is a
    function of the design and the bits of those inputs alone, so the fit
    reuses it (``centered_system.reuse_data_factors``).  Without the reuse
    (the mutation below) the same fit repeats inputs, and its coefficients
    are the same to the bit.
    """
    import contextlib

    import superglm.solvers.centered_system as centered_system
    from superglm import Tweedie

    rng = np.random.default_rng(3)
    n = 3_000
    x = rng.uniform(-1.0, 1.0, n)
    level = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "a": pd.Categorical([f"a{v}" for v in level]),
            "b": pd.Categorical([f"b{v}" for v in level]),
        }
    )
    mu = np.exp(0.3 * np.sin(2.0 * x) + 0.2 * level)
    y = rng.gamma(2.0, mu / 2.0) * (rng.uniform(size=n) < 0.3)
    passes: list[bytes] = []

    def inputs(*arrays) -> bytes:
        return b"|".join(
            b"-" if a is None else np.asarray(a, dtype=np.float64).tobytes() for a in arrays
        )

    plain, joint = rank.streamed_weighted_factor, rank.streamed_weighted_factor_rhs

    def recorded(chunks, weights, *, center=None, center_lo=None):
        passes.append(inputs(weights, center, center_lo))
        return plain(chunks, weights, center=center, center_lo=center_lo)

    def recorded_rhs(chunks, weights, response, *, center=None, center_lo=None):
        passes.append(inputs(weights, center, center_lo, response))
        return joint(chunks, weights, response, center=center, center_lo=center_lo)

    monkeypatch.setattr(rank, "streamed_weighted_factor", recorded)
    monkeypatch.setattr(rank, "streamed_weighted_factor_rhs", recorded_rhs)

    def fit() -> np.ndarray:
        passes.clear()
        model = SuperGLM(
            family=Tweedie(p=1.5),
            link="log",
            selection_penalty=0.0,
            features={"x": Spline(kind="ps", k=8), "a": Categorical(), "b": Categorical()},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        return np.asarray(model.result.beta)

    reused = fit()
    reused_passes = list(passes)
    assert len(set(reused_passes)) == len(reused_passes)
    monkeypatch.setattr(centered_system, "reuse_data_factors", contextlib.nullcontext)
    repeated = fit()
    assert len(passes) > len(set(passes)) == len(reused_passes)
    assert np.array_equal(reused.view(np.uint64), repeated.view(np.uint64))


def test_data_factor_reuse_keeps_the_four_most_recently_used(monkeypatch):
    """Weights A B C A D E A form five factors: a hit counts as a use.

    A PIRLS start shared by successive smoothing states comes back every
    third factor, with two other factors between its uses.  Mutation checks:
    keeping only the latest entry, or the latest formed rather than used,
    forms the last A again.
    """
    from superglm.solvers.centered_system import reuse_data_factors

    dm = _mixed_design(200, seed=13)
    rng = np.random.default_rng(4)
    weights = [rng.uniform(0.2, 3.0, dm.n) for _ in range(5)]
    leaves = _record_calls(monkeypatch, "_tsqr_leaf")
    with parallel_config(n_jobs=1), reuse_data_factors():
        factors = [grouped_weighted_factor(dm, weights[k]) for k in (0, 1, 2, 0, 3, 4, 0)]
    assert len(leaves) == 5
    for again in (factors[3], factors[6]):
        assert np.array_equal(again.view(np.uint64), factors[0].view(np.uint64))


def test_discrete_reml_builds_the_data_rank_factor_once(monkeypatch):
    """Only the published state certifies its data rank with observation rows.

    Two aliased factors leave the unpenalised data Gram unable to certify its
    own rank, so the fit needs the O(n p^2) data-rank factor.  The REML
    optimizer's terminal refit used to build it as well, for a ``rank_info``
    that ``finalize_reml_fit`` replaced unread; now only the terminal refit
    that is published builds it, and its decomposition is the factor's.
    """
    import superglm.solvers.irls_direct as irls_direct

    rng = np.random.default_rng(3)
    n = 4_000
    x = rng.uniform(-1.0, 1.0, n)
    level = rng.integers(0, 3, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "a": pd.Categorical([f"a{v}" for v in level]),
            "b": pd.Categorical([f"b{v}" for v in level]),
        }
    )
    y = rng.poisson(np.exp(0.3 * np.sin(2.0 * x) + 0.2 * level)).astype(float)
    calls: list[int] = []
    original = irls_direct.grouped_weighted_factor

    def counted(dm, W, **kwargs):
        calls.append(len(W))
        return original(dm, W, **kwargs)

    monkeypatch.setattr(irls_direct, "grouped_weighted_factor", counted)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={"x": Spline(kind="ps", k=8), "a": Categorical(), "b": Categorical()},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert calls == [n]
    data = model.result.rank_info.data
    assert data.method == "qr_svd"
    # Two aliased three-level factors: their second factor's two columns are
    # the first's, so the data rank is the width less two.
    assert data.rank == model.result.beta.size - 2
