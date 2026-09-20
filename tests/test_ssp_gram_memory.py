"""Saturated SSP Grams retain BLAS without observation-sized scratch."""

import tracemalloc
import weakref
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix import _group_matrix_core as core
from superglm._group_matrix._group_matrix_algebra import _weighted_row_chunk
from superglm._group_matrix._group_matrix_kernels import _csr_row_chunk
from superglm.group_matrix import SparseSSPGroupMatrix


@pytest.mark.parametrize("rows", [100, 9_999, 10_001])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_weighted_csr_subhalf_chunks_share_indices(rows, index_dtype):
    group = _group(20_000, 8, full=True)
    group.B.indices = group.B.indices.astype(index_dtype)
    group.B.indptr = group.B.indptr.astype(index_dtype)
    weights = np.linspace(0.5, 1.5, group.shape[0])
    chunk = _weighted_row_chunk(group.B, weights, 0, rows)
    assert np.shares_memory(chunk.indices, group.B.indices)
    assert chunk.indices.dtype == index_dtype
    assert chunk.indptr.dtype == index_dtype
    np.testing.assert_array_equal(chunk.toarray(), group.B[:rows].toarray() * weights[:rows, None])
    unweighted = _csr_row_chunk(group.B, 1, rows + 1)
    assert np.shares_memory(unweighted.data, group.B.data)
    assert np.shares_memory(unweighted.indices, group.B.indices)
    np.testing.assert_array_equal(unweighted.toarray(), group.B[1 : rows + 1].toarray())


@pytest.mark.parametrize("budget", [256 << 10, 1 << 20])
def test_projected_sparse_gram_accounts_for_subhalf_slice_scratch(monkeypatch, budget):
    n = 20_000
    basis = np.zeros((n, 8))
    basis[:, 0], basis[:, 1] = 1, 1 + 1e-8 * np.linspace(-1, 1, n)
    transform = np.zeros((8, 8))
    transform[:2, 0] = [1, -1]
    transform[0, 1:] = 1
    group = SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)
    weights = np.linspace(0.5, 1.5, n)
    group.gram(weights)
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", budget)
    tracemalloc.start()
    try:
        group.gram(weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Observation scratch is bounded by the cap. Allow fixed coefficient
    # products and Python/SciPy allocation bookkeeping, not a fraction of n.
    assert peak <= budget + (32 << 10) + 64 * transform.shape[1] ** 2


@pytest.mark.parametrize("budget", [256 << 10, 1 << 20])
def test_sensitive_int64_sparse_cross_bounds_actual_peak(monkeypatch, budget):
    n, p = 20_000, 8
    basis = np.zeros((n, p))
    basis[:, 0], basis[:, 1] = 1, 1 + 1e-8 * np.linspace(-1, 1, n)
    transform = np.zeros((p, p))
    transform[:2, 0], transform[0, 1:] = [1, -1], 1
    groups = [SparseSSPGroupMatrix(sp.csr_matrix(basis), transform.copy()) for _ in range(2)]
    for group in groups:
        group.B.indices = group.B.indices.astype(np.int64)
        group.B.indptr = group.B.indptr.astype(np.int64)
    weights = np.linspace(0.5, 1.5, n)
    algebra._cross_gram(*groups, weights)
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", budget)
    original = algebra._csr_row_chunk
    rows = []
    shared = []

    def record(csr, start, stop):
        shared.append(
            csr.indices.dtype == csr.indptr.dtype == np.dtype(np.int64)
            and any(np.shares_memory(csr.indices, group.B.indices) for group in groups)
            and any(np.shares_memory(csr.indptr, group.B.indptr) for group in groups)
        )
        rows.append(stop - start)
        return original(csr, start, stop)

    monkeypatch.setattr(algebra, "_csr_row_chunk", record)
    tracemalloc.start()
    try:
        result = algebra._cross_gram_sparse_ssp(*groups, weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result is not None and rows and max(rows) < n
    assert peak <= budget + (32 << 10) + 64 * p**2
    assert all(shared)


def _group(rows, width, *, full):
    basis = (1 + np.arange(rows * width).reshape(rows, width) % 11) / 16.0
    if not full:
        basis[::3, 0] = 0.0
    return SparseSSPGroupMatrix(sp.csr_matrix(basis), np.eye(width))


@pytest.mark.parametrize("full", [True, False], ids=["dense_view", "dense_copy"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_large_saturated_gram_bounds_scratch_and_keeps_blas(monkeypatch, full, index_dtype):
    budget = 64 << 10
    group = _group(8192, 32, full=full)
    group.B.indices = group.B.indices.astype(index_dtype)
    group.B.indptr = group.B.indptr.astype(index_dtype)
    weights = np.linspace(0.25, 1.0, group.shape[0])
    # Compile the range classifier before measuring Python/NumPy allocations.
    group.gram(weights)
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", budget, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("saturated large Grams must retain bounded BLAS execution")

    monkeypatch.setattr(core, "_csr_weighted_gram", forbidden)
    monkeypatch.setattr(core, "_exact_ssp_moments", forbidden)
    original = sp.csr_matrix.toarray
    rendered = []
    references = []

    def record_render(matrix, *args, **kwargs):
        assert all(reference() is None for reference in references)
        rendered.append(matrix.shape)
        result = original(matrix, *args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(sp.csr_matrix, "toarray", record_render)
    tracemalloc.start()
    try:
        result = group.gram(weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Include row scratch, coefficient products/indices and Python/SciPy overhead.
    assert peak <= 6 * budget + 16 * group._p_b**2 * 8
    assert result.shape == (32, 32)
    assert all(reference() is None for reference in references)
    if full:
        assert rendered == []
    else:
        assert len(rendered) > 1
        assert all(rows < group.shape[0] and rows * width * 8 <= budget for rows, width in rendered)


@pytest.mark.parametrize("full", [True, False], ids=["dense_view", "dense_copy"])
def test_numerical_fixture_dispatches_to_blocked_dense_gram(monkeypatch, full):
    group = _group(31, 5, full=full)
    weights = np.linspace(-0.5, 1.0, 31)
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", 256)

    def forbidden(*args, **kwargs):
        pytest.fail("the numerical fixture must exercise saturated dense products")

    monkeypatch.setattr(core, "_csr_weighted_gram", forbidden)
    monkeypatch.setattr(core, "_exact_ssp_moments", forbidden)
    original = sp.csr_matrix.toarray
    rendered = []

    def record_render(matrix, *args, **kwargs):
        rendered.append(matrix.shape)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", record_render)
    group.gram(weights)
    if full:
        assert rendered == []
    else:
        assert len(rendered) > 1
        assert sum(rows for rows, _ in rendered) == group.shape[0]
        assert all(rows < group.shape[0] and width == group._p_b for rows, width in rendered)


def _assert_exact_factor_target(group, weights, actual):
    def fractions(values):
        return np.vectorize(lambda value: Fraction(float(value)), otypes=[object])(values)

    basis, transform, mass = map(fractions, (group.B.toarray(), group.R_inv, weights))
    target = transform.T @ (basis.T @ (mass[:, None] * basis)) @ transform
    scale = abs(transform).T @ (abs(basis).T @ (abs(mass)[:, None] * abs(basis))) @ abs(transform)
    # Weighted dots, block accumulation, and both transform products. Counting
    # at most N blocks gives at most 2*N + 2*p + 4 rounded operations per term.
    count = 2 * len(weights) + 2 * basis.shape[1] + 4
    unit = Fraction(1, 2**53)
    gamma = count * unit / (1 - count * unit)
    for index in np.ndindex(actual.shape):
        assert abs(Fraction(float(actual[index])) - target[index]) <= gamma * scale[index]


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("signed", [True, False])
def test_blocked_gram_matches_exact_factor_target(monkeypatch, full, signed):
    group = _group(31, 5, full=full)
    group.R_inv = np.array([[1, 0, -1], [0, 1, 0.5], [0.5, -0.5, 1], [1, 0.5, 0], [0, 1, 1]])
    weights = (1 + np.arange(31) % 7) / 8.0
    weights[::7] = 0.0
    if signed:
        weights[1::2] *= -1
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", 256, raising=False)
    _assert_exact_factor_target(group, weights, group.gram(weights))


@pytest.mark.parametrize("full", [True, False])
def test_small_saturated_gram_keeps_existing_single_product(full):
    group = _group(19, 8, full=full)
    weights = np.linspace(-0.5, 1.0, 19)
    basis = group.B.toarray()
    expected = (basis * weights[:, None]).T @ basis
    lower = np.tril_indices(group._p_b, -1)
    expected[lower] = expected.T[lower]
    np.testing.assert_array_equal(group.gram(weights), expected)


@pytest.mark.parametrize("full", [True, False])
def test_blocked_gram_rechecks_live_values_and_cached_canonical_flags(monkeypatch, full):
    group = _group(31, 5, full=full)
    weights = np.linspace(-0.5, 1.0, 31)
    monkeypatch.setattr(core, "_MAX_SSP_GRAM_WORKSPACE_BYTES", 256)
    _assert_exact_factor_target(group, weights, group.gram(weights))
    group.B.data *= 0.5
    group.R_inv[0, 1] = 0.25
    _assert_exact_factor_target(group, weights, group.gram(weights))

    assert group.B.has_canonical_format
    # In-place permutation leaves SciPy's cached canonical flag stale. The
    # fresh Gram admission must inspect current buffers before using BLAS.
    group.B.indices[:2] = group.B.indices[1::-1]
    assert group.B.has_canonical_format
    calls = []
    original = core._csr_weighted_gram

    def record_fallback(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(core, "_csr_weighted_gram", record_fallback)
    actual = group.gram(weights)
    assert calls == [True]
    _assert_exact_factor_target(group, weights, actual)
