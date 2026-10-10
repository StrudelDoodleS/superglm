"""Saturated SSP blocks retain their raw weighted-Gram target under BLAS."""

import pickle
import weakref
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_core as core
from superglm._group_matrix._group_matrix_algebra import _gram_any_sign
from superglm.solvers.rank import decompose_symmetric
from tests._exact_reference import exact_weighted_gram


def _forbidden(*args, **kwargs):
    pytest.fail("unexpected scalar CSR kernel or dense materialization")


def _fraction_matrix(values):
    return np.vectorize(lambda value: Fraction(float(value)), otypes=[object])(values)


@pytest.mark.parametrize("signed", [False, True])
def test_sparse_raw_gram_cancellation_recomputes_projected_rows(signed):
    n = 20_000
    basis = np.zeros((n, 8))
    basis[:, 0] = 1
    basis[:, 1] = 1 + 1e-8 * np.linspace(-1, 1, n)
    transform = np.zeros((8, 2))
    transform[0] = [1, 1]
    transform[1] = [-1, 0]
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)
    weights = np.linspace(0.5, 1.5, n)
    if signed:
        weights[::2] *= -1
    projected = group.toarray()
    target = exact_weighted_gram(projected, projected, weights)
    # Signed moments have an absolute error scale from |W|, not a relative
    # accuracy target at a possibly zero signed diagonal.
    energy = exact_weighted_gram(projected, projected, abs(weights))
    scales = np.sqrt(np.diag(energy))
    expected = target / np.outer(scales, scales)
    gram, projected_rows = group._gram_with_projection(weights)
    assert projected_rows
    actual = gram / np.outer(scales, scales)
    assert np.linalg.norm(actual - expected, 2) <= 100 * np.finfo(float).eps * np.linalg.norm(
        energy / np.outer(scales, scales), 2
    )


def test_signed_diagonal_cancellation_does_not_require_projected_rows(monkeypatch):
    basis = np.zeros((4000, 8))
    basis[::2, 0], basis[1::2, 1] = 1, 1
    transform = np.zeros((8, 2))
    transform[:2] = [[1, 1], [1, -1]]
    weights = np.tile([1.0, -1.0], 2000)
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)
    monkeypatch.setattr(core, "_solver_space_gram", _forbidden)
    gram, projected = group._gram_with_projection(weights)
    assert not projected
    np.testing.assert_array_equal(gram, [[0, 4000], [4000, 0]])


def _assert_raw_target(actual, basis, transform, weights):
    """Enclose rounding against the exact represented B, W and R product.

    Weighting and the n-term raw dot contribute gamma_(n+1). Each of the
    two existing p-term transformation dots contributes gamma_p. These
    fixtures keep every nonzero arithmetic scale in the normal range.
    """
    b, r, w = map(_fraction_matrix, (basis, transform, weights))
    expected = r.T @ (b.T @ (w[:, None] * b)) @ r
    scale = abs(r).T @ (abs(b).T @ (abs(w)[:, None] * abs(b))) @ abs(r)
    count = basis.shape[0] + 1 + 2 * basis.shape[1]
    unit = Fraction(1, 2**53)
    gamma = count * unit / (1 - count * unit)
    assert np.all(np.isfinite(actual))
    for index in np.ndindex(actual.shape):
        assert abs(Fraction(float(actual[index])) - expected[index]) <= gamma * scale[index]


@pytest.mark.parametrize("nonzeros, dense_calls", [(20, 1), (18, 1), (17, 0), (4, 0)])
def test_saturation_dispatch_uses_the_existing_threshold(nonzeros, dense_calls, monkeypatch):
    basis = np.arange(1.0, 21.0).reshape(5, 4)
    basis.ravel()[nonzeros:] = 0
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), np.eye(4))
    observed = {"dense": 0, "csr": 0}
    original_dense = core._dense_if_saturated
    original_csr = core._csr_weighted_gram

    def dense(block):
        observed["dense"] += 1
        return original_dense(block)

    def sparse(*args):
        observed["csr"] += 1
        if dense_calls:
            _forbidden()
        return original_csr(*args)

    monkeypatch.setattr(core, "_dense_if_saturated", dense)
    monkeypatch.setattr(core, "_csr_weighted_gram", sparse)
    actual = group.gram(np.ones(5))
    assert observed == {"dense": int(nonzeros == 18), "csr": 1 - dense_calls}
    _assert_raw_target(actual, basis, np.eye(4), np.ones(5))


def test_full_storage_reuses_owned_values_without_dense_materialization(monkeypatch):
    basis = np.arange(1.0, 21.0).reshape(5, 4)
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), np.eye(4))
    monkeypatch.setattr(core, "_dense_if_saturated", _forbidden)
    monkeypatch.setattr(core, "_csr_weighted_gram", _forbidden)
    monkeypatch.setattr(core.SparseSSPGroupMatrix, "toarray", _forbidden)
    _assert_raw_target(group.gram(np.ones(5)), basis, group.R_inv, np.ones(5))
    group._data *= 0.5
    _assert_raw_target(group.gram(np.ones(5)), 0.5 * basis, group.R_inv, np.ones(5))


@pytest.mark.parametrize("signed", [False, True])
def test_dense_gram_matches_the_exact_raw_target_with_signed_weights(signed):
    basis = np.array([[0.1, 0.3, -0.7], [1.3, -0.2, 0.1], [-0.4, 0.6, 1.2], [0.7, 0.8, -0.3]])
    transform = np.array([[1.0, -0.25], [0.3, 1.5], [-0.5, 0.75]])
    weights = np.array([0.3, -0.7 if signed else 0.7, 1.2, 0.9])
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)
    _assert_raw_target(group.gram(weights), basis, transform, weights)


def test_signed_cancellation_keeps_raw_symmetry_and_separated_inertia(monkeypatch):
    # The last pair cancels exactly. The first two terms have opposite signs
    # and independent directions, so the mathematical inertia is (1, 1, 0).
    basis = np.array([[1.0, 0.25], [0.25, 1.0], [0.5, 0.75], [0.5, 0.75]])
    weights = np.array([1.0, -0.5, 2**20, -(2**20)], dtype=float)
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), np.eye(2))
    monkeypatch.setattr(core, "_csr_weighted_gram", _forbidden)
    actual = _gram_any_sign(group, weights)
    np.testing.assert_array_equal(actual, actual.T)
    _assert_raw_target(actual, basis, np.eye(2), weights)
    decomposition = decompose_symmetric(actual)
    assert decomposition.rank == 2
    assert np.count_nonzero(decomposition.retained_values > 0) == 1
    assert np.count_nonzero(decomposition.retained_values < 0) == 1


@pytest.mark.parametrize("storage", ["duplicates", "unsorted"])
def test_noncanonical_storage_keeps_the_existing_csr_route(storage, monkeypatch):
    indices = np.array([0, 0, 1, 0, 1, 1]) if storage == "duplicates" else np.array([1, 0, 0, 1])
    indptr = np.array([0, 3, 6]) if storage == "duplicates" else np.array([0, 2, 4])
    basis = sp.csr_matrix((np.arange(1.0, len(indices) + 1), indices, indptr), shape=(2, 2))
    assert not basis.has_canonical_format
    group = core.SparseSSPGroupMatrix(basis, np.eye(2))
    weights = np.array([0.5, -0.25])
    expected = core._csr_weighted_gram(group._data, group._indices, group._indptr, weights, 2)
    monkeypatch.setattr(core, "_dense_if_saturated", _forbidden)
    np.testing.assert_array_equal(group.gram(weights), expected)


def test_non_float64_weights_keep_the_existing_csr_route(monkeypatch):
    group = core.SparseSSPGroupMatrix(sp.csr_matrix([[1.0, 0.25], [0.5, 1.0]]), np.eye(2))
    weights = np.ones(2, dtype=np.float32)
    expected = core._csr_weighted_gram(group._data, group._indices, group._indptr, weights, 2)
    monkeypatch.setattr(core, "_dense_if_saturated", _forbidden)
    np.testing.assert_array_equal(group.gram(weights), expected)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.longdouble, np.complex128])
def test_transform_dtype_selects_the_existing_raw_operation_route(dtype, monkeypatch):
    transform = np.array([[1.0, 0.25], [-0.5, 1.0]], dtype=dtype)
    group = core.SparseSSPGroupMatrix(sp.csr_matrix([[1.0, 0.25], [0.5, 1.0]]), transform)
    weights = np.ones(2)
    raw = core._csr_weighted_gram(group._data, group._indices, group._indptr, weights, 2)
    expected = transform.T @ raw @ transform
    original = core._csr_weighted_gram
    calls = []

    def record(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(core, "_csr_weighted_gram", record)
    monkeypatch.setattr(core, "_dense_if_saturated", _forbidden)
    np.testing.assert_array_equal(group.gram(weights), expected)
    # longdouble aliases float64 on Windows and macOS ARM64.
    assert calls == ([] if transform.dtype == np.dtype(np.float64) else [True])


@pytest.mark.parametrize("exponent", [600, -600])
def test_exact_range_guard_precedes_dense_dispatch(exponent, monkeypatch):
    group = core.SparseSSPGroupMatrix(
        sp.csr_matrix([[np.ldexp(1.0, exponent)]]), np.array([[np.ldexp(1.0, -exponent)]])
    )
    monkeypatch.setattr(core, "_dense_if_saturated", _forbidden)
    monkeypatch.setattr(core, "_csr_weighted_gram", _forbidden)
    np.testing.assert_array_equal(group.gram(np.ones(1)), [[1.0]])


def test_gram_reads_live_owned_arrays_without_a_retained_dense_copy(monkeypatch):
    basis = np.array([[1.0, 0.25], [0.5, 1.0], [0.75, 0.5], [0.25, 0.5], [1.0, 0.0]])
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), np.eye(2))
    identities = tuple(id(getattr(group, name)) for name in ("B", "_data", "_indices", "_indptr"))
    references = []
    original = core._dense_if_saturated

    def observe(block):
        dense = original(block)
        references.append(weakref.ref(dense))
        assert np.shares_memory(block.data, group._data)
        assert np.shares_memory(block.data, group.B.data)
        return dense

    monkeypatch.setattr(core, "_dense_if_saturated", observe)
    weights = np.array([0.5, 1.0, -0.25, 0.75, 0.5])
    _assert_raw_target(group.gram(weights), basis, group.R_inv, weights)
    assert references and all(reference() is None for reference in references)
    group.B.data *= 2
    _assert_raw_target(group.gram(weights), 2 * basis, group.R_inv, weights)
    np.testing.assert_array_equal(group.matvec(np.ones(2)), (2 * basis) @ np.ones(2))
    np.testing.assert_array_equal(group.rmatvec(weights), (2 * basis).T @ weights)
    group._data *= 0.5
    group.R_inv[0, 1] = 0.25
    _assert_raw_target(group.gram(weights), basis, group.R_inv, weights)
    assert identities == tuple(
        id(getattr(group, name)) for name in ("B", "_data", "_indices", "_indptr")
    )
    assert all(reference() is None for reference in references)


def test_live_index_mutation_does_not_reuse_a_stale_canonical_flag(monkeypatch):
    group = core.SparseSSPGroupMatrix(sp.csr_matrix([[1.0, 0.25], [0.5, 1.0]]), np.eye(2))
    weights = np.array([1.0, -0.25])
    group.gram(weights)
    assert group.B.has_canonical_format
    group._indices[:2] = [1, 0]
    # SciPy's stored property is stale; the current arrays are not canonical.
    assert group.B.has_canonical_format
    expected = core._csr_weighted_gram(group._data, group._indices, group._indptr, weights, 2)
    monkeypatch.setattr(core, "_dense_if_saturated", _forbidden)
    np.testing.assert_array_equal(group.gram(weights), expected)


def test_pickle_and_row_subset_preserve_raw_design_and_dispatch(monkeypatch):
    basis = np.array([[1.0, 0.25], [0.5, 1.0], [0.75, 0.5]])
    transform = np.array([[1.0, 0.25], [-0.5, 1.0]])
    group = core.SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)
    group.gram(np.ones(3))
    restored = pickle.loads(pickle.dumps(group))
    indices = np.array([2, 0, 2])
    subset = restored.row_subset(indices)
    monkeypatch.setattr(core, "_csr_weighted_gram", _forbidden)
    for current, raw in ((restored, basis), (subset, basis[indices])):
        weights = np.array([0.5, 1.0, -0.25])
        _assert_raw_target(current.gram(weights), raw, transform, weights)
        np.testing.assert_array_equal(current.toarray(), raw @ transform)


def _wide_sparse_group(saturated, transform, n=2000, seed=3):
    """A k >= 100 basis like the exact-path Density x DrivAge tensor (k = 144)."""
    gen = np.random.default_rng(seed)
    k = transform.shape[0]
    basis = gen.uniform(0.1, 1.0, size=(n, k))
    if not saturated:
        # Sixteen adjacent columns per row, as a tensor of two cubic B-splines.
        start = gen.integers(0, k - 16, n)
        basis[np.arange(k) < start[:, None]] = 0
        basis[np.arange(k) >= start[:, None] + 16] = 0
    return core.SparseSSPGroupMatrix(sp.csr_matrix(basis), transform)


@pytest.mark.parametrize("saturated", [True, False])
@pytest.mark.parametrize("scale", [1.0, 1 / 3])
@pytest.mark.parametrize("signed", [False, True])
def test_one_term_transform_columns_never_screen_as_cancelling(saturated, scale, signed):
    # Each column of a scaled identity has one term, so each projected
    # diagonal is G_cc * r**2 with two roundings (none at r = 1), whatever k
    # is. Charging all k = 144 rows put gamma_288 over the 144-eps budget with
    # no cancellation, so the full-book tensor's Gram took projected rows and
    # each binned cross a row pass per column: 1,620 s of a 1,756 s fit.
    group = _wide_sparse_group(saturated, scale * np.eye(144))
    weights = np.random.default_rng(4).uniform(0.5, 1.5, group.shape[0])
    if signed:
        weights[::3] *= -1
    gram, projected = group._gram_with_projection(weights)
    assert not projected
    raw = (
        core._saturated_ssp_gram(group.B, weights)
        if saturated
        else core._csr_weighted_gram(group._data, group._indices, group._indptr, weights, 144)
    )
    raw[np.tril_indices(144, -1)] = raw.T[np.tril_indices(144, -1)]
    if scale == 1.0:
        # An identity projection is exact: no product or sum rounds.
        np.testing.assert_array_equal(gram, raw)


def test_one_term_transform_keeps_the_binned_raw_aggregate(monkeypatch):
    from superglm._group_matrix import _group_matrix_algebra as algebra
    from superglm._group_matrix._group_matrix_kernels import _csr_weighted_bincount

    group = _wide_sparse_group(True, np.eye(144))
    gen = np.random.default_rng(5)
    weights = gen.uniform(0.5, 1.5, group.shape[0])
    bins = gen.integers(0, 37, group.shape[0])
    monkeypatch.setattr(algebra, "_aggregate_group_matrix_columns", _forbidden)
    aggregate = algebra._agg_by_bin(group, bins, weights, 37)
    expected = _csr_weighted_bincount(
        group._data, group._indices, group._indptr, 144, bins, weights, 37
    )
    np.testing.assert_array_equal(aggregate, expected)


def test_dense_transforms_keep_the_full_term_count():
    # Every column of a dense transform has k terms, so the screen's decision
    # is the one it made before nonzeros were counted.
    gen = np.random.default_rng(6)
    for k, p in ((12, 11), (150, 149), (150, 20)):
        basis = gen.normal(size=(400, k))
        raw = basis.T @ (gen.uniform(0.5, 1.5, 400)[:, None] * basis)
        transform = gen.normal(size=(k, p))
        gram = transform.T @ raw @ transform
        envelope = np.diag(np.abs(transform).T @ np.abs(raw) @ np.abs(transform))
        count_u = 2 * k * np.finfo(float).eps / 2
        budget = max(100, p) * np.finfo(float).eps * np.abs(np.diag(gram))
        before = bool(np.any(count_u / (1 - count_u) * envelope > budget))
        assert core._ssp_projection_cancels(raw, transform, gram) == before


def test_projection_error_is_bounded_by_the_transform_column_terms():
    # The screen's bound: |fl(R' G R) - R' G R|_cc <= gamma_(2 m_c) (|R|'|G||R|)_cc
    # for m_c nonzeros in column c, against exact rationals. Entries of 1/3 and
    # 0.1 round in every product, so the one-term columns cannot pass by
    # exactness.
    gen = np.random.default_rng(8)
    k, terms = 120, (1, 1, 2, 5, 30, 120)
    raw = gen.normal(size=(k, k))
    raw = raw + raw.T
    transform = np.zeros((k, len(terms)))
    for column, count in enumerate(terms):
        rows = gen.choice(k, size=count, replace=False)
        transform[rows, column] = gen.choice([1 / 3, -0.1, 0.7], size=count)
    computed = np.diag(transform.T @ raw @ transform)
    exact_raw, exact_transform = _fraction_matrix(raw), _fraction_matrix(transform)
    unit = Fraction(1, 2**53)
    for column, count in enumerate(terms):
        r = exact_transform[:, column]
        exact = r @ exact_raw @ r
        scale = abs(r) @ abs(exact_raw) @ abs(r)
        gamma = 2 * count * unit / (1 - 2 * count * unit)
        error = abs(Fraction(float(computed[column])) - exact)
        assert error <= gamma * scale
        if count == 1:
            assert error > 0
