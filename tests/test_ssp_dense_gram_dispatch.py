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


def _forbidden(*args, **kwargs):
    pytest.fail("unexpected scalar CSR kernel or dense materialization")


def _fraction_matrix(values):
    return np.vectorize(lambda value: Fraction(float(value)), otypes=[object])(values)


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


@pytest.mark.parametrize("dtype", [np.float32, np.longdouble, np.complex128])
def test_non_float64_transforms_keep_the_existing_raw_operation_route(dtype, monkeypatch):
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
    assert calls == [True]


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
