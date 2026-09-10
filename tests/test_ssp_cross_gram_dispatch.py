"""Raw SSP cross products preserve live factor targets without column loops."""

import pickle
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.group_matrix import SparseSSPGroupMatrix


def _pair(kind="mixed"):
    rows = np.arange(12)[:, None]
    left = (1 + (rows + np.arange(3)) % 7) / 8
    left[(rows + np.arange(3)) % 3 == 0] = 0
    right = (1 + (2 * rows + np.arange(4)) % 9) / 8
    if kind == "sparse":
        right[(rows + np.arange(4)) % 3 != 0] = 0
    r_left = np.array([[1.0, 0.25], [-0.5, 1.0], [0.25, -0.5]])
    r_right = np.array([[1.0, 0.25, 0.5], [-0.5, 1.0, 0.25], [0.25, -0.5, 1.0], [0.5, 0.25, -0.5]])
    return (
        SparseSSPGroupMatrix(sp.csr_matrix(left), r_left),
        SparseSSPGroupMatrix(sp.csr_matrix(right), r_right),
    )


def _forbidden(*args, **kwargs):
    pytest.fail("eligible SSP cross must not generate columns or expand an observation design")


def _assert_target(actual, left, right, weights):
    def fractions(value):
        return np.vectorize(lambda item: Fraction(float(item)), otypes=[object])(value)

    bi, bj = map(fractions, (left.B.toarray(), right.B.toarray()))
    ri, rj, w = map(fractions, (left.R_inv, right.R_inv, weights))
    target = ri.T @ (bi.T @ (w[:, None] * bj)) @ rj
    scale = abs(ri).T @ (abs(bi).T @ (abs(w)[:, None] * abs(bj))) @ abs(rj)
    count = len(weights) + 1 + bi.shape[1] + bj.shape[1]
    unit = Fraction(1, 2**53)
    gamma = count * unit / (1 - count * unit)
    assert actual.shape == target.shape
    assert np.all(np.isfinite(actual))
    for index in np.ndindex(actual.shape):
        assert abs(Fraction(float(actual[index])) - target[index]) <= gamma * scale[index]


@pytest.mark.parametrize("kind", ["mixed", "sparse"])
@pytest.mark.parametrize("reverse", [False, True])
def test_raw_cross_dispatch_bypasses_columns_and_observation_expansion(kind, reverse, monkeypatch):
    left, right = _pair(kind)
    if reverse:
        left, right = right, left
    monkeypatch.setattr(algebra, "_cross_gram_by_columns", _forbidden)
    monkeypatch.setattr(SparseSSPGroupMatrix, "toarray", _forbidden)
    original_csr = sp.csr_matrix.toarray
    original_csc = sp.csc_matrix.toarray
    allocations = []

    def raw_only(matrix, *args, **kwargs):
        allocations.append(matrix.shape)
        assert matrix.shape[0] != len(weights)
        original = original_csr if type(matrix) is sp.csr_matrix else original_csc
        return original(matrix, *args, **kwargs)

    weights = np.linspace(0.25, 1.0, 12)
    with monkeypatch.context() as patch:
        patch.setattr(sp.csr_matrix, "toarray", raw_only)
        patch.setattr(sp.csc_matrix, "toarray", raw_only)
        profile = {}
        actual = algebra._cross_gram(left, right, weights, profile=profile)
    assert profile.get("block_cross_ssp_ssp_calls") == 1
    assert profile.get("block_cross_fallback_s", 0) == 0
    assert bool(allocations) == (kind == "sparse")
    _assert_target(actual, left, right, weights)


@pytest.mark.parametrize("kind", ["mixed", "sparse"])
@pytest.mark.parametrize("signed", [False, True])
def test_cross_matches_exact_live_factor_target_with_signed_and_zero_weights(kind, signed):
    left, right = _pair(kind)
    weights = np.array([0.0, 0.3, 0.7, 0.5, 0.2, 0.0, 1.5, 0.9, 0.25, 1.2, 0.5, 0.1])
    if signed:
        weights[1::2] *= -1
    _assert_target(algebra._cross_gram(left, right, weights), left, right, weights)
    _assert_target(algebra._cross_gram(right, left, weights), right, left, weights)


def test_cross_reads_live_B_and_R_instead_of_owned_gram_snapshots():
    left, right = _pair()
    weights = np.linspace(-0.5, 1.0, 12)
    algebra._cross_gram(left, right, weights)
    left.B.data *= 0.5
    right.B.data *= 2
    left._data[:] = 0
    right._data[:] = 0
    left.R_inv[0, 1] = 0.75
    right.R_inv[1, 0] = -0.25
    _assert_target(algebra._cross_gram(left, right, weights), left, right, weights)
    right.B = sp.csr_matrix(right.B.toarray()[:, ::-1])
    right.R_inv = right.R_inv[::-1].copy()
    _assert_target(algebra._cross_gram(left, right, weights), left, right, weights)


@pytest.mark.parametrize("mutation", ["indices", "indptr", "duplicates"])
def test_noncanonical_live_storage_keeps_the_column_route(mutation, monkeypatch):
    left, right = _pair()
    assert right.B.has_canonical_format
    algebra._cross_gram(left, right, np.ones(12))
    if mutation == "indices":
        right.B.indices[:2] = [1, 0]
    elif mutation == "indptr":
        right.B.indptr[1] += 1
    else:
        right.B.indices[1] = 0
    # The original object's cached property does not validate current storage.
    assert right.B.has_canonical_format
    expected = algebra._cross_gram_by_columns(left, right, np.ones(12))
    original = algebra._cross_gram_by_columns
    calls = []

    def fallback(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(algebra, "_cross_gram_by_columns", fallback)
    np.testing.assert_array_equal(algebra._cross_gram(left, right, np.ones(12)), expected)
    assert calls == [True]


@pytest.mark.parametrize("exponent", [-600, 600])
@pytest.mark.parametrize("reverse", [False, True])
def test_outside_range_preserves_the_finite_column_product(exponent, reverse, monkeypatch):
    left = SparseSSPGroupMatrix(
        sp.csr_matrix([[np.ldexp(1.0, exponent)]]), np.array([[np.ldexp(1.0, -exponent)]])
    )
    right = SparseSSPGroupMatrix(sp.csr_matrix([[1.0]]), np.ones((1, 1)))
    if reverse:
        left, right = right, left
    monkeypatch.setattr(algebra, "_weighted_row_chunk", _forbidden)
    np.testing.assert_array_equal(algebra._cross_gram(left, right, np.ones(1)), [[1.0]])


@pytest.mark.parametrize(
    "decline",
    [
        "weights32",
        "transform_wide",
        "basis32",
        "weight_shape",
        "group_subclass",
        "csr_subclass",
        "array_subclass",
        "budget",
    ],
)
def test_ineligible_pairs_decline_before_new_weighting(decline, monkeypatch):
    left, right = _pair()
    weights = np.ones(12)
    if decline == "weights32":
        weights = weights.astype(np.float32)
    elif decline == "transform_wide":
        right.R_inv = right.R_inv.astype(np.longdouble)
    elif decline == "basis32":
        right.B = right.B.astype(np.float32)
    elif decline == "weight_shape":
        weights = np.ones(1)
    elif decline == "group_subclass":

        class CustomGroup(SparseSSPGroupMatrix):
            pass

        right = CustomGroup(right.B, right.R_inv)
    elif decline == "csr_subclass":

        class CustomCSR(sp.csr_matrix):
            pass

        right.B = CustomCSR(right.B)
    elif decline == "array_subclass":

        class CustomArray(np.ndarray):
            def __getitem__(self, item):
                _forbidden()

        right.R_inv = right.R_inv.view(CustomArray)
    else:
        monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 1)
    monkeypatch.setattr(algebra, "_weighted_row_chunk", _forbidden)
    sentinel = np.full((left.shape[1], right.shape[1]), 17.0)
    calls = []

    def fallback(*args):
        calls.append(True)
        return sentinel

    monkeypatch.setattr(algebra, "_cross_gram_by_columns", fallback)
    assert algebra._cross_gram(left, right, weights) is sentinel
    assert calls == [True]


def test_pickle_and_row_subsets_preserve_live_raw_cross_dispatch(monkeypatch):
    left, right = pickle.loads(pickle.dumps(_pair()))
    rows = np.array([5, 2, 5, 11])
    left, right = left.row_subset(rows), right.row_subset(rows)
    weights = np.array([0.5, -0.25, 1.0, 0.0])
    monkeypatch.setattr(algebra, "_cross_gram_by_columns", _forbidden)
    _assert_target(algebra._cross_gram(left, right, weights), left, right, weights)
