"""Raw SSP cross products preserve live factor targets without column loops."""

import pickle
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
)
from tests._exact_reference import exact_matmul, exact_weighted_gram


@pytest.mark.parametrize("reverse", [False, True])
def test_sensitive_sparse_dense_cross_matches_the_projected_diagonal(reverse):
    eps = np.finfo(float).eps
    basis = np.zeros((3, 8))
    basis[:, 0], basis[:, 1] = 1, [1 + eps, 1, 1]
    transform = np.zeros((8, 2))
    transform[0], transform[1] = [1, 1], [-1, 0]
    groups = [
        SparseSSPGroupMatrix(sp.csr_matrix(basis), transform),
        DenseGroupMatrix(np.ones((3, 1))),
    ]
    if reverse:
        groups.reverse()
    design = np.hstack([group.toarray() for group in groups])
    target = exact_matmul((design.T, design))
    scale = np.sqrt(np.diag(target))
    reference = target / np.outer(scale, scale)
    actual = MatrixExecutionPlan(groups, n=3).moments(np.ones(3)).gram
    error = (actual - target) / np.outer(scale, scale)
    assert np.linalg.norm(error, 2) <= 100 * eps * np.linalg.norm(reference, 2)


@pytest.mark.parametrize("right_kind", ["sparse", "support"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_cancellation_sensitive_sparse_cross_uses_projected_rows(right_kind, reverse, signed):
    n = 20_000
    x = np.linspace(-1, 1, n)
    raw = np.zeros((n, 8))
    raw[:, 0], raw[:, 1] = 1, 1 + 1e-8 * x
    transform = np.zeros((8, 1))
    transform[:2, 0] = [1, -1]
    left = SparseSSPGroupMatrix(sp.csr_matrix(raw), transform)
    right = (
        SparseSSPGroupMatrix(sp.csr_matrix(np.sign(x)[:, None]), np.ones((1, 1)))
        if right_kind == "sparse"
        else DiscretizedSSPGroupMatrix(
            np.array([[-1.0], [1.0]]), np.ones((1, 1)), (x > 0).astype(int)
        )
    )
    if reverse:
        left, right = right, left
    weights = np.linspace(0.5, 1.5, n)
    if signed:
        weights[::2] *= -1
    design = np.hstack([left.toarray(), right.toarray()])
    target = exact_weighted_gram(design, design, weights)
    energy = exact_weighted_gram(design, design, abs(weights))
    scale = np.sqrt(np.diag(energy))
    plan = MatrixExecutionPlan((left, right), n=n)
    for _ in range(2):
        actual = plan.moments(weights, signed=signed).gram
        error = (actual - target) / np.outer(scale, scale)
        bound = 100 * np.finfo(float).eps * np.linalg.norm(energy / np.outer(scale, scale), 2)
        assert np.linalg.norm(error, 2) <= bound
        assert (
            abs(algebra._cross_gram(left, right, weights)[0, 0] - target[0, 1])
            <= bound * scale.prod()
        )
        weights *= 0.5
        target *= 0.5
        energy *= 0.5
        scale /= np.sqrt(2)


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


def test_cross_reads_live_B_and_R_after_public_and_private_mutations():
    left, right = _pair()
    weights = np.linspace(-0.5, 1.0, 12)
    algebra._cross_gram(left, right, weights)
    left.B.data *= 0.5
    right.B.data *= 2
    left._data[:] *= 0.5
    right._data[:] *= 0.5
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
        "transform32",
        pytest.param(
            "transform_wide",
            marks=pytest.mark.skipif(
                np.dtype(np.longdouble) == np.dtype(np.float64),
                reason="longdouble has the float64 dtype on this platform",
            ),
        ),
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
    elif decline == "transform32":
        right.R_inv = right.R_inv.astype(np.float32)
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


def _categorical_pair():
    left, _ = _pair()
    left.R_inv = np.array([[1.0, 0.25], [0.5, 1.0], [0.25, 0.5]])
    right = CategoricalGroupMatrix(np.array([0, 1, -1, 2] * 3), n_levels=3)
    return left, right


def _assert_categorical_target(actual, spline, category, weights):
    # Exact source products, independent of both the grouped and column kernels.
    fractions = np.vectorize(lambda value: Fraction(float(value)), otypes=[object])
    basis, transform, weight = map(fractions, (spline.B.toarray(), spline.R_inv, weights))
    design = basis @ transform
    envelope = abs(basis) @ abs(transform)
    target = np.zeros(actual.shape, dtype=object)
    scale = np.zeros(actual.shape, dtype=object)
    for row, code in enumerate(category.codes):
        if code < category.n_levels:
            target[:, code] += weight[row] * design[row]
            scale[:, code] += abs(weight[row]) * envelope[row]
    count = 2 * (len(weights) + basis.shape[1] + 2)
    unit = Fraction(1, 2**53)
    gamma = count * unit / (1 - count * unit)
    for index in np.ndindex(actual.shape):
        assert abs(Fraction(float(actual[index])) - target[index]) <= gamma * scale[index]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_sparse_categorical_cross_uses_one_grouped_pass(monkeypatch, reverse, signed):
    # Removing this route rebuilds and scans one observation vector per column.
    left, right = _categorical_pair()
    weights = np.linspace(0.0, 1.0, 12)
    if signed:
        weights[::2] *= -1
    monkeypatch.setattr(algebra, "_cross_gram_by_columns", _forbidden)
    monkeypatch.setattr(SparseSSPGroupMatrix, "toarray", _forbidden)
    grouped_calls = []
    grouped = algebra._csr_weighted_bincount

    def recorded(*args):
        grouped_calls.append(True)
        return grouped(*args)

    monkeypatch.setattr(algebra, "_csr_weighted_bincount", recorded)
    profile = {}
    actual = (
        algebra._cross_gram(right, left, weights, profile=profile).T
        if reverse
        else algebra._cross_gram(left, right, weights, profile=profile)
    )
    _assert_categorical_target(actual, left, right, weights)
    assert profile["block_cross_ssp_categorical_calls"] == 1
    assert grouped_calls == [True]


@pytest.mark.parametrize("change", ["values", "replace", "codes", "weights"])
def test_sparse_categorical_cross_observes_live_changes(change):
    left, right = _categorical_pair()
    weights = np.linspace(-0.5, 1.0, 12)
    for iteration in range(2):
        if iteration:
            if change == "values":
                left.B.data *= 0.5
                left.R_inv *= 2
            elif change == "replace":
                left.B = sp.csr_matrix(left.B.toarray()[:, ::-1])
                left.R_inv = left.R_inv[::-1].copy()
            elif change == "codes":
                right.codes[:] = np.roll(right.codes, 1)
            else:
                weights *= -0.5
        actual = algebra._cross_gram(left, right, weights)
        _assert_categorical_target(actual, left, right, weights)


@pytest.mark.parametrize(
    "decline",
    [
        "weights32",
        "transform32",
        "custom",
        "csr_subclass",
        "noncanonical",
        "budget",
        "extreme",
        "sensitive",
    ],
)
def test_sparse_categorical_cross_retains_guarded_column_fallback(monkeypatch, decline):
    left, right = _categorical_pair()
    weights = np.ones(12)
    if decline == "weights32":
        weights = weights.astype(np.float32)
    elif decline == "transform32":
        left.R_inv = left.R_inv.astype(np.float32)
    elif decline == "custom":

        class CustomCategory(CategoricalGroupMatrix):
            pass

        right = CustomCategory(np.where(right.codes == 3, -1, right.codes), 3)
    elif decline == "csr_subclass":

        class CustomCSR(sp.csr_matrix):
            pass

        left.B = CustomCSR(left.B)
    elif decline == "noncanonical":
        # A cached scipy canonical flag must not hide a subsequent index edit.
        assert left.B.has_canonical_format
        left.B.indices[1] = left.B.indices[0]
    elif decline == "budget":
        monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", 1)
    elif decline == "extreme":
        left.B.data *= np.ldexp(1.0, 1020)
        weights *= np.ldexp(1.0, -1020)
    else:
        raw = np.ones((12, 2))
        raw[:, 1] += np.arange(12) * np.finfo(float).eps
        left = SparseSSPGroupMatrix(sp.csr_matrix(raw), np.array([[1.0], [-1.0]]))
        assert left._gram_with_projection(weights)[1]
    calls = []
    original = algebra._cross_gram_by_columns

    def recorded(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(algebra, "_cross_gram_by_columns", recorded)
    actual = algebra._cross_gram(left, right, weights)
    _assert_categorical_target(actual, left, right, weights)
    assert calls == [True]
