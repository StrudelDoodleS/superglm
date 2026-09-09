"""Stored tensor algebra and bounded execution workspaces."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from superglm.group_matrix import DiscretizedTensorGroupMatrix


def _tensor(n1: int, n2: int, k1: int, k2: int, n: int, p: int):
    rng = np.random.default_rng(731)
    b1 = rng.normal(size=(n1, k1)) / np.sqrt(k1)
    b2 = rng.normal(size=(n2, k2)) / np.sqrt(k2)
    raw = np.array([np.kron(left, right) for left in b1 for right in b2])
    transform = np.linalg.qr(rng.normal(size=(k1 * k2, p)))[0]
    pairs = rng.integers(n1 * n2, size=n, dtype=np.intp)
    group = DiscretizedTensorGroupMatrix(
        b1, b2, pairs // n2, pairs % n2, raw, transform, pairs, tensor_id=731
    )
    return group, rng


def _assert_product_close(actual, expected, absolute_product, reduction_size):
    # Both the literal and factored evaluation accumulate floating-point
    # products. Bound their absolute error without assuming cancellation signs.
    tolerance = 16 * np.finfo(np.float64).eps * reduction_size * absolute_product
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize(
    "shape",
    [
        (7, 11, 2, 4, 83, 5),
        (13, 5, 5, 2, 97, 7),
        (2, 11, 3, 17, 89, 15),
        (11, 2, 17, 3, 89, 15),
    ],
)
def test_tensor_signed_gram_and_products_match_literal_stored_design(shape):
    group, rng = _tensor(*shape)
    weights = rng.normal(size=group.shape[0])
    vector = rng.normal(size=group.shape[1])
    design = group.B_unique[group.bin_idx] @ group.R_inv
    gram = design.T @ (weights[:, None] * design)
    scale = np.max(np.abs(design).T @ (np.abs(weights[:, None] * design)))
    _assert_product_close(group.gram(weights), gram, scale, max(*shape))
    expected = design @ vector
    scale = np.max(np.abs(design) @ np.abs(vector))
    _assert_product_close(group.matvec(vector), expected, scale, max(*shape))
    expected = design.T @ weights
    scale = np.max(np.abs(design).T @ np.abs(weights))
    _assert_product_close(group.rmatvec(weights), expected, scale, max(*shape))


def test_tensor_raw_gram_observes_mutable_marginal_tables():
    group, rng = _tensor(7, 9, 2, 3, 41, 4)
    grid = rng.normal(size=(7, 9))
    group._factored_gram_raw(grid)
    group.B1_unique_t[2, 0] += 0.25
    group.B2_unique_t[4, 1] -= 0.5
    raw = np.array(
        [np.kron(left, right) for left in group.B1_unique_t for right in group.B2_unique_t]
    )
    expected = raw.T @ (grid.ravel()[:, None] * raw)
    scale = np.max(np.abs(raw).T @ (np.abs(grid.ravel()[:, None] * raw)))
    _assert_product_close(group._factored_gram_raw(grid), expected, scale, grid.size)


@pytest.mark.parametrize("value", [np.int64(2**32), np.float32(1 + 2**-12)])
def test_tensor_raw_gram_preserves_weighting_dtype_promotion(value):
    b1 = np.ones((1, 1))
    b2 = np.array([[value]])
    indices = np.zeros(1, dtype=np.intp)
    group = DiscretizedTensorGroupMatrix(
        b1, b2, indices, indices, b1 * b2, b1, indices, tensor_id=731
    )
    grid = np.array([[-0.5]])
    # The weighted marginal product promotes the stored value before squaring.
    expected = np.array([[np.float64(value) * (-0.5 * np.float64(value))]])
    np.testing.assert_allclose(
        group._factored_gram_raw(grid), expected, rtol=2 * np.finfo(float).eps, atol=0
    )


@pytest.mark.parametrize(
    "left,right,weight,n1",
    [
        (1.0, 1e200, 1e-200, 1),
        (1.0, 1e-200, 1e200, 1),
        (1e100, 1e-100, 1e200, 2),
        (1e-100, 1e100, 1e-200, 2),
    ],
)
def test_tensor_reassociation_preserves_finite_extreme_scale_gram(left, right, weight, n1):
    b1 = np.zeros((n1, 1))
    b1[0, 0] = left
    b2 = np.array([[right]])
    grid = np.zeros((n1, 1))
    grid[0, 0] = -weight
    indices = np.zeros(1, dtype=np.intp)
    group = DiscretizedTensorGroupMatrix(
        b1, b2, indices, indices, b1 * b2, np.ones((1, 1)), indices, tensor_id=731
    )
    # The row tensor basis is finite; its literal weighted product is an
    # independent observable even when a marginal square/intermediate is not.
    row = left * right
    expected = np.array([[row * (-weight * row)]])
    with np.errstate(over="ignore", invalid="ignore"):
        actual = group._factored_gram_raw(grid)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=8 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.complex128, np.longdouble, np.int64])
def test_tensor_factored_matvec_preserves_numpy_dtype_behavior(dtype):
    group, rng = _tensor(3, 4, 2, 3, 2, 6)
    if dtype == np.int64:
        group.B2_unique_t = np.arange(12, dtype=dtype).reshape(4, 3) + 2**32
    else:
        group.B1_unique_t = group.B1_unique_t.astype(dtype)
        group.B2_unique_t = group.B2_unique_t.astype(dtype)
        group.R_inv = group.R_inv.astype(dtype)
    group.B_unique = np.array(
        [np.kron(left, right) for left in group.B1_unique_t for right in group.B2_unique_t]
    )
    vector = rng.normal(size=6).astype(dtype if dtype != np.int64 else np.float64)
    if dtype == np.complex128:
        vector += 1j * rng.normal(size=6)
    projected = group.B1_unique_t @ (group.R_inv @ vector).reshape(2, 3)
    expected = np.sum(projected[group.idx1] * group.B2_unique_t[group.idx2], axis=1)
    actual = group.matvec(vector)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_tensor_factored_matvec_preserves_finite_extreme_scale_sum():
    b1 = np.ones((1, 1))
    b2 = np.ones((3, 8))
    indices = np.zeros(1, dtype=np.intp)
    group = DiscretizedTensorGroupMatrix(
        b1, b2, indices, indices, b2, np.eye(8), indices, tensor_id=731
    )
    vector = np.array([1e308, 0, 1e308, -1e308, -1e308, 0, 0, 0])
    # NumPy's existing pairwise sum is finite; sequential accumulation of
    # these finite products overflows before the cancelling terms arrive.
    with np.errstate(over="ignore", invalid="ignore"):
        actual = group.matvec(vector)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, np.zeros(1))


def test_tensor_gram_workspace_has_no_bin_grid_times_basis_axis():
    group, rng = _tensor(128, 192, 3, 5, 17, 11)
    grid = rng.normal(size=(128, 192))
    group._factored_gram_raw(grid)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        result = group._factored_gram_raw(grid)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Two marginal outer tables, two contraction/output workspaces, and
    # interpreter overhead. In particular this excludes a (128, 192, 5) array.
    workspace = 8 * (128 * 3**2 + 192 * 5**2 + 2 * 128 * 5**2 + 3 * (3 * 5) ** 2)
    assert peak - before < workspace + 64 * 1024
    assert result.shape == (15, 15)


def test_tensor_matvec_workspace_has_no_observation_by_basis_arrays():
    group, rng = _tensor(192, 256, 3, 5, 20_000, 15)
    vector = rng.normal(size=15)
    group.matvec(vector)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        result = group.matvec(vector)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    workspace = 8 * (2 * group.shape[0] + 192 * 5 + 256 * 3 + 15)
    assert peak - before < workspace + 64 * 1024
    assert result.shape == (20_000,)
