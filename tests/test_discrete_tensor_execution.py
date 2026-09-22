"""Stored tensor algebra and bounded execution workspaces."""

from __future__ import annotations

import pickle
import tracemalloc
import weakref

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
from superglm.types import GroupSlice, TensorRawChannels


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


# ── Tensor x tensor cross-Gram over distinct margins: the channel route ────


_BAND = 4


def _banded_margin(rng, bins, k):
    """A margin whose centred basis is a width-4 raw band times a projection.

    Returns ``(offsets, values, projection, basis)`` with ``basis`` the band
    expanded into its ``k + 3`` raw columns and projected -- the identity the
    raw stage relies on, here true by construction.  Three spare columns give
    every row a choice of window, so the offsets vary.
    """
    k_raw = k + 3
    offsets = rng.integers(k_raw - _BAND + 1, size=bins, dtype=np.intp)
    values = rng.normal(size=(bins, _BAND))
    projection = rng.normal(size=(k_raw, k)) / np.sqrt(k_raw)
    expanded = np.zeros((bins, k_raw))
    np.put_along_axis(expanded, offsets[:, None] + np.arange(_BAND), values, axis=1)
    return offsets, values, projection, expanded @ projection


def _tensor_pair(n, left, right, *, shared=False, same_id=False, seed=911, banded=True):
    """Two factored tensors over the same ``n`` rows.

    ``left`` and ``right`` are ``(n1, n2, k1, k2, p)``.  Each stored joint
    support covers EVERY cell of its grid, so the displaced route's
    ``n_joint`` is fixed by the grid sizes rather than by which cells the
    draw observed.  ``shared`` makes the first margin's bin index the same
    array on both sides (one shared margin), or both bin indices with
    ``"both"``; ``same_id`` gives the right tensor the left one's margins,
    indices and id, differing only in its transform (a decomposed subgroup
    pair).  ``banded`` attaches both margins' raw band as ``raw_channels``
    (the production ``ps`` case) to both tensors, to neither with ``False``,
    or per side with a ``(left, right)`` pair.
    """
    rng = np.random.default_rng(seed)
    banded_left, banded_right = (banded, banded) if isinstance(banded, bool) else banded

    def build(shape, idx1, idx2, tensor_id, margins, with_band):
        n1, n2, k1, k2, p = shape
        (offsets1, values1, projection1, b1), (offsets2, values2, projection2, b2) = margins
        joint = np.einsum("ia,jb->ijab", b1, b2).reshape(n1 * n2, k1 * k2)
        transform = rng.normal(size=(k1 * k2, p)) / np.sqrt(k1 * k2)
        raw = TensorRawChannels(
            offsets1=offsets1,
            values1=values1,
            offsets2=offsets2,
            values2=values2,
            k2_raw=k2 + 3,
            projection=np.kron(projection1, projection2),
        )
        return DiscretizedTensorGroupMatrix(
            b1,
            b2,
            idx1,
            idx2,
            joint,
            transform,
            idx1 * n2 + idx2,
            tensor_id=tensor_id,
            raw_channels=raw if with_band else None,
        )

    def margins(shape):
        return _banded_margin(rng, shape[0], shape[2]), _banded_margin(rng, shape[1], shape[3])

    def draw(shape):
        return tuple(rng.integers(bins, size=n, dtype=np.intp) for bins in shape[:2])

    idx1, idx2 = draw(left)
    left_margins = margins(left)
    first = build(left, idx1, idx2, 1, left_margins, banded_left)
    if same_id:
        second = build(left, idx1, idx2, 1, left_margins, banded_right)
    else:
        other1, other2 = draw(right)
        second = build(
            right,
            idx1 if shared else other1,
            idx2 if shared == "both" else other2,
            2,
            margins(right),
            banded_right,
        )
    return first, second, rng


def _stable_cell_order(tensor):
    return np.argsort(tensor.idx1 * tensor.n_bins2 + tensor.idx2, kind="stable")


def test_tensor_cell_csr_is_a_stable_counting_sort():
    # 1480 cells for 120 rows: most cells are empty, so ptr carries runs of
    # equal offsets and the sort must still place every row.
    left, _right, _rng = _tensor_pair(120, (40, 37, 3, 3, 5), (33, 41, 2, 4, 6))
    ptr, order = left.cell_csr()
    np.testing.assert_array_equal(order, _stable_cell_order(left))
    counts = np.bincount(left.idx1 * left.n_bins2 + left.idx2, minlength=40 * 37)
    np.testing.assert_array_equal(ptr, np.concatenate([[0], np.cumsum(counts)]))
    assert ptr[-1] == left.shape[0]
    assert order.dtype == np.intp
    again = left.cell_csr()
    assert again[0] is ptr and again[1] is order


@pytest.mark.parametrize("index", ["idx1", "idx2"])
@pytest.mark.parametrize("replace", [False, True])
def test_tensor_cell_cache_revalidates_live_indices(index, replace):
    left, right, rng = _tensor_pair(400, (7, 5, 3, 4, 6), (6, 8, 3, 4, 5))
    weights = rng.normal(size=400)
    algebra._cross_gram(left, right, weights)
    old = left.cell_csr()
    values = getattr(left, index)
    changed = (values + 1) % (left.n_bins1 if index == "idx1" else left.n_bins2)
    if replace:
        setattr(left, index, changed)
    else:
        values[:] = changed
    # Tensor storage duplicates the row addresses for the packed-row methods.
    left.bin_idx = left.idx1 * left.n_bins2 + left.idx2
    ptr, order = left.cell_csr()
    np.testing.assert_array_equal(order, _stable_cell_order(left))
    assert ptr is not old[0] and order is not old[1]
    expected, bound = _dense_cross(left, right, weights)
    actual = algebra._cross_gram(left, right, weights)
    _assert_cross_matches(actual, expected, bound)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("factor", ["raw_values", "projection_map"])
def test_raw_channel_stage_declines_exceptional_source_factors(direction, factor):
    left, right, rng = _tensor_pair(40, (4, 5, 2, 2, 3), (6, 7, 2, 2, 3))
    band = right.raw_channels
    weights = rng.uniform(0.5, 1.5, 40)
    if factor == "raw_values":
        band.values1[:] *= np.ldexp(1.0, 500 * direction)
        band.values2[:] *= np.ldexp(1.0, 500 * direction)
        band.projection[:] *= np.ldexp(1.0, -1000 * direction)
        weights *= np.ldexp(1.0, 100 * direction)
    else:
        band.values1[:] *= np.ldexp(1.0, -300 * direction)
        band.values2[:] *= np.ldexp(1.0, -300 * direction)
        band.projection[:] *= np.ldexp(1.0, 600 * direction)
        right.R_inv *= np.ldexp(1.0, 500 * direction)
    expected, bound = _dense_cross(left, right, weights)
    profile = {}
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        actual = algebra._cross_gram(left, right, weights, profile=profile)
    _assert_cross_matches(actual, expected, bound)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == 0


def test_row_subset_does_not_inherit_the_cell_csr():
    n = 400
    left, right, rng = _tensor_pair(n, (7, 5, 3, 4, 6), (6, 8, 3, 4, 5))
    left.cell_csr()
    rows = rng.choice(n, size=n // 3, replace=False)
    left_sub = left.row_subset(rows)
    assert left_sub._cell_csr is None
    _ptr, order = left_sub.cell_csr()
    np.testing.assert_array_equal(order, _stable_cell_order(left_sub))

    # A malformed parent cache is rebuilt without affecting the subset's
    # independent order. The left tensor is the grid side (35 x 5 < 48 x 6).
    weights = rng.normal(size=n)
    expected, bound = _dense_cross(left, right, weights)
    ptr, order = left.cell_csr()
    left._cell_csr = (ptr, rng.permutation(order))
    profile = {}
    repaired = algebra._cross_gram(left, right, weights, profile=profile)
    assert profile["block_cross_tensor_tensor_channel_raw"] == 1
    _assert_cross_matches(repaired, expected, bound)
    right_sub = right.row_subset(rows)
    sub_expected, sub_bound = _dense_cross(left_sub, right_sub, weights[rows])
    profile = {}
    actual = algebra._cross_gram(left_sub, right_sub, weights[rows], profile=profile)
    assert profile["block_cross_tensor_tensor_channel_raw"] == 1
    _assert_cross_matches(actual, sub_expected, sub_bound)


def test_channel_accumulator_is_reused_across_blocks_in_a_build(monkeypatch):
    # Three blocks through one build cache: the second needs a larger scratch
    # (a 10 x 10 grid against 4 x 4), so the buffer grows once; the third fits
    # the grown buffer and reuses it.  Stale contents cannot leak because the
    # kernel writes every row, which the oracle check on each block pins.
    n = 300
    small = _tensor_pair(n, (4, 4, 2, 2, 3), (4, 4, 2, 2, 3))
    large = _tensor_pair(n, (10, 10, 2, 2, 3), (10, 10, 2, 2, 3), seed=5)
    weights = small[2].normal(size=n)
    scratches = []
    original = algebra._cell_hist_raw_kron

    def recorded(*args):
        scratches.append(args[-1])
        return original(*args)

    monkeypatch.setattr(algebra, "_cell_hist_raw_kron", recorded)
    cache = algebra._BlockWeightCache()
    for left, right, _rng in (small, large, small):
        expected, bound = _dense_cross(left, right, weights)
        actual = algebra._cross_gram(left, right, weights, cache=cache)
        _assert_cross_matches(actual, expected, bound)
    first, grown, reused = scratches
    assert grown.size > first.size
    assert not np.shares_memory(first, grown)
    assert np.shares_memory(grown, reused)
    assert reused.shape == first.shape


def test_cell_weights_are_permuted_once_per_grid_tensor_in_a_build(monkeypatch):
    # Two partners of one grid tensor through one build cache read the same
    # permuted weights -- one O(n) pass where there were two -- and a
    # different weight vector is permuted afresh.  The 4 x 4 tensor is the
    # grid side of every block (16 x 3 cells against 100 x 3 and 99 x 3).
    n = 300
    grid, first, rng = _tensor_pair(n, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    _grid, second, _rng = _tensor_pair(n, (4, 4, 2, 2, 3), (9, 11, 2, 2, 3), seed=5)
    weights = rng.normal(size=n)
    permuted = []
    original = algebra._cell_hist_raw_kron

    def recorded(ptr, bin1, bin2, w, *operands):
        permuted.append(w)
        return original(ptr, bin1, bin2, w, *operands)

    monkeypatch.setattr(algebra, "_cell_hist_raw_kron", recorded)
    profile = {}
    cache = algebra._BlockWeightCache(profile)
    for partner, w in ((first, weights), (second, weights), (first, weights.copy())):
        expected, bound = _dense_cross(grid, partner, w)
        actual = algebra._cross_gram(grid, partner, w, cache=cache, profile=profile)
        _assert_cross_matches(actual, expected, bound)
    assert profile["block_cross_tensor_tensor_channel_raw"] == 3
    assert profile["block_cell_weight_reuses"] == 1
    assert permuted[0] is permuted[1] and permuted[2] is not permuted[0]
    np.testing.assert_array_equal(permuted[0], weights[grid.cell_csr()[1]])


@pytest.mark.parametrize("cached", [False, True])
def test_channel_invariant_scans_run_once_per_assembly(monkeypatch, cached):
    grid, first, rng = _tensor_pair(300, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    _grid, second, _rng = _tensor_pair(300, (4, 4, 2, 2, 3), (9, 11, 2, 2, 3), seed=5)
    second.tensor_id = 3
    weights = rng.normal(size=300)
    own_margin = DiscretizedSSPGroupMatrix(grid.B1_unique_t, np.eye(2), grid.idx1)
    main = DiscretizedSSPGroupMatrix(rng.normal(size=(5, 2)), np.eye(2), rng.integers(5, size=300))
    calls = {"legacy": 0, "bounds": 0, "cells": 0}
    for name, key in (
        ("_tensor_operand_in_reassociation_range", "legacy"),
        ("_operand_exponent_bounds", "bounds"),
    ):
        original = getattr(algebra, name)

        def recorded(values, original=original, key=key):
            calls[key] += int(np.shares_memory(values, weights))
            return original(values)

        monkeypatch.setattr(algebra, name, recorded)
    original_cells = DiscretizedTensorGroupMatrix.cell_csr

    def recorded_cells(group):
        calls["cells"] += 1
        return original_cells(group)

    monkeypatch.setattr(DiscretizedTensorGroupMatrix, "cell_csr", recorded_cells)
    cache = algebra._BlockWeightCache() if cached else None
    for left, right in ((grid, first), (grid, second), (grid, first), (second, first)):
        algebra._cross_gram(left, right, weights, cache=cache)
    profile = {}
    for left, right in ((grid, main), (grid, own_margin), (main, own_margin)):
        algebra._cross_gram(left, right, weights, cache=cache, profile=profile)
    assert "block_cross_tensor_main_s" in profile
    assert "block_cross_tensor_own_margin_s" in profile
    assert profile["block_cross_disc_disc_hist_calls"] == 1
    # The seven cross calls each scan uncached weights once. The discrete
    # pair shares one range decision between its two support projections.
    assert calls == {
        "legacy": 1 if cached else 4,
        "bounds": 1 if cached else 7,
        "cells": 2 if cached else 4,
    }


@pytest.mark.parametrize("cross", ["tensor_main", "tensor_own_margin", "discrete_discrete"])
def test_mixed_cross_values_after_channel_primes_assembly_cache(cross):
    grid, partner, rng = _tensor_pair(300, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    weights = rng.normal(size=300)
    own_margin = DiscretizedSSPGroupMatrix(grid.B1_unique_t, rng.normal(size=(2, 2)), grid.idx1)
    main = DiscretizedSSPGroupMatrix(
        rng.normal(size=(5, 2)), rng.normal(size=(2, 2)), rng.integers(5, size=300)
    )
    cache = algebra._BlockWeightCache()
    algebra._cross_gram(grid, partner, weights, cache=cache)
    left, right = {
        "tensor_main": (grid, main),
        "tensor_own_margin": (grid, own_margin),
        "discrete_discrete": (main, own_margin),
    }[cross]
    x, y = left.toarray(), right.toarray()
    expected = x.T @ (weights[:, None] * y)
    scale = np.linalg.norm(abs(x).T @ (abs(weights[:, None]) * abs(y)), ord=np.inf)
    bound = 32 * np.finfo(float).eps * max(*x.shape, *y.shape) * scale
    actual = algebra._cross_gram(left, right, weights, cache=cache)
    assert np.linalg.norm(actual - expected, ord=np.inf) <= bound


@pytest.mark.parametrize("cached", [False, True])
def test_support_factor_scans_reuse_only_owned_factors_within_assembly(monkeypatch, cached):
    group = DiscretizedSSPGroupMatrix(
        np.array([[1.0, 0.25], [0.5, 1.0]]), np.eye(2), np.array([0, 1, 0])
    )
    partner = np.ones((2, 3))
    original = algebra._operand_exponent_bounds
    scans = {"basis": 0, "transform": 0, "partner": 0}

    def recorded(values):
        for name, operand in (
            ("basis", group.B_unique),
            ("transform", group.R_inv),
            ("partner", partner),
        ):
            scans[name] += int(values is operand)
        return original(values)

    monkeypatch.setattr(algebra, "_operand_exponent_bounds", recorded)
    for _ in range(2):
        cache = algebra._BlockWeightCache() if cached else None
        for value in (1.0, 2.0, 3.0):
            partner.fill(value)
            algebra._cross_support(group, cache, partner)
    assert scans == {
        "basis": 2 if cached else 6,
        "transform": 2 if cached else 6,
        "partner": 6,
    }


@pytest.mark.parametrize("replace", [False, True])
def test_support_factor_cache_observes_mutation_between_moment_assemblies(replace):
    from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan

    groups = [
        DiscretizedSSPGroupMatrix(
            np.array([[1.0, 0.25], [0.5, 1.0]]) * (index + 1),
            np.array([[1.0, 0.125], [-0.25, 1.0]]),
            np.array([0, 1, index, 0]),
        )
        for index in range(2)
    ]
    weights = np.array([1.0, 0.5, 2.0, 0.75])
    plan = MatrixExecutionPlan(groups, n=len(weights))
    for iteration in range(2):
        if iteration:
            for group in groups:
                for name in ("B_unique", "R_inv"):
                    changed = getattr(group, name) * 0.5
                    if replace:
                        setattr(group, name, changed)
                    else:
                        getattr(group, name)[:] = changed
            weights *= 2.0
        design = np.hstack([group.toarray() for group in groups])
        expected = design.T @ (weights[:, None] * design)
        scale = np.linalg.norm(abs(design).T @ (weights[:, None] * abs(design)), np.inf)
        bound = 16 * np.finfo(float).eps * max(design.shape) * scale
        actual = plan.moments(weights).gram
        assert np.linalg.norm(actual - expected, np.inf) <= bound


@pytest.mark.parametrize("cached", [False, True])
def test_discrete_cross_scans_each_owned_factor_once_per_assembly(monkeypatch, cached):
    groups = [
        DiscretizedSSPGroupMatrix(
            np.array([[1.0, 0.25], [0.5, 1.0]]) * (index + 1),
            np.eye(2),
            np.array([0, 1, index, 0]),
        )
        for index in range(2)
    ]
    factors = [factor for group in groups for factor in (group.B_unique, group.R_inv)]
    scans = [0] * len(factors)
    original = algebra._operand_exponent_bounds

    def recorded(values):
        for index, factor in enumerate(factors):
            scans[index] += int(values is factor)
        return original(values)

    monkeypatch.setattr(algebra, "_operand_exponent_bounds", recorded)
    weights = np.array([1.0, -0.5, 2.0, 0.75])
    for _ in range(2):
        cache = algebra._BlockWeightCache() if cached else None
        algebra._cross_gram(*groups, weights, cache=cache)
        algebra._cross_gram(*reversed(groups), weights, cache=cache)
    # Two assemblies, each visiting both orientations. A factor is scanned
    # once per call without a cache, or once per assembly with a cache.
    assert scans == [2 if cached else 4] * len(factors)


@pytest.mark.parametrize("replace", [False, True])
def test_discrete_cross_partner_range_cache_observes_next_assembly_mutation(replace):
    left = DiscretizedSSPGroupMatrix(np.ones((2, 1)), np.ones((1, 1)), np.array([0, 1]))
    right = DiscretizedSSPGroupMatrix(np.ones((2, 1)), np.ones((1, 1)), np.array([1, 0]))
    weights = np.array([0.5, 1.0])
    for exponent in (0, 1023):
        for group, name, value in (
            (left, "B_unique", np.ldexp(1.0, exponent)),
            (right, "R_inv", np.ldexp(1.0, -exponent)),
        ):
            if replace:
                setattr(group, name, np.full_like(getattr(group, name), value))
            else:
                getattr(group, name).fill(value)
        cache = algebra._BlockWeightCache()
        # Reciprocal factors keep the represented cross at exactly sum(weights),
        # but the high exponent forces the unchanged raw-association guard.
        with np.errstate(over="raise", invalid="raise", under="raise"):
            actual = algebra._cross_gram(left, right, weights, cache=cache)
        np.testing.assert_array_equal(actual, [[1.5]])
        if exponent:
            assert not cache._supports


@pytest.mark.parametrize("operand", ["B_unique", "R_inv", "partner"])
@pytest.mark.parametrize("exponent", [-1023, 1023])
def test_support_factor_cache_keeps_extreme_range_refusal(operand, exponent):
    group = DiscretizedSSPGroupMatrix(np.ones((2, 1)), np.ones((1, 1)), np.array([0, 1]))
    partner = np.ones((2, 1))
    cache = algebra._BlockWeightCache()
    assert algebra._cross_support(group, cache, partner)[1] is None
    if operand == "partner":
        # A reused weighted scratch object can change inside one assembly.
        partner[:] = np.ldexp(1.0, exponent)
    else:
        getattr(group, operand)[:] = np.ldexp(1.0, exponent)
        cache = algebra._BlockWeightCache()
    support, transform = algebra._cross_support(group, cache, partner)
    assert support is group.B_unique and transform is group.R_inv


@pytest.mark.parametrize("own_margin", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_tensor_main_partner_scans_run_once_per_assembly(monkeypatch, own_margin, cached):
    tensor, rng = _tensor(3, 4, 2, 2, 20, 3)
    main = DiscretizedSSPGroupMatrix(
        rng.normal(size=(3, 2)), np.eye(2), tensor.idx1 if own_margin else rng.integers(3, size=20)
    )
    cross = algebra._cross_gram_tensor_own_margin if own_margin else algebra._cross_gram_tensor_main
    factors = (tensor.B1_unique_t, tensor.B2_unique_t, tensor.R_inv)
    scans = [0, 0, 0]
    original = algebra._operand_exponent_bounds

    def recorded(values):
        for index, factor in enumerate(factors):
            scans[index] += int(values is factor)
        return original(values)

    monkeypatch.setattr(algebra, "_operand_exponent_bounds", recorded)
    weights = rng.normal(size=20)
    for _ in range(2):
        cache = algebra._BlockWeightCache() if cached else None
        for _ in range(3):
            assert cross(tensor, main, weights, cache) is not None
    assert scans == [2 if cached else 6] * 3


@pytest.mark.parametrize("own_margin", [False, True])
@pytest.mark.parametrize("replace", [False, True])
def test_tensor_main_partner_cache_observes_live_margins_next_assembly(own_margin, replace):
    tensor, rng = _tensor(3, 4, 2, 2, 20, 3)
    main = DiscretizedSSPGroupMatrix(
        rng.normal(size=(3, 2)), np.eye(2), tensor.idx1 if own_margin else rng.integers(3, size=20)
    )
    cross = algebra._cross_gram_tensor_own_margin if own_margin else algebra._cross_gram_tensor_main
    weights = rng.normal(size=20)
    for iteration in range(2):
        if iteration:
            for name in ("B1_unique_t", "B2_unique_t", "R_inv"):
                changed = getattr(tensor, name) * 0.5
                if replace:
                    setattr(tensor, name, changed)
                else:
                    getattr(tensor, name)[:] = changed
            weights *= 2.0
        # Evaluate live margins literally; the stored joint table deliberately
        # remains unchanged so it cannot hide stale marginal-factor reuse.
        x = main.B_unique[main.bin_idx] @ main.R_inv
        y = (
            np.array(
                [
                    np.kron(tensor.B1_unique_t[i], tensor.B2_unique_t[j])
                    for i, j in zip(tensor.idx1, tensor.idx2, strict=True)
                ]
            )
            @ tensor.R_inv
        )
        expected = x.T @ (weights[:, None] * y)
        scale = np.max(abs(x).T @ (abs(weights[:, None]) * abs(y)))
        actual = cross(tensor, main, weights, algebra._BlockWeightCache())
        _assert_product_close(actual, expected, scale, max(*x.shape, *y.shape))


def test_support_factor_cache_does_not_retain_weighted_partner():
    group = DiscretizedSSPGroupMatrix(np.ones((2, 1)), np.ones((1, 1)), np.array([0, 1]))
    partner = np.ones((2, 3))
    owner = weakref.ref(partner)
    cache = algebra._BlockWeightCache()
    algebra._cross_support(group, cache, partner)
    del partner
    assert owner() is None


@pytest.mark.parametrize("cached", [False, True])
def test_cached_weight_range_preserves_inclusive_legacy_endpoints(cached):
    left, right, _rng = _tensor_pair(32, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    cache = algebra._BlockWeightCache() if cached else None
    for value, admitted in (
        (2.0**-128, True),
        (np.nextafter(2.0**-128, 0.0), False),
        (2.0**128, True),
        (np.nextafter(2.0**128, np.inf), False),
    ):
        result = algebra._cross_gram_tensor_tensor_channels(left, right, np.full(32, value), cache)
        assert (result is not None) == admitted


def test_channel_assembly_retains_identity_key_owners():
    grid, partner, rng = _tensor_pair(300, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    weights = rng.normal(size=300)
    owner = weakref.ref(weights)
    cache = algebra._BlockWeightCache()
    algebra._cross_gram(grid, partner, weights, cache=cache)
    del weights
    assert owner() is not None  # A replacement must not alias a released identity key.
    del cache
    assert owner() is None


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("replace", [False, True])
def test_channel_invariants_observe_mutations_between_assemblies(cached, replace):
    grid, partner, rng = _tensor_pair(300, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    weights = rng.normal(size=300)
    for iteration in range(2):
        if iteration:
            for name, bins in (("idx1", grid.n_bins1), ("idx2", grid.n_bins2)):
                changed = (getattr(grid, name) + 1) % bins
                if replace:
                    setattr(grid, name, changed)
                else:
                    getattr(grid, name)[:] = changed
            grid.bin_idx = grid.idx1 * grid.n_bins2 + grid.idx2
            if replace:
                weights = weights * -0.75
            else:
                weights *= -0.75
        expected, bound = _dense_cross(grid, partner, weights)
        cache = algebra._BlockWeightCache() if cached else None
        actual = algebra._cross_gram(grid, partner, weights, cache=cache)
        _assert_cross_matches(actual, expected, bound)


@pytest.mark.parametrize("margin", ["B1_unique_t", "B2_unique_t"])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("transfer", ["direct", "subset", "pickle", "lambda"])
def test_tensor_cross_observes_changed_marginals(margin, replace, transfer):
    left, right, rng = _tensor_pair(64, (4, 4, 2, 2, 3), (10, 10, 2, 2, 4))
    weights = rng.normal(size=64)
    algebra._cross_gram(left, right, weights)
    values = getattr(right, margin)
    if replace:
        setattr(right, margin, 2.0 * values)
    else:
        values *= 2.0
    right.B_unique = np.einsum("ia,jb->ijab", right.B1_unique_t, right.B2_unique_t).reshape(
        right.n_bins1 * right.n_bins2, -1
    )
    if transfer == "subset":
        rows = np.arange(0, 64, 2)
        left, right = left.row_subset(rows), right.row_subset(rows)
        weights = weights[rows]
    elif transfer == "pickle":
        right = pickle.loads(pickle.dumps(right))
    elif transfer == "lambda":
        right.omega = np.eye(right.B_unique.shape[1])
        dm = DesignMatrix([right], 64, right.shape[1])
        groups = [GroupSlice("tensor", 0, right.shape[1])]
        rebuilt = rebuild_design_matrix_with_lambdas(dm, groups, {"tensor": 2.0}, np.ones(64), 1.0)
        right = rebuilt.group_matrices[0]
    expected, bound = _dense_cross(left, right, weights)
    actual = algebra._cross_gram(left, right, weights, cache=algebra._BlockWeightCache())
    _assert_cross_matches(actual, expected, bound)


@pytest.mark.parametrize("margin", ["B1_unique_t", "B2_unique_t"])
@pytest.mark.parametrize("changed", [False, True])
def test_raw_band_dispatch_checks_marginal_values(margin, changed):
    left, right, rng = _tensor_pair(64, (4, 4, 2, 2, 3), (10, 10, 2, 2, 4))
    setattr(right, margin, getattr(right, margin).copy() * (2 if changed else 1))
    right.B_unique = np.einsum("ia,jb->ijab", right.B1_unique_t, right.B2_unique_t).reshape(
        right.n_bins1 * right.n_bins2, -1
    )
    profile = {}
    algebra._cross_gram(left, right, rng.normal(size=64), profile=profile)
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == int(not changed)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1


def test_raw_band_dispatch_declines_legacy_pickle_without_marginal_state():
    left, right, rng = _tensor_pair(64, (4, 4, 2, 2, 3), (10, 10, 2, 2, 4))
    dict_state, slot_state = right.__getstate__()
    slot_state.pop("_raw_channel_state", None)
    restored = DiscretizedTensorGroupMatrix.__new__(DiscretizedTensorGroupMatrix)
    restored.__setstate__((dict_state, slot_state))
    profile = {}
    algebra._cross_gram(left, restored, rng.normal(size=64), profile=profile)
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == 0
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1


class _TaggedTensor(DiscretizedTensorGroupMatrix):
    __slots__ = ("tag_slot", "__dict__")


@pytest.mark.parametrize("protocol", [4, pickle.HIGHEST_PROTOCOL])
def test_tensor_pickle_preserves_subclass_dictionary(protocol):
    base, _, _ = _tensor_pair(32, (4, 4, 2, 2, 3), (10, 10, 2, 2, 3))
    tagged = _TaggedTensor.__new__(_TaggedTensor)
    tagged.__setstate__(base.__getstate__())
    tagged.label = {"name": "interaction", "revision": 3}
    tagged.tag_slot = "slot state"
    restored = pickle.loads(pickle.dumps(tagged, protocol=protocol))
    assert restored.label == tagged.label
    assert restored.tag_slot == tagged.tag_slot


def test_tensor_group_pickles_without_its_cell_csr_and_loads_from_before_the_band():
    n = 400
    left, right, rng = _tensor_pair(n, (7, 5, 3, 4, 6), (6, 8, 3, 4, 5))
    weights = rng.normal(size=n)
    expected, bound = _dense_cross(left, right, weights)
    left.cell_csr()
    restored = pickle.loads(pickle.dumps(left))
    assert restored._cell_csr is None
    np.testing.assert_array_equal(restored.raw_channels.values1, left.raw_channels.values1)
    profile = {}
    actual = algebra._cross_gram(restored, right, weights, profile=profile)
    assert profile["block_cross_tensor_tensor_channel_raw"] == 1
    _assert_cross_matches(actual, expected, bound)

    # A design pickled before the branch carries neither new slot.  Loaded,
    # the channel side has no band and the block takes the dense stage.
    _dict_state, slot_state = right.__getstate__()
    del slot_state["raw_channels"]
    old = DiscretizedTensorGroupMatrix.__new__(DiscretizedTensorGroupMatrix)
    old.__setstate__((None, slot_state))
    assert old.raw_channels is None and old._cell_csr is None
    profile = {}
    actual = algebra._cross_gram(left, old, weights, profile=profile)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == 0
    _assert_cross_matches(actual, expected, bound)


@pytest.mark.parametrize("case", ["unbanded", "banded_channel", "banded_grid"])
def test_raw_channel_stage_declines_to_the_dense_kernel_without_a_band(case):
    # The smaller grid is the grid side (30 x 5 < 100 x 6 cells), so the raw
    # kernel runs exactly when the LARGER tensor -- the channel side -- carries
    # the band; a band on the grid side alone is not consumed.
    n = 400
    banded = {"unbanded": False, "banded_channel": (False, True), "banded_grid": (True, False)}
    left, right, rng = _tensor_pair(n, (6, 5, 2, 3, 6), (10, 10, 2, 3, 5), banded=banded[case])
    assert (left.raw_channels is None) == (case != "banded_grid")
    assert (right.raw_channels is None) == (case != "banded_channel")
    weights = rng.normal(size=n)
    expected, bound = _dense_cross(left, right, weights)
    profile = {}
    actual = algebra._cross_gram(left, right, weights, profile=profile)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == int(case == "banded_channel")
    _assert_cross_matches(actual, expected, bound)


def test_raw_stage_declines_to_the_dense_stage_between_the_two_widths(monkeypatch):
    # 48 x 48 grids with 12 centred and 42 raw channels: a budget between
    # 2304 * 12 and 2304 * 42 cells admits the channel route at the stored
    # width but not the raw scratch, so the dense stage runs -- not the row
    # route.  The raw stage's wider scratch must never cost a block the
    # channel route it had before.
    n = 6000
    left, right, rng = _tensor_pair(n, (48, 48, 3, 4, 10), (48, 48, 4, 3, 11))
    stored = 48 * 48 * right.B_unique.shape[1]
    raw = 48 * 48 * right.raw_channels.projection.shape[0]
    assert stored < raw
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", (stored + raw) // 2)
    weights = rng.normal(size=n)
    expected, bound = _dense_cross(left, right, weights)
    profile = {}
    actual = algebra._cross_gram(left, right, weights, profile=profile)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert "block_cross_tensor_tensor_channel_raw" not in profile
    assert profile.get("block_cross_disc_disc_rows_calls", 0) == 0
    _assert_cross_matches(actual, expected, bound)


@pytest.mark.parametrize(
    "case,expected,comparisons",
    [("shared_over_cap", "channel", 0), ("shared_both_one_under_cap", "tensor_tensor", 1)],
)
def test_shared_margin_probe_skips_the_row_comparison_above_the_cell_cap(
    monkeypatch, case, expected, comparisons
):
    # Two 256 x 256 grids put every pairing's three-way histogram (16.8M
    # cells) over the compact helper's cap, so no O(n) index comparison may
    # run.  With both margins shared on 100 x 300 grids only the (2, 2)
    # pairing (3M cells) is under the cap: it alone is compared, and the
    # compact route is taken rather than declined for having two matches.
    n = 6000
    if case == "shared_over_cap":
        left, right, rng = _tensor_pair(n, (256, 256, 2, 2, 4), (256, 256, 2, 2, 3), shared=True)
    else:
        left, right, rng = _tensor_pair(n, (100, 300, 2, 2, 4), (100, 300, 2, 2, 3), shared="both")
    compared = []
    original = algebra._same_discrete_margin

    def counted(*args):
        compared.append(args[1:4:2])
        return original(*args)

    monkeypatch.setattr(algebra, "_same_discrete_margin", counted)
    weights = rng.normal(size=n)
    expected_block, bound = _dense_cross(left, right, weights)
    profile = {}
    actual = algebra._cross_gram(left, right, weights, profile=profile)
    assert _route(profile) == expected
    assert len(compared) == comparisons
    _assert_cross_matches(actual, expected_block, bound)


_ROUTE_COUNTERS = (
    ("channel", "block_cross_tensor_tensor_channel_calls"),
    ("rows", "block_cross_disc_disc_rows_calls"),
    ("hist", "block_cross_disc_disc_hist_calls"),
)


def _route(profile):
    """The one route a cross block took, read off its profile counters."""
    routes = [route for route, key in _ROUTE_COUNTERS if profile.get(key, 0)]
    if "block_cross_tensor_tensor_s" in profile:
        routes.append("tensor_tensor")
    assert len(routes) == 1, profile
    return routes[0]


def _dense_cross(left, right, weights):
    """The literal ``X_i.T @ diag(W) @ X_j`` and the repo's error bound for it."""
    x = left.toarray()
    y = right.toarray()
    expected = x.T @ (weights[:, None] * y)
    scale = np.linalg.norm(np.abs(x).T @ np.abs(weights[:, None] * y), ord=np.inf)
    reduction = max(len(weights), left.n_bins1 * left.n_bins2, right.n_bins1 * right.n_bins2)
    return expected, 32 * np.finfo(float).eps * reduction * scale


def _assert_cross_matches(actual, expected, bound):
    assert np.linalg.norm(actual - expected, ord=np.inf) <= bound
    assert np.linalg.norm(actual - expected) <= 1e-12 * np.linalg.norm(expected)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("distinct", "channel"),
        ("shared_fits", "tensor_tensor"),
        ("shared_declines", "channel"),
        ("same_id", "tensor_tensor"),
    ],
)
def test_distinct_margin_tensor_cross_gram_route(case, expected):
    # 48 x 48 grids put n_joint = 2304**2 above the histogram cell cap, so the
    # displaced route is the row-expanding one. 200 x 160 x 160 shared cells
    # put the compact shared-margin helper above ITS cap (5,000,000) while
    # 5 x 4 x 3 keep it in play; a shared tensor_id keeps the packed grid.
    n = 6000
    if case == "distinct":
        left, right, rng = _tensor_pair(n, (48, 48, 3, 4, 10), (48, 48, 4, 3, 11))
    elif case == "shared_fits":
        left, right, rng = _tensor_pair(n, (5, 4, 3, 2, 5), (5, 3, 2, 4, 6), shared=True)
    elif case == "shared_declines":
        left, right, rng = _tensor_pair(n, (200, 160, 2, 2, 4), (200, 160, 2, 2, 3), shared=True)
    else:
        left, right, rng = _tensor_pair(n, (7, 5, 3, 2, 5), (7, 5, 3, 2, 4), same_id=True)
    profile = {}
    algebra._cross_gram(left, right, rng.normal(size=n), profile=profile)
    assert _route(profile) == expected
    if expected == "channel":
        assert profile.get("block_cross_disc_disc_rows_calls", 0) == 0


_ORACLE_SHAPES = {
    "equal_k": ((7, 5, 3, 4, 6), (6, 8, 3, 4, 5), 400),
    "different_k": ((9, 4, 2, 5, 7), (5, 11, 4, 3, 4), 500),
    "support_wider_than_n": ((40, 37, 3, 3, 5), (33, 41, 2, 4, 6), 120),
    "single_bin": ((1, 1, 2, 3, 3), (1, 1, 3, 2, 4), 50),
    "tall_few_bins": ((3, 2, 2, 2, 3), (2, 3, 2, 2, 3), 5000),
}


def _weights(kind, rng, n):
    if kind == "uniform":
        return rng.uniform(0.5, 1.5, size=n)
    if kind == "zeros":
        return np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.5, 1.5, size=n))
    return rng.normal(size=n)


@pytest.mark.parametrize("banded", [True, False])
@pytest.mark.parametrize("kind", ["uniform", "zeros", "signed"])
@pytest.mark.parametrize("shape", list(_ORACLE_SHAPES))
def test_distinct_margin_tensor_cross_gram_matches_dense_oracle(shape, kind, banded):
    left_shape, right_shape, n = _ORACLE_SHAPES[shape]
    left, right, rng = _tensor_pair(n, left_shape, right_shape, banded=banded)
    weights = _weights(kind, rng, n)
    expected, bound = _dense_cross(left, right, weights)
    for first, second in ((left, right), (right, left)):
        profile = {}
        actual = algebra._cross_gram(first, second, weights, profile=profile)
        # The route assertions are what give the numeric half its teeth: the
        # displaced route and the dense stage match this oracle too.
        assert profile["block_cross_tensor_tensor_channel_calls"] == 1
        assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == int(banded)
        _assert_cross_matches(actual if first is left else actual.T, expected, bound)


@pytest.mark.parametrize("case", ["tie", "left_smaller", "right_smaller"])
def test_channel_route_grids_the_smaller_side_and_the_left_operand_on_a_tie(monkeypatch, case):
    # The two orientations sum in different orders and differ at about one
    # ulp, so which side is the grid is part of the numerical contract. In
    # production both grids are 256 x 256 with 81 columns and the counts tie.
    # Orientation is decided before the stage; the dense stage is the one
    # observed here, through the kernel that receives the grid's indices.
    small, large = (6, 5, 2, 3, 6), (10, 10, 2, 3, 5)
    if case == "tie":
        left, right, rng = _tensor_pair(300, small, (5, 6, 3, 2, 5), banded=False)
    elif case == "left_smaller":
        left, right, rng = _tensor_pair(300, small, large, banded=False)
    else:
        left, right, rng = _tensor_pair(300, large, small, banded=False)
    grids = []
    original = algebra._disc_disc_2d_hist_channels

    def recorded(*args):
        grids.append("left" if args[0] is left.idx1 else "right" if args[0] is right.idx1 else "?")
        return original(*args)

    monkeypatch.setattr(algebra, "_disc_disc_2d_hist_channels", recorded)
    weights = rng.normal(size=300)
    expected, bound = _dense_cross(left, right, weights)
    forward_profile, reverse_profile = {}, {}
    forward = algebra._cross_gram(left, right, weights, profile=forward_profile)
    reverse = algebra._cross_gram(right, left, weights, profile=reverse_profile)
    expected_grid = "right" if case == "right_smaller" else "left"
    # The smaller histogram is the grid whichever way the operands are
    # passed; only a tie follows the operand order.
    assert grids == [expected_grid, "right" if case == "tie" else expected_grid]
    assert forward_profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert reverse_profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert forward_profile.get("block_cross_tensor_tensor_channel_transposed", 0) == int(
        case == "right_smaller"
    )
    assert reverse_profile.get("block_cross_tensor_tensor_channel_transposed", 0) == int(
        case == "left_smaller"
    )
    _assert_cross_matches(forward, expected, bound)
    _assert_cross_matches(reverse.T, expected, bound)


@pytest.mark.parametrize("fits", [True, False])
def test_channel_route_honours_the_aggregate_cell_budget(monkeypatch, fits):
    n = 6000
    left, right, rng = _tensor_pair(n, (48, 48, 3, 4, 10), (48, 48, 4, 3, 11))
    cells = 48 * 48 * min(left.B_unique.shape[1], right.B_unique.shape[1])
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", cells if fits else cells - 1)
    weights = rng.normal(size=n)
    expected, bound = _dense_cross(left, right, weights)
    profile = {}
    actual = algebra._cross_gram(left, right, weights, profile=profile)
    assert profile.get("block_cross_tensor_tensor_channel_calls", 0) == int(fits)
    # A decline must be visible, not silent: at wider margins the route
    # backs off to the quadratic row route with no other symptom.
    assert profile.get("block_cross_tensor_tensor_channel_declines", 0) == int(not fits)
    assert profile.get("block_cross_disc_disc_rows_calls", 0) == int(not fits)
    _assert_cross_matches(actual, expected, bound)


def test_distinct_margin_tensor_cross_gram_bounds_its_transient():
    # 120,000 rows at 49 + 49 columns exceed one 64 MiB expansion chunk, so
    # the displaced route would hold two 85,598 x 49 row panels (67 MB) live.
    n = 120_000
    width = 7 * 7
    assert n > algebra._cross_expansion_chunk_rows(width, width, algebra._MAX_CROSS_EXPANSION_BYTES)
    left, right, rng = _tensor_pair(n, (64, 64, 7, 7, 40), (64, 64, 7, 7, 40))
    weights = rng.uniform(0.5, 1.5, size=n)
    algebra._cross_gram(left, right, weights)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        profile = {}
        result = algebra._cross_gram(left, right, weights, profile=profile)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert profile["block_cross_tensor_tensor_channel_raw"] == 1
    raw_width = right.raw_channels.projection.shape[0]
    cells = 64 * 64 * raw_width
    gathers = 3 * 8 * n
    tmp_bytes = 8 * 7 * 64 * raw_width
    # Without a build cache the raw scratch (cells doubles) is a fresh
    # allocation, and the three cell-order gathers are n-vectors (24 bytes a
    # row); both are traced.  What the ceiling pins is that no observation-row
    # panel (392 bytes a row here) is materialised.
    assert peak - before <= 8 * cells + gathers + tmp_bytes + 64 * 1024
    assert result.shape == (40, 40)


@pytest.mark.parametrize("budget", [256 << 10, 1 << 20])
@pytest.mark.parametrize("raw_fits", [False, True])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("cold", [False, True])
def test_raw_channel_admission_counts_retained_and_simultaneous_workspace(
    monkeypatch, budget, raw_fits, cached, cold
):
    bins = int(np.sqrt(budget / (2 * 8 * 36)))
    n = budget // (128 if raw_fits else 32)
    left, right, rng = _tensor_pair(n, (bins, bins, 3, 3, 6), (bins, bins, 3, 3, 6))
    weights = rng.uniform(0.5, 1.5, n)
    expected, bound = _dense_cross(left, right, weights)
    algebra._cross_gram(left, right, weights)  # Compile and build the retained cell index.
    band, right.raw_channels = right.raw_channels, None
    algebra._cross_gram(left, right, weights)  # Compile the dense fallback too.
    right.raw_channels = band
    retained = sum(array.nbytes for array in left.cell_csr())
    if cold:
        left._cell_csr = None
        retained = 0
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", budget // 8)
    tracemalloc.start()
    try:
        cache = algebra._BlockWeightCache() if cached else None
        profile = {}
        actual = algebra._cross_gram(left, right, weights, cache=cache, profile=profile)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak + retained <= budget + (32 << 10)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == int(raw_fits)
    _assert_cross_matches(actual, expected, bound)


def test_raw_channel_budget_includes_marginal_validation_before_histogram(monkeypatch):
    budget, n = 64 << 10, 64
    left, right, rng = _tensor_pair(n, (4, 4, 2, 2, 3), (65536, 1, 2, 2, 4))
    weights = rng.uniform(0.5, 1.5, n)
    algebra._cross_gram(left, right, weights)
    band, right.raw_channels = right.raw_channels, None
    algebra._cross_gram(left, right, weights)
    right.raw_channels = band
    retained = sum(array.nbytes for array in left.cell_csr())
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    tracemalloc.start()
    try:
        profile = {}
        algebra._cross_gram(left, right, weights, profile=profile)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak + retained <= budget + (4 << 10)
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == 0
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1


def test_raw_channel_budget_includes_new_cached_weights_during_stage_two(monkeypatch):
    budget, n = 72 << 10, 2000
    left, right, rng = _tensor_pair(n, (4, 4, 32, 3, 6), (10, 10, 2, 2, 6))
    weights = rng.uniform(0.5, 1.5, n)
    expected, bound = _dense_cross(left, right, weights)
    algebra._cross_gram(left, right, weights)
    band, right.raw_channels = right.raw_channels, None
    algebra._cross_gram(left, right, weights)
    right.raw_channels = band
    retained = sum(array.nbytes for array in left.cell_csr())
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    tracemalloc.start()
    try:
        cache = algebra._BlockWeightCache()
        profile = {}
        actual = algebra._cross_gram(left, right, weights, cache=cache, profile=profile)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak + retained <= budget + (4 << 10)
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == 0
    _assert_cross_matches(actual, expected, bound)


def test_channel_workspace_growth_and_dense_fallback_release_unused_buffers(monkeypatch):
    n, budget = 2048, 200 << 10
    small = _tensor_pair(n, (4, 4, 3, 3, 6), (4, 4, 3, 3, 6))
    large = _tensor_pair(n, (21, 21, 3, 3, 6), (21, 21, 3, 3, 6))
    weights = small[2].uniform(0.5, 1.5, n)
    expected = []
    for left, right, _rng in (small, large):
        algebra._cross_gram(left, right, weights)
        expected.append(_dense_cross(left, right, weights))
    retained = max(sum(a.nbytes for a in pair[0].cell_csr()) for pair in (small, large))
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    tracemalloc.start()
    try:
        cache = algebra._BlockWeightCache()
        for pair, oracle in ((small, expected[0]), (large, expected[1]), (small, expected[0])):
            profile = {}
            actual = algebra._cross_gram(*pair[:2], weights, cache=cache, profile=profile)
            assert profile["block_cross_tensor_tensor_channel_raw"] == 1
            _assert_cross_matches(actual, *oracle)
        # The stored-width stage fits after unused raw workspace is released.
        monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 64 << 10)
        large[1].raw_channels = None
        actual = algebra._cross_gram(*large[:2], weights, cache=cache)
        _assert_cross_matches(actual, *expected[1])
        assert cache._channel_scratch.size == 0 and not cache._cell_weights
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak + retained <= budget + (32 << 10)


@pytest.mark.parametrize("route", ["aggregate", "dtype", "range", "shared_margin", "same_id"])
def test_channel_buffers_are_released_before_other_tensor_routes(monkeypatch, route):
    n, budget = 4096, 256 << 10
    first = _tensor_pair(n, (20, 20, 3, 3, 3), (20, 21, 3, 3, 3))
    if route == "shared_margin":
        left, right, rng = _tensor_pair(
            n, (20, 30, 3, 3, 3), (20, 30, 3, 3, 3), shared=True, seed=5
        )
    elif route == "same_id":
        left, right, rng = _tensor_pair(
            n, (140, 140, 2, 2, 3), (140, 140, 2, 2, 3), same_id=True, seed=5
        )
    else:
        left, right, rng = _tensor_pair(n, (20, 20, 3, 3, 3), (20, 21, 3, 3, 3), seed=5)
    weights = rng.uniform(0.5, 1.5, n)
    later_weights = np.ldexp(weights, 140) if route == "range" else weights
    if route == "dtype":
        left.B1_unique_t = left.B1_unique_t.astype(np.float32)
        left.B_unique = np.array(
            [np.kron(a, b) for a in left.B1_unique_t for b in left.B2_unique_t]
        )
    expected, bound = _dense_cross(left, right, later_weights)
    algebra._cross_gram(*first[:2], weights)
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    if route in ("aggregate", "dtype", "range"):
        monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
    with monkeypatch.context() as warm:
        if route == "aggregate":
            warm.setattr(algebra, "_MAX_AGGREGATE_CELLS", 1)
        algebra._cross_gram(left, right, later_weights)
    tracemalloc.start()
    try:
        cache = algebra._BlockWeightCache()
        algebra._cross_gram(*first[:2], weights, cache=cache)
        assert cache._channel_scratch.size and cache._cell_weights
        if route == "aggregate":
            monkeypatch.setattr(algebra, "_MAX_AGGREGATE_CELLS", 1)
        profile = {}
        actual = algebra._cross_gram(left, right, later_weights, cache=cache, profile=profile)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= budget + (32 << 10)
    if route in ("aggregate", "dtype", "range"):
        assert profile["block_cross_tensor_tensor_channel_declines"] == 1
        assert profile["block_cross_disc_disc_rows_calls"] == 1
    else:
        assert "block_cross_tensor_tensor_s" in profile
    assert cache._channel_scratch.size == 0 and not cache._cell_weights
    _assert_cross_matches(actual, expected, bound)


def test_unaffordable_raw_stage_keeps_buffers_when_dense_stage_fits(monkeypatch):
    n = 512
    small = _tensor_pair(n, (5, 5, 3, 3, 3), (5, 5, 3, 3, 3))
    large = _tensor_pair(n, (17, 17, 3, 3, 3), (17, 17, 3, 3, 3), seed=5)
    weights = small[2].uniform(0.5, 1.5, n)
    cache = algebra._BlockWeightCache()
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", 64 << 10)
    algebra._cross_gram(*small[:2], weights, cache=cache)
    scratch, permutations = cache._channel_scratch, dict(cache._cell_weights)
    expected, bound = _dense_cross(*large[:2], weights)
    profile = {}
    actual = algebra._cross_gram(*large[:2], weights, cache=cache, profile=profile)
    _assert_cross_matches(actual, expected, bound)
    assert profile["block_cross_tensor_tensor_channel_calls"] == 1
    assert profile.get("block_cross_tensor_tensor_channel_raw", 0) == 0
    assert cache._channel_scratch is scratch
    assert cache._cell_weights.keys() == permutations.keys()
    assert all(cache._cell_weights[key] is value for key, value in permutations.items())
