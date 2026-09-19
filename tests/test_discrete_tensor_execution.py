"""Stored tensor algebra and bounded execution workspaces."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_algebra as algebra
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


# ── Tensor x tensor cross-Gram over distinct margins: the channel route ────


def _tensor_pair(n, left, right, *, shared=False, same_id=False, seed=911):
    """Two factored tensors over the same ``n`` rows.

    ``left`` and ``right`` are ``(n1, n2, k1, k2, p)``.  Each stored joint
    support covers EVERY cell of its grid, so the displaced route's
    ``n_joint`` is fixed by the grid sizes rather than by which cells the
    draw observed.  ``shared`` makes the first margin's bin index the same
    array on both sides (one shared margin); ``same_id`` gives the right
    tensor the left one's margins, indices and id, differing only in its
    transform (a decomposed subgroup pair).
    """
    rng = np.random.default_rng(seed)

    def build(shape, idx1, idx2, tensor_id, margins=None):
        n1, n2, k1, k2, p = shape
        if margins is None:
            margins = (
                rng.normal(size=(n1, k1)) / np.sqrt(k1),
                rng.normal(size=(n2, k2)) / np.sqrt(k2),
            )
        b1, b2 = margins
        joint = np.einsum("ia,jb->ijab", b1, b2).reshape(n1 * n2, k1 * k2)
        transform = rng.normal(size=(k1 * k2, p)) / np.sqrt(k1 * k2)
        return DiscretizedTensorGroupMatrix(
            b1, b2, idx1, idx2, joint, transform, idx1 * n2 + idx2, tensor_id=tensor_id
        )

    def draw(shape):
        return tuple(rng.integers(bins, size=n, dtype=np.intp) for bins in shape[:2])

    idx1, idx2 = draw(left)
    first = build(left, idx1, idx2, 1)
    if same_id:
        second = build(left, idx1, idx2, 1, (first.B1_unique_t, first.B2_unique_t))
    else:
        other1, other2 = draw(right)
        second = build(right, idx1 if shared else other1, other2, 2)
    return first, second, rng


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


@pytest.mark.parametrize("kind", ["uniform", "zeros", "signed"])
@pytest.mark.parametrize("shape", list(_ORACLE_SHAPES))
def test_distinct_margin_tensor_cross_gram_matches_dense_oracle(shape, kind):
    left_shape, right_shape, n = _ORACLE_SHAPES[shape]
    left, right, rng = _tensor_pair(n, left_shape, right_shape)
    weights = _weights(kind, rng, n)
    expected, bound = _dense_cross(left, right, weights)
    for first, second in ((left, right), (right, left)):
        profile = {}
        actual = algebra._cross_gram(first, second, weights, profile=profile)
        # The route assertion is what gives the numeric half its teeth: the
        # displaced route matches this oracle too.
        assert profile["block_cross_tensor_tensor_channel_calls"] == 1
        _assert_cross_matches(actual if first is left else actual.T, expected, bound)


@pytest.mark.parametrize("case", ["tie", "left_smaller", "right_smaller"])
def test_channel_route_grids_the_smaller_side_and_the_left_operand_on_a_tie(monkeypatch, case):
    # The two orientations sum in different orders and differ at about one
    # ulp, so which side is the grid is part of the numerical contract. In
    # production both grids are 256 x 256 with 81 columns and the counts tie.
    small, large = (6, 5, 2, 3, 6), (10, 10, 2, 3, 5)
    if case == "tie":
        left, right, rng = _tensor_pair(300, small, (5, 6, 3, 2, 5))
    elif case == "left_smaller":
        left, right, rng = _tensor_pair(300, small, large)
    else:
        left, right, rng = _tensor_pair(300, large, small)
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
    cells = 64 * 64 * width
    tmp_bytes = 8 * 7 * 64 * width
    # The stage-1 histogram (cells doubles) is allocated by numba's runtime,
    # which tracemalloc does not trace, so the ceiling is generous by that
    # term; what it pins is that no observation-row panel is materialised.
    assert peak - before <= 8 * cells + tmp_bytes + 64 * 1024
    assert result.shape == (40, 40)
