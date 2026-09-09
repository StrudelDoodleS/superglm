"""Work/dispatch regressions and independent signed discrete cross oracles."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix._cross_matrix_execution import CrossMatrixExecutionPlan
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import DiscretizedSCOPGroupMatrix, DiscretizedSSPGroupMatrix


def _group(kind, *, n, bins, width, seed):
    rng = np.random.default_rng(seed)
    support = rng.normal(size=(bins, width))
    support -= support.mean(axis=0)
    indices = rng.integers(bins, size=n, dtype=np.intp)
    if kind == "ssp":
        transform = rng.normal(size=(width, width - 1)) / np.sqrt(width)
        return DiscretizedSSPGroupMatrix(support, transform, indices), support[indices] @ transform
    return DiscretizedSCOPGroupMatrix(support, indices), support[indices]


def _plan(left, right, n):
    return CrossMatrixExecutionPlan(
        MatrixExecutionPlan([left], n=n, ordinary_tabmat=False),
        MatrixExecutionPlan([right], n=n, ordinary_tabmat=False),
    )


@pytest.mark.parametrize(
    "left_kind,right_kind", [("ssp", "ssp"), ("ssp", "scop"), ("scop", "ssp"), ("scop", "scop")]
)
@pytest.mark.parametrize(
    "bins,n,expected",
    [
        (128, 96, "histogram"),
        (256, 96, "rows"),
        (256, 512, "histogram"),
        (256, 2048, "histogram"),
        (256, 8192, "histogram"),
        (512, 2048, "rows"),
        (1024, 8192, "rows"),
    ],
)
def test_discrete_cross_dispatch_accounts_for_current_row_count(
    monkeypatch, left_kind, right_kind, bins, n, expected
):
    # Measured generic crossovers: setup matters for small grids, row gathering
    # for long chunks, and grid traffic for large supports. Near ties stay hist.
    left, _ = _group(left_kind, n=n, bins=bins, width=7, seed=10)
    right, _ = _group(right_kind, n=n, bins=bins, width=5, seed=11)
    calls = []
    for name, route in [
        ("_disc_disc_2d_hist", "histogram"),
        ("_support_support_raw_cross", "rows"),
    ]:
        original = getattr(algebra, name)

        def counted(*args, _original=original, _route=route, **kwargs):
            calls.append(_route)
            return _original(*args, **kwargs)

        monkeypatch.setattr(algebra, name, counted)
    profile = {}
    _plan(left, right, n).cross_moment(np.linspace(-1.0, 2.0, n), profile=profile)
    assert calls == [expected]
    assert profile.get("block_cross_disc_disc_hist_calls", 0) == (expected == "histogram")
    assert profile.get("block_cross_disc_disc_rows_calls", 0) == (expected == "rows")


@pytest.mark.parametrize("width,expected", [(4, "histogram"), (40, "rows")])
def test_discrete_cross_dispatch_accounts_for_basis_width(monkeypatch, width, expected):
    n = 512
    left, _ = _group("ssp", n=n, bins=256, width=width, seed=20)
    right, _ = _group("ssp", n=n, bins=256, width=width, seed=21)
    calls = []
    for name, route in [
        ("_disc_disc_2d_hist", "histogram"),
        ("_support_support_raw_cross", "rows"),
    ]:
        original = getattr(algebra, name)

        def counted(*args, _original=original, _route=route, **kwargs):
            calls.append(_route)
            return _original(*args, **kwargs)

        monkeypatch.setattr(algebra, name, counted)
    _plan(left, right, n).cross_moment(np.ones(n))
    assert calls == [expected]


@pytest.mark.parametrize(
    "left_kind,right_kind", [("ssp", "ssp"), ("ssp", "scop"), ("scop", "ssp"), ("scop", "scop")]
)
@pytest.mark.parametrize("n,bins", [(173, 257), (4096, 13)])
def test_signed_rectangular_discrete_cross_matches_stored_basis_oracle(
    left_kind, right_kind, n, bins
):
    left, x = _group(left_kind, n=n, bins=bins, width=7, seed=30)
    right, y = _group(right_kind, n=n, bins=bins + 4, width=5, seed=31)
    weights = np.random.default_rng(32).normal(size=n)
    weights[::7] = 0.0
    expected = x.T @ (weights[:, None] * y)
    # Absolute-product scale stays meaningful under cancellation and zero entries.
    scale = np.linalg.norm(np.abs(x).T @ (np.abs(weights[:, None] * y)), ord=np.inf)
    bound = 32 * np.finfo(float).eps * max(n, bins, 7) * scale
    actual = _plan(left, right, n).cross_moment(weights)
    assert actual.shape == (x.shape[1], y.shape[1])
    assert np.linalg.norm(actual - expected, ord=np.inf) <= bound
    reverse = _plan(right, left, n).cross_moment(weights)
    assert np.linalg.norm(reverse.T - expected, ord=np.inf) <= bound


def test_short_discrete_cross_expands_only_bounded_stored_support_panels(monkeypatch):
    n = 173
    left, _ = _group("ssp", n=n, bins=256, width=7, seed=40)
    right, _ = _group("ssp", n=n, bins=257, width=5, seed=41)
    budget = 9 * (7 + 5) * np.dtype(float).itemsize
    monkeypatch.setattr(algebra, "_MAX_CROSS_EXPANSION_BYTES", budget)
    panels = []
    original = algebra._expand_support_rows

    def recorded(support, indices):
        assert support is left.B_unique or support is right.B_unique
        panels.append((len(indices), support.shape[1]))
        return original(support, indices)

    monkeypatch.setattr(algebra, "_expand_support_rows", recorded)
    _plan(left, right, n).cross_moment(np.linspace(-2.0, 1.0, n))
    assert panels
    assert sum(rows for rows, width in panels if width == 7) == n
    assert sum(rows for rows, width in panels if width == 5) == n
    assert all(rows <= 9 for rows, _width in panels)


def test_histogram_cell_ceiling_still_overrides_favorable_compression(monkeypatch):
    n = 4096
    left, _ = _group("ssp", n=n, bins=13, width=7, seed=50)
    right, _ = _group("ssp", n=n, bins=17, width=5, seed=51)
    monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 13 * 17 - 1)
    profile = {}
    _plan(left, right, n).cross_moment(np.ones(n), profile=profile)
    assert profile["block_cross_disc_disc_rows_calls"] == 1
    assert profile.get("block_hist2d_builds", 0) == 0


@pytest.mark.parametrize(
    "left_dtype,right_dtype",
    [
        (np.int64, np.int64),
        (np.float32, np.float32),
        (np.float64, np.int64),
        (np.int64, np.float64),
        (np.float64, np.float32),
        (np.float32, np.float64),
    ],
)
@pytest.mark.parametrize("histogram_fits", [True, False])
def test_non_float64_support_retains_weighted_accumulation_precision(
    monkeypatch, left_dtype, right_dtype, histogram_fits
):
    n = 96
    indices = np.arange(n)
    support_left = np.arange(127 * 7, dtype=left_dtype).reshape(127, 7)
    support_right = np.arange(97 * 5, dtype=right_dtype).reshape(97, 5)
    left = DiscretizedSCOPGroupMatrix(support_left, indices)
    right = DiscretizedSCOPGroupMatrix(support_right, indices)
    weights = np.linspace(-1.0, 2.0, n)
    expected = support_left[indices].T @ (weights[:, None] * support_right[indices])
    scale = np.linalg.norm(
        np.abs(support_left[indices]).T @ np.abs(weights[:, None] * support_right[indices]),
        ord=np.inf,
    )
    if not histogram_fits:
        monkeypatch.setattr(algebra, "_MAX_DISC_DISC_HIST_CELLS", 1)
    profile = {}
    actual = _plan(left, right, n).cross_moment(weights, profile=profile)
    assert profile.get("block_cross_disc_disc_hist_calls", 0) == histogram_fits
    assert profile.get("block_cross_disc_disc_rows_calls", 0) == 0
    bound = 32 * np.finfo(float).eps * n * scale
    assert np.linalg.norm(actual - expected, ord=np.inf) <= bound


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("contiguous", [True, False])
def test_support_gather_uses_take_only_for_contiguous_storage(monkeypatch, dtype, contiguous):
    support = np.arange(31 * 14, dtype=dtype).reshape(31, 14)
    if not contiguous:
        support = support[:, ::2]
    indices = np.array([0, 7, 7, 12, 30], dtype=np.intp)
    expected = support[indices]
    calls = []
    original = np.take

    def recorded(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "take", recorded)
    actual = algebra._expand_support_rows(support, indices)
    assert len(calls) == int(contiguous)
    assert actual.dtype == support.dtype
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, support)


def test_noncontiguous_support_gather_does_not_copy_entire_support():
    support = np.ones((100_000, 14))[:, ::2]
    indices = np.arange(96, dtype=np.intp)
    tracemalloc.start()
    try:
        actual = algebra._expand_support_rows(support, indices)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Allow indexing metadata, but not a contiguous copy of the 5.6 MB support.
    assert peak <= actual.nbytes + 64_000
    np.testing.assert_array_equal(actual, np.ones((96, 7)))
