from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm._group_matrix._cross_matrix_execution import CrossMatrixExecutionPlan
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)


def _discrete_ssp(
    n: int,
    *,
    seed: int,
    shift: int = 0,
) -> DiscretizedSSPGroupMatrix:
    rng = np.random.default_rng(seed)
    n_bins = 17
    pattern = np.array([0, 3, 3, 8, 1, 16, 5, 8, 0, 12], dtype=np.intp)
    bin_idx = np.roll(np.resize(pattern, n), shift)
    return DiscretizedSSPGroupMatrix(
        rng.normal(size=(n_bins, 5)),
        rng.normal(size=(5, 3)),
        bin_idx,
    )


def _tensor(n: int, *, seed: int) -> DiscretizedTensorGroupMatrix:
    rng = np.random.default_rng(seed)
    n_bins1, n_bins2 = 6, 5
    B1 = rng.normal(size=(n_bins1, 2))
    B2 = rng.normal(size=(n_bins2, 2))
    idx1 = np.resize(np.array([0, 2, 2, 5, 1, 4, 0], dtype=np.intp), n)
    idx2 = np.resize(np.array([4, 1, 1, 0, 3, 2], dtype=np.intp), n)
    pair_idx = idx1 * n_bins2 + idx2
    B_joint = np.einsum("ia,jb->ijab", B1, B2, optimize=True).reshape(
        n_bins1 * n_bins2,
        4,
    )
    return DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        rng.normal(size=(4, 3)),
        pair_idx,
        tensor_id=5301,
    )


def _support_groups(n: int) -> tuple[list[object], list[object]]:
    rng = np.random.default_rng(5302)
    left_disc = _discrete_ssp(n, seed=5303)
    right_disc = _discrete_ssp(n, seed=5304, shift=3)
    tensor = _tensor(n, seed=5305)

    raw_basis = rng.normal(size=(n, 6))
    raw_basis[np.abs(raw_basis) < 0.52] = 0.0
    raw_ssp = SparseSSPGroupMatrix(
        sp.csr_matrix(raw_basis),
        rng.normal(size=(6, 3)),
    )

    spline_cat_basis = rng.normal(size=(n, 5))
    spline_cat_basis[np.abs(spline_cat_basis) < 0.58] = 0.0
    spline_cat = SplineCategoricalGroupMatrix(
        sp.csr_matrix(spline_cat_basis),
        rng.normal(size=(5, 3)),
        np.flatnonzero(np.arange(n) % 3 != 1),
    )
    discrete_spline_cat = DiscretizedSplineCategoricalGroupMatrix(
        left_disc.B_unique,
        left_disc.R_inv[:, :2],
        left_disc.bin_idx,
        np.flatnonzero(np.arange(n) % 4 != 2),
        n_rows=n,
    )
    return (
        [left_disc, tensor, raw_ssp, spline_cat, discrete_spline_cat],
        [right_disc, raw_ssp, discrete_spline_cat, spline_cat],
    )


def _plan(groups: list[object], *, n: int) -> MatrixExecutionPlan:
    return MatrixExecutionPlan(groups, n=n, ordinary_tabmat=False)


def _forbid_materialization(*_args, **_kwargs):
    raise AssertionError("compressed group was materialized at observation scale")


def test_compressed_cross_routes_never_call_toarray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 180
    left_groups, right_groups = _support_groups(n)
    left_dense = np.column_stack([group.toarray() for group in left_groups])
    right_dense = np.column_stack([group.toarray() for group in right_groups])
    weights = np.linspace(-1.0, 1.4, n)
    weights[::9] = 0.0

    for group_type in (
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
    ):
        monkeypatch.setattr(group_type, "toarray", _forbid_materialization)

    actual = CrossMatrixExecutionPlan(
        _plan(left_groups, n=n),
        _plan(right_groups, n=n),
    ).cross_moment(weights)

    np.testing.assert_allclose(
        actual,
        left_dense.T @ (weights[:, None] * right_dense),
        rtol=4e-13,
        atol=4e-12,
    )


def test_repeated_discrete_pairs_construct_one_histogram_per_weight_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 240
    left_group = _discrete_ssp(n, seed=5310)
    right_group = _discrete_ssp(n, seed=5311, shift=2)
    left = _plan([left_group, left_group, left_group], n=n)
    right = _plan([right_group, right_group], n=n)
    plan = CrossMatrixExecutionPlan(left, right)
    original = algebra._disc_disc_2d_hist
    calls = 0

    def counted_histogram(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(algebra, "_disc_disc_2d_hist", counted_histogram)

    first_profile: dict[str, float | int] = {}
    plan.cross_moment(np.linspace(-0.7, 1.3, n), profile=first_profile)
    assert calls == 1
    assert first_profile["block_hist2d_builds"] == 1
    assert first_profile["block_hist2d_reuses"] == 5

    second_profile: dict[str, float | int] = {}
    plan.cross_moment(np.linspace(0.1, 1.9, n), profile=second_profile)
    assert calls == 2
    assert second_profile["block_hist2d_builds"] == 1
    assert second_profile["block_hist2d_reuses"] == 5


def _retained_cross_allocation(n: int) -> tuple[int, set[str]]:
    left_group = _discrete_ssp(n, seed=5320)
    right_group = _discrete_ssp(n, seed=5321, shift=1)
    left = _plan([left_group, left_group], n=n)
    right = _plan([right_group, right_group], n=n)
    weights = np.linspace(-0.8, 1.2, n)
    gc.collect()
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    plan = CrossMatrixExecutionPlan(left, right)
    result = plan.cross_moment(weights)
    gc.collect()
    retained, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.shape == (left.p, right.p)
    assert result.nbytes == left.p * right.p * np.dtype(np.float64).itemsize
    assert not any(
        isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == n
        for value in vars(plan).values()
    )
    return max(0, retained - baseline), set(vars(plan))


def test_retained_plan_allocation_does_not_scale_with_observation_count() -> None:
    _retained_cross_allocation(64)  # warm imported kernels and allocator pools
    small_bytes, small_fields = _retained_cross_allocation(512)
    large_bytes, large_fields = _retained_cross_allocation(120_000)

    assert (
        small_fields
        == large_fields
        == {
            "left",
            "right",
            "n",
            "shape",
            "group_pairs",
            "_pair_entries",
            "_layout_frozen",
        }
    )
    assert large_bytes <= small_bytes + 64 * 1024


def test_self_cross_plan_matches_existing_signed_symmetric_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(5330)
    n = 160
    discrete = _discrete_ssp(n, seed=5331)
    groups = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        CategoricalGroupMatrix(
            np.resize(np.array([-1, 0, 2, 1, 0], dtype=np.intp), n),
            n_levels=3,
        ),
        discrete,
    ]
    plan = _plan(groups, n=n)
    weights = np.linspace(-1.2, 1.5, n)
    weights[::11] = 0.0
    cross_method: Callable[..., np.ndarray] = CrossMatrixExecutionPlan.cross_moment

    def forbidden_cross(*_args, **_kwargs):
        raise AssertionError("symmetric moments must not route through the rectangular plan")

    monkeypatch.setattr(CrossMatrixExecutionPlan, "cross_moment", forbidden_cross)
    symmetric = plan.moments(weights, signed=True).gram
    monkeypatch.setattr(CrossMatrixExecutionPlan, "cross_moment", cross_method)
    rectangular = CrossMatrixExecutionPlan(plan, plan).cross_moment(weights)

    np.testing.assert_allclose(rectangular, symmetric, rtol=3e-13, atol=3e-12)
    np.testing.assert_allclose(symmetric, symmetric.T, rtol=0.0, atol=3e-12)
