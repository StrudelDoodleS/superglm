from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import scipy.sparse as sp

import superglm
from superglm._group_matrix import CrossMatrixExecutionPlan
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.group_matrix import (
    CrossMatrixExecutionPlan as GroupMatrixCrossPlan,
)


def _dense_plans(
    *,
    n: int = 19,
) -> tuple[MatrixExecutionPlan, MatrixExecutionPlan, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5101)
    left_groups = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        DenseGroupMatrix(rng.normal(size=(n, 1))),
    ]
    right_groups = [
        DenseGroupMatrix(rng.normal(size=(n, 1))),
        DenseGroupMatrix(rng.normal(size=(n, 3))),
    ]
    left = MatrixExecutionPlan(left_groups, n=n, ordinary_tabmat=False)
    right = MatrixExecutionPlan(right_groups, n=n, ordinary_tabmat=False)
    left_dense = np.column_stack([group.toarray() for group in left_groups])
    right_dense = np.column_stack([group.toarray() for group in right_groups])
    return left, right, left_dense, right_dense


def test_cross_plan_is_available_only_from_internal_matrix_modules() -> None:
    assert GroupMatrixCrossPlan is CrossMatrixExecutionPlan
    assert not hasattr(superglm, "CrossMatrixExecutionPlan")


def test_cross_plan_rejects_mismatched_row_counts() -> None:
    left, _, _, _ = _dense_plans(n=9)
    right = MatrixExecutionPlan(
        [DenseGroupMatrix(np.ones((10, 2)))],
        n=10,
        ordinary_tabmat=False,
    )

    with pytest.raises(ValueError, match="row count"):
        CrossMatrixExecutionPlan(left, right)


@pytest.mark.parametrize(
    "weights",
    [
        np.ones(18),
        np.ones((19, 1)),
        np.concatenate((np.ones(18), [np.nan])),
        np.concatenate((np.ones(18), [np.inf])),
    ],
)
def test_cross_moment_rejects_invalid_weight_shape_or_values(weights: np.ndarray) -> None:
    left, right, _, _ = _dense_plans()
    plan = CrossMatrixExecutionPlan(left, right)

    with pytest.raises(ValueError, match="weights"):
        plan.cross_moment(weights)


def test_cross_moment_requires_signed_opt_in_for_negative_weights() -> None:
    left, right, left_dense, right_dense = _dense_plans()
    plan = CrossMatrixExecutionPlan(left, right)
    weights = np.linspace(-0.8, 1.2, left.n)

    with pytest.raises(ValueError, match="negative weights require signed=True"):
        plan.cross_moment(weights, signed=False)

    np.testing.assert_allclose(
        plan.cross_moment(weights),
        left_dense.T @ (weights[:, None] * right_dense),
        rtol=2e-14,
        atol=2e-14,
    )


def test_cross_plan_layout_is_immutable_and_pairs_are_constructed_once() -> None:
    left, right, _, _ = _dense_plans()
    plan = CrossMatrixExecutionPlan(left, right)
    pairs = plan.group_pairs

    assert isinstance(pairs, tuple)
    assert len(pairs) == len(left.group_spans) * len(right.group_spans)
    assert [(pair.left.index, pair.right.index) for pair in pairs] == [
        (left_index, right_index)
        for left_index in range(len(left.group_spans))
        for right_index in range(len(right.group_spans))
    ]
    with pytest.raises(AttributeError):
        plan.left = right
    with pytest.raises(AttributeError):
        plan.group_pairs = ()
    with pytest.raises(FrozenInstanceError):
        pairs[0].left = right.group_spans[0]

    plan.cross_moment(np.ones(left.n))
    assert plan.group_pairs is pairs


def test_dense_cross_moment_has_rectangular_global_shape_and_matches_oracle() -> None:
    left, right, left_dense, right_dense = _dense_plans()
    plan = CrossMatrixExecutionPlan(left, right)
    weights = np.linspace(0.2, 1.7, left.n)
    weights.setflags(write=False)

    actual = plan.cross_moment(weights, signed=False)

    assert plan.n == left.n == right.n
    assert plan.shape == (left.p, right.p)
    assert actual.shape == (left.p, right.p)
    np.testing.assert_allclose(
        actual,
        left_dense.T @ (weights[:, None] * right_dense),
        rtol=2e-14,
        atol=2e-14,
    )
    assert not weights.flags.writeable


_GROUP_KINDS = (
    "dense",
    "sparse",
    "categorical",
    "raw_spline",
    "ssp",
    "discrete_ssp",
    "raw_tensor",
    "discrete_tensor",
    "spline_categorical",
    "discrete_spline_categorical",
    "discrete_scop",
)


def _group_catalog(*, n: int = 48, seed: int = 5201) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sparse_values = rng.normal(size=(n, 3))
    sparse_values[rng.random(size=sparse_values.shape) < 0.62] = 0.0
    raw_spline_values = rng.normal(size=(n, 5))
    raw_spline_values[rng.random(size=raw_spline_values.shape) < 0.45] = 0.0
    ssp_basis = rng.normal(size=(n, 5))
    ssp_basis[np.abs(ssp_basis) < 0.45] = 0.0

    n_bins = 7
    bin_idx = np.resize(np.array([0, 1, 1, 4, 2, 6, 0, 4], dtype=np.intp), n)
    B_unique = rng.normal(size=(n_bins, 4))
    R_inv = rng.normal(size=(4, 3))

    raw_margin_1 = rng.normal(size=(n, 3))
    raw_margin_2 = rng.normal(size=(n, 2))
    raw_tensor_basis = np.einsum(
        "ni,nj->nij",
        raw_margin_1,
        raw_margin_2,
        optimize=True,
    ).reshape(n, 6)
    raw_tensor_basis[np.abs(raw_tensor_basis) < 0.35] = 0.0

    n_bins1, n_bins2 = 4, 3
    B1 = rng.normal(size=(n_bins1, 2))
    B2 = rng.normal(size=(n_bins2, 2))
    idx1 = np.resize(np.array([0, 1, 1, 3, 2, 0], dtype=np.intp), n)
    idx2 = np.resize(np.array([2, 1, 1, 0, 2, 0, 1], dtype=np.intp), n)
    pair_idx = idx1 * n_bins2 + idx2
    B_joint = np.einsum("ia,jb->ijab", B1, B2, optimize=True).reshape(
        n_bins1 * n_bins2,
        4,
    )

    spline_cat_rows = np.flatnonzero(np.arange(n) % 3 != 1)
    spline_cat_basis = rng.normal(size=(n, 5))
    spline_cat_basis[np.abs(spline_cat_basis) < 0.55] = 0.0
    discrete_cat_rows = np.flatnonzero(np.arange(n) % 4 != 2)

    return {
        "dense": DenseGroupMatrix(rng.normal(size=(n, 2))),
        "sparse": SparseGroupMatrix(sp.csr_matrix(sparse_values)),
        "categorical": CategoricalGroupMatrix(
            np.resize(np.array([-1, 0, 2, 1, 0, -1], dtype=np.intp), n),
            n_levels=3,
        ),
        "raw_spline": SparseGroupMatrix(sp.csr_matrix(raw_spline_values)),
        "ssp": SparseSSPGroupMatrix(
            sp.csr_matrix(ssp_basis),
            rng.normal(size=(5, 3)),
        ),
        "discrete_ssp": DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx),
        "raw_tensor": SparseSSPGroupMatrix(
            sp.csr_matrix(raw_tensor_basis),
            rng.normal(size=(6, 4)),
        ),
        "discrete_tensor": DiscretizedTensorGroupMatrix(
            B1,
            B2,
            idx1,
            idx2,
            B_joint,
            rng.normal(size=(4, 3)),
            pair_idx,
            tensor_id=5201,
        ),
        "spline_categorical": SplineCategoricalGroupMatrix(
            sp.csr_matrix(spline_cat_basis),
            rng.normal(size=(5, 3)),
            spline_cat_rows,
        ),
        "discrete_spline_categorical": DiscretizedSplineCategoricalGroupMatrix(
            B_unique,
            R_inv[:, :2],
            bin_idx,
            discrete_cat_rows,
            n_rows=n,
        ),
        "discrete_scop": DiscretizedSCOPGroupMatrix(
            rng.normal(size=(n_bins, 3)),
            bin_idx,
        ),
    }


def _single_group_plan(group: object, *, n: int) -> MatrixExecutionPlan:
    return MatrixExecutionPlan([group], n=n, ordinary_tabmat=False)


@pytest.mark.parametrize("left_kind", _GROUP_KINDS)
@pytest.mark.parametrize("right_kind", _GROUP_KINDS)
def test_every_group_pair_and_orientation_matches_signed_dense_oracle(
    left_kind: str,
    right_kind: str,
) -> None:
    n = 48
    groups = _group_catalog(n=n)
    left_group = groups[left_kind]
    right_group = groups[right_kind]
    left = _single_group_plan(left_group, n=n)
    right = _single_group_plan(right_group, n=n)
    weights = np.linspace(-1.1, 1.3, n)
    weights[::7] = 0.0
    profile: dict[str, float | int] = {}

    actual = CrossMatrixExecutionPlan(left, right).cross_moment(
        weights,
        profile=profile,
    )
    left_dense = left_group.toarray()
    right_dense = right_group.toarray()

    np.testing.assert_allclose(
        actual,
        left_dense.T @ (weights[:, None] * right_dense),
        rtol=3e-13,
        atol=3e-12,
    )
    category = f"{type(left_group).__name__}__{type(right_group).__name__}"
    assert profile[f"cross_pair_{category}_calls"] == 1
    assert profile[f"cross_pair_{category}_s"] >= 0.0
    assert profile["cross_route_specialized_calls"] + profile["cross_route_fallback_calls"] == 1


def test_discrete_histogram_is_reused_for_repeated_pairs_within_one_channel() -> None:
    n = 96
    groups = _group_catalog(n=n, seed=5202)
    left_group = groups["discrete_ssp"]
    right_group = DiscretizedSSPGroupMatrix(
        np.flipud(left_group.B_unique.copy()),
        np.eye(left_group.B_unique.shape[1]),
        np.roll(left_group.bin_idx.copy(), 3),
    )
    left = MatrixExecutionPlan([left_group, left_group], n=n, ordinary_tabmat=False)
    right = MatrixExecutionPlan([right_group, right_group], n=n, ordinary_tabmat=False)
    weights = np.linspace(-0.9, 1.1, n)
    profile: dict[str, float | int] = {}

    actual = CrossMatrixExecutionPlan(left, right).cross_moment(weights, profile=profile)
    left_dense = np.column_stack((left_group.toarray(), left_group.toarray()))
    right_dense = np.column_stack((right_group.toarray(), right_group.toarray()))

    np.testing.assert_allclose(
        actual,
        left_dense.T @ (weights[:, None] * right_dense),
        rtol=3e-13,
        atol=3e-12,
    )
    assert profile["block_hist2d_builds"] == 1
    assert profile["block_hist2d_reuses"] == 3
    assert profile["cross_route_specialized_calls"] == 4
    assert profile["cross_route_fallback_calls"] == 0
