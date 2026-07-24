"""Structured sufficient-statistic correctness and allocation guards."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_algebra as group_algebra
import superglm.solvers.structured as structured_module
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.solvers.structured import (
    build_scalar_structured_system,
    select_structured_group,
)
from superglm.types import GroupSlice, LinearConstraintSet


def _groups(group_matrices: list[GroupMatrix]) -> list[GroupSlice]:
    groups: list[GroupSlice] = []
    start = 0
    for index, gm in enumerate(group_matrices):
        end = start + gm.shape[1]
        groups.append(
            GroupSlice(
                name=f"group_{index}",
                start=start,
                end=end,
                penalized=isinstance(gm, RandomEffectGroupMatrix | SparseSSPGroupMatrix),
            )
        )
        start = end
    return groups


def _small_matrix_cases(n: int) -> list[tuple[str, GroupMatrix]]:
    rng = np.random.default_rng(843)
    numeric = DenseGroupMatrix(rng.normal(size=(n, 2)))
    categorical = CategoricalGroupMatrix(
        np.resize(np.array([-1, 0, 1, 2], dtype=np.intp), n),
        n_levels=3,
    )

    sparse_basis = sp.csr_matrix(rng.normal(size=(n, 4)))
    sparse_basis.data[np.abs(sparse_basis.data) < 0.45] = 0.0
    sparse_basis.eliminate_zeros()
    sparse_ssp = SparseSSPGroupMatrix(sparse_basis, np.eye(4))

    B_unique = rng.normal(size=(5, 3))
    bin_idx = np.arange(n, dtype=np.intp) % 5
    discrete_ssp = DiscretizedSSPGroupMatrix(B_unique, np.eye(3), bin_idx)

    B1 = rng.normal(size=(3, 2))
    B2 = rng.normal(size=(2, 2))
    idx1 = np.arange(n, dtype=np.intp) % 3
    idx2 = (np.arange(n, dtype=np.intp) // 2) % 2
    pair_idx = idx1 * 2 + idx2
    B_joint = np.vstack([np.kron(B1[i], B2[j]) for i in range(3) for j in range(2)])
    tensor = DiscretizedTensorGroupMatrix(
        B1,
        B2,
        idx1,
        idx2,
        B_joint,
        np.eye(4),
        pair_idx,
        tensor_id=91,
    )
    second_random_effect = RandomEffectGroupMatrix(
        np.arange(n, dtype=np.intp) % 3,
        n_levels=3,
    )
    return [
        ("numeric", numeric),
        ("categorical", categorical),
        ("sparse_ssp", sparse_ssp),
        ("discrete_ssp", discrete_ssp),
        ("tensor", tensor),
        ("second_random_effect", second_random_effect),
    ]


@pytest.mark.parametrize("weight_sign", ["positive", "arbitrary"])
@pytest.mark.parametrize(
    ("case_name", "small_factory"),
    [
        pytest.param(
            name,
            lambda n, selected=matrix: selected.row_subset(np.arange(n)),
            id=name,
        )
        for name, matrix in _small_matrix_cases(18)
    ],
)
def test_structured_summary_matches_materialized_weighted_gram(
    case_name: str,
    small_factory: Callable[[int], GroupMatrix],
    weight_sign: str,
):
    del case_name
    n = 18
    dominant = RandomEffectGroupMatrix(np.arange(n, dtype=np.intp) % 5, n_levels=5)
    small = small_factory(n)
    group_matrices = [small, dominant]
    groups = _groups(group_matrices)
    rng = np.random.default_rng(120)
    W = rng.uniform(0.2, 2.0, size=n)
    if weight_sign == "arbitrary":
        W[::3] *= -1.0
    Wz = rng.normal(size=n)
    X = np.hstack([gm.toarray() for gm in group_matrices])
    XtWX = X.T @ (W[:, None] * X)
    XtW = X.T @ W
    XtWz = X.T @ Wz

    system = build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index=1,
    )
    small_indices = system.operator.small_indices
    structured_indices = system.operator.structured_indices

    np.testing.assert_allclose(
        system.operator.A,
        XtWX[np.ix_(small_indices, small_indices)],
    )
    np.testing.assert_allclose(
        system.operator.C,
        XtWX[np.ix_(structured_indices, small_indices)],
    )
    np.testing.assert_allclose(
        system.operator.d,
        np.diag(XtWX)[structured_indices],
    )
    np.testing.assert_allclose(system.xtw_small, XtW[small_indices])
    np.testing.assert_allclose(system.xtw_structured, XtW[structured_indices])
    np.testing.assert_allclose(system.xtwz_small, XtWz[small_indices])
    np.testing.assert_allclose(system.xtwz_structured, XtWz[structured_indices])
    np.testing.assert_allclose(system.sum_w, np.sum(W))
    np.testing.assert_allclose(system.sum_wz, np.sum(Wz))


def test_select_structured_group_chooses_largest_random_effect_and_reports_ineligibility():
    n = 12
    group_matrices: list[GroupMatrix] = [
        RandomEffectGroupMatrix(np.arange(n) % 3, n_levels=3),
        DenseGroupMatrix(np.arange(n, dtype=np.float64)),
        RandomEffectGroupMatrix(np.arange(n) % 7, n_levels=7),
    ]
    groups = _groups(group_matrices)

    selected = select_structured_group(group_matrices, groups, mode="structured")

    assert selected.group_index == 2
    assert selected.group_name == groups[2].name
    assert selected.fallback_reason is None

    no_random_effect = [DenseGroupMatrix(np.arange(n, dtype=np.float64))]
    no_random_groups = _groups(no_random_effect)
    auto = select_structured_group(no_random_effect, no_random_groups, mode="auto")
    assert auto.group_index is None
    assert "RandomEffect" in auto.fallback_reason
    with pytest.raises(ValueError, match="RandomEffect"):
        select_structured_group(no_random_effect, no_random_groups, mode="structured")


def test_select_structured_group_rejects_constraint_geometry():
    n = 9
    group_matrices: list[GroupMatrix] = [
        DenseGroupMatrix(np.arange(n, dtype=np.float64)),
        RandomEffectGroupMatrix(np.arange(n) % 3, n_levels=3),
    ]
    groups = _groups(group_matrices)
    groups[0].constraints = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1))

    auto = select_structured_group(group_matrices, groups, mode="auto")

    assert auto.group_index is None
    assert "constraint" in auto.fallback_reason.lower()
    with pytest.raises(ValueError, match="constraint"):
        select_structured_group(group_matrices, groups, mode="structured")


def test_structured_summary_preserves_global_layout_with_multiple_small_groups():
    n = 18
    cases = dict(_small_matrix_cases(n))
    dominant = RandomEffectGroupMatrix(np.arange(n, dtype=np.intp) % 5, n_levels=5)
    group_matrices: list[GroupMatrix] = [
        cases["numeric"],
        dominant,
        cases["categorical"],
        cases["discrete_ssp"],
        cases["second_random_effect"],
    ]
    groups = _groups(group_matrices)
    rng = np.random.default_rng(340)
    W = rng.uniform(0.3, 1.7, size=n)
    Wz = rng.normal(size=n)
    X = np.hstack([matrix.toarray() for matrix in group_matrices])
    reference_gram = X.T @ (W[:, None] * X)

    system = build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index=1,
    )

    np.testing.assert_allclose(
        system.operator.A,
        reference_gram[np.ix_(system.operator.small_indices, system.operator.small_indices)],
    )
    np.testing.assert_allclose(
        system.operator.C,
        reference_gram[np.ix_(system.operator.structured_indices, system.operator.small_indices)],
    )


def test_structured_summary_supports_random_effect_only_model():
    dominant = RandomEffectGroupMatrix(
        np.array([0, 1, 1, 2, 0], dtype=np.intp),
        n_levels=3,
    )
    groups = _groups([dominant])
    W = np.array([0.5, 1.0, 1.5, 0.25, 2.0])
    Wz = np.array([1.0, -0.5, 0.75, 2.0, -1.0])

    system = build_scalar_structured_system(
        [dominant],
        groups,
        W,
        Wz,
        dominant_group_index=0,
    )

    assert system.operator.A.shape == (0, 0)
    assert system.operator.C.shape == (3, 0)
    np.testing.assert_allclose(
        system.operator.d,
        np.bincount(dominant.codes, weights=W, minlength=3),
    )
    np.testing.assert_allclose(
        system.xtwz_structured,
        np.bincount(dominant.codes, weights=Wz, minlength=3),
    )


class _GuardedRandomEffect(RandomEffectGroupMatrix):
    def gram(self, W):
        raise AssertionError("dominant random-effect gram must not be materialized")

    def toarray(self):
        raise AssertionError("dominant random-effect design must not be materialized")


def test_tabmat_selected_small_sandwich_avoids_dominant_materialization(monkeypatch):
    n_levels = 101
    n = n_levels * 2
    rng = np.random.default_rng(66)
    small = DenseGroupMatrix(rng.normal(size=(n, 2)))
    dominant = _GuardedRandomEffect(np.arange(n, dtype=np.intp) % n_levels, n_levels)
    group_matrices: list[GroupMatrix] = [dominant, small]
    groups = _groups(group_matrices)
    design = DesignMatrix(group_matrices, n=n, p=n_levels + 2)
    split = design.tabmat_split
    calls: list[np.ndarray | None] = []
    original_sandwich = type(split).sandwich

    def spy_sandwich(self, d, rows=None, cols=None):
        calls.append(None if cols is None else np.asarray(cols).copy())
        return original_sandwich(self, d, rows=rows, cols=cols)

    monkeypatch.setattr(type(split), "sandwich", spy_sandwich)
    W = rng.uniform(0.2, 1.8, size=n)
    Wz = rng.normal(size=n)

    system = build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index=0,
        tabmat_split=split,
    )

    assert any(
        cols is not None and np.array_equal(cols, system.operator.small_indices) for cols in calls
    )
    assert system.operator.A.shape == (2, 2)
    assert system.operator.C.shape == (n_levels, 2)


def test_native_ssp_aggregation_avoids_toarray_when_tabmat_is_ineligible(monkeypatch):
    n = 30
    rng = np.random.default_rng(92)
    B = sp.csr_matrix(rng.normal(size=(n, 4)))
    small = SparseSSPGroupMatrix(B, np.eye(4))
    dominant = _GuardedRandomEffect(np.arange(n, dtype=np.intp) % 6, n_levels=6)
    group_matrices: list[GroupMatrix] = [small, dominant]
    groups = _groups(group_matrices)
    reference = dominant.rmatvec(W := rng.uniform(0.5, 1.5, size=n))
    assert DesignMatrix(group_matrices, n=n, p=10).tabmat_split is None

    def fail_toarray(self):
        raise AssertionError("SSP cross aggregation must not call toarray")

    monkeypatch.setattr(SparseSSPGroupMatrix, "toarray", fail_toarray)
    system = build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        rng.normal(size=n),
        dominant_group_index=1,
    )

    np.testing.assert_allclose(system.xtw_structured, reference)
    assert system.operator.C.shape == (6, 4)


def test_large_dominant_builder_never_requests_full_p_by_p_storage(monkeypatch):
    n = 600
    n_levels = 5_000
    rng = np.random.default_rng(18)
    small = DenseGroupMatrix(rng.normal(size=(n, 2)))
    dominant = _GuardedRandomEffect(np.arange(n, dtype=np.intp), n_levels)
    group_matrices: list[GroupMatrix] = [small, dominant]
    groups = _groups(group_matrices)
    W = rng.uniform(0.4, 1.6, size=n)
    Wz = rng.normal(size=n)
    p = n_levels + 2

    build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index=1,
    )

    def fail_full_block(*args, **kwargs):
        raise AssertionError("full block Gram builder must not be used")

    monkeypatch.setattr(group_algebra, "_block_xtwx", fail_full_block)
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if shape == (p, p):
            raise AssertionError("full p x p allocation requested")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(structured_module.np, "zeros", guarded_zeros)

    system = build_scalar_structured_system(
        group_matrices,
        groups,
        W,
        Wz,
        dominant_group_index=1,
    )

    assert system.operator.A.shape == (2, 2)
    assert system.operator.C.shape == (n_levels, 2)
