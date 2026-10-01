"""Structured sufficient-statistic correctness and allocation guards."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_algebra as group_algebra
import superglm.solvers._structured.moments as structured_moments
from superglm.group_matrix import (
    DenseGroupMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.solvers.structured import (
    build_block_structured_system,
    build_nested_structured_layout,
    build_nested_structured_system,
    select_structured_group,
    structured_design_matvec,
    structured_design_rmatvec,
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


def _factor_smooth_matrix(
    n: int,
    *,
    levels: int,
    k: int = 5,
) -> FactorSmoothGroupMatrix:
    x = np.linspace(-1.0, 1.0, n)
    basis = np.column_stack([x**power for power in range(k)])
    return FactorSmoothGroupMatrix(
        sp.csr_matrix(basis),
        np.arange(n, dtype=np.intp) % levels,
        levels,
        natural_map=np.eye(k),
        levels=tuple(f"level-{index}" for index in range(levels)),
        repeated_penalty_components=(("wiggle", np.eye(k)),),
    )


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


def test_structured_selection_rejects_multiple_factor_smooths_before_assembly():
    matrices: list[GroupMatrix] = [
        _factor_smooth_matrix(80, levels=8),
        _factor_smooth_matrix(80, levels=6),
    ]
    groups = _groups(matrices)

    auto = select_structured_group(matrices, groups, mode="auto")
    assert auto.group_index is None
    assert "at most one FactorSmooth" in auto.fallback_reason
    assert groups[0].name in auto.fallback_reason
    assert groups[1].name in auto.fallback_reason

    with pytest.raises(ValueError, match="at most one FactorSmooth"):
        select_structured_group(matrices, groups, mode="structured")


def test_factor_smooth_is_dominant_candidate_even_when_random_effect_is_wider():
    matrices: list[GroupMatrix] = [
        RandomEffectGroupMatrix(np.arange(300) % 50, n_levels=50),
        _factor_smooth_matrix(300, levels=8),
    ]
    groups = _groups(matrices)

    selected = select_structured_group(matrices, groups, mode="structured")

    assert selected.group_index == 1
    assert selected.group_name == groups[1].name


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


class _GuardedRandomEffect(RandomEffectGroupMatrix):
    def gram(self, W):
        raise AssertionError("dominant random-effect gram must not be materialized")

    def toarray(self):
        raise AssertionError("dominant random-effect design must not be materialized")


def test_cached_dense_small_layout_fuses_design_vector_products(monkeypatch):
    rng = np.random.default_rng(918)
    n = 240
    n_levels = 61
    small = [
        DenseGroupMatrix(rng.normal(size=n)),
        DenseGroupMatrix(rng.normal(size=(n, 2))),
    ]
    dominant = RandomEffectGroupMatrix(np.arange(n) % n_levels, n_levels)
    matrices: list[GroupMatrix] = [*small, dominant]
    groups = _groups(matrices)
    layout = build_nested_structured_layout(matrices, groups, chain_group_indices=(2,))
    dense = np.column_stack([matrix.toarray() for matrix in matrices])
    beta = rng.normal(size=dense.shape[1])
    rows = rng.normal(size=n)

    def fail_separate_dense_dispatch(*_args, **_kwargs):
        raise AssertionError("cached dense-small vector product was not used")

    monkeypatch.setattr(DenseGroupMatrix, "matvec", fail_separate_dense_dispatch)
    monkeypatch.setattr(DenseGroupMatrix, "rmatvec", fail_separate_dense_dispatch)

    np.testing.assert_allclose(
        structured_design_matvec(layout, matrices, beta),
        dense @ beta,
    )
    np.testing.assert_allclose(
        structured_design_rmatvec(layout, matrices, rows),
        dense.T @ rows,
    )


def test_discrete_factor_smooth_shared_bin_cross_avoids_spline_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(429)
    n = 90
    n_levels = 6
    block_size = 4
    n_bins = 9
    bin_idx = np.arange(n, dtype=np.intp) % n_bins
    dominant = FactorSmoothGroupMatrix(
        rng.normal(size=(n_bins, 5)),
        (np.arange(n, dtype=np.intp) * 5 + 2) % n_levels,
        n_levels,
        natural_map=rng.normal(size=(5, block_size)),
        levels=tuple(f"level-{level}" for level in range(n_levels)),
        repeated_penalty_components=(("wiggle", np.eye(block_size)),),
        factor_basis="sz",
        bin_idx=bin_idx,
    )
    dense = DenseGroupMatrix(rng.normal(size=(n, 4)))
    spline = DiscretizedSSPGroupMatrix(
        rng.normal(size=(n_bins, 5)),
        rng.normal(size=(5, 3)),
        bin_idx.copy(),
    )
    matrices: list[GroupMatrix] = [dense, spline, dominant]
    groups = _groups(matrices)

    def fail_toarray(_self):
        raise AssertionError("optimized cross must not materialize observation rows")

    monkeypatch.setattr(DiscretizedSSPGroupMatrix, "toarray", fail_toarray)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", fail_toarray)

    system = build_block_structured_system(
        matrices,
        groups,
        rng.uniform(0.3, 1.8, size=n),
        rng.normal(size=n),
        dominant_group_index=2,
    )

    assert system.operator.C.shape == (n_levels, block_size, 7)


def test_large_dominant_builder_never_requests_full_p_by_p_storage(monkeypatch):
    """A lone random effect's chain of one never forms its level block or any p x p array."""
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
    layout = build_nested_structured_layout(group_matrices, groups, chain_group_indices=(1,))

    def fail_full_block(*args, **kwargs):
        raise AssertionError("full block Gram builder must not be used")

    monkeypatch.setattr(group_algebra, "_block_xtwx", fail_full_block)
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if shape == (p, p):
            raise AssertionError("full p x p allocation requested")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(structured_moments.np, "zeros", guarded_zeros)

    system = build_nested_structured_system(group_matrices, groups, W, Wz, layout=layout)

    assert system.operator.A.shape == (2, 2)
    assert system.operator.leaf.cross.shape == (n_levels, 2)
