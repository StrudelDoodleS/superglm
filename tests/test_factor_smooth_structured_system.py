"""Structured sufficient statistics for dominant factor smooth blocks."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
)
from superglm.solvers.structured import (
    BlockStructuredSystem,
    SumToZeroBlockStructuredSystem,
    build_block_structured_system,
    build_structured_system,
    get_block_structured_layout,
    materialize_compact_operator,
    select_structured_group,
    structured_design_matvec,
    structured_design_rmatvec,
)
from superglm.types import GroupSlice


def _dominant(
    *,
    discrete: bool,
    n: int = 48,
    factor_basis: str = "fs",
) -> FactorSmoothGroupMatrix:
    x_support = np.linspace(-1.0, 1.0, 9)
    basis_support = np.column_stack(
        [
            np.ones_like(x_support),
            x_support,
            x_support**2,
            x_support**3,
        ]
    )
    bin_idx = np.arange(n, dtype=np.intp) % len(x_support)
    codes = (np.arange(n, dtype=np.intp) * 5 + 1) % 6
    transform = np.array(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 0.8, -0.1, 0.0],
            [0.0, 0.0, 1.1, 0.15],
            [0.1, 0.0, 0.0, 0.9],
        ]
    )
    components = (("wiggle", np.diag([1.0, 1.0, 0.0, 0.0])),)
    kwargs = dict(
        codes=codes,
        n_levels=6,
        natural_map=transform,
        levels=tuple(f"level-{index}" for index in range(6)),
        repeated_penalty_components=components,
        factor_basis=factor_basis,
    )
    if discrete:
        return FactorSmoothGroupMatrix(
            basis_support,
            bin_idx=bin_idx,
            **kwargs,
        )
    return FactorSmoothGroupMatrix(
        sp.csr_matrix(basis_support[bin_idx]),
        **kwargs,
    )


def _design(*, discrete: bool, factor_basis: str = "fs"):
    rng = np.random.default_rng(417)
    dominant = _dominant(discrete=discrete, factor_basis=factor_basis)
    n = dominant.shape[0]
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        RandomEffectGroupMatrix(np.arange(n, dtype=np.intp) % 3, n_levels=3),
        CategoricalGroupMatrix((np.arange(n, dtype=np.intp) % 4) - 1, n_levels=3),
        SparseGroupMatrix(sp.random(n, 2, density=0.4, random_state=13, format="csr")),
        dominant,
    ]
    groups = []
    start = 0
    for index, matrix in enumerate(matrices):
        groups.append(GroupSlice(name=f"group-{index}", start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n, start)
    return rng, dm, groups, len(matrices) - 1


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_factor_smooth_structured_moments_match_dense_reference(
    discrete: bool,
    signed: bool,
) -> None:
    rng, dm, groups, dominant_index = _design(discrete=discrete)
    W = rng.uniform(0.2, 1.7, size=dm.n)
    if signed:
        W[::7] *= -0.35
    Wz = rng.normal(size=dm.n)

    system = build_block_structured_system(
        list(dm.group_matrices),
        groups,
        W,
        Wz,
        dominant_group_index=dominant_index,
    )

    reference = dm.toarray()
    gram = reference.T @ (W[:, None] * reference)
    xtw = reference.T @ W
    xtwz = reference.T @ Wz
    np.testing.assert_allclose(materialize_compact_operator(system.operator), gram, atol=2e-11)
    np.testing.assert_allclose(system.xtw_small, xtw[system.operator.small_indices])
    np.testing.assert_allclose(
        system.xtw_structured.ravel(),
        xtw[system.operator.structured_indices.ravel()],
    )
    np.testing.assert_allclose(system.xtwz_small, xtwz[system.operator.small_indices])
    np.testing.assert_allclose(
        system.xtwz_structured.ravel(),
        xtwz[system.operator.structured_indices.ravel()],
    )
    np.testing.assert_allclose(system.sum_w, np.sum(W))
    np.testing.assert_allclose(system.sum_wz, np.sum(Wz))


@pytest.mark.parametrize("discrete", [False, True])
def test_generic_structured_builder_dispatches_factor_smooth(discrete: bool) -> None:
    rng, dm, groups, dominant_index = _design(discrete=discrete)
    W = rng.uniform(0.3, 1.6, size=dm.n)
    Wz = rng.normal(size=dm.n)

    system = build_structured_system(
        list(dm.group_matrices),
        groups,
        W,
        Wz,
        dominant_group_index=dominant_index,
    )

    assert isinstance(system, BlockStructuredSystem)


@pytest.mark.parametrize("discrete", [False, True])
def test_sum_to_zero_structured_system_keeps_raw_blocks_and_public_moments(
    discrete: bool,
) -> None:
    rng, dm, groups, dominant_index = _design(
        discrete=discrete,
        factor_basis="sz",
    )
    W = rng.uniform(0.3, 1.6, size=dm.n)
    Wz = rng.normal(size=dm.n)

    system = build_structured_system(
        list(dm.group_matrices),
        groups,
        W,
        Wz,
        dominant_group_index=dominant_index,
    )
    reference = dm.toarray()
    gram = reference.T @ (W[:, None] * reference)
    xtw = reference.T @ W
    xtwz = reference.T @ Wz

    assert isinstance(system, SumToZeroBlockStructuredSystem)
    assert system.operator.D.shape == (6, 4, 4)
    assert system.raw_xtw_structured.shape == (6, 4)
    assert system.raw_xtwz_structured.shape == (6, 4)
    np.testing.assert_allclose(materialize_compact_operator(system.operator), gram, atol=2e-11)
    np.testing.assert_allclose(
        system.xtw_structured.ravel(),
        xtw[system.operator.structured_indices.ravel()],
    )
    np.testing.assert_allclose(
        system.xtwz_structured.ravel(),
        xtwz[system.operator.structured_indices.ravel()],
    )


def test_structured_selection_prefers_wider_factor_smooth_over_secondary_random_effect() -> None:
    _rng, dm, groups, dominant_index = _design(discrete=False)

    selection = select_structured_group(
        list(dm.group_matrices),
        groups,
        mode="structured",
    )

    assert selection.group_index == dominant_index
    assert selection.group_name == groups[dominant_index].name


def test_block_layout_is_reused_and_design_products_preserve_global_order() -> None:
    rng, dm, groups, dominant_index = _design(discrete=False)
    layout = get_block_structured_layout(dm, groups, dominant_group_index=dominant_index)
    again = get_block_structured_layout(dm, groups, dominant_group_index=dominant_index)
    beta = rng.normal(size=dm.p)
    rows = rng.normal(size=dm.n)

    assert again is layout
    np.testing.assert_allclose(
        structured_design_matvec(layout, dm.group_matrices, beta),
        dm.toarray() @ beta,
    )
    np.testing.assert_allclose(
        structured_design_rmatvec(layout, dm.group_matrices, rows),
        dm.toarray().T @ rows,
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_structured_moments_do_not_materialize_dominant_or_full_tabmat(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    rng, dm, groups, dominant_index = _design(discrete=discrete)
    W = rng.uniform(0.2, 1.4, size=dm.n)
    Wz = rng.normal(size=dm.n)

    def forbidden(_self) -> np.ndarray:
        raise AssertionError("dominant factor smooth was materialized")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden)
    system = build_block_structured_system(
        list(dm.group_matrices),
        groups,
        W,
        Wz,
        dominant_group_index=dominant_index,
    )

    assert system.operator.D.shape == (6, 4, 4)
    assert not dm._tabmat_built
