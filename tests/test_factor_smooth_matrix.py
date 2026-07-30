"""Compact matrix coverage for the reference implementation-style factor smooth interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import FactorSmooth
from superglm._frame import as_eager_frame
from superglm._group_matrix import _group_matrix_core
from superglm.dm_builder import build_design_matrix
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
)


def _materialize(
    basis: np.ndarray,
    codes: np.ndarray,
    natural_map: np.ndarray,
    n_levels: int,
) -> np.ndarray:
    natural_basis = basis @ natural_map
    n, block_size = natural_basis.shape
    result = np.zeros((n, n_levels * block_size))
    rows = np.arange(n)
    for column in range(block_size):
        result[rows, codes * block_size + column] = natural_basis[:, column]
    return result


def _matrix_fixture(*, discrete: bool) -> tuple[FactorSmoothGroupMatrix, np.ndarray]:
    rng = np.random.default_rng(871)
    support = np.array(
        [
            [1.0, 0.0, 0.2, 0.0],
            [0.4, 0.6, 0.0, 0.1],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.1, 0.5, 0.4],
            [0.0, 0.0, 0.2, 0.8],
        ]
    )
    bin_idx = np.array([0, 1, 2, 3, 4, 1, 3, 0, 4, 2, 2, 1], dtype=np.intp)
    codes = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 2, 0, 1], dtype=np.intp)
    natural_map = rng.normal(size=(4, 4))
    levels = ("alpha", "beta", "gamma")
    components = (
        ("wiggle", np.diag([1.7, 0.8, 0.0, 0.0])),
        ("null_0", np.diag([0.0, 0.0, 1.2, 0.0])),
        ("null_1", np.diag([0.0, 0.0, 0.0, 0.9])),
    )
    if discrete:
        gm = FactorSmoothGroupMatrix(
            support,
            codes,
            len(levels),
            natural_map=natural_map,
            levels=levels,
            repeated_penalty_components=components,
            bin_idx=bin_idx,
        )
    else:
        gm = FactorSmoothGroupMatrix(
            sp.csr_matrix(support[bin_idx]),
            codes,
            len(levels),
            natural_map=natural_map,
            levels=levels,
            repeated_penalty_components=components,
        )
    reference = _materialize(support[bin_idx], codes, natural_map, len(levels))
    return gm, reference


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_matrix_products_match_materialized_reference(discrete: bool) -> None:
    gm, reference = _matrix_fixture(discrete=discrete)
    rng = np.random.default_rng(317)
    beta = rng.normal(size=reference.shape[1])
    vector = rng.normal(size=reference.shape[0])
    weights = rng.uniform(0.2, 2.0, size=reference.shape[0])
    rhs = rng.normal(size=reference.shape[0])

    np.testing.assert_allclose(gm.matvec(beta), reference @ beta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gm.rmatvec(vector), reference.T @ vector, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        gm.gram(weights),
        reference.T @ (weights[:, None] * reference),
        rtol=1e-13,
        atol=1e-13,
    )
    gram, xtw, xt_rhs = gm.gram_rmatvec(weights, rhs)
    np.testing.assert_allclose(gram, reference.T @ (weights[:, None] * reference))
    np.testing.assert_allclose(xtw, reference.T @ weights)
    np.testing.assert_allclose(xt_rhs, reference.T @ rhs)
    np.testing.assert_allclose(gm.toarray(), reference)


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_local_sufficient_statistics_match_reference(discrete: bool) -> None:
    gm, reference = _matrix_fixture(discrete=discrete)
    rng = np.random.default_rng(919)
    weights = rng.uniform(0.2, 1.7, size=reference.shape[0])
    rhs = rng.normal(size=reference.shape[0])

    local_gram, local_xtw, local_rhs = gm.factor_smooth_sufficient_stats(weights, rhs)

    for level in range(gm.n_levels):
        sl = slice(level * gm.block_size, (level + 1) * gm.block_size)
        np.testing.assert_allclose(
            local_gram[level],
            reference[:, sl].T @ (weights[:, None] * reference[:, sl]),
        )
        np.testing.assert_allclose(local_xtw[level], reference[:, sl].T @ weights)
        np.testing.assert_allclose(local_rhs[level], reference[:, sl].T @ rhs)


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_cross_gram_matches_reference_without_materializing_term(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    gm, reference = _matrix_fixture(discrete=discrete)
    rng = np.random.default_rng(612)
    small = DenseGroupMatrix(rng.normal(size=(reference.shape[0], 3)))
    weights = rng.uniform(0.2, 1.8, size=reference.shape[0])

    def forbidden(_self) -> np.ndarray:
        raise AssertionError("compact factor smooth must not be materialized")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden)
    from superglm.group_matrix import _cross_gram

    np.testing.assert_allclose(
        _cross_gram(gm, small, weights),
        reference.T @ (weights[:, None] * small.M),
    )
    np.testing.assert_allclose(
        _cross_gram(small, gm, weights),
        small.M.T @ (weights[:, None] * reference),
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_row_subset_preserves_global_level_geometry(discrete: bool) -> None:
    gm, reference = _matrix_fixture(discrete=discrete)
    idx = np.array([0, 2, 5, 8], dtype=np.intp)

    subset = gm.row_subset(idx)

    assert subset.shape == (len(idx), gm.shape[1])
    assert subset.n_levels == gm.n_levels
    assert subset.block_size == gm.block_size
    assert subset.levels == gm.levels
    assert subset.repeated_penalty_components is gm.repeated_penalty_components
    np.testing.assert_allclose(subset.toarray(), reference[idx])


def test_exact_and_discrete_factor_smooth_matrices_agree_on_shared_support() -> None:
    exact, exact_reference = _matrix_fixture(discrete=False)
    discrete, discrete_reference = _matrix_fixture(discrete=True)
    rng = np.random.default_rng(119)
    beta = rng.normal(size=exact.shape[1])
    weights = rng.uniform(0.5, 1.5, size=exact.shape[0])

    np.testing.assert_array_equal(exact_reference, discrete_reference)
    np.testing.assert_allclose(exact.matvec(beta), discrete.matvec(beta))
    np.testing.assert_allclose(exact.rmatvec(weights), discrete.rmatvec(weights))
    np.testing.assert_allclose(exact.gram(weights), discrete.gram(weights))


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_vector_products_never_allocate_expanded_design(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    gm, _reference = _matrix_fixture(discrete=discrete)
    expanded_shape = gm.shape
    original_zeros = np.zeros
    original_empty = np.empty

    # Numba 0.64 re-types cached kernels when their NumPy globals are
    # monkeypatched and cannot type the Python allocation guards below.
    # Exercise the same kernel bodies through ``py_func`` so the guards still
    # see every allocation, including allocations made inside a kernel.
    kernel_names = (
        "_factor_smooth_csr_matvec",
        "_factor_smooth_csr_rmatvec",
        "_factor_smooth_csr_sufficient_stats",
        "_factor_smooth_support_matvec",
        "_factor_smooth_support_rmatvec",
        "_factor_smooth_support_cell_aggregates",
    )
    for kernel_name in kernel_names:
        kernel = getattr(_group_matrix_core, kernel_name)
        monkeypatch.setattr(_group_matrix_core, kernel_name, kernel.py_func)

    def guarded_zeros(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) == expanded_shape:
            raise AssertionError("allocated expanded factor-smooth design")
        return original_zeros(shape, *args, **kwargs)

    def guarded_empty(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) == expanded_shape:
            raise AssertionError("allocated expanded factor-smooth design")
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    monkeypatch.setattr(np, "empty", guarded_empty)
    gm.matvec(np.ones(gm.shape[1]))
    gm.rmatvec(np.ones(gm.shape[0]))
    gm.factor_smooth_sufficient_stats(np.ones(gm.shape[0]), np.ones(gm.shape[0]))


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_builder_keeps_all_levels_and_compact_penalties(discrete: bool) -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 8), 3)
    group = np.repeat(["zeta", "alpha", "mu"], 8)
    frame = pd.DataFrame({"x": x, "group": group})
    spec = FactorSmooth("x", group="group", k=6, m=2)

    result = build_design_matrix(
        as_eager_frame(frame),
        np.zeros(len(frame)),
        None,
        None,
        family="gaussian",
        link_spec=None,
        specs={},
        feature_order=[],
        interaction_specs={spec.name: spec},
        interaction_order=[spec.name],
        pending_interactions=[],
        model_discrete=discrete,
        n_bins_config=8,
        lambda2=0.1,
    )

    [gm] = result.dm.group_matrices
    assert isinstance(gm, FactorSmoothGroupMatrix)
    assert gm.levels == ("alpha", "mu", "zeta")
    assert gm.shape == (len(frame), 18)
    assert result.groups[0].name == "x:group:fs"
    assert result.groups[0].size == 18
    assert spec._levels == ["alpha", "mu", "zeta"]
    assert spec._natural_map.shape == (6, 6)
    assert [name for name, _block in gm.repeated_penalty_components] == [
        "wiggle",
        "null_0",
        "null_1",
    ]
    assert all(block.shape == (6, 6) for _name, block in gm.repeated_penalty_components)
    assert all(block.shape != (18, 18) for _name, block in gm.repeated_penalty_components)
    assert not hasattr(gm, "penalty_matrix")


def test_factor_smooth_natural_parameterization_diagonalizes_base_penalty() -> None:
    x = np.linspace(-2.0, 3.0, 41)
    group = np.resize(np.array(["a", "b", "c"], dtype=object), len(x))
    spec = FactorSmooth("x", group="group", k=7, m=2)
    info = spec.build(x, group, {}, sample_weight=np.ones(len(x)))

    raw_penalty = spec._spline._build_penalty()
    transformed = spec._natural_map.T @ raw_penalty @ spec._natural_map
    component_sum = sum(block for _name, block in info.repeated_penalty_components)

    np.testing.assert_allclose(
        transformed,
        info.repeated_penalty_components[0][1],
        atol=2e-10,
    )
    assert np.count_nonzero(np.diag(component_sum) > 0.0) == spec.k
    np.testing.assert_allclose(component_sum, np.diag(np.diag(component_sum)), atol=2e-10)


def test_factor_smooth_uses_reference_pspline_knot_spacing() -> None:
    x = np.linspace(-1.03, 1.03, 150)
    group = np.resize(np.array(["a", "b", "c"], dtype=object), len(x))
    spec = FactorSmooth("x", group="group", k=6, m=2)

    spec.build(x, group, {})

    expanded_lo = x.min() - 0.001 * np.ptp(x)
    expanded_hi = x.max() + 0.001 * np.ptp(x)
    knot_step = (expanded_hi - expanded_lo) / 3.0
    expected = np.linspace(
        expanded_lo - 3.0 * knot_step,
        expanded_hi + 3.0 * knot_step,
        10,
    )
    np.testing.assert_allclose(spec._spline._knots, expected, rtol=0.0, atol=2e-15)


def test_large_factor_smooth_local_grams_are_exactly_symmetric() -> None:
    n = 60_000
    x = np.resize(np.arange(18.0, 91.0), n)
    group = np.resize(np.array([f"region-{index}" for index in range(22)], dtype=object), n)
    spec = FactorSmooth("x", group="group", k=6)
    info = spec.build_discrete(x, group, {}, n_bins=256)
    gm = FactorSmoothGroupMatrix(
        info.factor_smooth_basis_unique,
        info.factor_smooth_codes,
        info.factor_smooth_n_levels,
        natural_map=info.factor_smooth_transform,
        levels=info.factor_smooth_levels,
        repeated_penalty_components=info.repeated_penalty_components,
        bin_idx=info.factor_smooth_bin_idx,
    )

    local_gram, _xtw, _rhs = gm.factor_smooth_sufficient_stats(
        np.linspace(0.1, 2.0, n),
        np.zeros(n),
    )

    np.testing.assert_array_equal(local_gram, local_gram.transpose(0, 2, 1))


def test_factor_smooth_is_excluded_from_tabmat_but_small_blocks_remain_eligible() -> None:
    gm, _reference = _matrix_fixture(discrete=False)
    dense = DenseGroupMatrix(np.ones((gm.shape[0], 2)))
    dm = DesignMatrix([dense, gm], gm.shape[0], dense.shape[1] + gm.shape[1])

    from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
    from superglm._group_matrix._group_matrix_tabmat import _build_tabmat_split

    plan = MatrixExecutionPlan(dm.group_matrices, n=dm.n, ordinary_tabmat=True)
    assert plan.ordinary_indices == (0,)
    assert _build_tabmat_split([gm]) is None
