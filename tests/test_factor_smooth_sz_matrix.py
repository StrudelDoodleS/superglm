"""Compact exact and discrete matrix algebra for SZ factor smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import FactorSmooth
from superglm._frame import as_eager_frame
from superglm.dm_builder import build_design_matrix
from superglm.factor_smooth_geometry import sum_to_zero_contrast
from superglm.group_matrix import (
    DenseGroupMatrix,
    FactorSmoothGroupMatrix,
    _cross_gram,
)


def _matrix_fixture(
    *,
    discrete: bool,
) -> tuple[FactorSmoothGroupMatrix, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(221)
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
    components = (("wiggle", np.diag([1.7, 0.8, 0.0, 0.0])),)
    kwargs = {
        "natural_map": natural_map,
        "levels": levels,
        "repeated_penalty_components": components,
        "factor_basis": "sz",
    }
    if discrete:
        gm = FactorSmoothGroupMatrix(
            support,
            codes,
            len(levels),
            bin_idx=bin_idx,
            **kwargs,
        )
    else:
        gm = FactorSmoothGroupMatrix(
            sp.csr_matrix(support[bin_idx]),
            codes,
            len(levels),
            **kwargs,
        )

    natural_basis = support[bin_idx] @ natural_map
    contrast = sum_to_zero_contrast(len(levels))
    reference = (contrast[codes, :, None] * natural_basis[:, None, :]).reshape(len(codes), -1)
    return gm, reference, natural_basis


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_matrix_products_match_dense_free_coordinate_oracle(discrete: bool) -> None:
    gm, reference, _natural_basis = _matrix_fixture(discrete=discrete)
    rng = np.random.default_rng(317)
    beta = rng.normal(size=reference.shape[1])
    vector = rng.normal(size=reference.shape[0])
    weights = rng.uniform(0.2, 2.0, size=reference.shape[0])
    rhs = rng.normal(size=reference.shape[0])

    assert gm.factor_basis == "sz"
    assert gm.shape == reference.shape
    np.testing.assert_allclose(gm.toarray(), reference, atol=1e-13)
    np.testing.assert_allclose(gm.matvec(beta), reference @ beta, atol=1e-13)
    np.testing.assert_allclose(gm.rmatvec(vector), reference.T @ vector, atol=1e-13)
    np.testing.assert_allclose(
        gm.gram(weights),
        reference.T @ (weights[:, None] * reference),
        atol=1e-12,
    )
    gram, xtw, xt_rhs = gm.gram_rmatvec(weights, rhs)
    np.testing.assert_allclose(gram, reference.T @ (weights[:, None] * reference))
    np.testing.assert_allclose(xtw, reference.T @ weights)
    np.testing.assert_allclose(xt_rhs, reference.T @ rhs)


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_raw_moments_and_public_cross_products_match_oracles(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    gm, reference, natural_basis = _matrix_fixture(discrete=discrete)
    rng = np.random.default_rng(919)
    weights = rng.uniform(0.2, 1.7, size=reference.shape[0])
    rhs = rng.normal(size=reference.shape[0])
    small = DenseGroupMatrix(rng.normal(size=(reference.shape[0], 3)))

    def forbidden(_self) -> np.ndarray:
        raise AssertionError("compact SZ operations must not materialize the design")

    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", forbidden)
    local_gram, local_xtw, local_rhs = gm.factor_smooth_sufficient_stats(weights, rhs)
    raw_cross = gm.factor_smooth_dense_cross_gram(weights, small.M)

    assert local_gram.shape == (gm.n_levels, gm.block_size, gm.block_size)
    assert local_xtw.shape == (gm.n_levels, gm.block_size)
    assert raw_cross.shape == (gm.n_levels, gm.block_size, small.shape[1])
    for level in range(gm.n_levels):
        mask = gm.codes == level
        local = natural_basis[mask]
        np.testing.assert_allclose(
            local_gram[level],
            local.T @ (weights[mask, None] * local),
        )
        np.testing.assert_allclose(local_xtw[level], local.T @ weights[mask])
        np.testing.assert_allclose(local_rhs[level], local.T @ rhs[mask])
        np.testing.assert_allclose(
            raw_cross[level],
            local.T @ (weights[mask, None] * small.M[mask]),
        )

    np.testing.assert_allclose(
        _cross_gram(gm, small, weights),
        reference.T @ (weights[:, None] * small.M),
    )
    np.testing.assert_allclose(
        _cross_gram(small, gm, weights),
        small.M.T @ (weights[:, None] * reference),
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_row_subset_preserves_global_contrast(discrete: bool) -> None:
    gm, reference, _natural_basis = _matrix_fixture(discrete=discrete)
    rows = np.array([0, 2, 5, 8], dtype=np.intp)

    subset = gm.row_subset(rows)

    assert subset.factor_basis == "sz"
    assert subset.n_levels == gm.n_levels
    assert subset.shape == (len(rows), gm.shape[1])
    np.testing.assert_allclose(subset.toarray(), reference[rows])


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_builder_passes_compact_factor_geometry(discrete: bool) -> None:
    x = np.tile(np.linspace(-1.0, 1.0, 8), 3)
    group = np.repeat(["zeta", "alpha", "mu"], 8)
    frame = pd.DataFrame({"x": x, "group": group})
    spec = FactorSmooth("x", group="group", basis="sz", k=6, m=2)

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
        weight_semantics="frequency",
    )

    [gm] = result.dm.group_matrices
    assert isinstance(gm, FactorSmoothGroupMatrix)
    assert gm.factor_basis == "sz"
    assert gm.levels == ("alpha", "mu", "zeta")
    assert gm.shape == (len(frame), 12)
    assert result.groups[0].size == 12


def test_factor_smooth_matrix_rejects_invalid_sz_geometry() -> None:
    basis = sp.csr_matrix(np.eye(4))
    common = {
        "natural_map": np.eye(4),
        "repeated_penalty_components": (("wiggle", np.eye(4)),),
    }

    with pytest.raises(ValueError, match=r"factor_basis must be 'fs' or 'sz'"):
        FactorSmoothGroupMatrix(
            basis,
            np.arange(4, dtype=np.intp),
            4,
            levels=("a", "b", "c", "d"),
            factor_basis="reference",
            **common,
        )

    with pytest.raises(ValueError, match="requires at least two"):
        FactorSmoothGroupMatrix(
            basis,
            np.zeros(4, dtype=np.intp),
            1,
            levels=("only",),
            factor_basis="sz",
            **common,
        )


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_vector_and_moment_products_never_allocate_expanded_geometry(
    monkeypatch: pytest.MonkeyPatch,
    discrete: bool,
) -> None:
    gm, _reference, _natural_basis = _matrix_fixture(discrete=discrete)
    forbidden_shapes = {gm.shape, (gm.shape[1], gm.shape[1])}
    original_zeros = np.zeros
    original_empty = np.empty

    def guarded_zeros(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) in forbidden_shapes:
            raise AssertionError("allocated expanded SZ geometry")
        return original_zeros(shape, *args, **kwargs)

    def guarded_empty(shape, *args, **kwargs):
        if tuple(np.atleast_1d(shape)) in forbidden_shapes:
            raise AssertionError("allocated expanded SZ geometry")
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    monkeypatch.setattr(np, "empty", guarded_empty)
    gm.matvec(np.ones(gm.shape[1]))
    gm.rmatvec(np.ones(gm.shape[0]))
    gm.factor_smooth_sufficient_stats(
        np.ones(gm.shape[0]),
        np.ones(gm.shape[0]),
    )
