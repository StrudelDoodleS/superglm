"""Compact all-level matrix tests for random effects."""

import numpy as np
import pytest
from numba.core import types  # type: ignore[import-untyped]
from numba.core.registry import CPUDispatcher  # type: ignore[import-untyped]

import superglm._group_matrix._group_matrix_kernels as group_kernels
import superglm.group_matrix as group_matrix
from superglm import LambdaPolicy, RandomEffect
from superglm._group_matrix._group_matrix_kernels import (
    _random_effect_sufficient_stats,
)
from superglm.dm_builder import _process_info


def test_group_matrix_warmup_reaches_every_module_owned_dispatcher() -> None:
    dispatchers = {
        name: value
        for name, value in vars(group_kernels).items()
        if isinstance(value, CPUDispatcher) and value.py_func.__module__ == group_kernels.__name__
    }
    assert dispatchers

    group_kernels._warmup_group_matrix_kernels()

    assert all(dispatcher.nopython_signatures for dispatcher in dispatchers.values())


def test_group_matrix_warmup_matches_constructor_owned_integer_dtypes() -> None:
    group_kernels._warmup_group_matrix_kernels()

    assert any(
        signature.args[1].dtype == types.int32 and signature.args[2].dtype == types.int32
        for signature in group_kernels._csr_weighted_gram.nopython_signatures
    )
    assert any(
        signature.args[0].dtype == types.int32
        and signature.args[1].dtype == types.int32
        and signature.args[4].dtype == types.intp
        for signature in group_kernels._pattern_support_summaries.nopython_signatures
    )
    assert any(
        signature.args[0].dtype == types.intp
        for signature in group_kernels._random_effect_sufficient_stats.nopython_signatures
    )


def test_random_effect_group_matrix_is_available():
    assert hasattr(group_matrix, "RandomEffectGroupMatrix")


def test_random_effect_group_matrix_matches_materialized_all_level_reference():
    codes = np.array([2, 0, 1, 2, 1], dtype=np.intp)
    gm = group_matrix.RandomEffectGroupMatrix(codes, n_levels=3)
    reference = np.eye(3)[codes]
    beta = np.array([0.4, -0.2, 0.8])
    rhs = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
    weights = np.array([0.5, 1.0, 2.0, 1.5, 0.25])

    np.testing.assert_allclose(gm.matvec(beta), reference @ beta)
    np.testing.assert_allclose(gm.rmatvec(rhs), reference.T @ rhs)
    np.testing.assert_allclose(gm.gram(weights), reference.T @ (weights[:, None] * reference))
    np.testing.assert_array_equal(gm.toarray(), reference)
    assert gm.shape == (5, 3)


def test_random_effect_group_matrix_row_subset_preserves_type_geometry_and_policy():
    policy = LambdaPolicy.fixed(4.0)
    gm = group_matrix.RandomEffectGroupMatrix(
        np.array([2, 0, 1, 2], dtype=np.intp),
        n_levels=3,
        lambda_policies={"_default": policy},
    )

    subset = gm.row_subset(np.array([0, 1, 3]))

    assert isinstance(subset, group_matrix.RandomEffectGroupMatrix)
    assert subset.n_levels == 3
    assert subset.lambda_policies == {"_default": policy}
    np.testing.assert_array_equal(subset.codes, [2, 0, 2])


@pytest.mark.parametrize("codes", [np.array([-1, 0]), np.array([0, 2])])
def test_random_effect_group_matrix_rejects_codes_outside_fitted_levels(codes):
    with pytest.raises(ValueError, match="codes"):
        group_matrix.RandomEffectGroupMatrix(codes, n_levels=2)


def test_design_builder_dispatches_random_effect_group_info_to_compact_type():
    policy = LambdaPolicy.fixed(2.0)
    info = RandomEffect(lambda_policy=policy).build(np.array(["b", "a", "c", "b"], dtype=object))

    gm, r_inv, n_cols = _process_info(
        info,
        sample_weight=np.ones(4),
        lambda2=0.1,
    )

    assert isinstance(gm, group_matrix.RandomEffectGroupMatrix)
    assert gm.lambda_policies == {"_default": policy}
    np.testing.assert_array_equal(gm.codes, [1, 0, 2, 1])
    assert r_inv is None
    assert n_cols == 3


def test_tabmat_represents_every_random_effect_level_without_drop_first():
    codes = np.arange(101, dtype=np.intp)
    gm = group_matrix.RandomEffectGroupMatrix(codes, n_levels=101)

    split = group_matrix.DesignMatrix([gm], n=101, p=101).tabmat_split
    categorical = split.matrices[0]

    assert categorical.drop_first is False
    np.testing.assert_array_equal(categorical.categories, np.arange(101))
    np.testing.assert_array_equal(categorical.toarray(), gm.toarray())


def test_tabmat_keeps_ordinary_categorical_drop_first_representation():
    codes = np.arange(101, dtype=np.intp)
    gm = group_matrix.CategoricalGroupMatrix(codes, n_levels=101)

    split = group_matrix.DesignMatrix([gm], n=101, p=101).tabmat_split
    categorical = split.matrices[0]

    assert categorical.drop_first is True
    np.testing.assert_array_equal(categorical.toarray(), gm.toarray())


def test_random_effect_sufficient_stats_fuse_weight_and_rhs_aggregation():
    codes = np.array([2, 0, 1, 2, 1, 2], dtype=np.intp)
    W = np.array([0.5, -1.0, 2.0, 1.5, -0.25, 0.75])
    Wz = np.array([-0.2, 1.0, 0.4, 2.0, -1.5, 0.3])

    level_W, level_Wz = _random_effect_sufficient_stats(codes, W, Wz, 4)

    np.testing.assert_allclose(level_W, np.bincount(codes, weights=W, minlength=4))
    np.testing.assert_allclose(level_Wz, np.bincount(codes, weights=Wz, minlength=4))
