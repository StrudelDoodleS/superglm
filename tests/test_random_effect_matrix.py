"""Compact all-level matrix tests for random effects."""

import numpy as np
import pytest

import superglm.group_matrix as group_matrix
from superglm import LambdaPolicy, RandomEffect
from superglm.dm_builder import _process_info


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
