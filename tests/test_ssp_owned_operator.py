"""Every SSP operation must use the same current represented design."""

import numpy as np
import pytest
import scipy.sparse as sp

from superglm._group_matrix import _group_matrix_algebra as algebra
from superglm.group_matrix import SparseSSPGroupMatrix


@pytest.mark.parametrize("exponent", [0, -600, 600])
@pytest.mark.parametrize("mutation", ["private", "public", "replace_values", "replace_basis"])
def test_ssp_mutation_keeps_gram_and_prediction_on_one_operator(exponent, mutation):
    basis = np.array([[1.0, 0.25], [0.5, 1.0], [0.75, 0.5]])
    group = SparseSSPGroupMatrix(
        sp.csr_matrix(np.ldexp(basis, exponent)), np.ldexp(np.eye(2), -exponent)
    )
    # Exercise dispatch once before mutation to catch retained stale values.
    group.gram(np.ones(3))
    if mutation == "private":
        group._data *= 2.0
    elif mutation == "public":
        group.B.data *= 2.0
    elif mutation == "replace_values":
        group.B.data = 2.0 * group.B.data
    else:
        group.B = 2.0 * group.B

    design = 2.0 * basis
    weights = np.array([0.5, 1.0, 0.25])
    vector = np.array([0.5, -0.25])
    expected_gram = design.T @ (weights[:, None] * design)
    np.testing.assert_array_equal(group.toarray(), design)
    np.testing.assert_array_equal(group.matvec(vector), design @ vector)
    np.testing.assert_array_equal(group.rmatvec(weights), design.T @ weights)
    np.testing.assert_array_equal(group.gram(weights), expected_gram)
    np.testing.assert_array_equal(group.row_subset(np.array([2, 0])).toarray(), design[[2, 0]])
    other = SparseSSPGroupMatrix(sp.csr_matrix(basis), np.eye(2))
    np.testing.assert_array_equal(
        algebra._cross_gram(group, other, weights), design.T @ (weights[:, None] * basis)
    )


def test_ssp_owns_float64_values_without_changing_the_input_basis():
    basis = sp.csr_matrix(np.array([[1.0, 0.25], [0.5, 1.0]], dtype=np.float32))
    expected = basis.toarray().astype(np.float64)
    group = SparseSSPGroupMatrix(basis, np.eye(2))
    basis.data *= 2.0
    assert group.B.dtype == np.dtype(np.float64)
    assert np.shares_memory(group.B.data, group._data)
    np.testing.assert_array_equal(group.toarray(), expected)
    np.testing.assert_array_equal(group.gram(np.ones(2)), expected.T @ expected)
