"""The exact spline interaction owns one CSR per fitted category level."""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from superglm import dm_builder
from superglm._group_matrix._group_matrix_range import group_range_matvec, group_row_range
from superglm.distributional.solver._formation import abs_group
from superglm.group_matrix import DesignMatrix, SplineCategoricalGroupMatrix
from superglm.types import GroupInfo, GroupSlice


@pytest.mark.parametrize("saturated", [False, True], ids=["sparse", "saturated"])
def test_builder_and_lambda_rebuild_adopt_the_level_csr(monkeypatch, saturated):
    basis = np.arange(1.0, 49.0).reshape(12, 4)
    if not saturated:
        basis *= np.eye(4)[np.arange(12) % 4]
    mask = np.arange(12) % 2 == 0
    weights = np.arange(1.0, 13.0)
    gathered = []
    compute_r_inv = dm_builder.compute_R_inv

    def record_level(level, *args):
        gathered.append(level)
        return compute_r_inv(level, *args)

    monkeypatch.setattr(dm_builder, "compute_R_inv", record_level)
    block, transform, width = dm_builder._process_info(
        GroupInfo(
            columns=None,
            n_cols=4,
            penalty_matrix=np.eye(4),
            reparametrize=True,
            spline_cat_basis=sp.csr_matrix(basis),
            spline_cat_mask=mask,
        ),
        group_name="x:f[B]",
        sample_weight=weights,
        lambda2=0.5,
    )
    reference = SplineCategoricalGroupMatrix(sp.csr_matrix(basis), transform, mask)
    np.testing.assert_array_equal(block.gram(weights), reference.gram(weights))
    assert (block._dense_level is not None) == saturated
    assert block.B_level is gathered[0]
    assert block._data is block.B_level.data
    assert block.B is None

    rebuilt = dm_builder.rebuild_design_matrix_with_lambdas(
        DesignMatrix([block], n=len(mask), p=width),
        [GroupSlice("x:f[B]", 0, width, feature_name="x:f")],
        {"x:f[B]": 2.0},
        weights,
    ).group_matrices[0]
    assert rebuilt.B_level is gathered[1] is block.B_level
    assert rebuilt._data is block.B_level.data
    assert rebuilt.B is None


def test_saved_level_rows_drive_subset_range_and_absolute_products():
    basis = np.arange(-12.0, 12.0).reshape(8, 3)
    rows = np.array([0, 3, 5, 7])
    info = GroupInfo(
        columns=None,
        n_cols=3,
        spline_cat_basis=sp.csr_matrix(basis),
        spline_cat_mask=np.isin(np.arange(len(basis)), rows),
    )
    block, _, _ = dm_builder._process_info(
        info, group_name="x:f[B]", sample_weight=np.ones(8), lambda2=0.0
    )
    block = pickle.loads(pickle.dumps(block))
    expected = np.zeros_like(basis)
    expected[rows] = basis[rows]
    index = np.array([7, 0, 3, 2, 7, 5, -1])
    subset = block.row_subset(index)
    selected = expected[index].copy()
    selected[-1] = 0.0  # Negative indices still do not match category membership.
    np.testing.assert_array_equal(subset.toarray(), selected)
    np.testing.assert_array_equal(subset.row_subset([4, 0, 0]).toarray(), selected[[4, 0, 0]])
    np.testing.assert_array_equal(block.row_subset(np.array(7)).toarray(), expected[7:8])
    np.testing.assert_array_equal(block.row_subset(np.array([1, 6])).toarray(), expected[[1, 6]])
    with pytest.raises(IndexError):
        block.row_subset(np.array([8]))
    np.testing.assert_array_equal(group_row_range(block, 2, 7).toarray(), expected[2:7])
    np.testing.assert_array_equal(group_range_matvec(block, 2, 7, np.ones(3)), expected[2:7].sum(1))
    np.testing.assert_array_equal(
        abs_group(block).matvec(2, 7, np.ones((3, 1))), np.abs(expected[2:7]).sum(1, keepdims=True)
    )
    assert block.B is subset.B is None
    assert block._data is block.B_level.data
