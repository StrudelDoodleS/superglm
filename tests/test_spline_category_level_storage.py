"""The exact spline interaction owns one CSR per fitted category level."""

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from superglm import SuperGLM, dm_builder
from superglm._group_matrix._group_matrix_range import group_range_matvec, group_row_range
from superglm.distributional.solver._formation import abs_group
from superglm.group_matrix import DesignMatrix, SplineCategoricalGroupMatrix
from superglm.types import GroupInfo, GroupSlice, LambdaPolicy, LinearConstraintSet


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


class _FixedQPLevel:
    """A user spec whose level block is a projected, in-term, fixed-lambda QP block."""

    def build(self, x, sample_weight=None):
        return GroupInfo(
            columns=None,
            n_cols=2,
            penalty_matrix=np.diag([1.0, 2.0]),
            reparametrize=True,
            projection=np.eye(3)[:, 1:],
            penalty_components=[("wiggle", np.diag([1.0, 2.0]))],
            lambda_policies={"wiggle": LambdaPolicy.fixed(7.0)},
            penalty_in_term_coordinates=True,
            monotone_engine="qp",
            constraints=LinearConstraintSet(A=np.eye(2), b=np.zeros(2)),
            spline_cat_basis=sp.csr_matrix(np.column_stack([np.ones_like(x), x, x * x])),
            spline_cat_mask=x > 0,
        )

    def _build_monotone_constraints_raw(self):
        return LinearConstraintSet(A=np.eye(3)[1:], b=np.zeros(2))

    def transform(self, x):
        return np.column_stack([x, x * x]) * (x > 0)[:, None]

    def reconstruct(self, beta):
        return {}


@pytest.mark.pins_behaviour
def test_a_fixed_qp_level_block_whitens_at_the_fit_lambda():
    """Pins master's whitening of a compact spline-by-level block from a user spec.

    The block is projected, in term coordinates and a QP block whose one
    component is fixed at 7. Only a generic projected block is whitened at
    its fixed QP lambdas; this level block is whitened at the fit's lambda.
    The values are master's (27633b021): the block's ``R_inv`` and the
    predictions after ``fit_path``.
    """
    x = np.linspace(-1.0, 1.0, 12)
    X = pd.DataFrame({"x": x})
    model = SuperGLM(features={"x": _FixedQPLevel()}, spline_penalty=0.2)
    model.fit_path(
        X, 1 + 2 * np.maximum(x, 0), sample_weight=np.arange(1.0, 13.0), lambda_seq=[0.05, 0.01]
    )
    np.testing.assert_allclose(
        model._dm.group_matrices[0].R_inv,
        [[0.0, 0.0], [1.235759364823351, -0.8079407647416339], [0.0, 1.3952108513201944]],
        rtol=1e-12,
    )
    expected = [1.0540477648990099] * 6 + [
        1.1480454357000967,
        1.3479476406541595,
        1.5610212173733053,
        1.7830619223847166,
        2.0088424717254707,
        2.232272322172331,
    ]
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-6)
