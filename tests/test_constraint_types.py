"""Tests for LinearConstraintSet type and GroupInfo constraint extensions."""

import numpy as np

from superglm.types import GroupInfo, LinearConstraintSet


class TestLinearConstraintSet:
    def test_create(self):
        A = np.array([[1, -1, 0], [0, 1, -1]])
        b = np.zeros(2)
        cs = LinearConstraintSet(A=A, b=b)
        assert cs.A.shape == (2, 3)
        assert cs.b.shape == (2,)

    def test_n_constraints(self):
        A = np.array([[1, -1, 0], [0, 1, -1]])
        b = np.zeros(2)
        cs = LinearConstraintSet(A=A, b=b)
        assert cs.n_constraints == 2

    def test_n_params(self):
        A = np.array([[1, -1, 0], [0, 1, -1]])
        b = np.zeros(2)
        cs = LinearConstraintSet(A=A, b=b)
        assert cs.n_params == 3

    def test_check_feasibility(self):
        A = np.array([[1, -1, 0], [0, 1, -1]])
        b = np.zeros(2)
        cs = LinearConstraintSet(A=A, b=b)
        # [3, 2, 1] satisfies both constraints (1>=0, 1>=0)
        assert cs.is_feasible(np.array([3.0, 2.0, 1.0]))
        # [1, 2, 3] violates both (-1>=0 fails)
        assert not cs.is_feasible(np.array([1.0, 2.0, 3.0]))

    def test_compose_with_projection(self):
        """A_solver = A_raw @ P maps constraints into projected space."""
        A_raw = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
        b = np.zeros(3)
        P = np.random.default_rng(42).standard_normal((4, 3))
        cs_raw = LinearConstraintSet(A=A_raw, b=b)
        cs_proj = cs_raw.compose(P)
        np.testing.assert_allclose(cs_proj.A, A_raw @ P)
        np.testing.assert_array_equal(cs_proj.b, b)
        assert cs_proj.n_params == 3


class TestGroupInfoConstraints:
    """GroupInfo carries optional constraint metadata."""

    def test_default_no_constraints(self):
        info = GroupInfo(columns=None, n_cols=5)
        assert info.constraints is None
        assert info.monotone_engine is None

    def test_with_constraints(self):
        A = np.eye(3)
        b = np.zeros(3)
        cs = LinearConstraintSet(A=A, b=b)
        info = GroupInfo(
            columns=None,
            n_cols=3,
            constraints=cs,
            monotone_engine="qp",
        )
        assert info.constraints is not None
        assert info.monotone_engine == "qp"

    def test_raw_to_solver_map(self):
        info = GroupInfo(
            columns=None,
            n_cols=3,
            raw_to_solver_map=np.eye(4, 3),
        )
        assert info.raw_to_solver_map.shape == (4, 3)
