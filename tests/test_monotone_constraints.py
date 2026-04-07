"""Tests for monotone constraint builders on BSplineSmooth and CRS."""

import numpy as np
import pytest

from superglm.features.spline import (
    BSplineSmooth,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
)
from superglm.types import LinearConstraintSet


class TestBSplineSmoothMonotoneConstraints:
    def test_monotone_increasing_builds(self):
        s = BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.constraints is not None
        assert info.monotone_engine == "qp"

    def test_monotone_decreasing_builds(self):
        s = BSplineSmooth(n_knots=8, monotone="decreasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.constraints is not None

    def test_no_monotone_no_constraints(self):
        s = BSplineSmooth(n_knots=8)
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.constraints is None
        assert info.monotone_engine is None

    def test_constraint_shape(self):
        """Constraints have correct dimensions: (K-1, n_cols) after projection."""
        s = BSplineSmooth(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        cs = info.constraints
        # Constraint params should match post-identifiability columns
        assert cs.n_params == info.n_cols

    def test_constraint_is_linear_constraint_set(self):
        s = BSplineSmooth(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert isinstance(info.constraints, LinearConstraintSet)

    def test_decreasing_flips_constraint(self):
        """Decreasing constraints are negation of increasing: coef_{i+1} <= coef_i."""
        s_inc = BSplineSmooth(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        s_dec = BSplineSmooth(
            n_knots=8,
            monotone="decreasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info_inc = s_inc.build(x)
        info_dec = s_dec.build(x)

        np.testing.assert_allclose(info_dec.constraints.A, -info_inc.constraints.A)

    def test_monotone_vector_is_feasible(self):
        """A monotonically increasing raw coefficient vector should be feasible."""
        s = BSplineSmooth(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)

        # Build a monotone-increasing raw coefficient vector that lies in
        # the column space of P (centered for sum-to-zero identifiability).
        K = info.raw_to_solver_map.shape[0]
        beta_raw = np.linspace(1, K, K)
        beta_raw -= beta_raw.mean()  # center to satisfy identifiability

        # Map to post-identifiability (solver) space via lstsq
        P = info.raw_to_solver_map
        beta_solver = np.linalg.lstsq(P, beta_raw, rcond=None)[0]

        assert info.constraints.is_feasible(beta_solver)

    def test_nonmonotone_vector_is_infeasible(self):
        """A non-monotone raw coefficient vector should be infeasible."""
        s = BSplineSmooth(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)

        # Build a non-monotone raw coefficient vector (dip in the middle),
        # centered to lie in the column space of P.
        K = info.raw_to_solver_map.shape[0]
        beta_raw = np.linspace(1, K, K)
        beta_raw[K // 2] = 0.0  # create a dip
        beta_raw -= beta_raw.mean()

        P = info.raw_to_solver_map
        beta_solver = np.linalg.lstsq(P, beta_raw, rcond=None)[0]

        assert not info.constraints.is_feasible(beta_solver)

    def test_postfit_mode_unchanged(self):
        """monotone_mode='postfit' still works as before (no constraints)."""
        s = BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="postfit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.constraints is None
        assert info.monotone_engine is None


class TestCRSMonotoneConstraints:
    """CubicRegressionSpline monotone constraint builder tests."""

    def test_monotone_increasing_builds(self):
        s = CubicRegressionSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.constraints is not None
        assert info.monotone_engine == "qp"

    def test_no_monotone_no_constraints(self):
        s = CubicRegressionSpline(n_knots=8)
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.constraints is None

    def test_constraint_shape(self):
        s = CubicRegressionSpline(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        cs = info.constraints
        assert cs.n_params == info.n_cols

    def test_monotone_vector_is_feasible(self):
        """A monotone-increasing centered raw vector should be feasible."""
        s = CubicRegressionSpline(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)

        K = info.raw_to_solver_map.shape[0]
        beta_raw = np.linspace(1, K, K)
        beta_raw -= beta_raw.mean()

        P = info.raw_to_solver_map
        beta_solver = np.linalg.lstsq(P, beta_raw, rcond=None)[0]

        assert info.constraints.is_feasible(beta_solver)

    def test_nonmonotone_vector_is_infeasible(self):
        """A non-monotone centered raw vector should be infeasible."""
        s = CubicRegressionSpline(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info = s.build(x)

        K = info.raw_to_solver_map.shape[0]
        beta_raw = np.linspace(1, K, K)
        beta_raw[K // 2] = 0.0  # create a dip
        beta_raw -= beta_raw.mean()

        P = info.raw_to_solver_map
        beta_solver = np.linalg.lstsq(P, beta_raw, rcond=None)[0]

        assert not info.constraints.is_feasible(beta_solver)

    def test_decreasing_flips_constraint(self):
        s_inc = CubicRegressionSpline(
            n_knots=8,
            monotone="increasing",
            monotone_mode="fit",
            penalty="none",
        )
        s_dec = CubicRegressionSpline(
            n_knots=8,
            monotone="decreasing",
            monotone_mode="fit",
            penalty="none",
        )
        x = np.linspace(0, 1, 200)
        info_inc = s_inc.build(x)
        info_dec = s_dec.build(x)
        np.testing.assert_allclose(info_dec.constraints.A, -info_inc.constraints.A)

    def test_ps_monotone_fit_uses_scop(self):
        """PSpline monotone_mode='fit' now builds SCOP reparameterization."""
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.scop_reparameterization is not None
        assert info.monotone_engine == "scop"
        assert info.constraints is None

    def test_ns_monotone_not_supported(self):
        """NaturalSpline does not accept monotone parameter."""
        with pytest.raises(TypeError):
            NaturalSpline(n_knots=8, monotone="increasing", monotone_mode="fit")


class TestPSplineSCOPConstraints:
    """PSpline builds SCOP reparameterization for monotone_mode='fit'."""

    def test_monotone_increasing_builds(self):
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.scop_reparameterization is not None
        assert info.monotone_engine == "scop"
        assert info.constraints is None

    def test_monotone_decreasing_builds(self):
        s = PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.scop_reparameterization is not None
        assert info.scop_reparameterization.direction == "decreasing"

    def test_no_monotone_no_scop(self):
        s = PSpline(n_knots=8)
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.scop_reparameterization is None
        assert info.monotone_engine is None

    def test_scop_reparam_q_matches_solver_dimension(self):
        """Solver-space reparam has q_eff = q_raw - 1 = n_cols."""
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.scop_reparameterization.q == info.n_cols

    def test_scop_penalty_shape_matches(self):
        """SCOP penalty matrix shape matches n_cols."""
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        S_scop = info.scop_reparameterization.penalty_matrix()
        assert S_scop.shape == (info.n_cols, info.n_cols)

    def test_reparametrize_false(self):
        """SCOP terms bypass standard SSP."""
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        assert info.reparametrize is False

    def test_columns_centered(self):
        """SCAM-style centering: columns of B_centered have zero mean."""
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        B = info.columns
        if hasattr(B, "toarray"):
            B = B.toarray()
        col_means = B.mean(axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-10)

    def test_forward_map_produces_monotone(self):
        """Random solver-space beta produces monotone gamma through the SCOP chain."""
        s = PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
        x = np.linspace(0, 1, 200)
        info = s.build(x)
        rng = np.random.default_rng(42)
        # beta in solver-space: length n_cols = q-1
        beta_eff = rng.standard_normal(info.n_cols)
        gamma_eff = info.scop_reparameterization.forward(beta_eff)
        # gamma_eff coefficients should be monotone (q_eff length)
        assert np.all(np.diff(gamma_eff) >= 0)
