"""Tests for SCOP reparameterization: beta -> beta_tilde -> gamma chain."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from superglm import Constraint, PSpline
from superglm.features._spline_constraints import curvature_difference_operator
from superglm.features._spline_subclass_ops import build_scop_reparameterization
from superglm.solvers.scop import build_scop_reparam


def _padded_irregular_knots(q: int, degree: int = 3) -> np.ndarray:
    n_interior = q - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1] ** 1.7
    lo_effective = -0.001
    hi_effective = 1.001
    inner = np.concatenate(([lo_effective], interior, [hi_effective]))
    lower = lo_effective - (interior[0] - lo_effective) * np.arange(degree, 0, -1)
    upper = hi_effective + (hi_effective - interior[-1]) * np.arange(1, degree + 1)
    return np.concatenate((lower, inner, upper))


def test_scop_convex_forward_produces_nonnegative_second_differences():
    beta = np.array([0.2, -0.1, -1.0, -0.8, -0.6, -0.4])
    knots = _padded_irregular_knots(6)
    reparam = build_scop_reparam(
        q=6,
        kind="convex",
        knots=knots,
        degree=3,
        domain=(0.0, 1.0),
    )
    gamma = reparam.forward(beta)
    curvature = curvature_difference_operator(knots, 3, domain=(0.0, 1.0))
    assert np.all(curvature @ gamma >= -1e-10)
    np.testing.assert_allclose(curvature @ reparam.Sigma[:, :2], 0.0, atol=2e-14)
    np.testing.assert_allclose(
        curvature @ reparam.Sigma[:, 2:],
        np.eye(4),
        atol=2e-14,
    )


def test_scop_concave_forward_produces_nonpositive_second_differences():
    beta = np.array([0.2, -0.1, -1.0, -0.8, -0.6, -0.4])
    knots = _padded_irregular_knots(6)
    reparam = build_scop_reparam(
        q=6,
        kind="concave",
        knots=knots,
        degree=3,
        domain=(0.0, 1.0),
    )
    gamma = reparam.forward(beta)
    signed_curvature = -curvature_difference_operator(knots, 3, domain=(0.0, 1.0))
    assert np.all(signed_curvature @ gamma >= -1e-10)
    np.testing.assert_allclose(signed_curvature @ reparam.Sigma[:, :2], 0.0, atol=2e-14)
    np.testing.assert_allclose(
        signed_curvature @ reparam.Sigma[:, 2:],
        np.eye(4),
        atol=2e-14,
    )


def test_scop_helper_retains_centered_affine_slope():
    knots = _padded_irregular_knots(6)
    x = np.linspace(0.0, 1.0, 40)
    basis = SciPyBSpline(knots, np.eye(6), 3)(x)
    spec = SimpleNamespace(
        _n_basis=6,
        _knots=knots,
        degree=3,
        _lo=0.0,
        _hi=1.0,
        constraint_kind="convex",
        monotone=None,
    )
    omega = np.eye(6)

    x_centered, scop_penalty, solver_reparam = build_scop_reparameterization(spec, basis, omega)

    assert x_centered.shape == (40, 5)
    assert spec._scop_null_dim == 1
    assert spec._scop_col_means.shape == (5,)
    active_lo = knots[3]
    active_hi = knots[6]
    expected_slope = (x - active_lo) / (active_hi - active_lo)
    np.testing.assert_allclose(
        x_centered[:, 0],
        expected_slope - expected_slope.mean(),
        atol=2e-15,
    )
    assert scop_penalty.shape == (5, 5)
    np.testing.assert_array_equal(scop_penalty[0], np.zeros(5))
    assert solver_reparam.q == 5
    assert solver_reparam.free_dim == 1


@pytest.mark.parametrize("kind", ["convex", "concave"])
@pytest.mark.parametrize("knot_case", ["default", "quantile", "explicit"])
def test_curvature_scop_sigma_certifies_exact_public_domain_geometry(kind, knot_case):
    x = np.linspace(0.0, 1.0, 300)
    kwargs = {}
    if knot_case == "quantile":
        x = x**5
        kwargs["knot_strategy"] = "quantile"
    elif knot_case == "explicit":
        kwargs["knots"] = np.array([1e-4, 0.003, 0.04, 0.25, 0.82, 0.98])

    spec = PSpline(
        n_knots=8,
        constraint=getattr(Constraint.fit, kind),
        **kwargs,
    )
    spec.build(x)
    curvature = curvature_difference_operator(
        spec._knots,
        spec.degree,
        domain=(spec._lo, spec._hi),
    )
    signed_curvature = curvature if kind == "convex" else -curvature
    sigma = np.asarray(spec._scop_Sigma)

    np.testing.assert_allclose(
        signed_curvature @ sigma[:, :2],
        0.0,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        signed_curvature @ sigma[:, 2:],
        np.eye(spec._n_basis - 2),
        rtol=3e-13,
        atol=3e-13,
    )


class TestSCOPForwardMap:
    """Forward map: beta -> gamma via cumulative exp."""

    def test_increasing_produces_monotone_gamma(self):
        """Any beta vector produces monotone increasing gamma."""
        rng = np.random.default_rng(42)
        q = 8
        beta = rng.standard_normal(q)
        reparam = build_scop_reparam(q, direction="increasing")
        gamma = reparam.forward(beta)
        assert np.all(np.diff(gamma) >= 0)

    def test_decreasing_produces_monotone_gamma(self):
        rng = np.random.default_rng(42)
        q = 8
        beta = rng.standard_normal(q)
        reparam = build_scop_reparam(q, direction="decreasing")
        gamma = reparam.forward(beta)
        assert np.all(np.diff(gamma) <= 0)

    def test_forward_shape(self):
        q = 10
        reparam = build_scop_reparam(q, direction="increasing")
        gamma = reparam.forward(np.zeros(q))
        assert gamma.shape == (q,)

    def test_beta_tilde_positivity(self):
        """beta_tilde components 2..q are always positive (exp)."""
        rng = np.random.default_rng(42)
        q = 8
        beta = rng.standard_normal(q) * 5  # large values
        reparam = build_scop_reparam(q, direction="increasing")
        beta_tilde = reparam.beta_tilde(beta)
        assert beta_tilde[0] == beta[0]
        assert np.all(beta_tilde[1:] > 0)


class TestSCOPJacobian:
    """Jacobian d(gamma)/d(beta) -- lower-triangular cumulative."""

    def test_jacobian_shape(self):
        q = 8
        reparam = build_scop_reparam(q, direction="increasing")
        beta = np.zeros(q)
        J = reparam.jacobian(beta)
        assert J.shape == (q, q)

    def test_jacobian_lower_triangular(self):
        q = 6
        reparam = build_scop_reparam(q, direction="increasing")
        beta = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.4])
        J = reparam.jacobian(beta)
        # Upper triangle (excluding diagonal) should be zero
        assert np.allclose(np.triu(J, k=1), 0)

    def test_jacobian_column_1_all_ones(self):
        q = 6
        reparam = build_scop_reparam(q, direction="increasing")
        beta = np.random.default_rng(42).standard_normal(q)
        J = reparam.jacobian(beta)
        np.testing.assert_allclose(J[:, 0], 1.0)

    def test_jacobian_column_i_has_exp(self):
        """Column i (i>=1) has exp(beta_i) in rows j>=i."""
        q = 5
        reparam = build_scop_reparam(q, direction="increasing")
        beta = np.array([0.1, 0.5, -0.3, 0.2, 0.8])
        J = reparam.jacobian(beta)
        for i in range(1, q):
            expected_val = np.exp(beta[i])
            for j in range(q):
                if j >= i:
                    np.testing.assert_allclose(J[j, i], expected_val, rtol=1e-12)
                else:
                    assert J[j, i] == 0.0

    def test_jacobian_vs_finite_differences(self):
        """Jacobian matches numerical finite differences."""
        q = 6
        reparam = build_scop_reparam(q, direction="increasing")
        beta = np.random.default_rng(42).standard_normal(q)
        J_analytic = reparam.jacobian(beta)

        eps = 1e-7
        J_fd = np.zeros((q, q))
        for i in range(q):
            beta_plus = beta.copy()
            beta_plus[i] += eps
            beta_minus = beta.copy()
            beta_minus[i] -= eps
            J_fd[:, i] = (reparam.forward(beta_plus) - reparam.forward(beta_minus)) / (2 * eps)

        np.testing.assert_allclose(J_analytic, J_fd, atol=1e-6)


class TestSCOPRoundTrip:
    """Feasible initialization: gamma -> beta via log recovery."""

    def test_roundtrip_monotone_sequence(self):
        """Forward then inverse recovers original beta."""
        q = 8
        reparam = build_scop_reparam(q, direction="increasing")
        beta_orig = np.array([1.0, 0.5, 0.3, 0.8, -0.2, 0.1, 0.4, 0.6])
        gamma = reparam.forward(beta_orig)
        beta_recovered = reparam.initialize_from_gamma(gamma)
        np.testing.assert_allclose(beta_recovered, beta_orig, atol=1e-10)

    def test_initialize_from_nonmonotone_gamma(self):
        """Non-monotone gamma gets clamped, producing a feasible beta."""
        q = 5
        reparam = build_scop_reparam(q, direction="increasing")
        gamma_bad = np.array([1.0, 3.0, 2.0, 4.0, 3.5])  # not monotone
        beta = reparam.initialize_from_gamma(gamma_bad)
        gamma_repaired = reparam.forward(beta)
        # Repaired gamma should be monotone
        assert np.all(np.diff(gamma_repaired) >= 0)


class TestSCOPPenalty:
    """SCOP-specific penalty on beta parameters."""

    def test_penalty_shape(self):
        q = 8
        reparam = build_scop_reparam(q, direction="increasing")
        S = reparam.penalty_matrix()
        assert S.shape == (q, q)

    def test_penalty_symmetric_psd(self):
        q = 8
        reparam = build_scop_reparam(q, direction="increasing")
        S = reparam.penalty_matrix()
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-10)

    def test_penalty_null_space(self):
        """Penalty has rank q-2: beta_1 is unpenalized, and a constant
        direction on (beta_2, ..., beta_q) is in the null space of the
        first-difference penalty."""
        q = 10
        reparam = build_scop_reparam(q, direction="increasing")
        S = reparam.penalty_matrix()
        eigvals = np.linalg.eigvalsh(S)
        n_zero = np.sum(np.abs(eigvals) < 1e-10)
        # beta_1 free (row/col 0 is zero) + constant beta_2=...=beta_q unpenalized
        assert n_zero == 2


class TestSCOPQPInitialization:
    """QP initialization in beta_tilde space."""

    def test_qp_init_produces_feasible_beta(self):
        """QP initialization from data produces a feasible starting point."""
        rng = np.random.default_rng(42)
        q = 8
        n = 200
        # Simulated B-spline design matrix and target
        B = rng.standard_normal((n, q))
        y = np.sort(rng.uniform(0, 5, n))
        reparam = build_scop_reparam(q, direction="increasing")
        beta_init = reparam.qp_initialize(B, y, lambda_penalty=0.1)
        gamma = reparam.forward(beta_init)
        assert np.all(np.diff(gamma) >= -1e-10)
