"""Tests for the active-set constrained penalized least-squares solver."""

import numpy as np

from superglm.solvers.constrained_qp import solve_constrained_qp


class TestUnconstrainedFallback:
    """When no constraints are active, QP reduces to unconstrained solve."""

    def test_unconstrained_matches_direct_solve(self):
        """With no binding constraints, QP = Cholesky solve."""
        rng = np.random.default_rng(42)
        p = 5
        H = rng.standard_normal((p, p))
        H = H.T @ H + np.eye(p)  # PD
        g = rng.standard_normal(p)

        # No constraints
        A = np.zeros((0, p))
        b = np.zeros(0)

        result = solve_constrained_qp(H, g, A, b)
        expected = np.linalg.solve(H, g)
        np.testing.assert_allclose(result.beta, expected, atol=1e-10)

    def test_interior_solution_ignores_constraints(self):
        """If unconstrained solution is feasible, return it."""
        H = np.eye(3)
        g = np.array([2.0, 3.0, 1.0])
        # Constraint: beta >= 0 (all nonneg)
        A = np.eye(3)
        b = np.zeros(3)

        result = solve_constrained_qp(H, g, A, b)
        # Unconstrained solution is [2, 3, 1] which satisfies beta >= 0
        np.testing.assert_allclose(result.beta, g, atol=1e-10)
        assert len(result.active_set) == 0


class TestSimpleConstraints:
    """Known-answer constrained QP problems."""

    def test_single_binding_constraint(self):
        """min 0.5 * (x1^2 + x2^2) - [-1, 2]^T x  s.t. x1 >= 0.

        Unconstrained: x* = [-1, 2]. But x1 >= 0, so x* = [0, 2].
        """
        H = np.eye(2)
        g = np.array([-1.0, 2.0])
        A = np.array([[1.0, 0.0]])  # x1 >= 0
        b = np.array([0.0])

        result = solve_constrained_qp(H, g, A, b)
        np.testing.assert_allclose(result.beta, [0.0, 2.0], atol=1e-10)
        assert 0 in result.active_set

    def test_monotone_constraint(self):
        """min 0.5 * ||beta - target||^2  s.t. beta monotone increasing.

        target = [3, 1, 2, 4] -> constrained solution is isotonic regression.
        """
        target = np.array([3.0, 1.0, 2.0, 4.0])
        H = np.eye(4)
        g = target
        # Adjacent differences: beta_{i+1} - beta_i >= 0
        A = np.diff(np.eye(4), axis=0)  # (3, 4)
        b = np.zeros(3)

        result = solve_constrained_qp(H, g, A, b)
        beta = result.beta
        # Must be monotone increasing
        assert np.all(np.diff(beta) >= -1e-10)
        # Known isotonic regression: [2, 2, 2, 4]
        np.testing.assert_allclose(beta, [2.0, 2.0, 2.0, 4.0], atol=1e-8)

    def test_penalized_monotone(self):
        """Penalized monotone: min 0.5 * beta^T (I + lambda*D'D) beta - g^T beta
        s.t. D @ beta >= 0.
        """
        p = 5
        lam = 0.1
        D = np.diff(np.eye(p), n=2, axis=0)
        H = np.eye(p) + lam * D.T @ D
        g = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        A = np.diff(np.eye(p), axis=0)  # monotone increasing
        b = np.zeros(p - 1)

        result = solve_constrained_qp(H, g, A, b)
        # Must be feasible
        assert np.all(np.diff(result.beta) >= -1e-10)


class TestWarmStart:
    """Warm-starting from a previous active set."""

    def test_warm_start_speeds_convergence(self):
        """With correct warm start, solver should converge immediately or nearly."""
        H = np.eye(3)
        g = np.array([-1.0, 2.0, 5.0])
        A = np.array([[1.0, 0.0, 0.0]])  # x1 >= 0
        b = np.array([0.0])

        # Cold start
        result_cold = solve_constrained_qp(H, g, A, b)
        # Warm start with known active set
        result_warm = solve_constrained_qp(H, g, A, b, active_set_init=result_cold.active_set)
        np.testing.assert_allclose(result_cold.beta, result_warm.beta, atol=1e-12)
        assert result_warm.n_iter <= result_cold.n_iter


class TestFeasibilityRestoration:
    """Solver handles infeasible starting points."""

    def test_infeasible_start_finds_feasible(self):
        """Even with infeasible initial beta, solver finds feasible solution."""
        H = np.eye(3)
        g = np.array([3.0, 2.0, 1.0])
        A = np.diff(np.eye(3), axis=0)  # monotone increasing
        b = np.zeros(2)

        # g = [3, 2, 1] is the unconstrained solution, which is decreasing
        result = solve_constrained_qp(H, g, A, b)
        assert np.all(np.diff(result.beta) >= -1e-10)

    def test_feasibility_with_rhs(self):
        """Constraints with nonzero b: A @ beta >= b."""
        H = np.eye(2)
        g = np.array([0.5, 0.5])
        A = np.eye(2)
        b = np.array([1.0, 1.0])  # beta >= 1

        result = solve_constrained_qp(H, g, A, b)
        np.testing.assert_allclose(result.beta, [1.0, 1.0], atol=1e-10)


class TestEdgeCases:
    """Edge cases and numerical stability."""

    def test_all_constraints_active(self):
        """All constraints binding at solution."""
        H = np.eye(3)
        g = np.array([0.0, 0.0, 0.0])
        A = np.eye(3)
        b = np.array([1.0, 2.0, 3.0])

        result = solve_constrained_qp(H, g, A, b)
        np.testing.assert_allclose(result.beta, [1.0, 2.0, 3.0], atol=1e-10)

    def test_singular_penalty(self):
        """H is positive semidefinite (not definite) -- regularization needed."""
        p = 4
        D = np.diff(np.eye(p), n=2, axis=0)
        H = D.T @ D  # rank-deficient
        H += 1e-8 * np.eye(p)  # small regularization
        g = np.array([1.0, 2.0, 3.0, 4.0])
        A = np.diff(np.eye(p), axis=0)
        b = np.zeros(p - 1)

        result = solve_constrained_qp(H, g, A, b)
        assert np.all(np.diff(result.beta) >= -1e-10)

    def test_returns_active_set(self):
        """Result includes the active constraint indices."""
        H = np.eye(2)
        g = np.array([-1.0, -2.0])
        A = np.eye(2)
        b = np.zeros(2)

        result = solve_constrained_qp(H, g, A, b)
        np.testing.assert_allclose(result.beta, [0.0, 0.0], atol=1e-10)
        assert set(result.active_set) == {0, 1}
