"""Tests for the active-set constrained penalized least-squares solver."""

import numpy as np
import pytest

from superglm.solvers.constrained_qp import (
    _is_feasible,
    _project_feasible,
    solve_constrained_qp,
)
from superglm.solvers.rank import decompose_gram


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


class TestRankDeficientHessian:
    """Singular H is rank-truncated by the shared policy, not raised on."""

    def test_unconstrained_singular_h_solves_the_normal_equations(self):
        """m == 0 with rank-deficient H must not raise."""
        H = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        g = np.array([1.0, 1.0])  # in range(H), so a solution exists

        result = solve_constrained_qp(H, g, np.zeros((0, 2)), np.zeros(0))

        assert np.all(np.isfinite(result.beta))
        # A rank-truncated solve of a consistent singular system is still an
        # exact solution of the normal equations, just not the minimum-norm one.
        np.testing.assert_allclose(H @ result.beta, g, atol=1e-12)
        assert result.converged

    def test_singular_h_returns_finite_solution_instead_of_raising(self):
        """Rank-deficient H must go through the rank policy, not raise LinAlgError."""
        H = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        g = np.array([1.0, 1.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([-10.0])  # inactive at the unconstrained solution

        result = solve_constrained_qp(H, g, A, b)

        assert np.all(np.isfinite(result.beta))
        np.testing.assert_allclose(H @ result.beta, g, atol=1e-12)
        assert np.all(A @ result.beta - b >= -1e-10)
        assert result.converged

    def test_singular_h_with_a_binding_constraint_reaches_the_optimum(self):
        """The active-set loop must also survive a rank-deficient H.

        Objective 0.5*(x1 + x2)^2 - (x1 + x2) subject to x1 >= 5 is minimized
        by any point with x1 == 5 and x1 + x2 == 1.
        """
        H = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        g = np.array([1.0, 1.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([5.0])  # binding

        result = solve_constrained_qp(H, g, A, b)

        assert np.all(np.isfinite(result.beta))
        assert result.beta[0] >= 5.0 - 1e-10
        np.testing.assert_allclose(float(np.sum(result.beta)), 1.0, atol=1e-10)
        assert result.converged

    def test_indefinite_h_error_names_this_function(self):
        """The rank policy's ValueError must be re-labelled for the caller."""
        H = np.array([[1.0, 0.0], [0.0, -1.0]])  # materially indefinite
        g = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="solve_constrained_qp requires a usable PSD H"):
            solve_constrained_qp(H, g, np.zeros((0, 2)), np.zeros(0))

    def test_well_conditioned_solution_is_unchanged_by_the_rank_policy(self):
        """The rank policy must not perturb a well-conditioned unconstrained solve."""
        H = np.array([[4.0, 1.0], [1.0, 3.0]])
        g = np.array([1.0, 2.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([-10.0])  # inactive

        result = solve_constrained_qp(H, g, A, b)

        np.testing.assert_allclose(result.beta, np.linalg.solve(H, g), rtol=1e-12)
        assert result.converged


class TestConvergenceFlag:
    """``QPResult.converged`` reports what its name says."""

    def test_iteration_starved_qp_reports_non_convergence(self):
        H = np.eye(3)
        g = np.array([5.0, 5.0, 5.0])
        A = -np.eye(3)
        b = np.array([-0.1, -0.1, -0.1])

        result = solve_constrained_qp(H, g, A, b, max_iter=1)

        assert not result.converged
        assert result.n_iter == 1
        # Exhaustion outranks feasibility.  This truncated point happens to be
        # feasible, and must still report non-convergence: the loop never
        # reached its stationarity/multiplier test, so nothing is certified.
        assert np.all(A @ result.beta - b >= -1e-12)

    def test_infeasible_projection_reports_non_convergence(self):
        """Mutually contradictory constraints cannot be projected onto."""
        H = np.eye(1)
        g = np.array([0.0])
        A = np.array([[1.0], [-1.0]])
        b = np.array([1.0, 1.0])  # x >= 1 and -x >= 1

        result = solve_constrained_qp(H, g, A, b)

        assert not result.converged
        # The flag is not merely "ran out of iterations": the active-set loop
        # terminated on its own KKT test, at a point that is still infeasible.
        assert result.n_iter < 200
        assert np.min(A @ result.beta - b) < -1e-6

    def test_feasible_solve_still_reports_convergence(self):
        """The projection path must not report spurious non-convergence."""
        H = np.eye(3)
        g = np.array([3.0, 2.0, 1.0])
        A = np.diff(np.eye(3), axis=0)  # monotone increasing; g is decreasing
        b = np.zeros(2)

        result = solve_constrained_qp(H, g, A, b)

        assert result.converged
        assert np.all(np.diff(result.beta) >= -1e-10)

    def test_projection_budget_overrun_that_the_loop_repairs_reports_convergence(self):
        """``converged`` describes the point returned, not the starting point.

        With more non-negativity constraints than the 100 projection sweeps
        can repair -- the ``A = np.eye(q)`` shape used by SCOP's solver-space
        ``qp_initialize`` -- the projection hands the active-set loop an
        infeasible start, and the loop then reaches a genuine feasible KKT
        point.  Latching the projection's own verdict reports a spurious
        non-convergence here and makes both call sites warn misleadingly.
        """
        p = 130
        rng = np.random.default_rng(32)
        M = rng.standard_normal((p, p))
        H = M.T @ M / p + np.eye(p)
        g = rng.standard_normal(p) - 0.6  # bias so >100 constraints are violated
        A = np.eye(p)
        b = np.zeros(p)

        # Precondition: the projection really does exhaust its sweep budget.
        beta_unc = np.linalg.solve(H, g)
        projected = _project_feasible(beta_unc, A, b, 1e-12)
        assert np.min(A @ projected - b) < -1e-9, "projection did not overrun its budget"

        result = solve_constrained_qp(H, g, A, b)

        # The loop terminated on its own KKT test, not on max_iter...
        assert result.n_iter < 200
        # ...at a point that is genuinely feasible...
        assert np.all(A @ result.beta - b >= -1e-12)
        # ...so this is a converged solve.
        assert result.converged


class TestCallSiteWarnings:
    """The three call sites surface ``converged`` instead of discarding it."""

    @staticmethod
    def _non_converging(calls):
        """Wrap the real solver, forcing ``converged=False`` on every call."""
        from superglm.solvers.constrained_qp import solve_constrained_qp as real_solve

        def fake_solve(*args, **kwargs):
            calls.append(1)
            result = real_solve(*args, **kwargs)
            result.converged = False
            return result

        return fake_solve

    def test_scop_raw_qp_initialize_warns_on_non_convergence(self, caplog, monkeypatch):
        import logging

        from superglm.solvers import scop

        calls: list[int] = []
        # raising=True (the default): if the symbol is ever moved back inside
        # the function body, this patch fails loudly instead of silently
        # leaving the real solver in place and vacuously passing.
        monkeypatch.setattr(scop, "solve_constrained_qp", self._non_converging(calls))

        reparam = scop.build_scop_reparam(6, kind="increasing")
        rng = np.random.default_rng(3)
        B = rng.normal(size=(40, 6))
        y = rng.normal(size=40)

        with caplog.at_level(logging.WARNING, logger="superglm.solvers.scop"):
            reparam.qp_initialize(B, y)

        assert calls, "the patched solver was never called"
        assert "did not converge" in caplog.text
        # The two SCOP initialization paths must be distinguishable in a log.
        assert "raw-space" in caplog.text
        assert "solver-space" not in caplog.text

    def test_scop_solver_qp_initialize_warns_on_non_convergence(self, caplog, monkeypatch):
        import logging

        from superglm.solvers import scop

        calls: list[int] = []
        monkeypatch.setattr(scop, "solve_constrained_qp", self._non_converging(calls))

        reparam = scop.build_scop_solver_reparam(6, kind="increasing")
        rng = np.random.default_rng(4)
        B = rng.normal(size=(40, reparam.q))
        y = rng.normal(size=40)

        with caplog.at_level(logging.WARNING, logger="superglm.solvers.scop"):
            reparam.qp_initialize(B, y)

        assert calls, "the patched solver was never called"
        assert "did not converge" in caplog.text
        assert "solver-space" in caplog.text
        assert "raw-space" not in caplog.text

    def test_irls_direct_warns_when_the_constrained_qp_does_not_converge(self, caplog, monkeypatch):
        import logging

        import pandas as pd

        from superglm import Constraint, SuperGLM
        from superglm.families import Gaussian
        from superglm.features.spline import BSplineSmooth
        from superglm.solvers import irls_direct

        calls: list[int] = []
        monkeypatch.setattr(irls_direct, "solve_constrained_qp", self._non_converging(calls))

        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 1, 80))
        y = 2.0 * x + rng.normal(0, 0.1, 80)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": BSplineSmooth(n_knots=6, constraint=Constraint.fit.increasing)},
        )

        with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
            model.fit(df[["x"]], df["y"])

        assert calls, "the patched solver was never called"
        assert "constrained QP did not converge" in caplog.text

    def test_irls_direct_qp_warning_is_latched_to_one_per_fit(self, caplog, monkeypatch):
        """The warning lives inside the IRLS loop; it must not repeat per iteration.

        ``irls_direct`` warns fire-once by convention -- the neighbouring SVD
        warning uses an ``== 3`` equality latch. Without a latch here, a fit
        whose QP never converges emits one identical WARNING per IRLS
        iteration, up to ``max_iter`` (default 200).
        """
        import logging

        import pandas as pd

        from superglm import Constraint, SuperGLM
        from superglm.families import Binomial
        from superglm.features.spline import BSplineSmooth
        from superglm.solvers import irls_direct

        calls: list[int] = []
        monkeypatch.setattr(irls_direct, "solve_constrained_qp", self._non_converging(calls))

        # Binomial needs several IRLS iterations, so the unlatched code would
        # warn several times; Gaussian converges in ~2 and barely discriminates.
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 1, 200))
        y = (rng.uniform(size=200) < 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))).astype(float)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Binomial(),
            selection_penalty=0,
            features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
        )

        with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
            model.fit(df[["x"]], df["y"])

        # Precondition: enough non-converging QP solves that an unlatched
        # warning would be clearly visible as a repeat.
        assert len(calls) >= 5, f"only {len(calls)} QP solves; test cannot discriminate"

        warnings = [
            record
            for record in caplog.records
            if "constrained QP did not converge" in record.getMessage()
        ]
        assert len(warnings) == 1, f"expected exactly 1 warning, got {len(warnings)}"


class TestInconsistentNormalEquations:
    """A rank-deficient H with g outside range(H) must not answer silently.

    ``decomposition.solve`` is a pseudo-inverse, so it returns a projection for
    a system that has no solution at all.  Routing the pure-H solves through
    the rank policy turned the pre-branch ``LinAlgError`` into a plausible
    wrong answer for this case; these tests pin the loud behaviour back.
    """

    # H = diag(1, 0) has null direction e2, and g = (0, 1) has all of its mass
    # there, so the objective 0.5*x1^2 - x2 decreases without bound as x2 grows.
    H_INCONSISTENT = np.diag([1.0, 0.0])
    G_INCONSISTENT = np.array([0.0, 1.0])

    def test_unconstrained_inconsistent_system_raises(self):
        with pytest.raises(ValueError, match="component in null\\(H\\)"):
            solve_constrained_qp(
                self.H_INCONSISTENT, self.G_INCONSISTENT, np.zeros((0, 2)), np.zeros(0)
            )

    def test_constrained_inconsistent_system_raises_rather_than_projecting(self):
        """The bounded-by-constraints case must raise too, not return H+g.

        With ``x2 <= 1`` the true optimum is ``[0, 1]`` with objective -1, but
        the pseudo-inverse answer ``[0, 0]`` has objective 0 and is feasible,
        so it would sail through the unconstrained-probe early return.  The
        active-set loop cannot recover it either: from an empty active set the
        only direction it forms is ``H+g - beta``, which lies in range(H) and
        is exactly zero here, so the loop stalls at the wrong point.
        """
        A = np.array([[0.0, -1.0]])  # -x2 >= -1, i.e. x2 <= 1
        b = np.array([-1.0])

        with pytest.raises(ValueError, match="unbounded below along that direction"):
            solve_constrained_qp(self.H_INCONSISTENT, self.G_INCONSISTENT, A, b)

    def test_error_names_the_rank_and_the_residual(self):
        with pytest.raises(ValueError) as excinfo:
            solve_constrained_qp(
                self.H_INCONSISTENT, self.G_INCONSISTENT, np.zeros((0, 2)), np.zeros(0)
            )
        message = str(excinfo.value)
        assert "rank 1 of 2" in message
        assert "Regularize H" in message

    def test_consistent_rank_deficient_system_still_solves(self):
        """The rank-truncation behaviour this branch added must be preserved.

        Same rank-deficient H, but g in range(H): the pseudo-inverse answer is
        a genuine solution of the normal equations and must not raise.
        """
        H = self.H_INCONSISTENT
        g = np.array([2.0, 0.0])  # orthogonal to null(H) = span(e2)

        result = solve_constrained_qp(H, g, np.zeros((0, 2)), np.zeros(0))

        np.testing.assert_allclose(H @ result.beta, g, atol=1e-12)
        assert result.converged

    @staticmethod
    def _ill_conditioned_rank_deficient(retained_condition, seed, *, null_mass=0.0):
        """H with an exact null direction and a controlled retained condition.

        ``g`` is built inside the retained span with O(1) coefficients, so it
        is consistent yet drives a large ``beta`` -- which is exactly what
        inflates a solve residual without making the system inconsistent.
        """
        p = 4
        rng = np.random.default_rng(seed)
        basis, _ = np.linalg.qr(rng.standard_normal((p, p)))
        eigenvalues = np.concatenate([[1.0], np.full(p - 2, 1.0 / retained_condition), [0.0]])
        H = basis @ np.diag(eigenvalues) @ basis.T
        H = 0.5 * (H + H.T)
        g = basis[:, : p - 1] @ rng.standard_normal(p - 1)
        if null_mass:
            g = g + null_mass * np.linalg.norm(g) * basis[:, p - 1]
        return H, g

    @pytest.mark.parametrize("retained_condition", [1e9, 1e10, 1e12])
    def test_consistent_but_ill_conditioned_systems_still_solve(self, retained_condition):
        """The gate must not refuse systems the rank policy can solve.

        ``decompose_gram`` truncates at ``gram_rcond = eps``, so it retains
        blocks conditioned far beyond ``factor_rcond``.  A residual-based gate
        refused almost all of these, because a residual is amplified by exactly
        that retained condition number.
        """
        refused = 0
        checked = 0
        for seed in range(20):
            H, g = self._ill_conditioned_rank_deficient(retained_condition, seed)
            if decompose_gram(H).rank >= H.shape[0]:
                continue  # gate not reached; nothing to assert
            checked += 1
            try:
                result = solve_constrained_qp(H, g, np.zeros((0, 4)), np.zeros(0))
            except ValueError:
                refused += 1
                continue
            assert np.all(np.isfinite(result.beta))

        assert checked >= 15, f"only {checked} systems reached the gate"
        assert refused == 0, f"{refused}/{checked} consistent systems refused"

    @pytest.mark.parametrize("retained_condition", [1e9, 1e10, 1e12])
    def test_inconsistent_systems_are_still_caught_at_the_same_conditions(self, retained_condition):
        """Loosening the threshold must not blind the gate."""
        caught = 0
        checked = 0
        for seed in range(20):
            H, g = self._ill_conditioned_rank_deficient(retained_condition, seed, null_mass=0.25)
            if decompose_gram(H).rank >= H.shape[0]:
                continue
            checked += 1
            with pytest.raises(ValueError, match="component in null"):
                solve_constrained_qp(H, g, np.zeros((0, 4)), np.zeros(0))
            caught += 1

        assert checked >= 15, f"only {checked} systems reached the gate"
        assert caught == checked

    def test_ridge_regularization_is_a_workable_escape(self):
        """The remedy the error message recommends must actually work."""
        H = self.H_INCONSISTENT + 1e-6 * np.eye(2)

        result = solve_constrained_qp(H, self.G_INCONSISTENT, np.zeros((0, 2)), np.zeros(0))

        assert np.all(np.isfinite(result.beta))
        np.testing.assert_allclose(H @ result.beta, self.G_INCONSISTENT, atol=1e-9)


class TestSymmetrization:
    """H is symmetrized once and used consistently on both solve paths."""

    def test_asymmetric_h_minimizes_its_symmetric_part(self):
        """Before the fix the decomposition saw 0.5*(H+H') but the KKT blocks
        saw raw H, so the two paths minimized different quadratics."""
        H = np.array([[2.0, 2.0], [0.0, 2.0]])  # symmetric part [[2,1],[1,2]]
        g = np.zeros(2)
        A = np.array([[1.0, 0.0]])  # x1 >= 1, binding
        b = np.array([1.0])

        result = solve_constrained_qp(H, g, A, b)

        # Optimum of 0.5 x' sym(H) x subject to x1 >= 1 is [1, -0.5].
        np.testing.assert_allclose(result.beta, [1.0, -0.5], atol=1e-10)
        assert result.converged

    def test_symmetric_input_is_untouched(self):
        """0.5*(H + H.T) must be bitwise identity for an exactly symmetric H."""
        rng = np.random.default_rng(11)
        M = rng.standard_normal((5, 5))
        H = M.T @ M + np.eye(5)
        H = 0.5 * (H + H.T)  # force exact symmetry
        g = rng.standard_normal(5)

        result = solve_constrained_qp(H, g, np.zeros((0, 5)), np.zeros(0))

        assert np.array_equal(0.5 * (H + H.T), H)
        np.testing.assert_allclose(result.beta, np.linalg.solve(H, g), rtol=1e-12)


class TestFeasibilityToleranceScaling:
    """The feasibility test is relative, so large constraint rows do not
    read as violations at a genuine KKT point."""

    def test_badly_scaled_constraints_do_not_report_spurious_non_convergence(self):
        """A blocking step lands on a constraint only to ~eps*|A_i @ beta|.

        With rows of order 1e3 that rounding already exceeds an absolute 1e-12
        tolerance, so the unscaled test called a genuine KKT point infeasible
        and made all three call sites warn.
        """
        rng = np.random.default_rng(0)
        p = 6
        M = rng.standard_normal((p, p))
        H = M.T @ M + 0.5 * np.eye(p)
        scale = 1e3
        g = rng.standard_normal(p) * scale
        A = np.diff(np.eye(p), axis=0)  # monotone increasing
        b = rng.standard_normal(p - 1) * scale

        result = solve_constrained_qp(H, g, A, b)

        raw_slack = A @ result.beta - b
        # Precondition: an absolute 1e-12 test really would call this a
        # violation, so the test exercises the scaling rather than passing
        # for free.
        assert np.min(raw_slack) < -1e-12, f"min slack {np.min(raw_slack):.3e} is not tight"
        # ...but it is a rounding-level violation relative to the row scale.
        row_scale = np.maximum(1.0, np.maximum(np.abs(b), np.abs(A @ result.beta)))
        assert np.min(raw_slack / row_scale) >= -1e-12
        assert result.converged

    def test_scaling_does_not_mask_a_real_violation(self):
        """A genuinely infeasible point must still report non-convergence."""
        H = np.eye(1)
        g = np.array([0.0])
        A = np.array([[1.0], [-1.0]])
        b = np.array([1e6, 1e6])  # x >= 1e6 and -x >= 1e6, badly scaled

        result = solve_constrained_qp(H, g, A, b)

        assert not result.converged

    def test_projection_and_convergence_share_one_feasibility_rule(self):
        """``_project_feasible`` hardcoded ``-1e-12`` while the caller used
        ``-tol``; they must now apply one predicate, since disagreement about
        what "feasible" means is what produced the spurious-warning class.

        The probe point sits a rounding-level distance below its constraint:
        1e-9 in absolute terms, which an absolute 1e-12 test rejects, but
        1e-13 relative to the row scale, which the shared rule accepts.
        """
        A = np.array([[1.0e4]])
        b = np.array([1.0e4])
        beta = np.array([1.0 - 1.0e-13])  # A @ beta - b == -1e-9

        raw_slack = float((A @ beta - b)[0])
        assert raw_slack < -1e-12, f"probe slack {raw_slack:.3e} is not tight enough"

        # Both the projection's stopping test and the caller's convergence
        # test must accept it -- so the projection leaves the point alone.
        assert _is_feasible(A, beta, b, 1e-12)
        np.testing.assert_array_equal(_project_feasible(beta, A, b, 1e-12), beta)
