"""Tests for the active-set constrained penalized least-squares solver."""

import numpy as np
import pytest

from superglm._fit_trace import MemoryTraceSink, NullTraceSink, TraceRun
from superglm.solvers.constrained_qp import (
    BLOCKING_TRACE_CHANNEL,
    _consistency_floor,
    _feasibility_scale,
    _feasibility_slack,
    _is_feasible,
    _null_space_mass,
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

        # An explicit generous max_iter: the property under test is "a
        # projection overrun that the loop repairs reports converged", not
        # "it repairs it within the default 200 iterations", and the
        # stationarity test is absolute (filed follow-up) so the count moves
        # with BLAS.
        result = solve_constrained_qp(H, g, A, b, max_iter=5000)

        # The loop terminated on its own KKT test, not on max_iter...
        assert result.n_iter < 5000
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

    def test_error_names_the_rank_and_the_null_mass(self):
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


class TestLowConditionConsistency:
    """The spectral floor must clear the eigensolver's own null-basis roundoff.

    The retained-condition term models error *amplification*, which vanishes as
    the retained block becomes well conditioned -- but ``eigh``'s backward
    error does not, so at low condition the floor bottomed out below the noise
    it was meant to sit above and refused consistent systems.
    """

    def test_well_conditioned_rank_one_system_is_not_refused(self):
        """Reported by review; reproduces platform-dependently.

        On this box the spectral mass measures ~5e-16 against the old 7.1e-15
        floor and passes; on the reporter's it measured 8.819e-15 and raised.
        That the same input lands either side of the floor depending on the
        eigensolver is itself the argument that the floor was inside the noise.
        """
        R = np.array([[-1.3181547011385448, 0.00814244848194327, 0.47345700571580956]])
        H = R.T @ R  # rank 1 of 3
        g = H @ np.array([0.3183325456193874, 0.4026397715796121, 0.9365457438408186])

        decomposition = decompose_gram(H)
        assert decomposition.rank < decomposition.width, "gate not reached"
        _, spectral = _null_space_mass(decomposition, g)
        floor = _consistency_floor(decomposition)
        assert floor > 20 * spectral, (
            f"floor {floor:.3e} leaves only {floor / max(spectral, 1e-300):.1f}x "
            f"over a measured roundoff of {spectral:.3e}"
        )

        result = solve_constrained_qp(H, g, np.zeros((0, 3)), np.zeros(0))

        np.testing.assert_allclose(H @ result.beta, g, atol=1e-12)
        assert result.converged

    @pytest.mark.parametrize("width", [3, 5, 8, 16, 32, 64, 120, 180])
    def test_consistent_systems_at_unit_condition_are_not_refused(self, width):
        """The floor's dimension term, checked across the production range.

        ``decomposition.width`` is the full parameter count, and the monotone
        ``irls_direct`` caller builds ``A`` over the whole coefficient vector,
        so in-tree widths reach 105-180.  Measured: the null-basis roundoff is
        *flat* in absolute terms across 16-180 (0.2-2.5 eps) rather than linear
        in width, so a linear floor over-provisions at the top of the range --
        the binding case is small widths, where it reaches 86.5 eps at width 5.
        """
        refused = 0
        checked = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            basis, _ = np.linalg.qr(rng.standard_normal((width, width)))
            eigenvalues = np.zeros(width)
            eigenvalues[0] = 1.0  # rank 1, perfectly conditioned retained block
            H = basis @ np.diag(eigenvalues) @ basis.T
            H = 0.5 * (H + H.T)
            g = H @ rng.standard_normal(width)  # in range(H) by construction
            if decompose_gram(H).rank >= width:
                continue
            checked += 1
            try:
                solve_constrained_qp(H, g, np.zeros((0, width)), np.zeros(0))
            except ValueError:
                refused += 1

        assert checked >= 15, f"only {checked} systems reached the gate"
        assert refused == 0, f"{refused}/{checked} consistent systems refused"


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


class TestStructuralAliasConsistency:
    """A structurally zero column's null vector is exact at every conditioning.

    ``rank._null_basis`` stacks exact unit vectors (for structurally zero
    columns) together with computed spectral directions whose accuracy decays
    as ``eps * retained condition``.  Measuring both against the spectral floor
    let an ill-conditioned retained block desensitize the exact half, so a
    genuinely unbounded objective was accepted as converged.
    """

    @staticmethod
    def _structural_alias_with_ill_conditioned_block(retained_condition, structural_mass, seed=0):
        """``blkdiag(K, 0)``: ``K`` carries a spectral truncation at the given
        retained condition, and the last column is structurally zero -- an
        exact unit null vector.  ``g`` is consistent on the ``K`` block and
        carries ``structural_mass`` on the exact null direction.
        """
        rng = np.random.default_rng(seed)
        basis, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        block = basis @ np.diag([1.0, 1.0 / retained_condition, 0.0]) @ basis.T
        block = 0.5 * (block + block.T)
        H = np.zeros((4, 4))
        H[:3, :3] = block
        g = np.zeros(4)
        g[:3] = basis[:, :2] @ rng.standard_normal(2)
        g[3] = structural_mass * np.linalg.norm(g[:3])
        return H, g

    @pytest.mark.parametrize(
        ("retained_condition", "structural_mass"),
        [(1e10, 1e-6), (1e11, 1e-5), (1e13, 1e-3)],
    )
    def test_structural_alias_caught_despite_ill_conditioned_retained_block(
        self, retained_condition, structural_mass
    ):
        H, g = self._structural_alias_with_ill_conditioned_block(
            retained_condition, structural_mass
        )
        decomposition = decompose_gram(H)

        # Preconditions. Without these the test could pass under a single
        # shared floor and prove nothing.
        assert decomposition.rank < decomposition.width, "gate not reached"
        assert _consistency_floor(decomposition) > structural_mass, (
            "the spectral floor is tighter than the injected mass here, so a "
            "shared floor would have caught this anyway"
        )
        structural, spectral = _null_space_mass(decomposition, g)
        assert spectral < structural, "the injected mass must be structural, not spectral"

        with pytest.raises(ValueError, match="structurally aliased column"):
            solve_constrained_qp(H, g, np.zeros((0, 4)), np.zeros(0))

    @pytest.mark.parametrize("retained_condition", [1e10, 1e11, 1e13])
    def test_consistent_system_with_a_structural_alias_still_solves(self, retained_condition):
        """The tight structural floor must not over-fire on a consistent g.

        A real caller's structural entry is *exactly* zero -- an identically
        zero design column gives an identically zero inner product -- so the
        floor's slack is defensive rather than load-bearing.  The probe uses a
        roundoff-scale ``1e-17`` instead of exact zero so that the assertion
        actually exercises the slack: at a floor of 0 this would be refused.
        """
        H, g = self._structural_alias_with_ill_conditioned_block(
            retained_condition, structural_mass=1e-17
        )
        decomposition = decompose_gram(H)
        assert decomposition.rank < decomposition.width, "gate not reached"
        structural, _ = _null_space_mass(decomposition, g)
        assert 0.0 < structural < 1e-15, f"probe mass {structural:.2e} is not roundoff-scale"

        result = solve_constrained_qp(H, g, np.zeros((0, 4)), np.zeros(0))

        assert np.all(np.isfinite(result.beta))
        assert result.converged


class TestLoopFeasibilityRouting:
    """The active-set loop uses the same feasibility measure as its boundaries.

    While the boundaries were relative and the loop body absolute, the loop
    treated rows as violated that ``_is_feasible`` considered satisfied, then
    blocked on them with a negative slack -- a backward step, which widens the
    deferred negative-``alpha`` bug rather than merely inheriting it.
    """

    def test_blocking_gate_is_scaled_so_a_numerically_still_row_is_skipped(self):
        """The *gate* is scaled: that is the half of the routing that changed.

        A directional derivative that is numerically zero relative to its row
        must not make the row a blocking candidate.  Under the absolute gate it
        did, and the row then contributed a negative ``alpha`` -- a backward
        step.  Pure arithmetic, so no BLAS can move it.
        """
        products = np.array([1.0e6])
        b = np.array([0.0])
        raw_step = np.array([-1.0e-9])
        tol = 1e-12

        scale = _feasibility_scale(products, b)
        assert scale[0] == 1.0e6, "fixture no longer exercises a scale above 1"
        assert raw_step[0] < -tol, "the absolute gate would make this a candidate"
        assert (raw_step / scale)[0] > -tol, "the scaled gate must skip it"

    def test_blocking_ratio_is_taken_from_the_raw_pair(self):
        """The *ratio* is not scaled: that half must stay bitwise as it was.

        Dividing numerator and denominator by the same row scale is
        algebraically neutral and numerically is not -- it rounds twice where
        the raw quotient rounds once.  On a row with slack above 1 the scaled
        numerator collapses to exactly 1.0.  These are measured values where
        the two forms genuinely disagree in the last bit.
        """
        products, b_i, raw_step = 3.0086616993496036, 0.0, -0.13181179586621902
        scale = max(1.0, abs(b_i), abs(products))

        raw_quotient = (products - b_i) / -raw_step
        scaled_quotient = ((products - b_i) / scale) / -(raw_step / scale)

        assert scale > 1.0
        assert (products - b_i) / scale == 1.0, "the scaled numerator collapses"
        assert raw_quotient != scaled_quotient, "fixture no longer demonstrates the double rounding"
        # The solver must use the raw form; see the ratio block's comment.
        assert raw_quotient == 22.825435914728093
        assert scaled_quotient == 22.82543591472809

    def test_an_infeasible_returned_point_is_never_reported_converged(self):
        """The converse implication, which is not a tautology.

        Asserting ``converged -> feasible`` would be: ``converged`` *is* that
        same ``_is_feasible`` call on the same arrays, so it can only agree
        with itself.  ``infeasible -> not converged`` is different, because the
        exhaustion return reaches ``False`` without consulting feasibility at
        all, and because a regression that hardcoded ``converged=True`` at
        either in-loop return would violate it.  No iteration count and no
        convergence outcome is asserted, so nothing here moves with BLAS.
        """
        fixtures = []

        # Mutually infeasible: returns an infeasible point, must report False.
        fixtures.append(
            (np.eye(1), np.array([0.0]), np.array([[1.0], [-1.0]]), np.array([1.0, 1.0]))
        )
        # Badly scaled, nonzero b: the scaling is observable here.
        rng = np.random.default_rng(8)
        p = 5
        M = rng.standard_normal((p, p))
        for scale in (1e2, 1e4):
            H = M.T @ M + 0.5 * np.eye(p)
            fixtures.append(
                (
                    H,
                    rng.standard_normal(p) * scale,
                    np.diff(np.eye(p), axis=0),
                    rng.standard_normal(p - 1) * scale,
                )
            )

        saw_infeasible = False
        for H, g, A, b in fixtures:
            result = solve_constrained_qp(H, g, A, b)
            assert np.all(np.isfinite(result.beta))
            if not _is_feasible(A, result.beta, b, 1e-12):
                saw_infeasible = True
                assert not result.converged, "an infeasible point was reported as a converged solve"

        assert saw_infeasible, "no fixture returned an infeasible point; the implication is vacuous"

    def test_zero_rhs_makes_the_scaling_exactly_inert(self):
        """Every in-tree caller passes b = 0; pin that this is the inert case.

        With ``b_i == 0`` the per-row scale is ``max(1, |A_i @ beta|)``.
        Measured across 7 monotone/convex/SCOP fits spanning Gaussian,
        Binomial and Poisson -- 15 QP calls, 189 constraint rows -- the max
        ``|A_i @ beta|`` inside the ratio block is 0.72, so the scale is
        exactly 1.0 there and the scaled quantities are bitwise the raw ones.
        That bound covers ``A @ beta`` inside the ratio block only; the
        feasibility test keys off ``A @ beta_new`` and ``_project_feasible``
        off its own iterates, which are different products and are covered by
        the byte-for-byte fitted-value check rather than by this bound.
        """
        rng = np.random.default_rng(3)
        p = 5
        M = rng.standard_normal((p, p))
        H = M.T @ M + 0.5 * np.eye(p)
        g = rng.standard_normal(p)
        A = np.diff(np.eye(p), axis=0)
        b = np.zeros(p - 1)

        result = solve_constrained_qp(H, g, A, b)

        products = A @ result.beta
        assert np.max(np.abs(products)) <= 1.0, "fixture no longer exercises scale == 1"
        np.testing.assert_array_equal(_feasibility_scale(products, b), np.ones(p - 1))
        np.testing.assert_array_equal(_feasibility_slack(A, result.beta, b), products - b)


class TestConstraintBoundedInconsistentSystem:
    """The bounded-but-inconsistent case is refused, deliberately.

    ``H = diag(1, 0)``, ``g = (0, 1)``, ``x2 <= 1`` has the finite optimum
    ``(0, 1)``.  Reaching it needs a null-space descent direction, which is a
    filed follow-up rather than a capability this solver has; master refused
    the same input with ``LinAlgError``, so refusing is not a regression.
    This test exists so that implementing the solve shows up as a deliberate
    change rather than a silent one.
    """

    def test_constraint_bounded_case_is_refused_and_says_so(self):
        H = np.diag([1.0, 0.0])
        g = np.array([0.0, 1.0])
        A = np.array([[0.0, -1.0]])  # -x2 >= -1, i.e. x2 <= 1
        b = np.array([-1.0])

        # The finite optimum exists and is strictly better than the projection.
        optimum = np.array([0.0, 1.0])
        assert 0.5 * optimum @ H @ optimum - g @ optimum == -1.0

        with pytest.raises(ValueError) as excinfo:
            solve_constrained_qp(H, g, A, b)

        message = str(excinfo.value)
        # The message must not claim the problem itself is unbounded: the
        # constraints here do bound it.
        assert "unconstrained objective is unbounded below" in message
        assert "constraints may still bound the problem" in message
        assert "null-space descent direction" in message


class TestBlockingDecisionTrace:
    """Pin the routing *mechanism* through the house tracing seam.

    Routing the loop's gates through ``_feasibility_scale`` is not observable
    as a numeric outcome: with the raw ratio it is bitwise inert on every
    nonzero-``b`` probe, and the only population that differs is ``b = 0``
    exhausted solves whose paths are chaotic.  Asserting anything there is how
    a fixture that passed locally exhausted ``max_iter`` on CI's BLAS.

    So these tests assert *properties of each decision*, re-derived by the
    hook rather than echoed from the loop.  A property must hold whatever path
    the search takes, so no assertion here can depend on BLAS: only the
    preconditions count records, and they carry wide margins.
    """

    @staticmethod
    def _decisions(scale=100.0, seeds=range(8), p=5, zero_rhs=False, order=1):
        """Aggregate blocking decisions over several small, fast fixtures.

        Aggregating rather than pinning one fixture is deliberate: a different
        BLAS may shift an individual search, but the batch still produces
        decisions, so the preconditions stay satisfied.
        """
        records = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            M = rng.standard_normal((p, p))
            H = M.T @ M + 0.5 * np.eye(p)
            g = rng.standard_normal(p) * scale
            A = np.diff(np.eye(p), n=order, axis=0)
            b = np.zeros(A.shape[0]) if zero_rhs else rng.standard_normal(A.shape[0]) * scale
            sink = MemoryTraceSink()
            solve_constrained_qp(H, g, A, b, _trace_run=TraceRun(f"qp-{seed}", sink=sink))
            for event in sink.events:
                assert event.event_kind == "step_decision"
                assert event.channel == BLOCKING_TRACE_CHANNEL
                records.append(event.payload)
        return records

    def test_no_step_the_convergence_test_accepts_reaches_the_blocking_search(self):
        """The full-step gate must be the convergence test, not an absolute one.

        Where the two disagree, the loop blocks on a row ``_is_feasible``
        considers satisfied and takes a negative ``alpha`` -- a backward step.

        The two can only disagree when ``tol * |b_i|`` exceeds ``tol``, so this
        needs a large ``b``; at ``1e2`` the absolute and scaled full-step tests
        agree on every fixture and the assertion would hold for either
        implementation.  Measured: every seed here witnesses the disagreement
        when the gate is reverted.
        """
        records = self._decisions(scale=1e6, seeds=range(6))

        assert len(records) >= 10, f"only {len(records)} decisions; no margin"
        accepted = [r for r in records if r["full_step_is_feasible"]]
        assert not accepted, (
            f"{len(accepted)} of {len(records)} blocking searches were entered "
            "for a step the convergence test accepts"
        )

    def test_recorded_alpha_is_the_raw_quotient_for_the_blocking_row(self):
        """The ratio must be the raw quotient, not the doubly-rounded one.

        Recomputed from the inputs the hook recorded, so this asserts the
        documented relation rather than a stored number.
        """
        records = self._decisions()

        checked = 0
        scaled_rows = 0
        distinguishing = 0
        for record in records:
            blocking = record["blocking_row"]
            if blocking < 0:
                continue
            index = record["considered_rows"].index(blocking)
            products = record["row_products"][index]
            b_i = record["row_b"][index]
            raw_step = record["row_raw_step"][index]
            row_scale = max(1.0, abs(b_i), abs(products))
            if row_scale > 1.0:
                scaled_rows += 1
                if (products - b_i) / -raw_step != ((products - b_i) / row_scale) / -(
                    raw_step / row_scale
                ):
                    distinguishing += 1
            assert record["alpha"] == (products - b_i) / -raw_step, (
                f"alpha {record['alpha']!r} is not the raw quotient "
                f"{(products - b_i) / -raw_step!r} for row {blocking}"
            )
            checked += 1

        assert checked >= 10, f"only {checked} blocking decisions; no margin"
        # Without a row whose scale exceeds 1 the two forms agree bitwise and
        # the assertion above is satisfied by any implementation.
        # Sharper than counting rows with scale > 1: require that the two
        # quotient forms actually disagree in floating point on some recorded
        # row, which is the thing the assertion above distinguishes.
        assert distinguishing >= 1, (
            f"scale > 1 on {scaled_rows} blocking rows but the raw and scaled "
            "quotients agree bitwise on all of them; the fixtures no longer "
            "distinguish the two forms"
        )

    def test_recorded_operands_agree_with_an_independent_slack(self):
        """Cross-check the ratio's *operands*, not just the ratio.

        ``products`` and ``raw_step`` are handed to the hook by the loop, so
        the alpha test -- which recomputes its expectation from those same
        recorded operands -- is self-consistent under any mutation of them.
        A slip such as ``products = A @ beta_new`` would give the loop a wrong
        slack, a wrong alpha and a wrong step, and that test would still pass.

        ``row_scaled_slack`` comes from an independent ``_feasibility_slack``
        call on the loop's *current* iterate, so reconstructing the raw slack
        from it and comparing against the recorded operands catches exactly
        that class of slip.
        """
        records = self._decisions(scale=1e6, seeds=range(6))

        checked = 0
        for record in records:
            for products, b_i, scaled_slack in zip(
                record["row_products"], record["row_b"], record["row_scaled_slack"]
            ):
                row_scale = max(1.0, abs(b_i), abs(products))
                reconstructed = scaled_slack * row_scale
                raw_slack = products - b_i
                # (x / s) * s is within an ulp of x; anything larger means the
                # two came from different iterates.
                assert abs(reconstructed - raw_slack) <= 8 * np.finfo(float).eps * max(
                    abs(raw_slack), row_scale
                ), (
                    f"recorded slack {raw_slack!r} disagrees with the "
                    f"independent measurement {reconstructed!r}"
                )
                checked += 1

        assert checked >= 10, f"only {checked} considered rows; no margin"

    def test_the_blocked_row_always_passes_the_scaled_gate(self):
        """The blocking gate must be scaled too, not only the full-step gate.

        A directional derivative that is numerically zero *relative to its row*
        must not make the row a blocking candidate; under the unscaled gate it
        did, and the row then contributed a negative ``alpha``.  The hook
        derives the scale itself, so a loop gating on the raw derivative
        records a ``blocking_row`` outside its own considered set.
        """
        # Needs a row whose derivative is numerically still *relative to its
        # row* while its slack is negative -- only then does the unscaled gate
        # select it, at a hugely negative alpha.  Measured to occur on the
        # ``b = 0`` first-difference shape the monotone spline path uses.
        records = self._decisions(scale=1e6, seeds=range(25), p=6, zero_rhs=True)

        blocked = [r for r in records if r["blocking_row"] >= 0]
        assert len(blocked) >= 10, f"only {len(blocked)} blocking decisions; no margin"
        offenders = [r for r in blocked if not r["blocking_is_considered"]]
        assert not offenders, (
            f"{len(offenders)} of {len(blocked)} decisions blocked on a row the "
            "scaled gate excludes"
        )
        for record in blocked:
            assert record["blocking_scaled_step"] < -1e-12, (
                f"row {record['blocking_row']} was blocked on with scaled "
                f"derivative {record['blocking_scaled_step']!r}"
            )

    def test_the_seam_is_off_by_default_and_inert_when_disabled(self):
        """Default off, and an attached-but-disabled run changes nothing."""
        rng = np.random.default_rng(2)
        p = 5
        M = rng.standard_normal((p, p))
        H = M.T @ M + 0.5 * np.eye(p)
        g = rng.standard_normal(p) * 100.0
        A = np.diff(np.eye(p), axis=0)
        b = rng.standard_normal(p - 1) * 100.0

        default = solve_constrained_qp(H, g, A, b)
        disabled_sink = NullTraceSink()
        disabled = solve_constrained_qp(H, g, A, b, _trace_run=TraceRun("off", sink=disabled_sink))
        enabled_sink = MemoryTraceSink()
        enabled = solve_constrained_qp(H, g, A, b, _trace_run=TraceRun("on", sink=enabled_sink))

        assert not disabled_sink.enabled
        assert enabled_sink.events, "the fixture must actually produce decisions"
        for other in (disabled, enabled):
            np.testing.assert_array_equal(default.beta, other.beta)
            assert default.n_iter == other.n_iter
            assert default.converged == other.converged
            assert default.active_set == other.active_set


class TestProductionShapedConsistency:
    """The floor must clear a *cancelling* ``g``, not just a single matvec.

    Both the original calibration and the unit-condition sweep build
    ``g = H @ x`` -- one matvec of roundoff.  In production
    ``g_vec = XtWz - rhs0 * XtW1 / sum_W`` is a cancelling combination whose
    relative error can sit orders above eps, and ``_null_space_mass`` projects
    the *computed* ``g``, so the measured mass carries ``g``'s own error on top
    of the basis error that ``_consistency_floor`` models.
    """

    @staticmethod
    def _profiled_fit(width, seed):
        """``H = X'WX`` with an exactly collinear column; ``g`` profiled.

        Consistent by construction: ``H v = 0`` implies ``X v = 0``, and the
        profiled ``g`` then satisfies ``g . v = 0`` identically.
        """
        rng = np.random.default_rng(seed)
        n = 4 * width
        X = rng.standard_normal((n, width))
        X[:, -1] = X[:, 0]  # exact null direction e_last - e_0
        W = rng.gamma(2.0, 1.0, n)
        z = X @ rng.standard_normal(width) + 0.1 * rng.standard_normal(n)
        H = X.T @ (W[:, None] * X)
        g = X.T @ (W * z) - ((W * z).sum() / W.sum()) * (X.T @ W)
        return H, g

    @pytest.mark.parametrize("width", [16, 64, 180])
    def test_profiled_rhs_at_production_width_is_not_refused(self, width):
        refused = 0
        checked = 0
        for seed in range(10):
            H, g = self._profiled_fit(width, seed)
            decomposition = decompose_gram(H)
            if decomposition.rank >= decomposition.width:
                continue
            checked += 1
            _, spectral = _null_space_mass(decomposition, g)
            assert _consistency_floor(decomposition) > 100 * spectral, (
                f"floor leaves under 100x over a measured roundoff of {spectral:.3e}"
            )
            try:
                solve_constrained_qp(H, g, np.zeros((0, width)), np.zeros(0))
            except ValueError:
                refused += 1

        assert checked >= 5, f"only {checked} systems reached the gate"
        assert refused == 0, f"{refused}/{checked} consistent profiled fits refused"


class TestRankDeficientKKT:
    """A truncated H can leave a null direction in the KKT system.

    ``np.linalg.solve`` *numerically succeeds* on such a system rather than
    raising, so the ``LinAlgError`` fallback never fires and the step drifts
    along the flat direction.  Admitting rank-deficient H is what routed
    singular systems into that solve: before this branch the pure-H solve
    raised first and the KKT system never saw one.
    """

    def test_binding_constraint_with_rank_one_h_stays_bounded_and_feasible(self):
        """Measured pre-fix: 200 iterations, ``|beta| ~ 4.1e43``, constraint
        violated by 1.0. ``converged`` was already ``False``, so a test that
        only checked the flag would have passed on that answer."""
        H = np.outer([2.0, 3.0, 5.0], [2.0, 3.0, 5.0])  # rank 1
        g = H @ np.array([-3.0, -3.0, -3.0])
        A = np.array([[1.0, 3.0, 2.0]])
        b = np.array([1.0])

        assert decompose_gram(H).rank == 1, "fixture must be rank deficient"

        result = solve_constrained_qp(H, g, A, b)

        # Property 1: the returned point satisfies the constraints.
        assert np.all(A @ result.beta - b >= -1e-9), (
            f"constraint violated by {np.min(A @ result.beta - b):.3e}"
        )
        # Property 2: the magnitude is bounded -- this is what a `converged`
        # check alone would miss.
        assert np.max(np.abs(result.beta)) < 1e3, (
            f"max|beta| = {np.max(np.abs(result.beta)):.3e} has drifted"
        )
        # And it is the optimum: 0.5 s^2 + 30 s at s = v @ beta is minimal at
        # s = -30, giving -450.
        objective = 0.5 * result.beta @ H @ result.beta - g @ result.beta
        np.testing.assert_allclose(objective, -450.0, rtol=1e-9)
        assert result.converged

    def test_full_rank_path_still_uses_the_direct_kkt_solve(self):
        """The rank-aware branch must not touch the full-rank majority."""
        rng = np.random.default_rng(4)
        p = 5
        M = rng.standard_normal((p, p))
        H = M.T @ M + np.eye(p)
        g = rng.standard_normal(p)
        A = np.eye(p)
        b = np.full(p, 0.5)  # binding, so the KKT branch is exercised

        decomposition = decompose_gram(H)
        assert decomposition.rank == decomposition.width, "fixture must be full rank"

        result = solve_constrained_qp(H, g, A, b)

        assert result.converged
        assert np.all(A @ result.beta - b >= -1e-10)


class TestIntegerHessian:
    """``H + H.T`` evaluates in the input dtype, so an integer H can wrap."""

    def test_large_integer_hessian_does_not_overflow(self):
        """``[[2**62]]`` as int64 sums to a negative number, and the PSD guard
        then rejects a valid one-dimensional problem."""
        H = np.array([[2**62]])
        assert (H + H.T)[0, 0] < 0, "fixture no longer demonstrates the wrap"
        g = np.array([1.0])

        result = solve_constrained_qp(H, g, np.zeros((0, 1)), np.zeros(0))

        np.testing.assert_allclose(result.beta, [1.0 / 2**62], rtol=1e-12)
        assert result.converged

    def test_integer_hessian_matches_its_float_cast(self):
        H = np.array([[4, 1], [1, 3]])
        g = np.array([1.0, 2.0])

        integer_result = solve_constrained_qp(H, g, np.zeros((0, 2)), np.zeros(0))
        float_result = solve_constrained_qp(H.astype(float), g, np.zeros((0, 2)), np.zeros(0))

        np.testing.assert_array_equal(integer_result.beta, float_result.beta)
