"""Tests for the active-set constrained penalized least-squares solver."""

import inspect

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
    _solve_saddle_least_squares,
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


class TestNullMassSurvivesExtremeGradientScale:
    """The breach test is a ratio of two norms, and both of them can saturate.

    ``np.linalg.norm`` forms ``sqrt(x.dot(x))``, so the squaring leaves the
    representable range an octave before the value does.  Underflow drives the
    ratio to ``0.0`` (both norms zero, denominator clamped to ``tiny``);
    overflow drives it to ``nan`` (``inf / inf``).  Neither is greater than the
    floor, so the guard this branch added to stop returning ``H+g`` as a
    stationary point is bypassed by its own arithmetic at both extremes.
    """

    H_STRUCTURAL = np.diag([1.0, 0.0])

    @staticmethod
    def _spectral_fixture():
        """Rank-deficient with the null direction *rotated off* the axes, so
        the deficiency is spectral rather than structural."""
        basis = np.linalg.qr(np.random.default_rng(0).standard_normal((3, 3)))[0]
        H = basis @ np.diag([1.0, 1.0, 0.0]) @ basis.T
        return 0.5 * (H + H.T), basis[:, 2]

    @pytest.mark.parametrize(
        "magnitude", [1e-300, 1e-200, 1e-170, 1e-160, 1.0, 1e160, 1e200, 1e300]
    )
    def test_a_structural_breach_is_caught_at_every_representable_scale(self, magnitude):
        """The mass is scale-free in exact arithmetic, so the answer must be
        the same at every magnitude: all of ``g`` is on the null direction."""
        g = np.array([0.0, magnitude])

        structural, _ = _null_space_mass(decompose_gram(self.H_STRUCTURAL), g)
        assert structural == pytest.approx(1.0, rel=1e-12)

        with pytest.raises(ValueError, match="structurally aliased column"):
            solve_constrained_qp(self.H_STRUCTURAL, g, np.zeros((0, 2)), np.zeros(0))

    @pytest.mark.parametrize("magnitude", [1e-300, 1e-200, 1.0, 1e200, 1e300])
    def test_a_spectral_breach_is_caught_at_every_representable_scale(self, magnitude):
        """Pre-fix, ``1e200`` returned ``max|beta| = 2.4e202`` instead."""
        H, null_direction = self._spectral_fixture()
        g = null_direction * magnitude

        _, spectral = _null_space_mass(decompose_gram(H), g)
        assert spectral == pytest.approx(1.0, rel=1e-12)

        with pytest.raises(ValueError, match="truncated spectral direction"):
            solve_constrained_qp(H, g, np.zeros((0, 3)), np.zeros(0))

    @pytest.mark.parametrize("magnitude", [1e-300, 1e-200, 1.0, 1e200, 1e300])
    def test_a_consistent_gradient_still_solves_at_every_scale(self, magnitude):
        """The converse: rescaling must not manufacture a breach either."""
        g = np.array([magnitude, 0.0])

        structural, spectral = _null_space_mass(decompose_gram(self.H_STRUCTURAL), g)
        assert (structural, spectral) == (0.0, 0.0)

        result = solve_constrained_qp(self.H_STRUCTURAL, g, np.zeros((0, 2)), np.zeros(0))
        np.testing.assert_array_equal(result.beta, [magnitude, 0.0])

    def test_a_zero_gradient_reports_zero_mass_without_dividing(self):
        """``g == 0`` has no exponent to normalize by and no inconsistency to
        report; it is the one input the old ``max(norm, tiny)`` clamp existed
        for, and it must keep giving exactly the same answer."""
        masses = _null_space_mass(decompose_gram(self.H_STRUCTURAL), np.zeros(2))
        assert masses == (0.0, 0.0)

        result = solve_constrained_qp(self.H_STRUCTURAL, np.zeros(2), np.zeros((0, 2)), np.zeros(0))
        np.testing.assert_array_equal(result.beta, [0.0, 0.0])

    def test_the_saturating_norm_is_the_mechanism(self):
        """Pin the mechanism alongside the outcome, so a numpy release that
        gave ``norm`` a scaled inner product would show up as a changed premise
        rather than as a silently redundant guard."""
        decomposition = decompose_gram(self.H_STRUCTURAL)
        underflowed = np.array([0.0, 1e-200])
        overflowed = np.array([0.0, 1e200])
        null_row = np.array([False, True])

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            # The squaring saturates; the true magnitude is representable.
            assert float(underflowed.dot(underflowed)) == 0.0
            assert float(np.hypot(*underflowed)) == 1e-200
            assert not np.isfinite(overflowed.dot(overflowed))
            assert float(np.hypot(*overflowed)) == 1e200

            # So the unscaled ratio reads 0.0 at one end and nan at the other,
            # and ``mass > floor`` is false for both.
            def naive_ratio(g):
                return float(np.linalg.norm(g[null_row])) / max(
                    float(np.linalg.norm(g)), np.finfo(float).tiny
                )

            assert naive_ratio(underflowed) == 0.0
            assert np.isnan(naive_ratio(overflowed))

        # The shipped ratio reports the true mass at both extremes.
        assert _null_space_mass(decomposition, underflowed)[0] == pytest.approx(1.0)
        assert _null_space_mass(decomposition, overflowed)[0] == pytest.approx(1.0)


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


class TestToleranceDomain:
    """``tol`` is a relative tolerance, and its predicate has a real domain.

    The normalized slack saturates at ``-1``, so ``slack >= -tol`` accepts
    every finite violation once ``tol >= 1``.  The absolute test this replaced
    could not become vacuous, so the domain arrived with the scale-aware slack
    and is checked at the public boundary.
    """

    H = np.array([[1.0]])
    G = np.array([-100.0])
    A = np.array([[1.0]])
    B = np.array([0.0])

    def test_the_slack_saturates_which_is_why_the_domain_exists(self):
        """The premise, measured rather than asserted: two violations three
        orders of magnitude apart normalize to the same ``-1.0``."""
        near = _feasibility_slack(self.A, np.array([-100.0]), self.B)
        far = _feasibility_slack(self.A, np.array([-1e30]), self.B)
        np.testing.assert_array_equal(near, [-1.0])
        np.testing.assert_array_equal(far, [-1.0])
        # ...so at tol = 1 the predicate accepts a violation of 1e30.
        assert _is_feasible(self.A, np.array([-1e30]), self.B, 1.0)

    @pytest.mark.parametrize("tol", [1.0, 1.0000000000000002, 2.0, 1e6, np.inf])
    def test_a_tolerance_at_or_above_one_is_rejected(self, tol):
        """Pre-fix ``tol = 1.0`` returned ``beta = [-100]`` -- the
        unconstrained answer, violating ``beta >= 0`` by 100 -- with
        ``converged=True`` and ``n_iter=0``."""
        with pytest.raises(ValueError, match=r"requires 0 < tol < 1"):
            solve_constrained_qp(self.H, self.G, self.A, self.B, tol=tol)

    @pytest.mark.parametrize("tol", [0.0, -1e-12, -1.0, -np.inf, np.nan])
    def test_a_non_positive_or_undefined_tolerance_is_rejected(self, tol):
        """``tol <= 0`` makes the step-norm test ``||step|| < tol`` unreachable,
        so the loop can only ever exhaust ``max_iter``; NaN is rejected by the
        same negated-range spelling."""
        with pytest.raises(ValueError, match=r"requires 0 < tol < 1"):
            solve_constrained_qp(self.H, self.G, self.A, self.B, tol=tol)

    @pytest.mark.parametrize("tol", [1e-15, 1e-12, 1e-8, 0.5, 0.999999])
    def test_the_open_interval_is_accepted_and_still_binds_the_constraint(self, tol):
        """The boundary is at 1, not below it: everything short of it must
        still solve, and solve to the constrained optimum."""
        result = solve_constrained_qp(self.H, self.G, self.A, self.B, tol=tol)

        np.testing.assert_array_equal(result.beta, [0.0])
        assert result.converged

    def test_the_default_lies_inside_the_domain(self):
        """The validated domain must not exclude the value every in-tree
        caller relies on; no caller passes ``tol`` at all."""
        default = inspect.signature(solve_constrained_qp).parameters["tol"].default
        assert 0.0 < default < 1.0
        assert default == 1e-12


class TestProjectionSelectsTheWorstViolation:
    """The scale-aware slack is a stopping test, not a ranking.

    At ``b = 0`` it evaluates to ``x / max(1, |x|)``, which is exactly ``-1.0``
    for *every* violation past 1, so feeding it to ``argmin`` makes rows
    violated by wildly different amounts indistinguishable and breaks the exact
    tie on the lowest index.  Selection therefore comes from the raw
    violations; only the stopping test stays scale-aware.
    """

    A = np.diff(np.eye(7), axis=0)  # rows e_{i+1} - e_i, the monotone shape
    B = np.zeros(6)
    # First differences -1.5, -14, -1.5, -14, -1.5, -14: every row past the
    # clamp, three of them nearly ten times worse than the other three.
    BETA = np.array([0.0, -1.5, -15.5, -17.0, -31.0, -32.5, -46.5])

    def test_the_clamp_erases_the_ordering(self):
        """Precondition: the raw violations differ 9x and the scaled ones do not."""
        raw = self.A @ self.BETA - self.B
        np.testing.assert_array_equal(raw, [-1.5, -14.0, -1.5, -14.0, -1.5, -14.0])
        # Exactly -1.0 on all six -- not merely close, which is what makes the
        # tie exact and hands the pick to the lowest index.
        np.testing.assert_array_equal(
            _feasibility_slack(self.A, self.BETA, self.B), np.full(6, -1.0)
        )
        assert int(np.argmin(raw)) == 1
        assert int(np.argmin(_feasibility_slack(self.A, self.BETA, self.B))) == 0

    def test_below_the_clamp_the_two_orderings_agree(self):
        """Control: the collapse is the cause, not a coincidence of the fixture."""
        beta = np.array([0.0, -0.3, -0.5, -0.9, -1.1, -1.3, -1.6])
        raw = self.A @ beta - self.B
        np.testing.assert_array_equal(raw, _feasibility_slack(self.A, beta, self.B))
        assert int(np.argmin(raw)) == int(np.argmin(_feasibility_slack(self.A, beta, self.B)))

    def test_the_sweep_budget_is_spent_on_the_worst_row(self):
        """The outcome the ranking buys, not the internal it is spelled with.

        Each sweep repairs one row and pushes part of that row's violation into
        its two neighbours, so the 100-sweep budget is finite currency and
        spending it on the worst row is what converts it into feasibility.
        Measured on this fixture: selecting on the raw violations leaves a
        worst residual violation of ``-2.84e-01``; selecting on the clamped
        slack leaves ``-4.70e+00``, 16.6x worse, because it cycles between rows
        0 and 1 -- both stuck at ``-1.0`` -- while rows 3 and 5, violated by 14,
        wait.  The 1.0 threshold below sits between the two with an order of
        magnitude of margin either side.
        """
        projected = _project_feasible(self.BETA, self.A, self.B, 1e-12)
        residual = float(np.min(self.A @ projected - self.B))
        assert residual > -1.0, f"worst residual violation {residual:.3e} is the clamped ordering's"

    def test_the_stopping_test_still_means_every_row(self):
        """Selection moved to the raw pair; the *stopping* test must not.

        Taking the stopping test from the selected row -- ``scaled[argmin(raw)]
        >= -tol`` -- is equivalent at ``b = 0``, where clamping is monotone in
        the raw violation so the raw argmin also attains the minimum slack.  It
        is not equivalent for a nonzero ``b``, and this is the case that
        separates them: row 1 is the worse raw violation (-1) but, against a
        row scale of 1000, is already satisfied to ``tol``, while row 0 is
        violated by half its scale.  Reading the stopping test off row 1 exits
        immediately and returns a point ``_is_feasible`` rejects.
        """
        A = np.eye(2)
        b = np.array([0.0, 1000.0])
        beta = np.array([-0.5, 999.0])

        raw = A @ beta - b
        scaled = _feasibility_slack(A, beta, b)
        np.testing.assert_array_equal(raw, [-0.5, -1.0])
        np.testing.assert_allclose(scaled, [-0.5, -0.001], rtol=0, atol=0)
        assert int(np.argmin(raw)) == 1 and int(np.argmin(scaled)) == 0

        projected = _project_feasible(beta, A, b, 0.01)
        assert _is_feasible(A, projected, b, 0.01), (
            f"projection returned {projected}, which its own caller calls infeasible"
        )

    @staticmethod
    def _master_project_feasible(beta, A, b):
        """``master``'s projection, transcribed verbatim from ``git show
        master:src/superglm/solvers/constrained_qp.py``.

        A reference implementation rather than a recorded array: the claim
        under test is that the shipped body *is* this one at ``b = 0``, and an
        argument stays true as fixtures move where a stored number does not.
        """
        beta = beta.copy()
        for _ in range(100):
            violations = A @ beta - b
            worst = np.argmin(violations)
            if violations[worst] >= -1e-12:
                break
            a = A[worst]
            deficit = b[worst] - a @ beta
            beta += deficit / (a @ a) * a
        return beta

    def test_the_projection_is_bitwise_masters_at_zero_rhs(self):
        """Every in-tree call site passes ``b = 0`` and the default ``tol``, and
        on that domain the scale-aware rewrite is the identity.

        ``products - 0.0`` is bitwise ``products``; the selection is the same
        raw ``argmin``; and the clamp ``v / max(1, |v|)`` is exactly ``v`` for
        ``|v| <= 1`` and exactly ``-1`` below, so for ``tol`` in ``(0, 1)`` the
        clamped and raw stopping tests are the same predicate row by row.  The
        default ``tol`` is ``1e-12``, which is the literal ``master`` hardcoded.
        """
        rng = np.random.default_rng(2026)
        checked = repaired = 0
        for _ in range(400):
            p = int(rng.integers(2, 10))
            shape = rng.integers(0, 4)
            if shape == 0:
                A = np.diff(np.eye(p), axis=0)
            elif shape == 1:
                A = np.eye(p)
            elif shape == 2:
                A = np.diff(np.eye(p), n=2, axis=0)
            else:
                # ``D @ P``, the in-tree shape: unequal row norms, so a
                # selection normalized by the row norm reorders the sweep.
                A = np.diff(np.eye(p), axis=0) @ np.linalg.qr(rng.standard_normal((p, p)))[0]
            if A.shape[0] == 0:
                continue
            A = A * 10.0 ** rng.uniform(-3, 3)
            b = np.zeros(A.shape[0])
            beta = rng.standard_normal(p) * 10.0 ** rng.uniform(-2, 4)

            shipped = _project_feasible(beta, A, b, 1e-12)
            reference = self._master_project_feasible(beta, A, b)
            assert shipped.tobytes() == reference.tobytes(), (
                f"projection diverged from master at b = 0: {shipped} vs {reference}"
            )
            checked += 1
            if not np.all(A @ beta - b >= -1e-12):
                repaired += 1

        assert checked >= 350, f"only {checked} fixtures were built"
        assert repaired >= 100, (
            f"only {repaired} fixtures started infeasible; the comparison is mostly vacuous"
        )


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

    def test_the_rank_deficient_branch_is_reachable_from_a_production_fit(self):
        """A retraction, pinned: this branch was described as synthetic-only.

        It is not.  A monotone ``BSplineSmooth`` on a covariate with a wide gap
        leaves whole B-spline basis functions with empty support, so ``XtWX``
        has exactly zero rows and columns -- and with the spline penalty at
        zero, ``S`` adds nothing back.  Every ingredient then lines up:
        ``g_vec`` is exactly ``0.0`` on those coordinates because ``XtWz`` and
        ``XtW1`` both are, the structural null mass is exactly ``0.0`` so the
        consistency gate passes rather than raising, and the constraint rows
        are binding, so the fit routes into ``kkt_may_be_singular = True`` with
        a non-empty active set.  Measured on the fixture below: 4 QP calls, all
        at rank 13 of 19, 6 exactly zero rows, active set 6, ``max|beta| <
        0.42``, worst slack ``-0.0``.

        What this does *not* claim: that the branch is load-bearing here.
        Pinning ``kkt_may_be_singular = False`` returns bitwise identical
        coefficients on this fixture -- ``np.linalg.solve`` has no rank cutoff
        and handles this particular singular saddle.  The branch's necessity is
        pinned by the rank-one fixture above; this one pins only that
        production reaches it, which is what the earlier claim got wrong.
        """
        import pandas as pd

        from superglm import Constraint, SuperGLM
        from superglm.features.spline import BSplineSmooth
        from superglm.solvers import irls_direct

        real_solve = irls_direct.solve_constrained_qp
        observed: list[dict] = []

        def recording_solve(H, g, A, b, **kwargs):
            H_sym = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
            decomposition = decompose_gram(H_sym)
            zero_rows = np.flatnonzero(np.all(H_sym == 0.0, axis=1))
            result = real_solve(H, g, A, b, **kwargs)
            observed.append(
                {
                    "rank": decomposition.rank,
                    "width": decomposition.width,
                    "zero_rows": zero_rows,
                    "g_on_zero_rows": np.asarray(g, dtype=float)[zero_rows],
                    "structural_mass": _null_space_mass(decomposition, np.asarray(g, float))[0],
                    "n_active": len(result.active_set),
                    "beta": result.beta,
                    "slack": float(np.min(A @ result.beta - b)),
                }
            )
            return result

        # A 0.6-wide gap in x with 16 knots empties the support of six basis
        # functions.  Poisson so the fit takes several IRLS iterations.
        x = np.concatenate([np.linspace(0.0, 0.2, 60), np.linspace(0.8, 1.0, 60)])
        y = np.round(np.exp(0.5 + 2.0 * x)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=0.0,
            features={"x": BSplineSmooth(n_knots=16, constraint=Constraint.fit.increasing)},
        )
        try:
            irls_direct.solve_constrained_qp = recording_solve
            model.fit(pd.DataFrame({"x": x}), y)
        finally:
            irls_direct.solve_constrained_qp = real_solve

        assert observed, "the constrained QP never ran"
        singular = [row for row in observed if row["rank"] < row["width"]]
        assert singular, (
            f"no QP solve was rank deficient; ranks {[(r['rank'], r['width']) for r in observed]}"
        )

        for row in singular:
            # The mechanism, not just the outcome: a structurally empty column.
            assert row["zero_rows"].size > 0, "rank deficiency is not structural"
            np.testing.assert_array_equal(row["g_on_zero_rows"], np.zeros(row["zero_rows"].size))
            # ...so the consistency gate sees exactly no structural mass and
            # admits the system instead of raising.
            assert row["structural_mass"] == 0.0
            # ...and the constraints are binding, which is what puts a
            # constraint-tangent direction into the KKT system at all.
            assert row["n_active"] > 0, "no constraint was active"
            # The answer stays usable.
            assert np.all(np.isfinite(row["beta"]))
            assert np.max(np.abs(row["beta"])) < 1e3, (
                f"max|beta| = {np.max(np.abs(row['beta'])):.3e} has drifted"
            )
            assert row["slack"] >= -1e-8, f"constraint violated by {row['slack']:.3e}"

        assert np.all(np.isfinite(model._result.beta))

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


class TestKKTEquilibration:
    """``lstsq``'s cutoff is relative to the matrix it is handed.

    Routing rank-deficient systems to ``lstsq`` fixed the drift above, but on
    the *unscaled* saddle matrix the cutoff measures the constraint block
    against the norm of ``H``.  When ``H`` dwarfs the constraint rows every
    constraint direction reads as noise and is discarded, and the answer
    ignores the constraints.  The matrix is nonsingular in these cases; only
    the tolerance was wrong.
    """

    def test_dominant_hessian_does_not_discard_the_constraint_block(self):
        """Measured pre-fix: ``beta = (-1, 0.5)``, violating ``A @ beta >= 0``
        by ``0.5``, from KKT singular values ``[1e16, 1, 1]`` against an
        ``rcond=None`` cutoff of ``3 * eps * 1e16 = 6.66`` -- both unit values
        truncated, rank 1 of 3 retained.  The feasible optimum attains the same
        objective, so the wrong answer was not even cheaper."""
        H = np.diag([1e16, 0.0])
        g = np.array([-1e16, 0.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([0.0])

        assert decompose_gram(H).rank == 1, "fixture must take the lstsq branch"

        result = solve_constrained_qp(H, g, A, b)

        assert np.all(A @ result.beta - b >= -1e-9), (
            f"constraint violated by {np.min(A @ result.beta - b):.3e}"
        )
        objective = 0.5 * result.beta @ H @ result.beta - g @ result.beta
        np.testing.assert_allclose(objective, -5e15, rtol=1e-12)
        np.testing.assert_allclose(result.beta, [-1.0, 1.0], rtol=1e-9)
        # ``converged`` is unchanged in meaning and now reports True on its own
        # terms: the loop reached its termination test and the point it
        # returned is feasible, which is the whole certificate.  Pre-fix it was
        # False *because* the point was infeasible.
        assert result.converged

    @pytest.mark.parametrize("exponent", [-20, -12, -8, -4, 0, 4, 8, 12, 16, 20])
    def test_feasibility_survives_hessian_constraint_scale_mismatch(self, exponent):
        """One passing example is not evidence that a scaling is right.

        Pre-fix this is feasible up to ``1e12`` and infeasible by ``0.5`` at
        ``1e16`` and ``1e20``; the cutoff crosses the constraint rows somewhere
        in between.  Post-fix every scale returns the exact optimum
        ``(-1, 1) * 1`` with the objective ``-0.5 * scale``.
        """
        scale = 10.0**exponent
        H = np.diag([scale, 0.0])
        g = np.array([-scale, 0.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([0.0])

        assert decompose_gram(H).rank == 1, "fixture must take the lstsq branch"

        result = solve_constrained_qp(H, g, A, b)

        slack = float((A @ result.beta - b)[0])
        assert slack >= -1e-9, f"infeasible by {slack:.3e} at scale {scale:.0e}"
        if exponent >= -12:
            # Below about ``eps`` the H block is lost in the roundoff of the
            # constraint block when the saddle matrix is *assembled*, so the
            # tangential step is not recoverable by any later rescaling. That
            # floor is unchanged by this fix and is asserted separately below.
            objective = 0.5 * result.beta @ H @ result.beta - g @ result.beta
            np.testing.assert_allclose(objective, -0.5 * scale, rtol=1e-12)

    def test_hessian_below_constraint_roundoff_is_a_known_precision_floor(self):
        """Not a regression, and not fixable by scaling: at ``||H|| <= eps *
        ||A||`` the ``H`` block rounds away as the saddle matrix is formed, so
        the constraint-tangent step is gone before any solve sees it.  Pinned
        so that the sweep above cannot quietly be weakened to hide it."""
        scale = 1e-16
        H = np.diag([scale, 0.0])
        g = np.array([-scale, 0.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([0.0])

        result = solve_constrained_qp(H, g, A, b)

        assert float((A @ result.beta - b)[0]) >= -1e-9
        objective = 0.5 * result.beta @ H @ result.beta - g @ result.beta
        # The optimum is -0.5 * scale; the reachable answer is 0.75 of it.
        np.testing.assert_allclose(objective, -0.75 * 0.5 * scale, rtol=1e-9)

    def test_retained_in_band_direction_is_not_truncated_by_the_kkt_solve(self):
        """The rank policy retains eigenvalues down to ``gram_rcond * lambda_max
        = eps * lambda_max``; ``lstsq(rcond=None)`` drops singular values below
        ``max(M, N) * eps * sigma_max``.  A direction in between is retained by
        the policy -- ``decomposition.solve`` uses it -- but discarded by the
        KKT solve, so the two solves inside one function disagree about what
        the retained subspace is.

        This is only reachable when the active rows leave the direction free:
        an in-band direction that an active constraint *pins* keeps a KKT
        singular value of order that constraint row's norm, not of order its
        ``H`` eigenvalue, so it never approaches either cutoff.  Here
        ``A_eq = [[1, 0, 0]]`` leaves ``e2`` free and ``H[1, 1] = eps`` puts it
        squarely in the band.

        Asserted on the saddle solve rather than end to end because
        ``decomposition.solve`` plus the feasibility projection happen to
        resolve ``e2`` before the loop's first KKT step in every end-to-end
        fixture tried, which would make an outer assertion pass against the
        unequilibrated solve too.

        Measured: unequilibrated, the KKT singular ratios are
        ``[1, 0.38, 1.4e-16, 0]`` against a cutoff of ``4 * eps = 8.9e-16``, so
        the direction is dropped and the step along ``e2`` is exactly ``0``.
        An explicit ``rcond=gram_rcond`` does *not* rescue it -- ``1.4e-16`` is
        below ``eps`` too -- but equilibration lifts the ratio to ``0.38``.
        """
        eps = np.finfo(float).eps
        target = -1.0e3
        H = np.diag([1.0, eps, 0.0])
        g = np.array([-1.0, eps * target, 0.0])
        A_eq = np.array([[1.0, 0.0, 0.0]])

        decomposition = decompose_gram(H)
        assert decomposition.rank == 2, "fixture must take the lstsq branch"
        # The policy itself uses the in-band direction, so the KKT solve must.
        np.testing.assert_allclose(decomposition.solve(g), [-1.0, target, 0.0], rtol=1e-9)

        KKT = np.zeros((4, 4))
        KKT[:3, :3] = H
        KKT[:3, 3:] = -A_eq.T
        KKT[3:, :3] = A_eq
        rhs = np.concatenate([g, [0.0]])

        solution = _solve_saddle_least_squares(KKT, rhs)

        np.testing.assert_allclose(solution[1], target, rtol=1e-9)

    def test_equilibration_is_bitwise_inert_on_an_already_balanced_saddle(self):
        """The scaling must not perturb systems that did not need it: every row
        inf-norm is 1 here, so the scale is exactly 1.0 and each multiply is
        exact.  This is what keeps the change confined to badly scaled saddles
        rather than spreading last-bit drift across the rank-deficient path."""
        KKT = np.array(
            [[1.0, 1.0, -1.0], [1.0, 1.0, -0.0], [1.0, 0.0, 0.0]],
        )
        rhs = np.array([0.25, -0.5, 0.125])
        assert np.all(np.abs(KKT).max(axis=1) == 1.0), "fixture must be balanced"

        equilibrated = _solve_saddle_least_squares(KKT, rhs)
        plain = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

        assert equilibrated.tobytes() == plain.tobytes()

    def test_degenerate_blocks_do_not_produce_non_finite_scales(self):
        """A structurally empty row has inf-norm 0 and must not be divided by.

        It keeps scale ``1.0`` rather than being clamped to ``tiny``, which
        would manufacture a ``6.7e153`` scale for a row carrying no
        information.  A denormal row is a real row and *is* normalized.
        """
        empty = np.zeros((3, 3))
        empty[0, 0] = 1.0
        assert _solve_saddle_least_squares(empty, np.array([1.0, 0.0, 0.0])).tolist() == [
            1.0,
            0.0,
            0.0,
        ]

        for H, g, A, b in [
            (np.zeros((2, 2)), np.zeros(2), np.array([[1.0, 1.0]]), np.array([1.0])),
            (
                np.diag([1e-300, 0.0]),
                np.array([-1e-300, 0.0]),
                np.array([[1.0, 1.0]]),
                np.array([0.0]),
            ),
            (np.diag([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([[1e18, 1e18]]), np.array([0.0])),
        ]:
            result = solve_constrained_qp(H, g, A, b)
            assert np.all(np.isfinite(result.beta)), f"non-finite beta for {np.diag(H)}"

    def test_equilibrated_entries_are_bounded_by_one(self):
        """The no-overflow argument, exercised rather than asserted in prose:
        ``|K[i, j]| <= min(m_i, m_j) <= sqrt(m_i * m_j)``, so every scaled entry
        is at most 1 whatever the dynamic range of the input.

        The bound is exact in real arithmetic; the two multiplies that apply it
        round, so the test allows a few ulps.  What matters for overflow is the
        magnitude, not the last bit -- the measured worst case is 1 + 2 ulp.
        """
        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(200):
            n = int(rng.integers(2, 9))
            raw = rng.standard_normal((n, n)) * 10.0 ** rng.uniform(-200, 200, (n, 1))
            raw = np.where(rng.random((n, n)) < 0.3, 0.0, raw)
            # |K| symmetric, as the saddle assembly guarantees.
            saddle = np.triu(np.abs(raw)) - np.triu(np.abs(raw), 1).T
            row_norm = np.abs(saddle).max(axis=1)
            nonzero = row_norm > 0.0
            scale = np.where(nonzero, 1.0 / np.sqrt(np.where(nonzero, row_norm, 1.0)), 1.0)
            scaled = saddle * scale[:, None] * scale[None, :]
            assert np.all(np.isfinite(scaled))
            worst = max(worst, float(np.abs(scaled).max()))
        assert worst <= 1.0 + 8.0 * np.finfo(float).eps, f"scaled entry reached {worst}"


class TestRankGateSeesCollinearityNotScale:
    """The gate keys off the *equilibrated* rank, so scale never reaches it.

    ``decompose_gram`` divides by ``sqrt(diag(H))`` before deciding rank, so
    ``H = diag(1, 1e-20)`` equilibrates to the identity and is reported full
    rank.  That is a real property of the rank policy, not a bug in it -- but
    it means a sweep that plants its small eigenvalue diagonally never leaves
    the ``np.linalg.solve`` side of the gate, whatever it scores.
    """

    @pytest.mark.parametrize("delta", [1e-20, 1e-18, 1e-16, 2.220446049250313e-16, 1e-12, 1e-8])
    @pytest.mark.parametrize("width", [2, 6, 12])
    def test_a_diagonally_planted_eigenvalue_never_drops_the_rank(self, delta, width):
        H = np.diag([1.0] * (width - 1) + [delta])

        decomposition = decompose_gram(H)

        assert decomposition.rank == decomposition.width, (
            f"raw condition {1.0 / delta:.1e} dropped the rank; the gate's "
            "scale-blindness no longer holds and the rationale beside it is stale"
        )

    def test_the_same_eigenvalue_planted_after_equilibration_does_drop_it(self):
        """The contrast that makes the point above a property rather than an
        accident: identical raw conditioning, opposite gate decision."""
        rng = np.random.default_rng(11)
        width = 6
        basis = np.linalg.qr(rng.standard_normal((width, width)))[0]
        equilibrated = basis @ np.diag([1.0] * (width - 1) + [1e-20]) @ basis.T
        scale = np.sqrt(np.diag(equilibrated))
        correlation = equilibrated / np.outer(scale, scale)
        H = 0.5 * (correlation + correlation.T)

        decomposition = decompose_gram(H)

        assert decomposition.rank < decomposition.width, (
            "fixture no longer plants inside the retention band"
        )


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


class TestLargeFiniteHessian:
    """The float cast fixed integer wrap; the sum still overflows in floats."""

    def test_large_finite_hessian_is_solved_rather_than_refused(self):
        """``[[1e308]] + [[1e308]].T`` is ``inf``, so the PSD guard refused a
        matrix that ``master``'s ``np.linalg.solve(H, g)`` answered exactly."""
        H = np.array([[1e308]])
        with np.errstate(over="ignore"):
            assert not np.isfinite(H + H.T).all(), "fixture no longer overflows"
        g = np.array([1.0])

        result = solve_constrained_qp(H, g, np.zeros((0, 1)), np.zeros(0))

        np.testing.assert_array_equal(result.beta, [1e-308])
        assert result.converged

    def test_large_finite_hessian_solves_the_constrained_problem(self):
        """The KKT blocks carry ``H_sym`` too, so the active-set path has to
        survive the same magnitude, not just the early unconstrained return."""
        H = np.diag([1e308, 1e308])
        g = np.array([-1e308, 1e308])
        A = np.eye(2)
        b = np.zeros(2)

        result = solve_constrained_qp(H, g, A, b)

        assert np.all(np.isfinite(result.beta))
        # Unconstrained optimum is ``g / diag(H) = (-1, 1)``; clipping the
        # violated first row to its bound gives ``(0, 1)``.
        np.testing.assert_allclose(result.beta, [0.0, 1.0], atol=1e-12, rtol=1e-12)
        assert _is_feasible(A, result.beta, b, 1e-12)
        assert result.converged

    def test_a_dense_hessian_past_the_overflow_bound_matches_the_scaled_problem(self):
        """Scaling ``H`` and ``g`` by a common constant leaves the optimum
        fixed, so the overflow regime has a reference answer to be checked
        against -- on a dense ``H`` with a binding active set, not the diagonal
        one-liner above.  Pre-fix the scaled solve returns all zeros: the
        symmetrization saturates to ``inf`` and equilibration turns that into
        ``nan``."""
        rng = np.random.default_rng(21)
        factor = rng.standard_normal((4, 4))
        H = factor.T @ factor + np.eye(4)
        H *= 1.5 / np.abs(H).max()
        g = rng.standard_normal(4)
        A = np.eye(4)
        b = np.zeros(4)

        reference = solve_constrained_qp(H, g, A, b)
        assert reference.active_set, "fixture must bind a constraint"

        # ``H`` and ``g`` are scaled by *different* powers of two, so ``H``
        # crosses the bound while ``H_sym @ beta`` stays representable.  That
        # matvec overflows for a dense ``H`` at this magnitude with an O(1)
        # ``beta``, which is a separate exposure this fix does not address.
        # ``b = 0`` makes the feasible cone scale-invariant, so the optimum
        # moves by exactly ``g_scale / H_scale``.
        h_scale, g_scale = 2.0**1023, 2.0**1000
        scaled_H = H * h_scale
        assert np.abs(scaled_H).max() > np.finfo(float).max / 2.0, "fixture is inside the bound"
        with np.errstate(over="ignore"):
            assert not np.isfinite(scaled_H + scaled_H.T).all(), "fixture no longer overflows"

        scaled = solve_constrained_qp(scaled_H, g * g_scale, A, b)

        assert np.all(np.isfinite(scaled.beta))
        np.testing.assert_allclose(
            scaled.beta * (h_scale / g_scale), reference.beta, rtol=1e-9, atol=1e-10
        )
        assert scaled.active_set == reference.active_set
        assert scaled.converged == reference.converged
