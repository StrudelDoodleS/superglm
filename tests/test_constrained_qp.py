"""Tests for the active-set constrained penalized least-squares solver."""

import functools
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
from superglm.solvers.rank import decompose_gram, needs_factor_certification


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
    """QP call sites distinguish KKT uncertainty from primal infeasibility."""

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

    def test_irls_direct_reports_incomplete_kkt_as_terminal_nonconvergence(
        self,
        caplog,
        monkeypatch,
    ):
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

        with caplog.at_level(logging.INFO, logger="superglm.solvers.irls_direct"):
            model.fit(df[["x"]], df["y"], max_iter=3)

        assert len(calls) == 3
        assert not model.result.converged
        assert model.result.termination_reason == "constraint_kkt_incomplete"

        x_grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 200)})
        assert np.min(np.diff(model.predict(x_grid))) >= -1e-10

        qp_records = [
            record
            for record in caplog.records
            if "constrained QP did not converge" in record.getMessage()
        ]
        assert len(qp_records) == 1
        assert qp_records[0].levelno == logging.INFO
        assert "KKT certificate is incomplete" in qp_records[0].getMessage()
        terminal_records = [
            record
            for record in caplog.records
            if "no complete constrained-QP KKT certificate" in record.getMessage()
        ]
        assert len(terminal_records) == 1
        assert terminal_records[0].levelno == logging.WARNING
        assert "approximately satisfied" not in caplog.text
        assert "violates hard constraints" not in caplog.text

    def test_irls_direct_qp_kkt_info_is_latched_to_one_per_fit(self, caplog, monkeypatch):
        """The KKT note lives inside the IRLS loop; it must not repeat per iteration.

        Without a latch, a fit whose QP never obtains a complete certificate
        emits one identical INFO record per IRLS
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
        # report several times; Gaussian converges in ~2 and barely discriminates.
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 1, 200))
        y = (rng.uniform(size=200) < 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))).astype(float)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Binomial(),
            selection_penalty=0,
            features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
        )

        with caplog.at_level(logging.INFO, logger="superglm.solvers.irls_direct"):
            model.fit(df[["x"]], df["y"], max_iter=6)

        # Precondition: enough non-converging QP solves that an unlatched
        # INFO record would be clearly visible as a repeat.
        assert len(calls) >= 5, f"only {len(calls)} QP solves; test cannot discriminate"

        records = [
            record
            for record in caplog.records
            if "constrained QP did not converge" in record.getMessage()
        ]
        assert len(records) == 1, f"expected exactly 1 KKT note, got {len(records)}"
        assert records[0].levelno == logging.INFO
        assert "KKT certificate is incomplete" in records[0].getMessage()
        assert not model.result.converged
        assert model.result.termination_reason == "constraint_kkt_incomplete"
        assert "approximately satisfied" not in caplog.text
        assert "violates hard constraints" not in caplog.text


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

        ``decompose_gram`` truncates at ``max(gram_rcond, n eps) * lambda_max``
        -- version 3 floors the bare ``eps`` this used to say at the
        eigensolver's bar, issue #356 -- so it still retains blocks
        conditioned far beyond ``factor_rcond``.  The ceiling on a retained
        Gram condition moved with it, from ``1/eps = 4.5e+15`` to
        ``1/(4 eps) = 1.1e+15`` at this fixture's width of 4, which is still
        **1126x above** the largest condition swept here.  A residual-based
        gate refused almost all of these, because a residual is amplified by
        exactly that retained condition number.
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

    Its per-row scale is ``max(1, |b|, |A_i| @ |beta|)`` -- the dot product's
    own error scale, issue #359 -- which depends on the row's *inputs* and not
    on the size of its violation.  So a row violated far worse than another can
    carry a far larger scale and rank below it, and feeding the scaled slack to
    ``argmin`` selects the wrong row to repair.  Selection therefore comes from
    the raw violations; only the stopping test stays scale-aware.

    Before #359 the scale was ``max(1, |b|, |A_i @ beta|)``, a monotone
    function of the violation itself, and the two orderings parted only where
    that clamp *saturated* at ``-1``.  They now part generically, which makes
    this class's decision more necessary rather than less.
    """

    A = np.diff(np.eye(7), axis=0)  # rows e_{i+1} - e_i, the monotone shape
    B = np.zeros(6)
    # First differences -1.5, -14, -1.5, -14, -1.5, -14: three rows violated
    # more than nine times worse than the other three.
    BETA = np.array([0.0, -1.5, -15.5, -17.0, -31.0, -32.5, -46.5])

    def test_the_row_scale_reorders_the_violations(self):
        """Precondition: the worst raw violation is not the worst scaled one."""
        raw = self.A @ self.BETA - self.B
        np.testing.assert_array_equal(raw, [-1.5, -14.0, -1.5, -14.0, -1.5, -14.0])
        # The scale grows along the rows -- |beta_i| + |beta_{i+1}| -- entirely
        # independently of which rows are violated worst.
        np.testing.assert_allclose(
            np.abs(self.A) @ np.abs(self.BETA), [1.5, 17.0, 32.5, 48.0, 63.5, 79.0]
        )
        scaled = _feasibility_slack(self.A, self.BETA, self.B)
        np.testing.assert_allclose(
            scaled,
            [
                -1.0,
                -0.8235294117647058,
                -0.046153846153846156,
                -0.2916666666666667,
                -0.023622047244094488,
                -0.17721518987341772,
            ],
        )
        # Row 1 is violated 9.3x worse than row 0 and still ranks second.
        assert int(np.argmin(raw)) == 1
        assert int(np.argmin(scaled)) == 0

    def test_the_two_orderings_agree_when_every_row_shares_one_scale(self):
        """Control: the reordering is the scale, not a coincidence of the fixture.

        When every row's dot-product scale falls under the ``max(1, .)`` floor
        the scale is 1 throughout, the slack is the raw violation, and the two
        selections coincide.  That is the well-scaled case the relative test is
        documented to leave alone.
        """
        A = np.eye(3)
        b = np.zeros(3)
        beta = np.array([-0.3, -0.5, -0.2])
        np.testing.assert_array_equal(np.abs(A) @ np.abs(beta), [0.3, 0.5, 0.2])
        raw = A @ beta - b
        scaled = _feasibility_slack(A, beta, b)
        np.testing.assert_array_equal(raw, scaled)
        assert int(np.argmin(raw)) == int(np.argmin(scaled)) == 1

    def test_the_sweep_budget_is_spent_on_the_worst_row(self):
        """The outcome the ranking buys, not the internal it is spelled with.

        Each sweep repairs one row and pushes part of that row's violation into
        its two neighbours, so the 100-sweep budget is finite currency and
        spending it on the worst row is what converts it into feasibility.
        Measured on this fixture: selecting on the raw violations leaves a
        worst residual violation of ``-2.84e-01``; selecting on the clamped
        slack leaves ``-4.70e+00``, 16.6x worse, because it cycles between rows
        0 and 1 -- both stuck at ``-1.0`` -- while rows 3 and 5, violated by 14,
        wait.  The 1.0 threshold below sits between the two, with ``3.5x``
        margin to the passing value and ``4.7x`` to the failing one.
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

    def test_the_projection_stops_no_later_than_the_absolute_predicate(self):
        """#359 made the stopping test weaker, and it must be weaker ONLY.

        This test used to require the projection to be bitwise the absolute
        implementation at ``b = 0``.  That is deliberately no longer true: the
        scale is now ``|A_i| @ |beta|``, which dominates ``|A_i @ beta|``, so a
        cancelling row is measured against the accuracy its dot product
        actually has and the stopping test accepts points the absolute one
        refused.  ``test_a_cancelling_row_is_measured_against_its_own_bound``
        pins that gain directly.

        What must still hold is the *direction* of the change.  The scale only
        ever grows, so every normalized violation moves toward zero and the
        shipped projection can only stop at or before the absolute one -- never
        after.  Requiring the shipped result to be feasible under the absolute
        predicate wherever the reference is states exactly that, and would fail
        for any scale that shrank a row instead of growing it.
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
            # Wherever the absolute predicate is satisfied by its own result,
            # the shipped one must be too: a scale that only grows can stop
            # earlier, never later.
            if np.all(A @ reference - b >= -1e-12):
                assert _is_feasible(A, shipped, b, 1e-12), (
                    f"shipped projection stopped LATER than the absolute one, which a "
                    f"dominating scale cannot do: {shipped} vs {reference}"
                )
            checked += 1
            if not np.all(A @ beta - b >= -1e-12):
                repaired += 1

        assert checked >= 350, f"only {checked} fixtures were built"
        assert repaired >= 100, (
            f"only {repaired} fixtures started infeasible; the comparison is mostly vacuous"
        )

    def test_a_cancelling_row_is_measured_against_its_own_bound(self):
        """The gain #359 buys, on a row built to cancel.

        ``A @ beta`` here is ``1e8 - 1e8 - 5e-12``: every term exact, the
        leading pair cancelling completely, and a result 20 orders under the
        terms that produced it.  The absolute predicate reads the ``-5e-12``
        output against a scale of 1 and calls it a violation five times its
        tolerance.  The dot product's own error bound is
        ``n * eps * (|A| @ |beta|) = 6.7e-08``, so the point is feasible to
        every digit the arithmetic has -- ``5e-12`` is ``7.5e-05`` of it.
        """
        # The operands are given rather than computed, deliberately.  A row
        # that cancels this hard has NO summation order it agrees with itself
        # on -- ``5e-12`` is far under ``ulp(1e8) = 1.5e-08``, so a BLAS that
        # adds the small term before the cancelling pair loses it entirely and
        # returns ``0.0``.  Measured: SKYLAKEX, HASWELL, SANDYBRIDGE, NEHALEM
        # and ZEN return ``-5e-12`` here while PRESCOTT and CORE2 return
        # ``0.0``.  Asserting a computed product would pin one BLAS's
        # associativity, which is the class of fixture this repository keeps
        # having to repair, so the products are supplied and only the scale
        # rule -- the thing #359 changed -- is under test.
        products = np.array([-5e-12])
        b = np.array([0.0])
        magnitude = np.array([2e8])  # |A_i| @ |beta| for that row

        absolute_scale = np.maximum(1.0, np.maximum(np.abs(b), np.abs(products)))
        np.testing.assert_array_equal(absolute_scale, [1.0])
        np.testing.assert_array_equal(
            _feasibility_scale(products, b, abs_products=magnitude), magnitude
        )

        # The old scale collapses to the max(1, .) floor, so the test is
        # absolute and refuses; the dot-product scale accepts.
        assert float((products - b)[0] / absolute_scale[0]) < -1e-12
        assert float((products - b)[0] / magnitude[0]) > -1e-12

        bound = 3 * float(np.finfo(float).eps) * float(magnitude[0])
        assert abs(products[0]) < bound / 1000.0, (
            f"violation {abs(products[0]):.3e} is not comfortably inside the "
            f"dot-product bound {bound:.3e}; the fixture no longer makes its point"
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

    def test_zero_rhs_is_inert_only_where_the_row_mass_is_under_one(self):
        """``b = 0`` is not sufficient for inertness, and #359 is why.

        **The claim this test used to carry has been withdrawn, not weakened.**
        It said the scaling was "exactly inert" at ``b = 0`` because the per-row
        scale was ``max(1, |A_i @ beta|)`` and a 189-row measurement put the
        worst ``|A_i @ beta|`` at 0.72.  The scale is now
        ``max(1, |A_i| @ |beta|)``, so that measurement is of the wrong
        quantity, and re-running it on the right one over the monotone and
        constraint fit suites gives **4003 of 8697 constraint rows -- 46% --
        with a scale above 1**, worst 23.3.  Inertness at ``b = 0`` is
        therefore false in general.

        What is still true, and is what this pins, is the condition for it:
        the scale is 1 exactly when the row's own coefficient mass is, so the
        guard below is on ``|A| @ |beta|`` and not on ``|A @ beta|``.  Those
        differ by exactly the dominance this change is about, so guarding the
        weaker one would let a BLAS change report the fixture healthy while
        the assertion failed for an unrelated-looking reason.
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
        magnitude = np.abs(A) @ np.abs(result.beta)
        assert np.max(magnitude) <= 1.0, (
            f"fixture no longer exercises scale == 1: worst row mass "
            f"{np.max(magnitude):.4g} (the products alone reach only "
            f"{np.max(np.abs(products)):.4g}, which is the quantity this guard "
            "used to check and is strictly weaker)"
        )
        np.testing.assert_array_equal(
            _feasibility_scale(products, b, abs_products=magnitude), np.ones(p - 1)
        )
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
            for products, b_i, scaled_slack, row_scale in zip(
                record["row_products"],
                record["row_b"],
                record["row_scaled_slack"],
                record["row_scale"],
            ):
                # Since #359 the scale is `max(1, |b|, |A_i| @ |beta|)` and is
                # not recoverable from the product, so it is read from the
                # trace.  That still cross-checks the operands: the slack comes
                # from an independent `_feasibility_slack` call and the scale
                # from the loop's own, so a slip that fed one a different
                # iterate breaks the reconstruction.
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


class TestStationarityIsRealNotATruncationArtifact:
    """``||step|| < tol`` means stationarity even when ``step`` comes from ``lstsq``.

    The loop terminates on a small step and then reports the KKT certificate.
    That inference is immediate for ``np.linalg.solve``, which either returns
    *the* solution or raises; it is not immediate for
    ``_solve_saddle_least_squares``, whose minimum-norm answer is small exactly
    when ``lstsq`` discards a direction the right-hand side needed.  Every early
    return on a rank-deficient ``H`` with a non-empty active set takes that
    path, so the distinction covers the whole rank-deficient population rather
    than a corner of it.

    The saddle system is nonetheless always consistent, for a structural reason
    recorded beside the early return: for PSD ``H = L L^T`` the range of
    ``P H Q`` sits inside the range of ``P H P``, so a right-hand side whose
    first block lies in ``range(H)`` and whose second lies in ``range(A_eq)``
    is always reachable.  These tests pin both halves -- the property, and the
    hypothesis it needs -- so that the argument beside the code cannot go stale
    silently.
    """

    @staticmethod
    def _kkt_violation(H, g, A, result, tol=1e-12):
        """Largest KKT violation of ``result``, derived from scratch.

        Returns ``(stationarity, dual, primal)`` as scale-free quantities.
        Nothing here reuses the solver's own arrays or predicates: the point of
        the test is to witness the certificate independently, and a check fed
        the loop's own arithmetic would agree with it by construction.
        """
        beta = result.beta
        gradient = H @ beta - g
        scale = max(np.linalg.norm(H @ beta), np.linalg.norm(g))
        if not result.active_set:
            # No active constraints: stationarity is the bare gradient.
            outside = np.linalg.norm(gradient)
            dual = 0.0
        else:
            A_eq = A[result.active_set]
            # Stationarity asks that the gradient lie in range(A_eq^T).  Its
            # component outside that range is what the projection at the drop
            # test silently discards, so measure it rather than the projection.
            multipliers = np.linalg.lstsq(A_eq.T, gradient, rcond=None)[0]
            outside = np.linalg.norm(A_eq.T @ multipliers - gradient)
            dual = float(-np.min(multipliers)) / max(np.linalg.norm(multipliers), 1.0)
        # A gradient that is itself negligible is stationary with zero
        # multipliers whatever its direction does, so normalise by the scale it
        # came from and not by the cancelled gradient.
        stationarity = outside / scale if scale > 0.0 else 0.0
        primal = float(-np.min(_feasibility_slack(A, beta, np.zeros(A.shape[0]))))
        return stationarity, dual, primal

    def test_a_converged_result_satisfies_the_kkt_conditions(self):
        """``converged=True`` is checked from outside, over a population.

        The non-tautological halves are stationarity and dual feasibility: the
        loop asserts both and this re-derives them from ``H``, ``g`` and the
        returned active set.  Primal feasibility is *not* asserted here, since
        ``converged`` is defined through the same predicate and re-testing it
        would prove nothing -- ``test_an_infeasible_returned_point_is_never_
        reported_converged`` carries that direction.
        """
        rng = np.random.default_rng(20260730)
        certified = 0
        rank_deficient = 0
        for _ in range(700):
            p = int(rng.integers(3, 10))
            rank = int(rng.integers(1, p))
            basis = np.linalg.qr(rng.standard_normal((p, p)))[0]
            spectrum = np.zeros(p)
            spectrum[:rank] = 10.0 ** rng.uniform(-2, 2, rank)
            H = basis @ np.diag(spectrum) @ basis.T
            H = 0.5 * (H + H.T)
            g = H @ rng.standard_normal(p)  # in range(H) by construction
            A = np.diff(np.eye(p), axis=0) if rng.random() < 0.5 else np.eye(p)
            b = np.zeros(A.shape[0])

            if decompose_gram(H).rank >= p:
                continue
            rank_deficient += 1
            try:
                result = solve_constrained_qp(H, g, A, b)
            except ValueError:
                continue
            if not result.converged:
                continue
            certified += 1
            stationarity, dual, _ = self._kkt_violation(H, g, A, result)
            assert stationarity < 1e-8, (
                f"converged=True on a point whose gradient is {stationarity:.3e} "
                f"outside range(A_eq^T) -- the small step was a truncation "
                f"artifact, not stationarity (active set {result.active_set})"
            )
            assert dual < 1e-6, f"converged=True with a negative multiplier: {dual:.3e}"

        # Liveness: the sweep must actually reach the path under test, or the
        # assertions above are vacuous.
        assert rank_deficient >= 400, f"only {rank_deficient} rank-deficient cases"
        assert certified >= 200, f"only {certified} converged solves to check"

    def test_the_saddle_system_is_consistent_whenever_g_is_in_range_h(self):
        """The structural claim, brute-forced away from the loop.

        ``lstsq`` on the assembled saddle leaves a rounding-level residual for
        every PSD ``H``, whatever its rank, provided the first block lies in
        ``range(H)``.  That is what licenses reading a small minimum-norm step
        as stationarity.
        """
        rng = np.random.default_rng(4242)
        worst = 0.0
        truncated = 0
        deficient_and_truncated = 0
        for _ in range(400):
            p = int(rng.integers(2, 8))
            rank = int(rng.integers(1, p + 1))
            factor = rng.standard_normal((p, rank))
            H = factor @ factor.T  # PSD, rank <= rank
            n_eq = int(rng.integers(1, p + 2))
            A_eq = rng.standard_normal((n_eq, p))
            if rng.random() < 0.5:  # linearly dependent active rows
                k = int(rng.integers(0, n_eq))
                A_eq[k] = A_eq[(k + 1) % n_eq] * float(rng.choice([-1.0, 2.0]))
            beta = rng.standard_normal(p)
            g = H @ rng.standard_normal(p)

            KKT = np.zeros((p + n_eq, p + n_eq))
            KKT[:p, :p] = H
            KKT[:p, p:] = -A_eq.T
            KKT[p:, :p] = A_eq
            rhs = np.concatenate([g - H @ beta, -(A_eq @ beta)])  # b_eq = 0

            sol, _, retained, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
            was_truncated = int(retained < p + n_eq)
            truncated += was_truncated
            deficient_and_truncated += was_truncated * int(rank < p)
            residual = np.linalg.norm(KKT @ sol - rhs)
            scale = np.linalg.norm(KKT) * np.linalg.norm(sol) + np.linalg.norm(rhs)
            worst = max(worst, residual / scale if scale > 0 else 0.0)

        # Liveness, twice.  ``lstsq`` must really be truncating, otherwise
        # consistency is trivial; and the truncation must reach the population
        # the argument is about -- a rank-deficient ``H``, not merely a
        # rank-deficient ``A_eq``, which truncates for its own reason and would
        # keep the first assertion alive on a full-rank sweep.
        assert truncated >= 100, f"only {truncated} of 400 saddles were truncated"
        assert deficient_and_truncated >= 50, (
            f"only {deficient_and_truncated} truncated saddles had a rank-deficient H"
        )
        assert worst < 1e-12, f"worst relative saddle residual {worst:.3e}"

    def test_a_g_outside_range_h_would_make_the_saddle_inconsistent(self):
        """The contrast, so the test above is not passing for a trivial reason.

        This is the case the consistency gate above the loop refuses outright.
        If that gate ever stopped refusing it, the saddle system would become
        inconsistent, ``lstsq`` would truncate a needed direction, and a small
        step would no longer mean stationarity.
        """
        H = np.diag([1.0, 0.0])  # range(H) = span(e0)
        A_eq = np.array([[1.0, 0.0]])
        beta = np.zeros(2)
        g = np.array([0.0, 1.0])  # entirely outside range(H)

        KKT = np.zeros((3, 3))
        KKT[:2, :2] = H
        KKT[:2, 2:] = -A_eq.T
        KKT[2:, :2] = A_eq
        rhs = np.concatenate([g - H @ beta, -(A_eq @ beta)])

        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        residual = np.linalg.norm(KKT @ sol - rhs)
        scale = np.linalg.norm(KKT) * np.linalg.norm(sol) + np.linalg.norm(rhs)
        assert residual / scale > 0.1, "fixture no longer exercises an inconsistent saddle"

        # And the solver refuses it rather than returning the artifact.
        with pytest.raises(ValueError, match="component in null"):
            solve_constrained_qp(H, g, np.array([[1.0, 0.0]]), np.zeros(1))


class TestInfeasibleEarlyReturnIsProjected:
    """The stationary-point early return repairs the point it is about to return.

    The loop can stop at a stationary point on a *subset* active set with
    another row materially violated.  ``converged=False`` disclosed that but did
    not fix it, and the ``irls_direct`` call site takes ``beta`` unconditionally
    -- so an infeasible answer meant a fitted model that was not monotone.
    Projecting at the return converts those into feasible, possibly suboptimal
    answers.

    The guard is precisely today's ``converged=False`` condition, so every solve
    that currently returns a feasible point is untouched by construction rather
    than by measurement.
    """

    @staticmethod
    @functools.cache
    def _rank_deficient_population(n=2700, seed=20260730):
        """Rank-deficient QPs shaped like the in-tree callers: ``b = 0``, structured ``A``.

        ``x = 0`` is feasible for every one of them, so an infeasible answer is
        a solver defect rather than an infeasible problem.

        **The ensemble grew from 900 to 2700 for #359, and the bars below did
        not move.**  Correcting the feasibility scale to the dot product's own
        error bound made fewer solves read as infeasible, which shrank the
        defect population this class samples: at ``n = 900`` it fell to 26
        firing and 4 repaired, under bars of 30 and 10.  Lowering the bars would
        have recorded the improvement as a weaker test.  Tripling the sample
        restores them with margin instead -- measured 93 firing and 34 repaired,
        3.1x and 3.4x -- so the assertions still mean what they meant.
        """
        rng = np.random.default_rng(seed)
        cases = []
        while len(cases) < n:
            p = int(rng.integers(3, 13))
            rank = int(rng.integers(1, p))
            basis = np.linalg.qr(rng.standard_normal((p, p)))[0]
            spectrum = np.zeros(p)
            spectrum[:rank] = 10.0 ** rng.uniform(-2, 2, rank)
            H = basis @ np.diag(spectrum) @ basis.T
            H = 0.5 * (H + H.T)
            if rng.random() < 0.45:
                zero = int(rng.integers(0, p))
                H[zero, :] = 0.0
                H[:, zero] = 0.0
            H = H * 10.0 ** rng.uniform(-6, 6)
            g = H @ rng.standard_normal(p)  # in range(H) by construction
            shape = rng.random()
            if shape < 0.4:
                A = np.diff(np.eye(p), axis=0)
            elif shape < 0.7:
                A = np.eye(p)
            else:
                A = np.diff(np.eye(p), n=2, axis=0)
            A = A * 10.0 ** rng.uniform(-3, 3)
            if A.shape[0] == 0:
                continue
            b = np.zeros(A.shape[0])
            try:
                if decompose_gram(H).rank >= p:
                    continue
                solve_constrained_qp(H, g, A, b, max_iter=1)
            except ValueError:
                continue  # inconsistent normal equations: refused by design
            cases.append((H, g, A, b))
        return cases

    def test_an_infeasible_early_return_is_repaired_where_the_projection_reaches(self):
        """Counted, not snapshotted: without the projection this count is 0.

        Every case here reaches the early return with a row still violated, so
        before the repair *none* of them was feasible on return.  The bound is
        deliberately far below the measured rate -- 223 of 420 across the two
        full 3950-case ensembles -- because the point being pinned is that the
        repair happens at all, not the exact share the sweep budget reaches.
        """
        fired = repaired = 0
        for H, g, A, b in self._rank_deficient_population():
            result = solve_constrained_qp(H, g, A, b)
            if result.converged or result.n_iter >= 200:
                continue  # not the early-return-infeasible population
            fired += 1
            if _is_feasible(A, result.beta, b, 1e-12):
                repaired += 1

        assert fired >= 30, f"only {fired} solves reached the early return infeasibly"
        assert repaired >= 10, (
            f"{repaired} of {fired} infeasible early returns came back feasible; "
            "without the projection at the return this is 0 by construction"
        )

    def test_a_repaired_point_is_still_not_reported_converged(self):
        """Feasible is not certified, and the flag must not start saying it is.

        ``converged`` reports the feasibility of the point the *loop* found,
        taken before the projection runs.  Reporting post-projection feasibility
        instead would flip exactly the repaired population to ``True`` -- a
        point that satisfies the constraints but is not a KKT point, which is
        the over-claim the flag exists to prevent.
        """
        checked = 0
        for H, g, A, b in self._rank_deficient_population():
            result = solve_constrained_qp(H, g, A, b)
            if result.converged or result.n_iter >= 200:
                continue
            if not _is_feasible(A, result.beta, b, 1e-12):
                continue  # the projection ran out of budget; nothing to over-claim
            checked += 1
            assert not result.converged, (
                "a projected point was reported converged: the flag has been "
                "moved to post-projection feasibility"
            )
        assert checked >= 10, f"only {checked} repaired points to check"

    def test_the_projection_does_not_run_again_on_a_feasible_solve(self, monkeypatch):
        """The inertness mechanism, witnessed rather than inferred.

        Every constrained solve runs ``_project_feasible`` once before the loop.
        A solve that returns a feasible point must not run it a second time --
        that is what makes this change bitwise inert on the currently-good
        population, and counting the calls witnesses it directly instead of
        re-deriving the guard's condition, which would agree by construction.
        """
        calls = {"n": 0}
        original = _project_feasible

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr("superglm.solvers.constrained_qp._project_feasible", counting)

        rng = np.random.default_rng(11)
        converged_solves = 0
        for _ in range(120):
            p = int(rng.integers(2, 9))
            M = rng.standard_normal((p, p))
            H = M.T @ M + (0.1 + rng.random()) * np.eye(p)  # full rank
            g = rng.standard_normal(p) * float(rng.choice([1.0, 1e2, 1e4]))
            A = np.diff(np.eye(p), axis=0) if rng.random() < 0.5 else np.eye(p)
            b = np.zeros(A.shape[0])

            calls["n"] = 0
            result = solve_constrained_qp(H, g, A, b)
            if not result.converged:
                continue
            converged_solves += 1
            assert calls["n"] <= 1, (
                f"a converged solve ran _project_feasible {calls['n']} times; "
                "the return-side projection fired on a feasible point"
            )

        assert converged_solves >= 60, f"only {converged_solves} converged solves swept"


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
        """The rank policy retains eigenvalues down to ``max(gram_rcond, n eps)
        * lambda_max``, which version 3 made ``n * eps * lambda_max`` (issue
        #356, where it was ``eps * lambda_max``);
        ``lstsq(rcond=None)`` drops singular values below
        ``max(M, N) * eps * sigma_max``.  The two are now the same SHAPE and
        still not the same number -- they are taken on different matrices, the
        Gram here and the KKT system there, so the band below is narrower than
        it was but has not closed.  A direction in between is retained by
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

    def test_the_same_eigenvalue_planted_after_equilibration_reaches_the_gate(self):
        """The contrast that makes the point above a property rather than an
        accident: identical raw conditioning, opposite gate decision.

        **THIS ASSERTION USED TO PASS BY LUCK, AND VERSION 3 IS WHAT MAKES IT
        A PROPERTY -- ISSUE #356.**  The planted eigenvalue equilibrates into
        pure round-off, and under version 2 the cut sat at ``eps * lambda_max``,
        which is BENEATH the eigensolver's own ``n eps lambda_max`` error bar.
        The residue landed either side of that cut depending on the machine --
        measured over 7 ``OPENBLAS_CORETYPE`` microkernels at ``0.10x to 1.47x``
        of it -- and ``np.maximum(w, 0.0)`` then dropped it only when round-off
        happened to come out NEGATIVE.  So the rank here was decided by a sign
        that is not data: 5 on six kernels and 6 on SKYLAKEX under numpy 2.5.2,
        and the mirror image under 2.4.2.

        With the cut floored at the bar, the residue is inside it on every
        configuration, so the drop is now a function of the data and this reads
        5 everywhere.

        **Two sweeps measure that residue and only one of them is the
        portability margin.**  Over 7 microkernels x both numpy generations it
        runs ``0.017x to 0.245x`` of the bar, i.e. 4.1x of clearance.  Over 7
        microkernels x {this matrix, this matrix with the residue reflected
        through its own eigenvector} it runs ``0.017x to 0.488x`` -- **2.05x**
        -- which is the figure ``rank.py``'s ``_eigensolver_relative_bar``
        records, and it is the one to quote: that module chose reflection
        precisely because it is the conservative stand-in for a BLAS this
        machine cannot run, where a numpy generation is not.  Quote 2.05x.
        """
        rng = np.random.default_rng(11)
        width = 6
        basis = np.linalg.qr(rng.standard_normal((width, width)))[0]
        equilibrated = basis @ np.diag([1.0] * (width - 1) + [1e-20]) @ basis.T
        scale = np.sqrt(np.diag(equilibrated))
        correlation = equilibrated / np.outer(scale, scale)
        H = 0.5 * (correlation + correlation.T)

        # ``eigh`` and NOT ``eigvalsh``: the with-vectors driver is the one
        # ``decompose_gram`` runs, the two are different LAPACK paths, and they
        # do not agree here.  Over the sweep ``eigvalsh`` reaches -7.252e-16 on
        # SANDYBRIDGE where ``eigh`` peaks at +5.504e-16, so a precondition
        # written on the wrong one would be describing a spectrum the gate
        # never sees.
        eigenvalues, _ = np.linalg.eigh(H)
        eigensolver_bar = width * np.finfo(np.float64).eps * float(np.max(np.abs(eigenvalues)))

        diagonal = np.diag([1.0] * (width - 1) + [1e-20])
        diagonal_scale = np.sqrt(np.diag(diagonal))
        diagonal_correlation = diagonal / np.outer(diagonal_scale, diagonal_scale)

        # PRECONDITIONS.  Neither of these is the assertion -- they establish
        # that the fixture still shows the gate what the contrast requires,
        # which is a clean 1.0 on one side and something under the
        # eigensolver's own resolution on the other.
        assert np.linalg.eigvalsh(diagonal_correlation).min() == pytest.approx(1.0, abs=1e-12), (
            "the diagonal plant no longer equilibrates to the identity, so the "
            "contrast this test draws has lost one of its two sides"
        )
        assert abs(float(eigenvalues.min())) < eigensolver_bar, (
            "the planted direction is no longer inside the eigensolver's "
            f"resolution: |{eigenvalues.min():.6e}| against a bar of "
            f"{eigensolver_bar:.6e}, so the gate is now being shown something "
            "it can resolve and the contrast no longer holds"
        )

        # THE ASSERTION, AND IT GOES THROUGH THE GATE.  An earlier revision of
        # this test stopped at the two preconditions above, which was a
        # mistake worth naming: they are arithmetic on arrays the test builds
        # itself, so ``decompose_gram`` was never called at all.  Under that
        # revision, deleting the rank truncation outright -- forcing
        # ``retained_mask`` all-True, a gate that can never drop a direction --
        # left this whole class green, and so did replacing ``decompose_gram``
        # with an unconditional ``raise``.  The old ``rank < width`` caught
        # both.  Removing a coin-flip assertion is right; removing the call is
        # not.
        #
        # ``needs_factor_certification`` is asserted BESIDE the rank, not
        # instead of it, and it is the module's OWN answer to this ambiguity --
        # its docstring is about normal equations that "retain a different
        # direction while reporting the same rank", which is this fixture.  The
        # diagonal plant declines to certify on all 14 configurations, and that
        # pair is the contrast this class is named for.
        #
        # **SUPERSEDED, VERSION 2:** the reason for reaching for certification
        # was that it "holds on BOTH sides of the rank flip -- the collinear
        # plant certifies on all 14 while its rank reads 5 on six kernels and 6
        # on SKYLAKEX".  Under version 3 there is no rank flip to hold on both
        # sides of; the rank reads 5 everywhere, which is what the paragraph
        # below the assertions records.
        #
        # **And on this fixture the certification assertion is now IMPLIED by
        # the rank one, so it is documentation rather than an independent
        # observation.**  All six columns are active and ``psd_semantics`` is
        # True here, so ``rank < 6`` forces ``resolution_limited`` through its
        # first clause, and ``_certification_required`` then returns True on
        # ``rank < width and resolution_limited``.  It cannot fail unless the
        # rank assertion already has.  Kept because it names the contrast in
        # the module's own vocabulary, not because it adds coverage.
        #
        # **AND ITS TEETH ARE STATED RATHER THAN ASSUMED.**  Mutation-checked:
        # replacing ``decompose_gram`` with a ``raise`` reds this, which is
        # what the revision that stopped at the preconditions failed to do.
        # Forcing ``retained_mask`` all-True -- a gate that never truncates --
        # does not red the certification assertion either, which is why the
        # rank assertion below is kept rather than replaced by it.
        #
        # **THAT GAP CLOSED WITH POLICY VERSION 3, AND THE RANK IS ASSERTED
        # AGAIN BECAUSE OF IT.**  The paragraph above was written against
        # version 2, where the gate dropped a direction only when
        # ``eigenvalue <= gram_rcond * max`` with ``gram_rcond = eps`` -- so
        # everything it could drop lay beneath the eigensolver's own
        # ``n eps max`` bar, no fixture could make truncation fire against a
        # RESOLVED eigenvalue, and the rank was round-off on every input.
        # Version 3 floors the cut AT that bar, so truncation now fires exactly
        # where the arithmetic can tell, and ``rank < width`` is a property of
        # the fixture rather than of the machine: measured 5 on all 7
        # microkernels under both numpy generations.  It is asserted first
        # because it is the one that catches the never-truncate mutation, which
        # certification alone does not.
        collinear = decompose_gram(H)
        assert collinear.rank < collinear.width, (
            f"the equilibrated plant was retained (rank {collinear.rank} of "
            f"{collinear.width}); under version 3 the residue sits at 0.017x to "
            "0.488x of the eigensolver bar on the conservative sweep, so dropping "
            "it is a property here"
        )
        assert needs_factor_certification(collinear), (
            "the equilibrated plant no longer reaches the certification band, "
            f"so the Gram route now believes it resolved this direction (rank "
            f"{collinear.rank} of {collinear.width})"
        )
        assert not needs_factor_certification(decompose_gram(diagonal_correlation)), (
            "the diagonal plant now demands certification, so the gate is "
            "seeing scale where the class docstring says it sees nothing"
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
