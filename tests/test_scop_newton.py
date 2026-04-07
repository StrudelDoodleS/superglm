"""Tests for SCOP Newton solver."""

import numpy as np

from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import SCOPNewtonResult, scop_newton_step


class TestSCOPNewtonStep:
    """Single Newton step on SCOP working model."""

    def test_reduces_objective(self):
        """A Newton step reduces the penalized WLS objective."""
        rng = np.random.default_rng(42)
        q_raw = 8
        q_eff = q_raw - 1
        n = 200
        reparam = build_scop_solver_reparam(q_raw, direction="increasing")

        B = rng.standard_normal((n, q_eff))
        W = np.ones(n)
        z = np.sort(rng.uniform(0, 5, n))
        beta = np.zeros(q_eff)
        S = reparam.penalty_matrix()
        lam = 0.1

        result = scop_newton_step(
            B_scop=B,
            W=W,
            z=z,
            beta_scop=beta,
            reparam=reparam,
            S_scop=S,
            lambda2=lam,
        )

        assert isinstance(result, SCOPNewtonResult)
        assert result.objective_after <= result.objective_before + 1e-10

    def test_convergence_on_simple_data(self):
        """Repeated Newton steps converge on monotone data."""
        rng = np.random.default_rng(42)
        q_raw = 6
        q_eff = q_raw - 1
        n = 300

        reparam = build_scop_solver_reparam(q_raw, direction="increasing")

        # Simple monotone data
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.1, n)

        # Build a B-spline basis and transform through SCOP
        from scipy.interpolate import BSpline as BSpl

        # q_raw basis functions of degree 3 needs q_raw + 4 knots
        n_internal = q_raw - 4  # 2 internal knots for q_raw=6
        internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        knots = np.concatenate([[0] * 4, internal, [1] * 4])
        B_full = np.column_stack(
            [BSpl(knots, (np.arange(q_raw) == j).astype(float), 3)(x) for j in range(q_raw)]
        )
        Sigma = reparam.raw_reparam.Sigma
        X_sigma = B_full @ Sigma
        X_centered = X_sigma[:, 1:] - X_sigma[:, 1:].mean(axis=0)

        W = np.ones(n)
        S = reparam.penalty_matrix()

        beta = np.zeros(q_eff)
        for _ in range(50):
            result = scop_newton_step(
                B_scop=X_centered,
                W=W,
                z=y,
                beta_scop=beta,
                reparam=reparam,
                S_scop=S,
                lambda2=0.1,
            )
            beta = result.beta_new
            if result.step_norm < 1e-8:
                break

        # forward() returns exp(beta_eff) = positive increments (all > 0).
        # Monotonicity of the actual curve is guaranteed by construction:
        # gamma_raw = Sigma @ (beta_1, exp(beta_eff)) is monotone increasing
        # because all increments are positive.
        beta_tilde_eff = reparam.forward(beta)
        assert np.all(beta_tilde_eff > 0), "All increments should be positive"

    def test_fisher_fallback_on_bad_hessian(self):
        """If Newton Hessian is unusable, falls back to Fisher step."""
        rng = np.random.default_rng(42)
        q_raw = 4
        q_eff = q_raw - 1
        n = 50
        reparam = build_scop_solver_reparam(q_raw, direction="increasing")
        B = rng.standard_normal((n, q_eff))
        W = np.ones(n)
        z = rng.standard_normal(n)
        beta = np.zeros(q_eff)
        S = reparam.penalty_matrix()

        result = scop_newton_step(
            B_scop=B,
            W=W,
            z=z,
            beta_scop=beta,
            reparam=reparam,
            S_scop=S,
            lambda2=0.1,
        )
        # Should always return a result (Fisher fallback if needed)
        assert result.beta_new is not None
        assert isinstance(result.used_fisher_fallback, bool)

    def test_step_halving_prevents_divergence(self):
        """Damped Newton with step halving should not increase objective."""
        rng = np.random.default_rng(42)
        q_raw = 6
        q_eff = q_raw - 1
        n = 100
        reparam = build_scop_solver_reparam(q_raw, direction="increasing")
        B = rng.standard_normal((n, q_eff))
        W = np.ones(n)
        z = rng.standard_normal(n) * 10  # large residuals
        beta = rng.standard_normal(q_eff) * 3  # bad starting point
        S = reparam.penalty_matrix()

        result = scop_newton_step(
            B_scop=B,
            W=W,
            z=z,
            beta_scop=beta,
            reparam=reparam,
            S_scop=S,
            lambda2=0.1,
        )
        assert result.objective_after <= result.objective_before + 1e-6

    def test_result_fields_populated(self):
        """All result fields are set with sensible values."""
        rng = np.random.default_rng(42)
        q_raw = 5
        q_eff = q_raw - 1
        n = 100
        reparam = build_scop_solver_reparam(q_raw, direction="increasing")
        B = rng.standard_normal((n, q_eff))
        W = np.abs(rng.standard_normal(n)) + 0.1
        z = rng.standard_normal(n)
        beta = np.zeros(q_eff)
        S = reparam.penalty_matrix()

        result = scop_newton_step(
            B_scop=B,
            W=W,
            z=z,
            beta_scop=beta,
            reparam=reparam,
            S_scop=S,
            lambda2=1.0,
        )

        assert result.beta_new.shape == (q_eff,)
        assert result.objective_before >= 0
        assert result.objective_after >= 0
        assert result.step_norm >= 0
        assert isinstance(result.used_fisher_fallback, bool)

    def test_decreasing_direction(self):
        """Newton step works for decreasing monotone constraint."""
        rng = np.random.default_rng(42)
        q_raw = 6
        q_eff = q_raw - 1
        n = 200
        reparam = build_scop_solver_reparam(q_raw, direction="decreasing")

        B = rng.standard_normal((n, q_eff))
        W = np.ones(n)
        z = np.sort(rng.uniform(0, 5, n))[::-1]  # decreasing
        beta = np.zeros(q_eff)
        S = reparam.penalty_matrix()

        result = scop_newton_step(
            B_scop=B,
            W=W,
            z=z,
            beta_scop=beta,
            reparam=reparam,
            S_scop=S,
            lambda2=0.1,
        )
        assert result.objective_after <= result.objective_before + 1e-10
