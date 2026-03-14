"""Tests for the EFS (Extended Fellner-Schall) REML optimizer path.

Split from test_reml.py — these test the BCD-based REML path
(lambda1 > 0) specifically.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import Spline


class TestEFSOptimizer:
    """Tests specific to the EFS REML path (lambda1 > 0)."""

    def test_efs_poisson_converges(self):
        """EFS should converge for Poisson with group lasso."""
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x1) + 0.2 * np.cos(x2))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit_reml(X, y, max_reml_iter=20)

        assert model._reml_result.converged
        assert len(model._reml_lambdas) == 2
        for name, lam in model._reml_lambdas.items():
            assert np.isfinite(lam), f"Non-finite lambda for {name}"
            assert lam > 0, f"Non-positive lambda for {name}"

    def test_efs_gamma_estimated_scale(self, capsys):
        """EFS should handle estimated-scale families (Gamma)."""
        rng = np.random.default_rng(123)
        n = 600
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        mu = np.exp(0.3 + 0.35 * np.sin(x1) + 0.15 * np.cos(x2))
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        y = np.maximum(y, 1e-4)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="gamma",
            lambda1=0.01,
            features={
                "x1": Spline(n_knots=6, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit_reml(X, y, max_reml_iter=12, verbose=True)

        out = capsys.readouterr().out
        assert "REML iter=" in out
        assert "(pirls=" in out
        assert "(cheap)" in out

        assert model.result.converged
        assert np.isfinite(model.result.phi)
        assert model.result.phi > 0

    def test_efs_scalar_groups_estimated(self):
        """EFS should estimate lambdas for scalar (rank-1) groups via select=True."""
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.4 * np.sin(x1))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp", select=True)},
        )
        model.fit_reml(X, y, max_reml_iter=20)

        assert model._reml_result.converged
        # Both linear (rank 1) and spline subgroups should have lambdas
        assert "x1:linear" in model._reml_lambdas
        assert "x1:spline" in model._reml_lambdas
        for name, lam in model._reml_lambdas.items():
            assert np.isfinite(lam), f"Non-finite lambda for {name}"
            assert lam > 0, f"Non-positive lambda for {name}"

    def test_efs_bootstrap_and_fast_convergence(self, capsys):
        """EFS bootstrap should give good initial lambdas → fast convergence."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x1))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=6, penalty="ssp")},
        )
        model.fit_reml(X, y, max_reml_iter=20, verbose=True)

        out = capsys.readouterr().out
        # Bootstrap line should appear
        assert "REML bootstrap:" in out
        assert model._reml_result.converged
        # With bootstrap, should converge quickly (≤ 6 REML iters)
        assert model._reml_result.n_reml_iter <= 6

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_efs_bad_starts_converge(self, family):
        """EFS should converge from adverse lambda2_init with enough iterations."""
        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 10, n)
        if family == "poisson":
            mu = np.exp(0.5 + 0.3 * np.sin(x1) + 0.2 * np.cos(x2))
            y = rng.poisson(mu).astype(float)
        else:
            mu = np.exp(0.3 + 0.35 * np.sin(x1) + 0.15 * np.cos(x2))
            y = rng.gamma(shape=5.0, scale=mu / 5.0)
            y = np.maximum(y, 1e-4)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        features = {
            "x1": Spline(n_knots=8, penalty="ssp"),
            "x2": Spline(n_knots=8, penalty="ssp"),
        }

        # Baseline: default start
        baseline = SuperGLM(family=family, lambda1=0.01, features=features)
        baseline.fit_reml(X, y, max_reml_iter=30)
        assert baseline._reml_result.converged

        # Very small and very large lambda2_init
        for init_val in [1e-6, 1e5]:
            m = SuperGLM(family=family, lambda1=0.01, features=features)
            m.fit_reml(X, y, max_reml_iter=30, lambda2_init=init_val)
            assert m._reml_result.converged, f"{family} lambda2_init={init_val} did not converge"

            # Deviance should agree with baseline
            dev_rel = abs(m.result.deviance - baseline.result.deviance) / abs(
                baseline.result.deviance
            )
            assert dev_rel < 1e-3, f"{family} init={init_val} deviance rel diff {dev_rel:.6f}"

    def test_efs_objective_consistent_after_cheap_exit(self):
        """REML objective should use fresh penalty caches after final DM rebuild."""
        from superglm.reml import build_penalty_caches
        from superglm.reml_optimizer import reml_laml_objective

        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x1))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x1": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X, y, max_reml_iter=20)
        assert model._reml_result.converged

        # Recompute objective with fresh caches from scratch
        reml_groups = [(i, g) for i, g in enumerate(model._groups) if g.penalized]
        fresh_caches = build_penalty_caches(model._dm.group_matrices, model._groups, reml_groups)
        fresh_obj = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            model._reml_result.pirls_result,
            model._reml_lambdas,
            np.ones(n),
            np.zeros(n),
            penalty_caches=fresh_caches,
        )

        # Should match the stored objective closely
        stored_obj = model._reml_result.objective
        rel_diff = abs(stored_obj - fresh_obj) / abs(fresh_obj)
        assert rel_diff < 1e-10, (
            f"Objective mismatch: stored={stored_obj:.10f} fresh={fresh_obj:.10f} "
            f"rel_diff={rel_diff:.2e}"
        )

    def test_efs_agrees_with_direct_at_lambda1_zero(self):
        """When lambda1=0, EFS and direct paths should give similar results.

        This validates the EFS update formula produces correct REML estimates
        by comparing against the Newton-based direct solver.
        """
        rng = np.random.default_rng(42)
        n = 1000
        x1 = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.4 * np.sin(x1))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1})

        features = {"x1": Spline(n_knots=8, penalty="ssp")}

        # Direct path (lambda1=0)
        m_direct = SuperGLM(family="poisson", lambda1=0, features=features)
        m_direct.fit_reml(X, y, max_reml_iter=20)

        # EFS path (lambda1=tiny, to trigger BCD)
        m_efs = SuperGLM(family="poisson", lambda1=1e-8, features=features)
        m_efs.fit_reml(X, y, max_reml_iter=20)

        assert m_direct._reml_result.converged
        assert m_efs._reml_result.converged

        # Deviance should agree closely
        dev_rel = abs(m_direct.result.deviance - m_efs.result.deviance) / abs(
            m_direct.result.deviance
        )
        assert dev_rel < 5e-3, f"Deviance rel diff {dev_rel:.6f}"

        # REML lambdas should be in the same ballpark (EFS fixed-point vs
        # Newton can settle at different points on a flat REML surface)
        for name in m_direct._reml_lambdas:
            lam_d = m_direct._reml_lambdas[name]
            lam_e = m_efs._reml_lambdas[name]
            log_diff = abs(np.log(lam_d) - np.log(lam_e))
            assert log_diff < 3.5, f"{name} log-lambda diff {log_diff:.4f}"
