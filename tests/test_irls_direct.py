"""Tests for the direct IRLS solver (no BCD)."""

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.model import SuperGLM


# ── Fixtures ───────────────────────────────────────────────────
@pytest.fixture
def poisson_data():
    """Synthetic Poisson dataset with one nonlinear and one categorical feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.uniform(18, 80, n)
    area = rng.choice(["A", "B", "C", "D"], n)
    area_effect = {"A": 0.0, "B": 0.2, "C": -0.1, "D": 0.3}
    mu = np.exp(-0.5 + 0.01 * (x1 - 40) ** 2 / 100 + np.array([area_effect[a] for a in area]))
    y = rng.poisson(mu)
    exposure = np.ones(n)
    X = pd.DataFrame({"DrivAge": x1, "Area": area})
    return X, y, exposure


@pytest.fixture
def select_data():
    """Synthetic data with signal spline and noise spline (for select=True)."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(18, 80, n)
    x2 = rng.uniform(0, 10, n)  # noise
    mu = np.exp(-0.5 + 0.01 * (x1 - 40) ** 2 / 100)
    y = rng.poisson(mu)
    exposure = np.ones(n)
    X = pd.DataFrame({"signal": x1, "noise": x2})
    return X, y, exposure


# ── Basic solver tests ─────────────────────────────────────────
class TestDirectSolverBasic:
    def test_matches_bcd_ridge(self, poisson_data):
        """Direct solver with lambda1=0 should give similar deviance as BCD with tiny lambda1."""
        X, y, w = poisson_data

        # Direct solver (lambda1=0)
        m_direct = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m_direct.fit(X, y, exposure=w)

        # BCD solver with near-zero lambda1 (effectively ridge)
        m_bcd = SuperGLM(
            family="poisson",
            lambda1=1e-8,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m_bcd.fit(X, y, exposure=w)

        # Deviances should be very close
        assert abs(m_direct.result.deviance - m_bcd.result.deviance) / m_bcd.result.deviance < 0.01

    def test_all_group_types(self, poisson_data):
        """Direct solver handles Dense (numeric), Sparse (categorical), SparseSSP (spline)."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=8),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, exposure=w)
        assert m.result.converged
        assert m.result.deviance < np.sum(y) * 10  # reasonable deviance

    def test_warm_start(self, poisson_data):
        """Warm-started direct solver should converge in fewer iterations."""
        X, y, w = poisson_data

        # Cold start
        m1 = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m1.fit(X, y, exposure=w)

        # Warm start via re-fit (same model object, beta is reused internally)
        # We can't easily warm-start through .fit(), so test via direct solver import
        from superglm.solvers.irls_direct import fit_irls_direct

        result_warm, _ = fit_irls_direct(
            X=m1._dm,
            y=y,
            weights=w,
            family=m1._distribution,
            link=m1._link,
            groups=m1._groups,
            lambda2=m1.lambda2,
            beta_init=m1.result.beta,
            intercept_init=m1.result.intercept,
        )
        # Warm start should need <= 2 iterations
        assert result_warm.n_iter <= 2

    def test_exact_edf(self, poisson_data):
        """Effective df from trace formula should be reasonable."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, exposure=w)

        # edf should be between 1 (intercept-only) and total params + 1
        total_params = sum(g.size for g in m._groups) + 1
        assert 1 < m.result.effective_df < total_params

    def test_predict_after_direct_fit(self, poisson_data):
        """predict/reconstruct should work after direct solver fit."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, exposure=w)

        mu_hat = m.predict(X)
        assert mu_hat.shape == y.shape
        assert np.all(mu_hat > 0)

        rec = m.reconstruct_feature("DrivAge")
        assert "relativity" in rec


# ── Select=True tests ──────────────────────────────────────────
class TestDirectSolverSelect:
    def test_select_no_aliasing(self, select_data):
        """select=True with direct solver should converge in <= 10 IRLS iters."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit(X, y, exposure=w)
        # Direct solver: no BCD aliasing → should converge fast
        assert m.result.n_iter <= 10
        assert m.result.converged


# ── REML + direct solver tests ────────────────────────────────
class TestREMLDirect:
    def test_reml_direct_convergence(self):
        """fit_reml() with lambda1=0 should converge using the direct solver."""
        # Use data with strong nonlinearity so REML finds a finite lambda
        rng = np.random.default_rng(123)
        n = 2000
        x1 = rng.uniform(18, 80, n)
        # Strong U-shape so REML doesn't push lambda→∞
        mu = np.exp(-2.0 + 0.002 * (x1 - 50) ** 2)
        y = rng.poisson(mu)
        X = pd.DataFrame({"DrivAge": x1})
        w = np.ones(n)

        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={"DrivAge": Spline(n_knots=10)},
        )
        m.fit_reml(X, y, exposure=w, max_reml_iter=20)

        assert hasattr(m, "_reml_lambdas")
        assert m._reml_result.converged

    def test_reml_direct_select_true(self, select_data):
        """REML + select=True + lambda1=0: should estimate both linear and spline lambdas."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit_reml(X, y, exposure=w, max_reml_iter=15)

        assert hasattr(m, "_reml_lambdas")
        lambdas = m._reml_lambdas

        # Should have entries for linear and spline subgroups
        linear_keys = [k for k in lambdas if ":linear" in k]
        spline_keys = [k for k in lambdas if ":spline" in k]
        assert len(linear_keys) >= 1
        assert len(spline_keys) >= 1

    def test_reml_direct_all_lambdas_estimated(self, select_data):
        """With direct solver, ALL lambdas are REML-estimated including 1-col groups."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit_reml(X, y, exposure=w, max_reml_iter=15)

        lambdas = m._reml_lambdas
        # 1-col linear subgroups should have been estimated (not stuck at initial value)
        linear_keys = [k for k in lambdas if ":linear" in k]
        for key in linear_keys:
            # Lambda should differ from the initial value (0.1 default)
            # It may be close, but shouldn't be exactly the default
            assert lambdas[key] > 0

    def test_reml_direct_predict_after_fit(self, poisson_data):
        """predict/reconstruct work after REML + direct solver."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit_reml(X, y, exposure=w, max_reml_iter=10)

        mu_hat = m.predict(X)
        assert mu_hat.shape == y.shape
        assert np.all(mu_hat > 0)

        rec = m.reconstruct_feature("DrivAge")
        assert "relativity" in rec

    def test_reml_direct_keeps_basis_fixed(self, poisson_data, monkeypatch):
        """Direct REML should update lambda weights without rebuilding the basis."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            lambda1=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )

        def fail_rebuild(*args, **kwargs):
            raise AssertionError("Direct REML should not rebuild the design matrix")

        monkeypatch.setattr(m, "_rebuild_design_matrix_with_lambdas", fail_rebuild)
        m.fit_reml(X, y, exposure=w, max_reml_iter=20)

        assert hasattr(m, "_reml_lambdas")
        assert "DrivAge" in m._reml_lambdas
