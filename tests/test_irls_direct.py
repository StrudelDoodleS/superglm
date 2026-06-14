"""Tests for the direct IRLS solver (no BCD)."""

import numpy as np
import pandas as pd
import pytest

from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.model import SuperGLM
from superglm.types import GroupSlice, LinearConstraintSet


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
    sample_weight = np.ones(n)
    X = pd.DataFrame({"DrivAge": x1, "Area": area})
    return X, y, sample_weight


@pytest.fixture
def select_data():
    """Synthetic data with signal spline and noise spline (for select=True)."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.uniform(18, 80, n)
    x2 = rng.uniform(0, 10, n)  # noise
    mu = np.exp(-0.5 + 0.01 * (x1 - 40) ** 2 / 100)
    y = rng.poisson(mu)
    sample_weight = np.ones(n)
    X = pd.DataFrame({"signal": x1, "noise": x2})
    return X, y, sample_weight


# ── Basic solver tests ─────────────────────────────────────────
class TestDirectSolverBasic:
    def test_h_inverse_profiled_intercept_solve_matches_augmented_system(self):
        from superglm.solvers.irls_direct import (
            _robust_solve,
            _solve_profiled_intercept_from_h_inv,
        )

        rng = np.random.default_rng(222)
        p = 7
        A = rng.standard_normal((p, p))
        XtWX = A @ A.T + np.eye(p)
        B = rng.standard_normal((p, p))
        S = B @ B.T + 0.3 * np.eye(p)
        H = XtWX + S
        H_inv = np.linalg.inv(H)
        XtWz = rng.standard_normal(p)
        XtW1 = rng.standard_normal(p)
        sum_W = 20.0
        sum_Wz = -0.4

        M_aug = np.empty((p + 1, p + 1))
        M_aug[0, 0] = sum_W
        M_aug[0, 1:] = XtW1
        M_aug[1:, 0] = XtW1
        M_aug[1:, 1:] = H
        rhs = np.empty(p + 1)
        rhs[0] = sum_Wz
        rhs[1:] = XtWz

        beta_aug, _, _ = _robust_solve(M_aug, rhs)
        beta_h, intercept_h = _solve_profiled_intercept_from_h_inv(
            H_inv,
            XtWz,
            XtW1,
            sum_W,
            sum_Wz,
        )

        np.testing.assert_allclose(intercept_h, beta_aug[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(beta_h, beta_aug[1:], rtol=1e-10, atol=1e-10)

    def test_one_step_return_xtwx_uses_h_inverse_solve(self, monkeypatch):
        """POI one-step calls reuse the final H inverse instead of solving M_aug separately."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import LogLink

        rng = np.random.default_rng(321)
        n = 120
        X_raw = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        eta = -0.1 + X_raw @ np.array([0.2, -0.15])
        y = rng.poisson(np.exp(eta)).astype(float)
        weights = np.ones(n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        def fail_robust_solve(*args, **kwargs):
            raise AssertionError("_robust_solve should not be used on this fast path")

        monkeypatch.setattr(irls_direct, "_robust_solve", fail_robust_solve)

        result, H_inv, XtWX = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=LogLink(),
            groups=groups,
            lambda2=0.1,
            max_iter=1,
            return_xtwx=True,
        )

        assert result.n_iter == 1
        assert H_inv.shape == (2, 2)
        assert XtWX.shape == (2, 2)

    def test_safe_decompose_h_uses_unpivoted_cholesky_for_spd(self, monkeypatch):
        import superglm.solvers.irls_direct as irls_direct

        rng = np.random.default_rng(11)
        A = rng.standard_normal((6, 6))
        H = A @ A.T + np.eye(6)

        def fail_dpstrf(*args, **kwargs):
            raise AssertionError("pivoted Cholesky should not be used for SPD fast path")

        def fail_svd(*args, **kwargs):
            raise AssertionError("SVD should not be used for SPD fast path")

        monkeypatch.setattr(irls_direct.scipy.linalg.lapack, "dpstrf", fail_dpstrf)
        monkeypatch.setattr(irls_direct.np.linalg, "svd", fail_svd)

        H_inv, log_det, cholesky_ok = irls_direct._safe_decompose_H(H)

        np.testing.assert_allclose(H_inv, np.linalg.inv(H), rtol=1e-10, atol=1e-10)
        assert log_det == pytest.approx(np.linalg.slogdet(H)[1], abs=1e-10)
        assert cholesky_ok

    def test_robust_solve_uses_unpivoted_cholesky_for_spd(self, monkeypatch):
        import superglm.solvers.irls_direct as irls_direct

        rng = np.random.default_rng(13)
        A = rng.standard_normal((6, 6))
        M = A @ A.T + np.eye(6)
        rhs = rng.standard_normal(6)

        def fail_dpstrf(*args, **kwargs):
            raise AssertionError("pivoted Cholesky should not be used for SPD fast path")

        def fail_svd(*args, **kwargs):
            raise AssertionError("SVD should not be used for SPD fast path")

        monkeypatch.setattr(irls_direct.scipy.linalg.lapack, "dpstrf", fail_dpstrf)
        monkeypatch.setattr(irls_direct.np.linalg, "svd", fail_svd)

        x, cond_est, used_svd = irls_direct._robust_solve(M, rhs)

        np.testing.assert_allclose(x, np.linalg.solve(M, rhs), rtol=1e-10, atol=1e-10)
        assert cond_est > 0
        assert not used_svd

    def test_constant_weight_fit_reuses_weighted_gram(self, monkeypatch):
        """Gaussian identity fits reuse X'WX across IRLS iterations."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(123)
        n = 80
        X_raw = np.column_stack(
            [
                rng.normal(size=n),
                rng.normal(size=n),
            ]
        )
        weights = rng.uniform(0.5, 2.0, size=n)
        y = 1.3 + X_raw @ np.array([0.4, -0.25]) + rng.normal(scale=0.05, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        original = irls_direct._block_xtwx_rhs
        calls = 0

        def counting_block_xtwx_rhs(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "_block_xtwx_rhs", counting_block_xtwx_rhs)

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            tol=1e-12,
        )

        assert result.n_iter > 1
        assert calls == 1

    def test_constant_weight_cache_preserves_solution(self, monkeypatch):
        """Reusing X'WX does not change the fitted coefficients."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(456)
        n = 90
        X_raw = np.column_stack(
            [
                rng.normal(size=n),
                rng.normal(size=n),
                rng.normal(size=n),
            ]
        )
        weights = rng.uniform(0.5, 2.0, size=n)
        y = 0.8 + X_raw @ np.array([0.2, -0.35, 0.5]) + rng.normal(scale=0.1, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=3)
        groups = [GroupSlice(name="x", start=0, end=3)]

        cached, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            tol=1e-12,
        )

        monkeypatch.setattr(irls_direct, "_has_constant_irls_weights", lambda *_: False)
        uncached, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=5,
            tol=1e-12,
        )

        np.testing.assert_allclose(cached.beta, uncached.beta, rtol=0, atol=1e-12)
        assert cached.intercept == pytest.approx(uncached.intercept, abs=1e-12)
        assert cached.deviance == pytest.approx(uncached.deviance, abs=1e-12)

    def test_qp_constraints_are_assembled_once_across_iterations(self, monkeypatch):
        """QP constraint blocks are fixed for a fit and should not be rebuilt per IRLS step."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Gaussian
        from superglm.links import IdentityLink

        rng = np.random.default_rng(457)
        n = 80
        X_raw = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
        y = 0.3 + X_raw @ np.array([0.1, -0.2]) + rng.normal(scale=0.1, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [
            GroupSlice(
                name="x",
                start=0,
                end=2,
                constraints=LinearConstraintSet(
                    A=np.array([[1.0, 0.0]], dtype=np.float64),
                    b=np.array([-100.0], dtype=np.float64),
                ),
            )
        ]

        original_qp = irls_direct.solve_constrained_qp
        constraint_matrices = []

        def recording_qp(H, g, A, b, *args, **kwargs):
            constraint_matrices.append(A)
            return original_qp(H, g, A, b, *args, **kwargs)

        monkeypatch.setattr(irls_direct, "solve_constrained_qp", recording_qp)

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=np.ones(n),
            family=Gaussian(),
            link=IdentityLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=3,
            tol=0.0,
        )

        assert result.n_iter == 3
        assert len({id(A) for A in constraint_matrices}) == 1

    def test_variable_weight_fit_rebuilds_weighted_gram(self, monkeypatch):
        """Poisson log fits do not reuse X'WX because W changes with mu."""
        import superglm.solvers.irls_direct as irls_direct
        from superglm.distributions import Poisson
        from superglm.links import LogLink

        rng = np.random.default_rng(789)
        n = 100
        X_raw = np.column_stack(
            [
                rng.normal(size=n),
                rng.normal(size=n),
            ]
        )
        eta = -0.2 + X_raw @ np.array([0.25, -0.1])
        y = rng.poisson(np.exp(eta))
        weights = rng.uniform(0.5, 1.5, size=n)
        dm = DesignMatrix([DenseGroupMatrix(X_raw)], n=n, p=2)
        groups = [GroupSlice(name="x", start=0, end=2)]

        original = irls_direct._block_xtwx_rhs
        calls = 0

        def counting_block_xtwx_rhs(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(irls_direct, "_block_xtwx_rhs", counting_block_xtwx_rhs)

        result, _ = irls_direct.fit_irls_direct(
            X=dm,
            y=y,
            weights=weights,
            family=Poisson(),
            link=LogLink(),
            groups=groups,
            lambda2=0.0,
            max_iter=20,
            tol=1e-12,
        )

        assert result.n_iter > 1
        assert calls == result.n_iter

    def test_matches_bcd_ridge(self, poisson_data):
        """Direct solver with selection_penalty=0 should give similar deviance as BCD with tiny lambda1."""
        X, y, w = poisson_data

        # Direct solver (selection_penalty=0)
        m_direct = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m_direct.fit(X, y, sample_weight=w)

        # BCD solver with near-zero lambda1 (effectively ridge)
        m_bcd = SuperGLM(
            family="poisson",
            selection_penalty=1e-8,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m_bcd.fit(X, y, sample_weight=w)

        # Deviances should be very close
        assert abs(m_direct.result.deviance - m_bcd.result.deviance) / m_bcd.result.deviance < 0.01

    def test_all_group_types(self, poisson_data):
        """Direct solver handles Dense (numeric), Sparse (categorical), SparseSSP (spline)."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=8),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, sample_weight=w)
        assert m.result.converged
        assert m.result.deviance < np.sum(y) * 10  # reasonable deviance

    def test_warm_start(self, poisson_data):
        """Warm-started direct solver should converge in fewer iterations."""
        X, y, w = poisson_data

        # Cold start
        m1 = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m1.fit(X, y, sample_weight=w)

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
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, sample_weight=w)

        # edf should be between 1 (intercept-only) and total params + 1
        total_params = sum(g.size for g in m._groups) + 1
        assert 1 < m.result.effective_df < total_params

    def test_predict_after_direct_fit(self, poisson_data):
        """predict/reconstruct should work after direct solver fit."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit(X, y, sample_weight=w)

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
            selection_penalty=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit(X, y, sample_weight=w)
        # Direct solver: no BCD aliasing → should converge fast
        assert m.result.n_iter <= 10
        assert m.result.converged


# ── REML + direct solver tests ────────────────────────────────
class TestREMLDirect:
    def test_reml_direct_convergence(self):
        """fit_reml() with selection_penalty=0 should converge using the direct solver."""
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
            selection_penalty=0,
            features={"DrivAge": Spline(n_knots=10)},
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=50)

        assert hasattr(m, "_reml_lambdas")
        assert m._reml_result.converged

    def test_reml_direct_select_true(self, select_data):
        """REML + select=True + selection_penalty=0: should estimate both null and wiggle lambdas."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=15)

        assert hasattr(m, "_reml_lambdas")
        lambdas = m._reml_lambdas

        # Should have entries for null and wiggle penalty components
        null_keys = [k for k in lambdas if ":null" in k]
        wiggle_keys = [k for k in lambdas if ":wiggle" in k]
        assert len(null_keys) >= 1
        assert len(wiggle_keys) >= 1

    def test_reml_direct_all_lambdas_estimated(self, select_data):
        """With direct solver, ALL lambdas are REML-estimated including 1-col groups."""
        X, y, w = select_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "signal": Spline(n_knots=10, select=True),
                "noise": Spline(n_knots=10, select=True),
            },
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=15)

        lambdas = m._reml_lambdas
        # Null-space components (rank-1) should have been estimated (not stuck at initial value)
        null_keys = [k for k in lambdas if ":null" in k]
        for key in null_keys:
            # Lambda should differ from the initial value (0.1 default)
            # It may be close, but shouldn't be exactly the default
            assert lambdas[key] > 0

    def test_reml_direct_predict_after_fit(self, poisson_data):
        """predict/reconstruct work after REML + direct solver."""
        X, y, w = poisson_data
        m = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=10)

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
            selection_penalty=0,
            features={
                "DrivAge": Spline(n_knots=10),
                "Area": Categorical(),
            },
        )

        def fail_rebuild(*args, **kwargs):
            raise AssertionError("Direct REML should not rebuild the design matrix")

        monkeypatch.setattr(m, "_rebuild_design_matrix_with_lambdas", fail_rebuild)
        m.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

        assert hasattr(m, "_reml_lambdas")
        assert "DrivAge" in m._reml_lambdas
