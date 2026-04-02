"""Tests for _robust_solve(), _safe_decompose_H(), and QR solver path."""

import logging

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.penalties.group_lasso import GroupLasso
from superglm.solvers.irls_direct import _robust_solve, _safe_decompose_H
from superglm.tweedie_profile import generate_tweedie_cpg


class TestRobustSolve:
    """Unit tests for the Cholesky + SVD fallback linear solver."""

    def test_well_conditioned_uses_cholesky(self):
        """Well-conditioned SPD system should use the fast Cholesky path."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10))
        M = A.T @ A + np.eye(10)  # SPD, cond ~ O(1)
        rhs = rng.standard_normal(10)

        x, cond_est, used_svd = _robust_solve(M, rhs)

        assert not used_svd
        np.testing.assert_allclose(M @ x, rhs, atol=1e-10)

    def test_ill_conditioned_diagonal_cholesky_exact(self):
        """Diagonal system: Cholesky is exact even at cond ~1e14 (no roundoff)."""
        n = 20
        s = np.logspace(0, -14, n)
        M = np.diag(s)
        rhs = np.ones(n)

        x, cond_est, used_svd = _robust_solve(M, rhs)

        # Cholesky on diagonal is element-wise sqrt → cho_solve is exact
        assert not used_svd
        np.testing.assert_allclose(M @ x, rhs, atol=1e-10)

    def test_ill_conditioned_dense_uses_svd(self):
        """Dense (non-diagonal) SPD with cond ~1e14 must trigger SVD fallback.

        The Cholesky diagonal ratio can underreport condition by orders of
        magnitude on non-diagonal matrices.  The residual check catches this.
        """
        rng = np.random.default_rng(123)
        n = 10
        # Random orthogonal basis — off-diagonal structure means diag(L)
        # does NOT reflect the true eigenvalues.
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.logspace(0, -14, n)
        M = (Q * s) @ Q.T  # symmetric, cond = 1e14
        rhs = rng.standard_normal(n)

        x, cond_est, used_svd = _robust_solve(M, rhs)

        assert used_svd, (
            f"Expected SVD fallback for cond~1e14 dense matrix, "
            f"but Cholesky was used (cond_est={cond_est:.1e})"
        )
        assert np.all(np.isfinite(x))

    def test_singular_system_pseudo_inverse(self):
        """Exactly singular system should still return a solution via SVD."""
        n = 10
        rng = np.random.default_rng(99)
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        s = np.ones(n)
        s[-1] = 0.0
        M = (U * s) @ U.T
        rhs = rng.standard_normal(n)

        x, cond_est, used_svd = _robust_solve(M, rhs)

        assert used_svd
        residual = M @ x - rhs
        assert np.linalg.norm(residual) < np.linalg.norm(rhs)

    def test_moderate_condition_uses_cholesky(self):
        """Moderate condition (~1e4) should stay on Cholesky — solve is accurate."""
        rng = np.random.default_rng(42)
        n = 10
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.logspace(0, -4, n)  # cond = 1e4
        M = (Q * s) @ Q.T
        rhs = rng.standard_normal(n)

        x, _, used_svd = _robust_solve(M, rhs)

        assert not used_svd
        np.testing.assert_allclose(M @ x, rhs, atol=1e-6)


class TestSafeDecomposeH:
    """Tests for _safe_decompose_H with residual-checked SVD fallback."""

    def test_well_conditioned_cholesky_path(self):
        """Well-conditioned H should use Cholesky and return correct inverse."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 8))
        H = A.T @ A + 5 * np.eye(8)

        H_inv, log_det, cholesky_ok = _safe_decompose_H(H)

        assert cholesky_ok
        np.testing.assert_allclose(H_inv @ H, np.eye(8), atol=1e-10)
        np.testing.assert_allclose(log_det, np.log(np.linalg.det(H)), rtol=1e-10)

    def test_ill_conditioned_diagonal_cholesky_exact(self):
        """Diagonal H: Cholesky inverse is exact even at cond ~1e14."""
        n = 10
        s = np.logspace(0, -14, n)
        H = np.diag(s)

        H_inv, log_det, cholesky_ok = _safe_decompose_H(H)

        # Cholesky on diagonal is exact — residual check passes
        assert cholesky_ok
        np.testing.assert_allclose(np.diag(H_inv), 1.0 / s, rtol=1e-10)

    def test_ill_conditioned_dense_svd_fallback(self):
        """Dense H with cond ~1e14 must trigger SVD (catches diagonal heuristic bug)."""
        rng = np.random.default_rng(123)
        n = 10
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.logspace(0, -14, n)
        H = (Q * s) @ Q.T

        H_inv, log_det, cholesky_ok = _safe_decompose_H(H)

        assert not cholesky_ok, "Expected SVD fallback for cond~1e14 dense matrix"
        assert np.all(np.isfinite(H_inv))
        assert np.isfinite(log_det)

    def test_singular_H_svd_fallback(self):
        """Rank-deficient H (Cholesky fails) should fall back to SVD."""
        n = 8
        rng = np.random.default_rng(77)
        A = rng.standard_normal((n, n - 2))
        H = A @ A.T

        H_inv, log_det, cholesky_ok = _safe_decompose_H(H)

        assert not cholesky_ok
        assert np.isfinite(log_det)
        assert np.all(np.isfinite(H_inv))

    def test_log_det_positive_eigenvalues_only(self):
        """log_det should only sum over positive singular values."""
        n = 6
        rng = np.random.default_rng(55)
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        s = np.array([100.0, 10.0, 1.0, 0.1, 1e-15, 1e-16])
        H = (U * s) @ U.T

        _, log_det, _ = _safe_decompose_H(H)

        expected = np.sum(np.log([100.0, 10.0, 1.0, 0.1]))
        np.testing.assert_allclose(log_det, expected, atol=1.0)


class TestQRSolverPath:
    """Tests for the QR-based direct solver (direct_solve='qr')."""

    @pytest.fixture()
    def small_tweedie_data(self):
        """5k Tweedie dataset for QR tests (small n only)."""
        rng = np.random.default_rng(2026)
        n = 5000
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = -1.0 + 0.5 * x1 - 0.3 * x2
        mu = np.exp(eta)
        y = generate_tweedie_cpg(n, mu=mu, phi=2.0, p=1.5, rng=rng)
        return pd.DataFrame({"x1": x1, "x2": x2}), y

    def test_qr_matches_gram_well_conditioned(self, small_tweedie_data):
        """QR and gram produce identical beta on well-conditioned data."""
        df, y = small_tweedie_data

        m_gram = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric(), "x2": Numeric()},
            direct_solve="gram",
        )
        m_gram.fit(df, y)

        m_qr = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Numeric(), "x2": Numeric()},
            direct_solve="qr",
        )
        m_qr.fit(df, y)

        np.testing.assert_allclose(m_qr._result.beta, m_gram._result.beta, atol=1e-10)
        np.testing.assert_allclose(m_qr._result.intercept, m_gram._result.intercept, atol=1e-10)
        np.testing.assert_allclose(m_qr._result.deviance, m_gram._result.deviance, rtol=1e-10)

    def test_qr_tweedie_convergence(self, small_tweedie_data):
        """QR path converges on Tweedie with zeros."""
        df, y = small_tweedie_data
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x1": Spline(n_knots=8), "x2": Spline(n_knots=8)},
            direct_solve="qr",
        )
        model.fit(df, y)
        assert model._result.converged
        assert np.isfinite(model._result.deviance)

    def test_auto_near_collinear_converges(self, caplog):
        """'auto' mode handles near-collinear data without SVD fallback.

        With pivoted Cholesky (Higham Ch. 10.3), near-collinear systems
        that previously triggered repeated SVD fallbacks are now handled
        directly by the rank-revealing decomposition.
        """
        rng = np.random.default_rng(42)
        n = 5000
        # Nested categoricals → near-collinearity
        region = rng.choice(10, n)
        sub_region = region * 3 + rng.choice(3, n)
        age = rng.uniform(18, 80, n)
        eta = -2.0 + rng.standard_normal(10)[region] * 0.3
        eta += rng.standard_normal(30)[sub_region] * 0.1
        mu = np.exp(eta)
        y = generate_tweedie_cpg(n, mu=mu, phi=2.0, p=1.5, rng=rng)

        df = pd.DataFrame(
            {
                "region": pd.Categorical(region),
                "sub_region": pd.Categorical(sub_region),
                "age": age,
            }
        )
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={
                "region": Categorical(),
                "sub_region": Categorical(),
                "age": Numeric(),
            },
            direct_solve="auto",
        )
        with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
            model.fit(df, y)
        assert model._result.converged
        # Pivoted Cholesky should handle this without repeated SVD fallbacks.
        # Lock in the improvement: assert the warning is absent.
        assert not any("consecutive SVD fallbacks" in r.message for r in caplog.records)

    def test_gram_no_warning(self, caplog):
        """'gram' mode suppresses SVD fallback warnings."""
        rng = np.random.default_rng(42)
        n = 5000
        region = rng.choice(10, n)
        sub_region = region * 3 + rng.choice(3, n)
        age = rng.uniform(18, 80, n)
        eta = -2.0 + rng.standard_normal(10)[region] * 0.3
        eta += rng.standard_normal(30)[sub_region] * 0.1
        mu = np.exp(eta)
        y = generate_tweedie_cpg(n, mu=mu, phi=2.0, p=1.5, rng=rng)

        df = pd.DataFrame(
            {
                "region": pd.Categorical(region),
                "sub_region": pd.Categorical(sub_region),
                "age": age,
            }
        )
        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={
                "region": Categorical(),
                "sub_region": Categorical(),
                "age": Numeric(),
            },
            direct_solve="gram",
        )
        with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
            model.fit(df, y)
        assert not any("consecutive SVD fallbacks" in r.message for r in caplog.records)

    def test_invalid_direct_solve_raises(self):
        """Invalid direct_solve value raises ValueError."""
        with pytest.raises(ValueError, match="direct_solve"):
            SuperGLM(direct_solve="invalid")
