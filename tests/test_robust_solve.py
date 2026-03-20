"""Tests for _robust_solve() and _safe_decompose_H() condition-aware solvers."""

import numpy as np

from superglm.solvers.irls_direct import _robust_solve, _safe_decompose_H


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
        assert cond_est < 1e10
        np.testing.assert_allclose(M @ x, rhs, atol=1e-10)

    def test_ill_conditioned_uses_svd(self):
        """System with cond ~1e14 should trigger SVD fallback."""
        # Build a matrix with prescribed condition number ~1e14
        rng = np.random.default_rng(123)
        n = 20
        U, _, Vt = np.linalg.svd(rng.standard_normal((n, n)))
        s = np.logspace(0, -14, n)  # cond = 1e14
        M = (U * s) @ U.T  # symmetric
        rhs = rng.standard_normal(n)

        x, cond_est, used_svd = _robust_solve(M, rhs)

        assert used_svd
        # SVD truncates near-zero singular values, so the residual in the
        # well-conditioned subspace should be small
        residual = M @ x - rhs
        assert np.linalg.norm(residual) < np.linalg.norm(rhs)
        assert np.all(np.isfinite(x))

    def test_singular_system_pseudo_inverse(self):
        """Exactly singular system should still return a solution via SVD."""
        n = 10
        # Rank-deficient: last eigenvalue is 0
        rng = np.random.default_rng(99)
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        s = np.ones(n)
        s[-1] = 0.0  # exactly singular
        M = (U * s) @ U.T
        rhs = rng.standard_normal(n)

        x, cond_est, used_svd = _robust_solve(M, rhs)

        assert used_svd
        # Solution should be in the column space of M
        residual = M @ x - rhs
        # The component in the range space should be solved exactly
        # (residual is only in the null space)
        assert np.linalg.norm(residual) < np.linalg.norm(rhs)

    def test_threshold_boundary(self):
        """Custom threshold: cond just below threshold uses Cholesky."""
        rng = np.random.default_rng(42)
        n = 5
        U, _, _ = np.linalg.svd(rng.standard_normal((n, n)))
        s = np.array([1e4, 1e3, 1e2, 1e1, 1.0])  # cond = 1e4
        M = (U * s) @ U.T
        rhs = rng.standard_normal(n)

        # With threshold 1e5 (> cond 1e4) → Cholesky
        _, _, used_svd = _robust_solve(M, rhs, cond_threshold=1e5)
        assert not used_svd

        # With threshold 1e3 (< cond 1e4) → SVD
        _, _, used_svd = _robust_solve(M, rhs, cond_threshold=1e3)
        assert used_svd


class TestSafeDecomposeH:
    """Tests for _safe_decompose_H with condition-aware SVD fallback."""

    def test_well_conditioned_cholesky_path(self):
        """Well-conditioned H should use Cholesky and return correct inverse."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 8))
        H = A.T @ A + 5 * np.eye(8)

        H_inv, log_det, cholesky_ok = _safe_decompose_H(H)

        assert cholesky_ok
        np.testing.assert_allclose(H_inv @ H, np.eye(8), atol=1e-10)
        np.testing.assert_allclose(log_det, np.log(np.linalg.det(H)), rtol=1e-10)

    def test_ill_conditioned_svd_fallback(self):
        """Ill-conditioned H should fall back to SVD and still produce usable inverse."""
        # Use diagonal matrix so condition number is exact and predictable
        n = 10
        s = np.logspace(0, -14, n)  # cond = 1e14 >> 1e10 threshold
        H = np.diag(s)

        H_inv, log_det, cholesky_ok = _safe_decompose_H(H)

        assert not cholesky_ok
        assert np.all(np.isfinite(H_inv))
        assert np.isfinite(log_det)

    def test_singular_H_svd_fallback(self):
        """Rank-deficient H (Cholesky fails) should fall back to SVD."""
        n = 8
        rng = np.random.default_rng(77)
        A = rng.standard_normal((n, n - 2))  # rank n-2
        H = A @ A.T  # rank-deficient

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

        # Only the first 4 eigenvalues should contribute
        expected = np.sum(np.log([100.0, 10.0, 1.0, 0.1]))
        np.testing.assert_allclose(log_det, expected, atol=1.0)
