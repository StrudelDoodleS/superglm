"""Tests for Wood (2011) Appendix B multi-penalty log-determinant kernel."""

import numpy as np

from superglm.multi_penalty import (
    logdet_s_gradient,
    logdet_s_hessian,
    similarity_transform_logdet,
)


def _make_psd(q: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Make a random PSD matrix of given rank."""
    A = rng.standard_normal((q, rank))
    return A @ A.T


class TestSimilarityTransformLogdet:
    """Core correctness tests for the Appendix B algorithm."""

    def test_full_rank_well_conditioned(self):
        """log|S|+ matches slogdet on full-rank well-conditioned S."""
        rng = np.random.default_rng(42)
        q = 10
        S1 = _make_psd(q, q, rng) + 0.1 * np.eye(q)
        S2 = _make_psd(q, q, rng) + 0.1 * np.eye(q)
        lambdas = np.array([2.0, 0.5])

        result = similarity_transform_logdet([S1, S2], lambdas)

        S_total = lambdas[0] * S1 + lambdas[1] * S2
        _, expected_logdet = np.linalg.slogdet(S_total)

        assert result.rank == q
        np.testing.assert_allclose(result.logdet_s_plus, expected_logdet, rtol=1e-10)

    def test_rank_deficient_single_penalty(self):
        """Rank-deficient S: log|S|+ uses positive eigenvalues only."""
        rng = np.random.default_rng(42)
        q = 10
        rank = 7
        S1 = _make_psd(q, rank, rng)
        lambdas = np.array([3.0])

        result = similarity_transform_logdet([S1], lambdas)

        # Expected: log of positive eigenvalues of lambda * S1
        eigvals = np.linalg.eigvalsh(lambdas[0] * S1)
        pos = eigvals[eigvals > 1e-8 * eigvals.max()]
        expected = float(np.sum(np.log(pos)))

        assert result.rank == rank
        np.testing.assert_allclose(result.logdet_s_plus, expected, rtol=1e-6)

    def test_two_overlapping_penalties(self):
        """Two penalties with overlapping range spaces: correct rank detection."""
        rng = np.random.default_rng(42)
        q = 8
        # S1 has rank 5, S2 has rank 5, but they share 3 directions
        A = rng.standard_normal((q, 5))
        B = np.copy(A)
        B[:, 3:] = rng.standard_normal((q, 2))  # replace 2 columns
        S1 = A @ A.T
        S2 = B @ B.T
        lambdas = np.array([1.0, 1.0])

        result = similarity_transform_logdet([S1, S2], lambdas)

        # Combined S should have rank = 7 (5 + 5 - 3 shared)
        S_total = S1 + S2
        eigvals = np.linalg.eigvalsh(S_total)
        expected_rank = int(np.sum(eigvals > 1e-8 * eigvals.max()))
        expected_logdet = float(np.sum(np.log(eigvals[eigvals > 1e-8 * eigvals.max()])))

        assert result.rank == expected_rank
        np.testing.assert_allclose(result.logdet_s_plus, expected_logdet, rtol=1e-6)

    def test_extreme_lambda_ratio(self):
        """Stable at extreme lambda ratios (1e-8 to 1e8)."""
        rng = np.random.default_rng(42)
        q = 8
        S1 = _make_psd(q, q, rng) + 0.01 * np.eye(q)
        S2 = _make_psd(q, q, rng) + 0.01 * np.eye(q)
        lambdas = np.array([1e8, 1e-8])

        result = similarity_transform_logdet([S1, S2], lambdas)

        S_total = lambdas[0] * S1 + lambdas[1] * S2
        _, expected = np.linalg.slogdet(S_total)

        assert result.rank == q
        # At extreme ratios, the small penalty contributes negligibly
        # so we allow slightly looser tolerance
        np.testing.assert_allclose(result.logdet_s_plus, expected, rtol=1e-6)

    def test_single_penalty_matches_eigvals(self):
        """Single-penalty case matches simple eigenvalue computation."""
        rng = np.random.default_rng(42)
        q = 10
        S1 = _make_psd(q, 8, rng)
        lam = 5.0
        lambdas = np.array([lam])

        result = similarity_transform_logdet([S1], lambdas)

        eigvals = np.linalg.eigvalsh(lam * S1)
        pos = eigvals[eigvals > 1e-10 * eigvals.max()]
        expected = float(np.sum(np.log(pos)))

        np.testing.assert_allclose(result.logdet_s_plus, expected, rtol=1e-8)

    def test_q_s_shape_matches_rank(self):
        """Q_s shape is (q, rank), not (q, q)."""
        rng = np.random.default_rng(42)
        q = 6
        S1 = _make_psd(q, 3, rng)  # rank 3
        lambdas = np.array([1.0])

        result = similarity_transform_logdet([S1], lambdas)

        assert result.Q_s.shape == (q, result.rank)
        # Columns should be orthonormal
        QtQ = result.Q_s.T @ result.Q_s
        np.testing.assert_allclose(QtQ, np.eye(result.rank), atol=1e-10)

    def test_all_zero_penalties(self):
        """Degenerate case: all-zero penalties → rank 0, logdet 0."""
        q = 5
        S1 = np.zeros((q, q))
        lambdas = np.array([1.0])

        result = similarity_transform_logdet([S1], lambdas)

        assert result.rank == 0
        assert result.logdet_s_plus == 0.0

    def test_pinv_plus_is_correct(self):
        """S_pinv_plus @ S ≈ I on the positive subspace."""
        rng = np.random.default_rng(42)
        q = 8
        S1 = _make_psd(q, q, rng) + 0.1 * np.eye(q)
        S2 = _make_psd(q, q, rng) + 0.1 * np.eye(q)
        lambdas = np.array([2.0, 0.5])

        result = similarity_transform_logdet([S1, S2], lambdas)
        S_total = lambdas[0] * S1 + lambdas[1] * S2

        product = result.S_pinv_plus @ S_total
        np.testing.assert_allclose(product, np.eye(q), atol=1e-10)

    def test_e_sqrt_satisfies_ete_equals_s(self):
        """E'E ≈ S for the stable square root."""
        rng = np.random.default_rng(42)
        q = 8
        S1 = _make_psd(q, q, rng) + 0.1 * np.eye(q)
        lambdas = np.array([2.0])

        result = similarity_transform_logdet([S1], lambdas)
        S_total = lambdas[0] * S1

        EtE = result.E_sqrt.T @ result.E_sqrt
        np.testing.assert_allclose(EtE, S_total, rtol=1e-6, atol=1e-10)


class TestLogdetDerivatives:
    """FD tests for gradient and Hessian of log|S|+."""

    @staticmethod
    def _setup(q=8, M=3, seed=42):
        rng = np.random.default_rng(seed)
        penalties = [_make_psd(q, q, rng) + 0.01 * np.eye(q) for _ in range(M)]
        lambdas = np.exp(rng.standard_normal(M))
        return penalties, lambdas

    def test_gradient_matches_fd(self):
        """Analytic gradient matches central FD of log|S|+ w.r.t. rho."""
        penalties, lambdas = self._setup()
        M = len(lambdas)
        rhos = np.log(lambdas)

        result = similarity_transform_logdet(penalties, lambdas)
        grad = logdet_s_gradient(result, penalties, lambdas)

        eps = 1e-5
        fd_grad = np.zeros(M)
        for j in range(M):
            rhos_p = rhos.copy()
            rhos_p[j] += eps
            rhos_m = rhos.copy()
            rhos_m[j] -= eps
            res_p = similarity_transform_logdet(penalties, np.exp(rhos_p))
            res_m = similarity_transform_logdet(penalties, np.exp(rhos_m))
            fd_grad[j] = (res_p.logdet_s_plus - res_m.logdet_s_plus) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-8)

    def test_hessian_matches_fd(self):
        """Analytic Hessian matches FD of gradient."""
        penalties, lambdas = self._setup()
        M = len(lambdas)
        rhos = np.log(lambdas)

        result = similarity_transform_logdet(penalties, lambdas)
        hess = logdet_s_hessian(result, penalties, lambdas)

        eps = 1e-4
        fd_hess = np.zeros((M, M))
        for j in range(M):
            rhos_p = rhos.copy()
            rhos_p[j] += eps
            rhos_m = rhos.copy()
            rhos_m[j] -= eps
            res_p = similarity_transform_logdet(penalties, np.exp(rhos_p))
            res_m = similarity_transform_logdet(penalties, np.exp(rhos_m))
            grad_p = logdet_s_gradient(res_p, penalties, np.exp(rhos_p))
            grad_m = logdet_s_gradient(res_m, penalties, np.exp(rhos_m))
            fd_hess[:, j] = (grad_p - grad_m) / (2 * eps)

        np.testing.assert_allclose(hess, fd_hess, rtol=0.05, atol=1e-6)

    def test_hessian_symmetric(self):
        """Hessian should be symmetric."""
        penalties, lambdas = self._setup()
        result = similarity_transform_logdet(penalties, lambdas)
        hess = logdet_s_hessian(result, penalties, lambdas)
        np.testing.assert_allclose(hess, hess.T, atol=1e-12)

    def test_gradient_rank_deficient(self):
        """Gradient is correct when S is rank-deficient."""
        rng = np.random.default_rng(42)
        q = 10
        S1 = _make_psd(q, 6, rng)
        S2 = _make_psd(q, 4, rng)
        lambdas = np.array([1.0, 2.0])
        rhos = np.log(lambdas)

        result = similarity_transform_logdet([S1, S2], lambdas)
        grad = logdet_s_gradient(result, [S1, S2], lambdas)

        eps = 1e-5
        fd_grad = np.zeros(2)
        for j in range(2):
            rhos_p = rhos.copy()
            rhos_p[j] += eps
            rhos_m = rhos.copy()
            rhos_m[j] -= eps
            res_p = similarity_transform_logdet([S1, S2], np.exp(rhos_p))
            res_m = similarity_transform_logdet([S1, S2], np.exp(rhos_m))
            fd_grad[j] = (res_p.logdet_s_plus - res_m.logdet_s_plus) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-4, atol=1e-6)
