"""Tests for Wood (2011) Appendix B multi-penalty log-determinant kernel."""

import numpy as np
import pandas as pd
import pytest

from superglm.multi_penalty import (
    logdet_s_gradient,
    logdet_s_hessian,
    similarity_transform_logdet,
)
from superglm.types import PenaltyComponent


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

    def test_q_plus_q_zero_orthogonal_decomposition(self):
        """Q_plus/Q_zero form a complete orthogonal decomposition of q-space."""
        rng = np.random.default_rng(42)
        q = 8
        rank = 5
        S1 = _make_psd(q, rank, rng)
        lambdas = np.array([2.0])

        result = similarity_transform_logdet([S1], lambdas)
        S_total = lambdas[0] * S1

        # Q_plus: (q, rank) penalized subspace
        assert result.Q_plus.shape == (q, result.rank)
        np.testing.assert_allclose(result.Q_plus.T @ result.Q_plus, np.eye(result.rank), atol=1e-10)

        # Q_zero: (q, q-rank) null space
        assert result.Q_zero.shape == (q, q - result.rank)
        np.testing.assert_allclose(
            result.Q_zero.T @ result.Q_zero, np.eye(q - result.rank), atol=1e-10
        )

        # Cross-orthogonality: Q_plus' Q_zero ≈ 0
        np.testing.assert_allclose(result.Q_plus.T @ result.Q_zero, 0.0, atol=1e-10)

        # Q_full is orthogonal: Q_full' Q_full ≈ I
        Q_full = result.Q_full
        assert Q_full.shape == (q, q)
        np.testing.assert_allclose(Q_full.T @ Q_full, np.eye(q), atol=1e-10)

        # S @ Q_zero ≈ 0 (null space of S)
        np.testing.assert_allclose(S_total @ result.Q_zero, 0.0, atol=1e-8)

        # Completeness: Q_plus Q_plus' + Q_zero Q_zero' ≈ I
        proj_sum = result.Q_plus @ result.Q_plus.T + result.Q_zero @ result.Q_zero.T
        np.testing.assert_allclose(proj_sum, np.eye(q), atol=1e-10)

    def test_weak_positive_eigenvalue_not_discarded(self):
        """Near-singular but full-rank S must not lose weak eigenvalues.

        Regression: diag(1.0, 1e-6) is full-rank. The initial Frobenius
        transform must not discard the 1e-6 eigenvalue, or log|S|+ will
        be wrong and S @ Q_zero will be nonzero.
        """
        q = 2
        S1 = np.diag([1.0, 1e-6])
        lambdas = np.array([1.0])

        result = similarity_transform_logdet([S1], lambdas)

        assert result.rank == q, f"Expected full rank {q}, got {result.rank}"
        expected_logdet = np.log(1.0) + np.log(1e-6)
        np.testing.assert_allclose(result.logdet_s_plus, expected_logdet, rtol=1e-6)

        # Q_zero should be empty (full rank → no null space)
        assert result.Q_zero.shape == (q, 0), (
            f"Full-rank S should have empty Q_zero, got shape {result.Q_zero.shape}"
        )

    def test_weak_positive_multi_penalty(self):
        """Two penalties where the combined S has a weak eigenvalue ~1e-8."""
        rng = np.random.default_rng(42)
        q = 5
        S1 = np.diag([1.0, 0.5, 0.1, 0.01, 1e-8])
        S2 = _make_psd(q, q, rng) * 1e-10  # very small second penalty
        lambdas = np.array([1.0, 1.0])

        result = similarity_transform_logdet([S1, S2], lambdas)
        S_total = S1 + S2

        # S_total is full rank (S1 diagonal + tiny S2 perturbation)
        eigvals = np.linalg.eigvalsh(S_total)
        expected_rank = int(np.sum(eigvals > np.finfo(float).eps * q * eigvals.max()))
        expected_logdet = float(np.sum(np.log(eigvals[eigvals > 1e-15 * eigvals.max()])))

        assert result.rank == expected_rank
        np.testing.assert_allclose(result.logdet_s_plus, expected_logdet, rtol=1e-4)

    def test_full_rank_q_zero_is_empty(self):
        """Full-rank S: Q_zero has zero columns."""
        rng = np.random.default_rng(42)
        q = 6
        S1 = _make_psd(q, q, rng) + 0.1 * np.eye(q)
        lambdas = np.array([1.0])

        result = similarity_transform_logdet([S1], lambdas)

        assert result.rank == q
        assert result.Q_plus.shape == (q, q)
        assert result.Q_zero.shape == (q, 0)
        assert result.Q_full.shape == (q, q)

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

    def test_tensor_scale_q50_q100(self):
        """Interaction-scale penalty blocks: correct at q=50 and q=100.

        Tensor product smooths (ti/te) can push penalty block size to
        q = k1 * k2, easily reaching 50-100. Three-way interactions go
        higher but are rare and expected to be slow.
        """
        rng = np.random.default_rng(42)
        for q in [50, 100]:
            # Two penalties with different rank structures (typical tensor setup)
            S1 = _make_psd(q, q, rng)
            S2 = _make_psd(q, q - 5, rng)  # rank-deficient margin
            lambdas = np.array([1.0, 2.0])

            result = similarity_transform_logdet([S1, S2], lambdas)
            S_total = lambdas[0] * S1 + lambdas[1] * S2
            _, expected = np.linalg.slogdet(S_total)

            assert result.rank == q
            np.testing.assert_allclose(result.logdet_s_plus, expected, rtol=1e-6)
            # Q_plus should be full rank, Q_zero empty
            assert result.Q_plus.shape == (q, q)
            assert result.Q_zero.shape == (q, 0)


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


# ═══════════════════════════════════════════════════════════════════
# Cut 2A: _build_penalty_matrix multi-penalty + S_override tests
# ═══════════════════════════════════════════════════════════════════


class TestBuildPenaltyMatrixMultiComponent:
    """Verify _build_penalty_matrix accumulates multiple PenaltyComponents."""

    def test_two_components_same_group(self):
        """Two PenaltyComponents for same group: S = lam1*Omega1 + lam2*Omega2."""
        from superglm.solvers.irls_direct import _build_penalty_matrix

        rng = np.random.default_rng(42)
        p_g = 6
        p = p_g  # single group fills the whole coefficient vector

        # Two different omega_ssp matrices (PSD)
        A1 = rng.standard_normal((p_g, p_g))
        omega1 = A1.T @ A1
        A2 = rng.standard_normal((p_g, p_g))
        omega2 = A2.T @ A2

        lam1, lam2 = 2.5, 0.3
        lambda2 = {"comp1": lam1, "comp2": lam2}

        sl = slice(0, p_g)
        pc1 = PenaltyComponent(
            name="comp1",
            group_name="grp",
            group_index=0,
            group_sl=sl,
            omega_raw=omega1,  # not used when omega_ssp is set
            omega_ssp=omega1,
        )
        pc2 = PenaltyComponent(
            name="comp2",
            group_name="grp",
            group_index=0,
            group_sl=sl,
            omega_raw=omega2,
            omega_ssp=omega2,
        )

        # group_matrices not used when omega_ssp is pre-computed,
        # but must be indexable
        S = _build_penalty_matrix(
            group_matrices=[None],
            groups=[],
            lambda2=lambda2,
            p=p,
            reml_penalties=[pc1, pc2],
        )

        expected = lam1 * omega1 + lam2 * omega2
        np.testing.assert_allclose(S[:p_g, :p_g], expected, rtol=1e-12)

    def test_zero_lambda_skipped(self):
        """Component with lambda=0 contributes nothing."""
        from superglm.solvers.irls_direct import _build_penalty_matrix

        p_g = 4
        omega = np.eye(p_g)
        sl = slice(0, p_g)
        pc = PenaltyComponent(
            name="comp",
            group_name="grp",
            group_index=0,
            group_sl=sl,
            omega_raw=omega,
            omega_ssp=omega,
        )

        S = _build_penalty_matrix(
            group_matrices=[None],
            groups=[],
            lambda2={"comp": 0.0},
            p=p_g,
            reml_penalties=[pc],
        )
        np.testing.assert_array_equal(S, np.zeros((p_g, p_g)))

    def test_legacy_path_unchanged(self):
        """Without reml_penalties, falls through to legacy single-penalty path."""
        from superglm.solvers.irls_direct import _build_penalty_matrix

        # Legacy path requires SSP group matrices; just verify an empty
        # group list produces all-zero S (no crash, no change)
        S = _build_penalty_matrix(
            group_matrices=[],
            groups=[],
            lambda2=1.0,
            p=5,
            reml_penalties=None,
        )
        np.testing.assert_array_equal(S, np.zeros((5, 5)))

    def test_omega_ssp_fallback_from_raw(self):
        """When omega_ssp is None, falls back to R_inv.T @ omega_raw @ R_inv."""
        from types import SimpleNamespace

        from superglm.solvers.irls_direct import _build_penalty_matrix

        rng = np.random.default_rng(99)
        p_g = 4
        R_inv = rng.standard_normal((p_g, p_g))
        omega_raw = np.eye(p_g)
        expected_ssp = R_inv.T @ omega_raw @ R_inv

        gm = SimpleNamespace(R_inv=R_inv)
        sl = slice(0, p_g)
        pc = PenaltyComponent(
            name="comp",
            group_name="grp",
            group_index=0,
            group_sl=sl,
            omega_raw=omega_raw,
            omega_ssp=None,  # force fallback
        )

        S = _build_penalty_matrix(
            group_matrices=[gm],
            groups=[],
            lambda2={"comp": 3.0},
            p=p_g,
            reml_penalties=[pc],
        )
        np.testing.assert_allclose(S[:p_g, :p_g], 3.0 * expected_ssp, rtol=1e-12)


@pytest.mark.slow
class TestFitIrlsDirectSOverrideParity:
    """S_override produces identical results to internal S build."""

    def test_parity(self):
        from superglm.features.spline import CubicRegressionSpline
        from superglm.model import SuperGLM
        from superglm.solvers.irls_direct import _build_penalty_matrix, fit_irls_direct

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + np.sin(2 * np.pi * x))
        y = rng.poisson(mu).astype(float)

        df = pd.DataFrame({"x": x})
        m = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=8)},
            family="poisson",
        )
        m.fit(df, y)

        weights = np.ones(n)
        offset = np.zeros(n)
        lambdas = {"x": 5.0}

        # Build S externally
        p = m._dm.p
        S = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p)

        # Fit without S_override (internal build)
        res_internal, inv_internal = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=weights,
            family=m._distribution,
            link=m._link,
            groups=m._groups,
            lambda2=lambdas,
            offset=offset,
        )

        # Fit with S_override (external build)
        res_override, inv_override = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=weights,
            family=m._distribution,
            link=m._link,
            groups=m._groups,
            lambda2=lambdas,
            offset=offset,
            S_override=S,
        )

        np.testing.assert_allclose(res_override.beta, res_internal.beta, rtol=1e-12)
        np.testing.assert_allclose(res_override.deviance, res_internal.deviance, rtol=1e-12)
        np.testing.assert_allclose(inv_override, inv_internal, rtol=1e-10)


@pytest.mark.slow
class TestInvertXtWXPlusPenaltySOverrideParity:
    """S_override in _invert_xtwx_plus_penalty matches internal path."""

    def test_parity(self):
        from superglm.features.spline import CubicRegressionSpline
        from superglm.model import SuperGLM
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            _invert_xtwx_plus_penalty,
            fit_irls_direct,
        )

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + np.sin(2 * np.pi * x))
        y = rng.poisson(mu).astype(float)

        df = pd.DataFrame({"x": x})
        m = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=8)},
            family="poisson",
        )
        m.fit(df, y)

        weights = np.ones(n)
        offset = np.zeros(n)
        lambdas = {"x": 5.0}

        # Get XtWX from a fit
        _, _, XtWX = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=weights,
            family=m._distribution,
            link=m._link,
            groups=m._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
        )

        # Build S externally
        p = XtWX.shape[0]
        S = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p)

        # Invert without S_override
        inv_internal = _invert_xtwx_plus_penalty(XtWX, m._dm.group_matrices, m._groups, lambdas)

        # Invert with S_override
        inv_override = _invert_xtwx_plus_penalty(
            XtWX, m._dm.group_matrices, m._groups, lambdas, S_override=S
        )

        np.testing.assert_allclose(inv_override, inv_internal, rtol=1e-12)
