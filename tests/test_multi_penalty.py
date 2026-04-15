"""Tests for Wood (2011) Appendix B multi-penalty log-determinant kernel."""

import numpy as np
import pandas as pd
import pytest

from superglm.reml.multi_penalty import (
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


class TestEndToEndMultiPenaltyDirect:
    """End-to-end: optimize_direct_reml with pc.name != group_name.

    Constructs two synthetic PenaltyComponents for one group (split the
    single omega into two halves), then runs the direct REML Newton optimizer
    through convergence.
    """

    @pytest.mark.slow
    def test_direct_reml_converges_with_multi_penalty(self):
        """optimize_direct_reml converges with two components per group."""
        from superglm import SuperGLM
        from superglm.features.spline import CubicRegressionSpline
        from superglm.group_matrix import SparseSSPGroupMatrix
        from superglm.reml import optimize_direct_reml
        from superglm.solvers.irls_direct import _build_penalty_matrix
        from superglm.types import PenaltyComponent

        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + np.sin(2 * np.pi * x1))
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"x1": x1})

        m = SuperGLM(
            features={"x1": CubicRegressionSpline(n_knots=8)},
            family="poisson",
        )
        m.fit(df, y)

        # Identify the penalised group
        reml_groups = []
        for i, (gm, g) in enumerate(zip(m._dm.group_matrices, m._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix) and gm.omega is not None:
                reml_groups.append((i, g))
        assert len(reml_groups) == 1
        idx, g = reml_groups[0]
        gm = m._dm.group_matrices[idx]
        omega_ssp_full = gm.R_inv.T @ gm.omega @ gm.R_inv

        # Split omega into two synthetic halves via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(omega_ssp_full)
        mid = len(eigvals) // 2
        omega1 = (eigvecs[:, :mid] * eigvals[:mid]) @ eigvecs[:, :mid].T
        omega2 = (eigvecs[:, mid:] * eigvals[mid:]) @ eigvecs[:, mid:].T
        # omega1 + omega2 ≈ omega_ssp_full
        np.testing.assert_allclose(omega1 + omega2, omega_ssp_full, atol=1e-12)

        # Build two PenaltyComponents with different names
        eps_thresh = np.finfo(float).eps ** (2 / 3)

        def _make_pc(name, omega_ssp):
            ev = np.linalg.eigvalsh(omega_ssp)
            thresh = eps_thresh * max(ev.max(), 1e-12)
            pos = ev[ev > thresh]
            return PenaltyComponent(
                name=name,
                group_name=g.name,
                group_index=idx,
                group_sl=g.sl,
                omega_raw=gm.omega,  # raw not used by direct path
                omega_ssp=omega_ssp,
                rank=float(len(pos)),
                log_det_omega_plus=float(np.sum(np.log(pos))) if len(pos) > 0 else 0.0,
                eigvals_omega=pos,
            )

        pc1 = _make_pc("x1:pen1", omega1)
        pc2 = _make_pc("x1:pen2", omega2)
        penalties = [pc1, pc2]

        # Build penalty caches (one per component)
        from superglm.reml import PenaltyCache

        penalty_caches = {
            pc.name: PenaltyCache(
                omega_ssp=pc.omega_ssp,
                log_det_omega_plus=pc.log_det_omega_plus,
                rank=pc.rank,
                eigvals_omega=pc.eigvals_omega,
            )
            for pc in penalties
        }
        penalty_ranks = {pc.name: pc.rank for pc in penalties}
        lambdas = {"x1:pen1": 1.0, "x1:pen2": 1.0}

        # Run optimize_direct_reml
        result = optimize_direct_reml(
            dm=m._dm,
            distribution=m._distribution,
            link=m._link,
            groups=m._groups,
            discrete=False,
            y=y,
            sample_weight=np.ones(n),
            offset_arr=np.zeros(n),
            reml_groups=reml_groups,
            penalty_ranks=penalty_ranks,
            lambdas=lambdas,
            max_reml_iter=15,
            reml_tol=1e-4,
            verbose=False,
            penalty_caches=penalty_caches,
            reml_penalties=penalties,
        )

        # Must converge
        assert result.converged, "Multi-penalty direct REML did not converge"

        # Final lambdas must have both component keys
        assert "x1:pen1" in result.lambdas
        assert "x1:pen2" in result.lambdas

        # Final S from multi-penalty should equal sum of component contributions
        p = m._dm.p
        S_multi = _build_penalty_matrix(
            m._dm.group_matrices,
            m._groups,
            result.lambdas,
            p,
            reml_penalties=penalties,
        )
        S_manual = result.lambdas["x1:pen1"] * omega1 + result.lambdas["x1:pen2"] * omega2
        np.testing.assert_allclose(S_multi[g.sl, g.sl], S_manual, rtol=1e-10)

    @pytest.mark.slow
    def test_component_independence(self):
        """Changing one component lambda changes S independently of the other."""
        from superglm.types import PenaltyComponent

        q = 6
        rng = np.random.default_rng(42)
        omega1 = _make_psd(q, 3, rng)
        omega2 = _make_psd(q, 3, rng)

        pc1 = PenaltyComponent(
            name="t:a",
            group_name="t",
            group_index=0,
            group_sl=slice(0, q),
            omega_raw=omega1,
            omega_ssp=omega1,
            rank=3.0,
        )
        pc2 = PenaltyComponent(
            name="t:b",
            group_name="t",
            group_index=0,
            group_sl=slice(0, q),
            omega_raw=omega2,
            omega_ssp=omega2,
            rank=3.0,
        )

        # Baseline
        lam_base = {"t:a": 1.0, "t:b": 1.0}
        S_base = np.zeros((q, q))
        for pc in [pc1, pc2]:
            S_base[pc.group_sl, pc.group_sl] += lam_base[pc.name] * pc.omega_ssp

        # Change only lambda for component "t:a"
        lam_a_changed = {"t:a": 5.0, "t:b": 1.0}
        S_a = np.zeros((q, q))
        for pc in [pc1, pc2]:
            S_a[pc.group_sl, pc.group_sl] += lam_a_changed[pc.name] * pc.omega_ssp

        # The difference should only involve omega1
        diff = S_a - S_base
        expected_diff = 4.0 * omega1  # (5-1) * omega1
        np.testing.assert_allclose(diff, expected_diff, atol=1e-12)


# ── GroupInfo penalty_components validation ────────────────────


class TestGroupInfoPenaltyComponents:
    """Validate GroupInfo.penalty_components field."""

    def test_valid_penalty_components(self):
        """penalty_components with matching shapes and sum accepted."""
        from superglm.types import GroupInfo

        q = 4
        rng = np.random.default_rng(42)
        omega1 = _make_psd(q, 2, rng)
        omega2 = _make_psd(q, 2, rng)
        omega_sum = omega1 + omega2

        info = GroupInfo(
            columns=None,
            n_cols=q,
            penalty_matrix=omega_sum,
            penalty_components=[("a", omega1), ("b", omega2)],
        )
        assert info.penalty_components is not None
        assert len(info.penalty_components) == 2

    def test_wrong_shape_raises(self):
        """penalty_components with wrong shape raise ValueError."""
        from superglm.types import GroupInfo

        q = 4
        rng = np.random.default_rng(42)
        omega_good = _make_psd(q, 2, rng)
        omega_bad = _make_psd(q + 1, 2, rng)

        with pytest.raises(ValueError, match="penalty_component.*shape"):
            GroupInfo(
                columns=None,
                n_cols=q,
                penalty_matrix=omega_good,
                penalty_components=[("a", omega_good), ("b", omega_bad)],
            )

    def test_sum_mismatch_raises(self):
        """penalty_components that don't sum to penalty_matrix raise ValueError."""
        from superglm.types import GroupInfo

        q = 4
        rng = np.random.default_rng(42)
        omega1 = _make_psd(q, 2, rng)
        omega2 = _make_psd(q, 2, rng)
        omega_wrong = omega1 + 2 * omega2  # wrong sum

        with pytest.raises(ValueError, match="does not match"):
            GroupInfo(
                columns=None,
                n_cols=q,
                penalty_matrix=omega_wrong,
                penalty_components=[("a", omega1), ("b", omega2)],
            )


# ── Tensor penalty component emission ─────────────────────────


class TestTensorPenaltyComponentEmission:
    """Verify TensorInteraction emits separate marginal penalty components."""

    def test_non_decompose_emits_components(self):
        """Non-decompose tensor build populates penalty_components."""
        from superglm.features.interaction import TensorInteraction
        from superglm.features.spline import CubicRegressionSpline

        rng = np.random.default_rng(42)
        n = 200
        x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)

        spec1 = CubicRegressionSpline(n_knots=4)
        spec2 = CubicRegressionSpline(n_knots=4)
        spec1.build_knots_and_penalty(x1, np.ones(n))
        spec2.build_knots_and_penalty(x2, np.ones(n))

        ti = TensorInteraction("x1", "x2", n_knots=(4, 4))
        info = ti.build(x1, x2, {"x1": spec1, "x2": spec2})

        # Non-decompose returns a single GroupInfo with penalty_components
        assert not isinstance(info, list)
        assert info.penalty_components is not None
        assert len(info.penalty_components) == 2

        # Semantic names
        suffixes = [s for s, _ in info.penalty_components]
        assert "margin_x1" in suffixes
        assert "margin_x2" in suffixes

        # Components sum to penalty_matrix
        comp_sum = sum(omega for _, omega in info.penalty_components)
        np.testing.assert_allclose(comp_sum, info.penalty_matrix, atol=1e-12)

    def test_decompose_does_not_emit_components(self):
        """decompose=True still splits bilinear/wiggly subgroups, with marginal penalties on wiggly."""
        from superglm.features.interaction import TensorInteraction
        from superglm.features.spline import CubicRegressionSpline

        rng = np.random.default_rng(42)
        n = 200
        x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)

        spec1 = CubicRegressionSpline(n_knots=4)
        spec2 = CubicRegressionSpline(n_knots=4)
        spec1.build_knots_and_penalty(x1, np.ones(n))
        spec2.build_knots_and_penalty(x2, np.ones(n))

        ti = TensorInteraction("x1", "x2", n_knots=(4, 4), decompose=True)
        infos = ti.build(x1, x2, {"x1": spec1, "x2": spec2})

        # decompose returns a list of GroupInfos (Mechanism A)
        assert isinstance(infos, list)
        assert len(infos) == 2
        bilinear, wiggly = infos
        assert bilinear.subgroup_name == "bilinear"
        assert bilinear.penalty_components is None
        assert wiggly.subgroup_name == "wiggly"
        assert wiggly.penalty_components is not None
        assert len(wiggly.penalty_components) == 2

    def test_multi_m_parent_raises(self):
        """Tensor interactions reject multi-penalty parent smooths."""
        from superglm.features.interaction import TensorInteraction
        from superglm.features.spline import CubicRegressionSpline

        rng = np.random.default_rng(42)
        n = 200
        x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)

        spec1 = CubicRegressionSpline(n_knots=4, m=(1, 2))
        spec2 = CubicRegressionSpline(n_knots=4)
        spec1.build_knots_and_penalty(x1, np.ones(n))
        spec2.build_knots_and_penalty(x2, np.ones(n))

        ti = TensorInteraction("x1", "x2", n_knots=(4, 4))
        with pytest.raises(NotImplementedError, match="single-penalty parent smooths"):
            ti.build(x1, x2, {"x1": spec1, "x2": spec2})

    def test_tensor_multi_m_parent_fit_raises(self):
        """fit_reml rejects multi-penalty tensor parents."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=6, m=(1, 2)),
                "x2": Spline(kind="cr", n_knots=6),
            },
            interactions=[("x1", "x2")],
        )
        with pytest.raises(NotImplementedError, match="single-penalty parent smooths"):
            model.fit_reml(X, y, max_reml_iter=30)


class TestBlockwiseHessianTrace:
    def test_tensor_multi_m_parent_discrete_raises(self):
        """Discrete path also rejects multi-penalty tensor parents."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            discrete=True,
            features={
                "x1": Spline(kind="cr", n_knots=6, m=(1, 2)),
                "x2": Spline(kind="cr", n_knots=6),
            },
            interactions=[("x1", "x2")],
        )
        with pytest.raises(NotImplementedError, match="single-penalty parent smooths"):
            model.fit_reml(X, y, max_reml_iter=30)

    def test_tensor_n_knots_override_multi_m_raises(self):
        """n_knots override still rejects multi-penalty tensor parents."""
        from superglm.features.interaction import TensorInteraction
        from superglm.features.spline import CubicRegressionSpline

        rng = np.random.default_rng(42)
        n = 200
        x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)

        spec1 = CubicRegressionSpline(n_knots=8, m=(1, 2))
        spec2 = CubicRegressionSpline(n_knots=8)
        spec1.build_knots_and_penalty(x1, np.ones(n))
        spec2.build_knots_and_penalty(x2, np.ones(n))

        ti = TensorInteraction("x1", "x2", n_knots=(4, 4))
        with pytest.raises(NotImplementedError, match="single-penalty parent smooths"):
            ti.build(x1, x2, {"x1": spec1, "x2": spec2})


# ── log|S|+ correctness for shared-block penalties ────────────


class TestLogdetSharedBlock:
    """Verify compute_logdet_s_plus gives correct joint log-det."""

    def test_joint_logdet_matches_eigendecomposition(self):
        """compute_logdet_s_plus matches naive eigendecomposition of sum(lam*omega)."""
        from superglm.reml import compute_logdet_s_plus

        q = 8
        rng = np.random.default_rng(42)
        omega1 = _make_psd(q, 5, rng) + 0.01 * np.eye(q)
        omega2 = _make_psd(q, 5, rng) + 0.01 * np.eye(q)

        lam1, lam2 = 3.0, 0.7

        pc1 = PenaltyComponent(
            name="g:a",
            group_name="g",
            group_index=0,
            group_sl=slice(0, q),
            omega_raw=omega1,
            omega_ssp=omega1,
            rank=float(q),
        )
        pc2 = PenaltyComponent(
            name="g:b",
            group_name="g",
            group_index=0,
            group_sl=slice(0, q),
            omega_raw=omega2,
            omega_ssp=omega2,
            rank=float(q),
        )
        penalties = [pc1, pc2]
        lambdas = {"g:a": lam1, "g:b": lam2}

        # Grouped path
        result = compute_logdet_s_plus(lambdas, penalties)

        # Naive eigendecomposition of the combined matrix
        S_combined = lam1 * omega1 + lam2 * omega2
        eigvals = np.linalg.eigvalsh(S_combined)
        expected = float(np.sum(np.log(eigvals[eigvals > 1e-10])))

        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_single_penalty_matches_additive_formula(self):
        """For single-penalty groups, compute_logdet_s_plus matches the fast shortcut."""
        from superglm.reml import PenaltyCache, cached_logdet_s_plus, compute_logdet_s_plus

        q = 6
        rng = np.random.default_rng(42)
        omega = _make_psd(q, 4, rng) + 0.01 * np.eye(q)
        eigvals = np.linalg.eigvalsh(omega)
        pos = eigvals[eigvals > 1e-10]

        pc = PenaltyComponent(
            name="g",
            group_name="g",
            group_index=0,
            group_sl=slice(0, q),
            omega_raw=omega,
            omega_ssp=omega,
            rank=float(len(pos)),
            log_det_omega_plus=float(np.sum(np.log(pos))),
            eigvals_omega=pos,
        )
        lambdas = {"g": 2.5}

        # New path
        result_new = compute_logdet_s_plus(lambdas, [pc])

        # Old additive path
        cache = PenaltyCache(
            omega_ssp=omega,
            log_det_omega_plus=pc.log_det_omega_plus,
            rank=pc.rank,
            eigvals_omega=pos,
        )
        result_old = cached_logdet_s_plus(lambdas, {"g": cache})

        np.testing.assert_allclose(result_new, result_old, rtol=1e-10)

    def test_tensor_pair_closed_form_matches_generic_shared_block(self):
        """Discrete tensor pairs use the closed-form logdet summary without changing values."""
        from superglm.group_matrix import DiscretizedTensorGroupMatrix
        from superglm.reml import compute_logdet_s_derivatives, compute_logdet_s_plus
        from superglm.reml.penalty_algebra import (
            build_tensor_pair_logdet_summaries,
            evaluate_tensor_pair_logdet_summaries,
        )

        p1, p2 = 4, 3
        rng = np.random.default_rng(42)
        S1 = _make_psd(p1, p1 - 1, rng)
        S2 = _make_psd(p2, p2 - 1, rng)
        omega_1 = np.kron(S1, np.eye(p2))
        omega_2 = np.kron(np.eye(p1), S2)
        q = p1 * p2

        gm = DiscretizedTensorGroupMatrix(
            B1_unique=np.eye(p1),
            B2_unique=np.eye(p2),
            idx1=np.zeros(q, dtype=np.intp),
            idx2=np.zeros(q, dtype=np.intp),
            B_joint=np.eye(q),
            R_inv=np.eye(q),
            pair_idx=np.arange(q, dtype=np.intp),
            tensor_id=7,
        )
        penalties = [
            PenaltyComponent(
                name="ti:margin_x1",
                group_name="ti",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=omega_1,
                omega_ssp=omega_1,
                rank=float(np.linalg.matrix_rank(omega_1)),
            ),
            PenaltyComponent(
                name="ti:margin_x2",
                group_name="ti",
                group_index=0,
                group_sl=slice(0, q),
                omega_raw=omega_2,
                omega_ssp=omega_2,
                rank=float(np.linalg.matrix_rank(omega_2)),
            ),
        ]
        lambdas = {"ti:margin_x1": 2.3, "ti:margin_x2": 0.6}

        generic_logdet = compute_logdet_s_plus(lambdas, penalties)
        generic_grad, generic_hess = compute_logdet_s_derivatives(lambdas, penalties)

        tensor_summaries = build_tensor_pair_logdet_summaries([gm], penalties)
        tensor_evals = evaluate_tensor_pair_logdet_summaries(tensor_summaries, lambdas)
        closed_logdet = compute_logdet_s_plus(
            lambdas, penalties, tensor_pair_evaluations=tensor_evals
        )
        closed_grad, closed_hess = compute_logdet_s_derivatives(
            lambdas,
            penalties,
            tensor_pair_evaluations=tensor_evals,
        )

        assert set(tensor_summaries) == {"ti"}
        assert set(tensor_evals) == {"ti"}
        np.testing.assert_allclose(closed_logdet, generic_logdet, rtol=1e-10, atol=1e-10)
        for name, value in generic_grad.items():
            np.testing.assert_allclose(closed_grad[name], value, rtol=1e-10, atol=1e-10)
        for pair, value in generic_hess.items():
            np.testing.assert_allclose(closed_hess[pair], value, rtol=1e-10, atol=1e-10)


# ── Multi-order derivative penalty tests ──────────────────────


class TestMultiOrderSplinePenalty:
    """Verify Spline(m=...) produces correct penalties and REML integration."""

    def test_single_m1_difference_penalty(self):
        """Spline(kind='bs', m=1) produces 1st-difference penalty."""
        from superglm.features.spline import BasisSpline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = BasisSpline(n_knots=6, m=1)
        spec._place_knots(x)
        omega = spec._build_penalty()
        D1 = np.diff(np.eye(spec._n_basis), n=1, axis=0)
        expected = D1.T @ D1
        np.testing.assert_allclose(omega, expected, atol=1e-12)

    def test_single_m3_difference_penalty(self):
        """Spline(kind='bs', m=3) produces 3rd-difference penalty."""
        from superglm.features.spline import BasisSpline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = BasisSpline(n_knots=6, m=3)
        spec._place_knots(x)
        omega = spec._build_penalty()
        D3 = np.diff(np.eye(spec._n_basis), n=3, axis=0)
        expected = D3.T @ D3
        np.testing.assert_allclose(omega, expected, atol=1e-12)

    def test_crs_m1_integrated_first_derivative(self):
        """Spline(kind='cr', m=1) produces integrated f' penalty."""
        from superglm.features.spline import CubicRegressionSpline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = CubicRegressionSpline(n_knots=6, m=1)
        spec._place_knots(x)
        omega = spec._build_penalty_for_order(1)
        # Should be PSD
        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals >= -1e-10)
        # Null space: constants only (f'(const) = 0)
        n_null = int(np.sum(eigvals < 1e-8))
        assert n_null == 1  # constant function has zero first derivative

    def test_default_m2_unchanged(self):
        """Default Spline(m=2) produces identical penalty to before."""
        from superglm import Spline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)

        spec_default = Spline(kind="bs", n_knots=6)
        spec_explicit = Spline(kind="bs", n_knots=6, m=2)

        info_d = spec_default.build(x)
        info_e = spec_explicit.build(x)

        np.testing.assert_allclose(info_d.penalty_matrix, info_e.penalty_matrix, atol=1e-12)

    def test_multi_m_emits_penalty_components(self):
        """Spline(m=(1,2)) emits penalty_components with two entries."""
        from superglm import Spline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)

        spec = Spline(kind="cr", n_knots=6, m=(1, 2))
        info = spec.build(x)

        assert info.penalty_components is not None
        assert len(info.penalty_components) == 2
        suffixes = [s for s, _ in info.penalty_components]
        assert "d1" in suffixes
        assert "d2" in suffixes

        # Components sum to penalty_matrix
        comp_sum = sum(om for _, om in info.penalty_components)
        np.testing.assert_allclose(comp_sum, info.penalty_matrix, atol=1e-12)

    def test_multi_m_bs_emits_components(self):
        """Spline(kind='bs', m=(2,3)) emits penalty_components."""
        from superglm import Spline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = Spline(kind="bs", n_knots=8, m=(2, 3))
        info = spec.build(x)

        assert info.penalty_components is not None
        assert len(info.penalty_components) == 2
        suffixes = [s for s, _ in info.penalty_components]
        assert "d2" in suffixes
        assert "d3" in suffixes

    def test_select_plus_multi_m_builds(self):
        """select=True + multi-m produces null + per-order components."""
        from superglm import Spline

        spec = Spline(kind="cr", n_knots=8, m=(1, 2), select=True)
        result = spec.build(np.linspace(0, 1, 200))

        assert result.penalty_components is not None
        suffixes = [s for s, _ in result.penalty_components]
        assert "null" in suffixes
        assert "d1" in suffixes
        assert "d2" in suffixes
        assert len(suffixes) == 3

        assert result.component_types == {"null": "selection"}

        # Components must sum to penalty_matrix
        comp_sum = sum(omega for _, omega in result.penalty_components)
        np.testing.assert_allclose(comp_sum, result.penalty_matrix, atol=1e-12)

    def test_select_plus_multi_m_high_order_raises(self):
        """select=True + m=(1,2,3) on BS raises (current capability policy)."""
        from superglm import Spline

        with pytest.raises(NotImplementedError, match="not supported for PSpline"):
            Spline(kind="bs", n_knots=8, m=(1, 2, 3), select=True)

    @pytest.mark.slow
    def test_select_plus_multi_m_fit_converges(self):
        """Spline(select=True, m=(1,2)) converges with separate lambdas."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Spline(kind="cr", n_knots=8, m=(1, 2), select=True)},
        )
        model.fit_reml(X, y, max_reml_iter=30)

        assert model._reml_result.converged
        lam = model._reml_lambdas
        # Should have null + d1 + d2 lambda keys
        assert "x:null" in lam
        assert "x:d1" in lam
        assert "x:d2" in lam

    @pytest.mark.slow
    def test_select_plus_multi_m_discrete_converges(self):
        """select=True + multi-m works on the discrete path."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            discrete=True,
            features={"x": Spline(kind="cr", n_knots=8, m=(1, 2), select=True)},
        )
        model.fit_reml(X, y, max_reml_iter=30)

        assert model._reml_result.converged
        lam = model._reml_lambdas
        assert "x:null" in lam
        assert "x:d1" in lam
        assert "x:d2" in lam

    def test_tensor_parent_multi_m_raises(self):
        """Tensor parents with multi-order penalties are rejected."""
        from superglm.features.spline import CubicRegressionSpline

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 200)
        spec = CubicRegressionSpline(n_knots=6, m=(1, 2))
        spec.build(x)
        with pytest.raises(NotImplementedError, match="single-penalty parent smooths"):
            spec.tensor_marginal_ingredients(x)

    @pytest.mark.slow
    def test_fit_reml_multi_m_converges(self):
        """fit_reml with Spline(m=(1,2)) converges with per-order lambdas."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Spline(kind="cr", n_knots=8, m=(1, 2))},
        )
        model.fit_reml(X, y, max_reml_iter=20)

        assert model._reml_result.converged
        lam = model._reml_lambdas
        assert "x:d1" in lam
        assert "x:d2" in lam

        pred = model.predict(X)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)

    @pytest.mark.slow
    def test_fit_reml_multi_m_three_orders(self):
        """fit_reml with m=(1,2,3) converges with three lambdas."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Spline(kind="cr", n_knots=8, m=(1, 2, 3))},
        )
        model.fit_reml(X, y, max_reml_iter=20)

        assert model._reml_result.converged
        lam = model._reml_lambdas
        assert len([k for k in lam if k.startswith("x:")]) == 3


class TestSelectionPenaltyRejectedInREML:
    """fit_reml() rejects sparse selection penalties in favor of direct REML."""

    @pytest.mark.slow
    def test_tensor_selection_penalty_rejected(self):
        """Tensor REML requires selection_penalty=0."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * np.cos(2 * np.pi * x2)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=1e-8,
            features={
                "x1": Spline(kind="cr", n_knots=6),
                "x2": Spline(kind="cr", n_knots=6),
            },
            interactions=[("x1", "x2")],
        )
        with pytest.raises(ValueError, match="selection_penalty=0"):
            model.fit_reml(X, y, max_reml_iter=30)

    @pytest.mark.slow
    def test_main_effect_selection_penalty_rejected(self):
        """The REML rejection is generic, not tensor-specific."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=1e-10,
            features={
                "x1": Spline(kind="cr", n_knots=6),
                "x2": Spline(kind="cr", n_knots=6),
            },
        )
        with pytest.raises(ValueError, match="selection_penalty=0"):
            model.fit_reml(X, y, max_reml_iter=30)

    @pytest.mark.slow
    def test_tensor_summary_reports_fitted_lambda(self):
        """summary() reports fitted REML lambda for tensor interaction terms."""
        from superglm import Spline, SuperGLM

        rng = np.random.default_rng(42)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + np.sin(2 * np.pi * x1) + 0.3 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=6),
                "x2": Spline(kind="cr", n_knots=6),
            },
            interactions=[("x1", "x2")],
        )
        model.fit_reml(X, y, max_reml_iter=30)

        # Tensor term should have per-marginal lambdas
        lam = model._reml_lambdas
        tensor_keys = [k for k in lam if "x1:x2" in k]
        assert len(tensor_keys) >= 2

        # summary() should report a non-None lambda for the tensor term
        summary = model.summary()
        ti_row = next(r for r in summary._coef_rows if r.name == "x1:x2")
        assert ti_row.smoothing_lambda is not None, (
            "Tensor term smoothing_lambda is None — fitted REML values not propagated to summary"
        )
