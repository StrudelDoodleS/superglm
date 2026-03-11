"""Finite-difference tests for REML gradient and Hessian.

Split from test_reml.py for maintainability — these tests are
computationally intensive and logically self-contained.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import CubicRegressionSpline
from superglm.group_matrix import SparseSSPGroupMatrix


class TestREMLFiniteDifference:
    """Verify analytic gradient and Hessian match finite differences."""

    @staticmethod
    def _setup_model(family, seed=42):
        """Build a fitted model with two CRS splines for FD checks."""
        from superglm.group_matrix import DiscretizedSSPGroupMatrix
        from superglm.reml import build_penalty_caches
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        rng = np.random.default_rng(seed)
        n = 800
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2)
        if family == "poisson":
            y = rng.poisson(mu).astype(float)
        elif family == "gamma":
            y = rng.gamma(shape=5.0, scale=mu / 5.0)
            y = np.maximum(y, 1e-4)
        else:
            raise ValueError(family)

        df = pd.DataFrame({"x1": x1, "x2": x2})
        m = SuperGLM(
            features={
                "x1": CubicRegressionSpline(n_knots=8),
                "x2": CubicRegressionSpline(n_knots=8),
            },
            family=family,
        )
        m.fit(df, y)

        exposure = np.ones(n)
        offset_arr = np.zeros(n)
        lambdas = {"x1": 10.0, "x2": 0.5}

        reml_groups = []
        penalty_ranks = {}
        for i, (gm, g) in enumerate(zip(m._dm.group_matrices, m._groups)):
            if g.penalized and isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                reml_groups.append((i, g))
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
                eigv = np.linalg.eigvalsh(omega_ssp)
                penalty_ranks[g.name] = float(np.sum(eigv > 1e-8 * max(eigv.max(), 1e-12)))

        penalty_caches = build_penalty_caches(m._dm.group_matrices, m._groups, reml_groups)

        pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=exposure,
            family=m._distribution,
            link=m._link,
            groups=m._groups,
            lambda2=lambdas,
            offset=offset_arr,
            return_xtwx=True,
        )

        p_dim = XtWX.shape[0]
        S = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p_dim)
        pq = float(pirls_result.beta @ S @ pirls_result.beta)
        M_p = sum(c.rank for c in penalty_caches.values())
        phi_hat = 1.0
        if not getattr(m._distribution, "scale_known", True):
            phi_hat = max((pirls_result.deviance + pq) / max(n - M_p, 1.0), 1e-10)

        return (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        )

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_gradient_matches_fd(self, family):
        """Analytic gradient matches central FD of objective (partial: fixed β, W)."""
        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model(family)

        grad = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )

        eps = 1e-5
        group_names = [g.name for _, g in reml_groups]
        fd_grad = np.zeros(len(reml_groups))
        for i, name in enumerate(group_names):
            rho_base = np.log(lambdas[name])
            lam_p, lam_m = lambdas.copy(), lambdas.copy()
            lam_p[name] = np.exp(rho_base + eps)
            lam_m[name] = np.exp(rho_base - eps)
            op = m._reml_laml_objective(
                y,
                pirls_result,
                lam_p,
                exposure,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )
            om = m._reml_laml_objective(
                y,
                pirls_result,
                lam_m,
                exposure,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )
            fd_grad[i] = (op - om) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_hessian_matches_fd(self, family):
        """Approximate outer Hessian matches full outer FD to within ~5%.

        The analytic Hessian includes the IFT correction (dβ̂/dρ = -H⁻¹ S β̂)
        but holds W fixed. FD re-solves PIRLS, so W changes. The residual
        includes both the fixed-W approximation and higher-order IFT terms.
        """
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model(family)

        grad = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        hess = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
        )

        eps = 1e-4
        group_names = [g.name for _, g in reml_groups]
        p_dim = XtWX.shape[0]
        M_p = sum(c.rank for c in penalty_caches.values())
        m_groups = len(reml_groups)
        fd_hess = np.zeros((m_groups, m_groups))

        for j in range(m_groups):
            rho_base = np.log(lambdas[group_names[j]])
            for sign in [+1, -1]:
                lam_pert = lambdas.copy()
                lam_pert[group_names[j]] = np.exp(rho_base + sign * eps)

                # Re-solve PIRLS at perturbed lambda (full outer FD)
                result_pert, inv_pert, xtwx_pert = fit_irls_direct(
                    X=m._dm,
                    y=y,
                    weights=exposure,
                    family=m._distribution,
                    link=m._link,
                    groups=m._groups,
                    lambda2=lam_pert,
                    offset=offset_arr,
                    beta_init=pirls_result.beta,
                    intercept_init=pirls_result.intercept,
                    return_xtwx=True,
                )

                phi_pert = 1.0
                if not getattr(m._distribution, "scale_known", True):
                    S_pert = _build_penalty_matrix(m._dm.group_matrices, m._groups, lam_pert, p_dim)
                    pq_pert = float(result_pert.beta @ S_pert @ result_pert.beta)
                    phi_pert = max((result_pert.deviance + pq_pert) / max(n - M_p, 1.0), 1e-10)

                grad_pert = m._reml_direct_gradient(
                    result_pert,
                    inv_pert,
                    lam_pert,
                    reml_groups,
                    penalty_ranks,
                    phi_hat=phi_pert,
                )
                if sign == 1:
                    grad_plus = grad_pert
                else:
                    grad_minus = grad_pert

            fd_hess[:, j] = (grad_plus - grad_minus) / (2 * eps)

        # Check diagonal and off-diagonal separately for tighter regression bounds.
        # Diagonal: rtol=5% is tight enough; atol=0.1 catches absolute drift.
        # Off-diagonal: relative to diagonal scale (small cross-terms need
        # scale-aware tolerance, not a blanket atol=0.5 that hides regressions).
        diag_analytic = np.diag(hess)
        diag_fd = np.diag(fd_hess)
        np.testing.assert_allclose(diag_analytic, diag_fd, rtol=0.05, atol=0.1)

        for i in range(m_groups):
            for j in range(m_groups):
                if i == j:
                    continue
                abs_err = abs(hess[i, j] - fd_hess[i, j])
                scale = max(abs(fd_hess[i, j]), abs(diag_fd.mean()), 1e-6)
                rel_err = abs_err / scale
                assert rel_err < 0.15, (
                    f"{family} Hessian[{i},{j}]: analytic={hess[i, j]:.6f}, "
                    f"fd={fd_hess[i, j]:.6f}, rel_err={rel_err:.4f}"
                )

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_total_gradient_matches_outer_fd(self, family):
        """Total gradient (partial + W correction) vs outer FD of objective.

        The outer FD re-solves PIRLS at perturbed ρ, so β̂ and W change.
        The total gradient should match the FD of f(ρ) = V(β̂(ρ), ρ) better
        than the partial gradient.

        For Gamma/log, dW/dη=0 so partial = total and both match equally.
        For Poisson/log, the W correction should reduce the discrepancy.
        """
        from superglm.solvers.irls_direct import fit_irls_direct

        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model(family)

        # Partial gradient (fixed W)
        grad_partial = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )

        # W correction
        w_corr = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            exposure,
            offset_arr,
        )
        if w_corr is not None:
            grad_total = grad_partial + w_corr[0]
        else:
            grad_total = grad_partial.copy()

        # Outer FD: re-solve PIRLS, evaluate V(ρ±ε), central difference
        eps = 1e-5
        group_names = [g.name for _, g in reml_groups]
        fd_grad = np.zeros(len(reml_groups))

        for i, name in enumerate(group_names):
            rho_base = np.log(lambdas[name])
            objs = {}
            for sign in [+1, -1]:
                lam_pert = lambdas.copy()
                lam_pert[name] = np.exp(rho_base + sign * eps)
                r_pert, _, xtwx_pert = fit_irls_direct(
                    X=m._dm,
                    y=y,
                    weights=exposure,
                    family=m._distribution,
                    link=m._link,
                    groups=m._groups,
                    lambda2=lam_pert,
                    offset=offset_arr,
                    beta_init=pirls_result.beta,
                    intercept_init=pirls_result.intercept,
                    return_xtwx=True,
                )
                objs[sign] = m._reml_laml_objective(
                    y,
                    r_pert,
                    lam_pert,
                    exposure,
                    offset_arr,
                    XtWX=xtwx_pert,
                    penalty_caches=penalty_caches,
                )
            fd_grad[i] = (objs[1] - objs[-1]) / (2 * eps)

        # Total gradient should be at least as close to outer FD as partial
        err_total = np.abs(grad_total - fd_grad)
        err_partial = np.abs(grad_partial - fd_grad)

        # For Gamma/log, W correction is zero → same error
        # For Poisson/log, total gradient should be closer or equal
        for i in range(len(reml_groups)):
            assert err_total[i] <= err_partial[i] + 1e-8, (
                f"{family} group {group_names[i]}: total gradient error "
                f"({err_total[i]:.6f}) should not exceed partial error "
                f"({err_partial[i]:.6f})"
            )

    def test_w_correction_zero_for_gamma_log(self):
        """Gamma with log link has dW/dη=0, so W correction must vanish."""
        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model("gamma")

        result = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            exposure,
            offset_arr,
        )
        assert result is None, "Gamma/log should have zero W correction"

    def test_w_correction_nonzero_for_poisson_log(self):
        """Poisson with log link has dW/dη=W, so W correction must be nonzero."""
        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model("poisson")

        result = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            exposure,
            offset_arr,
        )
        assert result is not None, "Poisson/log should have nonzero W correction"
        grad_correction, dH_extra = result
        assert np.any(np.abs(grad_correction) > 1e-6)
        assert len(dH_extra) == len(reml_groups)

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_total_hessian_matches_fd(self, family):
        """Hessian with dH_extra vs FD of total gradient (partial + W correction).

        Finite-differences the total gradient (including W correction) by
        re-solving PIRLS at perturbed ρ and recomputing both the partial
        gradient and W correction at each perturbation.  The analytic Hessian
        with dH_extra should match better than without (for Poisson; for
        Gamma the correction is zero so both are equivalent).
        """
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        (
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            XtWX,
            phi_hat,
            n,
        ) = self._setup_model(family)

        # Compute partial gradient + W correction at base point
        grad_partial = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        w_corr = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            exposure,
            offset_arr,
        )
        dH_extra = w_corr[1] if w_corr is not None else None

        # Analytic Hessian WITH dH_extra
        hess_with = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad_partial,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
            dH_extra=dH_extra,
        )

        # Analytic Hessian WITHOUT dH_extra (for comparison)
        hess_without = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad_partial,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
            dH_extra=None,
        )

        # FD of total gradient: re-solve PIRLS at perturbed ρ, recompute
        # both partial gradient and W correction
        eps = 1e-4
        group_names = [g.name for _, g in reml_groups]
        p_dim = XtWX.shape[0]
        M_p = sum(c.rank for c in penalty_caches.values())
        m_groups = len(reml_groups)
        fd_hess = np.zeros((m_groups, m_groups))

        for j in range(m_groups):
            rho_base = np.log(lambdas[group_names[j]])
            for sign in [+1, -1]:
                lam_pert = lambdas.copy()
                lam_pert[group_names[j]] = np.exp(rho_base + sign * eps)

                result_pert, inv_pert, xtwx_pert = fit_irls_direct(
                    X=m._dm,
                    y=y,
                    weights=exposure,
                    family=m._distribution,
                    link=m._link,
                    groups=m._groups,
                    lambda2=lam_pert,
                    offset=offset_arr,
                    beta_init=pirls_result.beta,
                    intercept_init=pirls_result.intercept,
                    return_xtwx=True,
                )

                phi_pert = 1.0
                if not getattr(m._distribution, "scale_known", True):
                    S_pert = _build_penalty_matrix(m._dm.group_matrices, m._groups, lam_pert, p_dim)
                    pq_pert = float(result_pert.beta @ S_pert @ result_pert.beta)
                    phi_pert = max((result_pert.deviance + pq_pert) / max(n - M_p, 1.0), 1e-10)

                # Total gradient = partial + W correction
                grad_pert = m._reml_direct_gradient(
                    result_pert,
                    inv_pert,
                    lam_pert,
                    reml_groups,
                    penalty_ranks,
                    phi_hat=phi_pert,
                )
                w_corr_pert = m._reml_w_correction(
                    result_pert,
                    inv_pert,
                    lam_pert,
                    reml_groups,
                    penalty_caches,
                    exposure,
                    offset_arr,
                )
                if w_corr_pert is not None:
                    grad_pert = grad_pert + w_corr_pert[0]

                if sign == 1:
                    grad_plus = grad_pert
                else:
                    grad_minus = grad_pert

            fd_hess[:, j] = (grad_plus - grad_minus) / (2 * eps)

        # Hessian with dH_extra should match FD at least as well as without
        diag_fd = np.diag(fd_hess)
        err_with = np.abs(np.diag(hess_with) - diag_fd)
        err_without = np.abs(np.diag(hess_without) - diag_fd)

        # For Poisson: with correction should be better or equal
        # For Gamma: correction is zero, so both should be equivalent
        for i in range(m_groups):
            assert err_with[i] <= err_without[i] + 1e-4, (
                f"{family} Hessian[{i},{i}]: with dH_extra err={err_with[i]:.6f} "
                f"exceeds without err={err_without[i]:.6f}"
            )

        # Both should be reasonably close to FD (within 15% relative)
        for i in range(m_groups):
            for j in range(m_groups):
                scale = max(abs(fd_hess[i, j]), abs(diag_fd.mean()), 1e-6)
                rel_err = abs(hess_with[i, j] - fd_hess[i, j]) / scale
                assert rel_err < 0.15, (
                    f"{family} total Hessian[{i},{j}]: analytic={hess_with[i, j]:.6f}, "
                    f"fd={fd_hess[i, j]:.6f}, rel_err={rel_err:.4f}"
                )
