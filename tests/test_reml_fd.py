"""Finite-difference tests for REML gradient and Hessian.

Split from test_reml.py for maintainability — these tests are
computationally intensive and logically self-contained.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Gamma, NegativeBinomial, Tweedie
from superglm.features.spline import CubicRegressionSpline
from superglm.group_matrix import SparseSSPGroupMatrix
from superglm.profiling.tweedie import generate_tweedie_cpg
from superglm.reml import compute_dW_deta


class TestREMLFiniteDifference:
    """Verify analytic gradient and Hessian match finite differences."""

    @staticmethod
    def _setup_model(family, seed=42):
        """Build a fitted model with two CRS splines for FD checks."""
        from superglm.group_matrix import DiscretizedSSPGroupMatrix
        from superglm.reml import build_penalty_caches
        from superglm.reml.penalty_algebra import compute_penalty_nullity
        from superglm.reml.scale import (
            prepare_gamma_reml_scale_data,
            profile_gamma_reml_scale,
        )
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
            family_obj = family
        elif family == "gamma":
            y = rng.gamma(shape=5.0, scale=mu / 5.0)
            y = np.maximum(y, 1e-4)
            family_obj = family
        elif family == "nb2":
            theta = 5.0
            y = rng.negative_binomial(n=theta, p=theta / (theta + mu)).astype(float)
            family_obj = NegativeBinomial(theta=theta)
        elif family == "tweedie":
            y = generate_tweedie_cpg(n, mu, phi=1.5, p=1.5, rng=rng)
            y = np.maximum(y, 0.0)
            family_obj = Tweedie(p=1.5)
        else:
            raise ValueError(family)

        df = pd.DataFrame({"x1": x1, "x2": x2})
        m = SuperGLM(
            features={
                "x1": CubicRegressionSpline(n_knots=8),
                "x2": CubicRegressionSpline(n_knots=8),
            },
            family=family_obj,
        )
        m.fit(df, y)

        sample_weight = np.ones(n)
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

        penalty_caches = build_penalty_caches(m._dm.group_matrices, reml_groups)

        pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
            X=m._dm,
            y=y,
            weights=sample_weight,
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
        assert pirls_result.reml_hessian_rank is not None
        M_p = compute_penalty_nullity(S, hessian_rank=pirls_result.reml_hessian_rank)
        phi_hat = 1.0
        if isinstance(m._distribution, Gamma):
            scale_profile = profile_gamma_reml_scale(
                prepare_gamma_reml_scale_data(y, sample_weight),
                pirls_result.deviance + pq,
                M_p,
            )
            phi_hat = scale_profile.phi
        elif not getattr(m._distribution, "scale_known", True):
            phi_hat = max((pirls_result.deviance + pq) / max(n - M_p, 1.0), 1e-10)

        return (
            m,
            y,
            sample_weight,
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

    @pytest.mark.parametrize("family", ["poisson", "gamma", "nb2", "tweedie"])
    def test_gradient_matches_fd(self, family):
        """Analytic gradient matches central FD of objective (partial: fixed β, W)."""
        (
            m,
            y,
            sample_weight,
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
            inverse_phi=1.0 / phi_hat,
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
                sample_weight,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )
            om = m._reml_laml_objective(
                y,
                pirls_result,
                lam_m,
                sample_weight,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )
            fd_grad[i] = (op - om) / (2 * eps)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("family", ["poisson", "gamma", "nb2", "tweedie"])
    def test_hessian_matches_fd(self, family):
        """Approximate outer Hessian matches full outer FD to within ~5%.

        The analytic Hessian includes the IFT correction (dβ̂/dρ = -H⁻¹ S β̂)
        but holds W fixed. FD re-solves PIRLS, so W changes. The residual
        includes both the fixed-W approximation and higher-order IFT terms.
        """
        from superglm.reml.penalty_algebra import compute_penalty_nullity
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        (
            m,
            y,
            sample_weight,
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
                    weights=sample_weight,
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
                    assert result_pert.reml_hessian_rank is not None
                    M_p = compute_penalty_nullity(
                        S_pert,
                        hessian_rank=result_pert.reml_hessian_rank,
                    )
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

    @pytest.mark.parametrize("family", ["poisson", "gamma", "nb2", "tweedie"])
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
            sample_weight,
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
            sample_weight,
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
                    weights=sample_weight,
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
                    sample_weight,
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

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_dW_deta_matches_fd(self, family):
        """Verify compute_dW_deta() against central finite differences of W(eta)."""
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.links import stabilize_eta

        (
            m,
            y,
            sample_weight,
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

        eta = stabilize_eta(
            m._dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            m._link,
        )
        mu = clip_mu(m._link.inverse(eta), m._distribution)

        dW_analytic = compute_dW_deta(m._link, m._distribution, mu, eta, sample_weight)

        eps = 1e-6

        def compute_W(eta_vals):
            mu_vals = clip_mu(m._link.inverse(eta_vals), m._distribution)
            g1 = m._link.deriv_inverse(eta_vals)
            V = np.maximum(m._distribution.variance(mu_vals), _VARIANCE_FLOOR)
            return sample_weight * g1**2 / V

        W_plus = compute_W(eta + eps)
        W_minus = compute_W(eta - eps)
        dW_fd = (W_plus - W_minus) / (2 * eps)

        if dW_analytic is None:
            # Gamma/log: dW/deta = 0, so FD should also be ~0
            np.testing.assert_allclose(dW_fd, 0.0, atol=1e-6)
        else:
            np.testing.assert_allclose(dW_analytic, dW_fd, rtol=1e-5, atol=1e-10)

    @pytest.mark.parametrize("family", ["poisson", "gamma", "nb2", "tweedie"])
    def test_d2W_deta2_analytic_matches_fd(self, family):
        """Analytic compute_d2W_deta2() matches FD of compute_dW_deta()."""
        from superglm.distributions import clip_mu
        from superglm.links import stabilize_eta
        from superglm.reml import compute_d2W_deta2

        (
            m,
            y,
            sample_weight,
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

        eta = stabilize_eta(
            m._dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            m._link,
        )
        mu = clip_mu(m._link.inverse(eta), m._distribution)

        # Analytic d²W/dη²
        d2W_analytic = compute_d2W_deta2(m._link, m._distribution, mu, eta, sample_weight)

        # FD of compute_dW_deta
        eps = 1e-5
        eta_p = eta + eps
        mu_p = clip_mu(m._link.inverse(eta_p), m._distribution)
        dW_p = compute_dW_deta(m._link, m._distribution, mu_p, eta_p, sample_weight)

        eta_m = eta - eps
        mu_m = clip_mu(m._link.inverse(eta_m), m._distribution)
        dW_m = compute_dW_deta(m._link, m._distribution, mu_m, eta_m, sample_weight)

        if dW_p is None or dW_m is None:
            # Gamma/log: dW/deta=0 everywhere, so d²W/deta²=0
            if d2W_analytic is not None:
                np.testing.assert_allclose(d2W_analytic, 0.0, atol=1e-6)
            return

        d2W_fd = (dW_p - dW_m) / (2 * eps)

        if d2W_analytic is None:
            np.testing.assert_allclose(d2W_fd, 0.0, atol=1e-4)
        else:
            np.testing.assert_allclose(d2W_analytic, d2W_fd, rtol=1e-4, atol=1e-8)

    def test_w_correction_zero_for_gamma_log(self):
        """Gamma with log link has dW/dη=0, so W correction must vanish."""
        (
            m,
            y,
            sample_weight,
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
            sample_weight,
            offset_arr,
        )
        assert result is None, "Gamma/log should have zero W correction"

    @pytest.mark.parametrize("family", ["poisson", "nb2", "tweedie"])
    def test_w_correction_nonzero(self, family):
        """Poisson/NB2/Tweedie with log link have dW/deta != 0."""
        (
            m,
            y,
            sample_weight,
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

        result = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
        )
        assert result is not None, f"{family}/log should have nonzero W correction"
        grad_correction, dH_extra = result
        assert np.any(np.abs(grad_correction) > 1e-6)
        assert len(dH_extra) == len(reml_groups)

    @pytest.mark.parametrize("family", ["poisson", "gamma", "nb2", "tweedie"])
    def test_total_hessian_matches_fd(self, family):
        """Hessian with dH_extra vs FD of total gradient (partial + W correction).

        Finite-differences the total gradient (including W correction) by
        re-solving PIRLS at perturbed ρ and recomputing both the partial
        gradient and W correction at each perturbation.  The analytic Hessian
        with dH_extra should match better than without (for Poisson; for
        Gamma the correction is zero so both are equivalent).
        """
        from superglm.reml.penalty_algebra import compute_penalty_nullity
        from superglm.solvers.irls_direct import (
            _build_penalty_matrix,
            fit_irls_direct,
        )

        (
            m,
            y,
            sample_weight,
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
            sample_weight,
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
                    weights=sample_weight,
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
                    assert result_pert.reml_hessian_rank is not None
                    M_p = compute_penalty_nullity(
                        S_pert,
                        hessian_rank=result_pert.reml_hessian_rank,
                    )
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
                    sample_weight,
                    offset_arr,
                )
                if w_corr_pert is not None:
                    grad_pert = grad_pert + w_corr_pert[0]

                if sign == 1:
                    grad_plus = grad_pert
                else:
                    grad_minus = grad_pert

            fd_hess[:, j] = (grad_plus - grad_minus) / (2 * eps)

        # Hessian with dH_extra should match FD at least as well as without.
        # For NB2/Tweedie the first-order W correction can occasionally be
        # marginally worse on individual diagonal entries due to higher-order
        # terms (dropped d²W/dρ²), so allow a small relative slack.
        diag_fd = np.diag(fd_hess)
        err_with = np.abs(np.diag(hess_with) - diag_fd)
        err_without = np.abs(np.diag(hess_without) - diag_fd)

        for i in range(m_groups):
            slack = 0.02 * abs(diag_fd[i]) if family in ("nb2", "tweedie") else 1e-4
            assert err_with[i] <= err_without[i] + slack, (
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

    def test_total_gradient_would_add_w_correction_twice_on_diagonal(self):
        """Passing total g adds exactly the separately differentiated W term."""
        (
            m,
            y,
            sample_weight,
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

        # Partial gradient (fixed W)
        grad_partial = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )

        # W correction (nonzero for Poisson/log)
        w_corr = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
        )
        assert w_corr is not None, "Poisson/log must have nonzero W correction"
        grad_total = grad_partial + w_corr[0]
        dH_extra = w_corr[1]

        # dH_extra already adds W_i to the inverse-product term.  Supplying
        # grad_total here would add the first derivative once more through
        # the fixed-W diagonal identity g_partial + rank/2.
        hess_total = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad_total,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
            dH_extra=dH_extra,
        )

        # The production call must use this fixed-W gradient.
        hess_partial = m._reml_direct_hessian(
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

        np.testing.assert_allclose(
            np.diag(hess_total) - np.diag(hess_partial),
            w_corr[0],
            rtol=1e-12,
            atol=1e-12,
        )

        # Off-diagonals must be identical (only the diagonal correction
        # depends on the gradient parameter)
        m_groups = len(reml_groups)
        for i in range(m_groups):
            for j in range(m_groups):
                if i != j:
                    assert hess_total[i, j] == hess_partial[i, j], (
                        f"Off-diagonal [{i},{j}] should not depend on gradient"
                    )

    def test_w_correction_order2_returns_three_tuple(self):
        """w_correction_order=2 returns (grad, dH_extra, dH2_cross) 3-tuple."""
        (
            m,
            y,
            sample_weight,
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
            sample_weight,
            offset_arr,
            w_correction_order=2,
        )
        assert result is not None
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        grad_corr, dH_extra, dH2_cross = result
        assert grad_corr.shape == (len(reml_groups),)
        assert dH2_cross is not None
        assert dH2_cross.shape == (len(reml_groups), len(reml_groups))
        # dH2_cross should be symmetric
        np.testing.assert_allclose(dH2_cross, dH2_cross.T, atol=1e-12)

    def test_w_correction_differentiates_entire_profiled_determinant(self):
        """The W correction includes both log|H_c| and log(sum(W))."""
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.links import stabilize_eta
        from superglm.solvers.irls_direct import _build_penalty_matrix

        (
            m,
            _y,
            sample_weight,
            offset_arr,
            lambdas,
            reml_groups,
            _penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            _XtWX,
            _phi_hat,
            _n,
        ) = self._setup_model("nb2")

        correction = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
            w_correction_order=1,
        )
        assert correction is not None

        group_name = reml_groups[0][1].name
        p = m._dm.p
        S_base = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p)
        doubled = lambdas.copy()
        doubled[group_name] *= 2.0
        S_i = _build_penalty_matrix(m._dm.group_matrices, m._groups, doubled, p) - S_base

        dbeta_i = -(XtWX_S_inv @ (S_i @ pirls_result.beta))
        assert pirls_result.rank_info is not None
        mean_x = np.asarray(pirls_result.rank_info.mean_x)
        deta_i = m._dm.matvec(dbeta_i) - float(mean_x @ dbeta_i)
        eta = stabilize_eta(
            m._dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            m._link,
        )
        X_aug = np.column_stack((np.ones(m._dm.n), m._dm.toarray()))

        def half_profiled_logdet(rho_step: float) -> float:
            eta_step = stabilize_eta(eta + rho_step * deta_i, m._link)
            mu_step = clip_mu(m._link.inverse(eta_step), m._distribution)
            W_step = (
                sample_weight
                * m._link.deriv_inverse(eta_step) ** 2
                / np.maximum(m._distribution.variance(mu_step), _VARIANCE_FLOOR)
            )
            lambda_step = lambdas.copy()
            lambda_step[group_name] *= float(np.exp(rho_step))
            S_step = _build_penalty_matrix(
                m._dm.group_matrices,
                m._groups,
                lambda_step,
                p,
            )
            H_aug = X_aug.T @ (W_step[:, None] * X_aug)
            H_aug[1:, 1:] += S_step
            sign, logdet = np.linalg.slogdet(H_aug)
            assert sign > 0.0
            return 0.5 * float(logdet)

        eps = 2.0e-5
        finite_difference = (half_profiled_logdet(eps) - half_profiled_logdet(-eps)) / (2.0 * eps)
        analytic = 0.5 * float(np.sum(XtWX_S_inv * S_i)) + correction[0][0]

        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-7, atol=2e-8)

    def test_w_correction_order2_includes_log_weight_sum_curvature(self):
        """Second derivatives include curvature of the profiled scalar factor."""
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.links import stabilize_eta
        from superglm.reml import compute_d2W_deta2
        from superglm.solvers.irls_direct import _build_penalty_matrix

        (
            m,
            _y,
            sample_weight,
            offset_arr,
            lambdas,
            reml_groups,
            _penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            _XtWX,
            _phi_hat,
            _n,
        ) = self._setup_model("nb2")
        correction = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
            w_correction_order=2,
        )
        assert correction is not None and correction[2] is not None

        group_name = reml_groups[0][1].name
        p = m._dm.p
        S_base = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p)
        doubled = lambdas.copy()
        doubled[group_name] *= 2.0
        S_i = _build_penalty_matrix(m._dm.group_matrices, m._groups, doubled, p) - S_base
        dbeta_i = -(XtWX_S_inv @ (S_i @ pirls_result.beta))

        assert pirls_result.rank_info is not None
        mean_x = np.asarray(pirls_result.rank_info.mean_x)
        sum_w = float(pirls_result.rank_info.sum_w)
        X_centered = m._dm.toarray() - mean_x
        deta_i = X_centered @ dbeta_i
        eta = stabilize_eta(
            m._dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            m._link,
        )
        mu = clip_mu(m._link.inverse(eta), m._distribution)
        dW_deta = compute_dW_deta(m._link, m._distribution, mu, eta, sample_weight)
        d2W_deta2 = compute_d2W_deta2(m._link, m._distribution, mu, eta, sample_weight)
        assert dW_deta is not None and d2W_deta2 is not None

        a_i = dW_deta * deta_i
        dmean_i = (X_centered.T @ a_i) / sum_w
        rhs = X_centered.T @ (dW_deta * deta_i**2) + 2.0 * (S_i @ dbeta_i)
        d2beta_ii = dbeta_i - XtWX_S_inv @ rhs
        d2eta_ii = X_centered @ d2beta_ii - float(dmean_i @ dbeta_i)
        d2w_ii = d2W_deta2 * deta_i**2 + dW_deta * d2eta_ii
        C_ii = X_centered.T @ (d2w_ii[:, None] * X_centered)
        C_ii -= 2.0 * sum_w * np.outer(dmean_i, dmean_i)
        matrix_curvature = 0.5 * float(np.sum(XtWX_S_inv * C_ii))

        def half_log_weight_sum(rho_step: float) -> float:
            eta_step = stabilize_eta(
                eta + rho_step * deta_i + 0.5 * rho_step**2 * d2eta_ii,
                m._link,
            )
            mu_step = clip_mu(m._link.inverse(eta_step), m._distribution)
            W_step = (
                sample_weight
                * m._link.deriv_inverse(eta_step) ** 2
                / np.maximum(m._distribution.variance(mu_step), _VARIANCE_FLOOR)
            )
            return 0.5 * float(np.log(np.sum(W_step)))

        eps = 2.0e-3
        scalar_curvature_fd = (
            half_log_weight_sum(eps) - 2.0 * half_log_weight_sum(0.0) + half_log_weight_sum(-eps)
        ) / eps**2

        np.testing.assert_allclose(
            correction[2][0, 0],
            matrix_curvature + scalar_curvature_fd,
            rtol=3e-5,
            atol=2e-8,
        )

    def test_w_correction_order2_matches_fd_of_centered_hessian_derivative(self):
        """Order-2 correction differentiates the profiled-intercept Hessian."""
        from superglm.solvers.irls_direct import fit_irls_direct

        (
            m,
            y,
            sample_weight,
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
        del penalty_ranks, XtWX, phi_hat, n

        base = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
            w_correction_order=2,
        )
        assert base is not None and base[2] is not None
        analytic = base[2]
        names = [group.name for _, group in reml_groups]
        eps = 1e-3
        fd = np.zeros_like(analytic)

        for j, name in enumerate(names):
            derivative_matrices = []
            for sign in (-1.0, 1.0):
                perturbed = lambdas.copy()
                perturbed[name] *= float(np.exp(sign * eps))
                result, inverse, _ = fit_irls_direct(
                    X=m._dm,
                    y=y,
                    weights=sample_weight,
                    family=m._distribution,
                    link=m._link,
                    groups=m._groups,
                    lambda2=perturbed,
                    offset=offset_arr,
                    beta_init=pirls_result.beta,
                    intercept_init=pirls_result.intercept,
                    return_xtwx=True,
                )
                correction = m._reml_w_correction(
                    result,
                    inverse,
                    perturbed,
                    reml_groups,
                    penalty_caches,
                    sample_weight,
                    offset_arr,
                    w_correction_order=1,
                )
                assert correction is not None
                derivative_matrices.append(correction[1])
            for i in range(len(names)):
                matrix_derivative = (derivative_matrices[1][i] - derivative_matrices[0][i]) / (
                    2.0 * eps
                )
                fd[i, j] = 0.5 * float(np.sum(XtWX_S_inv * matrix_derivative))

        np.testing.assert_allclose(analytic, fd, rtol=4e-3, atol=2e-5)

    def test_w_correction_order2_changes_hessian(self):
        """Order-2 W correction produces a different Hessian than order-1.

        For Poisson/log, d²W/dη² is nonzero, so the second-order
        cross-terms should alter the Hessian off-diagonals.
        """
        (
            m,
            y,
            sample_weight,
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

        # Order 1
        w1 = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
            w_correction_order=1,
        )
        grad_partial = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        grad1 = grad_partial + w1[0]
        hess1 = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad_partial,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
            dH_extra=w1[1],
        )

        # Order 2
        w2 = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
            w_correction_order=2,
        )
        grad2 = grad_partial + w2[0]  # gradient is same for both orders
        hess2 = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad_partial,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
            dH_extra=w2[1],
            dH2_cross=w2[2],
        )

        # Gradients must be identical
        np.testing.assert_allclose(grad1, grad2, atol=1e-14)

        # Hessians must differ (d²W/dη² is nonzero for Poisson/log)
        hess_diff = np.abs(hess2 - hess1)
        assert np.any(hess_diff > 1e-8), (
            "Poisson/log: order-2 Hessian should differ from order-1, "
            f"but max diff = {hess_diff.max():.2e}"
        )

    def test_direct_optimizer_does_not_double_count_w_gradient_in_hessian(
        self,
        monkeypatch,
    ):
        """The diagonal S_i term uses the fixed-W gradient, not total gradient."""
        import superglm.reml.direct as direct
        from superglm.reml.gradient import reml_direct_gradient
        from superglm.reml.penalty_algebra import coerce_reml_penalties

        (
            m,
            y,
            sample_weight,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            _pirls_result,
            _XtWX_S_inv,
            _XtWX,
            _phi_hat,
            _n,
        ) = self._setup_model("poisson")
        penalties = coerce_reml_penalties(
            reml_groups=reml_groups,
            group_matrices=m._dm.group_matrices,
            penalty_caches=penalty_caches,
        )
        original_hessian = direct.reml_direct_hessian
        checked = False

        def checked_hessian(*args, **kwargs):
            nonlocal checked
            expected_partial = reml_direct_gradient(
                args[0],
                kwargs["pirls_result"],
                args[2],
                args[3],
                phi_hat=kwargs["phi_hat"],
                reml_penalties=kwargs["reml_penalties"],
                penalty_caches=kwargs["penalty_caches"],
            )
            np.testing.assert_allclose(kwargs["gradient"], expected_partial, atol=1e-12)
            checked = True
            return original_hessian(*args, **kwargs)

        monkeypatch.setattr(direct, "reml_direct_hessian", checked_hessian)
        direct.optimize_direct_reml(
            dm=m._dm,
            distribution=m._distribution,
            link=m._link,
            groups=m._groups,
            discrete=False,
            y=y,
            sample_weight=sample_weight,
            offset_arr=offset_arr,
            reml_groups=reml_groups,
            penalty_ranks=penalty_ranks,
            lambdas=lambdas.copy(),
            max_reml_iter=1,
            reml_tol=1e-6,
            verbose=False,
            penalty_caches=penalty_caches,
            w_correction_order=2,
            reml_penalties=penalties,
        )
        assert checked

    def test_order2_hessian_matches_fd_of_total_gradient_for_poisson(self):
        """Full order-2 curvature matches the total-gradient finite difference."""
        from superglm.solvers.irls_direct import fit_irls_direct

        (
            m,
            y,
            sample_weight,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            XtWX_S_inv,
            _XtWX,
            phi_hat,
            n,
        ) = self._setup_model("poisson")
        grad_partial = m._reml_direct_gradient(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )
        correction = m._reml_w_correction(
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
            w_correction_order=2,
        )
        assert correction is not None and correction[2] is not None
        analytic = m._reml_direct_hessian(
            XtWX_S_inv,
            lambdas,
            reml_groups,
            grad_partial,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n,
            phi_hat=phi_hat,
            dH_extra=correction[1],
            dH2_cross=correction[2],
        )

        names = [group.name for _, group in reml_groups]
        eps = 1.0e-4
        finite_difference = np.zeros_like(analytic)
        for j, name in enumerate(names):
            gradients = []
            for sign in (-1.0, 1.0):
                perturbed = lambdas.copy()
                perturbed[name] *= float(np.exp(sign * eps))
                result, inverse, _ = fit_irls_direct(
                    X=m._dm,
                    y=y,
                    weights=sample_weight,
                    family=m._distribution,
                    link=m._link,
                    groups=m._groups,
                    lambda2=perturbed,
                    offset=offset_arr,
                    beta_init=pirls_result.beta,
                    intercept_init=pirls_result.intercept,
                    return_xtwx=True,
                )
                gradient = m._reml_direct_gradient(
                    result,
                    inverse,
                    perturbed,
                    reml_groups,
                    penalty_ranks,
                    phi_hat=phi_hat,
                )
                perturbed_correction = m._reml_w_correction(
                    result,
                    inverse,
                    perturbed,
                    reml_groups,
                    penalty_caches,
                    sample_weight,
                    offset_arr,
                    w_correction_order=1,
                )
                assert perturbed_correction is not None
                gradients.append(gradient + perturbed_correction[0])
            finite_difference[:, j] = (gradients[1] - gradients[0]) / (2.0 * eps)

        np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-8)

    def test_w_correction_is_stable_under_large_feature_translation(self, monkeypatch):
        """Signed centered moments must not cancel for translated columns."""
        import superglm.reml.w_derivatives as w_derivatives
        from superglm.distributions import Poisson
        from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
        from superglm.links import LogLink
        from superglm.solvers.irls_direct import fit_irls_direct
        from superglm.types import GroupSlice, PenaltyComponent

        rng = np.random.default_rng(123)
        x = np.linspace(-1.5, 1.5, 320)
        y = rng.poisson(np.exp(0.25 + 0.35 * x)).astype(float)
        weights = np.ones_like(x)
        offset = np.zeros_like(x)
        family = Poisson()
        link = LogLink()
        group = GroupSlice(name="x", start=0, end=1)
        penalty = PenaltyComponent(
            name="x",
            group_name="x",
            group_index=0,
            group_sl=slice(0, 1),
            omega_raw=np.ones((1, 1)),
            omega_ssp=np.ones((1, 1)),
            rank=1.0,
            log_det_omega_plus=0.0,
            eigvals_omega=np.ones(1),
        )
        lambdas = {"x": 4.0}
        stable_gram_calls = 0
        original_centered_gram_rhs = w_derivatives.centered_gram_rhs

        def counted_centered_gram_rhs(**kwargs):
            nonlocal stable_gram_calls
            stable_gram_calls += 1
            return original_centered_gram_rhs(**kwargs)

        monkeypatch.setattr(w_derivatives, "centered_gram_rhs", counted_centered_gram_rhs)

        def correction_for_shift(shift: float) -> tuple[np.ndarray, np.ndarray]:
            dm = DesignMatrix(
                [DenseGroupMatrix((x + shift)[:, None])],
                n=len(x),
                p=1,
            )
            result, inverse, _ = fit_irls_direct(
                X=dm,
                y=y,
                weights=weights,
                family=family,
                link=link,
                groups=[group],
                lambda2=lambdas,
                offset=offset,
                return_xtwx=True,
                reml_penalties=[penalty],
            )
            correction = w_derivatives.reml_w_correction(
                dm=dm,
                link=link,
                groups=[group],
                pirls_result=result,
                XtWX_S_inv=inverse,
                lambdas=lambdas,
                sample_weight=weights,
                offset_arr=offset,
                distribution=family,
                w_correction_order=2,
                reml_penalties=[penalty],
            )
            assert correction is not None and correction[2] is not None
            return correction[0], correction[2]

        base_gradient, base_hessian = correction_for_shift(0.0)
        assert stable_gram_calls == 0, "well-scaled designs must retain the execution-plan hot path"
        shifted_gradient, shifted_hessian = correction_for_shift(1.0e8)
        assert stable_gram_calls > 0
        np.testing.assert_allclose(shifted_gradient, base_gradient, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(shifted_hessian, base_hessian, rtol=2e-5, atol=2e-8)

    def test_fit_reml_w_correction_order2_converges(self):
        """fit_reml(w_correction_order=2) runs and converges on Poisson data."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + np.sin(2 * np.pi * x1))
        y = rng.poisson(mu).astype(float)
        df = pd.DataFrame({"x1": x1})

        from superglm.features.spline import CubicRegressionSpline

        m = SuperGLM(
            features={"x1": CubicRegressionSpline(n_knots=8)},
            family="poisson",
        )
        m.fit_reml(df, y, w_correction_order=2)
        assert m._result.converged
