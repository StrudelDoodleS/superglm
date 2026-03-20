"""Verify the IFT Hessian correction against full outer finite differences.

The analytic Hessian includes the IFT correction (dβ̂/dρ = -H⁻¹ S β̂)
but holds IRLS working weights W fixed (Laplace approximation).  FD
re-solves PIRLS at perturbed λ, so W changes too.  Tolerances account
for both the fixed-W approximation and higher-order IFT terms.
"""

import numpy as np
import pandas as pd
import pytest

from superglm.features.spline import CubicRegressionSpline
from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix
from superglm.model import SuperGLM
from superglm.reml import build_penalty_caches
from superglm.solvers.irls_direct import _build_penalty_matrix, fit_irls_direct


def _setup(family, seed=42):
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
    M_p = sum(c.rank for c in penalty_caches.values())
    phi_hat = 1.0
    if not getattr(m._distribution, "scale_known", True):
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
        M_p,
    )


def _full_outer_fd_hessian(
    m,
    y,
    sample_weight,
    offset_arr,
    lambdas,
    reml_groups,
    penalty_ranks,
    pirls_result,
    M_p,
    n,
    eps=1e-4,
):
    group_names = [g.name for _, g in reml_groups]
    m_groups = len(reml_groups)
    p_dim = m._dm.p
    fd_hess = np.zeros((m_groups, m_groups))

    for j in range(m_groups):
        rho_base = np.log(lambdas[group_names[j]])
        grads = {}
        for sign in [+1, -1]:
            lam_pert = lambdas.copy()
            lam_pert[group_names[j]] = np.exp(rho_base + sign * eps)
            r_pert, inv_pert, xtwx_pert = fit_irls_direct(
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
                pq_pert = float(r_pert.beta @ S_pert @ r_pert.beta)
                phi_pert = max((r_pert.deviance + pq_pert) / max(n - M_p, 1.0), 1e-10)
            grads[sign] = m._reml_direct_gradient(
                r_pert,
                inv_pert,
                lam_pert,
                reml_groups,
                penalty_ranks,
                phi_hat=phi_pert,
            )
        fd_hess[:, j] = (grads[1] - grads[-1]) / (2 * eps)
    return fd_hess


class TestIFTHessian:
    """Analytic Hessian (with IFT correction) vs full outer finite differences."""

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_hessian_diagonal_matches_fd(self, family):
        """Diagonal elements should match FD within 5% relative error."""
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
            M_p,
        ) = _setup(family)

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

        fd_hess = _full_outer_fd_hessian(
            m,
            y,
            sample_weight,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            pirls_result,
            M_p,
            n,
        )

        diag_analytic = np.diag(hess)
        diag_fd = np.diag(fd_hess)
        np.testing.assert_allclose(diag_analytic, diag_fd, rtol=0.05, atol=0.1)

    @pytest.mark.parametrize("family", ["poisson", "gamma"])
    def test_hessian_off_diagonal_matches_fd(self, family):
        """Off-diagonal elements should match FD within reasonable tolerance."""
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
            M_p,
        ) = _setup(family)

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

        fd_hess = _full_outer_fd_hessian(
            m,
            y,
            sample_weight,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            pirls_result,
            M_p,
            n,
        )

        m_groups = len(reml_groups)
        for i in range(m_groups):
            for j in range(m_groups):
                if i == j:
                    continue
                abs_err = abs(hess[i, j] - fd_hess[i, j])
                scale = max(abs(fd_hess[i, j]), abs(np.diag(fd_hess).mean()), 1e-6)
                rel_err = abs_err / scale
                assert rel_err < 0.15, (
                    f"{family} Hessian[{i},{j}]: analytic={hess[i, j]:.6f}, "
                    f"fd={fd_hess[i, j]:.6f}, rel_err={rel_err:.4f}"
                )
