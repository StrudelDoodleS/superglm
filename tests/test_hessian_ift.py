"""Verify the IFT Hessian correction against full outer finite differences."""

import numpy as np
import pandas as pd

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
        M_p,
    )


def full_outer_fd_hessian(
    m,
    y,
    exposure,
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


if __name__ == "__main__":
    for family in ["poisson", "gamma"]:
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

        fd_hess = full_outer_fd_hessian(
            m,
            y,
            exposure,
            offset_arr,
            lambdas,
            reml_groups,
            penalty_ranks,
            pirls_result,
            M_p,
            n,
        )

        diff = np.abs(hess - fd_hess)
        rel_err = diff / (np.abs(fd_hess) + 1e-10)
        print(f"{family}:")
        print(f"  max abs diff = {diff.max():.4f}")
        print(f"  max rel err  = {rel_err.max():.4f}")
        print(f"  Analytic diag: {np.diag(hess)}")
        print(f"  FD diag:       {np.diag(fd_hess)}")
        print(f"  Analytic:\n{hess}")
        print(f"  FD:\n{fd_hess}")
        print()
