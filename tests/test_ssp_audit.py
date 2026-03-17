"""Audits for SSP basis invariance.

These tests target the subtle question raised in review:

- the SSP basis uses a dense reparameterization ``R_inv``
- in this repo ``R_inv`` is built from ``G + lambda * Omega``
- so we want to verify that changing the valid SSP basis does not change
  the fitted function or the REML outer quantities when the smoothing
  problem itself is held fixed

The tests below compare mathematically equivalent SSP parameterizations:

1. an arbitrary orthogonal rotation of the fitted SSP basis
2. a different valid SSP preconditioner built from another scalar lambda

Both should leave the fitted function and REML criterion unchanged up to
numerical tolerance when the raw spline basis and raw penalty are the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.dm_builder import compute_projected_R_inv, compute_R_inv
from superglm.features.spline import CubicRegressionSpline
from superglm.group_matrix import DesignMatrix, SparseSSPGroupMatrix
from superglm.reml import build_penalty_caches
from superglm.reml_optimizer import (
    reml_direct_gradient,
    reml_direct_hessian,
    reml_laml_objective,
)
from superglm.solvers.irls_direct import fit_irls_direct


def _setup_poisson_spline_model(seed: int = 123):
    """Build a small smooth-only model with two SSP spline groups."""
    rng = np.random.default_rng(seed)
    n = 700
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    eta = 0.3 + 0.8 * np.sin(2 * np.pi * x1) - 0.5 * np.cos(2 * np.pi * x2)
    y = rng.poisson(np.exp(eta)).astype(float)
    exposure = np.ones(n)
    offset = np.zeros(n)

    df = pd.DataFrame({"x1": x1, "x2": x2})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "x1": CubicRegressionSpline(n_knots=8, penalty="ssp"),
            "x2": CubicRegressionSpline(n_knots=8, penalty="ssp"),
        },
    )
    model.fit(df, y, exposure=exposure)

    assert all(isinstance(gm, SparseSSPGroupMatrix) for gm in model._dm.group_matrices)
    return model, y, exposure, offset


def _orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    """Draw a deterministic orthogonal matrix with positive diagonal convention."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _equivalent_ssp_dm(
    dm: DesignMatrix,
    exposure: np.ndarray,
    *,
    basis_lambda: float | None = None,
    rotate_seed: int | None = None,
) -> DesignMatrix:
    """Construct an equivalent DesignMatrix under another valid SSP basis."""
    new_gms = []
    for idx, gm in enumerate(dm.group_matrices):
        if not isinstance(gm, SparseSSPGroupMatrix):
            raise TypeError(f"Expected SparseSSPGroupMatrix, got {type(gm).__name__}")

        if basis_lambda is None:
            r_inv_new = gm.R_inv.copy()
        else:
            if gm.projection is not None:
                projection = gm.projection
                omega_proj = projection.T @ gm.omega @ projection
                r_inv_local = compute_projected_R_inv(
                    gm.B,
                    projection,
                    omega_proj,
                    exposure,
                    basis_lambda,
                )
                r_inv_new = projection @ r_inv_local
            else:
                r_inv_new = compute_R_inv(gm.B, gm.omega, exposure, basis_lambda)

        if rotate_seed is not None:
            q = _orthogonal_matrix(r_inv_new.shape[1], rotate_seed + idx)
            r_inv_new = r_inv_new @ q

        gm_new = SparseSSPGroupMatrix(gm.B, r_inv_new)
        gm_new.omega = gm.omega
        gm_new.projection = gm.projection
        new_gms.append(gm_new)

    return DesignMatrix(new_gms, dm.n, dm.p)


def _reml_metadata(dm: DesignMatrix, groups: list) -> tuple[list, dict[str, float], dict]:
    """Build REML metadata for a given SSP design matrix."""
    reml_groups = []
    penalty_ranks = {}
    for i, (gm, g) in enumerate(zip(dm.group_matrices, groups, strict=False)):
        omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        eigvals = np.linalg.eigvalsh(omega_ssp)
        threshold = 1e-8 * max(eigvals.max(), 1e-12)
        penalty_ranks[g.name] = float(np.sum(eigvals > threshold))
        reml_groups.append((i, g))
    penalty_caches = build_penalty_caches(dm.group_matrices, groups, reml_groups)
    return reml_groups, penalty_ranks, penalty_caches


class TestSSPAudit:
    def test_direct_irls_fit_is_invariant_under_ssp_rotation(self):
        """Equivalent orthogonal SSP bases should give the same fitted spline."""
        model, y, exposure, offset = _setup_poisson_spline_model()
        dm_rot = _equivalent_ssp_dm(model._dm, exposure, rotate_seed=77)
        lambdas = {"x1": 0.7, "x2": 2.5}

        result_ref, _, _ = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
        )
        result_rot, _, _ = fit_irls_direct(
            X=dm_rot,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
        )

        eta_ref = model._dm.matvec(result_ref.beta) + result_ref.intercept + offset
        eta_rot = dm_rot.matvec(result_rot.beta) + result_rot.intercept + offset

        np.testing.assert_allclose(eta_ref, eta_rot, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(result_ref.deviance, result_rot.deviance, rtol=1e-10)

        for gm_ref, gm_rot, g in zip(
            model._dm.group_matrices, dm_rot.group_matrices, model._groups, strict=False
        ):
            beta_ref_raw = gm_ref.R_inv @ result_ref.beta[g.sl]
            beta_rot_raw = gm_rot.R_inv @ result_rot.beta[g.sl]
            np.testing.assert_allclose(beta_ref_raw, beta_rot_raw, rtol=1e-8, atol=1e-8)

    def test_reml_quantities_are_invariant_to_ssp_preconditioner_choice(self):
        """REML objective pieces should not care which valid SSP basis is used."""
        model, y, exposure, offset = _setup_poisson_spline_model()
        dm_alt = _equivalent_ssp_dm(model._dm, exposure, basis_lambda=8.0)
        lambdas = {"x1": 1.3, "x2": 0.4}

        result_ref, inv_ref, xtwx_ref = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
        )
        result_alt, inv_alt, xtwx_alt = fit_irls_direct(
            X=dm_alt,
            y=y,
            weights=exposure,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
        )

        reml_groups_ref, penalty_ranks_ref, penalty_caches_ref = _reml_metadata(
            model._dm, model._groups
        )
        reml_groups_alt, penalty_ranks_alt, penalty_caches_alt = _reml_metadata(
            dm_alt, model._groups
        )

        obj_ref = reml_laml_objective(
            model._dm,
            model._distribution,
            model._link,
            model._groups,
            y,
            result_ref,
            lambdas,
            exposure,
            offset,
            XtWX=xtwx_ref,
            penalty_caches=penalty_caches_ref,
        )
        obj_alt = reml_laml_objective(
            dm_alt,
            model._distribution,
            model._link,
            model._groups,
            y,
            result_alt,
            lambdas,
            exposure,
            offset,
            XtWX=xtwx_alt,
            penalty_caches=penalty_caches_alt,
        )

        grad_ref = reml_direct_gradient(
            model._dm.group_matrices,
            result_ref,
            inv_ref,
            lambdas,
            reml_groups_ref,
            penalty_ranks_ref,
        )
        grad_alt = reml_direct_gradient(
            dm_alt.group_matrices,
            result_alt,
            inv_alt,
            lambdas,
            reml_groups_alt,
            penalty_ranks_alt,
        )

        hess_ref = reml_direct_hessian(
            model._dm.group_matrices,
            model._distribution,
            inv_ref,
            lambdas,
            reml_groups_ref,
            grad_ref,
            penalty_ranks_ref,
            penalty_caches=penalty_caches_ref,
            pirls_result=result_ref,
            n_obs=len(y),
        )
        hess_alt = reml_direct_hessian(
            dm_alt.group_matrices,
            model._distribution,
            inv_alt,
            lambdas,
            reml_groups_alt,
            grad_alt,
            penalty_ranks_alt,
            penalty_caches=penalty_caches_alt,
            pirls_result=result_alt,
            n_obs=len(y),
        )

        np.testing.assert_allclose(obj_ref, obj_alt, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(grad_ref, grad_alt, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hess_ref, hess_alt, rtol=1e-7, atol=1e-8)
