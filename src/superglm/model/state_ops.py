"""Covariance and inference-state computations for fitted models."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR
from superglm.inference._term_covariance import compute_coef_covariance


def _build_S_from_penalties(model, lam2) -> NDArray | None:
    """Build full penalty matrix from model._reml_penalties if available.

    Returns None if model has no stored reml_penalties (non-REML fit or
    single-penalty where the legacy path is equivalent).
    """
    penalties = getattr(model, "_reml_penalties", None)
    if penalties is None:
        return None
    from superglm.reml.penalty_algebra import build_penalty_matrix

    return build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        lam2,
        model._dm.p,
        reml_penalties=penalties,
    )


def coef_covariance(model):
    """Phi-scaled Bayesian covariance for active coefficients."""
    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    S_full = _build_S_from_penalties(model, lam2)
    return compute_coef_covariance(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model.result,
        model._fit_weights,
        model._fit_offset,
        lam2,
        S_override=S_full,
    )


def fit_active_info(model):
    """Active design columns, weights, and (X'WX+S)^{-1} from fit state."""
    from superglm.distributions import clip_mu
    from superglm.inference.covariance import _penalised_xtwx_inv
    from superglm.links import stabilize_eta

    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    V = model._distribution.variance(mu)
    dmu_deta = model._link.deriv_inverse(eta)
    W = model._fit_weights * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    S_full = _build_S_from_penalties(model, lam2)
    X_a, XtWX_inv, XtWX_inv_aug, active_groups, _ = _penalised_xtwx_inv(
        model.result.beta,
        W,
        model._dm.group_matrices,
        model._groups,
        lam2,
        S_override=S_full,
    )
    return X_a, W, XtWX_inv, XtWX_inv_aug, active_groups


def fit_inference_info(model):
    """All coefficient-space inference quantities for model.summary().

    Self-contained: computes working weights W, then uses the gram path
    (per-group gram blocks + p³ inversion) instead of materialising the
    full n×p active design matrix.  This makes model.summary() O(n + p³)
    instead of O(n·p²).

    Returns a dict with:
        W : (n,) working weights
        XtWX_inv : (p_active, p_active) = (X'WX + S)^{-1}
        XtWX_inv_aug : (p_active+1, p_active+1) augmented inverse incl. intercept
        active_groups : list of GroupSlice re-indexed to active columns
        R_a : (p_active, p_active) upper-triangular Cholesky factor of X'WX
        edf : per-coefficient EDF vector
        edf1 : Wood's alternative EDF vector
        group_edf_map : per-group summed EDF dict
    """
    import scipy.linalg

    from superglm.distributions import clip_mu
    from superglm.inference.covariance import _penalised_xtwx_inv_gram
    from superglm.links import stabilize_eta

    # Compute working weights — one O(n) pass
    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    V = model._distribution.variance(mu)
    dmu_deta = model._link.deriv_inverse(eta)
    W = model._fit_weights * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    S_full = _build_S_from_penalties(model, lam2)

    # Gram path: per-group gram + cross-gram blocks, then invert.
    # O(n·p_g² per block + p³) — avoids the full n×p QR.
    # Returns XtWX and S directly so we don't need to recover them
    # from the (possibly truncated) pseudo-inverse.
    XtWX_inv, XtWX_inv_aug, active_groups, XtWX, S = _penalised_xtwx_inv_gram(
        model.result.beta,
        W,
        model._dm.group_matrices,
        model._groups,
        lam2,
        S_override=S_full,
    )

    p_a = XtWX_inv.shape[0]
    if p_a == 0:
        return {
            "W": W,
            "XtWX_inv": XtWX_inv,
            "XtWX_inv_aug": XtWX_inv_aug,
            "active_groups": active_groups,
            "R_a": np.empty((0, 0)),
            "edf": np.array([]),
            "edf1": np.array([]),
            "group_edf_map": {},
        }

    # EDF: F = (X'WX+S)^{-1} X'WX — use XtWX directly from the gram path,
    # which is correct even when XtWX_inv is a truncated pseudo-inverse.
    F = XtWX_inv @ XtWX
    edf = np.diag(F)
    edf1 = 2.0 * edf - np.sum(F * F, axis=1)

    # R factor via Cholesky of X'WX (O(p³) instead of O(n·p²) QR)
    try:
        R_a = scipy.linalg.cholesky(XtWX, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        # Near-singular: eigendecompose and build pseudo-R
        eigvals, eigvecs = np.linalg.eigh(XtWX)
        eigvals = np.maximum(eigvals, 0.0)
        R_a = (eigvecs * np.sqrt(eigvals)).T  # p×p, R'R = XtWX

    group_edf_map: dict[str, float] = {}
    for ag in active_groups:
        group_edf_map[ag.name] = float(np.sum(edf[ag.sl]))

    return {
        "W": W,
        "XtWX_inv": XtWX_inv,
        "XtWX_inv_aug": XtWX_inv_aug,
        "active_groups": active_groups,
        "R_a": R_a,
        "edf": edf,
        "edf1": edf1,
        "group_edf_map": group_edf_map,
    }


def group_edf(model) -> dict[str, float] | None:
    """Per-group effective degrees of freedom via F = (X'WX+S)^{-1} X'WX."""
    if model._dm is None or model._result is None:
        return None
    return cast(dict[str, float] | None, model._fit_inference_info["group_edf_map"])
