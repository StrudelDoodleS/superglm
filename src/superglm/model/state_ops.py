"""Covariance and inference-state computations for fitted models."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR
from superglm.inference._term_covariance import compute_coef_covariance
from superglm.types import GroupSlice


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


def _solver_space_working_weights(model) -> NDArray:
    """Working weights computed against the solver-space fit state."""
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta

    solver = model._solver_pirls_result()
    eta = model._dm.matvec(solver.beta) + solver.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    V = model._distribution.variance(mu)
    dmu_deta = model._link.deriv_inverse(eta)
    return model._fit_weights * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)


def _public_augmented_covariance(model, XtWX_inv_aug: NDArray, active_groups) -> NDArray:
    """Map solver-space augmented covariance into the public runtime state."""
    runtime_state = getattr(model, "_runtime_canonical_state", None)
    if runtime_state is None:
        return XtWX_inv_aug

    p_active = XtWX_inv_aug.shape[0] - 1
    if p_active <= 0:
        return XtWX_inv_aug

    intercept_shift = np.zeros(p_active, dtype=np.float64)
    for term_state in runtime_state.get("terms", {}).values():
        if not term_state.get("applied_to_public_model", False):
            continue
        for group_state in term_state.get("groups", []):
            active_group = next(
                (group for group in active_groups if group.name == group_state["group_name"]),
                None,
            )
            if active_group is None:
                continue
            column_means = np.asarray(group_state["column_means"], dtype=np.float64).ravel()
            intercept_shift[active_group.sl] = column_means

    if not np.any(intercept_shift):
        return XtWX_inv_aug

    transform = np.eye(XtWX_inv_aug.shape[0], dtype=np.float64)
    transform[0, 1:] = intercept_shift
    return transform @ XtWX_inv_aug @ transform.T


def _rank_active_state(model, rank_info, W: NDArray):
    """Materialize the explicitly selected fit state in rank-info order."""
    selected_names = set(rank_info.selected_group_names)
    active_groups: list[GroupSlice] = []
    active_arrays: list[NDArray] = []
    col = 0
    for gm, group in zip(model._dm.group_matrices, model._groups, strict=True):
        if group.name not in selected_names:
            continue
        active_arrays.append(gm.toarray())
        active_groups.append(
            GroupSlice(
                name=group.name,
                start=col,
                end=col + group.size,
                weight=group.weight,
                penalized=group.penalized,
                feature_name=group.feature_name,
                subgroup_type=group.subgroup_type,
                constraints=group.constraints,
                monotone_engine=group.monotone_engine,
                scop_reparameterization=group.scop_reparameterization,
            )
        )
        col += group.size
    X_active = np.hstack(active_arrays) if active_arrays else np.empty((len(W), 0))
    if col != len(rank_info.selected_columns):
        raise ValueError("rank metadata selected width does not match active groups")
    return X_active, active_groups


def _rank_augmented_covariance(model, rank_info, active_groups):
    """Transform centered retained covariance to solver and public intercepts."""
    p_active = len(rank_info.selected_columns)
    centered = np.zeros((p_active + 1, p_active + 1))
    if rank_info.sum_w > 0.0:
        centered[0, 0] = 1.0 / rank_info.sum_w
    centered[1:, 1:] = rank_info.augmented.pseudo_inverse()
    transform = np.eye(p_active + 1)
    transform[0, 1:] = -rank_info.mean_x[rank_info.selected_columns]
    solver_covariance = transform @ centered @ transform.T
    return _public_augmented_covariance(model, solver_covariance, active_groups)


def coef_covariance(model):
    """Phi-scaled Bayesian covariance for active coefficients."""
    solver = model._solver_pirls_result()
    if solver.rank_info is not None:
        W = _solver_space_working_weights(model)
        _, active_groups = _rank_active_state(model, solver.rank_info, W)
        covariance = solver.phi * solver.rank_info.augmented.pseudo_inverse()
        return covariance, active_groups
    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    S_full = _build_S_from_penalties(model, lam2)
    return compute_coef_covariance(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model._solver_pirls_result(),
        model._fit_weights,
        model._fit_offset,
        lam2,
        S_override=S_full,
    )


def fit_active_info(model):
    """Active design columns, weights, and (X'WX+S)^{-1} from fit state."""
    from superglm.inference.covariance import _penalised_xtwx_inv

    solver = model._solver_pirls_result()
    W = _solver_space_working_weights(model)
    if solver.rank_info is not None:
        X_active, active_groups = _rank_active_state(model, solver.rank_info, W)
        inverse = solver.rank_info.coefficient.pseudo_inverse()
        augmented = _rank_augmented_covariance(model, solver.rank_info, active_groups)
        return X_active, W, inverse, augmented, active_groups

    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    S_full = _build_S_from_penalties(model, lam2)
    X_a, XtWX_inv, XtWX_inv_aug, active_groups, _ = _penalised_xtwx_inv(
        solver.beta,
        W,
        model._dm.group_matrices,
        model._groups,
        lam2,
        S_override=S_full,
    )
    XtWX_inv_aug = _public_augmented_covariance(model, XtWX_inv_aug, active_groups)
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

    from superglm.inference.covariance import _penalised_xtwx_inv_gram

    solver = model._solver_pirls_result()
    W = _solver_space_working_weights(model)
    if solver.rank_info is not None:
        rank_info = solver.rank_info
        X_active, active_groups = _rank_active_state(model, rank_info, W)
        retained_inverse = rank_info.augmented.pseudo_inverse()
        inverse = rank_info.coefficient.pseudo_inverse()
        augmented = _rank_augmented_covariance(model, rank_info, active_groups)
        if X_active.shape[1] == 0:
            return {
                "W": W,
                "XtWX_inv": inverse,
                "XtWX_inv_aug": augmented,
                "active_groups": active_groups,
                "R_a": np.empty((0, 0)),
                "edf": np.array([]),
                "edf1": np.array([]),
                "group_edf_map": dict(rank_info.group_edf),
            }
        X_centered = X_active - rank_info.mean_x[rank_info.selected_columns]
        data_gram = X_centered.T @ (W[:, None] * X_centered)
        F = retained_inverse @ data_gram
        edf = rank_info.feature_edf[rank_info.selected_columns].copy()
        edf1 = 2.0 * edf - np.sum(F * F, axis=1)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (data_gram + data_gram.T))
        eigvals = np.maximum(eigvals, 0.0)
        R_a = (eigvecs * np.sqrt(eigvals)).T
        return {
            "W": W,
            "XtWX_inv": inverse,
            "XtWX_inv_aug": augmented,
            "active_groups": active_groups,
            "R_a": R_a,
            "edf": edf,
            "edf1": edf1,
            "group_edf_map": dict(rank_info.group_edf),
        }

    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    S_full = _build_S_from_penalties(model, lam2)

    # Gram path: per-group gram + cross-gram blocks, then invert.
    # O(n·p_g² per block + p³) — avoids the full n×p QR.
    # Returns XtWX and S directly so we don't need to recover them
    # from the (possibly truncated) pseudo-inverse.
    XtWX_inv, XtWX_inv_aug, active_groups, XtWX, S = _penalised_xtwx_inv_gram(
        solver.beta,
        W,
        model._dm.group_matrices,
        model._groups,
        lam2,
        S_override=S_full,
    )
    XtWX_inv_aug = _public_augmented_covariance(model, XtWX_inv_aug, active_groups)

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
