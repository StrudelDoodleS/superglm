"""Penalised (X'WX + S)^{-1} covariance utilities.

Extracted from ``metrics.py`` to break the inference <-> metrics circular
dependency.  These functions depend only on numpy, scipy, group_matrix, and
types — they have no dependency on inference results or metrics classes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    _block_xtwx,
)
from superglm.types import GroupSlice


def _second_diff_penalty(p: int) -> NDArray:
    """Second-difference penalty matrix D2'D2 for p basis functions."""
    D2 = np.diff(np.eye(p), n=2, axis=0)
    return D2.T @ D2


def _penalised_xtwx_inv(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, list[GroupSlice], list]:
    """Compute (X'WX + S)^{-1} via augmented QR + truncated SVD.

    Shared by ``model._coef_covariance`` and ``ModelMetrics._active_info``.

    Parameters
    ----------
    beta : (p,) array — full coefficient vector.
    W : (n,) array — working weights (dmu_deta² / V, sample_weight-scaled).
    group_matrices : list of GroupMatrix — design matrices per group.
    groups : list of GroupSlice — slices into beta for each group.
    lambda2 : float or dict[str, float]
        Smoothing penalty weight. A scalar applies to all groups; a dict
        maps group names to per-group lambdas (REML).

    Returns
    -------
    X_a : (n, p_active) dense active design matrix.
    XtWX_S_inv : (p_active, p_active) inverse of (X'WX + S).
    XtWX_S_inv_aug : (p_active+1, p_active+1) inverse of augmented system
        including intercept row/column. Element [0,0] is intercept variance,
        [1:,1:] are feature marginal variances (accounting for intercept).
    active_groups : list of GroupSlice re-indexed to X_a columns.
    active_gms : list of GroupMatrix for active groups.
    """
    active_cols: list[NDArray] = []
    active_groups_out: list[GroupSlice] = []
    active_gms: list = []
    active_group_names: list[str] = []
    col = 0
    for gm, g in zip(group_matrices, groups):
        if np.linalg.norm(beta[g.sl]) > 1e-12:
            arr = gm.toarray()
            active_cols.append(arr)
            active_gms.append(gm)
            active_group_names.append(g.name)
            p_g = arr.shape[1]
            active_groups_out.append(
                GroupSlice(
                    name=g.name,
                    start=col,
                    end=col + p_g,
                    weight=g.weight,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                    constraints=g.constraints,
                    monotone_engine=g.monotone_engine,
                    scop_reparameterization=g.scop_reparameterization,
                )
            )
            col += p_g

    if not active_cols:
        n = len(W)
        # Augmented inverse is 1×1: just the intercept variance
        w_sum = float(np.sum(W))
        aug_inv = np.array([[1.0 / w_sum]]) if w_sum > 0 else np.array([[0.0]])
        return np.empty((n, 0)), np.empty((0, 0)), aug_inv, [], []

    X_a = np.hstack(active_cols)
    p_a = X_a.shape[1]

    # Build sqrt(S) factor: L such that L'L = S (block-diagonal penalty)
    # Unpenalized groups (e.g. select=True null-space) get no penalty contribution.
    if S_override is not None:
        # S_override is full (p x p) — slice to active columns, then sqrt
        active_idx = []
        for ag, gname in zip(active_groups_out, active_group_names):
            orig_g = next(g for g in groups if g.name == gname)
            active_idx.extend(range(orig_g.start, orig_g.end))
        active_idx_arr = np.array(active_idx)
        S_active = S_override[np.ix_(active_idx_arr, active_idx_arr)]
        eigvals_s, eigvecs_s = np.linalg.eigh(S_active)
        eigvals_s = np.maximum(eigvals_s, 0.0)
        S_rows = np.sqrt(eigvals_s)[:, None] * eigvecs_s.T  # sqrt(S)
    else:
        S_rows = np.zeros((p_a, p_a))
        for gm_orig, ag, gname in zip(active_gms, active_groups_out, active_group_names):
            if not ag.penalized:
                continue

            if isinstance(lambda2, dict):
                lam_g = lambda2.get(gname, 0.0)
            else:
                lam_g = lambda2

            if lam_g == 0:
                continue

            if isinstance(gm_orig, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                R_inv = gm_orig.R_inv
                omega = gm_orig.omega
                if omega is None:
                    p_b = R_inv.shape[0]
                    omega = _second_diff_penalty(p_b)
                S_g = lam_g * R_inv.T @ omega @ R_inv
            elif ag.scop_reparameterization is not None:
                S_g = lam_g * ag.scop_reparameterization.penalty_matrix()
            else:
                continue

            eigvals_g, eigvecs_g = np.linalg.eigh(S_g)
            eigvals_g = np.maximum(eigvals_g, 0.0)
            L_g = np.sqrt(eigvals_g)[:, None] * eigvecs_g.T
            S_rows[ag.sl, ag.sl] = L_g

    # Augmented QR: [sqrt(W)*X; sqrt(S)] → R'R = X'WX + S
    A = np.vstack([X_a * np.sqrt(W)[:, None], S_rows])
    _, R = np.linalg.qr(A, mode="reduced")

    # Truncated SVD for numerical stability.
    # Regularize truncated directions so SEs are large-but-finite.
    _, s_R, Vh_R = np.linalg.svd(R, full_matrices=False)
    threshold = 1e-6 * s_R[0]
    regularized = 1.0 / threshold**2
    inv_s2 = np.where(s_R > threshold, 1.0 / s_R**2, regularized)
    XtWX_S_inv = (Vh_R.T * inv_s2[None, :]) @ Vh_R

    # Augmented (p+1)×(p+1) inverse including intercept row/column.
    # The augmented Fisher information is:
    #   F_aug = [[sum(W),  X'W1], [X'W1,  X'WX + S]]
    # where X'W1 = X_a' @ W (cross-product of features with intercept).
    # This is needed for correct SEs that account for intercept estimation.
    sqrtW = np.sqrt(W)
    X_aug = np.hstack([np.ones((len(W), 1)), X_a])
    S_aug_rows = np.zeros((p_a + 1, p_a + 1))
    S_aug_rows[1:, 1:] = S_rows  # no penalty on intercept
    A_aug = np.vstack([X_aug * sqrtW[:, None], S_aug_rows])
    _, R_aug = np.linalg.qr(A_aug, mode="reduced")
    _, s_aug, Vh_aug = np.linalg.svd(R_aug, full_matrices=False)
    threshold_aug = 1e-6 * s_aug[0]
    regularized_aug = 1.0 / threshold_aug**2
    inv_s2_aug = np.where(s_aug > threshold_aug, 1.0 / s_aug**2, regularized_aug)
    XtWX_S_inv_aug = (Vh_aug.T * inv_s2_aug[None, :]) @ Vh_aug

    return X_a, XtWX_S_inv, XtWX_S_inv_aug, active_groups_out, active_gms


def _penalised_xtwx_inv_gram(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
) -> tuple[NDArray, NDArray, list[GroupSlice], NDArray | None, NDArray | None]:
    """Fast (X'WX + S)^{-1} via per-group gram matrices.

    Same result as ``_penalised_xtwx_inv`` but avoids forming the dense
    (n, p) matrix. Computes X'WX block-by-block using the group gram
    kernels, then inverts the (p_active, p_active) system directly.
    Cost is O(n · sum(p_g²)) + O(p³) instead of O(n · p²).

    Does NOT return X_a (not needed for REML). For leverage/hat matrix
    diagnostics, use ``_penalised_xtwx_inv`` instead.

    Returns
    -------
    XtWX_S_inv : (p_active, p_active) inverse of (X'WX + S).
    XtWX_S_inv_aug : (p_active+1, p_active+1) inverse of augmented system
        including intercept row/column.
    active_groups : list of GroupSlice re-indexed to active columns.
    XtWX : (p_active, p_active) X'WX gram matrix, or None if p_active == 0.
    S : (p_active, p_active) penalty matrix, or None if p_active == 0.
    """
    # Identify active groups
    active_gms: list = []
    active_groups_out: list[GroupSlice] = []
    active_group_names: list[str] = []
    col = 0
    for gm, g in zip(group_matrices, groups):
        if np.linalg.norm(beta[g.sl]) > 1e-12:
            active_gms.append(gm)
            active_group_names.append(g.name)
            p_g = gm.shape[1]
            active_groups_out.append(
                GroupSlice(
                    name=g.name,
                    start=col,
                    end=col + p_g,
                    weight=g.weight,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                    constraints=g.constraints,
                    monotone_engine=g.monotone_engine,
                    scop_reparameterization=g.scop_reparameterization,
                )
            )
            col += p_g

    p_a = col
    if p_a == 0:
        w_sum = float(np.sum(W))
        aug_inv = np.array([[1.0 / w_sum]]) if w_sum > 0 else np.array([[0.0]])
        return np.empty((0, 0)), aug_inv, [], None, None

    # Build X'WX block-by-block: gram for diagonal, cross_gram for off-diagonal.
    # For discretized groups this is O(n_bins) per block instead of O(n·p²).
    XtWX = _block_xtwx(active_gms, active_groups_out, W)

    # Add penalty S (same logic as _penalised_xtwx_inv)
    if S_override is not None:
        active_idx = []
        for ag, gname in zip(active_groups_out, active_group_names):
            orig_g = next(g for g in groups if g.name == gname)
            active_idx.extend(range(orig_g.start, orig_g.end))
        active_idx_arr = np.array(active_idx)
        S = S_override[np.ix_(active_idx_arr, active_idx_arr)]
    else:
        S = np.zeros((p_a, p_a))
        for gm_orig, ag, gname in zip(active_gms, active_groups_out, active_group_names):
            if not ag.penalized:
                continue

            if isinstance(lambda2, dict):
                lam_g = lambda2.get(gname, 0.0)
            else:
                lam_g = lambda2

            if lam_g == 0:
                continue

            if isinstance(gm_orig, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                R_inv = gm_orig.R_inv
                omega = gm_orig.omega
                if omega is None:
                    p_b = R_inv.shape[0]
                    omega = _second_diff_penalty(p_b)
                S[ag.sl, ag.sl] = lam_g * R_inv.T @ omega @ R_inv
            elif ag.scop_reparameterization is not None:
                S_scop = ag.scop_reparameterization.penalty_matrix()
                S[ag.sl, ag.sl] = lam_g * S_scop

    # Invert (X'WX + S) via eigendecomposition.
    # Match the dense QR/SVD path: there we truncate singular values at
    # ``rtol * s_max``. Since ``eigvals(M) = s**2``, the equivalent cutoff
    # on the eigenvalue scale is ``rtol**2 * eig_max``.
    # Truncated directions get a large regularized inverse (1/threshold)
    # so that SEs are finite-but-huge rather than zero — correctly
    # signaling "undetermined" for near-separated coefficients.
    M = XtWX + S
    eigvals, eigvecs = np.linalg.eigh(M)
    threshold = (1e-6**2) * max(eigvals.max(), 1e-12)
    regularized = 1.0 / threshold
    with np.errstate(divide="ignore"):
        inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, regularized)
    XtWX_S_inv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T

    # Augmented (p+1)×(p+1) inverse including intercept row/column.
    # Build X'W1 via per-group rmatvec (avoids materializing dense X_a).
    XtW1 = np.empty(p_a)
    for gm, ag in zip(active_gms, active_groups_out):
        XtW1[ag.sl] = gm.rmatvec(W)
    sum_W = float(np.sum(W))

    M_aug = np.empty((p_a + 1, p_a + 1))
    M_aug[0, 0] = sum_W
    M_aug[0, 1:] = XtW1
    M_aug[1:, 0] = XtW1
    M_aug[1:, 1:] = M  # XtWX + S
    eigvals_aug, eigvecs_aug = np.linalg.eigh(M_aug)
    threshold_aug = (1e-6**2) * max(eigvals_aug.max(), 1e-12)
    regularized_aug = 1.0 / threshold_aug
    with np.errstate(divide="ignore"):
        inv_eigvals_aug = np.where(eigvals_aug > threshold_aug, 1.0 / eigvals_aug, regularized_aug)
    XtWX_S_inv_aug = (eigvecs_aug * inv_eigvals_aug[None, :]) @ eigvecs_aug.T

    return XtWX_S_inv, XtWX_S_inv_aug, active_groups_out, XtWX, S
