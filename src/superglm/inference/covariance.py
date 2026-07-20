"""Penalised (X'WX + S)^{-1} covariance utilities.

Extracted from ``metrics.py`` to break the inference <-> metrics circular
dependency.  These functions depend only on numpy, scipy, group_matrix, and
types — they have no dependency on inference results or metrics classes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.solvers.rank import decompose_factor, decompose_gram
from superglm.types import GroupSlice


def _second_diff_penalty(p: int) -> NDArray:
    """Second-difference penalty matrix D2'D2 for p basis functions."""
    D2 = np.diff(np.eye(p), n=2, axis=0)
    return D2.T @ D2


def _active_penalty_matrix(
    group_matrices: list,
    groups: list[GroupSlice],
    active_groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    *,
    S_override: NDArray | None = None,
    reml_penalties: list | None = None,
) -> NDArray:
    """Build ``S`` directly in compact active coordinates."""
    p_active = sum(group.size for group in active_groups)
    if p_active == 0:
        return np.empty((0, 0), dtype=np.float64)

    active_by_name = {group.name: group for group in active_groups}
    if S_override is not None:
        selected_columns = np.asarray(
            [
                column
                for group in groups
                if group.name in active_by_name
                for column in range(group.start, group.end)
            ],
            dtype=np.intp,
        )
        return np.asarray(S_override[np.ix_(selected_columns, selected_columns)], dtype=np.float64)

    S = np.zeros((p_active, p_active), dtype=np.float64)
    if reml_penalties is not None:
        represented_group_indices = {component.group_index for component in reml_penalties}
        for component in reml_penalties:
            active_group = active_by_name.get(component.group_name)
            if active_group is None:
                continue
            gm = group_matrices[component.group_index]
            lam = lambda2[component.name] if isinstance(lambda2, dict) else lambda2
            if lam == 0:
                continue
            omega = (
                component.omega_ssp
                if component.omega_ssp is not None
                else gm.R_inv.T @ component.omega_raw @ gm.R_inv
            )
            S[active_group.sl, active_group.sl] += lam * omega

        for group_index, group in enumerate(groups):
            if group_index in represented_group_indices:
                continue
            active_group = active_by_name.get(group.name)
            if active_group is None or group.scop_reparameterization is None or not group.penalized:
                continue
            lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam > 0:
                S[active_group.sl, active_group.sl] += (
                    lam * group.scop_reparameterization.penalty_matrix()
                )
        return S

    for gm, group in zip(group_matrices, groups, strict=True):
        active_group = active_by_name.get(group.name)
        if active_group is None or not group.penalized:
            continue
        lam = lambda2.get(group.name, 0.0) if isinstance(lambda2, dict) else lambda2
        if lam == 0:
            continue
        if isinstance(
            gm,
            SparseSSPGroupMatrix
            | SplineCategoricalGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix,
        ):
            omega = gm.omega
            if omega is None:
                omega = _second_diff_penalty(gm.R_inv.shape[0])
            S[active_group.sl, active_group.sl] = lam * gm.R_inv.T @ omega @ gm.R_inv
        elif group.scop_reparameterization is not None:
            S[active_group.sl, active_group.sl] = (
                lam * group.scop_reparameterization.penalty_matrix()
            )
    return S


def _penalised_xtwx_inv(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
    selected_group_names: set[str] | None = None,
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
        if (
            g.name in selected_group_names
            if selected_group_names is not None
            else np.linalg.norm(beta[g.sl]) > 1e-12
        ):
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

            if isinstance(
                gm_orig,
                SparseSSPGroupMatrix
                | SplineCategoricalGroupMatrix
                | DiscretizedSplineCategoricalGroupMatrix
                | DiscretizedSSPGroupMatrix,
            ):
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
    XtWX_S_inv = decompose_factor(A).pseudo_inverse()

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
    XtWX_S_inv_aug = decompose_factor(A_aug).pseudo_inverse()

    return X_a, XtWX_S_inv, XtWX_S_inv_aug, active_groups_out, active_gms


def _penalised_xtwx_inv_gram(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
    selected_group_names: set[str] | None = None,
    reml_penalties: list | None = None,
    compute_augmented: bool = True,
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
        if (
            g.name in selected_group_names
            if selected_group_names is not None
            else np.linalg.norm(beta[g.sl]) > 1e-12
        ):
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

    # The active group subset has its own solver coordinates, so use one
    # ephemeral plan and fuse X'W with its Gram when the augmented covariance
    # is requested.
    active_plan = MatrixExecutionPlan(active_gms, n=len(W))
    active_moments = active_plan.moments(W, include_xtw=compute_augmented)
    XtWX = active_moments.gram

    S = _active_penalty_matrix(
        group_matrices,
        groups,
        active_groups_out,
        lambda2,
        S_override=S_override,
        reml_penalties=reml_penalties,
    )

    M = XtWX + S
    XtWX_S_inv = decompose_gram(M).pseudo_inverse()

    if not compute_augmented:
        return XtWX_S_inv, np.empty((0, 0)), active_groups_out, XtWX, S

    # Augmented (p+1)×(p+1) inverse including intercept row/column.
    XtW1 = active_moments.xtw
    if XtW1 is None:  # pragma: no cover - guaranteed by compute_augmented
        raise RuntimeError("execution plan did not return X'W")
    sum_W = float(np.sum(W))

    M_aug = np.empty((p_a + 1, p_a + 1))
    M_aug[0, 0] = sum_W
    M_aug[0, 1:] = XtW1
    M_aug[1:, 0] = XtW1
    M_aug[1:, 1:] = M  # XtWX + S
    XtWX_S_inv_aug = decompose_gram(M_aug).pseudo_inverse()

    return XtWX_S_inv, XtWX_S_inv_aug, active_groups_out, XtWX, S
