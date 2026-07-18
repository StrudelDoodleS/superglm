"""Covariance and inference-state computations for fitted models."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR
from superglm.group_matrix import DesignMatrix
from superglm.model.fit_state import fitted_lambda2, fitted_penalty
from superglm.solvers.rank import (
    decompose_factor,
    decompose_gram,
    diagonal_of_square,
    needs_factor_certification,
    selected_group_name_set,
)
from superglm.types import GroupSlice


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

    feature_covariance = XtWX_inv_aug[1:, 1:]
    original_cross = XtWX_inv_aug[0, 1:].copy()
    shifted_cross = original_cross + intercept_shift @ feature_covariance
    result = np.array(XtWX_inv_aug, dtype=np.float64, copy=True)
    result[0, 0] = (
        XtWX_inv_aug[0, 0]
        + 2.0 * float(intercept_shift @ original_cross)
        + float(intercept_shift @ feature_covariance @ intercept_shift)
    )
    result[0, 1:] = shifted_cross
    result[1:, 0] = shifted_cross
    return result


def _grouped_active_state(model, selected_names: set[str]):
    """Return a compact grouped design and its original selected columns."""
    active_groups: list[GroupSlice] = []
    active_matrices = []
    rebuilt_columns: list[int] = []
    col = 0
    for gm, group in zip(model._dm.group_matrices, model._groups, strict=True):
        if group.name not in selected_names:
            continue
        active_matrices.append(gm)
        rebuilt_columns.extend(range(group.start, group.end))
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
    return (
        DesignMatrix(active_matrices, n=model._dm.n, p=col),
        active_groups,
        np.asarray(rebuilt_columns, dtype=np.intp),
    )


def _rank_active_state(model, rank_info):
    """Return the explicitly selected grouped design in rank-info order."""
    design, active_groups, rebuilt_columns = _grouped_active_state(
        model,
        set(rank_info.selected_group_names),
    )
    if not np.array_equal(
        rebuilt_columns,
        np.asarray(rank_info.selected_columns, dtype=np.intp),
    ):
        raise ValueError("rank metadata selected columns do not match active groups")
    return design, active_groups


def _legacy_active_state(model, solver, W: NDArray):
    """Rebuild old fit inference with grouped, centered coefficient algebra."""
    from superglm.inference.covariance import _active_penalty_matrix
    from superglm.solvers.centered_system import (
        build_centered_system,
        grouped_augmented_factor,
        grouped_weighted_factor,
    )

    selected_names = selected_group_name_set(
        solver,
        model._groups,
        penalty=fitted_penalty(model),
    )
    design, active_groups, _ = _grouped_active_state(model, selected_names)
    lam2 = fitted_lambda2(model)
    curvature = np.array(
        _active_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            active_groups,
            lam2,
            reml_penalties=getattr(model, "_reml_penalties", None),
        ),
        dtype=np.float64,
        copy=True,
    )
    from superglm.solvers.pirls import _add_selection_local_curvature

    original_by_name = {group.name: group for group in model._groups}
    original_active_groups = [original_by_name[group.name] for group in active_groups]
    _add_selection_local_curvature(
        curvature=curvature,
        penalty=fitted_penalty(model),
        beta=np.asarray(solver.beta, dtype=np.float64),
        original_groups=original_active_groups,
        active_groups=active_groups,
    )
    if design.p == 0:
        sum_w = float(np.sum(W))
        augmented = np.array([[1.0 / sum_w]]) if sum_w > 0.0 else np.array([[0.0]])
        data_rank = decompose_gram(np.empty((0, 0)))
        return (
            design,
            active_groups,
            np.empty((0, 0)),
            _public_augmented_covariance(model, augmented, active_groups),
            np.empty((0, 0)),
            np.empty((0, 0)),
            data_rank,
        )

    centered = build_centered_system(
        dm=design,
        W=W,
        z_off=np.zeros(design.n),
        penalty=curvature,
    )
    raw_gram, xtw1, _, _ = centered.raw_weighted_moments()
    coefficient_rank = decompose_gram(raw_gram + curvature)
    data_rank = decompose_gram(centered.data_gram)
    if needs_factor_certification(data_rank):
        certified = decompose_factor(grouped_weighted_factor(design, W, center=centered.mean_x))
        if certified.rank != data_rank.rank:
            data_rank = certified
    if needs_factor_certification(coefficient_rank):
        certified = decompose_factor(grouped_augmented_factor(design, W, curvature))
        if certified.rank != coefficient_rank.rank:
            coefficient_rank = certified
    if not np.any(curvature):
        profile_rank = data_rank
    else:
        profile_rank = decompose_gram(centered.data_gram + curvature)
        if needs_factor_certification(profile_rank):
            certified = decompose_factor(
                grouped_augmented_factor(
                    design,
                    W,
                    curvature,
                    center=centered.mean_x,
                )
            )
            if certified.rank != profile_rank.rank:
                profile_rank = certified
    coefficient_inverse = coefficient_rank.pseudo_inverse()
    profile_inverse = profile_rank.pseudo_inverse()
    mean_x = xtw1 / centered.sum_w
    intercept_cross = -(profile_inverse @ mean_x)
    augmented = np.empty((design.p + 1, design.p + 1), dtype=np.float64)
    augmented[0, 0] = 1.0 / centered.sum_w + float(mean_x @ profile_inverse @ mean_x)
    augmented[0, 1:] = intercept_cross
    augmented[1:, 0] = intercept_cross
    augmented[1:, 1:] = profile_inverse
    augmented = _public_augmented_covariance(model, augmented, active_groups)
    return (
        design,
        active_groups,
        coefficient_inverse,
        augmented,
        centered.data_gram,
        profile_inverse,
        data_rank,
    )


def _rank_centered_data_gram(X_active: DesignMatrix, W: NDArray) -> NDArray:
    """Build the selected centered Gram without a full observation-row matrix."""
    from superglm.solvers.centered_system import build_centered_system

    p_active = X_active.p
    if p_active == 0:
        return np.empty((0, 0))
    return build_centered_system(
        dm=X_active,
        W=W,
        z_off=np.zeros(len(W)),
        penalty=np.zeros((p_active, p_active)),
    ).data_gram


def _rank_edf1(rank_info, data_gram: NDArray, edf: NDArray) -> NDArray:
    """Alternative EDF in the decomposition's retained coefficient space."""
    import scipy.linalg

    decomposition = rank_info.augmented
    if decomposition.rank == 0:
        return np.zeros_like(edf)

    if decomposition.cholesky_factor is not None:
        active = decomposition.active_columns
        scale = decomposition.column_scale[active]
        retained_gram = data_gram[np.ix_(active, active)] / np.outer(scale, scale)
        retained_inverse = scipy.linalg.cho_solve(
            (decomposition.cholesky_factor, True),
            np.eye(decomposition.rank),
            check_finite=False,
        )
        retained_F = retained_inverse @ retained_gram
        edf1 = 2.0 * edf
        edf1[active] -= diagonal_of_square(retained_F)
        return edf1

    F = decomposition.pseudo_inverse() @ data_gram
    return 2.0 * edf - diagonal_of_square(F)


def _rank_augmented_covariance(model, rank_info, active_groups):
    """Transform centered retained covariance to solver and public intercepts."""
    p_active = len(rank_info.selected_columns)
    feature_covariance = rank_info.augmented.pseudo_inverse()
    mean_x = rank_info.mean_x[rank_info.selected_columns]
    intercept_cross = -(feature_covariance @ mean_x)
    solver_covariance = np.empty((p_active + 1, p_active + 1), dtype=np.float64)
    solver_covariance[0, 0] = (1.0 / rank_info.sum_w if rank_info.sum_w > 0.0 else 0.0) + float(
        mean_x @ feature_covariance @ mean_x
    )
    solver_covariance[0, 1:] = intercept_cross
    solver_covariance[1:, 0] = intercept_cross
    solver_covariance[1:, 1:] = feature_covariance
    return _public_augmented_covariance(model, solver_covariance, active_groups)


def coef_covariance(model):
    """Phi-scaled Bayesian covariance for active coefficients."""
    solver = model._solver_pirls_result()
    scop_inference = getattr(solver, "scop_inference", None)
    if scop_inference is not None:
        if solver.rank_info is not None:
            _, active_groups = _rank_active_state(model, solver.rank_info)
            selected = np.asarray(solver.rank_info.selected_columns, dtype=np.intp)
        else:
            _, active_groups, selected = _grouped_active_state(
                model,
                {group.name for group in model._groups},
            )
        mapped_covariance = np.asarray(scop_inference.augmented_inverse)[1:, 1:]
        covariance = solver.phi * mapped_covariance[np.ix_(selected, selected)]
        return covariance, active_groups
    if solver.rank_info is not None:
        _, active_groups = _rank_active_state(model, solver.rank_info)
        covariance = solver.phi * solver.rank_info.augmented.pseudo_inverse()
        return covariance, active_groups
    W = _solver_space_working_weights(model)
    _, active_groups, _, _, _, profile_inverse, _ = _legacy_active_state(model, solver, W)
    return solver.phi * profile_inverse, active_groups


def fit_active_info(model):
    """Grouped active design, weights, and (X'WX+S)^{-1} from fit state."""
    solver = model._solver_pirls_result()
    W = _solver_space_working_weights(model)
    scop_inference = getattr(solver, "scop_inference", None)
    if scop_inference is not None:
        if solver.rank_info is not None:
            X_active, active_groups = _rank_active_state(model, solver.rank_info)
            selected = np.asarray(solver.rank_info.selected_columns, dtype=np.intp)
        else:
            X_active, active_groups, selected = _grouped_active_state(
                model,
                {group.name for group in model._groups},
            )
        inverse = np.asarray(scop_inference.coefficient_inverse)[np.ix_(selected, selected)]
        augmented_indices = np.concatenate(([0], selected + 1))
        augmented = np.asarray(scop_inference.augmented_inverse)[
            np.ix_(augmented_indices, augmented_indices)
        ]
        augmented = _public_augmented_covariance(model, augmented, active_groups)
        return X_active, W, inverse, augmented, active_groups
    if solver.rank_info is not None:
        X_active, active_groups = _rank_active_state(model, solver.rank_info)
        inverse = solver.rank_info.coefficient.pseudo_inverse()
        augmented = _rank_augmented_covariance(model, solver.rank_info, active_groups)
        return X_active, W, inverse, augmented, active_groups

    X_active, active_groups, inverse, augmented, _, _, _ = _legacy_active_state(
        model,
        solver,
        W,
    )
    return X_active, W, inverse, augmented, active_groups


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
    solver = model._solver_pirls_result()
    W = _solver_space_working_weights(model)
    scop_inference = getattr(solver, "scop_inference", None)
    if scop_inference is not None:
        rank_info = solver.rank_info
        if rank_info is not None:
            X_active, active_groups = _rank_active_state(model, rank_info)
            selected = np.asarray(rank_info.selected_columns, dtype=np.intp)
            coefficient_estimable = rank_info.coefficient_estimable()
        else:
            X_active, active_groups, selected = _grouped_active_state(
                model,
                {group.name for group in model._groups},
            )
            coefficient_estimable = np.ones(len(solver.beta), dtype=bool)
        inverse = np.asarray(scop_inference.coefficient_inverse)[np.ix_(selected, selected)]
        augmented_indices = np.concatenate(([0], selected + 1))
        augmented = np.asarray(scop_inference.augmented_inverse)[
            np.ix_(augmented_indices, augmented_indices)
        ]
        augmented = _public_augmented_covariance(model, augmented, active_groups)
        edf = np.asarray(scop_inference.feature_edf)[selected].copy()
        edf1 = np.asarray(scop_inference.feature_edf1)[selected].copy()
        if X_active.p == 0:
            R_a = np.empty((0, 0))
        else:
            data_gram = _rank_centered_data_gram(X_active, W)
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
            "group_edf_map": dict(scop_inference.group_edf),
            "coefficient_estimable": coefficient_estimable,
        }
    if solver.rank_info is not None:
        rank_info = solver.rank_info
        X_active, active_groups = _rank_active_state(model, rank_info)
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
                "coefficient_estimable": rank_info.coefficient_estimable(),
            }
        data_gram = _rank_centered_data_gram(X_active, W)
        edf = rank_info.feature_edf[rank_info.selected_columns].copy()
        edf1 = _rank_edf1(rank_info, data_gram, edf)
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
            "coefficient_estimable": rank_info.coefficient_estimable(),
        }

    X_active, active_groups, inverse, augmented, data_gram, profile_inverse, data_rank = (
        _legacy_active_state(model, solver, W)
    )
    if X_active.p == 0:
        return {
            "W": W,
            "XtWX_inv": inverse,
            "XtWX_inv_aug": augmented,
            "active_groups": active_groups,
            "R_a": np.empty((0, 0)),
            "edf": np.array([]),
            "edf1": np.array([]),
            "group_edf_map": {},
            "coefficient_estimable": np.zeros(len(solver.beta), dtype=bool),
        }

    F = profile_inverse @ data_gram
    edf = np.diag(F)
    edf1 = 2.0 * edf - diagonal_of_square(F)

    eigvals, eigvecs = np.linalg.eigh(0.5 * (data_gram + data_gram.T))
    eigvals = np.maximum(eigvals, 0.0)
    R_a = (eigvecs * np.sqrt(eigvals)).T

    group_edf_map: dict[str, float] = {}
    for ag in active_groups:
        group_edf_map[ag.name] = float(np.sum(edf[ag.sl]))
    coefficient_estimable = np.zeros(len(solver.beta), dtype=bool)
    active_estimable = data_rank.coefficient_estimable()
    original_by_name = {group.name: group for group in model._groups}
    for active_group in active_groups:
        coefficient_estimable[original_by_name[active_group.name].sl] = active_estimable[
            active_group.sl
        ]

    return {
        "W": W,
        "XtWX_inv": inverse,
        "XtWX_inv_aug": augmented,
        "active_groups": active_groups,
        "R_a": R_a,
        "edf": edf,
        "edf1": edf1,
        "group_edf_map": group_edf_map,
        "coefficient_estimable": coefficient_estimable,
    }


def group_edf(model) -> dict[str, float] | None:
    """Per-group effective degrees of freedom via F = (X'WX+S)^{-1} X'WX."""
    if model._dm is None or model._result is None:
        return None
    return cast(dict[str, float] | None, model._fit_inference_info["group_edf_map"])
