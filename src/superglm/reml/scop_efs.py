"""SCOP-aware EFS REML optimizer.

Extends the standard Fellner-Schall EFS loop to handle SCOP monotone terms
alongside unconstrained SSP terms.

References
----------
Wood & Fasiolo (2017). A generalized Fellner-Schall method for smoothing
parameter optimization with application to shape constrained regression.
Biometrics 73(4), 1071-1081.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import _raw_centering_well_scaled
from superglm.distributions import _VARIANCE_FLOOR, Gamma, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import stabilize_eta
from superglm.reml.objective import reml_laml_objective
from superglm.reml.observed_geometry import classify_scop_reml_curvature
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    compute_logdet_s_derivatives,
)
from superglm.reml.result import REMLResult
from superglm.reml.scale import prepare_gamma_reml_scale_data
from superglm.reml.scop_geometry import (
    SCOPJointGeometry,
    SCOPModeScore,
    build_cached_scop_joint_geometry,
    build_observed_scop_joint_geometry,
    install_scop_postfit_inference,
    restrict_to_scop_resolved_range,
    scop_penalized_mode_score,
    scop_resolved_range_projector,
)
from superglm.solvers.centered_system import (
    grouped_augmented_factor,
    grouped_weighted_factor,
)
from superglm.solvers.dispersion import dispersion_likelihood_size
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    RankInfo,
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
)
from superglm.types import GroupSlice, PenaltyComponent

# These private thresholds intentionally mix units: absolute lambda scale for the
# floor guard, log-lambda-step scale for stability/plateau checks, and relative
# objective scale for outer-loop flatness checks.
_MULTI_SCOP_DISCRETE_LAMBDA_FLOOR = 1.0e-4
_MULTI_SCOP_DISCRETE_FLOOR_FACTOR = 1.05
_MULTI_SCOP_DISCRETE_LOG_STEP_TOL = 1.0e-3
_MULTI_SCOP_DISCRETE_MIN_STABLE_ITERS = 3
_MULTI_SCOP_DISCRETE_ACTIVE_PLATEAU_TOL = 5.0e-3
_MULTI_SCOP_DISCRETE_OBJ_REL_TOL = 1.0e-6
_SCOP_EFS_MAX_BACKTRACK_ATTEMPTS = 8
_SCOP_EFS_MAX_REFLECTED_ATTEMPTS = 4


@dataclass(frozen=True)
class _SCOPREMLFitContext:
    """Inputs shared by all coherent SCOP coefficient-mode evaluations."""

    dm: DesignMatrix
    distribution: Any
    link: Any
    groups: list[GroupSlice]
    y: NDArray
    sample_weight: NDArray
    offset_arr: NDArray
    pirls_tol: float
    max_pirls_iter: int
    reml_penalties: list[PenaltyComponent] | None
    convergence: str
    scop_joint: bool
    debug_recorder: Any
    likelihood_size: float
    gamma_scale_data: Any


@dataclass(frozen=True)
class _SCOPREMLMode:
    """One fitted coefficient mode and its lambda-coherent LAML geometry."""

    lambdas: dict[str, float]
    result: Any
    xtwx: NDArray
    centered_xtwx: NDArray
    fisher_mean_x: NDArray
    fisher_sum_w: float
    scop_states: dict[int, dict]
    penalty: NDArray
    penalty_components: list[PenaltyComponent]
    joint_geometry: SCOPJointGeometry
    hessian_inverse: NDArray
    evaluation: Any
    log_det_h: float
    hessian_rank: int
    curvature_source: str
    mode_score: SCOPModeScore

    @property
    def objective(self) -> float:
        return float(self.evaluation.value)


def _multi_scop_discrete_cleanup_enabled(*, discrete: bool, scop_term_count: int) -> bool:
    return bool(discrete and scop_term_count > 1)


def _multi_scop_discrete_cleanup_names(
    *,
    estimated_names: set[str],
    scop_states: dict[int, dict],
    scop_term_count: int,
) -> set[str]:
    """Return the estimated SCOP names eligible for the discrete cleanup path."""
    if not scop_states:
        return set()

    all_scop_discrete = all(st.get("bin_idx") is not None for st in scop_states.values())
    if not _multi_scop_discrete_cleanup_enabled(
        discrete=all_scop_discrete,
        scop_term_count=scop_term_count,
    ):
        return set()

    eligible_names = {
        st["group_name"]
        for st in scop_states.values()
        if st.get("bin_idx") is not None and st["group_name"] in estimated_names
    }
    return eligible_names if len(eligible_names) > 1 else set()


def _get_scop_penalty_metadata(st: dict) -> tuple[float, float, NDArray]:
    """Return cached SCOP penalty metadata, computing it once if needed."""
    cached_rank = st.get("penalty_rank")
    cached_log_det = st.get("penalty_log_det_omega_plus")
    cached_eigvals = st.get("penalty_eigvals_omega")
    if cached_rank is not None and cached_log_det is not None and cached_eigvals is not None:
        eigvals = np.asarray(cached_eigvals, dtype=np.float64)
        rank = float(cached_rank)
        if (
            eigvals.ndim == 1
            and len(eigvals) == int(rank)
            and np.all(eigvals > 0.0)
            and np.isfinite(cached_log_det)
        ):
            return rank, float(cached_log_det), eigvals

    S_scop = st["S_scop"]
    eps_thresh = np.finfo(float).eps ** (2 / 3)
    eigvals = np.linalg.eigvalsh(S_scop)
    thresh = eps_thresh * max(eigvals.max(), 1e-12)
    rank = float(np.sum(eigvals > thresh))
    n_pos = int(rank)

    if n_pos > 0:
        sorted_eig = np.sort(eigvals)[::-1]
        pos_eigvals = np.asarray(sorted_eig[:n_pos], dtype=np.float64)
        log_det = float(np.sum(np.log(np.maximum(pos_eigvals, 1e-300))))
    else:
        pos_eigvals = np.array([], dtype=np.float64)
        log_det = 0.0

    st["penalty_rank"] = rank
    st["penalty_log_det_omega_plus"] = log_det
    st["penalty_eigvals_omega"] = pos_eigvals
    return rank, log_det, pos_eigvals


def _scop_jacobian_diag(st: dict) -> NDArray:
    """Return diag(d gamma / d beta_eff), reusing cached gamma where available."""
    gamma_eff = st.get("gamma_eff")
    if gamma_eff is not None:
        gamma_eff = np.asarray(gamma_eff, dtype=np.float64)
        if gamma_eff.ndim == 1 and np.all(np.isfinite(gamma_eff)):
            return gamma_eff
    beta_eff = np.asarray(st["beta_eff"], dtype=np.float64)
    return np.exp(np.clip(beta_eff, -500, 500))


def _update_multi_scop_discrete_stability_counts(
    *,
    lambdas_old: dict[str, float],
    lambdas_new: dict[str, float],
    active_names: set[str],
    stable_counts: dict[str, int],
) -> dict[str, int]:
    """Track generic per-name lambda stability across freeze and plateau signals.

    A name is counted as stable when it is either near the absolute lambda floor
    or moving by only a small log-step. The near-floor branch resets when a
    lambda first enters the floor region so freezing responds only to
    consecutive near-floor iterations.
    """
    updated = dict(stable_counts)
    floor_threshold = _MULTI_SCOP_DISCRETE_LAMBDA_FLOOR * _MULTI_SCOP_DISCRETE_FLOOR_FACTOR
    for name in active_names:
        lam_old = max(lambdas_old[name], 1.0e-10)
        lam_new = max(lambdas_new[name], 1.0e-10)
        log_step = abs(np.log(lam_new) - np.log(lam_old))
        near_floor_old = lam_old <= floor_threshold
        near_floor_new = lam_new <= floor_threshold
        if near_floor_new:
            updated[name] = updated.get(name, 0) + 1 if near_floor_old else 1
        elif log_step < _MULTI_SCOP_DISCRETE_LOG_STEP_TOL:
            updated[name] = updated.get(name, 0) + 1
        else:
            updated[name] = 0
    return updated


def _freeze_multi_scop_discrete_lambdas(
    *,
    active_names: set[str],
    frozen_names: set[str],
    lambdas_new: dict[str, float],
    stable_counts: dict[str, int],
) -> tuple[set[str], set[str]]:
    """Freeze only floor-pinned names once the generic stability counter matures."""
    active_out = set(active_names)
    frozen_out = set(frozen_names)
    for name in list(active_names):
        lam_new = lambdas_new[name]
        near_floor = lam_new <= (
            _MULTI_SCOP_DISCRETE_LAMBDA_FLOOR * _MULTI_SCOP_DISCRETE_FLOOR_FACTOR
        )
        stable_long_enough = stable_counts.get(name, 0) >= _MULTI_SCOP_DISCRETE_MIN_STABLE_ITERS
        if near_floor and stable_long_enough:
            active_out.discard(name)
            frozen_out.add(name)
    return active_out, frozen_out


def _multi_scop_discrete_plateau_converged(
    *,
    obj_rel_change: float,
    lambdas_old: dict[str, float],
    lambdas_new: dict[str, float],
    active_names: set[str],
) -> bool:
    """Require objective flatness, then check active-set log-step stability."""
    if not active_names:
        return obj_rel_change < _MULTI_SCOP_DISCRETE_OBJ_REL_TOL
    active_changes = [
        abs(np.log(max(lambdas_new[name], 1.0e-10)) - np.log(max(lambdas_old[name], 1.0e-10)))
        for name in active_names
    ]
    max_active_change = max(active_changes) if active_changes else 0.0
    return (
        obj_rel_change < _MULTI_SCOP_DISCRETE_OBJ_REL_TOL
        and max_active_change < _MULTI_SCOP_DISCRETE_ACTIVE_PLATEAU_TOL
    )


def build_scop_penalty_components(
    scop_states: dict[int, dict],
) -> list[PenaltyComponent]:
    """Build PenaltyComponent objects for SCOP terms.

    For SCOP terms, omega_ssp = S_scop (first-diff penalty in beta_eff space).
    No R_inv transform -- SCOP bypasses SSP reparameterization.

    Parameters
    ----------
    scop_states : dict
        Keyed by group index. Each value has keys:
        "S_scop", "group_sl", "group_name", "beta_eff".

    Returns
    -------
    list[PenaltyComponent]
    """
    components = []

    for gi, st in scop_states.items():
        S_scop = st["S_scop"]
        rank, log_det, pos_eigvals = _get_scop_penalty_metadata(st)

        pc = PenaltyComponent(
            name=st["group_name"],
            group_name=st["group_name"],
            group_index=gi,
            group_sl=st["group_sl"],
            omega_raw=S_scop,
            omega_ssp=S_scop,
            rank=rank,
            log_det_omega_plus=log_det,
            eigvals_omega=pos_eigvals,
        )
        components.append(pc)

    return components


def _merge_scop_penalty_components(
    base_components: list[PenaltyComponent] | None,
    scop_components: list[PenaltyComponent],
) -> list[PenaltyComponent]:
    """Replace mapped-coordinate components for SCOP-owned coefficient blocks."""
    scop_group_indices = {component.group_index for component in scop_components}
    ordinary = [
        component
        for component in (base_components or [])
        if component.group_index not in scop_group_indices
    ]
    return ordinary + scop_components


def compute_scop_aware_penalty_quad(
    result_beta: NDArray,
    S: NDArray,
    scop_states: dict[int, dict],
    lambdas: dict[str, float],
    *,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> float:
    """Compute penalty quadratic with correct SCOP beta_eff contributions.

    For non-SCOP groups, result.beta @ S @ result.beta is correct (SSP space).
    For SCOP groups, result.beta contains gamma_eff = exp(beta_eff), but the
    penalty is defined on beta_eff: lambda * beta_eff^T @ S_scop @ beta_eff.
    We subtract the wrong gamma-space contribution and add the correct
    beta_eff-space contribution.

    Parameters
    ----------
    result_beta : (p,) coefficient vector (contains gamma for SCOP groups)
    S : (p, p) full penalty matrix (block-diagonal, includes lambda * S_scop blocks)
    scop_states : SCOP converged state dict
    lambdas : dict of lambda values keyed by component name
    """
    if not scop_states:
        return float(result_beta @ S @ result_beta)

    pq = float(result_beta @ S @ result_beta)

    for gi, st in scop_states.items():
        sl = st["group_sl"]
        beta_eff = st["beta_eff"]
        gamma_eff = result_beta[sl]

        # Remove the complete mapped-coordinate block, including every named
        # component that may overlap this SCOP group.
        pq -= float(gamma_eff @ S[sl, sl] @ gamma_eff)
        matching = (
            [component for component in reml_penalties if component.group_index == gi]
            if reml_penalties is not None
            else []
        )
        if matching:
            for component in matching:
                omega = component.omega_ssp
                if omega is None:
                    omega = component.omega_raw
                pq += lambdas[component.name] * float(beta_eff @ omega @ beta_eff)
        else:
            lam = lambdas.get(st["group_name"], 0.0)
            pq += lam * float(beta_eff @ st["S_scop"] @ beta_eff)

    return pq


def assemble_joint_hessian(
    XtWX_plus_S: NDArray,
    scop_states: dict[int, dict],
    *,
    XtW1: NDArray | None = None,
    sum_W: float | None = None,
) -> tuple[NDArray, dict[str, slice]]:
    """Assemble the intercept-profiled joint Hessian in latent coordinates.

    The XtWX_plus_S matrix is in gamma space for SCOP groups: the SCOP
    diagonal block has only ``lambda * S_scop`` (missing data curvature),
    and the cross-blocks ``X_linear^T W B_scop`` lack the SCOP Jacobian
    factor ``diag(exp(beta_eff))``.

    This function:

    1. Replaces each SCOP diagonal block with ``H_scop_penalized`` (the
       full Newton Hessian in beta_eff space, including data curvature).
    2. Transforms cross-blocks to beta_eff space by scaling columns
       (for ``H[other, scop]``) and rows (for ``H[scop, other]``) by
       ``j_diag = exp(beta_eff)``, the diagonal of the SCOP Jacobian
       ``d(gamma_eff)/d(beta_eff)``.

    Parameters
    ----------
    XtWX_plus_S : (p, p) ndarray
        The linear-system penalized Gram matrix.
    scop_states : dict
        SCOP converged state dict, keyed by group index. Each value
        must contain "group_sl", "H_scop_penalized", "group_name",
        and "beta_eff".

    Returns
    -------
    H_joint : (p, p) ndarray
        Joint Hessian with SCOP blocks and cross-blocks in beta_eff space.
    mapping : dict
        Maps group_name to the slice in H_joint for each SCOP group.
    """
    if (XtW1 is None) != (sum_W is None):
        raise ValueError("XtW1 and sum_W must be provided together")
    if not scop_states and XtW1 is None:
        return XtWX_plus_S, {}

    p = XtWX_plus_S.shape[0]
    H_joint = XtWX_plus_S.copy()
    mapping = {}

    # Collect all SCOP indices so we can identify "other" indices
    scop_slices = []
    for gi, st in scop_states.items():
        scop_slices.append(st["group_sl"])

    all_scop_idx = (
        np.concatenate([np.arange(sl.start, sl.stop) for sl in scop_slices])
        if scop_slices
        else np.empty(0, dtype=int)
    )
    other_idx = np.setdiff1d(np.arange(p), all_scop_idx)

    for gi, st in scop_states.items():
        sl = st["group_sl"]
        H_scop = st["H_scop_penalized"]
        name = st["group_name"]
        j_diag = _scop_jacobian_diag(st)

        # Replace diagonal SCOP block with full Newton Hessian
        H_joint[sl, sl] = H_scop
        mapping[name] = sl

        # Transform cross-blocks: gamma-space → beta_eff-space
        # H[other, scop] = X_other^T W B_scop  →  scale columns by j_diag
        if other_idx.size > 0:
            scop_idx = np.arange(sl.start, sl.stop)
            H_joint[np.ix_(other_idx, scop_idx)] *= j_diag[np.newaxis, :]
            H_joint[np.ix_(scop_idx, other_idx)] *= j_diag[:, np.newaxis]

    # Transform SCOP-SCOP cross-blocks: H_ij(beta_eff) = diag(j_i) @ H_ij(gamma) @ diag(j_j)
    scop_items = list(scop_states.items())
    for idx_a in range(len(scop_items)):
        gi_a, st_a = scop_items[idx_a]
        sl_a = st_a["group_sl"]
        j_a = _scop_jacobian_diag(st_a)
        for idx_b in range(idx_a + 1, len(scop_items)):
            gi_b, st_b = scop_items[idx_b]
            sl_b = st_b["group_sl"]
            j_b = _scop_jacobian_diag(st_b)
            idx_a_arr = np.arange(sl_a.start, sl_a.stop)
            idx_b_arr = np.arange(sl_b.start, sl_b.stop)
            H_joint[np.ix_(idx_a_arr, idx_b_arr)] *= j_a[:, np.newaxis] * j_b[np.newaxis, :]
            H_joint[np.ix_(idx_b_arr, idx_a_arr)] *= j_b[:, np.newaxis] * j_a[np.newaxis, :]

    if XtW1 is not None:
        assert sum_W is not None
        intercept_cross = np.asarray(XtW1, dtype=np.float64)
        if intercept_cross.shape != (p,):
            raise ValueError("XtW1 must match the slope coefficient dimension")
        if not np.all(np.isfinite(intercept_cross)):
            raise ValueError("XtW1 must be finite")
        if not np.isfinite(sum_W) or sum_W <= 0.0:
            raise ValueError("sum_W must be positive and finite")
        intercept_cross = intercept_cross.copy()
        for st in scop_states.values():
            sl = st["group_sl"]
            intercept_cross[sl] *= _scop_jacobian_diag(st)
        H_joint -= np.outer(intercept_cross, intercept_cross) / sum_W

    # Restrict to the range the SCOP steps could resolve. The diagonal blocks
    # arrive already restricted, but the cross-blocks assembled above still
    # couple other coefficients to a direction the solver froze, and that
    # leakage is enough to leave the joint matrix indefinite where consumers
    # decompose it. Projecting the assembled matrix covers both. A fit whose
    # steps discarded nothing is returned untouched.
    H_joint = restrict_to_scop_resolved_range(H_joint, scop_states)

    return H_joint, mapping


def _result_intercept_moments(
    result: Any,
    *,
    width: int,
    fisher_mean_x: NDArray | None = None,
    fisher_sum_w: float | None = None,
) -> tuple[NDArray, float]:
    """Recover intercept moments from the working cache or compatibility metadata."""
    if (fisher_mean_x is None) != (fisher_sum_w is None):
        raise ValueError("fisher_mean_x and fisher_sum_w must be provided together")
    if fisher_mean_x is not None:
        assert fisher_sum_w is not None
        mean_x = np.asarray(fisher_mean_x, dtype=np.float64)
        sum_w = float(fisher_sum_w)
        if mean_x.shape != (width,):
            raise RuntimeError("SCOP REML intercept geometry has the wrong width")
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise RuntimeError("SCOP REML intercept weight sum must be positive and finite")
        return sum_w * mean_x, sum_w

    rank_info = result.rank_info
    if rank_info is None:
        raise RuntimeError("SCOP REML requires retained intercept geometry")
    sum_w = float(rank_info.sum_w)
    mean_x = np.asarray(rank_info.mean_x, dtype=np.float64)
    if mean_x.shape != (width,):
        raise RuntimeError("SCOP REML intercept geometry has the wrong width")
    return sum_w * mean_x, sum_w


def _reml_evaluation_phi(
    evaluation: Any,
    *,
    scale_known: bool,
    fallback_likelihood_size: float,
) -> float:
    """Return the scale paired with one authoritative REML evaluation."""
    if scale_known:
        return 1.0
    if evaluation.profiled_scale is not None:
        return float(evaluation.profiled_scale.phi)
    penalty_nullity = float(evaluation.penalty_nullity or 0.0)
    return max(
        float(evaluation.penalized_deviance)
        / max(float(fallback_likelihood_size) - penalty_nullity, 1.0),
        1.0e-10,
    )


def _evaluate_scop_reml_mode(
    context: _SCOPREMLFitContext,
    lambdas: dict[str, float],
    *,
    result: Any,
    xtwx: NDArray,
    centered_xtwx: NDArray,
    fisher_mean_x: NDArray,
    fisher_sum_w: float | None = None,
    scop_states: dict[int, dict],
    penalty_components: list[PenaltyComponent] | None = None,
    penalty: NDArray | None = None,
    mode_score: SCOPModeScore | None = None,
    eta_unclipped: NDArray | None = None,
) -> _SCOPREMLMode:
    """Assemble and evaluate LAML from one lambda-coherent fitted mode."""
    if penalty_components is None:
        penalty_components = _merge_scop_penalty_components(
            context.reml_penalties,
            build_scop_penalty_components(scop_states),
        )
    if penalty is None:
        penalty = build_penalty_matrix(
            list(context.dm.group_matrices),
            context.groups,
            lambdas,
            context.dm.p,
            reml_penalties=penalty_components,
        )
    xtw1, sum_w = _result_intercept_moments(
        result,
        width=context.dm.p,
        fisher_mean_x=fisher_mean_x if fisher_sum_w is not None else None,
        fisher_sum_w=fisher_sum_w,
    )
    if mode_score is None:
        mode_score = scop_penalized_mode_score(
            dm=context.dm,
            distribution=context.distribution,
            link=context.link,
            y=context.y,
            sample_weight=context.sample_weight,
            offset_arr=context.offset_arr,
            result=result,
            latent_penalty=penalty,
            scop_states=scop_states,
            centered_fisher_gram=centered_xtwx,
            fisher_mean_x=fisher_mean_x,
            fisher_sum_w=sum_w,
            eta_unclipped=eta_unclipped,
        )

    def terminal_fisher_weights() -> NDArray:
        eta_raw = (
            context.dm.matvec(result.beta) + result.intercept + context.offset_arr
            if eta_unclipped is None
            else np.asarray(eta_unclipped, dtype=np.float64)
        )
        eta = stabilize_eta(eta_raw, context.link)
        mu = clip_mu(context.link.inverse(eta), context.distribution)
        variance = np.maximum(
            np.asarray(context.distribution.variance(mu), dtype=np.float64),
            _VARIANCE_FLOOR,
        )
        derivative = np.asarray(context.link.deriv_inverse(eta), dtype=np.float64)
        return context.sample_weight * derivative**2 / variance

    curvature = (
        classify_scop_reml_curvature(context.distribution, context.link)
        if scop_states
        else "fisher"
    )
    if curvature == "observed":
        joint_geometry = build_observed_scop_joint_geometry(
            dm=context.dm,
            distribution=context.distribution,
            link=context.link,
            y=context.y,
            sample_weight=context.sample_weight,
            offset_arr=context.offset_arr,
            result=result,
            penalty=penalty,
            scop_states=scop_states,
            fisher_XtWX=xtwx,
            fisher_XtW1=xtw1,
            fisher_sum_W=sum_w,
            centered_fisher_gram=centered_xtwx,
            fisher_mean_x=fisher_mean_x,
            eta_unclipped=eta_unclipped,
        )
        hessian_inverse = joint_geometry.hessian_inverse
        log_det_h = joint_geometry.log_det_H
        hessian_rank = joint_geometry.hessian_rank
        curvature_source = joint_geometry.curvature_source
    else:
        joint_geometry = build_cached_scop_joint_geometry(
            raw_fisher_gram=xtwx,
            fisher_xtw=xtw1,
            fisher_sum_w=sum_w,
            latent_penalty=penalty,
            scop_states=scop_states,
            centered_fisher_gram=centered_xtwx,
            fisher_mean_x=fisher_mean_x,
            dm=context.dm,
            fisher_weights=terminal_fisher_weights,
        )
        hessian_inverse = joint_geometry.hessian_inverse
        log_det_h = joint_geometry.log_det_H
        hessian_rank = joint_geometry.hessian_rank
        curvature_source = joint_geometry.curvature_source
    evaluation = reml_laml_objective(
        context.dm,
        context.distribution,
        context.link,
        context.groups,
        context.y,
        result,
        lambdas,
        context.sample_weight,
        context.offset_arr,
        XtWX=xtwx,
        XtW1=xtw1,
        sum_W=sum_w,
        log_det_H=log_det_h,
        hessian_rank=hessian_rank,
        S_override=penalty,
        reml_penalties=penalty_components,
        scop_states=scop_states,
        likelihood_size=context.likelihood_size,
        gamma_scale_data=context.gamma_scale_data,
        return_evaluation=True,
    )
    return _SCOPREMLMode(
        lambdas=lambdas.copy(),
        result=result,
        xtwx=xtwx,
        centered_xtwx=centered_xtwx,
        fisher_mean_x=fisher_mean_x,
        fisher_sum_w=sum_w,
        scop_states=scop_states,
        penalty=penalty,
        penalty_components=penalty_components,
        joint_geometry=joint_geometry,
        hessian_inverse=hessian_inverse,
        evaluation=evaluation,
        log_det_h=log_det_h,
        hessian_rank=hessian_rank,
        curvature_source=curvature_source,
        mode_score=mode_score,
    )


def _scop_mode_newton_relative(mode: _SCOPREMLMode) -> float:
    """Return the estimable-range Newton correction for mode certification.

    Componentwise relative scores are deliberately retained as a diagnostic,
    but they are not a sufficient convergence test at a flat SCOP boundary.
    There the exponential Jacobian and the exact penalty score both vanish,
    so harmless penalty-matvec noise can have an order-one *relative* score.
    The factor-certified pseudoinverse instead measures the coefficient
    correction on the estimable range, as required for the rank-deficient
    latent geometry described by Pya and Wood.
    """
    geometry = mode.joint_geometry
    score = mode.mode_score
    latent_beta = np.asarray(mode.result.beta, dtype=np.float64).copy()
    fisher_transformed_mean = np.asarray(mode.fisher_mean_x, dtype=np.float64).copy()
    for state in mode.scop_states.values():
        group_slice = state["group_sl"]
        latent_beta[group_slice] = np.asarray(state["beta_eff"], dtype=np.float64)
        fisher_transformed_mean[group_slice] *= _scop_jacobian_diag(state)

    geometry_mean = geometry.transformed_mean_x
    if geometry_mean is None:
        geometry_mean = geometry.transformed_intercept_cross / geometry.sum_w
    geometry_mean = np.asarray(geometry_mean, dtype=np.float64)

    # ``score.slopes`` is profiled with the retained Fisher mean. Reconstruct
    # the raw slope score, then profile it with the curvature actually used by
    # this mode (observed or Fisher) before applying that same pseudoinverse.
    raw_slope_score = score.slopes + fisher_transformed_mean * score.intercept
    profiled_score = raw_slope_score - geometry_mean * score.intercept
    # Stationarity is only meaningful where the solver could move. The SCOP
    # Newton step truncates its augmented factor at ``sqrt(eps)``, and the
    # discarded directions are ones the data cannot resolve -- no iteration
    # drives their score to zero, so requiring it here would reject every
    # boundary mode. This geometry's own estimable range is certified over the
    # whole centered model at its own scale, so it does not coincide with the
    # solver's; the solver's is the one the coefficients actually obey.
    #
    # The score is projected *before* the pseudoinverse, not the correction
    # after it: ``hessian_inverse`` is not diagonal in this basis, so a score
    # component along a discarded direction re-emerges as a correction pointing
    # somewhere else entirely, which no projection of the result can remove.
    resolved = scop_resolved_range_projector(mode.scop_states, len(profiled_score))
    if resolved is not None:
        profiled_score = resolved @ profiled_score

    slope_correction = geometry.hessian_inverse @ profiled_score
    if resolved is not None:
        # The correction must also lie where the solver can move.
        slope_correction = resolved @ slope_correction

    intercept_correction = (
        score.intercept - float(geometry.transformed_intercept_cross @ slope_correction)
    ) / geometry.sum_w

    slope_relative = float(
        np.max(
            np.abs(slope_correction) / np.maximum(1.0, np.abs(latent_beta)),
            initial=0.0,
        )
    )
    intercept_relative = abs(intercept_correction) / max(
        1.0,
        abs(float(mode.result.intercept)),
    )
    return max(slope_relative, intercept_relative)


def _scop_mode_tolerance(mode: _SCOPREMLMode, pirls_tol: float) -> float:
    """Return a rank-aware numerical floor for terminal mode certification.

    The authoritative observation-factor policy resolves directions only to
    ``sqrt(eps)`` relative accuracy. A joint rank-dimensional correction
    aggregates that factor/score roundoff at root-rank scale. This remains a
    numerical floor, not an alternative score-based convergence criterion.
    """
    numerical_floor = np.sqrt(max(1, mode.hessian_rank) * np.finfo(np.float64).eps)
    return max(10.0 * min(pirls_tol, 1.0e-10), float(numerical_floor))


def _fit_scop_reml_mode(
    context: _SCOPREMLFitContext,
    lambdas: dict[str, float],
    *,
    beta_init: NDArray | None,
    intercept_init: float | None,
    scop_state_init: dict[int, dict] | None,
    phase: str,
    reml_iteration: int,
    line_search_iteration: int | None = None,
    trial_alpha: float | None = None,
    require_converged: bool,
    _certification_retry: int = 0,
) -> _SCOPREMLMode | None:
    """Fit and evaluate one mode, optionally rejecting a failed inner solve."""
    debug_context: dict[str, Any] = {
        "phase": phase,
        "reml_iteration": reml_iteration,
    }
    if line_search_iteration is not None:
        debug_context["line_search_iteration"] = line_search_iteration
    if trial_alpha is not None:
        debug_context["trial_alpha"] = float(trial_alpha)

    trace_run = getattr(context.debug_recorder, "trace_run", None)
    trace_purpose = {
        "bootstrap": "reml_bootstrap",
        "candidate": "reml_candidate",
        "reml": "reml_candidate",
        "line_search": "reml_line_search",
        "final": "reml_final",
        "fixed": "reml_fixed",
    }.get(phase, f"reml_{phase}")
    working_cache: dict[str, Any] = {}
    irls_out: Any = fit_irls_direct(
        X=context.dm,
        y=context.y,
        weights=context.sample_weight,
        family=context.distribution,
        link=context.link,
        groups=context.groups,
        lambda2=lambdas,
        offset=context.offset_arr,
        beta_init=beta_init,
        intercept_init=intercept_init,
        tol=(
            min(context.pirls_tol, 1.0e-10)
            if any(group.monotone_engine == "scop" for group in context.groups)
            and classify_scop_reml_curvature(context.distribution, context.link) == "observed"
            else context.pirls_tol
        ),
        max_iter=context.max_pirls_iter,
        return_xtwx=True,
        return_scop_state=True,
        reml_penalties=context.reml_penalties,
        # A LAML evaluation requires a coefficient mode. Deviance plateaus can
        # precede latent SCOP stationarity, especially near an interpolating
        # fit where the penalty deliberately trades a tiny deviance increase
        # for a much smaller quadratic.
        convergence="coefficients",
        _scop_joint=context.scop_joint,
        scop_state_init=scop_state_init,
        debug_recorder=context.debug_recorder,
        debug_context=debug_context,
        trace_run=trace_run,
        trace_purpose=trace_purpose,
        _compute_scop_postfit_inference=False,
        compute_rank_info=False,
        _compute_fit_statistics=False,
        _compute_reml_geometry=False,
        cache_out=working_cache,
    )
    scop_states: dict[int, dict]
    if len(irls_out) == 4:
        result, _, xtwx, scop_states = irls_out
    else:
        result, _, xtwx = irls_out
        scop_states = {}

    if require_converged and not result.converged:
        return None
    rank_info = result.rank_info
    cached_mean_x = working_cache.get("mean_x")
    cached_sum_w = working_cache.get("sum_W")
    if cached_mean_x is None or cached_sum_w is None:
        if rank_info is None:
            raise RuntimeError("SCOP REML requires retained centered fit geometry")
        cached_mean_x = rank_info.mean_x
        cached_sum_w = rank_info.sum_w
    fisher_mean_x = np.asarray(cached_mean_x, dtype=np.float64)
    fisher_sum_w = float(cached_sum_w)
    if fisher_mean_x.shape != (context.dm.p,):
        raise RuntimeError("SCOP REML centered mean has the wrong width")
    if not np.isfinite(fisher_sum_w) or fisher_sum_w <= 0.0:
        raise RuntimeError("SCOP REML centered weight sum must be positive and finite")
    centered_xtwx = working_cache.get("centered_XtWX")
    if centered_xtwx is None:
        # Compatibility for injected/custom solvers. Production direct IRLS
        # always publishes the stable centered matrix through ``cache_out``.
        centered_xtwx = xtwx - fisher_sum_w * np.outer(
            fisher_mean_x,
            fisher_mean_x,
        )
    centered_xtwx = np.asarray(centered_xtwx, dtype=np.float64)
    penalty_components = _merge_scop_penalty_components(
        context.reml_penalties,
        build_scop_penalty_components(scop_states),
    )
    penalty = build_penalty_matrix(
        list(context.dm.group_matrices),
        context.groups,
        lambdas,
        context.dm.p,
        reml_penalties=penalty_components,
    )
    sum_w = fisher_sum_w
    if scop_states:
        mode_score = scop_penalized_mode_score(
            dm=context.dm,
            distribution=context.distribution,
            link=context.link,
            y=context.y,
            sample_weight=context.sample_weight,
            offset_arr=context.offset_arr,
            result=result,
            latent_penalty=penalty,
            scop_states=scop_states,
            centered_fisher_gram=centered_xtwx,
            fisher_mean_x=fisher_mean_x,
            fisher_sum_w=sum_w,
            eta_unclipped=working_cache.get("eta_unclipped"),
        )
    else:
        mode_score = SCOPModeScore(
            intercept=0.0,
            slopes=np.zeros(context.dm.p, dtype=np.float64),
            max_abs=0.0,
            relative_max=0.0,
        )
    mode = _evaluate_scop_reml_mode(
        context,
        lambdas,
        result=result,
        xtwx=xtwx,
        centered_xtwx=centered_xtwx,
        fisher_mean_x=fisher_mean_x,
        fisher_sum_w=fisher_sum_w,
        scop_states=scop_states,
        penalty_components=penalty_components,
        penalty=penalty,
        mode_score=mode_score,
        eta_unclipped=working_cache.get("eta_unclipped"),
    )
    mode_newton_relative = _scop_mode_newton_relative(mode)
    mode_tolerance = _scop_mode_tolerance(mode, context.pirls_tol)
    if mode_newton_relative > mode_tolerance:
        if _certification_retry < 3:
            # Rungs 0->1 and 1->2 tighten the inner tolerance and re-fit from the
            # mode that just failed. That rescues a residual left by loose inner
            # convergence -- measured, 29 of 43 retries -- but it cannot move a
            # fit already converged tighter than the bar, which returns the same
            # mode bit-identically however hard the tolerance is squeezed.
            #
            # Rung 2->3 is therefore the cold one. It holds the tightest
            # tolerance the ladder reached rather than squeezing further, so it
            # differs from its predecessor in exactly one respect: the starting
            # point. The warm start it drops comes from a bootstrap fitted at
            # lambda=1e-4, a long way from these lambdas, which is the plausible
            # reason the mode landed off-stationary in the first place.
            cold_rung = _certification_retry == 2
            retry_tolerance = 10.0 ** (-10 - min(_certification_retry, 1))
            retry_context = replace(
                context,
                pirls_tol=min(context.pirls_tol, retry_tolerance),
            )
            centered_scale = np.sqrt(np.maximum(np.diag(centered_xtwx), 0.0) / sum_w)
            warm_retry = not cold_rung and _raw_centering_well_scaled(fisher_mean_x, centered_scale)
            return _fit_scop_reml_mode(
                retry_context,
                lambdas,
                beta_init=result.beta.copy() if warm_retry else None,
                intercept_init=float(result.intercept) if warm_retry else None,
                scop_state_init=(scop_states if scop_states and warm_retry else None),
                phase=phase,
                reml_iteration=reml_iteration,
                line_search_iteration=line_search_iteration,
                trial_alpha=trial_alpha,
                require_converged=require_converged,
                _certification_retry=_certification_retry + 1,
            )
        if require_converged:
            return None
        raise RuntimeError(
            "SCOP coefficient mode failed latent penalized-score certification "
            f"(relative score={mode_score.relative_max:.3e}, "
            f"relative Newton correction={mode_newton_relative:.3e}, "
            f"tolerance={mode_tolerance:.3e})"
        )
    if trace_run is not None and trace_run.enabled:
        if result.state_id is None:  # pragma: no cover - trace contract
            raise RuntimeError("traced SCOP REML evaluation is missing its coefficient state ID")
        phi = _reml_evaluation_phi(
            mode.evaluation,
            scale_known=getattr(context.distribution, "scale_known", True),
            fallback_likelihood_size=context.likelihood_size,
        )
        trace_run.emit_lazy(
            "evaluation",
            lambda: {
                "state_id": result.state_id,
                "evaluation_id": result.evaluation_id,
                "solver": "scop_efs_reml",
                "phase": phase,
                "outer_iteration": reml_iteration,
                "line_search_iteration": line_search_iteration,
                "trial_alpha": trial_alpha,
                "objective": mode.objective,
                "lambdas": mode.lambdas,
                "dispersion": phi,
                # Candidate rank/EDF work is deliberately omitted. Only the
                # public terminal is hydrated with authoritative SCOP EDF.
                "effective_df": None,
                "curvature_source": mode.curvature_source,
                "mode_score_relative": mode.mode_score.relative_max,
            },
            channel="reml",
            purpose=trace_purpose,
            authoritative=False,
        )
    return mode


def _certified_terminal_rank(
    matrix: NDArray,
    factor_factory: Callable[[], NDArray],
) -> RankDecomposition:
    """Apply the shared Gram-first policy to one terminal-only rank claim."""
    decomposition = decompose_gram(matrix)
    if needs_factor_certification(decomposition):
        decomposition = decompose_factor(factor_factory())
    return decomposition


def _hydrate_scop_terminal_rank_info(
    context: _SCOPREMLFitContext,
    mode: _SCOPREMLMode,
    *,
    fisher_weights: Callable[[], NDArray],
) -> None:
    """Install generic estimability metadata once on the retained SCOP mode."""
    result = mode.result
    if result.rank_info is not None:
        raise RuntimeError("SCOP REML candidate unexpectedly retained public rank metadata")

    data_rank = _certified_terminal_rank(
        mode.centered_xtwx,
        lambda: grouped_weighted_factor(
            context.dm,
            fisher_weights(),
            center=mode.fisher_mean_x,
        ),
    )
    augmented_rank = _certified_terminal_rank(
        mode.centered_xtwx + mode.penalty,
        lambda: grouped_augmented_factor(
            context.dm,
            fisher_weights(),
            mode.penalty,
            center=mode.fisher_mean_x,
        ),
    )
    coefficient_rank = _certified_terminal_rank(
        mode.xtwx + mode.penalty,
        lambda: grouped_augmented_factor(
            context.dm,
            fisher_weights(),
            mode.penalty,
        ),
    )

    selected_columns = np.arange(context.dm.p, dtype=int)
    selected_columns.setflags(write=False)
    mean_x = np.array(mode.fisher_mean_x, dtype=np.float64, copy=True)
    mean_x.setflags(write=False)
    feature_edf = np.zeros(context.dm.p, dtype=np.float64)
    feature_edf.setflags(write=False)
    result.rank_info = RankInfo(
        policy_version=SHARED_RANK_POLICY.version,
        coordinate_space="solver",
        selected_columns=selected_columns,
        selected_group_names=tuple(group.name for group in context.groups),
        sum_w=mode.fisher_sum_w,
        mean_x=mean_x,
        intercept_edf=0.0,
        data=data_rank,
        augmented=augmented_rank,
        coefficient=coefficient_rank,
        feature_edf=feature_edf,
        group_edf={group.name: 0.0 for group in context.groups},
        objective_loss=None,
    )


def _finalize_scop_reml_mode(
    context: _SCOPREMLFitContext,
    mode: _SCOPREMLMode,
) -> Any:
    """Hydrate exactly one retained coefficient mode for public inference."""
    result = mode.result
    if result.scop_inference is not None or result.scop_geometry is not None:
        raise RuntimeError("SCOP REML terminal mode was already hydrated")

    cached_weights: NDArray | None = None

    def terminal_fisher_weights() -> NDArray:
        nonlocal cached_weights
        if cached_weights is None:
            eta = stabilize_eta(
                context.dm.matvec(result.beta) + result.intercept + context.offset_arr,
                context.link,
            )
            mu = clip_mu(context.link.inverse(eta), context.distribution)
            variance = np.maximum(
                np.asarray(context.distribution.variance(mu), dtype=np.float64),
                _VARIANCE_FLOOR,
            )
            derivative = np.asarray(context.link.deriv_inverse(eta), dtype=np.float64)
            cached_weights = context.sample_weight * derivative**2 / variance
        return cached_weights

    _hydrate_scop_terminal_rank_info(
        context,
        mode,
        fisher_weights=terminal_fisher_weights,
    )
    result.phi = _reml_evaluation_phi(
        mode.evaluation,
        scale_known=getattr(context.distribution, "scale_known", True),
        fallback_likelihood_size=context.likelihood_size,
    )
    result.log_det_H = mode.log_det_h
    result.reml_hessian_rank = mode.hessian_rank
    fisher_xtw = mode.fisher_sum_w * mode.fisher_mean_x
    install_scop_postfit_inference(
        result,
        raw_fisher_gram=mode.xtwx,
        centered_fisher_gram=mode.centered_xtwx,
        fisher_xtw=fisher_xtw,
        fisher_mean_x=mode.fisher_mean_x,
        fisher_sum_w=mode.fisher_sum_w,
        latent_penalty=mode.penalty,
        scop_states=mode.scop_states,
        groups=context.groups,
        observed_geometry=mode.joint_geometry,
        dm=context.dm,
        fisher_weights=terminal_fisher_weights,
    )
    if not np.isfinite(result.phi) or not np.isfinite(result.effective_df):
        raise RuntimeError("SCOP REML terminal hydration left non-finite fit statistics")
    return result


def _backtrack_scop_efs_candidate(
    context: _SCOPREMLFitContext,
    current: _SCOPREMLMode,
    proposed_lambdas: dict[str, float],
    *,
    reml_iteration: int,
    max_attempts: int = _SCOP_EFS_MAX_BACKTRACK_ATTEMPTS,
) -> tuple[_SCOPREMLMode, bool]:
    """Fit and score repeatedly damped log-lambda trials.

    The returned boolean is false only when every attempted converged candidate
    in both the proposed and reflected directions is uphill (or every inner
    solve fails).  Reflection is a safeguarded fallback for EFS directions,
    whose expected-curvature update need not remain a descent direction for
    the exact LAML objective near a mode.  In the failure case the exact
    current fitted mode is returned, so callers cannot accidentally publish an
    unevaluated lambda movement.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    changed_names = [
        name
        for name, proposed in proposed_lambdas.items()
        if name in current.lambdas and proposed != current.lambdas[name]
    ]
    if not changed_names:
        return current, True

    log_directions: dict[str, float] = {}
    for name in changed_names:
        old = float(current.lambdas[name])
        proposed = float(proposed_lambdas[name])
        if old <= 0.0 or proposed <= 0.0 or not np.isfinite(old + proposed):
            raise ValueError("SCOP EFS lambda trials must be positive and finite")
        log_directions[name] = float(np.log(proposed) - np.log(old))

    trial_number = 0
    for direction_sign in (1.0, -1.0):
        direction_attempts = (
            max_attempts
            if direction_sign > 0.0
            else min(max_attempts, _SCOP_EFS_MAX_REFLECTED_ATTEMPTS)
        )
        for attempt in range(direction_attempts):
            alpha = direction_sign * 0.5**attempt
            trial_lambdas = current.lambdas.copy()
            for name in changed_names:
                log_trial = np.log(current.lambdas[name]) + alpha * log_directions[name]
                trial_lambdas[name] = float(np.clip(np.exp(log_trial), 1.0e-6, 1.0e10))

            trial_number += 1
            candidate = _fit_scop_reml_mode(
                context,
                trial_lambdas,
                beta_init=current.result.beta,
                intercept_init=float(current.result.intercept),
                scop_state_init=current.scop_states if current.scop_states else None,
                phase="line_search",
                reml_iteration=reml_iteration,
                line_search_iteration=trial_number,
                trial_alpha=alpha,
                require_converged=True,
            )
            if candidate is None:
                continue
            tolerance = 1.0e-8 * max(abs(current.objective), 1.0)
            candidate_is_acceptable = (
                candidate.objective <= current.objective + tolerance
                if direction_sign > 0.0
                else candidate.objective < current.objective
            )
            if np.isfinite(candidate.objective) and candidate_is_acceptable:
                return candidate, True

    return current, False


def _is_scop_component(pc: PenaltyComponent, scop_states: dict[int, dict]) -> dict | None:
    """Return SCOP state dict if pc corresponds to a SCOP group, else None."""
    state = scop_states.get(pc.group_index)
    if state is None:
        return None
    if state["group_name"] != pc.group_name or state["group_sl"] != pc.group_sl:
        return None
    return state


def scop_efs_lambda_update(
    pc: PenaltyComponent,
    beta: NDArray,
    H_joint_inv: NDArray,
    inv_phi: float,
    lam_old: float,
    scop_states: dict[int, dict],
) -> float:
    """Fellner-Schall fixed-point update for one penalty component.

    .. deprecated:: Use ``_joint_efs_lambda_step`` for the main EFS loop.
        This function uses the old fixed-point formula and is kept only for
        backward compatibility.
    """
    omega_g = pc.omega_ssp
    sl = pc.group_sl

    scop_st = _is_scop_component(pc, scop_states)
    if scop_st is not None:
        beta_g = scop_st["beta_eff"]
    else:
        beta_g = beta[sl]

    if np.linalg.norm(beta_g) < 1e-12:
        return lam_old

    quad = float(beta_g @ omega_g @ beta_g)
    trace_term = float(np.trace(H_joint_inv[sl, sl] @ omega_g))

    r_j = pc.rank
    denom = inv_phi * quad + trace_term

    if denom > 1e-12:
        lam_raw = r_j / denom
    else:
        return lam_old

    log_step = np.log(max(lam_raw, 1e-10)) - np.log(max(lam_old, 1e-10))
    log_step = np.clip(log_step, -5.0, 5.0)
    lam_new = lam_old * float(np.exp(log_step))

    return lam_new


def fit_fixed_scop_reml(
    dm: DesignMatrix,
    distribution,
    link,
    groups: list[GroupSlice],
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    lambdas: dict[str, float],
    *,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    reml_penalties: list[PenaltyComponent] | None = None,
    convergence: str = "deviance",
    _scop_joint: bool = True,
    debug_recorder=None,
) -> REMLResult:
    """Evaluate and publish one fixed-lambda SCOP coefficient mode."""
    likelihood_size = dispersion_likelihood_size(distribution, sample_weight)
    gamma_scale_data = (
        prepare_gamma_reml_scale_data(y, sample_weight) if isinstance(distribution, Gamma) else None
    )
    context = _SCOPREMLFitContext(
        dm=dm,
        distribution=distribution,
        link=link,
        groups=groups,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        reml_penalties=reml_penalties,
        convergence=convergence,
        scop_joint=_scop_joint,
        debug_recorder=debug_recorder,
        likelihood_size=likelihood_size,
        gamma_scale_data=gamma_scale_data,
    )
    mode = _fit_scop_reml_mode(
        context,
        lambdas,
        beta_init=None,
        intercept_init=None,
        scop_state_init=None,
        phase="fixed",
        reml_iteration=0,
        require_converged=True,
    )
    if mode is None:
        raise RuntimeError("fixed-lambda SCOP fit did not converge to a coefficient mode")
    result = _finalize_scop_reml_mode(context, mode)
    step_norms = {
        state["group_name"]: float(state.get("last_step_norm", 0.0))
        for state in mode.scop_states.values()
    }
    fisher_fallbacks = sum(
        bool(state.get("last_fisher_fallback", False)) for state in mode.scop_states.values()
    )
    return REMLResult(
        lambdas=mode.lambdas.copy(),
        pirls_result=result,
        n_reml_iter=0,
        converged=bool(result.converged),
        lambda_history=[mode.lambdas.copy()],
        objective=float(mode.evaluation.value),
        reml_penalties=mode.penalty_components,
        scop_states=mode.scop_states if mode.scop_states else None,
        inner_iter_history=[int(result.n_iter)],
        objective_history=[float(mode.evaluation.value)],
        curvature_source=mode.curvature_source,
        termination_reason="fixed_lambdas",
        scop_step_norms=[step_norms] if step_norms else None,
        scop_fisher_fallbacks=int(fisher_fallbacks),
    )


def _joint_efs_lambda_step(
    all_pcs: list[PenaltyComponent],
    beta: NDArray,
    H_joint_inv: NDArray,
    phi: float,
    lambdas: dict[str, float],
    estimated_names: set[str],
    scop_states: dict[int, dict],
    alpha: dict[str, float],
    prev_dlsp: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Joint EFS lambda step using rEDF/pSp (Wood & Fasiolo 2017, scasm-style).

    Update on log scale::

        rEDF = rank - lambda * sEDF
        dlsp = log(phi) + log(rEDF) - log(pSp * lambda)
        log_lambda_new = log_lambda_old + alpha * dlsp

    with adaptive alpha (halve on sign-flip, grow on stable direction) and
    suppression detection.

    Parameters
    ----------
    all_pcs : list of PenaltyComponent
        All penalty components (SSP + SCOP).
    beta : (p,) coefficient vector (gamma for SCOP groups).
    H_joint_inv : (p, p) inverse of joint Hessian.
    phi : scale parameter (1.0 for known-scale families).
    lambdas : current lambda values keyed by component name.
    estimated_names : set of names to update.
    scop_states : SCOP converged state dict.
    alpha : per-component adaptive step size (mutated in place).
    prev_dlsp : previous step directions for sign-flip detection.

    Returns
    -------
    lambdas_new : updated lambda dict.
    alpha : updated adaptive step sizes.
    dlsp_accepted : step directions (for sign-flip tracking; caller should
        update prev_dlsp from the POST-DAMPING accepted step, not this raw value).
    """
    lambdas_new = lambdas.copy()
    dlsp_out: dict[str, float] = {}
    prior_edf, _ = compute_logdet_s_derivatives(lambdas, all_pcs)

    for pc in all_pcs:
        if pc.name not in estimated_names:
            continue

        omega_g = pc.omega_ssp
        sl = pc.group_sl

        scop_st = _is_scop_component(pc, scop_states)
        if scop_st is not None:
            beta_g = scop_st["beta_eff"]
        else:
            beta_g = beta[sl]

        if np.linalg.norm(beta_g) < 1e-12:
            dlsp_out[pc.name] = 0.0
            continue

        # pSp and sEDF
        pSp = float(beta_g @ omega_g @ beta_g)
        sEDF = float(np.trace(H_joint_inv[sl, sl] @ omega_g))

        # Residual EDF — keep raw for suppression check, floor for log.
        # The rank shortcut is exact only for an isolated penalty block.  For
        # overlapping components Wood--Fasiolo's generalized update requires
        # lambda_j tr(S_lambda^- S_j), i.e. the log|S|+ derivative.
        rEDF_raw = prior_edf[pc.name] - lambdas[pc.name] * sEDF
        rEDF_used = max(rEDF_raw, 1e-7)

        # Log-scale step: dlsp = log(phi) + log(rEDF) - log(pSp * lambda)
        pSp_lam = max(pSp * lambdas[pc.name], 1e-300)
        dlsp = np.log(max(phi, 1e-300)) + np.log(rEDF_used) - np.log(pSp_lam)

        # Suppression detection (scasm-style)
        if rEDF_raw < 0.05 and dlsp > 0:
            dlsp = 0.0
        if sEDF < 0.05 and dlsp < 0:
            dlsp = 0.0

        # Adaptive alpha damping
        if pc.name in prev_dlsp and prev_dlsp[pc.name] != 0.0:
            same_sign = dlsp * prev_dlsp[pc.name] > 0
            if not same_sign:
                alpha[pc.name] *= 0.5
            elif alpha.get(pc.name, 1.0) < 2.0:
                alpha[pc.name] = min(2.0, alpha.get(pc.name, 1.0) * 1.2)

        a_j = alpha.get(pc.name, 1.0)

        # Cap step magnitude
        scaled_step = a_j * dlsp
        max_step = 4.0
        if abs(scaled_step) > max_step:
            scaled_step = max_step * np.sign(scaled_step)

        # Apply step
        log_lam_new = np.log(max(lambdas[pc.name], 1e-10)) + scaled_step
        lambdas_new[pc.name] = float(np.clip(np.exp(log_lam_new), 1e-6, 1e10))
        dlsp_out[pc.name] = dlsp

    return lambdas_new, alpha, dlsp_out


def optimize_scop_efs_reml(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    sample_weight: NDArray,
    offset_arr: NDArray,
    lambdas: dict[str, float],
    estimated_names: set[str],
    *,
    max_reml_iter: int = 20,
    reml_tol: float = 1e-6,
    pirls_tol: float = 1e-6,
    max_pirls_iter: int = 100,
    verbose: bool = False,
    reml_penalties: list[PenaltyComponent] | None = None,
    convergence: str = "deviance",
    _scop_joint: bool = True,
    debug_recorder=None,
) -> REMLResult:
    """SCOP-aware EFS REML optimizer for monotone splines.

    Implements the Fellner-Schall fixed-point iteration (Wood & Fasiolo 2017)
    using ``fit_irls_direct`` with SCOP Newton inner solver. Each outer
    iteration:

    1. Fit via ``fit_irls_direct(return_xtwx=True, return_scop_state=True)``
    2. Build the joint Hessian with SCOP Newton blocks replacing linear blocks
    3. Compute EFS lambda updates using SCOP-aware quad and trace terms
    4. Step-damp via REML objective comparison
    5. Check convergence on max abs log-lambda change

    Parameters
    ----------
    dm : DesignMatrix
        Design matrix (discretized for SCOP).
    distribution : Distribution
        GLM family.
    link : Link
        Link function.
    groups : list of GroupSlice
        Group definitions for each feature.
    y : ndarray
        Response vector.
    sample_weight : ndarray
        Sample weights (exposure/frequency).
    offset_arr : ndarray
        Offset vector.
    lambdas : dict
        Initial smoothing parameters keyed by group name.
    estimated_names : set of str
        Names of lambda components to estimate (others held fixed).
    max_reml_iter : int
        Maximum outer EFS iterations.
    reml_tol : float
        Convergence tolerance on max abs log-lambda change.
    pirls_tol : float
        Convergence tolerance for inner IRLS solver.
    max_pirls_iter : int
        Maximum inner IRLS iterations.
    verbose : bool
        Print iteration progress.
    reml_penalties : list of PenaltyComponent, optional
        Pre-built penalty components for non-SCOP terms.
    convergence : str
        PIRLS convergence criterion: 'deviance' or 'coefficients'.

    Returns
    -------
    REMLResult
        Result with estimated lambdas, final PIRLS result, convergence info.
    """
    scale_known = getattr(distribution, "scale_known", True)
    likelihood_size = dispersion_likelihood_size(distribution, sample_weight)
    gamma_scale_data = (
        prepare_gamma_reml_scale_data(y, sample_weight) if isinstance(distribution, Gamma) else None
    )
    fit_context = _SCOPREMLFitContext(
        dm=dm,
        distribution=distribution,
        link=link,
        groups=groups,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        reml_penalties=reml_penalties,
        convergence=convergence,
        scop_joint=_scop_joint,
        debug_recorder=debug_recorder,
        likelihood_size=likelihood_size,
        gamma_scale_data=gamma_scale_data,
    )
    lambdas = lambdas.copy()

    # -- Bootstrap: one IRLS with minimal penalty -> one EFS step --
    # Fixed-policy lambdas keep their value; only estimated components get 1e-4.
    boot_lambdas = {
        name: (1e-4 if name in estimated_names else val) for name, val in lambdas.items()
    }
    boot_mode = _fit_scop_reml_mode(
        fit_context,
        boot_lambdas,
        beta_init=None,
        intercept_init=None,
        scop_state_init=None,
        phase="bootstrap",
        reml_iteration=0,
        require_converged=True,
    )
    if boot_mode is None:
        raise RuntimeError("SCOP REML bootstrap did not converge to a coefficient mode")
    boot_result = boot_mode.result
    boot_scop_states = boot_mode.scop_states
    all_pcs = boot_mode.penalty_components
    H_joint_inv_boot = boot_mode.hessian_inverse
    boot_evaluation = boot_mode.evaluation
    boot_phi = _reml_evaluation_phi(
        boot_evaluation,
        scale_known=scale_known,
        fallback_likelihood_size=fit_context.likelihood_size,
    )

    # One EFS step on bootstrap beta — uses rEDF/pSp formula for ALL terms
    # (including SCOP). This gives SCOP lambdas their first meaningful move.
    boot_alpha = {name: 1.0 for name in estimated_names}
    boot_lambdas_new, _, _ = _joint_efs_lambda_step(
        all_pcs,
        boot_result.beta,
        H_joint_inv_boot,
        boot_phi,
        {pc.name: boot_lambdas.get(pc.name, 1e-4) for pc in all_pcs},
        estimated_names,
        boot_scop_states,
        boot_alpha,
        {},
    )
    for name in estimated_names:
        if name in boot_lambdas_new:
            lambdas[name] = boot_lambdas_new[name]

    if verbose:
        lam_str = ", ".join(f"{pc.name}={lambdas[pc.name]:.4g}" for pc in all_pcs)
        print(f"  SCOP REML bootstrap: lambdas=[{lam_str}]")

    # -- Main EFS loop --
    lambda_history: list[dict[str, float]] = [lambdas.copy()]
    converged = False
    termination_reason = "max_reml_iter"
    n_reml_iter = 0
    warm_beta: NDArray | None = boot_result.beta.copy()
    warm_intercept: float = float(boot_result.intercept)
    warm_scop_states: dict[int, dict] | None = boot_scop_states if boot_scop_states else None
    retained_mode: _SCOPREMLMode | None = None
    current_mode: _SCOPREMLMode | None = None

    # Convergence diagnostics
    inner_iter_history: list[int] = []
    objective_history: list[float] = []
    scop_step_norms_history: list[dict[str, float]] = []
    total_fisher_fallbacks = 0
    managed_cleanup_active_history: list[list[str]] = []
    managed_cleanup_frozen_history: list[list[str]] = []
    managed_cleanup_freeze_iter: int | None = None

    # Adaptive EFS step state (per-component)
    efs_alpha: dict[str, float] = {name: 1.0 for name in estimated_names}
    efs_prev_dlsp: dict[str, float] = {}

    scop_term_count = sum(1 for group in groups if group.monotone_engine == "scop")
    managed_cleanup_names = _multi_scop_discrete_cleanup_names(
        estimated_names=estimated_names,
        scop_states=boot_scop_states,
        scop_term_count=scop_term_count,
    )
    managed_cleanup_active = bool(managed_cleanup_names)
    active_names: set[str] = set(estimated_names)
    frozen_names: set[str] = set()
    stable_counts: dict[str, int] = {name: 0 for name in managed_cleanup_names}

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        # Step 1: Fit the current lambda mode, unless the preceding accepted
        # line-search trial already produced this exact coherent state.
        if retained_mode is None:
            current_mode = _fit_scop_reml_mode(
                fit_context,
                lambdas,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                scop_state_init=warm_scop_states,
                phase="candidate",
                reml_iteration=n_reml_iter,
                require_converged=True,
            )
            if current_mode is None:
                raise RuntimeError("SCOP REML candidate did not converge to a coefficient mode")
        else:
            current_mode = retained_mode
            retained_mode = None

        result = current_mode.result
        scop_states = current_mode.scop_states
        beta = result.beta
        inner_iter_history.append(result.n_iter)

        # Collect SCOP diagnostics from this inner fit
        step_norms_this_iter: dict[str, float] = {}
        for gi, st in scop_states.items():
            step_norms_this_iter[st["group_name"]] = st.get("last_step_norm", 0.0)
            if st.get("last_fisher_fallback", False):
                total_fisher_fallbacks += 1
        scop_step_norms_history.append(step_norms_this_iter)

        # Steps 2--5: Reuse the penalty, latent Hessian, and LAML evaluation
        # assembled from the same fitted coefficient mode.
        H_joint_inv = current_mode.hessian_inverse
        all_pcs = current_mode.penalty_components
        current_evaluation = current_mode.evaluation
        phi = _reml_evaluation_phi(
            current_evaluation,
            scale_known=scale_known,
            fallback_likelihood_size=fit_context.likelihood_size,
        )

        # Step 6: Joint EFS lambda update (rEDF/pSp, scasm-style)
        # Only update components in active_names (frozen ones are skipped)
        lambdas_new, efs_alpha, raw_dlsp = _joint_efs_lambda_step(
            all_pcs,
            beta,
            H_joint_inv,
            phi,
            lambdas,
            active_names,
            scop_states,
            efs_alpha,
            efs_prev_dlsp,
        )

        # Step 7: Fit and score bounded log-scale trials.  Every candidate's
        # coefficients, SCOP blocks, penalty, and LAML geometry share the same
        # lambda state; a failed search retains the exact current mode.
        obj_curr = current_mode.objective
        objective_history.append(float(obj_curr))
        retained_mode, candidate_accepted = _backtrack_scop_efs_candidate(
            fit_context,
            current_mode,
            lambdas_new,
            reml_iteration=n_reml_iter,
        )
        lambdas_new = retained_mode.lambdas.copy()
        obj_after = retained_mode.objective

        # Update prev_dlsp from ACCEPTED (post-damping) step
        for name in estimated_names:
            if name in lambdas_new and name in lambdas:
                accepted_step = np.log(max(lambdas_new[name], 1e-10)) - np.log(
                    max(lambdas[name], 1e-10)
                )
                efs_prev_dlsp[name] = accepted_step

        # Step 7b: Multi-SCOP discrete cleanup — freeze floor-pinned components
        # after they have been stable for several accepted lambda updates.
        if managed_cleanup_active and candidate_accepted:
            frozen_names_before = set(frozen_names)
            managed_active_names = managed_cleanup_names - frozen_names
            stable_counts = _update_multi_scop_discrete_stability_counts(
                lambdas_old=lambdas,
                lambdas_new=lambdas_new,
                active_names=managed_active_names,
                stable_counts=stable_counts,
            )
            managed_active_names, frozen_names = _freeze_multi_scop_discrete_lambdas(
                active_names=managed_active_names,
                frozen_names=frozen_names,
                lambdas_new=lambdas_new,
                stable_counts=stable_counts,
            )
            frozen_names &= managed_cleanup_names
            active_names = set(estimated_names) - frozen_names
            # Record the first 1-based outer iteration where the frozen set
            # changes relative to the previous accepted iteration.
            if managed_cleanup_freeze_iter is None and frozen_names != frozen_names_before:
                managed_cleanup_freeze_iter = n_reml_iter
            # Histories store accepted post-update snapshots for this outer step.
            managed_cleanup_active_history.append(sorted(managed_cleanup_names - frozen_names))
            managed_cleanup_frozen_history.append(sorted(frozen_names))
        elif managed_cleanup_active:
            # A rejected line search retains the active set and must not age a
            # floor-stability counter as though a lambda update were accepted.
            active_names = set(estimated_names) - frozen_names
            managed_cleanup_active_history.append(sorted(managed_cleanup_names - frozen_names))
            managed_cleanup_frozen_history.append(sorted(frozen_names))
        else:
            active_names = set(estimated_names)
            frozen_names.clear()

        # Step 8: Convergence check — strict tolerance OR objective plateau
        # Strict convergence still checks the accepted update across all
        # estimated components, including any names that were frozen earlier.
        changes = [
            abs(np.log(lambdas_new[pc.name]) - np.log(lambdas[pc.name]))
            for pc in all_pcs
            if pc.name in lambdas
            and pc.name in lambdas_new
            and lambdas[pc.name] > 0
            and lambdas_new[pc.name] > 0
        ]
        max_change = max(changes) if changes else 0.0

        # Plateau detection: objective flat and lambda changes small
        obj_rel_change = 0.0
        if len(objective_history) >= 2:
            obj_prev = objective_history[-2]
            obj_curr_val = objective_history[-1]
            obj_rel_change = abs(obj_curr_val - obj_prev) / max(abs(obj_curr_val), 1.0)

        # Converge on strict lambda tolerance
        strict_converged = candidate_accepted and max_change < reml_tol
        if managed_cleanup_active:
            plateau_converged = (
                candidate_accepted
                and n_reml_iter >= 3
                and _multi_scop_discrete_plateau_converged(
                    obj_rel_change=obj_rel_change,
                    lambdas_old=lambdas,
                    lambdas_new=lambdas_new,
                    active_names=active_names,
                )
            )
        else:
            plateau_converged = (
                candidate_accepted
                and n_reml_iter >= 3
                and obj_rel_change < 1e-6
                and max_change < 0.01
            )

        if verbose:
            lam_str = ", ".join(f"{pc.name}={lambdas_new[pc.name]:.4g}" for pc in all_pcs)
            print(
                f"  SCOP REML iter={n_reml_iter}  max_change={max_change:.6f}"
                f"  obj_rel={obj_rel_change:.2e}  lambdas=[{lam_str}]"
            )

        if debug_recorder is not None and getattr(debug_recorder, "enabled_level", 0) >= 2:
            debug_recorder.append_jsonl(
                "reml",
                {
                    "iteration": n_reml_iter,
                    "objective_before": float(obj_curr),
                    "objective_after": float(obj_after),
                    "lambda_max_delta": float(max_change),
                    "objective_relative_change": float(obj_rel_change),
                    "strict_converged": bool(strict_converged),
                    "plateau_converged": bool(plateau_converged),
                    "candidate_accepted": bool(candidate_accepted),
                    "estimated_names": sorted(estimated_names),
                    "active_names": sorted(active_names),
                    "frozen_names": sorted(frozen_names),
                    "lambdas": {name: float(value) for name, value in lambdas_new.items()},
                },
            )

        lambda_history.append(lambdas_new.copy())

        if not candidate_accepted:
            termination_reason = "line_search_stalled"
            lambdas = lambdas_new
            break

        if strict_converged or plateau_converged:
            converged = True
            termination_reason = "lambda_tolerance" if strict_converged else "objective_plateau"
            lambdas = lambdas_new
            break

        # Step 9: Warm start for next iteration
        lambdas = lambdas_new
        warm_beta = retained_mode.result.beta.copy()
        warm_intercept = float(retained_mode.result.intercept)
        warm_scop_states = retained_mode.scop_states if retained_mode.scop_states else None

    # -- Terminal mode --
    # An accepted line-search state has already paid for both its converged
    # coefficient fit and its coherent LAML evaluation.  Reuse it directly;
    # the fallback is needed only when the outer loop did not run.
    final_mode = retained_mode
    if final_mode is None or final_mode.lambdas != lambdas:
        if current_mode is not None and current_mode.lambdas == lambdas:
            final_mode = current_mode
        else:
            final_mode = _fit_scop_reml_mode(
                fit_context,
                lambdas,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                scop_state_init=warm_scop_states,
                phase="final",
                reml_iteration=n_reml_iter,
                require_converged=True,
            )
            if final_mode is None:
                raise RuntimeError("SCOP REML final fit did not converge to a coefficient mode")

    final_result = _finalize_scop_reml_mode(fit_context, final_mode)
    final_scop_states = final_mode.scop_states
    final_all_pcs = final_mode.penalty_components
    final_evaluation = final_mode.evaluation

    return REMLResult(
        lambdas=lambdas,
        pirls_result=final_result,
        n_reml_iter=n_reml_iter,
        converged=converged,
        lambda_history=lambda_history,
        reml_penalties=final_all_pcs,
        scop_states=final_scop_states if final_scop_states else None,
        objective=float(final_evaluation.value),
        inner_iter_history=inner_iter_history,
        objective_history=objective_history,
        curvature_source=final_mode.curvature_source,
        termination_reason=termination_reason,
        scop_step_norms=scop_step_norms_history if scop_step_norms_history else None,
        scop_fisher_fallbacks=total_fisher_fallbacks,
        managed_cleanup_names=sorted(managed_cleanup_names) if managed_cleanup_names else None,
        managed_cleanup_frozen_names=sorted(frozen_names) if managed_cleanup_active else None,
        managed_cleanup_freeze_iter=managed_cleanup_freeze_iter,
        managed_cleanup_active_history=(
            managed_cleanup_active_history if managed_cleanup_active_history else None
        ),
        managed_cleanup_frozen_history=(
            managed_cleanup_frozen_history if managed_cleanup_frozen_history else None
        ),
    )
