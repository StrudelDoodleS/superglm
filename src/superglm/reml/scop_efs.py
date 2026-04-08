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

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DesignMatrix
from superglm.reml.objective import reml_laml_objective
from superglm.reml.penalty_algebra import compute_total_penalty_rank
from superglm.reml.result import REMLResult
from superglm.solvers.irls_direct import _build_penalty_matrix, _safe_decompose_H, fit_irls_direct
from superglm.types import GroupSlice, PenaltyComponent


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
    eps_thresh = np.finfo(float).eps ** (2 / 3)
    components = []

    for gi, st in scop_states.items():
        S_scop = st["S_scop"]
        eigvals = np.linalg.eigvalsh(S_scop)
        thresh = eps_thresh * max(eigvals.max(), 1e-12)
        rank = float(np.sum(eigvals > thresh))
        n_pos = int(rank)

        if n_pos > 0:
            sorted_eig = np.sort(eigvals)[::-1]
            pos_eigvals = sorted_eig[:n_pos]
            log_det = float(np.sum(np.log(np.maximum(pos_eigvals, 1e-300))))
        else:
            pos_eigvals = np.array([])
            log_det = 0.0

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


def compute_scop_aware_penalty_quad(
    result_beta: NDArray,
    S: NDArray,
    scop_states: dict[int, dict],
    lambdas: dict[str, float],
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
        S_scop = st["S_scop"]
        beta_eff = st["beta_eff"]
        gamma_eff = result_beta[sl]
        lam = lambdas.get(st["group_name"], 0.0)

        # Subtract wrong gamma-space contribution
        pq -= float(gamma_eff @ (lam * S_scop) @ gamma_eff)
        # Add correct beta_eff-space contribution
        pq += lam * float(beta_eff @ S_scop @ beta_eff)

    return pq


def assemble_joint_hessian(
    XtWX_plus_S: NDArray,
    scop_states: dict[int, dict],
) -> tuple[NDArray, dict[str, slice]]:
    """Assemble joint Hessian in the beta_eff coordinate system.

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
    if not scop_states:
        return XtWX_plus_S, {}

    p = XtWX_plus_S.shape[0]
    H_joint = XtWX_plus_S.copy()
    mapping = {}

    # Collect all SCOP indices so we can identify "other" indices
    scop_slices = []
    for gi, st in scop_states.items():
        scop_slices.append(st["group_sl"])

    all_scop_idx = np.concatenate([np.arange(sl.start, sl.stop) for sl in scop_slices])
    other_idx = np.setdiff1d(np.arange(p), all_scop_idx)

    for gi, st in scop_states.items():
        sl = st["group_sl"]
        H_scop = st["H_scop_penalized"]
        name = st["group_name"]
        beta_eff = st["beta_eff"]
        j_diag = np.exp(np.clip(beta_eff, -500, 500))

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
        j_a = np.exp(np.clip(st_a["beta_eff"], -500, 500))
        for idx_b in range(idx_a + 1, len(scop_items)):
            gi_b, st_b = scop_items[idx_b]
            sl_b = st_b["group_sl"]
            j_b = np.exp(np.clip(st_b["beta_eff"], -500, 500))
            idx_a_arr = np.arange(sl_a.start, sl_a.stop)
            idx_b_arr = np.arange(sl_b.start, sl_b.stop)
            H_joint[np.ix_(idx_a_arr, idx_b_arr)] *= j_a[:, np.newaxis] * j_b[np.newaxis, :]
            H_joint[np.ix_(idx_b_arr, idx_a_arr)] *= j_b[:, np.newaxis] * j_a[np.newaxis, :]

    return H_joint, mapping


def _is_scop_component(pc: PenaltyComponent, scop_states: dict[int, dict]) -> dict | None:
    """Return SCOP state dict if pc corresponds to a SCOP group, else None."""
    for gi, st in scop_states.items():
        if st["group_name"] == pc.name:
            return st
    return None


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

        # Residual EDF — keep raw for suppression check, floor for log
        rEDF_raw = pc.rank - lambdas[pc.name] * sEDF
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
    n = len(y)
    lambdas = lambdas.copy()

    # -- Bootstrap: one IRLS with minimal penalty -> one EFS step --
    # Fixed-policy lambdas keep their value; only estimated components get 1e-4.
    boot_lambdas = {
        name: (1e-4 if name in estimated_names else val) for name, val in lambdas.items()
    }
    boot_out = fit_irls_direct(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=boot_lambdas,
        offset=offset_arr,
        tol=pirls_tol,
        max_iter=max_pirls_iter,
        return_xtwx=True,
        return_scop_state=True,
        reml_penalties=reml_penalties,
        convergence=convergence,
        _scop_joint=_scop_joint,
    )

    # Unpack: with return_xtwx=True and SCOP -> 4-tuple
    if len(boot_out) == 4:
        boot_result, boot_H_inv, boot_XtWX, boot_scop_states = boot_out
    else:
        boot_result, boot_H_inv, boot_XtWX = boot_out
        boot_scop_states = {}

    # Build penalty matrix
    S_boot = _build_penalty_matrix(
        dm.group_matrices, groups, boot_lambdas, dm.p, reml_penalties=reml_penalties
    )

    # Build SCOP penalty components and merge with reml_penalties
    scop_pcs = build_scop_penalty_components(boot_scop_states)
    all_pcs = list(reml_penalties or []) + scop_pcs

    # Assemble joint Hessian for bootstrap
    H_joint_boot, _ = assemble_joint_hessian(boot_XtWX + S_boot, boot_scop_states)
    H_joint_inv_boot, _, _ = _safe_decompose_H(H_joint_boot)

    # Estimate phi for estimated-scale families
    boot_inv_phi = 1.0
    if not scale_known:
        pq_boot = compute_scop_aware_penalty_quad(
            boot_result.beta, S_boot, boot_scop_states, boot_lambdas
        )
        M_p = compute_total_penalty_rank(all_pcs)
        boot_phi = max((boot_result.deviance + pq_boot) / max(n - M_p, 1.0), 1e-10)
        boot_inv_phi = 1.0 / boot_phi

    # One EFS step on bootstrap beta — uses rEDF/pSp formula for ALL terms
    # (including SCOP). This gives SCOP lambdas their first meaningful move.
    boot_alpha = {name: 1.0 for name in estimated_names}
    boot_lambdas_new, _, _ = _joint_efs_lambda_step(
        all_pcs,
        boot_result.beta,
        H_joint_inv_boot,
        1.0 / boot_inv_phi if boot_inv_phi > 0 else 1.0,
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
    n_reml_iter = 0
    warm_beta: NDArray | None = boot_result.beta.copy()
    warm_intercept: float = float(boot_result.intercept)
    warm_scop_states: dict[int, dict] | None = boot_scop_states if boot_scop_states else None

    # Convergence diagnostics
    inner_iter_history: list[int] = []
    objective_history: list[float] = []
    scop_step_norms_history: list[dict[str, float]] = []
    total_fisher_fallbacks = 0

    # Adaptive EFS step state (per-component)
    efs_alpha: dict[str, float] = {name: 1.0 for name in estimated_names}
    efs_prev_dlsp: dict[str, float] = {}

    # Staged freeze state: temporarily freeze converged SSP lambdas,
    # iterate SCOP-only, then unfreeze for final joint cleanup.
    active_names: set[str] = set(estimated_names)
    frozen_names: set[str] = set()
    _unfreeze_pending = False
    _unfreeze_scheduled = False

    for reml_iter in range(max_reml_iter):
        n_reml_iter = reml_iter + 1

        # Step 1: Inner fit with SCOP state
        irls_out = fit_irls_direct(
            X=dm,
            y=y,
            weights=sample_weight,
            family=distribution,
            link=link,
            groups=groups,
            lambda2=lambdas,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
            tol=pirls_tol,
            max_iter=max_pirls_iter,
            return_xtwx=True,
            return_scop_state=True,
            reml_penalties=reml_penalties,
            convergence=convergence,
            _scop_joint=_scop_joint,
            scop_state_init=warm_scop_states,
        )

        if len(irls_out) == 4:
            result, H_inv, XtWX, scop_states = irls_out
        else:
            result, H_inv, XtWX = irls_out
            scop_states = {}

        beta = result.beta
        intercept = result.intercept
        inner_iter_history.append(result.n_iter)

        # Collect SCOP diagnostics from this inner fit
        step_norms_this_iter: dict[str, float] = {}
        for gi, st in scop_states.items():
            step_norms_this_iter[st["group_name"]] = st.get("last_step_norm", 0.0)
            if st.get("last_fisher_fallback", False):
                total_fisher_fallbacks += 1
        scop_step_norms_history.append(step_norms_this_iter)

        # Step 2: Build penalty matrix
        S = _build_penalty_matrix(
            dm.group_matrices, groups, lambdas, dm.p, reml_penalties=reml_penalties
        )

        # Step 3: Joint Hessian
        H_joint, _ = assemble_joint_hessian(XtWX + S, scop_states)
        H_joint_inv, log_det_H_joint, _ = _safe_decompose_H(H_joint)

        # Step 4: Build SCOP PCs and merge
        scop_pcs = build_scop_penalty_components(scop_states)
        all_pcs = list(reml_penalties or []) + scop_pcs

        # Step 5: Estimate phi for estimated-scale families
        inv_phi = 1.0
        if not scale_known:
            pq = compute_scop_aware_penalty_quad(beta, S, scop_states, lambdas)
            M_p = compute_total_penalty_rank(all_pcs)
            phi_hat = max((result.deviance + pq) / max(n - M_p, 1.0), 1e-10)
            inv_phi = 1.0 / phi_hat

        # Step 6: Joint EFS lambda update (rEDF/pSp, scasm-style)
        # Only update components in active_names (frozen ones are skipped)
        phi = 1.0 / inv_phi if inv_phi > 0 else 1.0
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

        # Step 7: Uphill-step guard via REML objective comparison
        obj_curr = reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            result,
            lambdas,
            sample_weight,
            offset_arr,
            XtWX=XtWX,
            reml_penalties=all_pcs,
            scop_states=scop_states,
        )
        objective_history.append(float(obj_curr))
        obj_trial = reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            result,
            lambdas_new,
            sample_weight,
            offset_arr,
            XtWX=XtWX,
            reml_penalties=all_pcs,
            scop_states=scop_states,
        )
        if obj_trial > obj_curr + 1e-8 * max(abs(obj_curr), 1.0):
            # Half-step on log scale
            for pc in all_pcs:
                if pc.name not in estimated_names:
                    continue
                log_old = np.log(max(lambdas[pc.name], 1e-10))
                log_new = np.log(max(lambdas_new[pc.name], 1e-10))
                lambdas_new[pc.name] = float(np.clip(np.exp(0.5 * (log_old + log_new)), 1e-6, 1e10))

        # Update prev_dlsp from ACCEPTED (post-damping) step
        for name in estimated_names:
            if name in lambdas_new and name in lambdas:
                accepted_step = np.log(max(lambdas_new[name], 1e-10)) - np.log(
                    max(lambdas[name], 1e-10)
                )
                efs_prev_dlsp[name] = accepted_step

        # Step 7b: Staged freezing — temporarily freeze converged components
        # so SCOP-only iterations are cheap, then unfreeze for final cleanup.
        freeze_threshold = 0.001  # log-scale
        if n_reml_iter >= 3 and not _unfreeze_scheduled:
            newly_frozen = []
            for name in list(active_names):
                if name in lambdas_new and name in lambdas:
                    ch = abs(
                        np.log(max(lambdas_new[name], 1e-10)) - np.log(max(lambdas[name], 1e-10))
                    )
                    if ch < freeze_threshold:
                        newly_frozen.append(name)
            for name in newly_frozen:
                active_names.discard(name)
                frozen_names.add(name)

            # If we have frozen components and only SCOP remains active,
            # schedule an unfreeze after SCOP stabilizes to re-couple
            if frozen_names and active_names and len(active_names) < len(estimated_names):
                _unfreeze_pending = True

        # Step 8: Convergence check — strict tolerance OR objective plateau
        # Use all components (not just still-estimated) for convergence decision
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

        # Unfreeze: if SCOP-only iterations have stabilized, re-couple for final joint cleanup
        if _unfreeze_pending and max_change < 0.01 and n_reml_iter >= 5:
            active_names = set(estimated_names)
            frozen_names.clear()
            _unfreeze_scheduled = True
            _unfreeze_pending = False

        # Converge on strict lambda tolerance
        strict_converged = max_change < reml_tol
        # OR: objective plateau — relative objective change < 1e-6 AND
        # lambda changes < 0.01 (1% on log scale) for at least 2 iterations
        plateau_converged = n_reml_iter >= 3 and obj_rel_change < 1e-6 and max_change < 0.01

        if verbose:
            lam_str = ", ".join(f"{pc.name}={lambdas_new[pc.name]:.4g}" for pc in all_pcs)
            print(
                f"  SCOP REML iter={n_reml_iter}  max_change={max_change:.6f}"
                f"  obj_rel={obj_rel_change:.2e}  lambdas=[{lam_str}]"
            )

        lambda_history.append(lambdas_new.copy())

        if strict_converged or plateau_converged:
            converged = True
            lambdas = lambdas_new
            break

        # Step 9: Warm start for next iteration
        lambdas = lambdas_new
        warm_beta = beta.copy()
        warm_intercept = float(intercept)
        warm_scop_states = scop_states if scop_states else None

    # -- Final refit --
    final_out = fit_irls_direct(
        X=dm,
        y=y,
        weights=sample_weight,
        family=distribution,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset_arr,
        beta_init=warm_beta,
        intercept_init=warm_intercept,
        tol=pirls_tol,
        max_iter=max_pirls_iter,
        return_xtwx=True,
        return_scop_state=True,
        reml_penalties=reml_penalties,
        convergence=convergence,
        _scop_joint=_scop_joint,
        scop_state_init=warm_scop_states,
    )

    if len(final_out) == 4:
        final_result, _, final_XtWX, final_scop_states = final_out
    else:
        final_result, _, final_XtWX = final_out
        final_scop_states = {}

    # Build final PCs for objective
    final_scop_pcs = build_scop_penalty_components(final_scop_states)
    final_all_pcs = list(reml_penalties or []) + final_scop_pcs

    return REMLResult(
        lambdas=lambdas,
        pirls_result=final_result,
        n_reml_iter=n_reml_iter,
        converged=converged,
        lambda_history=lambda_history,
        reml_penalties=final_all_pcs,
        scop_states=final_scop_states if final_scop_states else None,
        objective=reml_laml_objective(
            dm,
            distribution,
            link,
            groups,
            y,
            final_result,
            lambdas,
            sample_weight,
            offset_arr,
            XtWX=final_XtWX,
            reml_penalties=final_all_pcs,
            scop_states=final_scop_states,
        ),
        inner_iter_history=inner_iter_history,
        objective_history=objective_history,
        scop_step_norms=scop_step_norms_history if scop_step_norms_history else None,
        scop_fisher_fallbacks=total_fisher_fallbacks,
    )
