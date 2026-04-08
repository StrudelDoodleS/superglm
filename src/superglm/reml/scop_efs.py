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

import numpy as np
from numpy.typing import NDArray

from superglm.types import PenaltyComponent


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
    """Assemble block-diagonal joint Hessian from linear + SCOP blocks.

    The XtWX_plus_S matrix already has the SCOP block filled with
    lambda * S_scop from _build_penalty_matrix. We REPLACE that block
    with H_scop_penalized (the full Newton Hessian including data-term
    curvature). Cross-terms between linear and SCOP blocks are not
    included -- coupling is handled by the IRLS outer loop.

    Parameters
    ----------
    XtWX_plus_S : (p, p) ndarray
        The linear-system penalized Gram matrix.
    scop_states : dict
        SCOP converged state dict, keyed by group index. Each value
        must contain "group_sl", "H_scop_penalized", "group_name".

    Returns
    -------
    H_joint : (p, p) ndarray
        Joint Hessian with SCOP blocks replaced.
    mapping : dict
        Maps group_name to the slice in H_joint for each SCOP group.
    """
    if not scop_states:
        return XtWX_plus_S, {}

    H_joint = XtWX_plus_S.copy()
    mapping = {}

    for gi, st in scop_states.items():
        sl = st["group_sl"]
        H_scop = st["H_scop_penalized"]
        name = st["group_name"]
        H_joint[sl, sl] = H_scop
        mapping[name] = sl

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

    For SSP components: quad = beta_g^T @ omega_g @ beta_g (gamma space).
    For SCOP components: quad = beta_eff^T @ S_scop @ beta_eff (solver space).
    Trace term always uses H_joint_inv sliced at pc.group_sl.

    Parameters
    ----------
    pc : PenaltyComponent
        Penalty component (SSP or SCOP).
    beta : (p,) full coefficient vector (gamma for SCOP groups).
    H_joint_inv : (p, p) inverse of joint Hessian.
    inv_phi : 1/phi (scale parameter inverse).
    lam_old : current lambda for this component.
    scop_states : SCOP converged state dict.

    Returns
    -------
    lam_new : updated lambda (with uphill guard applied).
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

    # Uphill-step guard: clip log-step to [-5, 5]
    log_step = np.log(max(lam_raw, 1e-10)) - np.log(max(lam_old, 1e-10))
    log_step = np.clip(log_step, -5.0, 5.0)
    lam_new = lam_old * float(np.exp(log_step))

    return lam_new
