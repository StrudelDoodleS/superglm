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
