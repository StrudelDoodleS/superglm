"""From-scratch joint SCOP objective: the oracle the cached and delta objectives are checked against."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm._utils import weighted_row_sum


def safe_joint_objective(
    scop_items: list[tuple[int, dict]],
    W: NDArray,
    z_scop: NDArray,
    beta_joint: NDArray,
    slices: list[slice],
    lambdas_list: list[float],
) -> float:
    """Evaluate joint SCOP objective, returning inf on any non-finite value.

    Parameters
    ----------
    scop_items : list of (group_index, state_dict)
    W : (n,) observation weights
    z_scop : (n,) working response minus non-SCOP eta
    beta_joint : concatenated beta_eff for all groups
    slices : per-group slices into beta_joint
    lambdas_list : per-group lambda values
    """
    with np.errstate(over="ignore", invalid="ignore"):
        total_eta = np.zeros_like(z_scop)
        penalty = 0.0
        for (gi, st), sl_i, lam_i in zip(scop_items, slices, lambdas_list):
            beta_i = beta_joint[sl_i]
            gamma_i = st["reparam"].forward(beta_i)
            if not np.all(np.isfinite(gamma_i)):
                return np.inf
            eta_i = st["B_scop"] @ gamma_i
            if st["bin_idx"] is not None:
                eta_i = eta_i[st["bin_idx"]]
            total_eta += eta_i
            penalty += 0.5 * lam_i * float(beta_i @ st["S_scop"] @ beta_i)
        r = z_scop - total_eta
        data_term = 0.5 * weighted_row_sum(W, np.square, r)
        obj = data_term + penalty
    return float(obj) if np.isfinite(obj) else np.inf
