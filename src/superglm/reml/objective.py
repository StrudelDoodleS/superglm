"""REML/LAML objective function.

Laplace-approximate restricted maximum likelihood objective
(Wood 2011, Section 2, Eqs 4-5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, clip_mu
from superglm.group_matrix import (
    DesignMatrix,
    _block_xtwx,
)
from superglm.links import stabilize_eta
from superglm.reml.penalty_algebra import (
    build_penalty_matrix,
    cached_logdet_s_plus,
    compute_logdet_s_plus,
    compute_total_penalty_rank,
)
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice, PenaltyComponent


def reml_laml_objective(
    dm: DesignMatrix,
    distribution: Any,
    link: Any,
    groups: list[GroupSlice],
    y: NDArray,
    result: PIRLSResult,
    lambdas: dict[str, float],
    sample_weight: NDArray,
    offset_arr: NDArray,
    XtWX: NDArray | None = None,
    penalty_caches: dict | None = None,
    log_det_H: float | None = None,
    S_override: NDArray | None = None,
    reml_penalties: list[PenaltyComponent] | None = None,
    scop_states: dict[int, dict] | None = None,
) -> float:
    """Laplace REML/LAML objective up to additive constants.

    Wood (2011) Section 2, Eqs (4)-(5): V(rho) = -l(beta_hat) + 0.5*beta_hat'S*beta_hat +
    0.5*log|H| - 0.5*log|S|_+. Known-scale: nll + 0.5*(penalty_quad + logdet_m -
    logdet_s). Estimated-scale: phi profiled out -> 0.5*(n-Mp)*log(D+beta_hat'S*beta_hat)
    replaces the nll + 0.5*penalty_quad terms.

    Parameters
    ----------
    log_det_H : float, optional
        Precomputed log|X'WX + S| from ``_safe_decompose_H``.  Avoids
        redundant eigendecomposition when the caller already has it.

    Handles both known-scale families (Poisson, NB2 where phi=1) and
    estimated-scale families (Gamma, Tweedie) via phi-profiled REML.
    """
    eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset_arr, link)
    mu = clip_mu(link.inverse(eta), distribution)
    if XtWX is None:
        V = distribution.variance(mu)
        dmu_deta = link.deriv_inverse(eta)
        W = sample_weight * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)
        XtWX = _block_xtwx(dm.group_matrices, groups, W, tabmat_split=dm.tabmat_split)

    p = XtWX.shape[0]
    if S_override is not None:
        S = S_override
    else:
        S = build_penalty_matrix(
            dm.group_matrices, groups, lambdas, p, reml_penalties=reml_penalties
        )
    if scop_states:
        from superglm.reml.scop_efs import compute_scop_aware_penalty_quad

        penalty_quad = compute_scop_aware_penalty_quad(result.beta, S, scop_states, lambdas)
    else:
        penalty_quad = float(result.beta @ S @ result.beta)

    # log|S|_+ -- use multi-penalty-aware path when reml_penalties available
    if reml_penalties is not None:
        logdet_s = compute_logdet_s_plus(lambdas, reml_penalties)
    elif penalty_caches is not None:
        logdet_s = cached_logdet_s_plus(lambdas, penalty_caches)
    else:
        eigvals_s = np.linalg.eigvalsh(S)
        thresh_s = 1e-10 * max(eigvals_s.max(), 1e-12)
        pos_s = eigvals_s[eigvals_s > thresh_s]
        logdet_s = float(np.sum(np.log(pos_s))) if pos_s.size else 0.0

    # log|H| = log|X'WX + S| -- reuse precomputed value if available
    if log_det_H is not None:
        logdet_m = log_det_H
    elif scop_states:
        from superglm.reml.scop_efs import assemble_joint_hessian

        H_joint, _ = assemble_joint_hessian(XtWX + S, scop_states)
        eigvals_m = np.linalg.eigvalsh(H_joint)
        thresh_m = 1e-10 * max(eigvals_m.max(), 1e-12)
        pos_m = eigvals_m[eigvals_m > thresh_m]
        logdet_m = float(np.sum(np.log(pos_m))) if pos_m.size else 0.0
    else:
        M = XtWX + S
        eigvals_m = np.linalg.eigvalsh(M)
        thresh_m = 1e-10 * max(eigvals_m.max(), 1e-12)
        pos_m = eigvals_m[eigvals_m > thresh_m]
        logdet_m = float(np.sum(np.log(pos_m))) if pos_m.size else 0.0

    # phi-profiled REML for estimated-scale families
    scale_known = getattr(distribution, "scale_known", True)
    if not scale_known:
        n = len(y)
        if reml_penalties is not None:
            M_p = compute_total_penalty_rank(reml_penalties)
        elif penalty_caches is not None:
            M_p = sum(c.rank for c in penalty_caches.values())
        else:
            M_p = float(len(pos_s))
        d_plus_pq = max(result.deviance + penalty_quad, 1e-300)
        scale_term = 0.5 * max(n - M_p, 1.0) * np.log(d_plus_pq)
        return float(0.5 * (logdet_m - logdet_s) + scale_term)

    nll = -distribution.log_likelihood(y, mu, sample_weight, phi=1.0)
    return float(nll + 0.5 * (penalty_quad + logdet_m - logdet_s))
