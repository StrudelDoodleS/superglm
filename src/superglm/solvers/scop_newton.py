"""Stabilized full Newton solver for SCOP (Shape Constrained P-spline) terms.

One damped Newton step on the penalized WLS objective in SCOP solver space.

The objective is:
    L = 0.5 * sum(W * (z - B @ gamma_eff(beta))^2) + 0.5 * lambda * beta^T S beta

where gamma_eff = reparam.forward(beta) is the nonlinear SCOP map through
the exp-cumsum chain.

Full Newton (not Gauss-Newton) is required because Fisher scoring causes
convergence problems for SCOP terms (Pya & Wood 2015, p. 6, lines 470-476).
The second-order terms from d^2(gamma_eff)/d(beta)^2 are included in the
Hessian. If the resulting Hessian is not positive definite, we fall back to
Fisher scoring (Gauss-Newton) for that iteration only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from superglm.solvers.scop import SCOPSolverReparam


@dataclass
class SCOPNewtonResult:
    """Result of a single SCOP Newton step."""

    beta_new: NDArray
    """Updated solver-space parameters (q_eff,)."""

    objective_before: float
    """Penalized WLS objective before the step."""

    objective_after: float
    """Penalized WLS objective after the step."""

    step_norm: float
    """L2 norm of the parameter update."""

    used_fisher_fallback: bool
    """True if the full Newton Hessian was not PD and Fisher scoring was used."""

    H_penalized: NDArray | None = None
    """(q_eff, q_eff) penalized Hessian from the Newton step, or None."""


def _objective(
    B_scop: NDArray,
    W: NDArray,
    z: NDArray,
    beta_scop: NDArray,
    reparam: SCOPSolverReparam,
    S_scop: NDArray,
    lambda2: float,
    bin_idx: NDArray | None = None,
) -> float:
    """Penalized WLS objective for SCOP term.

    When *bin_idx* is provided, *B_scop* is ``(n_bins, q_eff)`` (bin-level
    design) while *W* and *z* are observation-level ``(n,)``.  Predictions
    are scattered via ``eta = (B_scop @ gamma_eff)[bin_idx]``.
    """
    gamma_eff = reparam.forward(beta_scop)
    eta = B_scop @ gamma_eff
    if bin_idx is not None:
        eta = eta[bin_idx]
    residual = z - eta
    data_term = 0.5 * np.sum(W * residual**2)
    penalty_term = 0.5 * lambda2 * (beta_scop @ S_scop @ beta_scop)
    return float(data_term + penalty_term)


def scop_newton_step(
    B_scop: NDArray,
    W: NDArray,
    z: NDArray,
    beta_scop: NDArray,
    reparam: SCOPSolverReparam,
    S_scop: NDArray,
    lambda2: float,
    max_halving: int = 10,
    bin_idx: NDArray | None = None,
) -> SCOPNewtonResult:
    """One damped full Newton step on the SCOP penalized WLS objective.

    Parameters
    ----------
    B_scop : (n, q_eff) or (n_bins, q_eff)
        Centered design matrix for the SCOP term.  When *bin_idx* is provided
        this is the bin-level design ``(n_bins, q_eff)``.
    W : (n,)
        IRLS working weights (always observation-level).
    z : (n,)
        Working response (always observation-level).
    beta_scop : (q_eff,)
        Current SCOP solver-space parameters.
    reparam : SCOPSolverReparam
        Solver-space reparameterization (forward, jacobian).
    S_scop : (q_eff, q_eff)
        SCOP penalty matrix in solver space.
    lambda2 : float
        Penalty weight (non-negative).
    max_halving : int
        Maximum number of step halvings for line search.
    bin_idx : (n,) array of int or None
        If provided, *B_scop* is ``(n_bins, q_eff)`` and predictions are
        scattered: ``eta = (B_scop @ gamma)[bin_idx]``.  Gradient and Hessian
        are computed via bin-level aggregation to avoid the full ``(n, q_eff)``
        matrix.

    Returns
    -------
    SCOPNewtonResult
        Updated parameters and diagnostics.
    """
    q_eff = len(beta_scop)
    beta = beta_scop.copy()

    # --- Forward map and residual ---
    # forward() = exp(beta), jacobian is diagonal: diag(exp(beta))
    gamma_eff = reparam.forward(beta)  # exp(beta), (q_eff,)
    j_diag = gamma_eff  # d(exp(b))/db = exp(b) = gamma_eff

    eta_bin = B_scop @ gamma_eff  # (n_bins,) or (n,)
    if bin_idx is not None:
        eta = eta_bin[bin_idx]  # scatter to (n,)
    else:
        eta = eta_bin
    residual = z - eta

    penalty_term = 0.5 * lambda2 * (beta @ S_scop @ beta)
    obj_before = 0.5 * np.sum(W * residual**2) + penalty_term

    # --- Weighted design products (exploit diagonal J) ---
    # J_eff = diag(j_diag), so J^T @ BtWB @ J = diag(j) @ BtWB @ diag(j)
    # and J^T @ r_eff = j_diag * r_eff (elementwise)
    if bin_idx is not None:
        n_bins = B_scop.shape[0]
        W_agg = np.bincount(bin_idx, weights=W, minlength=n_bins)
        Wr_agg = np.bincount(bin_idx, weights=W * residual, minlength=n_bins)
        BtWB = B_scop.T @ (B_scop * W_agg[:, None])  # (q_eff, q_eff)
        r_eff = B_scop.T @ Wr_agg  # (q_eff,)
    else:
        BtW = B_scop.T * W[np.newaxis, :]
        BtWB = BtW @ B_scop
        r_eff = BtW @ residual

    # --- Gradient (exploit diagonal J: J^T @ r = j * r) ---
    grad_data = -(j_diag * r_eff)  # elementwise, not matrix multiply
    grad = grad_data + lambda2 * (S_scop @ beta)

    # --- Gauss-Newton Hessian (exploit diagonal J: J^T BtWB J = j j^T * BtWB) ---
    jj = j_diag[:, None] * j_diag[None, :]  # (q_eff, q_eff) outer product
    H_gn = jj * BtWB + lambda2 * S_scop  # elementwise Hadamard, not matmul

    # --- Second-order correction (full Newton) ---
    # H_second[i,i] = grad_data[i] (diagonal correction)
    H_full = H_gn + np.diag(grad_data)

    # --- Attempt full Newton step ---
    used_fisher = False
    step = _solve_step(H_full, grad)
    if step is None:
        used_fisher = True
        H_fisher = H_gn + 1e-8 * np.eye(q_eff)
        step = _solve_step(H_fisher, grad)
        if step is None:
            H_fisher += 1e-4 * np.eye(q_eff)
            step = _solve_step(H_fisher, grad)
            if step is None:
                step = -1e-4 * grad

    # --- Damped step (step halving) ---
    # Avoid recomputing full _objective() — inline the cheap parts.
    # Only the data term changes; penalty term uses beta_new.
    alpha = 1.0
    beta_new = beta - alpha * step
    gamma_new = reparam.forward(beta_new)
    eta_bin_new = B_scop @ gamma_new
    if bin_idx is not None:
        eta_new = eta_bin_new[bin_idx]
    else:
        eta_new = eta_bin_new
    res_new = z - eta_new
    obj_new = 0.5 * np.sum(W * res_new**2) + 0.5 * lambda2 * (beta_new @ S_scop @ beta_new)

    for _ in range(max_halving):
        if obj_new <= obj_before + 1e-14:
            break
        alpha *= 0.5
        beta_new = beta - alpha * step
        gamma_new = reparam.forward(beta_new)
        eta_bin_new = B_scop @ gamma_new
        if bin_idx is not None:
            eta_new = eta_bin_new[bin_idx]
        else:
            eta_new = eta_bin_new
        res_new = z - eta_new
        obj_new = 0.5 * np.sum(W * res_new**2) + 0.5 * lambda2 * (beta_new @ S_scop @ beta_new)

    step_norm = float(np.linalg.norm(alpha * step))

    return SCOPNewtonResult(
        beta_new=beta_new,
        objective_before=float(obj_before),
        objective_after=float(obj_new),
        step_norm=step_norm,
        used_fisher_fallback=used_fisher,
        H_penalized=H_full,
    )


def _solve_step(H: NDArray, grad: NDArray) -> NDArray | None:
    """Solve H @ step = grad via Cholesky. Returns None if H is not PD."""
    try:
        L, low = cho_factor(H)
        return cho_solve((L, low), grad)
    except np.linalg.LinAlgError:
        return None
