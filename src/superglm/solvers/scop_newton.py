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

For multi-SCOP models, ``scop_joint_newton_step`` solves for ALL SCOP groups
simultaneously via a joint Hessian with proper cross-group blocks, replacing
the sequential Gauss-Seidel loop that causes slow convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, minres

from superglm.group_matrix import _disc_disc_2d_hist
from superglm.solvers.scop import SCOPSolverReparam


@dataclass
class _SCOPPrototypeConfig:
    """Private prototype switches for numerical experiments on the joint solve."""

    solve_mode: Literal["direct", "minres", "minres_inexact"] = "direct"
    iterative_q_total_min: int = 96
    iterative_rtol: float = 1e-10
    iterative_maxiter: int | None = None
    cross_block_rel_tol: float = 0.0


_PROTOTYPE_CONFIG = _SCOPPrototypeConfig()


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

    linear_solver: str = "direct"
    """Linear solver used for the Newton direction."""

    linear_iterations: int = 0
    """Iteration count for iterative solvers, else 0."""

    dropped_cross_blocks: int = 0
    """Number of cross-group blocks dropped by prototype sparsification."""


def configure_scop_prototype(
    *,
    solve_mode: Literal["direct", "minres", "minres_inexact"] | None = None,
    iterative_q_total_min: int | None = None,
    iterative_rtol: float | None = None,
    iterative_maxiter: int | None = None,
    cross_block_rel_tol: float | None = None,
) -> None:
    """Configure private prototype switches for SCOP numerical experiments."""
    global _PROTOTYPE_CONFIG
    cfg = _PROTOTYPE_CONFIG
    if solve_mode is not None:
        cfg.solve_mode = solve_mode
    if iterative_q_total_min is not None:
        cfg.iterative_q_total_min = iterative_q_total_min
    if iterative_rtol is not None:
        cfg.iterative_rtol = iterative_rtol
    if iterative_maxiter is not None:
        cfg.iterative_maxiter = iterative_maxiter
    if cross_block_rel_tol is not None:
        cfg.cross_block_rel_tol = cross_block_rel_tol


def reset_scop_prototype() -> None:
    """Reset prototype switches to the exact direct-solve defaults."""
    global _PROTOTYPE_CONFIG
    _PROTOTYPE_CONFIG = _SCOPPrototypeConfig()


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


def _safe_trial_objective(
    B_scop: NDArray,
    W: NDArray,
    z: NDArray,
    beta_trial: NDArray,
    reparam: SCOPSolverReparam,
    S_scop: NDArray,
    lambda2: float,
    bin_idx: NDArray | None,
) -> float:
    """Evaluate trial-step objective, returning np.inf on any non-finite quantity.

    Overflow in exp(beta_eff), residual**2, or the penalty term is treated
    as a rejected line-search proposal rather than a warning-producing path.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        gamma_trial = reparam.forward(beta_trial)
        if not np.all(np.isfinite(gamma_trial)):
            return np.inf
        eta_bin = B_scop @ gamma_trial
        if bin_idx is not None:
            eta_trial = eta_bin[bin_idx]
        else:
            eta_trial = eta_bin
        res = z - eta_trial
        data_term = 0.5 * np.sum(W * res**2)
        penalty_term = 0.5 * lambda2 * float(beta_trial @ S_scop @ beta_trial)
        obj = data_term + penalty_term
    if not np.isfinite(obj):
        return np.inf
    return float(obj)


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

    obj_before = _safe_trial_objective(B_scop, W, z, beta, reparam, S_scop, lambda2, bin_idx)

    # If the starting point is already non-finite, we cannot compute a
    # meaningful gradient or Hessian.  Return a no-op.
    if not np.isfinite(obj_before):
        return SCOPNewtonResult(
            beta_new=beta,
            objective_before=np.inf,
            objective_after=np.inf,
            step_norm=0.0,
            used_fisher_fallback=False,
            H_penalized=lambda2 * S_scop,
        )

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
    # Non-finite trial states (overflow in exp(beta_eff), residual**2, etc.)
    # are treated as rejected line-search proposals, never as accepted iterates.
    alpha = 1.0
    accepted = False

    for _ in range(max_halving + 1):  # +1 for the initial full step
        beta_trial = beta - alpha * step
        obj_trial = _safe_trial_objective(
            B_scop, W, z, beta_trial, reparam, S_scop, lambda2, bin_idx
        )
        if np.isfinite(obj_trial) and obj_trial <= obj_before + 1e-14:
            accepted = True
            break
        alpha *= 0.5

    if accepted:
        beta_new = beta_trial
        obj_new = obj_trial
        step_norm = float(np.linalg.norm(alpha * step))
    else:
        # All halvings failed — reject the step entirely
        beta_new = beta
        obj_new = obj_before
        step_norm = 0.0

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


def _build_minres_preconditioner(
    scop_items: list[tuple[int, dict]],
    joint_slices: list[slice],
    BtWBs: list[NDArray],
    j_diags: list[NDArray],
    lambdas_list: list[float],
) -> LinearOperator:
    """Build a block-Jacobi inverse preconditioner for MINRES.

    Uses the positive semidefinite Gauss-Newton diagonal blocks plus a small
    ridge, falling back to a diagonal inverse if Cholesky still fails.
    """
    block_solvers: list[tuple[str, object]] = []
    for idx, (_, st) in enumerate(scop_items):
        j_i = j_diags[idx]
        jj_ii = j_i[:, None] * j_i[None, :]
        block = jj_ii * BtWBs[idx] + lambdas_list[idx] * st["S_scop"]
        eye = np.eye(block.shape[0])
        solved = False
        for ridge in (1e-10, 1e-8, 1e-6, 1e-4):
            try:
                fac = cho_factor(block + ridge * eye)
                block_solvers.append(("chol", fac))
                solved = True
                break
            except np.linalg.LinAlgError:
                continue
        if not solved:
            diag = np.maximum(np.abs(np.diag(block)), 1e-8)
            block_solvers.append(("diag", 1.0 / diag))

    size = joint_slices[-1].stop if joint_slices else 0

    def _matvec(v: NDArray) -> NDArray:
        out = np.empty_like(v)
        for sl_i, (kind, solver) in zip(joint_slices, block_solvers, strict=True):
            block_v = v[sl_i]
            if kind == "chol":
                out[sl_i] = cho_solve(solver, block_v)
            else:
                out[sl_i] = solver * block_v
        return out

    return LinearOperator((size, size), matvec=_matvec, dtype=np.float64)


def _solve_step_minres(
    H: NDArray,
    grad: NDArray,
    scop_items: list[tuple[int, dict]],
    joint_slices: list[slice],
    BtWBs: list[NDArray],
    j_diags: list[NDArray],
    lambdas_list: list[float],
    *,
    rtol: float,
    maxiter: int | None,
    allow_inexact: bool,
) -> tuple[NDArray | None, int]:
    """Solve H @ step = grad with MINRES and a block-Jacobi preconditioner."""
    iters = 0

    def _callback(_x: NDArray) -> None:
        nonlocal iters
        iters += 1

    A = LinearOperator(H.shape, matvec=lambda v: H @ v, dtype=np.float64)
    M = _build_minres_preconditioner(scop_items, joint_slices, BtWBs, j_diags, lambdas_list)
    step, info = minres(
        A,
        grad,
        M=M,
        rtol=rtol,
        maxiter=maxiter,
        callback=_callback,
        check=False,
    )

    residual = np.linalg.norm(H @ step - grad)
    grad_norm = max(np.linalg.norm(grad), 1e-300)
    tol_mult = 10.0 if allow_inexact else 1.0
    residual_tol = tol_mult * max(rtol, 1e-12) * grad_norm
    if info == 0 and np.isfinite(residual) and residual <= residual_tol:
        return step, iters
    if allow_inexact and np.isfinite(residual) and residual <= residual_tol:
        return step, iters
    return None, iters


def _solve_joint_step(
    H: NDArray,
    grad: NDArray,
    scop_items: list[tuple[int, dict]],
    joint_slices: list[slice],
    BtWBs: list[NDArray],
    j_diags: list[NDArray],
    lambdas_list: list[float],
) -> tuple[NDArray | None, str, int]:
    """Solve the joint Newton direction using the active prototype policy."""
    cfg = _PROTOTYPE_CONFIG
    q_total = H.shape[0]

    use_iterative = cfg.solve_mode != "direct" and q_total >= cfg.iterative_q_total_min
    if not use_iterative:
        return _solve_step(H, grad), "direct", 0

    allow_inexact = cfg.solve_mode == "minres_inexact"
    step, iters = _solve_step_minres(
        H,
        grad,
        scop_items,
        joint_slices,
        BtWBs,
        j_diags,
        lambdas_list,
        rtol=cfg.iterative_rtol,
        maxiter=cfg.iterative_maxiter,
        allow_inexact=allow_inexact,
    )
    if step is not None:
        solver_name = "minres_inexact" if allow_inexact else "minres"
        return step, solver_name, iters

    return _solve_step(H, grad), "direct_fallback", iters


# ---------------------------------------------------------------------------
# Joint multi-group SCOP Newton step
# ---------------------------------------------------------------------------


def _safe_joint_objective(
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
        data_term = 0.5 * np.sum(W * r**2)
        obj = data_term + penalty
    return float(obj) if np.isfinite(obj) else np.inf


def _joint_objective_from_eta(
    W: NDArray,
    z_scop: NDArray,
    total_eta: NDArray,
    beta_joint: NDArray,
    slices: list[slice],
    lambdas_list: list[float],
    scop_items: list[tuple[int, dict]],
) -> float:
    """Evaluate the joint SCOP objective from cached total eta.

    This is mathematically identical to ``_safe_joint_objective`` but reuses
    the already assembled SCOP contribution when the caller has it.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        penalty = 0.0
        for (_, st), sl_i, lam_i in zip(scop_items, slices, lambdas_list):
            beta_i = beta_joint[sl_i]
            penalty += 0.5 * lam_i * float(beta_i @ st["S_scop"] @ beta_i)
        residual = z_scop - total_eta
        data_term = 0.5 * np.sum(W * residual**2)
        obj = data_term + penalty
    return float(obj) if np.isfinite(obj) else np.inf


def _safe_joint_trial_objective(
    scop_items: list[tuple[int, dict]],
    W: NDArray,
    z_scop: NDArray,
    beta_trial: NDArray,
    slices: list[slice],
    lambdas_list: list[float],
    total_eta_current: NDArray,
    gammas_current: list[NDArray],
) -> float:
    """Evaluate a trial joint objective reusing the current SCOP state.

    Instead of rebuilding the whole objective from scratch, this keeps the
    current total SCOP contribution and only applies per-group trial deltas.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        total_eta = total_eta_current.copy()
        penalty = 0.0
        for (_, st), sl_i, lam_i, gamma_curr in zip(
            scop_items, slices, lambdas_list, gammas_current
        ):
            beta_i = beta_trial[sl_i]
            gamma_trial = st["reparam"].forward(beta_i)
            if not np.all(np.isfinite(gamma_trial)):
                return np.inf
            gamma_delta = gamma_trial - gamma_curr
            eta_delta = st["B_scop"] @ gamma_delta
            if st["bin_idx"] is not None:
                eta_delta = eta_delta[st["bin_idx"]]
            total_eta += eta_delta
            penalty += 0.5 * lam_i * float(beta_i @ st["S_scop"] @ beta_i)
        residual = z_scop - total_eta
        data_term = 0.5 * np.sum(W * residual**2)
        obj = data_term + penalty
    return float(obj) if np.isfinite(obj) else np.inf


def _compute_cross_gram(
    st_i: dict,
    st_j: dict,
    W: NDArray,
) -> NDArray:
    """Compute cross-group weighted gram matrix B_i^T @ diag(W) @ B_j.

    Handles all combinations of discretized and dense groups.

    Parameters
    ----------
    st_i, st_j : SCOP state dicts with B_scop, bin_idx keys
    W : (n,) observation weights

    Returns
    -------
    (q_i, q_j) cross-gram matrix
    """
    B_i, bi_i = st_i["B_scop"], st_i["bin_idx"]
    B_j, bi_j = st_j["B_scop"], st_j["bin_idx"]

    if bi_i is not None and bi_j is not None:
        # Both discretized: 2D weight histogram
        nb_i = B_i.shape[0]
        nb_j = B_j.shape[0]
        W_2d = _disc_disc_2d_hist(bi_i, bi_j, W, nb_i, nb_j)
        return B_i.T @ W_2d @ B_j
    elif bi_i is None and bi_j is None:
        # Both dense
        return B_i.T @ (B_j * W[:, None])
    elif bi_i is not None and bi_j is None:
        # i discretized, j dense: scatter i to observation level
        B_i_full = B_i[bi_i]
        return B_i_full.T @ (B_j * W[:, None])
    else:
        # i dense, j discretized: scatter j to observation level
        B_j_full = B_j[bi_j]
        return B_i.T @ (B_j_full * W[:, None])


def scop_joint_newton_step(
    scop_states: dict[int, dict],
    W: NDArray,
    z_scop: NDArray,
    lambdas: dict[str, float] | float,
    groups: list,
    max_halving: int = 10,
) -> dict[int, SCOPNewtonResult]:
    """Joint Newton step for all SCOP groups simultaneously.

    Instead of sequential Gauss-Seidel updates (which cause slow convergence
    with cross-correlated SCOP terms), this solves for ALL SCOP beta_eff
    parameters in one Newton step with proper cross-group Hessian blocks.

    For a single SCOP group, degenerates to one block with no cross-terms,
    producing identical results to ``scop_newton_step``.

    Parameters
    ----------
    scop_states : dict[int, dict]
        Per-group SCOP state. Keys are group indices into ``groups``.
        Each dict has: B_scop, S_scop, beta_scop, reparam, bin_idx,
        group_sl, group_name.
    W : (n,) array
        IRLS working weights (observation-level).
    z_scop : (n,) array
        Working response minus non-SCOP eta contributions.
    lambdas : dict[str, float] or float
        Per-group penalty weights (keyed by group name), or a scalar
        applied to all groups.
    groups : list of GroupSlice
        Full group list (used for name lookup).
    max_halving : int
        Maximum number of step halvings for joint line search.

    Returns
    -------
    dict[int, SCOPNewtonResult]
        Per-group Newton results keyed by group index.
    """
    # --- Setup: ordered list of (group_index, state) pairs ---
    scop_items = sorted(scop_states.items())
    n_groups = len(scop_items)

    # Per-group dimensions and slice mapping into joint vector
    q_effs = []
    joint_slices = []
    lambdas_list = []
    offset = 0
    for gi, st in scop_items:
        q_i = len(st["beta_scop"])
        q_effs.append(q_i)
        joint_slices.append(slice(offset, offset + q_i))
        # Look up lambda for this group
        g_i = groups[gi]
        if isinstance(lambdas, dict):
            lam_i = lambdas.get(g_i.name, 0.0)
        else:
            lam_i = float(lambdas)
        lambdas_list.append(lam_i)
        offset += q_i

    q_total = offset

    # --- Step 1: Forward map, per-group j_diag, compute shared residual ---
    betas = []
    j_diags = []
    etas = []
    for gi, st in scop_items:
        beta_i = st["beta_scop"].copy()
        betas.append(beta_i)
        gamma_i = st["reparam"].forward(beta_i)
        j_diags.append(gamma_i)  # d(exp(b))/db = exp(b) = gamma
        eta_i = st["B_scop"] @ gamma_i
        if st["bin_idx"] is not None:
            eta_i = eta_i[st["bin_idx"]]
        etas.append(eta_i)

    # Shared residual: z_scop minus ALL SCOP group etas
    total_scop_eta = np.zeros_like(z_scop)
    for eta_i in etas:
        total_scop_eta += eta_i
    residual = z_scop - total_scop_eta

    # --- Check starting objective ---
    beta_joint = np.concatenate(betas)
    obj_before = _joint_objective_from_eta(
        W,
        z_scop,
        total_scop_eta,
        beta_joint,
        joint_slices,
        lambdas_list,
        scop_items,
    )

    if not np.isfinite(obj_before):
        # Non-finite starting point: return no-op for all groups
        results = {}
        for idx, (gi, st) in enumerate(scop_items):
            q_i = q_effs[idx]
            results[gi] = SCOPNewtonResult(
                beta_new=st["beta_scop"].copy(),
                objective_before=np.inf,
                objective_after=np.inf,
                step_norm=0.0,
                used_fisher_fallback=False,
                H_penalized=lambdas_list[idx] * st["S_scop"],
            )
        return results

    # --- Step 2: Per-group BtWB and gradient ---
    BtWBs = []
    r_effs = []
    grad_datas = []
    for idx, (gi, st) in enumerate(scop_items):
        B_i = st["B_scop"]
        bi_i = st["bin_idx"]
        j_i = j_diags[idx]

        if bi_i is not None:
            n_bins = B_i.shape[0]
            W_agg = np.bincount(bi_i, weights=W, minlength=n_bins)
            Wr_agg = np.bincount(bi_i, weights=W * residual, minlength=n_bins)
            BtWB_ii = B_i.T @ (B_i * W_agg[:, None])
            r_eff_i = B_i.T @ Wr_agg
        else:
            BtW_i = B_i.T * W[np.newaxis, :]
            BtWB_ii = BtW_i @ B_i
            r_eff_i = BtW_i @ residual

        BtWBs.append(BtWB_ii)
        r_effs.append(r_eff_i)
        grad_data_i = -(j_i * r_eff_i)
        grad_datas.append(grad_data_i)

    # --- Step 3: Assemble joint Hessian and gradient ---
    H = np.zeros((q_total, q_total))
    grad = np.zeros(q_total)
    dropped_cross_blocks = 0

    for idx, (gi, st) in enumerate(scop_items):
        sl_i = joint_slices[idx]
        j_i = j_diags[idx]
        lam_i = lambdas_list[idx]
        beta_i = betas[idx]

        # Diagonal block: J^T BtWB J + lambda*S + diag(grad_data)
        jj_ii = j_i[:, None] * j_i[None, :]
        block = jj_ii * BtWBs[idx] + lam_i * st["S_scop"]
        block[np.diag_indices_from(block)] += grad_datas[idx]
        H[sl_i, sl_i] = block

        # Gradient
        grad[sl_i] = grad_datas[idx] + lam_i * (st["S_scop"] @ beta_i)

    # Cross-group blocks
    cfg = _PROTOTYPE_CONFIG
    for a in range(n_groups):
        for b in range(a + 1, n_groups):
            gi_a, st_a = scop_items[a]
            gi_b, st_b = scop_items[b]
            sl_a = joint_slices[a]
            sl_b = joint_slices[b]
            j_a = j_diags[a]
            j_b = j_diags[b]

            BtWB_ab = _compute_cross_gram(st_a, st_b, W)
            if cfg.cross_block_rel_tol > 0.0:
                norm_scale = np.sqrt(
                    np.linalg.norm(BtWBs[a], ord="fro") * np.linalg.norm(BtWBs[b], ord="fro")
                )
                if norm_scale > 0.0:
                    rel_cross = np.linalg.norm(BtWB_ab, ord="fro") / norm_scale
                    if rel_cross < cfg.cross_block_rel_tol:
                        dropped_cross_blocks += 1
                        continue
            cross = BtWB_ab * j_a[:, np.newaxis] * j_b[np.newaxis, :]
            H[sl_a, sl_b] = cross
            H[sl_b, sl_a] = cross.T

    # --- Step 4: Solve ---
    used_fisher = False
    step, linear_solver, linear_iterations = _solve_joint_step(
        H,
        grad,
        scop_items,
        joint_slices,
        BtWBs,
        j_diags,
        lambdas_list,
    )

    if step is None:
        # Fisher fallback: drop second-order diag(grad_data) terms
        used_fisher = True
        H_gn = H.copy()
        for idx in range(n_groups):
            sl_i = joint_slices[idx]
            H_block = H_gn[sl_i, sl_i].copy()
            H_block[np.diag_indices_from(H_block)] -= grad_datas[idx]
            H_gn[sl_i, sl_i] = H_block
        H_fisher = H_gn + 1e-8 * np.eye(q_total)
        step, linear_solver, linear_iterations = _solve_joint_step(
            H_fisher,
            grad,
            scop_items,
            joint_slices,
            BtWBs,
            j_diags,
            lambdas_list,
        )
        if step is None:
            H_fisher += 1e-4 * np.eye(q_total)
            step, linear_solver, linear_iterations = _solve_joint_step(
                H_fisher,
                grad,
                scop_items,
                joint_slices,
                BtWBs,
                j_diags,
                lambdas_list,
            )
            if step is None:
                step = -1e-4 * grad
                linear_solver = "gradient_fallback"
                linear_iterations = 0

    # --- Step 5: Joint line search ---
    alpha = 1.0
    accepted = False

    for _ in range(max_halving + 1):
        beta_trial = beta_joint - alpha * step
        obj_trial = _safe_joint_trial_objective(
            scop_items,
            W,
            z_scop,
            beta_trial,
            joint_slices,
            lambdas_list,
            total_scop_eta,
            j_diags,
        )
        if np.isfinite(obj_trial) and obj_trial <= obj_before + 1e-14:
            accepted = True
            break
        alpha *= 0.5

    if accepted:
        beta_new_joint = beta_trial
        obj_new = obj_trial
    else:
        beta_new_joint = beta_joint
        obj_new = obj_before

    # --- Step 6: Distribute results ---
    results = {}
    for idx, (gi, st) in enumerate(scop_items):
        sl_i = joint_slices[idx]
        beta_new_i = beta_new_joint[sl_i]
        if accepted:
            step_norm_i = float(np.linalg.norm(alpha * step[sl_i]))
        else:
            step_norm_i = 0.0

        # Diagonal block of the joint Hessian
        H_block_i = H[sl_i, sl_i]

        results[gi] = SCOPNewtonResult(
            beta_new=beta_new_i,
            objective_before=float(obj_before),
            objective_after=float(obj_new),
            step_norm=step_norm_i,
            used_fisher_fallback=used_fisher,
            H_penalized=H_block_i,
            linear_solver=linear_solver,
            linear_iterations=linear_iterations,
            dropped_cross_blocks=dropped_cross_blocks,
        )

    return results
