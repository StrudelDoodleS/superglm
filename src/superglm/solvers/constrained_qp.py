"""Active-set constrained penalized least-squares solver.

Solves:
    minimize   0.5 * beta^T H beta - g^T beta
    subject to A @ beta >= b

where H is positive definite (or positive semidefinite with regularization).

Uses a primal active-set method:
1. Start with a feasible point (project if needed).
2. Solve the equality-constrained subproblem on the active set.
3. If the step is feasible, check multipliers to drop constraints.
4. If infeasible, find the blocking constraint and add it.

Warm-starting: pass active_set_init from a previous solve to skip
discovery iterations (the active set usually stabilizes after a few
IRLS iterations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class QPResult:
    """Result of a constrained QP solve."""

    beta: NDArray
    active_set: list[int] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = True


def _project_feasible(beta: NDArray, A: NDArray, b: NDArray) -> NDArray:
    """Project beta onto the feasible set {x : A @ x >= b}.

    Uses iterative constraint-by-constraint projection (Dykstra-like).
    For the small dense problems we handle, this converges quickly.
    """
    beta = beta.copy()
    for _ in range(100):
        violations = A @ beta - b
        worst = np.argmin(violations)
        if violations[worst] >= -1e-12:
            break
        # Project onto the violated constraint: a^T x >= b_i
        a = A[worst]
        deficit = b[worst] - a @ beta
        beta += deficit / (a @ a) * a
    return beta


def solve_constrained_qp(
    H: NDArray,
    g: NDArray,
    A: NDArray,
    b: NDArray,
    active_set_init: list[int] | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> QPResult:
    """Solve a convex QP with linear inequality constraints.

    Parameters
    ----------
    H : (p, p) NDArray
        Positive definite (or PSD + regularized) Hessian.
    g : (p,) NDArray
        Linear term (gradient at zero, with sign: objective is
        0.5 * beta^T H beta - g^T beta).
    A : (m, p) NDArray
        Constraint matrix. Constraints are A @ beta >= b.
    b : (m,) NDArray
        Constraint right-hand side.
    active_set_init : list[int] | None
        Warm-start active set from previous solve.
    max_iter : int
        Maximum active-set iterations.
    tol : float
        Tolerance for constraint satisfaction and multiplier signs.

    Returns
    -------
    QPResult
        Solution with beta, active_set, iteration count, convergence flag.
    """
    p = H.shape[0]
    m = A.shape[0]

    if m == 0:
        # No constraints — direct solve
        beta = np.linalg.solve(H, g)
        return QPResult(beta=beta, active_set=[], n_iter=0)

    # --- Unconstrained solution ---
    beta_unc = np.linalg.solve(H, g)
    if np.all(A @ beta_unc - b >= -tol):
        return QPResult(beta=beta_unc, active_set=[], n_iter=0)

    # --- Initialize active set ---
    if active_set_init is not None:
        active = list(active_set_init)
    else:
        active = []

    # --- Feasible starting point ---
    beta = _project_feasible(beta_unc, A, b)

    for it in range(max_iter):
        # --- Equality-constrained subproblem on active set ---
        if len(active) == 0:
            # No active constraints — unconstrained step
            step = np.linalg.solve(H, g) - beta
        else:
            A_eq = A[active]  # (|active|, p)
            b_eq = b[active]  # (|active|,)

            # Solve KKT system:
            # [H    -A_eq^T] [step  ] = [g - H @ beta]
            # [A_eq  0     ] [lambda] = [b_eq - A_eq @ beta]
            n_eq = len(active)
            KKT = np.zeros((p + n_eq, p + n_eq))
            KKT[:p, :p] = H
            KKT[:p, p:] = -A_eq.T
            KKT[p:, :p] = A_eq

            rhs = np.zeros(p + n_eq)
            rhs[:p] = g - H @ beta
            rhs[p:] = b_eq - A_eq @ beta

            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                # Singular KKT — use least-squares
                sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

            step = sol[:p]

        # --- Check step feasibility ---
        if np.linalg.norm(step) < tol:
            # At a stationary point. Check multipliers.
            if len(active) == 0:
                return QPResult(beta=beta, active_set=active, n_iter=it + 1)

            # Recompute multipliers at current point.
            # KKT stationarity: H*beta - g = A_eq' * lambda, lambda >= 0
            # => lambda = (A_eq @ A_eq^T)^{-1} @ A_eq @ (H @ beta - g)
            A_eq = A[active]
            residual = H @ beta - g
            try:
                multipliers = np.linalg.solve(A_eq @ A_eq.T, A_eq @ residual)
            except np.linalg.LinAlgError:
                multipliers = np.linalg.lstsq(A_eq @ A_eq.T, A_eq @ residual, rcond=None)[0]

            # Drop most negative multiplier (constraint wants to be inactive)
            min_mult = np.min(multipliers)
            if min_mult >= -tol:
                # All multipliers nonneg — KKT satisfied
                return QPResult(beta=beta, active_set=active, n_iter=it + 1)

            drop_idx = np.argmin(multipliers)
            active.pop(drop_idx)
            continue

        # --- Step ratio: find blocking constraint ---
        beta_new = beta + step
        violations = A @ beta_new - b

        if np.all(violations >= -tol):
            # Full step is feasible
            beta = beta_new
        else:
            # Find blocking constraint (first to be violated along step)
            alpha_min = 1.0
            blocking = -1

            for i in range(m):
                if i in active:
                    continue
                a_step = A[i] @ step
                if a_step < -tol:
                    # This constraint could be violated
                    slack = A[i] @ beta - b[i]
                    alpha = slack / (-a_step)
                    if alpha < alpha_min:
                        alpha_min = alpha
                        blocking = i

            if blocking >= 0:
                beta = beta + alpha_min * step
                active.append(blocking)
            else:
                beta = beta_new

    return QPResult(beta=beta, active_set=active, n_iter=max_iter, converged=False)
