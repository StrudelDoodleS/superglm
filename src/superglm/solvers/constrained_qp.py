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

from superglm.solvers.rank import decompose_gram


@dataclass
class QPResult:
    """Result of a constrained QP solve.

    ``converged`` means the full KKT certificate holds for ``beta``: the
    active-set loop reached its own termination test (a stationary step with
    no negative multiplier) *and* ``beta`` is feasible.  It is ``False`` when
    the loop exhausted ``max_iter``, and when the loop terminated but the
    returned point still violates a constraint -- which is what happens for a
    mutually infeasible constraint system.  In either case ``beta`` is the
    best available point, not a certified solution.
    """

    beta: NDArray
    active_set: list[int] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = True


def _project_feasible(beta: NDArray, A: NDArray, b: NDArray) -> NDArray:
    """Project beta onto the feasible set {x : A @ x >= b}.

    Uses iterative constraint-by-constraint projection (Dykstra-like).
    For the small dense problems we handle, this converges quickly.

    Each sweep repairs only the single worst violation, so the 100-sweep
    budget can be exhausted with the point still infeasible -- either because
    the constraints are mutually infeasible, or merely because there are more
    violated constraints than sweeps.  Those two cases are not distinguishable
    here and the active-set loop often recovers from the second, so the caller
    must test the feasibility of the point it finally returns rather than
    treating the starting point's status as the answer.
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
        Positive semidefinite Hessian. It is decomposed once through the
        shared rank policy, so a rank-deficient H is truncated rather than
        raising; a materially indefinite H raises ``ValueError``, because
        the problem is then not the convex QP this solver assumes.
        The rank policy symmetrizes its input as ``0.5 * (H + H.T)``, so an
        asymmetric H is solved as its symmetric part. Every in-tree caller
        builds H symmetric by construction (``XtWX + S``, ``X'X + lambda*P``).
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

    # Route the pure-H solves through the shared rank policy so a singular or
    # near-singular H is rank-truncated the way it is everywhere else in the
    # solver subsystem, rather than raising LinAlgError.  H does not change
    # during the solve, so one decomposition serves every pure-H solve below.
    try:
        decomposition = decompose_gram(H)
    except ValueError as exc:
        raise ValueError(f"solve_constrained_qp requires a usable PSD H: {exc}") from exc

    if m == 0:
        # No constraints — direct solve
        beta = decomposition.solve(g)
        return QPResult(beta=beta, active_set=[], n_iter=0)

    # --- Unconstrained solution ---
    beta_unc = decomposition.solve(g)
    if np.all(A @ beta_unc - b >= -tol):
        return QPResult(beta=beta_unc, active_set=[], n_iter=0)

    # --- Initialize active set ---
    if active_set_init is not None:
        active = list(active_set_init)
    else:
        active = []

    # --- Feasible starting point ---
    # This may still be infeasible; see _project_feasible.  Feasibility is
    # therefore re-tested on the point actually returned, below.
    beta = _project_feasible(beta_unc, A, b)

    for it in range(max_iter):
        # --- Equality-constrained subproblem on active set ---
        if len(active) == 0:
            # No active constraints — unconstrained step.  beta_unc is the same
            # quantity, already computed above the loop.
            step = beta_unc - beta
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
                # Stationary with no active constraint: the KKT certificate is
                # complete once the point is also feasible.
                return QPResult(
                    beta=beta,
                    active_set=active,
                    n_iter=it + 1,
                    converged=bool(np.all(A @ beta - b >= -tol)),
                )

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
                # All multipliers nonneg — stationarity and dual feasibility
                # hold; primal feasibility completes the KKT certificate.
                return QPResult(
                    beta=beta,
                    active_set=active,
                    n_iter=it + 1,
                    converged=bool(np.all(A @ beta - b >= -tol)),
                )

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

    # Exhaustion is unconditional non-convergence, feasible or not: the loop
    # never reached its stationarity/multiplier test, so there is no KKT
    # certificate to complete.  A merely feasible point is not a solution --
    # any interior point is feasible -- so consulting feasibility here would
    # report success for a search that was cut off mid-flight.
    return QPResult(beta=beta, active_set=active, n_iter=max_iter, converged=False)
