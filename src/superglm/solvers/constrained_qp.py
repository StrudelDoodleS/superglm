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

from superglm.solvers.rank import RankDecomposition, decompose_gram

_EPS = np.finfo(float).eps

# Headroom on the normal-equation consistency floor (see ``_consistency_floor``).
# The floor estimates the accuracy of the *computed null basis*, and this is the
# safety factor above that estimate.  It is deliberately a constant of its own
# rather than ``SHARED_RANK_POLICY.certification_band``, which happens to hold
# the same value but governs factor certification: retuning that band for its
# own purpose must not silently move this gate.
_NULL_BASIS_ACCURACY_SLACK = 32.0

# Consistency floor for the *structural* half of the null space.  Those basis
# vectors are exact unit vectors -- a structurally zero column gives ``H`` an
# identically zero row -- so no conditioning term belongs here; the only slack
# needed covers rounding in the norms themselves.
_STRUCTURAL_CONSISTENCY_FLOOR = _NULL_BASIS_ACCURACY_SLACK * _EPS


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


def _null_space_mass(decomposition: RankDecomposition, g: NDArray) -> tuple[float, float]:
    """Split ``g``'s share of ``null(H)`` into its structural and spectral parts.

    ``g`` is in ``range(H)`` -- the normal equations are consistent -- exactly
    when it is orthogonal to ``null(H)``, so project it onto the null basis the
    decomposition already computed.  This measures the inconsistency directly
    instead of inferring it from a solve residual: a residual is amplified by
    the retained condition number, which at the conditions the rank policy
    happily retains swamps the signal entirely.

    The two halves must not share a threshold, because they are not known to
    the same accuracy.  ``rank._null_basis`` stacks

    * one **exact unit vector** per structurally zero column (``column_scale``
      is 0 there, so ``H`` has an identically zero row and column), and
    * the **discarded spectral directions** ``D^-1 @ V_discarded``, whose
      computed accuracy degrades as ``eps * retained condition``.

    They have disjoint row supports -- the unit vectors live on the unscaled
    columns, the spectral directions on the scaled ones -- so the split is an
    orthogonal decomposition of ``null(H)`` and the two masses can be measured
    and thresholded independently. Judging the exact half against the spectral
    half's floor is what let an ill-conditioned retained block desensitize a
    structural alias where detection is exact.

    Returns ``(structural_mass, spectral_mass)``, each as a share of ``||g||``.
    """
    reference = max(float(np.linalg.norm(g)), np.finfo(float).tiny)
    structural = decomposition.column_scale == 0.0
    structural_mass = float(np.linalg.norm(g[structural])) / reference

    null_basis = decomposition.null_basis()
    scaled = ~structural
    spectral_mass = 0.0
    if null_basis.shape[1] and np.any(scaled):
        # Restricting to the scaled rows annihilates the unit-vector columns
        # and leaves exactly the spectral ones.
        spectral_basis = null_basis[scaled, :]
        retained_columns = np.linalg.norm(spectral_basis, axis=0) > 0.0
        if np.any(retained_columns):
            orthonormal, _ = np.linalg.qr(spectral_basis[:, retained_columns])
            spectral_mass = float(np.linalg.norm(orthonormal.T @ g[scaled])) / reference
    return structural_mass, spectral_mass


def _consistency_floor(decomposition: RankDecomposition) -> float:
    """Largest *spectral* null mass the decomposition's own roundoff can manufacture.

    This governs only the spectral half of the split in ``_null_space_mass``.
    The structural half is measured against ``_STRUCTURAL_CONSISTENCY_FLOOR``,
    because a structurally zero column's null vector is an exact unit vector
    and carries none of the error this floor models.

    The computed spectral null basis is accurate only to about ``eps`` times
    the retained condition number, so a genuinely consistent ``g`` still
    projects onto it by roughly that much.  Scaling the threshold with the
    retained conditioning -- rather than fixing it at ``factor_rcond`` -- is
    what keeps a well-posed but ill-conditioned rank-deficient system solvable,
    since the rank policy truncates at ``gram_rcond`` and so retains blocks far
    more ill-conditioned than ``factor_rcond`` would tolerate.

    Sensitivity degrades *gradually* as the retained conditioning grows; there
    is no sharp cutoff.  Measured detection of an injected **spectral** null
    component, by retained condition:

    ==============  ============  ===============  =================
    retained cond   median floor  1% mass caught   0.1% mass caught
    ==============  ============  ===============  =================
    ``1e9``         ``6.8e-06``   120/120          120/120
    ``1e10``        ``6.8e-05``   120/120          120/120
    ``1e11``        ``6.8e-04``   118/118          102/118
    ``1e12``        ``6.9e-03``   103/119            8/119
    ``1e13``        ``6.8e-02``     8/120            1/120
    ==============  ============  ===============  =================

    So a 0.1% spectral inconsistency is already partly missed at ``1e11`` and a
    1% one at ``1e12``, well below the ``~1e14`` at which the floor saturates
    at 1 and nothing spectral is detectable at all.  This is a real resolution
    limit rather than a tuning choice: at those conditions the spectral null
    basis is itself only accurate to the floor, and along such a direction the
    objective moves only in the fifth significant figure over a coefficient
    range of ``1e9``.  It fails open -- the solve proceeds rather than refusing
    a system it cannot adjudicate -- which is the safe direction, since
    refusing a solvable system is the defect this floor exists to prevent.

    None of that degradation applies to a structural alias: those are exact at
    every conditioning, which is why they are thresholded separately.
    """
    retained = decomposition.retained_values
    if retained is None or retained.size == 0:
        # Cholesky without truncated spectrum: nothing spectral was dropped.
        retained_condition = 1.0
    else:
        magnitudes = np.abs(retained)
        smallest = float(np.min(magnitudes))
        retained_condition = (
            float(np.max(magnitudes)) / smallest if smallest > 0.0 else float("inf")
        )
    return float(min(1.0, _NULL_BASIS_ACCURACY_SLACK * _EPS * max(1.0, retained_condition)))


def _feasibility_slack(A: NDArray, beta: NDArray, b: NDArray) -> NDArray:
    """Return ``A @ beta - b`` measured against a scale-aware tolerance.

    A step that lands *on* a constraint reproduces ``b_i`` only to about
    ``eps * |A_i @ beta|``, so a fixed absolute tolerance turns a genuine KKT
    point into a violation as soon as the constraint row is large: at
    ``|A_i @ beta| ~ 1e4`` an exactly-active constraint already reads ``-2e-12``.
    Comparing against ``tol * max(1, |b_i|, |A_i @ beta|)`` keeps the test
    meaningful under rescaling, and is identical to the absolute test for the
    well-scaled problems where the scale factor is 1.

    Returns the slack already divided by its per-row scale, so callers can
    compare it against a bare ``-tol``.
    """
    products = A @ beta
    return (products - b) / _feasibility_scale(products, b)


def _feasibility_scale(products: NDArray, b: NDArray) -> NDArray:
    """Per-row scale for the relative feasibility test: ``max(1, |b|, |A @ beta|)``.

    Exposed separately so the active-set loop can divide *both* its slack and
    its directional derivative by the same factor.  Scaling both leaves the
    step ratio ``slack / -a_step`` numerically unchanged while making the
    decisions built on them -- "is this row already satisfied", "does this step
    move the row at all" -- agree with ``_is_feasible``.
    """
    return np.maximum(1.0, np.maximum(np.abs(b), np.abs(products)))


def _is_feasible(A: NDArray, beta: NDArray, b: NDArray, tol: float) -> bool:
    """Whether ``beta`` satisfies ``A @ beta >= b`` to a scale-aware tolerance."""
    if A.shape[0] == 0:
        return True
    return bool(np.all(_feasibility_slack(A, beta, b) >= -tol))


def _project_feasible(beta: NDArray, A: NDArray, b: NDArray, tol: float) -> NDArray:
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

    Uses the same scale-aware stopping test as the caller's convergence check,
    so the two cannot disagree about what "feasible" means.
    """
    beta = beta.copy()
    for _ in range(100):
        violations = _feasibility_slack(A, beta, b)
        worst = int(np.argmin(violations))
        if violations[worst] >= -tol:
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
        raising, provided g lies in ``range(H)``.

        Three inputs raise ``ValueError`` rather than returning a plausible
        wrong answer: a materially indefinite H (the problem is then not the
        convex QP this solver assumes); a rank-deficient H whose g has a
        component outside ``range(H)`` (the objective is unbounded below along
        a null direction, and no search direction this method forms can follow
        it); and an H the rank policy cannot equilibrate.

        H is symmetrized once as ``0.5 * (H + H.T)`` and that symmetric part is
        used throughout -- decomposition, KKT blocks, residual and multiplier
        test alike -- so an asymmetric H is solved consistently as its
        symmetric part rather than as two different quadratics on the two
        paths. For an exactly symmetric H the symmetrization is bitwise
        identity. Every in-tree caller builds H symmetric by construction
        (``XtWX + S``, ``X'X + lambda*P``).
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
        Tolerance for constraint satisfaction and multiplier signs. The
        constraint test is relative: row ``i`` is satisfied when
        ``A_i @ beta - b_i >= -tol * max(1, |b_i|, |A_i @ beta|)``, so a
        badly scaled constraint system does not read as infeasible purely
        because its rows are large. This matters only for callers with a
        nonzero ``b``: at ``b_i == 0`` the relative test is algebraically
        identical to the absolute one for any ``tol`` in ``(0, 1)``, and every
        in-tree caller passes ``b = 0``.

    Returns
    -------
    QPResult
        Solution with beta, active_set, iteration count, convergence flag.
    """
    p = H.shape[0]
    m = A.shape[0]

    # The rank policy decomposes 0.5 * (H + H.T).  Materialize that symmetric
    # part once and use it for the KKT blocks, the residual and the multiplier
    # test too, so the two solve paths cannot disagree about which quadratic
    # they are minimizing.  For an exactly symmetric H this is bitwise H.
    H_sym = 0.5 * (H + H.T)

    # Route the pure-H solves through the shared rank policy so a singular or
    # near-singular H is rank-truncated the way it is everywhere else in the
    # solver subsystem, rather than raising LinAlgError.  H does not change
    # during the solve, so one decomposition serves every pure-H solve below.
    try:
        decomposition = decompose_gram(H_sym)
    except ValueError as exc:
        raise ValueError(f"solve_constrained_qp requires a usable PSD H: {exc}") from exc

    # --- Unconstrained solution ---
    beta_unc = decomposition.solve(g)

    # decomposition.solve is a pseudo-inverse, so it answers even when the
    # normal equations have no solution.  If H is rank-deficient and g has a
    # component outside range(H), the quadratic decreases without bound along
    # that null direction and H^+g is merely a projection, not a stationary
    # point.  Returning it as converged would be a silent wrong answer, so
    # detect the inconsistency before either early return.  Full rank means
    # range(H) is everything, so the check is only needed after truncation.
    if decomposition.rank < decomposition.width:
        # The two halves of null(H) are known to different accuracies, so they
        # get their own floors; see _null_space_mass.  Sharing one floor let an
        # ill-conditioned retained block desensitize a structural alias, where
        # detection is in fact exact.
        structural_mass, spectral_mass = _null_space_mass(decomposition, g)
        spectral_floor = _consistency_floor(decomposition)
        structural_breach = structural_mass > _STRUCTURAL_CONSISTENCY_FLOOR
        if structural_breach or spectral_mass > spectral_floor:
            kind, mass, floor = (
                ("a structurally aliased column", structural_mass, _STRUCTURAL_CONSISTENCY_FLOOR)
                if structural_breach
                else ("a truncated spectral direction", spectral_mass, spectral_floor)
            )
            raise ValueError(
                "solve_constrained_qp: H is rank-deficient (rank "
                f"{decomposition.rank} of {decomposition.width}) and g has a "
                f"component in null(H) along {kind} ({mass:.3e} of ||g||, "
                f"against a resolution floor of {floor:.3e}), so the objective "
                "is unbounded below along that direction. The unconstrained "
                "solve and the active-set loop's empty-active-set step both "
                "form directions inside range(H), so neither entry path can "
                "follow it. Regularize H (for example add a ridge term) or "
                "drop the aliased columns."
            )

    if m == 0:
        # No constraints — direct solve
        return QPResult(beta=beta_unc, active_set=[], n_iter=0)

    if _is_feasible(A, beta_unc, b, tol):
        return QPResult(beta=beta_unc, active_set=[], n_iter=0)

    # --- Initialize active set ---
    if active_set_init is not None:
        active = list(active_set_init)
    else:
        active = []

    # --- Feasible starting point ---
    # This may still be infeasible; see _project_feasible.  Feasibility is
    # therefore re-tested on the point actually returned, below.
    beta = _project_feasible(beta_unc, A, b, tol)

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
            KKT[:p, :p] = H_sym
            KKT[:p, p:] = -A_eq.T
            KKT[p:, :p] = A_eq

            rhs = np.zeros(p + n_eq)
            rhs[:p] = g - H_sym @ beta
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
                    converged=_is_feasible(A, beta, b, tol),
                )

            # Recompute multipliers at current point.
            # KKT stationarity: H*beta - g = A_eq' * lambda, lambda >= 0
            # => lambda = (A_eq @ A_eq^T)^{-1} @ A_eq @ (H @ beta - g)
            A_eq = A[active]
            residual = H_sym @ beta - g
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
                    converged=_is_feasible(A, beta, b, tol),
                )

            drop_idx = np.argmin(multipliers)
            active.pop(drop_idx)
            continue

        # --- Step ratio: find blocking constraint ---
        # Both tests below go through the same per-row scale the convergence
        # test uses.  Leaving them absolute while the boundaries went relative
        # made the loop treat a row as violated that _is_feasible considers
        # satisfied, so it would block on that row with a negative slack and
        # take a backward step.
        beta_new = beta + step

        if _is_feasible(A, beta_new, b, tol):
            # Full step is feasible
            beta = beta_new
        else:
            # Find blocking constraint (first to be violated along step)
            alpha_min = 1.0
            blocking = -1

            products = A @ beta
            row_scale = _feasibility_scale(products, b)
            scaled_slack = (products - b) / row_scale
            scaled_step = (A @ step) / row_scale

            for i in range(m):
                if i in active:
                    continue
                a_step = scaled_step[i]
                if a_step < -tol:
                    # This constraint could be violated.  Dividing slack and
                    # a_step by the same row scale leaves the ratio unchanged.
                    alpha = scaled_slack[i] / (-a_step)
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
