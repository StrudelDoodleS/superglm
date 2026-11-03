"""Active-set constrained penalized least-squares solver.

Solves ``min 0.5 beta' H beta - g' beta`` subject to ``A @ beta >= b`` with
``H`` positive semidefinite, by a primal active-set method (Nocedal and
Wright, *Numerical Optimization*, 2nd ed., 2006, Algorithm 16.3). A
rank-deficient ``H`` is rank-truncated by the shared rank policy, never
regularized; a materially indefinite one is refused. ``active_set_init``
warm-starts the working set from a previous solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
from numpy.typing import NDArray

from superglm._fit_trace import TraceRun
from superglm._roundoff import EPS as _EPS
from superglm._roundoff import TINY, UNIT_ROUNDOFF, gamma
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    RankDecomposition,
    _symmetric_part,
    decompose_factor,
    decompose_gram,
)

# Safety factor on the spectral consistency floor (see ``_consistency_floor``).
# Kept separate from ``SHARED_RANK_POLICY.certification_band`` and from the
# structural floor below, so retuning either cannot move this gate.
_NULL_BASIS_ACCURACY_SLACK = 32.0

# The smallest action the coordinate-weighted certificate divides by.
_SQRT_TINY = TINY**0.5

# Dimension term for the spectral floor: ``eigh``'s backward error grows with
# the width even for a perfectly conditioned retained block (worst case seen
# 17.3 eps per unit width; 512 leaves about 30x margin).
_SPECTRAL_DIMENSION_SLACK = 512.0

# Floor for the structural half of the null space, whose basis vectors are
# exact unit vectors: it covers only rounding in the two norms of the ratio.
# Independent of the spectral slack, though the values coincide.
_STRUCTURAL_NORM_ROUNDING_SLACK = 32.0
_STRUCTURAL_CONSISTENCY_FLOOR = _STRUCTURAL_NORM_ROUNDING_SLACK * _EPS


@dataclass
class QPResult:
    """Result of a constrained QP solve.

    ``converged`` means the returned beta passed primal feasibility, dual
    feasibility, stationarity and complementarity; exhaustion or an incomplete
    certificate returns ``False`` even at a feasible point. A failed
    feasibility check does not imply the constraints are mutually infeasible.

    ``rank``, ``width``, ``method``, ``condition`` (the factor-scale
    ``pre_truncation_condition``, as the unconstrained branch reports) and
    ``used_svd_fallback`` describe the ``H`` decomposition the solve ran on.
    The defaults mean no decomposition was reached.
    """

    beta: NDArray
    active_set: list[int] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = True
    rank: int = -1
    width: int = -1
    method: str = ""
    condition: float = 0.0
    used_svd_fallback: bool = False


def _null_space_mass(decomposition: RankDecomposition, g: NDArray) -> tuple[float, float]:
    """Split ``g``'s share of ``null(H)`` into its structural and spectral parts.

    The normal equations are consistent exactly when ``g`` is orthogonal to ``null(H)``, so
    ``g`` is projected on the null basis rather than read from a solve residual, which the
    retained condition amplifies. The basis holds exact unit vectors on structurally zero
    columns and ``D^-1 V_discarded`` on the scaled ones, with disjoint row supports. ``g`` is
    first scaled by a power of two (exact) so both norms lie in ``[0.5, sqrt(width)]`` and
    neither underflows to 0 or overflows to inf, hiding a breach. Returns
    ``(structural_mass, spectral_mass)``, each as a share of ``||g||``.
    """
    largest = float(np.max(np.abs(g), initial=0.0))
    if largest == 0.0:
        return 0.0, 0.0
    g = np.ldexp(g, -int(np.frexp(largest)[1]))

    reference = float(np.linalg.norm(g))
    structural = decomposition.column_scale == 0.0
    structural_mass = float(np.linalg.norm(g[structural])) / reference

    null_basis = decomposition.null_basis()
    scaled = ~structural
    spectral_mass = 0.0
    # Restricting to the scaled rows leaves exactly the spectral columns; with
    # no null columns, or no scaled columns, ``retained_columns`` is all-False.
    spectral_basis = null_basis[scaled, :]
    retained_columns = np.linalg.norm(spectral_basis, axis=0) > 0.0
    if np.any(retained_columns):
        orthonormal, _ = np.linalg.qr(spectral_basis[:, retained_columns])
        spectral_mass = float(np.linalg.norm(orthonormal.T @ g[scaled])) / reference
    return structural_mass, spectral_mass


def _consistency_floor(decomposition: RankDecomposition) -> float:
    """Largest spectral null mass the decomposition's own roundoff can manufacture.

    The computed spectral null basis is accurate to about ``eps`` times the retained condition
    (an eigenvalue ratio on the Gram scale, not ``pre_truncation_condition``'s factor scale), so
    the floor scales with it and a well-posed ill-conditioned rank-deficient system stays
    solvable; once the floor saturates at 1 (about ``1e14``) the test fails open rather than
    refuse a solvable system. The structural half uses ``_STRUCTURAL_CONSISTENCY_FLOOR``;
    ``retained_values is None`` means nothing spectral was dropped (the Cholesky paths).
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
    return float(
        min(
            1.0,
            _EPS
            * max(
                _NULL_BASIS_ACCURACY_SLACK * max(1.0, retained_condition),
                _SPECTRAL_DIMENSION_SLACK * decomposition.width,
            ),
        )
    )


def _constraint_products(
    A: NDArray, beta: NDArray, abs_A: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Ordinary binary64 row actions; callers scale exceptional rows below."""
    A = np.asarray(A, dtype=float)
    beta = np.asarray(beta, dtype=float)
    abs_A = np.abs(A) if abs_A is None else np.asarray(abs_A, dtype=float)
    magnitude_beta = np.abs(beta)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        products = A @ beta
        magnitude = abs_A @ magnitude_beta
    return products, magnitude


def _constraint_actions(A, beta, b, abs_A=None, step=None, *, common=False):
    """Use a common power of two for each exceptional row's action and RHS.

    Product mantissas lie in [1/4, 1), and the chosen exponent bounds the
    absolute sum including the direction. No physical product or reciprocal
    scale is materialized. Terms that underflow in these dimensionless units
    are below the existing relative dot allowance. Normal rows retain their
    original matvecs and blocking-ratio rounding. Stationarity requests one
    common exponent across all rows to preserve multiplier units.
    """
    A, beta, b = np.asarray(A, float), np.asarray(beta, float), np.asarray(b, float)
    products, magnitude = _constraint_products(A, beta, abs_A)
    action, motion = (
        _constraint_products(A, step, abs_A)
        if step is not None
        else (np.zeros_like(b), np.zeros_like(b))
    )
    bound = b.copy()
    unsafe = np.abs(b) > np.finfo(float).max / 4
    for vector, absolute in ((beta, magnitude), (step, motion)):
        if vector is None:
            continue
        unsafe |= ~np.isfinite(absolute) | (absolute > np.finfo(float).max / 4)
        unsafe |= (absolute < np.finfo(float).tiny) & np.any((A != 0) & (vector != 0), axis=1)
    if not np.any(unsafe):
        return products, magnitude, bound, action, motion
    if common:
        unsafe[:] = True
    coefficients, powers = np.frexp(A[unsafe])
    terms = []
    exponents = np.frexp(b[unsafe])[1]
    exponents = np.where(b[unsafe] != 0, exponents, -10000)
    for vector in (beta, step):
        if vector is None:
            continue
        mantissa, exponent = np.frexp(vector)
        value = coefficients * mantissa
        exponent = np.where(value != 0, powers + exponent, -10000)
        exponents = np.maximum(exponents, np.max(exponent, axis=1) + A.shape[1].bit_length())
        terms.append((value, exponent))
    if common:
        exponents[:] = np.max(exponents)
    for (value, exponent), signed, absolute in zip(
        terms, (products, action), (magnitude, motion), strict=False
    ):
        normalized = np.ldexp(value, exponent - exponents[:, None])
        signed[unsafe] = np.sum(normalized, axis=1)
        absolute[unsafe] = np.sum(np.abs(normalized), axis=1)
    bound[unsafe] = np.ldexp(b[unsafe], -exponents)
    return products, magnitude, bound, action, motion


def _feasibility_slack(
    A: NDArray,
    beta: NDArray,
    b: NDArray,
    *,
    abs_b: NDArray | None = None,
    abs_A: NDArray | None = None,
) -> NDArray:
    """Return the primal slack in homogeneous row-action units.

    The denominator is max(|b_i|, |A_i| @ |beta|). A positive rescaling
    of one constraint leaves this ratio unchanged. The absolute dot-product
    terms account for cancellation; a zero denominator means an exactly zero
    constraint action and has zero slack. Callers add their requested relative
    tolerance and the dimension-dependent dot-product rounding allowance.

    Optional abs_b and abs_A cache those exact input magnitudes.
    """
    products, magnitude, bounds, _, _ = _constraint_actions(A, beta, b, abs_A)
    scale = _feasibility_scale(products, bounds, abs_products=magnitude)
    slack = products - bounds
    return np.divide(slack, scale, out=np.zeros_like(slack), where=scale > 0.0)


def _feasibility_scale(
    products: NDArray,
    b: NDArray,
    *,
    abs_b: NDArray | None = None,
    abs_products: NDArray | None = None,
) -> NDArray:
    """Return max(|b|, |A| @ |beta|), without an additive unit floor.

    Production callers provide abs_products, the sum of absolute input terms.
    The products-only fallback remains for direct callers of this helper.
    Zero actions stay zero and must be handled explicitly by the caller.
    """
    magnitude = np.abs(products) if abs_products is None else abs_products
    return np.maximum(np.abs(b) if abs_b is None else abs_b, magnitude)


def _roundoff_tolerance(width: int) -> float:
    """A dot-product allowance in relative action units, with no magnitude floor."""
    if (width + 2) * UNIT_ROUNDOFF >= 1.0:
        raise ValueError("QP dimensions do not admit a dot-product error bound")
    return gamma(width + 2)


def _scaled_constraint_step(
    A: NDArray, beta: NDArray, step: NDArray, b: NDArray, abs_A: NDArray
) -> tuple[NDArray, NDArray]:
    """Normalize a directional action, including motion off an exact zero row."""
    products, magnitude, bounds, action, step_magnitude = _constraint_actions(
        A, beta, b, abs_A, step
    )
    scale = np.maximum(_feasibility_scale(products, bounds, abs_products=magnitude), step_magnitude)
    return np.divide(action, scale, out=np.zeros_like(action), where=scale > 0.0), scale


def _stationarity_certificate(
    H: NDArray,
    g: NDArray,
    beta: NDArray,
    active_matrix: NDArray,
    tol: float,
) -> tuple[bool, int | None, NDArray]:
    """Certify a nonnegative dual force in homogeneous stationarity units.

    Constraint rows are normalized before their multipliers are found, and the objective by the
    absolute actions of ``H beta`` and ``g``; with ``M = A_active.T`` in those coordinates,
    ``|M^+|`` maps score-action uncertainty to multiplier uncertainty, and a multiplier below
    that allowance releases its constraint. The clipped multipliers must then reproduce
    stationarity; an uncertain sign never certifies success. The multipliers are fitted in the
    per-coefficient test's units (each equation divided by ``max(|H||beta|, |g|)``), which
    matters when active rows touch coefficients orders of magnitude apart; those weights can
    span 1e32, and Householder QR is row-wise backward stable under them with column pivoting
    and rows sorted by decreasing infinity norm (Powell and Reid, 1969; Cox and Higham, Pitman
    Res. Notes Math. 380, 1998, Lemma 4.2 and Theorem 3.2).
    """
    active_matrix = np.asarray(active_matrix, dtype=float)
    products, h_action, g_values, _, _ = _constraint_actions(H, beta, g, common=True)
    g_action = np.abs(g_values)
    objective_scale = max(np.max(h_action, initial=0.0), np.max(g_action, initial=0.0))
    if objective_scale == 0.0:
        return True, None, np.zeros(len(active_matrix))
    if not np.isfinite(objective_scale):
        return False, None, np.zeros(len(active_matrix))
    residual = np.asarray((products - g_values) / objective_scale, dtype=float)
    h_action = np.asarray(h_action / objective_scale, dtype=float)
    g_action = np.asarray(g_action / objective_scale, dtype=float)
    row_scale = np.max(np.abs(active_matrix), axis=1, initial=0.0)
    rows = np.divide(
        active_matrix,
        row_scale[:, None],
        out=np.zeros_like(active_matrix),
        where=row_scale[:, None] > 0.0,
    )
    arithmetic = _roundoff_tolerance(len(beta) + len(rows))
    if len(rows):
        # A positive diagonal weighting cannot change the rank, but it can
        # push a genuine singular value under a cutoff relative to the
        # largest weighted one: the rank is decided unweighted, and the
        # basic solution on the leading pivoted columns has that rank.
        # Weights stay at most 1 / sqrt(TINY), so the weighted rows and any
        # product of two of them are finite (the square-root safe range of
        # LAPACK's la_constants; Anderson, ACM TOMS 44, 2017).
        own = np.maximum(h_action, g_action)
        weight = 1.0 / np.where(own > 0.0, np.maximum(own, _SQRT_TINY), 1.0)
        rank = np.linalg.matrix_rank(rows.T)
        weighted = rows.T * weight[:, None]
        order = np.argsort(-np.max(np.abs(weighted), axis=1, initial=0.0), kind="stable")
        q, r, pivots = scipy.linalg.qr(weighted[order], mode="economic", pivoting=True)
        basic = scipy.linalg.solve_triangular(r[:rank, :rank], q[:, :rank].T)
        inverse_action = np.zeros((len(rows), len(beta)))
        inverse_action[np.ix_(pivots[:rank], order)] = basic * weight[order]
        multipliers = inverse_action @ residual
        dual_action = np.abs(rows.T) @ np.abs(multipliers)
        score_scale = np.maximum(np.maximum(h_action, g_action), dual_action)
        score_error = arithmetic * (h_action + g_action + dual_action)
        multiplier_allowance = np.abs(inverse_action) @ (tol * score_scale + score_error)
        multiplier_allowance += arithmetic * np.abs(multipliers)
        if not np.all(np.isfinite(multipliers)) or not np.all(np.isfinite(multiplier_allowance)):
            return False, None, multipliers
        violating = multipliers < -multiplier_allowance
        if np.any(violating):
            worst = int(np.argmin(np.where(violating, multipliers, np.inf)))
            return False, worst, multipliers
        multipliers = np.maximum(multipliers, 0.0)
        dual = rows.T @ multipliers
        dual_action = np.abs(rows.T) @ multipliers
    else:
        multipliers = np.empty(0)
        dual = np.zeros_like(residual)
        dual_action = np.zeros_like(residual)
    score_scale = np.maximum(np.maximum(h_action, g_action), dual_action)
    allowance = tol * score_scale + arithmetic * (h_action + g_action + dual_action)
    stationary = bool(np.all(np.abs(residual - dual) <= allowance))
    return stationary, None, multipliers


def _certifies_kkt_point(
    H: NDArray,
    g: NDArray,
    beta: NDArray,
    A: NDArray,
    b: NDArray,
    active: list[int],
    tol: float,
    abs_A: NDArray,
) -> bool:
    """Whether ``beta`` passes the complete KKT certificate on ``active`` as it stands.

    Primal feasibility, stationarity with multipliers no more negative than
    their allowance, and complementarity: the three tests the small-step
    branch of ``solve_constrained_qp`` converges on, read without dropping a
    row or repairing ``beta``.
    """
    if not _is_feasible(A, beta, b, tol, abs_A=abs_A):
        return False
    stationary, drop_idx, multipliers = _stationarity_certificate(H, g, beta, A[active], tol)
    if not stationary or drop_idx is not None:
        return False
    return _complementarity_certificate(H, g, beta, A[active], b[active], multipliers, tol)


def _is_feasible(
    A: NDArray, beta: NDArray, b: NDArray, tol: float, *, abs_A: NDArray | None = None
) -> bool:
    """Whether ``beta`` satisfies ``A @ beta >= b`` to a scale-aware tolerance.

    ``abs_A`` is the loop-invariant ``np.abs(A)``.  The scale needs it on every
    call, and rebuilding it here is pure waste on the paths that already hold
    it -- the active-set loop, and ``irls_direct``, which tests the aggregate
    constraint system two to three times per IRLS iteration plus once per
    line-search halving.
    """
    return bool(
        np.all(
            _feasibility_slack(A, beta, b, abs_A=abs_A) >= -tol - _roundoff_tolerance(A.shape[1])
        )
    )


def _complementarity_certificate(
    H: NDArray,
    g: NDArray,
    beta: NDArray,
    active_matrix: NDArray,
    active_bounds: NDArray,
    multipliers: NDArray,
    tol: float,
) -> bool:
    """Measure dual-weighted slack against the represented objective actions.

    Complementarity has objective units. A positive rounding residue on a
    binding coordinate need not have small *relative primal slack*: its dual
    product must instead be small relative to the quadratic and linear actions.
    The multipliers use the normalized units of _stationarity_certificate.
    """
    if not len(multipliers):
        return True
    x = np.asarray(beta, dtype=float)
    rows = np.asarray(active_matrix, dtype=float)
    row_scale = np.max(np.abs(rows), axis=1, initial=0.0)
    row_scale = np.where(row_scale > 0.0, row_scale, 1.0)
    rows = rows / row_scale[:, None]
    dual = np.asarray(multipliers, dtype=float)
    _, h_action, score, _, _ = _constraint_actions(H, x, g, common=True)
    objective_scale = max(np.max(h_action, initial=0.0), np.max(np.abs(score), initial=0.0))
    if objective_scale == 0.0:
        return bool(np.all(dual == 0.0))
    # Complementarity is homogeneous in coefficient units as well. Scale
    # before the energy products, so |x| cannot overflow those reductions.
    x_scale = max(float(np.max(np.abs(x), initial=0.0)), np.finfo(float).tiny)
    x = x / x_scale
    bound_mantissa, bound_exponent = np.frexp(np.asarray(active_bounds, dtype=float))
    x_mantissa, x_exponent = np.frexp(x_scale)
    row_mantissa, row_exponent = np.frexp(row_scale)
    # Both divisors are applied before restoring range. Sequential physical
    # divisions can overflow even when their final dimensionless bound is small.
    with np.errstate(over="ignore", under="ignore"):
        bounds = np.ldexp(
            bound_mantissa / (x_mantissa * row_mantissa),
            bound_exponent - x_exponent - row_exponent,
        )
    row_action = np.abs(rows) @ np.abs(x) + np.abs(bounds)
    energy = max(
        np.abs(x) @ (h_action / objective_scale),
        np.abs(x) @ (np.abs(score) / objective_scale),
        dual @ row_action,
    )
    error = _roundoff_tolerance(len(beta) + len(dual)) * (dual @ row_action)
    gap = dual @ np.abs(rows @ x - bounds)
    return bool(np.isfinite(gap) and gap <= tol * energy + error)


def _solve_saddle_least_squares(KKT: NDArray, rhs: NDArray) -> NDArray:
    """Least-squares solve of a saddle system, symmetrically equilibrated first.

    ``lstsq``'s cutoff is relative to the largest singular value, so on an unscaled saddle
    matrix a large ``H`` hides every constraint direction (``H = diag(1e16, 0)``,
    ``A = [[1, 1]]`` keeps rank 1 of 3). One symmetric equilibration ``D K D``,
    ``D = diag(1 / sqrt(row inf-norm))`` (the first Ruiz step; Ruiz 2001), makes the cutoff
    separate directions by conditioning and keeps the saddle structure; equilibrated entries
    satisfy ``|K_ij| / sqrt(m_i m_j) <= 1``, and a zero row keeps scale 1. The solution is
    minimum-norm in equilibrated coordinates. ``rcond`` stays at ``lstsq``'s default, since an
    active constraint pins a near-null ``H`` direction at the constraint's scale.
    """
    row_norm = np.abs(KKT).max(axis=1)
    nonzero = row_norm > 0.0
    scale = np.where(nonzero, 1.0 / np.sqrt(np.where(nonzero, row_norm, 1.0)), 1.0)
    equilibrated = KKT * scale[:, None] * scale[None, :]
    return np.linalg.lstsq(equilibrated, rhs * scale, rcond=None)[0] * scale


def _project_feasible(beta: NDArray, A: NDArray, b: NDArray, tol: float) -> NDArray:
    """Repair a half-space violation, for at most 100 sweeps.

    Termination uses the same homogeneous primal predicate as the QP. Each
    repaired row is normalized before its squared norm is formed, preserving
    the projection under tiny or large constraint units. A violated zero row
    is impossible to repair and leaves an uncertified candidate. Exhaustion
    also requires the caller to check feasibility; neither case is success.
    """
    beta = beta.copy()
    # Loop-invariant: only ``A @ beta`` changes between sweeps.
    abs_A = np.abs(A)
    for _ in range(100):
        # Inlined rather than calling ``_feasibility_slack``, which would
        # recompute the matvec; the arithmetic is identical term for term.
        products, magnitude, bounds, _, _ = _constraint_actions(A, beta, b, abs_A)
        violations = products - bounds
        scale = _feasibility_scale(products, bounds, abs_products=magnitude)
        slack = np.divide(violations, scale, out=np.zeros_like(violations), where=scale > 0.0)
        if slack.min() >= -tol - _roundoff_tolerance(A.shape[1]):
            break
        worst = int(np.argmin(violations))
        # Project onto the violated constraint: a^T x >= b_i
        a = A[worst]
        row_scale = float(np.max(np.abs(a), initial=0.0))
        if row_scale == 0.0:
            break  # A violated structurally zero row cannot be repaired.
        normalized = a / row_scale
        deficit = b[worst] / row_scale - normalized @ beta
        beta += deficit / (normalized @ normalized) * normalized
    return beta


class _ConstraintBlocks(NamedTuple):
    """Coefficient blocks that no constraint row couples.

    ``columns[j]`` holds block ``j``'s coefficients and ``width[j]`` their
    count; ``row_label[i]`` is row ``i``'s block, ``-1`` for a zero row.
    """

    columns: list[NDArray]
    row_label: NDArray
    width: NDArray


def _constraint_blocks(A: NDArray) -> _ConstraintBlocks:
    """Coefficient blocks that no constraint row couples, each with its rows.

    These are the connected components of the bipartite row-column support
    graph of ``A``; in a fitted model, one shape-constrained term's
    coefficients. Each nonzero row lies in exactly one block, so a block's
    working rows constrain only that block's coefficients.
    """
    m, p = A.shape
    rows, cols = np.nonzero(A)
    # ``np.nonzero`` lists entries row by row, so the row nodes' adjacency is
    # already in CSR order; the column nodes have no stored entries.
    indptr = np.full(m + p + 1, rows.size, dtype=np.intp)
    indptr[0] = 0
    indptr[1 : m + 1] = np.cumsum(np.bincount(rows, minlength=m))
    graph = scipy.sparse.csr_matrix((np.ones(rows.size), m + cols, indptr), shape=(m + p, m + p))
    count, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)
    row_component, column_component = labels[:m], labels[m:]
    has_columns = np.zeros(count, dtype=bool)
    has_columns[column_component] = True
    components = np.unique(row_component[has_columns[row_component]])
    block_of = np.full(count, -1, dtype=np.intp)
    block_of[components] = np.arange(components.size)
    column_label = block_of[column_component]
    return _ConstraintBlocks(
        columns=[np.flatnonzero(column_label == j) for j in range(components.size)],
        row_label=block_of[row_component],
        width=np.bincount(column_label[column_label >= 0], minlength=components.size),
    )


def _homogeneous_feasible_start(
    beta: NDArray, A: NDArray, b: NDArray, tol: float
) -> NDArray | None:
    """Phase one for rows zero satisfies: each block that violates its rows restarts at zero.

    A primal active-set method needs a feasible start (Nocedal and Wright,
    *Numerical Optimization*, 2nd ed., 2006, section 16.5), found in general
    by a phase-one LP. A block whose rows all have ``b <= 0`` holds them at
    ``beta[block] = 0``, so no LP is needed: a homogeneous block (every
    ``b == 0``) is a cone with that point as its apex, and a cap, whose bounds
    contain a zero log effect, has ``b <= 0``. Blocks that satisfy their rows
    keep their coefficients. ``None`` when a violated block has a row zero
    violates (``b > 0``).
    """
    beta = beta.copy()
    blocks = _constraint_blocks(A)
    for j, block_cols in enumerate(blocks.columns):
        rows = np.flatnonzero(blocks.row_label == j)
        if _is_feasible(A[rows], beta, b[rows], tol):
            continue
        if np.any(b[rows] > 0.0):
            return None
        beta[block_cols] = 0.0
    return beta


def _raises_rank(distance: float) -> bool:
    """The shared factor rank rule for a unit vector at ``distance`` from an orthonormal span.

    The unit columns ``[Q, a]``, with ``a`` at relative distance ``rho`` from ``range(Q)``, have
    singular values 1 and ``sqrt(1 +- sqrt(1 - rho**2))``. ``decompose_factor``'s rule,
    ``sigma_min > factor_rcond * sigma_max``, reads ``rho > factor_rcond * (1 + sqrt(1 -
    rho**2))``, between ``factor_rcond`` and twice it, far above the projection's rounding.
    """
    cosine = np.sqrt(max(0.0, 1.0 - distance * distance))
    return distance > SHARED_RANK_POLICY.factor_rcond * (1.0 + cosine)


class _WorkingBasis:
    """An orthonormal basis of the working rows' span, in working-set order.

    Working-set rows must stay linearly independent (Nocedal and Wright,
    *Numerical Optimization*, 2nd ed., 2006, section 16.5; Gill and Wong, Math.
    Prog. Comp. 7, 2015, section 2): only then does releasing a row with a
    negative multiplier let the step move off it. In exact arithmetic a
    dependent row cannot block, since ``A_w @ step = 0``; rounding, or a
    violated row at an infeasible start, makes it appear to.

    ``q`` and ``r`` are the thin QR factors of the working rows, each scaled
    to unit norm, taken as columns. ``scipy.linalg.qr_insert`` and
    ``qr_delete`` update them as rows enter and leave (Golub and Van Loan,
    *Matrix Computations*, 4th ed., 2013, section 6.5), so testing a
    candidate is one projection, ``O(p k)``. A candidate raises the working
    rank when its distance from ``range(q)`` passes ``_raises_rank``.
    """

    def __init__(self, A: NDArray) -> None:
        row_scale = np.max(np.abs(A), axis=1, initial=0.0)
        rows = np.divide(
            A, row_scale[:, None], out=np.zeros_like(A), where=row_scale[:, None] > 0.0
        )
        norm = np.linalg.norm(rows, axis=1)
        self.rows = np.divide(
            rows, norm[:, None], out=np.zeros_like(rows), where=norm[:, None] > 0.0
        )
        self.q = np.zeros((A.shape[1], 0))
        self.r = np.zeros((0, 0))

    def start(self, working: list[int]) -> list[int]:
        """Factor a warm working set; rows dependent on those before them are left out.

        In unpivoted QR, ``|r_jj|`` is column ``j``'s distance from the span
        of the columns before it, so one factorization decides an independent
        set, as a previous solve returns. A set it cannot certify is built one
        row at a time instead.
        """
        k = len(working)
        if 0 < k <= self.q.shape[0]:
            q, r = scipy.linalg.qr(self.rows[working].T, mode="economic", check_finite=False)
            if all(_raises_rank(abs(float(r[j, j]))) for j in range(k)):
                self.q, self.r = q, r
                return list(working)
        kept: list[int] = []
        for row in working:
            if self.raises_rank(row):
                self.insert(row)
                kept.append(row)
        return kept

    def raises_rank(self, row: int) -> bool:
        """Whether row ``row`` of ``A`` is independent of the working rows."""
        candidate = self.rows[row]
        residual = candidate - self.q @ (self.q.T @ candidate)
        return _raises_rank(float(np.linalg.norm(residual)))

    def insert(self, row: int) -> None:
        """Append row ``row`` of ``A`` to the working rows."""
        self.q, self.r = scipy.linalg.qr_insert(
            self.q, self.r, self.rows[row], self.q.shape[1], which="col", check_finite=False
        )

    def delete(self, position: int) -> None:
        """Remove the working row at ``position``.

        ``qr_delete`` reads a square ``q`` as a full factor and keeps all its
        columns, so with as many working rows as coefficients the result still
        spans every direction. The thin factor of the ``k`` rows left is the
        leading ``k`` columns of ``q`` and rows of ``r``, the rest of ``r``
        being zero.
        """
        q, r = scipy.linalg.qr_delete(self.q, self.r, position, which="col", check_finite=False)
        k = r.shape[1]
        self.q, self.r = q[:, :k], r[:k]


def _zero_block_faces(
    beta: NDArray,
    A_eq: NDArray,
    b: NDArray,
    active: list[int],
    blocks: _ConstraintBlocks,
    full_rank: dict[tuple[int, ...], bool],
) -> NDArray:
    """Construct the exact zero point of each block on a homogeneous zero face.

    This is the ``n_eq >= p`` construction in ``solve_constrained_qp`` applied
    to one block at a time. Rows of other blocks have no entries in a block's
    columns. If a block's homogeneous working rows have full column rank there,
    the face fixes ``beta[block_cols] == 0`` exactly. Without the construction,
    cancellation residue around 1e-15 stays in the block. Primal slack is
    measured relative to that residue and so reads as violated, up to ``-1``.
    ``A_eq`` holds the normalized working rows in ``active`` order, and
    ``full_rank`` caches the rank decision by row tuple. A block with fewer
    working rows than coefficients cannot be fixed, so only blocks with as
    many are examined.
    """
    labels = blocks.row_label[active]
    counts = np.bincount(labels[labels >= 0], minlength=blocks.width.size)
    for j in np.flatnonzero(counts >= blocks.width):
        block_cols = blocks.columns[j]
        local = np.flatnonzero(labels == j)
        rows = tuple(active[k] for k in local)
        if np.any(b[list(rows)] != 0.0):
            continue
        if rows not in full_rank:
            rank = decompose_factor(A_eq[np.ix_(local, block_cols)]).rank
            full_rank[rows] = rank == block_cols.size
        if full_rank[rows] and np.any(beta[block_cols] != 0.0):
            beta = beta.copy()
            beta[block_cols] = 0.0
    return beta


BLOCKING_TRACE_CHANNEL = "constrained_qp_blocking"


def _emit_blocking_decision(
    trace_run: TraceRun,
    *,
    iteration: int,
    A: NDArray,
    b: NDArray,
    beta: NDArray,
    beta_new: NDArray,
    step: NDArray,
    tol: float,
    products: NDArray,
    raw_step: NDArray,
    working_rows: tuple[int, ...],
    blocking: int,
    alpha_min: float,
) -> None:
    """Record one blocking-constraint decision as a ``step_decision`` event.

    Every recorded value is re-derived from the raw inputs, never copied from
    the loop's own arithmetic, so the trace can witness a wrong gate.
    """
    assert trace_run is not None  # narrowed by the caller's `tracing` guard
    products, magnitude, bounds, raw_step, direction_magnitude = _constraint_actions(
        A, beta, b, step=step
    )
    derived_scale = _feasibility_scale(products, bounds, abs_products=magnitude)
    direction_scale = np.maximum(derived_scale, direction_magnitude)
    derived_scaled_step = np.divide(
        raw_step, direction_scale, out=np.zeros_like(raw_step), where=direction_scale > 0.0
    )
    active_lookup = set(working_rows)
    considered = [
        index
        for index in range(A.shape[0])
        if index not in active_lookup and derived_scaled_step[index] < -tol
    ]
    scaled_slack = _feasibility_slack(A, beta, b)
    trace_run.emit_lazy(
        "step_decision",
        lambda: {
            "iteration": iteration,
            # Independently evaluated: the loop must reach this block only for
            # a step the convergence test rejects.
            "full_step_is_feasible": _is_feasible(A, beta_new, b, tol),
            "considered_rows": tuple(considered),
            # The working rows the candidate was tested against, in order.
            "working_rows": working_rows,
            # The ratio inputs (common power-of-two units on exceptional rows),
            # so a reader can recompute alpha instead of trusting its value.
            "row_products": tuple(float(products[i]) for i in considered),
            "row_b": tuple(float(bounds[i]) for i in considered),
            "row_raw_step": tuple(float(raw_step[i]) for i in considered),
            "row_scaled_slack": tuple(float(scaled_slack[i]) for i in considered),
            # Recorded because ``max(|b|, |A_i| @ |beta|)`` cannot be rebuilt
            # from ``row_products`` and ``row_b``.
            "row_scale": tuple(float(derived_scale[i]) for i in considered),
            "direction_scale": tuple(float(direction_scale[i]) for i in considered),
            # These dimensionless actions remain representable even when the
            # original-unit diagnostic floats above underflow during export.
            "row_scaled_step": tuple(float(derived_scaled_step[i]) for i in considered),
            "blocking_row": int(blocking),
            # Independently derived: a row the scaled gate excludes must never
            # be the one blocked on.
            "blocking_is_considered": blocking < 0 or blocking in considered,
            "blocking_scaled_step": (
                float(derived_scaled_step[blocking]) if blocking >= 0 else 0.0
            ),
            "alpha": float(alpha_min),
        },
        channel=BLOCKING_TRACE_CHANNEL,
        # Fixed: this seam is test-only and no caller passes a purpose.
        purpose="constrained_qp",
    )


def solve_constrained_qp(
    H: NDArray,
    g: NDArray,
    A: NDArray,
    b: NDArray,
    active_set_init: list[int] | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
    *,
    _trace_run: TraceRun | None = None,
) -> QPResult:
    """Solve a convex quadratic subject to A @ beta >= b.

    H is symmetrized once, then decomposed by the shared PSD rank authority.
    A rank-deficient H is admitted only when g is consistent with its retained
    range. Material negative curvature and unresolved incompatible null scores
    are refused; constraints bounding an otherwise unbounded null direction
    remain outside this solver's capabilities.

    tol must lie in (0, 1). Primal feasibility uses the row-action scale
    max(|b_i|, |A_i| @ |beta|), plus a dimension-dependent rounding allowance.
    Dual signs and stationarity use normalized constraint rows and objective
    actions. Short coefficient steps, and a full step on a full-rank H, trigger
    the complete certificate rather than establish it. An exhausted or
    uncertified candidate is not converged.

    The working rows stay linearly independent (Nocedal and Wright 2006,
    section 16.5): the ratio test takes the least step among rows that raise
    the working rank (_WorkingBasis), ties to the lowest index. Each block
    whose homogeneous working rows have full column rank is set to its exact
    zero point (_zero_block_faces), and the certificate's multipliers are
    fitted in the certificate's own units (_stationarity_certificate).

    active_set_init supplies a warm active set; a row dependent on those
    before it is left out. The optional _trace_run emits blocking decisions
    only when explicitly enabled. Result rank metadata describes the shared H
    decomposition; no ridge is added.
    """
    # Checked at the public boundary rather than at each use: every predicate
    # below reads ``tol``, and a vacuous one is not detectable from the result.
    # Spelled as a negated range so a NaN ``tol`` is rejected as well.
    if not 0.0 < tol < 1.0:
        raise ValueError(
            f"solve_constrained_qp requires 0 < tol < 1, got {tol!r}. tol is a "
            "relative tolerance whose normalized slack saturates at -1, so "
            "tol >= 1 accepts every finite constraint violation and the solve "
            "reports its unconstrained answer as converged. It is also the "
            "step-norm and multiplier-sign threshold, neither of which is "
            "meaningful at 1."
        )

    p = H.shape[0]
    m = A.shape[0]

    # Symmetrize once, in float: ``H + H.T`` in an integer dtype wraps.
    # ``rank._symmetric_part`` keeps a finite float ``H`` from overflowing.
    H_asarray = np.asarray(H, dtype=float)
    H_sym = _symmetric_part(H_asarray)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    g = np.asarray(g, dtype=float)

    # One decomposition by the shared rank policy serves every pure-H solve;
    # a singular H is rank-truncated rather than raising LinAlgError.
    try:
        decomposition = decompose_gram(H_sym)
    except ValueError as exc:
        raise ValueError(f"solve_constrained_qp requires a usable PSD H: {exc}") from exc

    def _result(
        beta: NDArray,
        active_set: list[int],
        n_iter: int,
        converged: bool = True,
    ) -> QPResult:
        """Build a result carrying the geometry every pure-H solve ran on.

        Attribute reads only -- no extra factorization.  A closure rather than a
        splatted dict so each field keeps its own type at every return site.
        """
        return QPResult(
            beta=beta,
            active_set=active_set,
            n_iter=n_iter,
            converged=converged,
            rank=int(decomposition.rank),
            width=int(decomposition.width),
            method=str(decomposition.method),
            condition=float(decomposition.pre_truncation_condition),
            used_svd_fallback=bool(decomposition.used_svd_fallback),
        )

    # --- Unconstrained solution ---
    beta_unc = decomposition.solve(g)

    # ``decomposition.solve`` is a pseudo-inverse: if g has a component outside
    # range(H) the objective is unbounded below and H^+ g is not stationary, so
    # refuse that before either early return. Only truncated H can have one.
    if decomposition.rank < decomposition.width:
        # The two halves of null(H) are known to different accuracies, so they
        # get their own floors; see _null_space_mass.
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
                f"against a resolution floor of {floor:.3e}), so the "
                "unconstrained objective is unbounded below along that "
                "direction. The unconstrained solve and the active-set loop's "
                "empty-active-set step both form directions inside range(H), "
                "so neither entry path can follow it. Note the constraints may "
                "still bound the problem, in which case a finite optimum "
                "exists that this solver cannot reach: doing so needs a "
                "null-space descent direction, which is a filed follow-up "
                "rather than a capability it has today. Regularize H (for "
                "example add a ridge term) or drop the aliased columns."
            )

    unconstrained_stationary, _, _ = _stationarity_certificate(
        H_sym, g, beta_unc, np.empty((0, p)), tol
    )
    if m == 0:
        # No constraints: the unconstrained solve above is already the answer.
        return _result(beta_unc, [], 0, converged=unconstrained_stationary)

    if _is_feasible(A, beta_unc, b, tol):
        return _result(beta_unc, [], 0, converged=unconstrained_stationary)

    # --- Initialize active set ---
    # The working rows stay linearly independent in every solve; a warm set
    # is a hint, so a row dependent on those before it is left out.
    basis = _WorkingBasis(A)
    active = basis.start([] if active_set_init is None else list(active_set_init))

    # --- Feasible starting point ---
    # This may still be infeasible; see _project_feasible.  Feasibility is
    # therefore re-tested on the point actually returned, below.
    beta = _project_feasible(beta_unc, A, b, tol)

    # Resolved once: the hot loop must not pay a per-row or per-iteration
    # attribute lookup for a seam that is off in every production call.
    tracing = _trace_run is not None and _trace_run.enabled

    # Route rank-truncated systems to the equilibrated least-squares solve.
    # ``decompose_gram`` equilibrates before deciding rank, so it reports
    # collinearity, not scale: a full-rank H can still give a near-singular
    # KKT system, on which ``np.linalg.solve`` succeeds but can pick a far
    # larger representative on a near-flat face. Widening the gate would change
    # every near-singular fit and is a separate decision.
    kkt_may_be_singular = decomposition.rank < decomposition.width

    # ``|A|`` is loop-invariant; only the ``|beta|`` matvec is per iteration.
    abs_A = np.abs(A)
    zero_face_cache: dict[tuple[int, ...], bool] = {}
    blocks = _constraint_blocks(A)
    block_zero_face: dict[tuple[int, ...], bool] = {}
    # Whether ``beta`` minimizes on the current face (previous iteration took
    # its full step and kept the active set). The exact next step is then zero
    # (Nocedal and Wright 2006, Algorithm 16.3); in floating point it is the
    # saddle solve's rounding, which an ill-conditioned H can lift above the
    # ``tol`` step gate, so the loop certifies directly instead. Full-rank H
    # only: on a rank-deficient one a face minimum is a flat set.
    at_face_minimum = False

    for it in range(max_iter):
        # --- Equality-constrained subproblem on active set ---
        if len(active) == 0:
            # No active constraints — unconstrained step.  beta_unc is the same
            # quantity, already computed above the loop.
            step = beta_unc - beta
        else:
            A_eq = A[active]  # (|active|, p)
            b_eq = b[active]  # (|active|,)
            # Equality rows express geometry, not objective units. Normalize
            # before the saddle solve so equivalent constraint scaling does
            # not manufacture a primal residual on an active boundary.
            constraint_scale = np.max(np.abs(A_eq), axis=1, initial=0.0)
            constraint_scale = np.where(constraint_scale > 0.0, constraint_scale, 1.0)
            A_eq = A_eq / constraint_scale[:, None]
            b_eq = b_eq / constraint_scale

            n_eq = len(active)
            zero_face = False
            if n_eq >= p and np.all(b[active] == 0.0):
                # A full-column-rank homogeneous active face is exactly {0}.
                # Construct that point rather than a cancellation residue from
                # beta + step. Check the original bounds: normalization must
                # not erase a nonzero affine right-hand side into this branch.
                key = tuple(active)
                if key not in zero_face_cache:
                    zero_face_cache[key] = decompose_factor(A_eq).rank == p
                zero_face = zero_face_cache[key]
            if not zero_face:
                beta = _zero_block_faces(beta, A_eq, b, active, blocks, block_zero_face)
            if zero_face:
                beta = np.zeros_like(beta)
                step = np.zeros_like(beta)
            else:
                # Solve KKT system:
                # [H    -A_eq^T] [step  ] = [g - H @ beta]
                # [A_eq  0     ] [lambda] = [b_eq - A_eq @ beta]
                KKT = np.zeros((p + n_eq, p + n_eq))
                KKT[:p, :p] = H_sym
                KKT[:p, p:] = -A_eq.T
                KKT[p:, :p] = A_eq

                rhs = np.zeros(p + n_eq)
                rhs[:p] = g - H_sym @ beta
                rhs[p:] = b_eq - A_eq @ beta

                if kkt_may_be_singular:
                    # A truncated H can leave a constraint-tangent null
                    # direction on which np.linalg.solve succeeds and drifts,
                    # so take the minimum-norm solution directly.
                    sol = _solve_saddle_least_squares(KKT, rhs)
                else:
                    try:
                        sol = np.linalg.solve(KKT, rhs)
                    except np.linalg.LinAlgError:
                        # Singular KKT: the same equilibrated least-squares
                        # solve as the branch above.
                        sol = _solve_saddle_least_squares(KKT, rhs)

                step = sol[:p]

        # --- Check step feasibility ---
        # Judge movement per coordinate, relative to ``max(1, |beta_i|)``: an
        # absolute threshold is not scale-invariant, and at an active-boundary
        # KKT point cancellation in ``g - H @ beta`` leaves O(eps |beta|) steps.
        relative_step = step / np.maximum(1.0, np.abs(beta))
        small_step = bool(np.linalg.norm(relative_step) < tol)
        if (
            at_face_minimum
            and not small_step
            and not kkt_may_be_singular
            and _certifies_kkt_point(H_sym, g, beta, A, b, active, tol, abs_A)
        ):
            # The complete certificate, so a pass means the same here as at a
            # small step; a failure changes nothing and the loop steps on.
            return _result(beta, active, it + 1, converged=True)
        at_face_minimum = False
        if small_step:
            stationary, drop_idx, multipliers = _stationarity_certificate(
                H_sym, g, beta, A[active], tol
            )
            if drop_idx is not None:
                active.pop(drop_idx)
                basis.delete(drop_idx)
                continue
            feasible = _is_feasible(A, beta, b, tol)
            if not feasible:
                beta = _project_feasible(beta, A, b, tol)
                feasible = _is_feasible(A, beta, b, tol)
                # A repair is a new candidate: recheck its full dual and
                # stationarity certificate before assigning convergence.
                stationary, drop_idx, multipliers = _stationarity_certificate(
                    H_sym, g, beta, A[active], tol
                )
                if drop_idx is not None:
                    active.pop(drop_idx)
                    basis.delete(drop_idx)
                    continue
            complementary = _complementarity_certificate(
                H_sym, g, beta, A[active], b[active], multipliers, tol
            )
            if feasible and stationary and complementary:
                return _result(beta, active, it + 1, converged=True)
            if not feasible or not np.any(step != 0.0):
                return _result(beta, active, it + 1, converged=False)
            # A small but uncertified direction can still encounter a
            # blocking row; coefficient units cannot complete a dual solve.

        # --- Step ratio: find blocking constraint ---
        # Both tests use the per-row scale of the convergence test, so a row
        # ``_is_feasible`` accepts is never treated as violated.
        beta_new = beta + step

        if _is_feasible(A, beta_new, b, tol, abs_A=abs_A):
            # Full step is feasible
            beta = beta_new
            at_face_minimum = True
        else:
            # Find blocking constraint (first to be violated along step)
            alpha_min = 1.0
            blocking = -1

            products, _, bounds, raw_step, _ = _constraint_actions(A, beta, b, abs_A, step)
            # Same scale ``_is_feasible`` uses, which is what makes "does this
            # step move the row" agree with "is this row satisfied".
            scaled_step, _ = _scaled_constraint_step(A, beta, step, b, abs_A)
            # Gate on the scaled derivative, as _is_feasible does; take the
            # ratio from the raw pair, which rounds once.
            gated = scaled_step < -tol
            gated[active] = False
            candidates = np.flatnonzero(gated)
            alphas = (products[candidates] - bounds[candidates]) / -raw_step[candidates]
            ahead = alphas < alpha_min
            candidates, alphas = candidates[ahead], alphas[ahead]
            # The least step among rows that raise the working rank, ties to
            # the lowest index: rows are tested in that order until one does.
            for k in np.lexsort((candidates, alphas)):
                if basis.raises_rank(int(candidates[k])):
                    alpha_min = float(alphas[k])
                    blocking = int(candidates[k])
                    break

            if tracing:
                # After selection, before `active` is mutated, so the recorded
                # working rows are the set the decision was actually made
                # against.  Off by default, so this costs one bool test.
                _emit_blocking_decision(
                    _trace_run,
                    iteration=it,
                    A=A,
                    b=b,
                    beta=beta,
                    beta_new=beta_new,
                    step=step,
                    tol=tol,
                    products=products,
                    raw_step=raw_step,
                    working_rows=tuple(active),
                    blocking=blocking,
                    alpha_min=alpha_min,
                )

            if blocking >= 0:
                beta = beta + alpha_min * step
                active.append(blocking)
                basis.insert(blocking)
            else:
                beta = beta_new
                at_face_minimum = True

    # Exhaustion is non-convergence even at a feasible point: the KKT
    # certificate was never completed.
    return _result(beta, active, max_iter, converged=False)
