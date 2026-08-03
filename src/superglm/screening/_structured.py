"""Structured screening kernel for a spline x categorical candidate pair.

The dense path assembles the pair's ``(k, k)`` curvature with ``k = k_a * L``
and factorizes it, which is cubic in the level count and quadratic in memory.
Both are pure waste here, because with the categorical margin's treatment
menu the pair's bordered system

    K(lambda) = [[V + lambda S, C'],
                 [C,            M ]]

is EXACTLY a block-arrow matrix once its variables are grouped by level:

  * ``V`` is block-diagonal — levels have disjoint row support, so no tensor
    column of one level meets any column of another;
  * ``S = kron(S_a, I)`` is block-diagonal on the same grouping — one copy of
    the spline penalty per level, which is the varying-coefficient layout;
  * ``M``'s categorical-main block is diagonal, and ``C``'s categorical-main
    rows are level-local: level q's indicator meets level q's tensor columns
    and nothing else.

What is left coupling the levels is only the intercept and the spline main —
a border of ``1 + k_a`` columns, independent of ``L``.  So the pair is
``L`` blocks of size ``k_a + 1`` around a border of size ``1 + k_a``, and
:mod:`superglm.screening._arrow` factors that in time and memory LINEAR in
``L``.  The dense path's ``V`` alone is 2.4 TB at fifty thousand levels; the
same pair's blocks are 48 MB.

The two quantities the ladder needs both read off one such factorization:

    T(lambda)   = U_eff' A(lambda)^-1 U_eff
    edf(lambda) = rank(A) - lambda * tr(A(lambda)^-1 S)

the second because ``V_eff = A - lambda S`` identically, which trades a trace
against a DENSE ``V_eff`` for one against a block-diagonal ``S`` — and a
block-diagonal ``S`` needs only the diagonal blocks of the inverse, which the
arrow factorization already has.

This path is entered only where the dense path is refused, so every pair the
dense path can score is still scored by it, bit for bit.  The two agree to
the tolerance pinned in tests/test_structured_screening.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.screening._arrow import factor_arrow, psd_ranks
from superglm.screening._score_stat import ScreenedPair

_EDF_TOL = 1e-6
_EDF_ROUNDOFF_FACTOR = 64.0
_EDF_ABSOLUTE_DUST = np.finfo(np.float64).eps ** 2
_MAX_BISECT = 200
_TRACE_CHUNK_DOUBLES = 262_144

# The most steps one rung's bisection can take.  It halves the log of a
# bracket spanning 1e20 and stops when the two ends are within 1e-12 of each
# other relatively, so it exhausts at ceil(log2(ln(1e20) / ln(1 + 1e-12)))
# steps whatever the data does -- _EDF_TOL usually stops it sooner (measured
# 26 to 28 on the pairs that search), but nothing guarantees that, and a
# ceiling is what lets a caller decide BEFORE the search whether to pay for
# it rather than after.
_MAX_STEPS_PER_RUNG = min(_MAX_BISECT, 46)


class _UnstableStructuredEDFError(FloatingPointError):
    """The arrow rank and inverse disagree by more than numerical dust."""


def _edf_roundoff(*values: float) -> float:
    """Scale-aware dust allowance for EDF identities and ordering."""
    return max(
        _EDF_ABSOLUTE_DUST,
        _EDF_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * sum(abs(float(value)) for value in values),
    )


@dataclass(frozen=True)
class SplineCatPair:
    """Level-blocked moments of one spline x categorical pair.

    Every array is linear in the level count ``L``.  ``V``/``U``/``c``/``m``
    are per-level; ``border``/``u_border`` are the level-independent border,
    the overlap span's ``[intercept | spline main]`` corner.
    """

    V: NDArray  # (L, k_a, k_a) tensor curvature per level
    U: NDArray  # (L, k_a)      tensor score per level
    c: NDArray  # (L, k_a)      level indicator against level q's tensor columns
    m: NDArray  # (L,)          weight in each level
    S_a: NDArray  # (k_a, k_a)  the spline penalty, shared by every level
    border: NDArray  # (r, r)        border block of M, r = 1 + k_a
    u_border: NDArray  # (r,)
    u_cat: NDArray  # (L,)
    # None means the global centered-row rank lies inside its certified
    # numerical ambiguity band; the structured route must be refused.
    profiled_trace: float | None

    @property
    def dims(self) -> tuple[int, int]:
        return self.U.shape  # (L, k_a)


def _centered_level_factors(B: NDArray, W: NDArray) -> NDArray:
    """Return QR factors of each level's centered weighted spline rows.

    The raw-moment identity ``B' W B - (B' w)(B' w)' / sum(w)`` is unusable
    here: the trace this module needs can be twelve or more orders below both
    terms.  This two-pass form subtracts each level's mean from the basis
    rows first.  It then returns a square ``R_l`` whose Gram is the centered
    geometry, without forming that Gram:

        R_l' R_l = (sqrt(W_l) D_l)' (sqrt(W_l) D_l).

    Shifting every row by the first basis row before taking the mean makes the
    centering invariant to a large common offset without subtracting that
    offset twice.  Zero-mass levels contribute an exact zero matrix.
    """
    B = np.asarray(B, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, k = B.shape
    n_levels = W.shape[1]
    if n_levels == 0:
        return np.empty((0, k, k), dtype=np.float64)
    if n == 0 or k == 0:
        return np.zeros((n_levels, k, k), dtype=np.float64)

    shifted = B - B[0]
    mass = W.sum(axis=0)
    means = np.zeros((n_levels, k), dtype=np.float64)
    np.divide(W.T @ shifted, mass[:, None], out=means, where=mass[:, None] > 0.0)
    centered = shifted[:, None, :] - means[None, :, :]
    centered *= np.sqrt(W)[:, :, None]
    raw = np.linalg.qr(np.moveaxis(centered, 1, 0), mode="r")
    if raw.shape[1] == k:
        return raw
    factors = np.zeros((n_levels, k, k), dtype=np.float64)
    factors[:, : raw.shape[1], :] = raw
    return factors


def _combine_row_factors(left: NDArray, right: NDArray) -> NDArray:
    """Compact two weighted-row factors without squaring either one."""
    return np.linalg.qr(np.concatenate((left, right), axis=0), mode="r")


def _representative_projection(
    row_factor: NDArray,
    *,
    n_rows: int,
    n_levels: int,
) -> tuple[NDArray, NDArray] | None:
    """Return a stable representative span and its structural null action.

    Column-pivoted QR of the small global weighted-row factor chooses
    representative basis columns ``active``.  In pivot order,

        Z = Z_active [I, C],       R11 C = R12.

    A coefficient action using only the active rows is therefore sufficient:
    every other representative differs only by a null-space action and has
    the same fitted rows and residual energies.  ``null_action`` is
    ``I - P`` built structurally as active rows ``[0, -C]`` and inactive rows
    ``[0, I]``.  No cancelling ``I - H^+ H`` is formed.

    The cutoff is the square root of the Hermitian pseudo-inverse policy used
    by the dense path.  Rank is refused, rather than guessed, when a pivot
    intersects its QR backward-error interval.  Each Householder reduction
    contributes an additive ``O(eps * leading_scale)`` perturbation; the
    conservative operation depth below covers one ``n_rows x k`` local QR per
    level, the sequential ``2k x k`` suffix merges, and the final ``k x k``
    pivoted QR.  Relative to a ``sqrt(k*eps)`` cutoff that uncertainty is
    ``O(sqrt(eps))`` — the scale on which row/level order can otherwise flip
    a retained direction.  Refusal lets routing fall back instead of making
    that platform-specific rank choice observable.
    """
    k = row_factor.shape[1]
    _, pivoted, permutation = scipy.linalg.qr(
        row_factor,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(pivoted))
    if diagonal.size == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty_like(row_factor),
        )

    eps = np.finfo(np.float64).eps
    leading_scale = float(diagonal[0])
    if leading_scale <= np.finfo(np.float64).tiny:
        return (
            np.empty(0, dtype=np.intp),
            np.zeros_like(row_factor),
        )
    cutoff = np.sqrt(max(k, 1) * eps) * leading_scale
    reduction_depth = (
        max(int(n_rows), 1) * max(k, 1)
        + 2 * max(int(n_levels), 1) * max(k, 1) ** 2
        + max(k, 1) ** 2
    )
    uncertainty = 16.0 * eps * reduction_depth * leading_scale
    if np.any(np.abs(diagonal - cutoff) <= uncertainty):
        return None
    rank = int(np.count_nonzero(diagonal > cutoff))
    active = np.asarray(permutation[:rank], dtype=np.intp)
    inactive = np.asarray(permutation[rank:], dtype=np.intp)

    null_action = np.zeros((k, k), dtype=np.float64)
    if rank and inactive.size:
        relation = scipy.linalg.solve_triangular(
            pivoted[:rank, :rank],
            pivoted[:rank, rank:],
            check_finite=False,
        )
        null_action[np.ix_(active, inactive)] = -relation
    null_action[inactive, inactive] = 1.0
    return active, null_action


def _aligned_representative_actions(
    active: NDArray,
    local: NDArray,
    other: NDArray,
) -> tuple[NDArray, NDArray]:
    """Project local and complementary rows in one aligned QR coordinate.

    ``local`` and ``other`` are row factors for disjoint pieces of the same
    global centered design.  Stacking their representative columns and
    applying the resulting orthogonal coordinate to the aligned right-hand
    sides gives both least-squares actions directly:

        G = [local_active; other_active] = Q R
        A_q(active)  = R^-1 Q_local' local
        A_-q(active) = R^-1 Q_other' other.

    This is algebraically the same representative fit as ``H^+ H_q`` and
    ``H^+ H_-q``, but it never forms ``G'G`` or solves through ``R'R``.  The
    two right-hand sides share one triangular solve.
    """
    k = local.shape[1]
    projection = np.zeros((k, k), dtype=np.float64)
    complement = np.zeros((k, k), dtype=np.float64)
    if active.size == 0:
        return projection, complement

    aligned = np.concatenate((local[:, active], other[:, active]), axis=0)
    Q, triangular = np.linalg.qr(aligned, mode="reduced")
    right_hand_sides = np.concatenate(
        (Q[:k].T @ local, Q[k:].T @ other),
        axis=1,
    )
    actions = scipy.linalg.solve_triangular(
        triangular,
        right_hand_sides,
        check_finite=False,
    )
    projection[active] = actions[:, :k]
    complement[active] = actions[:, k:]
    return projection, complement


def _trace_chunk_width(n_rows: int, n_cols: int, n_levels: int) -> int:
    """Bound centered-row and factor temporaries by a small fixed chunk."""
    per_level = max(int(n_rows) * int(n_cols), int(n_cols) ** 2, 1)
    return max(1, min(int(n_levels), _TRACE_CHUNK_DOUBLES // per_level or 1))


def _profiled_curvature_trace(
    B: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> float | None:
    """Compute ``tr(V_eff)`` as an exact sum of squared residual norms.

    Profiling the intercept and categorical main centers the spline geometry
    separately inside every level.  Write that weighted centered design as
    ``Z_l = sqrt(W_l) D_l``.  A QR factor ``R_l`` has the same action norm:
    ``||Z_l A||_F = ||R_l A||_F`` for every coefficient action ``A``.

    A column-pivoted QR of the global row factor chooses a stable
    representative span without ever forming the ill-conditioned normal
    matrix ``H = sum_l Z_l' Z_l``.  For emitted interaction block ``q``, let
    ``A_q`` be its representative projection coefficients, ``A_-q`` the
    coefficients for all other levels, and ``N`` the structural null action
    returned by :func:`_representative_projection`.  The residual action is
    assembled additively as ``N + A_-q``.  The diagonal Schur-complement trace
    is the exact nonnegative factor norm

        ||R_q (N + A_-q)||_F^2 + ||R_-q A_q||_F^2.

    Summing those nonnegative terms over emitted levels is exactly
    ``tr(V - C' M^+ C)``.  Unlike that difference, it remains representable
    when the mains absorb all but round-off of the interaction.

    ``R_-q`` is a QR compaction of a prefix and precomputed suffix of actual
    row factors.  It is never obtained by subtracting level ``q`` from a
    global Gram.  Only that suffix is a level-sized trace scratch stack; the
    centered factors are recomputed in bounded chunks on the forward pass,
    then all trace scratch is discarded before the arrow ladder.  Work and
    memory remain linear in the level count; no SVD, normal-matrix inverse, or
    dense ``V_eff`` is formed.
    """
    B = np.asarray(B, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    level_rows = np.asarray(level_rows, dtype=np.intp)
    n_rows, k_a = B.shape
    n_levels = W_cell.shape[1]
    if level_rows.size == 0 or k_a == 0:
        return 0.0

    chunk = _trace_chunk_width(n_rows, k_a, n_levels)
    suffix = np.empty((n_levels + 1, k_a, k_a), dtype=np.float64)
    suffix[-1] = 0.0
    for stop in range(n_levels, 0, -chunk):
        start = max(0, stop - chunk)
        factors = _centered_level_factors(B, W_cell[:, start:stop])
        for level in range(stop - 1, start - 1, -1):
            suffix[level] = _combine_row_factors(factors[level - start], suffix[level + 1])

    representative = _representative_projection(
        suffix[0],
        n_rows=n_rows,
        n_levels=n_levels,
    )
    if representative is None:
        return None
    active, null_action = representative
    emitted = np.zeros(n_levels, dtype=bool)
    emitted[level_rows] = True
    prefix = np.zeros((k_a, k_a), dtype=np.float64)
    trace = 0.0
    correction = 0.0

    for start in range(0, n_levels, chunk):
        stop = min(n_levels, start + chunk)
        factors = _centered_level_factors(B, W_cell[:, start:stop])
        for level in range(start, stop):
            local = factors[level - start]
            if emitted[level]:
                other = _combine_row_factors(prefix, suffix[level + 1])
                projection, complement = _aligned_representative_actions(active, local, other)
                residual = null_action + complement
                term = float(
                    np.sum(np.square(local @ residual)) + np.sum(np.square(other @ projection))
                )
                # Neumaier-style compensated accumulation keeps the final
                # scalar independent of level order down to the factor error.
                updated = trace + term
                if abs(trace) >= abs(term):
                    correction += (trace - updated) + term
                else:
                    correction += (term - updated) + trace
                trace = updated
            prefix = _combine_row_factors(prefix, local)

    return trace + correction


def spline_cat_moments(
    B_a: NDArray,
    S_a: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> SplineCatPair:
    """Assemble the level-blocked moments straight from the pair's cell tables.

    ``level_rows[q]`` is the cell-table column the q-th contrast indicates —
    the treatment menu is one-hot with a zeroed base row, so selecting that
    column IS multiplying by the menu, and the ``(L, L-1)`` menu is never
    built.  Nor is the dense path's ``(n_a, L, L)`` curvature intermediate,
    which is diagonal in its last two axes: measured at 380,880 doubles
    carrying 690 nonzeros for a 24-level factor.

    A missing ``S_a`` is carried as a zero penalty rather than refused, which
    is what the dense path does with the same input: an unpenalized block has
    no bandwidth to scan, and :func:`structured_ladder` reports it at a
    single rung.  Refusing here would abort the whole sweep over one pair,
    where the contract is a NaN row.
    """
    B_a = np.asarray(B_a, dtype=np.float64)
    n_a, k_a = B_a.shape
    level_rows = np.asarray(level_rows, dtype=np.intp)
    Wq = W_cell[:, level_rows]
    Sq = S_cell[:, level_rows]

    # One GEMM for every level's k_a x k_a curvature: the outer products of
    # the spline menu are level-independent, so they are formed once and
    # contracted against each level's weights.
    AA = (B_a[:, :, None] * B_a[:, None, :]).reshape(n_a, k_a * k_a)
    V = (Wq.T @ AA).reshape(-1, k_a, k_a)
    del AA
    # V persists into the ladder, but its n_a*k_a^2 construction scratch has
    # been released before the one level-sized suffix stack below is formed.
    profiled_trace = _profiled_curvature_trace(B_a, W_cell, level_rows)

    w_row = W_cell.sum(axis=1)
    s_row = S_cell.sum(axis=1)
    r = 1 + k_a
    border = np.empty((r, r), dtype=np.float64)
    border[0, 0] = W_cell.sum()
    border[0, 1:] = w_row @ B_a
    border[1:, 0] = border[0, 1:]
    border[1:, 1:] = B_a.T @ (B_a * w_row[:, None])

    u_border = np.empty(r, dtype=np.float64)
    u_border[0] = S_cell.sum()
    u_border[1:] = B_a.T @ s_row

    return SplineCatPair(
        V=V,
        U=(B_a.T @ Sq).T,
        c=(B_a.T @ Wq).T,
        m=Wq.sum(axis=0),
        S_a=np.zeros((k_a, k_a)) if S_a is None else np.asarray(S_a, dtype=np.float64),
        border=border,
        u_border=u_border,
        u_cat=Sq.sum(axis=0),
        profiled_trace=profiled_trace,
    )


def _overlap_arrow(p: SplineCatPair):
    """The overlap curvature ``M`` in arrow form: one scalar block per level.

    ``M``'s categorical-main block is diagonal and its border is the same
    ``[intercept | spline main]`` corner, so ``M`` is an arrow matrix with
    ``g = 1``.  Profiling therefore costs O(L) rather than the O(L^3) a dense
    ``M^-1`` would.
    """
    L, k_a = p.dims
    E = np.empty((L, 1 + k_a, 1), dtype=np.float64)
    E[:, 0, 0] = p.m
    E[:, 1:, 0] = p.c
    return factor_arrow(p.m.reshape(L, 1, 1), E, p.border)


def _unpenalized_blocks(p: SplineCatPair) -> NDArray:
    """The ``(L, k_a + 1, k_a + 1)`` level blocks at ``lambda = 0``."""
    L, k_a = p.dims
    g = k_a + 1
    G = np.empty((L, g, g), dtype=np.float64)
    G[:, :k_a, :k_a] = p.V
    G[:, :k_a, k_a] = p.c
    G[:, k_a, :k_a] = p.c
    G[:, k_a, k_a] = p.m
    return G


def block_ranks(p: SplineCatPair) -> NDArray:
    """The rank of every level block, counted where the two terms BALANCE.

    A block is ``P_q + lambda T`` with both terms PSD, so its null space is
    ``null(P_q) & null(T)`` for every ``lambda > 0`` and its rank is the same
    throughout.  That lets it be counted ONCE, and counted on the one member
    of the family a relative eigenvalue cut can actually resolve: each term
    scaled to unit trace, so neither is judged against the other's magnitude.

    Counting it at a bracket edge instead — which is what reading it off the
    factorization does — loses a whole degree of freedom whenever the two
    terms are far enough apart there.  Measured on a 20-level pair with one
    level weighted 1/100th of the others: the block-rank sum reads 227 at
    each edge where the balanced count and ``numpy.linalg.matrix_rank`` both
    read 228, and the pair's ``edf0`` came back 17.99995 against the dense
    path's 18.99991.  The low edge is not safe either — it under-counted by
    one on the same pair at EQUAL weights, the mirror image of the same
    failure.  Rare levels are the routine case at high cardinality, so this
    is not an edge case; see :mod:`superglm.screening._arrow`.
    """
    _, k_a = p.dims
    tiny = np.finfo(np.float64).tiny
    G = _unpenalized_blocks(p)
    G /= np.maximum(np.einsum("lpp->l", G, optimize=True), tiny)[:, None, None]
    G[:, :k_a, :k_a] += p.S_a / max(float(np.trace(p.S_a)), tiny)
    return psd_ranks(G)


def _pair_arrow(p: SplineCatPair, lam: float, ranks: NDArray | None = None):
    """``K(lambda)`` in arrow form: one ``(k_a + 1)`` block per level.

    Level q's block holds its tensor coefficients beside its own contrast;
    the border holds the intercept and the spline main, the only two things
    every level shares.  ``C``'s spline-main rows are literally ``V``'s
    diagonal blocks — both are ``sum_i w_i A_i A_i'`` restricted to the level
    — so they are taken from the same array rather than reassembled.
    """
    L, k_a = p.dims
    g, r = k_a + 1, 1 + k_a
    G = _unpenalized_blocks(p)
    G[:, :k_a, :k_a] += lam * p.S_a
    E = np.empty((L, r, g), dtype=np.float64)
    E[:, 0, :k_a] = p.c
    E[:, 0, k_a] = p.m
    E[:, 1:, :k_a] = p.V
    E[:, 1:, k_a] = p.c
    return factor_arrow(G, E, p.border, block_ranks=ranks)


def _profile(p: SplineCatPair) -> tuple[NDArray, int]:
    """``(U_eff, rank(M))`` — the whole lambda-independent half of the work.

    ``U_eff = U - C' M^-1 u_m``.  Column ``(p, q)`` of ``C`` is nonzero in
    exactly three places — the intercept row, the spline-main rows, and level
    q's own contrast row — so the contraction never touches a level other
    than its own.  ``M`` depends on no lambda, so this is computed once and
    every rung of the ladder reuses it.
    """
    f = _overlap_arrow(p)
    L, _ = p.dims
    w_cat, w_border = f.solve(p.u_cat.reshape(L, 1), p.u_border)
    U_eff = p.U - (p.c * (w_border[0] + w_cat.reshape(L))[:, None] + p.V @ w_border[1:])
    return U_eff, f.rank


def _evaluate(
    p: SplineCatPair,
    U_eff: NDArray,
    rank_m: int,
    lam: float,
    ranks: NDArray | None = None,
) -> tuple[float, float]:
    """``(T, edf)`` at one lambda, from ONE arrow factorization.

    ``rank(V_eff + lambda S)`` is Guttman rank additivity on the bordered
    system — ``rank(K) = rank(M) + rank(A)``.  ``ranks`` carries the level
    blocks' contribution, counted once by :func:`block_ranks` because it does
    not depend on ``lambda`` and cannot be resolved reliably at the edges
    where the ladder brackets; the border's own rank comes from this
    factorization.
    """
    L, k_a = p.dims
    f = _pair_arrow(p, lam, block_ranks(p) if ranks is None else ranks)
    b = np.zeros((L, k_a + 1), dtype=np.float64)
    b[:, :k_a] = U_eff
    x, _ = f.solve(b, np.zeros(1 + k_a, dtype=np.float64))
    T = float(np.sum(U_eff * x[:, :k_a]))
    blocks = f.diag_blocks()[:, :k_a, :k_a]
    rank_term = float(f.rank - rank_m)
    penalty_term = lam * float(np.einsum("lpr,rp->", blocks, p.S_a, optimize=True))
    edf = rank_term - penalty_term
    # A PSD penalty and inverse require both penalty_term >= 0 and
    # 0 <= edf <= rank_term.  Violating either side means the factorization's
    # numerical rank and inverse action disagree, and accepting it can make a
    # ladder search converge to a plausible but wrong row.  Correct only
    # round-off at an endpoint; signal anything material so the caller refuses
    # this structured route.
    roundoff = _edf_roundoff(rank_term, penalty_term)
    if (
        not np.isfinite(penalty_term)
        or not np.isfinite(edf)
        or penalty_term < -roundoff
        or edf < -roundoff
        or edf > rank_term + roundoff
    ):
        raise _UnstableStructuredEDFError(
            f"structured EDF is numerically inconsistent: {edf} "
            f"(rank={rank_term}, penalty trace={penalty_term})"
        )
    if penalty_term < 0.0 or edf > rank_term:
        edf = rank_term
    if edf < 0.0:
        edf = 0.0
    return T, edf


def structured_ladder(
    p: SplineCatPair,
    *,
    budgets: tuple[float, ...] = (4.0,),
    max_evaluations: int | None = None,
) -> list[ScreenedPair] | None:
    """Score one spline x categorical pair at every budget, structurally.

    Mirrors :func:`penalized_score_statistic_ladder`'s contract — clamp a
    budget outside the bracket to the nearest edge and report the edf
    actually achieved — but every evaluation is an arrow factorization rather
    than a dense one.

    **Whether the ladder searches is not a function of the pair's dimensions,
    so the caller caps it and the decision is taken here.**  A rung whose
    budget falls inside the bracket bisects, and each step of that bisection
    is a fresh arrow factorization where the dense ladder's equivalent is
    ``O(k)`` on a prebuilt pencil.  Whether any rung does depends on ``edf``
    at maximum penalty, which is the dimension of the penalty's null space
    per level: measured at ``L - 1`` for ``ps``, ``bs`` and ``cr`` margins,
    where every rung clamps and the whole ladder is 2 evaluations — but at
    ZERO for ``ns``, whose penalty is full rank, so every rung searches and a
    400-level pair measured 106.  ``max_evaluations`` bounds the arrow
    factorizations this call may spend.  The bracket settles which rungs
    search, and the worst case for those is checked against the ceiling
    BEFORE the first bisection step, so a pair that cannot afford its search
    pays only for the bracket and returns ``None`` — the caller's cue for the
    same NaN row an unaffordable dense pair gets.  ``max_evaluations=None``
    means unbounded.

    A numerical failure reached while bisecting one target refuses that target,
    not independent targets or already certified edge clamps.  The returned
    list may therefore contain fewer entries than ``budgets``.  If no rung
    survives, ``None`` preserves the pair-refusal signal and lets a speculative
    structured route hand the dense path back.
    """
    if p.profiled_trace is None:
        return None

    U_eff, rank_m = _profile(p)
    ranks = block_ranks(p)

    def evaluate(lam: float) -> tuple[float, float] | None:
        try:
            return _evaluate(p, U_eff, rank_m, lam, ranks)
        except _UnstableStructuredEDFError:
            return None

    if not np.any(p.S_a):
        # No penalty to scan, exactly the predicate the dense ladder applies:
        # one rung, at the block's own achieved rank, with lambda0 = 0.  A
        # zero penalty would otherwise make the bracket below infinite and
        # every rung NaN, since inf * 0 is not a number.
        evaluated = evaluate(0.0)
        if evaluated is None:
            return None
        stat, rank = evaluated
        return [ScreenedPair(statistic=stat, edf0=rank, lambda0=0.0) for _ in budgets]

    # Use the curvature the pencil actually turns on.  ``profiled_trace`` is
    # assembled from nonnegative centered residual energies in
    # ``spline_cat_moments``; neither a dense V_eff nor the catastrophic
    # ``tr(V) - tr(C' M^+ C)`` difference is formed.  This direct scale fixes
    # the bracket itself, so the reachable-rank contract proposed in issue
    # #204 is unnecessary: there are no speculative endpoint evaluations to
    # classify, and evaluation counts, duplicate caching and genuine
    # unreachable clamps retain their existing contract.
    #
    # The 1e+-10 edges are the dense ladder's, kept identical so a pair the
    # two paths can both score gets the same lambda0.
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    scale = max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale

    evaluated_lo = evaluate(lo)
    evaluated_hi = evaluate(hi)
    if evaluated_lo is None or evaluated_hi is None:
        return None
    stat_lo, edf_lo = evaluated_lo
    stat_hi, edf_hi = evaluated_hi
    # More penalty cannot add effective dimensions.  The ladder itself only
    # resolves EDF to ``_EDF_TOL``, so sub-tolerance ordering noise is not
    # evidence of a different numerical branch.
    if edf_hi > edf_lo + max(_EDF_TOL, _edf_roundoff(edf_lo, edf_hi)):
        return None

    # DISTINCT search targets, not rungs: the ladder's budgets are permitted to
    # repeat, every copy of one bisects to the same lambda, and charging each
    # copy separately would let repeating a budget decide whether the pair is
    # screenable at all.  The same set drives the cache below, so a repeat
    # costs nothing rather than a second bisection.
    searchable = {float(b) for b in budgets if edf_hi < float(b) < edf_lo}
    if max_evaluations is not None and 2 + _MAX_STEPS_PER_RUNG * len(searchable) > max_evaluations:
        return None

    solved: dict[float, ScreenedPair | None] = {}
    out: list[ScreenedPair] = []
    for budget in budgets:
        edf0 = float(budget)
        if edf0 not in searchable:
            lam = lo if edf0 >= edf_lo else hi
            stat, achieved = (stat_lo, edf_lo) if lam == lo else (stat_hi, edf_hi)
            out.append(ScreenedPair(statistic=stat, edf0=achieved, lambda0=float(lam)))
            continue
        if edf0 not in solved:
            a, b = lo, hi
            edf_a, edf_b = edf_lo, edf_hi
            lam, stat, achieved = hi, stat_hi, edf_hi
            refused = False
            for _ in range(_MAX_BISECT):
                if b <= a * (1.0 + 1e-12):
                    break
                lam = float(np.sqrt(a * b))
                evaluated = evaluate(lam)
                if evaluated is None:
                    refused = True
                    break
                stat, achieved = evaluated
                # Preserve the same monotone certificate inside the shrinking
                # bracket; a finite in-range EDF can still be on a broken
                # numerical branch.  As at the endpoints, noise below the
                # ladder's own target tolerance is not a refusal.
                ordering_tolerance = max(_EDF_TOL, _edf_roundoff(edf_a, achieved, edf_b))
                if achieved > edf_a + ordering_tolerance or achieved < edf_b - ordering_tolerance:
                    refused = True
                    break
                achieved = min(max(achieved, edf_b), edf_a)
                if abs(achieved - edf0) <= _EDF_TOL:
                    break
                if achieved > edf0:
                    a = lam
                    edf_a = achieved
                else:
                    b = lam
                    edf_b = achieved
            # Width exhaustion is not convergence: a numerically discontinuous
            # EDF curve can keep the target bracketed while never attaining it.
            # Do not cache or publish a plausible nearest endpoint as the rung.
            if refused or abs(achieved - edf0) > _EDF_TOL:
                solved[edf0] = None
            else:
                solved[edf0] = ScreenedPair(
                    statistic=stat,
                    edf0=achieved,
                    lambda0=float(lam),
                )
        result = solved[edf0]
        if result is not None:
            out.append(result)
    # A failure while searching one target says nothing about already certified
    # edge clamps or independent targets.  Preserve those rungs; if none
    # survive, retain the existing pair-refusal signal so a speculative
    # structured route can hand the dense path back.
    return out or None
