"""Structured screening kernel for a spline x categorical candidate pair.

The dense path factorizes the pair's ``(k, k)`` curvature with ``k = k_a * L``,
cubic in the level count. With the categorical margin's treatment menu the
bordered system ``K(lambda) = [[V + lambda S, C'], [C, M]]`` is exactly
block-arrow once grouped by level: ``V`` and ``S = kron(S_a, I)`` are
block-diagonal (levels have disjoint row support), ``M``'s categorical-main
block is diagonal, and ``C``'s rows are level-local. Only the intercept and the
spline main couple the levels, a border of ``1 + k_a`` columns independent of
``L``, so :func:`_profile` and :func:`_evaluate` work in time and memory linear
in ``L``.

The ladder needs ``T(lambda) = U_eff' A(lambda)^-1 U_eff`` and
``edf(lambda) = tr(A(lambda)^-1 V_eff)``, both read off one row-space
factorization built from the design, never from the pair's moments (forming
those in float64 destroys the quantity before any rule runs). The edf is a sum
of Tikhonov filter factors ``a_j / (a_j + lambda s_j)``, each in ``[0, 1]``, not
a difference of traces; this is the standard form (Wood, *JRSS-B* 73:3-36,
2011, sec. 3.6; Hansen, *Regularization Tools* v4.1, 2008, sec. 2.4; Golub,
Heath and Wahba, *Technometrics* 21:215-223, 1979). With ``Zc_q`` level
``q``'s centered weighted spline rows, ``J = blockdiag(Zc_q)`` and ``P`` the
projector onto the residualized overlap span, ``V_eff = J'(I - P)J`` (Frisch-
Waugh-Lovell absorption) is the Gram of a residualized design. Everything is
built from QRs of ``[R_q ; sqrt(lambda) rootS]``-style stacks and never from a
sum or difference of Grams (Gill, Golub, Murray and Saunders, *Math. Comp.*
28:505-535, 1974, sec. 5.2); the block-angular structure is standard (Golub and
Plemmons, *LAA* 34:3-28, 1980; Scott and Tuma, *Numer. Algorithms* 79, 2018).

Those references assume a nonsingular pencil. Here ``S`` is only semidefinite,
and ``null(V_eff) & null(S)`` is nonempty on starved pairs (one direction of 78
on the suite's starved ``bs`` pair), while both operands are individually rank
deficient on most fixtures, so singularity cannot be decided reliably.

The singular-pencil policy:

1. On the common null space the generalized eigenvalue is 0/0 and the pencil is
   singular (Cao, *LAA* 92:187-196, 1987, for ``B`` semidefinite; the converse
   fails without semidefiniteness). Such a direction contributes zero to
   ``edf``, as ``tr(A^+ V_eff)`` with the Moore-Penrose inverse does, and the
   pair is scored. This is a choice about an undefined quantity, with no error
   bound and no published convention to adopt (LAPACK Users' Guide, 3rd ed.,
   sec. 2.3.5.3, calls the values undefined).
2. Only a broken evaluation is refused (:class:`_UnstableStructuredEDFError`:
   the delivered number is not a filter-factor sum), never a singular pencil.
   :func:`structured_ladder` also returns ``None`` where
   :func:`_representative_projection` cannot place the centered-row rank, and
   where the monotone-ordering guards fail on an indefinite delivered
   ``V_eff``.
3. Never deflate where the pencil is not singular. Cuts that decide a
   direction's side are documented bounds, not chosen tolerances:
   :func:`_penalty_root`, :func:`_psd_clip_allowance`, and
   :func:`_orthonormality_bound` (which only refuses, never cuts). Four more
   are conventions, which a new cut must say it is: :func:`_profile`'s
   ``max(shape) eps s_max``, :func:`_block_inverse_factors`' cut at the same
   convention, :func:`_absorption_floor`'s ``sqrt(d eps)`` and
   :func:`_representative_projection`'s ``sqrt(k eps)``, the dense path's own
   cutoff. A reproducible cut need not be derivable.
4. Where the cut itself is undecidable, say so. A penalty residue below
   ``eigh``'s error bar is taken at its magnitude; where ``eigh`` returns an
   exact ``0.0`` the direction leaves the penalty, is reported free (filter
   factor one, the opposite end from clause 1) and ``edf`` is too high. On
   ``_rank_one_penalty_pair(2, 10, 20, 1e-4, 3)`` this publishes 9.000000
   against a 40-digit oracle's 5.720073; every float64 route does the same,
   because ``eigh`` destroys the information.
5. The dense route follows the same policy: its pencil is a pivoted QR of
   ``[R_eff ; rootS]`` truncated at ``sqrt(k eps)``, so the common null space
   is discarded on every pair, and both routes root the penalty through
   ``superglm.screening._factor_kernels._penalty_root``.

Range guard: every term of the edf sum is a squared norm, so ``edf >= 0`` by
construction; the upper end ``L k_a`` still guards a deflation that misfires.
The two clips behind "by construction" are bounded, not trusted:
:func:`_psd_factor`'s ``max(w, 0)`` is held to :func:`_psd_clip_allowance`, and
:func:`_penalty_root`'s drop branch is refused at ``n eps ||S||_2`` and its
magnitude branch at ``2 n^2 eps |tr S|``.

Reproducibility: permuting level coordinates, or changing BLAS kernel or thread
count, moves ``edf`` by about 1e-12. Monotonicity in lambda holds for a
definite pencil (Kaufman and Rosset, *Biometrika* 101:771-784, 2014) and can
fail on this route when the delivered ``V_eff`` is indefinite (issues #263,
#269); the ladder guards for it. When the border Schur complement is nearly
singular (8 levels of 3 rows against an 11-column margin) no organization of
the arithmetic recovers the lost digits and the ladder refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._roundoff import EPS
from superglm.screening._factor_kernels import _combine_row_factors, _penalty_root
from superglm.screening._score_stat import ScreenedPair, _lambda_bracket

_EDF_TOL = 1e-6
_EDF_ROUNDOFF_FACTOR = 64.0
_PSD_CLIP_FACTOR = 8.0
_EDF_ABSOLUTE_DUST = np.finfo(np.float64).eps ** 2
_MAX_BISECT = 200
_TRACE_CHUNK_DOUBLES = 262_144

# Safety factor in :func:`_orthonormality_bound`: 4.9x the largest residual measured
# over the fixture bank, across 7 OPENBLAS_CORETYPE microkernels and 2 thread settings.
_ORTHONORMALITY_FACTOR = 2.0

# Smallest ratio of a singular value to the absorption floor at which the deflation
# count is reported as decided; inside it :func:`_evaluate` refuses (Foster and Davis,
# *ACM TOMS* 40(1), 2013, secs. 1-2: certify or warn). 1.01 clears the 4.6e-11
# cross-kernel spread of singular values; 1.5 refuses 73 of 601 lambdas.
_ABSORPTION_MARGIN = 1.01

# Most steps one rung's bisection can take (a 1e20 bracket halved in log to 1e-12
# relative width), so a caller can price a search before paying for it.
_MAX_STEPS_PER_RUNG = min(_MAX_BISECT, 46)


class _UnstableStructuredEDFError(FloatingPointError):
    """A filter-factor sum came back outside ``[0, k]``, so it is not one.

    Raised only when the delivered number is not a filter-factor sum: outside the
    range, non-finite, or resting on a projection larger than the backward error that
    would explain it. Never raised for a singular pencil (clause 2 of the module
    policy). The two guards in :func:`_evaluate` judge a computed sum against its
    bound; the two penalty guards in :func:`_profile` judge ``S_a`` alone, the
    caller's specification, and fire identically at every lambda. An indefinite
    derived ``V_eff`` is not a raise condition: refuse a changed model, evaluate a
    hard pencil. :func:`structured_ladder` also declines without raising.
    """


def _edf_roundoff(*values: float) -> float:
    """Scale-aware dust allowance for EDF identities and ordering."""
    return max(
        _EDF_ABSOLUTE_DUST,
        _EDF_ROUNDOFF_FACTOR * EPS * sum(abs(float(value)) for value in values),
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
    # Row-space factors, not moments: the ladder's edf and statistic are computed from
    # these and ``overlap_rows``/``score_rows``/``base_score``, never from ``V``/``c``/``m``.
    R: NDArray  # (L, k_a, k_a)     centered row factor of each emitted level
    overlap_rows: NDArray  # (., 1 + k_a)  compacted rows of the unemitted levels
    # score_rows[q] = Zc_q' zw_q (level q's centered rows against its score);
    # base_score = [1' s ; B' s] over the unemitted levels. Together they are all U_eff
    # needs from the design.
    score_rows: NDArray  # (L, k_a)      Zc_q' zw_q per emitted level
    base_score: NDArray  # (1 + k_a,)    unemitted levels' overlap score

    @property
    def dims(self) -> tuple[int, int]:
        return self.U.shape  # (L, k_a)


def _centered_level_means(shifted: NDArray, W: NDArray) -> NDArray:
    """Weighted mean of the shifted basis rows inside each level (zero for a zero-mass level).

    Shared by the factor and score centerings so both use one residualization.
    """
    n_levels = W.shape[1]
    mass = W.sum(axis=0)
    means = np.zeros((n_levels, shifted.shape[1]), dtype=np.float64)
    np.divide(W.T @ shifted, mass[:, None], out=means, where=mass[:, None] > 0.0)
    return means


def _centered_level_scores(
    B: NDArray, W_cell: NDArray, S_cell: NDArray, level_rows: NDArray
) -> NDArray:
    """``Zc_q' zw_q`` per emitted level, from the design and not from ``M^+``.

    Level ``q``'s contribution is ``(B - mean_q)' S_q``: the weight cancels
    algebraically, so a tiny-weight level's score is never divided by its weight.
    The same ``B[0]`` shift as :func:`_centered_level_factors` is applied. Centering
    the rows before the contraction was measured against 60-digit mpmath and agrees
    to every digit, because ``mean_q`` is accumulated from the same terms.
    """
    B = np.asarray(B, dtype=np.float64)
    level_rows = np.asarray(level_rows, dtype=np.intp)
    k = B.shape[1]
    if level_rows.size == 0 or k == 0 or B.shape[0] == 0:
        return np.zeros((level_rows.size, k), dtype=np.float64)
    shifted = B - B[0]
    Wq = np.asarray(W_cell, dtype=np.float64)[:, level_rows]
    Sq = np.asarray(S_cell, dtype=np.float64)[:, level_rows]
    means = _centered_level_means(shifted, Wq)
    return (shifted.T @ Sq).T - means * Sq.sum(axis=0)[:, None]


def _unemitted_overlap_score(B: NDArray, S_cell: NDArray, level_rows: NDArray) -> NDArray:
    """``[1' s ; B' s]`` summed over the levels the contrast menu does not emit.

    The right-hand-side counterpart of :func:`_unemitted_overlap_rows`; the weight
    cancels as in :func:`_centered_level_scores`.
    """
    B = np.asarray(B, dtype=np.float64)
    S_cell = np.asarray(S_cell, dtype=np.float64)
    k_a = B.shape[1]
    out = np.zeros(1 + k_a, dtype=np.float64)
    unemitted = np.ones(S_cell.shape[1], dtype=bool)
    unemitted[np.asarray(level_rows, dtype=np.intp)] = False
    columns = np.flatnonzero(unemitted)
    if columns.size == 0 or S_cell.shape[0] == 0:
        return out
    rows = S_cell[:, columns].sum(axis=1)
    out[0] = float(rows.sum())
    out[1:] = B.T @ rows
    return out


def _centered_level_factors(B: NDArray, W: NDArray) -> NDArray:
    """QR factors ``R_l`` of each level's centered weighted spline rows.

    ``R_l' R_l = (sqrt(W_l) D_l)' (sqrt(W_l) D_l)`` without forming the Gram: the
    raw-moment form ``B'WB - (B'w)(B'w)'/sum(w)`` cancels catastrophically. Rows are
    shifted by the first basis row so centering is invariant to a common offset.
    Zero-mass levels give an exact zero matrix.
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
    means = _centered_level_means(shifted, W)
    centered = shifted[:, None, :] - means[None, :, :]
    centered *= np.sqrt(W)[:, :, None]
    raw = np.linalg.qr(np.moveaxis(centered, 1, 0), mode="r")
    if raw.shape[1] == k:
        return raw
    factors = np.zeros((n_levels, k, k), dtype=np.float64)
    factors[:, : raw.shape[1], :] = raw
    return factors


def _reduce_row_factors(base: NDArray, blocks: NDArray) -> NDArray:
    """Compact ``base`` stacked under a batch of per-level row factors.

    A binary reduction tree on the ``R`` factors alone (communication-avoiding TSQR,
    Demmel, Grigori, Hoemmen and Langou, *SIAM J. Sci. Comput.* 34(1):A206-A239,
    2012, sec. 2): pair the blocks, factor every pair in one batched call, repeat.
    Operands stay ``(2w, w)`` for any level count, so ``L`` appears only as a batch
    count; TSQR is backward stable. Blocks shorter than ``w`` are zero-padded and
    taller ones are reduced once, batched, before entering the tree.
    """
    base = np.asarray(base, dtype=np.float64)
    blocks = np.asarray(blocks, dtype=np.float64)
    n, h, w = blocks.shape
    if n == 0 or w == 0:
        return base
    if h > w:
        blocks = np.linalg.qr(blocks, mode="r")
        h = w
    if h < w:
        padded = np.zeros((n, w, w), dtype=np.float64)
        padded[:, :h, :] = blocks
        blocks = padded
    while blocks.shape[0] > 1:
        pairs = blocks.shape[0] // 2
        merged = np.linalg.qr(blocks[: 2 * pairs].reshape(pairs, 2 * w, w), mode="r")
        odd = blocks[2 * pairs :]
        blocks = np.concatenate((merged, odd), axis=0) if odd.size else merged
    return _combine_row_factors(base, blocks[0])


def _representative_projection(
    row_factor: NDArray,
    *,
    n_rows: int,
    n_levels: int,
) -> tuple[NDArray, NDArray] | None:
    """A stable representative span and its structural null action, or ``None``.

    Column-pivoted QR of the small global weighted-row factor picks representative
    columns ``active`` (``Z = Z_active [I, C]``, ``R11 C = R12``); using only the
    active rows is sufficient because the rest differ by a null-space action.
    ``null_action`` is ``I - P`` built structurally (active rows ``[0, -C]``,
    inactive rows ``[0, I]``), with no cancelling ``I - H^+ H``. The cutoff is
    :func:`superglm.screening._factor_kernels._factor_rank_floor`, shared with the
    dense path. Rank is refused (``None``) when a pivot lies within its QR
    backward-error interval of the cutoff, so row or level order cannot flip a
    retained direction.
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
    # The band is capped at half the cutoff: uncapped it grows with the reduction depth
    # until it contains zero, and every pivot would read as ambiguous.
    uncertainty = min(16.0 * eps * reduction_depth, 0.5 * np.sqrt(max(k, 1) * eps))
    uncertainty *= leading_scale
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

    With ``G = [local_active; other_active] = Q R``, the two least-squares actions are
    ``A_q = R^-1 Q_local' local`` and ``A_-q = R^-1 Q_other' other``: the same fit as
    ``H^+ H_q`` and ``H^+ H_-q`` without forming ``G'G``, sharing one triangular solve.
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
    *,
    keep_factors: NDArray | None = None,
) -> float | None:
    """``tr(V_eff)`` as an exact sum of squared residual norms.

    For emitted level ``q`` the Schur-complement trace is
    ``||R_q (N + A_-q)||_F^2 + ||R_-q A_q||_F^2``, ``N`` the null action of
    :func:`_representative_projection`; summed over levels this is ``tr(V - C' M^+ C)`` as
    nonnegative terms, representable when the mains absorb all but round-off. ``R_-q`` is a QR
    compaction of prefix and suffix row factors, never a subtraction from a global Gram; work
    and memory are linear in the level count. ``keep_factors``, an ``(len(level_rows), k, k)``
    output, receives each ``R_q`` so the ladder needs no third chunked QR.
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
    emitted_index = np.full(n_levels, -1, dtype=np.intp)
    emitted_index[level_rows] = np.arange(level_rows.size, dtype=np.intp)
    prefix = np.zeros((k_a, k_a), dtype=np.float64)
    trace = 0.0
    correction = 0.0

    for start in range(0, n_levels, chunk):
        stop = min(n_levels, start + chunk)
        factors = _centered_level_factors(B, W_cell[:, start:stop])
        for level in range(start, stop):
            local = factors[level - start]
            if emitted[level]:
                if keep_factors is not None:
                    keep_factors[emitted_index[level]] = local
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


def _unemitted_overlap_rows(
    B: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> NDArray:
    """Compact the overlap rows of the levels the treatment-coded menu does not emit.

    Base levels keep their rows ``[sqrt(w) | sqrt(w) B]`` uncentered, so only their
    combined factor is needed; it is compacted chunk by chunk and never held. Emitted
    levels are centered by :func:`_centered_level_factors`.
    """
    B = np.asarray(B, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    n_rows, k_a = B.shape
    n_levels = W_cell.shape[1]
    width = 1 + k_a
    unemitted = np.ones(n_levels, dtype=bool)
    unemitted[np.asarray(level_rows, dtype=np.intp)] = False
    columns = np.flatnonzero(unemitted)
    factor = np.zeros((0, width), dtype=np.float64)
    if columns.size == 0 or n_rows == 0:
        return factor
    chunk = _trace_chunk_width(n_rows, width, columns.size)
    for start in range(0, columns.size, chunk):
        block = columns[start : start + chunk]
        root = np.sqrt(W_cell[:, block]).T[:, :, None]  # (levels, n_rows, 1)
        rows = np.concatenate((root, root * B[None, :, :]), axis=2)
        factor = _reduce_row_factors(factor, np.linalg.qr(rows, mode="r"))
    return factor


def spline_cat_moments(
    B_a: NDArray,
    S_a: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> SplineCatPair:
    """Assemble the level-blocked moments straight from the pair's cell tables.

    ``level_rows[q]`` is the cell-table column the q-th contrast indicates, so
    selecting it multiplies by the treatment menu without building it. A missing
    ``S_a`` is carried as a zero penalty: an unpenalized block has no bandwidth to
    scan and :func:`structured_ladder` reports a single rung.
    """
    B_a = np.asarray(B_a, dtype=np.float64)
    n_a, k_a = B_a.shape
    level_rows = np.asarray(level_rows, dtype=np.intp)
    Wq = W_cell[:, level_rows]
    Sq = S_cell[:, level_rows]

    # The trace pass runs first: building ``V`` after the trace's suffix is released keeps
    # the phase at two level-sized stacks, which is what ``_within_structured_budget`` counts.
    R = np.zeros((level_rows.size, k_a, k_a), dtype=np.float64)
    profiled_trace = _profiled_curvature_trace(B_a, W_cell, level_rows, keep_factors=R)
    overlap_rows = _unemitted_overlap_rows(B_a, W_cell, level_rows)
    score_rows = _centered_level_scores(B_a, W_cell, S_cell, level_rows)
    base_score = _unemitted_overlap_score(B_a, S_cell, level_rows)

    # One GEMM forms every level's k_a x k_a curvature from level-independent outer
    # products; the n_a*k_a^2 scratch is released before anything else level-sized.
    AA = (B_a[:, :, None] * B_a[:, None, :]).reshape(n_a, k_a * k_a)
    V = (Wq.T @ AA).reshape(-1, k_a, k_a)
    del AA

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
        R=R,
        overlap_rows=overlap_rows,
        score_rows=score_rows,
        base_score=base_score,
    )


@dataclass(frozen=True)
class _PairGeometry:
    """Everything the ladder needs that does not depend on ``lambda``.

    ``V_eff`` is the Gram of a residualized design (Frisch-Waugh-Lovell): with ``Zc_q`` level
    ``q``'s centered rows, ``J = blockdiag(Zc_q)``, ``P`` the projector onto the overlap span
    ``Abar = Q R`` and ``Psi_q = Q_q' Zc_q``, ``V_eff = J'(I - P)J = blockdiag(D_q) - Psi'Psi``
    and ``A(lambda) = blockdiag(N_q) - Psi'Psi``, ``N_q = D_q + lambda S_a``; ``Psi`` is bounded
    by the data scale as ``Q`` is orthonormal. ``coupling`` maps a level's row factor into the
    overlap coordinates (``Psi_q = (R_q coupling)' R_q``); ``base_gram`` factors
    ``sum_b Q_b' Q_b`` over unemitted levels. ``ceiling = L * k_a`` counts the filter factors,
    each in ``[0, 1]``, bounding ``edf`` without a rank decision (an O(L) numerical rank can
    come out low and refuse true values).
    """

    U_eff: NDArray  # (L, k_a)      the profiled score
    coupling: NDArray  # (k_a, r)   R_q -> the shared overlap coordinates
    base_gram: NDArray  # (., r)    factor of sum_b Q_b' Q_b, unemitted levels
    root_penalty: NDArray  # (., k_a)  rootS' rootS = the PSD part of S_a
    orthonormality: float  # ||sum_q Q_q' Q_q - I_r||_2, MEASURED; a diagnostic
    # ``s_max / s_min`` over the retained overlap basis. A diagnostic only: spent on
    # :func:`_orthonormality_bound` in :func:`_profile` and kept so a caller can
    # recompute that bound from the geometry.
    conditioning: float
    overlap_rank: int  # r
    ceiling: float  # L * k_a


def _profile(p: SplineCatPair) -> _PairGeometry:
    """The whole lambda-independent half of the work.

    The profiled score is the Frisch-Waugh-Lovell residual ``U_eff = g - Psi'(Q'zw)``,
    ``g_q = Zc_q' zw_q``; no Gram is inverted and the weight is never a denominator. The overlap
    span's orthonormal basis comes from a TSQR over the per-level row factors, rank-revealed at
    the ``matrix_rank`` cut ``max(shape) eps s_max`` (Wood, *JRSS-B* 73:3-36, 2011, sec. 3.3.1;
    Gill et al., *Math. Comp.* 28, 1974, sec. 5.2). Nothing is ``L``-sized. Raises
    :class:`_UnstableStructuredEDFError` if the penalty projection was not round-off or the
    overlap basis does not close within its bound.
    """
    L, k_a = p.dims

    width = 1 + k_a
    combined_base = np.asarray(p.overlap_rows, dtype=np.float64).reshape(-1, width)
    combined = combined_base
    if L:
        chunk = _trace_chunk_width(k_a, width, L)
        padded = np.zeros((min(chunk, L), k_a, width), dtype=np.float64)
        for start in range(0, L, chunk):
            block = p.R[start : start + chunk]
            padded[: block.shape[0], :, 1:] = block
            combined = _reduce_row_factors(combined, padded[: block.shape[0]])
    if combined.size:
        row_factor = np.linalg.qr(combined, mode="r")
        singular, right = np.linalg.svd(row_factor)[1:]
        cut = max(row_factor.shape) * np.finfo(np.float64).eps * float(singular[0])
        rank = int(np.count_nonzero(singular > cut))
        basis = right[:rank].T / singular[:rank]
        # Conditioning of the retained basis: an eps-sized error in ``row_factor`` reaches
        # ``Q_q = R_q basis`` amplified by ``singular[0] / singular[rank - 1]``. Both ends sit
        # above the rank cut, and unlike the measured residual this is reproducible. See
        # :func:`_orthonormality_bound`.
        conditioning = float(singular[0] / singular[rank - 1]) if rank else 1.0
    else:
        rank = 0
        conditioning = 1.0
        basis = np.zeros((width, 0), dtype=np.float64)

    # ``Q' zw`` is one ``rank``-vector: emitted levels reach the overlap span only through
    # their spline columns (centering annihilates the intercept), so the right-hand side is
    # ``base_score`` plus the emitted levels' centered scores in the spline slots.
    coupling = np.ascontiguousarray(basis[1:])
    overlap_score = np.array(p.base_score, dtype=np.float64)
    overlap_score[1:] += p.score_rows.sum(axis=0)
    direction = coupling @ (basis.T @ overlap_score)
    U_eff = p.score_rows - np.einsum("lji,lj->li", p.R, p.R @ direction, optimize=True)

    root_penalty, dropped, bar = _penalty_root(p.S_a)
    trace_S = float(np.trace(p.S_a))
    kept_S = float(np.sum(np.square(root_penalty)))
    clip = abs(trace_S - kept_S) / max(abs(trace_S), np.finfo(np.float64).tiny)
    # Both halves score one pencil (a single block-angular QR with penalty rows
    # ``sqrt(lambda) root_penalty``); what remains is that the projection itself is round-off,
    # otherwise the route scores a different model from the one specified. ``dropped`` is the
    # largest magnitude :func:`_penalty_root` removed outside the eigensolver's bar
    # ``n eps ||S||_2`` (LAPACK Users' Guide, 3rd ed., sec. 4.7), so the refusal keys to the
    # drop itself with no second constant (issue #323). A singular penalty does not reach
    # here: ``w = 0`` is inside the bar. The dense route drops the same direction and
    # publishes, so on such a pair only this route returns a NaN row.
    if dropped > 0.0:
        raise _UnstableStructuredEDFError(
            f"the penalty has a negative eigenvalue of {dropped}, which is "
            f"outside an eigensolver's backward error of {bar}: the direction was "
            "dropped from the penalty this route scores, so its edf would count "
            "a free direction the specified penalty does not leave free"
        )
    # The aggregate test covers the magnitude branch: at most ``n`` eigenvalues sit inside
    # the bar, each moves the trace by at most ``2 n eps ||S||_2``, and ``tr(S) >= ||S||_2``
    # in the cone, so the lift moves at most ``2 n^2 eps`` of trace. ``kept_S`` is read off
    # ``root_penalty``, so a wrong ``keep`` mask trips it.
    if clip > 2.0 * p.S_a.shape[0] ** 2 * np.finfo(np.float64).eps:
        raise _UnstableStructuredEDFError(
            f"the penalty's PSD projection removed {clip} of its trace, which is "
            "more than an eigensolver's backward error: the statistic and the edf "
            "would be scoring different pencils"
        )

    # ``sum_q Q_q' Q_q`` is exactly ``I_r`` in exact arithmetic, and every bound the ladder
    # rests on (``Psi N^+ Psi' <= I``, hence ``H >= 0``) is that identity, so its computed
    # residual is measured here.
    base_gram = combined_base @ basis
    closure = base_gram.T @ base_gram
    if L:
        chunk = _trace_chunk_width(k_a, rank, L)
        for start in range(0, L, chunk):
            carried = p.R[start : start + chunk] @ coupling
            closure += np.einsum("lkr,lks->rs", carried, carried, optimize=True)
    defect = float(np.linalg.norm(closure - np.eye(rank), 2)) if rank else 0.0
    # The residual is a diagnostic, not a cut: the answer depends only on whether it stays
    # inside the bound its own arithmetic predicts, which scales with the basis conditioning
    # so a badly conditioned but correct pair widens the bar and is not refused (not a
    # singularity test). Worst observed is 0.204 of this bound over 1382 profiles x 14
    # kernel/thread configurations.
    bound = _orthonormality_bound(L * k_a, rank, conditioning)
    if defect > bound:
        raise _UnstableStructuredEDFError(
            f"the overlap basis closes to {defect}, which is outside the "
            f"{bound} its own round-off allows at a conditioning of {conditioning}: "
            "sum_q Q_q' Q_q is not I_r, so H is not the Schur complement the "
            "deflation reads it as"
        )

    return _PairGeometry(
        U_eff=U_eff,
        coupling=coupling,
        base_gram=base_gram,
        root_penalty=root_penalty,
        orthonormality=defect,
        conditioning=conditioning,
        overlap_rank=rank,
        ceiling=float(L * k_a),
    )


def _block_inverse_factors(triangular: NDArray) -> NDArray:
    """Batched ``T^+`` for a stack of small triangular factors, at the LAPACK cut.

    ``T_q' T_q = D_q + lambda S_a``; a direction inside the ``matrix_rank`` cut is
    inverted to zero (clause 1 of the module policy). The border's ``keep`` in
    :func:`_filter_factor_sum` shares the convention but also gates two ``1.0``
    branches that must not move with it, so do not mirror an edit between them.
    """
    u, singular, vt = np.linalg.svd(triangular)
    cut = max(triangular.shape[-2:]) * np.finfo(np.float64).eps
    top = np.maximum(singular[..., :1], np.finfo(np.float64).tiny)
    keep = singular > cut * top
    inv = np.where(keep, 1.0 / np.where(keep, singular, 1.0), 0.0)
    return (np.swapaxes(vt, -1, -2) * inv[..., None, :]) @ np.swapaxes(u, -1, -2)


def _psd_clip_allowance(n: int, scale: float, closure: float) -> float:
    """Largest negative eigenvalue an exactly-PSD block can show at ``scale``.

    ``_psd_factor``'s ``max(w, 0)`` (Higham, *LAA* 103, 1988) also clips a block made indefinite
    by a misfired deflation, so the caller refuses beyond ``8 n eps scale + closure``: ``eigh``
    is within ``n eps ||M||_2`` (*LAPACK Users' Guide*, 3rd ed., sec. 4.7) and ``W_q`` reaches it
    through three chained products (Higham, *ASNA* 2nd ed., Thm 3.5), rounded up to 8.
    ``scale = max(||M||_2, ||H^+||_2)``, as ``Xi``'s own norm can be ``2 eps``. ``closure`` is
    the measured ``sum_q Q_q' Q_q - I`` residual, which moves an eigenvalue by
    ``closure * ||H^+||``.
    """
    return _PSD_CLIP_FACTOR * max(int(n), 1) * float(np.finfo(np.float64).eps) * np.maximum(
        scale, 0.0
    ) + np.maximum(closure, 0.0)


def _psd_factor(M: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Batched ``F`` with ``F F' = M`` for symmetric ``M``, negatives clipped.

    Returns the factor, the clipped magnitude ``max(-w_min, 0)`` (``||M - M_+||_2``) and
    the block's norm; the caller decides whether the clip was round-off, see
    :func:`_psd_clip_allowance`. Equilibrating ``M`` to a unit diagonal was tried and
    rejected: it trades one rung's error for its neighbours'.
    """
    w, Q = np.linalg.eigh(0.5 * (M + np.swapaxes(M, -1, -2)))
    clipped = np.maximum(-np.min(w, axis=-1, initial=0.0), 0.0)
    top = np.maximum(np.max(np.abs(w), axis=-1, initial=0.0), 0.0)
    return Q * np.sqrt(np.where(w > 0.0, w, 0.0))[..., None, :], clipped, top


def _orthonormality_bound(n_terms: int, rank: int, conditioning: float) -> float:
    """A priori bound on ``||sum_q Q_q' Q_q - I_r||_2``, used only to refuse, never to cut.

    It checks that the identity every bound here rests on holds to the accuracy its arithmetic
    predicts. Householder QR loses orthogonality at ``O(m rank^(3/2) eps)`` (Higham, *ASNA* 2nd
    ed., Lemma 19.3 and eq. 19.13; Yamamoto et al., *ETNA* 44:306-326, 2015, sec. 5.1.1), but
    ``Q_q = R_q V Sigma^-1`` is the ``A R^-1`` route, with error ``O(eps kappa)``, so the bound
    is linear in the stacked row count and in the retained basis's ``conditioning``.
    :data:`_ORTHONORMALITY_FACTOR` is a calibrated constant (Higham's "small integer constant")
    with ``rank^(3/2)`` folded in. ``conditioning`` is read off singular values far above the
    rank cut, so the bound is reproducible, though not bit-identical.
    """
    eps = float(np.finfo(np.float64).eps)
    return (
        _ORTHONORMALITY_FACTOR
        * max(int(n_terms), int(rank), 1)
        * eps
        * max(float(conditioning), 1.0)
    )


def _absorption_floor(n_terms: int, rank: int) -> float:
    """Cut below which a singular value of ``H``'s factor is not resolvable.

    A convention, not a bound: it separates a direction the pencil does not resolve (adding
    zero) from one it does (a whole degree of freedom), and depends on dimensions and ``eps``
    alone, so it is identical on every kernel and thread setting. ``H`` is delivered as
    ``R_H' R_H``, so ``sqrt(d eps)`` carries the ``matrix_rank`` convention ``d eps`` onto a
    singular value (Weyl; Golub and Van Loan, 4th ed., Cor. 2.4.4), plus the TSQR's additive
    ``d eps`` (Higham, *ASNA*, eq. 19.13). It is absolute, not LAPACK's relative
    ``RCOND * sigma_max``; as ``sigma_max(R_H) <= 1`` it can over-deflate a mostly-absorbed
    border but never under-deflate. :func:`_orthonormality_bound` as a cut would deflate the
    only direction of ``_near_absorbed_cells``, so it stays a refusal.
    """
    eps = float(np.finfo(np.float64).eps)
    dimension = max(int(n_terms), int(rank), 1)
    return float(np.sqrt(dimension * eps)) + dimension * eps


class _ReferenceVariance:
    """Squared smoother norm from diagonal blocks and a streaming QR tree.

    Off-diagonal blocks have the form ``Z_q M Z_t'``. Each tree merge adds
    their squared norms before compressing the joined rows, so no large
    traces are subtracted. Only final ladder rungs construct this collector.
    """

    def __init__(self, geometry: _PairGeometry):
        self.geometry = geometry
        self.squared_norm = 0.0
        self.correction = 0.0
        self.diagonal_error = 0.0
        self.leaf_norm_squared = 0.0
        self.leaf_error_squared = 0.0
        self.leaves = 0
        self.tree: list[NDArray | None] = []

    def configure(self, singular: NDArray, right: NDArray, keep: NDArray) -> None:
        self.rotation = right.T
        self.singular = singular
        self.small = keep & (singular < np.sqrt(0.5))
        r = singular.size
        occupied = singular * singular
        free = (1.0 - singular) * (1.0 + singular)
        large = keep & ~self.small
        safe = np.where(large, occupied, 1.0)
        first = np.where(self.small, 1.0, np.where(large, 1.0 / safe, 0.0))
        crossed = np.where(self.small, 0.0, np.where(large, -1.0 / safe, -1.0))
        second = np.where(self.small, -1.0, np.where(large, free / safe, free))
        self.metric = np.zeros((2 * r, 2 * r), dtype=np.float64)
        index = np.arange(r)
        self.metric[index, index] = first
        self.metric[index, r + index] = crossed
        self.metric[r + index, index] = crossed
        self.metric[r + index, r + index] = second
        self.metric_norm = float(
            np.max(
                0.5 * (np.abs(first + second) + np.hypot(first - second, 2 * crossed)), initial=0.0
            )
        )
        self.amplification = (
            max(1.0, 1.0 / float(np.min(singular[self.small]))) if np.any(self.small) else 1.0
        )

    def _add(self, term: float) -> None:
        updated = self.squared_norm + term
        if abs(self.squared_norm) >= abs(term):
            self.correction += (self.squared_norm - updated) + term
        else:
            self.correction += (term - updated) + self.squared_norm
        self.squared_norm = updated

    def add_diagonal(self, factor: NDArray, allowance: NDArray) -> None:
        gram = factor @ np.swapaxes(factor, -1, -2)
        self._add(float(np.sum(np.square(gram))))
        # Include the within-bound clip and the subsequent Gram product.
        mass = np.sum(np.square(factor), axis=(-2, -1))
        operation = _PSD_CLIP_FACTOR * factor.shape[-1] * np.finfo(np.float64).eps
        self.diagonal_error += float(np.sum(allowance + operation * mass))

    def _merge(self, left: NDArray, right: NDArray) -> NDArray:
        self._add(2.0 * float(np.sum(np.square(left @ self.metric @ right.T))))
        joined = np.vstack((left, right))
        return np.linalg.qr(joined, mode="r") if joined.shape[0] > joined.shape[1] else joined

    def add_low_rank(self, local: NDArray, overlap: NDArray, error_scale: NDArray) -> None:
        if not self.singular.size:
            return
        first = local @ self.rotation
        second = overlap @ self.rotation
        # The small-h metric is diag(1,-1). Its factor residual is amplified
        # by 1/sigma, rather than multiplying a raw inverse of size 1/h.
        first[..., self.small] = (first[..., self.small] - second[..., self.small]) / self.singular[
            self.small
        ]
        rows = np.concatenate((first, second), axis=-1)
        self.leaf_norm_squared += float(np.sum(np.square(rows)))
        operation = (
            _PSD_CLIP_FACTOR
            * (overlap.shape[-2] + 2 * self.singular.size)
            * np.finfo(np.float64).eps
            * self.amplification
        )
        self.leaf_error_squared += float(np.sum(np.square(operation * error_scale)))
        for leaf in rows:
            self.leaves += 1
            level = 0
            while level < len(self.tree) and (sibling := self.tree[level]) is not None:
                leaf = self._merge(sibling, leaf)
                self.tree[level] = None
                level += 1
            if level == len(self.tree):
                self.tree.append(leaf.copy())
            else:
                self.tree[level] = leaf.copy()

    def finish(self) -> None:
        joined = None
        for factor in self.tree:
            if factor is not None:
                joined = factor if joined is None else self._merge(factor, joined)
        self.tree.clear()
        self.squared_norm += self.correction
        self.correction = 0.0

    def value(self, edf: float) -> float:
        if edf == 0.0:
            return 0.0
        square = self.squared_norm
        if not np.isfinite(square) or square <= 0.0:
            raise _UnstableStructuredEDFError(
                "structured reference variance is not positive and finite"
            )

        # A consistency allowance under the assembly-error model, not a forward-error
        # certificate: QR depth, factor residual amplification and PSD assembly enter.
        width = max(2 * self.singular.size, 1)
        depth = max(self.leaves, 1).bit_length()
        rounding = _PSD_CLIP_FACTOR * width**2 * (depth + 1) * np.finfo(np.float64).eps
        rounding = rounding / (1.0 - rounding) if rounding < 1.0 else np.inf
        mass = self.leaf_norm_squared
        leaf_error = np.sqrt(self.leaf_error_squared) + rounding * np.sqrt(mass)
        diagonal_error = self.diagonal_error + _edf_roundoff(edf, self.geometry.ceiling)
        matrix_error = (
            diagonal_error
            + self.metric_norm * (2.0 * np.sqrt(mass) * leaf_error + leaf_error**2)
            + rounding * self.metric_norm * mass
        )
        ceiling = self.geometry.ceiling
        upper = min(ceiling, edf + diagonal_error)
        allowance = 2.0 * np.sqrt(max(upper, 0.0)) * matrix_error + matrix_error**2
        allowance += rounding * square
        lower = max(edf - diagonal_error, 0.0) ** 2 / ceiling
        if not np.isfinite(allowance) or square < lower - allowance or square > upper + allowance:
            raise _UnstableStructuredEDFError(
                f"structured squared filter sum {square} is inconsistent with EDF {edf} "
                f"(allowance={allowance}, ceiling={ceiling})"
            )
        return 2.0 * square


def _filter_factor_sum(
    p: SplineCatPair,
    geometry: _PairGeometry,
    lam: float,
    *,
    variance: _ReferenceVariance | None = None,
) -> tuple[float, float, float, float]:
    """``(T, edf, clip bound, cut margin)`` at one lambda, all from one QR per level.

    ``T = U_eff' A^+ U_eff`` and ``edf = tr(A^+ V_eff)`` share one block-angular factorization:
    with ``B = N^+ + N^+ Psi' H^+ Psi N^+`` (``A^+`` on ``range(A)``) and ``z_q = T_q^+' u_q``,
    ``T = sum_q ||z_q||^2 + ||H^(+/2) v||^2`` and ``edf = sum_q ||F_q chol(W_q)||_F^2`` over ALL
    levels, ``F_q = [Zc_q | Q_q]``, ``W_q = M_q' A^+ M_q``: squared norms only (Wood, *JRSS-B*
    73, 2011, sec. 3.6, App. H). Per emitted level one QR of
    ``[[R_q, Phi_q], [sqrt(lam) rootS, 0]]`` gives ``T_q' T_q = D_q + lam S_a``; summing ``X``
    gives ``H = I_r - Psi N^+ Psi'`` without subtraction (Golub and Plemmons, *LAA* 34, 1980).
    ``K_q = R_q T_q^+`` is contracted before the inverse (``||K_q|| <= 1``). The third value,
    ``sum_q ||F_q||_F^2 ||W_q - W_q+||_2``, bounds in df what :func:`_psd_factor`'s clip removed.
    """
    L, k_a = p.dims
    r = geometry.overlap_rank
    coupling = geometry.coupling
    root = np.sqrt(lam) * geometry.root_penalty
    height, width = k_a + root.shape[0], k_a + r
    chunk = _trace_chunk_width(height, width, L) if L else 1
    blocks = [(s, min(L, s + chunk)) for s in range(0, L, chunk)]

    # --- pass 1: one block-angular QR per level, and H's factor ------------
    contracted = np.empty((L, k_a, k_a), dtype=np.float64)
    cross = np.empty((L, k_a, r), dtype=np.float64)
    border = np.asarray(geometry.base_gram, dtype=np.float64)
    local = np.zeros((min(chunk, max(L, 1)), height, width), dtype=np.float64)
    # ``local_stat`` is ``sum_q u_q' N_q^+ u_q``; ``coupled_score`` is ``Psi N^+ U_eff``, the
    # border's share before ``H^+`` is known. Nothing L-sized is retained.
    local_stat = 0.0
    stat_correction = 0.0
    coupled_score = np.zeros(r, dtype=np.float64)
    for start, stop in blocks:
        rows = p.R[start:stop]
        phi = rows @ coupling
        view = local[: stop - start]
        view[:, :k_a, :k_a] = rows
        view[:, :k_a, k_a:] = phi
        view[:, k_a:, :k_a] = root
        # Scaling the leading column block to unit norm makes the QR's backward error relative
        # to ``X_q`` (bounded by 1) rather than ``R_q``; ``Y_q`` and ``X_q`` are untouched and
        # ``T_q`` is divided by the same factor, undone below.
        scale = np.linalg.norm(view[:, :, :k_a], axis=(1, 2))
        scale = np.where(scale > 0.0, scale, 1.0)
        view[:, :, :k_a] /= scale[:, None, None]
        factored = np.linalg.qr(view, mode="r")
        inverse = _block_inverse_factors(factored[:, :k_a, :k_a])
        block_contracted = (rows @ inverse) / scale[:, None, None]
        contracted[start:stop] = block_contracted
        cross[start:stop] = factored[:, :k_a, k_a:]
        border = _reduce_row_factors(border, factored[:, k_a:, k_a:])

        # ``||z_q||^2 = u_q' N_q^+ u_q`` as a squared norm; ``inverse`` is ``(T_q / scale)^+``.
        whitened = (
            np.einsum("lij,li->lj", inverse, geometry.U_eff[start:stop], optimize=True)
            / scale[:, None]
        )
        term = float(np.sum(np.square(whitened)))
        # Neumaier-compensated accumulation across chunk totals (chunking is deterministic).
        updated = local_stat + term
        if abs(local_stat) >= abs(term):
            stat_correction += (local_stat - updated) + term
        else:
            stat_correction += (term - updated) + local_stat
        local_stat = updated
        if r:
            coupled_score += np.einsum(
                "lkr,lk->r",
                phi,
                np.einsum("lij,lj->li", block_contracted, whitened, optimize=True),
                optimize=True,
            )
    local_stat += stat_correction

    # --- H's spectrum, from its factor's singular values -------------------
    if r:
        # Clause 1 of the module policy, at the border. A dropped direction loses only its
        # ``H^+`` coefficient (the Moore-Penrose reading of a 0/0 filter factor). The same
        # ``keep`` gates ``extended`` and ``xi``, which take the ``1.0`` branch and reach
        # ``edf`` (``I + H^+(I - H) = H^+ + P_null(H)``): they must not be zeroed with it.
        # :func:`_block_inverse_factors` shares the convention, not the formula. The cut is
        # :func:`_absorption_floor` because dropping a resolved direction costs a whole df.
        singular, right = np.linalg.svd(border)[1:]
        floor = _absorption_floor(L * k_a, r)
        keep = singular > floor
        # Both sides of the cut are reported: a rank is a property of the pair only where the
        # tolerance is near neither singular value (Foster and Davis, *ACM TOMS* 40(1), 2013,
        # sec. 1; Golub and Van Loan, 4th ed., sec. 5.4.1). An empty side scores ``inf``;
        # :func:`_evaluate` owns the refusal. The reciprocal is guarded rather than divided
        # because ``_UnstableStructuredEDFError`` is a ``FloatingPointError`` and a bare one
        # raised under ``errstate(divide="raise")`` would escape the ladder's except clause.
        resolved_side = singular[keep]
        absorbed_side = singular[~keep]
        largest_absorbed = float(absorbed_side.max()) if absorbed_side.size else 0.0
        margin = min(
            float(resolved_side.min() / floor) if resolved_side.size else np.inf,
            float(floor / largest_absorbed) if largest_absorbed > 0.0 else np.inf,
        )
        occupied = singular * singular
        free = (1.0 - singular) * (1.0 + singular)
        safe = np.where(keep, occupied, 1.0)
        # ``I + H^+ (I - H)`` is ``H^+`` plus the projector onto ``null(H)``; dropping the
        # projector is wrong by a whole df per deflated direction, upward (22.0 against a
        # certified 6.000 on the starved fixture).
        resolved = (right.T * np.where(keep, 1.0 / safe, 0.0)) @ right
        extended = (right.T * np.where(keep, 1.0 / safe, 1.0)) @ right
        xi = np.where(keep, free / safe, free)
        coupled = (right.T * xi) @ right
        coupled = 0.5 * (coupled + coupled.T)
        deflation = float(np.max(np.where(keep, 1.0 / safe, 1.0)))
    else:
        resolved = extended = coupled = np.zeros((0, 0), dtype=np.float64)
        deflation = 1.0
        margin = np.inf

    if variance is not None:
        if r:
            variance.configure(singular, right, keep)
        else:
            variance.configure(np.empty(0), np.empty((0, 0)), np.empty(0, dtype=bool))

    # ``||H^(+/2) v||^2`` for ``v = Psi N^+ U_eff``, a squared norm through ``H^+``'s
    # spectral factor; a deflated direction carries no score.
    statistic = local_stat
    if r:
        # The reciprocal is guarded as above: an exactly-zero singular value is reachable at
        # ``lam = 0`` on an unpenalized pair whose base level carries no mass.
        inverse_singular = np.where(keep, 1.0 / np.where(keep, singular, 1.0), 0.0)
        whitened_border = (right * inverse_singular[:, None]) @ coupled_score
        statistic += float(np.dot(whitened_border, whitened_border))

    # --- pass 2: one PSD factor per level, and the squared norms -----------
    total = 0.0
    correction = 0.0
    uncertified = 0.0
    small = keep & (singular < np.sqrt(0.5)) if r else np.empty(0, dtype=bool)
    residual_coordinates = bool(np.any(small))
    if residual_coordinates:
        first = np.where(small, -1.0, np.where(keep, 1.0 / safe, 0.0))
        crossed_scale = np.where(small, singular, -np.where(keep, 1.0 / safe, 1.0))
        second = np.where(small, free, xi)
    block = np.empty((min(chunk, max(L, 1)), width, width), dtype=np.float64)
    for start, stop in blocks:
        carried = p.R[start:stop] @ coupling
        Y = cross[start:stop]
        contract = contracted[start:stop]
        view = block[: stop - start]
        rows = np.concatenate((contract, carried), axis=2)
        original_mass = np.sum(np.square(rows), axis=(1, 2))
        if residual_coordinates:
            # For h = sigma**2 the retained-direction contribution is ||K Y - Phi||**2 / h -
            # ||Phi||**2. Using E = (K Y - Phi)/sigma keeps the congruent metric bounded when
            # h < 1/2; factoring the 1/h entries loses the small contraction to eps/h.
            rotated = Y @ right.T
            overlap = carried @ right.T
            residual = contract @ rotated - overlap
            overlap[..., small] = residual[..., small] / singular[small]
            rows = np.concatenate((contract, overlap), axis=2)
            view[:, :k_a, :k_a] = np.eye(k_a) + (rotated * first) @ np.swapaxes(rotated, -1, -2)
            crossed = rotated * crossed_scale
            view[:, :k_a, k_a:] = crossed
            view[:, k_a:, :k_a] = np.swapaxes(crossed, -1, -2)
            view[:, k_a:, k_a:] = np.diag(second)
        else:
            crossed = Y @ extended
            view[:, :k_a, :k_a] = np.eye(k_a) + Y @ resolved @ np.swapaxes(Y, -1, -2)
            view[:, :k_a, k_a:] = -crossed
            view[:, k_a:, :k_a] = -np.swapaxes(crossed, -1, -2)
            view[:, k_a:, k_a:] = coupled
        factor, negative, top = _psd_factor(view)
        carried_factor = rows @ factor
        term = float(np.sum(np.square(carried_factor)))
        # What the clip removed beyond round-off, in degrees of freedom: for the excess ``e_q``,
        # ``|tr(F_q (W - W_+) F_q')| <= ||F_q||_F^2 e_q`` (Weyl plus the trace bound). The scale
        # is the largest intermediate, ``||H^+|| max(1, ||Y_q||)^2``, since the products that
        # build the block can cancel.
        scale = np.maximum(top, deflation * np.square(1.0 + np.linalg.norm(Y, axis=(1, 2))))
        # ``deflation`` is ``max(1/sigma^2)`` over kept directions, so a tighter absorption
        # floor loosens this allowance quadratically; the clip is within allowance at all 400
        # lambdas on the two pairs the floor moves most.
        allowance = _psd_clip_allowance(width, scale, geometry.orthonormality * deflation)
        # Keep the pre-congruence allowance in EDF units. The new metric's
        # clip is charged against its own rows, not against the old metric's
        # norm: changing coordinates must not enlarge the accepted error.
        if residual_coordinates:
            allowance_edf = original_mass * allowance
            clipped_edf = np.sum(np.square(rows), axis=(1, 2)) * negative
            uncertified += float(np.sum(np.maximum(clipped_edf - allowance_edf, 0.0)))
            diagonal_error = allowance_edf + clipped_edf
        else:
            uncertified += float(np.sum(original_mass * np.maximum(negative - allowance, 0.0)))
            diagonal_error = original_mass * (allowance + negative)
        if variance is not None:
            variance.add_diagonal(carried_factor, diagonal_error)
            variance.add_low_rank(
                contract @ Y,
                carried,
                np.linalg.norm(contract, axis=(1, 2)) * np.linalg.norm(Y, axis=(1, 2))
                + np.linalg.norm(carried, axis=(1, 2)),
            )
        # Neumaier-style compensated accumulation: the chunk totals are
        # nonnegative, so this only keeps the scalar independent of chunking.
        updated = total + term
        if abs(total) >= abs(term):
            correction += (total - updated) + term
        else:
            correction += (term - updated) + total
        total = updated

    # The levels the menu does not emit all carry the SAME ``W_q = Xi``, so
    # their whole contribution is one term against the compacted Gram factor
    # rather than one term per level.
    factor, negative, top = _psd_factor(coupled)
    base_factor = geometry.base_gram @ factor
    total += float(np.sum(np.square(base_factor)))
    allowance = _psd_clip_allowance(
        r, np.maximum(top, deflation), geometry.orthonormality * deflation
    )
    excess = np.maximum(negative - allowance, 0.0)
    uncertified += float(np.sum(np.square(geometry.base_gram)) * float(excess))
    if variance is not None:
        variance.add_diagonal(
            base_factor[None, ...],
            np.asarray([np.sum(np.square(geometry.base_gram)) * (allowance + negative)]),
        )
        variance.add_low_rank(
            np.zeros_like(geometry.base_gram)[None, ...],
            geometry.base_gram[None, ...],
            np.asarray([np.linalg.norm(geometry.base_gram)]),
        )
        variance.finish()
    return statistic, total + correction, uncertified, margin


def _evaluate(p: SplineCatPair, geometry: _PairGeometry, lam: float) -> tuple[float, float]:
    """``(T, edf)`` at one lambda, both off one factorization of the design.

    Neither reads ``V``, ``c``, ``m`` or ``border``: a statistic from a pseudo-inverse of
    the float64 moments beside an edf from row-space factors was issue #298.
    """
    return _checked_evaluation(geometry, lam, _filter_factor_sum(p, geometry, lam))


def _checked_evaluation(
    geometry: _PairGeometry, lam: float, evaluated: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Apply the same EDF and rank guards to search and final evaluations."""
    T, edf, uncertified, margin = evaluated

    # The deflation count is only an answer where the cut separates (issue #280: one
    # direction sat 1.00489x above the floor on one microkernel and 1.018x below it on six
    # others). Where the bounds cannot confirm the rank, refuse (Foster and Davis,
    # *ACM TOMS* 40(1), 2013). A refusal costs the rung on a default ladder, the pair on a
    # one-rung ladder or the zero-penalty branch, and on the ordinary structured route a
    # NaN row, since the dense track was already ruled out by budget; only a speculative
    # handoff rescores densely. Over the fixture bank x 7 kernels x 2 thread settings the
    # tightest bracket-edge margin is 1.0350 and nothing falls inside 1.01; refusals are
    # interior (2 of 601 lambdas, 1 or 2 of 200). Re-measure if :data:`_ABSORPTION_MARGIN`
    # is raised.
    if margin < _ABSORPTION_MARGIN:
        raise _UnstableStructuredEDFError(
            f"the absorption floor separates the border's spectrum by only {margin}x "
            f"at lambda={lam}, inside the {_ABSORPTION_MARGIN}x this module requires "
            "of a count: the deflated rank is not decided by the pair"
        )

    # Every term is a filter factor, so the sum lies in ``[0, L * k_a]``; the lower end holds
    # by construction (squared norms). The upper end is still a guard: a misfired
    # deflation inverts an unresolved direction (2237 against a ceiling of 77 once). The
    # terms are nonnegative, so pairwise-summation error is ``log2(n) eps`` of the total,
    # covered by ``_EDF_ROUNDOFF_FACTOR`` up to ``n = 2**64``.
    roundoff = _edf_roundoff(edf, geometry.ceiling)
    # ``uncertified`` is the clip beyond its allowance, in df: ``max(w, 0)`` silences an
    # indefinite block as well as round-off, so the two are separated on a decidable scale.
    if not np.isfinite(edf) or edf < -roundoff or edf > geometry.ceiling + roundoff:
        raise _UnstableStructuredEDFError(
            f"structured EDF is not a filter-factor sum: {edf} (ceiling={geometry.ceiling})"
        )
    if not uncertified <= roundoff:
        raise _UnstableStructuredEDFError(
            f"the PSD clip removed {uncertified} df beyond its backward error, on {edf} "
            f"(allowance={roundoff}, ceiling={geometry.ceiling})"
        )
    # The dust band is two-sided: a value inside ``[-roundoff, roundoff]`` is collapsed to
    # zero, so a cancellation residue cannot supply a positive reference variance (3e-17
    # inflated z by about 1e8 while -3e-17 skipped the rung).
    if abs(edf) <= roundoff:
        return T, 0.0
    return T, min(edf, geometry.ceiling)


def _reference_variance(p: SplineCatPair, geometry: _PairGeometry, lam: float) -> float:
    variance = _ReferenceVariance(geometry)
    evaluated = _filter_factor_sum(p, geometry, lam, variance=variance)
    _, edf = _checked_evaluation(geometry, lam, evaluated)
    return variance.value(edf)


class _StructuredBudgetExceededError(RuntimeError):
    """The ladder needs more factor passes than the routing budget allows."""


def structured_ladder(
    p: SplineCatPair,
    *,
    budgets: tuple[float, ...] = (4.0,),
    max_evaluations: int | None = None,
    raise_on_budget: bool = False,
) -> list[ScreenedPair] | None:
    """Score one spline x categorical pair at every budget, structurally.

    Mirrors :func:`penalized_score_statistic_ladder`'s contract (a budget outside the
    bracket clamps to the nearest edge and reports the edf achieved), but every
    evaluation is an arrow factorization. Search evaluations take the statistic and edf
    from one factorization; each distinct final lambda adds one pass for its reference
    variance, with O(L (k_a + r)^3) extra work and O(k_a r + r^2 log L) tree state.

    ``max_evaluations`` bounds factor passes (``None`` is unbounded): the ladder reserves
    the worst-case search cost plus one variance pass per distinct final lambda, and a
    pair that cannot fit returns ``None``, or raises ``_StructuredBudgetExceededError``
    with ``raise_on_budget=True`` so the caller can bin instead. A numerical failure while
    bisecting one target refuses that target only, so the list may be shorter than
    ``budgets``; if no rung survives, or the lambda-independent half fails, the result
    is ``None`` and a speculative route hands the pair back to the dense path.
    """
    if p.profiled_trace is None:
        return None

    # One M factorization for the whole ladder: it carries U_eff and both
    # halves of V_eff, none of which depends on lambda.
    try:
        geometry = _profile(p)
    except _UnstableStructuredEDFError:
        return None

    def evaluate(lam: float) -> tuple[float, float] | None:
        try:
            return _evaluate(p, geometry, lam)
        except _UnstableStructuredEDFError:
            return None

    variances: dict[float, float | None] = {}

    def screened(stat: float, edf: float, lam: float) -> ScreenedPair | None:
        if lam not in variances:
            try:
                variances[lam] = _reference_variance(p, geometry, lam)
            except _UnstableStructuredEDFError:
                variances[lam] = None
        reference_variance = variances[lam]
        if reference_variance is None:
            return None
        return ScreenedPair(
            statistic=stat, edf0=edf, lambda0=float(lam), reference_variance=reference_variance
        )

    if not np.any(p.S_a):
        # No penalty to scan, as in the dense ladder: one rung at the block's achieved edf
        # (``tr(V_eff^+ V_eff)``, uncounted, so not an integer on a near-rank pair), with
        # lambda0 = 0; otherwise the bracket would be infinite (inf * 0).
        if max_evaluations is not None and max_evaluations < 2:
            if raise_on_budget:
                raise _StructuredBudgetExceededError
            return None
        evaluated = evaluate(0.0)
        if evaluated is None:
            return None
        stat, edf = evaluated
        result = screened(stat, edf, 0.0)
        return [result for _ in budgets] if result is not None else None

    # The bracket uses the profiled trace, a sum of nonnegative residual energies, so no
    # dense V_eff or ``tr(V) - tr(C' M^+ C)`` difference is formed. The 1e+-10 edges are
    # the dense ladder's, so a pair both paths score gets the same lambda0 (agreement to
    # about two ulp on the thin-level fixture). That fixture has full-rank overlap; with a
    # rank-deficient overlap the dense numerator also counts directions outside
    # ``range(X_o)``, so a structured route there would need re-measuring.
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    scale = max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300)
    lo, hi = _lambda_bracket(scale)

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

    # Distinct targets, not rungs: repeated budgets bisect to the same lambda, so they are
    # charged and cached once. Each distinct emitted lambda also needs one variance pass.
    searchable = {float(b) for b in budgets if edf_hi < float(b) < edf_lo}
    clamped = {lo if float(b) >= edf_lo else hi for b in budgets if float(b) not in searchable}
    # Each distinct emitted lambda also needs one variance factorization.
    required = 2 + (_MAX_STEPS_PER_RUNG + 1) * len(searchable) + len(clamped)
    if max_evaluations is not None and required > max_evaluations:
        if raise_on_budget:
            raise _StructuredBudgetExceededError
        return None

    solved: dict[float, ScreenedPair | None] = {}
    out: list[ScreenedPair] = []
    for budget in budgets:
        edf0 = float(budget)
        if edf0 not in searchable:
            lam = lo if edf0 >= edf_lo else hi
            stat, achieved = (stat_lo, edf_lo) if lam == lo else (stat_hi, edf_hi)
            result = screened(stat, achieved, lam)
            if result is not None:
                out.append(result)
            continue
        if edf0 not in solved:
            a, b = lo, hi
            edf_a, edf_b = edf_lo, edf_hi
            lam, stat, achieved = hi, stat_hi, edf_hi
            refused = False
            for _ in range(_MAX_BISECT):
                if b <= a * (1.0 + 1e-12):
                    break
                lam = float(np.sqrt(a) * np.sqrt(b))
                evaluated = evaluate(lam)
                if evaluated is None:
                    refused = True
                    break
                stat, achieved = evaluated
                # The same monotone certificate as at the endpoints; noise below the target tolerance
                # is not a refusal.
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
                solved[edf0] = screened(stat, achieved, lam)
        result = solved[edf0]
        if result is not None:
            out.append(result)
    # A failure on one target leaves certified clamps and other targets standing; if none
    # survive, ``None`` lets a speculative route hand the pair to the dense path.
    return out or None
