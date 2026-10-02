"""The penalized-mode score the observed-REML certificate and the PIRLS stop rule share.

One-engine design sections 3.8 and 3.9.  At a proposed mode ``(alpha, beta)``
with row score ``s_r = omega_r (y_r - mu_r) mu'(eta_r) / V(mu_r)`` the score of
the penalized likelihood in centred coordinates is ``g_0 = sum_r s_r`` for the
intercept and ``g = X~'s - S beta`` for the slopes, ``X~ = X - 1 mean_x'``.
Its relative form, per coefficient, is ``|g_0| / sum |s|`` and

    |g_j| / (zeta sqrt(max(D_jj, n_w res_j^2) + S_jj) + |S beta|_j),
    zeta = sum |s| / sqrt(n_w),

with ``D_jj = sum_r w_r x~_rj^2`` the centred working-weighted data curvature,
``n_w = sum_r w_r``, ``res_j = n eps |mean_j|`` the resolution of the centring
and ``S_jj`` the penalty's diagonal: the Jacobi-scaled score ``|g_j| /
sqrt(H_jj)`` in units of the row score's own scale ``zeta``, plus the
pre-cancellation magnitude of the penalty term.  For a column the data carry
``zeta sqrt(D_jj) = sum |s| scale_j`` with ``scale_j`` the centred column
scale, a pre-cancellation magnitude with the score's units, invariant to
translating a column and to rescaling the weights.  A coefficient without
data rows (a random-effect level of zero weight) keeps a scale through
``S_jj``: its score is the penalty term alone, and a scale of ``|S beta|_j``
alone would make it identically 1 whatever the rounding left in ``beta_j``
(stage-1 verifier, gram census).

**Centred by type.**  ``X~'s`` is formed on centred rows wherever a column's
type admits an offset: a ``DenseGroupMatrix`` column is centred row by row
before the product, in fixed chunks, so its rounding scales with ``|x -
mean|`` and not with ``|x|``.  Every other column type (one-hot, random
effect, spline bases) has entries bounded by its type and takes the
compact transpose product less ``mean_j sum s``.  No data-dependent switch
chooses between the two.

**The bar** (``bar_eff = max(bar, floor)``, per coefficient).  The inner
PIRLS only has to be accurate enough that the error it induces in the REML
criterion and its gradient is small against the outer stop, ``reml_tol (1 +
|V|)`` (the forcing-term argument of inexact Newton methods, Eisenstat &
Walker 1996, applied to the nested iteration: solving the inner problem
beyond that "oversolves" and buys nothing).  At a returned mode with score
``g~`` the implicit-function derivatives give, to first order, a gradient
error ``e_k = (d theta / d rho_k)' g~ / phi`` and an objective error
``a' H^-1 g~`` (``a = grad_beta log|H| / 2``, nonzero for every family whose
weights move with ``eta`` and largest under observed geometry), so a relative
score ``r`` bounds both by ``r`` times sensitivity sums of order ``p edf``
against ``reml_tol (1 + |V|)``; worst case that asks for ``r`` near
``reml_tol`` itself.  Newton's contraction leaves the returned score well
below the bar that stopped it, so the measured requirement is looser, and the
bar is the loosest decade that meets the REML tolerance on the measured fits
(stage-2 change record, "stop bar"): at 1e-6 a Gamma/log fit (67k rows) moved
its smoothing parameters 27x beyond what ``reml_tol`` resolves and a 67k-row
Gamma/log fit and a binomial/cauchit random-effect fit stopped
``line_search_failed``; at 1e-7 two Tweedie census fits lost convergence and a
binomial/cauchit fit's edf moved by 1.1e-2.  At 1e-8 every real-data speed
case converges with smoothing parameters, edf, deviance and SE within the
REML tolerance of its 1e-10 fit (whose raw-offset Poisson case never
converges: 204 PIRLS iterations against 15), and every auto census fit that
converges at 1e-10 converges, 228 of 230 within that tolerance (the other two,
flat Gamma/log criteria, differ in lambda by 2.6e-5 relative with objectives
equal to 1e-12 relative, as at 1e-9); six gram census fits at lambda 1e-7
beside weights 1e4, where gram's own log|H| is not certified (design §4.5),
converge only at the tighter bar.  The bar is never set
below the problem's noise: the floor is the componentwise backward-error
floor of the score at the iterate in the Oettli-Prager form Arioli, Duff and
Ruiz (1992) use for stopping, ``gamma_n (|X~|'|s|)_j + gamma_p (|S| |beta|)_j
+ u (|H~| |theta|)_j`` over the same scale, ``gamma_k = k u / (1 - k u)``
(Higham 2002, Lemma 3.1): Tisseur's (2001, Theorem 2.4) limiting residual of
Newton's method in floating point, ``psi + u |J| |v|``.  The representation
term takes ``|H~ theta|``'s row-wise form ``sum_r w_r |x~_rj| |x~_r' beta| +
(|S| |beta|)_j``, which is at most ``(|H~||theta|)_j``: the floor is never
overstated, so it never certifies a mode float64 could have resolved.  It is
evaluated only for coefficients that miss the bar.  A score that stops
contracting short of the bar (``stagnation_window``) ends the solve at its
limiting accuracy; the mode is then published as not converged, never
refused.

**Weak identification** (section 3.9).  A coefficient is weakly identified
when its curvature, data and penalty, is within the rounding of an
accumulation of its rows at the largest working weight:
``sum_r w_r x~_rj^2 + S_jj <= gamma_{n+} max_r(w_r) sum_{omega_r > 0}
x~_rj^2`` (``w`` the Fisher working weights, never negative; ``n+`` the rows
with positive prior weight).  Its column carries curvature only through rows
weighted at the rounding of the largest weight -- rows of prior weight 1e-15,
or rows whose fitted binomial or Poisson mean underflows -- and no solver can
place it: a Newton step along it is not finite.  The fit keeps and flags it,
and the certificate and the stop rule run over the identified coefficients.
Coefficients a factor truncated as weakly identified (``excluded``) join the
class.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import sparse

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    SparseSSPGroupMatrix,
)

_UNIT_ROUNDOFF = 2.0**-53
_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)
_CHUNK = 8192
# The bar of the mode certificate and the PIRLS stop rule: one bar, no margin
# between them (the certificate IS the stop rule's residual at the mode PIRLS
# returns).  Set by what the REML criterion needs, not by how far the
# arithmetic can push the score (module docstring, **The bar**): a fixed
# multiple of the REML stopping tolerance, the forcing-term scaling of an
# inexact Newton method, so a caller who asks the outer iteration for more
# asks the inner one for proportionally more.
REML_TOL_BAR_RATIO = 10.0
# The Newton engines' default ``reml_tol`` (``model.reml_execute``) and the bar
# it gives, which every PIRLS outside a REML fit uses.
_DEFAULT_REML_TOL = 1e-9
MODE_CERTIFICATION_BAR = REML_TOL_BAR_RATIO * _DEFAULT_REML_TOL
# No certificate asks for less than the candidate-grade REML tolerance (1e-6)
# gives, nor for more than 100 eps can express; the per-coefficient floor
# (Tisseur's limiting residual) takes over below that.
_LOOSEST_BAR = REML_TOL_BAR_RATIO * 1e-6


def mode_certification_bar(reml_tol: float | None = None) -> float:
    """The certificate's bar for a REML fit stopping at ``reml_tol (1 + |V|)``.

    ``REML_TOL_BAR_RATIO * reml_tol`` (``MODE_CERTIFICATION_BAR`` at the
    Newton engines' default, ``None``), clipped to ``[100 eps,
    _LOOSEST_BAR]``.
    """
    tolerance = _DEFAULT_REML_TOL if reml_tol is None else float(reml_tol)
    return min(max(REML_TOL_BAR_RATIO * tolerance, 100.0 * _EPS), _LOOSEST_BAR)


# The stop rule resolves floors and weak tests only once every relative score
# is within this of zero: a floor or a weakly identified column only decides
# anything once the ordinary coefficients have nearly converged, and resolving
# costs a pass per failing block.  An evaluation choice, never a route: the
# floor (Tisseur's limiting residual, module docstring) is ``~ n u`` times a
# column's absolute-to-centred mass ratio, so it reaches 1e-3 only on a column
# whose absolute mass is ``1e-3 / (n u)`` (``> 1e7`` at a million rows) times
# its centred mass.
MODE_RESOLVE_CAP = 1e-3


def stagnation_window(max_iter: int, bar: float = MODE_CERTIFICATION_BAR) -> int:
    """Iterations over which a resolved score must at least halve before PIRLS stops as stagnated.

    Newton's method in floating point contracts its error until the error
    reaches the limiting accuracy, and no further (Tisseur 2001, SIMAX 22(4),
    Corollary 2.3: with ``u kappa(J*)``, ``u ||J^-1|| phi`` and ``beta
    ||J*^-1|| ||v0 - v*||`` at most 1/8, Theorem 2.2's factor ``G`` evaluates
    to about 0.55, below 1, though not to the 1/2 of the paper's remark); the
    score then only moves within its noise (More & Wild 2011).  The window
    rests on the iteration budget, not on a contraction rate: Fisher scoring
    on a non-canonical link contracts linearly, at a rate ``rho``.  The
    slowest rate that can still take a resolved score (``MODE_RESOLVE_CAP``)
    to the bar within ``max_iter`` iterations is ``rho_max = (bar /
    cap)^(1 / max_iter)``, and at that rate the score halves every ``ln 2 /
    -ln rho_max`` iterations: that is the window.  A resolved score that has not halved within it cannot
    reach the bar in the budget, so the solve stops there (``"score_stagnated"``)
    and the caller publishes the mode as not converged.  At the default 100
    iterations and the default bar the window is 7.
    """
    ratio = math.log(max(MODE_RESOLVE_CAP / bar, 2.0))
    return max(2, math.ceil(math.log(2.0) * max(int(max_iter), 1) / ratio))


def _gamma(count: int) -> float:
    """Higham's ``gamma_k = k u / (1 - k u)`` for ``k`` roundings, ``u = 2^-53``."""
    product = count * _UNIT_ROUNDOFF
    return product / (1.0 - product) if product < 1.0 else float("inf")


def one_hot_columns(dm: DesignMatrix) -> NDArray:
    """``(p,)`` bool: the columns of one-hot blocks (``CategoricalGroupMatrix``, random effects included)."""
    mask = np.zeros(dm.p, dtype=bool)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        mask[offset : offset + width] = isinstance(matrix, CategoricalGroupMatrix)
        offset += width
    return mask


@njit(cache=True)
def _factor_smooth_level_sums(
    data, indices, indptr, basis, bin_idx, codes, natural_map, centre, rows
):
    """``(4, K k)``: per natural fs column, ``sum |x~| a``, ``sum |x~| b``, ``sum w x~^2``, ``sum o x~^2``.

    ``rows = (a, b, w, o)`` per row, all non-negative; ``x~ = x - centre``.  A
    column of level ``l`` is supported on that level's rows: the in-level terms
    are summed row by row, and off the level ``x~ = -centre`` multiplies the
    other levels' totals, taken as prefix plus suffix sums of per-level sums
    (no subtraction).  Every term is a term of the direct sum, so the result
    is within ``gamma_n`` of it.  Exact rows read CSR (``data, indices,
    indptr``), discrete rows a support basis through ``bin_idx``.
    """
    levels, k = centre.shape
    out = np.zeros((4, levels, k))
    level_totals = np.zeros((4, levels))
    z = np.empty(k)
    discrete = bin_idx.size > 0
    for row in range(codes.size):
        level = codes[row]
        for q in range(4):
            level_totals[q, level] += rows[q, row]
        for b in range(k):
            z[b] = 0.0
        if discrete:
            support = bin_idx[row]
            for raw in range(basis.shape[1]):
                value = basis[support, raw]
                if value != 0.0:
                    for b in range(k):
                        z[b] += value * natural_map[raw, b]
        else:
            for pointer in range(indptr[row], indptr[row + 1]):
                raw = indices[pointer]
                value = data[pointer]
                for b in range(k):
                    z[b] += value * natural_map[raw, b]
        for b in range(k):
            deviation = z[b] - centre[level, b]
            size = abs(deviation)
            square = deviation * deviation
            out[0, level, b] += size * rows[0, row]
            out[1, level, b] += size * rows[1, row]
            out[2, level, b] += square * rows[2, row]
            out[3, level, b] += square * rows[3, row]
    prefix = np.zeros((4, levels + 1))
    suffix = np.zeros((4, levels + 1))
    for q in range(4):
        for level in range(levels):
            prefix[q, level + 1] = prefix[q, level] + level_totals[q, level]
        for level in range(levels - 1, -1, -1):
            suffix[q, level] = suffix[q, level + 1] + level_totals[q, level]
    for level in range(levels):
        for b in range(k):
            size = abs(centre[level, b])
            for q in range(4):
                other = prefix[q, level] + suffix[q, level + 1]
                out[q, level, b] += (size if q < 2 else size * size) * other
    return out.reshape(4, levels * k)


def column_sums(
    dm: DesignMatrix,
    columns: NDArray,
    mean_x: NDArray,
    magnitudes: tuple[NDArray, NDArray, NDArray],
    positive: NDArray,
) -> NDArray:
    """``(4, len(columns))``: ``sum |x~| a``, ``sum |x~| b``, ``sum w x~^2``, ``sum_{o} x~^2`` per column.

    ``magnitudes = (a, b, w)``, all non-negative, ``positive`` the rows of
    positive prior weight, ``x~ = x - mean_x``.  A factor-smooth (``fs``) block
    takes every failing column in one pass over rows
    (``_factor_smooth_level_sums``); any other column is formed through its
    own group matrix (one product over that block, not the whole design).
    The sums have non-negative terms, so any order is within ``gamma_n``.
    """
    rows = np.vstack([*magnitudes, np.asarray(positive, dtype=np.float64)])
    columns = np.asarray(columns, dtype=np.intp)
    result = np.empty((4, len(columns)))
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        inside = np.flatnonzero((columns >= offset) & (columns < offset + width))
        if inside.size:
            local = columns[inside] - offset
            if isinstance(matrix, FactorSmoothGroupMatrix) and matrix.factor_basis == "fs":
                discrete = matrix.is_discrete
                empty_int = np.zeros(0, dtype=np.intp)
                empty = np.zeros(0)
                sums = _factor_smooth_level_sums(
                    empty if discrete else matrix._data,
                    empty_int if discrete else matrix._indices,
                    empty_int if discrete else matrix._indptr,
                    matrix.B_unique if discrete else np.zeros((0, 0)),
                    matrix.bin_idx if discrete else empty_int,
                    matrix.codes,
                    np.ascontiguousarray(matrix.natural_map),
                    np.ascontiguousarray(
                        mean_x[offset : offset + width].reshape(matrix.n_levels, matrix.block_size)
                    ),
                    rows,
                )
                result[:, inside] = sums[:, local]
            else:
                for position, column in zip(inside, local, strict=True):
                    unit = np.zeros(width)
                    unit[column] = 1.0
                    centred = (
                        np.asarray(matrix.matvec(unit), dtype=np.float64) - mean_x[offset + column]
                    )
                    size = np.abs(centred)
                    squares = centred**2
                    result[:, position] = (
                        float(size @ rows[0]),
                        float(size @ rows[1]),
                        float(rows[2] @ squares),
                        float(rows[3] @ squares),
                    )
        offset += width
    return result


def centred_data_score(dm: DesignMatrix, row_score: NDArray, mean_x: NDArray) -> NDArray:
    """``(X - 1 mean_x')' s`` by column type (module docstring)."""
    total = float(np.sum(row_score))
    result = dm.rmatvec(row_score) - mean_x * total
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            values = matrix.M
            centre = mean_x[offset : offset + width]
            accumulated = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                accumulated += (values[lo:hi] - centre).T @ row_score[lo:hi]
            result[offset : offset + width] = accumulated
        offset += width
    return result


def centred_matvec(dm: DesignMatrix, beta: NDArray, center: NDArray) -> NDArray:
    """``(X - 1 center') beta`` by column type (module docstring).

    A ``DenseGroupMatrix`` block is centred row by row before its product, so
    the result rounds at ``|x - center| |beta|``; every other block, whose
    entries and centre are bounded by its type, takes its own product less
    ``center' beta``.  With the centred intercept ``alpha`` this evaluates
    ``eta = alpha + X~ beta + offset`` without the cancellation of ``X beta``
    against the raw intercept that a column's offset forces (one-engine
    design §3.8: the PIRLS state is ``(alpha, beta)``).
    """
    beta = np.asarray(beta, dtype=np.float64)
    result = np.zeros(dm.n)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        part = beta[offset : offset + width]
        centre = center[offset : offset + width]
        if type(matrix) is DenseGroupMatrix:
            values = matrix.M
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                result[lo:hi] += (values[lo:hi] - centre) @ part
        else:
            result += matrix.matvec(part) - float(centre @ part)
        offset += width
    return result


def dense_columns(dm: DesignMatrix) -> NDArray:
    """``(p,)`` bool: the ``DenseGroupMatrix`` columns, the one type whose entries its type does not bound."""
    mask = np.zeros(dm.p, dtype=bool)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        mask[offset : offset + width] = type(matrix) is DenseGroupMatrix
        offset += width
    return mask


def _dense_rows(matrix, start, stop, centre, centre_lo):
    """Rows ``start:stop`` of a dense block, ``(x - c) - c_lo`` (``c_lo`` ``None``: ``x - c``)."""
    rows = matrix.M[start:stop] - centre
    return rows if centre_lo is None else rows - centre_lo


def dense_centred_matvec(
    dm: DesignMatrix, values: NDArray, center: NDArray, center_lo: NDArray | None = None
) -> NDArray:
    """``(X_d - 1 c_d') v_d`` over the ``DenseGroupMatrix`` blocks only, centred row by row.

    The dense blocks' share of ``centred_matvec`` in its fixed chunks, for a
    caller that applies every other block through its own (structured)
    product.  ``center_lo`` makes the centre an exact pair, rows ``(x - c) -
    c_lo`` (``centered_system.weighted_mean_pair``).
    """
    result = np.zeros(dm.n)
    values = np.asarray(values, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            part = values[offset : offset + width]
            centre = center[offset : offset + width]
            centre_lo = None if center_lo is None else center_lo[offset : offset + width]
            for start in range(0, dm.n, _CHUNK):
                stop = min(start + _CHUNK, dm.n)
                result[start:stop] += _dense_rows(matrix, start, stop, centre, centre_lo) @ part
        offset += width
    return result


def dense_centred_rmatvec(
    dm: DesignMatrix, rows: NDArray, center: NDArray, center_lo: NDArray | None = None
) -> NDArray:
    """``(X_d - 1 c_d')' r`` on the ``DenseGroupMatrix`` columns (zero elsewhere), centred row by row.

    ``center_lo`` as ``dense_centred_matvec``.
    """
    result = np.zeros(dm.p)
    rows = np.asarray(rows, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            centre = center[offset : offset + width]
            centre_lo = None if center_lo is None else center_lo[offset : offset + width]
            accumulated = np.zeros(width)
            for start in range(0, dm.n, _CHUNK):
                stop = min(start + _CHUNK, dm.n)
                block = _dense_rows(matrix, start, stop, centre, centre_lo)
                accumulated += block.T @ rows[start:stop]
            result[offset : offset + width] = accumulated
        offset += width
    return result


def prior_weighted_centre(dm: DesignMatrix, prior_weights: NDArray) -> NDArray:
    """``c0``: each dense column's shifted prior-weighted mean, 0 on every other column type.

    One-engine design §3.2: ``c0_j = x_ref,j + sum w (x_j - x_ref,j) / sum w``
    over the prior weights ``w``, ``x_ref`` the first row of positive weight.
    Computed once per design; a column constant on its weighted rows centres
    to exactly ``x_ref``.  Other column types have entries bounded by their
    type and need no centre.
    """
    weights = np.asarray(prior_weights, dtype=np.float64)
    centre = np.zeros(dm.p)
    positive = np.flatnonzero(weights > 0.0)
    total = float(np.sum(weights))
    if not positive.size or not total > 0.0:
        return centre
    reference = int(positive[0])
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            values = matrix.M
            anchor = np.asarray(values[reference], dtype=np.float64)
            accumulated = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                accumulated += (values[lo:hi] - anchor).T @ weights[lo:hi]
            centre[offset : offset + width] = anchor + accumulated / total
        offset += width
    return centre


def weighted_column_centring(
    dm: DesignMatrix, weights: NDArray, positive_prior: NDArray
) -> tuple[NDArray, float, NDArray]:
    """``(mean_x, sum_w, D)``: the ``weights``-weighted column means and centred diagonal.

    ``D_jj = sum_r w_r (x_rj - mean_j)^2``, the centring
    ``penalized_mode_residual`` scales a score by, formed from given working
    weights instead of a solve's centred system.  The means take one
    transpose product, a dense column's anchored at its first positive-weight
    row (``prior_weighted_centre``).  The diagonal takes one pass per block:
    a one-hot block in closed form, a factor-smooth ``fs`` block through its
    level sums (``column_sums``, centred row by row), and every other block,
    dense or not, over its rows in fixed chunks by the corrected two-pass
    algorithm (Chan, Golub & LeVeque 1983) about the rounded weighted mean
    ``m``, never a row of the data:

        D_jj = sum w (x - m)^2 - (sum w (x - m))^2 / sum w,

    clamped at 0.  Without the correction the two-pass relative error is
    within ``n u + n^2 kappa^2 u^2``, ``kappa^2 = sum w x^2 / D_jj`` (Chan,
    Golub & LeVeque 1983), and the correction (Bjorck's) reduces the
    second-order term.  Raw moments, ``sum w x^2 - sum_w m^2``, carried
    ``(n + 3) u sum w x^2``, ``kappa^2`` times larger: enough to leave a
    column that is nearly constant over the weighted mass with an inflated
    diagonal and so a deflated relative score.  Zeros when no weight is
    positive.
    """
    w = np.asarray(weights, dtype=np.float64)
    sum_w = float(np.sum(w))
    p = dm.p
    if not sum_w > 0.0:
        return np.zeros(p), 0.0, np.zeros(p)
    on_weight = dm.rmatvec(w)
    mean_x = on_weight / sum_w
    dense = np.zeros(p, dtype=bool)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        dense[offset : offset + width] = type(matrix) is DenseGroupMatrix
        offset += width
    if np.any(dense):
        mean_x[dense] = prior_weighted_centre(dm, w)[dense]
    diagonal = np.empty(p)
    one_hot = one_hot_columns(dm)
    centre = mean_x[one_hot]
    diagonal[one_hot] = (
        on_weight[one_hot] * (1.0 - centre) ** 2 + (sum_w - on_weight[one_hot]) * centre**2
    )
    # every other block in one pass of its own over its rows: an fs block
    # through its level sums, and any other block in fixed chunks of rows by
    # the corrected two-pass algorithm about the rounded mean -- never a
    # design product per column, never raw moments
    factor_smooth: list[NDArray] = []
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        columns = slice(offset, offset + width)
        offset += width
        if isinstance(matrix, CategoricalGroupMatrix):
            continue
        if isinstance(matrix, FactorSmoothGroupMatrix) and matrix.factor_basis == "fs":
            factor_smooth.append(np.arange(columns.start, columns.stop))
            continue
        dense_values = matrix.M if type(matrix) is DenseGroupMatrix else None
        centre = mean_x[columns]
        squares = np.zeros(width)
        firsts = np.zeros(width)
        # a wide block's chunk holds about 2^20 entries (8 MB), not 8192 rows
        chunk = max(256, min(_CHUNK, (1 << 20) // max(width, 1)))
        for lo in range(0, dm.n, chunk):
            hi = min(lo + chunk, dm.n)
            rows = (
                dense_values[lo:hi]
                if dense_values is not None
                else np.asarray(matrix.row_subset(np.arange(lo, hi)).toarray(), dtype=np.float64)
            ) - centre
            squares += w[lo:hi] @ rows**2
            firsts += w[lo:hi] @ rows
        diagonal[columns] = np.maximum(squares - firsts**2 / sum_w, 0.0)
    if factor_smooth:
        chosen = np.concatenate(factor_smooth)
        zeros = np.zeros(dm.n)
        diagonal[chosen] = column_sums(dm, chosen, mean_x, (zeros, zeros, w), positive_prior)[2]
    return mean_x, sum_w, diagonal


def centre_offset_mean(
    dm: DesignMatrix, weights: NDArray, sum_w: float, center: NDArray, mean_x: NDArray
) -> NDArray:
    """``d = sum W (x - c) / sum W``: a weighted column mean read about the state's centre ``c``.

    The centred intercept about ``c`` is ``alpha = mean_z - d' beta`` (one-engine
    design §3.8).  ``mean_x - c`` from the weighted mean ``mean_x`` rounds at
    ``u |c|``, the size of the column's offset, not of its spread: at a 1e6
    offset that is ``1e-10``, times the slope in every row's ``eta``.  Every
    ``DenseGroupMatrix`` column, by type (the only type whose entries are not
    bounded by their type), is differenced row by row before its weighted
    sum, in fixed chunks as ``centred_matvec``, so ``d`` rounds at ``gamma_n
    max|x - c|``.  Every other column keeps ``mean_x - c``.
    """
    offset_mean = np.asarray(mean_x, dtype=np.float64) - center
    w = np.asarray(weights, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            centre = center[offset : offset + width]
            values = matrix.M
            accumulated = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                accumulated += (values[lo:hi] - centre).T @ w[lo:hi]
            offset_mean[offset : offset + width] = accumulated / sum_w
        offset += width
    return offset_mean


def two_sum(a, b):
    """Knuth's TwoSum: ``s = fl(a + b)`` and its rounding error, ``a + b = s + e`` exactly.

    Branch-free and exact for any finite operands, also under underflow, with
    ``|e| <= u |s|`` (Ogita, Rump & Oishi 2005, Algorithm 3.1 and Theorem 3.4).
    """
    s = a + b
    b_virtual = s - a
    return s, (a - (s - b_virtual)) + (b - b_virtual)


def _scaled_ratio(numerator: tuple[float, int], denominator: tuple[float, int]) -> float:
    """``(N 2^K) / (D 2^L)`` without forming either scaled sum, and without raising.

    Every finite non-zero pair divides its significands as ``frexp``
    mantissas, whose quotient lies in ``(1/2, 2)``, so it neither under- nor
    overflows whichever sum sits at the larger scale: a weight total carried
    at ``2^-1126`` beside a numerator at ``2^0`` read ``1e10 * 2^-1126`` and
    rounded to 0, and a numerator at ``2^-1126`` beside a total at ``2^0``
    read ``2^146 / 2^-960 = inf`` before the guard tested the operands.  The
    exponents are added before scaling back: a result past the binary64 range
    is a signed infinity, one below it a signed zero, never the
    ``OverflowError`` ``math.ldexp`` raises.  A zero denominator has no
    quotient and returns ``nan``, which every caller reads as "fall back".
    """
    if denominator[0] == 0.0:
        return math.nan
    if numerator[0] == 0.0 or not (math.isfinite(numerator[0]) and math.isfinite(denominator[0])):
        return numerator[0] / denominator[0]
    numerator_mantissa, numerator_exponent = math.frexp(numerator[0])
    denominator_mantissa, denominator_exponent = math.frexp(denominator[0])
    mantissa, exponent = math.frexp(numerator_mantissa / denominator_mantissa)
    exponent += numerator_exponent - denominator_exponent + numerator[1] - denominator[1]
    if exponent > 1024:
        return math.copysign(math.inf, mantissa)
    if exponent < -1100:
        return math.copysign(0.0, mantissa)
    return math.ldexp(mantissa, exponent)


def _exact_sum(
    weights: NDArray, values: NDArray, shift: float, mode: int
) -> tuple[float, int, bool]:
    """``(S, K, ok)`` with the sum ``S 2^K``: unscaled in one pass, scaled only if that overflows."""
    from superglm.solvers._exact_sums import scaled_exact_sum, unscaled_exact_sum

    total, exponent, ok = unscaled_exact_sum(weights, values, shift, mode)
    if ok:
        return total, exponent, True
    return scaled_exact_sum(weights, values, shift, mode)


def compensated_weighted_mean(values: NDArray, weights: NDArray) -> float:
    """``m* = sum w v / sum w`` from exact products, refined once with an exact residual.

    Each sum runs in a compiled kernel (``_exact_sums``) that streams the rows
    and holds only Shewchuk's partials, so it needs no memory that grows with
    the rows.  Every product ``w_i v_i`` is split exactly into two floats and
    the partials are rounded once, as ``math.fsum`` rounds them.  The products
    are added unscaled, so no contribution underflows before the large terms
    cancel (``[1e150, -1e150, 1e-200]`` keeps its ``1e-200``); a product below
    ``2^-969``, where TwoProduct stops being exact, is formed on the operands'
    mantissas and carried exactly at a shifted scale.  Only if a
    split, product or partial overflows does that sum fall back to the scaled
    kernel: products on the operands' ``frexp`` mantissas, each sum scaled by
    its own largest power of two (so ``1e-300`` beside ``1e300`` survives
    there too).  The first quotient ``m0`` is then refined by the residual
    ``v - m0 = h + e`` (TwoSum, exact), whose products are formed the same
    way.  With ``n`` rows, ``u`` the unit roundoff and ``W = sum w``:

        |m - m*| <= (u + 10 u^2) |m*| + 2^-1074 (2 + 16 (n + 5) L / W),

    ``L = 1`` on the unscaled path (the merge of the scaled-up partials of
    products below ``2^-969``, half a unit of ``2^-1074`` per partial) and
    ``L = max |w v| + |m*| max w`` on the overflow fallback (a piece scaled
    below the normal range, at most ``2^-1075`` of its sum's largest power of
    two), plus the subnormal results (Higham 2002 §2.2).  On
    two levels of adjacent floats, on subnormal weights and on values of
    ``+-1e300`` the mean is correctly rounded.  Three passes over the rows,
    two more for a sum that overflows.  A quotient past the binary64 range,
    a zero weight sum, non-finite input or a non-finite residual falls back
    to ``np.average``.
    """
    from superglm.solvers._exact_sums import native_operand

    v = native_operand(values)
    w = native_operand(weights)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
        return float(np.average(v, weights=w))
    total, total_exponent, total_ok = _exact_sum(w, v, 0.0, 0)
    if not total_ok or total == 0.0:
        return float(np.average(v, weights=w))
    numerator, numerator_exponent, numerator_ok = _exact_sum(w, v, 0.0, 1)
    first = _scaled_ratio((numerator, numerator_exponent), (total, total_exponent))
    if not numerator_ok or not math.isfinite(first):
        return float(np.average(v, weights=w))
    residual, residual_exponent, residual_ok = _exact_sum(w, v, first, 2)
    if not residual_ok:
        return float(np.average(v, weights=w))
    mean = first + _scaled_ratio((residual, residual_exponent), (total, total_exponent))
    return mean if math.isfinite(mean) else first


def _anchored_total(chunks, weights: NDArray, anchor: NDArray) -> NDArray:
    """``sum w (x - anchor)`` over the row chunks, compensated across chunks (Kahan)."""
    total = np.zeros_like(anchor)
    compensation = np.zeros_like(anchor)
    for start, stop, block in chunks():
        contribution = (np.asarray(block, dtype=np.float64) - anchor).T @ weights[start:stop]
        corrected = contribution - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    return total


def corrected_two_pass_pair(chunks, weights: NDArray, sum_w: float, width: int):
    """``(anchor, lo)``: each column's weighted mean as an exact pair, the corrected two-pass way.

    ``chunks`` is a zero-argument callable returning ``(start, stop, rows)``
    blocks.  Pass one forms the shift ``sum w (x - x_ref) / sum w`` about
    ``x_ref``, the first row that carries weight, and rounds the mean,
    ``anchor = fl(x_ref + shift)``; pass two forms the remainder ``lo = sum w
    (x - anchor) / sum w`` (Chan, Golub & LeVeque 1983, the corrected
    two-pass algorithm on shifted data).  The anchor lies within ``u |m| +
    gamma_k sum |w| |x - x_ref| / |sum w|`` of the mean (``k`` the chunk
    length plus two, Higham 2002 §4.3), so ``x - anchor`` is exact where
    ``x`` and the anchor lie within a factor two (Sterbenz) and otherwise
    rounds at ``u |x - anchor|``, the centred row's own scale, and ``lo``
    carries ``gamma_k sum |w| |x - anchor| / |sum w|``.  The seed row enters
    only through pass one's rounding, which pass two absorbs: a row far from
    the mean with zero or negligible weight no longer sets the remainder's
    scale (it did as the anchor: Sol's ``x = [0, 1e16 - 2, 1e16, 1e16 + 2]``
    at ``w = [0, -0.1, 1, 1]`` read a centred Gram of 3.6 for 1.0526).  Zero
    weights enter neither sum, and a column constant on its weighted rows
    gives ``anchor = x_ref`` and ``lo = 0`` exactly.
    """
    weights = np.asarray(weights, dtype=np.float64)
    anchor = rounded_weighted_mean(chunks, weights, sum_w, width)
    return anchor, _anchored_total(chunks, weights, anchor) / sum_w


def rounded_weighted_mean(chunks, weights: NDArray, sum_w: float, width: int) -> NDArray:
    """Pass one of ``corrected_two_pass_pair``: the weighted mean, shifted by the first weighted row.

    ``fl(x_ref + sum w (x - x_ref) / sum w)``: within ``u |m| + gamma_k sum |w| |x
    - x_ref| / |sum w|`` of the mean, whichever row seeds it; zeros without a
    weighted row.
    """
    weights = np.asarray(weights, dtype=np.float64)
    reference = None
    for start, stop, block in chunks():
        carried = np.flatnonzero(weights[start:stop] != 0.0)
        if carried.size:
            reference = np.array(np.asarray(block, dtype=np.float64)[carried[0]], copy=True)
            break
    if reference is None:
        return np.zeros(width, dtype=np.float64)
    return reference + _anchored_total(chunks, weights, reference) / sum_w


def centred_intercept_remainder(
    y: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    alpha: float,
    contribution: NDArray,
) -> float | None:
    """``alpha_lo``: the compensated remainder of a Gaussian identity fit's centred intercept.

    With the slopes' centred contribution ``t = centred_matvec(...)`` as
    evaluated, the intercept that zeros the intercept score is ``alpha* =
    sum w (y - o - t) / sum w`` (the intercept is never penalized).  The
    fit's float ``alpha`` misses it by the forward error of the weighted means
    that formed it (BLAS dots, whose summation order is kernel dependent),
    and by at least the half ulp ``alpha*`` loses when it is not representable
    (the midpoint of two adjacent floats).  One step of iterative refinement
    with the residual formed error-free (Demmel et al. 2009 for least squares)
    recovers it.  Each row's residual ``d = y - o - alpha - t`` is carried as
    four floats by TwoSum and weighted by TwoProduct into one exact sum
    (``_exact_sums.weighted_residual_sum``), never added back into its head
    before the reduction (that absorbed the errors: ``y = [1e12, -1e12, 1]``
    predicted 0.3333062 for 1/3).  The sum and ``sum w`` are each rounded
    once, so

        |alpha + alpha_lo - alpha*| <= 3u |alpha_lo| + 2^-1074 (1 + 5 m / sum w),

    ``m`` the weighted rows, the last term a product's error below the normal
    range.  ``None`` when there is no positive weight or a part overflows,
    and the predictor stays ``alpha + t``.
    """
    from superglm.solvers._exact_sums import native_operand, weighted_residual_sum

    w = native_operand(weights)
    total, total_exponent, total_ok = _exact_sum(w, w, 0.0, 0)
    if not total_ok or not total > 0.0:
        return None
    response = native_operand(y)
    contribution = native_operand(contribution)
    has_offset = offset is not None and bool(np.any(offset))
    shift = native_operand(offset) if has_offset else native_operand(np.zeros(0))
    residual, residual_exponent, ok = weighted_residual_sum(
        w, response, shift, has_offset, float(alpha), contribution
    )
    if not ok:
        return None
    remainder = _scaled_ratio((residual, residual_exponent), (total, total_exponent))
    return remainder if math.isfinite(remainder) else None


def centred_warm_start(result) -> tuple[float, NDArray] | None:
    """``(alpha, c)`` of a PIRLS state that carries its centred predictor, else ``None``.

    A warm start's ``_centred_init`` (``irls_direct``): the next fit continues
    from ``alpha + (X - 1 c') beta`` rather than from the raw intercept,
    which rounds ``alpha - c' beta`` at ``|c' beta|``.  Pass it only with the
    state's own ``beta`` in the same design coordinates.
    """
    alpha = getattr(result, "centred_intercept", None)
    centre = getattr(result, "state_center", None)
    if alpha is None or centre is None:
        return None
    return float(alpha), np.asarray(centre, dtype=np.float64)


def linear_predictor(dm: DesignMatrix, result, offset: NDArray | None) -> NDArray:
    """The unclipped ``eta`` of a PIRLS result, from its centred state when it carries one.

    ``alpha + X~ beta + offset`` (``centred_matvec``) when the result records
    ``centred_intercept`` and ``state_center``; ``X beta + intercept +
    offset`` otherwise.  A published Gaussian identity fit also carries the
    intercept's remainder ``alpha_lo`` (``centred_intercept_remainder``) and
    evaluates the compensated pair as ``alpha + (X~ beta + alpha_lo)``: the
    remainder joins the rows at their own scale before the one rounding at
    ``|eta|``, so ``|eta - eta*| <= u |eta*| + u |X~ beta + alpha_lo|`` plus
    the remainder's bound.  ``offset`` ``None`` adds nothing.
    """
    alpha = getattr(result, "centred_intercept", None)
    center = getattr(result, "state_center", None)
    if alpha is None or center is None:
        eta = dm.matvec(result.beta) + result.intercept
    else:
        eta = centred_matvec(dm, result.beta, center)
        alpha_lo = getattr(result, "centred_intercept_lo", None)
        if alpha_lo is not None:
            eta += alpha_lo
        eta = alpha + eta
    return eta if offset is None else eta + offset


@dataclass(frozen=True)
class ModeResidual:
    """The relative penalized score of one iterate and what the certificate makes of it.

    ``relative`` and ``bar_effective`` are ``(p + 1,)``, intercept first;
    ``bar_effective`` is the fixed bar except where a floor was resolved above
    it.  ``weak`` ``(p,)`` marks the weakly identified coefficients this
    evaluation found and ``excluded`` those plus the ones the caller excluded.
    ``resolved`` says every coefficient that missed the fixed bar had its floor
    and weak test evaluated.  ``scale`` ``(p + 1,)`` holds the denominators of
    ``relative`` (``sum |s|`` for the intercept, then each slope's), each in
    its coordinate's own units.
    """

    intercept_score: float
    slope_score: NDArray
    relative: NDArray
    bar_effective: NDArray
    weak: NDArray
    excluded: NDArray
    resolved: bool
    bar: float
    scale: NDArray | None = None

    def ratio(self) -> float:
        """``max relative / bar_effective`` over the intercept and the identified slopes."""
        identified = np.concatenate(([True], ~self.excluded))
        values = self.relative[identified] / self.bar_effective[identified]
        return float(np.max(values, initial=0.0))

    @property
    def floor_binding(self) -> bool:
        """Whether a derived floor raised the bar for an identified coefficient."""
        identified = np.concatenate(([True], ~self.excluded))
        return bool(np.any(self.bar_effective[identified] > self.bar))


# The most weak slopes one evaluation forms a block for: above it the
# exclusion is refused (the safe direction), so the block, an ``n x k`` array
# and ``k`` design products, stays small.
_WEAK_BLOCK_LIMIT = 32


def _centred_block_gram(
    dm: DesignMatrix, columns: NDArray, mean_x: NDArray, weights: NDArray
) -> tuple[NDArray, NDArray]:
    """``(gram, support)`` for a few columns ``B`` formed directly.

    ``gram`` is ``sum w (x_B - m_B)(x_B - m_B)'`` less its correction, the
    corrected two-pass cross products (Chan, Golub & LeVeque 1983) about the
    rounded means ``m_B``, accumulated over fixed chunks of rows.
    ``support`` marks the rows where any column of ``B`` is nonzero: the rows
    a step in ``B`` changes, with the intercept held.  One design product per
    column, for the small block of weakly identified slopes
    ``penalized_mode_residual`` tests (``_WEAK_BLOCK_LIMIT``).
    """
    columns = np.asarray(columns, dtype=np.intp)
    weights = np.asarray(weights, dtype=np.float64)
    centred = np.empty((dm.n, len(columns)))
    support = np.zeros(dm.n, dtype=bool)
    for position, column in enumerate(columns):
        unit = np.zeros(dm.p)
        unit[column] = 1.0
        raw = np.asarray(dm.matvec(unit), dtype=np.float64)
        support |= raw != 0.0
        centred[:, position] = raw - mean_x[column]
    firsts = weights @ centred
    gram = np.zeros((len(columns), len(columns)))
    for lo in range(0, dm.n, _CHUNK):
        block = centred[lo : lo + _CHUNK]
        gram += (block * weights[lo : lo + _CHUNK, None]).T @ block
    total = float(np.sum(weights))
    if total > 0.0:
        gram = gram - np.outer(firsts, firsts) / total
    return 0.5 * (gram + gram.T), support


def _half_block_decrement(hessian: NDArray, gradient: NDArray) -> float:
    """``g' H^-1 g / 2`` over ``H``'s eigenvectors, its null space decided in ``u``.

    An eigenvalue within ``4 k u lambda_max`` of zero (``k`` the block's
    size) is null to the eigensolver's resolution: a projection of ``g`` on
    it beyond its own rounding, ``4 k u |g|``, cannot be bounded, so the gain
    is ``inf`` and the exclusion it would license is refused; one within that
    rounding adds nothing.  Deciding at exact zero would follow the sign
    LAPACK gives a null eigenvalue.  ``inf`` too when ``H`` or ``g`` is not
    finite.
    """
    if not (np.all(np.isfinite(hessian)) and np.all(np.isfinite(gradient))):
        return math.inf
    try:
        values, vectors = np.linalg.eigh(hessian)
    except np.linalg.LinAlgError:
        return math.inf
    k = len(values)
    resolution = 4.0 * k * _UNIT_ROUNDOFF
    largest = float(np.max(np.abs(values), initial=0.0))
    projection = vectors.T @ gradient
    null = values <= resolution * largest
    if np.any(null & (np.abs(projection) > resolution * float(np.linalg.norm(gradient)))):
        return math.inf
    kept = ~null
    return 0.5 * float(np.sum(projection[kept] ** 2 / values[kept]))


# Above this many joint cells of three or more one-hot blocks, the cells of
# all of them together are not decided; every level, reference and two-block
# bridge set still is.
_ROW_SET_CELL_LIMIT = 4096
# The cells elimination leaves (the incidence's core) are decided by one SVD,
# formed only while its cost ``m k min(m, k)`` stays within this many flops.
_ROW_SET_CORE_FLOPS = 2**31
# The most nonzeros the two-block bridge sets' directions hold in one design
# (about 50 MB retained as CSR; formation first holds them in Python
# dictionaries, several times that): past it a bridge is held without its direction
# (``RowSets.bounded``).  Only a long chain of single-cell links reaches it.
_ROW_SET_BRIDGE_NONZEROS = 2**22
# The design's row sets, in its ``_structured_layout_cache`` (``row_sets``).
_ROW_SETS_KEY = "row_sets"


@dataclass(frozen=True)
class RowSets:
    """The sets of rows a design's one-hot blocks move on their own (``row_sets``).

    ``blocks`` holds ``(first column, matrix)`` for each
    ``CategoricalGroupMatrix``.  The kept joint sets are unions of the
    design's distinct joint cells: ``row_cell`` gives each row's joint cell,
    or is ``None`` when no joint set is kept, and ``cell_sets`` is ``(cells,
    sets)``, sparse, 1 where a cell belongs to a set.  ``cell_directions`` is
    ``(sets, p)``, sparse (CSR): the slope part of a coefficient direction
    that moves the set's rows and no other row (its intercept part carries
    no penalty).  Held sparse, it costs its nonzeros, so no width of the
    design drops a set.  ``cell_codes`` is ``(sets, blocks)``: each set's
    code in each block it names (a block's ``n_levels`` for its reference),
    ``-1`` in a block it does not name.  ``bounded`` marks a set held
    without its direction, past ``_ROW_SET_BRIDGE_NONZEROS``.
    """

    blocks: tuple[tuple[int, CategoricalGroupMatrix], ...]
    row_cell: NDArray | None
    cell_sets: Any
    cell_directions: Any
    cell_codes: NDArray
    bounded: NDArray

    def directions(self, p: int) -> Iterator[NDArray | None]:
        """The slope parts ``(p,)`` of each block's reference direction, then of each joint set's.

        ``None`` for a ``bounded`` set.  One at a time: a single dense
        ``(p,)`` vector is alive, whatever the number of sets.
        """
        for start, matrix in self.blocks:
            reference = np.zeros(p)
            reference[start : start + matrix.n_levels] = -1.0
            yield reference
        held = self.cell_directions
        for row in range(held.shape[0]):
            yield None if self.bounded[row] else held[[row]].toarray().ravel()

    def named_columns(self, index: int, p: int) -> NDArray:
        """``(p,)`` bool: the columns of the blocks joint set ``index`` names."""
        columns = np.zeros(p, dtype=bool)
        for (start, matrix), code in zip(self.blocks, self.cell_codes[index], strict=True):
            if code >= 0:
                columns[start : start + matrix.n_levels] = True
        return columns


def row_sets(dm: DesignMatrix) -> RowSets:
    """The design's ``RowSets``, formed once per design and held in its ``_structured_layout_cache``.

    Owner: the design's one-hot blocks.  Lifetime: the design (the layout
    cache is not pickled).  Invalidation: none, since the cells, their
    elimination and their directions read the blocks' codes alone, never a
    weight, response, coefficient or penalty, and no fit changes a code.  A
    held entry is reused only for the same block objects at the same columns.
    """
    found = []
    offset = 0
    for matrix in dm.group_matrices:
        if isinstance(matrix, CategoricalGroupMatrix):
            found.append((offset, matrix))
        offset += matrix.shape[1]
    blocks = tuple(found)
    cache = getattr(dm, "_structured_layout_cache", None)
    held = cache.get(_ROW_SETS_KEY) if isinstance(cache, dict) else None
    if (
        isinstance(held, RowSets)
        and len(held.blocks) == len(blocks)
        and all(a == c and b is d for (a, b), (c, d) in zip(held.blocks, blocks, strict=True))
    ):
        return held
    sets = _form_row_sets(blocks, dm.p)
    if isinstance(cache, dict):
        cache[_ROW_SETS_KEY] = sets
    return sets


def _form_row_sets(blocks: tuple[tuple[int, CategoricalGroupMatrix], ...], p: int) -> RowSets:
    """The joint sets of two or more one-hot blocks: rows the model moves on their own (``row_sets``).

    A union ``R`` of joint cells is a set when its indicator lies in the
    range of the cells' incidence (columns: the intercept and every one-hot
    column; a block's reference has no column), so that some ``d`` moves
    ``R``'s rows by one and no other row.  A set that is the rows of one
    level, or of a block's reference, is judged there already: it is not
    kept.

    **Two blocks.**  The intercept and a pair of blocks' columns span every
    level indicator of both, the column space of the unsigned incidence of
    the levels' bipartite graph, whose edges are the joint cells present.
    Negating one block's columns makes it the signed incidence, with the same
    column space, where a row's leverage is its edge's effective resistance
    (Spielman & Srivastava 2011, Lemma 3; Kline, Saggio & Solvsten 2020,
    Example 4: below one exactly when a path avoids the edge).  So a joint
    cell of the pair is a set exactly when it is a bridge of that graph,
    found from depth-first lowpoints in linear time (Tarjan 1974): integer
    arithmetic, no rank decision and no size limit.  A bridge with an
    endpoint of degree one is that level's rows.  Its direction is a
    potential on one side ``X`` of the bridge, ``s`` on the first block's
    levels in ``X`` and ``-s`` on the second's: every cell inside or outside
    ``X`` keeps its predictor and the bridge moves by one.  A level's slope is
    its potential less its block reference's.  The side whose slopes have
    fewer nonzeros is held, while the design's total stays within
    ``_ROW_SET_BRIDGE_NONZEROS``; past it a bridge is held without its
    direction (``bounded``).  With three or more blocks every pair's bridges
    are still sets, since more columns only enlarge the range.

    **Three or more blocks** also have cells of all of them together, which
    no graph decides: their connectedness is a rank condition (Srivastava &
    Anderson 1970; Godolphin 2013), decided here by ``_joint_cell_sets``
    within ``_ROW_SET_CELL_LIMIT`` cells.
    """
    count_blocks = len(blocks)
    none = RowSets(
        blocks,
        None,
        sparse.csr_matrix((0, 0)),
        sparse.csr_matrix((0, p)),
        np.zeros((0, count_blocks), dtype=np.intp),
        np.zeros(0, dtype=bool),
    )
    if count_blocks < 2:
        return none
    stacked = np.column_stack([matrix.codes for _, matrix in blocks])
    cells, inverse = np.unique(stacked, axis=0, return_inverse=True)
    member_cells: list[NDArray] = []
    member_sets: list[NDArray] = []
    codes: list[NDArray] = []
    directions: list[dict[int, float] | None] = []
    budget = _ROW_SET_BRIDGE_NONZEROS
    for first in range(count_blocks):
        for second in range(first + 1, count_blocks):
            budget = _bridge_sets(
                cells, blocks, first, second, budget, member_cells, member_sets, codes, directions
            )
    if count_blocks > 2:
        for cell, direction in _joint_cell_sets(cells, blocks):
            member_cells.append(np.array([cell], dtype=np.intp))
            member_sets.append(np.array([len(codes)], dtype=np.intp))
            codes.append(np.asarray(cells[cell], dtype=np.intp))
            directions.append(direction)
    if not codes:
        return none
    kept = len(codes)
    rows_of = np.concatenate(member_cells)
    cell_sets = sparse.csr_matrix(
        (np.ones(len(rows_of)), (rows_of, np.concatenate(member_sets))), shape=(len(cells), kept)
    )
    rows, columns, values = [], [], []
    for index, direction in enumerate(directions):
        for column, value in (direction or {}).items():
            rows.append(index)
            columns.append(column)
            values.append(value)
    held = sparse.csr_matrix((values, (rows, columns)), shape=(kept, p))
    bounded = np.array([direction is None for direction in directions], dtype=bool)
    return RowSets(
        blocks, np.asarray(inverse).reshape(-1), cell_sets, held, np.stack(codes), bounded
    )


def _level_graph_bridges(
    heads: NDArray, tails: NDArray, nodes: int
) -> tuple[NDArray, NDArray, list[tuple[int, int, int, int]]]:
    """``(order, end, found)``: the bridges of a simple graph (Tarjan 1974).

    Edge ``e`` joins ``heads[e]`` and ``tails[e]``.  An iterative depth-first
    search numbers the nodes in preorder (``order``; a node's subtree is
    ``order[position : end]``) and keeps each node's lowpoint, the least
    preorder number its subtree reaches by one non-tree edge.  A tree edge
    into ``child`` is a bridge exactly when ``child``'s lowpoint exceeds its
    parent's number.  ``found`` holds ``(edge, child, component start,
    component end)`` per bridge, in preorder positions.
    """
    edges = len(heads)
    ends = np.concatenate([heads, tails])
    others = np.concatenate([tails, heads])
    ids = np.concatenate([np.arange(edges), np.arange(edges)])
    sort = np.argsort(ends, kind="stable")
    first = np.searchsorted(ends[sort], np.arange(nodes + 1)).tolist()
    neighbour = others[sort].tolist()
    edge_of = ids[sort].tolist()
    position = [-1] * nodes
    low = [0] * nodes
    via = [-1] * nodes
    end = [0] * nodes
    cursor = first[:-1]
    order: list[int] = []
    found: list[tuple[int, int, int, int]] = []
    for root in range(nodes):
        if position[root] >= 0 or first[root] == first[root + 1]:
            continue
        start = len(order)
        position[root] = low[root] = start
        order.append(root)
        stack = [root]
        bridges: list[tuple[int, int]] = []
        while stack:
            node = stack[-1]
            at = cursor[node]
            if at < first[node + 1]:
                cursor[node] = at + 1
                if edge_of[at] == via[node]:
                    continue
                reached = neighbour[at]
                if position[reached] < 0:
                    via[reached] = edge_of[at]
                    position[reached] = low[reached] = len(order)
                    order.append(reached)
                    stack.append(reached)
                elif position[reached] < low[node]:
                    low[node] = position[reached]
                continue
            stack.pop()
            end[node] = len(order)
            if stack:
                parent = stack[-1]
                low[parent] = min(low[parent], low[node])
                if low[node] > position[parent]:
                    bridges.append((via[node], node))
        found.extend((edge, child, start, len(order)) for edge, child in bridges)
    return np.asarray(order, dtype=np.intp), np.asarray(end, dtype=np.intp), found


def _bridge_sets(
    cells: NDArray,
    blocks: tuple[tuple[int, CategoricalGroupMatrix], ...],
    first: int,
    second: int,
    budget: int,
    member_cells: list[NDArray],
    member_sets: list[NDArray],
    codes: list[NDArray],
    directions: list[dict[int, float] | None],
) -> int:
    """Append the bridge sets of blocks ``first`` and ``second`` (``_form_row_sets``); the budget left."""
    (start_a, block_a), (start_b, block_b) = blocks[first], blocks[second]
    levels_a, levels_b = block_a.n_levels, block_b.n_levels
    key = cells[:, first] * (levels_b + 1) + cells[:, second]
    pairs, pair_of_cell = np.unique(key, return_inverse=True)
    pair_of_cell = np.asarray(pair_of_cell).reshape(-1)
    heads = pairs // (levels_b + 1)
    tails = levels_a + 1 + pairs % (levels_b + 1)
    nodes = levels_a + levels_b + 2
    order, end, found = _level_graph_bridges(heads, tails, nodes)
    if not found:
        return budget
    degree = np.bincount(heads, minlength=nodes) + np.bincount(tails, minlength=nodes)
    position = np.full(nodes, -1, dtype=np.intp)
    position[order] = np.arange(len(order))
    in_a = np.concatenate(([0], np.cumsum(order < levels_a)))
    in_b = np.concatenate(([0], np.cumsum((order > levels_a) & (order < nodes - 1))))
    reference_a, reference_b = int(position[levels_a]), int(position[nodes - 1])
    set_of_pair = np.full(len(pairs), -1, dtype=np.intp)
    for edge, child, low, high in found:
        if degree[heads[edge]] == 1 or degree[tails[edge]] == 1:
            continue  # the only cell of one of its levels: that level's set
        lo, hi = int(position[child]), int(end[child])
        sign = 1.0 if child <= levels_a else -1.0  # the first block's endpoint on side T
        sides = []
        for flip in (False, True):

            def inside(at: int, flip=flip, lo=lo, hi=hi, low=low, high=high) -> bool:
                return (low <= at < high and not lo <= at < hi) if flip else lo <= at < hi

            count_a = (
                (in_a[high] - in_a[low] - in_a[hi] + in_a[lo]) if flip else in_a[hi] - in_a[lo]
            )
            count_b = (
                (in_b[high] - in_b[low] - in_b[hi] + in_b[lo]) if flip else in_b[hi] - in_b[lo]
            )
            holds_a, holds_b = inside(reference_a), inside(reference_b)
            nonzeros = int(
                (levels_a - count_a if holds_a else count_a)
                + (levels_b - count_b if holds_b else count_b)
            )
            sides.append((nonzeros, flip, holds_a, holds_b))
        nonzeros, flip, holds_a, holds_b = min(sides, key=lambda side: side[0])
        named = np.full(len(blocks), -1, dtype=np.intp)
        named[first], named[second] = heads[edge], tails[edge] - levels_a - 1
        set_of_pair[edge] = len(codes)
        codes.append(named)
        if nonzeros > budget:
            directions.append(None)
            continue
        budget -= nonzeros
        side = np.concatenate((order[low:lo], order[hi:high])) if flip else order[lo:hi]
        potential = -sign if flip else sign
        side_a = side[side < levels_a]
        side_b = side[(side > levels_a) & (side < nodes - 1)] - levels_a - 1
        if holds_a:
            side_a, value_a = np.setdiff1d(np.arange(levels_a), side_a), -potential
        else:
            value_a = potential
        if holds_b:
            side_b, value_b = np.setdiff1d(np.arange(levels_b), side_b), potential
        else:
            value_b = -potential
        direction = {int(start_a + level): value_a for level in side_a.tolist()}
        direction.update({int(start_b + level): value_b for level in side_b.tolist()})
        directions.append(direction)
    set_of_cell = set_of_pair[pair_of_cell]
    held = np.flatnonzero(set_of_cell >= 0)
    member_cells.append(held)
    member_sets.append(set_of_cell[held])
    return budget


def _joint_cell_sets(
    cells: NDArray, blocks: tuple[tuple[int, CategoricalGroupMatrix], ...]
) -> list[tuple[int, dict[int, float]]]:
    """The cells of all the blocks together that are sets, with their slope directions (``_form_row_sets``).

    A cell is a set when its indicator ``e_c`` lies in the range of the
    cells' incidence ``M``.  A cell that is the only cell of a level, or of a
    block's reference rows, is that set's rows: it is not kept.

    **Elimination.**  A column that holds one remaining cell is that cell's
    indicator less the indicators of cells already eliminated, so ``d_c =
    e_column - sum d_eliminated`` and the cell is a set, exactly (integer
    arithmetic).  Eliminating it may leave another column with one cell.
    Elimination never decides a remaining cell's rank, because only the
    eliminated cell has a nonzero in its pivot column.  It decides every
    cell of nested blocks (each a duplicate) and of a saturated interaction.

    **The core** the elimination leaves is decided by one SVD.  The computed
    SVD is the exact SVD of ``M + E`` with ``||E||_2 <= p(m, k) eps ||M||_2``,
    and its leading ``r`` left singular vectors lie within an angle ``p(m,
    k) eps ||M||_2 / gap`` of the true subspace (LAPACK Users' Guide, 3rd
    ed., section 4.9.1), the gap being ``sigma_r`` for an incidence whose
    other singular values are zero.  With ``p(m, k) = max(m, k)``, the rank
    tolerance's own constant, a cell's leverage ``||U_c||^2`` is within
    ``tol / (sigma_r - tol) + gamma_{r+2} + r eps`` of its value (``tol =
    max(m, k) eps sigma_1``; the last two terms its sum and ``U``'s
    orthogonality).  A cell is a set exactly when its leverage is 1, so it is
    kept when ``1 - ||U_c||^2`` lies within that resolution.  A kept cell
    that is not a set can only be refused: its direction does not move its
    rows alone, so at the exact mode its test reads a nonzero residual.  A
    core cell's direction is the SVD's minimum-norm ``d``, less each
    eliminated cell's direction times the amount ``d`` moves it.  Formed
    within ``_ROW_SET_CELL_LIMIT`` cells, the core only while ``m k min(m,
    k) <= _ROW_SET_CORE_FLOPS``.
    """
    count = len(cells)
    if count > _ROW_SET_CELL_LIMIT:
        return []
    duplicate = np.zeros(count, dtype=bool)
    for position, (_, matrix) in enumerate(blocks):
        per_code = np.bincount(cells[:, position], minlength=matrix.n_levels + 1)
        duplicate |= per_code[cells[:, position]] == 1
    if np.all(duplicate):
        return []
    # incidence column 0 is the intercept, column 1 + j slope j
    columns_of: list[list[int]] = [[0] for _ in range(count)]
    cells_of: dict[int, list[int]] = {0: list(range(count))}
    for position, (start, matrix) in enumerate(blocks):
        for cell, code in enumerate(cells[:, position].tolist()):
            if code < matrix.n_levels:
                columns_of[cell].append(1 + start + code)
                cells_of.setdefault(1 + start + code, []).append(cell)
    live = np.ones(count, dtype=bool)
    remaining = {column: len(members) for column, members in cells_of.items()}
    queue = deque(column for column, size in remaining.items() if size == 1)
    direction: dict[int, dict[int, float]] = {}
    while queue:
        pivot = queue.popleft()
        if remaining[pivot] != 1:
            continue
        cell = next(member for member in cells_of[pivot] if live[member])
        moved = {pivot: 1.0}
        for member in cells_of[pivot]:
            if member != cell:
                for column, value in direction[member].items():
                    moved[column] = moved.get(column, 0.0) - value
        direction[cell] = moved
        live[cell] = False
        for column in columns_of[cell]:
            remaining[column] -= 1
            if remaining[column] == 1:
                queue.append(column)
    core = np.flatnonzero(live)
    if core.size and np.any(~duplicate[core]):
        _decide_core(core, columns_of, duplicate, live, direction)
    return [
        (cell, {column - 1: value for column, value in direction[cell].items() if column and value})
        for cell in sorted(cell for cell in direction if not duplicate[cell])
    ]


def _decide_core(
    core: NDArray,
    columns_of: list[list[int]],
    duplicate: NDArray,
    live: NDArray,
    direction: dict[int, dict[int, float]],
) -> None:
    """Add the directions of the core's sets (``_form_row_sets``) to ``direction``."""
    core_columns = sorted({column for cell in core.tolist() for column in columns_of[cell]})
    m, k = len(core), len(core_columns)
    if m * k * min(m, k) > _ROW_SET_CORE_FLOPS:
        return
    where = {column: index for index, column in enumerate(core_columns)}
    incidence = np.zeros((m, k))
    for row, cell in enumerate(core.tolist()):
        incidence[row, [where[column] for column in columns_of[cell]]] = 1.0
    left, singular, right = np.linalg.svd(incidence, full_matrices=False)
    tolerance = max(m, k) * _EPS * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    if rank == 0 or float(singular[rank - 1]) <= 2.0 * tolerance:
        return
    resolution = (
        tolerance / (float(singular[rank - 1]) - tolerance) + _gamma(rank + 2) + rank * _EPS
    )
    if resolution >= 0.5:
        return
    leverage = np.sum(left[:, :rank] ** 2, axis=1)
    eliminated = np.flatnonzero(~live).tolist()
    for row in np.flatnonzero((1.0 - leverage <= resolution) & ~duplicate[core]).tolist():
        solution = right[:rank].T @ (left[row, :rank] / singular[:rank])
        moved = {core_columns[i]: float(value) for i, value in enumerate(solution) if value}
        base = dict(moved)
        for cell in eliminated:
            carried = sum(base.get(column, 0.0) for column in columns_of[cell])
            if carried:
                for column, value in direction[cell].items():
                    moved[column] = moved.get(column, 0.0) - carried * value
        direction[int(core[row])] = moved


def _set_totals(sums: list, index: int) -> tuple[float, float, float, float, float, float]:
    """One set's (score, |score|, represented, count, rising, falling) totals."""
    own, absolute, represented, count, up, down = (float(total[index]) for total in sums)
    return own, absolute, represented, count, up, down


def row_set_quadratics(sets: RowSets, p: int, apply: Callable[..., NDArray]) -> NDArray:
    """A lower bound on ``d' S d`` along each of ``sets.directions(p)``, in ``apply``'s units.

    ``apply(v)`` is ``S v`` and ``apply(v, magnitude=True)`` is ``|S| |v|``,
    which bounds the product's rounding, so ``d' S d - gamma_{2p+4} |d|' |S|
    |d|`` is a lower bound.  Where the penalty is positive along ``d`` but
    that bound is not, the entry is ``nan``: penalized, with no curvature
    bound.  One penalty product per reference and per joint set.  A
    ``bounded`` set's entry is 0 where the penalty is zero on the columns of
    the blocks it names, so along any direction in them, and ``nan``
    otherwise.
    """
    out = []
    for index, direction in enumerate(sets.directions(p)):
        if direction is None:
            named = sets.named_columns(index - len(sets.blocks), p)
            with np.errstate(over="ignore", invalid="ignore"):
                reach = apply(named.astype(np.float64), magnitude=True)
            out.append(0.0 if np.all(reach[named] == 0.0) else math.nan)
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            product = float(direction @ apply(direction))
            size = float(np.abs(direction) @ apply(direction, magnitude=True))
            lower = product - _gamma(2 * p + 4) * size
        if not product > 0.0:
            out.append(0.0 if product == 0.0 else math.nan)
        else:
            out.append(lower if lower > 0.0 else math.nan)
    return np.asarray(out, dtype=np.float64)


def row_set_residual(
    *,
    sets: RowSets,
    row_score: NDArray,
    response: NDArray,
    fisher_weights: NDArray,
    positive_prior: NDArray,
    eta: NDArray,
    column_penalty: NDArray,
    column_penalty_size: NDArray,
    column_curvature: NDArray,
    set_curvature: NDArray,
    bar: float,
    underflow: float,
) -> float:
    """The largest relative score of a set of rows the one-hot blocks move on their own.

    The relative penalized score scales every coordinate by one global
    ``zeta``, which the heaviest rows set: a level carrying ``w`` of the
    weight ``W`` then passes about ``sqrt(W / w)`` bars from its own maximum.
    And a reference level has no column of its own, so a light reference
    level is read only through the intercept, whose scale the heavy levels
    set.  Every set ``R`` of rows the model can move independently is
    therefore certified on its own rows: the score along its indicator, less
    the penalty's gradient along the coefficient direction ``d_R`` that moves
    it,

        g_R = sum_{i in R} s_i - d_R' (S beta),

    within the bar of its own terms' size, ``sum_{i in R} |s_i| + |d_R|'
    (|S| |beta|)``, or of their rounding, ``gamma_{|R| + 2}`` of the rows'
    sum, ``u`` of their predictor's representation ``sum f_i |eta_i|`` and
    ``gamma_{p + 2}`` of the penalty's size.  The sets (``row_sets``, formed
    once per design), by one rule (an indicator in the span of the intercept
    and the one-hot columns):
    - each level of each one-hot block (``CategoricalGroupMatrix``, random
      effects included), ``d_R`` its column;
    - each block's reference rows, the rows no column of it holds, summed
      directly over those rows, ``d_R`` the intercept less the block's
      columns;
    - with two or more blocks, each union of joint cells whose indicator
      lies in that span and that is not already one of the sets above
      (``_form_row_sets``): every bridge of each pair of blocks' level
      graph, whatever the design's size, and with three or more blocks the
      cells of all of them together within ``_ROW_SET_CELL_LIMIT`` cells,
      such as a saturated interaction's, base cells included.
    A set held without its direction (``bounded``) is judged with ``|d_R'
    (S beta)|`` bounded by the sum of ``|S beta|`` over the columns of the
    blocks it names, which holds its direction, and with no penalty size in
    its scale: the refusing side, exact where the penalty's gradient is zero
    there.

    **A penalized set** is also certified by its distance to its own maximum.
    Precondition: the log-likelihood is concave in ``eta``, as binomial/log's
    is (the only caller; Gaussian/log's is not).  So the penalized objective
    along ``d_R`` is at least ``d_R' S d_R``-strongly concave, and its
    maximum along ``d_R`` lies within ``|g_R| / d_R' S d_R`` of the iterate,
    in the units of ``eta`` on ``R``'s rows.  That distance, with ``g_R``'s
    floor and ``underflow`` added, within ``bar`` passes the set
    (``set_curvature`` and ``diag S`` for a level: lower bounds on ``d_R' S
    d_R``).  It is what certifies a penalized set whose rows' scores vanish
    together with its penalty's gradient: a random-effect level inside a
    level without events, whose free column carries the separation while
    each step returns the random effect to zero.

    A set whose positive-weight rows' responses are all zero, or all one, and
    whose direction carries no penalty has no interior maximum (its supremum
    is at infinity, or at the mean space's boundary): it is the separation
    the weak test discloses, and it is not tested here.  That is read off
    the responses, not off the scores' signs: a non-event's score ``-w
    odds`` underflows to zero at small enough weights and means, and would
    hide a level's event rows behind it.  A separated set along a penalized
    direction has a finite penalized maximum and is tested.  A set whose bar
    falls within ``underflow`` and that the distance does not pass cannot be
    resolved and is refused (``inf``).  ``column_*`` are per column in the
    weights' units: the penalty gradient (with any active constraint's
    multipliers), its size ``|S| |beta|``, and ``diag S``; ``set_curvature``
    is ``row_set_quadratics`` in the same units.
    """
    groups = sets.blocks
    if not groups:
        return 0.0
    score = np.asarray(row_score, dtype=np.float64)
    positive = np.asarray(positive_prior, dtype=bool)
    absolute = np.abs(score)
    represented = np.asarray(fisher_weights, dtype=np.float64) * np.abs(eta)
    observed = np.asarray(response, dtype=np.float64)
    rising = (positive & (observed > 0.0)).astype(np.float64)
    falling = (positive & (observed < 1.0)).astype(np.float64)
    carried = positive.astype(np.float64)
    penalty = np.asarray(column_penalty, dtype=np.float64)
    size = np.asarray(column_penalty_size, dtype=np.float64)
    curvature = np.asarray(column_curvature, dtype=np.float64)
    quadratics = np.asarray(set_curvature, dtype=np.float64)
    p = len(penalty)
    worst = 0.0

    def judge(
        own,
        absolute_sum,
        represented_sum,
        count,
        up,
        down,
        direction_penalty,
        direction_size,
        quadratic,
        slack=0.0,
    ):
        nonlocal worst
        if count <= 0.0:
            return
        penalized = not quadratic <= 0.0  # nan: penalized, with no curvature bound
        if not penalized and (up == 0.0 or down == 0.0):
            return  # separated along an unpenalized direction: no interior maximum
        scale = absolute_sum + direction_size
        if not (math.isfinite(scale) and math.isfinite(direction_penalty) and math.isfinite(slack)):
            worst = math.inf
            return
        residual = abs(own - direction_penalty) + slack
        floor = (
            _gamma(int(count) + 2) * absolute_sum
            + _UNIT_ROUNDOFF * represented_sum
            + _gamma(p + 2) * direction_size
        )
        ratio = math.inf if bar * scale <= underflow else residual / max(bar * scale, floor)
        # only a normal ``bar d'Sd`` bounds the distance: below 2^-1022 the
        # product carries up to 2^-1075 of absolute error and the curvature's
        # own scaling may have rounded up, and at 0 it bounds nothing.  There
        # the set is judged by its relative score alone, the refusing side.
        if penalized and math.isfinite(quadratic) and bar * quadratic >= _TINY:
            ratio = min(ratio, (residual + floor + underflow) / (bar * quadratic))
        worst = max(worst, ratio)

    for position, (start, matrix) in enumerate(groups):
        levels = matrix.n_levels
        codes = matrix.codes
        columns = slice(start, start + levels)
        sums = [
            np.bincount(codes, weights=values, minlength=levels + 1)
            for values in (score, absolute, represented, carried, rising, falling)
        ]
        for level in range(levels + 1):
            if level < levels:
                column = start + level
                direction = (penalty[column], size[column], curvature[column])
            else:  # the reference rows: intercept less every column of the block
                direction = (
                    -float(np.sum(penalty[columns])),
                    float(np.sum(size[columns])),
                    float(quadratics[position]),
                )
            judge(*_set_totals(sums, level), *direction)
    if sets.row_cell is None:
        return worst
    held = sets.cell_directions
    membership = sets.cell_sets.T.tocsr()
    cells = sets.cell_sets.shape[0]
    # each cell's totals over its own rows, then each set's over its cells:
    # every term still passes through fewer additions than the set has rows
    sums = [
        np.asarray(membership @ np.bincount(sets.row_cell, weights=values, minlength=cells)).ravel()
        for values in (score, absolute, represented, carried, rising, falling)
    ]
    with np.errstate(over="ignore", invalid="ignore"):
        cell_penalty = np.asarray(held @ penalty).ravel()
        cell_size = np.asarray(abs(held) @ size).ravel()
    for index in range(held.shape[0]):
        slack = 0.0
        if sets.bounded[index]:
            with np.errstate(over="ignore", invalid="ignore"):
                slack = float(np.sum(np.abs(penalty[sets.named_columns(index, p)])))
        judge(
            *_set_totals(sums, index),
            float(cell_penalty[index]),
            float(cell_size[index]),
            float(quadratics[len(groups) + index]),
            slack,
        )
    return worst


@dataclass(frozen=True)
class TruncatedDirection:
    """Rows a direction the factorization truncates moves, judged on those rows (``truncated_direction_ratio``).

    The positive-weight rows the direction moves are held as half-open runs
    ``row_ranges`` of consecutive rows, ``row_count`` of them in all (``rows``
    expands them), so a light region of many rows costs a few runs, not an
    index each.  ``columns`` are the coefficients it leans on most (at most
    eight, by size), and ``information_ratio`` the Fisher information of the
    other rows that share those coefficients over that of the moved rows.
    ``at_maximum`` says the rows sit at their own maximum along the direction
    (weakly identified).  Otherwise the fit cannot be certified in float64;
    ``boundary`` then says every row it moves improves along it and some are
    events, which under the log link rise until one reaches ``eta = 0``: the
    rows' supremum is on the boundary of the parameter space, a probability of
    one, not an interior maximum.  ``unresolved_basis`` says the
    factorization's basis is too inaccurate to show whether the direction moves
    rows at all: ``rows`` are those it visibly moves, possibly none, and the
    claim is refused.  ``earlier`` marks a record judged at an iterate before
    the one the fit returned: history, which never decides the fit's verdict
    (``irls_direct``).
    """

    row_ranges: tuple[tuple[int, int], ...]
    row_count: int
    columns: tuple[int, ...]
    information_ratio: float
    at_maximum: bool
    boundary: bool = False
    earlier: bool = False
    unresolved_basis: bool = False

    @property
    def rows(self) -> tuple[int, ...]:
        """Every row the direction moves, in order."""
        return tuple(row for start, stop in self.row_ranges for row in range(start, stop))

    def __setstate__(self, state: dict) -> None:
        # v0.36.0 pickled every moved row as ``rows``; a saved model still loads
        state = dict(state)
        if "row_ranges" not in state:
            rows = np.asarray(state.pop("rows", ()), dtype=np.int64)
            state["row_ranges"] = row_ranges(rows)
            state["row_count"] = int(rows.size)
        state.setdefault("boundary", False)
        state.setdefault("earlier", False)
        state.setdefault("unresolved_basis", False)
        for name, value in state.items():
            object.__setattr__(self, name, value)


def _unresolved(rows: NDArray | None = None, columns: tuple[int, ...] = ()) -> TruncatedDirection:
    """A refusal's record when the judgement cannot be formed: the rows it can name, or none."""
    rows = np.zeros(0, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    return TruncatedDirection(
        row_ranges(rows), int(rows.size), columns, 0.0, False, unresolved_basis=True
    )


def row_ranges(rows: NDArray) -> tuple[tuple[int, int], ...]:
    """Increasing row indices as half-open runs ``(start, stop)`` of consecutive rows."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = rows[np.concatenate(([0], breaks))]
    stops = rows[np.concatenate((breaks - 1, [rows.size - 1]))] + 1
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))


# Each row's l1 sum on the design, in its ``_structured_layout_cache``.
_ROW_ABS_KEY = "row_abs_sums"


def row_abs_sums(dm: DesignMatrix) -> NDArray:
    """Each row's ``sum_j |x_ij|``, or a bound on it, formed once per design.

    A one-hot block contributes its row's own entry (1, or 0 on its base
    level) and a dense block ``|M| 1``, both exact.  A spline block stored as
    a basis ``B`` and a reparameterisation ``R`` (``X = B R``, sparse or
    binned) contributes ``|B| (|R| 1)``, which bounds its row's sum.  Any
    other block contributes the sum of its columns' largest entries, one
    product per column, which bounds every row.  Owner, lifetime and
    invalidation as ``row_sets``: it reads the design's entries alone.
    """
    cache = getattr(dm, "_structured_layout_cache", None)
    held = cache.get(_ROW_ABS_KEY) if isinstance(cache, dict) else None
    if isinstance(held, np.ndarray):
        return held
    sums = np.zeros(dm.n)
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if isinstance(matrix, CategoricalGroupMatrix):
            sums += np.asarray(matrix.matvec(np.ones(width)), dtype=np.float64)
        elif isinstance(matrix, DenseGroupMatrix):
            sums += np.abs(np.asarray(matrix.M, dtype=np.float64)) @ np.ones(width)
        elif type(matrix) is SparseSSPGroupMatrix:
            spread = np.abs(np.asarray(matrix.R_inv, dtype=np.float64)) @ np.ones(width)
            sums += abs(matrix.B) @ spread
        elif type(matrix) is DiscretizedSSPGroupMatrix:
            spread = np.abs(np.asarray(matrix.R_inv, dtype=np.float64)) @ np.ones(width)
            sums += (np.abs(np.asarray(matrix.B_unique, dtype=np.float64)) @ spread)[matrix.bin_idx]
        else:
            for column in range(width):
                unit = np.zeros(width)
                unit[column] = 1.0
                sums += float(np.max(np.abs(np.asarray(matrix.matvec(unit))), initial=0.0))
    sums.setflags(write=False)
    if isinstance(cache, dict):
        cache[_ROW_ABS_KEY] = sums
    return sums


def null_basis_angle(decomposition: Any, rows: int) -> float:
    """The angle within which a rank decision's discarded subspace is computed, or 1 without a gap.

    The computed invariant subspace of an eigenvalue cluster lies within
    ``p(n) eps ||A||_2 / gap`` of the true one, ``gap`` the distance from the
    cluster to the nearest other eigenvalue (LAPACK Users' Guide, 3rd ed.,
    section 4.7, after Parlett), and the singular subspace of a factor
    likewise with its singular values (section 4.9).  The discarded cluster
    holds every eigenvalue at or below the cutoff, so its gap is the smallest
    retained one less the cutoff: a light set's vanishing curvature lies
    inside the cluster and does not shrink it.  ``p(n)`` is the matrix's
    larger dimension.  Without a gap the subspace is not resolved: 1.
    """
    retained = np.asarray(
        decomposition.retained_values if decomposition.retained_values is not None else (),
        dtype=np.float64,
    )
    width = int(decomposition.width)
    if retained.size == 0:
        return 1.0
    if decomposition.method == "qr_svd":
        sigma = np.sqrt(retained)
        gap = float(np.min(sigma)) - float(decomposition.cutoff)
        size = max(rows, width)
        largest = float(np.max(sigma))
    else:
        gap = float(np.min(retained)) - float(decomposition.cutoff)
        size = width
        largest = float(np.max(retained))
    if not gap > 0.0:
        return 1.0
    return min(1.0, size * _EPS * largest / gap)


def _structural_columns(basis: NDArray, column_scale: NDArray | None) -> NDArray:
    """Which null-basis columns are structural: supported only on columns without centred data.

    ``rank._null_basis`` stacks the discarded spectral directions, supported
    on the active columns, then an exact unit vector per inactive column
    (``column_scale == 0``).  Those move no row by construction.
    """
    if column_scale is None:
        return np.zeros(basis.shape[1], dtype=bool)
    active = np.asarray(column_scale, dtype=np.float64) > 0.0
    return ~np.any(basis[active] != 0.0, axis=0)


def truncated_direction_ratio(
    *,
    dm: DesignMatrix,
    null_basis: NDArray,
    angle: float,
    mean_x: NDArray,
    row_score: NDArray,
    fisher_weights: NDArray,
    response: NDArray,
    positive_prior: NDArray,
    penalty_gradient: NDArray,
    penalty_size: NDArray,
    penalty_apply: Callable[[NDArray], NDArray],
    penalty_size_apply: Callable[[NDArray], NDArray],
    bar: float,
    underflow: float,
    column_scale: NDArray | None = None,
    eta: NDArray | None = None,
) -> tuple[float, tuple[TruncatedDirection, ...]]:
    """The directions the factorization truncates, each judged on the rows it moves.

    A light row set, weighing ~1e16 below the rows it shares coefficients
    with, adds curvature below the rounding of ``X'WX``: the factorization
    truncates its direction, and the relative score sums it beside the heavy
    rows' rounding.  ``null_basis`` ``V`` spans the truncated subspace, to
    within ``angle`` (``null_basis_angle``).  Its structural columns, unit
    vectors on columns without centred data (``column_scale`` 0), move no row
    and are dropped.  Each row's movement ``m = (X - 1 mean_x') V`` is read on
    the design, and the subspace is turned to the right singular vectors of
    that movement over the positive-weight rows, so that directions moving
    different rows by different amounts are judged apart: a direction two
    nearly collinear columns leave moves every row by ~1e-9, a light cut its
    own rows by ~1.  A turned direction's support is the rows it moves beyond
    the basis's error.  ``angle`` bounds ``sin theta``, so a computed ``d``
    lies within ``angle ||d||_2`` of the true subspace (read in the basis's
    own coordinates; the equilibrated ones the angle holds in are #431's),
    and row ``i``'s movement within ``l1_i angle ||d||_2`` of the true one,
    ``l1_i`` the row's own ``sum_j |x_ij - mean_j|`` (``row_abs_sums``).  The
    movement is formed as ``(X V) t``, so its rounding is ``gamma_{p+2} l1_i
    (max |V|) |t|`` through the turn ``t``.  The error is ``4 l1_i (angle
    ||d||_2 + gamma_{p+2} (max |V|) |t|)``.

    - **Structural.**  A direction that moves no row beyond its error is
      aliasing: none of its rows moves resolvably.  Unless that test is
      vacuous: a row moves at most ``l1_i ||d||_inf``, so once the error per
      unit of ``l1_i`` reaches ``||d||_inf`` no movement can be told from the
      error, and the claim is refused with a record (``unresolved_basis``)
      naming the rows the direction moves beyond their rounding, if any.
      Below that, an empty support can still hide a real movement within the
      basis's error (a light set beside a wide column at a moderate angle);
      telling it from an alias is left to #431.
    - **Not hidden.**  A direction whose rows' scores, each weighted by how far
      it moves the row, sum above ``bar`` times every row's score at its
      largest movement, ``sum |m_i s_i| > bar max |m_i| sum |s|``, is seen by
      the relative score, which governs as before.
    - **Separated.**  A hidden direction whose moved rows all respond 0 is a
      separated set, read off the responses (as ``row_set_residual`` reads
      them): not judged here.
    - **Hidden.**  The Newton step on the hidden directions' rows alone,
      ``delta = C^+ G`` with ``C = M' F M + D' S D`` and ``G = M' s - D' S beta``
      (score and Fisher weights of those rows only, brought to unit scale by
      a power of two), moves each row by ``M delta`` in ``eta``.  Within
      ``bar`` or its rounding (``gamma`` of the sums, the basis's error in the
      rows' movement and in the pull ``D' S beta``, the penalty's size, through
      ``|C^+|``) on every row, the rows sit at their own maximum: weakly
      identified.  Otherwise, if every row the step moves beyond its rounding
      improves along it (a response of 0 moving down, of 1 up):
      - along directions the penalty does not bend (``D' S D`` within its
        error), the step is a recession direction of the rows' likelihood:
        with responses of 0 only, a separation (left to the separated sets);
        with an event among them, under the log link, a supremum on the
        boundary ``eta = 0``, refused and disclosed as such;
      - along a penalized direction the penalized objective has a finite
        maximum along the step, which the Newton step estimates: a supremum
        on the boundary only when that step carries a moved row to ``eta >=
        0`` (``eta``, the rows' current predictor; without it, never).
      Else the fit cannot be certified in float64.

    Arithmetic that cannot form the judgement (a non-finite movement or
    score, scores and weights that vanish, or a scaled system that overflows)
    refuses (``inf``) with a record that names no rows (``unresolved_basis``).

    The penalty's bend ``d' S d`` is judged against its error: the basis's,
    ``2 r ||d||_2 ||S d||_2 + ||S||_inf (r ||d||_2)^2`` with ``r = 4 (angle +
    gamma_{p+2})`` (``||S||_inf`` from ``penalty_size_apply``, ``|S| |v|``,
    bounding ``||S||_2``), and its rounding.  Forming ``S d`` sums at most
    ``p`` terms per penalty component, scales each by its ``lambda`` and adds
    the components on a coordinate, at most ``p + 2`` of them; the product
    ``d' fl(S d)`` adds ``p`` more: ``gamma_{3p + 5} |d|' |S| |d|`` in all.

    Returns the largest ``|M delta| / max(bar, floor)`` over uncertifiable
    directions (0 where none) and the hidden directions found.  Nothing in
    the literature certifies a maximum beyond float64's resolution
    (Schwendinger, Grun & Hornik 2021 compare log-binomial solvers by the best
    log-likelihood any of them reaches): this detects it and refuses.
    """
    basis = np.asarray(null_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] == 0:
        return 0.0, ()
    basis = basis[:, ~_structural_columns(basis, column_scale)]
    if basis.shape[1] == 0:
        return 0.0, ()
    positive = np.asarray(positive_prior, dtype=bool)
    score = np.asarray(row_score, dtype=np.float64)
    fisher = np.asarray(fisher_weights, dtype=np.float64)
    observed = np.asarray(response, dtype=np.float64)
    mean = np.asarray(mean_x, dtype=np.float64)
    p = dm.p
    moved = np.column_stack(
        [
            np.asarray(dm.matvec(basis[:, k]), dtype=np.float64) - float(mean @ basis[:, k])
            for k in range(basis.shape[1])
        ]
    )
    # each row's own l1 sum of its centred entries
    reach = row_abs_sums(dm) + float(np.sum(np.abs(mean)))
    resolution = 4.0 * (angle + _gamma(p + 2))
    if not np.any(positive) or not np.all(np.isfinite(moved[positive])):
        return math.inf, (_unresolved(),)
    # the subspace turned to the right singular vectors of its movement
    _, _, right = np.linalg.svd(moved[positive], full_matrices=False)
    turn = right.T
    directions = basis @ turn
    movement = moved @ turn
    lengths = np.linalg.norm(directions, axis=0)
    peaks = np.max(np.abs(directions), axis=0)
    # row i's movement along direction k is within this, per unit of l1_i:
    # the basis's error, and the rounding of ``(X V) t`` through ``|t|``
    reach_error = 4.0 * (
        angle * lengths + _gamma(p + 2) * (np.max(np.abs(basis), axis=0) @ np.abs(turn))
    )
    error = reach[:, None] * reach_error[None, :]
    total = float(np.sum(np.abs(score[positive])))
    if not math.isfinite(total):
        return math.inf, (_unresolved(),)
    hidden: list[int] = []
    for k in range(directions.shape[1]):
        support = positive & (np.abs(movement[:, k]) > error[:, k])
        if not support.any():
            if reach_error[k] < peaks[k]:
                continue  # aliasing
            # every row's error reaches the most it can move: unresolved
            rounding = reach * (_gamma(p + 2) * float(peaks[k]))
            visible = np.flatnonzero(positive & (np.abs(movement[:, k]) > rounding))
            leaning = np.abs(directions[:, k])
            named = tuple(
                int(j)
                for j in np.argsort(-leaning)[:8]
                if leaning[j] > resolution * float(peaks[k])
            )
            return math.inf, (_unresolved(visible, named),)
        weighted = float(np.sum(np.abs(movement[support, k]) * np.abs(score[support])))
        largest_move = float(np.max(np.abs(movement[support, k])))
        if not math.isfinite(weighted) or weighted > bar * largest_move * total:
            continue  # seen by the relative score
        if np.all(observed[support] == 0.0):
            # a separated set, read off its responses as ``row_set_residual``
            # does: no interior maximum.  Not off its scores' or its step's
            # signs: at means near 0 the rows' scores and curvature vanish,
            # and the penalty's gradient or rounding sets the step's sign.
            continue
        hidden.append(k)
    if not hidden:
        return 0.0, ()
    support = positive & np.any(np.abs(movement[:, hidden]) > error[:, hidden], axis=1)
    local = movement[support][:, hidden]
    direction = directions[:, hidden]
    lengths = lengths[hidden]
    s_local, f_local = score[support], fisher[support]
    # the gradient's error from the rows' movement error, per direction
    movement_error = reach_error[hidden] * float(reach[support] @ np.abs(s_local))
    largest = float(np.max(np.concatenate([np.abs(s_local), f_local]), initial=0.0))
    if not (math.isfinite(largest) and largest > 0.0):
        return math.inf, (_unresolved(),)
    exponent = -int(np.frexp(largest)[1])
    gradient_at = np.asarray(penalty_gradient, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        s_local = np.ldexp(s_local, exponent)
        f_local = np.ldexp(f_local, exponent)
        pull = np.ldexp(direction.T @ gradient_at, exponent)
        pull_size = np.ldexp(
            np.abs(direction).T @ np.asarray(penalty_size, dtype=np.float64), exponent
        )
        # the basis's error reaches the pull too: ``|e' S beta| <= ||e||_2
        # ||S beta||_2`` with ``||e||_2 <= angle ||d||_2``
        pull_error = np.ldexp(resolution * lengths * float(np.linalg.norm(gradient_at)), exponent)
        movement_error = np.ldexp(movement_error, exponent)
        stiffness = np.column_stack(
            [
                np.ldexp(np.asarray(penalty_apply(direction[:, j]), dtype=np.float64), exponent)
                for j in range(direction.shape[1])
            ]
        )
        bending = direction.T @ stiffness
        curvature = local.T @ (f_local[:, None] * local) + bending
        gradient = local.T @ s_local - pull
        # the penalty's curvature along each direction, and its error:
        # the basis's, ``|d' S d - d*' S d*| = |e' S (2 d - e)|``, and the
        # rounding of forming ``S d`` and ``d' S d``
        bent_size = np.column_stack(
            [
                np.ldexp(
                    np.asarray(penalty_size_apply(np.abs(direction[:, j])), dtype=np.float64),
                    exponent,
                )
                for j in range(direction.shape[1])
            ]
        )
        penalty_norm = float(
            np.max(np.ldexp(np.asarray(penalty_size_apply(np.ones(p)), dtype=np.float64), exponent))
        )
        bending_error = (
            2.0 * resolution * lengths * np.linalg.norm(stiffness, axis=0)
            + penalty_norm * (resolution * lengths) ** 2
            + _gamma(3 * p + 5) * np.sum(np.abs(direction) * bent_size, axis=0)
        )
    if not (
        np.all(np.isfinite(curvature))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(pull_error))
        and np.all(np.isfinite(movement_error))
        and np.all(np.isfinite(bending_error))
    ):
        return math.inf, (_unresolved(),)
    curvature = 0.5 * (curvature + curvature.T)
    values, vectors = np.linalg.eigh(curvature)
    kept = values > float(np.max(np.abs(values), initial=0.0)) * 4.0 * len(values) * _EPS
    inverse = (vectors[:, kept] / values[kept]) @ vectors[:, kept].T
    steps = local @ (inverse @ gradient)
    rounding = (
        _gamma(len(s_local) + 2) * (np.abs(local).T @ np.abs(s_local))
        + movement_error
        + _gamma(p + 2) * pull_size
        + pull_error
        + np.ldexp(underflow, exponent)
    ).ravel()
    floor = np.abs(local) @ (np.abs(inverse) @ rounding) + _gamma(4 * len(s_local)) * np.abs(steps)
    limit = np.maximum(bar, floor)
    over = np.abs(steps) > limit
    rows = np.flatnonzero(support)
    weights = np.abs(direction).max(axis=1)
    # a coefficient the direction leans on beyond its basis's error
    significant = weights > resolution * float(np.max(weights))
    columns = tuple(int(j) for j in np.argsort(-weights)[:8] if significant[j])
    shared = np.zeros(dm.n, dtype=bool)
    for column in np.flatnonzero(significant):
        unit = np.zeros(p)
        unit[column] = 1.0
        shared |= np.asarray(dm.matvec(unit)) != 0.0
    shared &= positive & ~support
    information = float(np.sum(fisher[shared])) / max(float(np.sum(fisher[support])), _TINY)
    runs = row_ranges(rows)
    if not over.any():
        return 0.0, (TruncatedDirection(runs, int(rows.size), columns, information, True),)
    y_moved = observed[rows]
    # a recession direction of the rows' likelihood: every row the step moves
    # beyond its rounding improves (a row moving the wrong way, however
    # little, bounds the likelihood along the ray)
    improving = ((y_moved == 0.0) & (steps < 0.0)) | ((y_moved == 1.0) & (steps > 0.0))
    determined = np.abs(steps) > floor
    ratio = float(np.max(np.abs(steps) / limit))
    rising = improving & determined & (y_moved == 1.0)
    if np.all(improving | ~determined):
        if not np.any(np.diag(bending) > bending_error):
            if not np.any(rising):
                return 0.0, ()  # a separation: no interior maximum
            # events rising: under the log link their supremum is the boundary
            # eta = 0 at finite coefficients, not a separation
            record = TruncatedDirection(runs, int(rows.size), columns, information, False, True)
            return ratio, (record,)
        # the penalty bends a direction: the penalized maximum along the step
        # is finite, on the boundary only if the step carries a row past it
        if eta is not None and np.any(rising):
            reached = np.asarray(eta, dtype=np.float64)[rows] + steps
            if np.any(reached[rising] >= 0.0):
                record = TruncatedDirection(runs, int(rows.size), columns, information, False, True)
                return ratio, (record,)
    record = TruncatedDirection(runs, int(rows.size), columns, information, False)
    return ratio, (record,)


def penalized_mode_residual(
    *,
    dm: DesignMatrix,
    row_score: NDArray,
    fisher_weights: NDArray,
    positive_prior: NDArray,
    mean_x: NDArray,
    centered_scale: NDArray,
    alpha: float,
    eta_tilde: NDArray,
    penalty_score: NDArray,
    penalty_magnitude: NDArray,
    penalty_curvature: NDArray,
    sum_w: float,
    bar: float,
    excluded: NDArray | None = None,
    resolve_cap: float = float("inf"),
    column_shift: NDArray | None = None,
    decrement_noise: Callable[[NDArray], float] | None = None,
    penalty_block: Callable[[NDArray], NDArray] | None = None,
) -> ModeResidual:
    """Evaluate the shared relative score (module docstring) at one iterate.

    ``eta_tilde`` is ``X~ beta`` per row, ``alpha`` the centred intercept
    ``intercept + mean_x' beta``; ``penalty_score = S beta``,
    ``penalty_magnitude = |S| |beta|``, ``penalty_curvature = diag(S)`` and
    ``sum_w`` the working weights' sum ``centered_scale`` is relative to
    (``centered_scale_j^2 sum_w = D_jj``).  When every relative score is at most ``resolve_cap``
    (always, by default) the floors and weak tests are evaluated for every
    coefficient that misses the fixed bar: one-hot columns in closed form from
    transpose products, every other column from its own entries (one design
    product each).

    ``column_shift`` (``(p,)`` even non-negative integers, ``None`` for none)
    puts slope ``j`` in its own units: every data-side quantity of the column
    (its score, curvature, floors' sums and the weak test's weight) is
    multiplied by ``2^-shift_j``, and ``zeta`` by ``2^(-shift_j / 2)``, while
    ``penalty_*`` arrive already in those units.  Each relative score is a
    ratio of quantities of one degree in the weights and the penalty taken
    together, so the shift leaves it unchanged; it only keeps a penalty far
    larger than the data's scale from overflowing beside it.  The powers of
    two are exact, and the even shift keeps ``zeta``'s square root exact.

    ``decrement_noise`` (``None`` for none) maps a boolean row mask to the
    rounding noise of the objective over those rows, in the weights' units.
    When given, the weak test may exclude a slope only where the iterate is
    already at the mode along it, to that noise.  The weak test reads curvature alone, and at an unfinished
    iterate a slope's curvature can be tiny only because its rows' means are
    still far off.  Half the Newton decrement, ``lambda^2 / 2`` with
    ``lambda^2 = g' H^-1 g``, is the predicted gain of a Newton step (Boyd &
    Vandenberghe 2004, section 9.5.1).  It is formed over the block ``B`` of
    newly weak slopes at once, ``g_B' H_BB^-1 g_B / 2``: no sum of the
    coordinates' own decrements bounds it, since each ``g_j^2 / H_jj`` is a
    lower bound on the block's and two coupled slopes can hide a direction
    along their difference whose gain is far larger.
    - ``g_B`` is each slope's score less its rounding floor, ``sign(g_j)
      (|g_j| - floor_j)_+``, in the weights' units.
    - ``H_BB`` is the penalised curvature block of those slopes: their
      weighted, centred data Gram, formed directly from their columns by the
      corrected two-pass algorithm (the block is small), plus
      ``penalty_block(B)``, the penalty's block in the weights' units
      (``None``: its diagonal, ``penalty_curvature``).
    - The gain is ``inf`` when ``H_BB`` has an eigenvalue null to the
      eigensolver's resolution that ``g_B`` meets beyond its own rounding
      (``_half_block_decrement``), and when more than ``_WEAK_BLOCK_LIMIT``
      slopes are newly weak.
    - The noise is the objective's own over the rows the block touches,
      ``decrement_noise(S)`` for ``S`` the support of ``X_B``, the rows where
      any of its columns is nonzero.  A step ``d`` in ``B`` with the raw
      intercept held changes ``eta`` on ``S`` alone, so the objective's
      change is ``sum_{i in S} [l_i(eta_i + x_iB' d) - l_i(eta_i)]``, and
      evaluating it in float64 carries about ``gamma_{|S|}`` times ``sum_{i
      in S} |l_i|`` (each row's own few roundings add a few ``u |l_i|``).  The
      decrement itself is formed in centred coordinates, whose step also
      moves every row by ``-m_B' d``: the profiled gain, never below the
      held-intercept one, so measuring it against ``S``'s noise can only
      refuse more.  A gain within that noise cannot be told from none; one
      beyond it is a step the fit has not taken.  Rows outside ``S`` carry
      noise that a step on ``S`` alone never meets: a heavy level's rounding
      must not hide a light level's gain.  For one-hot levels ``S`` is the
      levels' own rows; for a dense or spline column it is its nonzero
      rows, possibly all of them.
    If the gain exceeds the noise, no slope of this evaluation is excluded as
    weak.
    """
    n, p = dm.n, dm.p
    total = float(np.sum(row_score))
    absolute = np.abs(row_score)
    intercept_scale = max(_TINY, float(np.sum(absolute)))

    def shifted(values, columns=slice(None)):
        if column_shift is None:
            return values
        return np.ldexp(np.asarray(values, dtype=np.float64), -np.asarray(column_shift)[columns])

    slope_score = shifted(centred_data_score(dm, row_score, mean_x)) - penalty_score
    resolution = n * _EPS * np.abs(mean_x)
    root_weight = math.sqrt(max(float(sum_w), _TINY))
    zeta = intercept_scale / root_weight
    column_zeta = (
        zeta
        if column_shift is None
        else np.ldexp(np.full(p, zeta), -(np.asarray(column_shift) // 2))
    )
    curvature_root = np.sqrt(
        shifted((np.maximum(centered_scale, resolution) * root_weight) ** 2)
        + np.maximum(np.asarray(penalty_curvature, dtype=np.float64), 0.0)
    )
    slope_scale = np.maximum(_TINY, column_zeta * curvature_root + np.abs(penalty_score))
    relative = np.concatenate(([abs(total) / intercept_scale], np.abs(slope_score) / slope_scale))
    bar_effective = np.full(p + 1, float(bar))
    weak = np.zeros(p, dtype=bool)
    excluded = np.zeros(p, dtype=bool) if excluded is None else np.asarray(excluded, dtype=bool)
    failing = np.flatnonzero(relative > bar)
    resolve = bool(np.max(relative, initial=0.0) <= resolve_cap)
    resolved = not failing.size or resolve
    if failing.size and resolve:
        weights = np.asarray(fisher_weights, dtype=np.float64)
        positive = np.asarray(positive_prior, dtype=bool)
        row_count = int(np.count_nonzero(positive))
        gamma_rows, gamma_penalty = _gamma(n), _gamma(p + 2)
        largest = float(np.max(weights, initial=0.0))
        predictor = weights * np.abs(eta_tilde)
        if failing[0] == 0:
            representation = float(np.sum(weights)) * abs(alpha)
            floor = (
                gamma_rows * intercept_scale + _UNIT_ROUNDOFF * representation
            ) / intercept_scale
            bar_effective[0] = max(bar, floor)
        slopes = failing[failing > 0] - 1
        slopes = slopes[~excluded[slopes]]
        if slopes.size:
            one_hot = one_hot_columns(dm)[slopes]
            evaluation = np.empty(len(slopes))
            represented = np.empty(len(slopes))
            curvature = np.empty(len(slopes))
            mass = np.empty(len(slopes))
            if np.any(one_hot):
                columns = slopes[one_hot]
                centre = mean_x[columns]
                inside = dm.rmatvec(absolute)[columns]
                on_predictor = dm.rmatvec(predictor)[columns]
                on_weight = dm.rmatvec(weights)[columns]
                on_rows = dm.rmatvec(positive.astype(np.float64))[columns]
                off = np.abs(centre)
                on = np.abs(1.0 - centre)
                evaluation[one_hot] = on * inside + off * (intercept_scale - inside)
                represented[one_hot] = on * on_predictor + off * (
                    float(np.sum(predictor)) - on_predictor
                )
                curvature[one_hot] = (
                    on_weight * on**2 + (float(np.sum(weights)) - on_weight) * off**2
                )
                mass[one_hot] = on_rows * on**2 + (row_count - on_rows) * off**2
            if not np.all(one_hot):
                sums = column_sums(
                    dm, slopes[~one_hot], mean_x, (absolute, predictor, weights), positive
                )
                evaluation[~one_hot], represented[~one_hot] = sums[0], sums[1]
                curvature[~one_hot], mass[~one_hot] = sums[2], sums[3]
            evaluation = shifted(evaluation, slopes)
            represented = shifted(represented, slopes)
            curvature = shifted(curvature, slopes)
            largest_column = shifted(np.full(len(slopes), largest), slopes)
            magnitude = penalty_magnitude[slopes]
            score_floor = (
                gamma_rows * evaluation
                + gamma_penalty * magnitude
                + _UNIT_ROUNDOFF * (represented + magnitude)
            )
            floor = score_floor / slope_scale[slopes]
            bar_effective[slopes + 1] = np.maximum(bar, floor)
            diagonal = np.asarray(penalty_curvature, dtype=np.float64)[slopes]
            newly_weak = curvature + diagonal <= _gamma(row_count) * largest_column * mass
            if decrement_noise is not None and np.any(newly_weak):
                chosen = slopes[newly_weak]
                resolvable = np.maximum(np.abs(slope_score[chosen]) - score_floor[newly_weak], 0.0)
                gain = 0.0
                support = None
                if len(chosen) > _WEAK_BLOCK_LIMIT:
                    gain = math.inf
                elif np.any(resolvable > 0.0):
                    gradient = np.copysign(resolvable, slope_score[chosen])
                    with np.errstate(over="ignore", invalid="ignore"):
                        # each coordinate's own units back to the weights'
                        if column_shift is not None:
                            gradient = np.ldexp(gradient, np.asarray(column_shift)[chosen])
                        if penalty_block is None:
                            penalty = np.diag(
                                np.asarray(penalty_curvature, dtype=np.float64)[chosen]
                                if column_shift is None
                                else np.ldexp(
                                    np.asarray(penalty_curvature, dtype=np.float64)[chosen],
                                    np.asarray(column_shift)[chosen],
                                )
                            )
                        else:
                            penalty = np.asarray(penalty_block(chosen), dtype=np.float64)
                        gram, support = _centred_block_gram(dm, chosen, mean_x, weights)
                        hessian = gram + penalty
                    gain = _half_block_decrement(hessian, gradient)
                if gain > 0.0 and not (support is not None and gain <= decrement_noise(support)):
                    newly_weak[:] = False
            weak[slopes] = newly_weak
    return ModeResidual(
        intercept_score=total,
        slope_score=slope_score,
        relative=relative,
        bar_effective=bar_effective,
        weak=weak,
        excluded=excluded | weak,
        resolved=resolved,
        bar=float(bar),
        scale=np.concatenate(([intercept_scale], slope_scale)),
    )


def weakly_identified_mask(
    *,
    dm: DesignMatrix,
    fisher_weights: NDArray,
    positive_prior: NDArray,
    mean_x: NDArray,
    penalty_diagonal: NDArray,
    unpenalized: NDArray,
) -> NDArray:
    """``(p,)`` bool: the §3.9 weak test (module docstring) at one mode.

    The disclosure a fit publishes once, from its final Fisher rows, for every
    family and backend.  One-hot columns take the closed forms
    ``penalized_mode_residual`` uses and a ``DenseGroupMatrix`` column is
    centred row by row in fixed chunks, every column of both.  A column of
    any other block (spline, tensor and factor-smooth bases) is formed by one
    design product only when ``unpenalized`` marks it: a penalized one is
    identified by its penalty, and a penalty at the rounding of the data is
    the border's step-5 truncation, disclosed there (design §3.6); no block's
    Gram is formed.  ``penalty_diagonal`` is ``diag(S)``.
    """
    weights = np.asarray(fisher_weights, dtype=np.float64)
    positive = np.asarray(positive_prior, dtype=bool)
    row_count = int(np.count_nonzero(positive))
    p = dm.p
    if row_count == 0 or p == 0:
        return np.zeros(p, dtype=bool)
    largest = float(np.max(weights, initial=0.0))
    total = float(np.sum(weights))
    rows = positive.astype(np.float64)
    curvature = np.zeros(p)
    mass = np.zeros(p)
    one_hot = one_hot_columns(dm)
    if np.any(one_hot):
        centre = mean_x[one_hot]
        on_weight = dm.rmatvec(weights)[one_hot]
        on_rows = dm.rmatvec(rows)[one_hot]
        off, on = np.abs(centre), np.abs(1.0 - centre)
        curvature[one_hot] = on_weight * on**2 + (total - on_weight) * off**2
        mass[one_hot] = on_rows * on**2 + (row_count - on_rows) * off**2
    offset = 0
    formed: list[NDArray] = []
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        columns = slice(offset, offset + width)
        offset += width
        if isinstance(matrix, CategoricalGroupMatrix):
            continue
        centre = mean_x[columns]
        if type(matrix) is DenseGroupMatrix:
            values = matrix.M
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                squares = (values[lo:hi] - centre) ** 2
                curvature[columns] += weights[lo:hi] @ squares
                mass[columns] += rows[lo:hi] @ squares
            continue
        formed.append(np.flatnonzero(unpenalized[columns]) + columns.start)
    if formed and np.concatenate(formed).size:
        chosen = np.concatenate(formed)
        zeros = np.zeros(dm.n)
        sums = column_sums(dm, chosen, mean_x, (zeros, zeros, weights), positive)
        curvature[chosen] = sums[2]
        mass[chosen] = sums[3]
    diagonal = np.asarray(penalty_diagonal, dtype=np.float64)
    bar = _gamma(row_count) * largest
    return (mass > 0.0) & (np.maximum(curvature, 0.0) + diagonal <= bar * mass)
