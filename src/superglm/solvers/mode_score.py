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
from dataclasses import dataclass

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    FactorSmoothGroupMatrix,
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


def offset_columns(dm: DesignMatrix, prior_weights: NDArray, center: NDArray) -> NDArray:
    """``(p,)`` bool: the dense columns whose centre lies beyond their spread, ``|c| > max|x - c|``.

    Over the rows of positive prior weight.  Only for such a column does the
    raw intercept lose a bit: ``eta`` formed through it rounds at ``(|c| +
    |x - c|) |beta|`` per row against the centred state's ``|x - c| |beta|``
    (one-engine design §3.8), within a factor two of each other otherwise.
    The centred readings of issue #430 (``centre_offset_mean``, a warm
    start's ``(alpha, c)``, the certificates' intercept) run when a design
    has such a column, and every other design computes as before, bit for
    bit.  A ``DenseGroupMatrix`` column is the only type whose entries are
    not bounded by their type; one pass over those columns, in fixed chunks.
    """
    center = np.asarray(center, dtype=np.float64)
    weights = np.asarray(prior_weights, dtype=np.float64)
    far = np.zeros(dm.p, dtype=bool)
    positive = weights > 0.0
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        centre = center[offset : offset + width]
        if type(matrix) is DenseGroupMatrix and np.any(centre != 0.0):
            values = matrix.M
            spread = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                rows = positive[lo:hi]
                if np.any(rows):
                    deviation = np.abs(values[lo:hi][rows] - centre)
                    spread = np.maximum(spread, np.max(deviation, axis=0))
            far[offset : offset + width] = np.abs(centre) > spread
        offset += width
    return far


def centre_offset_mean(
    dm: DesignMatrix,
    weights: NDArray,
    sum_w: float,
    center: NDArray,
    mean_x: NDArray,
    columns: NDArray,
) -> NDArray:
    """``d = sum W (x - c) / sum W``: a weighted column mean read about the state's centre ``c``.

    The centred intercept about ``c`` is ``alpha = mean_z - d' beta`` (one-engine
    design §3.8).  ``mean_x - c`` from the weighted mean ``mean_x`` rounds at
    ``u |c|``, the size of the column's offset, not of its spread: at a 1e6
    offset that is ``1e-10``, times the slope in every row's ``eta``.  A
    column ``columns`` marks (``offset_columns``: a dense column whose centre
    lies beyond its spread) is differenced row by row before its weighted
    sum, in fixed chunks as ``centred_matvec``, so ``d`` rounds at ``gamma_n
    max|x - c|``.  Every other column keeps ``mean_x - c``, which already
    rounds within a factor two of that, bit for bit as before.
    """
    offset_mean = np.asarray(mean_x, dtype=np.float64) - center
    w = np.asarray(weights, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        marked = np.asarray(columns[offset : offset + width], dtype=bool)
        if type(matrix) is DenseGroupMatrix and np.any(marked):
            centre = center[offset : offset + width]
            values = matrix.M
            accumulated = np.zeros(width)
            for lo in range(0, dm.n, _CHUNK):
                hi = min(lo + _CHUNK, dm.n)
                accumulated += (values[lo:hi] - centre).T @ w[lo:hi]
            block = offset_mean[offset : offset + width]
            block[marked] = accumulated[marked] / sum_w
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


def compensated_weighted_mean(values: NDArray, weights: NDArray) -> float:
    """``sum w v / sum w`` refined once with an error-free residual (``centred_intercept_remainder``).

    ``m0 = np.average(v, w)`` (pairwise or BLAS summation) errs by up to
    ``gamma_n mean_w |v|``, which on a two-level sample of adjacent floats
    is 1.5 ulp: the midpoint ``m*`` is no float, and ``m0`` landed an ulp
    beyond either neighbour.  One step of iterative refinement with the
    residual formed by TwoSum (``v - m0 = h + e`` exactly) returns ``m0 + d``,
    ``d = sum w (h + e) / sum w``:

        |m - m*| <= u |m*| + gamma_{n+2} sum w |v - m0| / sum w + O(u^2) mean_w |v|,

    which scales with the values' spread about their mean, not with ``|m|``.
    On the adjacent-float sample every quantity is exact and ``m`` is ``m*``
    correctly rounded.  ``m0`` when the residual is not finite.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    first = float(np.average(v, weights=w))
    head, error = two_sum(v, -first)
    with np.errstate(over="ignore", invalid="ignore"):
        correction = float(np.sum(w * head + w * error)) / float(np.sum(w))
    return first + correction if math.isfinite(correction) else first


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
    recovers it: ``y - o`` and ``- alpha`` are TwoSums, so the residual ``d =
    y - o - alpha - t`` rounds at its own size, and

        |alpha + alpha_lo - alpha*| <= gamma_{n+3} sum w |d| / sum w
                                       + gamma_n |alpha_lo| + O(u^2) mean_w |y - o|,

    which scales with the residuals, not with ``|eta|`` as ``alpha``'s own
    error does.  ``None`` when there is no positive weight or the residual
    overflows, and the predictor stays ``alpha + t``.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w))
    if not total > 0.0:
        return None
    response = np.asarray(y, dtype=np.float64)
    tail = 0.0
    if offset is not None and np.any(offset):
        response, tail = two_sum(response, -np.asarray(offset, dtype=np.float64))
    head, error = two_sum(response, -float(alpha))
    residual = (head - contribution) + (error + tail)
    with np.errstate(over="ignore", invalid="ignore"):
        remainder = float(np.sum(w * residual)) / total
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
    and weak test evaluated.
    """

    intercept_score: float
    slope_score: NDArray
    relative: NDArray
    bar_effective: NDArray
    weak: NDArray
    excluded: NDArray
    resolved: bool
    bar: float

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
    """
    n, p = dm.n, dm.p
    total = float(np.sum(row_score))
    absolute = np.abs(row_score)
    intercept_scale = max(_TINY, float(np.sum(absolute)))
    slope_score = centred_data_score(dm, row_score, mean_x) - penalty_score
    resolution = n * _EPS * np.abs(mean_x)
    root_weight = math.sqrt(max(float(sum_w), _TINY))
    zeta = intercept_scale / root_weight
    curvature_root = np.sqrt(
        (np.maximum(centered_scale, resolution) * root_weight) ** 2
        + np.maximum(np.asarray(penalty_curvature, dtype=np.float64), 0.0)
    )
    slope_scale = np.maximum(_TINY, zeta * curvature_root + np.abs(penalty_score))
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
            magnitude = penalty_magnitude[slopes]
            floor = (
                gamma_rows * evaluation
                + gamma_penalty * magnitude
                + _UNIT_ROUNDOFF * (represented + magnitude)
            ) / slope_scale[slopes]
            bar_effective[slopes + 1] = np.maximum(bar, floor)
            diagonal = np.asarray(penalty_curvature, dtype=np.float64)[slopes]
            weak[slopes] = curvature + diagonal <= _gamma(row_count) * largest * mass
    return ModeResidual(
        intercept_score=total,
        slope_score=slope_score,
        relative=relative,
        bar_effective=bar_effective,
        weak=weak,
        excluded=excluded | weak,
        resolved=resolved,
        bar=float(bar),
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
