"""Exact contiguous banding of fitted curves under a per-value tolerance.

Chooses bands of consecutive values so that every value's band factor (the
band's weighted mean) lies within that value's tolerance, using the fewest
bands and, among those, the least weighted squared error.  This is dynamic
programming over contiguous segments (Fisher 1958; Bellman 1961; Jagadish et
al. 1998).  Minimising the pair (band count, error) lexicographically keeps the
principle of optimality: a best banding's prefix is a best banding of that
prefix, or swapping in a better prefix would improve the whole.

Several curves can share one set of bands.  A segment is then allowed when
every curve's weighted mean over it meets that curve's own tolerances, and its
error is the sum of the curves' errors, so the same recursion holds with a
segment's feasibility the conjunction over curves and its cost the sum
(Terzi and Tsaparas 2006 segment a d-dimensional sequence at common
boundaries by the same sum over dimensions; Runge, Kostic, Combeau and Romano
2026 solve the fewest-valid-segments, then least-cost, recursion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

MAX_EXACT_VALUES = 5000
_UNIT_ROUNDOFF = 2.0**-53
_FACTOR_RTOL = 1e-6
_TOLERANCE_CEILING = float(np.finfo(np.float64).max) / 4.0
# Below this range the running moments w d and w d**2 stay finite for any band
# (5000 R**2 below the largest double); exp overflows long before, at ~709.
_RANGE_CEILING = 2.0**500


@dataclass(frozen=True)
class ExactBanding:
    """Bands chosen by :func:`exact_bands`.

    Attributes
    ----------
    starts : NDArray
        Index of each band's first value, ascending, starting at 0.
    factors : NDArray
        Shape ``(bands,)`` for one curve, ``(bands, curves)`` for several.
        The double each band publishes: its weighted mean, summed value by
        value in the frame of the band's first value and added back to it,
        then moved, if it misses a tolerance, to the nearest double that meets
        them all.  Every value is within its tolerance of its factor exactly,
        the tolerance being the double the solve used: ``tol`` itself, or when
        widened ``min(fl(tolerance_factor * tol), largest double / 4)``, curve
        by curve.  An unmoved factor differs from the exact mean by its final
        rounding, half an ulp, plus about ``2 k u`` times the band's spread,
        with ``u = 2**-53`` the unit roundoff, and :func:`_underflow_margin`;
        a moved one by less than one ulp more.  A band whose tolerances leave
        no double between them, for any curve, is never formed.
    tolerance_factor : float
        Multiplier applied to the tolerances: 1.0 unless ``max_bands`` forced
        a wider limit.  One factor serves every curve.
    sse : float
        Weighted squared error of the published factors, the sum of
        ``w * (s - factor)**2`` in the units of ``w``, over every curve.  Each
        curve's sum is formed directly from nonnegative terms and the curves'
        sums are added with one correctly rounded ``math.fsum``, so its
        relative error is at most
        ``(n + 5) u / (1 - (n + 5) u)`` (Higham 2002, Lemma 3.1 and section
        4.2), plus two absolute terms from underflow: ``n * 2**-1073`` times
        the largest weight from the terms, and ``2**-1075`` from the final
        rescale by the largest weight, which can round into the subnormals.
        It is always finite: weights that would put it past the largest double
        are refused.
    """

    starts: NDArray[np.intp]
    factors: NDArray[np.float64]
    tolerance_factor: float
    sse: float


# The certificate accounts for underflow (_underflow_margin), so a caller's
# np.seterr(under="raise") must not turn the subnormal products it covers into errors.
@np.errstate(under="ignore")
def exact_bands(s, w, tol, max_bands: int) -> ExactBanding:
    """Band the curve ``s`` (weights ``w``) so each value stays within ``tol``.

    Several curves share one banding when ``s`` and ``tol`` are ``(n, m)``:
    a band must then hold every curve's factor inside that curve's tolerances,
    and the weights are common to the curves.  One curve, a 1-D ``s``, gives
    the 1-D factors it always has.

    Parameters
    ----------
    s : array
        Curve values at distinct, ascending inputs: shape ``(n,)``, or
        ``(n, m)`` for ``m`` curves read at the same inputs.
    w : array
        Positive weight of each of the ``n`` values, shared by the curves.
    tol : array
        Nonnegative tolerance of each value, of ``s``'s shape: its band's
        factor, the band's weighted mean as a double, must lie within ``tol``
        of it.  A single value is always its own factor.
    max_bands : int
        Largest number of bands allowed.  When the tolerances need more, they
        are all widened by the smallest factor that fits, reported as
        ``tolerance_factor``.

    Raises
    ------
    ValueError
        For invalid inputs, for tolerances no banding within ``max_bands`` can
        meet, and when the banding's weighted squared error is past the
        largest double, as it can be for weights near it.  Only the weights'
        ratios set the bands, so dividing them all by a power of two that
        keeps them normal gives the same bands.
    """
    one_curve = np.ndim(s) == 1
    s, w, tol = _validated(s, w, tol, max_bands)
    # Only ratios of weights matter; scaling by the largest keeps w * d finite.
    scale = float(w.max())
    w = w / scale
    if np.any(w < np.finfo(np.float64).tiny):
        # Below the smallest normal double the rounding is absolute, not relative,
        # and the smallest weights would vanish from their bands' means.
        raise ValueError(
            "exact banding weights span more than a double can average: the smallest "
            "is below 2**-1022 of the largest"
        )
    starts, factors = _fewest_then_least(s, w, tol)
    factor = 1.0
    if len(starts) > max_bands:
        factor, (starts, factors) = _smallest_fitting_factor(s, w, tol, max_bands)
    normalised = _squared_error(s, w, starts, factors)
    # A Python float product: it overflows to inf silently, under any np.seterr.
    sse = normalised * scale
    if math.isinf(sse):
        raise ValueError(
            f"exact banding weights are too large: the banding's weighted squared error, "
            f"{normalised:.6g} times the largest weight {scale:.6g}, is past the largest "
            "double. Only the weights' ratios set the bands, so divide them all by a power "
            "of two near the largest, which leaves the bands unchanged."
        )
    return ExactBanding(
        starts=starts,
        factors=factors[:, 0] if one_curve else factors,
        tolerance_factor=factor,
        sse=sse,
    )


def _validated(s, w, tol, max_bands):
    """The inputs as float64 arrays, ``s`` and ``tol`` always ``(n, m)``."""
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    tol = np.asarray(tol, dtype=np.float64)
    if s.ndim not in (1, 2) or w.ndim != 1 or s.shape != tol.shape or len(s) != len(w):
        raise ValueError(
            "s, w and tol must have the same length: w a 1-D array, s and tol both "
            "1-D arrays or both (length, curves) arrays"
        )
    if s.ndim == 1:
        s, tol = s[:, None], tol[:, None]
    if len(s) == 0:
        raise ValueError("exact banding needs at least one value")
    if len(s) > MAX_EXACT_VALUES:
        raise ValueError(
            f"exact banding supports at most {MAX_EXACT_VALUES} distinct values, got {len(s)}"
        )
    if not (np.isfinite(s).all() and np.isfinite(w).all() and np.isfinite(tol).all()):
        raise ValueError("s, w and tol must be finite")
    if np.any(w <= 0.0):
        raise ValueError("exact banding weights must be positive")
    if np.any(tol < 0.0):
        raise ValueError("exact banding tolerances must be nonnegative")
    with np.errstate(over="ignore"):
        span = (s.max(axis=0) - s.min(axis=0)).max()
    if span > _RANGE_CEILING:
        raise ValueError(
            "exact banding needs a curve whose range is below 2**500, where its running "
            "moments stay finite; a log relativity never comes near it"
        )
    if isinstance(max_bands, bool) or not isinstance(max_bands, int | np.integer) or max_bands < 1:
        raise ValueError(f"max_bands must be a positive integer, got {max_bands!r}")
    return s, w, tol


def _fewest_then_least(s, w, tol, *, count_only: bool = False) -> Any:
    """Fewest bands meeting ``tol``, then least weighted squared error at their factors.

    ``s`` and ``tol`` are ``(n, m)`` for ``m`` curves sharing the bands, or 1-D
    for one, which returns 1-D factors.  Returns each band's start and the
    factor it publishes per curve, or with ``count_only`` just the number of
    bands: the same ``count`` the full pass forms, without the squared errors and
    factors that only choose among bands of equal count, which is what asking
    whether ``tol`` fits a cap needs.  ``count[j]`` and ``sse[j]`` describe the
    best banding of the first ``j`` values.  A band is allowed when every
    curve's factor meets that curve's tolerances, and it costs the sum over
    curves of its squared error, so feasibility is a conjunction and the cost
    a sum over curves and a segment's value is fixed by its own values alone:
    the recursion over the last band's start is Bellman's, with the last
    band's cost the sum over curves.

    For each end ``j`` the candidate starts run leftwards until some curve's
    tolerance window ``[max(s - tol), min(s + tol)]`` empties; the window only
    shrinks as a band grows, so every longer band is infeasible too.  Starts
    that closed at an earlier end are not scanned again, so an end costs the
    number of its open starts times the number of curves, and the pass is
    O(n W m) for windows of W starts and O(n^2 m) at worst.  Memory is
    O(n m).  The mean test is not hereditary (a band whose mean is
    inside its windows can hold a part whose mean is not), so every open start
    is tested, not only the longest; and the best count is not monotone in
    ``j`` either, so no start can be skipped on the strength of a smaller prefix.

    Everything is compared in the frame shifted by ``s[j]``, so the curve's
    level drops out of every rounding error, and a band is accepted only when
    its computed mean clears the window by more than that error: every
    accepted band keeps every value within its requested tolerance.  The cost
    of certifying is a tie at rounding level: a mean that meets a zero-width
    window exactly, as around a zero tolerance, is rejected unless the band is
    a plateau.  Each curve is certified on its own columns with the same
    arithmetic one curve gets, so the certificate holds curve by curve.

    A band also needs a double that meets every tolerance in it, for every
    curve, to carry as its factor.  The exact mean can fall between two
    doubles that each miss one tolerance, and such a band is not formed.

    A band's factor is its weighted mean, summed value by value in the frame
    of its first value, added back to that value and held to those doubles.
    Its cost is its squared error at that factor, ``M2 + W (factor - mean)**2``,
    where ``M2``, the squared error about the mean, is kept for every open
    start by West's (1979) weighted update as each end is added.  That
    update's rounding grows like ``k kappa u`` in the band's condition
    number, where the running moments' grew like ``k kappa**2 u`` (Chan,
    Golub and LeVeque 1983, Table 1).  The objective is the published
    factors' error, a sum over bands each fixed by its own values, so the
    principle of optimality holds for it.
    """
    one_curve = s.ndim == 1
    if one_curve:
        s, tol = s[:, None], tol[:, None]
    n, curves = s.shape
    count = np.zeros(n + 1, dtype=np.int64)
    sse = np.zeros(n + 1, dtype=np.float64)
    start = np.zeros(n, dtype=np.intp)
    published = np.zeros((n, curves), dtype=np.float64)
    never = np.iinfo(np.int64).max
    underflow = _underflow_margin(w)
    low_double, high_double = _representable_bounds(s, tol)
    # Per start a, the band [a, j] so far: its weight (common to the curves),
    # and per curve its first moment about s[a] and its squared error about
    # its mean.
    weight = np.zeros(n, dtype=np.float64)
    moment = np.zeros((n, curves), dtype=np.float64)
    square = np.zeros((n, curves), dtype=np.float64)
    floor = 0
    for j in range(n):
        # The starts still open are floor..j, reversed so that row k is start
        # j - k; the accumulations below run leftwards from j, so a row's
        # window is unchanged by cutting the rows past the open ones off.
        # d is exact for values within a factor two of s[j] (Sterbenz), and
        # otherwise off by at most u |d|.
        d_all = s[floor : j + 1][::-1] - s[j]
        t_all = tol[floor : j + 1][::-1]
        lo = np.maximum.accumulate(d_all - t_all, axis=0)
        hi = np.minimum.accumulate(d_all + t_all, axis=0)
        closed = (lo > hi).any(axis=1)
        m = int(np.argmax(closed)) if closed.any() else j - floor + 1
        # A window that has closed stays closed as its band grows, exactly; the
        # floor keeps it closed where rounding would reopen it, so every open
        # start's sums hold its whole band.
        floor = max(floor, j - m + 1)
        m = j - floor + 1
        open_starts = slice(floor, j + 1)
        if not count_only:
            x = s[j] - s[open_starts]  # value j in each open start's frame
            before = weight[open_starts]
            after = before + w[j]
            gap = x - np.divide(
                moment[open_starts],
                before[:, None],
                out=np.zeros((m, curves)),
                where=(before > 0.0)[:, None],
            )
            # West's increment w[j] * before / after, ordered so that its first
            # product is at least half the smaller weight, so it loses at most
            # one bit to underflow.
            lighter = np.minimum(before, w[j])
            square[open_starts] += (
                (lighter * (np.maximum(before, w[j]) / after))[:, None] * gap * gap
            )
            moment[open_starts] += w[j] * x
            weight[open_starts] = after
        d = d_all[:m]
        wr = w[floor : j + 1][::-1]
        cw = np.cumsum(wr)
        mean = np.cumsum(wr[:, None] * d, axis=0) / cw[:, None]
        # The mean of k shifted values in the solve's weights, w / scale, errs
        # by at most 2k u max|d| to first order: 2k - 1 roundings in its sums
        # and division (d[0] = 0, so the numerator's first addition is exact;
        # Higham 2002, Lemma 3.1) and u max|d| from rounding d.  Each window
        # edge errs by u max|d| from its d plus u times its own size.  The
        # caller's exact mean differs from the solve's, since w / scale rounds
        # each weight by a relative e_i, |e_i| <= u, unless the largest weight
        # is a power of two (exports pass 1).  That moves the mean by
        # sum w_i e_i (d_i - mean) / sum w_i (1 + e_i), at most u / (1 - u)
        # times max|d_i - mean| <= max d - min d <= 2 max|d|.  The margin takes
        # 4 (k + 2) u max|d|, more than twice those three, and 2 u times each
        # edge's size (x + 2 u |x| is increasing, so that covers every d - t
        # below lo and d + t above hi).  The two additions that form low_edge
        # and high_edge can round that 2 u |edge| term away; err's own roundings
        # are second order.  The (2k + 5) u max|d| left after the three errors
        # covers the edge's u |edge| instead wherever it decides the band, with
        # u max|d| to spare for the second-order terms: the caller's mean is
        # within max|d| of zero, so an edge within (2k + 3) max|d| of it is
        # within (2k + 4) max|d| of zero, and an edge further from it is far
        # past its own rounding.  Feasibility is then monotone in a widening
        # factor, and a plateau (d = 0) merges whatever its tolerances.  All of
        # this is per column: max|d| is each curve's own, and a column does not
        # see the others' values.
        length = np.arange(1, m + 1)
        max_d = np.maximum.accumulate(np.abs(d), axis=0)
        err = 4.0 * _UNIT_ROUNDOFF * (length + 2)[:, None] * max_d
        # max_d starts at 0 and never falls: the underflow margin goes on every
        # band past the plateau, which has none.
        err[max_d > 0.0] += underflow
        low_edge = lo[:m] + err + 2.0 * _UNIT_ROUNDOFF * np.abs(lo[:m])
        high_edge = hi[:m] - err - 2.0 * _UNIT_ROUNDOFF * np.abs(hi[:m])
        # The bounds are exact, so this test is too: no rounding margin.
        low_band = np.maximum.accumulate(low_double[floor : j + 1][::-1], axis=0)
        high_band = np.minimum.accumulate(high_double[floor : j + 1][::-1], axis=0)
        ok = ((low_edge <= mean) & (mean <= high_edge) & (low_band <= high_band)).all(axis=1)
        ok[0] = True  # a single value is its own mean
        candidate = np.where(ok, count[floor : j + 1][::-1] + 1, never)
        fewest = candidate.min()
        count[j + 1] = fewest
        if count_only:
            continue
        # Only the starts that reach the fewest count compete on error, so the
        # factor and the cost are formed for those alone; they stay in order of
        # increasing band length, so a tie goes to the shortest band, as before.
        best = np.flatnonzero(candidate == fewest)
        band_weight = weight[open_starts][::-1][best]
        shift = moment[open_starts][::-1][best] / band_weight[:, None]
        first_value = s[floor : j + 1][::-1][best]
        factor = np.clip(first_value + shift, low_band[best], high_band[best])
        miss = (factor - first_value) - shift
        cost = (square[open_starts][::-1][best] + band_weight[:, None] * miss * miss).sum(axis=1)
        total = sse[floor : j + 1][::-1][best] + cost
        k = int(np.argmin(total))
        sse[j + 1] = total[k]
        start[j] = j - best[k]
        published[j] = factor[k]
    if count_only:
        return int(count[n])
    starts: list[int] = []
    factors: list[NDArray[np.float64]] = []
    j = n - 1
    while j >= 0:
        starts.append(int(start[j]))
        factors.append(published[j])
        j = int(start[j]) - 1
    banded = np.array(factors[::-1], dtype=np.float64)
    return np.array(starts[::-1], dtype=np.intp), banded[:, 0] if one_curve else banded


def _squared_error(s, w, starts, factors) -> float:
    """Weighted squared error of ``s`` about its bands' factors, summed directly.

    Each term ``w (s - factor)**2`` is nonnegative and carries a relative error
    of at most ``gamma_4``, its difference entering twice, so no cancellation
    can occur; summing n of them in any order adds at most ``gamma_(n - 1)``
    (Higham 2002, Lemma 3.1 and eq. 4.4).  A curve's sum has relative error at
    most ``gamma_(n + 3)``, plus underflow; the curves' sums are nonnegative
    too and ``math.fsum`` rounds their total once, so the total's is at most
    ``gamma_(n + 4)``, plus underflow.  ``s`` and ``factors`` are ``(n, m)`` and
    ``(bands, m)``.
    """
    gap = s - np.repeat(factors, np.diff(np.append(starts, len(s))), axis=0)
    return math.fsum(float(np.sum(w * gap[:, c] * gap[:, c])) for c in range(s.shape[1]))


def _underflow_margin(w) -> float:
    """Absolute error bound on a band mean from products ``w * d`` that underflow.

    Each rounds by at most 2**-1075, and a band of ``k`` values weighs at least
    ``k * min(w)``, so its underflowed products move the mean by less than
    ``2**-1075 / min(w)``; 2**-1074 more covers the division's own rounding and
    this bound's.  ``w`` is scaled to a maximum of 1 and refused below the
    smallest normal double, so this is at most about ``2**-52``.
    """
    return 2.0**-1074 / float(w.min()) + 2.0**-1074


def _representable_bounds(s, tol):
    """Per value, the least double not below ``s - tol`` and the greatest not above ``s + tol``.

    A double meets a value's tolerance exactly when it lies between the two.
    Each bound is the rounded sum, moved one place when its rounding error
    fell on the wrong side.  Fast2Sum (Dekker 1971), with the larger
    magnitude first, gives that error exactly in round-to-nearest, through
    underflow (Hauser 1996), and none of its later steps overflow when the
    sum itself does not (Boldo, Graillat and Muller 2017, Theorem 5.1).  A sum
    past the largest double makes that double its bound, which binds no factor.
    A step into the subnormals flags underflow, though it is exact.
    """
    with np.errstate(over="ignore", under="ignore"):
        low = _rounded_toward(s, -tol, np.inf)
        high = _rounded_toward(s, tol, -np.inf)
    return low, high


def _rounded_toward(a, b, direction: float):
    """``a + b`` rounded to a double in ``direction``, +inf or -inf, exactly."""
    first = np.abs(a) >= np.abs(b)
    big = np.where(first, a, b)
    small = np.where(first, b, a)
    total = big + small
    error = small - (total - big)  # a + b == total + error, by Fast2Sum
    wrong_side = error > 0.0 if direction > 0.0 else error < 0.0
    return np.where(wrong_side, np.nextafter(total, direction), total)


def _widened(tol, factor: float):
    """``factor * tol``, each held at a quarter of the largest double.

    That still covers any curve's range, and a window edge can never become
    infinite.
    """
    with np.errstate(over="ignore"):
        return np.minimum(factor * tol, _TOLERANCE_CEILING)


def _fits(s, w, tol, factor: float, max_bands: int) -> bool:
    """Whether the fewest bands at ``factor * tol`` are at most ``max_bands``."""
    return _fewest_then_least(s, w, _widened(tol, factor), count_only=True) <= max_bands


def _smallest_fitting_factor(s, w, tol, max_bands: int):
    """Smallest multiplier of ``tol``, to relative 1e-6, whose banding fits ``max_bands``.

    Returns the factor and that banding (starts and factors).  Widening every tolerance only enlarges
    each feasible set (float64 rounding is monotone), so the fewest-band count
    never rises with the factor and bisection applies.  The search starts at
    ``2 * reach / tau``, with ``reach`` the widest curve's range plus
    :func:`_underflow_margin` and ``tau`` the least positive tolerance.  When
    that is a double, every positive-tolerance window there holds every
    possible band mean and every value of the curve, so a double as well, and
    no larger multiplier fits more: if it fails, zero tolerances are what stand
    in the way.  When it is past the largest double, the search starts at the
    largest double instead, and failing there means no multiplier a double can
    hold fits.  The bisection is geometric, so a factor near 1e30 takes about
    thirty passes.  A pass only has to say whether the fewest bands fit, which
    the band count alone answers (:func:`_fewest_then_least` with
    ``count_only``), and the banding is formed once, at the factor kept.
    """
    positive = tol[tol > 0.0]
    if positive.size == 0:
        raise ValueError(
            f"cannot fit {len(s)} values into {max_bands} bands at any tolerance: "
            "values with zero tolerance cannot share a band"
        )
    # Past the largest double the bound is taken as the largest double.
    reach = float((s.max(axis=0) - s.min(axis=0)).max()) + _underflow_margin(w)
    upper = max(2.0 * reach / float(positive.min()), 1.0)
    upper = min(upper, float(np.finfo(np.float64).max))
    # Fitting is monotone in the factor, so one pass settles every larger factor
    # (it fits) and every smaller one (it does not).  The bisection below is
    # walked on what is known, so its path, and its result, are those of a pass
    # at every trial; but its first trials sit far above the answer, where the
    # windows span the whole curve and a pass costs the most.  Doubling from
    # 1, where the caller found the bands too many, brackets the answer at a
    # tolerance close to it, and those trials are never run.
    does_not_fit, fits = 1.0, upper
    probe = 2.0
    while probe < upper:
        if _fits(s, w, tol, probe, max_bands):
            fits = probe
            break
        does_not_fit = probe
        probe *= 2.0
    else:
        if not _fits(s, w, tol, upper, max_bands):
            raise ValueError(
                f"cannot fit {len(s)} values into {max_bands} bands at any tolerance multiplier "
                "a double can hold: values with zero or vanishingly small tolerance cannot share "
                "a band"
            )
    lower = 1.0
    while upper / lower - 1.0 > _FACTOR_RTOL:
        # sqrt of each bound, since their product can overflow.
        middle = math.sqrt(lower) * math.sqrt(upper)
        if middle >= fits:
            fit = True
        elif middle <= does_not_fit:
            fit = False
        else:
            fit = _fits(s, w, tol, middle, max_bands)
            if fit:
                fits = middle
            else:
                does_not_fit = middle
        if fit:
            upper = middle
        else:
            lower = middle
    # The trials only counted bands; the banding is formed once, at the factor kept.
    return upper, _fewest_then_least(s, w, _widened(tol, upper))
