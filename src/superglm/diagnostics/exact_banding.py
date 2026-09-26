"""Exact contiguous banding of a fitted curve under a per-value tolerance.

Chooses bands of consecutive values so that every value's band factor (the
band's weighted mean) lies within that value's tolerance, using the fewest
bands and, among those, the least weighted squared error.  This is dynamic
programming over contiguous segments (Fisher 1958; Bellman 1961; Jagadish et
al. 1998).  Minimising the pair (band count, error) lexicographically keeps the
principle of optimality: a best banding's prefix is a best banding of that
prefix, or swapping in a better prefix would improve the whole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MAX_EXACT_VALUES = 5000
_EPS = float(np.finfo(np.float64).eps)
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
        Weighted mean of the curve in each band.  The certificate covers the
        exact mean; a factor is that mean rounded once to a double, so a
        tolerance below half a unit in the last place of the curve's level
        cannot be met by any double.
    tolerance_factor : float
        Multiplier applied to the tolerances: 1.0 unless ``max_bands`` forced
        a wider limit.
    sse : float
        Weighted squared error of the banding.  It is formed from running
        moments about each band's last value, so it loses accuracy when that
        value sits far from the band's mean relative to the band's spread; it
        only orders bandings with the fewest bands, never their count.
    """

    starts: NDArray[np.intp]
    factors: NDArray[np.float64]
    tolerance_factor: float
    sse: float


def exact_bands(s, w, tol, max_bands: int) -> ExactBanding:
    """Band the curve ``s`` (weights ``w``) so each value stays within ``tol``.

    Parameters
    ----------
    s : array
        Curve values at distinct, ascending inputs.
    w : array
        Positive weight of each value.
    tol : array
        Nonnegative tolerance of each value: its band's weighted mean must lie
        within ``tol`` of it.  A single value is always its own mean.
    max_bands : int
        Largest number of bands allowed.  When the tolerances need more, they
        are all widened by the smallest factor that fits, reported as
        ``tolerance_factor``.
    """
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
    starts, sse = _fewest_then_least(s, w, tol)
    factor = 1.0
    if len(starts) > max_bands:
        factor, (starts, sse) = _smallest_fitting_factor(s, w, tol, max_bands)
    ends = np.append(starts[1:], len(s))
    # Averaged in the band's own frame, so the factor carries its final rounding
    # rather than one scaled by the curve's level.
    factors = np.array(
        [
            s[a] + np.average(s[a:b] - s[a], weights=w[a:b])
            for a, b in zip(starts, ends, strict=True)
        ],
        dtype=np.float64,
    )
    return ExactBanding(starts=starts, factors=factors, tolerance_factor=factor, sse=sse * scale)


def _validated(s, w, tol, max_bands):
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    tol = np.asarray(tol, dtype=np.float64)
    if s.ndim != 1 or s.shape != w.shape or s.shape != tol.shape:
        raise ValueError("s, w and tol must be 1-D arrays of the same length")
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
    if s.max() - s.min() > _RANGE_CEILING:
        raise ValueError(
            "exact banding needs a curve whose range is below 2**500, where its running "
            "moments stay finite; a log relativity never comes near it"
        )
    if isinstance(max_bands, bool) or not isinstance(max_bands, int | np.integer) or max_bands < 1:
        raise ValueError(f"max_bands must be a positive integer, got {max_bands!r}")
    return s, w, tol


def _fewest_then_least(s, w, tol) -> tuple[NDArray[np.intp], float]:
    """Fewest bands meeting ``tol``, then least weighted squared error.

    ``count[j]`` and ``sse[j]`` describe the best banding of the first ``j``
    values.  For each end ``j`` the candidate starts run leftwards until the
    tolerance window ``[max(s - tol), min(s + tol)]`` empties; the window only
    shrinks as a band grows, so every longer band is infeasible too.  Memory is
    O(n); the window scan is O(n) per end, so a solve is O(n^2).

    Everything is compared in the frame shifted by ``s[j]``, so the curve's
    level drops out of every rounding error, and a band is accepted only when
    its computed mean clears the window by more than that error: every
    accepted band keeps every value within its requested tolerance.  The cost
    of certifying is a tie at rounding level: a mean that meets a zero-width
    window exactly, as around a zero tolerance, is rejected unless the band is
    a plateau.
    """
    n = len(s)
    count = np.zeros(n + 1, dtype=np.int64)
    sse = np.zeros(n + 1, dtype=np.float64)
    start = np.zeros(n, dtype=np.intp)
    never = np.iinfo(np.int64).max
    for j in range(n):
        # d is exact for values within a factor two of s[j] (Sterbenz), and
        # otherwise off by at most eps |d|.
        d_all = s[j::-1] - s[j]
        t_all = tol[j::-1]
        lo = np.maximum.accumulate(d_all - t_all)
        hi = np.minimum.accumulate(d_all + t_all)
        closed = lo > hi
        m = int(np.argmax(closed)) if closed.any() else j + 1
        d = d_all[:m]
        wr = w[j::-1][:m]
        cw = np.cumsum(wr)
        mean = np.cumsum(wr * d) / cw
        cost = np.maximum(np.cumsum(wr * d * d) - cw * mean * mean, 0.0)
        # The mean of k shifted values errs by at most about (k + 2) eps max|d|,
        # and each window edge by eps times its own size (x + eps |x| is
        # increasing, so that covers every d - t below lo and d + t above hi).
        # Feasibility is then monotone in a widening factor, and a plateau
        # (d = 0) merges whatever its tolerances.
        length = np.arange(1, m + 1)
        max_d = np.maximum.accumulate(np.abs(d))
        # A product w * d below the smallest normal double rounds absolutely, by
        # at most 2**-1075 each, so that goes in too; a plateau (d = 0) has none.
        underflow = np.where(max_d > 0.0, length * 2.0**-1074 / cw, 0.0)
        err = 2.0 * _EPS * (length + 2) * max_d + underflow
        low_edge = lo[:m] + err + _EPS * np.abs(lo[:m])
        high_edge = hi[:m] - err - _EPS * np.abs(hi[:m])
        ok = (low_edge <= mean) & (mean <= high_edge)
        ok[0] = True  # a single value is its own mean
        first = j - np.arange(m)
        candidate = np.where(ok, count[first] + 1, never)
        fewest = candidate.min()
        total = np.where(candidate == fewest, sse[first] + cost, np.inf)
        k = int(np.argmin(total))
        count[j + 1] = fewest
        sse[j + 1] = total[k]
        start[j] = first[k]
    starts: list[int] = []
    j = n - 1
    while j >= 0:
        starts.append(int(start[j]))
        j = int(start[j]) - 1
    return np.array(starts[::-1], dtype=np.intp), float(sse[n])


def _banding_within(s, w, tol, factor: float, max_bands: int):
    """The banding at ``factor * tol`` if it fits ``max_bands``, else None.

    A widened tolerance is held at a quarter of the largest double: that still
    covers any curve's range, and a window edge can never become infinite.
    """
    with np.errstate(over="ignore"):
        widened = np.minimum(factor * tol, _TOLERANCE_CEILING)
    starts, sse = _fewest_then_least(s, w, widened)
    return (starts, sse) if len(starts) <= max_bands else None


def _smallest_fitting_factor(s, w, tol, max_bands: int):
    """Smallest multiplier of ``tol``, to relative 1e-6, whose banding fits ``max_bands``.

    Returns the factor and that banding.  Widening every tolerance only enlarges
    each feasible set (float64 rounding is monotone), so the fewest-band count
    never rises with the factor and bisection applies.  At ``2 R / tau``, with
    ``R`` the curve's range and ``tau`` its least positive tolerance, every
    positive-tolerance window holds every possible band mean, so no larger
    factor fits more; if that fails, zero tolerances are what stand in the
    way.  The bisection is geometric, so a factor near 1e30 takes about thirty
    solves.
    """
    positive = tol[tol > 0.0]
    if positive.size == 0:
        raise ValueError(
            f"cannot fit {len(s)} values into {max_bands} bands at any tolerance: "
            "values with zero tolerance cannot share a band"
        )
    # Past the largest double the bound is taken as the largest double.
    upper = max(2.0 * float(s.max() - s.min()) / float(positive.min()), 1.0)
    upper = min(upper, float(np.finfo(np.float64).max))
    banding = _banding_within(s, w, tol, upper, max_bands)
    if banding is None:
        raise ValueError(
            f"cannot fit {len(s)} values into {max_bands} bands at any tolerance a double "
            "can hold: values with zero or vanishingly small tolerance cannot share a band"
        )
    lower = 1.0
    while upper / lower - 1.0 > _FACTOR_RTOL:
        # sqrt of each bound, since their product can overflow.
        middle = math.sqrt(lower) * math.sqrt(upper)
        trial = _banding_within(s, w, tol, middle, max_bands)
        if trial is None:
            lower = middle
        else:
            upper, banding = middle, trial
    return upper, banding
