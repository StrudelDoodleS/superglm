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


@dataclass(frozen=True)
class ExactBanding:
    """Bands chosen by :func:`exact_bands`.

    Attributes
    ----------
    starts : NDArray
        Index of each band's first value, ascending, starting at 0.
    factors : NDArray
        Weighted mean of the curve in each band.
    tolerance_factor : float
        Multiplier applied to the tolerances: 1.0 unless ``max_bands`` forced
        a wider limit.
    sse : float
        Weighted squared error of the banding.
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
    starts, sse = _fewest_then_least(s, w, tol)
    factor = 1.0
    if len(starts) > max_bands:
        factor, (starts, sse) = _smallest_fitting_factor(s, w, tol, max_bands)
    ends = np.append(starts[1:], len(s))
    factors = np.array(
        [np.average(s[a:b], weights=w[a:b]) for a, b in zip(starts, ends, strict=True)],
        dtype=np.float64,
    )
    return ExactBanding(starts=starts, factors=factors, tolerance_factor=factor, sse=sse)


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
    accepted band keeps every value within its requested tolerance.
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
        # The mean of k shifted values errs by at most about (k + 2) eps max|d|
        # and each window edge by eps (|d| + tol); a zero-tolerance plateau has
        # d = 0 and no error, so it still merges.
        k = np.arange(1, m + 1)
        err = (
            2.0
            * _EPS
            * ((k + 2) * np.maximum.accumulate(np.abs(d)) + np.maximum.accumulate(t_all[:m]))
        )
        ok = (lo[:m] + err <= mean) & (mean <= hi[:m] - err)
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
    """The banding at ``factor * tol`` if it fits ``max_bands``, else None."""
    starts, sse = _fewest_then_least(s, w, factor * tol)
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
        upper = math.inf
    else:
        upper = 2.0 * float(s.max() - s.min()) / float(positive.min())
        if not math.isfinite(upper):
            raise ValueError(
                f"cannot fit {len(s)} values into {max_bands} bands: the smallest positive "
                f"tolerance, {float(positive.min()):.3g}, is too small to widen in double precision"
            )
    banding = _banding_within(s, w, tol, upper, max_bands) if math.isfinite(upper) else None
    if banding is None:
        raise ValueError(
            f"cannot fit {len(s)} values into {max_bands} bands at any tolerance: "
            "values with zero tolerance cannot share a band"
        )
    lower = 1.0
    while upper / lower - 1.0 > _FACTOR_RTOL:
        middle = math.sqrt(lower * upper)
        trial = _banding_within(s, w, tol, middle, max_bands)
        if trial is None:
            lower = middle
        else:
            upper, banding = middle, trial
    return upper, banding
