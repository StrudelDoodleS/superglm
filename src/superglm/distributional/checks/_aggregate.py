"""One grouped aggregation for every table in the suite.

A grouped mean of a rate is a **ratio of sums**, ``sum(w y) / sum(w)``, never
the mean of the per-row ratios ``y_i``: with exposure weights the first is total
cost over total exposure and the second gives a one-day policy the same say as a
one-year policy.  Every decile, bin, level, band and segment mean in the
inference suite goes through :func:`grouped_ratio`, so no table can take the
wrong mean by accident.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _weight_vector(values: NDArray, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of row contributions")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def grouped_ratio(
    numerator: NDArray,
    denominator: NDArray,
    groups: NDArray,
    *,
    n_groups: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(sum_numerator, sum_denominator, ratio)`` per group.

    ``numerator`` and ``denominator`` are the already-weighted row
    contributions -- ``w * y`` and ``w`` for a weighted mean of ``y``, or
    ``w * y`` and ``w * mu_hat`` for an actual-over-expected ratio.  ``groups``
    holds non-negative integer codes as ``pandas.factorize`` and
    ``numpy.digitize`` produce them; ``n_groups`` declares the width of the
    result when the codes do not visit every group, which keeps a table's row
    count equal to its bin count.

    The ratio is ``NaN`` wherever the summed denominator is zero: an empty bin
    has no mean, and reporting one as zero would put a false number in a table.
    """
    values = _weight_vector(numerator, name="numerator")
    weights = _weight_vector(denominator, name="denominator")
    codes = np.asarray(groups)
    if codes.ndim != 1:
        raise ValueError("groups must be a one-dimensional array of group codes")
    if not (values.shape == weights.shape == codes.shape):
        raise ValueError("numerator, denominator and groups must have the same number of rows")
    if not np.issubdtype(codes.dtype, np.integer) or (codes.size and int(codes.min()) < 0):
        raise ValueError("groups must be non-negative integer codes")

    minimum = int(codes.max()) + 1 if codes.size else 0
    if n_groups is None:
        width = minimum
    else:
        width = int(n_groups)
        if width < minimum:
            raise ValueError(f"n_groups must be at least {minimum} to hold every group code")

    total = np.bincount(codes, weights=values, minlength=width).astype(np.float64, copy=False)
    exposure = np.bincount(codes, weights=weights, minlength=width).astype(np.float64, copy=False)
    ratio = np.full(width, np.nan, dtype=np.float64)
    np.divide(total, exposure, out=ratio, where=exposure != 0.0)
    return total, exposure, ratio


__all__ = ["grouped_ratio"]
