"""Private knot-placement helpers for spline feature specs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def weighted_quantile_knots(x: NDArray, n_knots: int, alpha: float) -> NDArray:
    """Compute interior knots via weighted quantiles of unique values."""
    ux, counts = np.unique(x, return_counts=True)
    if len(ux) < 2:
        return ux
    w = counts.astype(np.float64) ** alpha
    cw = np.cumsum(w)
    denom = cw[-1] - w[0]
    if denom <= 0:
        return ux[:1]
    cdf = (cw - w[0]) / denom
    probs = np.linspace(0, 1, n_knots + 2)[1:-1]
    raw = np.interp(probs, cdf, ux)
    return np.unique(raw)


def resolve_interior_knots(
    x: NDArray,
    *,
    lo: float,
    hi: float,
    n_knots: int,
    knot_strategy: str,
    knot_alpha: float,
    explicit_knots: NDArray | None,
    explicit_boundary: tuple[float, float] | None,
) -> tuple[NDArray, str]:
    """Resolve interior knots and the effective knot-placement strategy."""
    if explicit_knots is not None:
        return explicit_knots, "explicit"

    if knot_strategy in ("quantile", "quantile_rows", "quantile_tempered"):
        x_q = x[(x >= lo) & (x <= hi)] if explicit_boundary is not None else x
        if len(x_q) == 0:
            x_q = np.array([lo, hi])
        if knot_strategy == "quantile_tempered":
            interior = weighted_quantile_knots(x_q, n_knots, knot_alpha)
        else:
            probs = np.linspace(0, 100, n_knots + 2)[1:-1]
            source = np.unique(x_q) if knot_strategy == "quantile" else x_q
            interior = np.unique(np.percentile(source, probs))
        if len(interior) < n_knots:
            return np.linspace(lo, hi, n_knots + 2)[1:-1], "uniform"
        return interior, knot_strategy

    return np.linspace(lo, hi, n_knots + 2)[1:-1], "uniform"


__all__ = ["resolve_interior_knots", "weighted_quantile_knots"]
