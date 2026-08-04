"""Private knot-placement helpers for spline feature specs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MAX_EXACT_FREQUENCY_MASS = float(2**53)


def knot_geometry_data(
    x: NDArray,
    sample_weight: NDArray | None,
) -> tuple[NDArray, NDArray | None]:
    """Return rows and frequency masses that may determine knot geometry.

    A zero frequency weight represents zero replicated rows, so it must not
    affect either learned boundaries or data-adaptive knots. Positive weights
    stay compact; callers never expand rows.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    if sample_weight is None:
        return values, None

    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if weights.shape != values.shape:
        raise ValueError(f"sample_weight must have length {len(values)}, got {len(weights)}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    active = weights > 0.0
    if not np.any(active):
        raise ValueError("sample_weight must not be all zero")
    return values[active], weights[active]


def _unique_frequency_mass(
    x: NDArray,
    sample_weight: NDArray | None,
) -> tuple[NDArray, NDArray]:
    """Aggregate physical-row counts or frequency weights by sorted value."""
    ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if sample_weight is None:
        return ux, counts.astype(np.float64)
    mass = np.bincount(
        inverse,
        weights=np.asarray(sample_weight, dtype=np.float64),
        minlength=len(ux),
    )
    return ux, mass


def _interpolate_frequency_cdf(
    values: NDArray,
    mass: NDArray,
    probs: NDArray,
) -> NDArray:
    """Interpolate a compact weighted empirical CDF."""
    cumulative = np.cumsum(mass)
    denom = cumulative[-1] - mass[0]
    if denom <= 0.0:
        return values[:1]
    cdf = (cumulative - mass[0]) / denom
    return np.interp(probs, cdf, values)


def row_quantile_knots(
    x: NDArray,
    n_knots: int,
    sample_weight: NDArray | None = None,
) -> NDArray:
    """Compute row-quantile knots without expanding frequency-weighted rows.

    Integral frequency weights with total mass at most ``2**53`` reproduce
    ``np.percentile(np.repeat(x, weights), ...)`` up to floating-point
    interpolation. Non-integral masses use compact weighted-CDF interpolation;
    they have no literal row-replication oracle.
    """
    probs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    if sample_weight is None:
        return np.unique(np.quantile(x, probs))

    ux, mass = _unique_frequency_mass(x, sample_weight)
    if len(ux) < 2:
        return ux

    total_mass = float(np.sum(mass))
    integral_mass = np.all(mass == np.floor(mass))
    if integral_mass and total_mass <= _MAX_EXACT_FREQUENCY_MASS:
        cumulative = np.cumsum(mass)
        ranks = probs * (total_mass - 1.0)
        lower_rank = np.floor(ranks)
        upper_rank = np.ceil(ranks)
        fraction = ranks - lower_rank
        lower = ux[np.searchsorted(cumulative, lower_rank, side="right")]
        upper = ux[np.searchsorted(cumulative, upper_rank, side="right")]
        return np.unique(lower + fraction * (upper - lower))

    return np.unique(_interpolate_frequency_cdf(ux, mass, probs))


def weighted_quantile_knots(
    x: NDArray,
    n_knots: int,
    alpha: float,
    sample_weight: NDArray | None = None,
) -> NDArray:
    """Compute tempered-quantile knots from compact frequency mass.

    For integer frequency weights, aggregating mass by unique value before
    applying ``mass**alpha`` is algebraically identical to expanding the rows
    and recounting duplicates.
    """
    ux, counts = _unique_frequency_mass(x, sample_weight)
    if len(ux) < 2:
        return ux
    w = counts.astype(np.float64) ** alpha
    probs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    raw = _interpolate_frequency_cdf(ux, w, probs)
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
    sample_weight: NDArray | None = None,
) -> tuple[NDArray, str]:
    """Resolve interior knots and the effective knot-placement strategy."""
    if explicit_knots is not None:
        return explicit_knots, "explicit"

    if knot_strategy in ("quantile", "quantile_rows", "quantile_tempered"):
        if explicit_boundary is not None:
            inside = (x >= lo) & (x <= hi)
            x_q = x[inside]
            weight_q = None if sample_weight is None else sample_weight[inside]
        else:
            x_q = x
            weight_q = sample_weight
        if len(x_q) == 0:
            x_q = np.array([lo, hi])
            weight_q = None
        if knot_strategy == "quantile_tempered":
            interior = weighted_quantile_knots(
                x_q,
                n_knots,
                knot_alpha,
                sample_weight=weight_q,
            )
        elif knot_strategy == "quantile_rows":
            interior = row_quantile_knots(
                x_q,
                n_knots,
                sample_weight=weight_q,
            )
        else:
            probs = np.linspace(0, 100, n_knots + 2)[1:-1]
            interior = np.unique(np.percentile(np.unique(x_q), probs))
        if len(interior) < n_knots:
            return np.linspace(lo, hi, n_knots + 2)[1:-1], "uniform"
        return interior, knot_strategy

    return np.linspace(lo, hi, n_knots + 2)[1:-1], "uniform"


__all__ = [
    "knot_geometry_data",
    "resolve_interior_knots",
    "row_quantile_knots",
    "weighted_quantile_knots",
]
