"""Shared private array helpers used across superglm submodules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _ensure_array(a, dtype=np.float64) -> NDArray:
    """Convert input to a 1-D float64 array."""
    return np.asarray(a, dtype=dtype)


def _ensure_1d_float(x) -> NDArray:
    """Convert to 1-D float64, raveling if needed."""
    return np.asarray(x, dtype=np.float64).ravel()


def _default_weights(w, n: int) -> NDArray:
    """Return uniform weights if *w* is None, else ensure array."""
    if w is None:
        return np.ones(n, dtype=np.float64)
    return np.asarray(w, dtype=np.float64)


def _validate_strict_prior_weights(weights, n: int) -> NDArray:
    """Return one-dimensional, finite, strictly positive float64 prior weights."""
    message = f"weights must be finite and strictly positive, one-dimensional, and have length {n}"
    try:
        raw = np.asarray(weights)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if (
        raw.ndim != 1
        or raw.shape[0] != n
        or np.iscomplexobj(raw)
        or getattr(raw.dtype, "kind", None) in {"M", "m"}
    ):
        raise ValueError(message)
    try:
        validated = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(validated)) or np.any(validated <= 0.0):
        raise ValueError(message)
    return validated


def weighted_row_sum(weights, row_term: Callable[..., NDArray], *rows) -> float:
    """Return ``sum(w * row_term(*rows))`` with each row of weight exactly 0 an exact 0.

    A zero-weight row is not in the fit, so its term is never evaluated: a
    Gaussian identity mean or response past ``sqrt(DBL_MAX)`` there made its
    unit deviance ``inf``, and ``0 * inf`` turned the whole sum into NaN, which
    rejected every trial step (issue #369).  With no zero weight this is the
    unchanged expression.  With one, every other row keeps its product in its
    position and the zero rows hold ``+0.0``, which is what ``0 * t`` gave for a
    finite ``t >= 0``, so the sum differs from the old one only where that was NaN.
    """
    mass = np.asarray(weights, dtype=np.float64)
    carried = mass != 0.0
    if carried.all():
        return float(np.sum(weights * row_term(*rows)))
    terms = np.zeros_like(mass)
    terms[carried] = mass[carried] * row_term(
        *(np.asarray(row, dtype=np.float64)[carried] for row in rows)
    )
    return float(np.sum(terms))


def _ulp_size(values: NDArray) -> NDArray:
    """Return the local float64 spacing without overflowing at max-float."""
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        upward = np.nextafter(magnitude, np.inf) - magnitude
        downward = magnitude - np.nextafter(magnitude, 0.0)
    return np.where(np.isfinite(upward), np.maximum(upward, downward), downward)


def _explained_deviance(
    deviance: float,
    null_deviance: float,
    y: NDArray,
    null_mu: NDArray,
    weights: NDArray,
) -> float:
    """Return explained deviance with an ulp-scale exact-null convention.

    A mathematically constant response can acquire a positive null deviance
    solely because its floating-point weighted mean differs by one or two
    representable values. Treat that null model as exact only when the active
    response values are exactly constant and each agrees with its null
    prediction within eight local ulps. Requiring exact response constancy
    preserves genuine adjacent-float variation at every physical scale.
    """
    if null_deviance <= 0.0:
        return 0.0

    observed = np.asarray(y, dtype=np.float64)
    baseline = np.asarray(null_mu, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if observed.shape != baseline.shape or observed.shape != sample_weight.shape:
        raise ValueError("explained-deviance arrays must have identical shapes")

    active = sample_weight > 0.0
    if not np.any(active):
        return 0.0
    active_observed = observed[active]
    if not np.all(active_observed == active_observed[0]):
        return float(1.0 - deviance / null_deviance)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(observed - baseline)
    tolerance = 8.0 * np.maximum(_ulp_size(observed), _ulp_size(baseline))
    numerically_exact = bool(
        np.all(np.isfinite(difference[active])) and np.all(difference[active] <= tolerance[active])
    )
    if numerically_exact:
        return 0.0
    return float(1.0 - deviance / null_deviance)
