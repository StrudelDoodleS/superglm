"""Vectorized compound-Poisson normalizer used by Tweedie exact density."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

_SERIES_RTOL = 5.0e-15
_SERIES_MAX_TERMS = 100_000


def tweedie_log_series(
    log_t: NDArray,
    a: float,
    *,
    rtol: float = _SERIES_RTOL,
    max_terms: int = _SERIES_MAX_TERMS,
) -> tuple[NDArray, NDArray]:
    """Return log(sum terms) and E[J] for t**j/(j! Gamma(a*j)), j >= 1."""
    values = np.asarray(log_t, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise FloatingPointError("Tweedie exact series requires finite one-dimensional log(t)")
    if not np.isfinite(a) or a <= 0.0:
        raise FloatingPointError("Tweedie exact series requires finite a > 0")

    log_ratio_at_limit = (
        values - np.log(max_terms + 1.0) - gammaln(a * (max_terms + 1.0)) + gammaln(a * max_terms)
    )
    if np.any(log_ratio_at_limit >= 0.0):
        raise FloatingPointError("Tweedie exact series mode lies beyond the configured term limit")

    log_sum = np.full(values.shape, -np.inf, dtype=np.float64)
    log_first_moment = np.full(values.shape, -np.inf, dtype=np.float64)
    active = np.ones(values.shape, dtype=np.bool_)

    for j in range(1, max_terms + 1):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        active_log_t = values[indices]
        log_term = j * active_log_t - gammaln(j + 1.0) - gammaln(a * j)
        log_sum[indices] = np.logaddexp(log_sum[indices], log_term)
        log_first_moment[indices] = np.logaddexp(
            log_first_moment[indices],
            np.log(float(j)) + log_term,
        )

        log_ratio = active_log_t - np.log(j + 1.0) - gammaln(a * (j + 1.0)) + gammaln(a * j)
        declining = log_ratio < 0.0
        if not np.any(declining):
            continue

        declining_indices = indices[declining]
        declining_log_ratio = log_ratio[declining]
        ratio = np.exp(declining_log_ratio)
        log_one_minus_ratio = np.log1p(-ratio)
        log_next = log_term[declining] + declining_log_ratio
        log_mass_tail = log_next - log_one_minus_ratio
        log_moment_factor = np.logaddexp(
            np.log(j + 1.0) - log_one_minus_ratio,
            declining_log_ratio - 2.0 * log_one_minus_ratio,
        )
        log_moment_tail = log_next + log_moment_factor
        done = (log_mass_tail <= log_sum[declining_indices] + np.log(rtol)) & (
            log_moment_tail <= log_first_moment[declining_indices] + np.log(rtol)
        )
        active[declining_indices[done]] = False

    if np.any(active):
        raise FloatingPointError(
            "Tweedie exact series did not converge for "
            f"{int(np.count_nonzero(active))} row(s) within {max_terms} terms"
        )

    expected_j = np.exp(log_first_moment - log_sum)
    if not np.all(np.isfinite(log_sum)) or not np.all(np.isfinite(expected_j)):
        raise FloatingPointError("Tweedie exact series produced a non-finite result")
    return log_sum, expected_j
