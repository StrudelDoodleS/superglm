"""Vectorized compound-Poisson normalizer used by Tweedie exact density."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

_SERIES_RTOL = 5.0e-15
_SERIES_LOG_CUTOFF = 37.0
_SERIES_MAX_TERMS = 100_000
_SERIES_MAX_TOTAL_TERMS = 1_000_000
_SERIES_BATCH_TERMS = 262_144
_SERIES_MAX_SAFE_MODE = float(2**52)


def _log_series_term(log_t: NDArray, a: float, j: NDArray) -> NDArray:
    j_float = np.asarray(j, dtype=np.float64)
    return j_float * log_t - gammaln(j_float + 1.0) - gammaln(a * j_float)


def _select_budgeted_rows(
    counts: NDArray,
    log_modes: NDArray,
    values: NDArray,
    max_total_terms: int,
) -> NDArray:
    """Select deterministic all-or-none tie groups within the global budget."""
    feasible = np.flatnonzero(counts > 0)
    if feasible.size == 0 or max_total_terms <= 0:
        return np.empty(0, dtype=np.intp)

    order = np.lexsort(
        (
            values[feasible],
            log_modes[feasible],
            counts[feasible],
        )
    )
    ordered = feasible[order]
    ordered_counts = counts[ordered]
    ordered_modes = log_modes[ordered]
    ordered_values = values[ordered]
    group_break = (
        (ordered_counts[1:] != ordered_counts[:-1])
        | (ordered_modes[1:] != ordered_modes[:-1])
        | (ordered_values[1:] != ordered_values[:-1])
    )
    starts = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.flatnonzero(group_break).astype(np.intp) + 1,
        )
    )
    stops = np.concatenate((starts[1:], np.array([ordered.size], dtype=np.intp)))

    selected: list[NDArray] = []
    used = 0
    for start, stop in zip(starts, stops, strict=True):
        group = ordered[start:stop]
        work = int(np.sum(counts[group], dtype=np.int64))
        if used + work <= max_total_terms:
            selected.append(group)
            used += work
    if not selected:
        return np.empty(0, dtype=np.intp)
    return np.concatenate(selected)


def tweedie_log_series(
    log_t: NDArray,
    a: float,
    *,
    rtol: float = _SERIES_RTOL,
    max_terms: int = _SERIES_MAX_TERMS,
    max_total_terms: int = _SERIES_MAX_TOTAL_TERMS,
    batch_terms: int = _SERIES_BATCH_TERMS,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return log mass, E[J], and rows evaluated exactly around their modes."""
    values = np.asarray(log_t, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise FloatingPointError("Tweedie exact series requires finite one-dimensional log(t)")
    if not np.isfinite(a) or a <= 0.0:
        raise FloatingPointError("Tweedie exact series requires finite a > 0")
    if not np.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError("Tweedie exact series requires 0 < rtol < 1")
    if batch_terms <= 0:
        raise ValueError("Tweedie exact series requires batch_terms > 0")

    log_sum = np.full(values.shape, np.nan, dtype=np.float64)
    expected_j = np.full(values.shape, np.nan, dtype=np.float64)
    exact = np.zeros(values.shape, dtype=np.bool_)
    if values.size == 0 or max_terms < 3 or max_total_terms <= 0:
        return log_sum, expected_j, exact

    cutoff = max(_SERIES_LOG_CUTOFF, -float(np.log(rtol)))
    max_mode_from_width = ((max_terms - 1.0) / 2.0) ** 2 * (a + 1.0) / (2.0 * cutoff)
    max_mode = min(max_mode_from_width, _SERIES_MAX_SAFE_MODE)
    if not np.isfinite(max_mode) or max_mode <= 1.0:
        return log_sum, expected_j, exact

    with np.errstate(all="ignore"):
        log_mode = (values - a * np.log(a)) / (a + 1.0)
    candidates = np.flatnonzero(np.isfinite(log_mode) & (log_mode <= np.log(max_mode)))
    if candidates.size == 0:
        return log_sum, expected_j, exact

    guesses = np.maximum(
        1,
        np.floor(np.exp(log_mode[candidates])).astype(np.int64),
    )
    adjacent = np.stack((np.maximum(1, guesses - 1), guesses, guesses + 1))
    adjacent_log_terms = _log_series_term(values[candidates][None, :], a, adjacent)
    selected_peak = np.argmax(adjacent_log_terms, axis=0)
    candidate_columns = np.arange(candidates.size)
    modes = adjacent[selected_peak, candidate_columns]
    peaks = adjacent_log_terms[selected_peak, candidate_columns]

    finite_peak = np.isfinite(peaks)
    candidates = candidates[finite_peak]
    modes = modes[finite_peak]
    peaks = peaks[finite_peak]
    if candidates.size == 0:
        return log_sum, expected_j, exact

    radii = np.maximum(
        2,
        np.ceil(np.sqrt(2.0 * cutoff * modes / (a + 1.0)) + 2.0).astype(np.int64),
    )
    lower = np.maximum(1, modes - radii)
    upper = modes + radii
    bounded = np.zeros(candidates.size, dtype=np.bool_)
    for _ in range(32):
        lower_large = (lower > 1) & (
            _log_series_term(values[candidates], a, lower) > peaks - cutoff
        )
        upper_large = _log_series_term(values[candidates], a, upper) > peaks - cutoff
        expand = lower_large | upper_large
        bounded = ~expand
        if not np.any(expand):
            break
        radii[expand] *= 2
        lower[expand] = np.maximum(1, modes[expand] - radii[expand])
        upper[expand] = modes[expand] + radii[expand]

    counts = upper - lower + 1
    counts[(~bounded) | (counts > max_terms)] = 0
    chosen = _select_budgeted_rows(
        counts,
        log_mode[candidates],
        values[candidates],
        max_total_terms,
    )
    if chosen.size == 0:
        return log_sum, expected_j, exact

    chosen_rows = candidates[chosen]
    exact[chosen_rows] = True
    position = 0
    while position < chosen.size:
        remaining = chosen[position:]
        cumulative = np.cumsum(counts[remaining], dtype=np.int64)
        take = max(1, int(np.searchsorted(cumulative, batch_terms, side="right")))
        local_rows = remaining[:take]
        local_counts = counts[local_rows]
        starts = np.cumsum(local_counts, dtype=np.int64) - local_counts
        total = int(np.sum(local_counts, dtype=np.int64))
        repeated_starts = np.repeat(starts, local_counts)
        j = (
            np.repeat(lower[local_rows], local_counts)
            + np.arange(total, dtype=np.int64)
            - repeated_starts
        )
        repeated_rows = np.repeat(local_rows, local_counts)
        relative = np.exp(
            _log_series_term(values[candidates][repeated_rows], a, j) - peaks[repeated_rows]
        )
        mass = np.add.reduceat(relative, starts)
        moment = np.add.reduceat(relative * j, starts)
        output_rows = candidates[local_rows]
        log_sum[output_rows] = peaks[local_rows] + np.log(mass)
        expected_j[output_rows] = moment / mass
        position += take

    return log_sum, expected_j, exact
