"""Compiled Dunn-Smyth (2005) series for the Tweedie density, 1 < p < 2.

A positive response has density W(t) / y * exp(c w / phi) with
W(t) = sum_{j>=1} t^j / (j! Gamma(a j)) and a = (2 - p) / (p - 1). Every term
is positive, so starting at the peak term and summing outward until a term
falls _LOG_CUTOFF below it evaluates log W to machine accuracy; only the term
count grows, about 2 sqrt(2 * 37 * j_max / (a + 1)).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Terms 37 log-units below the peak change log W by < 1e-16 (Dunn & Smyth 2005).
_LOG_CUTOFF = 37.0
# Beyond 2**52 consecutive integers are no longer exact in float64.
_MAX_SAFE_MODE = float(2**52)
# A cap on one row's work: peak indices up to ~3.4e9 (a + 1). The density's
# accuracy switch to the saddlepoint (superglm._tweedie.saddlepoint_switch) sits
# below it for every p < 2 - 2e-9, so on that path it binds only nearer p = 2.
MAX_ROW_TERMS = 1_000_000
# lgamma(j + 1) + lgamma(a j) is shared by every row of one call. The accuracy
# switch bounds it by 1.2-1.7 j* entries; this cap binds only where j* > 9e5,
# p > 1.9999.
_TABLE_LIMIT = 1 << 20


@njit(cache=True)
def _term(j: int, log_t: float, a: float, log_base: NDArray) -> float:
    if j < log_base.size:
        return j * log_t - log_base[j]
    return j * log_t - (math.lgamma(j + 1.0) + math.lgamma(a * j))


@njit(cache=True)
def _row_moments(log_t: float, a: float, mode: int, log_base: NDArray):
    """(ok, log W, E[J], Var[J]) for one row, summed outward from the peak term.

    The terms are log-concave in j, so climbing from the estimated mode reaches
    the peak. The climb matters near p = 1: a is large there and one step off
    the peak already takes exp(q - peak) out of range. Two directions times a
    term walk is the algorithm itself, hence the nested loop.
    """
    peak = _term(mode, log_t, a, log_base)
    for direction in (1, -1):
        while mode + direction >= 1:
            neighbour = _term(mode + direction, log_t, a, log_base)
            if not neighbour > peak:
                break
            mode += direction
            peak = neighbour
    mass, first, second, n_terms = 1.0, 0.0, 0.0, 1
    for direction in (1, -1):
        j = mode + direction
        while j >= 1:
            if n_terms >= MAX_ROW_TERMS:
                return False, math.nan, math.nan, math.nan
            q = _term(j, log_t, a, log_base)
            relative = math.exp(q - peak)
            offset = float(j - mode)
            mass += relative
            first += relative * offset
            second += relative * offset * offset
            n_terms += 1
            if q <= peak - _LOG_CUTOFF:
                break
            j += direction
    mean_offset = first / mass
    variance = second / mass - mean_offset * mean_offset
    log_w = peak + math.log(mass)
    if not (math.isfinite(log_w) and math.isfinite(variance)):
        return False, math.nan, math.nan, math.nan
    return True, log_w, mode + mean_offset, variance


@njit(cache=True)
def _series_moments_kernel(log_t, a, log_max_mode, ok, log_w, mean_j, var_j) -> None:
    a_plus_one = a + 1.0
    a_log_a = a * math.log(a)
    modes = np.zeros(log_t.size, dtype=np.int64)  # 0 marks a row past log_max_mode
    table_size = 64
    for row in range(log_t.size):
        log_mode = (log_t[row] - a_log_a) / a_plus_one
        if log_mode > log_max_mode:
            continue
        mode = math.exp(log_mode)
        radius = math.sqrt(2.0 * _LOG_CUTOFF * mode / a_plus_one)
        modes[row] = max(1, int(math.floor(mode)))
        # A row's window is mode +- radius; one whose window sits entirely above
        # the table limit reads nothing from the table and does not size it.
        if mode - 4.0 * radius - 64.0 < _TABLE_LIMIT:
            table_size = max(table_size, int(min(mode + 4.0 * radius + 64.0, _TABLE_LIMIT)))
    log_base = np.empty(table_size, dtype=np.float64)
    log_base[0] = 0.0
    for j in range(1, table_size):
        log_base[j] = math.lgamma(j + 1.0) + math.lgamma(a * j)
    for row in range(log_t.size):
        if modes[row] == 0:
            ok[row], log_w[row], mean_j[row], var_j[row] = False, math.nan, math.nan, math.nan
            continue
        ok[row], log_w[row], mean_j[row], var_j[row] = _row_moments(
            log_t[row], a, modes[row], log_base
        )


def series_moments(
    log_t: NDArray, a: float, max_mode: float = math.inf
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per row: whether the series evaluated, log W, E[J] and Var[J].

    A row is refused when its peak index passes ``max_mode``, 2**52 or the work
    bound, where its term window 2 sqrt(2 * 37 j / (a + 1)) reaches MAX_ROW_TERMS.
    """
    work_bound = (0.5 * MAX_ROW_TERMS) ** 2 * (a + 1.0) / (2.0 * _LOG_CUTOFF)
    log_max_mode = math.log(min(max_mode, work_bound, _MAX_SAFE_MODE))
    log_t = np.ascontiguousarray(log_t, dtype=np.float64)
    ok = np.empty(log_t.size, dtype=np.bool_)
    log_w = np.empty(log_t.size, dtype=np.float64)
    mean_j = np.empty(log_t.size, dtype=np.float64)
    var_j = np.empty(log_t.size, dtype=np.float64)
    _series_moments_kernel(log_t, float(a), log_max_mode, ok, log_w, mean_j, var_j)
    return ok, log_w, mean_j, var_j


def warmup() -> None:
    """Compile the kernel (called by superglm.warmup)."""
    series_moments(np.array([-1.0, 2.0]), 1.0)
