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
# Covers peak indices up to ~3e9 (a + 1); stage 0 of the rebuild found no real row
# needing more (benchmarks/tweedie_nb_rebuild_receipts.json, decision a).
MAX_ROW_TERMS = 1_000_000
# lgamma(j + 1) + lgamma(a j) is shared by every row of one call; cap its size.
_TABLE_LIMIT = 1 << 20


@njit(cache=True)
def _term(j: int, log_t: float, a: float, log_base: NDArray) -> float:
    if j < log_base.size:
        return j * log_t - log_base[j]
    return j * log_t - (math.lgamma(j + 1.0) + math.lgamma(a * j))


@njit(cache=True)
def _climb(log_t: float, a: float, mode: int, log_base: NDArray):
    """(mode, peak term) reached by climbing from the estimated mode.

    The terms are log-concave in j, so the climb reaches the peak. It matters
    near p = 1: a is large there and one step off the peak already takes
    exp(q - peak) out of range. Two directions times a climb is the algorithm
    itself, hence the nested loop.
    """
    peak = _term(mode, log_t, a, log_base)
    for direction in (1, -1):
        while mode + direction >= 1:
            neighbour = _term(mode + direction, log_t, a, log_base)
            if not neighbour > peak:
                break
            mode += direction
            peak = neighbour
    return mode, peak


@njit(cache=True)
def _row_moments(log_t: float, a: float, mode: int, log_base: NDArray):
    """(ok, log W, E[J], Var[J]) for one row, summed outward from the peak term.

    Two directions times a term walk is the algorithm itself, hence the nested loop.
    """
    mode, peak = _climb(log_t, a, mode, log_base)
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
def _series_moments_kernel(log_t, a, ok, log_w, mean_j, var_j) -> None:
    a_plus_one = a + 1.0
    a_log_a = a * math.log(a)
    log_safe_mode = math.log(_MAX_SAFE_MODE)
    modes = np.zeros(log_t.size, dtype=np.int64)  # 0 marks a row past the work bound
    table_size = 64
    for row in range(log_t.size):
        log_mode = (log_t[row] - a_log_a) / a_plus_one
        if log_mode > log_safe_mode:
            continue
        mode = math.exp(log_mode)
        radius = math.sqrt(2.0 * _LOG_CUTOFF * mode / a_plus_one)
        if 2.0 * radius >= MAX_ROW_TERMS:
            continue
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
    log_t: NDArray, a: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per row: whether the series evaluated, log W, E[J] and Var[J]."""
    log_t = np.ascontiguousarray(log_t, dtype=np.float64)
    ok = np.empty(log_t.size, dtype=np.bool_)
    log_w = np.empty(log_t.size, dtype=np.float64)
    mean_j = np.empty(log_t.size, dtype=np.float64)
    var_j = np.empty(log_t.size, dtype=np.float64)
    _series_moments_kernel(log_t, float(a), ok, log_w, mean_j, var_j)
    return ok, log_w, mean_j, var_j


@njit(cache=True)
def _digamma_positive(value: float) -> float:
    """Return digamma(value) for a finite positive scalar."""
    result = 0.0
    x = value
    while x < 12.0:
        result -= 1.0 / x
        x += 1.0
    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    correction = inverse_squared * (
        1.0 / 12.0
        - inverse_squared
        * (
            1.0 / 120.0
            - inverse_squared
            * (
                1.0 / 252.0
                - inverse_squared
                * (
                    1.0 / 240.0
                    - inverse_squared * (5.0 / 660.0 - inverse_squared * (691.0 / 32760.0))
                )
            )
        )
    )
    return result + math.log(x) - 0.5 * inverse - correction


@njit(cache=True)
def _digamma_trigamma_positive(value: float) -> tuple[float, float]:
    """Return digamma(value) and trigamma(value) with one recurrence."""
    digamma_result = 0.0
    trigamma_result = 0.0
    x = value
    while x < 12.0:
        inverse = 1.0 / x
        digamma_result -= inverse
        trigamma_result += 1.0 / (x * x)
        x += 1.0

    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    digamma_correction = inverse_squared * (
        1.0 / 12.0
        - inverse_squared
        * (
            1.0 / 120.0
            - inverse_squared
            * (
                1.0 / 252.0
                - inverse_squared
                * (
                    1.0 / 240.0
                    - inverse_squared * (5.0 / 660.0 - inverse_squared * (691.0 / 32760.0))
                )
            )
        )
    )
    digamma = digamma_result + math.log(x) - 0.5 * inverse - digamma_correction

    trigamma_tail = inverse + 0.5 * inverse_squared
    trigamma_tail += (
        inverse
        * inverse_squared
        * (
            1.0 / 6.0
            - inverse_squared
            * (
                1.0 / 30.0
                - inverse_squared
                * (
                    1.0 / 42.0
                    - inverse_squared
                    * (
                        1.0 / 30.0
                        - inverse_squared * (5.0 / 66.0 - inverse_squared * (691.0 / 2730.0))
                    )
                )
            )
        )
    )
    return digamma, trigamma_result + trigamma_tail


@njit(cache=True)
def _term_p_channels(j: int, log_t_p: float, log_t_pp: float, inverse_r: float, order: int):
    """d q_j / d log phi, d q_j / dp, d2 q_j / d log phi dp and d2 q_j / dp2.

    q_j = j log t - lgamma(j + 1) - lgamma(a j) with a = 1/r - 1, so da/dp = -1/r^2
    and d2a/dp2 = 2/r^3; order one needs no trigamma.
    """
    j_float = float(j)
    inverse_r2 = inverse_r * inverse_r
    if order == 1:
        digamma = _digamma_positive((inverse_r - 1.0) * j_float)
        return -j_float * inverse_r, j_float * (log_t_p + digamma * inverse_r2), math.nan, math.nan
    digamma, trigamma = _digamma_trigamma_positive((inverse_r - 1.0) * j_float)
    q_pp = (
        j_float * log_t_pp
        - j_float * j_float * trigamma * inverse_r2 * inverse_r2
        - 2.0 * j_float * digamma * inverse_r2 * inverse_r
    )
    return (
        -j_float * inverse_r,
        j_float * (log_t_p + digamma * inverse_r2),
        j_float * inverse_r2,
        q_pp,
    )


@njit(cache=True)
def _centred_update(moments, relative: float, mass: float, channels, anchor, order: int):
    """Add one term to peak-anchored weighted means and co-moments (weighted Welford)."""
    mean_rho, mean_p, mean_rho_p, mean_pp, var_rho, cov, var_p = moments
    ratio = relative / mass
    centred_rho = channels[0] - anchor[0]
    centred_p = channels[1] - anchor[1]
    delta_rho = centred_rho - mean_rho
    delta_p = centred_p - mean_p
    mean_rho += ratio * delta_rho
    mean_p += ratio * delta_p
    if order == 2:
        var_rho += relative * delta_rho * (centred_rho - mean_rho)
        cov += relative * delta_rho * (centred_p - mean_p)
        var_p += relative * delta_p * (centred_p - mean_p)
        mean_rho_p += ratio * ((channels[2] - anchor[2]) - mean_rho_p)
        mean_pp += ratio * ((channels[3] - anchor[3]) - mean_pp)
    return mean_rho, mean_p, mean_rho_p, mean_pp, var_rho, cov, var_p


# Refusal codes of row_p_moments; the LSS point kernel reports them as its own.
SERIES_MODE_RANGE = 2
SERIES_MAX_TERMS = 4
_NO_TABLE = np.empty(0, dtype=np.float64)


@njit(cache=True)
def _p_failure(status: int):
    nan = math.nan
    return status, nan, nan, nan, nan, nan, nan, nan, nan, 0


@njit(cache=True)
def row_p_moments(log_t, a, log_t_p, log_t_pp, inverse_r, order, max_terms, log_cutoff):
    """One row with its own power: (status, log W, E and Cov of the term log-derivatives, terms).

    The tuple is (status, log W, E[q_rho], E[q_p], E[q_rho,p], E[q_pp], Var[q_rho],
    Cov[q_rho, q_p], Var[q_p], terms) under weights proportional to exp(q_j), rho =
    log phi. Unrequested orders stay NaN and never evaluate their special functions.
    Two directions times a term walk is the algorithm itself, hence the nested loop.
    """
    log_mode = (log_t - a * math.log(a)) / (a + 1.0)
    if log_mode > math.log(_MAX_SAFE_MODE):
        return _p_failure(SERIES_MODE_RANGE)
    mode, peak = _climb(log_t, a, max(1, int(math.floor(math.exp(log_mode)))), _NO_TABLE)
    anchor = (math.nan, math.nan, math.nan, math.nan)
    if order >= 1:
        anchor = _term_p_channels(mode, log_t_p, log_t_pp, inverse_r, order)
    moments = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mass, terms = 1.0, 1
    for direction in (-1, 1):
        j = mode + direction
        while j >= 1:
            if terms >= max_terms:
                return _p_failure(SERIES_MAX_TERMS)
            if j >= _MAX_SAFE_MODE:
                return _p_failure(SERIES_MODE_RANGE)
            relative_log = _term(j, log_t, a, _NO_TABLE) - peak
            relative = math.exp(relative_log)
            mass += relative
            terms += 1
            if order >= 1:
                channels = _term_p_channels(j, log_t_p, log_t_pp, inverse_r, order)
                moments = _centred_update(moments, relative, mass, channels, anchor, order)
            if relative_log <= -log_cutoff:
                break
            j += direction
    mean_rho, mean_p, mean_rho_p, mean_pp, var_rho, cov, var_p = moments
    return (
        0,
        peak + math.log(mass),
        anchor[0] + mean_rho,
        anchor[1] + mean_p,
        anchor[2] + mean_rho_p if order == 2 else math.nan,
        anchor[3] + mean_pp if order == 2 else math.nan,
        var_rho / mass if order == 2 else math.nan,
        cov / mass if order == 2 else math.nan,
        var_p / mass if order == 2 else math.nan,
        terms,
    )


def warmup() -> None:
    """Compile the kernel (called by superglm.warmup)."""
    series_moments(np.array([-1.0, 2.0]), 1.0)
