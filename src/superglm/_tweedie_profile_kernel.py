"""Compiled exact sufficient statistics for Tweedie profile likelihoods."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

PROFILE_KERNEL_OK = 0
PROFILE_KERNEL_WORK_LIMIT = 1
PROFILE_KERNEL_UNSAFE_MODE = 2
PROFILE_KERNEL_NONFINITE = 3

_LOG_CUTOFF = 37.0
_MAX_SAFE_MODE = float(2**52)
_DEFAULT_MAX_TERMS = 100_000
_DEFAULT_MAX_TOTAL_TERMS = 1_000_000


@dataclass(frozen=True)
class ExactProfileStatistics:
    """Aggregate mean-NLL derivatives from one exact compound-Poisson sweep."""

    status: int
    nll: float
    gradient_p: float
    gradient_log_phi: float
    hessian_pp: float
    hessian_log_phi_log_phi: float
    hessian_p_log_phi: float
    n_positive: int
    n_terms: int


@njit(cache=True)
def _digamma_positive(value: float) -> float:
    """Return digamma(value) for a finite positive scalar."""
    if not math.isfinite(value) or value <= 0.0:
        return math.nan
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
def _trigamma_positive(value: float) -> float:
    """Return trigamma(value) for a finite positive scalar."""
    if not math.isfinite(value) or value <= 0.0:
        return math.nan
    result = 0.0
    x = value
    while x < 12.0:
        result += 1.0 / (x * x)
        x += 1.0
    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    tail = inverse + 0.5 * inverse_squared
    tail += (
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
    return result + tail


@njit(cache=True)
def _log_series_term(j: int, log_t: float, a: float) -> float:
    j_float = float(j)
    return j_float * log_t - math.lgamma(j_float + 1.0) - math.lgamma(a * j_float)


@njit(cache=True)
def _series_term_derivatives(
    j: int,
    log_t: float,
    log_t_p: float,
    log_t_pp: float,
    inverse_r: float,
) -> tuple[float, float, float, float, float]:
    """Return q, q_p, q_u, q_pp, and q_pu for one series term."""
    j_float = float(j)
    a = inverse_r - 1.0
    aj = a * j_float
    digamma_aj = _digamma_positive(aj)
    trigamma_aj = _trigamma_positive(aj)
    inverse_r_squared = inverse_r * inverse_r
    inverse_r_cubed = inverse_r_squared * inverse_r
    q = j_float * log_t - math.lgamma(j_float + 1.0) - math.lgamma(aj)
    q_p = j_float * (log_t_p + digamma_aj * inverse_r_squared)
    q_u = -j_float * inverse_r
    q_pp = (
        j_float * log_t_pp
        - trigamma_aj * j_float * j_float * inverse_r_squared * inverse_r_squared
        - 2.0 * digamma_aj * j_float * inverse_r_cubed
    )
    q_pu = j_float * inverse_r_squared
    return q, q_p, q_u, q_pp, q_pu


@njit(cache=True)
def _series_term_is_finite(term) -> bool:
    return (
        math.isfinite(term[0])
        and math.isfinite(term[1])
        and math.isfinite(term[2])
        and math.isfinite(term[3])
        and math.isfinite(term[4])
    )


@njit(cache=True)
def _failure_tuple(status: int, n_positive: int, n_terms: int):
    return (
        status,
        math.inf,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        n_positive,
        n_terms,
    )


@njit(cache=True)
def _exact_profile_statistics_kernel(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64],
    p: float,
    log_phi: float,
    max_terms: int,
    max_total_terms: int,
):
    n_observations = y.size
    r = p - 1.0
    s = 2.0 - p
    inverse_r = 1.0 / r
    a = inverse_r - 1.0
    phi_inverse = math.exp(-log_phi)
    if (
        not math.isfinite(phi_inverse)
        or not math.isfinite(a)
        or a <= 0.0
        or max_terms < 3
        or max_total_terms <= 0
    ):
        return _failure_tuple(PROFILE_KERNEL_NONFINITE, 0, 0)

    log_likelihood = 0.0
    likelihood_p = 0.0
    likelihood_u = 0.0
    likelihood_pp = 0.0
    likelihood_uu = 0.0
    likelihood_pu = 0.0
    n_positive = 0
    total_terms = 0

    inverse_r_squared = inverse_r * inverse_r
    inverse_r_cubed = inverse_r_squared * inverse_r
    inverse_s = 1.0 / s
    inverse_s_squared = inverse_s * inverse_s
    inverse_s_cubed = inverse_s_squared * inverse_s

    for row in range(n_observations):
        response = y[row]
        mean = mu[row]
        weight = weights[row]
        log_mean = math.log(mean)
        effective_inverse_phi = weight * phi_inverse
        mean_to_s = math.exp(s * log_mean)
        if not math.isfinite(mean_to_s) or not math.isfinite(effective_inverse_phi):
            return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)

        if response == 0.0:
            zero_base = mean_to_s * inverse_s
            zero_base_p = mean_to_s * (-log_mean * inverse_s + inverse_s_squared)
            zero_base_pp = mean_to_s * (
                log_mean * log_mean * inverse_s
                - 2.0 * log_mean * inverse_s_squared
                + 2.0 * inverse_s_cubed
            )
            row_log_likelihood = -effective_inverse_phi * zero_base
            row_p = -effective_inverse_phi * zero_base_p
            row_u = effective_inverse_phi * zero_base
            row_pp = -effective_inverse_phi * zero_base_pp
            row_uu = row_log_likelihood
            row_pu = effective_inverse_phi * zero_base_p
        else:
            n_positive += 1
            log_response = math.log(response)
            log_weight = math.log(weight)
            log_r = math.log(r)
            log_t = -a * (log_r - log_response) - math.log(s) + inverse_r * (log_weight - log_phi)
            term_difference = log_r - log_response - log_weight + log_phi
            log_t_p = (term_difference - s) * inverse_r_squared + inverse_s
            log_t_pp = (2.0 * (-term_difference) + 2.0 + s) * inverse_r_cubed + inverse_s_squared
            log_mode = (log_t - a * math.log(a)) / (a + 1.0)
            if not math.isfinite(log_mode) or log_mode > math.log(_MAX_SAFE_MODE):
                return _failure_tuple(PROFILE_KERNEL_UNSAFE_MODE, n_positive, total_terms)
            mode_value = math.exp(log_mode)
            if not math.isfinite(mode_value):
                return _failure_tuple(PROFILE_KERNEL_UNSAFE_MODE, n_positive, total_terms)
            mode = max(1, int(math.floor(mode_value)))

            estimated_radius = int(math.ceil(math.sqrt(2.0 * _LOG_CUTOFF * mode / (a + 1.0)) + 2.0))
            estimated_terms = min(mode, estimated_radius) + estimated_radius + 1
            if estimated_terms > max_terms or total_terms + estimated_terms > max_total_terms:
                return _failure_tuple(PROFILE_KERNEL_WORK_LIMIT, n_positive, total_terms)

            peak = _log_series_term(mode, log_t, a)
            if mode > 1:
                below = _log_series_term(mode - 1, log_t, a)
                if below > peak:
                    mode -= 1
                    peak = below
            above = _log_series_term(mode + 1, log_t, a)
            if above > peak:
                mode += 1
                peak = above

            if not math.isfinite(peak):
                return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)
            anchor = _series_term_derivatives(
                mode,
                log_t,
                log_t_p,
                log_t_pp,
                inverse_r,
            )
            if not _series_term_is_finite(anchor):
                return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)
            _, anchor_p, anchor_u, anchor_pp, anchor_pu = anchor

            mass = 1.0
            delta_p_mass = 0.0
            delta_u_mass = 0.0
            delta_p_squared_mass = 0.0
            delta_u_squared_mass = 0.0
            delta_pu_mass = 0.0
            q_pp_mass = anchor_pp
            q_pu_mass = anchor_pu
            row_terms = 1
            total_terms += 1

            j = mode + 1
            while True:
                if row_terms >= max_terms or total_terms >= max_total_terms:
                    return _failure_tuple(PROFILE_KERNEL_WORK_LIMIT, n_positive, total_terms)
                term = _series_term_derivatives(
                    j,
                    log_t,
                    log_t_p,
                    log_t_pp,
                    inverse_r,
                )
                if not _series_term_is_finite(term):
                    return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)
                q, q_p, q_u, q_pp, q_pu = term
                relative = math.exp(q - peak)
                delta_p = q_p - anchor_p
                delta_u = q_u - anchor_u
                mass += relative
                delta_p_mass += relative * delta_p
                delta_u_mass += relative * delta_u
                delta_p_squared_mass += relative * delta_p * delta_p
                delta_u_squared_mass += relative * delta_u * delta_u
                delta_pu_mass += relative * delta_p * delta_u
                q_pp_mass += relative * q_pp
                q_pu_mass += relative * q_pu
                row_terms += 1
                total_terms += 1
                if q <= peak - _LOG_CUTOFF:
                    break
                j += 1

            j = mode - 1
            while j >= 1:
                if row_terms >= max_terms or total_terms >= max_total_terms:
                    return _failure_tuple(PROFILE_KERNEL_WORK_LIMIT, n_positive, total_terms)
                term = _series_term_derivatives(
                    j,
                    log_t,
                    log_t_p,
                    log_t_pp,
                    inverse_r,
                )
                if not _series_term_is_finite(term):
                    return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)
                q, q_p, q_u, q_pp, q_pu = term
                relative = math.exp(q - peak)
                delta_p = q_p - anchor_p
                delta_u = q_u - anchor_u
                mass += relative
                delta_p_mass += relative * delta_p
                delta_u_mass += relative * delta_u
                delta_p_squared_mass += relative * delta_p * delta_p
                delta_u_squared_mass += relative * delta_u * delta_u
                delta_pu_mass += relative * delta_p * delta_u
                q_pp_mass += relative * q_pp
                q_pu_mass += relative * q_pu
                row_terms += 1
                total_terms += 1
                if q <= peak - _LOG_CUTOFF:
                    break
                j -= 1

            if not math.isfinite(mass) or mass <= 0.0:
                return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)
            mean_delta_p = delta_p_mass / mass
            mean_delta_u = delta_u_mass / mass
            series_p = anchor_p + mean_delta_p
            series_u = anchor_u + mean_delta_u
            series_pp = q_pp_mass / mass + (
                delta_p_squared_mass / mass - mean_delta_p * mean_delta_p
            )
            series_uu = delta_u_squared_mass / mass - mean_delta_u * mean_delta_u
            series_pu = q_pu_mass / mass + (delta_pu_mass / mass - mean_delta_p * mean_delta_u)
            log_sum = peak + math.log(mass)

            mean_to_minus_r = math.exp(-r * log_mean)
            first_base = response * mean_to_minus_r
            canonical = -first_base * inverse_r - mean_to_s * inverse_s
            canonical_p = first_base * (log_mean * inverse_r + inverse_r_squared) + mean_to_s * (
                log_mean * inverse_s - inverse_s_squared
            )
            canonical_pp = first_base * (
                -log_mean * log_mean * inverse_r
                - 2.0 * log_mean * inverse_r_squared
                - 2.0 * inverse_r_cubed
            ) + mean_to_s * (
                -log_mean * log_mean * inverse_s
                + 2.0 * log_mean * inverse_s_squared
                - 2.0 * inverse_s_cubed
            )

            row_log_likelihood = log_sum - log_response + effective_inverse_phi * canonical
            row_p = series_p + effective_inverse_phi * canonical_p
            row_u = series_u - effective_inverse_phi * canonical
            row_pp = series_pp + effective_inverse_phi * canonical_pp
            row_uu = series_uu + effective_inverse_phi * canonical
            row_pu = series_pu - effective_inverse_phi * canonical_p

        if not (
            math.isfinite(row_log_likelihood)
            and math.isfinite(row_p)
            and math.isfinite(row_u)
            and math.isfinite(row_pp)
            and math.isfinite(row_uu)
            and math.isfinite(row_pu)
        ):
            return _failure_tuple(PROFILE_KERNEL_NONFINITE, n_positive, total_terms)
        log_likelihood += row_log_likelihood
        likelihood_p += row_p
        likelihood_u += row_u
        likelihood_pp += row_pp
        likelihood_uu += row_uu
        likelihood_pu += row_pu

    inverse_n = 1.0 / n_observations
    return (
        PROFILE_KERNEL_OK,
        -log_likelihood * inverse_n,
        -likelihood_p * inverse_n,
        -likelihood_u * inverse_n,
        -likelihood_pp * inverse_n,
        -likelihood_uu * inverse_n,
        -likelihood_pu * inverse_n,
        n_positive,
        total_terms,
    )


def _warmup_tweedie_profile() -> None:
    _digamma_positive(1.0)
    _trigamma_positive(1.0)
    _log_series_term(1, 0.0, 1.0)
    term = _series_term_derivatives(1, 0.0, 0.0, 0.0, 2.0)
    _series_term_is_finite(term)
    _failure_tuple(PROFILE_KERNEL_WORK_LIMIT, 0, 0)
    arrays = (
        np.array([0.0, 1.0], dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
    )
    for _ in range(2):
        raw = _exact_profile_statistics_kernel(
            *arrays, 1.5, 0.0, _DEFAULT_MAX_TERMS, _DEFAULT_MAX_TOTAL_TERMS
        )
        status = int(raw[0])
        if status != PROFILE_KERNEL_OK:
            raise RuntimeError(f"Tweedie profile kernel warmup returned status {status}")
        for values in arrays:
            values.setflags(write=False)


def exact_profile_statistics(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    p: float,
    log_phi: float,
    *,
    max_terms: int = _DEFAULT_MAX_TERMS,
    max_total_terms: int = _DEFAULT_MAX_TOTAL_TERMS,
) -> ExactProfileStatistics:
    """Return exact aggregate profile statistics or a bounded failure status."""
    y_array = np.ascontiguousarray(y, dtype=np.float64)
    mu_array = np.ascontiguousarray(mu, dtype=np.float64)
    weight_array = np.ascontiguousarray(weights, dtype=np.float64)
    if y_array.ndim != 1 or y_array.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    if mu_array.shape != y_array.shape or weight_array.shape != y_array.shape:
        raise ValueError("mu and weights must match y")
    if (
        not np.all(np.isfinite(y_array))
        or np.any(y_array < 0.0)
        or not np.all(np.isfinite(mu_array))
        or np.any(mu_array <= 0.0)
        or not np.all(np.isfinite(weight_array))
        or np.any(weight_array <= 0.0)
    ):
        raise ValueError("Tweedie profile arrays must be finite with y >= 0, mu > 0, weights > 0")
    p_float = float(p)
    log_phi_float = float(log_phi)
    if not np.isfinite(p_float) or not 1.0 < p_float < 2.0:
        raise ValueError("p must be finite and strictly between 1 and 2")
    if not np.isfinite(log_phi_float):
        raise ValueError("log_phi must be finite")
    if max_terms < 0 or max_total_terms < 0:
        raise ValueError("series work limits must be non-negative")

    return _exact_profile_statistics_prevalidated(
        y_array,
        mu_array,
        weight_array,
        p_float,
        log_phi_float,
        max_terms=int(max_terms),
        max_total_terms=int(max_total_terms),
    )


def _exact_profile_statistics_prevalidated(
    y: NDArray[np.float64],
    mu: NDArray[np.float64],
    weights: NDArray[np.float64],
    p: float,
    log_phi: float,
    *,
    max_terms: int = _DEFAULT_MAX_TERMS,
    max_total_terms: int = _DEFAULT_MAX_TOTAL_TERMS,
) -> ExactProfileStatistics:
    """Call the compiled kernel for already validated contiguous float arrays."""
    raw = _exact_profile_statistics_kernel(
        y,
        mu,
        weights,
        p,
        log_phi,
        max_terms,
        max_total_terms,
    )
    return ExactProfileStatistics(
        status=int(raw[0]),
        nll=float(raw[1]),
        gradient_p=float(raw[2]),
        gradient_log_phi=float(raw[3]),
        hessian_pp=float(raw[4]),
        hessian_log_phi_log_phi=float(raw[5]),
        hessian_p_log_phi=float(raw[6]),
        n_positive=int(raw[7]),
        n_terms=int(raw[8]),
    )
