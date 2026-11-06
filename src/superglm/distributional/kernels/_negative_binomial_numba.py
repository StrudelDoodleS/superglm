"""Compiled NB2 rows: the near-Poisson theta series, and the complete row evaluation.

Each row runs the same float64 operation sequence, in the same order, as the
reference loop in ``tests/_nb_series_reference.py``: left-to-right products and
sums, no fast-math (so no FMA or reassociation), and libm ``pow`` wherever the
reference uses ``**``. Exponents arrive as data (``exponents[k] == float(k)``)
because LLVM rewrites ``pow(x, 2.0)`` with a literal exponent as ``x * x`` and
would unroll a constant trip count.

Kernels are serial and release the GIL; ``superglm._parallel.pooled_row_blocks``
runs them over fixed blocks of ``_PARALLEL_BLOCK`` rows with private scratch,
so every worker count gives the same bits.

``evaluate_rows`` is ``evaluate_negative_binomial_rows``'s array code one row at a
time: the same domain tests, formulas and certificates, with the recurrence summed
in blocks of ``RECURRENCE_BLOCK`` terms, each left to right (error at most
``gamma_(RECURRENCE_BLOCK - 1) sum |t_i|``, Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., chapter 4), and the ``B = ceil(n / RECURRENCE_BLOCK)``
block sums added by cascaded summation with ``TwoSum`` (Ogita, Rump and Oishi, *Accurate
Sum and Dot Product*, SIAM J. Sci. Comput. 26(6), 2005, Algorithm 4.1; error at most
``u |sum| + gamma_(B - 1)^2 sum |p_i|``, Proposition 4.5, in their ``eps = u``).  A long
sum, which the remainder terms added after it cancel, thus keeps an error that no longer
grows with the count: plain left-to-right sums would give ``gamma_(n - 1) sum |t_i|``.  A
row of at most ``RECURRENCE_BLOCK`` terms is one block, and its sums are as a plain loop
gives them.  The kernel writes only rows it can certify, and answers any other row with
``KERNEL_DECLINED``, which the caller meets by running its array route, whose refusals are
the package's.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm._parallel import pooled_row_blocks
from superglm._roundoff import EPS as _EPS
from superglm._roundoff import SMALLEST_SUBNORMAL as _SUBNORMAL
from superglm._roundoff import UNIT_ROUNDOFF as _UNIT_ROUNDOFF
from superglm._roundoff import gamma as _gamma

# Rows per parallel block: large enough to amortise waking the workers. Rows
# are independent, so the block size changes no bit.
_PARALLEL_BLOCK = 1024

# Int64 values, not Python ints: numba would type each Python int constant as its own literal.
KERNEL_OK = np.int64(0)
KERNEL_DECLINED = np.int64(1)

# The effective parameters the row evaluation accepts, and the Poisson limit
# beyond which a row is evaluated at a surrogate size; the array route reads them from here.
DOMAIN_MIN = math.ldexp(1.0, -450)
DOMAIN_MAX = math.ldexp(1.0, 450)
POISSON_LIMIT_RATIO = _EPS
POISSON_SURROGATE_EXPONENT = 60
# Terms of the recurrence summed left to right before the block joins the running total.
RECURRENCE_BLOCK = 256
# The fixed-order series replaces the recurrence where it beats it by its bound.
SERIES_ORDER = 12
SERIES_RATIO = 1.0 / 16.0
# ``x/(1+x) - log1p(x)`` by Horner's rule: coefficients of ``x^2 .. x^SERIES_ORDER``.
_REMAINDER_COEFFICIENTS = tuple(
    (-1.0) ** (order + 1) * (order - 1.0) / order for order in range(2, SERIES_ORDER + 1)
)
_GAMMA_8 = _gamma(8)
_GAMMA_SERIES = _gamma(2 * (2 * SERIES_ORDER + 2))


@njit(cache=True)
def _gamma_bound(operations: float) -> float:
    """``n eps / (1 - n eps)``, the kernel's ``_gamma`` (written in ``eps = 2u``)."""
    scaled = operations * _EPS
    return scaled / (1.0 - scaled)


@njit(cache=True)
def _scaled_power_sums(
    count: int,
    bernoulli: NDArray[np.float64],
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> None:
    """Normalized float64 power sums of ``0..count-1`` and their roundoff bounds, in place."""
    order = values.shape[0]
    for power in range(order):
        values[power] = 0.0
        errors[power] = 0.0
    if count < 2:
        values[0] = float(count)
        return
    inverse_count = 1.0 / float(count)
    for power in range(order):
        coefficients[power] = 0.0
    for power in range(order):
        choose = 1.0
        for term in range(power + 1):
            coefficients[term] = choose * bernoulli[term]
            if term < power:
                choose *= float(power + 1 - term) / float(term + 1)
        polynomial = coefficients[power]
        absolute = abs(polynomial)
        for term in range(power - 1, -1, -1):
            polynomial = coefficients[term] + inverse_count * polynomial
            absolute = abs(coefficients[term]) + inverse_count * absolute
        denominator = float(power + 1)
        values[power] = polynomial / denominator
        errors[power] = _gamma_bound(float(4 * power + 8)) * absolute / denominator + _SUBNORMAL


@njit(cache=True)
def _series_row(
    count: int,
    row_mean: float,
    row_theta: float,
    scale: float,
    remainder_error: float,
    remainder_scale: float,
    derivative_order: int,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    sums: NDArray[np.float64],
    sum_errors: NDArray[np.float64],
    coefficients: NDArray[np.float64],
):
    """One row's series: ``(use_score, use_hessian, natural_score, natural_hessian,
    direct_score, direct_hessian)``, the Hessian entries read only at order two.

    The scratch is touched only for counts above one.
    """
    x = scale / row_theta
    polynomial = 0.0
    absolute = 0.0
    coefficient_error_absolute = 0.0
    weighted = 0.0
    transformed = 0.0
    weighted_absolute = 0.0
    weighted_error = 0.0
    if count > 1:  # the power sums are read only for counts above one
        _scaled_power_sums(count, bernoulli, sums, sum_errors, coefficients)
    count_scale = float(count) / scale if count > 1 else 0.0
    for order in range(series_order, 1, -1):
        if count > 1:
            offset_factor = math.pow(count_scale, exponents[order])
            offset = sums[order - 1] * offset_factor
            offset_error = (
                abs(offset_factor) * sum_errors[order - 1]
                + _gamma_bound(float(order + 2)) * abs(offset)
                + _SUBNORMAL
            )
        else:
            offset = 0.0
            offset_error = 0.0
        mean_term = math.pow(row_mean / scale, exponents[order - 1]) * (
            ((order - 1.0) / order * row_mean - count) / scale
        )
        sign = -1.0 if order % 2 == 0 else 1.0
        coefficient = sign * (offset + mean_term)
        coefficient_abs = abs(offset) + abs(mean_term)
        polynomial = coefficient + x * polynomial
        absolute = coefficient_abs + x * absolute
        coefficient_error_absolute = offset_error + x * coefficient_error_absolute
        if derivative_order == 2:
            weighted = order * coefficient + x * weighted
            transformed = (1.0 - order) * coefficient + x * transformed
            weighted_absolute = order * coefficient_abs + x * weighted_absolute
            weighted_error = order * offset_error + x * weighted_error

    natural_score = x * x * polynomial
    direct_score = scale * x * polynomial
    order_value = exponents[series_order]
    next_order_value = exponents[series_order + 1]
    mean_ratio = row_mean / row_theta
    shifted = count - 1 if count > 1 else 0
    offset_ratio = shifted / row_theta
    delta = abs(row_mean - count)
    score_tail = (
        count / row_theta * math.pow(offset_ratio, order_value) / (1.0 - offset_ratio)
        + math.pow(mean_ratio, next_order_value) / next_order_value
        + delta / row_theta * math.pow(mean_ratio, order_value) / (1.0 - mean_ratio)
    )
    series_score_error = (
        x
        * x
        * (_gamma_bound(80.0) * absolute + (1.0 + _gamma_bound(32.0)) * coefficient_error_absolute)
        + score_tail
        + _SUBNORMAL
    )
    total = row_mean + row_theta
    recurrence_score_scale = count / total * ((row_mean + shifted) / row_theta) + remainder_scale
    recurrence_score_error = (
        _gamma_bound(float(count + 64)) * recurrence_score_scale + remainder_error
    )
    use_score = series_score_error < recurrence_score_error
    if derivative_order == 1:
        return use_score, False, natural_score, 0.0, direct_score, 0.0

    two = exponents[2]
    natural_hessian = -(x * x / row_theta) * weighted
    direct_hessian = scale * x * transformed
    hessian_tail = (
        count
        / math.pow(row_theta, two)
        * math.pow(offset_ratio, order_value)
        * (next_order_value - series_order * offset_ratio)
        / math.pow(1.0 - offset_ratio, two)
        + math.pow(mean_ratio, next_order_value) / row_theta / (1.0 - mean_ratio)
        + delta
        / math.pow(row_theta, two)
        * math.pow(mean_ratio, order_value)
        * (next_order_value - series_order * mean_ratio)
        / math.pow(1.0 - mean_ratio, two)
    )
    series_hessian_error = (
        x
        * x
        / row_theta
        * (_gamma_bound(96.0) * weighted_absolute + (1.0 + _gamma_bound(32.0)) * weighted_error)
        + hessian_tail
        + _SUBNORMAL
    )
    recurrence_hessian_scale = (
        recurrence_score_scale * (1.0 / row_theta + 1.0 / total)
        + (row_mean / total) * mean_ratio / total
    )
    use_hessian = series_hessian_error < (
        _gamma_bound(float(count + 80)) * recurrence_hessian_scale + _SUBNORMAL
    )
    return (
        use_score,
        use_hessian,
        natural_score,
        natural_hessian,
        direct_score,
        direct_hessian,
    )


@njit(cache=True, nogil=True)
def theta_series_serial(
    rows: NDArray[np.int64],
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    candidate_scale: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    derivative_order: int,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    use: NDArray[np.bool_],
    natural: NDArray[np.float64],
    direct: NDArray[np.float64],
) -> None:
    """Fill ``use``, ``natural`` and ``direct`` on the candidate ``rows``, one row at a time."""
    sums = np.empty(series_order)
    sum_errors = np.empty(series_order)
    coefficients = np.empty(series_order)
    for index in range(rows.shape[0]):
        row = rows[index]
        (
            use_score,
            use_hessian,
            natural_score,
            natural_hessian,
            direct_score,
            direct_hessian,
        ) = _series_row(
            counts[row],
            mean[row],
            theta[row],
            candidate_scale[row],
            remainder_error[row],
            remainder_scale[row],
            derivative_order,
            series_order,
            exponents,
            bernoulli,
            sums,
            sum_errors,
            coefficients,
        )
        natural[row, 0] = natural_score
        direct[row, 0] = direct_score
        use[row, 0] = use_score
        if derivative_order == 2:
            natural[row, 1] = natural_hessian
            direct[row, 1] = direct_hessian
            use[row, 1] = use_hessian


@njit(cache=True)
def _tail_moment_one(ratio: float, start: float) -> float:
    """``sum_{m >= M} m r^m = r^M (M/(1-r) + r/(1-r)^2)`` for ``0 <= r < 1``."""
    rest = 1.0 - ratio
    return math.pow(ratio, start) * (start / rest + ratio / (rest * rest))


@njit(cache=True)
def _tail_moment_two(ratio: float, start: float) -> float:
    """``sum_{m >= M} m^2 r^m = r^M (M^2/(1-r) + 2M r/(1-r)^2 + r(1+r)/(1-r)^3)``."""
    rest = 1.0 - ratio
    return math.pow(ratio, start) * (
        start * start / rest
        + 2.0 * start * ratio / (rest * rest)
        + ratio * (1.0 + ratio) / (rest * rest * rest)
    )


@njit(cache=True)
def _log_third_row(
    row: int,
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    series_ratio: float,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    third: NDArray[np.float64],
    error: NDArray[np.float64],
    used_series: NDArray[np.bool_],
    sums: NDArray[np.float64],
    sum_errors: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> None:
    count = counts[row]
    row_mean = mean[row]
    row_theta = theta[row]
    total = row_mean + row_theta
    q = row_mean / total
    p = row_theta / total
    r = (row_theta - row_mean) / total
    s = q * p
    # Closed forms in (a, b) = (log mean, log theta); each is a product of
    # singly rounded quotients and differences of the exact inputs.
    third[row, 0] = -(count + row_theta) * r * s
    third[row, 1] = s * (count * r - 2.0 * row_mean * p)
    third[row, 2] = -(count - row_mean) * r * s
    shifted = count - 1 if count > 1 else 0

    # Recurrence: l_bbb = sum_j alpha_j (mu - j)/t [(q - alpha_j)^2 - alpha_j beta_j - p q]
    # + theta (R + 2 q^3), alpha_j = theta/(theta + j), beta_j = j/(theta + j).  The
    # cheap a-priori scale uses alpha <= 1, alpha beta <= 1/4, |mu - j| <= mu + y - 1.
    bracket_scale = (1.0 + q) * (1.0 + q) + 0.25 + p * q
    recurrence_scale = count * ((row_mean + shifted) / total) * bracket_scale + row_theta * (
        remainder_scale[row] + 2.0 * q * q * q
    )
    recurrence_error = _gamma_bound(float(count + 32)) * recurrence_scale + (
        1.0 + _gamma_bound(4.0)
    ) * (row_theta * remainder_error[row])

    candidate = max(row_mean, float(shifted)) / row_theta <= series_ratio
    series_value = 0.0
    series_error = math.inf
    if candidate:
        scale = row_mean if row_mean >= shifted else float(shifted)
        x = scale / row_theta
        weighted = 0.0
        weighted_absolute = 0.0
        weighted_error = 0.0
        _scaled_power_sums(count, bernoulli, sums, sum_errors, coefficients)
        count_scale = float(count) / scale if count > 1 else 0.0
        for order in range(series_order, 1, -1):
            if count > 1:
                offset_factor = math.pow(count_scale, exponents[order])
                offset = sums[order - 1] * offset_factor
                offset_error = (
                    abs(offset_factor) * sum_errors[order - 1]
                    + _gamma_bound(float(order + 2)) * abs(offset)
                    + _SUBNORMAL
                )
            else:
                offset = 0.0
                offset_error = 0.0
            power = math.pow(row_mean / scale, exponents[order - 1])
            mean_term = power * (((order - 1.0) / order * row_mean - count) / scale)
            mean_scale = power * (((order - 1.0) / order * row_mean + count) / scale)
            sign = -1.0 if order % 2 == 0 else 1.0
            coefficient = sign * (offset + mean_term)
            factor = float((order - 1) * (order - 1))
            weighted = factor * coefficient + x * weighted
            weighted_absolute = factor * (abs(offset) + mean_scale) + x * weighted_absolute
            weighted_error = factor * offset_error + x * weighted_error
        lead = scale * x
        mean_ratio = row_mean / row_theta
        offset_ratio = shifted / row_theta
        start = exponents[series_order]
        tail = (
            count * _tail_moment_two(offset_ratio, start)
            + row_mean * _tail_moment_one(mean_ratio, start)
            + abs(row_mean - count) * _tail_moment_two(mean_ratio, start)
        )
        series_value = lead * weighted
        series_error = (
            lead
            * (_gamma_bound(64.0) * weighted_absolute + (1.0 + _gamma_bound(32.0)) * weighted_error)
            + tail
            + _SUBNORMAL
        )

    if series_error < recurrence_error:
        third[row, 3] = series_value
        error[row] = (1.0 + _gamma_bound(64.0)) * series_error
        used_series[row] = True
        return
    value = 0.0
    for j in range(count):
        offset_value = float(j)
        shifted_theta = row_theta + offset_value
        alpha = row_theta / shifted_theta
        beta = offset_value / shifted_theta
        difference = q - alpha
        bracket = difference * difference - alpha * beta - p * q
        value += alpha * ((row_mean - offset_value) / total) * bracket
    value += row_theta * (remainder[row] + 2.0 * q * q * q)
    third[row, 3] = value
    error[row] = (1.0 + _gamma_bound(64.0)) * recurrence_error
    used_series[row] = False


@njit(cache=True, nogil=True)
def _log_third_rows(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    series_ratio: float,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    start: int,
    stop: int,
    third: NDArray[np.float64],
    error: NDArray[np.float64],
    used_series: NDArray[np.bool_],
) -> None:
    """``log_third_derivatives`` on rows ``start..stop``, into the caller's outputs."""
    sums = np.empty(series_order)
    sum_errors = np.empty(series_order)
    coefficients = np.empty(series_order)
    for row in range(start, stop):
        _log_third_row(
            row,
            counts,
            mean,
            theta,
            remainder,
            remainder_error,
            remainder_scale,
            series_ratio,
            series_order,
            exponents,
            bernoulli,
            third,
            error,
            used_series,
            sums,
            sum_errors,
            coefficients,
        )


def log_third_derivatives(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    series_ratio: float,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Unit-mass ``(l_aaa, l_aab, l_abb, l_bbb)`` in log coordinates, with ``l_bbb``'s error bound.

    ``l_bbb`` takes the near-Poisson series or the count recurrence, whichever
    certifies the smaller absolute error; the other three channels are closed
    forms.  Rows are independent, so every worker count gives the same bits.
    """
    n_rows = counts.shape[0]
    third = np.empty((n_rows, 4))
    error = np.empty(n_rows)
    used_series = np.zeros(n_rows, dtype=np.bool_)
    pooled_row_blocks(
        lambda start, stop: _log_third_rows(
            counts, mean, theta, remainder, remainder_error, remainder_scale, series_ratio,
            series_order, exponents, bernoulli, start, stop, third, error, used_series,
        ),
        n_rows,
        _PARALLEL_BLOCK,
    )  # fmt: skip
    return third, error, used_series


@njit(cache=True)
def _tail_moment_three(ratio: float, start: float) -> float:
    """``sum_{m >= M} m^3 r^m = r^M (M^3/(1-r) + 3M^2 r/(1-r)^2 + 3M r(1+r)/(1-r)^3 + r(1+4r+r^2)/(1-r)^4)``."""
    rest = 1.0 - ratio
    return math.pow(ratio, start) * (
        start * start * start / rest
        + 3.0 * start * start * ratio / (rest * rest)
        + 3.0 * start * ratio * (1.0 + ratio) / (rest * rest * rest)
        + ratio * (1.0 + 4.0 * ratio + ratio * ratio) / (rest * rest * rest * rest)
    )


@njit(cache=True)
def _log_fourth_row(
    row: int,
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    series_ratio: float,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    fourth: NDArray[np.float64],
    error: NDArray[np.float64],
    sums: NDArray[np.float64],
    sum_errors: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> None:
    count = counts[row]
    row_mean = mean[row]
    row_theta = theta[row]
    total = row_mean + row_theta
    q = row_mean / total
    p = row_theta / total
    shifted = count - 1 if count > 1 else 0

    # Recurrence: l_bbbb = sum_j T_j [D^3 - 3 D E - E'] + theta (R + q^2 - 4 q^3 + 6 q^4)
    # with T_j = alpha_j (mu - j)/t, D = q - alpha_j, E = alpha_j beta_j + p q and
    # E' = alpha_j beta_j (beta_j - alpha_j) + p q (q - p), one log-theta derivative
    # beyond the third's T_j [D^2 - E].  |D| <= 1 + q, E <= 1/4 + p q, |E'| <= 1/4 + p q.
    bracket_scale = (1.0 + q) ** 3 + 3.0 * (1.0 + q) * (0.25 + p * q) + 0.25 + p * q
    remainder_polynomial = q * q * (1.0 + 4.0 * q + 6.0 * q * q)
    recurrence_scale = count * ((row_mean + shifted) / total) * bracket_scale + row_theta * (
        remainder_scale[row] + remainder_polynomial
    )
    recurrence_error = _gamma_bound(float(count + 40)) * recurrence_scale + (
        1.0 + _gamma_bound(4.0)
    ) * (row_theta * remainder_error[row])

    candidate = max(row_mean, float(shifted)) / row_theta <= series_ratio
    series_value = 0.0
    series_error = math.inf
    if candidate:
        scale = row_mean if row_mean >= shifted else float(shifted)
        x = scale / row_theta
        weighted = 0.0
        weighted_absolute = 0.0
        weighted_error = 0.0
        _scaled_power_sums(count, bernoulli, sums, sum_errors, coefficients)
        count_scale = float(count) / scale if count > 1 else 0.0
        for order in range(series_order, 1, -1):
            if count > 1:
                offset_factor = math.pow(count_scale, exponents[order])
                offset = sums[order - 1] * offset_factor
                offset_error = (
                    abs(offset_factor) * sum_errors[order - 1]
                    + _gamma_bound(float(order + 2)) * abs(offset)
                    + _SUBNORMAL
                )
            else:
                offset = 0.0
                offset_error = 0.0
            power = math.pow(row_mean / scale, exponents[order - 1])
            mean_term = power * (((order - 1.0) / order * row_mean - count) / scale)
            mean_scale = power * (((order - 1.0) / order * row_mean + count) / scale)
            sign = -1.0 if order % 2 == 0 else 1.0
            coefficient = sign * (offset + mean_term)
            factor = float((order - 1) * (order - 1) * (order - 1))
            weighted = -factor * coefficient + x * weighted
            weighted_absolute = factor * (abs(offset) + mean_scale) + x * weighted_absolute
            weighted_error = factor * offset_error + x * weighted_error
        lead = scale * x
        mean_ratio = row_mean / row_theta
        offset_ratio = shifted / row_theta
        start = exponents[series_order]
        tail = (
            count * _tail_moment_three(offset_ratio, start)
            + row_mean * _tail_moment_two(mean_ratio, start)
            + abs(row_mean - count) * _tail_moment_three(mean_ratio, start)
        )
        series_value = lead * weighted
        series_error = (
            lead
            * (_gamma_bound(64.0) * weighted_absolute + (1.0 + _gamma_bound(32.0)) * weighted_error)
            + tail
            + _SUBNORMAL
        )

    if series_error < recurrence_error:
        fourth[row] = series_value
        error[row] = (1.0 + _gamma_bound(64.0)) * series_error
        return
    value = 0.0
    pq = p * q
    pq_slope = pq * (q - p)
    for j in range(count):
        offset_value = float(j)
        shifted_theta = row_theta + offset_value
        alpha = row_theta / shifted_theta
        beta = offset_value / shifted_theta
        d = q - alpha
        e = alpha * beta + pq
        e_slope = alpha * beta * (beta - alpha) + pq_slope
        value += alpha * ((row_mean - offset_value) / total) * (d * d * d - 3.0 * d * e - e_slope)
    value += row_theta * (remainder[row] + q * q * (1.0 - 4.0 * q + 6.0 * q * q))
    fourth[row] = value
    error[row] = (1.0 + _gamma_bound(64.0)) * recurrence_error


@njit(cache=True, nogil=True)
def _log_fourth_rows(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    series_ratio: float,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    start: int,
    stop: int,
    fourth: NDArray[np.float64],
    error: NDArray[np.float64],
) -> None:
    """``log_fourth_derivative`` on rows ``start..stop``, into the caller's outputs."""
    sums = np.empty(series_order)
    sum_errors = np.empty(series_order)
    coefficients = np.empty(series_order)
    for row in range(start, stop):
        _log_fourth_row(
            row,
            counts,
            mean,
            theta,
            remainder,
            remainder_error,
            remainder_scale,
            series_ratio,
            series_order,
            exponents,
            bernoulli,
            fourth,
            error,
            sums,
            sum_errors,
            coefficients,
        )


def log_fourth_derivative(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    series_ratio: float,
    series_order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit-mass ``l_bbbb`` in ``b = log theta`` with its error bound, row by row.

    The near-Poisson series of the theta score differentiated three times more
    in ``log theta`` (coefficients ``(1 - n)^3 c_n``) or the count recurrence,
    whichever bounds the smaller error, as ``l_bbb`` chooses.  Rows are
    independent, so every worker count gives the same bits.
    """
    n_rows = counts.shape[0]
    fourth = np.empty(n_rows)
    error = np.empty(n_rows)
    pooled_row_blocks(
        lambda start, stop: _log_fourth_rows(
            counts, mean, theta, remainder, remainder_error, remainder_scale, series_ratio,
            series_order, exponents, bernoulli, start, stop, fourth, error,
        ),
        n_rows,
        _PARALLEL_BLOCK,
    )  # fmt: skip
    return fourth, error


# Cody-Waite split of log(2): the high part has trailing zero bits, so an
# integer multiple of it below 2**20 is exact.
_LN2_HIGH = 0.6931471803691238
_LN2_LOW = 1.9082149292705877e-10
_RESCALE_ABOVE = 2.0**300
_RESCALE_BY = 2.0**-300


@njit(cache=True)
def _size_information_row(mean: float, size: float, max_terms: int) -> tuple[float, int]:
    """``theta**2 I_theta_theta`` of one NB2 law, and the series terms it took.

    ``I = (1/theta) sum_{j>=1} P(Y > j) j / ((theta + j)**2 (theta + j - 1))``, Lawless's
    (1987, eq. 2.8) form with non-negative terms so the Poisson limit does not cancel. Summed
    by parts it is ``sum_{y>=2} P(y) C(y-1)``, one forward pmf pass, with
    ``C(k) = sum_{j<=k} j theta/((theta+j)**2 (theta+j-1))``. The tail past ``y`` is at most
    ``P(y) (C(y-1) r/(1-r) + r/(4 (theta+y-1) (1-r)**2))``, ``r = p max(1, (theta+y)/(y+1))``;
    the series stops when that is at most ``u`` times the partial sum. ``P(y)`` is carried as
    ``v 2**e`` so it never underflows. ``terms = -1`` when ``max_terms`` runs out.
    """
    p = mean / (mean + size)
    exponent_value = -size * math.log1p(mean / size)
    exponent = 0
    if exponent_value < -700.0:
        exponent = int(math.ceil(exponent_value / 0.6931471805599453))
        exponent_value = (exponent_value - exponent * _LN2_HIGH) - exponent * _LN2_LOW
    value = math.exp(exponent_value)
    cumulative = 0.0
    partial = 0.0
    y = 0
    while y < max_terms:
        y += 1
        # P(y) = P(y - 1) p (theta + y - 1) / y, carried as value * 2**exponent.
        value = value * (p * ((size + (y - 1.0)) / y))
        if value > _RESCALE_ABOVE:
            value *= _RESCALE_BY
            partial *= _RESCALE_BY
            exponent += 300
        if y < 2:
            continue
        j = y - 1.0
        shifted = size + j
        cumulative += j * size / (shifted * shifted * (size + (j - 1.0)))
        partial += value * cumulative
        growth = (size + y) / (y + 1.0)
        ratio = p * growth if growth > 1.0 else p
        if ratio < 1.0:
            spare = 1.0 - ratio
            tail = value * (cumulative * (ratio / spare) + ratio / (4.0 * shifted * spare * spare))
            if tail <= _UNIT_ROUNDOFF * partial:
                return math.ldexp(partial, exponent), y
    return 0.0, -1


@njit(cache=True, nogil=True)
def _size_information_rows(
    mean: NDArray[np.float64],
    size: NDArray[np.float64],
    max_terms: int,
    start: int,
    stop: int,
    information: NDArray[np.float64],
    terms: NDArray[np.int64],
) -> None:
    """``expected_size_information`` on rows ``start..stop``, into the caller's outputs."""
    for row in range(start, stop):
        information[row], terms[row] = _size_information_row(mean[row], size[row], max_terms)


def expected_size_information(
    mean: NDArray[np.float64],
    size: NDArray[np.float64],
    max_terms: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """``theta**2 I_theta_theta`` row by row, with each row's series term count.

    Rows are independent and each is one fixed operation sequence, so every
    worker count gives the same bits.
    """
    n_rows = mean.shape[0]
    information = np.empty(n_rows)
    terms = np.empty(n_rows, dtype=np.int64)
    pooled_row_blocks(
        lambda start, stop: _size_information_rows(
            mean, size, max_terms, start, stop, information, terms
        ),
        n_rows,
        _PARALLEL_BLOCK,
    )
    return information, terms


@njit(cache=True)
def _ratio_remainder_row(ratio: float):
    """``(x/(1+x) - log1p(x), error, scale)`` by the lower-error direct or Horner route."""
    rational = ratio / (1.0 + ratio) if ratio <= 1.0 else 1.0 - 1.0 / (1.0 + ratio)
    logarithm = math.log1p(ratio)
    result = rational - logarithm
    scale = abs(rational) + abs(logarithm)
    error = _GAMMA_8 * scale + _SUBNORMAL
    if ratio < 1.0:
        coefficients = _REMAINDER_COEFFICIENTS
        polynomial = coefficients[-1]
        absolute = abs(coefficients[-1])
        for index in range(len(coefficients) - 2, -1, -1):
            polynomial = coefficients[index] + ratio * polynomial
            absolute = abs(coefficients[index]) + ratio * absolute
        series = ratio * ratio * polynomial
        series_scale = ratio * ratio * absolute
        tail = SERIES_ORDER / (SERIES_ORDER + 1.0) * ratio ** (SERIES_ORDER + 1)
        series_error = _GAMMA_SERIES * series_scale + tail + _SUBNORMAL
        if series_error < error:
            return series, series_error, series_scale
    return result, error, scale


@njit(cache=True)
def _fold(total: float, carry: float, block: float):
    """``total + block`` rounded, and the rounding's error (``TwoSum``) added to ``carry``."""
    folded = total + block
    shift = folded - total
    return folded, carry + ((total - (folded - shift)) + (block - shift))


@njit(cache=True)
def _recurrence_row(count: int, mean: float, theta: float, order: int, derivatives: bool):
    """The recurrence's sums over ``j < count``: ``(value, natural score, natural Hessian,
    direct score, direct Hessian)``; the derivative sums only where ``derivatives``.

    Blocks of ``RECURRENCE_BLOCK`` terms are summed left to right and joined by ``_fold``.
    """
    total = mean + theta
    inverse_total = 1.0 / total
    value = score = hessian = direct_score = direct_hessian = 0.0
    value_carry = score_carry = hessian_carry = direct_score_carry = direct_hessian_carry = 0.0
    for first in range(0, count, RECURRENCE_BLOCK):
        block_value = block_score = block_hessian = 0.0
        block_direct_score = block_direct_hessian = 0.0
        for index in range(first, min(first + RECURRENCE_BLOCK, count)):
            offset = float(index)
            # log1p((j - mean)/total) == log((theta + j)/total); below -1/2 the log1p
            # argument nears -1 and the quotient form is better conditioned.
            argument = (offset - mean) / total
            if argument < -0.5:
                block_value += math.log((theta + offset) / total)
            else:
                block_value += math.log1p(argument)
            if derivatives:
                inverse_offset = 1.0 / (theta + offset)
                score_term = (mean - offset) * inverse_offset * inverse_total
                direct_term = (theta * inverse_offset) * ((mean - offset) / total)
                block_score += score_term
                block_direct_score += direct_term
                if order == 2:
                    block_hessian += -score_term * (inverse_offset + inverse_total)
                    block_direct_hessian += direct_term * (
                        1.0 - theta * inverse_offset - theta * inverse_total
                    )
        value, value_carry = _fold(value, value_carry, block_value)
        if derivatives:
            score, score_carry = _fold(score, score_carry, block_score)
            direct_score, direct_score_carry = _fold(
                direct_score, direct_score_carry, block_direct_score
            )
            if order == 2:
                hessian, hessian_carry = _fold(hessian, hessian_carry, block_hessian)
                direct_hessian, direct_hessian_carry = _fold(
                    direct_hessian, direct_hessian_carry, block_direct_hessian
                )
    return (
        value + value_carry,
        score + score_carry,
        hessian + hessian_carry,
        direct_score + direct_score_carry,
        direct_hessian + direct_hessian_carry,
    )


@njit(cache=True)
def _returns_to(reconstructed: float, target: float, scale: float, bound: float, floor: float):
    """Whether ``reconstructed`` is finite and within ``bound`` of ``target``, in units of the channel's ``scale``."""
    return math.isfinite(reconstructed) and abs(reconstructed - target) <= (
        bound * (scale + abs(target) + abs(reconstructed)) + floor
    )


@njit(cache=True, nogil=True, error_model="numpy")
def evaluate_rows(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    weights: NDArray[np.float64],
    prior: bool,
    order: int,
    exponents: NDArray[np.float64],
    bernoulli: NDArray[np.float64],
    start: int,
    stop: int,
    optimizing: NDArray[np.float64],
    score: NDArray[np.float64],
    hessian: NDArray[np.float64],
):
    """Rows ``start..stop`` of ``evaluate_negative_binomial_rows``: ``(status, first declined row)``.

    ``score`` and ``hessian`` are read at orders one and two.  A row is declined
    where the array route refuses it: a parameter outside ``[DOMAIN_MIN,
    DOMAIN_MAX]``, a Poisson limit outside the range, a non-finite channel, or a
    channel its natural form cannot return to the direct log-link form within
    the roundoff bound.
    """
    sums = np.empty(SERIES_ORDER)
    sum_errors = np.empty(SERIES_ORDER)
    coefficients = np.empty(SERIES_ORDER)
    for row in range(start, stop):
        count = counts[row]
        c = float(count)
        row_mean = mean[row]
        row_theta = theta[row]
        weight = weights[row]
        if not (DOMAIN_MIN <= row_mean <= DOMAIN_MAX and DOMAIN_MIN <= weight <= DOMAIN_MAX):
            return KERNEL_DECLINED, row
        scale = weight if prior else 1.0
        multiplier = 1.0 if prior else weight
        effective_mean = scale * row_mean
        if not DOMAIN_MIN <= effective_mean <= DOMAIN_MAX:
            return KERNEL_DECLINED, row
        reach = max(effective_mean, max(c - 1.0, 0.0))
        effective_theta = scale * row_theta
        poisson = reach < POISSON_LIMIT_RATIO * effective_theta
        if poisson:
            effective_theta = min(math.ldexp(reach, POISSON_SURROGATE_EXPONENT), DOMAIN_MAX)
            if reach >= POISSON_LIMIT_RATIO * effective_theta:
                return KERNEL_DECLINED, row
        elif not DOMAIN_MIN <= row_theta <= DOMAIN_MAX:
            return KERNEL_DECLINED, row
        if not DOMAIN_MIN <= effective_theta <= DOMAIN_MAX:
            return KERNEL_DECLINED, row
        total = effective_mean + effective_theta

        remainder = remainder_error = remainder_scale = 0.0
        use_score = use_hessian = False
        series_natural_score = series_natural_hessian = 0.0
        series_direct_score = series_direct_hessian = 0.0
        derivatives = False
        if order >= 1:
            remainder, remainder_error, remainder_scale = _ratio_remainder_row(
                effective_mean / effective_theta
            )
            if reach / effective_theta <= SERIES_RATIO:
                (
                    use_score,
                    use_hessian,
                    series_natural_score,
                    series_natural_hessian,
                    series_direct_score,
                    series_direct_hessian,
                ) = _series_row(
                    count,
                    effective_mean,
                    effective_theta,
                    reach,
                    remainder_error,
                    remainder_scale,
                    order,
                    SERIES_ORDER,
                    exponents,
                    bernoulli,
                    sums,
                    sum_errors,
                    coefficients,
                )
            derivatives = not (use_score and (order == 1 or use_hessian))
        value, natural_score, natural_hessian, direct_score, direct_hessian = _recurrence_row(
            count, effective_mean, effective_theta, order, derivatives
        )
        effective_value = (
            value
            + c * math.log(effective_mean)
            - effective_theta * math.log1p(effective_mean / effective_theta)
        )
        row_value = multiplier * effective_value
        if not math.isfinite(row_value):
            return KERNEL_DECLINED, row
        optimizing[row] = row_value
        if order == 0:
            continue

        theta_score = series_natural_score if use_score else natural_score + remainder
        direct_theta_score = (
            series_direct_score if use_score else direct_score + effective_theta * remainder
        )
        term_scale = c / total * ((effective_mean + max(c - 1.0, 0.0)) / effective_theta)
        direct_theta_score_scale = effective_theta * (term_scale + remainder_scale)
        difference = c - effective_mean
        mean_score = (effective_theta / total) * (difference / effective_mean)
        direct_mean_score = (effective_theta / total) * difference
        factor = multiplier * scale
        score_mean = factor * mean_score
        score_theta = factor * theta_score
        target_mean = multiplier * direct_mean_score
        target_theta = multiplier * direct_theta_score
        mean_score_scale = (effective_theta / total) * (abs(c) + effective_mean)
        scale_mean = multiplier * mean_score_scale
        scale_theta = multiplier * direct_theta_score_scale
        if not (math.isfinite(score_mean) and math.isfinite(score_theta)):
            return KERNEL_DECLINED, row

        hessian_mean = hessian_cross = hessian_theta = 0.0
        if order == 2:
            ratio_mean = effective_mean / total
            ratio_theta = effective_theta / total
            hessian_remainder = (ratio_mean) * (effective_mean / effective_theta) / total
            direct_hessian_remainder = (
                effective_theta * (effective_mean / total) ** 2 + effective_theta * remainder
            )
            theta_hessian = (
                series_natural_hessian if use_hessian else natural_hessian + hessian_remainder
            )
            direct_theta_hessian = (
                series_direct_hessian if use_hessian else direct_hessian + direct_hessian_remainder
            )
            direct_theta_hessian_scale = (
                effective_theta
                * term_scale
                * (1.0 + effective_theta * (1.0 / effective_theta + 1.0 / total))
                + effective_theta * (effective_mean / total) ** 2
                + effective_theta * remainder_scale
            )
            count_over_mean = c / effective_mean
            mean_hessian = -(effective_theta / total) * (
                count_over_mean / effective_mean + (count_over_mean - 1.0) / total
            )
            cross_hessian = difference / (total * total)
            hessian_scale_factor = factor * scale
            hessian_mean = hessian_scale_factor * mean_hessian
            hessian_cross = hessian_scale_factor * cross_hessian
            hessian_theta = hessian_scale_factor * theta_hessian
            direct_mean_hessian = -(ratio_mean * ratio_theta * (c + effective_theta))
            direct_cross_hessian = ratio_mean * ratio_theta * difference
            target_hessian_mean = multiplier * direct_mean_hessian
            target_hessian_cross = multiplier * direct_cross_hessian
            target_hessian_theta = multiplier * direct_theta_hessian
            scale_hessian_mean = multiplier * (abs(direct_mean_hessian) + mean_score_scale)
            scale_hessian_cross = multiplier * abs(direct_cross_hessian)
            scale_hessian_theta = multiplier * (
                direct_theta_hessian_scale + direct_theta_score_scale
            )
            if not (
                math.isfinite(hessian_mean)
                and math.isfinite(hessian_cross)
                and math.isfinite(hessian_theta)
            ):
                return KERNEL_DECLINED, row

        if poisson:  # a Poisson-limit row has no theta channels
            score_theta = 0.0
            target_theta = 0.0
            if order == 2:
                hessian_cross = hessian_theta = 0.0
                target_hessian_cross = target_hessian_theta = 0.0

        # The direct log-link channels the natural channels must return to.
        reconstructed_mean = score_mean * row_mean
        reconstructed_theta = score_theta * row_theta
        bound = _gamma_bound(c + 128.0)
        floor = 16.0 * _SUBNORMAL
        if not (
            _returns_to(reconstructed_mean, target_mean, scale_mean, bound, floor)
            and _returns_to(reconstructed_theta, target_theta, scale_theta, bound, floor)
        ):
            return KERNEL_DECLINED, row
        if order == 2:
            reconstructed_hessian_mean = (
                hessian_mean * row_mean
            ) * row_mean + score_mean * row_mean
            reconstructed_hessian_cross = (hessian_cross * row_mean) * row_theta
            reconstructed_hessian_theta = (
                hessian_theta * row_theta
            ) * row_theta + score_theta * row_theta
            bound = _gamma_bound(c + 160.0)
            floor = 24.0 * _SUBNORMAL
            if not (
                _returns_to(
                    reconstructed_hessian_mean,
                    target_hessian_mean,
                    scale_hessian_mean,
                    bound,
                    floor,
                )
                and _returns_to(
                    reconstructed_hessian_cross,
                    target_hessian_cross,
                    scale_hessian_cross,
                    bound,
                    floor,
                )
                and _returns_to(
                    reconstructed_hessian_theta,
                    target_hessian_theta,
                    scale_hessian_theta,
                    bound,
                    floor,
                )
            ):
                return KERNEL_DECLINED, row
        score[row, 0] = score_mean
        score[row, 1] = score_theta
        if order == 2:
            hessian[row, 0] = hessian_mean
            hessian[row, 1] = hessian_cross
            hessian[row, 2] = hessian_theta
    return KERNEL_OK, np.int64(-1)


# The kernels the worker pool runs (``nogil``, never ``prange``), each writing
# only its own rows' outputs.
_POOLED_ROW_KERNELS = (
    theta_series_serial,
    evaluate_rows,
    _log_third_rows,
    _log_fourth_rows,
    _size_information_rows,
)


__all__ = [
    "evaluate_rows",
    "expected_size_information",
    "log_fourth_derivative",
    "log_third_derivatives",
    "theta_series_serial",
]
