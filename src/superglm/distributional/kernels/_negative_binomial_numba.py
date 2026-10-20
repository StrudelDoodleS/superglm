"""Compiled near-Poisson theta series for the NB2 row kernel.

One row is the same float64 operation sequence, in the same order, as the
per-row reference loop it replaces (kept as ``tests/_nb_series_reference.py``):
every product, quotient and sum is written in the reference's left-to-right
order, no fast-math flag is set (so no FMA contraction or reassociation), and
every power the reference takes with ``**`` is a libm ``pow`` call here too.
The exponents arrive as data (``exponents[k] == float(k)``) rather than as
literals: LLVM rewrites ``pow(x, 2.0)`` with a literal exponent as ``x * x``,
which need not equal libm's ``pow`` in the last bit, and a constant trip count
would let it unroll the series loop into literal exponents. Rows are
independent, so the parallel core gives the serial core's bits at any thread
count.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange  # type: ignore[import-untyped]
from numpy.typing import NDArray

_EPS = 2.220446049250313e-16
_TINY = 5e-324
_PARALLEL_BLOCK = 4096


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
        errors[power] = _gamma_bound(float(4 * power + 8)) * absolute / denominator + _TINY


@njit(cache=True)
def _theta_series_row(
    row: int,
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
    sums: NDArray[np.float64],
    sum_errors: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> None:
    count = counts[row]
    row_mean = mean[row]
    row_theta = theta[row]
    scale = candidate_scale[row]
    x = scale / row_theta
    polynomial = 0.0
    absolute = 0.0
    coefficient_error_absolute = 0.0
    weighted = 0.0
    transformed = 0.0
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
                + _TINY
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

    natural[row, 0] = x * x * polynomial
    direct[row, 0] = scale * x * polynomial
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
        + _TINY
    )
    total = row_mean + row_theta
    recurrence_score_scale = (
        count / total * ((row_mean + shifted) / row_theta) + remainder_scale[row]
    )
    recurrence_score_error = (
        _gamma_bound(float(count + 64)) * recurrence_score_scale + remainder_error[row]
    )
    use[row, 0] = series_score_error < recurrence_score_error
    if derivative_order == 1:
        return

    two = exponents[2]
    natural[row, 1] = -(x * x / row_theta) * weighted
    direct[row, 1] = scale * x * transformed
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
        + _TINY
    )
    recurrence_hessian_scale = (
        recurrence_score_scale * (1.0 / row_theta + 1.0 / total)
        + (row_mean / total) * mean_ratio / total
    )
    use[row, 1] = series_hessian_error < (
        _gamma_bound(float(count + 80)) * recurrence_hessian_scale + _TINY
    )


@njit(cache=True)
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
        _theta_series_row(
            rows[index],
            counts,
            mean,
            theta,
            candidate_scale,
            remainder_error,
            remainder_scale,
            derivative_order,
            series_order,
            exponents,
            bernoulli,
            use,
            natural,
            direct,
            sums,
            sum_errors,
            coefficients,
        )


@njit(cache=True, parallel=True)
def theta_series_parallel(
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
    """The serial core's rows in blocks across threads; each row's arithmetic is unchanged."""
    n_blocks = (rows.shape[0] + _PARALLEL_BLOCK - 1) // _PARALLEL_BLOCK
    for chunk in prange(n_blocks):  # ty: ignore[not-iterable] -- Numba loop primitive
        sums = np.empty(series_order)
        sum_errors = np.empty(series_order)
        coefficients = np.empty(series_order)
        stop = min((chunk + 1) * _PARALLEL_BLOCK, rows.shape[0])
        for index in range(chunk * _PARALLEL_BLOCK, stop):
            _theta_series_row(
                rows[index],
                counts,
                mean,
                theta,
                candidate_scale,
                remainder_error,
                remainder_scale,
                derivative_order,
                series_order,
                exponents,
                bernoulli,
                use,
                natural,
                direct,
                sums,
                sum_errors,
                coefficients,
            )


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
                    + _TINY
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
            + _TINY
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


@njit(cache=True, parallel=True)
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
    forms.  Rows are independent, so every thread count gives the same bits.
    """
    n_rows = counts.shape[0]
    third = np.empty((n_rows, 4))
    error = np.empty(n_rows)
    used_series = np.zeros(n_rows, dtype=np.bool_)
    n_blocks = (n_rows + _PARALLEL_BLOCK - 1) // _PARALLEL_BLOCK
    for chunk in prange(n_blocks):  # ty: ignore[not-iterable] -- Numba loop primitive
        sums = np.empty(series_order)
        sum_errors = np.empty(series_order)
        coefficients = np.empty(series_order)
        stop = min((chunk + 1) * _PARALLEL_BLOCK, n_rows)
        for row in range(chunk * _PARALLEL_BLOCK, stop):
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
                    + _TINY
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
            + _TINY
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


@njit(cache=True, parallel=True)
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
    independent, so every thread count gives the same bits.
    """
    n_rows = counts.shape[0]
    fourth = np.empty(n_rows)
    error = np.empty(n_rows)
    n_blocks = (n_rows + _PARALLEL_BLOCK - 1) // _PARALLEL_BLOCK
    for chunk in prange(n_blocks):  # ty: ignore[not-iterable] -- Numba loop primitive
        sums = np.empty(series_order)
        sum_errors = np.empty(series_order)
        coefficients = np.empty(series_order)
        stop = min((chunk + 1) * _PARALLEL_BLOCK, n_rows)
        for row in range(chunk * _PARALLEL_BLOCK, stop):
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
    return fourth, error


# Cody-Waite split of log(2): the high part has trailing zero bits, so an
# integer multiple of it below 2**20 is exact.
_LN2_HIGH = 0.6931471803691238
_LN2_LOW = 1.9082149292705877e-10
_UNIT_ROUNDOFF = 1.1102230246251565e-16
_RESCALE_ABOVE = 2.0**300
_RESCALE_BY = 2.0**-300


@njit(cache=True)
def _size_information_row(mean: float, size: float, max_terms: int) -> tuple[float, int]:
    """``theta**2 I_theta_theta`` of one NB2 law, and the series terms it took.

    ``I = (1/theta) sum_{j>=1} P(Y > j) j / ((theta + j)**2 (theta + j - 1))``,
    which equals Lawless's (1987, eq. 2.8) ``sum_j P(Y > j)/(theta + j)**2 -
    mu/(theta (theta + mu))`` but has only non-negative terms, so the Poisson
    limit, where both of Lawless's sums approach ``mu/theta**2``, does not
    cancel. Summed by parts over ``y`` it is ``sum_{y>=2} P(y) C(y-1)``,
    ``C(k) = sum_{j=1..k} c_j``, ``c_j = j theta/((theta+j)**2 (theta+j-1))``,
    which one forward pass of the pmf recurrence accumulates.

    The identity is summation by parts with the NB2 recurrence ``(y + 1)
    P(y + 1) = p (theta + y) P(y)``, which gives ``E[Y/(theta + Y - 1)] = p``,
    ``p = mu/(mu + theta)``.

    The tail past ``y`` is at most ``P(y) (C(y-1) r/(1-r) + r/(4 (theta+y-1)
    (1-r)**2))``: every later pmf ratio is at most ``r = p max(1, (theta+y)/
    (y+1))`` and ``c_j <= 1/(4 (theta+j-1))``. The series stops once that bound
    is at most ``u`` times the partial sum, so truncation adds at most ``u``
    relative error to the terms' own rounding. ``P(y)`` is carried as ``v 2**e``
    (power-of-two rescaling is exact), so a ``P(0)`` below the normal range
    does not underflow. Returns ``terms = -1`` when ``max_terms`` runs out
    first.
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


@njit(cache=True, parallel=True)
def expected_size_information(
    mean: NDArray[np.float64],
    size: NDArray[np.float64],
    max_terms: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """``theta**2 I_theta_theta`` row by row, with each row's series term count.

    Rows are independent and each is one fixed operation sequence, so every
    thread count gives the same bits.
    """
    n_rows = mean.shape[0]
    information = np.empty(n_rows)
    terms = np.empty(n_rows, dtype=np.int64)
    n_blocks = (n_rows + _PARALLEL_BLOCK - 1) // _PARALLEL_BLOCK
    for chunk in prange(n_blocks):  # ty: ignore[not-iterable] -- Numba loop primitive
        stop = min((chunk + 1) * _PARALLEL_BLOCK, n_rows)
        for row in range(chunk * _PARALLEL_BLOCK, stop):
            information[row], terms[row] = _size_information_row(mean[row], size[row], max_terms)
    return information, terms


__all__ = [
    "expected_size_information",
    "log_fourth_derivative",
    "log_third_derivatives",
    "theta_series_parallel",
    "theta_series_serial",
]
