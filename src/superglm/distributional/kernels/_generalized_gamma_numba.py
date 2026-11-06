"""Compiled rows of the Prentice generalized gamma, in both parametrisations.

The port of ``generalized_gamma``'s array rows: the same series, branch
thresholds and expressions, evaluated one row at a time, so the two differ only
by rounding and read side by side. ``error_model="numpy"`` keeps the array
code's IEEE arithmetic: a division by a product that underflowed gives an
infinity or NaN that flags the row, not an exception.

The row kernel is serial and releases the GIL; ``pooled_rows`` runs it over fixed
row blocks and each row writes only its own outputs, so every worker count gives
the same bits. The mean loading and ``log(sigma)`` depend on ``(sigma, Q)`` alone,
and a book has few distinct scales, so a block keeps what it formed in a small
table keyed on the bits of ``sigma``: a row reads the same bits as forming them anew.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.distributional.kernels._tweedie_numba import _digamma_positive

# Digest of the compiled callee above, stamped by ``tools/generate_severity_families.py``:
# Numba's on-disk cache keys a compiled caller on its own file only.
_CALLEE_DIGEST = "4eae5ae0add65ab2f88b8b9f7ad2466fcf605ce46e6a7588e1db21207c7245d7"

# Int64 values, not Python ints: numba would type each Python int constant as its own literal.
KERNEL_OK = np.int64(0)
KERNEL_MEAN_LOADING = np.int64(1)

_STIRLING_SWITCH = 8.0
_U_SERIES_RADIUS = 1.0
_U_SERIES_TERMS = 25
_V_SERIES_RADIUS = 0.25
_V_SERIES_TERMS = 40
_M3_SERIES_RADIUS = 0.8
_M3_SERIES_TERMS = 48
_ZERO_SHAPE = 1.0e-8
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_M3_A_COEFFICIENTS = tuple(1.0 / (2 * j + 3) for j in range(_M3_SERIES_TERMS))
_M3_B_COEFFICIENTS = tuple((2 * j + 2) / (2 * j + 5) for j in range(_M3_SERIES_TERMS))
_BERNOULLI = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
    43867.0 / 244188.0,
    -174611.0 / 125400.0,
)

# Taylor coefficients of the series functions of u = Q w and v = sigma Q.
_R2 = tuple(1.0 / math.factorial(j + 2) for j in range(_U_SERIES_TERMS))
_E1 = tuple(1.0 / math.factorial(j + 1) for j in range(_U_SERIES_TERMS))
_U2 = tuple((j + 1) / math.factorial(j + 2) for j in range(_U_SERIES_TERMS))
_T3 = tuple(-(j + 1) / math.factorial(j + 3) for j in range(_U_SERIES_TERMS))
_T3_D1 = tuple(-(j + 1) * (j + 2) / math.factorial(j + 4) for j in range(_U_SERIES_TERMS))
_L1 = tuple((-1.0) ** j / (j + 1) for j in range(_V_SERIES_TERMS))
_L1_D1 = tuple((-1.0) ** (j + 1) * (j + 1) / (j + 2) for j in range(_V_SERIES_TERMS))
_L2 = tuple((-1.0) ** (j + 1) / (j + 2) for j in range(_V_SERIES_TERMS))

# One row per power of the argument, one column per function: the fused Horner loops run the
# independent chains of a function family side by side, each in the order of a loop alone.
_U_COEFFICIENTS = np.array([_R2, _E1, _T3, _U2, _T3_D1]).T.copy()
_L_COEFFICIENTS = np.array([_L1, _L1_D1, _L2]).T.copy()
_M3_COEFFICIENTS = np.array([_M3_A_COEFFICIENTS, _M3_B_COEFFICIENTS]).T.copy()

# The per-block table of mean loadings: open addressing on a multiplicative hash of the bits of
# sigma, never fuller than three quarters, so a probe always reaches an empty slot.
_TABLE_SLOTS = 512
_TABLE_LIVE = 384
_TABLE_MASK = _TABLE_SLOTS - 1
_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)
_HASH_SHIFT = np.uint64(64 - 9)

_NO_SCORE = (0.0, 0.0, 0.0)
_NO_HESSIAN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@njit(cache=True, error_model="numpy")
def _u_series(u, order):
    """``(R2, E1, T3, U2, T3')`` for ``|u| < 1`` by Horner's rule in ``u``, only what ``order`` reads."""
    last = _U_SERIES_TERMS - 1
    r2 = _U_COEFFICIENTS[last, 0]
    if order == 0:
        for j in range(last - 1, -1, -1):
            r2 = r2 * u + _U_COEFFICIENTS[j, 0]
        return r2, 0.0, 0.0, 0.0, 0.0
    e1 = _U_COEFFICIENTS[last, 1]
    t3 = _U_COEFFICIENTS[last, 2]
    if order == 1:
        for j in range(last - 1, -1, -1):
            r2 = r2 * u + _U_COEFFICIENTS[j, 0]
            e1 = e1 * u + _U_COEFFICIENTS[j, 1]
            t3 = t3 * u + _U_COEFFICIENTS[j, 2]
        return r2, e1, t3, 0.0, 0.0
    u2 = _U_COEFFICIENTS[last, 3]
    t3d = _U_COEFFICIENTS[last, 4]
    for j in range(last - 1, -1, -1):
        r2 = r2 * u + _U_COEFFICIENTS[j, 0]
        e1 = e1 * u + _U_COEFFICIENTS[j, 1]
        t3 = t3 * u + _U_COEFFICIENTS[j, 2]
        u2 = u2 * u + _U_COEFFICIENTS[j, 3]
        t3d = t3d * u + _U_COEFFICIENTS[j, 4]
    return r2, e1, t3, u2, t3d


@njit(cache=True, error_model="numpy")
def _l_series(v):
    """``(L1, L1', L2)`` for ``|v| < 1/4`` by Horner's rule in ``v``."""
    last = _V_SERIES_TERMS - 1
    l1 = _L_COEFFICIENTS[last, 0]
    l1d = _L_COEFFICIENTS[last, 1]
    l2 = _L_COEFFICIENTS[last, 2]
    for j in range(last - 1, -1, -1):
        l1 = l1 * v + _L_COEFFICIENTS[j, 0]
        l1d = l1d * v + _L_COEFFICIENTS[j, 1]
        l2 = l2 * v + _L_COEFFICIENTS[j, 2]
    return l1, l1d, l2


@njit(cache=True, error_model="numpy")
def _m3_series(t2):
    """``(A(t2), B(t2))``, the atanh series of ``M3`` and of its derivative, by Horner's rule in ``t2``."""
    last = _M3_SERIES_TERMS - 1
    a = _M3_COEFFICIENTS[last, 0]
    b = _M3_COEFFICIENTS[last, 1]
    for j in range(last - 1, -1, -1):
        a = a * t2 + _M3_COEFFICIENTS[j, 0]
        b = b * t2 + _M3_COEFFICIENTS[j, 1]
    return a, b


@njit(cache=True, error_model="numpy")
def _all_finite(values):
    for value in values:
        if not math.isfinite(value):
            return False
    return True


@njit(cache=True, error_model="numpy")
def _stirling_d2_series(x):
    """The Bernoulli series for ``trigamma(x) - 1/x - 1/(2x^2)``, valid for ``x >= 8``."""
    inv = 1.0 / x
    inv2 = inv * inv
    acc = 0.0
    power = inv2 * inv
    for j in range(len(_BERNOULLI)):
        acc = acc + (2 * j + 1) * (2 * j + 2) * _BERNOULLI[j] * power
        power = power * inv2
    return acc


@njit(cache=True, error_model="numpy")
def _stirling_triplet(x):
    """``(S(x), S'(x), S''(x))`` of the Stirling remainder; series for ``x >= 8``.

    Below the switch ``S`` and ``S'`` are the direct differences and ``S''``
    walks the recurrence ``S''(x) = S''(x + 1) + 1/(2 x^2 (x + 1)^2)`` onto the
    series. A non-positive ``x`` gives NaN in ``S`` and ``S'``, as ``log`` does.
    """
    if x >= _STIRLING_SWITCH:
        inv2 = 1.0 / (x * x)
        s0 = 0.0
        s1 = 0.0
        power0 = 1.0 / x
        power1 = inv2
        for j in range(len(_BERNOULLI)):
            s0 = s0 + _BERNOULLI[j] * power0
            s1 = s1 - (2 * j + 1) * _BERNOULLI[j] * power1
            power0 = power0 * inv2
            power1 = power1 * inv2
        return s0, s1, _stirling_d2_series(x)
    log_x = math.log(x)
    s0 = math.lgamma(x) - (x - 0.5) * log_x + x - _HALF_LOG_TWO_PI
    s1 = _digamma_positive(x) - log_x + 0.5 / x
    steps = 0.0
    for j in range(int(_STIRLING_SWITCH)):
        lower = x + j
        upper = lower + 1.0
        steps = steps + 0.5 / (lower * lower * upper * upper)
    return s0, s1, _stirling_d2_series(x + _STIRLING_SWITCH) + steps


@njit(cache=True, error_model="numpy")
def _shape_terms(q):
    """``(k, S(k), S'(k), S''(k), s0, k^2 S'(k), k^3 S''(k))``, including ``Q = 0``.

    The raw remainders feed the mean loading, which zeroes them below the
    zero-shape switch; the scaled ``s0, k2s1, k3s2`` feed the location rows and
    take their ``Q -> 0`` limits there.
    """
    if abs(q) < _ZERO_SHAPE:
        return 1.0, 0.0, 0.0, 0.0, q * q / 12.0, -1.0 / 12.0, 1.0 / 6.0
    k = 1.0 / (q * q)
    s0k, s1k, s2k = _stirling_triplet(k)
    return k, s0k, s1k, s2k, s0k, k * k * s1k, k * k * k * s2k


@njit(cache=True, error_model="numpy")
def _u_channels(u, e, order):
    """``(R2, E1, T3, U2, T3')`` of ``u = Q w``; inside the series, only what ``order`` reads."""
    if abs(u) < _U_SERIES_RADIUS:
        return _u_series(u, order)
    em1 = math.expm1(u)
    square = u * u
    cube = square * u
    return (
        (em1 - u) / square,
        em1 / u,
        (2.0 * (em1 - u) - u * em1) / cube,
        (u * e - em1) / square,
        (4.0 * u * em1 - square * e - 6.0 * em1 + 6.0 * u) / (cube * u),
    )


@njit(cache=True, error_model="numpy")
def _log_mean_loading(sigma, q, k, s0k, s1k, s2k):
    """``(log C, C_sigma, C_Q, C_sigma_sigma, C_sigma_Q, C_QQ)``, with their ``Q = 0`` limits."""
    v = sigma * q
    if abs(q) < _ZERO_SHAPE:
        q2 = s0x = s1x = s2x = inv_q = 0.0
    else:
        q2 = q * q
        s0x, s1x, s2x = _stirling_triplet(k + sigma / q)
        inv_q = 1.0 / q
    log1p_v = math.log1p(v)
    if abs(v) < _V_SERIES_RADIUS:
        l1, l1d, l2 = _l_series(v)
    else:
        l1 = log1p_v / v
        l1d = (v / (1.0 + v) - log1p_v) / (v * v)
        l2 = (log1p_v - v) / (v * v)
    if abs(v) < _M3_SERIES_RADIUS:
        # The atanh form of log1p with t = v/(2 + v); B = A'(t)/t.
        t = v / (2.0 + v)
        t2 = t * t
        a, b = _m3_series(t2)
        one = 1.0 - t
        one2 = one * one
        m3 = -0.25 * one2 * a
        m3d = 0.25 * one2 * one * a - 0.125 * one2 * one2 * t * b
    else:
        m3 = (v - (1.0 + 0.5 * v) * log1p_v) / (v * v * v)
        m3d = (0.5 * l1d - 3.0 * m3) / v
    d_s1 = s1x - s1k
    sig2 = sigma * sigma
    one_v = 1.0 + v
    half_v = 1.0 + 0.5 * v
    inv_q2 = inv_q * inv_q
    inv_q3 = inv_q2 * inv_q
    logc = sig2 * (l2 + l1) - 0.5 * log1p_v + s0x - s0k
    c_s = sigma * l1 - q / (2.0 * one_v) + s1x * inv_q
    c_q = (
        2.0 * sig2 * sigma * m3
        - sigma * half_v / one_v
        - 2.0 * inv_q3 * half_v * d_s1
        + 0.5 * sigma
        - sigma * s1k * inv_q2
    )
    c_ss = 1.0 / one_v + q2 / (2.0 * one_v * one_v) + s2x * inv_q2
    c_sq = (
        sig2 * l1d - 1.0 / (2.0 * one_v * one_v) - s2x * (2.0 + v) * inv_q2 * inv_q2 - s1x * inv_q2
    )
    c_qq = (
        2.0 * sig2 * sig2 * m3d
        + sig2 / (2.0 * one_v * one_v)
        + 6.0 * half_v * d_s1 * inv_q2 * inv_q2
        - sigma * d_s1 * inv_q3
        + 2.0 * inv_q3 * inv_q3 * half_v * ((2.0 + v) * s2x - 2.0 * s2k)
        + 2.0 * sigma * s2k * inv_q3 * inv_q2
        + 2.0 * sigma * s1k * inv_q3
    )
    return logc, c_s, c_q, c_ss, c_sq, c_qq


@njit(cache=True, error_model="numpy")
def _location_row(y, mu, sigma, log_sigma, q, m, s0, k2s1, k3s2, order):
    """``(valid, optimizing, score, hessian)`` of one ``(mu, sigma, Q)`` row, unit carrier left out."""
    w = (math.log(y) - mu) / sigma
    u = q * w
    e = math.exp(u)
    r2, e1, t3, u2, t3d = _u_channels(u, e, order)
    w2 = w * w
    optimizing = m * (-log_sigma - s0 - w2 * r2)
    score = _NO_SCORE
    hessian = _NO_HESSIAN
    if order >= 1:
        w3 = w2 * w
        score = (
            m * w * e1 / sigma,
            m * (w2 * e1 - 1.0) / sigma,
            m * (2.0 * q * k2s1 + w3 * t3),
        )
        if order == 2:
            s2 = sigma * sigma
            hessian = (
                m * (-e / s2),
                m * (-w * (e + e1) / s2),
                m * (w2 * u2 / sigma),
                m * ((1.0 - 2.0 * w2 * e1 - w2 * e) / s2),
                m * (w3 * u2 / sigma),
                m * (-4.0 * k3s2 - 6.0 * k2s1 + w3 * w * t3d),
            )
    valid = (
        math.isfinite(e)
        and math.isfinite(optimizing)
        and _all_finite(score)
        and _all_finite(hessian)
    )
    return valid, optimizing, score, hessian


@njit(cache=True, error_model="numpy")
def _mean_chain(me, score, hessian, c_s, c_q, c_ss, c_sq, c_qq, order):
    """Score and packed Hessian in ``(m, sigma, Q)`` through ``mu = log m - log C``."""
    s_mu, s_sig, s_q = score
    h_mm, h_ms, h_mq, h_ss, h_sq, h_qq = hessian
    chained = (s_mu / me, s_sig - s_mu * c_s, s_q - s_mu * c_q)
    if order < 2:
        return chained, _NO_HESSIAN
    return chained, (
        (h_mm - s_mu) / (me * me),
        (h_ms - h_mm * c_s) / me,
        (h_mq - h_mm * c_q) / me,
        h_ss - 2.0 * h_ms * c_s + h_mm * c_s * c_s - s_mu * c_ss,
        h_sq - h_ms * c_q - h_mq * c_s + h_mm * c_s * c_q - s_mu * c_sq,
        h_qq - 2.0 * h_mq * c_q + h_mm * c_q * c_q - s_mu * c_qq,
    )


@njit(cache=True, error_model="numpy")
def distinct_pairs(first_key, second_key, limit):
    """``(count, rows, inverse)``: the first row of each distinct bit pattern of ``(first, second)``.

    ``inverse[row]`` indexes ``rows``. Past ``limit`` distinct pairs it stops and returns
    ``count = limit + 1``, with ``rows`` and ``inverse`` unfinished.
    """
    n = len(first_key)
    size = 16
    index_bits = 4
    while size < 2 * limit + 2:
        size *= 2
        index_bits += 1
    shift = np.uint64(64 - index_bits)
    first_bits = first_key.view(np.int64)
    second_bits = second_key.view(np.int64)
    table = np.full(size, -1, dtype=np.int64)
    rows = np.empty(min(n, limit + 1), dtype=np.int64)
    inverse = np.empty(n, dtype=np.int64)
    count = 0
    for row in range(n):
        mixed = (
            np.uint64(first_bits[row]) * _HASH_MULTIPLIER + np.uint64(second_bits[row])
        ) * _HASH_MULTIPLIER
        slot = np.int64(mixed >> shift)
        while True:
            held = table[slot]
            if held < 0:
                if count >= limit:
                    return limit + 1, rows, inverse
                table[slot] = count
                rows[count] = row
                held = count
                count += 1
                break
            first_row = rows[held]
            if (
                first_bits[first_row] == first_bits[row]
                and second_bits[first_row] == second_bits[row]
            ):
                break
            slot = (slot + 1) & (size - 1)
        inverse[row] = held
    return count, rows[:count], inverse


@njit(cache=True, error_model="numpy")
def _loading_slot(bits, stamp, keys, generation):
    """``(slot, found)``: where ``bits`` sits in the live table, or the empty slot it would take."""
    slot = np.int64((np.uint64(bits) * _HASH_MULTIPLIER) >> _HASH_SHIFT)
    while stamp[slot] == generation and keys[slot] != bits:
        slot = (slot + 1) & _TABLE_MASK
    return slot, stamp[slot] == generation


@njit(cache=True, nogil=True, error_model="numpy")
def _evaluate_rows(
    response: NDArray[np.float64],
    first: NDArray[np.float64],
    sigma: NDArray[np.float64],
    shape: NDArray[np.float64],
    multiplier: NDArray[np.float64],
    mean_form: bool,
    order: int,
    start: int,
    stop: int,
    optimizing: NDArray[np.float64],
    score: NDArray[np.float64],
    hessian: NDArray[np.float64],
    valid: NDArray[np.bool_],
):
    """Rows ``start..stop`` into zeroed outputs: ``(status, failing row)``.

    ``first`` is the mean or the location. Only valid rows are written, so the
    others keep their zero placeholders and ``valid = False``. The shape terms
    are kept from the row before while its ``Q`` repeats, and the loading and
    ``log(sigma)`` from the block's table while ``Q`` is unchanged: the same bits
    as forming them anew.
    """
    q_previous = 0.0
    k, s0k, s1k, s2k, s0, k2s1, k3s2 = _shape_terms(q_previous)
    sigma_previous = math.nan
    log_sigma = 0.0
    logc = c_s = c_q = c_ss = c_sq = c_qq = 0.0
    slots = _TABLE_SLOTS if mean_form else 1
    stamp = np.zeros(slots, dtype=np.int64)
    keys = np.empty(slots, dtype=np.int64)
    entries = np.empty((slots, 7), dtype=np.float64)
    sigma_bits = sigma.view(np.int64)
    generation = 1
    live = 0
    for row in range(start, stop):
        q = shape[row]
        row_sigma = sigma[row]
        if q != q_previous:
            k, s0k, s1k, s2k, s0, k2s1, k3s2 = _shape_terms(q)
            q_previous = q
            sigma_previous = math.nan
            generation += 1  # the table's loadings belong to the shape before
            live = 0
        location = first[row]
        if mean_form:
            if not (q >= 0.0 or row_sigma * -q < 1.0):  # the mean does not exist
                continue
            if row_sigma != sigma_previous:
                slot, found = _loading_slot(sigma_bits[row], stamp, keys, generation)
                if found:
                    logc, c_s, c_q = entries[slot, 0], entries[slot, 1], entries[slot, 2]
                    c_ss, c_sq, c_qq = entries[slot, 3], entries[slot, 4], entries[slot, 5]
                    log_sigma = entries[slot, 6]
                else:
                    logc, c_s, c_q, c_ss, c_sq, c_qq = _log_mean_loading(
                        row_sigma, q, k, s0k, s1k, s2k
                    )
                    if not _all_finite((logc, c_s, c_q, c_ss, c_sq, c_qq)):
                        return KERNEL_MEAN_LOADING, row
                    log_sigma = math.log(row_sigma)
                    if live < _TABLE_LIVE:
                        stamp[slot] = generation
                        keys[slot] = sigma_bits[row]
                        entries[slot, 0], entries[slot, 1], entries[slot, 2] = logc, c_s, c_q
                        entries[slot, 3], entries[slot, 4], entries[slot, 5] = c_ss, c_sq, c_qq
                        entries[slot, 6] = log_sigma
                        live += 1
                sigma_previous = row_sigma
            location = math.log(location) - logc
        elif row_sigma != sigma_previous:
            log_sigma = math.log(row_sigma)
            sigma_previous = row_sigma
        row_valid, row_value, row_score, row_hessian = _location_row(
            response[row], location, row_sigma, log_sigma, q, multiplier[row], s0, k2s1, k3s2, order
        )
        if not row_valid:
            continue
        if mean_form and order >= 1:
            row_score, row_hessian = _mean_chain(
                first[row], row_score, row_hessian, c_s, c_q, c_ss, c_sq, c_qq, order
            )
            if not (_all_finite(row_score) and _all_finite(row_hessian)):
                continue
        optimizing[row] = row_value
        valid[row] = True
        if order >= 1:
            score[row, 0], score[row, 1], score[row, 2] = row_score
            if order == 2:
                (
                    hessian[row, 0],
                    hessian[row, 1],
                    hessian[row, 2],
                    hessian[row, 3],
                    hessian[row, 4],
                    hessian[row, 5],
                ) = row_hessian
    return KERNEL_OK, np.int64(-1)


@njit(cache=True, nogil=True, error_model="numpy")
def mean_direction_rows(
    mean: NDArray[np.float64],
    score_mu: NDArray[np.float64],
    hessian: NDArray[np.float64],
    loading: NDArray[np.float64],
    direction: NDArray[np.float64],
    packed_location: NDArray[np.float64],
    start: int,
    stop: int,
    out: NDArray[np.float64],
):
    """Mean-form ``dH(v)`` in ``(m, sigma, Q)`` from the location form's, rows ``start..stop``.

    ``mu = log m - log C(sigma, Q)`` has Jacobian ``J`` (identity except the ``mu`` row
    ``(1/m, -C_s, -C_Q)``) and second and third derivatives only in ``mu``. With
    ``w = J v``, ``Z_i = sum_k mu_ik v_k`` and ``mu_ij(v) = sum_k mu_ijk v_k``:

        dH(v) = J' dH_loc(w) J + Z (H J)_mu' + (H J)_mu Z' + mu_ij (H w)_mu + s_mu mu_ij(v).

    ``hessian`` is the location Hessian at unit mass, packed ``(n, 6)``, ``loading`` the
    columns ``C_s, C_Q, C_ss, C_sQ, C_QQ, C_sss, C_ssQ, C_sQQ, C_QQQ``, ``direction`` the
    mean-form directions ``(d, n, 3)`` and ``packed_location`` the location form's
    ``dH_loc(w)`` along ``w = J v``, ``(d, n, 6)``. Each row writes its own ``out[:, row]``.
    """
    for row in range(start, stop):
        m = mean[row]
        c_s, c_q = loading[row, 0], loading[row, 1]
        c_ss, c_sq, c_qq = loading[row, 2], loading[row, 3], loading[row, 4]
        c_sss, c_ssq, c_sqq, c_qqq = (
            loading[row, 5],
            loading[row, 6],
            loading[row, 7],
            loading[row, 8],
        )
        h_mm, h_ms, h_mq = hessian[row, 0], hessian[row, 1], hessian[row, 2]
        # (H J)_mu, and the ``mu`` second derivatives (nonzero only in these entries).
        g_m = h_mm / m
        g_s = h_ms - h_mm * c_s
        g_q = h_mq - h_mm * c_q
        d_mm = -1.0 / (m * m)
        for index in range(direction.shape[0]):
            v_m, v_s, v_q = (
                direction[index, row, 0],
                direction[index, row, 1],
                direction[index, row, 2],
            )
            w_m = v_m / m - c_s * v_s - c_q * v_q
            z_m = d_mm * v_m
            z_s = -c_ss * v_s - c_sq * v_q
            z_q = -c_sq * v_s - c_qq * v_q
            h_w = h_mm * w_m + h_ms * v_s + h_mq * v_q
            t_mm = 2.0 / (m * m * m) * v_m
            t_ss = -(c_sss * v_s + c_ssq * v_q)
            t_sq = -(c_ssq * v_s + c_sqq * v_q)
            t_qq = -(c_sqq * v_s + c_qqq * v_q)
            location = packed_location[index, row]
            # (L J) for the entries J' (L J) reads, L the symmetric matrix of the packed location
            # direction; J' (L J) is then a row of it at 1/m, -C_s, -C_Q beside the identity rows.
            lj_00 = location[0] / m
            lj_01 = location[0] * -c_s + location[1]
            lj_02 = location[0] * -c_q + location[2]
            lj_11 = location[1] * -c_s + location[3]
            lj_12 = location[1] * -c_q + location[4]
            lj_22 = location[2] * -c_q + location[5]
            # full = J' (L J) + Z g' + g Z' + mu_ij (H w)_mu + s_mu mu_ij(v), upper triangle.
            out[index, row, 0] = lj_00 / m + 2.0 * z_m * g_m + d_mm * h_w + score_mu[row] * t_mm
            out[index, row, 1] = lj_01 / m + z_m * g_s + g_m * z_s
            out[index, row, 2] = lj_02 / m + z_m * g_q + g_m * z_q
            out[index, row, 3] = (
                -c_s * lj_01 + lj_11 + 2.0 * z_s * g_s - c_ss * h_w + score_mu[row] * t_ss
            )
            out[index, row, 4] = (
                -c_s * lj_02 + lj_12 + z_s * g_q + g_s * z_q - c_sq * h_w + score_mu[row] * t_sq
            )
            out[index, row, 5] = (
                -c_q * lj_02 + lj_22 + 2.0 * z_q * g_q - c_qq * h_w + score_mu[row] * t_qq
            )
    return KERNEL_OK, np.int64(-1)


# The kernels the worker pool runs (``nogil``, never ``prange``).
_POOLED_ROW_KERNELS = (_evaluate_rows, mean_direction_rows)
