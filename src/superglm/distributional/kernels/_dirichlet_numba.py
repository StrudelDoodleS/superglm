"""Dirichlet-only compiled arithmetic, with Python fallbacks of the same formulas.

All scratch belongs to one call: O(K^2) per row, plus packed batch outputs.
No parameter, weight, basis or penalty-dependent cache is retained. Numba's
disk cache contains code only; the callee digest covers imported primitives.
Fast-math/reassociation is deliberately disabled: split products and the
nonoverlapping summation expansion require ordinary binary64 rounding.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba.extending import overload, register_jitable

from superglm.distributional.kernels._severity_primitives import (
    _EULER,
    _SRES,
    _ZETA,
    gnorm,
    sres_0123,
)

_CALLEE_DIGEST = "c7b29bd4d7452aa2445465854f122a2dcc01ce1ec33e56ceb24f7ed364f16565"
_U = 2.0**-53
# A routing envelope, never an admission limit. Within it intermediate
# powers/products in the row and four-channel jets stay in range. Outside,
# the Python implementation retains NumPy's intermediate-overflow exceptions
# (including an overflow in an unselected direction channel). Final-channel
# finiteness alone cannot detect those. No K-dependent cutoff is imposed.
_FAST_MIN = 2.0**-64
_FAST_MAX = 2.0**64


def _sum(values):
    return math.fsum(map(float, values))


@njit(cache=True)
def _fsum(values):
    """Nonoverlapping FastTwoSum expansion, including half-even final rounding.

    The expansion and collapse follow the binary64 algorithm documented with
    CPython's math.fsum (Modules/mathmodule.c). Each partial is a float64
    rounding residual, not a wider dtype or arbitrary-precision value.
    Scratch is at most len(values) floats, dies with this reduction, and works
    for arbitrary K. Subnormal addition is exact. Nonfinite input/intermediate
    sums return NaN so the caller replays the Python path and its refusal.
    """
    partials = np.empty(len(values))
    size = 0
    for value in values:
        x = float(value)
        if not math.isfinite(x):
            return math.nan
        write = 0
        for j in range(size):
            y = partials[j]
            if abs(x) < abs(y):
                x, y = y, x
            high = x + y
            if not math.isfinite(high):
                return math.nan
            low = y - (high - x)
            if low != 0.0:
                partials[write] = low
                write += 1
            x = high
        size = write
        if x != 0.0:
            partials[size] = x
            size += 1
    high, low = 0.0, 0.0
    if size:
        size -= 1
        high = partials[size]
        while size:
            size -= 1
            x, y = high, partials[size]
            high = x + y
            low = y - (high - x)
            if low != 0.0:
                break
        if size and (
            (low < 0.0 and partials[size - 1] < 0.0) or (low > 0.0 and partials[size - 1] > 0.0)
        ):
            doubled = 2.0 * low
            rounded = high + doubled
            if rounded - high == doubled:
                high = rounded
    return high


@overload(_sum)
def _compiled_sum(values):
    def implementation(values):
        return _fsum(values)

    return implementation


@register_jitable
def _product_parts(a, b):
    """Exact product of frexp mantissas; every split operation stays normal."""
    product = a * b
    split_a = 134217729.0 * a
    hi_a = split_a - (split_a - a)
    lo_a = a - hi_a
    split_b = 134217729.0 * b
    hi_b = split_b - (split_b - b)
    lo_b = b - hi_b
    error = ((hi_a * hi_b - product) + hi_a * lo_b + lo_a * hi_b) + lo_a * lo_b
    return product, error


@register_jitable
def _close_ratio(x, alpha, j):
    """(y_j / mu_j - 1) from sum_k (a_k x_j - a_j x_k).

    Products are split before scaling by the exponent of a_j. Their sum
    keeps the ulp residual of both row totals instead of subtracting two
    rounded totals. The caller only enters when the ratio is near one, so
    the scaled positive/negative sums cannot overflow. Underflow of remote
    summands costs at most half a subnormal in these scaled coordinates.
    """
    aj, exponent = math.frexp(float(alpha[j]))
    xj, xje = math.frexp(float(x[j]))
    parts = []
    for k in range(len(x)):
        if k == j:
            continue
        ak, ake = math.frexp(float(alpha[k]))
        xk, xke = math.frexp(float(x[k]))
        p, e = _product_parts(ak, xj)
        q, f = _product_parts(aj, xk)
        parts.extend(
            (
                math.ldexp(p, ake + xje - exponent),
                math.ldexp(e, ake + xje - exponent),
                -math.ldexp(q, xke),
                -math.ldexp(f, xke),
            )
        )
    return _sum(parts) / aj / _sum(x)


@register_jitable
def _r_minus_log1p(r):
    if abs(r) > 0.125:
        return r - math.log1p(r)
    # Alternating Taylor series, |r| <= 1/8; 20 terms leave < u of r^2/2.
    term = r * r
    terms = []
    for degree in range(2, 22):
        terms.append(term / degree)
        term *= -r
    return _sum(terms)


@register_jitable
def _scaled_product(*values):
    mantissa, exponent = 1.0, 0
    for value in values:
        m, e = math.frexp(float(value))
        mantissa *= m
        exponent += e
    return math.ldexp(mantissa, exponent)


@register_jitable
def _power_difference(mu, rho, p):
    return -rho * _sum([mu**i for i in range(p)])


@register_jitable
def _residual_difference_asymptotic(a, total, other, order):
    mu, rho = a / total, other / total
    inverse = 1 / a
    power = inverse
    terms = [(-0.5 if order == 0 else 0.5) * _power_difference(mu, rho, order + 1)]
    for i, coefficient in enumerate(_SRES[order]):
        terms.append(coefficient * power * _power_difference(mu, rho, order + 2 * (i + 1)))
        power *= inverse * inverse
    return _sum(terms)


@register_jitable
def _residual_difference(a, total, other, order):
    """Scalar counterpart of the directional recurrence below."""
    if a >= 10:
        return _residual_difference_asymptotic(a, total, other, order)
    shift = math.ceil(10 - a)
    left, right = a + shift, total + shift
    shifted = _residual_difference_asymptotic(left, right, other, order)
    if order == 0:
        terms = [(a / left) * shifted, a * (math.log1p(other / left) - math.log1p(other / a))]
        for i in range(shift):
            terms.append((a / (a + i)) * (other / (total + i)))
    else:
        terms = [(a / left) ** 2 * shifted, a * (other / total), -(a / left) * (a / right) * other]
        for i in range(shift):
            terms.append(
                -(a / (a + i)) * (a / (total + i)) * (other / (a + i) + other / (total + i))
            )
    return _sum(terms)


@register_jitable
def _row_state(x, logs, alpha, order=2):
    total = _sum(alpha)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("nonrepresentable concentration sum")
    mu = alpha / total
    if np.any(mu == 0):
        raise ValueError("nonrepresentable positive mean share")
    k = len(alpha)
    residual = np.empty((k, 4))
    for j in range(k):
        residual[j] = sres_0123(float(alpha[j]))
    total_residual = np.array(sres_0123(total))
    ratios = np.empty(len(alpha))
    divergence = []
    for j, a in enumerate(alpha):
        # The logs only select the close branch; they never decide a close ratio.
        rough = logs[j] + math.log(total) - math.log(float(a))
        if abs(rough) < 0.5:
            r = _close_ratio(x, alpha, j)
            ratios[j] = math.log1p(r)
            divergence.append(a * _r_minus_log1p(r))
        else:
            ratios[j] = rough
            # a*(r-log1p(r)), without forming a possibly overflowing y/mu.
            divergence.append(_sum((total * math.exp(logs[j]), -a, -a * rough)))
    value_terms = np.empty(2 * k + 2)
    value_terms[0], value_terms[1] = -_sum(divergence), -gnorm(total)
    for j in range(k):
        value_terms[j + 2] = gnorm(float(alpha[j]))
        value_terms[j + k + 2] = -logs[j]
    value = _sum(value_terms)
    scaled_score = np.empty(k)
    scaled_hessian = np.empty((k, k))
    for j in range(k if order else 0):
        other = _sum(np.delete(alpha, j))
        if mu[j] > 0.5:
            # A dominant component can have a residual smaller than one ulp
            # of either sres term, even though its scaled score is representable.
            score_residual = _residual_difference(float(alpha[j]), total, other, 0)
        else:
            score_residual = mu[j] * total_residual[0] - residual[j, 0]
        scaled_score[j] = alpha[j] * ratios[j] + score_residual
        if order < 2:
            continue
        hessian_residual = (
            _residual_difference(float(alpha[j]), total, other, 1)
            if mu[j] > 0.5
            else mu[j] ** 2 * total_residual[1] - residual[j, 1]
        )
        scaled_hessian[j, j] = -alpha[j] * (other / total) + hessian_residual
        for column in range(j + 1, k):
            scaled_hessian[j, column] = scaled_hessian[column, j] = (
                alpha[j] * mu[column] + mu[j] * mu[column] * total_residual[1]
            )
    return value, scaled_score, scaled_hessian, total, mu, residual, total_residual, ratios


# A two-direction jet (value, a, b, ab). Each entry is a scalar, so the
# directional evaluator stores only O(K^2), never a third/fourth tensor.
@register_jitable
def _mul(x, y):
    return np.array(
        [
            x[0] * y[0],
            x[1] * y[0] + x[0] * y[1],
            x[2] * y[0] + x[0] * y[2],
            _sum((x[3] * y[0], x[1] * y[2], x[2] * y[1], x[0] * y[3])),
        ]
    )


@register_jitable
def _div(x, y):
    value = x[0] / y[0]
    first = (x[1] - value * y[1]) / y[0]
    second = (x[2] - value * y[2]) / y[0]
    mixed = _sum((x[3], -value * y[3], -first * y[2], -second * y[1])) / y[0]
    return np.array([value, first, second, mixed])


@register_jitable
def _residual_jet(alpha, residual, order):
    m = order
    value = residual[m]
    a = alpha[0]
    if a < 0.25:
        # Differentiate the origin series itself: sres0+sres1 would
        # cancel -1+1 and erase the derivative of a tiny shape.
        slope = -a * (_EULER + math.log(a) + 1) if m == 0 else -a
        curvature = -a if m == 0 else 0.0
        power = a
        for i, coefficient in enumerate(_ZETA):
            power *= a
            degree = i + 2
            term = coefficient * power * (1 if i % 2 == 0 else -1)
            factor = 1 if m == 0 else degree - 1
            slope += degree * factor * term
            curvature += degree * (degree - 1) * factor * term
    elif a >= 10:
        slope_terms, curvature_terms = [], []
        inverse = 1 / a
        power = inverse
        for i, coefficient in enumerate(_SRES[m]):
            degree = 2 * i + 1
            term = coefficient * power
            slope_terms.append(-degree * term)
            curvature_terms.append(degree * (degree + 1) * term)
            power *= inverse * inverse
        slope, curvature = _sum(slope_terms), _sum(curvature_terms)
    else:
        slope = (m + 1) * value + residual[m + 1]
        curvature = m * (m + 1) * value + 2 * (m + 1) * residual[m + 1] + residual[m + 2]
    da, db, dab = alpha[1:] / alpha[0]
    return np.array([value, slope * da, slope * db, curvature * da * db + slope * dab])


@register_jitable
def _log1p_jet(x):
    denominator = 1 + x[0]
    return np.array(
        [
            math.log1p(x[0]),
            x[1] / denominator,
            x[2] / denominator,
            x[3] / denominator - (x[1] / denominator) * (x[2] / denominator),
        ]
    )


@register_jitable
def _power_difference_jet(mu, rho, p):
    one = np.array([1.0, 0.0, 0.0, 0.0])
    series, term = one.copy(), one.copy()
    for _ in range(1, p):
        term = _mul(term, mu)
        series += term
    return -_mul(rho, series)


@register_jitable
def _residual_difference_jet_asymptotic(a, total, other, order):
    one = np.array([1.0, 0.0, 0.0, 0.0])
    mu = _div(a, total)
    rho = _div(other, total)
    mu[1:] = -rho[1:]
    constant = -0.5 if order == 0 else 0.5
    result = constant * _power_difference_jet(mu, rho, order + 1)
    inverse = _div(one, a)
    inverse2 = _mul(inverse, inverse)
    power = inverse.copy()
    for i, coefficient in enumerate(_SRES[order]):
        result += coefficient * _mul(power, _power_difference_jet(mu, rho, order + 2 * (i + 1)))
        power = _mul(power, inverse2)
    return result


@register_jitable
def _residual_difference_jet(a, total, other, order):
    """a^(m+1) [psires_m(A)-psires_m(a)], preserving A-a separately.

    The asymptotic difference factors mu^p-1 as -rho sum(mu^i).
    Below ten, shift both arguments together and difference the recurrence's
    reciprocal terms before summation. These are float64 formulas even when
    a+other rounds back to a. This same expression supplies optional directions.
    """
    one = np.array([1.0, 0.0, 0.0, 0.0])
    if a[0] >= 10:
        return _residual_difference_jet_asymptotic(a, total, other, order)
    shift = math.ceil(10 - a[0])
    shifted_a, shifted_total = a + shift * one, total + shift * one
    ratio = _div(a, shifted_a)
    shifted = _residual_difference_jet_asymptotic(shifted_a, shifted_total, other, order)
    if order == 0:
        result = _mul(ratio, shifted)
        result += _mul(a, _log1p_jet(_div(other, shifted_a)) - _log1p_jet(_div(other, a)))
    else:
        result = _mul(_mul(ratio, ratio), shifted)
        result += _mul(a, _div(other, total)) - _mul(_mul(ratio, _div(a, shifted_total)), other)
    for i in range(shift):
        left, right = a + i * one, total + i * one
        if order == 0:
            result += _mul(_div(a, left), _div(other, right))
        else:
            result -= _mul(
                _mul(_div(a, left), _div(a, right)), _div(other, left) + _div(other, right)
            )
    return result


@njit(cache=True, error_model="numpy")
def _evaluate_batch(response, alpha, mass, order, logs):
    """Packed outputs; false rows must be replayed by the Python boundary."""
    n, k = alpha.shape
    pairs = k * (k + 1) // 2
    values = np.zeros(n)
    score = np.zeros((n, k if order else 0))
    hessian = np.zeros((n, pairs if order == 2 else 0))
    valid = np.zeros(n, dtype=np.bool_)
    for row in range(n):
        a = alpha[row]
        if np.any(a < _FAST_MIN) or np.any(a > _FAST_MAX) or not np.isfinite(a).all():
            continue
        if mass[row] > _FAST_MAX:
            continue
        total = _sum(a)
        if not math.isfinite(total) or total <= 0 or np.any(a / total == 0):
            continue
        value, scaled_s, scaled_h, _, _, _, _, _ = _row_state(response[row], logs[row], a, order)
        value *= mass[row]
        ok = math.isfinite(value)
        if order:
            for j in range(k):
                natural = (scaled_s[j] / a[j]) * mass[row]
                target = scaled_s[j] * mass[row]
                reconstructed = natural * a[j]
                if (
                    not math.isfinite(natural)
                    or not math.isfinite(target)
                    or not math.isfinite(reconstructed)
                    or abs(reconstructed - target) > 128 * _U * abs(target)
                ):
                    ok = False
                score[row, j] = natural
        if order == 2:
            p = 0
            for j in range(k):
                for column in range(j, k):
                    natural = _scaled_product(
                        scaled_h[j, column], mass[row], 1 / a[j], 1 / a[column]
                    )
                    target = scaled_h[j, column] * mass[row]
                    reconstructed = _scaled_product(natural, a[j], a[column])
                    if (
                        not math.isfinite(natural)
                        or not math.isfinite(target)
                        or not math.isfinite(reconstructed)
                        or abs(reconstructed - target) > 256 * _U * abs(target)
                    ):
                        ok = False
                    hessian[row, p] = natural
                    p += 1
        if ok:
            values[row] = value
            valid[row] = True
        else:
            score[row] = 0
            hessian[row] = 0
    return values, score, hessian, valid


def _curvature_python(
    response, theta, multiplier, link_derivatives, first, second, double, log_shares
):
    """Observed predictor-curvature directions, using separable links to order four."""
    n, k = theta.shape
    result = np.empty((n, k * (k + 1) // 2))
    pairs = np.empty((k * (k + 1) // 2, 2), dtype=np.int64)
    p = 0
    for j in range(k):
        for column in range(j, k):
            pairs[p, 0], pairs[p, 1] = j, column
            p += 1
    for row in range(n):
        alpha = theta[row]
        _, _, _, _, _, residual, total_residual, ratios = _row_state(
            response[row], log_shares[row], alpha
        )
        d1, d2, d3, d4 = [d[row] for d in link_derivatives]
        v, w = first[row], second[row]
        ajet = np.column_stack((alpha, d1 * v, d1 * w, d2 * v * w))
        totaljet = np.array([_sum(ajet[:, c]) for c in range(4)])
        mujet, otherjet = [], []
        complementjets = []
        for j in range(k):
            complement = np.array([_sum(np.delete(ajet[:, c], j)) for c in range(4)])
            mu, rho = _div(ajet[j], totaljet), _div(complement, totaljet)
            if mu[0] > 0.5:
                mu[1:] = -rho[1:]
            else:
                rho[1:] = -mu[1:]
            mujet.append(mu)
            otherjet.append(rho)
            complementjets.append(complement)
        rtotal = _residual_jet(totaljet, total_residual, 1)
        bjet = np.empty((len(pairs), 4))
        for p, (j, column) in enumerate(pairs):
            if j == column:
                remainder = (
                    _residual_difference_jet(ajet[j], totaljet, complementjets[j], 1)
                    if mujet[j][0] > 0.5
                    else _mul(_mul(mujet[j], mujet[j]), rtotal)
                    - _residual_jet(ajet[j], residual[j], 1)
                )
                b = -_mul(ajet[j], otherjet[j]) + remainder
            else:
                b = _mul(ajet[j], mujet[column]) + _mul(_mul(mujet[j], mujet[column]), rtotal)
            bjet[p] = b
        # Differentiate a*log(y/mu) and the regular residual separately.
        # In dS=h*S+B*h the origin's +1 and -1 erase a tiny direction.
        r0total = _residual_jet(totaljet, total_residual, 0)
        sjet = np.empty((k, 4))
        for j in range(k):
            mu = mujet[j]
            ra, rb = -mu[1] / mu[0], -mu[2] / mu[0]
            ratiojet = np.array([ratios[j], ra, rb, -mu[3] / mu[0] + ra * rb])
            remainder = (
                _residual_difference_jet(ajet[j], totaljet, complementjets[j], 0)
                if mu[0] > 0.5
                else _mul(mu, r0total) - _residual_jet(ajet[j], residual[j], 0)
            )
            sjet[j] = _mul(ajet[j], ratiojet) + remainder
        qjet, zjet = [], []
        for j in range(k):
            qjet.append(
                _div(np.array([d1[j], d2[j] * v[j], d2[j] * w[j], d3[j] * v[j] * w[j]]), ajet[j])
            )
            zjet.append(
                _div(np.array([d2[j], d3[j] * v[j], d3[j] * w[j], d4[j] * v[j] * w[j]]), ajet[j])
            )
        for p, (j, column) in enumerate(pairs):
            h = _mul(_mul(qjet[j], qjet[column]), bjet[p])
            if j == column:
                h += _mul(zjet[j], sjet[j])
            result[row, p] = -multiplier[row] * h[3 if double else 1]
    return result


_curvature_batch = njit(cache=True, error_model="numpy")(_curvature_python)
