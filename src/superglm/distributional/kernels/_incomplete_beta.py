"""Order-two log beta tails in ``(a, b, logit(x))``.

The DLMF 8.17.22--23 fraction is differentiated on raw jets, internally in
log shapes. Every requested channel must settle for three full iterations;
an integer shape's zero *value* coefficient does not end its derivatives.
This is an empirically tested analytic route, not a global error certificate.
Failure (including the work cap) returns NaNs, never the last iterate.

The prefactor uses the two-component deviance and ``gnorm`` instead of a
difference of large log gammas. All arithmetic is portable binary64.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from superglm.distributional.kernels._response_tails import Jet, _checked_core_order
from superglm.distributional.kernels._severity_primitives import gnorm, psires_0123

# Compiled callees' source/constant digest; the generator's caller list is
# maintained by the integration owner. Numba otherwise keys only this file.
_CALLEE_DIGEST = "b51ca08b6124df2a6867c9dafbb618962377ba619a33115b1ce694b305921446"

MAX_TERMS = 100_000
_U = 2.0**-53
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@njit(cache=True)
def _log1pmx(x):
    if abs(x) > 0.125:
        return math.log1p(x) - x
    term = -x * x / 2.0
    value = term
    for k in range(3, 40):
        term *= -x * (k - 1.0) / k
        value += term
        if abs(term) <= _U * abs(value):
            break
    return value


@njit(cache=True)
def _logistic(z):
    e = math.exp(-abs(z))
    small = e / (1.0 + e)
    large = 1.0 / (1.0 + e)
    soft = math.log1p(e)
    if z >= 0.0:
        return large, small, -soft, -z - soft
    return small, large, z - soft, -soft


@njit(cache=True)
def _deviance_parts(a, b, z):
    """``a log(A/x)+b log(B/y)`` and the two log ratios, A=a/(a+b).

    Near the mean, cancelling linear terms are removed analytically before
    summing two nonnegative ``-log1pmx`` terms. Ratios of the small component
    are retained even when the large probability rounds to one.
    """
    t = a + b
    A, B = a / t, b / t
    delta = z - (math.log(a) - math.log(b))
    if abs(delta) <= 0.5:
        em = math.expm1(delta)
        v = em / (1.0 + A * em)
        da, db = B * v, -A * v
        deviance = -a * _log1pmx(da) - b * _log1pmx(db)
        lx = -math.log1p(B * math.expm1(-delta))
        ly = -math.log1p(A * em)
    else:
        _, _, logx, logy = _logistic(z)
        lx = logx + math.log1p(b / a)
        ly = logy + math.log1p(a / b)
        deviance = -a * lx - b * ly
    return deviance, lx, ly


@njit(cache=True)
def _constant(value):
    out = np.zeros(10)
    out[0] = value
    return out


@njit(cache=True)
def _mul(f, g, order):
    out = _constant(f[0] * g[0])
    if order >= 1:
        for i in range(3):
            out[1 + i] = f[1 + i] * g[0] + f[0] * g[1 + i]
    if order == 2:
        for col, (i, j) in enumerate(_PAIRS):
            out[4 + col] = (
                f[4 + col] * g[0] + f[0] * g[4 + col] + f[1 + i] * g[1 + j] + f[1 + j] * g[1 + i]
            )
    return out


@njit(cache=True)
def _unary(f, value, first, second, order):
    out = _constant(value)
    if order >= 1:
        out[1:4] = first * f[1:4]
    if order == 2:
        for col, (i, j) in enumerate(_PAIRS):
            out[4 + col] = first * f[4 + col] + second * f[1 + i] * f[1 + j]
    return out


@njit(cache=True, error_model="numpy")
def _inverse(f, order):
    # Express derivatives as ratios, avoiding cubed subnormal denominators.
    out = _constant(1.0 / f[0])
    if order >= 1:
        out[1:4] = -(f[1:4] / f[0]) * out[0]
    if order == 2:
        for col, (i, j) in enumerate(_PAIRS):
            out[4 + col] = (2 * (f[1 + i] / f[0]) * (f[1 + j] / f[0]) - f[4 + col] / f[0]) * out[0]
    return out


@njit(cache=True)
def _log(f, order):
    out = _constant(math.log(f[0]))
    if order >= 1:
        out[1:4] = f[1:4] / f[0]
    if order == 2:
        for col, (i, j) in enumerate(_PAIRS):
            out[4 + col] = f[4 + col] / f[0] - out[1 + i] * out[1 + j]
    return out


@njit(cache=True)
def _prefactor(a, b, z, order):
    t = a + b
    x, y, _, _ = _logistic(z)
    deviance, lx, ly = _deviance_parts(a, b, z)
    out = _constant(-deviance + gnorm(a) + gnorm(b) - gnorm(t) - math.log(a))
    if order == 0:
        return out
    ra, rb, rt = psires_0123(a), psires_0123(b), psires_0123(t)
    ga = a * (lx - ra[0] + rt[0])
    gb = b * (ly - rb[0] + rt[0])
    out[1] = ga - 1.0
    out[2] = gb
    # a*y-b*x, evaluated without subtracting nearly equal large products.
    delta = z - (math.log(a) - math.log(b))
    if abs(delta) <= 0.5:
        out[3] = -a * (b / t) * math.expm1(delta) / (1 + (a / t) * math.expm1(delta))
    else:
        out[3] = a * y - b * x
    if order == 2:
        ab_t = a * (b / t)
        out[4] = ga - ab_t + a * (a * (rt[1] - ra[1]))
        out[5] = ab_t + a * (b * rt[1])
        out[6] = a * y
        out[7] = gb - ab_t + b * (b * (rt[1] - rb[1]))
        out[8] = -b * x
        out[9] = -(t * x) * y
    return out


@njit(cache=True, error_model="numpy")
def _fraction(a, b, z, order, max_terms):
    aa, bb = _constant(a), _constant(b)
    aa[1], aa[4] = a, a
    bb[2], bb[7] = b, b
    x, y, _, _ = _logistic(z)
    xx = _constant(x)
    xx[3], xx[9] = x * y, x * y * (y - x)
    one = _constant(1.0)
    d = _inverse(one - _mul(_mul(aa + bb, xx, order), _inverse(aa + one, order), order), order)
    c, h = one.copy(), d.copy()
    settled = 0
    n = 1 if order == 0 else (4 if order == 1 else 10)
    for m in range(1, max_terms + 1):
        previous = h.copy()
        mm = _constant(float(m))
        m2 = _constant(2.0 * m)
        even = _mul(_mul(mm, bb - mm, order), xx, order)
        even = _mul(even, _inverse(_mul(aa + m2 - one, aa + m2, order), order), order)
        odd = -_mul(_mul(aa + mm, aa + bb + mm, order), xx, order)
        odd = _mul(odd, _inverse(_mul(aa + m2, aa + m2 + one, order), order), order)
        for coefficient in (even, odd):
            d = _inverse(one + _mul(coefficient, d, order), order)
            c = one + _mul(coefficient, _inverse(c, order), order)
            h = _mul(h, _mul(d, c, order), order)
        if not np.all(np.isfinite(h[:n])) or h[0] <= 0.0:
            return np.full(10, np.nan), m
        # Channels are in dimensionless log shapes and log odds. A tiny
        # coefficient is judged against the value as well as its own scale.
        good = True
        for j in range(n):
            scale = max(abs(h[0]), abs(h[j]))
            if abs(h[j] - previous[j]) > 32 * _U * scale:
                good = False
        settled = settled + 1 if good else 0
        if settled >= 3:
            result = _prefactor(a, b, z, order) + _log(h, order)
            if not np.all(np.isfinite(result[:n])) or result[0] > 0.0:
                return np.full(10, np.nan), m
            return result, m
    return np.full(10, np.nan), max_terms


@njit(cache=True)
def _complement(f, order):
    # The measured fraction envelope does not resolve a complement whose
    # amplification exceeds 1e4. A count-law consumer has a positive-sum
    # fallback (including the small-size atom at zero); no rounded CDF is
    # turned into a falsely precise tiny tail here.
    if not f[0] <= -1e-4:
        return np.full(10, np.nan)
    value = math.log(-math.expm1(f[0])) if f[0] > -math.log(2.0) else math.log1p(-math.exp(f[0]))
    ratio = math.exp(f[0] - value)
    return _unary(f, value, -ratio, -ratio * (1 + ratio), order)


@njit(cache=True)
def _beta_log_shape_row(a, b, z, upper, order, max_terms):
    """Raw log-shape/log-odds jet and fraction work count (no Python state)."""
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 and math.isfinite(z)):
        return np.full(10, np.nan), 0
    # DLMF's convergence switch, using log odds to retain x near one.
    swap = z > math.log(a + 1.0) - math.log(b + 1.0)
    if swap:
        f, cells = _fraction(b, a, -z, order, max_terms)
        f = f[np.array([0, 2, 1, 3, 7, 5, 8, 4, 6, 9])]
        f[3], f[6], f[8] = -f[3], -f[6], -f[8]
    else:
        f, cells = _fraction(a, b, z, order, max_terms)
    if swap != upper:
        f = _complement(f, order)
    return f, cells


def log_beta_tail_logit_jets(a, b, logit_x, *, upper: bool, order: int) -> Jet:
    """Log regularized beta tail and raw packed partials in ``(a,b,logit_x)``.

    Broadcast scalar or one-dimensional arguments. Unresolved rows are NaN.
    Orders are exactly integers 0--2; no numerical differencing is used.
    """
    order = _checked_core_order(order)
    arrays = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=float)) for v in (a, b, logit_x))
    )
    if any(v.ndim != 1 for v in arrays):
        raise ValueError("beta arguments must be scalars or row vectors")
    out = np.empty((len(arrays[0]), 10))
    for row, (aa, bb, zz) in enumerate(zip(*arrays, strict=True)):
        out[row], _ = _beta_log_shape_row(aa, bb, zz, upper, order, MAX_TERMS)
    if order == 0:
        return (out[:, 0],)
    a, b, _ = arrays
    scale = np.column_stack((a, b, np.ones_like(a)))
    score = out[:, 1:4] / scale
    if order == 1:
        return out[:, 0], score
    hessian = out[:, 4:].copy()
    hessian[:, 0] -= out[:, 1]
    hessian[:, 3] -= out[:, 2]
    for col, (i, j) in enumerate(_PAIRS):
        hessian[:, col] = (hessian[:, col] / scale[:, i]) / scale[:, j]
    return out[:, 0], score, hessian
