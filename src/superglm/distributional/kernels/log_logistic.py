"""Log-logistic/Fisk law, natural scale/median and positive tail exponent."""

from math import comb

import numpy as np
from scipy.special import expit

from superglm.distributional.kernels._common import readonly
from superglm.distributional.kernels._declared import log_logistic as generated
from superglm.distributional.kernels._declared_adapter import (
    ScalarJetResult,
    finite_positive_exp,
    generated_density,
    jet_order,
    log_response_ratio,
    positive_parameters,
    positive_window_jets,
    probabilities,
)

PARAMETERS = generated.PARAMETERS
COORDINATES = generated.COORDINATES


def carrier(y, multiplier):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.asarray(multiplier) * np.log(y)


def density_jets(y, theta, multiplier, *, order):
    y, theta = np.asarray(y, dtype=float), np.asarray(theta, dtype=float)
    result = generated_density(generated, y, theta, multiplier, order=order)
    valid = (
        result.valid & np.isfinite(y) & (y > 0) & np.all(np.isfinite(theta) & (theta > 0), axis=1)
    )
    if order:
        valid &= np.all((theta >= 2.0**-200) & (theta <= 2.0**200), axis=1)
    return ScalarJetResult(result.jets, result.carrier, valid)


def _finite_window_jets(lo, hi, theta, order):
    """Exact endpoint formula, differentiated in (log scale, shape) then scaled.

    Put l=log(lo/s), h=log(hi/s), w=log(hi/lo), d=a*w. The probability is
    expit(a*h)*expit(-a*l)*(1-exp(-d)). The gap d is formed independently
    of the two endpoint logits, including when the responses are adjacent.
    No interior response-valued quadrature nodes are formed.

    Endpoint logistic derivatives are v=p(1-p), D=v*(1-2p), E=v*(1-6v).
    For G=log(1-exp(-a*w)), set g=d/expm1(d). Its first four derivatives
    are g/a, -g(g+d)/a², g(g+d)(2g+d)/a³, and
    -g(g+d)[6g(g+d)+d²]/a⁴. This scaled form avoids overflow of 1/expm1(d).
    The log-scale derivative's difference of sigmoids is a sum of tanh's
    near zero, preserving the small score of a concentrated median window.
    Finally d/ds=s^-1 d/d(log s); the falling-factorial operators below
    perform every pure and mixed natural-coordinate transformation.
    """
    s, a = theta.T
    ell, h, w = log_response_ratio(lo, s), log_response_ratio(hi, s), log_response_ratio(hi, lo)
    zl, zh, d = a * ell, a * h, a * w
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        # For d<u the omitted -d/2 correction to log(d) is below u;
        # separate logs also retain a gap whose product underflows.
        gap = np.where(d < 2.0**-53, np.log(a) + np.log(w), np.log(-np.expm1(-d)))
        value = -np.logaddexp(0.0, -zh) - np.logaddexp(0.0, zl) + gap
        result = [value]
        if order == 0:
            return tuple(result)
        pl, ph = expit(zl), expit(zh)
        ql, qh = expit(-zl), expit(-zh)
        tl, th = np.tanh(0.5 * zl), np.tanh(0.5 * zh)
        t = np.where((zl * zh < 0) & (np.maximum(abs(zl), abs(zh)) > 1), pl - qh, 0.5 * (tl + th))
        vl, vh = pl * ql, ph * qh
        g = np.where(d < 2.0**-53, 1.0, d * np.exp(-d) / (-np.expm1(-d)))
        t10 = a * t
        t01 = h * qh - ell * pl + g / a
        result.append(np.column_stack((t10 / s, t01)))
        if order == 1:
            return tuple(result)
        t20 = -(a**2) * (vl + vh)
        t11 = t + a * (ell * vl + h * vh)
        t02 = -(ell**2 * vl + h**2 * vh) - (g * (g + d)) / a**2
        result.append(np.column_stack(((t20 - t10) / s**2, t11 / s, t02)))
        if order == 2:
            return tuple(result)
        dl, dh = -vl * tl, -vh * th
        t30 = a**3 * (dl + dh)
        t21 = -2 * a * (vl + vh) - a**2 * (ell * dl + h * dh)
        t12 = 2 * (ell * vl + h * vh) + a * (ell**2 * dl + h**2 * dh)
        t03 = -(ell**3 * dl + h**3 * dh) + (g * (g + d) * (2 * g + d)) / a**3
        result.append(
            np.column_stack(((t30 - 3 * t20 + 2 * t10) / s**3, (t21 - t11) / s**2, t12 / s, t03))
        )
        if order == 3:
            return tuple(result)
        el, eh = vl * (1 - 6 * vl), vh * (1 - 6 * vh)
        t40 = -(a**4) * (el + eh)
        t31 = 3 * a**2 * (dl + dh) + a**3 * (ell * el + h * eh)
        t22 = -2 * (vl + vh) - 4 * a * (ell * dl + h * dh) - a**2 * (ell**2 * el + h**2 * eh)
        t13 = 3 * (ell**2 * dl + h**2 * dh) + a * (ell**3 * el + h**3 * eh)
        t04 = -(ell**4 * el + h**4 * eh) - (g * (g + d) * (6 * g * (g + d) + d**2)) / a**4
        result.append(
            np.column_stack(
                (
                    (t40 - 6 * t30 + 11 * t20 - 6 * t10) / s**4,
                    (t31 - 3 * t21 + 2 * t11) / s**3,
                    (t22 - t12) / s**2,
                    t13 / s,
                    t04,
                )
            )
        )
    return tuple(result)


def _window_derivatives_admitted(lo, hi, theta, order):
    """Admission also accounts for use as a truncation normalizer.

    For small a*log(y/s), finite-window and density pure shape jets share
    (-1)^(j-1)(j-1)!/a^j. Subtraction loses the conditional O(log(y/s)^j)
    remainder. With reach=a*max(1,|l|,|h|), the relative rounding scale is
    O(u/reach^j). Require reach>=2^-8 for core jets (64u/reach² < sqrt(u))
    and >=1/8 for fourth jets. This is a conservative numerical restriction,
    not a change to the positive mathematical shape domain. Order zero is
    unaffected. No finite placeholder is presented as a valid derivative.
    """
    reach = theta[:, 1] * np.maximum(
        1.0,
        np.maximum(
            abs(log_response_ratio(lo, theta[:, 0])), abs(log_response_ratio(hi, theta[:, 0]))
        ),
    )
    return reach >= (0.125 if order > 2 else 2.0**-8)


def window_jets(lower, upper, theta, *, order):
    order = jet_order(order)
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    lo, hi = (np.broadcast_to(x, (n,)).astype(float) for x in (lower, upper))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)):
        raise ValueError("window endpoints require lower <= upper and no NaN")
    finite = (lo > 0) & (hi > lo) & np.isfinite(hi)
    result = (np.empty(n),) + tuple(np.empty((n, comb(1 + d, d))) for d in range(1, order + 1))
    valid = np.ones(n, dtype=bool)
    for closed_interval, selected in ((True, finite), (False, ~finite)):
        rows = np.flatnonzero(selected)
        if not len(rows):
            continue
        if closed_interval:
            sub = _finite_window_jets(lo[rows], hi[rows], theta[rows], order)
            if order:
                valid[rows] &= _window_derivatives_admitted(lo[rows], hi[rows], theta[rows], order)
        else:
            sub, valid[rows] = positive_window_jets(
                generated, lo[rows], hi[rows], theta[rows], order=order, rate=np.ones(len(rows))
            )
        for target, source in zip(result, sub, strict=True):
            target[rows] = source
    valid &= np.all(np.isfinite(theta) & (theta > 0), axis=1)
    if order:
        valid &= np.all((theta >= 2.0**-200) & (theta <= 2.0**200), axis=1)
    for channel in result:
        valid &= np.isfinite(channel) if channel.ndim == 1 else np.all(np.isfinite(channel), axis=1)
    return result, valid


def analytic_window_admitted(lower, upper, theta):
    theta = np.asarray(theta)
    lo, hi = (np.broadcast_to(x, (len(theta),)) for x in (lower, upper))
    finite = (lo > 0) & (hi > lo) & np.isfinite(hi)
    admitted = np.ones(len(theta), dtype=bool)
    admitted[finite] = _window_derivatives_admitted(lo[finite], hi[finite], theta[finite], 4)
    return admitted


def cdf(y, theta):
    theta = positive_parameters(theta)
    y = np.broadcast_to(np.asarray(y, dtype=float), (len(theta),))
    if np.isnan(y).any():
        raise ValueError("CDF response must not be NaN")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        z = theta[:, 1] * log_response_ratio(np.maximum(y, 0.0), theta[:, 0])
    return readonly(expit(z))


def quantile(p, theta):
    theta = positive_parameters(theta)
    p = probabilities(p, len(theta))
    scale, shape = theta.T
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        shift = (np.log(p) - np.log1p(-p)) / shape
        near = np.abs(shift) <= 0.5
        value = np.empty(len(theta))
        # Preserve the scale and its small displacement independently; exp(log(s))
        # can already lie below s and destroy quantile ordering around the median.
        value[near] = scale[near] + scale[near] * np.expm1(shift[near])
        value[~near] = finite_positive_exp(np.log(scale[~near]) + shift[~near], "quantile")
    if np.any(~np.isfinite(value) | (value <= 0)):
        raise ValueError("finite quantile is unresolved in the numerical range")
    return readonly(value)


def positive_moment_exists(theta, r):
    if not np.isfinite(r) or r <= 0:
        raise ValueError("moment order must be finite and positive")
    return positive_parameters(theta)[:, 1] > r


def _log_mean_loading(a):
    r = 1 / a
    # Close to a=1, (a-1)/a retains the distance from sin(pi)'s zero.
    # Away from that endpoint sinc directly retains the limit as a→infinity.
    near = a < 2
    result = np.empty_like(a)
    result[near] = (
        np.log(np.pi) - np.log(a[near]) - np.log(np.sin(np.pi * ((a[near] - 1) / a[near])))
    )
    result[~near] = -np.log(np.sinc(r[~near]))
    return result


def mean(theta):
    theta = positive_parameters(theta)
    finite = positive_moment_exists(theta, 1.0)
    result = np.full(len(theta), np.inf)
    s, a = theta[finite].T
    result[finite] = finite_positive_exp(np.log(s) + _log_mean_loading(a), "mean")
    return readonly(result)


def variance(theta):
    theta = positive_parameters(theta)
    finite = positive_moment_exists(theta, 2.0)
    s, a = theta[finite].T
    x = np.pi / a
    small = x <= 0.01
    log_factor = np.empty_like(x)
    # Var/mean^2 = tan(x)/x - 1. Its series avoids subtracting equal
    # moments for concentrated laws. At x<=.01 the omitted positive x^10
    # term is <.027*x^8 relative to x^2/3 (<3e-18). The log expression
    # also avoids underflow of x^2 before monetary scale rescales it.
    z = x[small] ** 2
    log_factor[small] = 2 * np.log(x[small]) + np.log(
        1 / 3 + z * (2 / 15 + z * (17 / 315 + z * (62 / 2835)))
    )
    near = (a < 4) & ~small
    delta = np.pi * ((a[near] - 2) / (2 * a[near]))
    log_factor[near] = np.log(1 / (np.tan(delta) * x[near]) - 1)
    body = ~small & ~near
    log_factor[body] = np.log(np.tan(x[body]) / x[body] - 1)
    result = np.full(len(theta), np.inf)
    result[finite] = finite_positive_exp(
        2 * np.log(s) + 2 * _log_mean_loading(a) + log_factor, "variance"
    )
    return readonly(result)
