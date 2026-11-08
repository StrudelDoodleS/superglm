"""Log-logistic/Fisk law, natural scale/median and positive tail exponent."""

import numpy as np
from scipy.special import expit

from superglm.distributional.kernels._common import readonly
from superglm.distributional.kernels._declared import log_logistic as generated
from superglm.distributional.kernels._declared_adapter import (
    ScalarJetResult,
    finite_positive_exp,
    generated_density,
    log_response_ratio,
    narrow_windows,
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


def _rate(theta):
    # On t=log(y), l'=a*(1-2*expit(a*(t-log(s)))) and
    # |l''|<=a*a/2. Higher derivatives are bounded logistic polynomials;
    # 6*a covers the six spatial derivatives in three-node quadrature.
    return np.maximum(1.0, 6 * np.asarray(theta)[:, 1])


def window_jets(lower, upper, theta, *, order):
    return positive_window_jets(generated, lower, upper, theta, order=order, rate=_rate(theta))


def analytic_window_admitted(lower, upper, theta):
    return ~narrow_windows(lower, upper, _rate(theta))


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
