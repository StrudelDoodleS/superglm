"""Severity families declared for the generator: the whole per-family input.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.
The ``NAIVE`` declarations write the same laws in textbook form, without the
stable primitives, to measure what the primitives buy in the tails.
"""

from __future__ import annotations

import sympy as sp
from scipy import special, stats

from sevgen import Declaration, log1p
from sevprims import gdev, gnorm, log1mexp, softplus

log = sp.log


def _weibull_z(y, scale, shape):
    return shape * log(y / scale)


WEIBULL = Declaration(
    name="weibull",
    parameters=("scale", "shape"),
    log_density=lambda y, scale, shape: (
        log(shape) - log(y) + _weibull_z(y, scale, shape) - sp.exp(_weibull_z(y, scale, shape))
    ),
    log_survival=lambda c, scale, shape: -sp.exp(_weibull_z(c, scale, shape)),
    log_cdf=lambda c, scale, shape: log1mexp(sp.exp(_weibull_z(c, scale, shape))),
    scipy_law=lambda scale, shape: stats.weibull_min(c=shape, scale=scale),
)


def _weibull_mean_z(y, mean, shape):
    return shape * (log(y / mean) + sp.loggamma(1 + 1 / shape))


WEIBULL_MEAN = Declaration(
    name="weibull_mean",
    parameters=("mean", "shape"),
    log_density=lambda y, mean, shape: (
        log(shape)
        - log(y)
        + _weibull_mean_z(y, mean, shape)
        - sp.exp(_weibull_mean_z(y, mean, shape))
    ),
    log_survival=lambda c, mean, shape: -sp.exp(_weibull_mean_z(c, mean, shape)),
    log_cdf=lambda c, mean, shape: log1mexp(sp.exp(_weibull_mean_z(c, mean, shape))),
    scipy_law=lambda mean, shape: stats.weibull_min(
        c=shape, scale=mean / special.gamma(1.0 + 1.0 / shape)
    ),
)


def _burr_z(y, scale, c):
    return c * log(y / scale)


BURR12 = Declaration(
    name="burr12",
    parameters=("scale", "c", "k"),
    log_density=lambda y, scale, c, k: (
        log(c) + log(k) - log(y) + _burr_z(y, scale, c) - (k + 1) * softplus(_burr_z(y, scale, c))
    ),
    log_survival=lambda y, scale, c, k: -k * softplus(_burr_z(y, scale, c)),
    log_cdf=lambda y, scale, c, k: log1mexp(k * softplus(_burr_z(y, scale, c))),
    scipy_law=lambda scale, c, k: stats.burr12(c=c, d=k, scale=scale),
)

LOG_LOGISTIC = Declaration(
    name="log_logistic",
    parameters=("scale", "shape"),
    log_density=lambda y, scale, shape: (
        log(shape) - log(y) + _burr_z(y, scale, shape) - 2 * softplus(_burr_z(y, scale, shape))
    ),
    log_survival=lambda y, scale, shape: -softplus(_burr_z(y, scale, shape)),
    log_cdf=lambda y, scale, shape: -softplus(-_burr_z(y, scale, shape)),
    scipy_law=lambda scale, shape: stats.fisk(c=shape, scale=scale),
)

# Lomax is Burr XII at c = 1: log1p(y/scale) = softplus(log(y/scale)), whose
# derivatives are bounded expit products (compare LOMAX_LOG1P below).
LOMAX = Declaration(
    name="lomax",
    parameters=("scale", "shape"),
    log_density=lambda y, scale, shape: (
        log(shape) - log(y) + log(y / scale) - (shape + 1) * softplus(log(y / scale))
    ),
    log_survival=lambda y, scale, shape: -shape * softplus(log(y / scale)),
    log_cdf=lambda y, scale, shape: log1mexp(shape * softplus(log(y / scale))),
    scipy_law=lambda scale, shape: stats.lomax(c=shape, scale=scale),
)

# GammaLS's own law and parameters: mean and coefficient of variation, a = 1/cv^2.
GAMMA = Declaration(
    name="gamma",
    parameters=("mean", "cv"),
    log_density=lambda y, mean, cv: (
        gnorm(cv**-2) - cv**-2 * gdev(y, mean) - log(y)
    ),
    scipy_law=lambda mean, cv: stats.gamma(a=cv**-2.0, scale=mean * cv**2.0),
)

FAMILIES = (WEIBULL, WEIBULL_MEAN, BURR12, LOG_LOGISTIC, LOMAX, GAMMA)

# ---------------------------------------------------------------- textbook forms

NAIVE_GAMMA = Declaration(
    name="naive_gamma",
    parameters=("mean", "cv"),
    log_density=lambda y, mean, cv: (
        cv**-2 * log(cv**-2)
        - sp.loggamma(cv**-2)
        + (cv**-2 - 1) * log(y)
        - cv**-2 * log(mean)
        - cv**-2 * y / mean
    ),
    scipy_law=GAMMA.scipy_law,
)

NAIVE_BURR12 = Declaration(
    name="naive_burr12",
    parameters=("scale", "c", "k"),
    log_density=lambda y, scale, c, k: (
        log(c) + log(k) - log(scale) + (c - 1) * log(y / scale)
        - (k + 1) * log(1 + (y / scale) ** c)
    ),
    log_survival=lambda y, scale, c, k: -k * log(1 + (y / scale) ** c),
    log_cdf=lambda y, scale, c, k: log(1 - (1 + (y / scale) ** c) ** (-k)),
    scipy_law=BURR12.scipy_law,
)

NAIVE_LOMAX = Declaration(
    name="naive_lomax",
    parameters=("scale", "shape"),
    log_density=lambda y, scale, shape: (
        log(shape) - log(scale) - (shape + 1) * log(1 + y / scale)
    ),
    log_survival=lambda y, scale, shape: -shape * log(1 + y / scale),
    log_cdf=lambda y, scale, shape: log(1 - (1 + y / scale) ** (-shape)),
    scipy_law=LOMAX.scipy_law,
)

LOMAX_LOG1P = Declaration(
    name="lomax_log1p",
    parameters=("scale", "shape"),
    log_density=lambda y, scale, shape: (
        log(shape) - log(scale) - (shape + 1) * log1p(y / scale)
    ),
    log_survival=lambda y, scale, shape: -shape * log1p(y / scale),
    log_cdf=lambda y, scale, shape: log1mexp(shape * log1p(y / scale)),
    scipy_law=LOMAX.scipy_law,
)

NAIVE = (NAIVE_GAMMA, NAIVE_BURR12, NAIVE_LOMAX, LOMAX_LOG1P)
