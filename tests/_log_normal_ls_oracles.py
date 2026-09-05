"""Independent references for the log-normal kernel tests.

Nothing here is allowed to reuse the kernel's algebra: the density comes from
``scipy.stats.lognorm``, the derivatives from mpmath's numerical
differentiation of a textbook log density, and the mean loading from a
quadrature of ``E[exp(sigma Z)]`` that never mentions ``sigma**2 / 2``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

PACKED = ((0, 0), (0, 1), (1, 1))
HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def scipy_log_density(y, mu, sigma):
    """``log f(y)`` from scipy's own log-normal implementation."""
    return stats.lognorm(s=sigma, scale=math.exp(mu)).logpdf(np.asarray(y, dtype=float))


def _mp_optimizing(mp, y, mu, sigma):
    w = (mp.log(y) - mu) / sigma
    return -mp.log(sigma) - w * w / 2


def mp_log_optimizing(mp, y, mu, sigma):
    """``-log sigma - w^2 / 2`` at 60 digits."""
    with mp.workdps(60):
        return float(_mp_optimizing(mp, mp.mpf(y), mp.mpf(mu), mp.mpf(sigma)))


def mp_location_derivatives(mp, y, mu, sigma):
    """``(score, packed Hessian)`` in ``(mu, sigma)`` by mpmath differentiation."""
    with mp.workdps(60):

        def f(location, scale):
            return _mp_optimizing(mp, mp.mpf(y), location, scale)

        point = (mp.mpf(mu), mp.mpf(sigma))
        score = [float(mp.diff(f, point, order)) for order in ((1, 0), (0, 1))]
        hessian = [float(mp.diff(f, point, order)) for order in ((2, 0), (1, 1), (0, 2))]
    return np.array(score), np.array(hessian)


def mp_mean_derivatives(mp, y, mean, sigma):
    """``(score, packed Hessian)`` in ``(m, sigma)`` with ``mu = log m - log C``."""
    with mp.workdps(60):

        def f(first, scale):
            return _mp_optimizing(mp, mp.mpf(y), mp.log(first) - scale * scale / 2, scale)

        point = (mp.mpf(mean), mp.mpf(sigma))
        score = [float(mp.diff(f, point, order)) for order in ((1, 0), (0, 1))]
        hessian = [float(mp.diff(f, point, order)) for order in ((2, 0), (1, 1), (0, 2))]
    return np.array(score), np.array(hessian)


def mp_log_mean_loading(mp, sigma):
    """``(log C, dlog C/dsigma, d2 log C/dsigma2)`` from a quadrature of ``E[e^(sigma Z)]``.

    The value is the definition of the loading; the two derivatives are central
    differences of that same quadrature at 30 digits with ``h = 1e-6``, whose
    truncation error is far below the ``1e-12`` the tests assert.
    """
    with mp.workdps(30):

        def log_loading(scale):
            integral = mp.quad(
                lambda z: mp.exp(scale * z - z * z / 2) / mp.sqrt(2 * mp.pi), [-30, 0, 30]
            )
            return mp.log(integral)

        s = mp.mpf(sigma)
        h = mp.mpf("1e-6")
        value, plus, minus = log_loading(s), log_loading(s + h), log_loading(s - h)
        return (
            float(value),
            float((plus - minus) / (2 * h)),
            float((plus - 2 * value + minus) / (h * h)),
        )
