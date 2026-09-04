"""Independent references for the epsilon-skew two-piece normal.

Nothing here shares code with the kernel: the density is written in its
textbook piecewise form, its normalisation is checked by quadrature rather
than asserted, and the expectations that the expected information must match
are taken by quadrature against that same density.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats

PACKED = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def piecewise_density(w, eps):
    """``f_W``: the right piece is the wide one for ``eps > 0``."""
    return stats.norm.pdf(w / (1.0 - eps)) if w < 0.0 else stats.norm.pdf(w / (1.0 + eps))


def normalising_mass(eps):
    """``int f_W``, by quadrature: the family carries no extra constant."""
    left = integrate.quad(lambda w: piecewise_density(w, eps), -60.0, 0.0)[0]
    right = integrate.quad(lambda w: piecewise_density(w, eps), 0.0, 60.0)[0]
    return left + right


def textbook_log_density(t, mu, sigma, eps):
    """``log f_T`` for the real-line variate, from the piecewise density."""
    return float(np.log(piecewise_density((t - mu) / sigma, eps) / sigma))


def expectation_by_quadrature(integrand, eps):
    """``E[g(W)]`` against the piecewise density, split at the kink."""
    left = integrate.quad(lambda w: integrand(w) * stats.norm.pdf(w / (1.0 - eps)), -40.0, 0.0)[0]
    right = integrate.quad(lambda w: integrand(w) * stats.norm.pdf(w / (1.0 + eps)), 0.0, 40.0)[0]
    return left + right


def sn2_log_density(t, mu, sigma_sn2, nu):
    """gamlss ``SN2`` log density from its published form (Fernandez-Steel).

    ``f = (c/sigma) exp(-(nu z)^2/2)`` for ``t < mu`` and
    ``(c/sigma) exp(-(z/nu)^2/2)`` for ``t >= mu``, ``z = (t - mu)/sigma``,
    ``c = sqrt(2) nu / (sqrt(pi) (1 + nu^2))``.  Written here so the mapping to
    our epsilon-skew coordinates is pinned without R.
    """
    z = (t - mu) / sigma_sn2
    c = math.sqrt(2.0) * nu / (math.sqrt(math.pi) * (1.0 + nu * nu))
    exponent = -0.5 * (nu * z) ** 2 if t < mu else -0.5 * (z / nu) ** 2
    return math.log(c / sigma_sn2) + exponent


def mp_log_density(t, mu, sigma, eps, dps=50):
    """Optimising log density ``-log sigma - u^2/2`` at high precision."""
    import mpmath as mp

    with mp.workdps(max(dps, mp.mp.dps)):
        t, mu, sigma, eps = (mp.mpf(v) for v in (t, mu, sigma, eps))
        z = (t - mu) / sigma
        u = z / ((1 - eps) if z < 0 else (1 + eps))
        return -mp.log(sigma) - u**2 / 2


def mp_log_mean_loading(sigma, eps, dps=50):
    """``log K`` in closed form at high precision."""
    import mpmath as mp

    with mp.workdps(max(dps, mp.mp.dps)):
        sigma, eps = mp.mpf(sigma), mp.mpf(eps)
        a1, a2 = sigma * (1 - eps), sigma * (1 + eps)
        left = (1 - eps) * mp.e ** (a1**2 / 2) * mp.ncdf(-a1)
        right = (1 + eps) * mp.e ** (a2**2 / 2) * mp.ncdf(a2)
        return mp.log(left + right)


def mp_log_mean_loading_by_quadrature(sigma, eps, dps=30):
    """``log E[e^(sigma W)]`` straight from the density: no closed form used."""
    import mpmath as mp

    with mp.workdps(dps):
        sigma, eps = mp.mpf(sigma), mp.mpf(eps)

        def phi(w):
            return mp.e ** (-(w**2) / 2) / mp.sqrt(2 * mp.pi)

        left = mp.quad(lambda w: mp.e ** (sigma * w) * phi(w / (1 - eps)), [-mp.inf, 0])
        right = mp.quad(lambda w: mp.e ** (sigma * w) * phi(w / (1 + eps)), [0, mp.inf])
        return mp.log(left + right)


def mp_mean_log_density(y, mean, sigma, eps, dps=50):
    """Mean-form optimising log density, reparametrised inside the reference."""
    import mpmath as mp

    with mp.workdps(max(dps, mp.mp.dps)):
        mu = mp.log(mp.mpf(mean)) - mp_log_mean_loading(sigma, eps, dps=dps)
        return mp_log_density(mp.log(mp.mpf(y)), mu, sigma, eps, dps=dps)


def mp_packed(function, point, dps=50):
    """``(score, packed Hessian)`` of a three-argument function at ``point``."""
    import mpmath as mp

    orders = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    with mp.workdps(dps):
        pt = tuple(mp.mpf(v) for v in point)
        first = [float(mp.diff(function, pt, o)) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        second = [float(mp.diff(function, pt, o)) for o in orders]
    return first, second


def mp_loading_derivatives(sigma, eps, dps=50):
    """``(K_s, K_e, K_ss, K_se, K_ee)`` by high-precision differentiation."""
    import mpmath as mp

    with mp.workdps(dps):
        pt = (mp.mpf(sigma), mp.mpf(eps))
        orders = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
        return [float(mp.diff(lambda a, b: mp_log_mean_loading(a, b), pt, o)) for o in orders]
