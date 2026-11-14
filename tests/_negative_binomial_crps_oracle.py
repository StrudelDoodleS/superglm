"""Independent NB2 CRPS: the lattice sum ``sum_k (F(k) - 1{y <= k})^2`` in mpmath.

No production code is imported. The pmf comes from the ratio
``f(k + 1) / f(k) = r (k + theta) / (k + 1)``, ``r = mu / (mu + theta)``, from
``f(0) = (1 - r)^theta``. Survival probabilities are summed down from the
cut ``K``, never formed as ``1 - F``, so a rare claim's ``P(Y > 0)`` keeps its
digits. Past ``K`` every pmf ratio is at most ``q_K = max(r, r (K + theta) / (K + 1))
< 1``, so ``S(K) <= f(K) q_K / (1 - q_K)``. Leaving ``S(K)`` out of every
``S(k)`` and the cells past ``K`` out of the sum moves the score by at most
``S(K) (2 mu + (K + 1) S(K)) + S(K)^2 / (1 - q_K^2)``; the sum stops once that is
below ``10^-35`` of ``min(f(0), f(1))^2``, which the score exceeds (below ``y = 1``
it holds ``S(0)^2 >= f(1)^2``, above it ``F(0)^2 = f(0)^2`` over a whole cell).

``nb2_crps_by_quadrature`` serves laws too wide for the lattice. It writes the
score as ``E|Y - y| - E|Y - Y'| / 2`` (Gneiting and Raftery 2007, eq. 21), with
``E|Y - y| = mu - y + 2 (y F(k) - mu G(k - 1))`` (``k f(k) = mu g(k - 1)``, ``G``
the NB2 CDF of size ``theta + 1``) and ``E|Y - Y'| / 2 = (mu / p) 2F1(theta + 1,
1/2; 2; -4r / p^2)`` (Wei and Held 2014). Euler's integral gives the ``2F1`` and
the beta density's integral gives each regularized incomplete beta, both by
``mp.quad`` with breakpoints at the integrand's peak and scale.
"""

from __future__ import annotations

import math

import mpmath as mp

_DIGITS = 60
# At a fixed precision: a module importing this one may have changed the ambient one.
with mp.workprec(53):
    _RELATIVE_TAIL = mp.mpf(10) ** -35


def nb2_crps(mean: float, size: float, response: float) -> float:
    """``CRPS(NB2(mean, size), response)`` for a real response, to binary64."""
    with mp.workdps(_DIGITS):
        mu, theta, y = mp.mpf(mean), mp.mpf(size), mp.mpf(response)
        r = mu / (mu + theta)
        pmf = [mp.exp(theta * mp.log(theta / (mu + theta)))]
        pmf.append(pmf[0] * r * theta)
        scale = min(pmf) ** 2
        floor_y = int(mp.floor(y))
        while True:
            k = len(pmf) - 1
            ratio = max(r, r * (k + theta) / (k + 1))
            if k >= max(floor_y, int(mu)) + 1 and ratio < 1:
                tail = pmf[-1] * ratio / (1 - ratio)  # bounds S(k)
                moved = tail * (2 * mu + (k + 1) * tail) + tail**2 / (1 - ratio**2)
                if moved < _RELATIVE_TAIL * scale:
                    break
            pmf.append(pmf[-1] * r * (k + theta) / (k + 1))
        cut = len(pmf) - 1
        survival = [mp.mpf(0)] * (cut + 1)  # S(k) less the uncounted tail past the cut
        for k in range(cut - 1, -1, -1):
            survival[k] = survival[k + 1] + pmf[k + 1]
        total = mp.mpf(0) if y >= 0 else -y
        cdf = mp.mpf(0)
        for k in range(cut):
            cdf += pmf[k]
            if k + 1 <= y:  # the whole cell [k, k + 1) lies below y
                total += cdf**2
            elif k >= y:  # the whole cell lies at or above y
                total += survival[k] ** 2
            else:  # y splits the cell
                total += (y - k) * cdf**2 + (k + 1 - y) * survival[k] ** 2
        return float(total)


def _beta_integral(a, b, lower, upper):
    """``int t^(a-1) (1-t)^(b-1) dt / B(a, b)`` over ``[lower, upper]``, ``a >= 1``, ``upper <= 1/2``."""
    log_beta = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)

    def density(t):
        return mp.exp((a - 1) * mp.log(t) + (b - 1) * mp.log1p(-t) - log_beta)

    points = {lower, upper}
    mode = (a - 1) / (a + b - 2) if a + b > 2 else mp.mpf(0)
    if 0 < mode < 1:
        spread = mp.sqrt(mode * (1 - mode) / (a + b - 2))
        for j in (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64):
            points.update({mode - j * spread, mode + j * spread})
    for end in (lower, upper):  # the decay scale where the peak lies outside the range
        slope = abs((a - 1) / end - (b - 1) / (1 - end)) if 0 < end < 1 else 0
        if slope:
            for c in (1, 4, 16, 64, 256, 1024):
                points.update({end - c / slope, end + c / slope})
    nodes = sorted(t for t in points if lower <= t <= upper)
    return mp.quad(density, nodes)


def _regularized_beta(a, b, x, y):
    """``I_x(a, b)``, ``y = 1 - x``, integrated on the smaller of ``x`` and ``y``.

    ``I_x(a, b) = 1 - I_y(b, a)`` (DLMF 8.17.4); the larger of the two would
    round to one at a law far from even odds.  For ``a < 1`` the integral is the
    upper one, away from ``t^(a-1)``'s pole.
    """
    if x > y:
        return 1 - _regularized_beta(b, a, y, x)
    if a < 1:
        return 1 - _upper_beta_with_small_a(a, b, x)
    return _beta_integral(a, b, mp.mpf(0), x)


def _upper_beta_with_small_a(a, b, x):
    """``int_x^1 t^(a-1) (1-t)^(b-1) dt / B(a, b)`` in ``t = x e^u``, where it is smooth."""
    log_beta = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)

    def density(u):
        t = min(x * mp.exp(u), mp.mpf(1))  # the top node rounds to just past one
        return mp.exp(a * mp.log(t) + (b - 1) * mp.log1p(-t) - log_beta) if t < 1 else 0

    top = -mp.log(x)
    cuts = {mp.log(c / (b * x)) for c in (0.25, 1, 4, 16, 64, 256)}
    return mp.quad(density, [mp.mpf(0), *sorted(u for u in cuts if 0 < u < top), top])


def _half_mean_difference(mu, theta):
    """``E|Y - Y'| / 2 = (mu / p) 2F1(theta + 1, 1/2; 2; -z)``, ``z = 4r / p^2``, by Euler's integral.

    ``(4 / pi) int_0^1 sqrt(1 - s^2) (1 + z s^2)^-(theta + 1) ds`` (from ``t = s^2``),
    with ``s = cot(e) / sqrt(z)`` and ``e = e0 e^u``, ``e0 = atan(1 / sqrt(z))``, is
    ``(2 mu / (pi sqrt(r))) int sqrt(1 - cot(e)^2 / z) sin(e)^(2 theta) e du``: smooth
    in ``u`` however large ``z`` is.
    """
    p, r = theta / (mu + theta), mu / (mu + theta)
    z = 4 * r / p**2
    e0 = mp.atan(1 / mp.sqrt(z))
    top = mp.log(mp.pi / 2 / e0)

    def integrand(u):
        e = e0 * mp.exp(u)
        return mp.sqrt(max(1 - mp.cot(e) ** 2 / z, 0)) * mp.exp(2 * theta * mp.log(mp.sin(e))) * e

    cuts = {mp.mpf(2) ** j for j in range(-4, 12)}
    # sin(e)^(2 theta) is confined within about 1 / sqrt(theta) of pi / 2.
    cuts |= {
        mp.log((mp.pi / 2 - c / mp.sqrt(theta)) / e0)
        for c in (0.5, 1, 2, 4, 8, 16, 32, 64)
        if c / mp.sqrt(theta) < mp.pi / 2 - e0
    }
    nodes = [mp.mpf(0), *sorted(u for u in cuts if 0 < u < top), top]
    return 2 * mu / (mp.pi * mp.sqrt(r)) * mp.quad(integrand, nodes)


def nb2_crps_by_quadrature(mean: float, size: float, response: float) -> float:
    """``CRPS(NB2(mean, size), response)`` for a real response, to binary64.

    ``mean - E|Y - Y'| / 2`` keeps about ``log10(mean / size)`` fewer digits
    than its terms, and about ``log10(1 / mean)`` fewer for a rare claim, whose
    score at zero is of order ``mean^2``; ``mean G`` against ``y F`` loses about
    ``log10(mean)``.  Where the size dwarfs the mean, ``sin(e)`` differs from
    one by about ``mean / size``.  The working precision adds them.
    """
    lost = abs(math.log10(mean / size)) + abs(math.log10(mean))
    with mp.workdps(_DIGITS + int(lost)):
        mu, theta, y = mp.mpf(mean), mp.mpf(size), mp.mpf(response)
        p, r = theta / (mu + theta), mu / (mu + theta)
        k = int(mp.floor(y))
        if k < 0:
            below = mp.mpf(0)
        elif k == 0:
            below = y * mp.exp(theta * mp.log(p))
        else:
            below = y * _regularized_beta(theta, k + 1, p, r) - mu * _regularized_beta(
                theta + 1, k, p, r
            )
        return float(mu - y + 2 * below - _half_mean_difference(mu, theta))


__all__ = ["nb2_crps", "nb2_crps_by_quadrature"]
