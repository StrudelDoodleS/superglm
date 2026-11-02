"""Independent references for the Prentice generalized gamma.

Nothing here shares code with the kernel: the textbook density is written in
its naive form (which loses digits near Q = 0 in float64, which is why the
mpmath versions exist).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

PACKED = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def textbook_log_density(y, mu, sigma, q):
    """Prentice (1974) log density; accepts complex mu/sigma/q for complex-step."""
    k = 1.0 / (q * q)
    w = (np.log(y) - mu) / sigma
    u = q * w
    return (
        np.log(np.abs(q) if not isinstance(q, complex) else q)
        + k * np.log(k)
        - special.loggamma(k)
        - np.log(sigma)
        - np.log(y)
        + k * (u - np.exp(u))
    )


def complex_step_score(y, params, index, step=1.0e-30):
    perturbed = [complex(p) for p in params]
    perturbed[index] = perturbed[index] + 1j * step
    return float(np.imag(textbook_log_density(y, *perturbed)) / step)


def mp_log_density(y, mu, sigma, q, dps=60):
    import mpmath as mp

    with mp.workdps(max(dps, mp.mp.dps)):
        y, mu, sigma, q = (mp.mpf(a) for a in (y, mu, sigma, q))
        k = 1 / q**2
        w = (mp.log(y) - mu) / sigma
        u = q * w
        return (
            mp.log(abs(q))
            + k * mp.log(k)
            - mp.loggamma(k)
            - mp.log(sigma)
            - mp.log(y)
            + k * (u - mp.exp(u))
        )


def mp_log_mean_loading(sigma, q, dps=60):
    import mpmath as mp

    with mp.workdps(max(dps, mp.mp.dps)):
        sigma, q = mp.mpf(sigma), mp.mpf(q)
        k = 1 / q**2
        c = sigma / q
        return c * mp.log(q**2) + mp.loggamma(k + c) - mp.loggamma(k)


def mp_expected_shortfall(p, mu, sigma, q, dps=30):
    """``E[Y | Y > y_p]`` and its ratio to ``E[Y]``, by quadrature of the density.

    ``W`` has log density ``log|q| + k log k - log Gamma(k) + k (q w - e^(q w))``,
    ``k = q^-2``, which is exactly ``-log(2 pi)/2 - R(k) - (e^t - 1 - t)/q^2`` with
    ``t = q w`` and ``R`` the Stirling remainder of ``log Gamma`` (the standard
    normal at ``q = 0``).  That form cancels nothing at large ``k``, so a modest
    ``dps`` suffices.  The quantile ``w_p`` solves ``int_w^inf f = 1 - p`` by
    Newton's method, and the shortfall is
    ``e^mu int_(w_p)^inf e^(sigma w) f(w) dw / (1 - p)``, the limits cut where the
    integrand is below ``e^-800``.  Shares no formula with
    the incomplete-gamma or uniform engines.
    """
    import mpmath as mp

    with mp.workdps(max(dps, mp.mp.dps)):
        p, mu, sigma, q = (mp.mpf(a) for a in (p, mu, sigma, q))
        half_log_two_pi = mp.log(2 * mp.pi) / 2
        if q == 0:
            remainder = mp.mpf(0)
        else:
            k = 1 / q**2
            if k > 1e8:
                remainder = 1 / (12 * k) - 1 / (360 * k**3) + 1 / (1260 * k**5)
            else:
                remainder = mp.loggamma(k) - ((k - 0.5) * mp.log(k) - k + half_log_two_pi)
        guard = 2 * int(mp.log10(1 / abs(q))) if q != 0 else 0

        def log_f(w):
            if q == 0:
                excess = w * w / 2
            else:
                with mp.workdps(mp.mp.dps + guard):
                    t = q * w
                    excess = (mp.expm1(t) - t) / q**2
            return -half_log_two_pi - remainder - excess

        def integral(lower, tilt):
            # The integrand is below e^-800 beyond 40 of the tilted centre on either
            # side, and an infinite limit makes the quadrature sample e^t at huge t.
            centre = mp.mpf(tilt)
            lower = centre - 40 if lower is None else lower
            points = [lower, lower + 1, lower + 4, centre, centre + 4, centre + 12]
            points = sorted({b for b in points if b >= lower} | {max(lower, centre) + 40})
            return mp.quad(lambda t: mp.exp(log_f(t) + tilt * t), points)

        survival = 1 - p
        z = -mp.sqrt(2) * mp.erfinv(2 * survival - 1)
        w_p = mp.findroot(
            lambda w: integral(w, 0) - survival,
            z - q * (z * z + 2) / 6,
            solver="newton",
            df=lambda w: -mp.exp(log_f(w)),
        )
        moment = integral(w_p, sigma)
        mean = integral(None, sigma)
        return mp.exp(mu) * moment / survival, moment / (survival * mean)


def _information_dps(q):
    """Working digits for the cross-entropy reference.

    ``loggamma(k + c) - loggamma(k)`` cancels ``log10 k = 2 log10(1/|Q|)``
    digits, and ``mp.diff`` spends more on its step, so the budget grows with
    ``-log10|Q|``.  Checked against a reference computed at ``dps + 40``.
    """
    return int(60 + 20 * max(0.0, -math.log10(abs(q))))


def _mp_cross_entropy(mu, sigma, q):
    """``theta' -> E_theta[log f_theta']`` in location coordinates, in closed form.

    ``E[log Y]`` is a digamma of ``k = Q^-2`` and ``E[e^u']`` is the
    ``Q'/sigma'``-th moment of the law, so nothing here repeats the packed
    information's algebra.  The body raises no precision of its own: it must run
    at whatever precision ``mp.diff`` has set, or every difference rounds away.
    """
    import mpmath as mp

    def cross_entropy(mu2, sigma2, q2):
        k = 1 / q**2
        e_log_y = mu + (sigma / q) * (mp.digamma(k) - mp.log(k))
        k2 = 1 / q2**2
        power = q2 / sigma2  # E[e^u'] = e^(-power mu') E[Y^power]
        order = power * sigma / q  # the moment order of the Gamma(k, 1) variate
        log_e_exp_u = (
            power * (mu - mu2) - order * mp.log(k) + mp.loggamma(k + order) - mp.loggamma(k)
        )
        return (
            mp.log(abs(q2))
            + k2 * mp.log(k2)
            - mp.loggamma(k2)
            - mp.log(sigma2)
            - e_log_y
            + k2 * (power * (e_log_y - mu2) - mp.exp(log_e_exp_u))
        )

    return cross_entropy


_INFORMATION_ORDERS = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]


def mp_expected_information(sigma, q, mu=0.3, dps=None):
    """Packed ``-E[H]`` in ``(mu, sigma, Q)`` from the cross-entropy Hessian.

    ``I(theta) = -d^2/dtheta'^2 E_theta[log f_theta']`` at ``theta' = theta``.
    An independent route to the same matrix: it never touches the digamma and
    trigamma identities the kernel packs.
    """
    import mpmath as mp

    with mp.workdps(_information_dps(q) if dps is None else dps):
        point = (mp.mpf(mu), mp.mpf(sigma), mp.mpf(q))
        cross_entropy = _mp_cross_entropy(*point)
        return [-float(mp.diff(cross_entropy, point, order)) for order in _INFORMATION_ORDERS]


def mp_mean_expected_information(mean, sigma, q, dps=None):
    """Packed ``-E[H]`` in ``(mean, sigma, Q)``, from the same cross-entropy Hessian.

    Independent of the kernel's Jacobian congruence: the reparametrisation
    ``mu = log m - log C(sigma, Q)`` is applied inside the function that is
    differentiated, not to the differentiated result.
    """
    import mpmath as mp

    dps = _information_dps(q) if dps is None else dps
    with mp.workdps(dps):
        point = (mp.mpf(mean), mp.mpf(sigma), mp.mpf(q))
        mu = mp.log(point[0]) - mp_log_mean_loading(point[1], point[2], dps=dps)
        location = _mp_cross_entropy(mu, point[1], point[2])

        def cross_entropy(mean2, sigma2, q2):
            return location(mp.log(mean2) - mp_log_mean_loading(sigma2, q2, dps=dps), sigma2, q2)

        return [-float(mp.diff(cross_entropy, point, order)) for order in _INFORMATION_ORDERS]


def mp_derivatives(function, point, orders, dps=60):
    """mpmath partial derivatives of ``function`` at ``point`` for each order tuple.

    ``mp.diff`` raises the working precision above ``dps`` so that its step is
    visible; the reference functions above therefore only ever raise precision
    (``max(dps, mp.mp.dps)``), never clamp it back, or every difference would
    round to zero.
    """
    import mpmath as mp

    with mp.workdps(dps):
        pt = tuple(mp.mpf(p) for p in point)
        return [float(mp.diff(function, pt, order)) for order in orders]
