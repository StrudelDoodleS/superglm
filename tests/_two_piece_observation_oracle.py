"""Independent piecewise-normal observation law; no production numerical helpers.

mpmath differentiates the normalized likelihood itself, including moving joins
and the mean-to-location map. Float64 CDF/integration references for the small
fit fixtures live here too, independently of the production jets.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
from scipy import special

PACKED = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def mp_location(first, scale, skew, parametrisation):
    if parametrisation == "location":
        return first
    a, b = scale * (1 - skew), scale * (1 + skew)
    loading = (1 - skew) * mp.exp(a * a / 2) * mp.ncdf(-a)
    loading += (1 + skew) * mp.exp(b * b / 2) * mp.ncdf(b)
    return mp.log(first) - mp.log(loading)


def mp_endpoint(value, transform):
    value = mp.mpf(value)
    return value if transform == "identity" else mp.log(value) if value > 0 else -mp.inf


def mp_window(lower, upper, theta, transform="identity", parametrisation="location"):
    first, scale, skew = theta
    mu = mp_location(first, scale, skew, parametrisation)
    lo, hi = mp_endpoint(lower, transform), mp_endpoint(upper, transform)

    def interval(a, b):
        # The survival representation avoids losing small same-side tails.
        if a >= 0:
            return (mp.erfc(a / mp.sqrt(2)) - mp.erfc(b / mp.sqrt(2))) / 2
        return (mp.erfc(-b / mp.sqrt(2)) - mp.erfc(-a / mp.sqrt(2))) / 2

    mass = mp.mpf(0)
    if lo < mu:
        s = 1 - skew
        mass += s * interval((lo - mu) / (scale * s), (min(hi, mu) - mu) / (scale * s))
    if hi > mu:
        s = 1 + skew
        mass += s * interval((max(lo, mu) - mu) / (scale * s), (hi - mu) / (scale * s))
    return mass


def mp_contribution(y, lower, upper, side, theta, transform, parametrisation):
    first, scale, skew = theta
    if side:
        a, b = (y, upper) if side > 0 else (lower, y)
        value = mp.log(mp_window(a, b, theta, transform, parametrisation))
    else:
        mu = mp_location(first, scale, skew, parametrisation)
        t = mp_endpoint(y, transform)
        width = scale * (1 - skew if t < mu else 1 + skew)
        value = -mp.log(scale) - mp.log(2 * mp.pi) / 2 - ((t - mu) / width) ** 2 / 2
        if transform == "log":
            value -= mp.log(mp.mpf(y))
    return value - mp.log(mp_window(lower, upper, theta, transform, parametrisation))


def mp_jet(function, theta, dps=70):
    with mp.workdps(dps):
        point = tuple(mp.mpf(x) for x in theta)
        score = [
            float(mp.diff(function, point, tuple(int(k == i) for k in range(3)))) for i in range(3)
        ]
        hessian = [
            float(mp.diff(function, point, tuple(int(k == i) + int(k == j) for k in range(3))))
            for i, j in PACKED
        ]
        return float(function(*point)), np.array(score), np.array(hessian)


def linked_loglik(
    eta, response, lower, upper, side, *, transform, parametrisation, floor=0.01, bound=0.9
):
    """Ordinary-domain independent CDF likelihood, also analytic on complex paths.

    Branches are selected by the real point. Complex steps differentiate this
    textbook likelihood without using family/link/normal-window derivatives.
    Fit fixtures have moderate finite endpoints, so CDF subtraction is safe.
    """
    first = np.exp(eta[:, 0]) if parametrisation == "mean" else eta[:, 0]
    scale, skew = floor + np.exp(eta[:, 1]), bound * np.tanh(0.5 * eta[:, 2])
    mu = first
    if parametrisation == "mean":
        a, b = scale * (1 - skew), scale * (1 + skew)
        loading = (1 - skew) * np.exp(a * a / 2) * special.ndtr(-a)
        loading += (1 + skew) * np.exp(b * b / 2) * special.ndtr(b)
        mu = np.log(first) - np.log(loading)
    if transform == "log":
        t, lo, hi = np.log(response), np.log(lower), np.log(upper)
    else:
        t, lo, hi = response, lower, upper

    def cdf(point):
        left = point < mu.real
        z = (point - mu) / (scale * np.where(left, 1 - skew, 1 + skew))
        return np.where(left, (1 - skew) * special.ndtr(z), 1 - (1 + skew) * special.ndtr(-z))

    z = (t - mu) / (scale * np.where(t < mu.real, 1 - skew, 1 + skew))
    exact = -np.log(scale) - 0.5 * np.log(2 * np.pi) - z * z / 2
    if transform == "log":
        exact -= np.log(response)
    c_lo, c_hi, c_y = cdf(lo), cdf(hi), cdf(t)
    numerator = np.where(side > 0, c_hi - c_y, c_y - c_lo)
    return np.where(side == 0, exact, np.log(numerator)) - np.log(c_hi - c_lo)


def complex_gradient(function, point):
    step = 1e-25
    return np.array(
        [
            function(np.asarray(point, dtype=complex) + 1j * step * unit).imag / step
            for unit in np.eye(len(point))
        ]
    )


def reference_hessian(function, point, step=2e-4):
    def difference(h):
        return np.column_stack(
            [
                (
                    complex_gradient(function, point + h * unit)
                    - complex_gradient(function, point - h * unit)
                )
                / (2 * h)
                for unit in np.eye(len(point))
            ]
        )

    coarse, fine = difference(step), difference(step / 2)
    result = (4 * fine - coarse) / 3
    return 0.5 * (result + result.T)


def bounded_sample(*, varying=False, skew=0.35):
    """Stratified inverse-CDF draws using the textbook epsilon-skew law."""
    x = np.repeat(np.linspace(-1, 1, 12), 16)
    mu = 0.3 + (0.5 * np.sin(np.pi * x) if varying else np.zeros(len(x)))
    sigma = 0.65
    lower, upper = np.full(len(x), -1.6), np.full(len(x), 2.2)

    def cdf(t):
        z = (t - mu) / sigma
        return np.where(
            z < 0,
            (1 - skew) * special.ndtr(z / (1 - skew)),
            1 - (1 + skew) * special.ndtr(-z / (1 + skew)),
        )

    q = np.tile((np.arange(16) + 0.5) / 16, 12)
    p = cdf(lower) + q * (cdf(upper) - cdf(lower))
    z = np.where(
        p < (1 - skew) / 2,
        (1 - skew) * special.ndtri(p / (1 - skew)),
        -(1 + skew) * special.ndtri((1 - p) / (1 + skew)),
    )
    y = mu + sigma * z
    side = np.where(y < -0.5, -1, np.where(y > 1.25, 1, 0))
    y = np.clip(y, -0.5, 1.25)
    return x, y, lower, upper, side
