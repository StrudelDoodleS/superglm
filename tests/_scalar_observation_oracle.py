"""Independent textbook scalar laws for the all-family observation audit.

No production probability, endpoint, derivative or link helper is used here.
Small central stencils at 60 decimal digits resolve derivatives far below the
float64 outcome tolerances. The compound law reuses Task 4's independent fixed
Poisson sum; its ordinary-domain fixtures have rate < 6 and 100 terms.
"""

from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np

from tests._tweedie_window_oracle import density as tweedie_density
from tests._tweedie_window_oracle import window as tweedie_window
from tests._two_piece_observation_oracle import mp_contribution as two_piece_contribution


def contribution(law, form, y, lo, hi, side, theta, prior=1):
    y, lo, hi, prior = map(mp.mpf, (float(y), float(lo), float(hi), float(prior)))
    theta = tuple(map(mp.mpf, theta))
    if law == "two_piece_normal" or law == "two_piece_log_normal":
        transform = "identity" if law == "two_piece_normal" else "log"
        return two_piece_contribution(y, lo, hi, side, theta, transform, form)
    first, second = theta[:2]

    def normal_cdf(z):
        return mp.erfc(-z / mp.sqrt(2)) / 2

    def cdf(z):
        if z == -mp.inf or (law != "gaussian" and z < 0):
            return mp.mpf(0)
        if z == mp.inf:
            return mp.mpf(1)
        if law == "gaussian":
            return normal_cdf((z - first) * mp.sqrt(prior) / second)
        if law == "nb":
            # The observed rate is binary64 k/w. Enumerate the moderate count
            # lattice directly, comparing represented rates, never floor(w*z).
            k = 0
            while float(k / float(prior)) <= float(z):
                k += 1
            r, mu = prior * second, prior * first
            return mp.fsum(
                mp.rf(r, j) / mp.factorial(j) * (r / (r + mu)) ** r * (mu / (r + mu)) ** j
                for j in range(k)
            )
        if z == 0:
            return mp.mpf(0)
        if law == "gamma":
            a = prior / second**2
            return mp.gammainc(a, 0, z * a / first, regularized=True)
        if law == "log_normal":
            mu = first if form == "location" else mp.log(first) - second**2 / 2
            return normal_cdf((mp.log(z) - mu) / second)
        if law == "gg":
            q = theta[2]
            a = 1 / q**2
            mu = first
            if form == "mean":
                loading = mp.loggamma(a + second / q) - mp.loggamma(a) - second / q * mp.log(a)
                mu = mp.log(first) - loading
            x = a * mp.exp(q * (mp.log(z) - mu) / second)
            return mp.gammainc(a, 0, x, regularized=True)
        if law == "gpd":
            return 1 - (1 + second * z / first) ** (-1 / second)
        if law == "lomax":
            return 1 - (1 + z / first) ** (-second)
        if law == "log_logistic":
            return 1 / (1 + (z / first) ** (-second))
        raise AssertionError(law)

    def density():
        if law == "gaussian":
            sd = second / mp.sqrt(prior)
            return mp.exp(-(((y - first) / sd) ** 2) / 2) / (mp.sqrt(2 * mp.pi) * sd)
        if law == "gamma":
            a = prior / second**2
            return mp.exp(-y * a / first) * y ** (a - 1) * (a / first) ** a / mp.gamma(a)
        if law == "log_normal":
            mu = first if form == "location" else mp.log(first) - second**2 / 2
            return mp.exp(-(((mp.log(y) - mu) / second) ** 2) / 2) / (
                y * second * mp.sqrt(2 * mp.pi)
            )
        if law == "gg":
            q = theta[2]
            a = 1 / q**2
            mu = first
            if form == "mean":
                mu = (
                    mp.log(first)
                    - mp.loggamma(a + second / q)
                    + mp.loggamma(a)
                    + second / q * mp.log(a)
                )
            z = q * (mp.log(y) - mu) / second
            return abs(q) * a**a * mp.exp(a * (z - mp.exp(z))) / (second * y * mp.gamma(a))
        if law == "gpd":
            return (1 + second * y / first) ** (-1 / second - 1) / first
        if law == "lomax":
            return second / first * (1 + y / first) ** (-second - 1)
        if law == "log_logistic":
            z = y / first
            return second / first * z ** (second - 1) / (1 + z**second) ** 2
        if law == "nb":
            k, r, mu = int(float(y) * float(prior)), prior * second, prior * first
            return mp.rf(r, k) / mp.factorial(k) * (r / (r + mu)) ** r * (mu / (r + mu)) ** k
        if law == "tweedie":
            return tweedie_density(y, *theta, weight=prior)
        raise AssertionError(law)

    def window(a, b):
        if law == "tweedie":
            return tweedie_window(a, b, *theta, weight=prior)
        return cdf(b) - cdf(a)

    numerator = density() if side == 0 else window(y if side > 0 else lo, hi if side > 0 else y)
    return mp.log(numerator) - mp.log(window(lo, hi))


def inverse_links(eta, specifications):
    """Independent smooth inverse maps, specified explicitly by the fixture."""
    result = []
    for x, (kind, lower, upper) in zip(eta, specifications, strict=True):
        if kind == "identity":
            result.append(x)
        elif kind == "log":
            result.append(lower + mp.exp(x))
        else:
            result.append(lower + (upper - lower) / (1 + mp.exp(-x)))
    return result


def jet(function, point):
    with mp.workdps(60):
        point = tuple(mp.mpf(float(x)) for x in point)
        h = mp.mpf("1e-12")
        value = function(*point)

        def at(i, sign, j=None, other=0):
            moved = list(point)
            moved[i] += sign * h
            if j is not None:
                moved[j] += other * h
            return function(*moved)

        plus = [at(i, 1) for i in range(len(point))]
        minus = [at(i, -1) for i in range(len(point))]
        score = [(p - m) / (2 * h) for p, m in zip(plus, minus, strict=True)]
        hessian = [
            (plus[i] - 2 * value + minus[i]) / h**2
            if i == j
            else (at(i, 1, j, 1) - at(i, 1, j, -1) - at(i, -1, j, 1) + at(i, -1, j, -1))
            / (4 * h**2)
            for i, j in combinations_with_replacement(range(len(point)), 2)
        ]
        return float(value), np.array(score, float), np.array(hessian, float)
