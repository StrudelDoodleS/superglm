"""Independent, high-precision NB probabilities and normalized row likelihoods.

No production recurrence, endpoint helper, or derivative formula is imported.
The oracle differentiates mpmath beta integrals/finite PMF sums at 65 digits.
"""

from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np


def pmf(k, mean, size):
    return mp.exp(
        mp.loggamma(k + size)
        - mp.loggamma(size)
        - mp.loggamma(k + 1)
        - size * mp.log1p(mean / size)
        + k * (mp.log(mean) - mp.log(mean + size))
    )


def probability(lower, upper, mean, size):
    if lower == 0 and upper == np.inf:
        return mp.mpf(1)
    if np.isfinite(upper) and upper - lower <= 128:
        return mp.fsum(pmf(k, mean, size) for k in range(int(lower), int(upper) + 1))
    p = size / (size + mean)
    if upper == np.inf:
        if lower <= 128:
            return -mp.expm1(-size * mp.log1p(mean / size)) - mp.fsum(
                pmf(k, mean, size) for k in range(1, int(lower))
            )
        return mp.betainc(lower, size, 0, mean / (size + mean), regularized=True)
    return mp.betainc(size, upper + 1, 0, p, regularized=True) - (
        mp.betainc(size, lower, 0, p, regularized=True) if lower else 0
    )


def derivatives(function, point, order=2):
    with mp.workdps(65):
        point = tuple(mp.mpf(float(x)) for x in point)
        scales = tuple(abs(x) if x else mp.mpf(1) for x in point)

        def scaled(*delta):
            return function(*(x + s * d for x, s, d in zip(point, scales, delta, strict=True)))

        out = [np.array([float(function(*point))])]
        for degree in range(1, order + 1):
            out.append(
                np.array(
                    [
                        [
                            float(
                                mp.diff(
                                    scaled,
                                    (mp.mpf(0),) * len(point),
                                    tuple(key.count(i) for i in range(len(point))),
                                )
                                / mp.fprod(scales[i] for i in key)
                            )
                            for key in combinations_with_replacement(range(len(point)), degree)
                        ]
                    ]
                )
            )
        return tuple(out)


def _log_probability(lower, upper, mean, size):
    if lower == 0 and np.isfinite(upper) and upper > mean and size > 0.1:
        tail = mp.betainc(upper + 1, size, 0, mean / (size + mean), regularized=True)
        return mp.log1p(-tail)
    return mp.log(probability(lower, upper, mean, size))


def window_jet(lower, upper, mean, size, weight=1.0):
    return derivatives(
        lambda m, r: _log_probability(lower, upper, weight * m, weight * r),
        (mean, size),
    )


def normalized_jet(count, observed, truncation, mean, size, weight=1.0):
    def likelihood(m, r):
        numerator = (
            pmf(count, weight * m, weight * r)
            if count is not None
            else probability(*observed, weight * m, weight * r)
        )
        return mp.log(numerator / probability(*truncation, weight * m, weight * r))

    return derivatives(likelihood, (mean, size))


def beta_jet(a, b, logit_x, upper):
    def log_tail(aa, bb, zz):
        x = 1 / (1 + mp.exp(-zz))
        return mp.log(mp.betainc(aa, bb, x if upper else 0, 1 if upper else x, regularized=True))

    return derivatives(log_tail, (a, b, logit_x))
