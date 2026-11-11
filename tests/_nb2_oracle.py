"""Independent 80-digit NB2 references for the unit deviance and the log density.

Each value is the textbook formula evaluated in mpmath; no production code is
imported, so a cancellation or underflow in float64 cannot reach the reference.
"""

from __future__ import annotations

import mpmath as mp


def unit_deviance(y: float, mu: float, theta: float, dps: int = 80) -> float:
    """``2 [y log(y / mu) - (y + theta) log((y + theta) / (mu + theta))]``."""
    with mp.workdps(dps):
        count, mean, size = mp.mpf(y), mp.mpf(mu), mp.mpf(theta)
        count_term = count * mp.log(count / mean) if count > 0 else mp.mpf(0)
        size_term = (count + size) * mp.log((count + size) / (mean + size))
        return float(2 * (count_term - size_term))


def log_density(y: float, mu: float, theta: float, dps: int = 80) -> float:
    """``log Gamma(y + theta) - log Gamma(theta) - log Gamma(y + 1)
    + theta log(theta / (mu + theta)) + y log(mu / (mu + theta))``."""
    with mp.workdps(dps):
        count, mean, size = mp.mpf(y), mp.mpf(mu), mp.mpf(theta)
        return float(
            mp.loggamma(count + size)
            - mp.loggamma(size)
            - mp.loggamma(count + 1)
            + size * mp.log(size / (mean + size))
            + count * mp.log(mean / (mean + size))
        )
