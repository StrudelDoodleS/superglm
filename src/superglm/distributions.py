"""Exponential dispersion family distributions.

Each distribution provides V(mu) (variance function) and d(y, mu)
(unit deviance) needed by the PIRLS solver.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln


@runtime_checkable
class Distribution(Protocol):
    """Protocol for exponential dispersion family distributions."""

    def variance(self, mu: NDArray) -> NDArray:
        """V(mu) — variance as a function of the mean."""
        ...

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Per-observation unit deviance d(y, mu)."""
        ...

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Sum of weighted log-likelihood contributions."""
        ...


class Poisson:
    """Poisson distribution. V(mu) = mu."""

    def variance(self, mu: NDArray) -> NDArray:
        return mu.copy()

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        d = np.zeros_like(y, dtype=float)
        pos = y > 0
        d[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos]))
        d[~pos] = 2 * mu[~pos]
        return d

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Poisson log-likelihood (phi is ignored, fixed at 1)."""
        return float(np.sum(weights * (y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1))))


class Gamma:
    """Gamma distribution. V(mu) = mu^2."""

    def variance(self, mu: NDArray) -> NDArray:
        return mu**2

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        return 2 * (-np.log(y / mu) + (y - mu) / mu)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Gamma log-likelihood. Shape k = 1/phi."""
        k = 1.0 / phi
        return float(np.sum(weights * (
            k * np.log(k * y / mu) - k * y / mu - np.log(y) - gammaln(k)
        )))


class Tweedie:
    """Tweedie distribution. V(mu) = mu^p, with p in (1, 2).

    Parameters
    ----------
    p : float
        Power parameter. Must be in (1, 2).
        p → 1 approaches Poisson, p → 2 approaches Gamma.
    """

    def __init__(self, p: float):
        if not 1 < p < 2:
            raise ValueError(f"Tweedie p must be in (1, 2), got {p}")
        self.p = p

    def variance(self, mu: NDArray) -> NDArray:
        return np.power(mu, self.p)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        p = self.p
        t1 = np.where(y > 0, np.power(y, 2 - p) / ((1 - p) * (2 - p)), 0.0)
        t2 = y * np.power(mu, 1 - p) / (1 - p)
        t3 = np.power(mu, 2 - p) / (2 - p)
        return 2 * (t1 - t2 + t3)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Tweedie log-likelihood via tweedie_logpdf."""
        from superglm.tweedie_profile import tweedie_logpdf

        logpdf = tweedie_logpdf(y, mu, phi, self.p, weights=weights)
        return float(np.sum(weights * logpdf))


DISTRIBUTION_SHORTCUTS: dict[str, type] = {
    "poisson": Poisson,
    "gamma": Gamma,
}


def resolve_distribution(
    family: str | Distribution, tweedie_p: float | None = None
) -> Distribution:
    """Convert string shorthand to distribution object, or pass through."""
    if not isinstance(family, str):
        return family
    if family in DISTRIBUTION_SHORTCUTS:
        return DISTRIBUTION_SHORTCUTS[family]()
    if family == "tweedie":
        if tweedie_p is None:
            raise ValueError("Tweedie distribution requires tweedie_p=")
        return Tweedie(p=tweedie_p)
    raise ValueError(
        f"Unknown distribution '{family}'. Use 'poisson', 'gamma', 'tweedie', or pass a Distribution object."
    )
