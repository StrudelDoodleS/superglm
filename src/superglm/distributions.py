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

    @property
    def default_link(self) -> str:
        """Name of the canonical/default link function."""
        ...

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

    @property
    def default_link(self) -> str:
        return "log"

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

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        return mu**2

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        return 2 * (-np.log(y / mu) + (y - mu) / mu)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Gamma log-likelihood. Shape k = 1/phi."""
        k = 1.0 / phi
        return float(
            np.sum(weights * (k * np.log(k * y / mu) - k * y / mu - np.log(y) - gammaln(k)))
        )


class NegativeBinomial:
    """Negative Binomial (NB2). V(mu) = mu + mu^2/theta.

    Parameters
    ----------
    theta : float
        Overdispersion parameter (>0). Larger theta = less overdispersion.
        As theta -> inf, approaches Poisson.
    """

    def __init__(self, theta: float):
        if theta <= 0:
            raise ValueError(f"NB theta must be > 0, got {theta}")
        self.theta = theta

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        return mu + mu**2 / self.theta

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        theta = self.theta
        d = np.where(
            y > 0,
            2
            * (
                y * np.log(np.maximum(y, 1e-300) / mu)
                - (y + theta) * np.log((y + theta) / (mu + theta))
            ),
            2 * theta * np.log(theta / (mu + theta)),
        )
        return d

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        theta = self.theta
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (mu + theta))
            + y * np.log(mu / (mu + theta))
        )
        return float(np.sum(weights * ll))


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

    @property
    def default_link(self) -> str:
        return "log"

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
    family: str | Distribution,
    tweedie_p: float | None = None,
    nb_theta: float | None = None,
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
    if family == "negative_binomial":
        if nb_theta is None:
            raise ValueError("NB distribution requires nb_theta=")
        return NegativeBinomial(theta=nb_theta)
    raise ValueError(
        f"Unknown distribution '{family}'. Use 'poisson', 'gamma', 'tweedie', "
        f"'negative_binomial', or pass a Distribution object."
    )
