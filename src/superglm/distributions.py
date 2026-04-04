"""Exponential dispersion family distributions.

Each distribution provides V(mu) (variance function) and d(y, mu)
(unit deviance) needed by the PIRLS solver.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

# ── Numerical guard constants for positive-mean families ─────────
_POSITIVE_INIT_MIN = 1e-12  # floor for initial_mean (replaces 0.1 pseudo-response)
_POSITIVE_MU_MIN = 1e-50  # clip_mu lower bound (log → eta ≈ -115)
_POSITIVE_MU_MAX = 1e50  # clip_mu upper bound (log → eta ≈ +115)
_VARIANCE_FLOOR = 1e-100  # V(mu) floor for IRLS working weights


@runtime_checkable
class Distribution(Protocol):
    """Protocol for exponential dispersion family distributions.

    Required: scale_known, default_link, variance, deviance_unit,
    log_likelihood.

    Optional: variance_derivative (V'(μ), used by REML W(ρ) correction;
    if absent, the correction is skipped for custom distribution objects).
    variance_second_derivative (V''(μ), used by second-order W(ρ)
    correction; Wood 2011, Appendix D).
    """

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion parameter φ is known (True) or estimated (False)."""
        ...

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
    def scale_known(self) -> bool:
        return True

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ."""
        return mu.copy()

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 1."""
        return np.ones_like(mu)

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 0. Wood (2011) Appendix D."""
        return np.zeros_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Unit deviance: 2[y log(y/μ) - (y - μ)]."""
        d = np.zeros_like(y, dtype=float)
        pos = y > 0
        d[pos] = 2 * (y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos]))
        d[~pos] = 2 * mu[~pos]
        return d

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Poisson log-likelihood (φ fixed at 1)."""
        return float(np.sum(weights * (y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1))))


class Gaussian:
    """Gaussian distribution. V(mu) = 1."""

    @property
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "identity"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = 1."""
        return np.ones_like(mu)

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 0."""
        return np.zeros_like(mu)

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 0. Wood (2011) Appendix D."""
        return np.zeros_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Gaussian unit deviance: (y - μ)^2."""
        return (y - mu) ** 2

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Gaussian log-likelihood with dispersion φ = σ²."""
        phi_safe = max(phi, 1e-300)
        resid2 = (y - mu) ** 2
        ll = -0.5 * (np.log(2 * np.pi * phi_safe) + resid2 / phi_safe)
        return float(np.sum(weights * ll))


class Gamma:
    """Gamma distribution. V(mu) = mu^2."""

    @property
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ²."""
        return mu**2

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 2μ."""
        return 2.0 * mu

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 2. Wood (2011) Appendix D."""
        return 2.0 * np.ones_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Unit deviance: 2[-log(y/μ) + (y - μ)/μ]."""
        return 2 * (-np.log(y / mu) + (y - mu) / mu)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Gamma log-likelihood. Shape k = 1/φ."""
        k = 1.0 / phi
        return float(
            np.sum(weights * (k * np.log(k * y / mu) - k * y / mu - np.log(y) - gammaln(k)))
        )


class NegativeBinomial:
    """Negative Binomial (NB2). V(mu) = mu + mu^2/theta.

    Parameters
    ----------
    theta : float or "auto"
        Overdispersion parameter (>0). Larger theta = less overdispersion.
        As theta -> inf, approaches Poisson. Pass ``"auto"`` to estimate
        theta via profile likelihood during ``fit()``.
    """

    def __init__(self, theta: float | str):
        if theta != "auto":
            if theta <= 0:
                raise ValueError(f"NB theta must be > 0, got {theta}")
        self.theta = theta

    @property
    def scale_known(self) -> bool:
        return True  # NB2 variance V(mu) = mu + mu²/θ captures overdispersion; φ=1

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ + μ²/θ."""
        return mu + mu**2 / self.theta

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 1 + 2μ/θ."""
        return 1.0 + 2.0 * mu / self.theta

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 2/θ. Wood (2011) Appendix D."""
        return (2.0 / self.theta) * np.ones_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """NB2 unit deviance."""
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
        """NB2 log-likelihood: Σ w[log Γ(y+θ) - log Γ(θ) - log Γ(y+1) + θ log(θ/(μ+θ)) + y log(μ/(μ+θ))]."""
        theta = self.theta
        ll = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (mu + theta))
            + y * np.log(mu / (mu + theta))
        )
        return float(np.sum(weights * ll))


class Binomial:
    """Binomial (Bernoulli) distribution. V(mu) = mu * (1 - mu).

    For use with binary y in {0, 1}.  This is a Bernoulli GLM (n_trials=1);
    sample_weight is case/frequency weight, not binomial trials.
    """

    @property
    def scale_known(self) -> bool:
        return True

    @property
    def default_link(self) -> str:
        return "logit"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ(1 − μ)."""
        return mu * (1 - mu)

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 1 − 2μ."""
        return 1.0 - 2.0 * mu

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = -2. Wood (2011) Appendix D."""
        return -2.0 * np.ones_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Bernoulli unit deviance: 2[y log(y/μ) + (1-y) log((1-y)/(1-μ))]."""
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        # For Bernoulli y in {0,1}: d = -2[y·log(μ) + (1-y)·log(1-μ)]
        return -2 * (y * np.log(mu_safe) + (1 - y) * np.log(1 - mu_safe))

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Bernoulli log-likelihood."""
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        ll = y * np.log(mu_safe) + (1 - y) * np.log(1 - mu_safe)
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
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μᵖ."""
        return np.power(mu, self.p)

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = p·μᵖ⁻¹."""
        return self.p * np.power(mu, self.p - 1)

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = p(p-1)·μᵖ⁻². Wood (2011) Appendix D."""
        return self.p * (self.p - 1) * np.power(mu, self.p - 2)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Tweedie unit deviance."""
        p = self.p
        t1 = np.where(y > 0, np.power(y, 2 - p) / ((1 - p) * (2 - p)), 0.0)
        t2 = y * np.power(mu, 1 - p) / (1 - p)
        t3 = np.power(mu, 2 - p) / (2 - p)
        return 2 * (t1 - t2 + t3)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Tweedie log-likelihood via exact Wright-Bessel evaluation."""
        from superglm.profiling.tweedie import tweedie_logpdf

        logpdf = tweedie_logpdf(y, mu, phi, self.p, weights=weights)
        return float(np.sum(logpdf))


DISTRIBUTION_SHORTCUTS: dict[str, type] = {
    "poisson": Poisson,
    "gaussian": Gaussian,
    "gamma": Gamma,
    "binomial": Binomial,
}


def resolve_distribution(
    family: str | Distribution,
) -> Distribution:
    """Convert string shorthand to distribution object, or pass through.

    Parameter-free families can be specified as strings (``"poisson"``,
    ``"gaussian"``, ``"gamma"``, ``"binomial"``).  Parameterized families
    (Tweedie, NB2) must be passed as Distribution objects::

        from superglm import families
        resolve_distribution(families.tweedie(p=1.5))
    """
    if not isinstance(family, str):
        return family
    if family in DISTRIBUTION_SHORTCUTS:
        return DISTRIBUTION_SHORTCUTS[family]()
    if family in ("tweedie", "negative_binomial"):
        raise ValueError(
            f"'{family}' requires parameters.  "
            f"Use families.tweedie(p=...) or families.nb2(theta=...) instead of a string."
        )
    raise ValueError(
        f"Unknown distribution '{family}'. Use 'poisson', 'gaussian', 'gamma', 'binomial', "
        f"or pass a Distribution object (e.g. families.tweedie(p=1.5))."
    )


# ── Family-aware helpers ───────────────────────────────────────────


def validate_response(y: NDArray, family: Distribution) -> None:
    """Validate the response vector for the given family.

    Raises ValueError for invalid responses (e.g. non-binary for binomial,
    negative for Poisson/Gamma).
    """
    if isinstance(family, Binomial):
        bad = ~np.isin(y, [0, 1])
        if np.any(bad):
            n_bad = int(np.sum(bad))
            vals = np.unique(y[bad])[:5]
            raise ValueError(
                f"Binomial family requires y in {{0, 1}}, "
                f"but found {n_bad} invalid values (e.g. {vals})."
            )


def initial_mean(y: NDArray, weights: NDArray, family: Distribution) -> float:
    """Weighted mean of y, clipped to the valid range for the family.

    For positive-response families (Poisson, Gamma, NB, Tweedie), use the raw
    weighted mean with only a small positive floor so sparse or near-separated
    fits are not biased upward by an arbitrary pseudo-response.
    For binomial, the raw weighted mean is clipped to (eps, 1-eps).
    For Gaussian, use the raw weighted mean with no positivity clipping.
    """
    if isinstance(family, Binomial):
        y_bar = float(np.average(y, weights=weights))
        return np.clip(y_bar, 1e-3, 1 - 1e-3)
    if isinstance(family, Gaussian):
        return float(np.average(y, weights=weights))
    return max(float(np.average(y, weights=weights)), _POSITIVE_INIT_MIN)


def clip_mu(mu: NDArray, family: Distribution) -> NDArray:
    """Clip predicted means to a valid range for the family.

    For positive-mean families, the bounds must be wide enough that the
    IRLS can converge for near-separated categorical levels.
    """
    if isinstance(family, Binomial):
        return np.clip(mu, 1e-7, 1 - 1e-7)
    if isinstance(family, Gaussian):
        return mu
    return np.clip(mu, _POSITIVE_MU_MIN, _POSITIVE_MU_MAX)
