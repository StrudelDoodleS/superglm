"""Link functions for GLMs.

Each link provides the mapping between the linear predictor (eta) and the
mean (mu), plus the derivative needed by the PIRLS working weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.distributions import Distribution


@runtime_checkable
class Link(Protocol):
    """Protocol for GLM link functions.

    Required methods (must be present for isinstance check):
        link, inverse, deriv, deriv_inverse

    Optional methods (detected at runtime via hasattr):
        deriv2_inverse — d²μ/dη², used by REML W(ρ) correction.
        If absent, the W(ρ) correction is skipped and REML falls back
        to the fixed-W Laplace approximation.
    """

    def link(self, mu: NDArray) -> NDArray:
        """mu -> eta (forward link)."""
        ...

    def inverse(self, eta: NDArray) -> NDArray:
        """eta -> mu (inverse link)."""
        ...

    def deriv(self, mu: NDArray) -> NDArray:
        """d_eta/d_mu — derivative of forward link w.r.t. mu."""
        ...

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        """d_mu/d_eta — derivative of inverse link w.r.t. eta."""
        ...


class LogLink:
    """Log link: eta = log(mu), mu = exp(eta)."""

    def link(self, mu: NDArray) -> NDArray:
        return np.log(mu)

    def inverse(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def deriv(self, mu: NDArray) -> NDArray:
        return 1.0 / mu

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return np.exp(eta)


class IdentityLink:
    """Identity link: eta = mu, mu = eta."""

    def link(self, mu: NDArray) -> NDArray:
        return mu.copy()

    def inverse(self, eta: NDArray) -> NDArray:
        return eta.copy()

    def deriv(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return np.ones_like(eta)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return np.zeros_like(eta)


class LogitLink:
    """Logit link: eta = log(mu / (1-mu)), mu = expit(eta)."""

    def link(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.log(mu_safe / (1 - mu_safe))

    def inverse(self, eta: NDArray) -> NDArray:
        from scipy.special import expit

        return expit(eta)

    def deriv(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return 1.0 / (mu_safe * (1 - mu_safe))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        from scipy.special import expit

        p = expit(eta)
        return p * (1 - p)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        from scipy.special import expit

        p = expit(eta)
        return p * (1 - p) * (1 - 2 * p)


class ProbitLink:
    """Probit link: eta = Phi^{-1}(mu), mu = Phi(eta).

    Uses the standard normal CDF.  Canonical link for binomial in some
    traditions (latent-variable / threshold model interpretation).
    """

    def link(self, mu: NDArray) -> NDArray:
        from scipy.stats import norm

        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return norm.ppf(mu_safe)

    def inverse(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm

        return norm.cdf(eta)

    def deriv(self, mu: NDArray) -> NDArray:
        from scipy.stats import norm

        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return 1.0 / norm.pdf(norm.ppf(mu_safe))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm

        return norm.pdf(eta)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        from scipy.stats import norm

        return -eta * norm.pdf(eta)


class CloglogLink:
    """Complementary log-log link: eta = log(-log(1 - mu)).

    The canonical link for the extreme-value / Gompertz model.
    Asymmetric alternative to logit for binary responses.
    """

    def link(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.log(-np.log(1 - mu_safe))

    def inverse(self, eta: NDArray) -> NDArray:
        return 1 - np.exp(-np.exp(eta))

    def deriv(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return 1.0 / ((1 - mu_safe) * (-np.log(1 - mu_safe)))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        # d/deta [1 - exp(-exp(eta))] = exp(eta) * exp(-exp(eta))
        ee = np.exp(eta)
        return ee * np.exp(-ee)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        # d²/deta² = exp(eta - exp(eta)) * (1 - exp(eta))
        ee = np.exp(eta)
        return ee * np.exp(-ee) * (1 - ee)


class CauchitLink:
    """Cauchit link: eta = tan(pi*(mu - 0.5)), mu = 0.5 + arctan(eta)/pi.

    The quantile function of the standard Cauchy distribution.
    Heavy-tailed alternative to logit for binary responses.
    """

    def link(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.tan(np.pi * (mu_safe - 0.5))

    def inverse(self, eta: NDArray) -> NDArray:
        return 0.5 + np.arctan(eta) / np.pi

    def deriv(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.pi / np.cos(np.pi * (mu_safe - 0.5)) ** 2

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return 1.0 / (np.pi * (1 + eta**2))

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return -2 * eta / (np.pi * (1 + eta**2) ** 2)


class InverseLink:
    """Inverse (reciprocal) link: eta = 1/mu, mu = 1/eta.

    Canonical link for the Gamma distribution.
    """

    def link(self, mu: NDArray) -> NDArray:
        return 1.0 / mu

    def inverse(self, eta: NDArray) -> NDArray:
        return 1.0 / eta

    def deriv(self, mu: NDArray) -> NDArray:
        return -1.0 / mu**2

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return -1.0 / eta**2

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return 2.0 / eta**3


class InverseSquaredLink:
    """Inverse-squared link: eta = 1/mu^2, mu = 1/sqrt(eta).

    Canonical link for the inverse Gaussian distribution.
    """

    def link(self, mu: NDArray) -> NDArray:
        return 1.0 / mu**2

    def inverse(self, eta: NDArray) -> NDArray:
        return 1.0 / np.sqrt(eta)

    def deriv(self, mu: NDArray) -> NDArray:
        return -2.0 / mu**3

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return -0.5 * eta ** (-1.5)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return 0.75 * eta ** (-2.5)


class SqrtLink:
    """Square-root link: eta = sqrt(mu), mu = eta^2.

    Variance-stabilising link for Poisson data.
    """

    def link(self, mu: NDArray) -> NDArray:
        return np.sqrt(mu)

    def inverse(self, eta: NDArray) -> NDArray:
        return eta**2

    def deriv(self, mu: NDArray) -> NDArray:
        return 0.5 / np.sqrt(np.maximum(mu, 1e-15))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return 2.0 * eta

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return 2.0 * np.ones_like(eta)


class PowerLink:
    """Power link: eta = mu^p, mu = eta^(1/p).

    Generalises identity (p=1), sqrt (p=0.5), inverse (p=-1),
    inverse-squared (p=-2).  The log link is the p→0 limit but is
    handled separately for numerical reasons.

    Parameters
    ----------
    power : float
        The power parameter.  Must not be 0 (use LogLink instead).
    """

    def __init__(self, power: float = 1.0):
        if power == 0:
            raise ValueError("PowerLink(power=0) is the log link — use LogLink instead.")
        self.power = power

    def link(self, mu: NDArray) -> NDArray:
        return np.power(mu, self.power)

    def inverse(self, eta: NDArray) -> NDArray:
        return np.power(np.maximum(eta, 1e-15), 1.0 / self.power)

    def deriv(self, mu: NDArray) -> NDArray:
        return self.power * np.power(mu, self.power - 1)

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        p = self.power
        return (1.0 / p) * np.power(np.maximum(eta, 1e-15), (1.0 / p) - 1)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        p = self.power
        q = 1.0 / p
        return q * (q - 1) * np.power(np.maximum(eta, 1e-15), q - 2)


class NegativeBinomialLink:
    """Negative binomial link: eta = log(mu / (mu + theta)).

    Canonical link for NB2(θ), parametrised so that mu > 0 maps to
    eta in (-inf, 0).

    Parameters
    ----------
    theta : float
        The NB overdispersion parameter (must be > 0).
    """

    def __init__(self, theta: float = 1.0):
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}")
        self.theta = theta

    def link(self, mu: NDArray) -> NDArray:
        return np.log(mu / (mu + self.theta))

    def inverse(self, eta: NDArray) -> NDArray:
        # mu = theta * exp(eta) / (1 - exp(eta))
        # Use expit-style stable computation: exp(eta)/(1 - exp(eta)) = -1/(1 - exp(-eta)) + 1
        e = np.exp(np.clip(eta, -30, 0 - 1e-10))
        return self.theta * e / (1 - e)

    def deriv(self, mu: NDArray) -> NDArray:
        return self.theta / (mu * (mu + self.theta))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        e = np.exp(np.clip(eta, -30, 0 - 1e-10))
        return self.theta * e / (1 - e) ** 2

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        e = np.exp(np.clip(eta, -30, 0 - 1e-10))
        return self.theta * e * (1 + e) / (1 - e) ** 3


_LOG_LINK_ETA_MIN = -80.0
_LOG_LINK_ETA_MAX = 80.0


def stabilize_eta(eta: NDArray, link: Link) -> NDArray:
    """Clip eta only where the inverse link needs protection.

    For the log link, the bounds must be wide enough that the IRLS can reach
    the true MLE for near-separated categories.  exp(-80) ≈ 2e-35 is safely
    above float64 subnormal range for all practical distributions, and well
    beyond the -37 / -43 regime seen in real actuarial data.
    """
    if isinstance(link, IdentityLink):
        return eta
    if isinstance(link, LogLink):
        return np.clip(eta, _LOG_LINK_ETA_MIN, _LOG_LINK_ETA_MAX)
    if isinstance(link, InverseLink | InverseSquaredLink):
        return np.clip(eta, 1e-12, 1e12)
    if isinstance(link, PowerLink):
        if link.power == 1.0:
            return eta
        return np.clip(eta, 1e-12, 1e12)
    if isinstance(link, NegativeBinomialLink):
        return np.clip(eta, -30.0, -1e-10)
    # Catch-all for custom links
    return np.clip(eta, _LOG_LINK_ETA_MIN, _LOG_LINK_ETA_MAX)


_LINK_SHORTCUTS: dict[str, type] = {
    "log": LogLink,
    "identity": IdentityLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "cloglog": CloglogLink,
    "cauchit": CauchitLink,
    "inverse": InverseLink,
    "inverse_squared": InverseSquaredLink,
    "sqrt": SqrtLink,
}


def resolve_link(link: str | Link | None, family: Distribution) -> Link:
    """Resolve a link specification to a Link object.

    Parameters
    ----------
    link : str, Link, or None
        If a Link object, pass through. If a string, look up by name.
        If None, use the family's default link.
    family : Distribution
        The distribution, used to determine the default link.
    """
    if isinstance(link, Link):
        return link
    if link is None:
        link = family.default_link
    if isinstance(link, str):
        if link not in _LINK_SHORTCUTS:
            raise ValueError(
                f"Unknown link '{link}'. Use one of {list(_LINK_SHORTCUTS)} or pass a Link object."
            )
        return _LINK_SHORTCUTS[link]()
    raise TypeError(f"Expected str, Link, or None, got {type(link)}")
