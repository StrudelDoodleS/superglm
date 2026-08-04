"""Link functions for GLMs.

Each link provides the mapping between the linear predictor (eta) and the
mean (mu), plus the derivative needed by the PIRLS working weights.
"""

from __future__ import annotations

from collections.abc import Callable
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
        deriv3_inverse — d³μ/dη³, used by second-order W(ρ) correction
        (Wood 2011, Appendix D).
        If absent, the correction is skipped and REML falls back
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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = exp(η). Wood (2011) Appendix D."""
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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = 0. Wood (2011) Appendix D."""
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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = μ(1-μ)(1 - 6μ + 6μ²). Wood (2011) Appendix D."""
        from scipy.special import expit

        p = expit(eta)
        return p * (1 - p) * (1 - 6 * p + 6 * p**2)


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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = (η² - 1)·φ(η). Wood (2011) Appendix D."""
        from scipy.stats import norm

        return (eta**2 - 1) * norm.pdf(eta)


class CloglogLink:
    """Complementary log-log link: eta = log(-log(1 - mu)).

    The canonical link for the extreme-value / Gompertz model.
    Asymmetric alternative to logit for binary responses.
    """

    def link(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(np.asarray(mu, dtype=np.float64), 1e-15, 1 - 1e-15)
        return np.log(-np.log1p(-mu_safe))

    def inverse(self, eta: NDArray) -> NDArray:
        with np.errstate(over="ignore"):
            return -np.expm1(-np.exp(np.asarray(eta, dtype=np.float64)))

    def deriv(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(np.asarray(mu, dtype=np.float64), 1e-15, 1 - 1e-15)
        return 1.0 / ((1.0 - mu_safe) * (-np.log1p(-mu_safe)))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        # d/deta [1 - exp(-exp(eta))] = exp(eta) * exp(-exp(eta))
        ee = np.exp(eta)
        return ee * np.exp(-ee)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        # d²/deta² = exp(eta - exp(eta)) * (1 - exp(eta))
        ee = np.exp(eta)
        return ee * np.exp(-ee) * (1 - ee)

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = exp(η-eη)·((1-eη)² - eη). Wood (2011) Appendix D."""
        ee = np.exp(eta)
        return ee * np.exp(-ee) * ((1 - ee) ** 2 - ee)


class CauchitLink:
    """Cauchit link: eta = tan(pi*(mu - 0.5)), mu = 0.5 + arctan(eta)/pi.

    The quantile function of the standard Cauchy distribution.
    Heavy-tailed alternative to logit for binary responses.
    """

    def link(self, mu: NDArray) -> NDArray:
        mu_safe = np.clip(np.asarray(mu, dtype=np.float64), 1e-15, 1 - 1e-15)
        result = np.empty_like(mu_safe)
        lower = mu_safe < 0.25
        upper = mu_safe > 0.75
        central = ~(lower | upper)
        result[lower] = -1.0 / np.tan(np.pi * mu_safe[lower])
        upper_tail = np.maximum(1.0 - mu_safe[upper], 1e-15)
        result[upper] = 1.0 / np.tan(np.pi * upper_tail)
        result[central] = np.tan(np.pi * (mu_safe[central] - 0.5))
        return result

    def inverse(self, eta: NDArray) -> NDArray:
        eta_values = np.asarray(eta, dtype=np.float64)
        result = np.empty_like(eta_values)
        lower = eta_values < -1.0
        upper = eta_values > 1.0
        central = ~(lower | upper)
        result[lower] = np.arctan(-1.0 / eta_values[lower]) / np.pi
        result[upper] = 1.0 - np.arctan(1.0 / eta_values[upper]) / np.pi
        result[central] = 0.5 + np.arctan(eta_values[central]) / np.pi
        return result

    def deriv(self, mu: NDArray) -> NDArray:
        eta = self.link(mu)
        return np.pi * (1.0 + eta**2)

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        eta_values = np.asarray(eta, dtype=np.float64)
        result = np.empty_like(eta_values)
        large = np.abs(eta_values) > 1.0
        reciprocal = 1.0 / eta_values[large]
        result[large] = reciprocal**2 / (np.pi * (1.0 + reciprocal**2))
        result[~large] = 1.0 / (np.pi * (1.0 + eta_values[~large] ** 2))
        return result

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        eta_values = np.asarray(eta, dtype=np.float64)
        result = np.empty_like(eta_values)
        large = np.abs(eta_values) > 1.0
        reciprocal = 1.0 / eta_values[large]
        result[large] = -2.0 * reciprocal**3 / (np.pi * (1.0 + reciprocal**2) ** 2)
        result[~large] = -2.0 * eta_values[~large] / (np.pi * (1.0 + eta_values[~large] ** 2) ** 2)
        return result

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = 2(3η² - 1) / (π(1+η²)³). Wood (2011) Appendix D."""
        eta_values = np.asarray(eta, dtype=np.float64)
        result = np.empty_like(eta_values)
        large = np.abs(eta_values) > 1.0
        reciprocal = 1.0 / eta_values[large]
        result[large] = (
            2.0 * reciprocal**4 * (3.0 - reciprocal**2) / (np.pi * (1.0 + reciprocal**2) ** 3)
        )
        result[~large] = (
            2.0
            * (3.0 * eta_values[~large] ** 2 - 1.0)
            / (np.pi * (1.0 + eta_values[~large] ** 2) ** 3)
        )
        return result


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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = -6/η⁴. Wood (2011) Appendix D."""
        return -6.0 / eta**4


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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = -15/8 · η^{-7/2}. Wood (2011) Appendix D."""
        return -1.875 * eta ** (-3.5)


class SqrtLink:
    """Square-root link: eta = sqrt(mu), mu = eta^2.

    Variance-stabilising link for Poisson data.
    """

    def link(self, mu: NDArray) -> NDArray:
        return np.sqrt(mu)

    def inverse(self, eta: NDArray) -> NDArray:
        return eta**2

    def deriv(self, mu: NDArray) -> NDArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.5 / np.abs(np.sqrt(np.asarray(mu)))

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        return 2.0 * eta

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        return 2.0 * np.ones_like(eta)

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = 0. Wood (2011) Appendix D."""
        return np.zeros_like(eta)


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
        if self.power == 1.0:
            return np.asarray(eta).copy()
        return np.power(np.maximum(eta, 1e-15), 1.0 / self.power)

    def deriv(self, mu: NDArray) -> NDArray:
        return self.power * np.power(mu, self.power - 1)

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        p = self.power
        if p == 1.0:
            return np.ones_like(eta)
        return (1.0 / p) * np.power(np.maximum(eta, 1e-15), (1.0 / p) - 1)

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        p = self.power
        if p == 1.0:
            return np.zeros_like(eta)
        q = 1.0 / p
        return q * (q - 1) * np.power(np.maximum(eta, 1e-15), q - 2)

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = q(q-1)(q-2)·η^{q-3}, q=1/p. Wood (2011) Appendix D."""
        p = self.power
        if p == 1.0:
            return np.zeros_like(eta)
        q = 1.0 / p
        return q * (q - 1) * (q - 2) * np.power(np.maximum(eta, 1e-15), q - 3)


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

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        """d³μ/dη³ = θe(1 + 4e + e²)/(1-e)⁴. Wood (2011) Appendix D."""
        e = np.exp(np.clip(eta, -30, 0 - 1e-10))
        return self.theta * e * (1 + 4 * e + e**2) / (1 - e) ** 4


_LOG_LINK_ETA_MIN = -80.0
_LOG_LINK_ETA_MAX = 80.0
_SQRT_LINK_ETA_MAX = 0.5 * float(np.sqrt(np.finfo(np.float64).max))
_CAUCHIT_PROBABILITY_EPS = 1e-15
_CAUCHIT_LINK_ETA_MAX = float(abs(CauchitLink().link(np.array([_CAUCHIT_PROBABILITY_EPS]))[0]))
_CLOGLOG_PROBABILITY_EPS = 1e-6
_CLOGLOG_LINK_ETA_MAX = float(CloglogLink().link(np.array([1.0 - _CLOGLOG_PROBABILITY_EPS]))[0])


def _identity_eta(eta: NDArray, _link: Link) -> NDArray:
    return eta


def _log_eta(eta: NDArray, _link: Link) -> NDArray:
    return np.clip(eta, _LOG_LINK_ETA_MIN, _LOG_LINK_ETA_MAX)


def _binary_eta(eta: NDArray, _link: Link) -> NDArray:
    # Keeps inverse-link derivatives away from numerical zero in IRLS.
    return np.clip(eta, -20.0, 20.0)


def _cloglog_eta(eta: NDArray, _link: Link) -> NDArray:
    # The positive tail saturates much earlier than logit: cap it at the
    # representable probability endpoint used by CloglogLink.link().
    return np.clip(
        np.asarray(eta, dtype=np.float64),
        -20.0,
        _CLOGLOG_LINK_ETA_MAX,
    )


def _cauchit_eta(eta: NDArray, _link: Link) -> NDArray:
    # Unlike the exponential-tail links, cauchit's inverse approaches the
    # endpoints polynomially.  Its useful eta range is therefore set by the
    # same probability resolution used by CauchitLink.link(), not logit's band.
    return np.clip(
        np.asarray(eta, dtype=np.float64),
        -_CAUCHIT_LINK_ETA_MAX,
        _CAUCHIT_LINK_ETA_MAX,
    )


def _positive_eta(eta: NDArray, _link: Link) -> NDArray:
    return np.clip(eta, 1e-12, 1e12)


def _sqrt_eta(eta: NDArray, _link: Link) -> NDArray:
    # The inverse μ = η² has two fitted branches even though the forward link
    # returns the principal square root.  Preserve both and bound only where
    # squaring would overflow, rather than inheriting a binary-link band.
    return np.clip(
        np.asarray(eta, dtype=np.float64),
        -_SQRT_LINK_ETA_MAX,
        _SQRT_LINK_ETA_MAX,
    )


def _power_eta(eta: NDArray, link: Link) -> NDArray:
    assert isinstance(link, PowerLink)
    if link.power == 1.0:
        return eta
    return np.clip(eta, 1e-12, 1e12)


def _negative_binomial_eta(eta: NDArray, _link: Link) -> NDArray:
    return np.clip(eta, -30.0, -1e-10)


_STABILIZE_ETA_BY_LINK_TYPE: dict[type, Callable[[NDArray, Link], NDArray]] = {
    IdentityLink: _identity_eta,
    LogLink: _log_eta,
    LogitLink: _binary_eta,
    ProbitLink: _binary_eta,
    CloglogLink: _cloglog_eta,
    CauchitLink: _cauchit_eta,
    InverseLink: _positive_eta,
    InverseSquaredLink: _positive_eta,
    SqrtLink: _sqrt_eta,
    PowerLink: _power_eta,
    NegativeBinomialLink: _negative_binomial_eta,
}


def stabilize_eta(eta: NDArray, link: Link) -> NDArray:
    """Clip eta only where the inverse link needs protection.

    For the log link, the bounds must be wide enough that the IRLS can reach
    the true MLE for near-separated categories.  exp(-80) ≈ 2e-35 is safely
    above float64 subnormal range for all practical distributions, and well
    beyond the -37 / -43 regime seen in real actuarial data.
    """
    for link_type, stabilizer in _STABILIZE_ETA_BY_LINK_TYPE.items():
        if isinstance(link, link_type):
            return stabilizer(eta, link)
    # Unregistered custom links retain a conservative fallback.
    return np.clip(eta, -20.0, 20.0)


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
