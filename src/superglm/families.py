"""Convenience constructors for distribution families.

Usage::

    from superglm import families

    model = SuperGLM(family=families.poisson(), ...)
    model = SuperGLM(family=families.tweedie(p=1.5), ...)
    model = SuperGLM(family=families.nb2(theta=1.0), ...)

Simple (parameter-free) families can also be specified as strings::

    model = SuperGLM(family="poisson", ...)
"""

from superglm.distributions import (
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
)


def poisson() -> Poisson:
    """Poisson distribution (variance = mean)."""
    return Poisson()


def gaussian() -> Gaussian:
    """Gaussian (Normal) distribution."""
    return Gaussian()


def gamma() -> Gamma:
    """Gamma distribution."""
    return Gamma()


def binomial() -> Binomial:
    """Binomial distribution for binary classification."""
    return Binomial()


def nb2(theta: float | str = "auto") -> NegativeBinomial:
    """Negative Binomial (NB2) distribution.

    Parameters
    ----------
    theta : float or "auto"
        Overdispersion parameter.  ``"auto"`` estimates theta via
        profile likelihood during ``fit()``.
    """
    return NegativeBinomial(theta=theta)


def tweedie(p: float) -> Tweedie:
    """Tweedie distribution.

    Parameters
    ----------
    p : float
        Power parameter, must be in (1, 2).
        p → 1 approaches Poisson, p → 2 approaches Gamma.
    """
    return Tweedie(p=p)
