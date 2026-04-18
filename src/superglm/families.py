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
    """Create a Poisson family object."""
    return Poisson()


def gaussian() -> Gaussian:
    """Create a Gaussian family object."""
    return Gaussian()


def gamma() -> Gamma:
    """Create a Gamma family object."""
    return Gamma()


def binomial() -> Binomial:
    """Create a Binomial family object."""
    return Binomial()


def nb2(theta: float | str = "auto") -> NegativeBinomial:
    """Create a negative binomial (NB2) family object.

    Parameters
    ----------
    theta : float or "auto"
        Overdispersion parameter.  ``"auto"`` estimates theta via
        profile likelihood during ``fit()``.
    """
    return NegativeBinomial(theta=theta)


def tweedie(p: float) -> Tweedie:
    """Create a Tweedie family object.

    Parameters
    ----------
    p : float
        Power parameter, must be in (1, 2).
        p → 1 approaches Poisson, p → 2 approaches Gamma.
    """
    return Tweedie(p=p)
