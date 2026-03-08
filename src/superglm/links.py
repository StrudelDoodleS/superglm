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


_LINK_SHORTCUTS: dict[str, type] = {
    "log": LogLink,
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
