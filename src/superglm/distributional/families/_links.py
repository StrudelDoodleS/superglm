"""Reusable link functions shared by the distributional families.

``BoundedLogitLink`` maps an open interval ``(lower, upper)`` to the real line
with a logit, so a parameter that misbehaves at both edges of its support can
carry an unconstrained linear predictor.  The inverse is strictly interior for
every finite ``eta``, which is what ``_validate_link_support`` requires of a
bounded link at its ``eta = +-20`` probe.

``BoundedPowerLink`` (Tweedie) is deliberately left alone: it has a byte-identical
golden fixture behind it and no reason to move.

Imports only numpy, scipy and the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

_PROBE = 20.0


def _finite_wall(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} wall must be a finite real number")
    try:
        wall = float(value)  # ty: ignore[invalid-argument-type] -- validated conversion boundary
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} wall must be a finite real number") from exc
    if not math.isfinite(wall):
        raise ValueError(f"{name} wall must be a finite real number")
    return wall


@dataclass(frozen=True)
class BoundedLogitLink:
    """Logit between two configured walls: ``value = lower + (upper - lower) expit(eta)``."""

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        lower = _finite_wall(self.lower, name="lower")
        upper = _finite_wall(self.upper, name="upper")
        if not lower < upper:
            raise ValueError("bounded logit walls must be strictly ordered")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        edges = self.inverse(np.array([-_PROBE, _PROBE]))
        if not (edges[0] > lower and edges[1] < upper):
            raise ValueError(
                "bounded logit walls are too close together for the inverse to stay strictly "
                "inside them in float64"
            )

    @property
    def _span(self) -> float:
        return self.upper - self.lower

    def _interior(self, value: NDArray, *, name: str) -> NDArray[np.float64]:
        try:
            values = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} input must be finite and strictly between the walls") from exc
        if (
            not np.all(np.isfinite(values))
            or np.any(values <= self.lower)
            or np.any(values >= self.upper)
        ):
            raise ValueError(f"{name} input must be finite and strictly between the walls")
        return values

    def link(self, mu: NDArray) -> NDArray[np.float64]:
        values = self._interior(mu, name="link")
        return np.log(values - self.lower) - np.log(self.upper - values)

    def inverse(self, eta: NDArray) -> NDArray[np.float64]:
        probability = expit(np.asarray(eta, dtype=np.float64))
        return self.lower + self._span * probability

    def deriv(self, mu: NDArray) -> NDArray[np.float64]:
        values = self._interior(mu, name="derivative")
        return self._span / ((values - self.lower) * (self.upper - values))

    def deriv_inverse(self, eta: NDArray) -> NDArray[np.float64]:
        probability = expit(np.asarray(eta, dtype=np.float64))
        return self._span * probability * (1.0 - probability)

    def deriv2_inverse(self, eta: NDArray) -> NDArray[np.float64]:
        probability = expit(np.asarray(eta, dtype=np.float64))
        return self._span * probability * (1.0 - probability) * (1.0 - 2.0 * probability)

    def deriv3_inverse(self, eta: NDArray) -> NDArray[np.float64]:
        """``d^3 mu / d eta^3``; Wood (2011) Appendix D uses it in the REML W-correction."""
        probability = expit(np.asarray(eta, dtype=np.float64))
        return (
            self._span
            * probability
            * (1.0 - probability)
            * (1.0 - 6.0 * probability + 6.0 * probability * probability)
        )


__all__ = ["BoundedLogitLink"]
