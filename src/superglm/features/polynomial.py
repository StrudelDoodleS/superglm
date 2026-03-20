"""Orthogonal polynomial feature using Legendre basis.

Stable alternative to P-splines for features with simple monotone or
quadratic shapes.  Group lasso selects or removes the entire polynomial
as a unit.  Degree 2-3 is the typical insurance choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.typing import NDArray

from superglm.types import GroupInfo


class Polynomial:
    """Orthogonal polynomial feature (Legendre basis).

    Scales x to [-1, 1] using training-data min/max, then builds a
    Legendre polynomial basis of degrees 1 through ``degree`` (the
    degree-0 constant is excluded — the model intercept handles it).

    Group size = ``degree``.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree.  2 (quadratic) or 3 (cubic) are the
        standard insurance choices.  Higher values are allowed but rarely
        useful.
    """

    def __init__(self, degree: int = 3):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        self.degree = degree
        self._lo: float = 0.0
        self._hi: float = 1.0

    def __repr__(self) -> str:
        return f"Polynomial(degree={self.degree})"

    def _scale(self, x: NDArray) -> NDArray:
        """Scale x to [-1, 1] using stored min/max."""
        span = self._hi - self._lo
        if span < 1e-12:
            return np.zeros_like(x)
        return 2.0 * (x - self._lo) / span - 1.0

    def _basis(self, x_scaled: NDArray) -> NDArray:
        """Legendre basis for degrees 1..degree (exclude degree 0)."""
        # legvander returns columns for degrees 0, 1, ..., degree
        return legvander(x_scaled, self.degree)[:, 1:]

    def build(
        self,
        x: NDArray[np.floating],
        exposure: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Build Legendre basis columns after learning min/max from *x*."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self._lo, self._hi = float(x.min()), float(x.max())
        cols = self._basis(self._scale(x))
        return GroupInfo(columns=cols, n_cols=self.degree)

    def transform(self, x: NDArray) -> NDArray:
        """Scale *x* using fitted min/max and return the Legendre basis matrix."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return self._basis(self._scale(x))

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        """Evaluate the fitted polynomial on a grid and return relativities."""
        x_grid = np.linspace(self._lo, self._hi, n_points)
        P_grid = self._basis(self._scale(x_grid))
        log_rels = P_grid @ beta
        return {
            "x": x_grid,
            "log_relativity": log_rels,
            "relativity": np.exp(log_rels),
            "degree": self.degree,
            "coefficients": beta,
        }
