"""B-spline basis with optional SSP reparametrisation.

Knots are penalised via P-spline (Eilers & Marx, 1996), so 15-20 interior
knots is a safe default. More knots gives the penalty more flexibility to
capture the shape — it will not cause overfitting because the second-difference
penalty controls smoothness, not the knot count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.types import GroupInfo


class Spline:
    """P-spline feature with B-spline basis and difference penalty.

    Parameters
    ----------
    n_knots : int
        Number of interior knots. 15-20 is recommended. Because the
        second-difference penalty controls smoothness, adding more knots
        does not cause overfitting — it just gives the smoother more
        flexibility to capture the true shape.
    degree : int
        B-spline polynomial degree. 3 (cubic, default) gives C2 smooth
        curves and is the standard choice. 1 (linear) and 2 (quadratic)
        are acceptable alternatives. Values above 3 are not advised —
        numerical issues increase with no practical benefit.
    knot_strategy : str
        "quantile" places knots at data quantiles, "uniform" spaces evenly.
    penalty : str
        "ssp" enables SSP reparametrisation, "none" disables it.
    """

    def __init__(
        self,
        n_knots: int = 15,
        degree: int = 3,
        knot_strategy: str = "quantile",
        penalty: str = "ssp",
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.knot_strategy = knot_strategy
        self.penalty = penalty

        self._knots: NDArray = np.array([])
        self._n_basis: int = 0
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._R_inv: NDArray | None = None

    def build(self, x: NDArray, exposure: NDArray | None = None) -> GroupInfo:
        x = np.asarray(x, dtype=np.float64).ravel()
        self._lo, self._hi = float(x.min()), float(x.max())
        pad = (self._hi - self._lo) * 1e-6

        if self.knot_strategy == "quantile":
            probs = np.linspace(0, 100, self.n_knots + 2)[1:-1]
            interior = np.percentile(x, probs)
        else:
            interior = np.linspace(self._lo, self._hi, self.n_knots + 2)[1:-1]

        self._knots = np.concatenate([
            np.repeat(self._lo - pad, self.degree + 1),
            interior,
            np.repeat(self._hi + pad, self.degree + 1),
        ])
        self._n_basis = len(self._knots) - self.degree - 1

        x_clip = np.clip(x, self._knots[0], self._knots[-1])
        B = BSpl.design_matrix(x_clip, self._knots, self.degree).toarray()

        D2 = np.diff(np.eye(self._n_basis), n=2, axis=0)
        omega = D2.T @ D2

        return GroupInfo(
            columns=B, n_cols=self._n_basis,
            penalty_matrix=omega,
            reparametrize=(self.penalty == "ssp"),
        )

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        self._R_inv = R_inv

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
        x_grid = np.linspace(self._lo, self._hi, n_points)
        x_clip = np.clip(x_grid, self._knots[0], self._knots[-1])
        B_grid = BSpl.design_matrix(x_clip, self._knots, self.degree).toarray()
        log_rels = B_grid @ beta_orig
        return {
            "x": x_grid,
            "log_relativity": log_rels,
            "relativity": np.exp(log_rels),
            "knots_interior": self._knots[self.degree + 1:-(self.degree + 1)],
            "coefficients_original": beta_orig,
        }

    def evaluate(self, x: NDArray, beta: NDArray) -> NDArray:
        beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
        x = np.asarray(x, dtype=np.float64).ravel()
        x_clip = np.clip(x, self._knots[0], self._knots[-1])
        B = BSpl.design_matrix(x_clip, self._knots, self.degree).toarray()
        return B @ beta_orig
