"""Saved-state classes of the retired SCOP (shape constrained P-spline) engine.

SCOP (Pya & Wood 2015, "Shape constrained additive models", Statistics and
Computing, 25, 543-559) fitted shape-constrained P-splines up to 0.41.0. A
model it fitted keeps these maps on its groups (``scop_reparameterization``),
and inference reads their penalty, so the classes stay for such a model to
load, predict and summarize. Nothing builds them now: a refit goes to the QP
engine.

- ``SCOPReparameterization``: the raw-space map, ``gamma = N b + s Sigma exp(c)``.
- ``SCOPSolverReparam``: the solver-space map after the constant direction
  is dropped. Monotone kinds keep no identity coordinate, curvature kinds one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _difference_penalty(width: int, start: int, shape_dim: int) -> NDArray:
    """First-difference penalty ``D'D`` on the trailing ``shape_dim`` coordinates."""
    S = np.zeros((width, width), dtype=np.float64)
    if shape_dim > 1:
        D = np.diff(np.eye(shape_dim), axis=0)
        S[start:, start:] = D.T @ D
    return S


@dataclass
class SCOPReparameterization:
    """Raw-space SCOP reparameterization for one shape-constrained smooth."""

    q: int
    kind: str
    null_dim: int
    null_basis: NDArray
    sigma_shape: NDArray
    shape_sign: float

    @property
    def Sigma(self) -> NDArray:
        """The full ``beta_tilde -> gamma`` map."""
        return np.concatenate([self.null_basis, self.shape_matrix], axis=1)

    @property
    def shape_dim(self) -> int:
        return self.q - self.null_dim

    @property
    def shape_matrix(self) -> NDArray:
        return self.shape_sign * self.sigma_shape

    def penalty_matrix(self) -> NDArray:
        """SCOP penalty matrix on the shape block."""
        return _difference_penalty(self.q, self.null_dim, self.shape_dim)


@dataclass
class SCOPSolverReparam:
    """Solver-space SCOP reparameterization after constant removal."""

    q: int
    raw_reparam: SCOPReparameterization

    @property
    def free_dim(self) -> int:
        """Number of identity-mapped coordinates retained in solver space."""
        return self.raw_reparam.null_dim - 1

    @property
    def shape_dim(self) -> int:
        """Number of positivity-mapped shape coordinates."""
        return self.q - self.free_dim

    def penalty_matrix(self) -> NDArray:
        """Penalty matrix in solver space on the shape block."""
        return _difference_penalty(self.q, self.free_dim, self.shape_dim)
