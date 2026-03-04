"""
Core types and protocols.

Every feature spec (Spline, Categorical, Numeric) implements FeatureSpec.
The solver never sees feature types — only GroupInfo objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ── What a feature spec must provide ────────────────────────────
@runtime_checkable
class FeatureSpec(Protocol):
    """Protocol for feature specifications.

    build()       -> columns + metadata for the solver
    reconstruct() -> turn fitted coefficients back into interpretable output
    """

    def build(
        self,
        x: NDArray[np.floating],
        exposure: NDArray[np.floating] | None = None,
    ) -> "GroupInfo": ...

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]: ...


# ── What the solver actually sees ───────────────────────────────
@dataclass
class GroupInfo:
    """Uniform interface between feature specs and the solver.

    The solver receives a list of these. It doesn't know or care
    whether the columns came from a spline, a one-hot encoding,
    or a single numeric feature.
    """

    columns: NDArray[np.floating]         # (n, p_g) design sub-matrix
    n_cols: int                           # p_g — group size
    penalty_matrix: NDArray | None = None # (p_g, p_g) for SSP, else None
    reparametrize: bool = False           # whether to apply SSP transform

    def __post_init__(self):
        assert self.columns.shape[1] == self.n_cols
        if self.penalty_matrix is not None:
            assert self.penalty_matrix.shape == (self.n_cols, self.n_cols)


# ── Group bookkeeping for the solver ────────────────────────────
@dataclass
class GroupSlice:
    """Tracks where each group lives in the full coefficient vector."""

    name: str
    start: int
    end: int
    weight: float = 1.0  # sqrt(p_g) by default, or adaptive weight

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def sl(self) -> slice:
        return slice(self.start, self.end)
