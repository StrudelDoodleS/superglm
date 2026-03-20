"""Numeric feature: single column, group of size 1."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo


class Numeric:
    """A single continuous feature used as-is.

    Group lasso on a size-1 group = standard L1 soft-thresholding.
    The column is passed through without any transformation.
    """

    def __repr__(self) -> str:
        return "Numeric()"

    def build(
        self,
        x: NDArray[np.floating],
        exposure: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Build a single-column design matrix."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return GroupInfo(columns=x.reshape(-1, 1), n_cols=1)

    def transform(self, x: NDArray) -> NDArray:
        """Transform new data (pass-through)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return x.reshape(-1, 1)

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Reconstruct the coefficient on the original scale."""
        b = float(beta[0])
        return {
            "coef": b,
            "relativity_per_unit": float(np.exp(b)),
        }
