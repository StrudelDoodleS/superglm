"""Categorical feature: one-hot dummies as a single group.

Group lasso can zero out the entire factor (all levels shrink to base
simultaneously), which is the correct variable selection behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo


class Categorical:
    """One-hot encoded categorical feature.

    Parameters
    ----------
    base : str
        How to choose the reference level.
        'most_exposed' - level with highest total exposure (default, best for insurance)
        'first'        - alphabetically first level
        Or pass a specific level name as a string.
    """

    def __init__(self, base: str = "most_exposed"):
        self.base = base
        self._levels: list[str] = []
        self._base_level: str = ""
        self._non_base: list[str] = []

    def build(
        self,
        x: NDArray,
        exposure: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        x = np.asarray(x).ravel()
        self._levels = sorted(np.unique(x).tolist())

        if len(self._levels) < 2:
            raise ValueError(f"Categorical needs >= 2 levels, got {len(self._levels)}")

        # Choose base level (reuse from prior fit if already determined)
        if self._base_level and self._base_level in self._levels:
            pass
        elif self.base == "most_exposed" and exposure is not None:
            exp_by_level = {lev: float(exposure[x == lev].sum()) for lev in self._levels}
            self._base_level = max(exp_by_level, key=exp_by_level.get)
        elif self.base == "most_exposed" and exposure is None:
            self._base_level = self._levels[0]
        elif self.base == "first":
            self._base_level = self._levels[0]
        elif self.base in self._levels:
            self._base_level = self.base
        else:
            raise ValueError(f"Base '{self.base}' not found in levels: {self._levels}")

        self._non_base = [lev for lev in self._levels if lev != self._base_level]

        # One-hot encode (excluding base)
        columns = np.column_stack(
            [(x == lev).astype(np.float64) for lev in self._non_base]
        )

        return GroupInfo(columns=columns, n_cols=len(self._non_base))

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Coefficients -> relativity table."""
        relativities = {self._base_level: 1.0}
        log_rels = {self._base_level: 0.0}
        for i, lev in enumerate(self._non_base):
            log_rels[lev] = float(beta[i])
            relativities[lev] = float(np.exp(beta[i]))
        return {
            "base_level": self._base_level,
            "levels": self._levels,
            "log_relativities": log_rels,
            "relativities": relativities,
        }
