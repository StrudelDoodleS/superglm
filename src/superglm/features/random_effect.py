"""Random-effect feature specification."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.types import GroupInfo, LambdaPolicy


class RandomEffect:
    """All-level categorical effect with a REML-estimated variance component."""

    requires_reml = True

    def __init__(
        self,
        *,
        unseen: Literal["population", "error"] = "population",
        missing: Literal["error"] = "error",
        lambda_policy: LambdaPolicy | None = None,
    ):
        if unseen not in ("population", "error"):
            raise ValueError(f"unseen must be 'population' or 'error', got {unseen!r}")
        if missing != "error":
            raise ValueError(f"missing must be 'error', got {missing!r}")
        if lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy or None")

        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self._levels: list[Any] = []
        self._level_to_code: dict[Any, int] = {}

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Factorize all fitted levels without dropping a reference category."""
        del sample_weight
        values = np.asarray(x).ravel()
        codes, uniques = pd.factorize(values, sort=True)
        if np.any(codes < 0):
            raise ValueError("RandomEffect column contains missing values (NaN or None).")
        self._levels = uniques.tolist()
        self._level_to_code = {level: code for code, level in enumerate(self._levels)}
        return GroupInfo(
            columns=None,
            n_cols=len(self._levels),
            penalized=True,
            cat_codes=codes.astype(np.intp, copy=False),
            lambda_policies=(
                None if self._lambda_policy is None else {"_default": self._lambda_policy}
            ),
            structured_kind="random_effect",
        )

    def validate_prediction_values(self, x: NDArray) -> None:
        """Reject missing values without applying the unseen-level policy."""
        values = np.asarray(x).ravel()
        if np.any(pd.isna(values)):
            raise ValueError("RandomEffect column contains missing values (NaN or None).")

    def _prediction_codes(self, x: NDArray) -> NDArray[np.intp]:
        values = np.asarray(x).ravel()
        self.validate_prediction_values(values)
        codes = pd.Index(self._levels).get_indexer(values).astype(np.intp, copy=False)
        unseen_mask = codes < 0
        if self.unseen == "error" and np.any(unseen_mask):
            unseen = pd.unique(values[unseen_mask]).tolist()
            raise ValueError(f"Encountered unseen RandomEffect levels: {unseen}.")
        return codes

    def score(
        self,
        x: NDArray,
        beta: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Select fitted level effects without materializing one-hot columns."""
        codes = self._prediction_codes(x)
        effects = np.zeros(len(codes), dtype=np.float64)
        known = codes >= 0
        effects[known] = np.asarray(beta, dtype=np.float64)[codes[known]]
        return effects

    def transform(self, x: NDArray) -> NDArray[np.floating]:
        """Materialize a small all-level one-hot reference matrix."""
        codes = self._prediction_codes(x)
        transformed = np.zeros((len(codes), len(self._levels)), dtype=np.float64)
        known = codes >= 0
        transformed[np.flatnonzero(known), codes[known]] = 1.0
        return transformed

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Return one fitted effect for every represented level."""
        effects = {
            level: float(value)
            for level, value in zip(self._levels, np.asarray(beta).ravel(), strict=True)
        }
        return {
            "levels": self._levels.copy(),
            "effects": effects,
            "log_relativities": effects.copy(),
            "relativities": {level: float(np.exp(value)) for level, value in effects.items()},
        }
