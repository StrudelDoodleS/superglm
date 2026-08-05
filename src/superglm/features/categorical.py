"""Categorical feature: one-hot dummies as a single group.

Group lasso can zero out the entire factor (all levels shrink to base
simultaneously), which is the correct variable selection behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.types import GroupInfo


def _validate_observed_categorical_levels(
    observed: set,
    known_levels: set,
    *,
    context: str = "",
) -> None:
    """Raise for observed levels outside a fitted categorical domain."""
    unseen = observed - known_levels
    if unseen:
        # Sort by str: a domain may legitimately mix types (numeric bands beside a
        # labelled special, `order=[1..6], specials=["MISSING"]`), and a bare
        # sorted() raises TypeError comparing int to str -- turning a clean report
        # of a genuinely unseen level into a crash from the error path itself.
        msg = (
            f"Encountered unseen categorical levels at predict time: "
            f"{sorted(unseen, key=str)}. All levels must be among those seen "
            f"during fit: {sorted(known_levels, key=str)}."
        )
        if context:
            msg = f"[{context}] {msg}"
        raise ValueError(msg)


def _validate_categorical_levels(x: NDArray, known_levels: set, *, context: str = "") -> None:
    """Raise ValueError if x contains levels not seen during fit.

    Parameters
    ----------
    x : array-like
        Categorical values to validate.
    known_levels : set
        Levels seen during build() / fit().
    context : str, optional
        Feature name for error message context.
    """
    # Check for missing values before np.unique — np.unique raises TypeError
    # on mixed object arrays like ["B", np.nan].
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in x):
        msg = "Categorical column contains missing values (NaN or None)."
        if context:
            msg = f"[{context}] {msg}"
        raise ValueError(msg)

    try:
        observed = set(np.unique(x).tolist())
    except TypeError:
        # np.unique sorts, and an object column holding more than one type has no
        # order: `["MISSING", 1]` raises TypeError comparing str to int. Membership
        # is all this function needs, so fall back to a plain set. Kept as a
        # fallback rather than the default because np.unique is much faster on the
        # homogeneous columns that are the common case.
        observed = set(np.asarray(x).ravel().tolist())
    _validate_observed_categorical_levels(
        observed,
        known_levels,
        context=context,
    )


def _grouping_labels(x: NDArray) -> NDArray:
    """Return string labels used by LevelGrouping, preserving missing-value checks."""
    import pandas as pd

    x = np.asarray(x).ravel()
    if np.asarray(pd.isna(x)).any():
        raise ValueError("Categorical column contains missing values (NaN or None).")
    return pd.Series(x).astype(str).to_numpy()


def _resolve_categorical_labels(
    x: NDArray,
    grouping,
    *,
    known_levels: set | None = None,
    context: str = "",
) -> NDArray:
    """Resolve one categorical parent's raw labels into its fitted level domain.

    A grouped categorical's public input contract is always the ORIGINAL
    labels.  Group labels are not accepted as an alternative input domain:
    they may themselves be original labels with a different mapping.  Validate
    the raw domain first, apply the mapping exactly once, then (when fitted
    levels are supplied) certify that the collapsed level was present at fit.
    """
    x = np.asarray(x).ravel()
    if grouping is None:
        if known_levels is not None:
            _validate_categorical_levels(x, known_levels, context=context)
        return x

    import pandas as pd

    labels = _grouping_labels(x)
    _validate_observed_categorical_levels(
        set(pd.unique(labels).tolist()),
        set(grouping.all_original_levels),
        context=context,
    )
    mapping = grouping.original_to_group
    resolved = pd.Series(labels, copy=False).map(mapping).to_numpy()
    if known_levels is not None:
        _validate_observed_categorical_levels(
            set(pd.unique(resolved).tolist()),
            known_levels,
            context=context,
        )
    return resolved


class Categorical:
    """One-hot encoded categorical feature.

    Parameters
    ----------
    base : str
        How to choose the reference level.
        'most_exposed' - level with highest total sample_weight (default, best for insurance)
        'first'        - alphabetically first level
        Or pass a specific level name as a string.
    """

    def __init__(self, base: str = "most_exposed", grouping=None):
        self.base = base
        self._grouping = grouping
        self._levels: list[str] = []
        self._base_level: str = ""
        self._non_base: list[str] = []

    def __repr__(self) -> str:
        n = len(self._levels)
        if n:
            return f"Categorical(base={self.base!r}, {n} levels, ref={self._base_level!r})"
        return f"Categorical(base={self.base!r})"

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Build sparse one-hot design columns, choosing the base level from *x*."""
        import pandas as pd

        x = _resolve_categorical_labels(x, self._grouping)

        # Single-pass O(n) factorize — avoids O(n log n) sort + O(n * levels) loop
        codes, uniques = pd.factorize(x, sort=True)

        # pd.factorize encodes NaN/None as -1 in codes. Reject them here
        # so they don't silently corrupt the design matrix.
        if (codes == -1).any():
            raise ValueError("Categorical column contains missing values (NaN or None).")

        self._levels = uniques.tolist()

        if len(self._levels) < 2:
            raise ValueError(f"Categorical needs >= 2 levels, got {len(self._levels)}")

        # Choose base level (reuse from prior fit if already determined)
        if self._base_level and self._base_level in self._levels:
            pass
        elif self.base == "most_exposed" and sample_weight is not None:
            # O(n) bincount instead of O(n * levels) dict comprehension
            exposure_per_level = np.bincount(
                codes, weights=sample_weight, minlength=len(self._levels)
            )
            self._base_level = self._levels[int(np.argmax(exposure_per_level))]
        elif self.base == "most_exposed" and sample_weight is None:
            self._base_level = self._levels[0]
        elif self.base == "first":
            self._base_level = self._levels[0]
        elif self.base in self._levels:
            self._base_level = self.base
        else:
            raise ValueError(f"Base '{self.base}' not found in levels: {self._levels}")

        self._non_base = [lev for lev in self._levels if lev != self._base_level]

        # Remap codes: drop base level, produce 0-based codes for non-base levels.
        # Base-level observations are excluded from the design matrix entirely
        # (absorbed into the intercept), so we encode them as -1.
        base_idx = self._levels.index(self._base_level)
        n_levels = len(self._non_base)
        remap = np.empty(len(self._levels), dtype=np.intp)
        remap[base_idx] = -1
        col = 0
        for i in range(len(self._levels)):
            if i != base_idx:
                remap[i] = col
                col += 1
        cat_codes = remap[codes]

        return GroupInfo(columns=None, n_cols=n_levels, cat_codes=cat_codes)

    def transform(self, x: NDArray) -> NDArray:
        """One-hot encode using levels learned during build()."""
        x = _resolve_categorical_labels(
            x,
            self._grouping,
            known_levels=set(self._levels),
        )
        return np.column_stack([(x == lev).astype(np.float64) for lev in self._non_base])

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted categorical contribution directly on new data."""
        import pandas as pd

        x = _resolve_categorical_labels(
            x,
            self._grouping,
            known_levels=set(self._levels),
        )

        codes = pd.Categorical(x, categories=self._levels).codes
        level_effects = np.zeros(len(self._levels), dtype=np.float64)
        for i, lev in enumerate(self._non_base):
            level_effects[self._levels.index(lev)] = float(beta[i])
        return level_effects[codes]

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
