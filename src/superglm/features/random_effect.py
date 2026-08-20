"""Random-effect feature specification."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.types import GroupInfo, LambdaPolicy


class RandomEffect:
    """All-level categorical effect with a REML-estimated variance component.

    ``levels=`` binds the level universe (spec 2026-08-11, §3.1) from an
    explicit sequence, a data column, or a categorical dtype.  A declared level
    with no training rows is not pinned the way an unpenalized dummy is: it
    keeps its own coefficient and shrinks to the population value through the
    variance component, exactly as a thinly observed level does.

    Notes
    -----
    When a REML-estimated ``RandomEffect`` is fitted beside an unpenalised
    ``Categorical`` whose levels include some with exposure but no positive
    response (under a log link with a zero-mass family such as Tweedie or
    Poisson), those levels separate -- their coefficients have no finite MLE
    -- and the marginal likelihood becomes nearly flat in this term's
    variance.  The fitted variance component is then poorly determined, and
    for the estimated-scale Tweedie criterion it is additionally biased
    upward relative to exact-likelihood REML.  ``fit_reml`` warns on that
    configuration; treat the published ``tau_squared`` with care there.
    """

    requires_reml = True

    def __init__(
        self,
        *,
        levels=None,
        unseen: Literal["population", "error"] = "population",
        missing: Literal["error"] = "error",
        lambda_policy: LambdaPolicy | None = None,
    ):
        from superglm.features._level_source import resolve_level_source

        if unseen not in ("population", "error"):
            raise ValueError(f"unseen must be 'population' or 'error', got {unseen!r}")
        if missing != "error":
            raise ValueError(f"missing must be 'error', got {missing!r}")
        if lambda_policy is not None and not isinstance(lambda_policy, LambdaPolicy):
            raise TypeError("lambda_policy must be a LambdaPolicy or None")

        self.unseen = unseen
        self.missing = missing
        self._lambda_policy = lambda_policy
        self._declared_levels: list | None = (
            None if levels is None else resolve_level_source(levels, context="RandomEffect")
        )
        self._level_source: str = "declared" if levels is not None else "inferred"
        self._levels: list[Any] = []
        self._level_to_code: dict[Any, int] = {}

    def adopt_dtype_categories(self, categories: list) -> None:
        """Adopt a dtype-declared universe unless one is already declared."""
        if self._declared_levels is None:
            from superglm.features._level_source import resolve_level_source

            self._declared_levels = resolve_level_source(list(categories), context="RandomEffect")
            self._level_source = "dtype"

    def apply_level_binding(self, binding) -> None:
        """Adopt a full-frame universe when nothing more specific declared one.

        Only the levels are read: a penalized term has no base level, so its
        bindings carry ``base=None`` and there is nothing to pin.
        """
        if self._declared_levels is None and binding.levels is not None:
            self._declared_levels = list(binding.levels)
            self._level_source = "full-frame"

    def resolve_binding(self, values: NDArray, sample_weight=None):
        """Compute this spec's full-frame binding without mutating the spec."""
        import copy

        from superglm.types import LevelBinding

        # Build on a throwaway copy so the universe and its NaN checks stay
        # single-sourced in `build`.
        probe = copy.deepcopy(self)
        probe.build(values, sample_weight=sample_weight)
        return LevelBinding(levels=tuple(probe._levels), base=None)

    def _declared_codes(self, values: NDArray) -> NDArray[np.intp]:
        """Code *values* against the bound universe, rejecting anything outside it."""
        codes = pd.Index(self._levels).get_indexer(values).astype(np.intp, copy=False)
        if np.any(codes < 0):
            # A -1 under a bound universe is either a broken column or data the
            # declaration does not admit; those are different bugs.
            outside = values[codes < 0]
            if np.any(pd.isna(outside)):
                raise ValueError("RandomEffect column contains missing values (NaN or None).")
            raise ValueError(
                f"Training data contains levels outside the declared level universe: "
                f"{sorted(set(outside.tolist()), key=str)}. Declared: "
                f"{sorted(self._levels, key=str)}. Widen levels= or fix the column."
            )
        return codes

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Factorize all fitted levels without dropping a reference category."""
        del sample_weight
        values = np.asarray(x).ravel()
        if self._declared_levels is not None:
            self._levels = list(self._declared_levels)
            codes = self._declared_codes(values)
        else:
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
