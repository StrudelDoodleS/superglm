"""sklearn-compatible wrapper for SuperGLM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.model import SuperGLM
from superglm.penalties.base import Penalty
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso


_PENALTY_SHORTCUTS = {
    "group_lasso": GroupLasso,
    "sparse_group_lasso": SparseGroupLasso,
    "ridge": Ridge,
}


class SuperGLMRegressor(BaseEstimator, RegressorMixin):
    """Penalised GLM with pluggable penalties for variable selection.

    Automatically detects feature types from the DataFrame:
    - object/category/string dtype columns -> Categorical (one-hot with base level)
    - columns listed in ``spline_features`` -> Spline (P-spline basis)
    - all other numeric columns -> Numeric (single standardized column)

    Parameters
    ----------
    family : str
        Distribution family: "poisson", "gamma", or "tweedie".
    tweedie_p : float or None
        Tweedie power parameter, required if family="tweedie". Must be in (1, 2).
    penalty : str or Penalty object
        Penalty to use. Pass a string shorthand ("group_lasso", "sparse_group_lasso",
        "ridge") for convenience, or a penalty object directly for full control
        (e.g. ``GroupLasso(lambda1=0.1, flavor=Adaptive(expon=2))``).
    lambda1 : float or None
        Regularisation strength. Used when ``penalty`` is a string shorthand.
        Ignored when ``penalty`` is a Penalty object (lambda1 is on the object).
        If None, auto-calibrated to 10%% of lambda_max.
    lambda2 : float
        Smoothing penalty weight for spline SSP reparametrisation.
    spline_features : list of str or None
        Column names to treat as spline features. If None, no splines.
    n_knots : int or list of int
        Number of interior knots for spline features. If int, applied to all
        spline features. If list, must match length of ``spline_features``.
        15-20 is a safe default — knots are penalised via P-spline
        (Eilers & Marx), so more knots gives the penalty more to work with
        without overfitting.
    degree : int
        B-spline polynomial degree. Default 3 (cubic) gives C2 smooth curves.
        1 (linear) and 2 (quadratic) are fine. >3 is not advised.
    categorical_base : str
        Base level strategy for categoricals: "most_exposed" picks the level
        with highest total sample_weight (falls back to most frequent if no
        weights), "first" picks alphabetically first.
    standardize_numeric : bool
        Whether to center and scale numeric features before fitting.
    offset : str, list of str, or None
        Column name(s) in X to use as offset (fixed term in the linear
        predictor, not penalised). Multiple columns are summed. Offset
        columns are excluded from features.
    """

    def __init__(
        self,
        family: str = "poisson",
        tweedie_p: float | None = None,
        penalty: str | Penalty = "group_lasso",
        lambda1: float | None = None,
        lambda2: float = 0.1,
        spline_features: list[str] | None = None,
        n_knots: int | list[int] = 15,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        standardize_numeric: bool = True,
        offset: str | list[str] | None = None,
    ):
        self.family = family
        self.tweedie_p = tweedie_p
        self.penalty = penalty
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.spline_features = spline_features
        self.n_knots = n_knots
        self.degree = degree
        self.categorical_base = categorical_base
        self.standardize_numeric = standardize_numeric
        self.offset = offset

    def _resolve_penalty(self) -> Penalty:
        """Convert string shorthand to penalty object, or validate object."""
        if isinstance(self.penalty, str):
            if self.penalty not in _PENALTY_SHORTCUTS:
                raise ValueError(
                    f"Unknown penalty '{self.penalty}'. "
                    f"Use one of {list(_PENALTY_SHORTCUTS)} or pass a Penalty object."
                )
            return _PENALTY_SHORTCUTS[self.penalty](lambda1=self.lambda1)
        return self.penalty

    def fit(self, X: pd.DataFrame, y: NDArray, sample_weight: NDArray | None = None) -> SuperGLMRegressor:
        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=np.float64)

        # Extract offset columns
        offset_array = self._extract_offset(X)

        # Resolve per-feature n_knots
        spline_cols = self.spline_features or []
        knots_map = self._resolve_knots(spline_cols)

        # Resolve penalty
        penalty_obj = self._resolve_penalty()

        # Build core model
        self._model = SuperGLM(
            family=self.family,
            penalty=penalty_obj,
            lambda2=self.lambda2,
            tweedie_p=self.tweedie_p,
        )

        # Auto-detect and register features
        self._feature_types: dict[str, str] = {}
        feature_cols = [c for c in X.columns if c not in self._offset_cols]

        for col in feature_cols:
            if col in spline_cols:
                self._model.add_feature(col, Spline(
                    n_knots=knots_map[col], degree=self.degree, penalty="ssp",
                ))
                self._feature_types[col] = f"Spline(n_knots={knots_map[col]}, degree={self.degree})"
            elif X[col].dtype.kind in ("O", "U") or isinstance(X[col].dtype, pd.CategoricalDtype):
                base = self.categorical_base
                if base == "most_exposed" and sample_weight is None:
                    base = "first"
                self._model.add_feature(col, Categorical(base=base))
                self._feature_types[col] = f"Categorical(base={base})"
            else:
                self._model.add_feature(col, Numeric(standardize=self.standardize_numeric))
                self._feature_types[col] = f"Numeric(standardize={self.standardize_numeric})"

        self._print_feature_summary()

        self._model.fit(X[feature_cols], y, exposure=sample_weight, offset=offset_array)

        # sklearn attributes
        self.n_features_in_ = len(feature_cols)
        self.feature_names_in_ = np.array(feature_cols)
        self.intercept_ = self._model.result.intercept
        self.coef_ = self._model.result.beta
        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        check_is_fitted(self)
        X = pd.DataFrame(X)
        offset_array = self._extract_offset(X, fitting=False)
        feature_cols = [c for c in self.feature_names_in_]
        return self._model.predict(X[feature_cols], offset=offset_array)

    def summary(self) -> dict[str, Any]:
        check_is_fitted(self)
        return self._model.summary()

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        check_is_fitted(self)
        return self._model.reconstruct_feature(name)

    def _extract_offset(self, X: pd.DataFrame, fitting: bool = True) -> NDArray | None:
        if self.offset is None:
            if fitting:
                self._offset_cols: list[str] = []
            return None

        if isinstance(self.offset, str):
            cols = [self.offset]
        else:
            cols = list(self.offset)

        if fitting:
            self._offset_cols = cols

        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Offset column(s) not found in X: {missing}")

        offset_array = np.zeros(len(X), dtype=np.float64)
        for c in cols:
            offset_array += np.asarray(X[c], dtype=np.float64)
        return offset_array

    def _resolve_knots(self, spline_cols: list[str]) -> dict[str, int]:
        if not spline_cols:
            return {}
        if isinstance(self.n_knots, int):
            return {col: self.n_knots for col in spline_cols}
        if len(self.n_knots) != len(spline_cols):
            raise ValueError(
                f"n_knots has length {len(self.n_knots)} but spline_features "
                f"has length {len(spline_cols)}. Must match or pass a single int."
            )
        return dict(zip(spline_cols, self.n_knots))

    def _print_feature_summary(self) -> None:
        print("SuperGLM features:")
        for col, desc in self._feature_types.items():
            print(f"  {col:<20s} → {desc}")
