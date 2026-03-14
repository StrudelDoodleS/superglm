"""sklearn-compatible wrapper for SuperGLM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from superglm.model import SuperGLM
from superglm.penalties.base import Penalty


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
    degree : int
        B-spline polynomial degree. Default 3 (cubic).
    categorical_base : str
        Base level strategy for categoricals: "most_exposed" or "first".
    standardize_numeric : bool
        Whether to center and scale numeric features before fitting.
    offset : str, list of str, or None
        Column name(s) in X to use as offset. Multiple columns are summed.
        Offset columns are excluded from features.
    """

    def __init__(
        self,
        family: str = "poisson",
        tweedie_p: float | None = None,
        nb_theta: float | str | None = None,
        penalty: str | Penalty = "group_lasso",
        lambda1: float | None = None,
        lambda2: float = 0.1,
        spline_features: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        standardize_numeric: bool = True,
        offset: str | list[str] | None = None,
    ):
        self.family = family
        self.tweedie_p = tweedie_p
        self.nb_theta = nb_theta
        self.penalty = penalty
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.spline_features = spline_features
        self.n_knots = n_knots
        self.degree = degree
        self.categorical_base = categorical_base
        self.standardize_numeric = standardize_numeric
        self.offset = offset

    def fit(
        self, X: pd.DataFrame, y: NDArray, sample_weight: NDArray | None = None
    ) -> SuperGLMRegressor:
        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=np.float64)

        # Extract offset columns before passing to SuperGLM
        offset_array = self._extract_offset(X)
        feature_cols = [c for c in X.columns if c not in self._offset_cols]

        # Adjust categorical base when no weights
        cat_base = self.categorical_base
        if cat_base == "most_exposed" and sample_weight is None:
            cat_base = "first"

        # Build core model — delegates all feature config
        self._model = SuperGLM(
            family=self.family,
            penalty=self.penalty,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            tweedie_p=self.tweedie_p,
            nb_theta=self.nb_theta,
            splines=self.spline_features or [],
            n_knots=self.n_knots,
            degree=self.degree,
            categorical_base=cat_base,
            standardize_numeric=self.standardize_numeric,
        )

        self._model.fit(X[feature_cols], y, exposure=sample_weight, offset=offset_array)

        # sklearn attributes
        self.n_features_in_ = len(feature_cols)
        self.feature_names_in_ = np.array(feature_cols)
        self.intercept_ = self._model.result.intercept
        self.coef_ = self._model.result.beta

        # Expose feature types for backward compat with tests
        from superglm.features.spline import _SplineBase

        self._feature_types = {}
        for name in self._model._feature_order:
            spec = self._model._specs[name]
            if isinstance(spec, _SplineBase):
                cls_name = type(spec).__name__
                self._feature_types[name] = (
                    f"{cls_name}(n_knots={spec.n_knots}, degree={spec.degree})"
                )
            elif type(spec).__name__ == "Categorical":
                self._feature_types[name] = f"Categorical(base={spec.base})"
            else:
                self._feature_types[name] = f"Numeric(standardize={spec.standardize})"

        return self

    def predict(self, X: pd.DataFrame) -> NDArray:
        check_is_fitted(self)
        X = pd.DataFrame(X)
        offset_array = self._extract_offset(X, fitting=False)
        feature_cols = [c for c in self.feature_names_in_]
        return self._model.predict(X[feature_cols], offset=offset_array)

    def diagnostics(self) -> dict[str, Any]:
        check_is_fitted(self)
        return self._model.diagnostics()

    def summary(self, alpha: float = 0.05):
        check_is_fitted(self)
        return self._model.summary(alpha=alpha)

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


class SuperGLMClassifier(BaseEstimator, ClassifierMixin):
    """Penalised binomial GLM for binary classification.

    Uses ``SuperGLM(family="binomial")`` under the hood.  Implements the
    sklearn classifier contract: ``classes_``, ``predict``, ``predict_proba``,
    ``decision_function``.

    ``sample_weight`` is a case/frequency weight, not binomial n_trials.

    Parameters
    ----------
    penalty : str or Penalty object
        Penalty to use (default ``"group_lasso"``).
    lambda1 : float or None
        Regularisation strength.  ``None`` auto-calibrates.
    lambda2 : float
        Smoothing penalty weight for splines.
    spline_features : list of str or None
        Column names to treat as spline features.
    n_knots : int or list of int
        Number of interior knots for spline features.
    degree : int
        B-spline degree.
    categorical_base : str
        Base level strategy for categoricals.
    standardize_numeric : bool
        Whether to center and scale numeric features.
    offset : str, list of str, or None
        Column name(s) in X to use as offset.
    threshold : float
        Classification threshold for ``predict()`` (default 0.5).
    """

    def __init__(
        self,
        penalty: str | Penalty = "group_lasso",
        lambda1: float | None = None,
        lambda2: float = 0.1,
        spline_features: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        standardize_numeric: bool = True,
        offset: str | list[str] | None = None,
        threshold: float = 0.5,
    ):
        self.penalty = penalty
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.spline_features = spline_features
        self.n_knots = n_knots
        self.degree = degree
        self.categorical_base = categorical_base
        self.standardize_numeric = standardize_numeric
        self.offset = offset
        self.threshold = threshold

    def fit(
        self, X: pd.DataFrame, y: NDArray, sample_weight: NDArray | None = None
    ) -> SuperGLMClassifier:
        from superglm.distributions import Binomial, validate_response

        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=np.float64)

        validate_response(y, Binomial())

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"SuperGLMClassifier requires both classes in y, but got only {self.classes_}."
            )

        # Extract offset columns before passing to SuperGLM
        offset_array = self._extract_offset(X)
        feature_cols = [c for c in X.columns if c not in self._offset_cols]

        cat_base = self.categorical_base
        if cat_base == "most_exposed" and sample_weight is None:
            cat_base = "first"

        self._model = SuperGLM(
            family="binomial",
            penalty=self.penalty,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            splines=self.spline_features or [],
            n_knots=self.n_knots,
            degree=self.degree,
            categorical_base=cat_base,
            standardize_numeric=self.standardize_numeric,
        )

        self._model.fit(X[feature_cols], y, exposure=sample_weight, offset=offset_array)

        self.n_features_in_ = len(feature_cols)
        self.feature_names_in_ = np.array(feature_cols)
        self.intercept_ = self._model.result.intercept
        self.coef_ = self._model.result.beta

        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray:
        """Return class probabilities, shape (n_samples, 2)."""
        check_is_fitted(self)
        X = pd.DataFrame(X)
        offset_array = self._extract_offset(X, fitting=False)
        feature_cols = list(self.feature_names_in_)
        p1 = self._model.predict(X[feature_cols], offset=offset_array)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: pd.DataFrame) -> NDArray:
        """Return class labels (0 or 1) using the threshold."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def decision_function(self, X: pd.DataFrame) -> NDArray:
        """Return log-odds (linear predictor)."""
        check_is_fitted(self)
        X = pd.DataFrame(X)
        offset_array = self._extract_offset(X, fitting=False)
        feature_cols = list(self.feature_names_in_)
        # Build eta directly
        blocks = []
        for name in self._model._feature_order:
            spec = self._model._specs[name]
            blocks.append(spec.transform(np.asarray(X[feature_cols][name])))
        for iname in self._model._interaction_order:
            ispec = self._model._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            blocks.append(
                ispec.transform(np.asarray(X[feature_cols][p1]), np.asarray(X[feature_cols][p2]))
            )
        eta = np.hstack(blocks) @ self._model.result.beta + self._model.result.intercept
        if offset_array is not None:
            eta = eta + offset_array
        return eta

    def diagnostics(self) -> dict[str, Any]:
        check_is_fitted(self)
        return self._model.diagnostics()

    def summary(self, alpha: float = 0.05):
        check_is_fitted(self)
        return self._model.summary(alpha=alpha)

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
