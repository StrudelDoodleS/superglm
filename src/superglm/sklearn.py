"""sklearn-compatible wrappers for SuperGLM.

Two wrappers are provided:

- ``SuperGLMRegressor`` — count / continuous regression (Poisson, Gamma, …).
  Rejects ``family="binomial"``; use the classifier for that.
- ``SuperGLMClassifier`` — binary classification (Binomial)

Both accept **DataFrame** or **ndarray** input:

- **DataFrame mode** (preferred): feature types are auto-detected from dtype.
  Object/category columns become ``Categorical``, columns in
  ``spline_features`` become ``BasisSpline``, everything else becomes
  ``Numeric``.
- **ndarray mode**: pass ``feature_names`` for readable column names
  (otherwise synthetic ``x0, x1, …`` names are generated).  Categoricals
  are *not* inferred from values — specify ``categorical_features``
  explicitly.  Unspecified columns default to ``Numeric``.

**Penalty default**: ``penalty=None`` means *no feature-selection penalty*.
Set ``selection_penalty`` to a positive value and the wrapper auto-upgrades
to ``"group_lasso"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from superglm.distributions import Distribution
from superglm.model import SuperGLM
from superglm.penalties.base import Penalty

# ── Family validation ─────────────────────────────────────────────

_REGRESSOR_FAMILIES = frozenset({"poisson", "gaussian", "gamma"})


def _validate_regressor_family(family) -> None:
    """Raise if *family* is not a valid regression family."""
    from superglm.distributions import (
        Binomial,
        Distribution,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        Tweedie,
    )

    if isinstance(family, Distribution):
        if isinstance(family, Binomial):
            raise ValueError(
                "Binomial family is not supported in SuperGLMRegressor; "
                "use SuperGLMClassifier instead."
            )
        if isinstance(family, Poisson | Gaussian | Gamma | NegativeBinomial | Tweedie):
            return
        raise ValueError(
            f"Unknown distribution type {type(family).__name__} for SuperGLMRegressor."
        )
    # String path
    if family == "binomial":
        raise ValueError(
            "family='binomial' is not supported in SuperGLMRegressor; "
            "use SuperGLMClassifier instead."
        )
    if family not in _REGRESSOR_FAMILIES:
        allowed = ", ".join(sorted(_REGRESSOR_FAMILIES))
        raise ValueError(
            f"Unknown family '{family}' for SuperGLMRegressor. "
            f"String families: {allowed}. "
            f"For parameterized families use families.tweedie(p=...) or families.nb2(theta=...)."
        )


# ── Input normalisation helpers ───────────────────────────────────


def _normalize_X(
    X,
    *,
    feature_names: list[str] | None,
    resolved_columns: list[str] | None,
    fitting: bool,
) -> tuple[pd.DataFrame, list[str], bool]:
    """Convert *X* to a DataFrame.

    Returns ``(dataframe, column_names, synthetic_names)``.
    *synthetic_names* is True when column names were auto-generated.
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy() if fitting else X
        return df, list(df.columns), False

    # Densify sparse matrices (e.g. from ColumnTransformer)
    import scipy.sparse

    if scipy.sparse.issparse(X):
        X = X.toarray()

    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if fitting:
        if feature_names is not None:
            if len(feature_names) != arr.shape[1]:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries "
                    f"but X has {arr.shape[1]} columns."
                )
            cols = list(feature_names)
            synthetic = False
        else:
            cols = [f"x{i}" for i in range(arr.shape[1])]
            synthetic = True
    else:
        cols = resolved_columns
        if arr.shape[1] != len(cols):
            raise ValueError(f"X has {arr.shape[1]} columns but model was fit with {len(cols)}.")
        synthetic = False

    return pd.DataFrame(arr, columns=cols), cols, synthetic


def _resolve_refs(
    refs: list[str | int] | None,
    columns: list[str],
    synthetic: bool,
    param_name: str,
) -> list[str] | None:
    """Resolve string / integer feature references to column names."""
    if refs is None:
        return None
    resolved: list[str] = []
    for ref in refs:
        if isinstance(ref, int | np.integer):
            if ref < 0 or ref >= len(columns):
                raise IndexError(
                    f"{param_name}: index {ref} out of range for {len(columns)} columns."
                )
            resolved.append(columns[int(ref)])
        elif isinstance(ref, str):
            if ref not in columns:
                if synthetic:
                    raise ValueError(
                        f"{param_name}: string ref '{ref}' used with "
                        f"ndarray input.  Pass feature_names or use "
                        f"integer indices."
                    )
                raise ValueError(f"{param_name}: '{ref}' not found in columns.")
            resolved.append(ref)
        else:
            raise TypeError(f"{param_name}: expected str or int, got {type(ref).__name__}")
    return resolved


def _resolve_offset(
    offset_param,
    X: pd.DataFrame,
    columns: list[str],
    synthetic: bool,
    *,
    fitting: bool,
    stored_cols: list[str] | None = None,
) -> tuple[NDArray | None, list[str]]:
    """Extract offset array and resolved column names."""
    if offset_param is None:
        return None, []

    if isinstance(offset_param, str | int | np.integer):
        raw_refs = [offset_param]
    else:
        raw_refs = list(offset_param)

    if fitting:
        resolved = _resolve_refs(raw_refs, columns, synthetic, "offset")
    else:
        resolved = stored_cols

    missing = [c for c in resolved if c not in X.columns]
    if missing:
        raise ValueError(f"Offset column(s) not found in X: {missing}")

    arr = np.zeros(len(X), dtype=np.float64)
    for c in resolved:
        arr += np.asarray(X[c], dtype=np.float64)
    return arr, resolved


_SHORTHAND_TUNING_DEFAULTS = {"n_knots": 10, "degree": 3, "categorical_base": "most_exposed"}


def _validate_wrapper_feature_config(
    features,
    *,
    spline_features,
    categorical_features,
    numeric_features,
    n_knots,
    degree,
    categorical_base,
) -> None:
    """Raise if ``features`` is mixed with shorthand wrapper arguments."""
    if features is None:
        return

    # Check shorthand feature lists
    set_lists = []
    if spline_features is not None:
        set_lists.append("spline_features")
    if categorical_features is not None:
        set_lists.append("categorical_features")
    if numeric_features is not None:
        set_lists.append("numeric_features")

    if set_lists:
        raise ValueError(
            f"Pass either features=... or the wrapper shorthand feature "
            f"arguments ({', '.join(set_lists)}), not both."
        )

    # Check shorthand tuning params that differ from defaults
    changed = []
    if n_knots != _SHORTHAND_TUNING_DEFAULTS["n_knots"]:
        changed.append(f"n_knots={n_knots!r}")
    if degree != _SHORTHAND_TUNING_DEFAULTS["degree"]:
        changed.append(f"degree={degree!r}")
    if categorical_base != _SHORTHAND_TUNING_DEFAULTS["categorical_base"]:
        changed.append(f"categorical_base={categorical_base!r}")

    if changed:
        raise ValueError(
            f"Pass either features=... or the wrapper shorthand feature "
            f"arguments ({', '.join(changed)}), not both. "
            f"With features=..., configure spline/categorical options "
            f"directly on the feature specs."
        )


def _resolve_native_feature_cols(
    features: dict,
    columns: list[str],
    offset_cols: list[str],
    synthetic: bool,
) -> list[str]:
    """Validate and return feature columns for the native ``features=`` path.

    Checks that every key in *features* exists in *columns* (after
    excluding offset columns) and returns the keys in their original
    dict order.
    """
    available = set(columns) - set(offset_cols)
    missing = [k for k in features if k not in available]
    if missing:
        if synthetic:
            raise ValueError(
                f"features= keys {missing} not found in columns. "
                f"X is an ndarray with auto-generated column names "
                f"({', '.join(columns[:3])}, …). Pass feature_names= "
                f"so column names match the features= keys, or use "
                f"the synthetic names (x0, x1, …) as keys."
            )
        raise ValueError(
            f"features= keys {missing} not found in X columns. "
            f"Available (non-offset) columns: "
            f"{sorted(available)}."
        )
    return list(features.keys())


def _resolve_wrapper_penalty(
    penalty,
    selection_penalty,
    spline_penalty,
):
    """Resolve penalty defaults with auto-upgrade.

    When ``penalty is None`` and a positive ``selection_penalty`` is given,
    the penalty is auto-upgraded to ``"group_lasso"`` for convenience.
    When neither is set, the wrapper defaults to an unpenalised fit
    (``selection_penalty=0``).
    """
    if penalty is None:
        if selection_penalty is not None and selection_penalty > 0:
            penalty = "group_lasso"
        elif selection_penalty is None:
            selection_penalty = 0.0

    return penalty, selection_penalty, spline_penalty


def _build_features_or_splines(
    feature_cols: list[str],
    spline_names: list[str] | None,
    categorical_names: list[str] | None,
    numeric_names: list[str] | None,
    n_knots: int | list[int],
    degree: int,
    cat_base: str,
    *,
    force_explicit: bool,
    X_df: pd.DataFrame | None = None,
) -> tuple[dict | None, list[str] | None]:
    """Build an explicit ``features`` dict or fall back to auto-detect.

    Returns ``(features_dict, splines_list)``.  Exactly one is non-None.
    When *force_explicit* is True (ndarray input or user provided explicit
    ``categorical_features`` / ``numeric_features``), all columns are
    mapped to feature specs.  Unspecified columns in DataFrame mode are
    auto-detected from dtype; in ndarray mode they default to Numeric.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.spline import BasisSpline

    spline_list = spline_names or []
    cat_list = categorical_names or []
    num_list = numeric_names or []

    if not force_explicit:
        # DataFrame auto-detect mode
        return None, spline_list or None

    # Explicit mode ─ build full features dict
    specified = set(spline_list) | set(cat_list) | set(num_list)
    unspecified = [c for c in feature_cols if c not in specified]

    # Resolve n_knots
    if isinstance(n_knots, int):
        nk_list = [n_knots] * len(spline_list)
    else:
        nk_list = list(n_knots)
        if len(nk_list) != len(spline_list):
            raise ValueError(
                f"n_knots has length {len(nk_list)} but "
                f"spline_features has length {len(spline_list)}."
            )

    features: dict = {}
    for i, name in enumerate(spline_list):
        features[name] = BasisSpline(n_knots=nk_list[i], degree=degree)
    for name in cat_list:
        features[name] = Categorical(base=cat_base)
    for name in num_list:
        features[name] = Numeric()

    # Unspecified columns: auto-detect from dtype when DataFrame is
    # available, otherwise default to Numeric (ndarray mode).
    for name in unspecified:
        if X_df is not None and X_df[name].dtype.kind in ("O", "S", "U"):
            features[name] = Categorical(base=cat_base)
        else:
            features[name] = Numeric()

    return features, None


# ── Shared fit / predict helpers ─────────────────────────────────


def _fit_common(
    wrapper,
    X,
    y: NDArray,
    sample_weight: NDArray | None,
    *,
    family: str | Distribution,
) -> None:
    """Shared fit logic for both wrapper classes.

    Validates feature config, normalizes inputs, resolves offset/penalty/
    features, instantiates and fits the core SuperGLM model, and sets
    common sklearn fitted attributes on *wrapper*.
    """
    # ── Validate feature config ──────────────────────────────
    _validate_wrapper_feature_config(
        wrapper.features,
        spline_features=wrapper.spline_features,
        categorical_features=wrapper.categorical_features,
        numeric_features=wrapper.numeric_features,
        n_knots=wrapper.n_knots,
        degree=wrapper.degree,
        categorical_base=wrapper.categorical_base,
    )

    # ── Normalise inputs ──────────────────────────────────────
    input_is_dataframe = isinstance(X, pd.DataFrame)
    X_df, columns, synthetic = _normalize_X(
        X,
        feature_names=wrapper.feature_names,
        resolved_columns=None,
        fitting=True,
    )
    wrapper._resolved_columns_ = columns
    wrapper._synthetic_names_ = synthetic

    y = np.asarray(y, dtype=np.float64)

    # Resolve offset
    offset_array, offset_cols = _resolve_offset(
        wrapper.offset,
        X_df,
        columns,
        synthetic,
        fitting=True,
    )
    wrapper._offset_cols_ = offset_cols

    # Resolve penalty
    penalty, resolved_sel, resolved_spl = _resolve_wrapper_penalty(
        wrapper.penalty,
        wrapper.selection_penalty,
        wrapper.spline_penalty,
    )

    if wrapper.features is not None:
        # ── Native features= path ────────────────────────────
        feature_cols = _resolve_native_feature_cols(
            wrapper.features,
            columns,
            offset_cols,
            synthetic,
        )

        model_kwargs: dict[str, Any] = dict(
            family=family,
            penalty=penalty,
            selection_penalty=resolved_sel,
            spline_penalty=resolved_spl,
            features=wrapper.features,
        )
    else:
        # ── Shorthand wrapper path ────────────────────────────
        feature_cols = [c for c in columns if c not in offset_cols]

        spline_names = _resolve_refs(
            wrapper.spline_features,
            columns,
            synthetic,
            "spline_features",
        )
        cat_names = _resolve_refs(
            wrapper.categorical_features,
            columns,
            synthetic,
            "categorical_features",
        )
        num_names = _resolve_refs(
            wrapper.numeric_features,
            columns,
            synthetic,
            "numeric_features",
        )

        # Categorical base fallback
        cat_base = wrapper.categorical_base
        if cat_base == "most_exposed" and sample_weight is None:
            cat_base = "first"

        # Build features dict or use auto-detect
        force_explicit = synthetic or (
            wrapper.categorical_features is not None or wrapper.numeric_features is not None
        )
        features_dict, splines_list = _build_features_or_splines(
            feature_cols,
            spline_names,
            cat_names,
            num_names,
            wrapper.n_knots,
            wrapper.degree,
            cat_base,
            force_explicit=force_explicit,
            X_df=X_df if input_is_dataframe else None,
        )

        model_kwargs = dict(
            family=family,
            penalty=penalty,
            selection_penalty=resolved_sel,
            spline_penalty=resolved_spl,
            n_knots=wrapper.n_knots,
            degree=wrapper.degree,
            categorical_base=cat_base,
        )
        if features_dict is not None:
            model_kwargs["features"] = features_dict
        else:
            model_kwargs["splines"] = splines_list or []

    wrapper._model = SuperGLM(**model_kwargs)
    wrapper._model.fit(
        X_df[feature_cols],
        y,
        sample_weight=sample_weight,
        offset=offset_array,
    )

    # sklearn attributes
    wrapper.n_features_in_ = len(feature_cols)
    wrapper.feature_names_in_ = np.array(feature_cols)
    wrapper.intercept_ = wrapper._model.result.intercept
    wrapper.coef_ = wrapper._model.result.beta


def _prepare_predict(
    wrapper,
    X,
) -> tuple[pd.DataFrame, list[str], NDArray | None]:
    """Normalize X and resolve offset for prediction.

    Returns ``(X_df, feature_cols, offset_array)``.
    """
    check_is_fitted(wrapper)
    X_df, _, _ = _normalize_X(
        X,
        feature_names=wrapper.feature_names,
        resolved_columns=wrapper._resolved_columns_,
        fitting=False,
    )
    offset_array, _ = _resolve_offset(
        wrapper.offset,
        X_df,
        wrapper._resolved_columns_,
        False,
        fitting=False,
        stored_cols=wrapper._offset_cols_,
    )
    feature_cols = list(wrapper.feature_names_in_)
    return X_df, feature_cols, offset_array


# ── Regressor ─────────────────────────────────────────────────────


class SuperGLMRegressor(BaseEstimator, RegressorMixin):
    """Penalised GLM for count / continuous regression.

    Accepts **DataFrame** or **ndarray** input.  See module docstring for
    details on ndarray mode and penalty defaults.

    Parameters
    ----------
    family : str or Distribution
        Distribution family.  Strings: ``"poisson"``, ``"gamma"``,
        ``"gaussian"``.  For parameterized families use objects:
        ``families.tweedie(p=1.5)``, ``families.nb2(theta=1.0)``.
        For binary classification use ``SuperGLMClassifier``.
    penalty : str, Penalty, or None
        Penalty type.  ``None`` (default) means no feature-selection
        penalty.  Set ``selection_penalty`` to a positive value to
        auto-upgrade to ``"group_lasso"``.
    selection_penalty : float or None
        Group penalty strength (feature selection).
    spline_penalty : float or None
        Within-group spline smoothing.  Defaults to 0.1.
    features : dict[str, FeatureSpec] or None
        Native-style feature specs.  Mutually exclusive with the
        shorthand wrapper arguments (``spline_features``,
        ``categorical_features``, ``numeric_features``, non-default
        ``n_knots``/``degree``/``categorical_base``).
    spline_features : list of str or int, or None
        Columns to treat as spline features (by name or index).
    categorical_features : list of str or int, or None
        Columns to treat as categorical (by name or index).
        Required for ndarray input if any columns are categorical.
    numeric_features : list of str or int, or None
        Columns to treat as numeric (by name or index).
        Unspecified columns default to numeric.
    feature_names : list of str or None
        Column names for ndarray input.  Ignored for DataFrames.
    n_knots : int or list of int
        Interior knots for spline features.
    degree : int
        B-spline degree (default 3).
    categorical_base : str
        Base level strategy (``"most_exposed"`` or ``"first"``).
    offset : str, int, list, or None
        Offset column(s) by name or index.
    """

    def __init__(
        self,
        family: str | Distribution = "poisson",
        penalty: str | Penalty | None = None,
        selection_penalty: float | None = None,
        spline_penalty: float | None = None,
        features: dict | None = None,
        spline_features: list[str | int] | None = None,
        categorical_features: list[str | int] | None = None,
        numeric_features: list[str | int] | None = None,
        feature_names: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        offset: str | int | list[str | int] | None = None,
    ):
        self.family = family
        self.penalty = penalty
        self.selection_penalty = selection_penalty
        self.spline_penalty = spline_penalty
        self.features = features
        self.spline_features = spline_features
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.feature_names = feature_names
        self.n_knots = n_knots
        self.degree = degree
        self.categorical_base = categorical_base
        self.offset = offset

    def fit(
        self,
        X,
        y: NDArray,
        sample_weight: NDArray | None = None,
    ) -> SuperGLMRegressor:
        _validate_regressor_family(self.family)
        _fit_common(self, X, y, sample_weight, family=self.family)

        # Expose feature types for backward compat
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
                self._feature_types[name] = "Numeric()"

        return self

    def predict(self, X) -> NDArray:
        X_df, feature_cols, offset_array = _prepare_predict(self, X)
        return self._model.predict(X_df[feature_cols], offset=offset_array)

    def diagnostics(self) -> dict[str, Any]:
        check_is_fitted(self)
        return self._model.diagnostics()

    def summary(self, alpha: float = 0.05):
        check_is_fitted(self)
        return self._model.summary(alpha=alpha)

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        check_is_fitted(self)
        return self._model.reconstruct_feature(name)


# ── Classifier ────────────────────────────────────────────────────


class SuperGLMClassifier(BaseEstimator, ClassifierMixin):
    """Penalised binomial GLM for binary classification.

    Uses ``SuperGLM(family="binomial")`` under the hood.  Implements the
    sklearn classifier contract: ``classes_``, ``predict``,
    ``predict_proba``, ``decision_function``.

    Accepts **DataFrame** or **ndarray** input.  See module docstring for
    details on ndarray mode and penalty defaults.

    Parameters
    ----------
    penalty : str, Penalty, or None
        Penalty type.  ``None`` (default) means no feature-selection
        penalty.  Setting ``selection_penalty`` auto-upgrades to
        ``"group_lasso"``.
    selection_penalty, spline_penalty : float or None
        See ``SuperGLMRegressor``.
    features : dict[str, FeatureSpec] or None
        Native-style feature specs.  Mutually exclusive with shorthand
        wrapper arguments.  See ``SuperGLMRegressor`` for details.
    spline_features : list of str or int, or None
        Columns to treat as spline features.
    categorical_features : list of str or int, or None
        Columns to treat as categorical.
    numeric_features : list of str or int, or None
        Columns to treat as numeric.
    feature_names : list of str or None
        Column names for ndarray input.
    n_knots : int or list of int
        Interior knots for spline features.
    degree : int
        B-spline degree.
    categorical_base : str
        Base level strategy.
    offset : str, int, list, or None
        Offset column(s).
    threshold : float
        Classification threshold for ``predict()`` (default 0.5).
    """

    def __init__(
        self,
        penalty: str | Penalty | None = None,
        selection_penalty: float | None = None,
        spline_penalty: float | None = None,
        features: dict | None = None,
        spline_features: list[str | int] | None = None,
        categorical_features: list[str | int] | None = None,
        numeric_features: list[str | int] | None = None,
        feature_names: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        offset: str | int | list[str | int] | None = None,
        threshold: float = 0.5,
    ):
        self.penalty = penalty
        self.selection_penalty = selection_penalty
        self.spline_penalty = spline_penalty
        self.features = features
        self.spline_features = spline_features
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.feature_names = feature_names
        self.n_knots = n_knots
        self.degree = degree
        self.categorical_base = categorical_base
        self.offset = offset
        self.threshold = threshold

    def fit(
        self,
        X,
        y: NDArray,
        sample_weight: NDArray | None = None,
    ) -> SuperGLMClassifier:
        from superglm.distributions import Binomial, validate_response

        y = np.asarray(y, dtype=np.float64)
        validate_response(y, Binomial())

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"SuperGLMClassifier requires both classes in y, but got only {self.classes_}."
            )

        _fit_common(self, X, y, sample_weight, family="binomial")
        return self

    def predict_proba(self, X) -> NDArray:
        """Return class probabilities, shape ``(n_samples, 2)``."""
        X_df, feature_cols, offset_array = _prepare_predict(self, X)
        p1 = self._model.predict(X_df[feature_cols], offset=offset_array)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> NDArray:
        """Return class labels (0 or 1) using the threshold."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def decision_function(self, X) -> NDArray:
        """Return log-odds (linear predictor)."""
        X_df, feature_cols, offset_array = _prepare_predict(self, X)
        X_feat = X_df[feature_cols]

        blocks = []
        for name in self._model._feature_order:
            spec = self._model._specs[name]
            blocks.append(spec.transform(np.asarray(X_feat[name])))
        for iname in self._model._interaction_order:
            ispec = self._model._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            blocks.append(
                ispec.transform(
                    np.asarray(X_feat[p1]),
                    np.asarray(X_feat[p2]),
                )
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
