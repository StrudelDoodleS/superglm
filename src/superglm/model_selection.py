"""Cross-validation with pluggable splitters and scorers."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import FrameLike, as_eager_frame
from superglm.distributions import Tweedie, weighted_log_likelihood

# The full-frame binding pass lives in a neutral module so the public
# SuperGLM.bind_levels runs this exact resolution without importing this one.
from superglm.model.binding_ops import resolve_level_bindings as _resolve_level_bindings
from superglm.solvers.dispersion import dispersion_likelihood_size, model_weight_semantics
from superglm.validation import _normalized_gini

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResult:
    """Structured result from :func:`superglm.cross_validate`.

    Attributes
    ----------
    fold_scores : DataFrame
        One row per fold with columns: ``fold``, ``n_train``, ``n_test``,
        ``fit_time_s``, ``score_time_s``, ``converged``, ``n_iter``,
        ``effective_df``, plus one column per requested metric.
    mean_scores : dict
        Equal-weight mean of each per-fold metric across folds. Built-in
        deviance and negative log-likelihood are normalized within each fold
        by the likelihood size the model's declared ``weight_semantics``
        implies: ``sum(sample_weight)`` under ``"frequency"``, the count of
        positive-weight validation rows under ``"prior"``.
    pooled_scores : dict
        Supported overall pooled metrics, computed as ratio-of-sums rather than
        mean-of-fold-ratios, with the same denominator.
    std_scores : dict
        Standard deviation of each metric across folds.
    fold_indices : list[tuple[ndarray, ndarray]] or None
        Per-fold ``(train_idx, test_idx)`` pairs from the CV splitter.
    curve_similarity : dict or None
        Fold-by-fold term similarity diagnostics for comparable main effects.
        A correlation is NaN where two curves share fewer than two finite
        points or either is flat there, a term a fold's penalty zeroes say.
    oof_predictions : ndarray or None
        Out-of-fold predictions (response scale), same length as *y*.
        ``None`` unless ``return_oof=True``.
    estimators : list or None
        Fitted model per fold. ``None`` unless ``return_estimators=True``.
    n_rows : int or None
        Number of rows the folds index: the length of ``y``.
    data_fingerprint : str or None
        SHA-256 of the rows the folds index: the row count, the frame library
        and the columns of ``X`` the model reads (``fingerprint_columns``),
        then the response, sample weights and offsets as little-endian
        float64, with unit weights standing in for
        ``sample_weight=None`` and zeros for ``offset=None``. Equal
        fingerprints mean the same rows in the same order, which is what lets
        a later consumer, such as the editor's Run CV, replay
        ``fold_indices`` on data it holds. The recipe reads values, never a
        library's row hash or dtype text, so a pandas or polars upgrade does
        not change it.
    splitter : str or None
        Class name of the splitter that drew the folds.
    fingerprint_columns : tuple of str or None
        The columns of ``X`` the fingerprint covers: the model's declared
        features and interaction parents, or every column when the model
        takes its features from ``X``. Other columns may hold anything.
        ``data_fingerprint`` is ``None`` when these columns cannot be hashed.
    fit_mode : {"fit", "fit_reml"} or None
        The fit method each fold was fitted with.
    fingerprint_version : int or None
        The version of the recipe ``data_fingerprint`` was made with. A
        fingerprint of another version cannot be compared with these rows.
    builtin_scores : tuple of str or None
        The score columns the built-in scorers computed, in ``scoring``
        order. A column a callable wrote is not one of them, even one named
        ``"deviance"``, ``"nll"`` or ``"gini"``.

    ``n_rows``, ``data_fingerprint``, ``splitter``, ``fingerprint_columns``,
    ``fit_mode``, ``fingerprint_version`` and ``builtin_scores`` are ``None``
    on a result made before they were recorded.
    """

    fold_scores: pd.DataFrame
    mean_scores: dict[str, float]
    pooled_scores: dict[str, float]
    std_scores: dict[str, float]
    fold_indices: list[tuple[NDArray, NDArray]] | None = None
    curve_similarity: dict[str, Any] | None = None
    oof_predictions: NDArray | None = None
    estimators: list | None = None
    n_rows: int | None = None
    data_fingerprint: str | None = None
    splitter: str | None = None
    fingerprint_columns: tuple[str, ...] | None = None
    fit_mode: str | None = None
    fingerprint_version: int | None = None
    builtin_scores: tuple[str, ...] | None = None

    def plot_terms_by_fold(
        self,
        X: FrameLike,
        *,
        y: NDArray | None = None,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        terms: str | list[str] | None = None,
        engine: str = "plotly",
        **kwargs,
    ):
        """Plot fold-specific main effects using the shared comparison engine.

        The stored folds are replayed on ``X`` and ``sample_weight``, so they
        must be the data given to :func:`cross_validate`, in the same order.
        Their row count is always checked. Pass ``y`` (and ``offset``, if the
        cross-validation had one) to check the rows themselves against
        ``data_fingerprint``; without ``y`` a reordered ``X`` of the same
        length cannot be told apart.

        Raises
        ------
        ValueError
            If ``X`` or ``sample_weight`` has another row count than the folds
            index, or, when ``y`` is passed, the data differ from the data the
            folds were drawn on.
        """
        if self.estimators is None:
            raise RuntimeError("return_estimators=True is required for plot_terms_by_fold().")

        from superglm.plotting.comparison import plot_term_comparison

        models = {
            f"fold_{fold}": est for fold, est in enumerate(self.estimators) if est is not None
        }
        if not models:
            raise RuntimeError("No fitted fold estimators are available to plot.")
        frame = as_eager_frame(X)
        weight_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        expected = _fold_row_count(self.n_rows, self.fold_indices or [])
        for name, rows in (("X", frame), ("sample_weight", weight_arr)):
            if expected is not None and rows is not None and len(rows) != expected:
                raise ValueError(_FOLD_ROWS.format(name=name, rows=len(rows), expected=expected))
        if y is not None and self.data_fingerprint is not None:
            if self.fingerprint_version != FINGERPRINT_VERSION:
                raise ValueError(_FOLD_VERSION)
            try:
                held = _data_fingerprint(frame, y, weight_arr, offset, self.fingerprint_columns)
            except (KeyError, TypeError, ValueError):
                held = None
            if held != self.data_fingerprint:
                raise ValueError(_FOLD_DATA.format(rows=expected))
        support_by_label: dict[str, dict[str, Any]] = {}
        for fold, indices in enumerate(self.fold_indices or []):
            label = f"fold_{fold}"
            if label not in models:
                continue
            train_idx, _test_idx = indices
            support_by_label[label] = {
                "X": frame.take_rows(train_idx),
                "sample_weight": None if weight_arr is None else weight_arr[train_idx],
            }

        return plot_term_comparison(
            models=models,
            terms=terms,
            X=X,
            sample_weight=sample_weight,
            support_by_label=support_by_label,
            engine=engine,
            **kwargs,
        )


_FOLD_ROWS = (
    "{name} has {rows:,} rows, but the folds were drawn on {expected:,}; pass the data given "
    "to cross_validate."
)
_FOLD_VERSION = (
    "This result's data fingerprint predates this version of superglm or comes from another "
    "one, so these rows cannot be checked; run cross_validate again with this version, or pass "
    "no y."
)
_FOLD_DATA = (
    "These {rows:,} rows are not the ones the folds were drawn on: the columns, dtypes, row "
    "order or values of X, y, sample_weight or offset differ (a pandas frame and a polars one "
    "differ too); pass the data given to cross_validate."
)


def _fold_row_count(n_rows: int | None, folds: Sequence[tuple[NDArray, NDArray]]) -> int | None:
    """The recorded row count, or one past the largest index an older result holds."""
    if n_rows is not None or not folds:
        return n_rows
    return 1 + max(int(np.max(np.concatenate(fold))) for fold in folds)


def _fingerprint_columns(model, frame) -> tuple[str, ...]:
    """The columns a model reads: its declared features and interaction parents.

    A model that takes its features from ``X`` reads every column. Fold
    replay needs only the rows the model reads to be the same, so a column it
    never reads stays out of the fingerprint and may hold anything.
    """
    if not getattr(model, "_features_explicit", False):
        return tuple(sorted(frame.columns, key=repr))
    names = [*model._specs]
    for spec in getattr(model, "_interaction_specs", {}).values():
        names.extend(spec.parent_names)
    return tuple(sorted(dict.fromkeys(names), key=repr))


# The recipe of _data_fingerprint. A result records it beside its fingerprint,
# and one made by another recipe is refused as such, never as other data.
# Version 3 keeps apart equal values a grouped categorical reads as two
# levels, -0.0 and 0.0, or 1 and 1.0 in an object column; version 2 did not.
FINGERPRINT_VERSION = 3


def _data_fingerprint(X, y, sample_weight=None, offset=None, columns=None) -> str:
    """SHA-256 of the recipe, the row count, the frame's columns, then y, weights and offsets.

    The response and weights alone do not identify the rows: two rows with
    the same response and weight can swap their features, and stored fold
    indices would then fall on different rows. The frame enters as its
    library ("pandas" or "polars", whose frames of the same values differ)
    and then ``columns`` (every column when ``None``) in name order, so that
    reordering columns alone does not change it, each as its name and
    :func:`_column_bytes`. Unit weights stand in for ``sample_weight=None``
    and zeros for ``offset=None``, which is how every scorer reads them, so
    rows supplied with those explicit values match. Every variable-length
    part carries its length, so the boundaries between the parts are fixed.
    """
    frame = as_eager_frame(X)
    response = np.asarray(y, dtype=np.float64).ravel()
    n = response.size
    weights = np.ones(n) if sample_weight is None else sample_weight
    offsets = np.zeros(n) if offset is None else offset
    names = sorted(frame.columns if columns is None else columns, key=repr)
    frame.require_columns(tuple(names))
    digest = hashlib.sha256(_framed(b"superglm.cv-fingerprint"))
    for count in (FINGERPRINT_VERSION, n, len(names)):
        digest.update(count.to_bytes(8, "little"))
    digest.update(_framed(frame.backend.encode("utf-8")))
    for name in names:
        digest.update(_framed(repr(name).encode("utf-8")))
        digest.update(_column_bytes(frame, name))
    for column in (response, weights, offsets):
        digest.update(np.ascontiguousarray(column, dtype="<f8").ravel().tobytes())
    return digest.hexdigest()


def _column_bytes(frame, name) -> bytes:
    """One column as bytes no pandas or polars version changes: a type tag, then its values.

    The tag is this recipe's own, from the NumPy array the model reads: a
    number's kind and width (``int32``, ``float64``), ``bool``, or ``values``
    for anything else, text included, whatever its dtype is called. A
    pandas categorical or polars Enum adds its declared categories, which
    the fit takes as the level universe. Numbers are written at a fixed
    width, little-endian: integers as 64-bit, floats as float64 (exact) with
    one NaN; anything else as codes in order of first appearance (missing
    values -1) and the text of each code's first value (:func:`_value_codes`).
    A grouped categorical reads a level by its text, so equal values that
    print differently stay apart: ``-0.0`` and ``0.0`` in a float column, and
    in an object column also ``1`` and ``1.0``.
    """
    values = frame.column_array(name)
    kind = values.dtype.kind
    if kind in "iu":
        tag = f"{'int' if kind == 'i' else 'uint'}{8 * values.dtype.itemsize}"
        data = np.ascontiguousarray(values, dtype="<i8" if kind == "i" else "<u8").tobytes()
    elif kind == "f":
        tag = f"float{8 * values.dtype.itemsize}"
        floats = np.array(values, dtype="<f8")
        floats[np.isnan(floats)] = np.nan
        data = floats.tobytes()
    elif kind == "b":
        tag, data = "bool", np.ascontiguousarray(values, dtype="<u1").tobytes()
    else:
        codes, uniques = _value_codes(values)
        tag = "values"
        data = np.ascontiguousarray(codes, dtype="<i8").tobytes() + _texts(uniques)
    categories = frame.column_declared_categories(name)
    declared = b"" if categories is None else _texts(categories)
    return _framed(tag.encode("utf-8")) + _framed(declared) + _framed(data)


def _value_codes(values) -> tuple[NDArray[np.intp], NDArray]:
    """``pandas.factorize`` codes, split where equal values differ in type or text.

    Equal text is the same text, so a column of ``str`` is coded as
    ``pandas.factorize`` codes it. Other equal values can print differently
    (``-0.0`` and ``0.0``, ``1``, ``1.0`` and ``True``), and each type and
    text then takes its own code. Codes follow first appearance, missing
    values are -1, and the second result holds each code's first value.
    """
    codes, uniques = pd.factorize(values, sort=False, use_na_sentinel=True)
    if all(type(value) is str for value in uniques):
        return codes, uniques
    present = codes >= 0
    kept = values[present]
    keys = [f"{code}:{_value_text(value)}" for code, value in zip(codes[present], kept)]
    split, _ = pd.factorize(np.asarray(keys, dtype=object), sort=False)
    codes[present] = split
    return codes, kept[np.unique(split, return_index=True)[1]]


def _value_text(value) -> str:
    """A value's type name and text."""
    return f"{type(value).__name__}:{value}"


def _texts(values) -> bytes:
    """The count, then each value's type name and text, UTF-8 and length-prefixed."""
    items = [_value_text(value).encode("utf-8", "surrogatepass") for value in values]
    return len(items).to_bytes(8, "little") + b"".join(_framed(item) for item in items)


def _framed(data: bytes) -> bytes:
    """``data`` after its length, so where it ends is fixed."""
    return len(data).to_bytes(8, "little") + data


# ── Model cloning ────────────────────────────────────────────────


def _clone_model(model):
    """Create a fresh (unfitted) copy of *model* preserving constructor config."""
    return model.clone_unfitted()


# ── Built-in scorers ─────────────────────────────────────────────


def _scoring_weights(model, sample_weight, n_rows: int) -> tuple[NDArray, float]:
    """Return scoring weights and the family's validation likelihood size."""
    weights = (
        np.ones(n_rows, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    denominator = dispersion_likelihood_size(
        weights,
        weight_semantics=model_weight_semantics(model),
    )
    if denominator <= 0.0:
        raise ValueError("validation sample_weight must have positive likelihood size")
    return weights, denominator


def _score_deviance(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Mean unit deviance under the family's sample-weight contract."""
    mu = model.predict(X_val, offset=offset)
    dev = model._distribution.deviance_unit(y_val, mu)
    weights, denominator = _scoring_weights(model, sample_weight, len(y_val))
    return float(np.sum(weights * dev) / denominator)


def _score_nll(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Mean negative log-likelihood under the family's weight contract."""
    mu = model.predict(X_val, offset=offset)
    weights, denominator = _scoring_weights(model, sample_weight, len(y_val))
    # A predefined or custom split can put an off-lattice row only in
    # validation, so no fold's fit ever warns while both the per-fold and the
    # pooled NLL quietly use an interpolated pseudo-density.
    from superglm.model.input_validation import check_weight_contract

    check_weight_contract(
        np.asarray(y_val, dtype=np.float64),
        weights,
        model._distribution,
        model_weight_semantics(model),
    )
    ll = weighted_log_likelihood(
        model._distribution,
        y_val,
        mu,
        weights,
        model.result.phi,
        weight_semantics=model_weight_semantics(model),
    )
    return float(-ll / denominator)


def _score_gini(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Tie-collapsed normalized Gini for binary/frequency models."""
    mu = model.predict(X_val, offset=offset)
    return _normalized_gini(y_val, mu, sample_weight)


def _pooled_deviance_parts(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Return numerator and denominator for pooled deviance aggregation."""
    mu = model.predict(X_val, offset=offset)
    dev = model._distribution.deviance_unit(y_val, mu)
    weights, denominator = _scoring_weights(model, sample_weight, len(y_val))
    return float(np.sum(weights * dev)), denominator


def _pooled_nll_parts(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Return numerator and denominator for pooled negative log-likelihood."""
    mu = model.predict(X_val, offset=offset)
    weights, denominator = _scoring_weights(model, sample_weight, len(y_val))
    # A predefined or custom split can put an off-lattice row only in
    # validation, so no fold's fit ever warns while both the per-fold and the
    # pooled NLL quietly use an interpolated pseudo-density.
    from superglm.model.input_validation import check_weight_contract

    check_weight_contract(
        np.asarray(y_val, dtype=np.float64),
        weights,
        model._distribution,
        model_weight_semantics(model),
    )
    ll = weighted_log_likelihood(
        model._distribution,
        y_val,
        mu,
        weights,
        model.result.phi,
        weight_semantics=model_weight_semantics(model),
    )
    return float(-ll), denominator


_RESERVED_COLUMNS = frozenset(
    {
        "fold",
        "n_train",
        "n_test",
        "fit_time_s",
        "score_time_s",
        "converged",
        "n_iter",
        "effective_df",
    }
)

_BUILTIN_SCORERS: dict[str, Callable] = {
    "deviance": _score_deviance,
    "nll": _score_nll,
    "gini": _score_gini,
}

_POOLED_PARTS: dict[str, Callable] = {
    "deviance": _pooled_deviance_parts,
    "nll": _pooled_nll_parts,
}


def _resolve_scorers(
    scoring: str | Callable | Sequence[str | Callable],
) -> dict[str, Callable]:
    """Normalize *scoring* into a {name: callable} dict."""
    if isinstance(scoring, str):
        scoring = (scoring,)
    elif callable(scoring) and not isinstance(scoring, list | tuple):
        scoring = (scoring,)

    resolved: dict[str, Callable] = {}
    unnamed_count = 0
    for s in scoring:
        if isinstance(s, str):
            if s not in _BUILTIN_SCORERS:
                raise ValueError(
                    f"Unknown scorer {s!r}. "
                    f"Built-in scorers: {list(_BUILTIN_SCORERS)}. "
                    f"Or pass a callable."
                )
            resolved[s] = _BUILTIN_SCORERS[s]
        elif callable(s):
            name = getattr(s, "__name__", None) or f"scorer_{unnamed_count}"
            if name in resolved:
                unnamed_count += 1
                name = f"{name}_{unnamed_count}"
            resolved[name] = s
            unnamed_count += 1
        else:
            raise TypeError(f"Scorer must be a string or callable, got {type(s)}")

    if not resolved:
        raise ValueError("scoring must contain at least one scorer")
    return resolved


# ── Main function ─────────────────────────────────────────────────


def cross_validate(
    model,
    X: FrameLike,
    y: NDArray,
    *,
    cv,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    groups: NDArray | None = None,
    fit_mode: str = "fit",
    scoring: str | Callable | Sequence[str | Callable] = ("deviance",),
    return_estimators: bool = False,
    return_oof: bool = False,
    error_score: float | str = np.nan,
) -> CrossValidationResult:
    """Cross-validate a SuperGLM model with a pluggable splitter.

    Parameters
    ----------
    model : SuperGLM
        An unfitted (or fitted) model. A fresh clone is created for each fold;
        the input model is never mutated.
    X : pandas or eager Polars DataFrame
        Feature matrix.
    y : array-like
        Response variable.
    cv : splitter
        Object with a ``.split(X, y, groups)`` method yielding
        ``(train_idx, test_idx)`` tuples. Any sklearn splitter works.
    sample_weight : array-like, optional
        Sliced per fold; splitters operate on the physical compact rows, and
        they are read under the model's declared ``weight_semantics``. Under
        ``"frequency"``, integer values are likelihood-equivalent to literal
        row replication
        within a fixed train/validation partition and fixed feature geometry.
        Under ``"prior"`` they state a precision,
        ``Var(Y_i) = phi * V(mu_i) / w_i``; a Tweedie fit additionally
        requires them finite and strictly positive.
    offset : array-like, optional
        Offset term, sliced per fold.
    groups : array-like, optional
        Group labels forwarded to ``cv.split()``.
    fit_mode : {"fit", "fit_reml"}
        Which fit method to call on each fold estimator.
    scoring : str, callable, or sequence thereof
        Metrics to evaluate. Built-in: ``"deviance"``, ``"nll"``, ``"gini"``.
        Built-in deviance and NLL divide their weighted totals by the
        likelihood size the declared contract implies: ``sum(sample_weight)``
        under ``"frequency"``, the count of positive-weight rows under
        ``"prior"``. Gini remains a separately weighted ranking metric.
        Callables must follow ``scorer(model, X, y, *, sample_weight, offset) -> float | dict``.
    return_estimators : bool
        If True, keep the fitted model from each fold.
    return_oof : bool
        If True, collect out-of-fold predictions.
    error_score : float or "raise"
        Value to assign when a fold fails. ``"raise"`` propagates the error.

    Returns
    -------
    CrossValidationResult
        Per-fold scores, mean/std aggregates, and optionally out-of-fold
        predictions and fitted estimators.
    """
    # ── Validation ────────────────────────────────────────────────
    if not hasattr(cv, "split") or not callable(cv.split):
        raise TypeError("cv must be a splitter object with a .split() method")

    if fit_mode not in ("fit", "fit_reml"):
        raise ValueError(f"fit_mode must be 'fit' or 'fit_reml', got {fit_mode!r}")

    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    frame = as_eager_frame(X)

    if sample_weight is not None:
        try:
            raw_weight = np.asarray(sample_weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sample_weight must be a numeric one-dimensional array") from exc
        if raw_weight.ndim != 1:
            raise ValueError("sample_weight must be one-dimensional")
        if len(raw_weight) != n:
            raise ValueError(f"sample_weight length {len(raw_weight)} != y length {n}")
        if np.iscomplexobj(raw_weight) or getattr(raw_weight.dtype, "kind", None) in {"M", "m"}:
            raise ValueError("sample_weight must contain only real numeric values")
        try:
            sample_weight = np.asarray(raw_weight, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sample_weight must contain only real numeric values") from exc
        if not np.all(np.isfinite(sample_weight)):
            raise ValueError("sample_weight must contain only finite values")
        if isinstance(model._distribution, Tweedie) and (model_weight_semantics(model) == "prior"):
            if np.any(sample_weight <= 0.0):
                raise ValueError("Tweedie sample_weight must be strictly positive")
        elif np.any(sample_weight < 0.0):
            raise ValueError("sample_weight must be nonnegative")

    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if len(offset) != n:
            raise ValueError(f"offset length {len(offset)} != y length {n}")

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError(f"groups length {len(groups)} != y length {n}")

    scorers = _resolve_scorers(scoring)
    score_names = list(scorers.keys())

    # One vocabulary for every fold, resolved before the split: a level thin
    # enough to miss a training fold would otherwise invent a per-fold universe
    # and kill that fold at predict time.  A binding the caller already made on
    # a wider frame (bind_levels, spec §9) outranks this one: the frame here is
    # whatever slice CV was handed, so this pass fills gaps, never overwrites.
    existing = dict(getattr(getattr(model, "_config", None), "level_bindings", None) or ())
    level_bindings = {**_resolve_level_bindings(model, frame, sample_weight), **existing}

    # ── Fold loop ─────────────────────────────────────────────────
    fold_records: list[dict[str, Any]] = []
    fold_indices_list: list[tuple[NDArray, NDArray]] = []
    estimators_list: list | None = [] if return_estimators else None
    oof: NDArray | None = np.full(n, np.nan) if return_oof else None
    pooled_numerators: dict[str, float] = {
        name: 0.0 for name in score_names if name in _POOLED_PARTS
    }
    pooled_denominators: dict[str, float] = {
        name: 0.0 for name in score_names if name in _POOLED_PARTS
    }
    # Whether a built-in scorer wrote each score column last. A column a
    # callable writes, its dict's keys included, is not the built-in scorer's,
    # whatever it is named; one a built-in scorer writes after it again is.
    # Every fold scores in one order, so the last fold's writers are all folds'.
    written_by_builtin = {
        name: scorer is _BUILTIN_SCORERS.get(name) for name, scorer in scorers.items()
    }

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        fold_indices_list.append((train_idx.copy(), test_idx.copy()))

        record: dict[str, Any] = {
            "fold": fold_i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }

        # Slice data
        X_train = frame.take_rows(train_idx)
        X_test = frame.take_rows(test_idx)
        y_train = y[train_idx]
        y_test = y[test_idx]
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        sw_test = sample_weight[test_idx] if sample_weight is not None else None
        off_train = offset[train_idx] if offset is not None else None
        off_test = offset[test_idx] if offset is not None else None

        try:
            # Clone and fit
            est = _clone_model(model)
            if level_bindings:
                est._config = est._config.with_value(level_bindings=tuple(level_bindings.items()))
            t0 = time.perf_counter()
            fit_fn = getattr(est, fit_mode)
            fit_fn(X_train, y_train, sample_weight=sw_train, offset=off_train)
            record["fit_time_s"] = time.perf_counter() - t0
            record["converged"] = est._result.converged
            record["n_iter"] = est._result.n_iter
            record["effective_df"] = est._result.effective_df

            # Score
            t1 = time.perf_counter()
            for sname, sfn in scorers.items():
                pooled_fn = _POOLED_PARTS.get(sname)
                if pooled_fn is not None and sfn is _BUILTIN_SCORERS.get(sname):
                    # One evaluation per fold. The per-fold score is exactly
                    # the pooled parts' quotient -- both scorers divide the
                    # same numerator by the same denominator -- so calling the
                    # two separately predicted twice, and once the contract
                    # check landed in each, warned twice about one condition.
                    #
                    # Guarded on the scorer actually being the built-in.
                    # `_resolve_scorers` already normalises a name to its
                    # built-in today, so this cannot currently differ -- it is
                    # here so that a future resolver change cannot silently
                    # route a custom callable through the built-in's parts.
                    numerator, denominator = pooled_fn(
                        est,
                        X_test,
                        y_test,
                        sample_weight=sw_test,
                        offset=off_test,
                    )
                    record[sname] = float(numerator / denominator)
                    written_by_builtin[sname] = True
                    pooled_numerators[sname] += numerator
                    pooled_denominators[sname] += denominator
                    continue
                result = sfn(
                    est,
                    X_test,
                    y_test,
                    sample_weight=sw_test,
                    offset=off_test,
                )
                if isinstance(result, dict):
                    for k, v in result.items():
                        if k in _RESERVED_COLUMNS:
                            raise ValueError(
                                f"Scorer returned reserved column name {k!r}. "
                                f"Reserved: {_RESERVED_COLUMNS}"
                            )
                        record[k] = v
                    written_by_builtin.update(dict.fromkeys(result, False))
                else:
                    record[sname] = float(result)
                    written_by_builtin[sname] = sfn is _BUILTIN_SCORERS.get(sname)
                    pooled_fn = _POOLED_PARTS.get(sname)
                    if pooled_fn is not None:
                        numerator, denominator = pooled_fn(
                            est,
                            X_test,
                            y_test,
                            sample_weight=sw_test,
                            offset=off_test,
                        )
                        pooled_numerators[sname] += numerator
                        pooled_denominators[sname] += denominator
            record["score_time_s"] = time.perf_counter() - t1

            # OOF predictions
            if oof is not None:
                oof[test_idx] = est.predict(X_test, offset=off_test)

            if estimators_list is not None:
                estimators_list.append(est)

        except Exception as exc:
            if error_score == "raise":
                raise
            logger.warning(f"Fold {fold_i} failed: {exc!r}. Setting scores to {error_score}.")
            record["fit_time_s"] = np.nan
            record["score_time_s"] = np.nan
            record["converged"] = False
            record["n_iter"] = 0
            record["effective_df"] = np.nan
            for sname in score_names:
                record[sname] = error_score
            if estimators_list is not None:
                estimators_list.append(None)

        fold_records.append(record)

    # ── Assemble result ───────────────────────────────────────────
    fold_scores = pd.DataFrame(fold_records)

    # Compute mean/std only over score columns that are present
    present_score_cols = [c for c in fold_scores.columns if c in score_names]
    # Also include any extra keys from dict-returning scorers
    extra_cols = [
        c for c in fold_scores.columns if c not in _RESERVED_COLUMNS and c not in present_score_cols
    ]
    all_score_cols = present_score_cols + extra_cols

    mean_scores = {c: float(fold_scores[c].mean()) for c in all_score_cols}
    pooled_scores = {
        name: pooled_numerators[name] / pooled_denominators[name]
        for name in pooled_numerators
        if pooled_denominators[name] > 0.0
    }
    std_scores = {c: float(fold_scores[c].std(ddof=0)) for c in all_score_cols}

    curve_similarity = None
    if estimators_list is not None:
        from superglm.plotting.curve_similarity import build_cv_curve_similarity

        curve_similarity = build_cv_curve_similarity(
            models=estimators_list,
            X=X,
            sample_weight=sample_weight,
            n_points=200,
        )

    columns = _fingerprint_columns(model, frame)
    try:
        fingerprint = _data_fingerprint(frame, y, sample_weight, offset, columns)
    except (TypeError, ValueError):
        # A column the model reads that cannot be hashed: the result keeps
        # its folds and scores, and a consumer checks the row count only.
        fingerprint = None

    return CrossValidationResult(
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        pooled_scores=pooled_scores,
        std_scores=std_scores,
        fold_indices=fold_indices_list,
        curve_similarity=curve_similarity,
        oof_predictions=oof,
        estimators=estimators_list,
        n_rows=n,
        data_fingerprint=fingerprint,
        splitter=type(cv).__name__,
        fingerprint_columns=columns,
        fit_mode=fit_mode,
        fingerprint_version=None if fingerprint is None else FINGERPRINT_VERSION,
        builtin_scores=tuple(name for name in score_names if written_by_builtin[name]),
    )
