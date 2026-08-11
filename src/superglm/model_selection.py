"""Cross-validation with pluggable splitters and scorers."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributions import Tweedie
from superglm.solvers.dispersion import dispersion_likelihood_size
from superglm.validation import _normalized_gini

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResult:
    """Structured result from :func:`cross_validate`.

    Attributes
    ----------
    fold_scores : DataFrame
        One row per fold with columns: ``fold``, ``n_train``, ``n_test``,
        ``fit_time_s``, ``score_time_s``, ``converged``, ``n_iter``,
        ``effective_df``, plus one column per requested metric.
    mean_scores : dict
        Equal-weight mean of each per-fold metric across folds. Built-in
        deviance and negative log-likelihood are normalized within each fold
        by ``sum(sample_weight)`` for non-Tweedie frequency weights and by the
        physical validation-row count for Tweedie EDM prior weights.
    pooled_scores : dict
        Supported overall pooled metrics, computed as ratio-of-sums rather than
        mean-of-fold-ratios, with the same family-specific denominator.
    std_scores : dict
        Standard deviation of each metric across folds.
    fold_indices : list[tuple[ndarray, ndarray]] or None
        Per-fold ``(train_idx, test_idx)`` pairs from the CV splitter.
    curve_similarity : dict or None
        Fold-by-fold term similarity diagnostics for comparable main effects.
    oof_predictions : ndarray or None
        Out-of-fold predictions (response scale), same length as *y*.
        ``None`` unless ``return_oof=True``.
    estimators : list or None
        Fitted model per fold. ``None`` unless ``return_estimators=True``.
    """

    fold_scores: pd.DataFrame
    mean_scores: dict[str, float]
    pooled_scores: dict[str, float]
    std_scores: dict[str, float]
    fold_indices: list[tuple[NDArray, NDArray]] | None = None
    curve_similarity: dict[str, Any] | None = None
    oof_predictions: NDArray | None = None
    estimators: list | None = None

    def plot_terms_by_fold(
        self,
        X: FrameLike,
        *,
        sample_weight: NDArray | None = None,
        terms: str | list[str] | None = None,
        engine: str = "plotly",
        **kwargs,
    ):
        """Plot fold-specific main effects using the shared comparison engine."""
        if self.estimators is None:
            raise RuntimeError("return_estimators=True is required for plot_terms_by_fold().")

        from superglm.plotting.comparison import plot_term_comparison

        models = {
            f"fold_{fold}": est for fold, est in enumerate(self.estimators) if est is not None
        }
        if not models:
            raise RuntimeError("No fitted fold estimators are available to plot.")
        frame = as_eager_frame(X)
        support_by_label: dict[str, dict[str, Any]] = {}
        weight_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
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


# ── Model cloning ────────────────────────────────────────────────


def _clone_model(model):
    """Create a fresh (unfitted) copy of *model* preserving constructor config."""
    return model.clone_unfitted()


# ── Full-frame level binding ─────────────────────────────────────


def _auto_detected_templates(model, frame: EagerFrame, sample_weight) -> list[tuple[Any, Any]]:
    """Return the specs the fit path's own auto-detection builds on *frame*.

    Classification is not reimplemented here: a throwaway clone runs the very
    detection each fold will run, so a column that becomes categorical for the
    fold becomes categorical for the binding pass too.
    """
    if getattr(model._config, "splines", None) is None:
        # Without the splines shorthand an empty feature set is a configuration
        # error the fold fit reports; there is nothing to detect or bind.
        return []

    from superglm.model.base import auto_detect

    probe = _clone_model(model)
    try:
        auto_detect(probe, frame, sample_weight)
    except Exception as exc:
        # Detection failures belong to the fold that raises them, where
        # error_score decides the outcome; binding must not preempt that.
        logger.debug(f"Level binding skipped: feature auto-detection failed: {exc!r}")
        return []
    return [(name, probe._specs[name]) for name in probe._feature_order]


def _resolve_level_bindings(model, frame: EagerFrame, sample_weight) -> dict[Any, Any]:
    """Bind level universes and most-exposed bases on the full pre-split frame.

    Sharing the level SET across folds is R factor semantics: the vocabulary is
    a property of the data column, not of the training subset, so no target
    information crosses folds (spec 2026-08-11, §3.5).  Quantities that do
    depend on training rows -- knots, penalties, coefficients -- keep binding
    per fold.
    """
    config = getattr(model, "_config", None)
    if config is None:
        return {}
    templates: list[tuple[Any, Any]] = list(config.feature_templates)
    if not templates:
        templates = _auto_detected_templates(model, frame, sample_weight)

    available = set(frame.columns)
    bindings: dict[Any, Any] = {}
    for name, spec in templates:
        # Terms that declare their own universe (OrderedCategorical) or hold no
        # universe at all (numeric, spline) never grow the hook.
        if not hasattr(spec, "resolve_binding") or name not in available:
            continue
        probe = copy.deepcopy(spec)
        declared = frame.column_declared_categories(name)
        if declared is not None and hasattr(probe, "adopt_dtype_categories"):
            probe.adopt_dtype_categories(declared)
        try:
            bindings[name] = probe.resolve_binding(frame.column_array(name), sample_weight)
        except Exception as exc:
            # A column the whole frame cannot bind (missing values, data outside
            # a declared universe) is a fit error, and stays one: it is reported
            # per fold under the caller's error_score rather than raised here.
            logger.debug(f"Level binding skipped for feature {name!r}: {exc!r}")
    return bindings


# ── Built-in scorers ─────────────────────────────────────────────


def _scoring_weights(model, sample_weight, n_rows: int) -> tuple[NDArray, float]:
    """Return scoring weights and the family's validation likelihood size."""
    weights = (
        np.ones(n_rows, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    denominator = dispersion_likelihood_size(model._distribution, weights)
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
    ll = model._distribution.log_likelihood(y_val, mu, weights, phi=model.result.phi)
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
    ll = model._distribution.log_likelihood(y_val, mu, weights, phi=model.result.phi)
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
        Sliced per fold; splitters operate on the physical compact rows. For
        non-Tweedie families these are nonnegative case/frequency weights:
        within a fixed train/validation partition and fixed feature geometry,
        integer values are likelihood-equivalent to literal row replication.
        For Tweedie they are finite, strictly positive EDM prior weights, with
        ``Var(Y_i) = phi * mu_i**p / w_i``.
    offset : array-like, optional
        Offset term, sliced per fold.
    groups : array-like, optional
        Group labels forwarded to ``cv.split()``.
    fit_mode : {"fit", "fit_reml"}
        Which fit method to call on each fold estimator.
    scoring : str, callable, or sequence thereof
        Metrics to evaluate. Built-in: ``"deviance"``, ``"nll"``, ``"gini"``.
        Built-in deviance and NLL divide their weighted totals by
        ``sum(sample_weight)`` for non-Tweedie frequency weights and by the
        physical row count for Tweedie prior weights. Gini remains a separately
        weighted ranking metric.
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
        if isinstance(model._distribution, Tweedie):
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
    # and kill that fold at predict time.
    level_bindings = _resolve_level_bindings(model, frame, sample_weight)

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
                else:
                    record[sname] = float(result)
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

    return CrossValidationResult(
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        pooled_scores=pooled_scores,
        std_scores=std_scores,
        fold_indices=fold_indices_list,
        curve_similarity=curve_similarity,
        oof_predictions=oof,
        estimators=estimators_list,
    )
