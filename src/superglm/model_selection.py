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
        Mean of each metric across folds.
    std_scores : dict
        Standard deviation of each metric across folds.
    oof_predictions : ndarray or None
        Out-of-fold predictions (response scale), same length as *y*.
        ``None`` unless ``return_oof=True``.
    estimators : list or None
        Fitted model per fold. ``None`` unless ``return_estimators=True``.
    """

    fold_scores: pd.DataFrame
    mean_scores: dict[str, float]
    std_scores: dict[str, float]
    oof_predictions: NDArray | None = None
    estimators: list | None = None


# ── Model cloning ────────────────────────────────────────────────


def _clone_model(model):
    """Create a fresh (unfitted) copy of *model* preserving constructor config."""
    new_penalty = copy.deepcopy(model.penalty)

    interactions: list[tuple[str, str]] | None = None
    if model._interaction_order:
        interactions = [
            model._interaction_specs[iname].parent_names for iname in model._interaction_order
        ]
    elif model._pending_interactions:
        interactions = list(model._pending_interactions)

    # A fitted auto-detect model has both _specs (populated at fit time) and
    # _splines (the original constructor arg).  The constructor forbids passing
    # both, so we must choose: if _specs exist use features=, otherwise fall
    # back to the splines= auto-detect path.
    if model._specs:
        features = {k: copy.deepcopy(v) for k, v in model._specs.items()}
        splines = None
    else:
        features = None
        splines = list(model._splines) if model._splines else None

    return type(model)(
        family=model.family,
        link=model.link,
        penalty=new_penalty,
        features=features,
        splines=splines,
        n_knots=model._n_knots,
        degree=model._degree,
        categorical_base=model._categorical_base,
        interactions=interactions,
        active_set=model._active_set,
        direct_solve=model._direct_solve,
        discrete=model._discrete,
        n_bins=model._n_bins,
    )


# ── Built-in scorers ─────────────────────────────────────────────


def _score_deviance(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Mean weighted unit deviance on the validation set."""
    mu = model.predict(X_val, offset=offset)
    dev = model._distribution.deviance_unit(y_val, mu)
    if sample_weight is not None:
        return float(np.sum(sample_weight * dev) / np.sum(sample_weight))
    return float(np.mean(dev))


def _score_nll(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Mean negative log-likelihood on the validation set."""
    mu = model.predict(X_val, offset=offset)
    w = sample_weight if sample_weight is not None else np.ones(len(y_val))
    ll = model._distribution.log_likelihood(y_val, mu, w, phi=model.result.phi)
    return float(-ll / np.sum(w))


def _score_gini(model, X_val, y_val, *, sample_weight=None, offset=None):
    """Gini coefficient (2 * AUC - 1) for binary/frequency models."""
    mu = model.predict(X_val, offset=offset)
    w = sample_weight if sample_weight is not None else np.ones(len(y_val))
    total_wy = np.sum(w * y_val)
    if total_wy == 0.0:
        return 0.0
    order = np.argsort(mu)
    y_sorted = y_val[order]
    w_sorted = w[order]
    total_w = np.sum(w_sorted)
    cum_wy = np.cumsum(w_sorted * y_sorted)
    gini = 1.0 - 2.0 * np.sum(w_sorted * cum_wy) / (total_w * total_wy)
    return float(gini)


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
    X: pd.DataFrame,
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
    X : DataFrame
        Feature matrix.
    y : array-like
        Response variable.
    cv : splitter
        Object with a ``.split(X, y, groups)`` method yielding
        ``(train_idx, test_idx)`` tuples. Any sklearn splitter works.
    sample_weight : array-like, optional
        Frequency weights, sliced per fold.
    offset : array-like, optional
        Offset term, sliced per fold.
    groups : array-like, optional
        Group labels forwarded to ``cv.split()``.
    fit_mode : {"fit", "fit_reml"}
        Which fit method to call on each fold estimator.
    scoring : str, callable, or sequence thereof
        Metrics to evaluate. Built-in: ``"deviance"``, ``"nll"``, ``"gini"``.
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
    """
    # ── Validation ────────────────────────────────────────────────
    if not hasattr(cv, "split") or not callable(cv.split):
        raise TypeError("cv must be a splitter object with a .split() method")

    if fit_mode not in ("fit", "fit_reml"):
        raise ValueError(f"fit_mode must be 'fit' or 'fit_reml', got {fit_mode!r}")

    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if len(sample_weight) != n:
            raise ValueError(f"sample_weight length {len(sample_weight)} != y length {n}")

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

    # ── Fold loop ─────────────────────────────────────────────────
    fold_records: list[dict[str, Any]] = []
    estimators_list: list | None = [] if return_estimators else None
    oof: NDArray | None = np.full(n, np.nan) if return_oof else None

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)

        record: dict[str, Any] = {
            "fold": fold_i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }

        # Slice data
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        sw_test = sample_weight[test_idx] if sample_weight is not None else None
        off_train = offset[train_idx] if offset is not None else None
        off_test = offset[test_idx] if offset is not None else None

        try:
            # Clone and fit
            est = _clone_model(model)
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
    std_scores = {c: float(fold_scores[c].std(ddof=0)) for c in all_score_cols}

    return CrossValidationResult(
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        std_scores=std_scores,
        oof_predictions=oof,
        estimators=estimators_list,
    )
