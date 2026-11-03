"""Validation scores and Final fit rows for SuperLSS models.

A SuperLSS has no single mean predictor, so deviance and AIC do not describe
it. Its splits are scored by the log score and the CRPS (Gneiting and Raftery,
2007), both lower-is-better, and by actual over expected, which reads the mean
alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.editor.lss.view import LSSModelView

LSS_METRIC_LABELS = {
    "log_score": "Log score",
    "crps": "CRPS",
    "actual_expected": "Actual / expected",
}


def _fitted(model: Any):
    return model.lss if isinstance(model, LSSModelView) else model


def lss_dataset_metrics(model: Any, dataset) -> dict[str, float]:
    """Score ``dataset`` with a SuperLSS (or the editor's view of one).

    ``log_score`` is the negative log-likelihood and ``crps`` the CRPS, each the
    weighted contributions of ``model.scores`` over the total weight.
    ``actual_expected`` is ``sum(w * y) / sum(w * mean)``. A family without a
    distribution function has no CRPS, which is then NaN.
    """
    lss = _fitted(model)
    offsets = dataset.offset
    weights = dataset.sample_weight
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    w = np.ones(y.size) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    total = float(np.sum(w))
    kwargs = {"sample_weight": weights, "offsets": offsets}
    try:
        scores = lss.scores(dataset.X, y, which=("log", "crps"), **kwargs)
    except NotImplementedError:
        scores = lss.scores(dataset.X, y, which=("log",), **kwargs)
    mean = np.asarray(lss.predict(dataset.X, offsets=offsets), dtype=np.float64)
    expected = float(np.sum(w * mean))
    crps = np.nansum(scores["crps"]) / total if "crps" in scores else np.nan
    return {
        "log_score": float(np.nansum(scores["log"]) / total),
        "crps": float(crps),
        "actual_expected": float(np.sum(w * y) / expected),
    }


def lss_metric_signature(model: Any) -> tuple[Any, ...]:
    """The fitted state that can change a SuperLSS's scores: its family and coefficients."""
    lss = _fitted(model)
    coefficients = np.asarray(lss._require_fitted().coefficients, dtype=np.float64)
    return (type(lss.family).__qualname__, repr(lss.family.to_config()), coefficients.tobytes())


def offset_departs(column: Any, neutral: float) -> bool:
    """Whether a split's offsets (an array, or arrays by parameter) hold anything but ``neutral``."""
    arrays = column.values() if isinstance(column, Mapping) else (column,)
    return any(bool(np.any(np.asarray(a, dtype=np.float64) != neutral)) for a in arrays)


def stack_offsets(datasets: Sequence[Any]) -> dict[str, NDArray[np.float64]] | None:
    """The splits' per-parameter offsets stacked in order, zero where a split lacks one."""
    columns = [dataset.offset for dataset in datasets]
    parameters = dict.fromkeys(key for column in columns if column for key in column)
    if not parameters:
        return None
    return {
        key: np.concatenate(
            [
                np.asarray(column[key], dtype=np.float64)
                if column and key in column
                else np.zeros(dataset.n_obs)
                for dataset, column in zip(datasets, columns, strict=True)
            ]
        )
        for key in parameters
    }
