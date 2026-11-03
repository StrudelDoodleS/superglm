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

from superglm.distributional.checks.scores import _scoring_rows
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    resolve_likelihood_weights,
)
from superglm.editor.lss.view import LSSModelView

LSS_METRIC_LABELS = {
    "log_score": "Log score",
    "crps": "CRPS",
    "actual_expected": "Actual / expected",
}


def _fitted(model: Any):
    return model.lss if isinstance(model, LSSModelView) else model


def lss_split_reason(model: Any, dataset) -> str | None:
    """Why ``dataset`` cannot be scored by a SuperLSS (or the editor's view of one); None if it can.

    The split's weights are resolved under the fit's weight contract and bound
    to its family, as scoring does: a family with no law for those weights,
    such as non-unit prior weights on the log-normal, refuses them.
    """
    lss = _fitted(model)
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    resolved = resolve_likelihood_weights(
        dataset.sample_weight,
        n_observations=y.size,
        contract=lss._require_fitted().fit_state.weight_contract,
    )
    try:
        lss.family.bind_likelihood(y[resolved.input_positions], resolved, COMPLETE_OBSERVATION)
    except UnsupportedLikelihoodContractError as exc:
        detail = str(exc).rstrip(".")
        return (
            f"The {dataset.label} split cannot be scored with its weights. "
            f"{detail[:1].upper()}{detail[1:]}."
        )
    return None


def lss_dataset_metrics(model: Any, dataset) -> dict[str, float]:
    """Score ``dataset`` with a SuperLSS (or the editor's view of one).

    ``log_score`` is the negative log-likelihood and ``crps`` the CRPS, each the
    sum of ``model.scores`` contributions over the library's aggregation mass:
    the total weight under frequency weights, the number of scored rows under
    prior weights, whose rows already score their own weighted law.
    ``actual_expected`` is ``sum(w * y) / sum(w * mean)``. A family without a
    distribution function has no CRPS, which is then NaN. A split
    ``lss_split_reason`` refuses scores NaN throughout.
    """
    lss = _fitted(model)
    if lss_split_reason(lss, dataset) is not None:
        return dict.fromkeys(LSS_METRIC_LABELS, float("nan"))
    offsets = dataset.offset
    weights = dataset.sample_weight
    y = np.asarray(dataset.y, dtype=np.float64).ravel()
    w = np.ones(y.size) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    kwargs = {"sample_weight": weights, "offsets": offsets}
    mass = float(
        _scoring_rows(lss._require_fitted(), dataset.X, y, **kwargs).aggregation_mass.sum()
    )
    log = lss.scores(dataset.X, y, which=("log",), **kwargs)["log"]
    try:
        crps = lss.scores(dataset.X, y, which=("crps",), **kwargs)["crps"]
        crps_score = float(np.nansum(crps) / mass)
    except NotImplementedError:
        crps_score = float("nan")
    mean = np.asarray(lss.predict(dataset.X, offsets=offsets), dtype=np.float64)
    return {
        "log_score": float(np.nansum(log) / mass),
        "crps": crps_score,
        "actual_expected": float(np.sum(w * y) / np.sum(w * mean)),
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
