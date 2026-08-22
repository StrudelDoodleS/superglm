"""Evaluation split inputs for editor reports and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from superglm._frame import as_eager_frame

_SPLIT_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "test": "Test",
}


@dataclass(frozen=True)
class EvaluationDataset:
    """A user-supplied or retained-data evaluation split."""

    name: str
    label: str
    X: Any
    y: Any
    sample_weight: Any = None
    offset: Any = None
    source: str = "supplied"

    @property
    def n_obs(self) -> int:
        return len(self.X)


def _validate_evaluation_weights(sample_weight, n_obs: int, *, family=None, weight_semantics):
    """Normalize split weights and enforce the family-specific public contract."""
    if sample_weight is None:
        return None

    from superglm.distributions import Tweedie

    if isinstance(family, Tweedie) and weight_semantics == "prior":
        from superglm._utils import _validate_strict_prior_weights

        return _validate_strict_prior_weights(sample_weight, n_obs)

    from superglm.model.input_validation import _finite_vector

    weights = _finite_vector("sample_weight", sample_weight, n_obs)
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must not be all zero")
    return weights


def coerce_evaluation_data(
    *,
    train_data=None,
    validation_data=None,
    test_data=None,
    family=None,
    weight_semantics: str,
) -> dict[str, EvaluationDataset]:
    """Normalize plain ``(X, y[, w[, offset]])`` tuples by split name."""
    datasets: dict[str, EvaluationDataset] = {}
    for name, data in (
        ("train", train_data),
        ("validation", validation_data),
        ("test", test_data),
    ):
        dataset = coerce_dataset(
            name,
            data,
            family=family,
            weight_semantics=weight_semantics,
        )
        if dataset is not None:
            datasets[name] = dataset
    return datasets


def coerce_dataset(
    name: str,
    data,
    *,
    family=None,
    weight_semantics: str,
) -> EvaluationDataset | None:
    """Convert a split tuple into an :class:`EvaluationDataset`."""
    if data is None:
        return None
    if not isinstance(data, tuple | list):
        raise TypeError(f"{name}_data must be a tuple like (X, y, sample_weight).")
    if len(data) not in {2, 3, 4}:
        raise ValueError(f"{name}_data must contain 2, 3, or 4 values.")

    X, y = data[0], data[1]
    sample_weight = data[2] if len(data) >= 3 else None
    offset = data[3] if len(data) == 4 else None
    n_obs = int(np.asarray(y).shape[0])
    frame = as_eager_frame(X)
    if len(frame) != n_obs:
        raise ValueError(f"{name}_data X and y lengths differ.")
    sample_weight = _validate_evaluation_weights(
        sample_weight,
        n_obs,
        family=family,
        weight_semantics=weight_semantics,
    )
    if offset is not None and len(offset) != n_obs:
        raise ValueError(f"{name}_data offset length differs from y.")
    return EvaluationDataset(
        name=name,
        label=_SPLIT_LABELS.get(name, name.title()),
        X=frame.native,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )


def evaluation_datasets(session) -> list[EvaluationDataset]:
    """Return explicit splits, with retained fit data as a train fallback."""
    explicit = getattr(session, "_evaluation_data", {})
    datasets: list[EvaluationDataset] = []
    retained = retained_fit_dataset(session)
    if "train" in explicit:
        datasets.append(explicit["train"])
    elif retained is not None:
        datasets.append(retained)
    for name in ("validation", "test"):
        if name in explicit:
            datasets.append(explicit[name])
    return datasets


def default_metrics_dataset(session) -> EvaluationDataset | None:
    """Prefer validation metrics, otherwise train/retained fit metrics."""
    explicit = getattr(session, "_evaluation_data", {})
    if "validation" in explicit:
        return explicit["validation"]
    for dataset in evaluation_datasets(session):
        return dataset
    return None


def training_export_dataset(session) -> EvaluationDataset | None:
    """Return training data suitable for rating-table export.

    Validation and test splits are deliberately excluded: rating tables describe
    the fitted portfolio, so callers must provide ``train_data`` or retain fit data.
    """
    explicit = getattr(session, "_evaluation_data", {})
    if "train" in explicit:
        return explicit["train"]
    return retained_fit_dataset(session)


def named_metrics_dataset(session, name: str | None) -> EvaluationDataset | None:
    """Resolve an optional split name for metric/report calls."""
    if name in (None, "", "default"):
        return default_metrics_dataset(session)
    for dataset in evaluation_datasets(session):
        if dataset.name == name:
            return dataset
    return None


def retained_fit_dataset(session) -> EvaluationDataset | None:
    """Expose retained fit data as an in-sample train split."""
    reference_model = getattr(session, "reference_model", session.model)
    if reference_model is None:
        return None
    X = getattr(reference_model, "_fit_X_ref", None)
    y = getattr(reference_model, "_fit_y_ref", None)
    if X is None or y is None:
        return None
    return EvaluationDataset(
        name="train",
        label="Train",
        X=X,
        y=y,
        sample_weight=getattr(reference_model, "_fit_weights", None),
        offset=getattr(reference_model, "_fit_offset", None),
        source="retained_fit_data",
    )


__all__ = [
    "EvaluationDataset",
    "coerce_dataset",
    "coerce_evaluation_data",
    "default_metrics_dataset",
    "evaluation_datasets",
    "named_metrics_dataset",
    "retained_fit_dataset",
    "training_export_dataset",
]
