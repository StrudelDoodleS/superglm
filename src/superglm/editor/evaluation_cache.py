"""Bounded scalar evaluation cache for one editor widget."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from superglm.editor._types import EditableTerm
from superglm.editor.evaluation import EvaluationDataset


@dataclass(frozen=True)
class EvaluationKey:
    """Identify one model/split metric dictionary."""

    role: Literal["original", "current"]
    model_revision: int
    dataset_epoch: int
    split: str
    metric_signature: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class EditMaterializationRequest:
    """Immutable plot-scale snapshot needed to construct one edited model."""

    model_revision: int
    edit_epoch: int
    base_model: Any
    terms: dict[str, EditableTerm]
    dataset: EvaluationDataset | None


@dataclass(frozen=True, slots=True)
class DatasetMetricRequest:
    """Immutable model/dataset references for one scalar metric calculation."""

    key: EvaluationKey
    model: Any
    dataset: EvaluationDataset


def model_metric_signature(model) -> tuple[Any, ...]:
    """Return the fitted state that can change scalar model metrics."""
    family = model._distribution
    link = model._link
    return (
        type(family).__module__,
        type(family).__qualname__,
        getattr(family, "p", None),
        getattr(family, "theta", None),
        type(link).__module__,
        type(link).__qualname__,
        float(model.result.phi),
        float(model.result.effective_df),
    )


class EvaluationCache:
    """Retain original scalars and scalars for at most one current revision."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._original: dict[EvaluationKey, dict[str, float]] = {}
        self._current: dict[EvaluationKey, dict[str, float]] = {}
        self._current_revision: int | None = None

    def advance_current_revision(self, revision: int) -> None:
        """Select a revision and discard all prior current-model values."""
        revision = int(revision)
        with self._lock:
            if self._current_revision != revision:
                self._current.clear()
                self._current_revision = revision

    def get(self, key: EvaluationKey) -> dict[str, float] | None:
        """Return an isolated copy of a cached metric dictionary, if present."""
        with self._lock:
            target = self._original if key.role == "original" else self._current
            payload = target.get(key)
            return None if payload is None else dict(payload)

    def put(self, key: EvaluationKey, values: dict[str, Any]) -> bool:
        """Store scalarized values unless a current-model key is superseded."""
        payload = {str(name): float(value) for name, value in values.items()}
        with self._lock:
            if key.role == "current" and key.model_revision != self._current_revision:
                return False
            target = self._original if key.role == "original" else self._current
            target[key] = payload
            return True

    def persistent_values_are_scalar(self) -> bool:
        """Report whether every retained value is a scalar float."""
        with self._lock:
            return all(
                isinstance(value, float) and np.ndim(value) == 0
                for mapping in (self._original, self._current)
                for payload in mapping.values()
                for value in payload.values()
            )


__all__ = [
    "DatasetMetricRequest",
    "EditMaterializationRequest",
    "EvaluationCache",
    "EvaluationKey",
    "model_metric_signature",
]
