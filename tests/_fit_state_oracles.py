"""Behavioral oracles shared by fit-state transaction tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class InjectedFitFailure(RuntimeError):  # noqa: N818 - matches the fit-state design vocabulary
    """Deliberate failure raised after a selected fit phase."""


@dataclass(frozen=True)
class ModelBehaviorSnapshot:
    dict_object: dict[str, Any]
    fit_state: object
    fit_revision: int
    predictions: np.ndarray
    beta: np.ndarray
    intercept: float
    deviance: float
    summary: object
    projections: dict[str, object]


def snapshot_model_behavior(model, X: pd.DataFrame) -> ModelBehaviorSnapshot:
    """Capture public behavior and fitted-object identities before a failed refit."""
    summary = model.summary()
    projection_names = (
        "_result",
        "_solver_result",
        "_dm",
        "_groups",
        "_specs",
        "_distribution",
        "_link",
        "_fit_weights",
        "_fit_offset",
        "_fit_stats",
        "_runtime_canonical_state",
        "_prediction_plan",
        "_fast_prediction_state",
        "_fit_mu",
        "_fit_null_mu",
        "_summary_cache",
    )
    result = model.result
    return ModelBehaviorSnapshot(
        dict_object=model.__dict__,
        fit_state=model._fit_state,
        fit_revision=model._fit_revision,
        predictions=np.asarray(model.predict(X), dtype=np.float64).copy(),
        beta=np.asarray(result.beta, dtype=np.float64).copy(),
        intercept=float(result.intercept),
        deviance=float(result.deviance),
        summary=summary,
        projections={name: getattr(model, name) for name in projection_names},
    )


def assert_model_behavior_unchanged(
    model,
    X: pd.DataFrame,
    before: ModelBehaviorSnapshot,
) -> None:
    """Assert the strong exception guarantee for an already-fitted model."""
    assert model.__dict__ is before.dict_object
    assert model._fit_state is before.fit_state
    assert model._fit_revision == before.fit_revision
    for name, previous in before.projections.items():
        assert getattr(model, name) is previous, name
    assert model.summary() is before.summary
    np.testing.assert_array_equal(model.result.beta, before.beta)
    assert model.result.intercept == before.intercept
    assert model.result.deviance == before.deviance
    np.testing.assert_array_equal(model.predict(X), before.predictions)
