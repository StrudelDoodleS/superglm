"""Persistence helpers for editor sessions and edited model copies."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from superglm._frame import as_eager_frame
from superglm.editor.apply import materialize_edit_request
from superglm.editor.io import (
    jsonable,
    record_from_payload,
    record_to_payload,
    term_to_payload,
    validate_loaded_term,
)
from superglm.model import SuperGLM

if TYPE_CHECKING:
    from superglm.editor.evaluation import EvaluationDataset


@dataclass(frozen=True)
class ModelArtifactValidation:
    """Successful checks performed on serialized model bytes."""

    artifact_round_trip: bool
    prediction_rows: int
    scope: str


def joblib_load_bytes(data: bytes):
    """Load joblib bytes through the monkeypatchable artifact boundary."""
    import joblib

    return joblib.load(io.BytesIO(data))


def _spread_row_indices(n_rows: int, max_rows: int) -> np.ndarray:
    """Choose deterministic rows spread across the complete dataset."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if n_rows <= 0:
        return np.empty(0, dtype=np.intp)
    n_selected = min(int(n_rows), int(max_rows))
    if n_selected == n_rows:
        return np.arange(n_rows, dtype=np.intp)
    return np.linspace(0, n_rows - 1, num=n_selected, dtype=np.intp)


def _slice_rows(values: Any, indices: np.ndarray):
    """Take only selected rows from pandas or array-like inputs."""
    try:
        frame = as_eager_frame(values)
    except ValueError:
        pass
    else:
        return frame.take_rows(indices)
    return np.asarray(values)[indices]


def _validate_artifact_contract(model, loaded_model) -> None:
    """Validate cheap fitted-model invariants after a joblib round trip."""
    try:
        if not isinstance(loaded_model, SuperGLM) or type(loaded_model) is not type(model):
            raise ValueError("loaded object has the wrong model type")

        original_result = model.result
        loaded_result = loaded_model.result
        if loaded_result is None:
            raise ValueError("loaded model has no fitted result")
        if not callable(getattr(loaded_model, "predict", None)):
            raise ValueError("loaded model has no callable predict")
        if not np.isfinite(loaded_result.intercept):
            raise ValueError("loaded model intercept is non-finite")

        original_beta = np.asarray(original_result.beta)
        loaded_beta = np.asarray(loaded_result.beta)
        if loaded_beta.shape != original_beta.shape:
            raise ValueError("loaded model beta shape differs")
        if not np.all(np.isfinite(loaded_beta)):
            raise ValueError("loaded model beta is non-finite")

        if list(loaded_model._feature_order) != list(model._feature_order):
            raise ValueError("loaded model feature order differs")
        original_specs = model._specs
        loaded_specs = loaded_model._specs
        if list(loaded_specs) != list(original_specs):
            raise ValueError("loaded model feature spec keys/order differ")
        if any(type(loaded_specs[name]) is not type(spec) for name, spec in original_specs.items()):
            raise ValueError("loaded model feature spec type differs")
    except ValueError as exc:
        raise ValueError(f"artifact validation failed: {exc}") from exc
    except Exception as exc:
        raise ValueError("artifact validation failed: invalid fitted model contract") from exc


def serialize_validated_model(
    model,
    *,
    dataset: EvaluationDataset | None = None,
    max_rows: int = 512,
) -> tuple[bytes, ModelArtifactValidation]:
    """Serialize and reload a fitted model, optionally comparing bounded predictions."""
    import joblib

    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")

    buffer = io.BytesIO()
    try:
        joblib.dump(model, buffer)
        data = bytes(buffer.getvalue())
        loaded_model = joblib_load_bytes(data)
    except Exception as exc:
        raise ValueError("artifact validation failed: joblib round trip failed") from exc

    _validate_artifact_contract(model, loaded_model)

    n_rows = 0 if dataset is None else dataset.n_obs
    if n_rows == 0:
        return data, ModelArtifactValidation(
            artifact_round_trip=True,
            prediction_rows=0,
            scope="artifact",
        )

    indices = _spread_row_indices(n_rows, max_rows)
    X_rows = _slice_rows(dataset.X, indices)
    offset_rows = None if dataset.offset is None else _slice_rows(dataset.offset, indices)
    try:
        original_predictions = np.asarray(
            model.predict(X_rows, offset=offset_rows),
            dtype=np.float64,
        )
        loaded_predictions = np.asarray(
            loaded_model.predict(X_rows, offset=offset_rows),
            dtype=np.float64,
        )
        valid_predictions = (
            original_predictions.shape == loaded_predictions.shape
            and np.all(np.isfinite(original_predictions))
            and np.all(np.isfinite(loaded_predictions))
            and np.allclose(
                original_predictions,
                loaded_predictions,
                rtol=1e-12,
                atol=1e-12,
            )
        )
    except Exception as exc:
        raise ValueError("prediction validation failed: model scoring failed") from exc
    if not valid_predictions:
        raise ValueError("prediction validation failed: round-trip predictions differ")

    return data, ModelArtifactValidation(
        artifact_round_trip=True,
        prediction_rows=len(indices),
        scope="artifact+predictions",
    )


def save_model(session, path: str | Path, *, model_override=None) -> Path:
    """Write the edited model copy as a joblib artifact."""
    from superglm.editor.evaluation import default_metrics_dataset

    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(".joblib")
    model = edited_model_for_export(session, model_override=model_override)
    data, _ = serialize_validated_model(
        model,
        dataset=default_metrics_dataset(session),
    )
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_bytes(data)
    return target


def edited_model_for_export(session, *, model_override=None):
    """Return an edited model copy scored on the editor's default split when available."""
    if model_override is not None:
        return model_override
    request = session.capture_materialization_request()
    if request is None:
        return session.model
    return materialize_edit_request(request)


def save_session(session, path: str | Path) -> None:
    """Write an auditable JSON edit artifact."""
    payload = {
        "format": "superglm.editor.v1",
        "n_points": session.n_points,
        "centering": session.centering,
        "terms": [term_to_payload(term) for term in session.terms.values()],
        "level_orders": {name: list(labels) for name, labels in session._level_orders.items()},
        "selection": {
            name: session._selection[name].astype(int).tolist() for name in session.terms
        },
        "history": [record_to_payload(record) for record in session.history],
    }
    Path(path).write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def load_session(session_cls, path: str | Path, *, model):
    """Load an edit artifact against a fitted model."""
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "superglm.editor.v1":
        raise ValueError(f"Unsupported editor artifact format: {payload.get('format')!r}")

    term_payloads = payload["terms"]
    session = session_cls.from_model(
        model,
        terms=[term["name"] for term in term_payloads],
        n_points=int(payload.get("n_points", 200)),
        centering=str(payload.get("centering", "native")),
    )
    session._level_orders = {
        str(name): [str(label) for label in labels]
        for name, labels in payload.get("level_orders", {}).items()
    }
    session._reapply_level_orders()
    for term_payload in term_payloads:
        term = session.terms[term_payload["name"]]
        validate_loaded_term(term, term_payload)
        term.edited_log_effect = np.asarray(term_payload["edited_log_effect"], dtype=np.float64)
        if term_payload.get("weights") is not None:
            term.weights = np.asarray(term_payload["weights"], dtype=np.float64)
        if term_payload.get("ci_lower_log_effect") is not None:
            term.ci_lower_log_effect = np.asarray(
                term_payload["ci_lower_log_effect"],
                dtype=np.float64,
            )
        if term_payload.get("ci_upper_log_effect") is not None:
            term.ci_upper_log_effect = np.asarray(
                term_payload["ci_upper_log_effect"],
                dtype=np.float64,
            )

    selection = payload.get("selection", {})
    for name, indices in selection.items():
        if name in session.terms:
            session.select_indices(name, indices)

    session.history = [record_from_payload(record) for record in payload.get("history", [])]
    session.redo_stack = []
    return session
