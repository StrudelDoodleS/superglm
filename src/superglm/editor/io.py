"""Serialization helpers for editor sessions and JSON APIs."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm.editor._types import EditableTerm, EditRecord


def term_to_payload(term: EditableTerm) -> dict[str, Any]:
    return {
        "name": term.name,
        "kind": term.kind,
        "x": None if term.x is None else term.x.tolist(),
        "levels": term.levels,
        "original_log_effect": term.original_log_effect.tolist(),
        "edited_log_effect": term.edited_log_effect.tolist(),
        "weights": None if term.weights is None else term.weights.tolist(),
        "ci_lower_log_effect": (
            None if term.ci_lower_log_effect is None else term.ci_lower_log_effect.tolist()
        ),
        "ci_upper_log_effect": (
            None if term.ci_upper_log_effect is None else term.ci_upper_log_effect.tolist()
        ),
        "metadata": term.metadata,
    }


def record_to_payload(record: EditRecord) -> dict[str, Any]:
    return {
        "term": record.term,
        "operation": record.operation,
        "indices": record.indices.astype(int).tolist(),
        "before": record.before.tolist(),
        "after": record.after.tolist(),
        "params": record.params,
    }


def record_from_payload(payload: dict[str, Any]) -> EditRecord:
    return EditRecord(
        term=str(payload["term"]),
        operation=str(payload["operation"]),
        indices=np.asarray(payload["indices"], dtype=np.intp),
        before=np.asarray(payload["before"], dtype=np.float64),
        after=np.asarray(payload["after"], dtype=np.float64),
        params=dict(payload.get("params", {})),
    )


def validate_loaded_term(term: EditableTerm, payload: dict[str, Any]) -> None:
    # Loaded artifacts are only valid against the same term shape/levels/grid.
    # Reject mismatches before mutating the fresh session.
    if term.kind != payload["kind"]:
        raise ValueError(
            f"Loaded term {term.name!r} has kind {payload['kind']!r}, "
            f"but model exposes {term.kind!r}."
        )
    edited = np.asarray(payload["edited_log_effect"], dtype=np.float64)
    if edited.shape != term.edited_log_effect.shape:
        raise ValueError(
            f"Loaded term {term.name!r} has shape {edited.shape}, "
            f"but model exposes {term.edited_log_effect.shape}."
        )
    original = np.asarray(payload.get("original_log_effect"), dtype=np.float64)
    if original.shape != term.original_log_effect.shape or not np.allclose(
        original,
        term.original_log_effect,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(f"Loaded baseline for term {term.name!r} does not match the fitted model.")
    for key in ("ci_lower_log_effect", "ci_upper_log_effect"):
        if payload.get(key) is None:
            continue
        ci = np.asarray(payload[key], dtype=np.float64)
        if ci.shape != term.edited_log_effect.shape:
            raise ValueError(f"Loaded {key} for term {term.name!r} does not match the model.")
    if term.levels is not None and list(payload.get("levels") or []) != term.levels:
        raise ValueError(f"Loaded levels for term {term.name!r} do not match the model.")
    if term.x is not None:
        loaded_x = np.asarray(payload.get("x"), dtype=np.float64)
        if loaded_x.shape != term.x.shape or not np.allclose(loaded_x, term.x):
            raise ValueError(f"Loaded x grid for term {term.name!r} does not match the model.")


def jsonable(value):
    # Shared conversion for session artifacts and HTTP JSON responses.
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return value
