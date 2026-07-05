"""Persistence helpers for editor sessions and edited model copies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from superglm.editor.io import (
    jsonable,
    record_from_payload,
    record_to_payload,
    term_to_payload,
    validate_loaded_term,
)


def save_model(session, path: str | Path) -> Path:
    """Write the edited model copy as a joblib artifact."""
    import joblib

    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(".joblib")
    target.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(session.to_model(), target)
    return target


def save_session(session, path: str | Path) -> None:
    """Write an auditable JSON edit artifact."""
    payload = {
        "format": "superglm.editor.v1",
        "n_points": session.n_points,
        "centering": session.centering,
        "terms": [term_to_payload(term) for term in session.terms.values()],
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
