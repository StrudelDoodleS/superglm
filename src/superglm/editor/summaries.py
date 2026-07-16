"""Summary payloads for the editor side panel."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm.editor.apply import materialize_edit_request
from superglm.editor.terms import native_log_effect_values

_UNSET = object()


def summary_payload(
    widget,
    source: str,
    *,
    model_override=None,
    offset_terms_override: list[str] | None = None,
    offset_labels_override: list[dict[str, Any]] | None = None,
    collapse_info_override: Any = _UNSET,
) -> dict[str, Any]:
    # The editor has one working lane: the in-force editable model. The
    # immutable original remains available as a reference payload for plots,
    # deltas, and explicit audit calls.
    if source == "selected":
        source = "in_force"
    if source in {"edited", "collapse"}:
        source = "in_force"
    source = source if source in {"original", "in_force", "refit"} else "in_force"
    if source == "original":
        model = widget.session.reference_model if model_override is None else model_override
        label = "Original"
        offset_terms: list[str] = []
        offset_labels: list[dict[str, Any]] = []
        collapse_info = None
    elif source == "in_force":
        model = _in_force_summary_model(widget.session, model_override=model_override)
        label = "In-force edit model"
        offset_terms = []
        offset_labels = []
        collapse_info = (
            getattr(widget, "_in_force_info", None)
            if collapse_info_override is _UNSET
            else collapse_info_override
        )
    elif source == "refit":
        model = widget._offset_refit_model if model_override is None else model_override
        label = "Fixed-offset refit"
        offset_terms = (
            list(widget._offset_refit_terms)
            if offset_terms_override is None
            else list(offset_terms_override)
        )
        offset_labels = (
            list(widget._offset_refit_labels)
            if offset_labels_override is None
            else list(offset_labels_override)
        )
        collapse_info = None
        if model is None:
            return {
                "available": False,
                "source": "refit",
                "label": label,
                "error": "No fixed-offset refit has been run for the current edits.",
            }
    else:
        model = widget._collapsed_refit_model
        label = "Collapsed-level refit"
        offset_terms = []
        offset_labels = []
        collapse_info = widget._collapsed_refit_info
        if model is None:
            return {
                "available": False,
                "source": "collapse",
                "label": label,
                "error": "No collapsed-level refit has been run for the current selection.",
            }

    if model is None or getattr(model, "_result", None) is None:
        return {
            "available": False,
            "source": source,
            "label": label,
            "error": "No fitted model is available.",
        }

    summary = model.summary()
    compact = _compact_summary_payload(summary, source, offset_terms, offset_labels, model=model)
    return {
        "available": True,
        "source": source,
        "label": label,
        "html": summary._repr_html_(),
        "compact": compact,
        "offset_terms": offset_terms,
        "offset_labels": offset_labels,
        "collapse": collapse_info,
        "note": _summary_note(source),
    }


def _in_force_summary_model(session, *, model_override=None):
    if model_override is not None:
        return model_override
    if not session.edited_terms():
        return session.model
    request = session.capture_materialization_request()
    assert request is not None
    return materialize_edit_request(request)


def offset_label_payload(session, terms: list[str]) -> list[dict[str, Any]]:
    # Edited offset terms are absent from the refit coefficient table, so expose
    # their fixed factors separately for audit/display.
    labels: list[dict[str, Any]] = []
    for name in terms:
        term = session.terms[name]
        values, omitted = _offset_values(term)
        labels.append(
            {
                "term": name,
                "kind": term.kind,
                "scale": "log edited relativity",
                "values": values,
                "omitted": omitted,
            }
        )
    return labels


def _offset_values(term) -> tuple[list[dict[str, Any]], int]:
    log_effect = native_log_effect_values(term)
    if term.levels is not None:
        return [
            {
                "label": str(label),
                "log_offset": float(value),
                "relativity": float(np.exp(value)),
            }
            for label, value in zip(term.levels, log_effect, strict=False)
        ], 0

    if term.x is not None:
        x = np.asarray(term.x, dtype=np.float64).ravel()
        if x.size > 80:
            idx = np.unique(np.linspace(0, x.size - 1, 80, dtype=np.intp))
        else:
            idx = np.arange(x.size, dtype=np.intp)
        return [
            {
                "x": float(x[i]),
                "log_offset": float(log_effect[i]),
                "relativity": float(np.exp(log_effect[i])),
            }
            for i in idx
        ], int(x.size - idx.size)

    return [
        {
            "index": int(i),
            "log_offset": float(value),
            "relativity": float(np.exp(value)),
        }
        for i, value in enumerate(log_effect)
    ], 0


def _compact_summary_payload(
    summary,
    source: str,
    offset_terms: list[str],
    offset_labels: list[dict[str, Any]],
    *,
    model=None,
) -> dict[str, Any]:
    # The browser renders this typed payload instead of scraping the notebook
    # HTML. The raw HTML remains available behind the "Full summary" disclosure.
    info = getattr(summary, "_info", {})
    rows = [_compact_summary_row(row) for row in getattr(summary, "_coef_rows", [])]
    if model is not None:
        rows = _with_reference_rows(rows, model)
    return {
        "source": source,
        "model": {
            "family": _compact_scalar(info.get("family")),
            "link": _compact_scalar(info.get("link")),
            "method": _display_method(info.get("method", "ML")),
            "penalty": _compact_scalar(info.get("penalty")),
            "n_obs": _compact_scalar(info.get("n_obs")),
            "effective_df": _compact_scalar(info.get("effective_df")),
            "deviance": _compact_scalar(info.get("deviance")),
            "aic": _compact_scalar(info.get("aic")),
            "bic": _compact_scalar(info.get("bic")),
            "log_likelihood": _compact_scalar(info.get("log_likelihood")),
            "tweedie_p": _compact_scalar(info.get("tweedie_p")),
            "tweedie_p_ci": _compact_profile_ci(info.get("tweedie_p_ci")),
            "tweedie_p_ci_status": _compact_scalar(info.get("tweedie_p_ci_status")),
            "tweedie_p_method": _compact_scalar(info.get("tweedie_p_method")),
            "nb_theta": _compact_scalar(info.get("nb_theta")),
        },
        "rows": rows,
        "offset_terms": offset_terms,
        "offset_labels": offset_labels,
    }


def _with_reference_rows(rows: list[dict[str, Any]], model) -> list[dict[str, Any]]:
    existing = {row["name"] for row in rows}
    additions: dict[str, list[dict[str, Any]]] = {}
    for term, spec in getattr(model, "_specs", {}).items():
        base_level = str(getattr(spec, "_base_level", "") or "")
        if not base_level:
            continue
        row_name = f"{term}[{base_level}]"
        if row_name in existing:
            continue
        additions.setdefault(str(term), []).append(_compact_reference_row(str(term), base_level))

    if not additions:
        return rows

    inserted: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        group = str(row.get("group") or "")
        if group in additions and group not in inserted:
            out.extend(additions[group])
            inserted.add(group)
        out.append(row)
    for group, group_rows in additions.items():
        if group not in inserted:
            out.extend(group_rows)
    return out


def _compact_reference_row(term: str, level: str) -> dict[str, Any]:
    return {
        "name": f"{term}[{level}]",
        "group": term,
        "kind": "reference",
        "coef": 0.0,
        "se": None,
        "se_label": "ref",
        "stat": None,
        "stat_label": "",
        "p_value": None,
        "sig_code": "",
        "sig_class": "sig-reference",
        "quasi_separated": False,
        "active": True,
        "n_params": 0,
        "ref_df": None,
        "edf": None,
    }


def _compact_summary_row(row) -> dict[str, Any]:
    # Spline rows are group-level Wald tests, not coefficient rows. They get a
    # p-value/significance class but no coefficient SE cell.
    p_value = _finite_float(row.wald_p if row.is_spline else row.p)
    stat = _finite_float(row.wald_chi2 if row.is_spline else row.z)
    return {
        "name": str(row.name),
        "group": str(row.group or ""),
        "kind": "spline" if row.is_spline else "coef",
        "coef": _finite_float(row.coef),
        "se": None if row.is_spline else _finite_float(row.se),
        "se_label": "curve" if row.is_spline else "",
        "stat": stat,
        "stat_label": "chi2" if row.is_spline else ("z" if stat is not None else ""),
        "p_value": p_value,
        "sig_code": _summary_sig_code(p_value, bool(row.quasi_separated)),
        "sig_class": _summary_sig_class(p_value, bool(row.quasi_separated)),
        "quasi_separated": bool(row.quasi_separated),
        "active": bool(row.active),
        "n_params": int(row.n_params or 0),
        "ref_df": _finite_float(row.ref_df),
        "edf": _row_edf(row),
    }


def _compact_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, int | str | bool):
        return value
    return str(value)


def _compact_profile_ci(value: Any) -> list[Any] | None:
    if not isinstance(value, tuple | list) or len(value) < 2:
        return None
    return [_compact_scalar(value[0]), _compact_scalar(value[1])]


def _display_method(value: Any) -> str:
    method = str(_compact_scalar(value) or "")
    return "MLE" if method == "ML" else method


def _row_edf(row) -> float | None:
    edf = _finite_float(row.edf)
    if edf is not None:
        return edf
    if str(row.name) == "Intercept":
        return 1.0
    return None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _summary_sig_code(p_value: float | None, quasi_separated: bool) -> str:
    if quasi_separated:
        return "QS"
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


def _summary_sig_class(p_value: float | None, quasi_separated: bool) -> str:
    if quasi_separated:
        return "sig-qs"
    if p_value is None:
        return "sig-unknown"
    if p_value < 0.001:
        return "sig-strong"
    if p_value < 0.01:
        return "sig-medium"
    if p_value < 0.05:
        return "sig-standard"
    if p_value < 0.1:
        return "sig-weak"
    return "sig-none"


def _summary_note(source: str) -> str:
    if source == "in_force":
        return (
            "Current editable model. Manual coefficient edits make inference stale; "
            "structural refits refresh inference for the edited feature definition."
        )
    if source == "edited":
        return "Edited-copy inference is stale for manually edited coefficients."
    if source == "refit":
        return (
            "Edited terms are fixed as offset factors on the link scale; "
            "intervals are conditional for the remaining fitted terms."
        )
    if source == "collapse":
        return (
            "Selected categorical levels were collapsed in the feature definition; "
            "the full model was refit, so summary inference is for the collapsed model."
        )
    return "Original fitted model summary."
