"""Shared refit dispatch for editor model replacements."""

from __future__ import annotations

from typing import Any

from superglm.editor.terms import resolve_refit_method


def fit_refit_model(
    source_model,
    refit_model,
    *,
    method: str,
    X,
    y,
    sample_weight=None,
    offset=None,
    fit_kwargs: dict[str, Any] | None = None,
) -> str:
    """Fit a refit model with the same fit/fit_reml mode resolution everywhere."""
    resolved_method = resolve_refit_method(source_model, method)
    kwargs = dict(fit_kwargs or {})
    if resolved_method == "fit_reml":
        refit_model.fit_reml(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            **kwargs,
        )
    elif resolved_method == "fit":
        refit_model.fit(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            **kwargs,
        )
    else:
        raise ValueError("method must be 'auto', 'fit', or 'fit_reml'.")
    return resolved_method
