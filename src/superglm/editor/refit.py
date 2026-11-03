"""Shared refit dispatch for editor model replacements."""

from __future__ import annotations

from typing import Any

from superglm.distributional.api import SuperLSS
from superglm.editor.errors import EditorValueError
from superglm.editor.terms import resolve_refit_method

# The penalties a Refit was given explicitly ({"lambda1": ..., "lambda2": ...}),
# kept on the refitted model and carried by later Refits, so Run CV and Final
# fit keep the analyst's choice instead of the opened model's declaration.
EXPLICIT_PENALTY_ATTRIBUTE = "_editor_explicit_penalty"


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
    """Fit a refit model with the same fit/fit_reml mode resolution everywhere.

    A SuperLSS replays the fit its source was made with; ``fit_kwargs`` and
    ``method`` do not apply (``editor.lss.refit``).
    """
    if isinstance(refit_model, SuperLSS):
        refit_model._replay_fit(X, y, sample_weight, offset)
        return refit_model._fit_settings["method"]
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
        raise EditorValueError("method must be 'auto', 'fit', or 'fit_reml'.")
    return resolved_method
