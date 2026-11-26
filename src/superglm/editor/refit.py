"""Shared refit dispatch for editor model replacements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from superglm.distributional.api import SuperLSS
from superglm.editor.errors import EditorValueError
from superglm.editor.terms import resolve_refit_method
from superglm.model import fit_method
from superglm.model.fit_state import configured_lambda2, fitted_lambda2

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
    _record_held_smoothing(source_model, refit_model, resolved_method)
    return resolved_method


def holds_searched_smoothing(model) -> bool:
    """Whether an ``"auto"`` refit of ``model`` holds smoothing a search chose.

    True for a ``fit_ncv`` fit, whose refit is a ``fit``, and for a refit that held such
    smoothing; false where the refit replays the search (``fit_reml``) or nothing searched.
    """
    traits = fit_method.traits_of(model)
    if fit_method.FIT_METHODS[traits.refit_as].criterion is not None:
        return False
    return traits.criterion is not None or fit_method.smoothing_held_from(model) is not None


def _record_held_smoothing(source_model, refit_model, resolved_method: str) -> None:
    """Record which search chose the smoothing a refit holds, where the source's own refit holds it.

    A source fitted by a search its refit does not replay (``fit_ncv``, whose ``refit_as`` is
    ``fit``), or a refit that already held one, hands its smoothing to a ``fit``. When that
    refit declares the source's smoothing unchanged, its record names the search that chose it.
    A refit by another method than the source's own, or at smoothing the analyst gave, records
    nothing. A term whose lambda policy the refit dropped declares its one penalty under its own
    name, which the source keyed ``"<name>:wiggle"``.
    """
    if resolved_method != fit_method.traits_of(source_model).refit_as:
        return
    if fit_method.FIT_METHODS[resolved_method].criterion is not None:
        return
    chosen_by = (
        fit_method.method_of(source_model)
        if fit_method.search_criterion(source_model) is not None
        else fit_method.smoothing_held_from(source_model)
    )
    held, declared = fitted_lambda2(source_model), configured_lambda2(refit_model)
    if chosen_by is None or not isinstance(held, Mapping) or not isinstance(declared, Mapping):
        return
    if declared and all(
        held.get(name, held.get(f"{name}:wiggle")) == value for name, value in declared.items()
    ):
        fit_method.annotate(refit_model, smoothing_held_from=chosen_by)
