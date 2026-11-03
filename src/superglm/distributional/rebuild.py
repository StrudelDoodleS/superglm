"""Rebuild a SuperLSS with some terms replaced, and replay the fit that made it.

A term is named ``"<parameter>:<term>"``, the layout's own composite key. A
rebuilt model is unfitted: it carries no coefficients and no hand edit, only the
templates and the settings the source was fitted with.

Models are typed ``Any``: ``api`` imports this module, so it cannot import ``api``
(``tests/test_distributional_family_kernel_architecture.py`` forbids import cycles
even under ``TYPE_CHECKING``).
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from superglm.distributional.binding import _bind_predictor_template
from superglm.distributional.predictor import Predictor
from superglm.terms import normalize_terms
from superglm.types import FeatureSpec

# Arguments of fit and fit_reml that describe the data or the rows, not the estimator.
_NOT_SETTINGS = frozenset(
    {
        "self",
        "X",
        "y",
        "sample_weight",
        "offsets",
        "truncation",
        "censored",
        "phase_recorder",
    }
)


def fit_settings(method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """The estimator keywords of a ``fit`` or ``fit_reml`` call, from its ``locals()``.

    ``fit_reml(method=...)`` only accepts ``"efs"``, the default, so it is not recorded;
    ``"method"`` names the entry point instead. A mapping (``lambdas``) is recorded as
    a plain dict of its items and an array as a copy, so a later change to the
    caller's object cannot change the replay; any other value is recorded as given.
    """
    settings = {
        name: _recorded(value)
        for name, value in params.items()
        if name not in _NOT_SETTINGS and name != "method"
    }
    return {"method": method, **settings}


def _recorded(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return np.array(value)
    return value


def _split_key(key: str) -> tuple[str, str]:
    parameter, separator, term = key.partition(":")
    if not separator or not parameter or not term:
        raise KeyError(
            f"{key!r} is not a term key; expected '<parameter>:<term>', such as 'mean:age'"
        )
    return parameter, term


def _unfitted_spec(key: str, label: str, value: Any) -> FeatureSpec:
    """The unfitted feature spec behind a replacement: a spec, or a declaration such as ``s(...)``."""
    if isinstance(value, FeatureSpec):
        return copy.deepcopy(value)
    features = normalize_terms((value,)).features
    if len(features) != 1 or label not in features:
        raise ValueError(
            f"The replacement for {key!r} must be one term on {label!r}; got {list(features)}."
        )
    return copy.deepcopy(features[label])


def _replaced_templates(source: Any, replacements: Mapping[str, Any]) -> tuple[Predictor, ...]:
    by_parameter: dict[str, dict[str, FeatureSpec]] = {}
    for key, value in replacements.items():
        parameter, term = _split_key(key)
        template = next((p for p in source._predictors if p.name == parameter), None)
        if template is None:
            raise KeyError(
                f"{key!r} names no predictor; the model has {[p.name for p in source._predictors]}"
            )
        if term in template.interaction_specs:
            raise ValueError(f"Interaction terms cannot be replaced: {key}")
        if term not in template.features:
            raise KeyError(f"{key!r} names no term of the {parameter!r} predictor")
        by_parameter.setdefault(parameter, {})[term] = _unfitted_spec(key, term, value)
    templates = []
    for template in source.predictors:
        features = {**template.features, **by_parameter.get(template.name, {})}
        templates.append(dataclasses.replace(template, features=features))
    return tuple(templates)


def _filtered_settings(
    settings: Mapping[str, Any] | None, replacements: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The source's settings without the fixed smoothing strengths of the replaced terms."""
    if settings is None:
        return None
    prefixes = tuple(f"{key}#" for key in replacements)
    kept = dict(settings)
    if kept.get("lambdas") is not None:
        kept["lambdas"] = {k: v for k, v in kept["lambdas"].items() if not k.startswith(prefixes)}
    return kept


BOUNDED_FIT_NOT_REPLAYED = (
    "This model was fitted with truncated or censored rows, and a replay records no bounds, "
    "so it would fit the complete-data likelihood instead. Refit it with fit() or fit_reml() "
    "and the same truncation= and censored=."
)


def clone_with_replaced_terms(source: Any, replacements: Mapping[str, Any]) -> Any:
    """An unfitted copy of ``source`` with the named terms' specs replaced.

    A source fitted on truncated or censored rows is refused: its replay would
    drop the bounds and fit another likelihood.
    """
    state = None if source._model is None else source._model.fit_state
    if state is not None and (state.truncated_row_count or state.censored_row_count):
        raise ValueError(BOUNDED_FIT_NOT_REPLAYED)
    templates = _replaced_templates(source, replacements)
    family = source._family
    clone = type(source)(
        family,
        *(_bind_predictor_template(family, template) for template in templates),
        weight_semantics=source._weight_contract.semantics,
        discrete=source._discrete,
        n_bins=source._n_bins_config(),
        separation=source._separation,
        coefficient_curvature=source._coefficient_curvature,
    )
    clone._fit_settings = _filtered_settings(source._fit_settings, replacements)
    return clone


def replay_fit(model: Any, X: Any, y: Any, sample_weight: Any, offsets: Any) -> Any:
    """Fit ``model`` the way its source was fitted and return it."""
    settings = model._fit_settings
    if settings is None:
        raise RuntimeError(
            "This model has no recorded fit settings, so its fit cannot be replayed. "
            "Fit it with fit() or fit_reml() first."
        )
    kwargs = {k: v for k, v in settings.items() if k != "method"}
    fit = getattr(model, settings["method"])
    return fit(X, y, sample_weight, offsets, **kwargs)
