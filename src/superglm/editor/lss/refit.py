"""Refit a SuperLSS with staged term changes.

A staged change is a draft: an unfitted spec for one term of one predictor.
Refit puts the drafts into an unfitted copy of the model and replays the fit the
model came from (``SuperLSS._replay_fit``), so the settings that made it
(method, outer iteration, smoothing controls) apply to the refit too.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any

from superglm._frame import as_eager_frame
from superglm.distributional.api import SuperLSS
from superglm.distributional.layout import _qualify_local_component
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.weights import LikelihoodWeightError, resolve_likelihood_weights
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.availability import WEIGHTS_DIFFER, refit_reason
from superglm.editor.lss.view import LSSModelView, split_key
from superglm.editor.refit import fit_refit_model

_REPLAYS = (
    "Refit replays the fit this model was made with ({method}), so {given} cannot be changed "
    "here. Fit the model again in code to change {them}."
)


def _require_replayable(source: SuperLSS, method: str, options: Mapping[str, Any]) -> None:
    """Refuse an option a replay cannot take, rather than ignore it."""
    own = source._fit_settings["method"]
    given = [name for name, value in options.items() if value is not ...]
    if method not in {"auto", own}:
        given.insert(0, "method")
    if given:
        raise EditorValueError(
            _REPLAYS.format(
                method=own, given=", ".join(given), them="it" if len(given) == 1 else "them"
            )
        )


def lss_refit_with_drafts(
    view: LSSModelView,
    drafts: Mapping[str, Any],
    *,
    X,
    y,
    sample_weight,
    offsets,
    method: str = "auto",
    **options: Any,
) -> SuperLSS:
    """A fitted copy of the model with each term of ``drafts`` (keyed ``parameter:term``) replaced.

    The copy replays the source's fit on ``X``, ``y``, ``sample_weight`` and
    ``offsets`` (a dict by parameter). ``method`` and ``options`` (``lambda1``,
    ``lambda2`` and fit keywords) may only repeat what the source was fitted with;
    ``...`` stands for an option not given. A model that records no fit settings,
    or rows without the offsets or weights it was fitted with, are refused. The
    source is not changed.
    """
    reason = refit_reason(view, sample_weight, offsets, y)
    if reason is not None:
        raise EditorValueError(reason)
    _require_replayable(view.lss, method, options)
    refit = view.lss._clone_with_replaced_terms(drafts)
    fit_refit_model(
        view.lss, refit, method=method, X=X, y=y, sample_weight=sample_weight, offset=offsets
    )
    return refit


FIXED_STRENGTHS_COVER_NO_NEW_PENALTY = (
    "This model was fitted with fixed smoothing strengths (fit with lambdas=), and this change "
    "to {term!r} adds or removes a penalty those strengths do not cover, so Refit could not "
    "replay the fit. Refit the model with fit_reml() to make this change."
)


def _refuse_changed_penalties(lss: SuperLSS, key: str, compiled: Any) -> None:
    """Refuse a draft whose penalties differ from the term's when the fit fixed every strength.

    ``fit`` estimates no strength, so its replay needs one for each penalty
    the draft has, and only the replaced term's own carry over.
    """
    settings = lss._fit_settings
    if settings is None or settings.get("method") != "fit":
        return
    prefix = f"{key}#"
    fitted = {
        name for name in lss._require_fitted().layout.penalty_names if name.startswith(prefix)
    }
    draft = {
        _qualify_local_component(component, predictor=compiled).name
        for component in compiled.penalties
    }
    if draft != fitted:
        raise EditorValueError(FIXED_STRENGTHS_COVER_NO_NEW_PENALTY.format(term=key))


def lss_probe_build(view: LSSModelView, key: str, replacement, X, sample_weight) -> None:
    """Build ``replacement`` for ``key`` on the refit's rows, as the fit's own compile does.

    The draft is the one term of a one-term predictor of the term's parameter,
    compiled by ``compile_predictors`` with the model's weight contract and
    binning. A spline's knots are placed and its shaped ranges certified, so a
    draft the fit would refuse is refused when it is staged. A build the library
    refuses raises its own error.
    """
    parameter, column = split_key(key)
    lss = view.lss
    frame = as_eager_frame(X)
    frame.require_columns((column,))
    try:
        resolved = resolve_likelihood_weights(
            sample_weight, n_observations=len(frame), contract=lss._weight_contract
        )
    except LikelihoodWeightError as exc:
        # No refit could run on these weights, so the change is refused when it is staged.
        raise EditorValueError(WEIGHTS_DIFFER) from exc
    if len(resolved.input_positions) < len(frame):
        frame = as_eager_frame(frame.take_rows(resolved.input_positions))
    template = next(p for p in lss._predictors if p.name == parameter)
    one_term = dataclasses.replace(
        template,
        features={column: replacement},
        interactions=(),
        interaction_specs={},
        interaction_order=(),
    )
    spec = next(p for p in lss.family.parameters if p.name == parameter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (compiled,) = compile_predictors(
            frame,
            resolved,
            (spec,),
            (one_term,),
            model_discrete=lss._discrete,
            n_bins_config=lss._n_bins_config(),
        )
    _refuse_changed_penalties(lss, key, compiled)
