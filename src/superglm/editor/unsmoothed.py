"""The Unsmoothed line: a spline term's curve with its smoothing switched off, drawn over it.

The line is the model refitted with the term's smoothing parameter at 0, on
the same basis and knots, and every other term's smoothing parameter held
where the fit in force put it: a plain penalised fit at fixed smoothing
parameters, so nothing is selected again. With its penalty at 0 a P-spline is
B-spline regression (Eilers and Marx, "Flexible smoothing with B-splines and
penalties", Statistical Science 11(2), 1996), whose coefficients the rows
determine only when the design has full column rank: the Schoenberg-Whitney
condition, that some rows can be matched one to each basis function, inside
its support, as ``scipy.interpolate.make_lsq_spline`` documents. Across a gap
in the data it fails, the penalty was all that held the curve there, and the
line is refused rather than drawn; the refit's own rank decision on its
penalised system decides. A selection penalty is lifted from the term, as
Free levels lifts it, so the line is not shrunk.

The line is drawn under the same centring rule as the curve. Hand edits are
not part of it: it is a fit. An ordered term's levels fitted free, joined by
a line, are Free levels' (:mod:`superglm.editor.free_levels`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from superglm.editor.errors import EditorClientError, EditorTypeError, EditorValueError
from superglm.editor.free_levels import _lift_selection
from superglm.editor.refit import fit_refit_model
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import clone_with_replaced_features
from superglm.features.spline import _SplineBase
from superglm.inference._term_types import _safe_exp
from superglm.model.fit_state import fitted_lambda2

_NOT_SMOOTHED = "Unsmoothed is for a spline term: {term!r} has no smoothing to switch off."
_ORDERED = (
    "On an ordered term, Free levels draws its levels fitted free, joined by a line: choose "
    "Free levels above the chart."
)
_NO_DATA = (
    "The unsmoothed line refits the model, which needs its training data: open the editor with "
    "train_data, or from a model fitted with its data kept."
)
_NOT_FITTED = (
    "The model could not be fitted with the smoothing of {term!r} switched off, so there is no "
    "unsmoothed line to draw."
)
_UNDETERMINED = (
    "With its smoothing off, the curve of {term!r} is not determined: some of its basis "
    "functions have too few rows under them, as across a gap in the data. Fewer knots, or knots "
    "where the rows are, would determine it."
)
_REML_ONLY = (
    "The unsmoothed line holds every other term's smoothing where the fit put it, in a plain "
    "fit, and {terms} can only be fitted by REML, which chooses it again."
)
_UNCONVERGED = (
    "The refit stopped before it converged, so the line is where it stopped: raise the model's "
    "max_iter to settle it."
)
_SHRUNK = (
    "The model's selection penalty cannot be lifted from {term!r} alone, so it shrinks the line."
)


def unsmoothed_available(spec) -> bool:
    """Whether ``spec`` takes the Unsmoothed line: a spline term."""
    return isinstance(spec, _SplineBase)


def unsmoothed_job(session, name: str) -> Callable[[], dict[str, Any]]:
    """Read what ``name``'s unsmoothed line needs from the session; the call returned draws it.

    The call does the one fit and reads nothing more from the session, so a
    caller holding the editor's lock may release it first. Refusals are
    fixed sentences (:class:`EditorValueError`), here or from the call.
    """
    term = session._require_term(name)
    spec = session.model._specs[name]
    if isinstance(spec, OrderedCategorical):
        raise EditorTypeError(_ORDERED)
    if not unsmoothed_available(spec):
        raise EditorTypeError(_NOT_SMOOTHED.format(term=name))
    return _spline_job(session, name, term)


def _spline_job(session, name: str, term) -> Callable[[], dict[str, Any]]:
    """The refit of a spline term with its smoothing parameter 0 and every other one held."""
    source = session.model
    reml_only = _reml_only_terms(source)
    if reml_only:
        raise EditorValueError(_REML_ONLY.format(terms=", ".join(map(repr, reml_only))))
    try:
        X, y, sample_weight, offset = session._resolve_refit_data(None, None, None, None)
    except RuntimeError as exc:
        # No training data was given and the model kept none.
        raise EditorValueError(_NO_DATA) from exc
    if y is None:
        raise EditorValueError(_NO_DATA)
    refit = clone_with_replaced_features(source, {}, lambda2=unsmoothed_lambdas(source, name))
    shrunk = _lift_selection(refit, source, name)
    grid = np.asarray(term.x, dtype=np.float64)
    n_points, centering = session.n_points, session.centering

    def run() -> dict[str, Any]:
        try:
            fit_refit_model(
                source,
                refit,
                method="fit",
                X=X,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
            )
        except EditorClientError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise EditorValueError(_NOT_FITTED.format(term=name)) from exc
        if not _curve_determined(refit, name):
            raise EditorValueError(_UNDETERMINED.format(term=name))
        inference = refit.term_inference(
            name, with_se=False, n_points=n_points, centering=centering
        )
        x = np.asarray(inference.x, dtype=np.float64).ravel()
        values = np.asarray(inference.log_relativity, dtype=np.float64).ravel()
        if x.shape != grid.shape or not np.array_equal(x, grid):
            order = np.argsort(x)
            values = np.interp(grid, x[order], values[order])
        notes = []
        if not bool(getattr(refit.result, "converged", True)):
            notes.append(_UNCONVERGED)
        if shrunk:
            notes.append(_SHRUNK.format(term=name))
        return {
            "term": name,
            "x": grid.tolist(),
            "y": [float(value) for value in np.asarray(_safe_exp(values))],
            "note": " ".join(notes) or None,
        }

    return run


def unsmoothed_lambdas(model, name: str) -> dict[str, float]:
    """Every smoothing parameter where ``model``'s fit put it, and ``name``'s at 0.

    A fit holds one parameter for every penalised group, or one per penalty
    component, keyed ``<group>`` or ``<group>:<component>``. A key belongs to
    the group whose name it starts with, the longest such name deciding,
    since an interaction's group ``a:b`` starts like a component of ``a``.
    """
    fitted = fitted_lambda2(model)
    owners = {group.name: group.feature_name for group in model._groups}
    if not isinstance(fitted, dict):
        fitted = dict.fromkeys(owners, float(fitted))

    def owner(key: str):
        if key in owners:
            return owners[key]
        prefixes = [group for group in owners if key.startswith(f"{group}:")]
        return owners[max(prefixes, key=len)] if prefixes else None

    lambdas = {key: 0.0 if owner(key) == name else float(value) for key, value in fitted.items()}
    lambdas.update({group: 0.0 for group, feature in owners.items() if feature == name})
    return lambdas


def _curve_determined(model, name: str) -> bool:
    """Whether the refit's penalised system determines every coefficient of ``name``.

    A coefficient selection set to zero is determined; one the fit kept is
    when its coordinate is estimable on the fit's penalised Gram, the rank
    decision the fit itself made. A fit that kept no rank record cannot say,
    and is taken as determined.
    """
    rank = getattr(model.result, "rank_info", None)
    if rank is None:
        return True
    selected = np.asarray(rank.selected_columns, dtype=np.intp)
    if rank.augmented.rank >= selected.size:
        return True
    position = {int(column): i for i, column in enumerate(selected)}
    for group in model._groups:
        if group.feature_name != name:
            continue
        for column in range(group.start, group.end):
            if column not in position:
                continue
            contrast = np.zeros(selected.size)
            contrast[position[column]] = 1.0
            if not rank.augmented.is_estimable(contrast):
                return False
    return True


def _reml_only_terms(model) -> list[str]:
    """The terms a plain fit refuses: variance components, and a declared lambda policy."""
    terms = [(name, model._specs[name]) for name in model._feature_order] + [
        (name, model._interaction_specs[name]) for name in model._interaction_order
    ]
    return [
        str(name)
        for name, spec in terms
        if getattr(spec, "requires_reml", False)
        or getattr(spec, "_lambda_policy", None) is not None
    ]
