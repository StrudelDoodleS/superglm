"""The Unsmoothed line: a term's curve with its smoothing switched off, drawn over its curve.

An ordered term with a spline basis draws each level's free estimate: the
model refitted with the term as a plain categorical, every other term kept,
the factor fitted unconstrained that practice sets beside a smoothed one
(Anderson et al., *A Practitioner's Guide to Generalized Linear Models*, CAS
2007, sections 2.27-2.33). It is the very fit Free levels makes
(:mod:`superglm.editor.free_levels`), and the editor shares it between the two.

A numeric spline draws the model refitted with the term's smoothing parameter
at 0, on the same basis and knots, and every other term's smoothing parameter
held where the fit in force put it: a plain penalised fit at fixed smoothing
parameters, so nothing is selected again. With its penalty at 0 a P-spline is
B-spline regression (Eilers and Marx, "Flexible smoothing with B-splines and
penalties", Statistical Science 11(2), 1996), whose coefficients the rows
determine only when the design has full column rank: the Schoenberg-Whitney
condition, that some rows can be matched one to each basis function, inside
its support, as ``scipy.interpolate.make_lsq_spline`` documents. Across a gap
in the data it fails, the penalty was all that held the curve there, and the
line is refused rather than drawn; the refit's own rank decision on its
penalised system decides. A selection penalty is lifted from the term in both
cases, as Free levels lifts it, so the line is not shrunk.

The line is drawn on the chart's reference, as the opened model's line is: a
numeric spline's under the same centring rule as its curve, an ordered term's
relative to the reference level. A level the free fit cannot estimate (no rows
of positive weight, every response at the family's bound, or rows another
term covers exactly) is a gap in the line, and a note names it. Hand edits are
not part of it: it is a fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from superglm.editor.errors import EditorClientError, EditorTypeError, EditorValueError
from superglm.editor.free_levels import (
    FreeFit,
    FreeFitWording,
    _determined,
    _lift_selection,
    _values,
    prepare_free_fit,
)
from superglm.editor.refit import fit_refit_model
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import clone_with_replaced_features, special_labels
from superglm.features.spline import _SplineBase
from superglm.inference._term_types import _safe_exp
from superglm.model.fit_state import fitted_lambda2

FREE = "free"
SPLINE = "spline"

_NOT_SMOOTHED = (
    "Unsmoothed is for a spline term or an ordered term with a spline basis: {term!r} has no "
    "smoothing to switch off."
)
_NO_DATA = (
    "The unsmoothed line refits the model, which needs its training data: open the editor with "
    "train_data, or from a model fitted with its data kept."
)
_ONE_LEVEL = (
    "The unsmoothed line of {term!r} needs rows of positive weight in at least two of its "
    "levels, and the data the refit reads has them in {count}."
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
_GAP_NO_ROWS = (
    "The line skips {levels}: the data the refit reads has no rows of positive weight for {whom}."
)
_GAP_SEPARATED = (
    "The line skips {levels}: every response on {whose} rows is {value}, so {whose} free value "
    "has no finite estimate."
)
_GAP_ALIASED = (
    "The line skips {levels}: the model's other terms cover the same rows, so the data cannot "
    "separate {whose} value from theirs."
)
_UNCONVERGED = (
    "The refit stopped before it converged, so the line is where it stopped: raise the model's "
    "max_iter to settle it."
)
_SHRUNK = (
    "The model's selection penalty cannot be lifted from {term!r} alone, so it shrinks the line."
)

_WORDING = FreeFitWording(
    operation="draw the unsmoothed line",
    no_data=_NO_DATA,
    one_level=_ONE_LEVEL,
    not_fitted=_NOT_FITTED,
)


def unsmoothed_kind(spec) -> str | None:
    """``"free"`` for an ordered term with a spline basis, ``"spline"`` for a spline term, else None."""
    if isinstance(spec, OrderedCategorical):
        return FREE if spec.basis_kind == "spline" else None
    return SPLINE if isinstance(spec, _SplineBase) else None


@dataclass(frozen=True)
class Unsmoothed:
    """One term's unsmoothed line, and the free fit it was drawn from, which Free levels can share."""

    payload: dict[str, Any]
    free_fit: FreeFit | None


@dataclass(frozen=True)
class _Levels:
    """An ordered term's levels on its curve as the chart draws them, read with the session.

    ``levels`` are on the curve in axis order; ``group`` takes each to its
    level or group in the fit in force; ``declared`` are those levels and
    groups in the term's order; ``curve`` is the fitted curve in the model's
    own centring at each of them, on the log scale, and ``offset`` what the
    chart's centring adds to it; ``reference`` is the chart's reference.
    """

    levels: list[str]
    group: dict[str, str]
    declared: list[str]
    curve: dict[str, float]
    offset: dict[str, float]
    reference: str


def unsmoothed_job(session, name: str, free_fit: FreeFit | None = None) -> Callable[[], Unsmoothed]:
    """Read what ``name``'s unsmoothed line needs from the session; the call returned draws it.

    The call does the one fit, or none for an ordered term whose free fit is
    given, and reads nothing more from the session, so a caller holding the
    editor's lock may release it first. Refusals are fixed sentences
    (:class:`EditorValueError`), here or from the call.
    """
    term = session._require_term(name)
    spec = session.model._specs[name]
    kind = unsmoothed_kind(spec)
    if kind is None:
        raise EditorTypeError(_NOT_SMOOTHED.format(term=name))
    if kind == SPLINE:
        return _spline_job(session, name, term)
    shown = _shown_levels(term, spec)
    if free_fit is not None:
        return lambda: Unsmoothed(free_line(name, shown, free_fit), free_fit)
    fit = prepare_free_fit(session, name, _WORDING)

    def run() -> Unsmoothed:
        found = fit()
        return Unsmoothed(free_line(name, shown, found), found)

    return run


def free_line_payload(session, name: str, fit: FreeFit) -> dict[str, Any]:
    """``name``'s unsmoothed line from a free fit already made, as Free levels makes it."""
    term = session._require_term(name)
    return free_line(name, _shown_levels(term, session.model._specs[name]), fit)


def _shown_levels(term, spec: OrderedCategorical) -> _Levels:
    specials = special_labels(spec)
    grouping = getattr(spec, "_grouping", None)
    levels = [str(level) for level in term.levels if str(level) not in specials]
    group = {
        level: level if grouping is None else str(grouping.original_to_group.get(level, level))
        for level in levels
    }
    shown = np.asarray(term.original_log_effect, dtype=np.float64)
    native = np.asarray(term.metadata.get("native_original_log_effect", shown), dtype=np.float64)
    curve: dict[str, float] = {}
    offset: dict[str, float] = {}
    for level, value, drawn in zip(term.levels, native, shown, strict=True):
        if str(level) in group and group[str(level)] not in curve:
            curve[group[str(level)]] = float(value)
            offset[group[str(level)]] = float(drawn - value)
    return _Levels(
        levels=levels,
        group=group,
        declared=[str(level) for level in spec._ordered_levels],
        curve=curve,
        offset=offset,
        reference=str(spec._base_level),
    )


def free_line(name: str, shown: _Levels, fit: FreeFit) -> dict[str, Any]:
    """The line through each level's free estimate, relative to the chart's reference.

    Each level is drawn at ``f(level) - f(anchor) + c(anchor)``, with ``f``
    the free fit's log-relativities and ``c`` the chart's curve. The anchor is
    the chart's reference, where the model's own centring puts the curve at
    zero exactly (its evaluation there is round-off) and the free fit's is
    zero, so the line is the free fit's own relativities, moved only by what
    the chart's centring adds. Where the free fit has no value at the
    reference, its own reference, else its first level, anchors the line on
    the curve.
    """
    free_model = fit.model
    free_spec = free_model._specs[name]
    pinned = {str(level) for level in getattr(free_spec, "_pinned_levels", ())}
    at_bound = {label: bound for bound, labels in fit.separated.items() for label in labels}
    estimated = {str(level) for level in free_spec._levels} - pinned - set(at_bound)
    # Whether selection kept the term, from the fit's own record, so the
    # comparison's covariance is not formed for the line.
    rank = getattr(free_model.result, "rank_info", None)
    kept = set() if rank is None else set(rank.selected_group_names)
    active = any(g.name in kept for g in free_model._groups if g.feature_name == name)
    labels, aliased = _determined(
        free_model, name, [label for label in shown.declared if label in estimated], active=active
    )
    values = dict(zip(labels, _values(free_model, name, labels).tolist(), strict=True))
    anchor = next(
        (
            label
            for label in (shown.reference, str(free_spec._base_level), *labels)
            if label in values
        ),
        None,
    )
    shift = 0.0
    if anchor is not None:
        on_curve = 0.0 if anchor == shown.reference else shown.curve.get(anchor, 0.0)
        shift = shown.offset.get(anchor, 0.0) + on_curve - values[anchor]
    y: list[float | None] = []
    skipped: dict[str, list[str]] = {}
    for level in shown.levels:
        group = shown.group[level]
        if group in values:
            y.append(float(_safe_exp(values[group] + shift)))
            continue
        y.append(None)
        if group in at_bound:
            reason = f"bound:{at_bound[group]}"
        else:
            reason = "aliased" if group in aliased else "no rows"
        skipped.setdefault(reason, []).append(level)
    notes = [_gap_note(reason, levels) for reason, levels in skipped.items()]
    if not bool(getattr(free_model.result, "converged", True)):
        notes.append(_UNCONVERGED)
    if fit.shrunk:
        notes.append(_SHRUNK.format(term=name))
    return {
        "term": name,
        "kind": FREE,
        "levels": shown.levels,
        "x": None,
        "y": y,
        "gaps": [level for levels in skipped.values() for level in levels],
        "note": " ".join(notes) or None,
    }


def _gap_note(reason: str, levels: list[str]) -> str:
    one = len(levels) == 1
    named = ", ".join(levels)
    if reason == "no rows":
        return _GAP_NO_ROWS.format(levels=named, whom="it" if one else "them")
    whose = "its" if one else "their"
    if reason == "aliased":
        return _GAP_ALIASED.format(levels=named, whose=whose)
    value = "0" if reason == "bound:zero" else "1"
    return _GAP_SEPARATED.format(levels=named, whose=whose, value=value)


def _spline_job(session, name: str, term) -> Callable[[], Unsmoothed]:
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

    def run() -> Unsmoothed:
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
        payload = {
            "term": name,
            "kind": SPLINE,
            "levels": None,
            "x": grid.tolist(),
            "y": [float(value) for value in _safe_exp(values)],
            "gaps": [],
            "note": " ".join(notes) or None,
        }
        return Unsmoothed(payload, None)

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
