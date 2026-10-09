"""Knots of a spline term: their count and placement rule, or their positions.

A knot change is a structural change that waits for Refit like a shaped
range. It rebuilds the term from its declaration with new knot settings,
keeping its shaped ranges, constraint and penalty; an ordered term keeps its
order, grouping, special levels and reference. Positions travel in chart
coordinates: a numeric term's own values, or an ordered term's display
positions, where smooth level ``i`` sits at ``i`` and the spline's own axis
maps linearly between neighbouring levels.
"""

from __future__ import annotations

import copy
import math
import warnings
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.dm_builder import knot_geometry_weight, resolve_discrete_n_bins, should_discretize
from superglm.editor._types import EditableTerm
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorValueError
from superglm.features._spline_ranges import RangeError
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import (
    EDITOR_KNOTS_ATTRIBUTE,
    SHAPE_FROZEN_ATTRIBUTE,
    base_names_level,
    declared_spline,
    interaction_users,
    pristine_basis,
    rebuilt_ordered_spec,
    respaced_spline,
    source_spline,
    stated_knots,
)
from superglm.features.spline import (
    CardinalCRSpline,
    NaturalSpline,
    PSpline,
    _BSplineBase,
    _SplineBase,
)

STRATEGIES = ("uniform", "quantile", "quantile_rows", "quantile_tempered")
_RULE_TEXT = {
    "uniform": "even spacing",
    "quantile": "quantiles of values",
    "quantile_rows": "quantiles of rows",
    "quantile_tempered": "tempered quantiles",
}
# Operations that change an ordered term's levels, and so the axis its knots sit on.
LEVEL_OPERATIONS = frozenset({"collapse", "ungroup", "special", "on_curve"})
# The display step a knot snaps to on an ordered term's level axis.
_ORDERED_STEP = 0.1
# Unit roundoff of float64, u = 2**-53: the bound on the relative error of one rounding.
_UNIT_ROUNDOFF = 2.0**-53

_NOT_SPLINE = "Knots are for spline terms and ordered terms with a spline basis."
_INTERACTION = "A term used by an interaction keeps its knots."
_NO_AXIS = (
    "This term's levels are not drawn in the order of its axis, so its knots are set in code."
)
_NO_DECLARATION = (
    "This model does not keep the term's declaration; refit it with this version of superglm "
    "to move its knots."
)
_NO_RANGE = "Every value of this term's column is the same, so it has no range to place knots on."
_LEVELS_WAITING = (
    "A waiting change on {term!r} changes its levels; refit it before changing the knots."
)
_FORMS = "Give a knot count and placement rule, a list of positions, or reset."
_COUNT = "The knot count must be a whole number of at least 1."
_RULE = (
    "Choose how the knots are placed: even spacing, quantiles of values, quantiles of rows "
    "or tempered quantiles."
)
_ALPHA = "Tempered quantiles take an alpha from 0 to 1."
_ORDERED_MAX = "{term!r} has {levels} levels on its curve, so it takes at most {most} knots."
_POSITIONS = "Knot positions must be numbers inside the range of {term!r}."
_CROWDED = (
    "The knot at {x} sits closer than {gap} to the knot or end beside it on {term!r}; "
    "move it further from them."
)
_CROWDED_ORDERED = (
    "A knot sits closer than a tenth of a level to the knot or end beside it on {term!r}; "
    "move it further from them."
)
_EVEN_ONLY_NS = (
    '{term!r} is a natural spline (kind="ns"), whose penalty needs evenly spaced knots; '
    'change its count here, or declare it with kind="cr" or kind="ps" to place its knots '
    "freely."
)
_EVEN_ONLY_ORDER = (
    "The penalty order of {term!r} is above its degree, which needs evenly spaced knots; "
    "change its count here, or lower m in code to place its knots freely."
)
_SHAPED_EDGE = (
    "A knot falls too close to the edge of a shaped range or the end of the axis. "
    "Move it, or undo the range first."
)
_COLLAPSED = (
    "{rule} put several of the {count} knots of {term!r} on one value{where}, so the fit "
    "would fall back to even spacing. {remedy}"
)
_KNOTS_PAST_CURVE = (
    "That change leaves the curve of {term!r} running from {first} to {last}, and {count} of "
    "its knots at fixed positions {verb} outside it. Move or remove {them}, or place the knots "
    "by a rule, before this change."
)
_KNOTS_OVER_LEVELS = (
    "That change leaves {levels} levels on the curve of {term!r}, which take at most {most} "
    "knots, and it has {count} at fixed positions. Remove knots, or place them by a rule, "
    "before this change."
)
_NO_REFERENCE = (
    "The opened model does not keep the declaration of {term!r}, so there is nothing to reset to."
)
_WEIGHTS_DIMENSION = "sample_weight must be one-dimensional."
_WEIGHTS_LENGTH = "sample_weight must have length {rows}, got {got}."


def knots_unavailable_reason(model, name: str, term: EditableTerm) -> str | None:
    """Why ``name``'s knots cannot be changed in the editor, as one sentence, or None."""
    spec = model._specs[name]
    if source_spline(spec) is None:
        return _NOT_SPLINE
    if interaction_users(model, name):
        return _INTERACTION
    if isinstance(spec, OrderedCategorical) and _ordered_axis(spec, term) is None:
        return _NO_AXIS
    fitted = spec._basis_spline if isinstance(spec, OrderedCategorical) else spec
    boundary = fitted.fitted_boundary
    if boundary is not None and not boundary[1] > boundary[0]:
        return _NO_RANGE
    if declared_spline(model, name) is None:
        return _NO_DECLARATION
    return None


def knots_payload(session, name: str, term: EditableTerm) -> dict[str, Any]:
    """The Knots tool's state for one term, in chart coordinates (the browser contract)."""
    model = session.model
    reason = knots_unavailable_reason(model, name, term)
    if reason is not None:
        return {
            **dict.fromkeys(
                (
                    "positions",
                    "count",
                    "strategy",
                    "alpha",
                    "lo",
                    "hi",
                    "min_gap",
                    "basis",
                    "waiting_basis",
                )
            ),
            "available": False,
            "reason": reason,
            "from_editor": False,
            "max_count": None,
            "resettable": False,
            "even_only": None,
        }
    spec = model._specs[name]
    axis = _Axis.of(spec, term)
    fitted = spec._basis_spline if isinstance(spec, OrderedCategorical) else spec
    knots = np.asarray(fitted.fitted_base_knots, dtype=np.float64)
    lo, hi = (axis.to_chart(edge) for edge in fitted.fitted_boundary)
    return {
        "available": True,
        "reason": None,
        "positions": [float(v) for v in axis.to_chart(knots)],
        "count": int(knots.size),
        "strategy": _strategy(fitted),
        "alpha": float(fitted.knot_alpha),
        "from_editor": bool(getattr(source_spline(spec), EDITOR_KNOTS_ATTRIBUTE, False)),
        "lo": float(lo),
        "hi": float(hi),
        "min_gap": axis.min_gap,
        "max_count": axis.max_count,
        "resettable": _resettable(session, name),
        "even_only": _even_only_reason(name, _waiting_spline(session, name)),
        "basis": _basis_payload(fitted, axis, float(lo), float(hi)),
        "waiting_basis": _waiting_basis_payload(session, name, fitted, axis, float(lo), float(hi)),
    }


def _waiting_basis_payload(session, name: str, fitted, axis: _Axis, lo: float, hi: float):
    """How the browser rebuilds the basis waiting changes put in force, or None.

    None while it is built as the one in force, and while a waiting change
    moves an ordered term's levels, whose axis the knots in force no longer
    sit on.
    """
    waiting = [step for step in getattr(session, "pending", ()) if step.term == name]
    if not waiting or any(step.operation in LEVEL_OPERATIONS for step in waiting):
        return None
    spline = source_spline(waiting[-1].draft_spec)
    if spline is None:
        return None
    drawn = _basis_payload(spline, axis, lo, hi)
    return None if drawn == _basis_payload(fitted, axis, lo, hi) else drawn


def _basis_payload(spline: _SplineBase, axis: _Axis, lo: float, hi: float) -> dict[str, Any] | None:
    """How the browser rebuilds the term's B-spline basis to draw it; None for a cardinal spline.

    ``ends`` is "open" for a P-spline or B-spline: the boundary is widened by 0.001 of its range
    and the knots carry on past it at the first and last spacing. It is "clamped" for a cubic
    regression or natural spline: each end of the boundary is repeated ``degree + 1`` times (a
    natural spline also widens it by 1e-6 of its range, far below a pixel, which the browser
    leaves out). ``boundary`` is in chart coordinates. ``level_values`` are an ordered term's
    smooth levels on the spline's own axis, through which the browser maps the knots and the
    boundary; None on a numeric term, whose chart axis is the spline's.
    """
    if isinstance(spline, CardinalCRSpline):
        return None
    return {
        "degree": int(spline.degree),
        "ends": "open" if isinstance(spline, _BSplineBase) else "clamped",
        "boundary": [lo, hi],
        "level_values": None if axis.values is None else [float(v) for v in axis.values],
    }


def pending_knots(session, name: str) -> dict[str, Any] | None:
    """The knots a waiting change on ``name`` would put in force, or None.

    None also once a later waiting change moves the term's levels, whose axis
    the stored positions no longer match.
    """
    for step in reversed(getattr(session, "pending", ())):
        if step.term != name:
            continue
        if step.operation in LEVEL_OPERATIONS:
            return None
        if step.operation == "knots":
            meta = step.metadata
            # Read as TermKnots.from_editor will be after a Refit: a reset is not by hand.
            spline = source_spline(step.draft_spec)
            return {
                "positions": list(meta["chart_positions"]),
                "count": int(meta["count"]),
                "strategy": meta["strategy"],
                "alpha": meta["alpha"],
                "from_editor": bool(getattr(spline, EDITOR_KNOTS_ATTRIBUTE, False)),
            }
    return None


def knots_feature_spec(
    model,
    term: EditableTerm,
    params: dict[str, Any],
    *,
    X,
    sample_weight=None,
    draft_spec=None,
    reference_model=None,
    levels_waiting: bool = False,
    waiting_positions=(),
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``term`` with new knots, and the waiting change's metadata.

    ``params`` is one of ``{"count", "strategy"[, "alpha"]}``, ``{"positions"}``
    (chart coordinates) or ``{"reset": True}``, which takes the knots of
    ``reference_model``, the opened model. ``draft_spec`` is the term's spec as
    waiting changes leave it; ``levels_waiting`` says one of them changes its
    levels. ``X`` and ``sample_weight`` are the refit's data, on which the new
    knots are placed now, so a refusal comes before the Refit.
    ``waiting_positions`` are the chart positions of a waiting knot change on
    the term: with the knots in force, the knots it shows, which a list of
    positions may keep where they are.
    """
    name = term.name
    _require_not_interaction_parent(model, name, operation="change the knots")
    reason = knots_unavailable_reason(model, name, term)
    if reason is not None:
        raise EditorValueError(reason)
    if levels_waiting:
        raise EditorValueError(_LEVELS_WAITING.format(term=name))
    fitted = model._specs[name]
    spec = fitted if draft_spec is None else draft_spec
    ordered = isinstance(spec, OrderedCategorical)
    source = pristine_basis(spec) if ordered else _numeric_declaration(model, name, draft_spec)
    axis = _Axis.of(fitted, term)
    form = _form(params)
    if form == "reset":
        basis, marked = _reset_basis(reference_model, name, source)
    elif form == "rule":
        basis, marked = _rule_basis(name, spec, source, axis, params), True
    else:
        shown = {*_in_force_chart(fitted, axis), *map(float, waiting_positions)}
        basis = _positions_basis(name, fitted, source, axis, params["positions"], shown)
        marked = True
    if marked:
        setattr(basis, EDITOR_KNOTS_ATTRIBUTE, True)
    else:
        basis.__dict__.pop(EDITOR_KNOTS_ATTRIBUTE, None)
    replacement = _hosted(spec, basis, name, X) if ordered else basis
    placed, strategy = _placed_knots(model, name, replacement, X, sample_weight)
    if form == "rule" and params["strategy"] != "uniform" and strategy == "uniform":
        raise EditorValueError(
            _collapsed(model, name, spec, source, params, X, sample_weight, placed.size)
        )
    count = int(placed.size)
    alpha = float(basis.knot_alpha)
    if form == "rule":
        label = f"{count} knots, {_RULE_TEXT[strategy]}"
    elif form == "positions":
        label = "knots placed by hand"
    else:
        label = "knots as declared"
    label = f"{label} in {name}"
    return replacement, {
        "format": "superglm.editor.knots.v1",
        "term": name,
        "form": form,
        "count": count,
        "strategy": strategy,
        "alpha": alpha,
        "positions": [float(v) for v in placed],
        "chart_positions": [float(v) for v in axis.to_chart(placed)],
        "label": label,
        "message": f"The knots of {name} were changed in the editor and the full model was refit.",
    }


# -- The three forms --------------------------------------------------------------


def _form(params: dict[str, Any]) -> str:
    keys = set(params)
    if keys == {"reset"} and params["reset"] is True:
        return "reset"
    if keys == {"positions"}:
        return "positions"
    if keys in ({"count", "strategy"}, {"count", "strategy", "alpha"}):
        return "rule"
    raise EditorValueError(_FORMS)


def _rule_basis(name: str, spec, source: _SplineBase, axis: _Axis, params) -> _SplineBase:
    count, strategy = params["count"], params["strategy"]
    if not _is_integer(count) or count < 1:
        raise EditorValueError(_COUNT)
    if strategy not in STRATEGIES:
        raise EditorValueError(_RULE)
    alpha = params.get("alpha", source.knot_alpha)
    if not _is_real(alpha) or not 0.0 <= alpha <= 1.0:
        raise EditorValueError(_ALPHA)
    _require_count(name, axis, int(count))
    if strategy != "uniform":
        _require_uneven_allowed(name, source)
    return respaced_spline(
        source, n_knots=int(count), knot_strategy=strategy, knot_alpha=float(alpha)
    )


def _positions_basis(
    name: str, fitted, source: _SplineBase, axis: _Axis, positions, shown: set[float]
) -> _SplineBase:
    """``source`` with knots at ``positions`` (chart coordinates).

    Each knot must lie inside the term's range. A knot the change places,
    one the term does not show already (``shown``), keeps the grid step at
    its place (``_Axis.gap_between``) from the knots or ends beside it; the
    knots it keeps are left where they are, as close as a rule placed them.
    """
    inner = fitted._basis_spline if isinstance(fitted, OrderedCategorical) else fitted
    lo, hi = (float(axis.to_chart(edge)) for edge in inner.fitted_boundary)
    if not isinstance(positions, list | tuple) or not positions:
        raise EditorValueError(_COUNT)
    if not all(_is_real(v) and math.isfinite(v) for v in positions):
        raise EditorValueError(_POSITIONS.format(term=name))
    chart = np.sort(np.asarray(positions, dtype=np.float64))
    if not (chart[0] > lo and chart[-1] < hi):
        raise EditorValueError(_POSITIONS.format(term=name))
    _require_room(name, axis, chart, shown, lo, hi)
    _require_count(name, axis, int(chart.size))
    _require_uneven_allowed(name, source)
    return respaced_spline(source, knots=axis.to_axis(chart))


def _require_room(name: str, axis: _Axis, chart: NDArray, shown: set[float], lo, hi) -> None:
    """Refuse a knot the change places nearer a knot or end beside it than the step there."""
    fences = np.concatenate(([lo], chart, [hi]))
    for index in range(1, fences.size - 1):
        left, x, right = (float(v) for v in fences[index - 1 : index + 2])
        apart = left < x < right
        if x in shown and apart:
            continue
        gap = axis.gap_between(left, right) if right > left else axis.gap_between(lo, hi)
        # A knot one step from a neighbour reads short by at most five roundings of
        # u M, with M = max(|lo|, |hi|): one for each of the two values, two for
        # their difference (|difference| <= 2 M) and one for the step (step <= M).
        # The rounding sits in the values, not the step. Where it reaches the step,
        # float64 cannot tell a knot from its neighbour, so none is accepted.
        tight = gap - 5.0 * _UNIT_ROUNDOFF * max(abs(lo), abs(hi))
        if not apart or tight <= 0.0 or x - left < tight or right - x < tight:
            if axis.values is not None:
                raise EditorValueError(_CROWDED_ORDERED.format(term=name))
            raise EditorValueError(_CROWDED.format(x=f"{x:g}", gap=_gap_text(gap), term=name))


def _in_force_chart(fitted, axis: _Axis) -> list[float]:
    """The knots in force in chart coordinates, as the payload sends them."""
    inner = fitted._basis_spline if isinstance(fitted, OrderedCategorical) else fitted
    return [float(v) for v in axis.to_chart(np.asarray(inner.fitted_base_knots, dtype=np.float64))]


def _reset_basis(reference_model, name: str, source: _SplineBase) -> tuple[_SplineBase, bool]:
    """``source`` with the opened model's knot settings, and whether those were the editor's."""
    reference = None if reference_model is None else declared_spline(reference_model, name)
    if reference is None:
        raise EditorValueError(_NO_REFERENCE.format(term=name))
    stated = stated_knots(reference)
    basis = respaced_spline(
        source,
        knots=None if stated is None else stated,
        n_knots=reference.n_knots,
        knot_strategy=reference.knot_strategy,
        knot_alpha=reference.knot_alpha,
    )
    return basis, bool(getattr(reference, EDITOR_KNOTS_ATTRIBUTE, False))


# -- Checks -----------------------------------------------------------------------


def _collapsed(model, name: str, spec, source, params, X, sample_weight, count: int) -> str:
    """The refusal for a quantile rule that would fall back to even spacing.

    The library spaces the knots evenly when a quantile rule gives fewer
    distinct knots than asked, as it does where many rows share one value.
    The sentence names that value on a numeric term, and the most knots the
    rule places without falling back.
    """
    strategy, alpha = params["strategy"], params.get("alpha", source.knot_alpha)
    most = 0
    for fewer in range(count - 1, 0, -1):
        basis = respaced_spline(
            source, n_knots=fewer, knot_strategy=strategy, knot_alpha=float(alpha)
        )
        replacement = (
            _hosted(spec, basis, name, X) if isinstance(spec, OrderedCategorical) else basis
        )
        if _placed_knots(model, name, replacement, X, sample_weight)[1] != "uniform":
            most = fewer
            break
    remedy = f"Choose {most} knots or fewer, or another rule." if most else "Choose another rule."
    return _COLLAPSED.format(
        rule=_RULE_TEXT[strategy].capitalize(),
        count=count,
        term=name,
        where=_crowded_value(model, name, spec, X, sample_weight),
        remedy=remedy,
    )


def _crowded_value(model, name: str, spec, X, sample_weight) -> str:
    """ "(60% of its rows are at 18)" for a numeric term's most shared value; empty otherwise."""
    if isinstance(spec, OrderedCategorical):
        return ""
    x = np.asarray(as_eager_frame(X).column_array(name), dtype=np.float64).ravel()
    weights = knot_geometry_weight(sample_weight, model._weight_semantics)
    keep = np.isfinite(x) if weights is None else np.isfinite(x) & (np.asarray(weights) > 0)
    values, mass = np.unique(x[keep], return_counts=True)
    if weights is not None and np.any(np.asarray(weights)[keep] != 1.0):
        mass = np.bincount(
            np.searchsorted(values, x[keep]),
            weights=np.asarray(weights)[keep],
            minlength=values.size,
        )
    top = int(np.argmax(mass))
    return f" ({mass[top] / mass.sum():.0%} of its rows are at {values[top]:g})"


def _require_count(name: str, axis: _Axis, count: int) -> None:
    if axis.max_count is not None and count > axis.max_count:
        raise EditorValueError(
            _ORDERED_MAX.format(term=name, levels=axis.max_count + 1, most=axis.max_count)
        )


def _require_uneven_allowed(name: str, source: _SplineBase) -> None:
    reason = _even_only_reason(name, source)
    if reason is not None:
        raise EditorValueError(reason)


def _even_only_reason(name: str, source: _SplineBase) -> str | None:
    """Why ``source`` takes only evenly spaced knots, or None when it takes any.

    A natural spline keeps the standard difference penalty, and a P-spline
    whose penalty order exceeds its degree has no general difference penalty.
    """
    if isinstance(source, NaturalSpline):
        return _EVEN_ONLY_NS.format(term=name)
    if isinstance(source, PSpline) and max(source._m_orders) > source.degree:
        return _EVEN_ONLY_ORDER.format(term=name)
    return None


def _waiting_spline(session, name: str) -> _SplineBase:
    """The spline the next knot change is made on: the last waiting draft's, else declared.

    A waiting basis change can make the term a natural spline, whose knots
    are evenly spaced only.
    """
    waiting = next(
        (step.draft_spec for step in reversed(session.pending) if step.term == name), None
    )
    spline = None if waiting is None else source_spline(waiting)
    declared = declared_spline(session.model, name) if spline is None else spline
    if declared is None:  # pragma: no cover - knots_unavailable_reason refuses it first
        raise EditorValueError(_NO_DECLARATION)
    return declared


def stated_knots_refusal(name: str, replacement) -> str | None:
    """Why a level change's draft of an ordered term cannot keep its stated knots, or None.

    Knots at fixed positions stay where they are while a level change moves
    the curve's levels under them: the change is refused while more of them
    remain than the levels on the curve take, or any lies outside the curve's
    first and last levels, naming which. Knots placed by a rule are placed
    again on the new levels.
    """
    spline = source_spline(replacement) if isinstance(replacement, OrderedCategorical) else None
    stated = None if spline is None else spline._explicit_knots
    smooth = list(getattr(replacement, "_smooth_levels", ()))
    if stated is None or len(smooth) < 2:
        return None
    knots = np.asarray(stated, dtype=np.float64)
    if knots.size > len(smooth) - 1:
        return _KNOTS_OVER_LEVELS.format(
            levels=len(smooth), term=name, most=len(smooth) - 1, count=int(knots.size)
        )
    values = np.asarray([replacement._level_to_value[level] for level in smooth])
    first, last = smooth[int(np.argmin(values))], smooth[int(np.argmax(values))]
    outside = int(np.count_nonzero((knots <= values.min()) | (knots >= values.max())))
    if outside == 0:
        return None
    return _KNOTS_PAST_CURVE.format(
        term=name,
        first=first,
        last=last,
        count=outside,
        verb="lies" if outside == 1 else "lie",
        them="it" if outside == 1 else "them",
    )


def _resettable(session, name: str) -> bool:
    """Whether the knots in force, or waiting, differ from the opened model's."""
    current = declared_spline(session.model, name)
    waiting = next(
        (step.draft_spec for step in reversed(session.pending) if step.term == name), None
    )
    if waiting is not None:
        current = source_spline(waiting) if isinstance(waiting, OrderedCategorical) else waiting
    reference = declared_spline(session.reference_model, name)
    if current is None or reference is None:
        return False
    return _settings(current) != _settings(reference)


def _settings(spline: _SplineBase) -> tuple:
    stated = stated_knots(spline)
    if stated is not None:
        return ("stated", tuple(str(v) for v in stated))
    return (spline.knot_strategy, int(spline.n_knots), float(spline.knot_alpha))


# -- Placement --------------------------------------------------------------------


def _numeric_declaration(model, name: str, draft_spec) -> _SplineBase:
    """The spline a numeric term's knots are changed on: its draft, else its declaration."""
    if isinstance(draft_spec, _SplineBase):
        return draft_spec
    declared = declared_spline(model, name)
    if declared is None:  # pragma: no cover - knots_unavailable_reason refuses it first
        raise EditorValueError(_NO_DECLARATION)
    return declared


def _hosted(spec: OrderedCategorical, basis, name: str, X) -> OrderedCategorical:
    """A fresh ordered term around ``basis``, keeping order, specials, grouping and base."""
    frame = as_eager_frame(X)
    frame.require_columns((name,))
    return rebuilt_ordered_spec(
        spec,
        grouping=getattr(spec, "_grouping", None),
        base=spec.base,
        data=frame.column_array(name),
        basis=basis,
        level=base_names_level(spec),
    )


def _placed_knots(
    model, name: str, replacement, X, sample_weight, *, raw: bool = False
) -> tuple[NDArray, str]:
    """The interior knots ``replacement`` takes on the refit's data, and the rule that placed them.

    The knots are on the spline's own axis; the rule is ``"explicit"`` for
    stated knots, and ``"uniform"`` where a quantile rule fell back to it.

    Placing them is the fit's first step, so a placement the library refuses,
    such as a knot against a shaped range's edge, is refused here.
    """
    built = _placed_spline(model, name, replacement, X, sample_weight, raw=raw)
    return np.asarray(built.fitted_base_knots, dtype=np.float64), _strategy(built)


def placed_geometry(model, name: str, spline: _SplineBase, X, sample_weight):
    """The base knots and boundary the refit gives numeric ``spline`` on its data."""
    built = _placed_spline(model, name, spline, X, sample_weight, raw=False)
    return np.asarray(built.fitted_base_knots, dtype=np.float64), built.fitted_boundary


def _require_weights_shape(weights: NDArray | None, rows: int) -> None:
    """Refuse weights of the wrong shape in the fit's sentences, before a probe indexes them.

    The fit refuses them before it builds anything; a probe that indexes the
    weights by the column's rows would otherwise fail first, and name the
    waiting change rather than the weights.
    """
    if weights is None:
        return
    if weights.ndim != 1:
        raise EditorValueError(_WEIGHTS_DIMENSION)
    if len(weights) != rows:
        raise EditorValueError(_WEIGHTS_LENGTH.format(rows=rows, got=len(weights)))


def _placed_spline(model, name: str, replacement, X, sample_weight, *, raw: bool):
    """A copy of ``replacement``'s spline with its knots placed on the refit's data."""
    probe = copy.deepcopy(replacement)
    column = as_eager_frame(X).column_array(name)
    reporting = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    _require_weights_shape(reporting, len(column))
    # The fit's own rule for the weights that place knots.
    weights = knot_geometry_weight(reporting, model._weight_semantics)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(probe, OrderedCategorical):
                probe._build_with_geometry(
                    column, reporting_weight=reporting, geometry_weight=weights
                )
                built = probe._basis_spline
            else:
                x = np.asarray(column, dtype=np.float64).ravel()
                keep = np.isfinite(x)
                binned = should_discretize(probe, model._discrete)
                n_bins = resolve_discrete_n_bins(name, probe, model._n_bins) if binned else None
                probe._place_knots(x[keep], None if weights is None else weights[keep], n_bins)
                built = probe
    except RangeError as exc:
        if raw:
            raise
        raise EditorValueError(_SHAPED_EDGE) from exc
    return built


def probe_build(model, name: str, replacement, X, sample_weight) -> None:
    """Build ``replacement`` on the refit's data as the fit's design compile does, alone.

    A spline term, or an ordered term on a spline basis, is compiled as the
    one term of a design, by the fit's own builder with the model's binning,
    smoothing and weight settings: its knots placed, its shaped ranges
    certified, its penalty, shrinkage and smoothing policy built. So a change
    the fit would refuse is refused when it is staged. Other terms have
    nothing to place. A build the library refuses raises its own error.
    """
    if not (
        isinstance(replacement, _SplineBase)
        or (isinstance(replacement, OrderedCategorical) and source_spline(replacement) is not None)
    ):
        return
    from superglm._predictor_compiler import compile_predictor_design
    from superglm.model.fit_state import configured_lambda2
    from superglm.solvers.dispersion import PRIOR_WEIGHTS

    frame = as_eager_frame(X)
    frame.require_columns((name,))
    n = len(frame.column_array(name))
    weights = (
        np.ones(n, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    _require_weights_shape(weights, n)
    bindings = getattr(model, "_level_bindings", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile_predictor_design(
            frame,
            weights,
            geometry_weight=knot_geometry_weight(weights, model._weight_semantics),
            polynomial_weight=weights,
            categorical_reporting_weight=weights,
            ordered_reporting_weight=weights,
            specs={name: replacement},
            feature_order=[name],
            interaction_specs={},
            interaction_order=[],
            pending_interactions=[],
            model_discrete=model._discrete,
            n_bins_config=model._n_bins,
            lambda2=configured_lambda2(model),
            level_bindings=dict(bindings) if bindings else None,
            physical_rows=model._weight_semantics == PRIOR_WEIGHTS,
        )


def _strategy(spline: _SplineBase) -> str:
    """The rule that placed a built spline's knots, or ``"explicit"`` for stated ones."""
    if stated_knots(spline) is not None:
        return "explicit"
    frozen = getattr(spline, SHAPE_FROZEN_ATTRIBUTE, None)
    if frozen is not None:
        return frozen
    return str(getattr(spline, "_knot_strategy_actual", spline.knot_strategy))


# -- Chart coordinates ------------------------------------------------------------


class _Axis:
    """Chart coordinates of a term's knot axis: identity, or an ordered term's level positions."""

    def __init__(self, values: NDArray | None, max_count: int | None = None):
        self.values = values
        self.max_count = max_count

    @classmethod
    def of(cls, spec, term: EditableTerm) -> _Axis:
        if not isinstance(spec, OrderedCategorical):
            return cls(None)
        values = _ordered_axis(spec, term)
        # The spline sees one value per level on the curve, a group being one.
        return cls(values, None if values is None else len(spec._smooth_levels) - 1)

    def to_axis(self, chart) -> NDArray:
        chart = np.asarray(chart, dtype=np.float64)
        if self.values is None:
            return chart
        return np.interp(chart, np.arange(self.values.size, dtype=np.float64), self.values)

    def to_chart(self, value) -> NDArray:
        value = np.asarray(value, dtype=np.float64)
        if self.values is None:
            return value
        return np.interp(value, self.values, np.arange(self.values.size, dtype=np.float64))

    @property
    def min_gap(self) -> float | None:
        """An ordered term's snap step and least gap, a tenth of a level; None on a numeric term."""
        return None if self.values is None else _ORDERED_STEP

    def gap_between(self, left: float, right: float) -> float:
        """The snap step, and least gap, of a knot between the knots or ends ``left`` and ``right``.

        A tenth of a level on an ordered term. On a numeric term two significant
        figures of the space between them, so knots that a rule put close
        together where the data is dense move in steps that suit them, and
        knots far apart in steps that suit those.
        """
        if self.values is not None:
            return _ORDERED_STEP
        return decade_step(right - left)


def _ordered_axis(spec: OrderedCategorical, term: EditableTerm) -> NDArray | None:
    """The axis values of the levels an ordered term draws on its curve, in display order, or None.

    The chart draws a grouped term with its groups expanded, each original
    level at its own place, so the knots sit on that expanded axis: an
    original level at its own value, as the term was declared. None unless
    the chart shows those levels first, in the axis's order, with strictly
    increasing values: the mapping a knot position needs.
    """
    grouping = getattr(spec, "_grouping", None)
    if grouping is None:
        shown = list(spec._smooth_levels)
        values = [spec._level_to_value[level] for level in shown]
    else:
        original = getattr(spec, "_original_level_to_value", None) or {}
        shown = [
            member
            for group in spec._smooth_levels
            for member in grouping.group_to_originals.get(str(group), ())
        ]
        if any(str(member) not in original for member in shown):
            return None
        values = [original[str(member)] for member in shown]
    if term.levels is None or term.levels[: len(shown)] != [str(level) for level in shown]:
        return None
    axis = np.asarray(values, dtype=np.float64)
    if axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        return None
    return axis


def decade_step(width: float) -> float:
    """Two significant figures of ``width``: the power of ten a decade below its leading digit.

    The exponent from ``log10`` is checked against the correctly rounded
    powers of ten (``float("1e…")``), so the step is the same in the browser,
    which does the same, for a width that sits on a power of ten.
    """
    exponent = math.floor(math.log10(width))
    if float(f"1e{exponent + 1}") <= width:
        exponent += 1
    elif float(f"1e{exponent}") > width:
        exponent -= 1
    return float(f"1e{exponent - 1}")


def _gap_text(gap: float) -> str:
    return f"{gap:g}"


def _is_integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
