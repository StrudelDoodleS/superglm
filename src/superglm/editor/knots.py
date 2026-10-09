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
from superglm.dm_builder import resolve_discrete_n_bins, should_discretize
from superglm.editor._types import EditableTerm
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorValueError
from superglm.features._spline_ranges import RangeError
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.rebuild import (
    EDITOR_KNOTS_ATTRIBUTE,
    base_names_level,
    declared_spline,
    interaction_users,
    pristine_basis,
    rebuilt_ordered_spec,
    respaced_spline,
    source_spline,
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

_NOT_SPLINE = "Knots are for spline terms and ordered terms with a spline basis."
_INTERACTION = "A term used by an interaction keeps its knots."
_NO_AXIS = (
    "This term's levels are not drawn in the order of its axis, so its knots are set in code."
)
_NO_DECLARATION = (
    "This model does not keep the term's declaration; refit it with this version of superglm "
    "to move its knots."
)
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
_POSITIONS = (
    "Knot positions must be numbers inside the range of {term!r}, at least {gap} apart "
    "and at least {gap} from its ends."
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
_NO_REFERENCE = (
    "The opened model does not keep the declaration of {term!r}, so there is nothing to reset to."
)


def knots_unavailable_reason(model, name: str, term: EditableTerm) -> str | None:
    """Why ``name``'s knots cannot be changed in the editor, as one sentence, or None."""
    spec = model._specs[name]
    if source_spline(spec) is None:
        return _NOT_SPLINE
    if interaction_users(model, name):
        return _INTERACTION
    if isinstance(spec, OrderedCategorical) and _ordered_axis(spec, term) is None:
        return _NO_AXIS
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
                ("positions", "count", "strategy", "alpha", "lo", "hi", "min_gap", "basis")
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
        "min_gap": axis.min_gap(float(lo), float(hi)),
        "max_count": axis.max_count,
        "resettable": _resettable(session, name),
        "even_only": _even_only_reason(name, declared_spline(model, name)),
        "basis": _basis_payload(fitted, axis, float(lo), float(hi)),
    }


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
            return {
                "positions": list(meta["chart_positions"]),
                "count": int(meta["count"]),
                "strategy": meta["strategy"],
                "alpha": meta["alpha"],
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
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``term`` with new knots, and the waiting change's metadata.

    ``params`` is one of ``{"count", "strategy"[, "alpha"]}``, ``{"positions"}``
    (chart coordinates) or ``{"reset": True}``, which takes the knots of
    ``reference_model``, the opened model. ``draft_spec`` is the term's spec as
    waiting changes leave it; ``levels_waiting`` says one of them changes its
    levels. ``X`` and ``sample_weight`` are the refit's data, on which the new
    knots are placed now, so a refusal comes before the Refit.
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
        basis, marked = _positions_basis(name, fitted, source, axis, params["positions"]), True
    if marked:
        setattr(basis, EDITOR_KNOTS_ATTRIBUTE, True)
    else:
        basis.__dict__.pop(EDITOR_KNOTS_ATTRIBUTE, None)
    replacement = _hosted(spec, basis, name, X) if ordered else basis
    placed, strategy = _placed_knots(model, name, replacement, X, sample_weight)
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


def _positions_basis(name: str, fitted, source: _SplineBase, axis: _Axis, positions):
    inner = fitted._basis_spline if isinstance(fitted, OrderedCategorical) else fitted
    lo, hi = (float(axis.to_chart(edge)) for edge in inner.fitted_boundary)
    gap = axis.min_gap(lo, hi)
    if not isinstance(positions, list | tuple) or not positions:
        raise EditorValueError(_COUNT)
    if not all(_is_real(v) and math.isfinite(v) for v in positions):
        raise EditorValueError(_POSITIONS.format(term=name, gap=_gap_text(gap)))
    chart = np.sort(np.asarray(positions, dtype=np.float64))
    # A relative slack of 1e-9 accepts a gap the browser snapped to exactly the step.
    tight = gap * (1.0 - 1e-9)
    if chart[0] - lo < tight or hi - chart[-1] < tight or np.any(np.diff(chart) < tight):
        raise EditorValueError(_POSITIONS.format(term=name, gap=_gap_text(gap)))
    _require_count(name, axis, int(chart.size))
    _require_uneven_allowed(name, source)
    return respaced_spline(source, knots=axis.to_axis(chart))


def _reset_basis(reference_model, name: str, source: _SplineBase) -> tuple[_SplineBase, bool]:
    """``source`` with the opened model's knot settings, and whether those were the editor's."""
    reference = None if reference_model is None else declared_spline(reference_model, name)
    if reference is None:
        raise EditorValueError(_NO_REFERENCE.format(term=name))
    stated = reference._named_knots or reference._explicit_knots
    basis = respaced_spline(
        source,
        knots=None if stated is None else stated,
        n_knots=reference.n_knots,
        knot_strategy=reference.knot_strategy,
        knot_alpha=reference.knot_alpha,
    )
    return basis, bool(getattr(reference, EDITOR_KNOTS_ATTRIBUTE, False))


# -- Checks -----------------------------------------------------------------------


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
    stated = spline._named_knots or spline._explicit_knots
    if stated is not None:
        return ("stated", tuple(str(v) for v in stated))
    return (spline.knot_strategy, int(spline.n_knots), float(spline.knot_alpha))


# -- Placement --------------------------------------------------------------------


def _numeric_declaration(model, name: str, draft_spec) -> _SplineBase:
    """The unfitted spline a numeric term's knots are changed on: its draft, else its declaration."""
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


def _placed_knots(model, name: str, replacement, X, sample_weight) -> tuple[NDArray, str]:
    """The interior knots ``replacement`` takes on the refit's data, and the rule that placed them.

    The knots are on the spline's own axis; the rule is ``"explicit"`` for
    stated knots, and ``"uniform"`` where a quantile rule fell back to it.

    Placing them is the fit's first step, so a placement the library refuses,
    such as a knot against a shaped range's edge, is refused here.
    """
    probe = copy.deepcopy(replacement)
    column = as_eager_frame(X).column_array(name)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(probe, OrderedCategorical):
                probe.build(column, sample_weight=weights)
                built = probe._basis_spline
            else:
                x = np.asarray(column, dtype=np.float64).ravel()
                keep = np.isfinite(x)
                binned = should_discretize(probe, model._discrete)
                n_bins = resolve_discrete_n_bins(name, probe, model._n_bins) if binned else None
                probe._place_knots(x[keep], None if weights is None else weights[keep], n_bins)
                built = probe
    except RangeError as exc:
        raise EditorValueError(_SHAPED_EDGE) from exc
    return np.asarray(built.fitted_base_knots, dtype=np.float64), _strategy(built)


def _strategy(spline: _SplineBase) -> str:
    """The rule that placed a built spline's knots, or ``"explicit"`` for stated ones."""
    if spline._explicit_knots is not None or spline._named_knots is not None:
        return "explicit"
    return str(getattr(spline, "_knot_strategy_actual", spline.knot_strategy))


# -- Chart coordinates ------------------------------------------------------------


class _Axis:
    """Chart coordinates of a term's knot axis: identity, or an ordered term's level positions."""

    def __init__(self, values: NDArray | None):
        self.values = values
        self.max_count = None if values is None else int(values.size) - 1

    @classmethod
    def of(cls, spec, term: EditableTerm) -> _Axis:
        return cls(_ordered_axis(spec, term) if isinstance(spec, OrderedCategorical) else None)

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

    def min_gap(self, lo: float, hi: float) -> float:
        """The snap step: a tenth of a level, or three significant figures of a numeric span."""
        if self.values is not None:
            return _ORDERED_STEP
        return 10.0 ** (math.floor(math.log10(hi - lo)) - 2)


def _ordered_axis(spec: OrderedCategorical, term: EditableTerm) -> NDArray | None:
    """The axis values of an ordered term's smooth levels, in display order, or None.

    None unless the chart shows the smooth levels first, in the axis's order,
    with strictly increasing values: the mapping a knot position needs.
    """
    smooth = list(spec._smooth_levels)
    if term.levels is None or term.levels[: len(smooth)] != [str(level) for level in smooth]:
        return None
    values = np.asarray([spec._level_to_value[level] for level in smooth], dtype=np.float64)
    if values.size < 2 or not np.all(np.diff(values) > 0.0):
        return None
    return values


def _gap_text(gap: float) -> str:
    return f"{gap:g}"


def _is_integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
