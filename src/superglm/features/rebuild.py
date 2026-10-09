"""Rebuild a term's spec with new structural decisions.

A structural decision -- how a categorical's levels are grouped, which level
is the reference, where new levels go, the polynomial ranges of a spline --
is built into a fresh, unfitted spec made from the term's declaration, never
by mutating a fitted one, and the new spec goes into a clone of the model.
The editor's collapse, ungroup, reference and shape steps and
:meth:`superglm.structure.Structure.apply` build terms here, so the two share
one implementation and the library never imports the editor.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from superglm.features._spline_ranges import PolynomialRange
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping, native_by_text
from superglm.features.ordered_categorical import (
    _CLAMP_WARNING_PREFIX,
    OrderedCategorical,
    _spline_kind_name,
)
from superglm.features.spline import CardinalCRSpline, Spline, _SplineBase

# The declared bases a fit resolves to a level itself.
SYMBOLIC_BASE_POLICIES = frozenset({"first", "most_exposed"})
# The levels a structural decision took off an ordered term's curve, by label,
# each with what it needs to go back: its value on the axis, and the levels
# before it in the term's order, nearest first. Levels declared special have
# neither, so they cannot go onto the curve. A plain dict of tuples, so a
# pickled spec needs no class from here.
FREED_LEVELS_ATTRIBUTE = "_freed_levels"

TOO_FEW_POINTS = "Select at least two points to shape a range."


class RangePlacementError(ValueError):
    """A range placed where the term cannot take it.

    An edge that is not a single band of an ordered term, or a range that
    overlaps another without covering the same span. The message is one
    sentence that can be shown to a user as it stands.
    """


# -- The model -------------------------------------------------------------------


def clone_with_replaced_features(model, replacements: dict[str, Any], *, lambda1=..., lambda2=...):
    """Clone a model and replace feature specs before fitting.

    Each replacement is deep-copied in, so fitting the clone never touches the
    caller's spec: a waiting step's draft stays unfitted, and a refit that is
    undone and run again fits a fresh copy of the same draft.
    """
    new_model = model._clone_without_features(set(), lambda1=lambda1, lambda2=lambda2)
    for term, replacement in replacements.items():
        new_model._specs[term] = copy.deepcopy(replacement)
    # A bind_levels binding was resolved on the old spec. Its base names a
    # level of the old grouping, which a replacement whose base is still
    # "most_exposed" would pin though it may no longer be a level, so a
    # replaced term's base resolves again where it is fitted. Its universe
    # carries over only to a replacement that declares none.
    kept = []
    for name, binding in getattr(new_model, "_level_bindings", None) or ():
        if name not in replacements:
            kept.append((name, binding))
        elif getattr(replacements[name], "_declared_levels", None) is None:
            kept.append((name, dataclasses.replace(binding, base=None)))
    bindings = tuple(kept) or None
    new_model._level_bindings = bindings
    new_model._config = new_model._config.with_value(
        feature_templates=tuple(
            (name, new_model._specs[name]) for name in new_model._feature_order
        ),
        level_bindings=bindings,
    )
    new_model._config_revision += 1
    return new_model


def interaction_users(model, term: str) -> list[str]:
    """The interactions that use ``term`` as a parent."""
    return [
        str(name)
        for name, spec in getattr(model, "_interaction_specs", {}).items()
        if term in getattr(spec, "parent_names", ())
    ]


# -- Categorical terms ---------------------------------------------------------


def rebuilt_categorical(
    spec: Categorical,
    fitted: Categorical,
    *,
    base,
    grouping,
    data,
    unseen: str | None = None,
    levels: list | None = None,
    level: bool = False,
) -> Categorical:
    """A fresh Categorical like ``spec`` with this grouping and base.

    It keeps ``levels=`` and ``unseen=``, which a collapse or ungroup used to
    drop; ``unseen`` replaces the policy and ``levels`` the declared universe.
    Grouped, the design speaks the grouping's text labels; ungrouped, the base
    goes back to its native value, so an integer level stays 3, not "3".
    ``level=True`` says ``base`` names a level or group, as a reference a fit
    resolved does, even when it reads "first" or "most_exposed".
    """
    named = level or str(base) not in SYMBOLIC_BASE_POLICIES
    if grouping is None and named:
        base = _native_levels(spec, fitted, data).get(str(base), base)
    rebuilt = Categorical(
        base=base,
        grouping=grouping,
        levels=_declared_universe(spec, grouping) if levels is None else levels,
        unseen=spec.unseen if unseen is None else unseen,
    )
    rebuilt._base_is_level = named and str(base) in SYMBOLIC_BASE_POLICIES
    return rebuilt


def base_names_level(spec) -> bool:
    """Whether ``spec``'s ``base=`` names a level or group rather than a base policy.

    It names a policy when it reads "first" or "most_exposed", unless a rebuild
    marked it as a level (:func:`rebuilt_categorical`).
    """
    return getattr(spec, "_base_is_level", False) or str(spec.base) not in SYMBOLIC_BASE_POLICIES


def _declared_universe(spec: Categorical, grouping) -> list | None:
    """The universe a rebuild of ``spec`` declares: its own, widened where it accepted more.

    A universe bound from a frame or a dtype is not a declaration: the term
    accepted every label its grouping maps, those the frame lacks included.
    So did a term declared before construction refused a grouping wider than
    ``levels=`` (:func:`accepted_levels`). The rebuilt term declares its
    universe, and a declaration names every label its grouping maps, so those
    labels follow the declared ones, which keep their order and so the design's.
    """
    declared = spec._declared_levels
    if spec._level_source == "declared" and accepted_levels(spec) == declared:
        return declared
    return _with_grouped(declared, grouping)


def accepted_levels(spec: Categorical) -> list | None:
    """The levels ``spec`` accepts by declaration: ``levels=``, and what its grouping adds.

    Construction refuses a grouping that maps labels ``levels=`` leaves out,
    but a term pickled before it did (superglm 0.36.1 and earlier), or one
    whose universe a frame bound, keeps such a grouping: it fits and scores
    those labels through their group. They follow the declared levels. None
    when nothing is declared.
    """
    return _with_grouped(spec._declared_levels, getattr(spec, "_grouping", None))


def _with_grouped(declared: list | None, grouping) -> list | None:
    """``declared``, then the labels ``grouping`` maps that it leaves out."""
    if declared is None or grouping is None:
        return declared
    named = {str(level) for level in declared}
    return [*declared, *(raw for raw in grouping.all_original_levels if raw not in named)]


def _native_levels(spec: Categorical, fitted: Categorical, data) -> dict[str, Any]:
    """Each level's native value by its text: declared, then fitted, then seen in ``data``.

    A draft has no fitted ``_levels``, so the in-force spec and the column stand
    in for it. A grouped spec's fitted levels are group labels, not raw ones.
    """
    fitted_levels = fitted._levels if getattr(fitted, "_grouping", None) is None else []
    observed = pd.unique(np.asarray(data).ravel()).tolist()
    return native_by_text(spec._declared_levels or [], fitted_levels, observed)


# -- Ordered terms ---------------------------------------------------------------


def rebuilt_ordered_spec(
    spec: OrderedCategorical,
    *,
    grouping: LevelGrouping | None,
    base: Any,
    data,
    basis=None,
    level: bool = False,
    freed: tuple[str, ...] = (),
    returned: tuple[str, ...] = (),
) -> OrderedCategorical:
    """A fresh, unfitted OrderedCategorical like ``spec`` with this grouping and base.

    ``basis`` replaces the inner basis (a shaped range). By default the pristine
    declared basis is cloned. A fitted spec is never mutated: its resolved base
    is sticky and would silently survive a changed ``base``. ``level=True``
    says ``base`` names a band or group, as :func:`rebuilt_categorical`'s does.

    ``freed`` names levels on the curve to take off it, as special levels, and
    ``returned`` special levels to put back on it, by label; each returned
    level must be one :func:`freed_levels` records. The levels taken off keep
    what they need to go back on the spec, ``FREED_LEVELS_ATTRIBUTE``.
    """
    named = level or str(base) not in SYMBOLIC_BASE_POLICIES
    values, native_base = _ordered_original_values(spec, grouping, data, base, named=named)
    # Clone the RAW declarations, not the string-coerced ``_specials``. A special
    # declared as ``9`` on a float column matches through its raw label -- the
    # string view renders 9.0 as "9.0", which never equals "9" -- so rebuilding
    # from the coerced form silently drops that fallback and the special's
    # indicator comes back all-zero on a refit.
    specials = list(spec._special_raw) or list(spec._specials)
    shown = list(spec._special_display)
    record = freed_levels(spec)
    order = full_level_order(spec) if freed else []
    positional = not isinstance(getattr(spec, "_spline_obj", None), _SplineBase)
    for label in returned:
        at = next(i for i, special in enumerate(shown) if str(special) == label)
        value = _returned_value(record.pop(label), values, positional=positional)
        del specials[at]
        display = shown.pop(at)
        values[display if grouping is None else label] = value
    if returned and grouping is not None:
        grouping = _grouping_in_axis_order(grouping, values)
    for label in freed:
        key = next(key for key in values if str(key) == label)
        before = order[: order.index(label)]
        record[label] = (values.pop(key), tuple(reversed(before)))
        specials.append(key)
        shown.append(key)
    # A special the declaration also named in order= or values= is reported
    # under that domain spelling (9.0 beside 1.0 and 2.0) and matches rows
    # through its raw label (9). The smooth's values lack it, so it is named
    # in values= again, which takes it out of the smooth as the declaration
    # did and keeps the spelling; the value given is never read.
    values.update(dict.fromkeys(shown, 0.0))
    source = pristine_basis(spec) if basis is None else basis
    # Collapsing levels shrinks the level count, so the pristine spline's
    # ``n_knots`` routinely exceeds the new ``n_levels - 1`` and construction
    # clamps it. That clamp is the caller's own basis being re-fitted to the
    # levels the caller just asked to merge, not a configuration mistake, and
    # the user-facing construction already warned if the original declaration
    # over-specified. Do not repeat it from an internal rebuild.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=re.escape(_CLAMP_WARNING_PREFIX),
            category=UserWarning,
        )
        rebuilt = OrderedCategorical(
            values=values,
            basis=source,
            base=native_base,
            grouping=grouping,
            specials=specials or None,
        )
    rebuilt._base_is_level = named and str(base) in SYMBOLIC_BASE_POLICIES
    if record:
        setattr(rebuilt, FREED_LEVELS_ATTRIBUTE, record)
    return rebuilt


def freed_levels(spec) -> dict[str, tuple[float, tuple[str, ...]]]:
    """The special levels of ``spec`` a structural decision took off its curve.

    Each label maps to its value on the axis and the levels before it in the
    term's order, nearest first: what putting it back needs. A copy, so a
    caller can change it freely.
    """
    return dict(getattr(spec, FREED_LEVELS_ATTRIBUTE, None) or {})


def full_level_order(spec: OrderedCategorical) -> list[str]:
    """The term's levels in order: those on its curve, and those taken off it in their place.

    Levels declared special have no place and are left out. Each level taken
    off goes back after the nearest level before it that is in the list, so
    one taken off next to another goes back beside it.
    """
    order = [str(level) for level in spec._declared_smooth_levels]
    for label, (_value, before) in freed_levels(spec).items():
        after = next((name for name in before if name in order), None)
        order.insert(0 if after is None else order.index(after) + 1, label)
    return order


def _grouping_in_axis_order(grouping: LevelGrouping, values: dict) -> LevelGrouping:
    """``grouping`` in axis order: a level put back on the curve takes its place.

    A term's bands follow its grouping's order, its levels as shown follow the
    grouping's originals, and a grouping made while a level was special lists
    that level last in both. Each original sits at its value and each group at
    its members' mean; a special, with no value, keeps its place after them.
    """
    axis = {str(key): float(at) for key, at in values.items()}

    def position(label) -> float:
        members = [
            axis[str(m)] for m in grouping.group_to_originals.get(label, [label]) if str(m) in axis
        ]
        return sum(members) / len(members) if members else float("inf")

    return dataclasses.replace(
        grouping,
        grouped_levels=sorted(grouping.grouped_levels, key=position),
        all_original_levels=sorted(
            grouping.all_original_levels, key=lambda original: axis.get(str(original), float("inf"))
        ),
    )


def states_positional_breaks(spec) -> bool:
    """Whether ``spec`` states Piecewise breaks by position, which a level leaving the curve moves."""
    from superglm.features.piecewise import Piecewise

    basis = getattr(spec, "_spline_obj", None)
    breaks = getattr(basis, "breaks", None) if isinstance(basis, Piecewise) else None
    return isinstance(breaks, list) and any(not isinstance(entry, str) for entry in breaks)


def _returned_value(
    entry: tuple[float, tuple[str, ...]], values: dict, *, positional: bool
) -> float:
    """The axis value a level put back on the curve takes.

    A spline's axis keeps the declared values, so the level takes its own. A
    Piecewise or Polynomial axis numbers the bands 0..L-1 again on every
    build, so its old number may now be another band's: the level goes
    between the nearest level before it still on the curve and the next band.
    """
    value, before = entry
    if not positional:
        return value
    axis = {str(key): float(at) for key, at in values.items()}
    after = next((axis[name] for name in before if name in axis), None)
    if after is None:
        return min(axis.values(), default=1.0) - 1.0
    later = [at for at in axis.values() if at > after]
    return after + 0.5 if not later else (after + min(later)) / 2.0


def pristine_basis(spec: OrderedCategorical):
    """A copy of the spline ``spec`` was declared with, before its level count clamped it."""
    # Clone from the pristine caller-supplied spline, not the clamped inner
    # copy: the new spec re-clamps against ITS OWN level count, which can
    # exceed the current one when a grouping is being undone. `_spline_obj` is
    # always set on a spec this version constructed; the fallback covers a
    # pre-0.24 pickle, whose `_basis_spline` read refuses a step-mode spec
    # loudly instead of silently cloning it onto the default P-spline.
    #
    # Read it with `getattr`, not `spec._spline_obj`: an attribute-less read
    # only reaches the fallback when the key EXISTS and is None, so a pickle
    # old enough to predate the attribute raised a bare AttributeError naming
    # `_spline_obj` -- still loud, but without the migration sentence, and
    # never reaching `_basis_spline` where that sentence lives.
    spline_obj = getattr(spec, "_spline_obj", None)
    if spline_obj is not None:
        source = copy.deepcopy(spline_obj)
    else:
        # A shortcut-era pickle has no pristine declaration, only the inner
        # spline -- and its `n_knots` was already clamped to the level count it
        # was BUILT against. Cloning that alone keeps the reduced basis, which
        # is wrong in exactly the direction ungrouping goes: back to MORE
        # levels. The removed shortcut path rebuilt from the then-plain
        # `n_knots` attribute, i.e. the count the caller REQUESTED, and that
        # entry survives in the pickled __dict__ because the class property
        # only shadows it. Recover it and let __init__ re-clamp against the new
        # level count, so the clamp is still enforced -- just against the right
        # number of levels.
        source = copy.deepcopy(spec._basis_spline)
        requested = spec.__dict__.get("n_knots")
        if isinstance(requested, int | np.integer) and int(requested) > source.n_knots:
            source.n_knots = int(requested)
    return source


def _ordered_original_values(
    spec: OrderedCategorical,
    grouping: LevelGrouping | None,
    data,
    base,
    *,
    named: bool,
) -> tuple[dict[Any, float], Any]:
    original_values = getattr(spec, "_original_level_to_value", None)
    if original_values is not None:
        values = {str(k): float(v) for k, v in original_values.items()}
    else:
        values = {str(k): float(v) for k, v in spec._level_to_value.items()}
    if grouping is not None:
        return values, base

    native_by_label = native_by_text(np.asarray(data, dtype=object).ravel())
    native_values = {native_by_label.get(label, label): value for label, value in values.items()}
    native_base = native_by_label.get(str(base), base) if named else base
    return native_values, native_base


def special_labels(spec: OrderedCategorical) -> set[str]:
    """The free (special) levels of ``spec``, in every spelling the editor displays."""
    # Both namespaces: displayed levels arrive in the DISPLAY spelling, so
    # matching only the str-coerced `_specials` leaves a guard INERT on a float
    # domain ("9" vs "9.0") -- and a guard that fails open here silently smooths
    # a level that `specials=` still reports as free.
    return {str(level) for level in spec._specials} | {
        str(level) for level in spec._special_display
    }


# -- Polynomial ranges -----------------------------------------------------------


def shape_unavailable_reason(model, name: str) -> str | None:
    """Why ``name`` cannot take polynomial ranges, as one sentence, or None when it can."""
    source = source_spline(model._specs[name])
    if source is None:
        return "Shapes need a spline term."
    if isinstance(source, CardinalCRSpline):
        return "Shapes are not available for cardinal cubic regression splines."
    if source.constraint_kind is not None:
        return "Remove the term's shape constraint to add shaped ranges."
    if source.select:
        return "Remove select=True from the term to add shaped ranges."
    if max(source._m_orders) > source.degree:
        # A shaped term is rebuilt with a derivative penalty, whose order the
        # degree bounds; a difference penalty (ps) is not bounded so.
        return "Shapes need a penalty order no higher than the spline's degree."
    if interaction_users(model, name):
        return "A term used by an interaction cannot be reshaped."
    return None


def source_spline(spec) -> _SplineBase | None:
    """The spline a term is declared with: its own spec, or an ordered term's basis."""
    basis = getattr(spec, "_spline_obj", None) if isinstance(spec, OrderedCategorical) else spec
    return basis if isinstance(basis, _SplineBase) else None


def current_ranges(spec) -> tuple[PolynomialRange, ...]:
    """The ranges in force, in axis order: band names on an ordered term, values otherwise."""
    source = source_spline(spec)
    if source is None:
        return ()
    if not isinstance(spec, OrderedCategorical):
        return source.polynomial_ranges
    return tuple(sorted(source.polynomial_ranges, key=lambda r: spec._range_edge_value(r.lo)))


def band_edges(spec: OrderedCategorical, name: str, lo, hi) -> tuple[str, str]:
    """Two single bands in axis order; a group, a special or an unknown label refuses."""
    lo, hi = str(lo), str(hi)
    if {lo, hi} & special_labels(spec):
        raise RangePlacementError(
            f"A shaped range covers only the bands of {name}; "
            "leave its special levels out of the selection."
        )
    try:
        at = {lo: spec._range_edge_value(lo), hi: spec._range_edge_value(hi)}
    except ValueError as exc:
        raise RangePlacementError(
            f"A shaped range must start and end on single bands of {name}; "
            "ungroup the bands at its ends first."
        ) from exc
    if at[lo] == at[hi]:
        raise RangePlacementError(TOO_FEW_POINTS)
    lo, hi = sorted((lo, hi), key=at.__getitem__)
    return lo, hi


def merged_ranges(
    existing: tuple[PolynomialRange, ...],
    new: PolynomialRange,
    position: Callable[[Any], float],
) -> list[PolynomialRange]:
    """``existing`` plus ``new``: the same range is replaced, any other overlap refused."""
    span = (position(new.lo), position(new.hi))
    kept = []
    for current in existing:
        at = (position(current.lo), position(current.hi))
        if at == span:
            continue
        if at[0] < span[1] and span[0] < at[1]:
            raise RangePlacementError(
                f"This range overlaps the {current.label} range "
                f"{edge_text(current.lo)}–{edge_text(current.hi)}. "
                "Undo it or choose a range outside it."
            )
        kept.append(current)
    return [*kept, new]


def shaped_spline(source: _SplineBase, ranges, *, knots, boundary) -> _SplineBase:
    """``source``'s settings with ``ranges``; a ``ps``/``ns`` source becomes ``bs``.

    Range edges repeat knots, which the equal-spacing difference penalties
    cannot take; a ``bs`` with the same knots, degree and penalty order is
    the derivative-penalty spline whose penalty can skip the pinned ranges.
    """
    return Spline(
        kind="cr" if _spline_kind_name(source) == "cr" else "bs",
        n_knots=source.n_knots,
        knots=knots,
        boundary=boundary,
        degree=source.degree,
        knot_strategy=source.knot_strategy,
        knot_alpha=source.knot_alpha,
        penalty=source.penalty,
        extrapolation=source.extrapolation,
        discrete=source.discrete,
        n_bins=source.n_bins,
        m=source._m_orders,
        lambda_policy=source._lambda_policy,
        polynomial_ranges=ranges,
    )


def edge_text(edge) -> str:
    """A range edge as a sentence names it: a band as it is, a number as ``%g``."""
    return edge if isinstance(edge, str) else f"{edge:g}"
