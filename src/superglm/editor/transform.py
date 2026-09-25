"""Give one term a stated shape: piecewise polynomial, spline at breaks, or polynomial."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm._frame import as_eager_frame
from superglm.editor._types import EditableTerm
from superglm.editor.collapse import (
    _pristine_basis,
    _require_not_interaction_parent,
    rebuilt_ordered_spec,
    special_labels,
)
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.features.constraint import ConstraintSpec
from superglm.features.ordered_categorical import OrderedCategorical, _spline_kind_name
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline, _SplineBase

FORMS = ("piecewise", "spline", "polynomial")
MAX_SEGMENT_DEGREE = 3
MAX_POLYNOMIAL_DEGREE = 5
# Read by model/report_ops.py BY NAME, so the model layer never imports the
# editor: a basis carrying it had its shape chosen in the editor from this
# data, so its tests are conditional on it (spec §3.3).
EDITOR_CHOSEN_SHAPE_ATTRIBUTE = "_editor_chosen_shape"
# A linear Numeric term is drawn as a single point, so it has no axis to place
# a break on: the Breaks tool stays off for it.
_NUMERIC_KINDS = (_SplineBase, Polynomial, Piecewise)


def transformed_feature_spec(
    model, term: EditableTerm, *, form, breaks, degrees=None, degree=None, X
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec that gives ``term`` the requested shape, with the step's metadata."""
    spec = model._specs[term.name]
    ordered = isinstance(spec, OrderedCategorical)
    if not ordered and not isinstance(spec, _NUMERIC_KINDS):
        raise EditorTypeError(f"Breaks need an ordered or numeric axis; {term.name!r} has neither.")
    _require_not_interaction_parent(model, term.name, operation="transform")
    if form not in FORMS:
        raise EditorValueError(f"Choose a piecewise, spline or polynomial form, got {form!r}.")
    source = _source_spline(spec)
    # Polynomial and ordered terms state no outside rule: the library default.
    extrapolation = getattr(spec, "extrapolation", "clip")
    if form == "polynomial":
        basis = _polynomial(term.name, source, breaks, degree)
    else:
        axis = _break_axis(spec, term) if ordered else None
        _validate_breaks(term, breaks, axis)
        pins = _piecewise_pins(spec, breaks)
        basis = (
            _piecewise(term.name, source, axis, breaks, degrees, extrapolation, pins)
            if form == "piecewise"
            else _knotted_spline(source, breaks, extrapolation)
        )
        setattr(basis, EDITOR_CHOSEN_SHAPE_ATTRIBUTE, True)
    replacement = _hosted(spec, basis, term.name, X) if ordered else basis
    return replacement, _metadata(term.name, form, breaks, degrees, degree)


def transform_payload(spec, term: EditableTerm) -> dict[str, Any] | None:
    """What the Breaks tool starts from; None for a term it cannot transform."""
    ordered = isinstance(spec, OrderedCategorical)
    if not ordered and not isinstance(spec, _NUMERIC_KINDS):
        return None
    axis = _break_axis(spec, term) if ordered else None
    return {"axis": axis, "piecewise": _current_piecewise(spec, axis)}


def _current_piecewise(spec, axis: list[str] | None) -> dict[str, list] | None:
    """The stated breaks and segment degrees of a piecewise term, in the route's vocabulary."""
    if isinstance(spec, Piecewise):
        return {
            "breaks": [float(knot) for knot in spec._knots[1:-1]],
            "degrees": [1] * (len(spec._knots) - 1),
        }
    basis = getattr(spec, "_spline_obj", None)
    # Int-mode breaks are placed from the data: there are no stated positions.
    if not isinstance(basis, Piecewise) or not isinstance(basis.breaks, list):
        return None
    return {
        "breaks": [b if isinstance(b, str) else axis[int(b)] for b in basis.breaks],
        "degrees": list(basis.degrees or [1] * (len(basis.breaks) + 1)),
    }


def _source_spline(spec) -> _SplineBase | None:
    """The spline the term is fitted with, whose settings and constraint carry over."""
    basis = _pristine_basis(spec) if isinstance(spec, OrderedCategorical) else spec
    return basis if isinstance(basis, _SplineBase) else None


def _break_axis(spec: OrderedCategorical, term: EditableTerm) -> list[str]:
    """The displayed ordered bands a break can name: every level but the specials."""
    specials = special_labels(spec)
    return [level for level in term.levels if level not in specials]


def _validate_breaks(term: EditableTerm, breaks: list, axis: list[str] | None) -> None:
    if not breaks:
        raise EditorValueError("Add at least one break.")
    if axis is None:
        _validate_numeric_breaks(term, breaks)
    else:
        _validate_band_breaks(term.name, breaks, axis)


def _validate_band_breaks(name: str, breaks: list, axis: list[str]) -> None:
    interior = axis[1:-1]
    if any(b not in interior for b in breaks):
        raise EditorValueError(
            f"Breaks must be interior bands of {name} (not its first or last band)."
        )
    if np.any(np.diff([interior.index(b) for b in breaks]) <= 0):
        raise EditorValueError(f"Breaks must be strictly increasing along {name}'s bands.")


def _validate_numeric_breaks(term: EditableTerm, breaks: list) -> None:
    lo, hi = float(np.min(term.x)), float(np.max(term.x))
    # A NaN fails both comparisons and an infinity lies outside, so this also
    # refuses every non-finite break.
    if not all(_is_real(b) and lo < b < hi for b in breaks):
        raise EditorValueError(
            f"Breaks must be finite numbers strictly inside the fitted range {lo:g} to {hi:g}."
        )
    if np.any(np.diff(np.asarray(breaks, dtype=np.float64)) <= 0):
        raise EditorValueError("Breaks must be strictly increasing.")


def _polynomial(name: str, source, breaks: list, degree) -> Polynomial:
    _refuse_constraint(name, source, "polynomial")
    if breaks:
        raise EditorValueError("A polynomial has no breaks; clear them or choose another form.")
    if not _is_int(degree) or not 1 <= degree <= MAX_POLYNOMIAL_DEGREE:
        raise EditorValueError("Choose a polynomial degree from 1 to 5.")
    return Polynomial(degree=int(degree))


def _piecewise(
    name: str, source, axis, breaks: list, degrees, extrapolation: str, pins: dict[str, Any]
) -> Piecewise:
    """Straight segments on a numeric axis; stated per-segment degrees on bands.

    All-flat and consecutive-flat segments stay the library's call.
    """
    _refuse_constraint(name, source, "piecewise")
    if axis is None:
        if degrees is not None and any(d != 1 for d in degrees):
            raise EditorValueError(
                "On a numeric feature, piecewise segments are straight lines; "
                "per-segment degrees need an ordered term."
            )
        return Piecewise(breaks=list(breaks), extrapolation=extrapolation, **pins)
    degrees = [1] * (len(breaks) + 1) if degrees is None else list(degrees)
    n = len(breaks)
    if len(degrees) != n + 1 or not all(_is_int(d) for d in degrees):
        raise EditorValueError(f"State one degree per segment: {n} breaks make {n + 1} segments.")
    if not all(0 <= d <= MAX_SEGMENT_DEGREE for d in degrees):
        raise EditorValueError("Segment degrees run from 0 (flat) to 3 (cubic).")
    return Piecewise(breaks=list(breaks), degrees=degrees, extrapolation=extrapolation)


def _refuse_constraint(name: str, source, form: str) -> None:
    """A shape constraint is a market rule: never drop it silently."""
    kind = None if source is None else source.constraint_kind
    if kind is not None:
        raise EditorValueError(
            f"{name} carries the {kind} constraint, which a {form} can't hold. "
            "Choose Spline, or remove the constraint first."
        )


def _piecewise_pins(spec, breaks: list) -> dict[str, Any]:
    """A numeric Piecewise source's pinned outer knots, and its base while a knot keeps it."""
    if not isinstance(spec, Piecewise):
        return {}
    pins = {"lower": spec.lower, "upper": spec.upper}
    # The outer knots are refit from the same pins and data, so they stay put.
    knots = {float(spec._knots[0]), *breaks, float(spec._knots[-1])}
    if isinstance(spec.base, str) or float(spec.base) in knots:
        pins["base"] = spec.base
    return pins


def _knotted_spline(source, knots: list, extrapolation: str) -> _SplineBase:
    """Knots at the breaks; a source spline's kind, settings and constraint carry over."""
    if source is None:
        return Spline(kind="cr", knots=list(knots), extrapolation=extrapolation)
    constraint = (
        None
        if source.constraint_kind is None
        else ConstraintSpec(mode=source.constraint_mode, kind=source.constraint_kind)
    )
    return Spline(
        kind=_spline_kind_name(source),
        knots=list(knots),
        degree=source.degree,
        penalty=source.penalty,
        select=source.select,
        extrapolation=source.extrapolation,
        boundary=source._explicit_boundary,
        discrete=source.discrete,
        n_bins=source.n_bins,
        m=source._m_orders,
        constraint=constraint,
        lambda_policy=source._lambda_policy,
    )


def _hosted(spec: OrderedCategorical, basis, name: str, X) -> OrderedCategorical:
    """A fresh ordered term around ``basis``, keeping order, specials, grouping, base."""
    frame = as_eager_frame(X)
    frame.require_columns((name,))
    return rebuilt_ordered_spec(
        spec,
        grouping=getattr(spec, "_grouping", None),
        base=spec.base,
        data=frame.column_array(name),
        basis=basis,
    )


def _metadata(name: str, form: str, breaks: list, degrees, degree) -> dict[str, Any]:
    n = len(breaks)
    plural = "s" if n != 1 else ""
    labels = {
        "piecewise": f"transform {name} to piecewise ({n} break{plural})",
        "spline": f"transform {name} to spline, knots at {n} break{plural}",
        "polynomial": f"transform {name} to polynomial, degree {degree}",
    }
    return {
        "format": "superglm.editor.term_transform.v1",
        "term": name,
        "form": form,
        "breaks": breaks,
        "degrees": degrees,
        "degree": degree,
        "label": labels[form],
        "message": f"{name} was transformed in the editor and the full model was refit.",
    }


def _is_int(value) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)
