"""The basis of a spline term: its kind, and whether it shrinks (``select=True``).

A basis change is a structural change that waits for Refit like a knot
change. It rebuilds the term's spline from its declaration as another kind,
or with the double penalty turned on or off, keeping its knots, boundary,
penalty order, constraint, shaped ranges and smoothing settings; an ordered
term keeps its order, grouping, special levels and reference.

The kinds the editor offers are the P-spline (``ps``), the B-spline (``bs``),
the cubic regression spline (``cr``) and the natural spline (``ns``). The
cardinal cubic regression spline (``cr_cardinal``) is shown where code
declared it, and cannot be chosen.
"""

from __future__ import annotations

from typing import Any

from superglm.editor._types import EditableTerm
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorValueError
from superglm.editor.knots import LEVEL_OPERATIONS, _hosted, _numeric_declaration
from superglm.features.ordered_categorical import OrderedCategorical, _spline_kind_name
from superglm.features.rebuild import (
    CUBIC_KINDS,
    EDITOR_BASIS_ATTRIBUTE,
    declared_spline,
    interaction_users,
    pristine_basis,
    rebased_spline,
    source_spline,
)
from superglm.features.spline import _SplineBase

# The kinds a term can be switched to, in the order the dropdown lists them.
KINDS = ("ps", "bs", "cr", "ns")
_NAMES = {
    "ps": "P-spline",
    "bs": "B-spline",
    "cr": "cubic regression spline",
    "ns": "natural spline",
    "cr_cardinal": "cardinal cubic regression spline",
}

_NOT_SPLINE = "The basis kind is set for spline terms and ordered terms with a spline basis."
_NO_DECLARATION = (
    "This model does not keep the term's declaration; refit it with this version of superglm "
    "to change its basis."
)
_LEVELS_WAITING = (
    "A waiting change on {term!r} changes its levels; refit it before changing the basis."
)
_FORMS = "Give a basis kind or a shrinkage setting, one at a time."
_KIND = (
    'Choose the basis kind: P-spline ("ps"), B-spline ("bs"), cubic regression ("cr") '
    'or natural ("ns").'
)
_SELECT = "Shrinkage is on (true) or off (false)."
_SAME_KIND = "{term!r} is already a {name}."
_SAME_SELECT = "Shrinkage is already {state} for {term!r}."
_NS_CONSTRAINT = (
    "A natural spline takes no shape constraint, and {term!r} has one; choose another kind, "
    "or remove the constraint in code."
)
_RANGES_KIND = (
    "Shaped ranges need a B-spline or a cubic regression spline, and {term!r} has shaped "
    "ranges; choose one of those kinds, or undo the ranges first."
)
_ORDER_KIND = (
    "A {name}'s penalty needs a penalty order m no higher than its degree, {degree}, and "
    "{term!r} has m={m}; choose a P-spline or a natural spline, or lower m in code."
)
_EVEN_KIND = (
    "{penalty} needs evenly spaced knots, and the knots of {term!r} are not evenly spaced; "
    "place them by even spacing first, or choose another kind."
)
_SELECT_NS = (
    "A natural spline cannot take shrinkage. To shrink {term!r}, choose another kind; to make "
    "it a natural spline, turn Shrink off first."
)
_SELECT_ORDER = (
    "Shrinkage on a {name} needs a penalty order m of {need}, and {term!r} has m={m}; "
    "choose a cubic regression spline, or change m in code."
)
_SELECT_RANGES = (
    "Shrinkage cannot be combined with shaped ranges; undo the ranges of {term!r} first."
)
_SELECT_CONSTRAINT = (
    "Shrinkage cannot be combined with a shape constraint the fit enforces, which {term!r} "
    "has; leave Shrink off, or apply the constraint after the fit (Constraint.postfit) in code."
)
_POLICY_NULL = (
    'The lambda_policy of {term!r} sets the shrinkage penalty ("null"), which it has only '
    "with Shrink on; change lambda_policy in code to turn Shrink off."
)
_NOT_BUILT = (
    "A {name} cannot take the other settings of {term!r}; choose another kind, or change the "
    "term in code."
)


def basis_unavailable_reason(model, name: str) -> str | None:
    """Why ``name``'s basis cannot be changed in the editor, as one sentence, or None.

    A term used by an interaction is refused separately, naming the
    interactions (``_require_not_interaction_parent``).
    """
    if source_spline(model._specs[name]) is None:
        return _NOT_SPLINE
    if declared_spline(model, name) is None:
        return _NO_DECLARATION
    return None


def basis_payload(session, name: str) -> dict[str, Any]:
    """The basis fields of a term's ``knots`` payload (the browser contract).

    ``kind`` and ``select`` are the basis in force; ``kinds`` are the kinds
    the term can be switched to; ``select_available`` says whether Shrink
    can be turned on or off on the term as its waiting changes leave it, and
    ``select_reason`` why not.
    """
    model = session.model
    if basis_unavailable_reason(model, name) is not None or interaction_users(model, name):
        return {
            "kind": None,
            "select": None,
            "kinds": [],
            "select_available": False,
            "select_reason": None,
        }
    spec = model._specs[name]
    fitted = spec._basis_spline if isinstance(spec, OrderedCategorical) else spec
    reason = _levels_waiting(session, name)
    if reason is None:
        shown = _waiting_spline(session, name)
        reason = _select_refusal(name, shown, _spline_kind_name(shown), not shown.select)
    return {
        "kind": _spline_kind_name(fitted),
        "select": bool(fitted.select),
        "kinds": list(KINDS),
        "select_available": reason is None,
        "select_reason": reason,
    }


def pending_basis(session, name: str) -> dict[str, Any] | None:
    """The basis the waiting changes on ``name`` put in force, while it differs, or None.

    A shaped range waiting on a P-spline or natural spline makes it a
    B-spline, so it shows here too.
    """
    draft = _waiting_draft(session, name)
    waiting = None if draft is None else source_spline(draft)
    fitted = source_spline(session.model._specs[name])
    if waiting is None or fitted is None:
        return None
    shown = (_spline_kind_name(waiting), bool(waiting.select))
    if shown == (_spline_kind_name(fitted), bool(fitted.select)):
        return None
    return {"kind": shown[0], "select": shown[1]}


def basis_feature_spec(
    model,
    term: EditableTerm,
    params: dict[str, Any],
    *,
    X,
    draft_spec=None,
    reference_model=None,
    levels_waiting: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``term`` with another basis, and the waiting change's metadata.

    ``params`` is ``{"kind": "ps" | "bs" | "cr" | "ns"}`` or
    ``{"select": True | False}``. ``draft_spec`` is the term's spec as
    waiting changes leave it; ``levels_waiting`` says one of them changes an
    ordered term's levels. A ``ps`` or ``bs`` spline takes the degree of the
    term's declaration in ``reference_model``, the opened model, as a
    structure file applied to that declaration does; ``cr`` and ``ns`` are
    cubic. ``X`` is the refit's data, which an ordered term is rebuilt on.
    """
    name = term.name
    _require_not_interaction_parent(model, name, operation="change the basis")
    reason = basis_unavailable_reason(model, name)
    if reason is not None:
        raise EditorValueError(reason)
    fitted = model._specs[name]
    if levels_waiting and isinstance(fitted, OrderedCategorical):
        raise EditorValueError(_LEVELS_WAITING.format(term=name))
    spec = fitted if draft_spec is None else draft_spec
    ordered = isinstance(spec, OrderedCategorical)
    source = pristine_basis(spec) if ordered else _numeric_declaration(model, name, draft_spec)
    current = _spline_kind_name(source)
    kind, select = _form(name, params, current, bool(source.select))
    degree = _degree(kind, source, reference_model, name)
    refusal = _kind_refusal(name, source, kind, degree) if kind != current else None
    refusal = refusal or _select_refusal(name, source, kind, select)
    if refusal is not None:
        raise EditorValueError(refusal)
    try:
        basis = rebased_spline(source, kind=kind, select=select, degree=degree)
    except (ValueError, NotImplementedError) as exc:
        raise EditorValueError(_NOT_BUILT.format(name=_NAMES[kind], term=name)) from exc
    setattr(basis, EDITOR_BASIS_ATTRIBUTE, True)
    replacement = _hosted(spec, basis, name, X) if ordered else basis
    if kind != current:
        label = f"kind {kind} in {name}"
    else:
        label = f"shrinkage {'on' if select else 'off'} in {name}"
    return replacement, {
        "format": "superglm.editor.basis.v1",
        "term": name,
        "kind": kind,
        "select": select,
        "label": label,
        "message": f"The basis of {name} was changed in the editor and the full model was refit.",
    }


# -- The two forms ----------------------------------------------------------------


def _form(name: str, params: dict[str, Any], kind: str, select: bool) -> tuple[str, bool]:
    """The kind and shrinkage ``params`` ask for: one of them changed, the other kept."""
    if set(params) == {"kind"}:
        if params["kind"] not in KINDS:
            raise EditorValueError(_KIND)
        if params["kind"] == kind:
            raise EditorValueError(_SAME_KIND.format(term=name, name=_NAMES[kind]))
        return str(params["kind"]), select
    if set(params) == {"select"}:
        if not isinstance(params["select"], bool):
            raise EditorValueError(_SELECT)
        if params["select"] == select:
            raise EditorValueError(_SAME_SELECT.format(state="on" if select else "off", term=name))
        return kind, params["select"]
    raise EditorValueError(_FORMS)


def _degree(kind: str, source: _SplineBase, reference_model, name: str) -> int:
    """The degree ``kind`` is built with: cubic, or the declaration's for ``ps`` and ``bs``."""
    if kind in CUBIC_KINDS:
        return 3
    declared = None if reference_model is None else declared_spline(reference_model, name)
    return int((source if declared is None else declared).degree)


# -- Checks -----------------------------------------------------------------------


def _kind_refusal(name: str, source: _SplineBase, kind: str, degree: int) -> str | None:
    """Why ``source`` cannot become a ``kind`` spline of ``degree``, or None."""
    orders = tuple(source._m_orders)
    if kind == "ns" and source.constraint_kind is not None:
        return _NS_CONSTRAINT.format(term=name)
    if kind in ("ps", "ns") and source.polynomial_ranges:
        return _RANGES_KIND.format(term=name)
    if kind in ("bs", "cr") and max(orders) > degree:
        return _ORDER_KIND.format(name=_NAMES[kind], degree=degree, term=name, m=_m_text(orders))
    if _uneven(source):
        if kind == "ns":
            return _EVEN_KIND.format(penalty="A natural spline's penalty", term=name)
        if kind == "ps" and max(orders) > degree:
            return _EVEN_KIND.format(
                penalty="A P-spline's penalty of an order above its degree", term=name
            )
    return None


def _select_refusal(name: str, source: _SplineBase, kind: str, select: bool) -> str | None:
    """Why a ``kind`` spline built from ``source`` cannot have ``select``, or None."""
    if not select:
        policy = source._lambda_policy
        return (
            _POLICY_NULL.format(term=name)
            if isinstance(policy, dict) and "null" in policy
            else None
        )
    orders = tuple(source._m_orders)
    if kind == "ns":
        return _SELECT_NS.format(term=name)
    if source.polynomial_ranges:
        return _SELECT_RANGES.format(term=name)
    if source.constraint_kind is not None and source.constraint_mode == "fit":
        return _SELECT_CONSTRAINT.format(term=name)
    if kind in ("ps", "bs") and max(orders) > 2:
        need = "2 or less"
    elif kind == "cr_cardinal" and orders != (2,):
        need = "2"
    else:
        return None
    return _SELECT_ORDER.format(name=_NAMES[kind], need=need, term=name, m=_m_text(orders))


def _uneven(source: _SplineBase) -> bool:
    """Whether ``source``'s knots are stated, or placed by a rule other than even spacing."""
    stated = source._named_knots is not None or source._explicit_knots is not None
    return stated or source.knot_strategy != "uniform"


def _levels_waiting(session, name: str) -> str | None:
    """The refusal while a waiting change moves an ordered term's levels, else None."""
    if not isinstance(session.model._specs[name], OrderedCategorical):
        return None
    waiting = {step.operation for step in session.pending if step.term == name}
    return _LEVELS_WAITING.format(term=name) if LEVEL_OPERATIONS & waiting else None


def _waiting_draft(session, name: str):
    """The last waiting change's draft for ``name``, or None when none waits."""
    return next((step.draft_spec for step in reversed(session.pending) if step.term == name), None)


def _waiting_spline(session, name: str) -> _SplineBase:
    """The spline ``name`` has as its waiting changes leave it: the last draft's, else declared."""
    draft = _waiting_draft(session, name)
    waiting = None if draft is None else source_spline(draft)
    if waiting is not None:
        return waiting
    declared = declared_spline(session.model, name)
    if declared is None:  # pragma: no cover - basis_unavailable_reason refuses it first
        raise EditorValueError(_NO_DECLARATION)
    return declared


def _m_text(orders: tuple[int, ...]) -> str:
    return str(orders[0]) if len(orders) == 1 else str(orders)
