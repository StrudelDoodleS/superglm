"""Which fit method published a model's fit: one tag, one table of moves, one writer.

``model._last_fit_meta`` holds the tag under ``"method"``, beside the facts that fit
recorded (``"discrete"``, the direct backend, the smoothing strategy, the initial
precision findings). Only this module writes it: ``record`` makes the moves,
``annotate`` adds facts to the recorded fit and ``copy_record`` gives another model
a copy. Every other module reads the tag through ``method_of`` and what it implies
through ``traits_of``.

The moves (``MOVES``): from any state, a successful ``fit``, ``fit_path``,
``fit_reml`` or ``fit_ncv`` leads to that method, and a failed fit leaves the state
as it was, because a fit runs in a workspace that is discarded when it fails. The
publication refit of ``estimate_p`` or ``estimate_theta`` is one of those fits: it
records the method it refits with. A model never fitted, or saved before fits
recorded their method, holds no tag (``None``) and reads with the traits of ``fit``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

FitMethod = Literal["fit", "fit_path", "fit_reml", "fit_ncv"]
FitState = FitMethod | None
FAILED = "failed"


@dataclass(frozen=True)
class FitMethodTraits:
    """What a fit method implies for the readers of a fitted model.

    ``summary_label`` is the method the summary prints. ``criterion`` is the one a
    smoothing search result (``model._reml_result.criterion``) must carry, or None
    for a method that searches no smoothing. ``profiles_scale`` is whether the
    likelihood scale is the REML profile. ``refit_as`` is the method an ``"auto"``
    or ``"inherit"`` refit replays. ``enforces_caps`` is whether the fit enforced
    the terms' bounds and caps.
    """

    summary_label: str
    criterion: str | None
    profiles_scale: bool
    refit_as: FitMethod
    enforces_caps: bool


FIT_METHODS: Mapping[FitMethod, FitMethodTraits] = MappingProxyType(
    {
        "fit": FitMethodTraits("ML", None, False, "fit", True),
        "fit_path": FitMethodTraits("ML", None, False, "fit", False),
        "fit_reml": FitMethodTraits("REML", "reml", True, "fit_reml", True),
        "fit_ncv": FitMethodTraits("NCV", "ncv", False, "fit", True),
    }
)

MOVES: Mapping[FitState, Mapping[str, FitState]] = MappingProxyType(
    {
        state: MappingProxyType({**{method: method for method in FIT_METHODS}, FAILED: state})
        for state in (None, *FIT_METHODS)
    }
)


def next_state(state: FitState, event: str) -> FitState:
    """The state ``event`` leads to from ``state``; a tag the table lacks moves as ``None``."""
    moves = MOVES.get(state, MOVES[None])
    if event not in moves:
        raise ValueError(f"unknown fit event {event!r}; expected one of {sorted(moves)}")
    return moves[event]


def record(model, method: FitMethod, **facts: Any) -> None:
    """Record that ``method`` published the model's fit, with the facts it recorded."""
    if method not in FIT_METHODS:
        raise ValueError(f"unknown fit method {method!r}; expected one of {sorted(FIT_METHODS)}")
    model._last_fit_meta = {
        "method": next_state(method_of(model), method),
        "discrete": model._discrete,
        **facts,
    }


def annotate(model, **facts: Any) -> None:
    """Add facts to the recorded fit without moving its method."""
    model._last_fit_meta.update(facts)


def copy_record(source, target) -> None:
    """Give ``target`` a copy of ``source``'s record, where ``source`` holds one."""
    meta = getattr(source, "_last_fit_meta", None)
    if meta is not None:
        target._last_fit_meta = dict(meta)


def method_of(model) -> FitState:
    """The recorded fit method, or None for a model that holds no record."""
    return (getattr(model, "_last_fit_meta", None) or {}).get("method")


def traits_of(model) -> FitMethodTraits:
    """The recorded method's traits; a model without a known method reads as ``fit``."""
    return FIT_METHODS.get(method_of(model), FIT_METHODS["fit"])


def search_criterion(model) -> str | None:
    """The criterion of the smoothing search the model's fit holds, or None where it holds none.

    A search result is the recorded method's (``FitMethodTraits.criterion``). A model that
    holds a result and no recorded method was saved before fits recorded theirs, when REML
    was the only search, and reads as REML's.
    """
    if getattr(model, "_reml_result", None) is None:
        return None
    return traits_of(model).criterion or "reml"


def smoothing_held_from(model) -> FitMethod | None:
    """The search method whose chosen smoothing this ``fit`` holds, recorded by the editor's refit.

    None for any fit that did not record one: a search method's own fit, and a fit at
    smoothing the analyst set.
    """
    return (getattr(model, "_last_fit_meta", None) or {}).get("smoothing_held_from")


def check_search_result(model) -> None:
    """Refuse a recorded method whose smoothing search result has another criterion.

    Runs only where both exist: a ``fit_reml`` fit with nothing to estimate has no
    result, and a model saved before fits recorded their method has no tag. A result
    without a ``criterion`` predates the field and was made by REML.
    """
    result = getattr(model, "_reml_result", None)
    method = method_of(model)
    traits = FIT_METHODS.get(method) if method is not None else None
    if result is None or traits is None:
        return
    criterion = getattr(result, "criterion", "reml")
    if criterion != traits.criterion:
        raise RuntimeError(
            f"fit candidate smoothing result has criterion {criterion!r}, "
            f"not the one {method} searches by"
        )
