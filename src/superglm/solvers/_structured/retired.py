"""Saved-model shims for retired structured classes.

A fitted model pickles its retained linear system. A deleted class named in
that state stays importable at its pickled path through the module's
``__getattr__`` (PEP 562) as an inert ``RetiredStructuredState`` subclass.
``SuperGLM.__setstate__`` drops such state
(``model.retired_state.mark_retired_linear_state``) and the first inference
call rebuilds it at the saved coefficients and smoothing parameters, with a
one-time notice. Predictions, coefficients and relativities never read it.
The same stand-ins serve the v0.35.0 Tweedie classes in
``superglm.reml.scale`` and ``superglm.profiling.tweedie``.
"""

from __future__ import annotations

from typing import Any

_RETIRED: dict[tuple[str, str], type] = {}

REBUILT_NOTICE = (
    "This model was saved by an earlier superglm build whose structured solver "
    "state has been retired: its linear system was rebuilt with the current solver "
    "at the saved coefficients and smoothing parameters, so standard errors, "
    "leverage and summaries use the current engine."
)


class RetiredStructuredState:
    """Inert stand-in for a retired structured solver object restored from a pickle."""

    retired_name: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__["_retired_args"] = (args, kwargs)

    def __setstate__(self, state: Any) -> None:
        self.__dict__["_retired_state"] = state

    def __repr__(self) -> str:
        return f"<retired structured state {self.retired_name!r}>"


class RetiredMethod:
    """A bound method of a retired object, restored from a pickle; calling it says so."""

    def __init__(self, owner: str, name: str) -> None:
        self.owner = owner
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{self.owner}.{self.name} was saved by an earlier superglm build and its code has "
            "been retired, so it cannot be evaluated; refit the model to recompute it."
        )


class RetiredSearchState(RetiredStructuredState):
    """Stand-in for retired search state that a pickle also binds methods of.

    A pickled bound method restores as ``getattr(instance, name)`` (Python
    ``pickle`` docs), so any public or private attribute of the stand-in
    answers with a ``RetiredMethod``; dunder lookups keep the default protocol.
    """

    def __getattr__(self, name: str) -> RetiredMethod:
        if name.startswith("__"):
            raise AttributeError(name)
        return RetiredMethod(self.retired_name, name)


def retired_class(
    module: str, name: str, base: type[RetiredStructuredState] = RetiredStructuredState
) -> type:
    """The ``RetiredStructuredState`` subclass standing in for ``module.name`` (one per name)."""
    key = (module, name)
    cls = _RETIRED.get(key)
    if cls is None:
        cls = type(name, (base,), {"retired_name": name, "__module__": module})
        _RETIRED[key] = cls
    return cls


def module_getattr(
    module: str, names: frozenset[str], base: type[RetiredStructuredState] = RetiredStructuredState
):
    """A module ``__getattr__`` that serves ``names`` as retired stand-ins (PEP 562)."""

    def module_attribute(name: str) -> type:
        if name in names:
            return retired_class(module, name, base)
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return module_attribute


def holds_retired_state(value: Any) -> bool:
    """Whether a retained linear state (or any of its fields) is a retired stand-in."""
    if value is None:
        return False
    if isinstance(value, RetiredStructuredState):
        return True
    fields = getattr(value, "__dict__", None)
    if not isinstance(fields, dict):
        return False
    return any(isinstance(item, RetiredStructuredState) for item in fields.values())
