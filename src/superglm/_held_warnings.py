"""Warnings a fit attempt holds back until its caller decides whether to keep the attempt.

A fit can try a start and discard it (the SCOP REML bootstrap,
``reml/scop_efs.py``), and a discarded start must not speak for the fit that
is published.  ``warnings.catch_warnings`` would hold its warnings back by
swapping the process-wide filters and ``showwarning``, which is not
thread-safe: two overlapping fits (the editor runs its jobs on threads) can
leave the process with the other's recording state, swallowing every later
warning.  The hold here is a context variable, so it belongs to one thread or
task and changes no global state.  Only superglm's own warnings are held:
those raised through ``warn`` inside it are recorded instead of shown.
Everything else passes through as it would without the hold: NumPy's
floating-point warnings under the caller's own ``np.seterr`` settings (and
their text), a custom family's warnings, a third-party library's.
``replay`` re-issues held warnings with
``warnings.warn_explicit`` at the frame each was raised from, with that
frame's module name and registry, exactly as ``warnings.warn`` would have, so
the caller's filters and once-per-location rules apply as if nothing had been
held.  The module name is the frame's ``__name__``, ``"<string>"`` without
one (as ``warnings.warn``), never ``None``, which Python 3.13 drops.
"""

from __future__ import annotations

import contextvars
import sys
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HeldWarning:
    """One warning, with the location ``warnings.warn`` would have given it."""

    message: Warning | str
    category: type[Warning]
    filename: str
    lineno: int
    module_globals: dict[str, Any]


_HOLD: contextvars.ContextVar[list[HeldWarning] | None] = contextvars.ContextVar(
    "superglm_held_warnings", default=None
)


def _record(held: list[HeldWarning], message, category, depth: int) -> None:
    """Record ``message`` at the frame ``depth`` levels above the caller of this function."""
    try:
        frame = sys._getframe(depth + 1)
    except ValueError:
        filename, lineno, module_globals = "sys", 1, sys.__dict__
    else:
        filename, lineno = frame.f_code.co_filename, frame.f_lineno
        module_globals = frame.f_globals
    if isinstance(message, Warning):
        category = type(message)
    held.append(HeldWarning(message, category, filename, lineno, module_globals))


def warn(message: Warning | str, category: type[Warning] = UserWarning, stacklevel: int = 1):
    """``warnings.warn``, unless this context holds warnings back: then it records it."""
    held = _HOLD.get()
    if held is None:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
        return
    _record(held, message, category, stacklevel)


@contextmanager
def hold() -> Iterator[list[HeldWarning]]:
    """Hold back, in this context only, superglm's own warnings raised inside the block."""
    held: list[HeldWarning] = []
    token = _HOLD.set(held)
    try:
        yield held
    finally:
        _HOLD.reset(token)


def replay(held: Iterable[HeldWarning]) -> None:
    """Issue held warnings where they were raised, through the caller's filters."""
    for record in held:
        module_globals = record.module_globals
        warnings.warn_explicit(
            record.message,
            record.category,
            record.filename,
            record.lineno,
            module=module_globals.get("__name__", "<string>"),
            registry=module_globals.setdefault("__warningregistry__", {}),
            module_globals=module_globals,
        )
