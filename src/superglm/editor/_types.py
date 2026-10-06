"""Shared editor data types."""

from __future__ import annotations

import itertools
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Step ids are seven hex digits, n -> (n * M + salt) mod 2**28. M is odd, so it
# is a unit modulo 2**28 and the map is a bijection: the first 2**28 counters
# give distinct ids. M scatters consecutive steps; the salt differs per process.
_STEP_ID_BITS = 28
_STEP_ID_MULTIPLIER = 0x9E3779B
_STEP_ID_SALT = secrets.randbits(_STEP_ID_BITS)
_step_counter = itertools.count()
_step_counter_lock = threading.Lock()


def new_step_id() -> str:
    """Seven hex digits naming one history entry, unique within the process."""
    with _step_counter_lock:
        n = next(_step_counter)
    return f"{(n * _STEP_ID_MULTIPLIER + _STEP_ID_SALT) % (1 << _STEP_ID_BITS):07x}"


@dataclass
class EditableTerm:
    """Editable link-scale representation of one fitted 1D main effect."""

    name: str
    kind: str
    original_log_effect: NDArray
    edited_log_effect: NDArray
    x: NDArray | None = None
    levels: list[str] | None = None
    weights: NDArray | None = None
    ci_lower_log_effect: NDArray | None = None
    ci_upper_log_effect: NDArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.edited_log_effect.size)

    @property
    def relativity(self) -> NDArray:
        return np.exp(self.edited_log_effect)

    def copy(self) -> EditableTerm:
        return EditableTerm(
            name=self.name,
            kind=self.kind,
            original_log_effect=self.original_log_effect.copy(),
            edited_log_effect=self.edited_log_effect.copy(),
            x=None if self.x is None else self.x.copy(),
            levels=None if self.levels is None else list(self.levels),
            weights=None if self.weights is None else self.weights.copy(),
            ci_lower_log_effect=(
                None if self.ci_lower_log_effect is None else self.ci_lower_log_effect.copy()
            ),
            ci_upper_log_effect=(
                None if self.ci_upper_log_effect is None else self.ci_upper_log_effect.copy()
            ),
            metadata=dict(self.metadata),
        )


@dataclass
class EditRecord:
    """One reversible edit to a term."""

    term: str
    operation: str
    indices: NDArray[np.intp]
    before: NDArray
    after: NDArray
    params: dict[str, Any] = field(default_factory=dict)
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        """The automatic message, e.g. ``"shift area"``."""
        return f"{self.operation.replace('_', ' ')} {self.term}"


@dataclass(frozen=True)
class PendingStep:
    """One structural change waiting for a Refit (spec D1).

    ``params`` name levels, groups and edges by label, as the builder resolved
    them. ``draft_spec`` is the term's unfitted spec after this change, built
    on the term's previous draft; it is never fitted itself, because a refit
    fits a deep copy. ``history_position`` is ``len(history)`` when it was
    staged: its place in time among the live edits. ``metadata`` is the
    builder's step metadata, stamped on a refit that applies this change
    alone. ``predictor`` is reserved for the SuperLSS editor.
    """

    operation: str
    term: str
    label: str
    params: dict[str, Any]
    draft_spec: Any
    history_position: int
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    predictor: str | None = None


@dataclass(frozen=True)
class SessionState:
    """The editor state on one side of a structural step."""

    model: Any
    terms: dict[str, EditableTerm]
    selection: dict[str, NDArray[np.intp]]
    level_orders: dict[str, list[str]]
    history: list[EditRecord]
    redo_stack: list[EditRecord]
    pending: tuple[PendingStep, ...] = ()
    pending_redo: tuple[PendingStep, ...] = ()


@dataclass(frozen=True)
class StructuralStep:
    """One structural change on the undo timeline and the state on its far side.

    On the undo stack ``state`` is the state before the step; on the redo
    stack it is the state the step left. ``changes`` are the waiting changes a
    refit applied, oldest first; a change refitted at once on its own is the
    step itself and shares its ``step_id``.
    """

    state: SessionState
    operation: str
    term: str | None
    label: str
    changes: tuple[PendingStep, ...] = ()
    step_id: str = field(default_factory=new_step_id)
    created_at: float = field(default_factory=time.time)
