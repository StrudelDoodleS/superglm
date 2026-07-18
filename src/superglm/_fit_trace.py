"""Dependency-free primitives for authoritative fit traces.

The numerical solvers use :class:`TraceRun` as a run-local allocator and event
sequencer.  A disabled run has one predictable branch and, through
``emit_lazy``, does not construct diagnostic payloads at all.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

SCHEMA_VERSION = 1

_EVENT_KINDS = frozenset(
    {
        "evaluation",
        "step_decision",
        "state_commit",
        "terminal",
        "run_failed",
    }
)
_STATE_BEARING_EVENT_KINDS = frozenset({"evaluation", "state_commit", "terminal"})


@dataclass(frozen=True)
class TraceEvent:
    """One immutable event in a globally ordered fit run."""

    schema_version: int
    run_id: str
    sequence: int
    timestamp: float
    event_kind: str
    channel: str
    purpose: str
    authoritative: bool
    payload: Mapping[str, object]


class TraceSink(Protocol):
    """Minimal sink contract used by :class:`TraceRun`."""

    enabled: bool

    def append(self, event: TraceEvent) -> None:
        """Persist or retain ``event``."""


class NullTraceSink:
    """Disabled sink whose append operation is allocation-free."""

    enabled = False

    def append(self, event: TraceEvent) -> None:
        del event


@dataclass
class MemoryTraceSink:
    """In-memory sink intended for tests and compact diagnostics."""

    events: list[TraceEvent] = field(default_factory=list)
    enabled: bool = True

    def append(self, event: TraceEvent) -> None:
        self.events.append(event)


def _freeze_value(value):
    """Detach common mutable containers without importing numerical packages."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)

    # Numerical arrays appear only in deliberately requested diagnostic or
    # terminal payloads.  Duck typing keeps this module dependency-free while
    # ensuring later solver mutation cannot rewrite trace history.
    copy = getattr(value, "copy", None)
    setflags = getattr(value, "setflags", None)
    if callable(copy) and callable(setflags) and hasattr(value, "shape"):
        detached = copy()
        detached.setflags(write=False)
        return detached
    return value


def _freeze_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze_value(value) for key, value in payload.items()})


PayloadFactory = Callable[[], Mapping[str, object]]


class TraceRun:
    """Allocate identities and serialize events for one fit attempt."""

    def __init__(
        self,
        run_id: str,
        *,
        sink: TraceSink | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        self.sink: TraceSink = sink if sink is not None else NullTraceSink()
        self.clock = clock
        self._sequence = 0
        self._state_id = 0
        self._evaluation_id = 0
        self._basis_id = 0

    @property
    def enabled(self) -> bool:
        """Whether this run will materialize and append events."""
        return self.sink.enabled

    def next_state_id(self) -> int:
        self._state_id += 1
        return self._state_id

    def next_evaluation_id(self) -> int:
        self._evaluation_id += 1
        return self._evaluation_id

    def next_basis_id(self) -> int:
        self._basis_id += 1
        return self._basis_id

    def emit(
        self,
        event_kind: str,
        *,
        channel: str,
        purpose: str = "fit",
        authoritative: bool = True,
        **payload: object,
    ) -> TraceEvent | None:
        """Validate, freeze, and append an event when tracing is enabled."""
        if not self.enabled:
            return None
        if event_kind not in _EVENT_KINDS:
            raise ValueError(f"unknown event_kind {event_kind!r}")
        if not isinstance(channel, str) or not channel:
            raise ValueError("channel must be a non-empty string")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError("purpose must be a non-empty string")
        if not isinstance(authoritative, bool):
            raise TypeError("authoritative must be bool")

        if event_kind in _STATE_BEARING_EVENT_KINDS:
            if "state_id" not in payload:
                raise ValueError(f"{event_kind} requires state_id")
            state_id = payload.get("state_id")
            if not isinstance(state_id, int) or isinstance(state_id, bool):
                raise TypeError(f"{event_kind} state_id must be an integer")
            if state_id <= 0:
                raise ValueError(f"{event_kind} state_id must be a positive integer")

        self._sequence += 1
        event = TraceEvent(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            sequence=self._sequence,
            timestamp=float(self.clock()),
            event_kind=event_kind,
            channel=channel,
            purpose=purpose,
            authoritative=authoritative,
            payload=_freeze_payload(payload),
        )
        self.sink.append(event)
        return event

    def emit_lazy(
        self,
        event_kind: str,
        payload_factory: PayloadFactory,
        *,
        channel: str,
        purpose: str = "fit",
        authoritative: bool = True,
    ) -> TraceEvent | None:
        """Append a lazily constructed event, skipping the factory when disabled."""
        if not self.enabled:
            return None
        payload = payload_factory()
        if not isinstance(payload, Mapping):
            raise TypeError("payload_factory must return a mapping")
        return self.emit(
            event_kind,
            channel=channel,
            purpose=purpose,
            authoritative=authoritative,
            **payload,
        )


__all__ = [
    "SCHEMA_VERSION",
    "MemoryTraceSink",
    "NullTraceSink",
    "TraceEvent",
    "TraceRun",
    "TraceSink",
]
