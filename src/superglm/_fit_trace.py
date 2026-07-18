"""Dependency-free primitives for authoritative fit traces.

The numerical solvers use :class:`TraceRun` as a run-local allocator and event
sequencer.  A disabled run has one predictable branch and, through
``emit_lazy``, does not construct diagnostic payloads at all.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _require_safe_path_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SAFE_PATH_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe non-empty file-name component")
    return value


def _jsonable(value):
    """Project frozen event values onto JSON-compatible standard types."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, frozenset | set):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return value


class JSONLTraceSink:
    """Append canonical events to channel-local JSONL files.

    Sequence numbers are allocated by the shared :class:`TraceRun`, so rows in
    separate channel files still have one unambiguous total order.
    """

    enabled = True

    def __init__(self, base_dir: str | Path, run_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.run_id = _require_safe_path_component(run_id, name="run_id")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.base_dir / f"{self.run_id}_run.json"
        if not metadata_path.exists():
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": self.run_id,
                        "canonical_trace": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def append(self, event: TraceEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError(
                f"trace event run_id {event.run_id!r} does not match sink {self.run_id!r}"
            )
        channel = _require_safe_path_component(event.channel, name="channel")
        payload = {
            "schema_version": event.schema_version,
            "run_id": event.run_id,
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "event_kind": event.event_kind,
            "channel": event.channel,
            "purpose": event.purpose,
            "authoritative": event.authoritative,
            "payload": _jsonable(event.payload),
        }
        path = self.base_dir / f"{self.run_id}_{channel}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


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
    "JSONLTraceSink",
    "MemoryTraceSink",
    "NullTraceSink",
    "TraceEvent",
    "TraceRun",
    "TraceSink",
]
