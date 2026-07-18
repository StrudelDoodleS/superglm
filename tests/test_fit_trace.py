"""Unit tests for dependency-free authoritative fit-trace primitives."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from superglm._fit_trace import MemoryTraceSink, NullTraceSink, TraceRun


def test_trace_run_assigns_one_sequence_across_channels() -> None:
    sink = MemoryTraceSink()
    run = TraceRun("run-1", sink=sink, clock=lambda: 12.5)

    first = run.emit(
        "evaluation",
        channel="pirls",
        state_id=run.next_state_id(),
        deviance=2.0,
    )
    second = run.emit(
        "evaluation",
        channel="reml",
        state_id=run.next_state_id(),
        objective=4.0,
    )

    assert first is sink.events[0]
    assert second is sink.events[1]
    assert [event.sequence for event in sink.events] == [1, 2]
    assert [event.timestamp for event in sink.events] == [12.5, 12.5]
    assert [event.payload["state_id"] for event in sink.events] == [1, 2]


def test_trace_run_allocates_independent_monotonic_id_spaces() -> None:
    run = TraceRun("run-1")

    assert [run.next_state_id(), run.next_state_id()] == [1, 2]
    assert [run.next_evaluation_id(), run.next_evaluation_id()] == [1, 2]
    assert [run.next_basis_id(), run.next_basis_id()] == [1, 2]


def test_null_sink_never_materializes_lazy_payload() -> None:
    called = False

    def payload() -> dict[str, object]:
        nonlocal called
        called = True
        return {"state_id": 1, "beta": [1.0]}

    event = TraceRun("run-1", sink=NullTraceSink()).emit_lazy(
        "evaluation",
        payload,
        channel="pirls",
    )

    assert event is None
    assert not called


def test_numerical_event_requires_state_identity() -> None:
    run = TraceRun("run-1", sink=MemoryTraceSink())

    with pytest.raises(ValueError, match="state_id"):
        run.emit("evaluation", channel="pirls", deviance=1.0)


@pytest.mark.parametrize("event_kind", ["evaluation", "state_commit", "terminal"])
def test_state_bearing_events_require_positive_integer_identity(event_kind: str) -> None:
    run = TraceRun("run-1", sink=MemoryTraceSink())

    with pytest.raises(ValueError, match="positive integer"):
        run.emit(event_kind, channel="pirls", state_id=0)
    with pytest.raises(TypeError, match="integer"):
        run.emit(event_kind, channel="pirls", state_id="1")


def test_unknown_event_kind_and_empty_channel_are_rejected() -> None:
    run = TraceRun("run-1", sink=MemoryTraceSink())

    with pytest.raises(ValueError, match="event_kind"):
        run.emit("made_up", channel="pirls", state_id=1)
    with pytest.raises(ValueError, match="channel"):
        run.emit("evaluation", channel="", state_id=1)


def test_trace_event_and_nested_payload_are_immutable() -> None:
    source = {"state_id": 1, "lambdas": [1.0, 2.0], "meta": {"phase": "outer"}}
    sink = MemoryTraceSink()
    event = TraceRun("run-1", sink=sink).emit("evaluation", channel="reml", **source)
    assert event is not None

    source["lambdas"].append(3.0)
    source["meta"]["phase"] = "changed"
    assert event.payload["lambdas"] == (1.0, 2.0)
    assert event.payload["meta"]["phase"] == "outer"

    with pytest.raises(TypeError):
        event.payload["new"] = "value"
    with pytest.raises(TypeError):
        event.payload["meta"]["phase"] = "changed"
    with pytest.raises(FrozenInstanceError):
        event.sequence = 100


def test_lazy_enabled_sink_materializes_payload_once() -> None:
    calls = 0
    sink = MemoryTraceSink()

    def payload() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"state_id": 1, "deviance": 2.0}

    event = TraceRun("run-1", sink=sink).emit_lazy(
        "evaluation",
        payload,
        channel="pirls",
        purpose="fit",
        authoritative=False,
    )

    assert calls == 1
    assert event is sink.events[0]
    assert event.purpose == "fit"
    assert not event.authoritative


def test_run_id_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="run_id"):
        TraceRun("")
