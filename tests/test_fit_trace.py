"""Unit tests for dependency-free authoritative fit-trace primitives."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from superglm._fit_trace import JSONLTraceSink, MemoryTraceSink, NullTraceSink, TraceRun
from superglm.model.reml_debug import load_reml_debug_run


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


def test_jsonl_sink_preserves_run_sequence_across_channels(tmp_path) -> None:
    sink = JSONLTraceSink(tmp_path, "run-1")
    run = TraceRun("run-1", sink=sink, clock=lambda: 0.0)
    run.emit(
        "evaluation",
        channel="pirls",
        state_id=run.next_state_id(),
        deviance=1.0,
    )
    run.emit(
        "evaluation",
        channel="reml",
        state_id=run.next_state_id(),
        objective=2.0,
    )

    loaded = load_reml_debug_run(tmp_path, "run-1")

    assert [event["sequence"] for event in loaded.events] == [1, 2]
    assert [event["channel"] for event in loaded.events] == ["pirls", "reml"]
    assert loaded.pirls_rows[0]["deviance"] == 1.0
    assert loaded.reml_rows[0]["objective"] == 2.0


def test_jsonl_sink_detaches_array_like_payloads(tmp_path) -> None:
    class ArrayLike:
        shape = (2,)

        def __init__(self, values):
            self.values = list(values)

        def copy(self):
            return ArrayLike(self.values)

        def setflags(self, *, write):
            assert not write

        def tolist(self):
            return list(self.values)

    sink = JSONLTraceSink(tmp_path, "run-1")
    run = TraceRun("run-1", sink=sink)
    values = ArrayLike([1.0, 2.0])
    run.emit("terminal", channel="fit", state_id=1, beta=values)
    values.values[0] = 99.0

    loaded = load_reml_debug_run(tmp_path, "run-1")
    assert loaded.events[0]["payload"]["beta"] == [1.0, 2.0]


def test_loader_accepts_legacy_rows_without_schema_version(tmp_path) -> None:
    (tmp_path / "old_run.json").write_text('{"method":"fit_reml"}', encoding="utf-8")
    (tmp_path / "old_reml.jsonl").write_text(
        '{"iteration":1,"objective_after":3.0}\n',
        encoding="utf-8",
    )

    loaded = load_reml_debug_run(tmp_path, "old")

    assert loaded.events == []
    assert loaded.reml_rows[0]["iteration"] == 1


def test_loader_rejects_duplicate_canonical_sequences(tmp_path) -> None:
    sink = JSONLTraceSink(tmp_path, "run-1")
    run = TraceRun("run-1", sink=sink)
    event = run.emit("evaluation", channel="pirls", state_id=1, deviance=1.0)
    assert event is not None
    pirls_path = tmp_path / "run-1_pirls.jsonl"
    row = pirls_path.read_text(encoding="utf-8")
    (tmp_path / "run-1_reml.jsonl").write_text(row, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate canonical trace sequence"):
        load_reml_debug_run(tmp_path, "run-1")


def test_loader_ignores_longer_run_ids_with_the_same_prefix(tmp_path) -> None:
    short = TraceRun("run", sink=JSONLTraceSink(tmp_path, "run"))
    long = TraceRun("run_child", sink=JSONLTraceSink(tmp_path, "run_child"))
    short.emit("evaluation", channel="pirls", state_id=1, deviance=1.0)
    long.emit("evaluation", channel="pirls", state_id=1, deviance=2.0)

    loaded = load_reml_debug_run(tmp_path, "run")

    assert len(loaded.events) == 1
    assert loaded.events[0]["payload"]["deviance"] == 1.0


def test_loader_rejects_noncontiguous_canonical_sequences(tmp_path) -> None:
    sink = JSONLTraceSink(tmp_path, "run-1")
    run = TraceRun("run-1", sink=sink)
    run.emit("evaluation", channel="pirls", state_id=1, deviance=1.0)
    run.emit("evaluation", channel="reml", state_id=2, objective=2.0)
    (tmp_path / "run-1_pirls.jsonl").unlink()

    with pytest.raises(ValueError, match="noncontiguous canonical trace sequence"):
        load_reml_debug_run(tmp_path, "run-1")
