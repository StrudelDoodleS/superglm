from __future__ import annotations

import gc
import threading
import weakref
from dataclasses import FrozenInstanceError

import pytest

from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey, EvidenceOutcome


def test_evidence_records_are_immutable():
    key = EvidenceKey(1, "metrics", "validation")
    outcome = EvidenceOutcome("complete", key, {"value": 1})

    with pytest.raises(FrozenInstanceError):
        key.model_revision = 2
    with pytest.raises(FrozenInstanceError):
        outcome.status = "superseded"


def test_same_running_evidence_key_shares_one_computation():
    coordinator = EvidenceCoordinator("dedup-running")
    release = threading.Event()
    started = threading.Event()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)
        return {"value": 1}

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), compute)
        assert started.wait(timeout=5)
        second = coordinator.submit(EvidenceKey(1, "metrics", "validation"), compute)
        assert first is second
        release.set()
        assert first.result(timeout=5).payload == {"value": 1}
        assert calls == 1
    finally:
        release.set()
        coordinator.close()


def test_same_pending_evidence_key_shares_one_computation():
    coordinator = EvidenceCoordinator("dedup-pending")
    release = threading.Event()
    started = threading.Event()
    calls: list[str] = []

    def run_first():
        calls.append("first")
        started.set()
        release.wait(timeout=5)
        return {"value": 1}

    def run_pending():
        calls.append("pending")
        return {"value": 2}

    def should_not_run():
        calls.append("duplicate")
        return {"value": 3}

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), run_first)
        assert started.wait(timeout=5)
        pending = coordinator.submit(EvidenceKey(2, "metrics", "validation"), run_pending)
        duplicate = coordinator.submit(EvidenceKey(2, "metrics", "validation"), should_not_run)
        assert pending is duplicate
        release.set()
        assert first.result(timeout=5).status == "complete"
        assert pending.result(timeout=5).payload == {"value": 2}
        assert calls == ["first", "pending"]
    finally:
        release.set()
        coordinator.close()


def test_latest_pending_revision_replaces_intermediate_work():
    coordinator = EvidenceCoordinator("coalesce")
    release = threading.Event()
    started = threading.Event()
    calls: list[int] = []

    def compute(revision):
        def run():
            calls.append(revision)
            if revision == 1:
                started.set()
                release.wait(timeout=5)
            return {"revision": revision}

        return run

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), compute(1))
        assert started.wait(timeout=5)
        second = coordinator.submit(EvidenceKey(2, "metrics", "validation"), compute(2))
        third = coordinator.submit(EvidenceKey(3, "metrics", "validation"), compute(3))
        second_outcome = second.result(timeout=5)
        assert second_outcome == EvidenceOutcome(
            "superseded", EvidenceKey(2, "metrics", "validation")
        )
        assert second_outcome.payload is None
        release.set()
        assert first.result(timeout=5).status == "complete"
        assert third.result(timeout=5).payload == {"revision": 3}
        assert calls == [1, 3]
        assert coordinator.max_active == 1
    finally:
        release.set()
        coordinator.close()


def test_superseding_pending_work_releases_its_compute_payload():
    coordinator = EvidenceCoordinator("release-pending")
    release = threading.Event()
    started = threading.Event()

    class CapturedPayload:
        pass

    def run_first():
        started.set()
        release.wait(timeout=5)
        return {"revision": 1}

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), run_first)
        assert started.wait(timeout=5)

        captured = CapturedPayload()
        captured_ref = weakref.ref(captured)

        def run_pending(value=captured):
            return {"value": value}

        second = coordinator.submit(EvidenceKey(2, "metrics", "validation"), run_pending)
        del captured
        del run_pending

        third = coordinator.submit(EvidenceKey(3, "metrics", "validation"), lambda: {"revision": 3})
        assert second.result(timeout=5).status == "superseded"
        gc.collect()
        assert captured_ref() is None

        release.set()
        assert first.result(timeout=5).status == "complete"
        assert third.result(timeout=5).status == "complete"
    finally:
        release.set()
        coordinator.close()


def test_compute_exception_propagates_and_pending_work_still_runs():
    coordinator = EvidenceCoordinator("exception")
    release = threading.Event()
    started = threading.Event()
    error = ValueError("evidence failed")

    def fail():
        started.set()
        release.wait(timeout=5)
        raise error

    try:
        failed = coordinator.submit(EvidenceKey(1, "metrics", "validation"), fail)
        assert started.wait(timeout=5)
        pending = coordinator.submit(EvidenceKey(2, "metrics", "validation"), lambda: {"value": 2})
        release.set()

        with pytest.raises(ValueError, match="evidence failed") as exc_info:
            failed.result(timeout=5)
        assert exc_info.value is error
        assert pending.result(timeout=5).payload == {"value": 2}
        assert coordinator.max_active == 1
    finally:
        release.set()
        coordinator.close()


def test_close_is_idempotent_and_supersedes_pending_work():
    coordinator = EvidenceCoordinator("close")
    release = threading.Event()
    started = threading.Event()

    def run_first():
        started.set()
        release.wait(timeout=5)
        return {"value": 1}

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), run_first)
        assert started.wait(timeout=5)
        pending = coordinator.submit(EvidenceKey(2, "metrics", "validation"), lambda: {"value": 2})

        coordinator.close()
        coordinator.close()

        assert pending.result(timeout=5) == EvidenceOutcome(
            "superseded", EvidenceKey(2, "metrics", "validation")
        )
        with pytest.raises(RuntimeError, match="Evidence coordinator is closed"):
            coordinator.submit(EvidenceKey(3, "metrics", "validation"), lambda: {})

        release.set()
        assert first.result(timeout=5).payload == {"value": 1}
    finally:
        release.set()
        coordinator.close()


def test_close_from_completion_callback_allows_promoted_work_to_finish():
    coordinator = EvidenceCoordinator("close-promoted")
    release = threading.Event()
    started = threading.Event()
    promoted_calls = 0

    def run_first():
        started.set()
        release.wait(timeout=5)
        return {"value": 1}

    def run_promoted():
        nonlocal promoted_calls
        promoted_calls += 1
        return {"value": 2}

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), run_first)
        assert started.wait(timeout=5)
        promoted = coordinator.submit(EvidenceKey(2, "metrics", "validation"), run_promoted)
        first.add_done_callback(lambda _completed: coordinator.close())

        release.set()
        assert first.result(timeout=5).status == "complete"
        assert promoted.result(timeout=5).payload == {"value": 2}
        assert promoted_calls == 1
    finally:
        release.set()
        coordinator.close()
