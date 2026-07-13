"""Bounded background execution for editor evidence cache misses."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    """Identify one evidence calculation for a model revision."""

    model_revision: int
    kind: str
    discriminator: str


@dataclass(frozen=True, slots=True)
class EvidenceOutcome:
    """Describe completed work or a pending request replaced by newer work."""

    status: Literal["complete", "superseded"]
    key: EvidenceKey
    payload: dict[str, Any] | None = None


@dataclass(slots=True)
class _WorkItem:
    key: EvidenceKey
    compute: Callable[[], dict[str, Any]]
    future: Future[EvidenceOutcome]


class EvidenceCoordinator:
    """Run one evidence calculation while retaining only the latest pending one."""

    def __init__(self, name: str) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"superglm-{name}",
        )
        self._running: _WorkItem | None = None
        self._pending: _WorkItem | None = None
        self._closed = False
        self._active = 0
        self._max_active = 0

    @property
    def max_active(self) -> int:
        """Return the greatest number of concurrently executing calculations."""
        with self._lock:
            return self._max_active

    def submit(
        self,
        key: EvidenceKey,
        compute: Callable[[], dict[str, Any]],
    ) -> Future[EvidenceOutcome]:
        """Return shared work for ``key`` or retain it as the latest pending work."""
        superseded: _WorkItem | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("Evidence coordinator is closed.")
            if self._running is not None and self._running.key == key:
                return self._running.future
            if self._pending is not None and self._pending.key == key:
                return self._pending.future

            item = _WorkItem(key, compute, Future())
            if self._running is None:
                self._start_locked(item)
            else:
                superseded = self._pending
                self._pending = item

        if superseded is not None:
            self._set_superseded(superseded)
        return item.future

    def _start_locked(self, item: _WorkItem) -> None:
        self._running = item
        worker = self._executor.submit(self._execute, item)
        worker.add_done_callback(lambda completed: self._finish(item, completed))

    def _execute(self, item: _WorkItem) -> EvidenceOutcome:
        with self._lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
        try:
            payload = item.compute()
            return EvidenceOutcome("complete", item.key, payload)
        finally:
            with self._lock:
                self._active -= 1

    def _finish(self, item: _WorkItem, worker: Future[EvidenceOutcome]) -> None:
        try:
            outcome = worker.result()
        except BaseException as exc:
            outcome = None
            error: BaseException | None = exc
        else:
            error = None

        with self._lock:
            if self._running is item:
                self._running = None
                pending = self._pending
                self._pending = None
                if pending is not None and not self._closed:
                    self._start_locked(pending)

        try:
            if error is not None:
                item.future.set_exception(error)
            else:
                assert outcome is not None
                item.future.set_result(outcome)
        except InvalidStateError:
            # A caller may cancel its public future while the private worker is running.
            pass

    @staticmethod
    def _set_superseded(item: _WorkItem) -> None:
        try:
            item.future.set_result(EvidenceOutcome("superseded", item.key))
        except InvalidStateError:
            # Cancellation is already a terminal state for the public future.
            pass

    def close(self) -> None:
        """Reject new work and supersede queued work without waiting for the runner."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = self._pending
            self._pending = None

        if pending is not None:
            self._set_superseded(pending)
        self._executor.shutdown(wait=False)


__all__ = ["EvidenceCoordinator", "EvidenceKey", "EvidenceOutcome"]
