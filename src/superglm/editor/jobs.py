"""Cancellable background jobs for the editor app.

Run CV and Final fit take one fit per fold, or one fit on every row, so they
run off the request thread, each on its own daemon thread, and the browser
polls them by id. A job never holds the widget lock while it works: ``work``
runs on state captured beforehand, and ``publish`` takes the lock itself and
stores the result only if that state is still current. A cancel is a flag
the work checks between steps (between folds for Run CV); once the work has
returned, publication can no longer be cancelled, so a cancelled job never
publishes anything. A finished job evicts the finished jobs of its kind
before it, so the runner keeps the last of each kind.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from superglm.editor.errors import EditorClientError, EditorKeyError, EditorValueError
from superglm.editor.io import jsonable

_LOGGER = logging.getLogger(__name__)

_ALREADY_RUNNING = "That job is already running. Wait for it, or cancel it first."
_UNKNOWN_JOB = "Unknown job. It may have finished and been replaced by a newer one."
_INTERNAL_ERROR = "internal editor error"


class JobCancelledError(Exception):
    """Raised by :meth:`JobContext.check` once the job's cancel flag is set."""


@dataclass(frozen=True)
class JobContext:
    """What a job's work sees: its id, a cancel check and a progress channel."""

    job_id: str
    _cancel: threading.Event = field(repr=False)
    _report: Callable[[dict[str, Any]], None] = field(repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        """Raise :class:`JobCancelledError` if a cancel was requested."""
        if self._cancel.is_set():
            raise JobCancelledError

    def progress(self, phase: str, **details: Any) -> None:
        """Append one progress entry the status route returns."""
        self._report({"phase": phase, **details})


class JobRunner:
    """Start, poll and cancel the editor's background jobs."""

    def __init__(self, *, name: str, wait_timeout: float = 30.0) -> None:
        self._name = name
        self._wait_timeout = float(wait_timeout)
        self._condition = threading.Condition(threading.RLock())
        self._jobs: dict[str, dict[str, Any]] = {}
        self._flags: dict[str, threading.Event] = {}
        self._counter = 0
        self._closed = False

    def start(
        self,
        kind: str,
        work: Callable[[JobContext], Any],
        publish: Callable[[Any], dict[str, Any]],
    ) -> str:
        """Run ``work`` on a new thread, then ``publish`` its value; return the job id."""
        with self._condition:
            if self._closed:
                raise RuntimeError("The job runner is closed.")
            if any(
                job["kind"] == kind and job["status"] == "running" for job in self._jobs.values()
            ):
                raise EditorValueError(_ALREADY_RUNNING)
            self._counter += 1
            job_id = f"{kind}-{self._counter}"
            flag = threading.Event()
            self._flags[job_id] = flag
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": kind,
                "status": "running",
                "progress": [],
                "result": None,
                "error": None,
                "cancel_requested": False,
                "publishing": False,
                "started_at": time.time(),
                "finished_at": None,
            }
        threading.Thread(
            target=self._run,
            args=(job_id, work, publish, flag),
            name=f"superglm-{self._name}-{job_id}",
            daemon=True,
        ).start()
        return job_id

    def status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
        """Return a job's status; with ``wait``, first wait until it stops running."""
        with self._condition:
            self._require(job_id)
            if wait:
                self._condition.wait_for(
                    lambda: self._jobs.get(job_id, {}).get("status") != "running",
                    timeout=self._wait_timeout,
                )
            return self._snapshot(self._require(job_id))

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a running job to stop at its next check; a finished job is left as it is."""
        with self._condition:
            job = self._require(job_id)
            if job["status"] == "running" and not job["publishing"]:
                self._flags[job_id].set()
                job["cancel_requested"] = True
                self._condition.notify_all()
            return {
                "job_id": job_id,
                "status": job["status"],
                "cancel_requested": job["cancel_requested"],
            }

    def latest(self, kind: str) -> dict[str, Any] | None:
        """The most recently started job of ``kind``, or None."""
        with self._condition:
            jobs = [job for job in self._jobs.values() if job["kind"] == kind]
            return self._snapshot(jobs[-1]) if jobs else None

    def close(self) -> None:
        """Ask every running job to stop; refuse new ones."""
        with self._condition:
            self._closed = True
            for flag in self._flags.values():
                flag.set()
            self._condition.notify_all()

    def _run(self, job_id, work, publish, flag: threading.Event) -> None:
        context = JobContext(job_id, flag, lambda entry: self._append(job_id, entry))
        try:
            value = work(context)
            with self._condition:
                if flag.is_set():
                    raise JobCancelledError
                self._jobs[job_id]["publishing"] = True
            result = publish(value)
        except JobCancelledError:
            self._finish(job_id, "cancelled")
        except EditorClientError as exc:
            self._finish(job_id, "failed", error=exc.public_message)
        except Exception:
            _LOGGER.exception("Unhandled SuperGLM editor job error.")
            self._finish(job_id, "failed", error=_INTERNAL_ERROR)
        else:
            self._finish(job_id, "done", result=result)

    def _append(self, job_id: str, entry: dict[str, Any]) -> None:
        with self._condition:
            self._jobs[job_id]["progress"].append(jsonable(entry))
            self._condition.notify_all()

    def _finish(self, job_id: str, status: str, *, result=None, error: str | None = None) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.update(
                status=status,
                result=jsonable(result),
                error=error,
                publishing=False,
                finished_at=time.time(),
            )
            self._flags.pop(job_id, None)
            evicted = [
                other_id
                for other_id, other in self._jobs.items()
                if other_id != job_id
                and other["kind"] == job["kind"]
                and other["status"] != "running"
            ]
            for other_id in evicted:
                del self._jobs[other_id]
            self._condition.notify_all()

    def _require(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(str(job_id))
        if job is None:
            raise EditorKeyError(_UNKNOWN_JOB)
        return job

    @staticmethod
    def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
        snapshot = {key: value for key, value in job.items() if key != "publishing"}
        snapshot["progress"] = list(job["progress"])
        return jsonable(snapshot)


__all__ = ["JobCancelledError", "JobContext", "JobRunner"]
