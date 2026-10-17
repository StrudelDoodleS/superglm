"""Gate for benchboard fit records: free on a machine without benchboard.

A fit method opens ``with fit_record(self, "<method>", locals()):`` around its
body, so no frame is added between the caller and the fit and every warning's
``stacklevel`` still lands on the caller.

Recording is on inside a benchboard job that captures fits
(``BENCHBOARD_FIT_DIR`` and ``BENCHBOARD_TASK_ID`` set), and on any machine
where benchboard's state directory exists (``BENCHBOARD_DIR``, default
``~/.cache/benchboard``), where a fit outside a job spools to
``fits/unowned/`` for benchboard to import.  ``SUPERGLM_FIT_RECORD=0`` turns it
off.  Otherwise the gate is a cached directory check and a shared null
context, and the recorder (``superglm.diagnostics._fit_record``) is never
imported.  Any failure inside the recorder is swallowed, and the fit's own
result or exception passes through unchanged.
"""

from __future__ import annotations

import contextlib
import os

_OFF = contextlib.nullcontext()
_unowned: dict[tuple[str | None, str | None], str | None] = {}


def destination() -> tuple[str, str | None] | None:
    """(spool directory, task id) for this fit's record, or None when recording is off.

    A None task id means the fit runs outside a benchboard job.
    """
    if os.environ.get("SUPERGLM_FIT_RECORD") == "0":
        return None
    spool, task = os.environ.get("BENCHBOARD_FIT_DIR"), os.environ.get("BENCHBOARD_TASK_ID")
    if spool and task:
        return spool, task
    key = (os.environ.get("BENCHBOARD_DIR"), os.environ.get("HOME"))
    if key not in _unowned:
        base = key[0] or os.path.join(os.path.expanduser("~"), ".cache", "benchboard")
        _unowned[key] = os.path.join(base, "fits", "unowned") if os.path.isdir(base) else None
    unowned = _unowned[key]
    return None if unowned is None else (unowned, None)


class _Recording:
    __slots__ = ("method", "model", "params", "place", "recorder", "token")

    def __init__(self, model, method, params, place):
        self.model, self.method, self.params, self.place = model, method, params, place
        self.recorder = self.token = None

    def __enter__(self):
        try:
            from superglm.diagnostics import _fit_record

            self.token = _fit_record.begin(self.model, self.method, self.params, self.place)
            self.recorder = _fit_record
        except Exception:  # noqa: BLE001, S110 - recording must never stop a fit
            pass
        return self

    def __exit__(self, kind, error, traceback):
        recorder = self.recorder
        if self.token is not None and recorder is not None:
            try:
                recorder.end(self.token, self.model, error)
            except Exception:  # noqa: BLE001, S110
                pass
        return False


def fit_record(model, method: str, params: dict):
    place = destination()
    # A power-search candidate (estimate_p under REML, and a later interval()) is ranked and
    # discarded; the estimate_p call that runs it is the record.
    if place is None or getattr(model, "_in_power_search", False):
        return _OFF
    return _Recording(model, method, params, place)
