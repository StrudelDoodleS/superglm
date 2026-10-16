"""Gate for benchboard fit records: free unless ``BENCHBOARD_FIT_DIR`` is set.

A fit method opens ``with fit_record(self, "<method>", locals()):`` around its
body, so no frame is added between the caller and the fit and every warning's
``stacklevel`` still lands on the caller.  With the variable unset this is one
environment lookup and a shared null context, and the recorder
(``superglm.diagnostics._fit_record``) is never imported.  With it set, the
recorder spools one record per fit; any failure inside the recorder is
swallowed, and the fit's own result or exception passes through unchanged.
"""

from __future__ import annotations

import contextlib
import os

_OFF = contextlib.nullcontext()


class _Recording:
    __slots__ = ("method", "model", "params", "recorder", "token")

    def __init__(self, model, method, params):
        self.model, self.method, self.params = model, method, params
        self.recorder = self.token = None

    def __enter__(self):
        try:
            from superglm.diagnostics import _fit_record

            self.token = _fit_record.begin(self.model, self.method, self.params)
            self.recorder = _fit_record
        except Exception:  # noqa: BLE001, S110 - recording must never stop a fit
            pass
        return self

    def __exit__(self, kind, error, traceback):
        if self.token is not None:
            try:
                self.recorder.end(self.token, self.model, error)
            except Exception:  # noqa: BLE001, S110
                pass
        return False


def fit_record(model, method: str, params: dict):
    if not os.environ.get("BENCHBOARD_FIT_DIR"):
        return _OFF
    return _Recording(model, method, params)
