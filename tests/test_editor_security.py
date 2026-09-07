"""Browser errors must not disclose arbitrary backend exception details."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, SuperGLM
from superglm.editor import EditorSession


@pytest.fixture(scope="module")
def security_model():
    x = np.linspace(-1.0, 1.0, 30)
    frame = pd.DataFrame({"x": x})
    y = 1.0 + 0.4 * x + 0.05 * np.sin(9.0 * x)
    return SuperGLM(family="gaussian", selection_penalty=0.0, features={"x": Numeric()}).fit(
        frame, y
    )


@pytest.fixture
def security_widget(security_model):
    widget = EditorSession.from_model(security_model, terms=["x"]).widget()
    try:
        yield widget
    finally:
        widget.close()


def request_json(widget, path, payload=None):
    request = urllib.request.Request(
        f"{widget.url}{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"X-SuperGLM-Editor-Token": widget._token, "Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.code, json.loads(response.read())


@pytest.mark.parametrize(
    "exception_type", [ValueError, KeyError, TypeError, IndexError, FileNotFoundError, RuntimeError]
)
@pytest.mark.parametrize(
    "path,payload,method",
    [
        ("/op", {"operation": "reset"}, "_operate"),
        ("/download_export", None, "_export_bytes"),
        ("/download_model", None, "_export_bytes"),
    ],
)
def test_backend_errors_are_private(
    security_widget, monkeypatch, caplog, exception_type, path, payload, method
):
    detail = "synthetic backend detail /internal/model.bin"

    def fail(*_args, **_kwargs):
        raise exception_type(detail)

    monkeypatch.setattr(security_widget, method, fail)
    status, body = request_json(security_widget, path, payload)

    assert status == 500
    assert body == {"error": "internal editor error"}
    assert detail in caplog.text


def test_state_backend_error_uses_private_json_response(security_widget, monkeypatch, caplog):
    detail = "synthetic state failure /internal/model.bin"

    def fail():
        raise ValueError(detail)

    monkeypatch.setattr(security_widget, "_state", fail)
    status, body = request_json(security_widget, "/state")

    assert status == 500
    assert body == {"error": "internal editor error"}
    assert detail in caplog.text


@pytest.mark.parametrize("exception_type", [ValueError, FileNotFoundError, RuntimeError])
def test_background_profile_errors_are_private(
    security_widget, monkeypatch, caplog, exception_type
):
    detail = "synthetic profile failure /internal/model.bin"

    def fail(*_args, **_kwargs):
        raise exception_type(detail)

    monkeypatch.setattr(security_widget, "_profile_distribution", fail)
    status, started = request_json(
        security_widget, "/profile_distribution/start", {"parameter": "tweedie_p"}
    )
    assert status == 200
    status, completed = request_json(
        security_widget, f"/profile_distribution/status/{started['job_id']}?wait=true"
    )

    assert status == 200
    assert completed["status"] == "error"
    assert completed["error"] == "internal editor error"
    assert completed["finished_at"] is not None
    assert detail in caplog.text


@pytest.mark.parametrize(
    "path,payload,message",
    [
        ("/term", {"term": "missing"}, "Unknown editable term"),
        ("/select", {"term": "x", "indices": [-1]}, "indices out of range"),
        ("/op", {"operation": "missing"}, "Unknown editor operation"),
        ("/download_export?filename=../model.joblib", None, "directory separators"),
        ("/term", {}, "term"),
        ("/control", {"term": "x", "handle_index": "bad", "value": 1}, "handle_index"),
        ("/select", {"term": "x", "indices": ["bad"]}, "indices"),
        ("/drag", {"term": "x", "delta": "bad"}, "delta"),
        ("/summary", {"level_display": "bad"}, "level_display"),
    ],
)
def test_invalid_requests_keep_helpful_messages(security_widget, path, payload, message):
    status, body = request_json(security_widget, path, payload)

    assert status == 400
    assert message in body["error"]


def test_malformed_body_does_not_echo_exception_or_input(security_widget):
    status, body = request_json(security_widget, "/op", ["synthetic input detail"])

    assert status == 400
    assert body == {"error": "Invalid request body or parameters."}


def test_background_profile_keeps_deliberate_validation_message(security_widget):
    status, started = request_json(
        security_widget, "/profile_distribution/start", {"parameter": "missing"}
    )
    assert status == 200
    _, completed = request_json(
        security_widget, f"/profile_distribution/status/{started['job_id']}?wait=true"
    )

    assert completed["status"] == "error"
    assert "parameter must be" in completed["error"]


@pytest.mark.parametrize("mode", ["synchronous", "complete", "error"])
def test_profile_trace_backend_messages_stay_private(security_widget, monkeypatch, caplog, mode):
    caplog.set_level("DEBUG", logger="superglm.editor.widget")
    message = "synthetic optimizer failure /internal/model.bin"
    fallback = "synthetic fallback failure /internal/cache.bin"
    numerical = {"step": 0, "p": 1.5, "phi": 0.2, "nll": 0.1, "phi_converged": False}
    row = {**numerical, "phi_message": message, "phi_fallback_reason": fallback}
    result = SimpleNamespace(search_trace=[row])

    def profile(_parameter, **options):
        if "trace_callback" in options:
            options["trace_callback"](row)
        if mode == "error":
            raise RuntimeError("synthetic profile stopped")
        return result

    monkeypatch.setattr(security_widget.session, "reprofile_distribution", profile)
    if mode == "synchronous":
        status, body = request_json(
            security_widget, "/profile_distribution", {"parameter": "tweedie_p"}
        )
        assert body["profile_trace"] == [numerical]
    else:
        status, started = request_json(
            security_widget, "/profile_distribution/start", {"parameter": "tweedie_p"}
        )
        assert status == 200
        status, body = request_json(
            security_widget, f"/profile_distribution/status/{started['job_id']}?wait=true"
        )
        assert body["status"] == mode
        assert body["trace"] == [numerical]
        if mode == "complete":
            assert body["result"]["profile_trace"] == [numerical]
        else:
            assert body["error"] == "internal editor error"

    assert status == 200
    assert message not in json.dumps(body)
    assert fallback not in json.dumps(body)
    assert message in caplog.text
    assert fallback in caplog.text
    assert result.search_trace == [row]
    assert row["phi_message"] == message
    assert row["phi_fallback_reason"] == fallback
