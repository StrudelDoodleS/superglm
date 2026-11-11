from __future__ import annotations

import gc
import threading
import weakref
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.evaluation import default_metrics_dataset
from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey, EvidenceOutcome


@pytest.fixture
def evidence_widget():
    rng = np.random.default_rng(20260805)
    n = 180
    X = pd.DataFrame({"x_spline": np.linspace(0.0, 10.0, n)})
    y = 0.4 + 0.2 * np.sin(X["x_spline"].to_numpy()) + rng.normal(0.0, 0.03, n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x_spline": Spline(n_knots=7)},
    )
    model.fit(X, y)
    session = EditorSession.from_model(
        model,
        terms=["x_spline"],
        train_data=(X, y, None, None),
    )
    session.select_indices("x_spline", [10, 11, 12])
    session.shift("x_spline", 0.2)
    widget = session.widget()
    try:
        yield widget
    finally:
        widget.close()


def _assert_selection_completes_while_blocked(widget, evidence_thread):
    selected = threading.Event()
    errors: list[BaseException] = []

    def select():
        try:
            widget._select("x_spline", [20, 21])
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            selected.set()

    selection_thread = threading.Thread(target=select)
    selection_thread.start()
    assert selected.wait(timeout=0.1)
    selection_thread.join(timeout=1)
    assert errors == []
    assert evidence_thread.is_alive()


def test_materialization_does_not_hold_widget_lock(evidence_widget, monkeypatch):
    import superglm.editor.apply as apply_module

    widget = evidence_widget
    dataset = default_metrics_dataset(widget.session)
    assert dataset is not None
    started = threading.Event()
    release = threading.Event()
    captured: dict[str, object] = {}
    errors: list[BaseException] = []
    original = apply_module.apply_edits_to_model_copy_with_data

    def blocked(model, terms, **kwargs):
        captured.update(kwargs)
        started.set()
        release.wait(timeout=5)
        return original(model, terms, **kwargs)

    def request_metrics():
        try:
            widget._metrics("deviance", "in_force")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(apply_module, "apply_edits_to_model_copy_with_data", blocked)
    evidence_thread = threading.Thread(target=request_metrics)
    evidence_thread.start()
    try:
        assert started.wait(timeout=5)
        _assert_selection_completes_while_blocked(widget, evidence_thread)
    finally:
        release.set()
        evidence_thread.join(timeout=5)

    assert errors == []
    assert captured["X"] is dataset.X
    assert captured["y"] is dataset.y
    assert captured["sample_weight"] is dataset.sample_weight
    assert captured["offset"] is dataset.offset


def test_scoring_does_not_hold_widget_lock(evidence_widget, monkeypatch):
    import superglm.editor.metrics as metrics_module

    widget = evidence_widget
    widget._summary("in_force")
    dataset = default_metrics_dataset(widget.session)
    assert dataset is not None
    started = threading.Event()
    release = threading.Event()
    captured_datasets = []
    errors: list[BaseException] = []
    original = metrics_module.compute_dataset_metrics

    def blocked(model, captured_dataset):
        captured_datasets.append(captured_dataset)
        started.set()
        release.wait(timeout=5)
        return original(model, captured_dataset)

    def request_metrics():
        try:
            widget._metrics("deviance", "in_force")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(metrics_module, "compute_dataset_metrics", blocked)
    evidence_thread = threading.Thread(target=request_metrics)
    evidence_thread.start()
    try:
        assert started.wait(timeout=5)
        _assert_selection_completes_while_blocked(widget, evidence_thread)
    finally:
        release.set()
        evidence_thread.join(timeout=5)

    assert errors == []
    assert captured_datasets
    captured_dataset = captured_datasets[0]
    assert captured_dataset is dataset
    assert captured_dataset.X is dataset.X
    assert captured_dataset.y is dataset.y
    assert captured_dataset.sample_weight is dataset.sample_weight
    assert captured_dataset.offset is dataset.offset


def test_summary_does_not_hold_widget_lock(evidence_widget, monkeypatch):
    widget = evidence_widget
    model, _revision = widget._current_model_for_evidence()
    assert model is not None
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original = model.summary

    def blocked(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    def request_summary():
        try:
            widget._summary("in_force")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(model, "summary", blocked)
    evidence_thread = threading.Thread(target=request_summary)
    evidence_thread.start()
    try:
        assert started.wait(timeout=5)
        _assert_selection_completes_while_blocked(widget, evidence_thread)
    finally:
        release.set()
        evidence_thread.join(timeout=5)

    assert errors == []


def test_download_serialization_does_not_hold_widget_lock(evidence_widget, monkeypatch):
    import joblib

    widget = evidence_widget
    model, _revision = widget._current_model_for_evidence()
    assert model is not None
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original = joblib.dump

    def blocked(value, target):
        started.set()
        release.wait(timeout=5)
        return original(value, target)

    def request_download():
        try:
            widget._export_bytes("joblib", "edited.joblib")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(joblib, "dump", blocked)
    evidence_thread = threading.Thread(target=request_download)
    evidence_thread.start()
    try:
        assert started.wait(timeout=5)
        _assert_selection_completes_while_blocked(widget, evidence_thread)
    finally:
        release.set()
        evidence_thread.join(timeout=5)

    assert errors == []


def test_report_is_superseded_when_revision_changes_during_summary(
    evidence_widget,
    monkeypatch,
):
    widget = evidence_widget
    model, source = widget._current_model_for_evidence()
    revision = widget.session.model_revision
    assert model is not None
    started = threading.Event()
    release = threading.Event()
    payloads = []
    errors: list[BaseException] = []
    original = model.summary

    def blocked(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    def request_report():
        try:
            payloads.append(
                widget._report(
                    "final",
                    model_revision=revision,
                    request_sequence=17,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(model, "summary", blocked)
    evidence_thread = threading.Thread(target=request_report)
    evidence_thread.start()
    try:
        assert started.wait(timeout=5)
        widget._operate("shift_up", "x_spline")
        assert evidence_thread.is_alive()
    finally:
        release.set()
        evidence_thread.join(timeout=5)

    assert errors == []
    assert payloads == [
        {
            "status": "superseded",
            "model_revision": revision,
            "request_sequence": 17,
            "in_force": source.token,
        }
    ]


def test_a_report_asked_for_before_an_edit_and_its_undo_is_answered(evidence_widget, monkeypatch):
    """An answer describes the model it was asked about, whatever the revision says.

    The page names that model by its ``in_force`` token. An edit and its undo while the report is
    built, or before the request reaches the server, put the very model back, so the report is
    answered and names the same model; a request about a model no longer in force is superseded.
    """
    widget = evidence_widget
    model, source = widget._current_model_for_evidence()
    asked = widget._state()
    started, release = threading.Event(), threading.Event()
    payloads, errors = [], []
    original = model.summary

    def blocked(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return original(*args, **kwargs)

    def edit_and_undo():
        widget._operate("shift_up", "x_spline")
        widget._operate("undo", "x_spline")

    def request_report():
        try:
            payloads.append(
                widget._report(
                    "final",
                    model_revision=asked["model_revision"],
                    request_sequence=17,
                    in_force=asked["in_force"],
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(model, "summary", blocked)
    evidence_thread = threading.Thread(target=request_report)
    evidence_thread.start()
    try:
        assert started.wait(timeout=5)
        edit_and_undo()
    finally:
        release.set()
        evidence_thread.join(timeout=5)
    monkeypatch.undo()
    late = widget._report(
        "validation", model_revision=asked["model_revision"], in_force=asked["in_force"]
    )
    widget._operate("shift_up", "x_spline")
    edited = widget._report(
        "validation", model_revision=asked["model_revision"], in_force=asked["in_force"]
    )

    assert errors == []
    assert asked["in_force"] == source.token
    assert widget.session.model_revision == asked["model_revision"] + 3
    assert [payload.get("status") for payload in payloads] == [None]
    assert (payloads[0]["in_force"], payloads[0]["request_sequence"]) == (source.token, 17)
    assert (late.get("status"), late["in_force"]) == (None, source.token)
    assert (edited["status"], edited["in_force"]) == ("superseded", source.token)


def test_a_final_report_keeps_the_edit_state_of_the_model_it_describes(monkeypatch):
    """An edited model's Final report shows the summary refitted on the training rows.

    Whether the model is edited is decided with the model the report describes. An edit back to
    the fitted values and its undo while the report is built put that model back, so they must
    not give it the unedited model's summary, read on the validation rows.
    """
    import superglm.editor.widget as widget_module

    rng = np.random.default_rng(20260805)
    X = pd.DataFrame({"x_spline": rng.uniform(0.0, 10.0, 240)})
    y = 0.4 + 0.2 * np.sin(X["x_spline"].to_numpy()) + rng.normal(0.0, 0.03, 240)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x_spline": Spline(n_knots=7)},
    ).fit(X.iloc[:180], y[:180])
    session = EditorSession.from_model(
        model,
        terms=["x_spline"],
        train_data=(X.iloc[:180], y[:180], None, None),
        validation_data=(X.iloc[180:], y[180:], None),
    )
    session.select_indices("x_spline", [10, 11, 12])
    session.shift("x_spline", 0.2)
    widget = session.widget()
    datasets = widget_module.evaluation_datasets
    materialize, score = (
        widget._model_materialized_for_dataset,
        widget._dataset_metrics_for_evidence,
    )
    moves: list[str] = []

    def edit_back_while_deciding(*args):
        if not moves:
            moves.append("back")
            with widget._lock:
                session.shift("x_spline", -0.2)
        return datasets(*args)

    def undo_then(step):
        def undone(*args):
            if moves == ["back"]:
                moves.append("undo")
                with widget._lock:
                    session.undo()
            return step(*args)

        return undone

    try:
        asked = widget._state()
        steady = widget._report("final", in_force=asked["in_force"])
        monkeypatch.setattr(widget_module, "evaluation_datasets", edit_back_while_deciding)
        monkeypatch.setattr(widget, "_model_materialized_for_dataset", undo_then(materialize))
        monkeypatch.setattr(widget, "_dataset_metrics_for_evidence", undo_then(score))
        raced = widget._report("final", in_force=asked["in_force"])
    finally:
        monkeypatch.undo()
        widget.close()

    assert moves == ["back", "undo"]
    assert steady["summary"]["compact"]["model"]["n_obs"] == 180
    assert (raced.get("status"), raced["in_force"]) == (None, asked["in_force"])
    assert raced["summary"]["compact"]["model"] == steady["summary"]["compact"]["model"]


class _FinalFitContext:
    """A job context that never cancels."""

    def check(self):
        return None

    def progress(self, *args, **kwargs):
        return None


def _split_session():
    """A Gaussian spline with 180 training and 60 validation rows, nothing edited."""
    rng = np.random.default_rng(20260805)
    X = pd.DataFrame({"x_spline": rng.uniform(0.0, 10.0, 240)})
    y = 0.4 + 0.2 * np.sin(X["x_spline"].to_numpy()) + rng.normal(0.0, 0.03, 240)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x_spline": Spline(n_knots=7)},
    ).fit(X.iloc[:180], y[:180])
    session = EditorSession.from_model(
        model,
        terms=["x_spline"],
        train_data=(X.iloc[:180], y[:180], None, None),
        validation_data=(X.iloc[180:], y[180:], None),
    )
    session.select_indices("x_spline", [10, 11, 12])
    return session


def _run_final_fit(widget):
    from superglm.editor.cv import capture_final_fit, run_final_fit

    with widget._lock:
        plan = capture_final_fit(widget.session)
    widget._final_fit = run_final_fit(plan, _FinalFitContext())


def _report_with_a_change_put_back(monkeypatch, widget, report, change, put_back):
    """The steady reply, then the reply when ``change`` lands as the report is built and
    ``put_back`` restores the model before the final check."""
    import superglm.editor.widget as widget_module

    build = widget_module.report_payload
    moves: list[str] = []

    def build_between(*args, **kwargs):
        with widget._lock:
            change()
        moves.append("change")
        try:
            return build(*args, **kwargs)
        finally:
            with widget._lock:
                put_back()
            moves.append("put back")

    asked = widget._state()
    steady = widget._report(report, in_force=asked["in_force"])
    monkeypatch.setattr(widget_module, "report_payload", build_between)
    try:
        raced = widget._report(report, in_force=asked["in_force"])
    finally:
        monkeypatch.undo()
    assert moves == ["change", "put back"]
    assert (steady.get("status"), steady["in_force"]) == (None, asked["in_force"])
    assert (raced.get("status"), raced["in_force"]) == (None, asked["in_force"])
    return steady, raced


def test_a_validation_report_compares_the_edit_of_the_model_it_describes(monkeypatch):
    """An edit back to the fitted values and its undo while the report is built must not hide the
    edited model's change column."""
    session = _split_session()
    session.shift("x_spline", 0.2)
    widget = session.widget()
    try:
        steady, raced = _report_with_a_change_put_back(
            monkeypatch,
            widget,
            "validation",
            lambda: session.shift("x_spline", -0.2),
            session.undo,
        )
    finally:
        widget.close()

    assert steady["compares_edit"] is True
    assert raced["compares_edit"] is True
    assert raced["splits"] == steady["splits"]


@pytest.mark.parametrize("fitted_edited", [False, True], ids=["fit-then-edit", "fit-on-edit"])
def test_a_final_report_says_whether_the_final_fit_describes_its_model(monkeypatch, fitted_edited):
    """A Final fit is stale exactly when it was run on another model than the one the report
    describes, whatever is in force for a moment while the report is built.

    fit-then-edit: run unedited, then edited; an undo and its redo while the report is built.
    fit-on-edit: run on the edited model; an edit and its undo while the report is built.
    """
    session = _split_session()
    widget = session.widget()
    try:
        if fitted_edited:
            session.shift("x_spline", 0.2)
            _run_final_fit(widget)
            change, put_back = (lambda: session.shift("x_spline", 0.1)), session.undo
        else:
            _run_final_fit(widget)
            session.shift("x_spline", 0.2)
            change, put_back = session.undo, session.redo
        steady, raced = _report_with_a_change_put_back(
            monkeypatch, widget, "final", change, put_back
        )
    finally:
        widget.close()

    assert steady["final_fit"]["stale"] is not fitted_edited
    assert raced["final_fit"]["stale"] is steady["final_fit"]["stale"]


def test_evidence_records_are_immutable():
    key = EvidenceKey(1, "metrics", "validation")
    outcome = EvidenceOutcome("complete", key, {"value": 1})

    with pytest.raises(FrozenInstanceError):
        key.model = 2
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
