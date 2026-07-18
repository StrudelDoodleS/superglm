import json
from pathlib import Path

import pandas as pd
import pytest

from superglm import Constraint, PSpline, SuperGLM


@pytest.fixture(autouse=True)
def _restore_debug_level():
    from superglm._debug import get_debug_level, set_debug_level

    previous_level = get_debug_level()
    yield
    set_debug_level(previous_level)


def _make_demo_data() -> tuple[pd.DataFrame, object]:
    x = pd.DataFrame({"x": [0.0, 0.5, 1.0, 0.25, 0.75] * 20})
    y = x["x"].to_numpy() ** 2 + 0.1
    return x, y


def _make_scop_model() -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=8, constraint=Constraint.fit.convex)},
    )


def _make_unconstrained_model(*, discrete: bool = True) -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={"x": PSpline(n_knots=8)},
    )


def test_debug_level_zero_emits_no_reml_trace(tmp_path: Path, monkeypatch):
    from superglm._debug import set_debug_level

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(0)

    x, y = _make_demo_data()
    model = _make_scop_model()
    model.fit_reml(x, y, max_reml_iter=4)

    assert not list(tmp_path.glob("*run.json"))
    assert not list(tmp_path.glob("*.jsonl"))


def test_debug_level_one_keeps_unconstrained_reml_summary_only(tmp_path: Path, monkeypatch):
    from superglm._debug import set_debug_level

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(1)

    x, y = _make_demo_data()
    model = _make_unconstrained_model()
    model.fit_reml(x, y, max_reml_iter=4)

    run_files = list(tmp_path.glob("*run.json"))

    assert run_files
    assert not list(tmp_path.glob("*.jsonl"))

    run_payload = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run_payload["debug_level"] == 1
    assert run_payload["method"] == "fit_reml"
    assert run_payload["reml_group_names"] == ["x"]


def test_debug_level_two_writes_non_scop_reml_and_pirls_traces(tmp_path: Path, monkeypatch):
    from superglm._debug import set_debug_level

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x, y = _make_demo_data()
    model = _make_unconstrained_model()
    model.fit_reml(x, y, max_reml_iter=4)

    assert list(tmp_path.glob("*run.json"))
    assert list(tmp_path.glob("*reml.jsonl"))
    assert list(tmp_path.glob("*pirls.jsonl"))
    assert not list(tmp_path.glob("*scop.jsonl"))


def test_debug_level_two_never_replays_a_coefficient_solve(tmp_path: Path, monkeypatch):
    """Tracing must observe optimizer work, not run an extra post-fit PIRLS call."""
    from superglm._debug import set_debug_level
    from superglm.model import reml_execute

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)
    monkeypatch.setattr(
        reml_execute,
        "fit_irls_direct",
        lambda *_args, **_kwargs: pytest.fail("debug tracing replayed a coefficient solve"),
    )

    x, y = _make_demo_data()
    model = _make_unconstrained_model(discrete=False)
    model.fit_reml(x, y, max_reml_iter=3)


def test_debug_level_two_records_one_ordered_actual_run_and_selected_terminal(
    tmp_path: Path,
    monkeypatch,
):
    """Canonical rows must identify the actual retained REML state."""
    from superglm._debug import set_debug_level
    from superglm.model.reml_debug import load_reml_debug_run

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x, y = _make_demo_data()
    model = _make_unconstrained_model(discrete=False)
    model.fit_reml(x, y, max_reml_iter=3)

    run_file = next(tmp_path.glob("*run.json"))
    run_id = run_file.name.removesuffix("_run.json")
    run = load_reml_debug_run(tmp_path, run_id)
    assert run.events
    assert [event["sequence"] for event in run.events] == list(range(1, len(run.events) + 1))
    assert any(
        event["channel"] == "pirls"
        and event["event_kind"] == "state_commit"
        and event["authoritative"]
        for event in run.events
    )
    terminal = [
        event
        for event in run.events
        if event["channel"] == "reml"
        and event["event_kind"] == "terminal"
        and event["purpose"] == "fit_reml"
    ]
    assert len(terminal) == 1
    payload = terminal[0]["payload"]
    assert payload["state_id"] == model._solver_result.state_id
    assert payload["objective"] == pytest.approx(model._reml_result.objective)
    assert payload["lambdas"] == pytest.approx(model._reml_lambdas)
    assert payload["dispersion"] == pytest.approx(model._solver_result.phi)
    assert payload["effective_df"] == pytest.approx(model._solver_result.effective_df)
    assert all(not event["payload"].get("trace_replay", False) for event in run.events)


def test_debug_level_two_writes_reml_trace_files(tmp_path: Path, monkeypatch):
    from superglm._debug import set_debug_level

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x, y = _make_demo_data()
    model = _make_scop_model()
    model.fit_reml(x, y, max_reml_iter=4)

    run_files = list(tmp_path.glob("*run.json"))
    reml_files = list(tmp_path.glob("*reml.jsonl"))
    pirls_files = list(tmp_path.glob("*pirls.jsonl"))
    scop_files = list(tmp_path.glob("*scop.jsonl"))

    assert run_files
    assert reml_files
    assert pirls_files
    assert scop_files

    run_payload = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert run_payload["debug_level"] == 2
    assert run_payload["method"] == "fit_reml"

    reml_rows = [
        json.loads(line)
        for line in reml_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert reml_rows
    assert reml_rows[0]["iteration"] >= 1

    pirls_rows = [
        json.loads(line)
        for line in pirls_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert pirls_rows
    assert pirls_rows[0]["iteration"] >= 1

    scop_rows = [
        json.loads(line)
        for line in scop_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert scop_rows
    assert scop_rows[0]["step_norm"] >= 0.0


def test_debug_tests_do_not_leak_enabled_state():
    from superglm._debug import get_debug_level

    assert get_debug_level() == 0
