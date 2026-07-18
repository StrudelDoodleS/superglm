import json
from pathlib import Path

import pandas as pd
import pytest

from superglm import Constraint, LambdaPolicy, PSpline, SuperGLM


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


@pytest.mark.parametrize("discrete", [False, True])
def test_debug_level_two_records_one_ordered_actual_run_and_selected_terminal(
    tmp_path: Path,
    monkeypatch,
    discrete: bool,
):
    """Canonical rows must identify the actual retained REML state."""
    from superglm._debug import set_debug_level
    from superglm.model.reml_debug import load_reml_debug_run

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x, y = _make_demo_data()
    model = _make_unconstrained_model(discrete=discrete)
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


def test_debug_level_two_orders_actual_direct_reml_solver_phases(
    tmp_path: Path,
    monkeypatch,
):
    """The canonical stream must cover real optimizer work through final refit."""
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
    state_commit_purposes = [
        event["purpose"]
        for event in run.events
        if event["channel"] == "pirls" and event["event_kind"] == "state_commit"
    ]

    expected = ["reml_bootstrap", "reml_candidate", "reml_line_search", "reml_final"]
    first_positions = [state_commit_purposes.index(purpose) for purpose in expected]
    assert first_positions == sorted(first_positions)

    pirls_state_ids = {
        event["payload"]["state_id"]
        for event in run.events
        if event["channel"] == "pirls" and event["event_kind"] == "evaluation"
    }
    outer_evaluations = [
        event
        for event in run.events
        if event["channel"] == "reml" and event["event_kind"] == "evaluation"
    ]
    assert {event["purpose"] for event in outer_evaluations} >= {
        "reml_candidate",
        "reml_line_search",
    }
    assert all(event["payload"]["state_id"] in pirls_state_ids for event in outer_evaluations)


def test_failed_reml_attempt_never_emits_a_terminal_success(tmp_path: Path, monkeypatch):
    """A terminal row is a publication claim, so workspace failure must omit it."""
    from superglm._debug import set_debug_level
    from superglm.model import fit_ops
    from superglm.model.reml_debug import load_reml_debug_run

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    def fail_during_postfit(*_args, **_kwargs):
        raise RuntimeError("post-fit validation failed")

    monkeypatch.setattr(fit_ops, "_canonicalize_fitted_model", fail_during_postfit)
    x, y = _make_demo_data()
    model = _make_unconstrained_model(discrete=False)
    with pytest.raises(RuntimeError, match="post-fit validation failed"):
        model.fit_reml(x, y, max_reml_iter=3)

    run_file = next(tmp_path.glob("*run.json"))
    run_id = run_file.name.removesuffix("_run.json")
    run = load_reml_debug_run(tmp_path, run_id)
    assert not [event for event in run.events if event["event_kind"] == "terminal"]


def test_terminal_trace_sink_failure_cannot_turn_an_installed_fit_into_an_exception(
    monkeypatch,
    caplog,
):
    """External trace I/O is best-effort once the no-fail state swap has occurred."""
    from superglm.model import fit_ops

    def fail_terminal(*_args, **_kwargs):
        raise OSError("trace disk is full")

    monkeypatch.setattr(fit_ops, "record_reml_terminal", fail_terminal)
    x, y = _make_demo_data()
    model = _make_unconstrained_model(discrete=False)

    returned = model.fit_reml(x, y, max_reml_iter=2)

    assert returned is model
    assert model._fit_revision == 1
    assert model._fit_state.revision == 1
    assert "terminal trace" in caplog.text


def test_terminal_trace_and_logging_failures_cannot_escape_after_install(monkeypatch):
    """Even a raising custom log handler is outside the fitted-state transaction."""
    import logging

    from superglm.model import fit_ops

    class RaisingHandler(logging.Handler):
        def emit(self, _record):
            raise RuntimeError("logging backend failed")

    monkeypatch.setattr(
        fit_ops,
        "record_reml_terminal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("trace disk is full")),
    )
    handler = RaisingHandler()
    fit_ops.logger.addHandler(handler)
    try:
        x, y = _make_demo_data()
        model = _make_unconstrained_model(discrete=False)
        returned = model.fit_reml(x, y, max_reml_iter=2)
    finally:
        fit_ops.logger.removeHandler(handler)

    assert returned is model
    assert model._fit_revision == 1


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
    assert any(row.get("iteration", 0) >= 1 for row in reml_rows)

    pirls_rows = [
        json.loads(line)
        for line in pirls_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert pirls_rows
    assert any(row.get("iteration", 0) >= 1 for row in pirls_rows)

    scop_rows = [
        json.loads(line)
        for line in scop_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert scop_rows
    assert any(row.get("step_norm", -1.0) >= 0.0 for row in scop_rows)


def test_scop_debug_records_actual_candidate_states_and_one_installed_terminal(
    tmp_path: Path,
    monkeypatch,
):
    """SCOP must participate in the same canonical trace contract as direct REML."""
    from superglm._debug import set_debug_level
    from superglm.model.reml_debug import load_reml_debug_run

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)

    x, y = _make_demo_data()
    model = _make_scop_model()
    model.fit_reml(x, y, max_reml_iter=3)

    run_file = next(tmp_path.glob("*run.json"))
    run_id = run_file.name.removesuffix("_run.json")
    run = load_reml_debug_run(tmp_path, run_id)
    assert [event["sequence"] for event in run.events] == list(range(1, len(run.events) + 1))
    pirls_state_ids = {
        event["payload"]["state_id"]
        for event in run.events
        if event["channel"] == "pirls" and event["event_kind"] == "evaluation"
    }
    outer = [
        event
        for event in run.events
        if event["channel"] == "reml" and event["event_kind"] == "evaluation"
    ]
    assert outer
    assert all(event["payload"]["state_id"] in pirls_state_ids for event in outer)
    assert all(event["payload"]["effective_df"] is None for event in outer)
    terminals = [
        event
        for event in run.events
        if event["channel"] == "reml"
        and event["event_kind"] == "terminal"
        and event["purpose"] == "fit_reml"
    ]
    assert len(terminals) == 1
    payload = terminals[0]["payload"]
    assert payload["state_id"] == model._solver_result.state_id
    assert payload["objective"] == pytest.approx(model._reml_result.objective)
    assert payload["lambdas"] == pytest.approx(model._reml_lambdas)
    assert payload["dispersion"] == pytest.approx(model._solver_result.phi)
    assert payload["effective_df"] == pytest.approx(model._solver_result.effective_df)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_fixed_scop_terminal_trace_identifies_its_single_evaluated_mode(
    tmp_path: Path,
    monkeypatch,
    family: str,
):
    """Fixed-lambda objective, coefficients, scale, and terminal share one identity."""
    import numpy as np

    from superglm._debug import set_debug_level
    from superglm.model.reml_debug import load_reml_debug_run

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)
    rng = np.random.default_rng(20260801)
    x = np.linspace(0.0, 1.0, 180)
    frame = pd.DataFrame({"x": x})
    y = (
        0.2 + 1.3 * x + rng.normal(0.0, 0.12, len(x))
        if family == "gaussian"
        else rng.poisson(np.exp(-0.4 + 1.0 * x))
    )
    model = SuperGLM(
        family=family,
        selection_penalty=0.0,
        discrete=True,
        features={
            "x": PSpline(
                n_knots=7,
                constraint=Constraint.fit.increasing,
                lambda_policy=LambdaPolicy(mode="fixed", value=1.5),
            )
        },
    )

    model.fit_reml(frame, y)

    run_file = next(tmp_path.glob("*run.json"))
    run_id = run_file.name.removesuffix("_run.json")
    run = load_reml_debug_run(tmp_path, run_id)
    fixed_evaluations = [
        event
        for event in run.events
        if event["channel"] == "reml"
        and event["event_kind"] == "evaluation"
        and event["purpose"] == "reml_fixed"
    ]
    terminals = [
        event
        for event in run.events
        if event["channel"] == "reml"
        and event["event_kind"] == "terminal"
        and event["purpose"] == "fit_reml"
    ]

    assert len(fixed_evaluations) == 1
    assert len(terminals) == 1
    evaluated = fixed_evaluations[0]["payload"]
    terminal = terminals[0]["payload"]
    assert model._reml_result.termination_reason == "fixed_lambdas"
    assert model._reml_result.pirls_result is model._solver_result
    assert evaluated["state_id"] == terminal["state_id"] == model._solver_result.state_id
    assert (
        evaluated["evaluation_id"]
        == terminal["evaluation_id"]
        == model._solver_result.evaluation_id
    )
    assert evaluated["objective"] == pytest.approx(model._reml_result.objective)
    assert terminal["objective"] == pytest.approx(model._reml_result.objective)
    assert evaluated["lambdas"] == pytest.approx(model._reml_lambdas)
    assert terminal["lambdas"] == pytest.approx(model._reml_lambdas)
    assert evaluated["dispersion"] == pytest.approx(model._solver_result.phi)
    assert terminal["dispersion"] == pytest.approx(model._solver_result.phi)


def test_profiled_theta_reml_emits_terminal_only_after_final_install(tmp_path, monkeypatch):
    """The profile wrapper must close the canonical trace of its installed REML refit."""
    import numpy as np

    from superglm import NegativeBinomial, Spline
    from superglm._debug import set_debug_level
    from superglm.model.reml_debug import load_reml_debug_run
    from superglm.profiling.nb import NBProfileResult

    monkeypatch.setenv("SUPERGLM_DEBUG_DIR", str(tmp_path))
    set_debug_level(2)
    monkeypatch.setattr(
        "superglm.profiling.nb.estimate_nb_theta",
        lambda *_args, **_kwargs: NBProfileResult(
            theta_hat=2.5,
            nll=1.0,
            n_evaluations=1,
            converged=True,
        ),
    )
    x = np.linspace(-1.0, 1.0, 100)
    X = pd.DataFrame({"x": x})
    y = np.resize(np.array([1.0, 2.0, 3.0, 4.0]), x.size)
    model = SuperGLM(
        family=NegativeBinomial(theta="auto"),
        selection_penalty=0.0,
        features={"x": Spline(n_knots=5)},
    )

    model.estimate_theta(X, y, fit_mode="reml")

    run_file = next(tmp_path.glob("*run.json"))
    run_id = run_file.name.removesuffix("_run.json")
    run = load_reml_debug_run(tmp_path, run_id)
    terminals = [event for event in run.events if event["event_kind"] == "terminal"]
    assert len(terminals) == 1
    assert terminals[0]["payload"]["state_id"] == model._solver_result.state_id
    assert terminals[0]["payload"]["objective"] == pytest.approx(model._reml_result.objective)


def test_debug_tests_do_not_leak_enabled_state():
    from superglm._debug import get_debug_level

    assert get_debug_level() == 0
