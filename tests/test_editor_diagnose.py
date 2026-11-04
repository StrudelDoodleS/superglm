"""The Diagnose tab: what it reads from a fit, on both model types, and that it refits nothing."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import GammaLS, SuperGLM, SuperLSS, cat, cross_validate, re, s
from superglm._lambda_box import LAMBDA_CEILING
from superglm.editor import edit
from superglm.editor.cv import capture_cv_run, run_cv
from superglm.editor.diagnose import (
    SETTLED_REML,
    _glm_cell,
    _lss_cell,
    _term_cell,
    fit_trace,
    supplied_fold_traces,
    worst_rows,
)
from superglm.model_selection import fold_adapter
from superglm.terms import ti


class _Context:
    def check(self):
        return None

    def progress(self, *args, **kwargs):
        return None


def _rows(n: int = 1200, seed: int = 20261104) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "size": rng.uniform(0.0, 5.0, n),
            "region": rng.choice(list("abcd"), n),
        }
    )
    mean = np.exp(1.0 + 0.3 * np.sin(X["age"].to_numpy() / 10.0))
    return X, rng.gamma(4.0, mean / 4.0)


def _lss() -> tuple[SuperLSS, pd.DataFrame, np.ndarray]:
    X, y = _rows()
    family = GammaLS()
    model = SuperLSS(
        family, family.mean(s("age", k=6), s("size", k=5), cat("region")), family.scale()
    )
    return model.fit_reml(X, y), X, y


def _glm() -> tuple[SuperGLM, pd.DataFrame, np.ndarray]:
    X, y = _rows()
    model = SuperGLM(family="gamma", terms=[s("age", k=6), s("size", k=5), cat("region")])
    return model.fit_reml(X, y), X, y


def _no_refit(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the Diagnose tab must not refit")

    for model_type in (SuperLSS, SuperGLM):
        monkeypatch.setattr(model_type, "fit_reml", refuse)
        monkeypatch.setattr(model_type, "fit", refuse)


@pytest.mark.parametrize(
    ("edf", "lam", "shape"),
    [
        (3.2, 1.0, "curve"),
        (1.0, 3.6e6, "straight"),
        (1.04, 50.0, "straight"),
        (1.06, 50.0, "curve"),
        (0.01, 1e10, "flat"),
        (5.0, None, "unsmoothed"),
    ],
)
def test_a_term_reads_as_the_shape_its_edf_leaves_it(edf, lam, shape):
    assert _term_cell(edf, lam, 1)["shape"] == shape


def test_only_a_one_dimensional_null_space_is_a_straight_line():
    assert _term_cell(2.0, 1e10, 2)["shape"] == "simplest"
    assert _term_cell(2.5, 1e10, 2)["shape"] == "curve"
    # A shrunk random effect has no unpenalized shape to reach.
    assert _term_cell(1.0, 50.0, 0)["shape"] == "curve"
    assert _term_cell(0.01, 50.0, 0)["shape"] == "flat"


def test_a_quadratic_null_space_at_a_fixed_large_smoothing_is_not_called_straight():
    X, y = _rows()
    family = GammaLS()
    model = SuperLSS(family, family.mean(s("age", kind="bs", k=6, m=3)), family.scale())
    model.fit(X, y, lambdas={"mean:age#wiggle": 1e10})
    cell = fit_trace(model, fit_id="model", label="This model")["terms"]["mean:age"]
    assert cell["shape"] == "simplest"
    assert cell["at_limit"] is False


def _component(lam, null_space, outcome="finite"):
    return SimpleNamespace(final_lambda=lam, null_space_dimension=null_space, outcome=outcome)


def test_a_superlss_term_reads_its_own_components():
    assert _lss_cell(3.0, ())["shape"] == "unsmoothed"
    assert _lss_cell(1.0, [_component(1e7, 1)])["at_limit"] is False
    assert _lss_cell(1.0, [_component(1e10, 1, "upper_bound")])["at_limit"] is True
    # An unknown null space claims nothing.
    assert _lss_cell(1.0, [_component(1e7, None)])["shape"] == "curve"
    # Several penalties: the largest shows, any at its limit marks, no simplest shape is claimed.
    several = _lss_cell(1.0, [_component(1e10, 4, "upper_bound"), _component(3.0, 5)])
    assert several == {"edf": 1.0, "lambda": 1e10, "shape": "curve", "at_limit": True}


def test_a_superglm_term_with_several_penalties_claims_no_simplest_shape():
    penalties = [(SimpleNamespace(rank=16.0), 2.0), (SimpleNamespace(rank=15.0), 9.0)]
    cell = _glm_cell(2.4, 20, penalties, searched=True)
    assert cell == {"edf": 2.4, "lambda": 9.0, "shape": "curve", "at_limit": False}


def _with_terms():
    X, y = _rows()
    model = SuperGLM(
        family="gamma",
        terms=[s("age", k=6), s("size", k=5), ti("age", "size"), cat("region"), re("group")],
    )
    return model.fit_reml(X.assign(group=X["region"]), y)


def test_a_superglm_term_reads_only_the_smoothing_it_owns():
    model = _with_terms()
    lambdas = model._reml_result.lambdas
    terms = fit_trace(model, fit_id="model", label="This model")["terms"]
    assert set(terms) == {"age", "size", "age:size", "region", "group"}
    assert terms["age"]["lambda"] == lambdas["age"]
    assert terms["size"]["lambda"] == lambdas["size"]
    assert terms["age:size"]["lambda"] == max(
        lambdas["age:size:margin_age"], lambdas["age:size:margin_size"]
    )
    assert terms["region"]["lambda"] is None
    assert terms["group"]["lambda"] == lambdas["group"]
    for name in ("age", "size"):
        assert terms[name]["edf"] == pytest.approx(model.term_inference(name, with_se=False).edf)
    assert terms["age:size"]["edf"] == pytest.approx(model._group_edf["age:size"])


def test_a_group_wide_fixed_smoothing_reaches_each_of_its_penalties():
    X, y = _rows()
    model = SuperGLM(family="gamma", terms=[s("age", k=6), s("size", k=5), ti("age", "size")])
    model.lambda2 = {"age": 1.0, "size": 1.0, "age:size": 5.0}
    terms = fit_trace(model.fit(X, y), fit_id="model", label="This model")["terms"]
    assert terms["age:size"]["lambda"] == 5.0
    assert terms["age"]["lambda"] == 1.0


def test_a_fit_that_released_its_design_says_its_smoothing_was_not_kept():
    X, y = _rows()
    fixed = SuperGLM(
        family="gamma", terms=[s("age", k=6)], spline_penalty=1000.0, retain_fit_state=False
    ).fit(X, y)
    cell = fit_trace(fixed, fit_id="model", label="This model")["terms"]["age"]
    assert cell["shape"] == "unrecorded"
    assert cell["lambda"] is None
    assert cell["edf"] == pytest.approx(fixed.term_inference("age", with_se=False).edf)
    searched = SuperGLM(family="gamma", terms=[s("age", k=6)], retain_fit_state=False).fit_reml(
        X, y
    )
    terms = fit_trace(searched, fit_id="model", label="This model")["terms"]
    assert terms["age"]["lambda"] == searched._reml_result.lambdas["age"]


def test_a_searched_term_s_shape_reads_its_own_penalty_s_null_space(monkeypatch):
    model = _with_terms()
    forced = {**model._group_edf, "age": 1.0, "group": 1.0}
    monkeypatch.setitem(model.__dict__, "_group_edf", forced)
    terms = fit_trace(model, fit_id="model", label="This model")["terms"]
    assert terms["age"]["shape"] == "straight"
    assert terms["group"]["shape"] == "curve"


def test_a_superglm_limit_is_the_search_s_own_ceiling(monkeypatch):
    model, _, _ = _glm()
    reml = model._reml_result
    capped = {**reml.lambdas, "age": LAMBDA_CEILING, "size": 1e6}
    monkeypatch.setattr(model, "_reml_result", dataclasses.replace(reml, lambdas=capped))
    terms = fit_trace(model, fit_id="model", label="This model")["terms"]
    assert terms["age"]["at_limit"] is True
    assert terms["size"]["at_limit"] is False


def test_a_finding_of_a_new_severity_does_not_stop_the_trace(monkeypatch):
    model, _, _ = _lss()
    report = model.diagnose()
    finding = SimpleNamespace(severity="notice", headline="h", observed="o", actions=())
    monkeypatch.setattr(
        model, "diagnose", lambda: SimpleNamespace(profile=report.profile, findings=[finding])
    )
    assert (
        fit_trace(model, fit_id="model", label="This model")["findings"][0]["severity"] == "notice"
    )


@pytest.mark.parametrize("make", [_lss, _glm], ids=["SuperLSS", "SuperGLM"])
def test_a_trace_reads_the_fit_s_own_search(make):
    model, _, _ = make()
    trace = fit_trace(model, fit_id="model", label="This model")
    if isinstance(model, SuperLSS):
        fitted = model._require_fitted()
        assert trace["rounds"] == fitted.fitted_result.n_smoothing_iter
        assert trace["path"][-1] == fitted.smoothing.objective
        assert trace["stop"] == fitted.smoothing.convergence_reason
        assert trace["phases"] and trace["findings"]
        assert set(trace["terms"]) == set(fitted.inference.term_edf)
    else:
        reml = model._reml_result
        assert trace["rounds"] == reml.n_reml_iter
        assert trace["path"] == reml.objective_history
        assert trace["stop"] == reml.termination_reason
        assert trace["phases"] == [] and trace["findings"] == []
        assert trace["terms"]["region"]["shape"] == "unsmoothed"
        assert trace["terms"]["age"]["lambda"] == reml.lambdas["age"]
    assert trace["stop_text"] and "_" not in trace["stop_text"]
    path = trace["path"]
    assert trace["late_gain"] == path[len(path) // 2] - path[-1]
    settled = trace["settled_round"]
    assert abs(path[settled] - path[-1]) <= SETTLED_REML
    assert settled == 0 or abs(path[settled - 1] - path[-1]) > SETTLED_REML


def test_a_fit_with_no_search_says_so():
    X, y = _rows()
    model = SuperGLM(family="gamma", terms=[s("age", k=6), cat("region")], spline_penalty=1.0)
    trace = fit_trace(model.fit(X, y), fit_id="model", label="This model")
    assert trace["rounds"] == 0
    assert trace["path"] == []
    assert trace["late_gain"] is None and trace["settled_round"] is None
    assert trace["stop_text"] == "no smoothing search ran"


def test_a_fixed_smoothing_is_read_without_a_search():
    X, y = _rows()
    model = SuperGLM(family="gamma", terms=[s("age", k=6), cat("region")], spline_penalty=1000.0)
    terms = fit_trace(model.fit(X, y), fit_id="model", label="This model")["terms"]
    assert terms["age"]["lambda"] == 1000.0
    assert terms["age"]["shape"] != "unsmoothed"
    assert terms["age"]["at_limit"] is False
    assert terms["region"]["shape"] == "unsmoothed"
    heavy = SuperGLM(family="gamma", terms=[s("age", k=6)], spline_penalty=1e9).fit(X, y)
    assert (
        fit_trace(heavy, fit_id="model", label="This model")["terms"]["age"]["shape"] == "straight"
    )


@pytest.mark.parametrize("make", [_lss, _glm], ids=["SuperLSS", "SuperGLM"])
def test_supplied_folds_are_read_without_refitting(make, monkeypatch):
    model, X, y = make()
    cv = cross_validate(
        model,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=1),
        fit_mode="fit_reml",
        return_estimators=True,
    )
    session = edit(model, train_data=(X, y), cv=cv)
    widget = session.widget()
    _no_refit(monkeypatch)
    try:
        payload = widget._report("diagnose")
        again = widget._report("diagnose")
    finally:
        widget.close()
    assert payload["report"] == "diagnose"
    assert [fit["id"] for fit in payload["fits"]] == ["model", "fold-1", "fold-2", "fold-3"]
    assert payload["fits"][1]["rows"] == len(cv.fold_indices[0][0])
    json.dumps(payload)
    # A supplied result's folds are read once.
    assert payload["fits"][1] is again["fits"][1]


def test_each_fold_lists_the_held_out_rows_it_predicted_worst():
    model, X, y = _lss()
    y = y.copy()
    splitter = KFold(3, shuffle=True, random_state=1)
    planted = int(list(splitter.split(X))[1][1][0])
    y[planted] *= 1000.0
    cv = cross_validate(model, X, y, cv=splitter, return_estimators=True, fit_mode="fit_reml")
    session = edit(model, train_data=(X, y), cv=cv)
    widget = session.widget()
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    worst = payload["fits"][2]["worst_rows"]
    assert worst["rows"][0]["row"] == planted
    assert worst["rows"][0]["response"] == pytest.approx(y[planted])
    values = dict(zip(worst["columns"], worst["rows"][0]["values"], strict=True))
    assert values == {name: X[name].iloc[planted] for name in ("age", "size", "region")}
    assert payload["cost_label"] == "Log score"
    assert planted not in {row["row"] for row in payload["fits"][1]["worst_rows"]["rows"]}


def test_a_superglm_ranks_rows_by_weighted_deviance_and_lists_their_weights():
    model, X, y = _glm()
    weights = np.ones(len(y))
    weights[:50] = 4.0
    adapter = fold_adapter(model)
    positions = np.arange(100, 200)
    listed = worst_rows(
        adapter,
        model,
        X.iloc[positions],
        y[positions],
        weights[positions],
        None,
        positions,
        ["age"],
    )
    mu = model.predict(X.iloc[positions])
    unit = model._distribution.deviance_unit(y[positions], mu)
    worst = positions[np.argmax(weights[positions] * unit)]
    assert adapter.row_cost_label == "Deviance"
    assert listed["rows"][0]["row"] == worst
    assert listed["rows"][0]["cost"] == pytest.approx(np.max(weights[positions] * unit))
    assert listed["rows"][0]["values"] == [X["age"].iloc[worst]]


@pytest.mark.parametrize("make", [_lss, _glm], ids=["SuperLSS", "SuperGLM"])
def test_a_row_of_zero_weight_is_not_ranked_or_predicted(make):
    model, X, y = make()
    rows = X.iloc[:60].copy()
    rows.loc[rows.index[:12], "age"] = np.nan
    weights = np.ones(60)
    weights[:12] = 0.0
    listed = worst_rows(
        fold_adapter(model), model, rows, y[:60], weights, None, np.arange(60), ["age"]
    )
    assert listed["rows"]
    assert all(row["row"] >= 12 for row in listed["rows"])
    assert all(row["weight"] == 1.0 for row in listed["rows"])


def test_a_fold_with_no_weight_has_no_rows_to_list():
    model, X, y = _glm()
    listed = worst_rows(
        fold_adapter(model), model, X.iloc[:20], y[:20], np.zeros(20), None, np.arange(20), []
    )
    assert listed["rows"] == []


def test_a_fold_that_failed_has_no_trace():
    model, X, y = _glm()
    cv = cross_validate(
        model,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=1),
        fit_mode="fit_reml",
        return_estimators=True,
    )
    failed = dataclasses.replace(cv, estimators=[None, *cv.estimators[1:]])
    traces = supplied_fold_traces(failed, None)
    assert [trace["id"] for trace in traces] == ["fold-2", "fold-3"]
    assert "worst_rows" not in traces[0]


def test_fold_models_with_no_fold_indices_are_traced_without_rows():
    model, X, y = _glm()
    cv = cross_validate(
        model,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=1),
        fit_mode="fit_reml",
        return_estimators=True,
    )
    session = edit(model, train_data=(X, y), cv=cv)
    bare = dataclasses.replace(cv, fold_indices=None)
    traces = supplied_fold_traces(bare, session.cv_check.rows)
    assert [trace["id"] for trace in traces] == ["fold-1", "fold-2", "fold-3"]
    assert traces[0]["rows"] is None
    assert "worst_rows" not in traces[0]


def test_a_supplied_fold_that_cannot_be_read_leaves_the_tab_open(monkeypatch):
    model, X, y = _glm()
    cv = cross_validate(
        model,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=1),
        fit_mode="fit_reml",
        return_estimators=True,
    )
    session = edit(model, train_data=(X, y), cv=cv)
    real = fit_trace

    def refuses_one(fitted, *, fit_id, **kwargs):
        if fit_id == "fold-2":
            raise ValueError("this fold cannot be read")
        return real(fitted, fit_id=fit_id, **kwargs)

    monkeypatch.setattr("superglm.editor.diagnose.fit_trace", refuses_one)
    widget = session.widget()
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    assert [fit["id"] for fit in payload["fits"]] == ["model", "fold-1", "fold-3"]
    assert payload["note"].endswith("No trace for Fold 2: see the log.")


def test_a_result_with_no_fold_models_shows_this_model_and_says_why():
    model, X, y = _glm()
    cv = cross_validate(model, X, y, cv=KFold(3, shuffle=True, random_state=1))
    session = edit(model, train_data=(X, y), cv=cv)
    widget = session.widget()
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    assert [fit["id"] for fit in payload["fits"]] == ["model"]
    assert "return_estimators=True" in payload["note"]


def test_no_result_shows_this_model():
    model, X, y = _glm()
    widget = edit(model, train_data=(X, y)).widget()
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    assert [fit["id"] for fit in payload["fits"]] == ["model"]
    assert "cross_validate()" in payload["note"]
    assert payload["cost_label"] == "Deviance"


@pytest.mark.parametrize("make", [_lss, _glm], ids=["SuperLSS", "SuperGLM"])
def test_run_cv_keeps_each_fold_s_trace_and_worst_rows(make):
    model, X, y = make()
    cv = cross_validate(model, X, y, cv=KFold(3, shuffle=True, random_state=1), fit_mode="fit_reml")
    session = edit(model, train_data=(X, y), cv=cv)
    run = run_cv(capture_cv_run(session), _Context())
    assert [fit["id"] for fit in run.fold_fits] == ["fold-1", "fold-2", "fold-3"]
    for fit, (train, test), rounds in zip(
        run.fold_fits, cv.fold_indices, run.result.fold_scores["n_reml_iter"], strict=True
    ):
        assert fit["rows"] == len(train)
        assert fit["rounds"] == rounds
        assert fit["seconds"] > 0.0
        assert {row["row"] for row in fit["worst_rows"]["rows"]} <= set(test.tolist())
    widget = session.widget()
    widget._cv_run = run
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    assert [fit["id"] for fit in payload["fits"]] == ["model", "fold-1", "fold-2", "fold-3"]
    assert payload["note"] == "Folds from Run CV on the current model."


def test_folds_from_an_older_model_say_so(monkeypatch):
    model, X, y = _glm()
    cv = cross_validate(
        model,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=1),
        fit_mode="fit_reml",
        return_estimators=True,
    )
    session = edit(model, train_data=(X, y), cv=cv)
    run = run_cv(capture_cv_run(session), _Context())
    widget = session.widget()
    try:
        assert (
            widget._report("diagnose")["note"] == "Folds as supplied with edit(model, cv=result)."
        )
        session.reference_model = object()
        assert "from the model as first opened" in widget._report("diagnose")["note"]
        widget._cv_run = run
        assert widget._report("diagnose")["note"] == "Folds from Run CV on the current model."
        widget._cv_run = dataclasses.replace(run, model_revision=run.model_revision - 1)
        note = widget._report("diagnose")["note"]
    finally:
        widget.close()
    assert "earlier version of this model" in note


@pytest.mark.parametrize("target", ["fit_trace", "worst_rows"])
def test_a_fold_that_cannot_be_traced_keeps_its_scores(monkeypatch, target):
    model, X, y = _glm()
    session = edit(
        model,
        train_data=(X, y),
        cv=cross_validate(
            model, X, y, cv=KFold(3, shuffle=True, random_state=1), fit_mode="fit_reml"
        ),
    )

    def fails(*args, **kwargs):
        raise ValueError("this fit cannot be traced")

    monkeypatch.setattr(f"superglm.editor.cv.{target}", fails)
    run = run_cv(capture_cv_run(session), _Context())
    assert run.fold_fits == ()
    assert np.isfinite(run.result.fold_scores["deviance"]).all()
    monkeypatch.undo()
    widget = session.widget()
    widget._cv_run = run
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    assert [fit["id"] for fit in payload["fits"]] == ["model"]
    assert payload["note"] == (
        "Folds from Run CV on the current model. No trace for Fold 1, Fold 2, Fold 3: see the log."
    )
