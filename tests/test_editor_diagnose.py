"""The Diagnose tab: what it reads from a fit, on both model types, and that it refits nothing."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import GammaLS, SuperGLM, SuperLSS, cat, cross_validate, s
from superglm.editor import edit
from superglm.editor.cv import capture_cv_run, run_cv
from superglm.editor.diagnose import (
    SETTLED_REML,
    _term_cell,
    fit_trace,
    supplied_fold_traces,
)


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


def test_a_large_smoothing_is_at_the_limit_and_a_small_one_is_not():
    assert _term_cell(1.0, 1e6, 1)["at_limit"] is True
    assert _term_cell(3.0, 10.0, 1)["at_limit"] is False
    assert _term_cell(3.0, None, 1)["at_limit"] is False
    assert _term_cell(1.0, 10.0, 1, at_cap=True)["at_limit"] is True


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
    assert trace["stop_text"] == "there was no smoothing to choose"


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
    assert worst[0]["row"] == planted
    assert worst[0]["response"] == pytest.approx(y[planted])
    costs = [row["cost"] for row in worst]
    assert costs == sorted(costs, reverse=True)
    assert set(payload["columns"]) == {"age", "size", "region"}
    assert len(worst[0]["values"]) == len(payload["columns"])
    assert payload["cost_label"] == "Log score"
    assert planted not in {row["row"] for row in payload["fits"][1]["worst_rows"]}


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
        assert {row["row"] for row in fit["worst_rows"]} <= set(test.tolist())
    widget = session.widget()
    widget._cv_run = run
    try:
        payload = widget._report("diagnose")
    finally:
        widget.close()
    assert [fit["id"] for fit in payload["fits"]] == ["model", "fold-1", "fold-2", "fold-3"]
    assert payload["note"] == "Folds from Run CV on the current model."
