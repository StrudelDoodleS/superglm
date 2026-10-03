"""The Cross-validation tab: carried edits, stored folds, Run CV and Final fit."""

from __future__ import annotations

import dataclasses
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import Categorical, Numeric, Spline, SuperGLM, cross_validate
from superglm.editor import EditorSession

_U = np.finfo(np.float64).eps / 2
# A fold that never saw a level holds it pinned, and the fit says so.
_EXPECTED_PIN = "ignore:.*pinned to.*:UserWarning"


def _model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "age": Spline(n_knots=6),
            "power": Numeric(),
            "region": Categorical(base="first"),
        },
    )


@pytest.fixture(scope="module")
def cv_frame():
    """600 rows: train 0-399, validation 400-499, test 500-599.

    Region's rows meet C first; the model orders its levels A, B, C.
    """
    rng = np.random.default_rng(20261003)
    n = 600
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "power": rng.normal(0.0, 1.0, n),
            "region": rng.choice(["C", "A", "B"], n, p=[0.4, 0.35, 0.25]),
        }
    )
    eta = (
        -0.5
        + 0.2 * np.sin(X["age"].to_numpy() / 12.0)
        + 0.1 * X["power"].to_numpy()
        + np.select([X["region"] == "B", X["region"] == "C"], [0.25, -0.15], 0.0)
    )
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    w = rng.uniform(0.5, 1.5, n)
    return X, y, w


@pytest.fixture(scope="module")
def cv_fit(cv_frame):
    """The model on the train rows, and a 3-fold cross_validate() of it there."""
    X, y, w = cv_frame
    train = slice(0, 400)
    model = _model().fit(X.iloc[train], y[train], sample_weight=w[train])
    supplied = cross_validate(
        _model(),
        X.iloc[train],
        y[train],
        cv=KFold(3, shuffle=True, random_state=0),
        sample_weight=w[train],
        scoring=("deviance", "gini", "nll"),
        return_estimators=True,
    )
    return model, supplied


def _splits(cv_frame):
    X, y, w = cv_frame
    return {
        "train_data": (X.iloc[:400], y[:400], w[:400]),
        "validation_data": (X.iloc[400:500], y[400:500], w[400:500]),
        "test_data": (X.iloc[500:], y[500:], w[500:]),
    }


def _post_json(url: str, payload: dict):
    from superglm.editor.widget import _LIVE_WIDGETS

    origin = url.rsplit("/", 1)[0]
    token = next(widget._token for widget in _LIVE_WIDGETS if widget.url == origin)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-SuperGLM-Editor-Token": token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str):
    from superglm.editor.widget import _LIVE_WIDGETS

    origin = url.rsplit("/", 1)[0]
    token = next(widget._token for widget in _LIVE_WIDGETS if widget.url == origin)
    request = urllib.request.Request(url, headers={"X-SuperGLM-Editor-Token": token})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_error(url: str, payload: dict) -> tuple[int, dict]:
    with pytest.raises(urllib.error.HTTPError) as error:
        _post_json(url, payload)
    return error.value.code, json.loads(error.value.read().decode("utf-8"))


# ── Carrying hand edits onto another fit (D5) ────────────────────


def test_carried_spline_edit_is_clamped_past_a_narrower_fold_range(cv_frame, cv_fit):
    from superglm.editor.apply import _as_dense
    from superglm.editor.carry import model_with_edited_curves
    from superglm.plotting.comparison import _feature_beta

    X, y, w = cv_frame
    model, _supplied = cv_fit
    session = EditorSession.from_model(
        model, terms=["age"], train_data=(X.iloc[:400], y[:400], w[:400])
    )
    term = session.terms["age"]
    # A straight line lies in every spline basis, so the carried curve can
    # match it exactly; only its centring constant may move.
    line = 0.01 * term.x
    session.set_values("age", np.arange(term.size), line)
    narrow = X["age"].to_numpy()[:400] < 55.0
    fold = _model().fit(X.iloc[:400][narrow], y[:400][narrow], sample_weight=w[:400][narrow])

    carried = model_with_edited_curves(
        fold,
        {"age": session.terms["age"].copy()},
        X.iloc[:400][narrow],
        y[:400][narrow],
        w[:400][narrow],
        n_points=session.n_points,
    )

    spec = carried._specs["age"]
    lo, hi = spec.fitted_boundary
    beta = _feature_beta(carried, "age")
    curve = spec.score(term.x, beta)
    inside = term.x <= hi
    assert hi < term.x[-1]
    assert np.all(np.isfinite(curve))
    # Past the fold's range the spline holds its end value, as it does at
    # predict time: equal to it up to two dot products' rounding (Higham sec. 3.1).
    end_row = _as_dense(spec.transform(np.array([hi])))[0]
    end_tol = 2 * end_row.size * _U * np.sum(np.abs(end_row * beta))
    assert np.max(np.abs(curve[~inside] - spec.score(np.array([hi]), beta)[0])) <= end_tol
    # Inside it, the edited line plus one constant. A least-squares fit of a
    # representable target is accurate to about cond(design) * n * u * |target|
    # (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., sec. 20.1).
    grid = np.linspace(lo, hi, session.n_points)
    design = np.column_stack([np.ones(grid.size), _as_dense(spec.transform(grid))])
    tol = np.linalg.cond(design) * grid.size * _U * np.max(np.abs(line))
    assert np.ptp(curve[inside] - line[inside]) <= tol
    assert np.all(np.isfinite(carried.predict(X.iloc[:400][~narrow])))


def test_carried_level_edit_keeps_its_change_when_the_reference_moves():
    from superglm.editor.carry import model_with_edited_curves

    rng = np.random.default_rng(7)

    def rows(n, shares):
        x = rng.uniform(0.0, 10.0, n)
        region = rng.choice(["A", "B", "C"], n, p=shares)
        eta = -0.4 + 0.1 * np.sin(x) + np.select([region == "B", region == "C"], [0.3, -0.2], 0.0)
        return pd.DataFrame({"x": x, "region": region}), rng.poisson(np.exp(eta)).astype(float)

    X_train, y_train = rows(400, [0.5, 0.3, 0.2])
    X_more, y_more = rows(400, [0.1, 0.7, 0.2])
    X_all = pd.concat([X_train, X_more], ignore_index=True)
    y_all = np.concatenate([y_train, y_more])

    def fit(X, y):
        features = {"x": Spline(n_knots=6), "region": Categorical(base="most_exposed")}
        return SuperGLM(family="poisson", selection_penalty=0.0, features=features).fit(X, y)

    model = fit(X_train, y_train)
    refit = fit(X_all, y_all)
    assert (model._specs["region"]._base_level, refit._specs["region"]._base_level) == ("A", "B")
    session = EditorSession.from_model(model, train_data=(X_train, y_train))
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    term = session.terms["region"]

    carried = model_with_edited_curves(
        refit, {"region": term.copy()}, X_all, y_all, n_points=session.n_points
    )

    probe = pd.DataFrame({"x": [5.0] * 3, "region": term.levels})
    log_carried = np.log(carried.predict(probe))
    change = log_carried - np.log(refit.predict(probe))
    edit = term.edited_log_effect - term.original_log_effect
    exposure = np.array([np.sum(X_all["region"] == level) for level in term.levels], dtype=float)
    tol = 64 * _U * max(1.0, np.max(np.abs(log_carried)))
    # The edited relativities hold exactly...
    np.testing.assert_allclose(
        log_carried - log_carried[0],
        term.edited_log_effect - term.edited_log_effect[0],
        rtol=0.0,
        atol=tol,
    )
    # ...and so does the edit's exposure-weighted change, whatever base the refit chose.
    assert abs(np.average(change, weights=exposure) - np.average(edit, weights=exposure)) <= tol


# ── Stored folds and the supplied result's rows ──────────────────


def test_stored_folds_replays_indices_with_a_hook_between_folds(cv_frame, cv_fit):
    from superglm.editor.cv import StoredFolds

    X, y, w = cv_frame
    _model_unused, supplied = cv_fit
    folds = tuple(supplied.fold_indices)
    events = []

    def score(model, X_val, y_val, *, sample_weight=None, offset=None):
        events.append(("score", len(y_val)))
        return {"rows": float(len(y_val))}

    replayed = cross_validate(
        _model(),
        X.iloc[:400],
        y[:400],
        cv=StoredFolds(folds, before_fold=lambda index: events.append(("before", index))),
        sample_weight=w[:400],
        scoring=score,
    )

    expected = []
    for index, (_train, test) in enumerate(folds):
        expected += [("before", index), ("score", len(test))]
    assert events == expected
    for (train, test), (again_train, again_test) in zip(folds, replayed.fold_indices, strict=True):
        np.testing.assert_array_equal(again_train, train)
        np.testing.assert_array_equal(again_test, test)


def test_cv_data_is_checked_against_the_folds(cv_frame, cv_fit):
    from superglm.editor.cv import (
        FINGERPRINT_MISMATCH,
        NO_CV,
        NO_FINGERPRINT,
        ROWS_MISMATCH,
        TRAIN_ROWS_MISMATCH,
    )

    X, y, w = cv_frame
    model, supplied = cv_fit
    rows = (X.iloc[:400], y[:400], w[:400])

    matched = EditorSession.from_model(model, terms=["region"], train_data=rows, cv=supplied)
    assert (matched.cv_check.reason, matched.cv_check.note) == (None, None)
    assert matched.cv_check.rows.n_obs == 400

    fewer = EditorSession.from_model(
        model, terms=["region"], cv=supplied, cv_data=(X.iloc[:399], y[:399], w[:399])
    )
    assert fewer.cv_check.rows is None
    assert fewer.cv_check.reason == ROWS_MISMATCH.format(rows=399, expected=400)

    wider = EditorSession.from_model(
        model, terms=["region"], train_data=(X.iloc[:500], y[:500], w[:500]), cv=supplied
    )
    assert wider.cv_check.reason == TRAIN_ROWS_MISMATCH.format(rows=500, expected=400)

    reordered = (X.iloc[:400][::-1], y[:400][::-1], w[:400][::-1])
    moved = EditorSession.from_model(model, terms=["region"], cv=supplied, cv_data=reordered)
    assert moved.cv_check.reason == FINGERPRINT_MISMATCH

    older = dataclasses.replace(supplied, n_rows=None, data_fingerprint=None)
    noted = EditorSession.from_model(model, terms=["region"], cv=older, cv_data=reordered)
    assert noted.cv_check.reason is None
    assert noted.cv_check.note == NO_FINGERPRINT

    assert EditorSession.from_model(model, terms=["region"]).cv_check.reason == NO_CV
    with pytest.raises(TypeError, match="not a splitter"):
        EditorSession.from_model(model, cv=KFold(3))


def test_edit_takes_split_data_and_a_cv_result(cv_frame, cv_fit):
    from superglm.editor import edit
    from superglm.editor.evaluation import evaluation_datasets

    model, supplied = cv_fit
    session = edit(model, ["region"], cv=supplied, **_splits(cv_frame))

    assert session.cv is supplied
    assert [dataset.name for dataset in evaluation_datasets(session)] == [
        "train",
        "validation",
        "test",
    ]
    assert session.cv_check.reason is None


# ── The tab from a supplied result ───────────────────────────────


def test_cv_report_shows_a_supplied_result_least_stable_first(cv_frame, cv_fit):
    from superglm.plotting.curve_similarity import _summarize_against_fold_mean

    X, _y, _w = cv_frame
    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    widget = session.widget()
    try:
        report = _post_json(f"{widget.url}/report", {"report": "cv"})
    finally:
        widget.close()

    assert report["report"] == "cv"
    assert report["header"] == {"supplied": True, "n_folds": 3, "splitter": "KFold", "n_rows": 400}
    assert [metric["name"] for metric in report["metrics"]] == ["deviance", "gini", "nll"]
    [result] = report["results"]
    assert (result["label"], result["origin"], result["stale"]) == (
        "As supplied",
        "supplied",
        False,
    )
    assert [fold["n_test"] for fold in result["folds"]] == [
        len(t) for _, t in supplied.fold_indices
    ]
    assert result["mean"]["deviance"] == supplied.mean_scores["deviance"]
    assert result["pooled"]["deviance"] == supplied.pooled_scores["deviance"]
    assert report["run_cv"] == {"available": True, "reason": None, "note": None}

    relativities = report["relativities"]
    assert relativities["origin"] == "supplied"
    # A linear term is one slope, so it has no curve to compare.
    assert {item["name"] for item in relativities["terms"]} == {"age", "region"}
    spreads = [item["spread"] for item in relativities["terms"]]
    assert spreads == sorted(spreads, reverse=True)
    region = next(item for item in relativities["terms"] if item["name"] == "region")
    assert list(pd.unique(X["region"])) != ["A", "B", "C"]
    assert region["levels"] == ["A", "B", "C"]
    weights = np.asarray(region["weights"])
    curves = {fold["label"]: np.asarray(fold["values"]) for fold in region["folds"]}
    for values in curves.values():
        log_values = np.log(values)
        assert abs(np.average(log_values, weights=weights)) <= 64 * _U * np.max(np.abs(log_values))
    assert region["spread"] == _summarize_against_fold_mean(curves, weights)["rmse_to_mean"].mean()


def test_cv_report_says_how_to_get_what_is_missing(cv_frame, cv_fit):
    from superglm.editor.cv import NO_CV, NO_ESTIMATORS

    model, supplied = cv_fit
    bare = dataclasses.replace(supplied, estimators=None)
    with_bare = EditorSession.from_model(model, cv=bare, **_splits(cv_frame)).widget()
    without = EditorSession.from_model(model, **_splits(cv_frame)).widget()
    try:
        bare_report = with_bare._report("cv")
        empty_report = without._report("cv")
    finally:
        with_bare.close()
        without.close()

    assert bare_report["relativities"] == {
        "available": False,
        "origin": None,
        "stale": False,
        "note": NO_ESTIMATORS,
        "terms": [],
    }
    assert (empty_report["note"], empty_report["results"]) == (NO_CV, [])
    assert empty_report["run_cv"]["reason"] == NO_CV
    assert empty_report["final_fit"]["available"] is True


# ── A fold that never saw a level ────────────────────────────────


@pytest.fixture(scope="module")
def rare_level(cv_frame, cv_fit):
    """The train rows with region D on three rows of the first fold's test rows only."""
    X, y, w = cv_frame
    _model_unused, supplied = cv_fit
    X_rows = X.iloc[:400].reset_index(drop=True).copy()
    y_rows, w_rows = y[:400], w[:400]
    first_test = supplied.fold_indices[0][1]
    X_rows.loc[first_test[y_rows[first_test] > 0][:3], "region"] = "D"
    return X_rows, y_rows, w_rows, supplied


def _rare_level_folds(rare_level, how: str) -> list[SuperGLM]:
    """The fold models: fitted on their own rows ("unseen"), or by cross_validate ("pinned").

    cross_validate shares one level universe across folds, so there the first
    fold holds D pinned to its base; a hand-rolled fold loop never sees D.
    """
    X_rows, y_rows, w_rows, supplied = rare_level
    if how == "unseen":
        return [
            _model().fit(X_rows.iloc[train], y_rows[train], sample_weight=w_rows[train])
            for train, _test in supplied.fold_indices
        ]
    return cross_validate(
        _model(),
        X_rows,
        y_rows,
        cv=KFold(3, shuffle=True, random_state=0),
        sample_weight=w_rows,
        return_estimators=True,
    ).estimators


@pytest.mark.filterwarnings(_EXPECTED_PIN)
@pytest.mark.parametrize("how", ["unseen", "pinned"])
def test_cv_report_gives_a_fold_that_never_saw_a_level_a_gap_there(rare_level, how):
    """A level only one fold's test rows hold: that fold has no value there.

    The fold keeps its place in the term's chart with a gap (null) at the
    level, every curve is centred on the levels all folds share, and the
    ranking reads each fold on the levels it has.
    """
    from superglm.plotting.curve_similarity import _summarize_against_fold_mean

    X_rows, y_rows, w_rows, supplied = rare_level
    fold_models = _rare_level_folds(rare_level, how)
    first = fold_models[0]._specs["region"]
    if how == "unseen":
        assert "D" not in first._levels
    else:
        assert list(first._pinned_levels) == ["D"]
    assert all("D" not in fold._specs["region"]._pinned_levels for fold in fold_models[1:])
    model = _model().fit(X_rows, y_rows, sample_weight=w_rows)
    result = dataclasses.replace(supplied, estimators=fold_models)
    session = EditorSession.from_model(model, cv=result, train_data=(X_rows, y_rows, w_rows))
    widget = session.widget()
    try:
        report = _post_json(f"{widget.url}/report", {"report": "cv"})
    finally:
        widget.close()

    region = next(item for item in report["relativities"]["terms"] if item["name"] == "region")
    assert region["levels"] == ["A", "B", "C", "D"]
    folds = {fold["label"]: fold["values"] for fold in region["folds"]}
    assert list(folds) == ["Fold 1", "Fold 2", "Fold 3"]
    assert folds["Fold 1"][3] is None
    assert all(value is not None for value in folds["Fold 1"][:3])
    assert all(value is not None for label in ("Fold 2", "Fold 3") for value in folds[label])
    weights = np.asarray(region["weights"])
    shared = slice(0, 3)
    curves = {}
    for label, values in folds.items():
        log_values = np.log(np.asarray(values, dtype=np.float64))
        centre = np.average(log_values[shared], weights=weights[shared])
        assert abs(centre) <= 64 * _U * np.max(np.abs(log_values[shared]))
        curves[label] = np.exp(log_values)
    expected = _summarize_against_fold_mean(curves, weights)["rmse_to_mean"].mean()
    assert np.isfinite(region["spread"])
    assert abs(region["spread"] - expected) <= 64 * _U * expected


@pytest.mark.filterwarnings(_EXPECTED_PIN)
def test_carried_level_edit_leaves_a_pinned_level_at_its_pin(rare_level):
    """Run CV's carry onto a fold that holds a level pinned: the rest of the edit lands."""
    from superglm.editor.carry import model_with_edited_curves

    X_rows, y_rows, w_rows, supplied = rare_level
    fold = _rare_level_folds(rare_level, "pinned")[0]
    train = supplied.fold_indices[0][0]
    model = _model().fit(X_rows, y_rows, sample_weight=w_rows)
    session = EditorSession.from_model(model, train_data=(X_rows, y_rows, w_rows))
    session.select_levels("region", ["C", "D"])
    session.shift("region", 0.1)
    term = session.terms["region"]

    carried = model_with_edited_curves(
        fold,
        {"region": term.copy()},
        X_rows.iloc[train],
        y_rows[train],
        w_rows[train],
        n_points=session.n_points,
    )

    probe = pd.DataFrame({"age": [40.0] * 4, "power": [0.0] * 4, "region": ["A", "B", "C", "D"]})
    log_carried = np.log(carried.predict(probe))
    edited = term.edited_log_effect[[term.levels.index(level) for level in "ABC"]]
    tol = 64 * _U * max(1.0, np.max(np.abs(log_carried)))
    np.testing.assert_allclose(
        log_carried[:3] - log_carried[0], edited - edited[0], rtol=0.0, atol=tol
    )
    # D has no coefficient in this fold: it predicts as the base level, A.
    assert log_carried[3] == log_carried[0]


@pytest.mark.filterwarnings(_EXPECTED_PIN)
def test_carried_ordered_edit_leaves_a_pinned_special_at_its_pin():
    """The same for an ordered term's special level that one fold never saw.

    The special has no column in that fold, so its target is no row of the
    projection: an edit that also moves it carries exactly as one that does not.
    """
    from superglm import OrderedCategorical
    from superglm.editor.carry import model_with_edited_curves

    rng = np.random.default_rng(5)
    n = 300
    band = rng.choice(["1", "2", "3", "4", "5"], n).astype(object)
    y = rng.poisson(np.exp(-0.3 + 0.1 * band.astype(int))).astype(np.float64)
    band[[i for i in range(100) if y[i] > 0][:3]] = "MISSING"  # the first fold's test rows
    X = pd.DataFrame({"band": band})

    def model():
        feature = OrderedCategorical(
            order=["1", "2", "3", "4", "5"], specials=["MISSING"], basis=Spline(n_knots=2)
        )
        return SuperGLM(family="poisson", selection_penalty=0.0, features={"band": feature})

    fold = cross_validate(model(), X, y, cv=KFold(3), return_estimators=True).estimators[0]
    assert list(fold._specs["band"]._pinned_specials) == ["MISSING"]
    train = np.arange(100, n)
    probe = pd.DataFrame({"band": ["1", "2", "3", "4", "5", "MISSING"]})
    carried = []
    for levels in (["5"], ["5", "MISSING"]):
        session = EditorSession.from_model(model().fit(X, y), train_data=(X, y))
        session.select_levels("band", levels)
        session.shift("band", 0.1)
        edited = {"band": session.terms["band"].copy()}
        carried.append(model_with_edited_curves(fold, edited, X.iloc[train], y[train]))

    only_five, with_special = (np.log(model_.predict(probe)) for model_ in carried)
    assert np.all(np.isfinite(only_five))
    assert not np.array_equal(only_five, np.log(fold.predict(probe)))
    np.testing.assert_array_equal(with_special, only_five)


# ── Background jobs ──────────────────────────────────────────────


def _blocking_job(entered, release, published):
    """A job that stops after its first step until ``release`` is set."""

    def work(context):
        context.progress("fold", fold=1, n_folds=2)
        entered.set()
        assert release.wait(30)
        context.check()
        context.progress("fold", fold=2, n_folds=2)
        return "value"

    def publish(value):
        published.append(value)
        return {"published": value}

    return work, publish


def test_job_runner_cancel_between_steps_publishes_nothing():
    from superglm.editor.jobs import JobRunner

    runner = JobRunner(name="test")
    entered, release, published = threading.Event(), threading.Event(), []
    job_id = runner.start("cv", *_blocking_job(entered, release, published))
    assert entered.wait(30)

    requested = runner.cancel(job_id)
    release.set()
    finished = runner.status(job_id, wait=True)
    runner.close()

    assert requested == {"job_id": job_id, "status": "running", "cancel_requested": True}
    assert finished["status"] == "cancelled"
    assert finished["progress"] == [{"phase": "fold", "fold": 1, "n_folds": 2}]
    assert published == []


def test_job_runner_runs_one_job_per_kind_and_keeps_the_last_of_each():
    from superglm.editor.errors import EditorKeyError, EditorValueError
    from superglm.editor.jobs import JobRunner

    runner = JobRunner(name="test")
    entered, release, published = threading.Event(), threading.Event(), []
    first = runner.start("cv", *_blocking_job(entered, release, published))
    assert entered.wait(30)
    with pytest.raises(EditorValueError, match="already running"):
        runner.start("cv", *_blocking_job(entered, release, published))
    other = runner.start("final_fit", lambda context: "other", lambda value: {"value": value})
    release.set()
    assert runner.status(first, wait=True)["result"] == {"published": "value"}
    assert runner.status(other, wait=True)["status"] == "done"

    second = runner.start("cv", *_blocking_job(entered, release, published))

    assert runner.status(second, wait=True)["status"] == "done"
    with pytest.raises(EditorKeyError):
        runner.status(first)
    assert runner.latest("cv")["job_id"] == second
    assert runner.latest("final_fit")["job_id"] == other
    assert published == ["value", "value"]
    runner.close()


def test_job_runner_reports_fixed_sentences_when_a_job_fails():
    from superglm.editor.errors import EditorValueError
    from superglm.editor.jobs import JobRunner

    def leak(context):
        raise RuntimeError("backend detail that must not reach the browser")

    def refuse(context):
        raise EditorValueError("A sentence written for the browser.")

    runner = JobRunner(name="test")
    leaked = runner.status(runner.start("cv", leak, lambda value: {}), wait=True)
    refused = runner.status(runner.start("final_fit", refuse, lambda value: {}), wait=True)
    runner.close()

    assert (leaked["status"], leaked["error"]) == ("failed", "internal editor error")
    assert (refused["status"], refused["error"]) == (
        "failed",
        "A sentence written for the browser.",
    )


def test_job_routes_run_a_job_off_the_widget_lock(cv_fit):
    model, _supplied = cv_fit
    widget = EditorSession.from_model(model, terms=["region"]).widget()
    entered, release, published = threading.Event(), threading.Event(), []
    widget._job_starters["probe"] = lambda: _blocking_job(entered, release, published)
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "probe"})
        assert entered.wait(30)
        # The work is mid-run; a request that takes the widget lock still answers.
        assert _get_json(f"{widget.url}/state")["selected_term"] == "region"
        cancelled = _post_json(f"{widget.url}/job_cancel", {"job_id": started["job_id"]})
        release.set()
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        unknown = _post_error(f"{widget.url}/job_start", {"kind": "nonsense"})
    finally:
        release.set()
        widget.close()

    assert started["status"] == "running"
    assert cancelled["cancel_requested"] is True
    assert finished["status"] == "cancelled"
    assert published == []
    assert unknown == (400, {"error": "Unknown job kind."})
