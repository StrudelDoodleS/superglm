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


def _model(selection_penalty=0.0) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=selection_penalty,
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
    # Without a fingerprint the row count is still checked, and refuses with no note.
    older_fewer = EditorSession.from_model(
        model, terms=["region"], cv=older, cv_data=(X.iloc[:399], y[:399], w[:399])
    )
    assert older_fewer.cv_check.rows is None
    assert (older_fewer.cv_check.reason, older_fewer.cv_check.note) == (
        ROWS_MISMATCH.format(rows=399, expected=400),
        None,
    )
    older_wider = EditorSession.from_model(
        model, terms=["region"], train_data=(X.iloc[:500], y[:500], w[:500]), cv=older
    )
    assert (older_wider.cv_check.reason, older_wider.cv_check.note) == (
        TRAIN_ROWS_MISMATCH.format(rows=500, expected=400),
        None,
    )

    assert EditorSession.from_model(model, terms=["region"]).cv_check.reason == NO_CV
    with pytest.raises(TypeError, match="not a splitter"):
        EditorSession.from_model(model, cv=KFold(3))


def test_cv_data_with_rows_swapped_between_equal_responses_is_refused():
    # Rows 0 and 100 share their response and weight, so swapping them leaves
    # y and the weights byte for byte the same. Replaying the GroupKFold folds
    # on the swapped frame would put groups A and F on both sides of a fold.
    from sklearn.model_selection import GroupKFold

    from superglm.editor.cv import FINGERPRINT_MISMATCH

    n = 120
    X = pd.DataFrame({"group": np.repeat(list("ABCDEF"), 20), "x": np.linspace(-1.0, 1.0, n)})
    y = np.tile([0.0, 1.0], n // 2)

    def template():
        return SuperGLM(family="binomial", selection_penalty=0.0, features={"x": Numeric()})

    model = template().fit(X, y)
    supplied = cross_validate(template(), X, y, cv=GroupKFold(3), groups=X["group"].to_numpy())
    swapped = X.copy()
    swapped.iloc[[0, 100]] = X.iloc[[100, 0]].to_numpy()
    groups = swapped["group"].to_numpy()
    leaked = [set(groups[train]) & set(groups[test]) for train, test in supplied.fold_indices]
    assert {"A", "F"} <= set().union(*leaked)

    same = EditorSession.from_model(model, cv=supplied, cv_data=(X.copy(), y, np.ones(n)))
    assert (same.cv_check.reason, same.cv_check.note) == (None, None)
    moved = EditorSession.from_model(model, cv=supplied, cv_data=(swapped, y))
    assert moved.cv_check.reason == FINGERPRINT_MISMATCH
    offset = EditorSession.from_model(model, cv=supplied, cv_data=(X, y, None, np.full(n, 0.5)))
    assert offset.cv_check.reason == FINGERPRINT_MISMATCH


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


@pytest.mark.parametrize("unseen", ["error", "base"])
def test_cv_report_reads_integer_coded_fold_models_after_a_collapse(unseen):
    """A collapse gives the in-force term text labels; fold models fitted on integers still score.

    Under ``unseen="error"`` the text labels were refused and the term left
    the list; under ``"base"`` every fold read flat at 1.
    """
    from superglm.editor.cv import capture_cv_view, cv_tab_payload

    rng = np.random.default_rng(7)
    n = 1200
    cls = rng.choice([10, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1])
    X = pd.DataFrame({"age": rng.uniform(18.0, 80.0, n), "cls": cls})
    eta = -0.5 + np.select([cls == 2, cls == 3], [0.3, -0.2], 0.0)
    y = rng.poisson(np.exp(eta)).astype(np.float64)

    def make():
        return SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"age": Spline(n_knots=6), "cls": Categorical(base=1, unseen=unseen)},
        )

    supplied = cross_validate(
        make(), X, y, cv=KFold(4, shuffle=True, random_state=0), return_estimators=True
    )
    session = EditorSession.from_model(make().fit(X, y), cv=supplied, train_data=(X, y))

    def cls_folds():
        view = capture_cv_view(session, run=None, final_fit=None)
        terms = cv_tab_payload(view, jobs={})["relativities"]["terms"]
        item = next((item for item in terms if item["name"] == "cls"), None)
        assert item is not None, [item["name"] for item in terms]
        return item["levels"], np.array([fold["values"] for fold in item["folds"]])

    levels, before = cls_folds()
    session.select_levels("cls", ["2", "3"])
    session.replace_with_collapsed_levels("cls", method="fit")
    assert session.terms["cls"].metadata["native_levels"] == ["1", "2", "3", "10"]
    after_levels, after = cls_folds()

    # The supplied folds are the same models, read on the same levels.
    assert after_levels == levels == ["1", "2", "3", "10"]
    assert np.ptp(after[:, 2]) > 0.0
    np.testing.assert_allclose(after, before, rtol=64 * _U)


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
    """The train rows with region D on three rows of the first fold's test rows only.

    The result is cross_validate on these rows: the splitter draws the same
    folds on the same row count, and the result's fingerprint holds the D rows.
    """
    X, y, w = cv_frame
    _model_unused, supplied = cv_fit
    X_rows = X.iloc[:400].reset_index(drop=True).copy()
    y_rows, w_rows = y[:400], w[:400]
    first_test = supplied.fold_indices[0][1]
    X_rows.loc[first_test[y_rows[first_test] > 0][:3], "region"] = "D"
    with pytest.warns(UserWarning, match="pinned to"):
        on_rows = cross_validate(
            _model(),
            X_rows,
            y_rows,
            cv=KFold(3, shuffle=True, random_state=0),
            sample_weight=w_rows,
            scoring=("deviance", "gini", "nll"),
        )
    for (train, test), (again_train, again_test) in zip(
        supplied.fold_indices, on_rows.fold_indices, strict=True
    ):
        np.testing.assert_array_equal(again_train, train)
        np.testing.assert_array_equal(again_test, test)
    return X_rows, y_rows, w_rows, on_rows


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


def test_job_runner_cancel_after_the_work_returns_publishes_nothing():
    """A cancel that lands after the work's last check of its own still stops the publish."""
    from superglm.editor.jobs import JobRunner

    runner = JobRunner(name="test")
    started, job, published = threading.Event(), [], []

    def work(context):
        context.check()
        assert started.wait(30)
        runner.cancel(job[0])
        return "value"

    job.append(runner.start("cv", work, lambda value: published.append(value) or {}))
    started.set()
    finished = runner.status(job[0], wait=True)
    runner.close()

    assert finished["status"] == "cancelled"
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


# ── Run CV and Final fit ─────────────────────────────────────────


class _Context:
    """A job context that never cancels and records progress."""

    def __init__(self):
        self.entries = []

    def check(self):
        return None

    def progress(self, phase, **details):
        self.entries.append({"phase": phase, **details})


@pytest.fixture
def fit_rows(monkeypatch):
    """The row count of every SuperGLM.fit call, in order."""
    rows = []
    fit = SuperGLM.fit

    def counted(self, X, y, *args, **kwargs):
        rows.append(len(y))
        return fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(SuperGLM, "fit", counted)
    return rows


def test_run_cv_reproduces_the_supplied_scores_when_nothing_is_edited(cv_frame, cv_fit, fit_rows):
    from superglm.editor.cv import capture_cv_run, run_cv

    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    context = _Context()

    run = run_cv(capture_cv_run(session), context)

    assert fit_rows == [len(train) for train, _test in supplied.fold_indices]
    assert [entry["fold"] for entry in context.entries if entry["phase"] == "fold"] == [1, 2, 3]
    # The same folds, structure and scorers in this thread: the same numbers.
    for name in ("deviance", "gini", "nll"):
        np.testing.assert_array_equal(run.result.fold_scores[name], supplied.fold_scores[name])
    assert run.result.pooled_scores == supplied.pooled_scores
    assert run.result.splitter == "KFold"


def test_run_cv_job_puts_the_hand_edits_back_on_every_fold(cv_frame, cv_fit, fit_rows):
    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    widget = session.widget()
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = _post_json(f"{widget.url}/report", {"report": "cv"})
    finally:
        widget.close()

    assert finished["status"] == "done"
    assert [entry["fold"] for entry in finished["progress"] if entry["phase"] == "fold"] == [
        1,
        2,
        3,
    ]
    assert len(fit_rows) == 3
    supplied_result, current = report["results"]
    assert (current["label"], current["origin"], current["stale"]) == (
        "Current model",
        "run",
        False,
    )
    for edited_fold, supplied_fold in zip(current["folds"], supplied_result["folds"], strict=True):
        assert edited_fold["scores"]["deviance"] != supplied_fold["scores"]["deviance"]
    assert report["relativities"]["origin"] == "run"
    region = next(item for item in report["relativities"]["terms"] if item["name"] == "region")
    # Every fold carries the edited curve, so the folds agree on region exactly.
    for fold in region["folds"]:
        np.testing.assert_allclose(fold["values"], region["edited"], rtol=64 * _U)
    # Its spread measures nothing, so region reads as held, after the measured terms.
    assert (region["held"], region["spread"], region["min_correlation"]) == (True, None, None)
    age, held = report["relativities"]["terms"]
    assert (age["name"], age["held"], held["name"]) == ("age", False, "region")
    assert age["spread"] > 0.0


def test_run_cv_puts_the_hand_edits_back_on_each_fold_with_that_folds_training_rows(
    cv_frame, cv_fit
):
    """Each fold's edits are carried with the rows that fold was fitted on.

    The oracle replays every fold by hand: the same clone and fit as
    ``cross_validate``, the edited curves carried with that fold's training
    rows, then the built-in scores on its test rows. The same operations in
    the same thread give the same numbers.
    """
    from superglm.editor.carry import model_with_edited_curves
    from superglm.editor.cv import capture_cv_run, run_cv
    from superglm.model_selection import _BUILTIN_SCORERS, _clone_model

    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    session.select_indices("age", list(range(120, 200)))
    session.shift("age", 0.3)
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    plan = capture_cv_run(session)

    run = run_cv(plan, _Context())

    frame = plan.rows.X
    y = np.asarray(plan.rows.y, dtype=np.float64)
    w = np.asarray(plan.rows.sample_weight, dtype=np.float64)
    for index, (train, test) in enumerate(plan.folds):
        fold = _clone_model(plan.model)
        getattr(fold, plan.fit_mode)(frame.iloc[train], y[train], sample_weight=w[train])
        fold = model_with_edited_curves(
            fold,
            plan.edited,
            frame.iloc[train],
            y[train],
            w[train],
            None,
            n_points=plan.n_points,
        )
        for name in ("deviance", "gini", "nll"):
            expected = _BUILTIN_SCORERS[name](
                fold, frame.iloc[test], y[test], sample_weight=w[test], offset=None
            )
            assert run.result.fold_scores[name][index] == expected, (index, name)


def test_run_cv_is_refused_with_its_reason(cv_frame, cv_fit):
    from superglm.editor.cv import ROWS_MISMATCH

    X, y, w = cv_frame
    model, supplied = cv_fit
    waiting = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    waiting.stage_structural("collapse", "region", {"levels": ["B", "C"], "group_label": None})
    mismatched = EditorSession.from_model(
        model, cv=supplied, cv_data=(X.iloc[:399], y[:399], w[:399])
    )
    seen = {}
    for name, session in {"waiting": waiting, "mismatched": mismatched}.items():
        widget = session.widget()
        try:
            report = widget._report("cv")
            seen[name] = (report, _post_error(f"{widget.url}/job_start", {"kind": "cv"}))
        finally:
            widget.close()

    report, refused = seen["waiting"]
    assert report["run_cv"] == {
        "available": False,
        "reason": "Refit first: 1 change is waiting.",
        "note": None,
    }
    assert refused == (400, {"error": "Refit first: 1 change is waiting."})
    assert report["final_fit"]["note"] == "Final fit: 1 waiting change is not included."
    report, refused = seen["mismatched"]
    assert report["run_cv"]["reason"] == ROWS_MISMATCH.format(rows=399, expected=400)
    assert refused == (400, {"error": report["run_cv"]["reason"]})


def test_run_cv_cancelled_mid_run_publishes_nothing(cv_frame, cv_fit, fit_rows, monkeypatch):
    from superglm.editor.jobs import JobContext

    model, supplied = cv_fit
    widget = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame)).widget()
    progress = JobContext.progress

    def cancel_before_the_second_fold(self, phase, **details):
        progress(self, phase, **details)
        if details.get("fold") == 2:
            widget._job_cancel(self.job_id)

    monkeypatch.setattr(JobContext, "progress", cancel_before_the_second_fold)
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = widget._report("cv")
    finally:
        widget.close()

    assert finished["status"] == "cancelled"
    assert [entry["fold"] for entry in finished["progress"]] == [1, 2]
    assert len(fit_rows) == 1
    assert [result["origin"] for result in report["results"]] == ["supplied"]
    assert report["relativities"]["origin"] == "supplied"
    assert report["jobs"]["cv"]["status"] == "cancelled"


@pytest.mark.parametrize("how", ["carry", "fit"])
def test_run_cv_stops_at_a_fold_that_fails_and_publishes_nothing(how, monkeypatch):
    """A fold that cannot be fitted or scored stops Run CV with its reason.

    ``carry``: the CV rows reach past the editor's train rows, and the edited
    term refuses them (``extrapolation="error"``) when the hand edits are put
    back on the first fold. ``fit``: the second fold's fit fails. Either way
    the folds that did score are not averaged as if they were all.
    """
    from superglm import Piecewise

    rng = np.random.default_rng(1)
    n = 1200
    age = rng.uniform(18.0, 80.0, n)
    X = pd.DataFrame({"age": age, "power": rng.normal(0.0, 1.0, n)})
    y = rng.poisson(np.exp(-0.5 + 0.01 * (age - 50.0))).astype(np.float64)
    supplied = cross_validate(
        _model_without_region(), X, y, cv=KFold(3, shuffle=True, random_state=0)
    )
    if how == "carry":
        inner = (age > 25.0) & (age < 70.0)
        piecewise = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={
                "age": Piecewise(breaks=[30.0, 50.0], extrapolation="error"),
                "power": Numeric(),
            },
        ).fit(X[inner], y[inner])
        session = EditorSession.from_model(
            piecewise, cv=supplied, cv_data=(X, y), train_data=(X[inner], y[inner])
        )
        session.select_indices("age", [1])
        session.shift("age", 0.1)
        fold, reason = 1, "Term 'age' received values outside the rated range"
    else:
        session = EditorSession.from_model(
            _model_without_region().fit(X, y), cv=supplied, cv_data=(X, y)
        )
        fit = SuperGLM.fit
        calls = []

        def second_fit_fails(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return fit(self, *args, **kwargs)

        monkeypatch.setattr(SuperGLM, "fit", second_fit_fails)
        fold, reason = 2, "That fold could not be fitted or scored."
    widget = session.widget()
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = widget._report("cv")
    finally:
        widget.close()

    assert finished["status"] == "failed"
    assert finished["error"].startswith(
        f"Run CV stopped at fold {fold} and kept no result. {reason}"
    )
    assert widget._cv_run is None
    assert [result["origin"] for result in report["results"]] == ["supplied"]


def _model_without_region() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(n_knots=6), "power": Numeric()},
    )


def test_run_cv_result_is_dropped_when_the_model_changes_mid_run(cv_frame, cv_fit, monkeypatch):
    from superglm.editor.cv import SUPERSEDED
    from superglm.editor.jobs import JobContext

    model, supplied = cv_fit
    widget = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame)).widget()
    progress = JobContext.progress

    def edit_during_the_first_fold(self, phase, **details):
        progress(self, phase, **details)
        if details.get("fold") == 1:
            widget._drag("region", [1], delta=0.1)

    monkeypatch.setattr(JobContext, "progress", edit_during_the_first_fold)
    try:
        started = _post_json(f"{widget.url}/job_start", {"kind": "cv"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        report = widget._report("cv")
    finally:
        widget.close()

    assert (finished["status"], finished["error"]) == ("failed", SUPERSEDED)
    assert [result["origin"] for result in report["results"]] == ["supplied"]


def test_final_fit_refits_train_and_validation_and_export_offers_it(cv_frame, cv_fit, fit_rows):
    from superglm.editor.cv import FINAL_NOT_RUN, FINAL_STALE
    from superglm.editor.errors import EditorValueError
    from superglm.editor.persistence import joblib_load_bytes

    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    session.select_levels("region", ["C"])
    session.shift("region", 0.1)
    edited = session.terms["region"].edited_log_effect.copy()
    widget = session.widget()
    try:
        with pytest.raises(EditorValueError) as before:
            widget._export_bytes("final")
        started = _post_json(f"{widget.url}/job_start", {"kind": "final_fit"})
        finished = _post_json(
            f"{widget.url}/job_status", {"job_id": started["job_id"], "wait": True}
        )
        state = _get_json(f"{widget.url}/state")
        section = widget._report("final")["final_fit"]
        exported = widget._export_bytes("final")
        history = session.editor_history_records()
        kept = widget._final_fit.model
        widget._drag("region", [0], delta=0.05)
        with pytest.raises(EditorValueError) as stale:
            widget._export_bytes("final")
    finally:
        widget.close()

    assert before.value.public_message == FINAL_NOT_RUN
    assert finished["status"] == "done"
    # Train and validation rows, one fit; the test split stays held out (D6).
    assert finished["result"]["n_rows"] == 500
    assert fit_rows == [500]
    assert state["final_fit"] == {"available": True, "stale": False}
    assert (section["n_rows"], section["splits"], section["carried"]) == (
        500,
        ["train", "validation"],
        ["region"],
    )
    assert exported.filename == "superglm_final_model.joblib"
    final_model = joblib_load_bytes(exported.data)
    # Like the edited model's export, it carries the editor's history; the
    # Final fit model the widget keeps is left without it.
    assert final_model._editor_history == history
    assert not hasattr(kept, "_editor_history")
    probe = pd.DataFrame({"age": [40.0] * 3, "power": [0.0] * 3, "region": ["A", "B", "C"]})
    log_mu = np.log(final_model.predict(probe))
    # The hand edit is put back as set: the final model's region relativities are the edited ones.
    np.testing.assert_allclose(
        log_mu - log_mu[0],
        edited - edited[0],
        rtol=0.0,
        atol=64 * _U * max(1.0, np.max(np.abs(log_mu))),
    )
    assert stale.value.public_message == FINAL_STALE


@pytest.mark.parametrize(
    ("column", "lacking"),
    [
        ("offset", "validation"),
        ("offset", "train"),
        ("sample_weight", "validation"),
        ("sample_weight", "train"),
    ],
)
def test_final_fit_refuses_an_offset_or_weights_only_one_split_carries(cv_frame, column, lacking):
    """Stacking the splits must not fill in an offset or weights one split lacks.

    A Poisson book fitted on log exposure, with the validation rows passed
    without it, would otherwise be fitted on exposure 1 there.
    """
    from superglm.editor.cv import FINAL_SPLIT_MISSING, capture_cv_view, capture_final_fit
    from superglm.editor.errors import EditorValueError

    X, y, w = cv_frame
    values = np.log(w) if column == "offset" else w
    given = {
        split: None if split == lacking else values[rows]
        for split, rows in (("train", slice(0, 400)), ("validation", slice(400, 500)))
    }
    model = _model().fit(X.iloc[:400], y[:400], **{column: given["train"]})

    def split(name, rows):
        extra = {"sample_weight": None, "offset": None, column: given[name]}
        return (X.iloc[rows], y[rows], extra["sample_weight"], extra["offset"])

    session = EditorSession.from_model(
        model,
        train_data=split("train", slice(0, 400)),
        validation_data=split("validation", slice(400, 500)),
    )
    have = "train" if lacking == "validation" else "validation"
    expected = FINAL_SPLIT_MISSING.format(
        column="offsets" if column == "offset" else "sample weights", have=have, lack=lacking
    )

    assert capture_cv_view(session, run=None, final_fit=None).final_reason == expected
    with pytest.raises(EditorValueError) as refused:
        capture_final_fit(session)
    assert refused.value.public_message == expected


def test_final_fit_stacks_a_split_without_weights_beside_unit_weights(cv_frame):
    """A model that kept unweighted fit data holds weights of 1: the validation rows match them.

    Guards the refusal above against refusing what the fill gets exactly right.
    """
    from superglm.editor.cv import _union_rows, capture_cv_view, capture_final_fit

    X, y, _w = cv_frame
    model = _model().fit(X.iloc[:400], y[:400])
    session = EditorSession.from_model(model, validation_data=(X.iloc[400:500], y[400:500]))

    assert capture_cv_view(session, run=None, final_fit=None).final_reason is None
    _X, stacked_y, weights, offset = _union_rows(capture_final_fit(session).datasets)
    assert stacked_y.size == 500 and offset is None
    np.testing.assert_array_equal(weights, np.ones(500))


def test_run_cv_and_final_fit_fit_off_the_widget_lock(cv_frame, cv_fit, monkeypatch):
    """While each fit of a real Run CV or Final fit runs, another thread can take the lock.

    The widget lock is re-entrant, so a probe from the job's own thread would
    get it even if the job held it; the probe takes it from a thread of its own.
    """
    model, supplied = cv_fit
    widget = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame)).widget()
    fit = SuperGLM.fit
    taken = []

    def probed(self, X, y, *args, **kwargs):
        got = []

        def take():
            # An unheld lock is taken at once; the timeout only bounds a regression.
            if widget._lock.acquire(timeout=5):
                widget._lock.release()
                got.append(True)

        other = threading.Thread(target=take)
        other.start()
        other.join()
        taken.append(bool(got))
        return fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(SuperGLM, "fit", probed)
    finished = {}
    try:
        for kind in ("cv", "final_fit"):
            started = _post_json(f"{widget.url}/job_start", {"kind": kind})
            status = widget._job_status(started["job_id"], wait=True)
            finished[kind] = status["status"]
    finally:
        widget.close()

    assert finished == {"cv": "done", "final_fit": "done"}
    # Three fold fits, then the Final fit.
    assert taken == [True, True, True, True]


def test_run_cv_and_final_fit_refit_the_structure_from_the_last_refit(cv_frame, cv_fit):
    """After a Refit that collapses region B and C, every fold and the Final fit keep them as one.

    The source model fits B and C apart, so only the refitted structure
    gives them one value.
    """
    from superglm.editor.cv import capture_cv_run, capture_final_fit, run_cv, run_final_fit

    model, supplied = cv_fit
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    session.stage_structural("collapse", "region", {"levels": ["B", "C"], "group_label": None})
    session.refit_pending(method="fit")

    run = run_cv(capture_cv_run(session), _Context())
    final = run_final_fit(capture_final_fit(session), _Context())

    region = next(item for item in run.terms if item["name"] == "region")
    assert region["levels"] == ["A", "B", "C"]
    assert [fold["label"] for fold in region["folds"]] == ["Fold 1", "Fold 2", "Fold 3"]
    for fold in region["folds"]:
        _a, b, c = fold["values"]
        np.testing.assert_allclose(c, b, rtol=64 * _U)
    probe = pd.DataFrame({"age": [40.0] * 3, "power": [0.0] * 3, "region": ["A", "B", "C"]})
    log_mu = np.log(final.model.predict(probe))
    np.testing.assert_allclose(
        log_mu[2], log_mu[1], rtol=0.0, atol=64 * _U * max(1.0, np.max(np.abs(log_mu)))
    )


@pytest.mark.parametrize(("fit_mode", "selection"), [("fit", "auto"), ("fit_reml", 0.0)])
def test_run_cv_and_final_fit_recalibrate_the_declared_penalties_after_a_refit(
    cv_frame, fit_mode, selection
):
    """A Refit that changes nothing leaves Run CV and Final fit as they were.

    The Refit's model declares the selection penalty and smoothing its own fit
    chose on all the training rows. Each fold and the Final fit choose them
    again on their own rows, as the opened model declares.
    """
    from superglm.editor.cv import capture_cv_run, capture_final_fit, run_cv, run_final_fit

    X, y, w = cv_frame
    rows = (X.iloc[:400], y[:400])
    model = getattr(_model(selection), fit_mode)(*rows, sample_weight=w[:400])
    supplied = cross_validate(
        _model(selection),
        *rows,
        cv=KFold(3, shuffle=True, random_state=0),
        sample_weight=w[:400],
        fit_mode=fit_mode,
        scoring=("deviance", "gini", "nll"),
    )
    session = EditorSession.from_model(model, cv=supplied, **_splits(cv_frame))
    before = run_final_fit(capture_final_fit(session), _Context()).model
    session.stage_structural("set_reference", "region", {"level": "A"})
    session.refit_pending()

    run = run_cv(capture_cv_run(session), _Context())
    after = run_final_fit(capture_final_fit(session), _Context()).model

    for name in ("deviance", "gini", "nll"):
        np.testing.assert_array_equal(run.result.fold_scores[name], supplied.fold_scores[name])
    assert after.selection_penalty_ == before.selection_penalty_
    np.testing.assert_array_equal(after.predict(X.iloc[500:]), before.predict(X.iloc[500:]))


def test_run_cv_takes_the_in_force_fit_method_and_the_supplied_scorers(cv_frame, cv_fit):
    from superglm.editor.cv import capture_cv_run

    X, y, w = cv_frame
    model, supplied = cv_fit
    reml = _model().fit_reml(X.iloc[:400], y[:400], sample_weight=w[:400])
    scores = supplied.fold_scores
    deviance_only = dataclasses.replace(supplied, fold_scores=scores.drop(columns=["gini", "nll"]))
    no_builtins = dataclasses.replace(
        supplied, fold_scores=scores.drop(columns=["deviance", "gini", "nll"])
    )

    def plan(fitted, result):
        return capture_cv_run(EditorSession.from_model(fitted, cv=result, **_splits(cv_frame)))

    assert plan(model, supplied).fit_mode == "fit"
    assert plan(reml, supplied).fit_mode == "fit_reml"
    assert plan(model, deviance_only).scoring == ("deviance",)
    assert plan(model, no_builtins).scoring == ("deviance", "gini", "nll")


@pytest.mark.filterwarnings(_EXPECTED_PIN)
def test_run_cv_scores_every_fold_when_one_fold_never_trained_on_a_level(rare_level):
    """Run CV on rows where region D sits in the first fold's test rows only.

    cross_validate's shared universe gives that fold D with no training
    rows, so it holds D pinned. The edit, which moves D too, still lands on
    the rest of that fold's region: every fold scores, and the first fold's
    region curve has a gap at D instead of a value.
    """
    from superglm.editor.cv import capture_cv_run, run_cv

    X_rows, y_rows, w_rows, supplied = rare_level
    model = _model().fit(X_rows, y_rows, sample_weight=w_rows)
    session = EditorSession.from_model(model, cv=supplied, train_data=(X_rows, y_rows, w_rows))
    session.select_levels("region", ["C", "D"])
    session.shift("region", 0.1)
    assert session.cv_check.reason is None

    run = run_cv(capture_cv_run(session), _Context())

    for name in ("deviance", "gini", "nll"):
        assert np.all(np.isfinite(run.result.fold_scores[name]))
    region = next(item for item in run.terms if item["name"] == "region")
    assert region["levels"] == ["A", "B", "C", "D"]
    folds = {fold["label"]: np.asarray(fold["values"]) for fold in region["folds"]}
    assert list(folds) == ["Fold 1", "Fold 2", "Fold 3"]
    assert np.isnan(folds["Fold 1"][3])
    assert not np.isnan(folds["Fold 2"]).any() and not np.isnan(folds["Fold 3"]).any()
    # Each fold carries the edit wherever it has a value.
    for values in folds.values():
        valued = ~np.isnan(values)
        np.testing.assert_allclose(values[valued], region["edited"][valued], rtol=64 * _U)


# ── The tab in the app ───────────────────────────────────────────


def test_app_has_a_cross_validation_tab_and_a_final_fit_export():
    from pathlib import Path

    import superglm.editor

    root = Path(superglm.editor.__file__).parent / "app"
    html = (root / "index.html").read_text()

    assert html.index('id="validationTab"') < html.index('id="cvTab"') < html.index('id="finalTab"')
    assert 'data-view="cv"' in html
    assert 'id="exportFinalFit" type="radio" name="exportFormat" value="final"' in html
    assert '<link rel="stylesheet" href="/assets/styles/cv.css">' in html


def test_cv_report_folds_carry_their_own_fold_number(cv_frame, cv_fit):
    """A term's fold curves name their fold, so a fold missing from a term
    keeps its colour and its place in the tab's charts."""
    from superglm.editor.cv import capture_cv_view, cv_tab_payload

    model, supplied = cv_fit
    estimators = list(supplied.estimators)
    estimators[1] = None
    session = EditorSession.from_model(
        model, cv=dataclasses.replace(supplied, estimators=estimators), **_splits(cv_frame)
    )

    view = capture_cv_view(session, run=None, final_fit=None)
    terms = cv_tab_payload(view, jobs={})["relativities"]["terms"]

    assert {item["name"] for item in terms} == {"age", "region"}
    for item in terms:
        assert [(fold["fold"], fold["label"]) for fold in item["folds"]] == [
            (0, "Fold 1"),
            (2, "Fold 3"),
        ]
