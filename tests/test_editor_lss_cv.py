"""The Cross-validation tab and Run CV on a SuperLSS."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from sklearn.model_selection import KFold

from superglm import (
    GammaLS,
    LogNormalLS,
    NegativeBinomialLS,
    OrderedCategorical,
    Spline,
    SuperLSS,
    cat,
    cross_validate,
    s,
    term,
)
from superglm.distributional.cross_validation import _POOLED_PARTS
from superglm.editor import edit
from superglm.editor.carry import model_with_edited_curves
from superglm.editor.cv import (
    FINGERPRINT_MISMATCH,
    capture_cv_run,
    fold_log_curves,
    fold_term_items,
    run_cv,
    run_cv_reason,
    weighted_mean,
)
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.display import fold_chart_display
from superglm.editor.lss.view import LSSModelView
from superglm.model_selection import _take_offset
from tests._lss_editor_fixtures import REGIONS, gamma_case, lognormal_case, nb_case, open_case

FOLDS = KFold(3, shuffle=True, random_state=4)
SCORES = ("nll", "crps", "gini", "actual_expected")
KNOTS = {"count": 5, "strategy": "uniform"}


class _Context:
    def check(self):
        return None

    def progress(self, *args, **kwargs):
        return None


def _supplied(case, **kw):
    return cross_validate(
        case.model,
        case.X,
        case.y,
        cv=FOLDS,
        offset=case.offsets,
        fit_mode="fit_reml",
        return_estimators=True,
        error_score="raise",
        **kw,
    )


@pytest.fixture(scope="module")
def lognormal_cv():
    return _supplied(lognormal_case())


@pytest.fixture(scope="module")
def nb_cv():
    return _supplied(nb_case())


def test_the_tab_reports_a_supplied_result_in_the_validation_tab_labels(lognormal_cv):
    widget = open_case(lognormal_case(), cv=lognormal_cv).widget()
    try:
        report = widget._report("cv")
    finally:
        widget.close()
    assert report["available"] and report["run_cv"]["available"]
    assert not report["results"][0]["stale"]
    assert [
        (m["name"], m["label"], m["short"], m["lower_is_better"]) for m in report["metrics"]
    ] == [
        ("nll", "Log score", "Log score", True),
        ("crps", "CRPS", "CRPS", True),
        ("gini", "Gini", "Gini", False),
        ("actual_expected", "Actual / expected", "A/E", None),
    ]
    result = report["results"][0]
    assert result["label"] == "As supplied"
    assert len(result["folds"]) == 3
    for name in SCORES:
        assert result["mean"][name] == pytest.approx(lognormal_cv.mean_scores[name])
    assert set(result["pooled"]) == {"nll", "crps", "actual_expected"}
    items = report["relativities"]["terms"]
    assert report["relativities"]["origin"] == "supplied"
    assert {item["name"] for item in items} == {
        "mean:age",
        "mean:region",
        "mean:power",
        "scale:age",
        "scale:region",
    }
    labels = {item["name"]: item["label"] for item in items}
    assert labels["mean:age"] == "age · Average"
    assert labels["scale:region"] == "region · Spread"
    # The average terms come first; each group runs least stable first.
    assert [item["group"] for item in items] == ["Average"] * 3 + ["Spread"] * 2
    assert all(item["role"] == item["group"].lower() for item in items)
    assert {item["short"] for item in items} == {"age", "region", "power"}
    for group in ("Average", "Spread"):
        spreads = [item["fold_gap"] for item in items if item["group"] == group]
        assert spreads == sorted(spreads, reverse=True)
    assert all(len(item["folds"]) == 3 for item in items)


@pytest.mark.parametrize(
    ("case", "fixture"), [(lognormal_case, "lognormal_cv"), (nb_case, "nb_cv")]
)
def test_fold_curves_are_each_fold_models_own_term_effects(case, fixture, request):
    fitted = case()
    cv = request.getfixturevalue(fixture)
    session = open_case(fitted, cv=cv)
    for fold, (train, _test) in zip(cv.estimators, cv.fold_indices, strict=True):
        curves = fold_log_curves(fold, session.terms)
        assert set(curves) == set(session.terms)
        for key, values in curves.items():
            parameter, column = key.split(":")
            term = session.terms[key]
            effect = fold.term_inference(
                parameter, column, X_train=fitted.X.iloc[train], simultaneous=False
            )
            if term.levels is None:
                # The tab's reader on the sweep's own grid reads what inference reports.
                reader = LSSModelView(fold, None)
                own_grid = reader._specs[key].score(effect.x, reader.term_coefficients(key))
                assert np.ptp(np.asarray(own_grid) - effect.effect) < 1e-9
                # And the tab reads the editor's grid: the sweep is fine enough to follow it.
                expected = np.interp(term.x, effect.x, effect.effect)
                assert np.ptp(values - expected) < 2e-2
            else:
                by_level = dict(zip(effect.levels, effect.effect, strict=True))
                position = {label: i for i, label in enumerate(term.levels)}
                for label, index in position.items():
                    if label in by_level:
                        assert values[index] == pytest.approx(by_level[label], abs=1e-9)


def test_average_terms_stay_first_even_when_a_spread_term_moves_most(nb_cv):
    # Fold curves that barely move in the mean and move a lot in theta.
    session = open_case(nb_case())
    terms = {key: session.terms[key] for key in ("mean:age", "theta:age")}
    base = {key: term.original_log_effect for key, term in terms.items()}
    curves = {
        fold: {
            "mean:age": base["mean:age"]
            + 1e-3 * fold * np.linspace(0.0, 1.0, terms["mean:age"].size),
            "theta:age": base["theta:age"] + fold * np.linspace(0.0, 1.0, terms["theta:age"].size),
        }
        for fold in range(3)
    }
    items = fold_term_items(terms, curves, display=fold_chart_display)
    assert items[0]["fold_gap"] < items[1]["fold_gap"]
    assert [item["name"] for item in items] == ["mean:age", "theta:age"]
    # Without a display there are no groups, and the order is by stability alone.
    plain = fold_term_items(terms, curves)
    assert [item["name"] for item in plain] == ["theta:age", "mean:age"]
    assert {item["group"] for item in plain} == {None}


def test_a_theta_curve_is_drawn_as_the_spread_the_editor_shows(nb_cv):
    widget = open_case(nb_case(), cv=nb_cv).widget()
    try:
        items = {item["name"]: item for item in widget._report("cv")["relativities"]["terms"]}
    finally:
        widget.close()
    session = open_case(nb_case())
    theta = "theta:age"
    term = session.terms[theta]
    weights = np.asarray(term.weights, dtype=np.float64)
    log = term.original_log_effect
    expected = np.exp(-(log - weighted_mean(log, weights)))
    assert items[theta]["fit"] == pytest.approx(expected, rel=1e-12)
    assert items[theta]["label"] == "age · Spread"
    mean = session.terms["mean:age"]
    log = mean.original_log_effect
    expected = np.exp(log - weighted_mean(log, np.asarray(mean.weights, dtype=np.float64)))
    assert items["mean:age"]["fit"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("case", "fixture"), [(lognormal_case, "lognormal_cv"), (nb_case, "nb_cv")]
)
def test_run_cv_on_the_unchanged_model_repeats_the_supplied_scores(case, fixture, request):
    cv = request.getfixturevalue(fixture)
    session = open_case(case(), cv=cv)
    assert run_cv_reason(session) is None
    run = run_cv(capture_cv_run(session), _Context())
    assert run.result.fit_mode == "fit_reml"
    for name in cv.builtin_scores:
        assert run.result.mean_scores[name] == pytest.approx(cv.mean_scores[name], rel=1e-9)
    for name in cv.pooled_scores:
        assert run.result.pooled_scores[name] == pytest.approx(cv.pooled_scores[name], rel=1e-9)
    assert not run.carried
    assert all(item["label"] for item in run.terms)


def test_run_cv_puts_hand_edits_back_on_each_fold_before_scoring():
    case = gamma_case()
    cv = cross_validate(
        case.model,
        case.X,
        case.y,
        cv=FOLDS,
        fit_mode="fit_reml",
        scoring=("nll", "crps", "actual_expected"),
        warm_start=False,
        error_score="raise",
    )
    session = open_case(case, cv=cv)
    key = "scale:age"
    term = session.terms[key]
    session.select_indices(key, list(range(term.size // 3, 2 * term.size // 3)))
    session.shift(key, 0.4)
    plan = capture_cv_run(session)
    assert list(plan.edited) == [key]
    run = run_cv(plan, _Context())
    assert run.carried == (key,)
    held = next(item for item in run.terms if item["name"] == key)
    assert held["held"]
    assert held["fold_gap"] is None
    # Scored as a fold fitted, edited and read by hand.
    train, test = cv.fold_indices[1]
    fold = case.model._clone_with_replaced_terms({})
    fold.fit_reml(case.X.iloc[train], case.y[train], retain_rows=False)
    edited = model_with_edited_curves(
        fold,
        plan.edited,
        case.X.iloc[train],
        case.y[train],
        None,
        None,
        n_points=session.n_points,
    )
    expected = edited.scores(case.X.iloc[test], case.y[test], which=("log",))["log"].mean()
    assert run.result.fold_scores.loc[1, "nll"] == pytest.approx(expected, rel=1e-9)
    # The edit moved the score: the unedited fold scores otherwise.
    assert abs(run.result.mean_scores["nll"] - cv.mean_scores["nll"]) > 1e-3


def test_an_ordered_term_is_folded_scored_and_drawn_like_any_level_term():
    case = lognormal_case()
    family = LogNormalLS()
    ordered = OrderedCategorical(order=list(REGIONS), basis=Spline(kind="cr", k=4))
    model = SuperLSS(
        family, family.mean(s("age", k=6)), family.scale(term("region", ordered))
    ).fit_reml(case.X, case.y, outer="efs+newton")
    cv = cross_validate(
        model,
        case.X,
        case.y,
        cv=FOLDS,
        fit_mode="fit_reml",
        return_estimators=True,
        error_score="raise",
    )
    session = edit(model, train_data=(case.X, case.y), cv=cv)
    train, _test = cv.fold_indices[0]
    effect = cv.estimators[0].term_inference(
        "scale", "region", X_train=case.X.iloc[train], simultaneous=False
    )
    curve = fold_log_curves(cv.estimators[0], session.terms)["scale:region"]
    # A fold curve holds the term's own centring; the chart re-centres every curve.
    assert np.ptp(np.asarray(curve) - effect.effect) < 1e-9
    widget = session.widget()
    try:
        items = {item["name"]: item for item in widget._report("cv")["relativities"]["terms"]}
    finally:
        widget.close()
    assert items["scale:region"]["kind"] == "levels"
    assert items["scale:region"]["levels"] == list(REGIONS)
    assert items["scale:region"]["group"] == "Spread"
    run = run_cv(capture_cv_run(session), _Context())
    assert run.result.mean_scores["nll"] == pytest.approx(cv.mean_scores["nll"], rel=1e-9)


def test_a_hand_edit_marks_the_supplied_folds_as_from_before_it(lognormal_cv):
    session = open_case(lognormal_case(), cv=lognormal_cv)
    session.select_indices("scale:age", [20, 21, 22])
    session.shift("scale:age", 0.2)
    widget = session.widget()
    try:
        assert widget._report("cv")["results"][0]["stale"]
    finally:
        widget.close()


def test_run_cv_waits_for_refit_and_refuses_other_rows(nb_cv):
    case = nb_case()
    session = open_case(case, cv=nb_cv)
    session.stage_structural("knots", "mean:age", KNOTS)
    reason = run_cv_reason(session)
    assert reason is not None and reason.startswith("Refit first")
    with pytest.raises(EditorValueError) as info:
        capture_cv_run(session)
    assert info.value.public_message == reason

    # The folds were drawn on rows with these offsets; a session whose rows lack them cannot replay.
    other = {"mean": case.offsets["mean"] + 0.5}
    moved = open_case(case, cv=nb_cv, cv_data=(case.X, case.y, None, other))
    assert moved.cv_check.reason == FINGERPRINT_MISMATCH.format(data="CV data", rows=len(case.y))


class _Recording(_Context):
    def __init__(self):
        self.entries = []

    def progress(self, phase, **details):
        self.entries.append((phase, details))


def test_run_cv_reports_each_fold_as_it_finishes(lognormal_cv):
    session = open_case(lognormal_case(), cv=lognormal_cv)
    context = _Recording()
    run = run_cv(capture_cv_run(session), context)
    done = [details for phase, details in context.entries if phase == "fold_done"]
    assert [details["fold"] for details in done] == [1, 2, 3]
    table = run.result.fold_scores
    for index, details in enumerate(done):
        assert details["n_folds"] == 3
        assert details["converged"] == bool(table.loc[index, "converged"])
        assert details["n_reml_iter"] == table.loc[index, "n_reml_iter"]
        assert details["effective_df"] == pytest.approx(table.loc[index, "effective_df"])
        train, test = lognormal_cv.fold_indices[index]
        assert (details["n_train"], details["n_test"]) == (len(train), len(test))
        assert details["fit_time_s"] > 0.0
        for name in SCORES:
            assert details["scores"][name] == pytest.approx(table.loc[index, name])
    # Each fold's report comes after it starts, and before the next one does.
    phases = [(phase, details.get("fold")) for phase, details in context.entries]
    assert phases[:4] == [("fold", 1), ("fold_done", 1), ("fold", 2), ("fold_done", 2)]


def test_the_tab_knows_whether_the_folds_split_the_rows(lognormal_cv):
    case = lognormal_case()
    payload = _payload(case, lognormal_cv)
    assert payload["results"][0]["held_out"] is True
    assert [fold["n_reml_iter"] for fold in payload["results"][0]["folds"]] == list(
        lognormal_cv.fold_scores["n_reml_iter"]
    )
    # Rows no fold holds out, or rows two folds hold out, are not a split of the book.
    train, test = lognormal_cv.fold_indices[0]
    lambdas = {"lambdas": dict(case.model.smoothing_parameters_)}
    for folds in ([(train, test), (train, test)], [(train, test)]):
        other = cross_validate(
            case.model, case.X, case.y, cv=_Stored(folds), fit_kwargs=lambdas, error_score="raise"
        )
        assert _payload(case, other)["results"][0]["held_out"] is False


def _payload(case, cv):
    widget = open_case(case, cv=cv).widget()
    try:
        return widget._report("cv")
    finally:
        widget.close()


class _Stored:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X=None, y=None, groups=None):
        yield from self.folds


def _finished(widget, kind: str) -> dict:
    job = widget._job_start(kind)["job_id"]
    status = widget._job_status(job, wait=True)
    while status["status"] == "running":
        status = widget._job_status(job, wait=True)
    return status


def test_a_count_family_has_no_crps_on_the_tab(nb_cv):
    assert "crps" not in nb_cv.fold_scores.columns
    widget = open_case(nb_case(), cv=nb_cv).widget()
    try:
        names = [metric["name"] for metric in widget._report("cv")["metrics"]]
    finally:
        widget.close()
    assert names == ["nll", "gini", "actual_expected"]


def test_the_widget_runs_cv_and_final_fit_as_background_jobs(nb_cv):
    widget = open_case(nb_case(), cv=nb_cv).widget()
    try:
        assert _finished(widget, "cv")["status"] == "done"
        report = widget._report("cv")
        assert [result["label"] for result in report["results"]] == ["As supplied", "Current model"]
        assert report["relativities"]["origin"] == "run"
        status = widget._job_status(report["jobs"]["cv"]["job_id"])
        assert status["result"]["headline"]["label"] == "Log score"
        assert status["result"]["headline"]["value"] == pytest.approx(
            report["results"][1]["mean"]["nll"]
        )
        assert status["elapsed_s"] > 0.0
        assert widget._state()["jobs_running"] == []

        assert _finished(widget, "final_fit")["status"] == "done"
        assert widget._report("final")["final_fit"]["available"]
    finally:
        widget.close()


def test_offsets_are_taken_by_rows_whatever_their_shape():
    rows = np.array([2, 0])
    assert _take_offset(None, rows) is None
    assert _take_offset(np.array([1.0, 2.0, 3.0]), rows).tolist() == [3.0, 1.0]
    assert _take_offset({"mean": np.array([1.0, 2.0, 3.0])}, rows)["mean"].tolist() == [3.0, 1.0]


@pytest.mark.parametrize(
    ("family", "key", "value"),
    [
        (GammaLS(), "mean:age", "relativity"),
        (GammaLS(), "scale:age", "relativity of the coefficient of variation"),
        (LogNormalLS(), "scale:age", "relativity of σ"),
        (NegativeBinomialLS(), "theta:age", "relativity of overdispersion (1/θ)"),
        (None, "scale:age", "relativity of the spread"),
    ],
)
def test_a_fold_curve_says_what_it_is_a_relativity_of(family, key, value):
    assert fold_chart_display(key, family=family).value == value


def test_run_cv_reads_no_label_of_a_row_of_zero_weight():
    case = lognormal_case()
    X = case.X.copy()
    X["region"] = X["region"].astype(object)
    X.loc[X.index[0], "region"] = None
    weights = np.ones(len(case.y))
    weights[0] = 0.0
    family = LogNormalLS()
    model = SuperLSS(family, family.mean(s("age", k=6), cat("region")), family.scale())
    model.fit_reml(X, case.y, sample_weight=weights)
    cv = cross_validate(
        model, X, case.y, cv=FOLDS, sample_weight=weights, fit_mode="fit_reml", error_score="raise"
    )
    session = edit(model, train_data=(X, case.y, weights), cv=cv)
    run = run_cv(capture_cv_run(session), _Context())
    assert np.isfinite(run.result.fold_scores["nll"]).all()


def test_run_cv_keeps_a_folds_scores_when_actual_expected_is_undefined(monkeypatch, lognormal_cv):
    monkeypatch.setitem(_POOLED_PARTS, "actual_expected", lambda *args, **kwargs: (1.0, 0.0))
    session = open_case(lognormal_case(), cv=lognormal_cv)
    run = run_cv(capture_cv_run(session), _Context())
    assert run.result.fold_scores["actual_expected"].isna().all()
    assert np.isfinite(run.result.fold_scores["nll"]).all()


def test_run_cv_pools_a_ratio_of_negative_totals(monkeypatch, lognormal_cv):
    monkeypatch.setitem(_POOLED_PARTS, "actual_expected", lambda *args, **kwargs: (-1.0, -2.0))
    session = open_case(lognormal_case(), cv=lognormal_cv)
    run = run_cv(capture_cv_run(session), _Context())
    assert run.result.pooled_scores["actual_expected"] == pytest.approx(0.5)


def test_run_cv_names_a_level_outside_an_ordered_term():
    case = lognormal_case()
    family = LogNormalLS()
    ordered = OrderedCategorical(order=list(REGIONS), basis=Spline(kind="cr", k=4))
    model = SuperLSS(family, family.mean(term("region", ordered)), family.scale()).fit_reml(
        case.X, case.y, outer="efs+newton"
    )
    cv_rows = case.X.assign(region=case.X["region"].astype(object))
    cv_rows.loc[cv_rows.index[:40], "region"] = "elsewhere"
    cv = cross_validate(model, case.X, case.y, cv=FOLDS, fit_mode="fit_reml")
    session = edit(
        model,
        train_data=(case.X, case.y),
        cv=dataclasses.replace(cv, data_fingerprint=None, fingerprint_version=None),
        cv_data=(cv_rows, case.y),
    )
    with pytest.raises(EditorValueError, match="outside its ordered levels"):
        run_cv(capture_cv_run(session), _Context())
