"""Validation scores, Final fit and the capabilities of an edited SuperLSS."""

from __future__ import annotations

import dataclasses
import io

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from superglm import Categorical, GammaLS, LogNormalLS, SuperLSS, cat, collapse_levels, s, term
from superglm.editor import edit
from superglm.editor.cv import capture_cv_run, capture_final_fit, final_fit_reason, run_final_fit
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.availability import LSS_UNAVAILABLE, SESSION_TOOLS
from superglm.editor.lss.validation import LSS_METRIC_LABELS, lss_dataset_metrics
from superglm.editor.reports import final_fit_report_payload, validation_report_payload
from tests._lss_editor_fixtures import (
    EPS,
    FitCase,
    gamma_case,
    holdout_rows,
    lognormal_case,
    nb_case,
    open_case,
)
from tests.test_editor import editor_frame, editor_model  # noqa: F401


@pytest.fixture(scope="module")
def frequency_lognormal() -> FitCase:
    """A log-normal fitted under frequency weights, the contract that scores integer weights."""
    case = lognormal_case()
    rng = np.random.default_rng(7)
    weights = rng.integers(1, 3, len(case.y)).astype(np.float64)
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=8), cat("region")),
        family.scale(s("age", k=6)),
        weight_semantics="frequency",
    ).fit_reml(case.X, case.y, sample_weight=weights, outer="efs+newton")
    return FitCase(model, case.X, case.y, weights, None)


def _shift(session, key, delta, indices):
    session.select_indices(key, list(indices))
    session.shift(key, delta)


def _job(widget, kind):
    started = widget._job_start(kind)
    status = widget._job_status(started["job_id"], wait=True)
    assert status["status"] == "done", status
    return status


def test_validation_metrics_match_direct_computation(frequency_lognormal):
    case = frequency_lognormal
    split = holdout_rows(case, 200)
    split = dataclasses.replace(split, sample_weight=np.ceil(split.sample_weight))
    got = lss_dataset_metrics(case.model, split)

    par = case.model.predict_parameters(split.X)
    sigma = par["scale"].to_numpy()
    mu = np.log(par["mean"].to_numpy()) - sigma**2 / 2
    # The log score is a loss: the negative log-likelihood, lower for a better fit.
    nll = -stats.lognorm(s=sigma, scale=np.exp(mu)).logpdf(split.y)
    w = split.sample_weight
    assert got["log_score"] == pytest.approx(np.average(nll, weights=w), rel=1e3 * EPS)
    assert got["actual_expected"] == pytest.approx(
        np.sum(w * split.y) / np.sum(w * par["mean"]), rel=64 * EPS
    )
    # CRPS of a log-normal law by quadrature: the integral of (F(x) - 1{y <= x})^2.
    crps = _lognormal_crps(sigma[:5], mu[:5], split.y[:5])
    scores = case.model.scores(split.X.iloc[:5], split.y[:5], sample_weight=w[:5])
    np.testing.assert_allclose(scores["crps"], w[:5] * crps, rtol=1e-6)
    assert got["crps"] == pytest.approx(
        np.nansum(case.model.scores(split.X, split.y, sample_weight=w)["crps"]) / w.sum()
    )
    assert set(got) == set(LSS_METRIC_LABELS)


def _lognormal_crps(sigma, mu, y):
    """CRPS as the integral of F^2 below y plus (1 - F)^2 above it."""
    out = []
    for s_, m_, y_ in zip(sigma, mu, y, strict=True):
        law = stats.lognorm(s=s_, scale=np.exp(m_))
        below = integrate.quad(lambda x: law.cdf(x) ** 2, 0.0, y_, epsabs=0, epsrel=1e-10)[0]
        above = integrate.quad(lambda x: law.sf(x) ** 2, y_, np.inf, epsabs=0, epsrel=1e-10)[0]
        out.append(below + above)
    return np.array(out)


def test_a_zero_weight_row_is_neither_predicted_nor_scored(frequency_lognormal):
    case = frequency_lognormal
    split = holdout_rows(case, 120)
    split = dataclasses.replace(split, sample_weight=np.ceil(split.sample_weight))
    # One more row with weight zero, a missing covariate and a missing response.
    extra = split.X.iloc[[0]].assign(age=np.nan)
    padded = dataclasses.replace(
        split,
        X=pd.concat([split.X, extra], ignore_index=True),
        y=np.append(split.y, np.nan),
        sample_weight=np.append(split.sample_weight, 0.0),
    )
    got = lss_dataset_metrics(case.model, padded)
    want = lss_dataset_metrics(case.model, split)
    assert got == pytest.approx(want, rel=64 * EPS)


def test_prior_weighted_split_is_averaged_over_rows_not_weight():
    """A prior weight sharpens a row's own law (shape times w); rows count once each."""
    case = gamma_case()
    split = holdout_rows(case, 150)
    got = lss_dataset_metrics(case.model, split)

    par = case.model.predict_parameters(split.X)
    w = split.sample_weight
    cv = par["scale"].to_numpy() / np.sqrt(w)
    mean = par["mean"].to_numpy()
    law = stats.gamma(a=1.0 / cv**2, scale=mean * cv**2)
    # Each row's mean log score; the weights enter the law, not the average.
    assert got["log_score"] == pytest.approx(np.mean(-law.logpdf(split.y)), rel=1e3 * EPS)
    assert got["log_score"] != pytest.approx(np.average(-law.logpdf(split.y), weights=w), rel=1e-3)
    # The mean is unchanged by a prior weight, so A/E reads the weights as exposure.
    assert got["actual_expected"] == pytest.approx(
        np.sum(w * split.y) / np.sum(w * mean), rel=64 * EPS
    )


def test_a_family_without_a_distribution_function_has_no_crps():
    case = nb_case()
    split = dataclasses.replace(holdout_rows(case, 100), sample_weight=None)
    got = lss_dataset_metrics(case.model, split)
    assert np.isnan(got["crps"])
    assert np.isfinite(got["log_score"]) and np.isfinite(got["actual_expected"])


def test_spread_edit_moves_log_score_but_not_actual_expected():
    case = lognormal_case()
    split = dataclasses.replace(holdout_rows(case, 200), sample_weight=None)
    session = open_case(case, validation_data=(split.X, split.y))
    _shift(session, "scale:age", 0.4, range(10, 150))

    report = validation_report_payload(session)
    row = next(r for r in report["splits"] if r["name"] == "validation")
    original, edited = row["metrics"]["original"], row["metrics"]["edited"]
    assert list(report["metric_labels"]) == list(LSS_METRIC_LABELS)
    assert set(original) == set(LSS_METRIC_LABELS)
    assert abs(edited["log_score"] - original["log_score"]) > 1e-3
    # The mean form fixes the mean when only the spread moves.
    assert edited["actual_expected"] == pytest.approx(original["actual_expected"], rel=64 * EPS)
    assert row["metrics"]["delta"]["log_score"] == edited["log_score"] - original["log_score"]


def test_cv_run_is_refused_for_superlss():
    session = open_case(lognormal_case())
    with pytest.raises(EditorValueError) as info:
        capture_cv_run(session)
    assert info.value.public_message == LSS_UNAVAILABLE["cv"]
    widget = session.widget()
    try:
        with pytest.raises(EditorValueError) as info:
            widget._report("cv")
        assert info.value.public_message == LSS_UNAVAILABLE["cv"]
        assert widget._metrics("deviance")["error"] == LSS_UNAVAILABLE["metrics"]
        with pytest.raises(EditorValueError) as info:
            widget._job_start("cv")
        assert info.value.public_message == LSS_UNAVAILABLE["cv"]
        assert {"metrics", "cv"} <= set(SESSION_TOOLS)
        unavailable = widget._state()["unavailable"]
        assert unavailable["cv"] == LSS_UNAVAILABLE["cv"]
        assert unavailable["metrics"] == LSS_UNAVAILABLE["metrics"]
    finally:
        widget.close()


class _Context:
    def check(self):
        return None

    def progress(self, phase, **details):
        return None


@pytest.mark.filterwarnings("ignore::superglm.distributional.revision.EditedCoefficientsWarning")
def test_final_fit_refits_on_all_rows_and_carries_edits(monkeypatch):
    case = lognormal_case()
    split = dataclasses.replace(holdout_rows(case, 200), sample_weight=None)
    session = open_case(case, validation_data=(split.X, split.y))
    _shift(session, "scale:age", 0.3, range(20, 120))

    calls = []
    replay = SuperLSS._replay_fit

    def counted(self, X, y, sample_weight=None, offsets=None):
        calls.append((len(y), sample_weight, offsets))
        return replay(self, X, y, sample_weight, offsets)

    monkeypatch.setattr(SuperLSS, "_replay_fit", counted)
    final = run_final_fit(capture_final_fit(session), _Context())

    assert calls == [(800, None, None)]
    assert type(final.model) is SuperLSS
    assert (final.n_rows, final.splits, final.carried) == (
        800,
        ("train", "validation"),
        ("scale:age",),
    )
    stacked = _stack(case, split)
    plain = case.model._clone_with_replaced_terms({})
    replay(plain, stacked, np.concatenate([case.y, split.y]))
    kept = final.model.term_inference("scale", "age", X_train=stacked)
    refit = plain.term_inference("scale", "age", X_train=stacked)
    # The drawn bump survives the refit, as far as a six-column basis carries a box.
    drawn = session.terms["scale:age"].x
    gap = kept.effect - refit.effect
    inside = (kept.x > drawn[40]) & (kept.x < drawn[100])
    assert inside.sum() > 40 and gap[inside].min() > 0.0 and gap[inside].mean() > 0.1
    assert gap[inside].mean() > gap[kept.x > drawn[170]].mean() + 0.1


def _stack(case, split):
    import pandas as pd

    return pd.concat([case.X, split.X], ignore_index=True)


def test_final_fit_stacks_parameter_offsets(monkeypatch):
    case = nb_case()
    split = holdout_rows(case, 100)
    split = dataclasses.replace(split, sample_weight=None)
    session = open_case(case, validation_data=(split.X, split.y, None, split.offset))
    seen = []
    replay = SuperLSS._replay_fit

    def counted(self, X, y, sample_weight=None, offsets=None):
        seen.append(offsets)
        return replay(self, X, y, sample_weight, offsets)

    monkeypatch.setattr(SuperLSS, "_replay_fit", counted)
    final = run_final_fit(capture_final_fit(session), _Context())
    assert final.n_rows == 700
    (offsets,) = seen
    assert list(offsets) == ["mean"]
    np.testing.assert_array_equal(
        offsets["mean"], np.concatenate([case.offsets["mean"], split.offset["mean"]])
    )


def test_final_fit_refuses_a_split_missing_one_parameters_offsets():
    case = nb_case()
    split = dataclasses.replace(holdout_rows(case, 100), sample_weight=None)
    session = open_case(case, validation_data=(split.X, split.y, None, {}))
    reason = final_fit_reason(session)
    assert reason is not None and "needs mean offsets on both or on neither" in reason
    with pytest.raises(EditorValueError, match="needs mean offsets on both"):
        capture_final_fit(session)


def test_final_fit_places_a_validation_only_level_in_the_unseen_group():
    case = gamma_case()
    grouping = collapse_levels(case.X["region"], groups={"Other": ["R5", "R6"]})
    family = GammaLS()
    region = Categorical(grouping=grouping, unseen="Other")
    model = SuperLSS(
        family, family.mean(s("age", k=6), term("region", region)), family.scale()
    ).fit_reml(case.X, case.y)
    split = dataclasses.replace(holdout_rows(case, 120), sample_weight=None)
    X = split.X.copy()
    X.loc[X.index[:10], "region"] = "R9"
    session = edit(model, train_data=(case.X, case.y), validation_data=(X, split.y))
    assert final_fit_reason(session) is None
    final = run_final_fit(capture_final_fit(session), _Context())
    assert final.n_rows == len(case.y) + 120
    # R9 is predicted as the Other group, as the in-force model places it.
    probe = X.iloc[:1].assign(region="R5")
    np.testing.assert_allclose(
        final.model.predict(X.iloc[:1]), final.model.predict(probe), rtol=64 * EPS
    )


def test_final_fit_reads_only_the_columns_the_model_uses():
    case = lognormal_case()
    split = dataclasses.replace(holdout_rows(case, 150), sample_weight=None)
    # A column the model never reads, present in the training rows only.
    X = case.X.assign(row_id=np.arange(len(case.y)))
    session = edit(case.model, train_data=(X, case.y), validation_data=(split.X, split.y))
    final = run_final_fit(capture_final_fit(session), _Context())
    assert final.n_rows == len(case.y) + 150


@pytest.mark.filterwarnings("ignore::superglm.distributional.revision.EditedCoefficientsWarning")
def test_the_widget_runs_final_fit_and_exports_it():
    case = lognormal_case()
    split = dataclasses.replace(holdout_rows(case, 150), sample_weight=None)
    session = open_case(case, validation_data=(split.X, split.y))
    _shift(session, "scale:age", 0.3, range(20, 120))
    widget = session.widget()
    try:
        assert widget._state()["final_fit"] == {"available": False, "stale": False}
        _job(widget, "final_fit")
        report = widget._report("final")
        assert report["final_fit"]["available"] is True
        assert report["final_fit"]["n_rows"] == 750
        assert report["final_fit"]["carried"] == ["scale:age"]
        assert "summary" not in report
        assert list(report["metric_labels"]) == list(LSS_METRIC_LABELS)
        result = widget._export_bytes("final")
    finally:
        widget.close()

    loaded = joblib.load(io.BytesIO(result.data))
    assert type(loaded) is SuperLSS
    assert [record["predictor"] for record in loaded._editor_history] == ["scale"]
    assert loaded._editor_history[-1]["term"] == "scale:age"


def test_final_fit_report_before_a_run_has_no_summary():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        payload = final_fit_report_payload(widget)
    finally:
        widget.close()
    assert "summary" not in payload
    assert payload["final_fit"]["available"] is False


def test_capabilities_only_for_superlss(editor_model):  # noqa: F811
    widget = open_case(lognormal_case()).widget()
    try:
        assert widget._state()["capabilities"] == {"cv": False, "final_fit": True}
    finally:
        widget.close()
    from superglm.editor import edit

    glm = edit(editor_model).widget()
    try:
        assert "capabilities" not in glm._state()
    finally:
        glm.close()


def test_state_carries_the_final_fit_plan_for_superlss_only(editor_model):  # noqa: F811
    # Available: the fit rows and a validation split with the same columns.
    case = nb_case()
    split = holdout_rows(case, 100)
    ready = open_case(case, validation_data=(split.X, split.y, None, split.offset)).widget()
    # Unavailable: the validation split lacks the offsets the train rows hold.
    lacking = open_case(case, validation_data=(split.X, split.y, None, None))
    blocked = lacking.widget()
    try:
        assert final_fit_reason(ready.session) is None
        assert ready._state()["final_fit_plan"] == {"available": True, "reason": None}
        reason = final_fit_reason(lacking)
        assert reason is not None
        assert blocked._state()["final_fit_plan"] == {"available": False, "reason": reason}
    finally:
        ready.close()
        blocked.close()
    from superglm.editor import edit

    glm = edit(editor_model).widget()
    try:
        assert "final_fit_plan" not in glm._state()
    finally:
        glm.close()


@pytest.mark.parametrize("report", ["validation", "final"])
def test_a_split_with_responses_outside_the_family_is_shown_unavailable(report):
    case = gamma_case()
    split = dataclasses.replace(holdout_rows(case, 100), sample_weight=None)
    y = split.y.copy()
    y[:3] = 0.0
    session = open_case(case, validation_data=(split.X, y))
    widget = session.widget()
    try:
        rows = {row["name"]: row for row in widget._report(report)["splits"]}
        refused = rows["validation"]
        assert refused["available"] is False and "metrics" not in refused
        assert refused["error"] == (
            "The Validation split cannot be scored: its responses include values the model's "
            "family does not take. Remove those rows from the split to score it."
        )
        assert np.isfinite(rows["train"]["metrics"]["edited"]["log_score"])
    finally:
        widget.close()


@pytest.mark.parametrize("make", [lognormal_case, nb_case])
@pytest.mark.parametrize("report", ["validation", "final"])
def test_a_split_whose_weights_the_family_refuses_is_shown_unavailable(make, report):
    # Non-unit prior weights have no law under a non-reproductive family or NB2 counts.
    case = make()
    split = holdout_rows(case, 100)
    session = open_case(case, validation_data=(split.X, split.y, split.sample_weight, split.offset))
    widget = session.widget()
    try:
        payload = widget._report(report)
        rows = {row["name"]: row for row in payload["splits"]}
        refused = rows["validation"]
        assert refused["available"] is False and "metrics" not in refused
        assert refused["error"] == (
            "The Validation split cannot be scored with its weights: the model's family has no "
            "likelihood for weights like these. Pass the split without weights to score it."
        )
        assert "available" not in rows["train"]
        assert np.isfinite(rows["train"]["metrics"]["edited"]["log_score"])
        again = {row["name"]: row for row in widget._report(report)["splits"]}
        assert again["validation"] == refused
    finally:
        widget.close()
