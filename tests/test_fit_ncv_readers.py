"""What every reader of a fitted model says of a ``fit_ncv`` fit, in each state the fit can end in.

The readers are the summary, ``reml_diagnostics()``, both forms of the training telemetry, the
per-group ``diagnostics()``, the fit recorder, the editor's Diagnose tab and its refit, and the
readers of a model whose coefficients were edited after fitting.  Each is read in six states
(a converged search, one stopped at its step limit, one stopped on its line search, a search
that found no smoothing and was published at its start, a converged fit with
``retain_fit_state=False``, and a search that met its test on coefficient fits the solver
ended at the limit of float64); the summary also in two more (a final coefficient fit short
of its test after a converged search, and a smoothing parameter on a bound).  Each says which
method chose the smoothing, gives the search's verdict in ``fit_ncv``'s words and presents
nothing as a REML score.  What the same readers say of ``fit`` and ``fit_reml`` models is
pinned in ``tests/test_fit_method_table.py``, whose module collects without ``fit_ncv``.
"""

from __future__ import annotations

import copy
import gzip
import json
import warnings
from dataclasses import replace
from functools import cache

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, cross_validate
from superglm.diagnostics.convergence import ncv_bound_message, ncv_nonconvergence_message
from superglm.diagnostics.fit_report import status_sentence
from superglm.editor import EditorSession
from superglm.editor import free_levels as free_levels_module
from superglm.editor.cv import capture_cv_run, capture_final_fit, run_cv, run_final_fit
from superglm.editor.diagnose import fit_trace
from superglm.editor.refit import fit_refit_model
from superglm.features.rebuild import clone_with_replaced_features
from superglm.model import fit_method, ncv_ops
from superglm.model.fit_state import fitted_lambda2
from superglm.model.telemetry_ops import metrics_for_logging
from superglm.ncv import engine
from superglm.ncv import search as search_module
from superglm.types import LambdaPolicy
from tests.test_editor_cv import _Context
from tests.test_fit_ncv import curve_book, curve_fit, curve_model, run
from tests.test_ncv_engine import always, solver_short
from tests.test_ncv_search import NarrowBox

FIVE = ("converged", "step_limit", "line_search", "failed_search", "compact")
SIX = (*FIVE, "at_the_limit")
NOT_REML = (
    "The smoothing was chosen by fit_ncv, by held-out prediction, not by REML, so this fit "
    "has no REML score; model.ncv_diagnostics() reports its search."
)
# How each state's statement opens; None where the fit converged.
LEADS = {
    "converged": None,
    "step_limit": (
        "fit_ncv did not converge: its smoothing search stopped after 1 step because it "
        "reached the max_ncv_iter limit."
    ),
    "line_search": (
        "fit_ncv did not converge: its smoothing search stopped after {n} steps because no "
        "further step lowered the held-out deviance."
    ),
    "failed_search": (
        "fit_ncv chose no smoothing: its held-out deviance could not be computed at the 1 "
        "smoothing setting it tried."
    ),
    "compact": None,
    "at_the_limit": (
        "fit_ncv did not converge: its smoothing search met its test after {n} steps, but the "
        "coefficient fits at {tried} of the {tried} smoothing settings it tried, and the final "
        "coefficient fit, stopped at the limit of floating-point accuracy for this data, short "
        "of their convergence test (termination_reason='score_stagnated')."
    ),
    "final_fit_short": (
        "fit_ncv did not converge: its smoothing search met its test after {n} steps, but the "
        "final coefficient fit at that smoothing did not meet its convergence test"
    ),
    "bound": None,
}
# The Diagnose tab's "Stopped because ..." for each state.
STOP_WORDS = {
    "converged": (
        "the held-out deviance no longer falls in any direction of the smoothing, to within "
        "the set tolerance"
    ),
    "step_limit": "it reached its limit of 1 smoothing round (max_ncv_iter) before settling",
    "line_search": "no further step lowered the held-out deviance",
    "failed_search": "the held-out deviance could not be computed at any smoothing it tried",
}
STOP_WORDS["compact"] = STOP_WORDS["at_the_limit"] = STOP_WORDS["converged"]
# Each state's verdicts: the search's, and its coefficient fits'.
VERDICTS = {
    "converged": (True, True),
    "step_limit": (False, True),
    "line_search": (False, True),
    "failed_search": (False, False),
    "compact": (True, True),
    "at_the_limit": (True, False),
}
NCV_CONVERGED = (
    "Converged. fit_ncv chose the smoothing by held-out prediction, and its search met its "
    "stopping test."
)


def _stalled(find):
    """The search, made to report that its line search found no step from where it ended."""

    def stalled(problem, nb, **settings):
        outcome = find(problem, nb, **settings)
        result = replace(outcome.result, termination_reason="line_search", converged=False)
        return replace(outcome, result=result)

    return stalled


def _uncertified(solve):
    """The final coefficient fit, made to end at its iteration limit, uncertified."""

    def uncertified(**kwargs):
        result, inverse = solve(**kwargs)
        return replace(result, converged=False, termination_reason="max_iter"), inverse

    return uncertified


def fit_in(state: str):
    """A ``fit_ncv`` fit of the curve book that ends in ``state``."""
    family = "poisson" if state == "failed_search" else "gaussian"
    b = curve_book(family)
    settings = {"retain_fit_state": False} if state == "compact" else {}
    arguments = {"groups": b.labels}
    if state == "step_limit":
        arguments["max_ncv_iter"] = 1
    if state == "failed_search":
        arguments["max_pirls_iter"] = 1
    with pytest.MonkeyPatch.context() as patch:
        if state == "line_search":
            patch.setattr(ncv_ops, "search_smoothing", _stalled(ncv_ops.search_smoothing))
        if state == "final_fit_short":
            patch.setattr(ncv_ops, "fit_irls_direct", _uncertified(ncv_ops.fit_irls_direct))
        if state == "at_the_limit":
            # Every coefficient solve, of the search and the final one, ends stagnated.
            patch.setattr(engine, "fit_irls_direct", solver_short(always, []))
            patch.setattr(ncv_ops, "fit_irls_direct", solver_short(always, []))
        if state == "bound":
            box = NarrowBox(high=curve_fit().ncv_diagnostics()["lambdas"]["t"] / 100.0)
            patch.setattr(search_module, "_fit_box", lambda problem: box)
        model, _ = run(curve_model(family, **settings), b, **arguments)
    return model


fitted_in = cache(fit_in)


def expected_statement(state: str, model) -> str | None:
    """The state's statement in ``fit_ncv``'s words, checked against how it opens."""
    lead = LEADS[state]
    message = ncv_nonconvergence_message(model._reml_result)
    if lead is None:
        assert message is None
        return None
    report = model.ncv_diagnostics()
    assert message.startswith(lead.format(n=report["n_iter"], tried=report["n_evaluations"]))
    assert "REML" not in message and "fit_reml" not in message
    return message


# ── the summary ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", (*SIX, "final_fit_short", "bound"))
def test_summary_names_the_method_and_gives_the_verdict_in_its_own_words(state):
    model = fitted_in(state)
    report = model.ncv_diagnostics()
    summary = model.summary()
    info = summary._info
    assert info["method"] == "NCV"
    assert (info["converged"], info["n_iter"]) == (report["converged"], report["n_iter"])
    message = expected_statement(state, model)
    bound = ncv_bound_message(model._reml_result)
    if state == "bound":
        assert bound.startswith(
            "fit_ncv's smoothing search ended at a limit of its range on 't' at the largest "
            "smoothing parameter the search allows"
        )
    else:
        assert bound is None
    notes = [text for style, text in summary._notes() if style == "warning"]
    assert notes == [note for note in (message, bound) if note is not None]
    assert "REML" not in str(summary) and "fit_reml" not in " ".join(notes)


def test_summary_discloses_a_search_that_failed_and_a_final_fit_that_failed_apart():
    """The search verdict and the coefficient verdict are both stated, whichever failed."""
    outer = fitted_in("step_limit").ncv_diagnostics()
    inner = fitted_in("final_fit_short").ncv_diagnostics()
    assert (outer["search_converged"], outer["coefficients_converged"]) == (False, True)
    assert (inner["search_converged"], inner["coefficients_converged"]) == (True, False)
    outer_note = fitted_in("step_limit").summary()._info["convergence_note"]
    inner_note = fitted_in("final_fit_short").summary()._info["convergence_note"]
    assert "its smoothing search stopped" in outer_note and "final coefficient" not in outer_note
    assert "its smoothing search met its test" in inner_note
    assert "the final coefficient fit at that smoothing did not meet" in inner_note
    limit = fitted_in("at_the_limit").ncv_diagnostics()
    assert (limit["search_converged"], limit["coefficients_converged"]) == (True, False)
    assert limit["coefficient_termination"] == "score_stagnated" and not limit["converged"]
    assert limit["work"]["stagnated_fits"] == limit["n_evaluations"] and limit["heldout_available"]


# ── reml_diagnostics(), the telemetry and diagnostics() ───────────────────────


@pytest.mark.parametrize("state", SIX)
def test_reml_diagnostics_answers_that_the_criterion_was_not_reml(state):
    model = fitted_in(state)
    report = model.ncv_diagnostics()
    assert model.reml_diagnostics() == {
        "enabled": False,
        "criterion": "ncv",
        "lambdas": report["lambdas"],
        "lambda_history": [],
        "profile": {},
        "converged": report["converged"],
        "termination_reason": report["termination_reason"],
        "convergence_note": expected_statement(state, model),
        "terminal_refit_termination": report["coefficient_termination"],
        "note": NOT_REML,
    }


@pytest.mark.parametrize("state", SIX)
def test_telemetry_files_the_search_as_fit_ncv_s_in_both_forms(state):
    model = fitted_in(state)
    report = model.ncv_diagnostics()
    payload = model.training_telemetry()
    json.dumps(payload, allow_nan=False)
    assert payload["fit"]["method"] == "fit_ncv"
    assert payload["reml"] == model.reml_diagnostics()
    search = payload["ncv"]
    for key in ("method", "termination_reason", "n_iter", "n_evaluations", "converged"):
        assert search[key] == report[key]
    assert search["criterion"] == report["criterion"] and search["lambdas"] == report["lambdas"]
    assert (search["search_converged"], search["coefficients_converged"]) == VERDICTS[state]
    metrics = metrics_for_logging(model)
    assert not [key for key in metrics if ".reml." in key]
    want = {
        "train.ncv.n_iter": float(report["n_iter"]),
        "train.ncv.n_evaluations": float(report["n_evaluations"]),
        "train.ncv.converged": float(report["converged"]),
        "train.ncv.search_converged": float(report["search_converged"]),
        "train.ncv.coefficients_converged": float(report["coefficients_converged"]),
    }
    if report["criterion"] is not None:
        want["train.ncv.criterion"] = report["criterion"]
        want["train.ncv.criterion_mean"] = report["criterion_mean"]
    assert {key: value for key, value in metrics.items() if ".ncv." in key} == want


@pytest.mark.parametrize("state", SIX)
def test_diagnostics_label_the_search_by_its_method(state):
    model = fitted_in(state)
    report = model.ncv_diagnostics()
    entry = model.diagnostics()["_model"]
    assert entry["smoothing_search"] == "fit_ncv"
    assert (entry["n_iter"], entry["converged"], entry["termination_reason"]) == (
        report["n_iter"],
        report["converged"],
        report["termination_reason"],
    )
    assert entry["convergence_note"] == expected_statement(state, model)
    assert entry["smoothing_bound_note"] is None


# ── the fit recorder ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", SIX)
def test_the_fit_record_carries_the_search_as_fit_ncv_s(state, monkeypatch, tmp_path):
    monkeypatch.delenv("SUPERGLM_FIT_RECORD", raising=False)
    monkeypatch.setenv("BENCHBOARD_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("BENCHBOARD_FIT_DIR", str(tmp_path))
    monkeypatch.setenv("BENCHBOARD_TASK_ID", "ncv-readers")
    model = fit_in(state)
    report = model.ncv_diagnostics()
    records = [json.loads(gzip.decompress(p.read_bytes())) for p in tmp_path.glob("*.json.gz")]
    (fit,) = [fit for record in records for fit in record["fits"] if fit["method"] == "fit_ncv"]
    summary = fit["summary"]
    assert fit["status"] == "success"
    assert (summary["n_iter"], summary["converged"], summary["termination_reason"]) == (
        report["n_iter"],
        report["converged"],
        report["termination_reason"],
    )
    assert summary["objective"] == report["criterion"]
    assert summary["n_penalties"] == len(report["lambdas"])
    assert summary["work"] == report["work"] and summary["work"]["inner_solves"] > 0
    at_the_limit = report["n_evaluations"] if state == "at_the_limit" else 0
    assert summary["work"]["stagnated_fits"] == at_the_limit
    verdicts = summary["ncv"]["search_converged"], summary["ncv"]["coefficients_converged"]
    assert verdicts == VERDICTS[state]
    assert summary["ncv"] == {
        "search_converged": report["search_converged"],
        "coefficients_converged": report["coefficients_converged"],
        "coefficient_termination": report["coefficient_termination"],
        "coefficient_n_iter": int(model.result.n_iter),
        "n_evaluations": report["n_evaluations"],
        "infeasible_trials": report["infeasible_trials"],
        "criterion_mean": report["criterion_mean"],
        "neighbourhoods": json.loads(json.dumps(report["neighbourhoods"])),
        "heldout_available": report["heldout_available"],
    }


# ── the editor ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", SIX)
def test_the_diagnose_tab_names_fit_ncv_and_charts_no_reml_score(state):
    model = fitted_in(state)
    report = model.ncv_diagnostics()
    trace = fit_trace(model, fit_id="model", label="All rows")
    assert (trace["search"], trace["criterion"], trace["path"]) == (
        "Held-out prediction (fit_ncv)",
        "ncv",
        [],
    )
    assert (trace["late_gain"], trace["settled_round"]) == (None, None)
    assert (trace["rounds"], trace["stop"]) == (report["n_iter"], report["termination_reason"])
    assert trace["stop_text"] == STOP_WORDS[state]
    if report["converged"]:
        assert (trace["status"], trace["verdict"], trace["explanation"]) == (
            "converged",
            NCV_CONVERGED,
            None,
        )
    else:
        assert (trace["status"], trace["verdict"]) == (
            "not_converged",
            status_sentence("not_converged"),
        )
        assert trace["explanation"] == expected_statement(state, model)
    assert "REML" not in json.dumps(trace, default=str)


def test_an_auto_refit_from_the_editor_holds_the_chosen_smoothing_with_fit_and_says_so():
    model = fitted_in("converged")
    b = curve_book()
    refit = clone_with_replaced_features(model, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        method = fit_refit_model(model, refit, method="auto", X=b.frame, y=b.y)
    assert method == "fit" and refit._reml_result is None
    assert fitted_lambda2(refit) == model.ncv_diagnostics()["lambdas"]
    np.testing.assert_allclose(refit.predict(b.frame), model.predict(b.frame), rtol=1e-10)
    trace = fit_trace(refit, fit_id="model", label="All rows")
    assert (trace["status"], trace["verdict"]) == (
        "fixed",
        "No smoothing search. The smoothing is held where fit_ncv chose it.",
    )
    again = clone_with_replaced_features(refit, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_refit_model(refit, again, method="auto", X=b.frame, y=b.y)
    assert fit_trace(again, fit_id="model", label="All rows")["verdict"] == trace["verdict"]
    elsewhere = clone_with_replaced_features(model, {}, lambda2=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_refit_model(model, elsewhere, method="auto", X=b.frame, y=b.y)
    assert fit_trace(elsewhere, fit_id="model", label="All rows")["verdict"] == status_sentence(
        "fixed_fit"
    )


def test_an_editor_refit_of_drafts_holds_a_fixed_lambda_policy_at_the_chosen_smoothing():
    """fit_ncv's drafts refit is a fit at the source's smoothing, a fixed policy's value included.

    The reference is a ``fit`` of the same drafted terms without the policy at those values.  A
    fit_reml source's refit replays fit_reml, keeping the policy and searching the other term.
    """
    rng = np.random.default_rng(41)
    frame = pd.DataFrame({"x": rng.uniform(size=400), "z": rng.uniform(size=400)})
    y = np.sin(6 * frame.x.to_numpy()) + np.cos(4 * frame.z.to_numpy()) + rng.normal(0, 0.3, 400)
    refits = {}
    for method in ("fit_ncv", "fit_reml"):
        features = {
            "x": Spline(kind="ps", n_knots=8, lambda_policy=LambdaPolicy.fixed(3.0)),
            "z": Spline(kind="ps", n_knots=6),
        }
        model = SuperGLM(family="gaussian", features=features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(model, method)(frame, y)
            session = EditorSession.from_model(model, terms=["x", "z"], train_data=(frame, y))
            session.stage_structural("knots", "x", {"count": 6, "strategy": "quantile_rows"})
            session.stage_structural("knots", "z", {"count": 5, "strategy": "quantile_rows"})
            drafts = {term: copy.deepcopy(session.draft_spec(term)) for term in ("x", "z")}
            session.refit_pending()
        refits[method] = (model, session.model, drafts)

    source, refit, drafts = refits["fit_ncv"]
    chosen = source.ncv_diagnostics()["lambdas"]
    assert chosen["x:wiggle"] == 3.0
    assert refit._editor_step["method"] == "fit" and refit._reml_result is None
    assert fitted_lambda2(refit) == {"x": 3.0, "z": chosen["z"]}
    drafts["x"]._lambda_policy = None
    reference = SuperGLM(family="gaussian", features=drafts)
    reference.lambda2 = {"x": 3.0, "z": chosen["z"]}
    reference.fit(frame, y)
    np.testing.assert_allclose(refit.predict(frame), reference.predict(frame), rtol=1e-10)
    assert fit_trace(refit, fit_id="model", label="All rows")["verdict"] == (
        "No smoothing search. The smoothing is held where fit_ncv chose it."
    )

    source, refit, _ = refits["fit_reml"]
    assert refit._editor_step["method"] == "fit_reml" and refit._reml_result is not None
    assert fitted_lambda2(refit)["x:wiggle"] == 3.0
    assert refit._specs["x"]._lambda_policy == LambdaPolicy.fixed(3.0)


def test_an_editor_refit_at_the_analysts_smoothing_drops_a_lambda_policy():
    """A drafts refit of a fit_ncv model given ``lambda2`` is a ``fit`` at that smoothing, x's policy dropped.

    The reference is a ``fit`` of the same drafted terms without the policy at ``lambda2 = 1``.
    """
    rng = np.random.default_rng(41)
    frame = pd.DataFrame({"x": rng.uniform(size=400), "z": rng.uniform(size=400)})
    y = np.sin(6 * frame.x.to_numpy()) + np.cos(4 * frame.z.to_numpy()) + rng.normal(0, 0.3, 400)
    features = {
        "x": Spline(kind="ps", n_knots=8, lambda_policy=LambdaPolicy.fixed(3.0)),
        "z": Spline(kind="ps", n_knots=6),
    }
    model = SuperGLM(family="gaussian", features=features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_ncv(frame, y)
        session = EditorSession.from_model(model, terms=["x", "z"], train_data=(frame, y))
        session.stage_structural("knots", "x", {"count": 6, "strategy": "quantile_rows"})
        drafts = {term: copy.deepcopy(session.draft_spec(term)) for term in ("x", "z")}
        session.refit_pending(lambda2=1.0)
    refit = session.model
    assert refit._editor_step["method"] == "fit" and refit._specs["x"]._lambda_policy is None
    drafts["x"]._lambda_policy = None
    reference = SuperGLM(family="gaussian", features=drafts)
    reference.lambda2 = 1.0
    reference.fit(frame, y)
    np.testing.assert_allclose(refit.predict(frame), reference.predict(frame), rtol=1e-10)


def test_the_editors_run_cv_replays_a_fit_ncv_result_with_its_groups():
    """Run CV with no edit gives back a fit_ncv result's fold scores: each fold holds out its groups.

    The groups share a level, so holding out single rows instead chooses other smoothing.
    """
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(30), 3)
    frame = pd.DataFrame({"x": rng.uniform(size=90)})
    y = np.sin(6 * frame.x.to_numpy()) + rng.normal(0, 0.6, 30)[groups] + rng.normal(0, 0.3, 90)
    model = SuperGLM(family="gaussian", features={"x": Spline(kind="ps", n_knots=10)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        supplied = cross_validate(
            model, frame, y, cv=GroupKFold(3), groups=groups, fit_mode="fit_ncv"
        )
        model.fit_ncv(frame, y, groups=groups)
        session = EditorSession.from_model(model, train_data=(frame, y), cv=supplied)
        run = run_cv(capture_cv_run(session), _Context())
    np.testing.assert_array_equal(supplied.ncv_groups, groups)
    np.testing.assert_allclose(
        run.result.fold_scores["deviance"], supplied.fold_scores["deviance"], rtol=1e-12
    )


def test_the_editors_final_fit_of_a_fit_ncv_model_holds_the_chosen_smoothing():
    """Final fit on the rows fit_ncv chose its smoothing on, with no edit, gives back the same model.

    A ``fit`` refit holds the smoothing, so its fitted values are the source's to the solve's rounding.
    """
    rng = np.random.default_rng(41)
    frame = pd.DataFrame({"x": rng.uniform(size=400)})
    y = np.sin(6 * frame.x.to_numpy()) + rng.normal(0, 1, 400)
    model = SuperGLM(family="gaussian", features={"x": Spline(kind="ps", n_knots=12)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_ncv(frame, y)
        session = EditorSession.from_model(model, train_data=(frame, y))
        final = run_final_fit(capture_final_fit(session), _Context())
    chosen = model.ncv_diagnostics()["lambdas"]
    assert fitted_lambda2(final.model) == chosen
    np.testing.assert_allclose(final.model.predict(frame), model.predict(frame), rtol=1e-10)


def test_compare_with_free_levels_holds_a_fixed_lambda_policy_at_the_chosen_smoothing(monkeypatch):
    """fit_ncv's free-levels fit is a fit at the source's smoothing, another term's policy included.

    The reference is a ``fit`` of the spline, without its policy at its value, and the levels as a
    plain categorical; a Gaussian identity fit is one weighted least-squares solve, so both agree to
    its rounding.  A fit_reml source's free fit replays fit_reml and keeps the policy.
    """
    bands = [f"b{i:02d}" for i in range(8)]
    rng = np.random.default_rng(43)
    frame = pd.DataFrame({"x": rng.uniform(size=600), "band": rng.choice(bands, 600)})
    step = np.array([bands.index(band) for band in frame.band])
    y = np.sin(6 * frame.x.to_numpy()) + 0.1 * step + rng.normal(0, 0.3, 600)
    fitted = []
    real = free_levels_module.fit_refit_model
    monkeypatch.setattr(
        free_levels_module,
        "fit_refit_model",
        lambda source, refit, **kwargs: fitted.append(refit) or real(source, refit, **kwargs),
    )
    free = {}
    for method in ("fit_ncv", "fit_reml"):
        features = {
            "x": Spline(kind="ps", n_knots=8, lambda_policy=LambdaPolicy.fixed(3.0)),
            "band": OrderedCategorical(order=bands, basis=Spline(kind="ps", n_knots=4)),
        }
        model = SuperGLM(family="gaussian", features=features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(model, method)(frame, y)
            session = EditorSession.from_model(model, train_data=(frame, y))
            free[method] = free_levels_module.free_level_comparison(session, "band")
    ncv_fit, reml_fit = fitted

    reference = SuperGLM(
        family="gaussian",
        features={
            "x": Spline(kind="ps", n_knots=8),
            "band": Categorical(base=ncv_fit._specs["band"]._base_level),
        },
    )
    reference.lambda2 = {"x": 3.0}
    reference.fit(frame, y)
    inference = reference.term_inference("band")
    expected = dict(zip(inference.levels, inference.log_relativity, strict=True))
    scale = 1.0 + max(abs(value) for value in expected.values())
    assert fit_method.method_of(ncv_fit) == "fit" and fitted_lambda2(ncv_fit)["x"] == 3.0
    np.testing.assert_allclose(
        free["fit_ncv"]["y"], [expected[band] for band in bands], rtol=0, atol=1e-6 * scale
    )

    assert fit_method.method_of(reml_fit) == "fit_reml"
    assert fitted_lambda2(reml_fit)["x:wiggle"] == 3.0
    assert reml_fit._specs["x"]._lambda_policy == LambdaPolicy.fixed(3.0)


# ── after a coefficient edit ──────────────────────────────────────────────────


@cache
def edited():
    """The bound fit with its curve shifted by hand over part of its range in the editor."""
    session = EditorSession.from_model(copy.deepcopy(fitted_in("bound")), terms=["t"])
    session.select_x("t", 0.2, 0.5)
    session.shift("t", 0.35)
    return session.to_model()


def test_an_edit_withdraws_the_held_out_results_and_the_criterion():
    """The held-out results described the fitted coefficients, not the edited ones."""
    model = edited()
    report = model.ncv_diagnostics()
    assert report["termination_reason"] == "coefficients_revised" and not report["converged"]
    assert not (report["search_converged"] or report["coefficients_converged"])
    assert report["criterion"] is report["criterion_mean"] is report["gradient"] is None
    assert not report["heldout_available"] and report["bound_pressure"] == []
    assert report["at_bound"] == fitted_in("bound").ncv_diagnostics()["at_bound"]
    assert report["lambdas"] == fitted_in("bound").ncv_diagnostics()["lambdas"]
    for reader in (model.ncv_heldout, model.ncv_heldout_groups):
        with pytest.raises(RuntimeError, match="the coefficients were revised after fitting"):
            reader()
    assert ncv_bound_message(model._reml_result).endswith(
        "if that shape is expected, use that simpler term; otherwise check the term's data."
    )
    assert model.reml_diagnostics()["convergence_note"] is None
    assert model.reml_diagnostics()["criterion"] == "ncv"
    assert fit_trace(model, fit_id="model", label="All rows")["status"] == "edited"
    summary = model.summary()
    assert summary._info["method"] == "NCV" and "convergence_note" not in summary._info
    assert "still falling" not in summary._info["smoothing_bound_note"]
