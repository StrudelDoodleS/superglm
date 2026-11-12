"""The distribution panel's payload: rows, fan columns, link-scale predictors and the amount axis."""

from __future__ import annotations

import functools
import json
import math
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, GammaLS, LogNormalLS, SuperLSS, TweedieLSS, cat, term
from superglm.distributional.families import BoundedLogitLink, LomaxLSS
from superglm.editor import edit, notebook
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.panel import distribution_panel_payload, fast_path, link_spec
from superglm.editor.payloads import session_payload
from superglm.editor.server import create_editor_app
from superglm.links import IdentityLink, InverseLink, LogitLink
from tests._lss_editor_fixtures import (
    EPS,
    FitCase,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
    tweedie_case,
)
from tests.test_editor import editor_frame, editor_model  # noqa: F401

COUNTS_REASON = "Count distributions are not drawn in this panel."


def _shift(session, key, delta):
    """Shift every point of the term's curve."""
    session.select_indices(key, list(range(session.terms[key].size)))
    session.shift(key, delta)


def _edited(session):
    """The model in force, as the widget materializes it."""
    return lss_model_with_edits(session.model, session.terms)


def _panel(session, feature):
    return distribution_panel_payload(session, feature, edited=_edited(session))


def _frame(session, column, values):
    """The reference row repeated once per value, with ``column`` set to the values."""
    frame = pd.concat([session.model.reference_row()] * len(values), ignore_index=True)
    frame[column] = list(values)
    return frame


def _tolerance(session, eta):
    """Row-wise dot products differ between batch shapes by gamma_n times their size."""
    n_coefficients = session.model._fitted.layout.n_coefficients
    return 4 * n_coefficients * EPS * max(1.0, float(np.max(np.abs(eta))))


@functools.cache
def grades_case() -> FitCase:
    """Fifteen grades of unequal size; each row of the three smallest stands for twelve observations."""
    counts = np.array([80, 70, 60, 50, 45, 40, 36, 32, 28, 24, 20, 16, 12, 10, 8])
    rng = np.random.default_rng(20261114)
    grade = np.repeat(np.arange(len(counts)), counts)
    X = pd.DataFrame({"grade": np.array([f"G{index:02d}" for index in grade])})
    weight = np.where(grade >= 12, 12.0, 1.0)
    y = np.exp(7.0 + 0.04 * grade + 0.5 * rng.standard_normal(len(grade)))
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(cat("grade")),
        family.scale(cat("grade")),
        weight_semantics="frequency",
    ).fit_reml(X, y, sample_weight=weight, outer="efs+newton")
    return FitCase(model, X, y, weight, None)


def test_numeric_rows_are_eight_even_values_over_the_training_range():
    case = lognormal_case()
    session = open_case(case)
    panel = _panel(session, "age")
    low, high = case.X["age"].min(), case.X["age"].max()

    assert (panel["feature"], panel["family"], panel["kind"]) == (
        "age",
        "lognormal_mean",
        "numeric",
    )
    assert panel["empty_reason"] is None and panel["cap_note"] is None
    np.testing.assert_array_equal([row["x"] for row in panel["rows"]], np.linspace(low, high, 8))
    np.testing.assert_array_equal(
        [column["x"] for column in panel["columns"]], np.linspace(low, high, 96)
    )
    assert all(row["level"] is None for row in panel["rows"])
    assert all(column["level"] is None for column in panel["columns"])
    assert panel["rows"][0]["label"] == f"{low:.4g}" and panel["rows"][-1]["label"] == f"{high:.4g}"
    assert panel["term_keys"] == {"mean": "mean:age", "scale": "scale:age"}
    for which in ("fitted", "edited"):
        assert np.shape(panel["eta"][which]["rows"]) == (8, 2)
        assert np.shape(panel["eta"][which]["columns"]) == (96, 2)
    assert panel["amount_unit"] is None


def test_a_feature_in_one_parameter_lists_only_that_term():
    panel = _panel(open_case(lognormal_case()), "power")
    assert panel["term_keys"] == {"mean": "mean:power"}
    assert np.shape(panel["eta"]["fitted"]["rows"]) == (8, 2)


def test_parameters_carry_their_labels_roles_and_links():
    lognormal = _panel(open_case(lognormal_case()), "age")["parameters"]
    assert lognormal == [
        {
            "name": "mean",
            "label": "Mean",
            "role": "average",
            "link": {"kind": "log"},
            "display": {"kind": "relativity", "link": "log"},
            "value_label": "Mean",
        },
        {
            "name": "scale",
            "label": "CV",
            "role": "spread",
            "link": {"kind": "floored_log", "floor": LogNormalLS().scale_floor},
            "display": {"kind": "relativity", "link": "floored_log"},
            "value_label": "σ",
        },
    ]
    gamma = _panel(open_case(gamma_case()), "age")
    assert gamma["family"] == "gamma"
    assert [parameter["link"] for parameter in gamma["parameters"]] == [{"kind": "log"}] * 2


def test_a_tweedie_panel_names_three_parameters_and_draws_no_power_curve():
    case = tweedie_case()
    panel = _panel(open_case(case), "age")
    low, high = case.model.family.power_lower, case.model.family.power_upper
    assert panel["family"] == "tweedie"
    assert panel["parameters"] == [
        {
            "name": "mean",
            "label": "Mean",
            "role": "average",
            "link": {"kind": "log"},
            "display": {"kind": "relativity", "link": "log"},
            "value_label": "Mean",
        },
        {
            "name": "dispersion",
            "label": "Dispersion",
            "role": "spread",
            "link": {"kind": "log"},
            "display": {"kind": "relativity", "link": "log"},
            "value_label": "Dispersion",
        },
        {
            "name": "power",
            "label": "Power",
            "role": "shape",
            "link": {"kind": "bounded_logit", "lower": low, "upper": high},
            "display": {"kind": "additive", "link": "bounded_power"},
            "value_label": "Power",
        },
    ]
    assert panel["term_keys"] == {"mean": "mean:age", "dispersion": "dispersion:age"}
    assert np.shape(panel["eta"]["fitted"]["rows"]) == (8, 3)
    # The link the browser reads is the model's own: lower + (upper - lower) / (1 + exp(-eta)).
    eta = np.asarray(panel["eta"]["fitted"]["columns"])
    natural = case.model.predict_parameters(
        _frame(open_case(case), "age", [c["x"] for c in panel["columns"]])
    )
    np.testing.assert_allclose(
        low + (high - low) / (1.0 + np.exp(-eta[:, 2])), natural["power"], rtol=0, atol=8 * EPS
    )


def test_every_link_has_a_spec_and_one_the_browser_cannot_invert_is_other():
    assert link_spec(IdentityLink()) == {"kind": "identity"}
    assert link_spec(LogitLink()) == {"kind": "logit"}
    assert link_spec(BoundedLogitLink(lower=-0.9, upper=0.9)) == {
        "kind": "bounded_logit",
        "lower": -0.9,
        "upper": 0.9,
    }
    assert link_spec(InverseLink()) == {"kind": "other"}


def test_an_overdispersion_is_walked_as_its_reciprocal():
    theta = _panel(open_case(nb_case()), "age")["parameters"][1]
    assert theta["display"] == {"kind": "inverse_theta", "link": "log"}
    assert theta["value_label"] == "Overdispersion"


@pytest.mark.parametrize(
    ("family", "drawn"),
    [
        (LogNormalLS(), "lognormal_mean"),
        (LogNormalLS(parametrisation="location"), None),
        (GammaLS(), "gamma"),
        (TweedieLSS(), "tweedie"),
        (LomaxLSS(), None),
    ],
    ids=lambda value: value if isinstance(value, str | None) else type(value).__name__,
)
def test_the_fast_path_table_names_only_the_browser_s_families(family, drawn):
    assert fast_path(family) == drawn


def test_levels_are_capped_at_twelve_by_exposure():
    case = grades_case()
    session = open_case(case)
    panel = _panel(session, "grade")

    exposure = pd.Series(case.sample_weight).groupby(case.X["grade"].to_numpy()).sum()
    kept = sorted(exposure.sort_values(ascending=False, kind="stable").index[:12])
    by_count = sorted(case.X["grade"].value_counts().index[:12])
    assert kept != by_count, "the fixture must tell exposure from row count"

    assert panel["kind"] == "levels"
    assert panel["cap_note"] == "Showing the 12 levels with the most exposure of 15."
    assert [row["level"] for row in panel["rows"]] == kept
    assert [row["label"] for row in panel["rows"]] == kept
    assert all(row["x"] is None for row in panel["rows"])
    assert panel["columns"] == [{"x": None, "level": level} for level in kept]
    assert np.shape(panel["eta"]["fitted"]["rows"]) == (12, 2)
    assert panel["eta"]["fitted"]["columns"] == panel["eta"]["fitted"]["rows"]
    assert panel["eta"]["edited"]["columns"] == panel["eta"]["edited"]["rows"]
    expected = case.model.predict_link(_frame(session, "grade", kept)).to_numpy()
    np.testing.assert_allclose(
        panel["eta"]["fitted"]["rows"], expected, rtol=0, atol=_tolerance(session, expected)
    )


def test_levels_within_the_cap_come_without_a_note():
    case = lognormal_case()
    panel = _panel(open_case(case), "region")
    assert panel["kind"] == "levels" and panel["cap_note"] is None
    assert [row["level"] for row in panel["rows"]] == sorted(case.X["region"].unique())


@pytest.mark.parametrize(
    ("case", "feature"),
    [
        (lognormal_case, "age"),
        (gamma_case, "age"),
        (lognormal_case, "region"),
        (tweedie_case, "age"),
        (tweedie_case, "region"),
    ],
)
def test_fitted_eta_matches_predict_link_at_the_reference_row(case, feature):
    fit = case()
    session = open_case(fit)
    panel = _panel(session, feature)
    rows = [row["x"] if row["level"] is None else row["level"] for row in panel["rows"]]
    columns = [col["x"] if col["level"] is None else col["level"] for col in panel["columns"]]

    for name, values in (("rows", rows), ("columns", columns)):
        expected = fit.model.predict_link(_frame(session, feature, values)).to_numpy()
        np.testing.assert_allclose(
            panel["eta"]["fitted"][name], expected, rtol=0, atol=_tolerance(session, expected)
        )
    reference = session.model.reference_row().iloc[0]
    assert {entry["column"] for entry in panel["reference"]} == set(reference.index) - {feature}
    for entry in panel["reference"]:
        assert entry["value"] == reference[entry["column"]]


def test_unedited_panel_has_the_same_predictors_fitted_and_edited():
    panel = _panel(open_case(lognormal_case()), "age")
    assert panel["eta"]["edited"] == panel["eta"]["fitted"]


def test_edited_eta_follows_a_spread_edit():
    case = lognormal_case()
    session = open_case(case)
    _shift(session, "scale:age", 0.3)
    panel = _panel(session, "age")

    for name in ("rows", "columns"):
        fitted = np.asarray(panel["eta"]["fitted"][name])
        edited = np.asarray(panel["eta"]["edited"][name])
        np.testing.assert_array_equal(edited[:, 0], fitted[:, 0])
        assert np.all(np.isfinite(edited))
    moved = np.asarray(panel["eta"]["edited"]["columns"])[:, 1]
    fitted_scale = np.asarray(panel["eta"]["fitted"]["columns"])[:, 1]
    assert np.max(np.abs(moved - fitted_scale)) > 0.1

    frame = _frame(session, "age", [column["x"] for column in panel["columns"]])
    expected = _edited(session).predict_link(frame).to_numpy()
    np.testing.assert_allclose(
        panel["eta"]["edited"]["columns"], expected, rtol=0, atol=_tolerance(session, expected)
    )


def test_amount_axis_spans_whole_decades_around_the_quantiles():
    case = lognormal_case()
    session = open_case(case)
    _shift(session, "scale:age", 1.0)
    panel = _panel(session, "age")

    frame = _frame(session, "age", [row["x"] for row in panel["rows"]])
    fitted, edited = case.model, _edited(session)
    low = [model.predict_quantile(frame, 0.001) for model in (fitted, edited)]
    high = [model.predict_quantile(frame, 0.999) for model in (fitted, edited)]
    q_low, q_high = float(np.min(low)), float(np.max(high))
    axis = panel["amount_axis"]

    assert isinstance(axis["min_log10"], int) and isinstance(axis["max_log10"], int)
    assert axis["min_log10"] == math.floor(math.log10(q_low))
    assert axis["max_log10"] == math.ceil(math.log10(q_high))
    assert 10.0 ** axis["min_log10"] <= q_low < 10.0 ** (axis["min_log10"] + 1)
    assert 10.0 ** (axis["max_log10"] - 1) < q_high <= 10.0 ** axis["max_log10"]
    assert axis["clipped_below"] is False
    # The edit widens the spread past the fitted model's decades, so the axis must cover it.
    assert axis["max_log10"] > math.ceil(math.log10(float(np.max(high[0]))))


def _quantile_extremes(session, panel, models):
    """The lowest 0.001 and highest 0.999 quantile of ``models`` over the panel's rows."""
    frame = _frame(session, "age", [row["x"] for row in panel["rows"]])
    low = min(float(np.min(model.predict_quantile(frame, 0.001))) for model in models)
    high = max(float(np.max(model.predict_quantile(frame, 0.999))) for model in models)
    return low, high


def test_amount_axis_of_an_ordinary_gamma_is_not_clipped():
    case = gamma_case()
    session = open_case(case)
    panel = _panel(session, "age")
    q_low, q_high = _quantile_extremes(session, panel, [case.model])
    assert panel["amount_axis"] == {
        "min_log10": math.floor(math.log10(q_low)),
        "max_log10": math.ceil(math.log10(q_high)),
        "clipped_below": False,
    }


def test_a_wide_spread_caps_the_axis_at_ten_decades_and_says_so():
    case = gamma_case()
    session = open_case(case)
    _shift(session, "scale:age", 1.5)
    panel = _panel(session, "age")
    q_low, q_high = _quantile_extremes(session, panel, [case.model, _edited(session)])
    top = math.ceil(math.log10(q_high))
    assert q_low > 0.0 and top - math.floor(math.log10(q_low)) > 10, "the edit must widen past ten"
    assert panel["amount_axis"] == {"min_log10": top - 10, "max_log10": top, "clipped_below": True}


def test_an_underflowing_lower_quantile_gives_a_capped_axis():
    case = gamma_case()
    session = open_case(case)
    _shift(session, "scale:age", 3.0)
    panel = _panel(session, "age")
    q_low, q_high = _quantile_extremes(session, panel, [case.model, _edited(session)])
    assert q_low == 0.0, "the edit must push a 0.001 quantile below the smallest float64"
    top = math.ceil(math.log10(q_high))
    assert panel["amount_axis"] == {"min_log10": top - 10, "max_log10": top, "clipped_below": True}


def _tweedie_zero_mass(model, frame):
    """P(Y = 0) per row from the compound Poisson-gamma rate mu^(2-p) / (phi (2-p))."""
    natural = model.predict_parameters(frame)
    power = natural["power"].to_numpy()
    rate = natural["mean"].to_numpy() ** (2.0 - power) / (
        natural["dispersion"].to_numpy() * (2.0 - power)
    )
    return np.exp(-rate)


def _tweedie_axis(session, panel, models):
    """The axis the rule gives: whole decades from the positive part's 0.001 quantile up to the
    law's 0.999 quantile (the positive part's, where the law's zero mass reaches it)."""
    frame = _frame(session, "age", [row["x"] for row in panel["rows"]])
    low, high = [], []
    for model in models:
        zero = _tweedie_zero_mass(model, frame)
        low.append(model.predict_quantile(frame, zero + 0.001 * (1.0 - zero)))
        level = np.where(zero < 0.999, 0.999, zero + 0.999 * (1.0 - zero))
        high.append(model.predict_quantile(frame, level))
    q_low, q_high = float(np.min(low)), float(np.max(high))
    assert q_low > 0.0
    return {
        "min_log10": math.floor(math.log10(q_low)),
        "max_log10": math.ceil(math.log10(q_high)),
        "clipped_below": False,
    }


def test_a_tweedie_amount_axis_starts_from_the_positive_part_of_the_law():
    case = tweedie_case()
    session = open_case(case)
    _shift(session, "dispersion:age", 1.0)
    panel = _panel(session, "age")
    edited = _edited(session)
    frame = _frame(session, "age", [row["x"] for row in panel["rows"]])
    assert np.all(_tweedie_zero_mass(edited, frame) > 0.5), "no claim must be the likelier outcome"
    assert float(np.max(edited.predict_quantile(frame, 0.001))) == 0.0
    assert panel["amount_axis"] == _tweedie_axis(session, panel, [case.model, edited])


def test_a_tweedie_risk_that_almost_never_claims_still_gets_a_positive_axis():
    case = tweedie_case()
    session = open_case(case)
    _shift(session, "dispersion:age", 8.0)
    panel = _panel(session, "age")
    edited = _edited(session)
    frame = _frame(session, "age", [row["x"] for row in panel["rows"]])
    assert float(np.min(_tweedie_zero_mass(edited, frame))) > 0.999
    assert panel["amount_axis"] == _tweedie_axis(session, panel, [case.model, edited])


def test_amount_unit_is_the_one_the_editor_was_opened_with():
    assert _panel(open_case(lognormal_case(), amount_unit="€"), "age")["amount_unit"] == "€"


def test_counts_use_one_year_and_are_empty_until_pr3():
    case = nb_case()
    session = open_case(case)
    panel = _panel(session, "age")

    assert panel["family"] == "negative_binomial"
    assert panel["kind"] == "empty" and panel["empty_reason"] == COUNTS_REASON
    assert panel["amount_axis"] is None
    assert [parameter["name"] for parameter in panel["parameters"]] == ["mean", "theta"]
    assert [parameter["label"] for parameter in panel["parameters"]] == [
        "Mean",
        "Overdispersion",
    ]
    assert panel["parameters"][0]["link"] == {"kind": "log"}

    xs = [row["x"] for row in panel["rows"]]
    frame = _frame(session, "age", xs)
    n = len(xs)
    one_year = case.model.predict_link(frame, offsets={"mean": np.zeros(n)}).to_numpy()
    half_year = case.model.predict_link(frame, offsets={"mean": np.full(n, np.log(0.5))})
    fitted = np.asarray(panel["eta"]["fitted"]["rows"])
    np.testing.assert_allclose(fitted, one_year, rtol=0, atol=_tolerance(session, one_year))
    assert np.max(np.abs(fitted[:, 0] - half_year.to_numpy()[:, 0])) > 0.5


@pytest.mark.parametrize("source", ["columns", "arrays"])
def test_panel_neutral_offsets_preserve_swept_feature_values(source):
    from types import SimpleNamespace

    from superglm.editor.lss.panel import _eta

    rng = np.random.default_rng(91)
    X = pd.DataFrame({"x": np.linspace(-1, 1, 80)})
    mean = np.exp(2 + 0.3 * X.x)
    scale = 0.4 * np.exp(X.x)
    y = rng.gamma(shape=1 / scale**2, scale=mean * scale**2)
    family = GammaLS()
    offset = "x" if source == "columns" else X.x.to_numpy()
    model = SuperLSS(family, family.mean("x", offset=offset), family.scale(offset=offset)).fit(X, y)
    view = SimpleNamespace(reference_row=lambda: pd.DataFrame({"x": [0.4]}))
    values = [-0.9, 0.2, 0.8]
    actual = _eta(model, view, "x", values)
    coef = model.coef_
    expected = np.column_stack(
        [
            coef["mean:(intercept)"] + coef["mean:x"] * np.array(values),
            np.full(len(values), coef["scale:(intercept)"]),
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def _count_link_predictions(monkeypatch):
    """Count actual link-design evaluations, including the neutral panel route."""
    from superglm.distributional.model import DenseDistributionalModel

    calls = []
    predict_link = DenseDistributionalModel._predict_eta_resolved

    def spy(self, coefficients, X, offsets):
        calls.append(len(X))
        return predict_link(self, coefficients, X, offsets)

    monkeypatch.setattr(DenseDistributionalModel, "_predict_eta_resolved", spy)
    return calls


def test_panel_makes_two_prediction_calls(monkeypatch):
    session = open_case(lognormal_case())
    _shift(session, "scale:age", 0.3)
    edited = _edited(session)
    calls = _count_link_predictions(monkeypatch)

    distribution_panel_payload(session, "age", edited=edited)
    assert calls == [8 + 96, 8 + 96]
    calls.clear()
    distribution_panel_payload(session, "region", edited=edited)
    assert calls == [7, 7]


def test_panel_predicts_once_when_nothing_is_edited(monkeypatch):
    session = open_case(lognormal_case())
    calls = _count_link_predictions(monkeypatch)
    distribution_panel_payload(session, "age", edited=session.model.lss)
    assert calls == [8 + 96]


def test_a_composite_key_names_its_feature():
    session = open_case(lognormal_case())
    assert _panel(session, "scale:age") == _panel(session, "age")


def test_an_unknown_feature_is_refused_with_the_features_to_choose_from():
    session = open_case(lognormal_case())
    with pytest.raises(EditorKeyError, match="age, power, region"):
        _panel(session, "colour")


def test_a_term_without_a_one_dimensional_sweep_is_empty_not_an_error():
    session = open_case(lognormal_case())
    for key in ("mean:age", "scale:age"):
        session.terms[key].x = None
    panel = _panel(session, "age")
    assert panel["kind"] == "empty" and panel["empty_reason"]
    assert panel["rows"] == [] and panel["columns"] == [] and panel["amount_axis"] is None
    assert panel["eta"] == {
        "fitted": {"rows": [], "columns": []},
        "edited": {"rows": [], "columns": []},
    }


def test_a_feature_read_as_numbers_and_as_levels_is_empty():
    session = open_case(lognormal_case())
    session.terms["scale:age"].x = None
    session.terms["scale:age"].levels = ["a", "b"]
    assert _panel(session, "age")["kind"] == "empty"


def test_a_superglm_session_has_no_distribution_panel(editor_model):  # noqa: F811
    from superglm.editor import EditorSession

    widget = EditorSession.from_model(editor_model).widget()
    try:
        with pytest.raises(EditorValueError, match="SuperLSS"):
            widget._distribution("x_spline")
    finally:
        widget.close()


def test_route_is_a_read_route():
    assert "/distribution" in notebook.READ_ROUTES
    assert "/distribution" not in notebook.CHANGE_ROUTES
    routes = {
        (route.path, frozenset(route.methods or ())) for route in create_editor_app(object()).routes
    }
    assert ("/distribution", frozenset({"POST"})) in routes


def _post(widget, body):
    request = urllib.request.Request(
        f"{widget.url}/distribution",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-SuperGLM-Editor-Token": widget._token},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_widget_serves_the_model_in_force():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        status, unedited = _post(widget, {"feature": "age"})
        assert status == 200 and unedited["eta"]["edited"] == unedited["eta"]["fitted"]

        _shift(session, "scale:age", 0.3)
        status, edited = _post(widget, {"feature": "age"})
        assert status == 200
        assert edited["eta"]["fitted"] == unedited["eta"]["fitted"]
        assert edited["eta"]["edited"] != edited["eta"]["fitted"]
        assert edited == json.loads(json.dumps(widget._distribution("age")))

        assert _post(widget, {}) == (400, {"error": "Missing required field: feature."})
        status, refusal = _post(widget, {"feature": "colour"})
        assert status == 400 and "colour" in refusal["error"]
    finally:
        widget.close()


def test_a_spread_curve_is_unmoved_by_an_edit_to_another_term_and_the_ridges_follow_it():
    # Shifting scale:age moves the spread at the reference row. The region curve is a relativity,
    # so it stays where it was; the region ridges show the edited model, and still differ
    # between levels by the curve's own factors.
    session = open_case(gamma_case())
    before = session_payload(session)["scale:region"]
    _shift(session, "scale:age", 0.3)
    panel = _panel(session, "region")
    curve = session_payload(session)["scale:region"]
    assert curve["y"] == before["y"]
    scale = panel["parameters"].index(next(p for p in panel["parameters"] if p["name"] == "scale"))
    edited = np.asarray(panel["eta"]["edited"]["rows"])[:, scale]
    fitted = np.asarray(panel["eta"]["fitted"]["rows"])[:, scale]
    shown = np.log([curve["y"][curve["levels"].index(row["level"])] for row in panel["rows"]])
    np.testing.assert_allclose(
        shown - shown[0], edited - edited[0], rtol=0, atol=_tolerance(session, edited)
    )
    np.testing.assert_allclose(edited - fitted, 0.3, rtol=0, atol=_tolerance(session, edited))


def test_a_declared_level_absent_from_the_rows_is_read_as_its_own_value():
    case = gamma_case()
    X = case.X.assign(grade=np.resize([1, 2], len(case.y)))
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(term("grade", Categorical(levels=[1, 2, 3]))),
        family.scale(),
    ).fit_reml(X, case.y)
    session = edit(model, train_data=(X, case.y))
    payload = _panel(session, "grade")
    row = [row["label"] for row in payload["rows"]].index("3")
    own_value = model.predict_link(X.iloc[:1].assign(grade=3)).to_numpy()[0]
    np.testing.assert_allclose(payload["eta"]["fitted"]["rows"][row], own_value, rtol=1e-12)
