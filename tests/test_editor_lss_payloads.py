"""SuperLSS terms in the editor payload: display units, conversions back, GLM unchanged."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import GammaLS, Numeric, OrderedCategorical, Spline, SuperLSS, term
from superglm.distributional.terms import _resolve_term, _term_design
from superglm.editor import EditorSession, edit
from superglm.editor.cv import fold_log_curves, fold_term_items
from superglm.editor.display import ADDITIVE
from superglm.editor.lss import INVERSE_THETA, RELATIVITY, display_for
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.display import fold_chart_display
from superglm.editor.lss.risk import risk_payload
from superglm.editor.operations import running_extreme_values
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import (
    EPS,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
    tweedie_case,
)
from tests.test_editor import (  # noqa: F401
    editor_frame,
    editor_log_link_frame,
    editor_log_link_model,
    editor_model,
)
from tests.test_lss_terms import mixed_band_fit, scale_ratio_to_reference

FIXTURE = Path(__file__).parent / "editor_frontend" / "fixtures" / "display_transforms.json"
TOLERANCE = 64 * EPS
# Every key of a SuperGLM term's payload; a SuperLSS key on it would change a SuperGLM page.
GLM_TERM_KEYS = {
    "kind", "term_type", "x", "x_domain", "y", "original_y", "previous_y", "controls",
    "spline_view", "ci_lower_y", "ci_upper_y", "weights", "exposure", "handle_indices", "impact",
    "n_points", "levels", "level_groups", "level_order_changed", "reference", "shape", "rules", "knots",
    "pending", "unseen", "unsmoothed", "effective_df", "edited", "x_label", "y_label", "title",
    "group_display",
}  # fmt: skip
LSS_FIELDS = {
    "parameter",
    "parameter_label",
    "parameter_role",
    "term_label",
    "display",
}


def _parameter_at(case, session, term, parameter):
    frame = pd.concat([session.model.reference_row()] * len(term["x"]), ignore_index=True)
    frame["age"] = term["x"]
    offsets = None if case.offsets is None else dict.fromkeys(case.offsets, np.zeros(len(frame)))
    return case.model.predict_parameters(frame, offsets=offsets)[parameter].to_numpy()


def _ratios(values):
    """Each value over the first: the relativity of a parameter between two points."""
    values = np.asarray(values, dtype=np.float64)
    return values / values[0]


def test_lognormal_scale_is_the_relativity_of_sigma_above_its_floor():
    case = lognormal_case()
    session = open_case(case)
    term = session_payload(session)["scale:age"]
    floor = case.model.family.scale_floor
    sigma = _parameter_at(case, session, term, "scale")
    np.testing.assert_allclose(_ratios(term["y"]), _ratios(sigma - floor), rtol=64 * TOLERANCE)
    np.testing.assert_array_equal(term["y"], np.exp(session.terms["scale:age"].edited_log_effect))
    assert term["y_label"] == "relativity of σ"
    assert term["display"] == {"kind": "relativity", "link": "floored_log"}


def test_gamma_scale_is_the_relativity_of_its_cv():
    case = gamma_case()
    session = open_case(case)
    term = session_payload(session)["scale:age"]
    scale = _parameter_at(case, session, term, "scale")
    np.testing.assert_allclose(_ratios(term["y"]), _ratios(scale), rtol=64 * TOLERANCE)
    np.testing.assert_array_equal(term["y"], np.exp(session.terms["scale:age"].edited_log_effect))
    assert term["y_label"] == "relativity of the CV"


def test_tweedie_dispersion_is_shown_as_relativities():
    case = tweedie_case()
    session = open_case(case)
    payload = session_payload(session)
    term = payload["dispersion:age"]
    # φ carries the amount's units, so the curve is its relativity.
    phi = _parameter_at(case, session, term, "dispersion")
    np.testing.assert_allclose(_ratios(term["y"]), _ratios(phi), rtol=64 * TOLERANCE)
    assert term["y_label"] == "relativity of the dispersion"
    assert (term["parameter"], term["parameter_label"], term["parameter_role"]) == (
        "dispersion",
        "Dispersion",
        "spread",
    )
    assert term["display"]["kind"] == "relativity"
    # Power has no curves: it is never a term of the payload.
    assert not any(key.startswith("power:") for key in payload if isinstance(key, str))


def _far_from_origin_fit():
    """A CV slope on a covariate near 1000: ``beta x`` is about 710, past where ``exp`` overflows."""
    rng = np.random.default_rng(578)
    t = np.linspace(0.0, 1.0, 360)
    X = pd.DataFrame({"x": 1000.0 + t})
    y = rng.gamma(np.exp(2.0 - 2.0 * t), np.exp(1.0) * np.exp(-2.0 + 2.0 * t))
    family = GammaLS()
    model = SuperLSS(family, family.mean(), family.scale(term("x", Numeric()))).fit_reml(X, y)
    return model, X, y


def test_a_spread_band_far_from_the_link_origin_brackets_its_curve():
    model, X, y = _far_from_origin_fit()
    term_payload = session_payload(edit(model, train_data=(X, y)))["scale:x"]
    curve = np.asarray(term_payload["y"], dtype=np.float64)
    lower = np.asarray(term_payload["ci_lower_y"], dtype=np.float64)
    upper = np.asarray(term_payload["ci_upper_y"], dtype=np.float64)
    # Read against the training median, the curve is a relativity of 1 there.
    x = np.asarray(term_payload["x"], dtype=np.float64)
    slope = model.coef_["scale:x"]
    np.testing.assert_allclose(curve, np.exp(slope * (x - np.median(X["x"]))), rtol=TOLERANCE)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= curve) and np.all(curve <= upper)
    assert np.all(upper > lower)


def test_a_linear_term_reads_the_same_on_the_editor_cv_and_explain_tabs():
    model, X, y = _far_from_origin_fit()
    session = edit(model, train_data=(X, y))
    shown = session_payload(session)["scale:x"]
    curve = np.asarray(shown["y"], dtype=np.float64)
    assert np.all(np.isfinite(curve))

    # The Cross-validation tab draws the in-force curve, and a fold sits at its level.
    curves = {0: fold_log_curves(model, session.terms)}
    (item,) = fold_term_items(
        {"scale:x": session.terms["scale:x"]},
        curves,
        display=partial(fold_chart_display, model=model),
    )
    np.testing.assert_allclose(item["fit"], curve, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(item["folds"][0]["values"], curve, rtol=1e-9, atol=0.0)

    # The Explain tab's walk moves x alone from the reference risk, the training median.
    picked = [0, 120, 199]
    reply = risk_payload(session, [{"x": float(shown["x"][i])} for i in picked], edited=model)
    (feature,) = [f for f in reply["features"] if f["column"] == "x"]
    assert feature["reference"] == np.median(X["x"])
    column = [f["column"] for f in reply["features"]].index("x")
    parameter = [p["name"] for p in reply["parameters"]].index("scale")
    moves = np.array([risk["effects"]["fitted"][column][parameter] for risk in reply["risks"]])
    # Each move is a difference of two linear predictors built from terms near 710.
    np.testing.assert_allclose(np.log(curve[picked]), moves, rtol=0.0, atol=1e3 * 710 * EPS)


def test_the_editor_opens_on_a_column_linear_in_one_parameter_and_levels_in_another():
    model, X, y = mixed_band_fit()
    session = edit(model, train_data=(X, y))
    payload = session_payload(session)["scale:band"]
    x = np.asarray(payload["x"], dtype=np.float64)
    curve = np.asarray(payload["y"], dtype=np.float64)
    assert np.all(np.isfinite(curve))
    ends = [0, len(x) - 1]
    np.testing.assert_array_equal(x[ends], [1.0, 4.0])
    np.testing.assert_allclose(
        curve[ends], scale_ratio_to_reference(model, X, x[ends]), rtol=1e-12, atol=0.0
    )
    # The Explain tab walks the same column from the same reference.
    reply = risk_payload(session, [{"band": 1, "z": 0.5}], edited=model)
    assert {feature["column"] for feature in reply["features"]} == {"band", "z"}


def test_a_linear_term_reads_its_own_column_alone_for_its_anchor(monkeypatch):
    # Mutation check: building the anchor from a whole sweep and a two-row design, as
    # 9b1ed58eb did, reads every other column again and builds two more designs per
    # linear term: 14 reads, 9 designs and 5 sweeps to open, 16, 10 and 6 for the tables.
    from superglm import cat
    from superglm.distributional import terms as terms_module

    rng = np.random.default_rng(664)
    X = pd.DataFrame(
        {
            "a": rng.uniform(18.0, 90.0, 300),
            "b": rng.uniform(0.0, 1.0, 300),
            "c": rng.choice(list("ABCD"), 300),
        }
    )
    y = rng.gamma(10.0, np.exp(0.01 * X["a"] + 0.2 * X["b"]) / 10.0)
    family = GammaLS()
    model = SuperLSS(
        family,
        family.mean(term("a", Numeric()), cat("c")),
        family.scale(term("b", Numeric())),
    ).fit_reml(X, y)

    counts = dict.fromkeys(("_reference_value", "_term_design", "_sweep"), 0)
    for name in counts:
        original = getattr(terms_module, name)

        def counting(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(terms_module, name, counting)
    monkeypatch.setattr("superglm.editor.lss.view._reference_value", terms_module._reference_value)

    def measured(action):
        counts.update(dict.fromkeys(counts, 0))
        action()
        return dict(counts)

    # One sweep and one design per term, each holding the two other columns, as at
    # 0.43.0; each linear term then reads its own column once for its curve's anchor
    # and once for the anchor the editor carries through a Refit.
    opened = measured(lambda: session_payload(edit(model, train_data=(X, y))))
    assert opened == {"_reference_value": 3 * 2 + 2 * 2, "_term_design": 3, "_sweep": 3}
    # The tables add the categorical term's base row, and read each anchor once.
    tables = measured(lambda: model.rating_table_payload(X))
    assert tables == {"_reference_value": 4 * 2 + 2, "_term_design": 4, "_sweep": 4}


def test_nb_theta_is_the_relativity_of_the_overdispersion():
    case = nb_case()
    session = open_case(case)
    term = session_payload(session)["theta:age"]
    theta = _parameter_at(case, session, term, "theta")
    np.testing.assert_allclose(_ratios(term["y"]), _ratios(1.0 / theta), rtol=64 * TOLERANCE)
    np.testing.assert_array_equal(term["y"], np.exp(-session.terms["theta:age"].edited_log_effect))
    assert term["y_label"] == "relativity of the overdispersion"
    # 1/theta falls as the effect rises, so the interval's ends are the other way round.
    assert np.all(np.asarray(term["ci_lower_y"]) <= np.asarray(term["y"]) + TOLERANCE)
    assert np.all(np.asarray(term["ci_upper_y"]) >= np.asarray(term["y"]) - TOLERANCE)


@pytest.mark.parametrize("case", [lognormal_case, gamma_case, nb_case, tweedie_case])
def test_mean_terms_stay_relativities(case):
    session = open_case(case())
    session.select_indices("mean:age", [4, 5])
    session.shift("mean:age", 0.2)
    payload = session_payload(session)
    term = payload["mean:age"]
    np.testing.assert_array_equal(term["y"], np.exp(session.terms["mean:age"].edited_log_effect))
    np.testing.assert_array_equal(
        term["original_y"], np.exp(session.terms["mean:age"].original_log_effect)
    )
    assert term["y_label"] == "relativity" and "y_format" not in term and "baseline" not in term
    assert term["parameter"] == "mean"


def test_lss_terms_carry_their_fields():
    case = lognormal_case()
    term = session_payload(open_case(case))["scale:region"]
    assert LSS_FIELDS <= set(term)
    assert term["parameter"] == "scale" and term["parameter_role"] == "spread"
    assert term["term_label"] == term["x_label"] == term["title"] == "region"
    display = display_for(open_case(case).model, "scale:region")
    assert term["display"] == display.to_payload()


def _transforms():
    return [RELATIVITY, INVERSE_THETA]


@pytest.mark.parametrize("transform", _transforms(), ids=lambda t: t.kind)
def test_display_round_trips(transform):
    effects = np.linspace(-2.5, 2.5, 41)
    back = transform.from_display(transform.to_display(effects))
    np.testing.assert_allclose(back, effects, rtol=1e3 * EPS, atol=1e3 * EPS)


@pytest.mark.parametrize("transform", _transforms(), ids=lambda t: t.kind)
def test_a_value_at_or_below_zero_clamps_to_the_smallest_relativity(transform):
    # A display value at or below zero is past every relativity's range: it takes the effect of
    # the smallest one, never that of its absolute value, and larger values never take smaller
    # shown effects.
    floor = transform.from_display(0.0)
    assert np.isfinite(floor)
    np.testing.assert_array_equal(transform.from_display([-3.0, -1.0, -0.3, -1e-9]), floor)
    probes = np.array([-3.0, -1.0, 0.0, 1e-9, 0.01, 0.3, 1.0, 3.0])
    direction = 1.0 if transform.increasing else -1.0
    assert np.all(np.diff(direction * transform.from_display(probes)) >= 0.0)
    assert np.all(np.isfinite(transform.to_display(transform.from_display([0.0, -3.0]))))


def test_a_dragged_relativity_converts_back_to_the_link_scale():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        transform = display_for(session.model, "scale:age")
        shown = float(session_payload(session)["scale:age"]["y"][5]) * 1.3
        widget._drag("scale:age", [5], values=[shown])
        edited = session.terms["scale:age"].edited_log_effect[5]
        np.testing.assert_allclose(edited, transform.from_display(shown), rtol=TOLERANCE)
        np.testing.assert_allclose(transform.to_display(edited), shown, rtol=1e3 * EPS)
        widget._drag("scale:age", [6], values=[0.0])  # below the floor: clamps, stays finite
        assert np.isfinite(session.terms["scale:age"].edited_log_effect[6])
    finally:
        widget.close()


def test_control_handle_converts_back_to_the_link_scale():
    session = open_case(nb_case())
    widget = session.widget()
    try:
        transform = display_for(session.model, "theta:age")
        before = widget.terms["theta:age"]["controls"]
        shown = before["y"][3] * 1.2
        widget._control("theta:age", 3, shown)
        controls = session.control_points("theta:age", n_handles=before["count"])
        np.testing.assert_allclose(
            transform.to_display(controls["log_effect"][3]), shown, rtol=1e3 * EPS
        )
    finally:
        widget.close()


def test_average_is_taken_in_display_units():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        transform = display_for(session.model, "scale:age")
        widget._select("scale:age", [3, 4, 5])
        before = transform.to_display(session.terms["scale:age"].edited_log_effect[[3, 4, 5]])
        weights = session.terms["scale:age"].weights[[3, 4, 5]]
        widget._operate("average", "scale:age")
        after = transform.to_display(session.terms["scale:age"].edited_log_effect[[3, 4, 5]])
        np.testing.assert_allclose(after, np.average(before, weights=weights), rtol=1e3 * EPS)
    finally:
        widget.close()


def test_lss_tool_payloads_follow_the_tools_a_superlss_offers():
    payload = session_payload(open_case(lognormal_case()))
    for key in ("mean:age", "mean:region", "scale:age"):
        term = payload[key]
        # Shaped ranges are offered on the spline terms, as on a SuperGLM's.
        assert term["shape"]["available"] is (key != "mean:region")
        assert term["unsmoothed"] is False and term["unseen"] is None
        assert term["rules"] is None
    assert payload["mean:region"]["reference"]["level"] == "R0"
    assert payload["mean:age"]["knots"]["available"] is True
    assert payload["scale:age"]["knots"]["available"] is True
    assert payload["mean:region"]["knots"]["available"] is False


def test_glm_payload_has_no_parameter_fields(editor_log_link_model):  # noqa: F811
    session = EditorSession.from_model(editor_log_link_model, terms=["x_spline"])
    term = session_payload(session)["x_spline"]
    assert not (LSS_FIELDS | {"y_format", "baseline"}) & set(term)
    assert set(term) == GLM_TERM_KEYS
    np.testing.assert_array_equal(term["y"], np.exp(session.terms["x_spline"].edited_log_effect))
    assert term["y_label"] == "relativity" and term["title"] == "x_spline"


#: Effects each transform is pinned at.
_EFFECTS = [-6.0, -1.5, -0.25, 0.0, 0.25, 1.5, 3.0]


def _fixture_content() -> dict:
    out = {}
    for transform in [*_transforms(), ADDITIVE]:
        values = transform.to_display(_EFFECTS)
        # Values at and past what a relativity can take convert to a finite effect.
        probes = [-3.0, -1.0, -0.3, 0.0, 1e-9, 0.01, 0.5, 1.0, 2.0]
        out[transform.kind] = {
            "kind": transform.kind,
            "effects": _EFFECTS,
            "display": [float(v) for v in values],
            "probes": probes,
            "effects_from_probes": [float(v) for v in transform.from_display(probes)],
        }
    return out


def _ordered_gamma_session():
    """A gamma model with an ordered spline term on both parameters, on the fit's rows."""
    rng = np.random.default_rng(20261104)
    order = ["low", "mid", "high", "top", "peak"]
    band = rng.choice(order, 1500, p=[0.35, 0.3, 0.2, 0.1, 0.05])
    X = pd.DataFrame({"band": band})
    position = X["band"].map({level: index for index, level in enumerate(order)}).to_numpy()
    cv = np.exp(-0.5 + 0.1 * position)
    y = rng.gamma(1.0 / cv**2, np.exp(1.0 + 0.2 * position) * cv**2)
    family = GammaLS()
    basis = Spline(kind="cr", k=4)
    model = SuperLSS(
        family,
        family.mean(term("band", OrderedCategorical(order=order, basis=basis))),
        family.scale(term("band", OrderedCategorical(order=order, basis=basis))),
    ).fit_reml(X, y)
    return edit(model, train_data=(X, y)), order


@pytest.mark.parametrize("key", ["mean:band", "scale:band"])
def test_an_ordered_term_reads_no_effect_at_its_reference_level(key):
    session, _ = _ordered_gamma_session()
    term_payload = session_payload(session)[key]
    base = term_payload["levels"].index(term_payload["reference"]["level"])
    np.testing.assert_allclose(term_payload["y"][base], 1.0, rtol=TOLERANCE, atol=0.0)


def test_an_ordered_spread_curve_is_the_models_relativity_at_each_level():
    session, order = _ordered_gamma_session()
    term_payload = session_payload(session)["scale:band"]
    rows = pd.DataFrame({"band": order})
    predicted = np.exp(session.model.lss.predict_link(rows)["scale"].to_numpy())
    index = term_payload["levels"].index(term_payload["reference"]["level"])
    np.testing.assert_allclose(
        term_payload["y"], predicted / predicted[index], rtol=64 * TOLERANCE, atol=0.0
    )


@pytest.mark.parametrize("key", ["mean:band", "scale:band"])
def test_an_ordered_spline_handle_writes_the_weighted_projection_of_its_curve(key):
    session, order = _ordered_gamma_session()
    term_payload = session_payload(session)[key]
    assert term_payload["spline_view"]["available"] is True
    assert term_payload["spline_view"]["fits_levels"] is True
    controls = session.control_points(key)
    middle = len(controls["log_effect"]) // 2
    session.move_control_point(key, middle, controls["log_effect"][middle] + 0.3)

    edited = session.terms[key]
    moved = edited.edited_log_effect - edited.original_log_effect
    assert np.max(np.abs(moved)) > 0.01

    parameter = key.split(":")[0]
    rows = pd.DataFrame({"band": order})
    view = session.model
    before = view.lss.predict_link(rows)[parameter].to_numpy()
    after = lss_model_with_edits(view, session.terms).predict_link(rows)[parameter].to_numpy()

    # The model holds what the term's columns and the intercept can carry of the curve.
    _, state, term_slice = _resolve_term(view._fitted, parameter, "band")
    columns = np.column_stack(
        [np.ones(len(order)), _term_design(view._fitted, rows, state, term_slice)]
    )
    root = np.sqrt(np.asarray(edited.weights, dtype=np.float64))[:, None]
    carried, *_ = np.linalg.lstsq(root * columns, root[:, 0] * moved, rcond=None)
    np.testing.assert_allclose(after - before, columns @ carried, rtol=0.0, atol=1e-9)
    assert np.abs(after - before).max() > 0.01


def test_a_small_edit_of_an_ordered_term_moves_the_model_by_that_much():
    session, order = _ordered_gamma_session()
    view = session.model
    edited = session.terms["scale:band"]
    edited.edited_log_effect = edited.original_log_effect + 1e-9
    rows = pd.DataFrame({"band": order})
    before = view.lss.predict_link(rows)["scale"].to_numpy()
    after = lss_model_with_edits(view, session.terms).predict_link(rows)["scale"].to_numpy()
    np.testing.assert_allclose(after - before, 1e-9, rtol=0.0, atol=1e-11)


def test_display_fixture_is_current():
    content = _fixture_content()
    if os.environ.get("SUPERGLM_WRITE_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(content, indent=2) + "\n")
    stored = json.loads(FIXTURE.read_text())
    assert stored.keys() == content.keys()
    for kind, entry in content.items():
        for field, value in entry.items():
            if isinstance(value, list):
                np.testing.assert_allclose(stored[kind][field], value, rtol=TOLERANCE, atol=0.0)
            elif isinstance(value, float):
                assert stored[kind][field] == pytest.approx(value, rel=TOLERANCE)
            else:
                assert stored[kind][field] == value


def _shown_after(operation, key="theta:age"):
    """The shown curve before and after ``operation`` on every point of an NB spread term,
    its display transform and the term's weights.

    A dip at the middle point is drawn first, so the curve both rises and falls.
    """
    session = open_case(nb_case())
    widget = session.widget()
    try:
        transform = display_for(session.model, key)
        assert not transform.increasing
        middle = session.terms[key].size // 2
        shown = transform.to_display(session.terms[key].edited_log_effect[middle])
        widget._drag(key, [middle], values=[0.5 * float(shown)])
        widget._select(key, list(range(session.terms[key].size)))
        before = transform.to_display(session.terms[key].edited_log_effect.copy())
        widget._operate(operation, key)
        after = transform.to_display(session.terms[key].edited_log_effect)
        return before, after, transform, session.terms[key].weights
    finally:
        widget.close()


@pytest.mark.parametrize(("operation", "factor"), [("shift_up", 1.05), ("shift_down", 0.95)])
def test_shift_moves_a_falling_display_the_way_it_names(operation, factor):
    before, after, _, _ = _shown_after(operation)
    np.testing.assert_allclose(after / before, factor, rtol=1e3 * EPS, atol=0.0)


@pytest.mark.parametrize(("operation", "sign"), [("increasing", 1.0), ("decreasing", -1.0)])
def test_monotone_fit_makes_the_shown_curve_monotone(operation, sign):
    before, after, transform, _ = _shown_after(operation)
    # The curve rises and falls, so each fit has to move it.
    assert np.any(np.diff(before) < 0.0) and np.any(np.diff(before) > 0.0)
    steps = sign * np.diff(after)
    assert np.all(steps >= -1e3 * EPS * np.max(np.abs(after)))
    # 1/theta falls as the effect rises: the shown fit is the effect's fit the other way.
    effect = transform.from_display(before)
    other = "decreasing" if operation == "increasing" else "increasing"
    index = np.arange(effect.size, dtype=np.intp)
    expected = running_extreme_values(effect, index, other)
    np.testing.assert_allclose(after, transform.to_display(expected), rtol=1e3 * EPS, atol=0.0)


@pytest.mark.parametrize(("operation", "pick"), [("snap_highest", np.max), ("snap_lowest", np.min)])
def test_snap_takes_the_shown_extreme(operation, pick):
    before, after, _, _ = _shown_after(operation)
    np.testing.assert_allclose(after, pick(before), rtol=1e3 * EPS, atol=0.0)


def test_directional_tools_on_a_rising_display_are_unchanged():
    session = open_case(gamma_case())
    widget = session.widget()
    try:
        transform = display_for(session.model, "scale:age")
        assert transform.increasing
        widget._select("scale:age", [2, 3, 4])
        before = session.terms["scale:age"].edited_log_effect[[2, 3, 4]].copy()
        widget._operate("shift_up", "scale:age")
        after = session.terms["scale:age"].edited_log_effect[[2, 3, 4]]
        np.testing.assert_allclose(after - before, np.log(1.05), rtol=1e3 * EPS, atol=0.0)
    finally:
        widget.close()


def test_history_names_a_falling_display_by_its_shown_direction():
    session = open_case(nb_case())
    widget = session.widget()
    try:
        widget._select("theta:age", [1, 2])
        widget._operate("shift_up", "theta:age")
        entry = next(e for e in widget._state()["timeline"] if e.get("operation") == "shift")
        assert entry["params"]["delta"] < 0.0
        assert entry["display_decreasing"] is True
        widget._select("mean:age", [1, 2])
        widget._operate("shift_up", "mean:age")
        entries = [e for e in widget._state()["timeline"] if e.get("term") == "mean:age"]
        assert entries[-1]["display_decreasing"] is False
    finally:
        widget.close()


def test_editing_one_curve_of_a_parameter_leaves_the_others_as_they_were():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        before = session_payload(session)["scale:age"]
        region = session_payload(session)["scale:region"]
        reference = session.model.reference_row()["region"].iloc[0]
        index = region["levels"].index(reference)
        widget._drag("scale:region", [index], values=[region["y"][index] * 1.3])
        after = session_payload(session)["scale:age"]
        assert after["y"] == before["y"]
        assert after["original_y"] == before["original_y"]
    finally:
        widget.close()


def test_a_drag_converts_with_the_same_relativity_it_shows():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        region = session_payload(session)["scale:region"]
        widget._drag("scale:region", [0], values=[region["y"][0] * 1.3])
        target = float(session_payload(session)["scale:age"]["y"][5]) * 1.2
        widget._drag("scale:age", [5], values=[target])
        edited = session.terms["scale:age"].edited_log_effect[5]
        np.testing.assert_allclose(np.exp(edited), target, rtol=1e3 * EPS)
        np.testing.assert_allclose(
            session_payload(session)["scale:age"]["y"][5], target, rtol=1e3 * EPS
        )
    finally:
        widget.close()
