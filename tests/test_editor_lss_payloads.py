"""SuperLSS terms in the editor payload: display units, conversions back, GLM unchanged."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm.editor import EditorSession
from superglm.editor.lss import RELATIVITY, DisplayTransform, display_for
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.availability import LSS_UNAVAILABLE
from superglm.editor.operations import isotonic_values
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import EPS, gamma_case, lognormal_case, nb_case, open_case
from tests.test_editor import editor_frame, editor_model  # noqa: F401

FIXTURE = Path(__file__).parent / "editor_frontend" / "fixtures" / "display_transforms.json"
TOLERANCE = 64 * EPS
# Every key of a SuperGLM term's payload; a SuperLSS key on it would change a SuperGLM page.
GLM_TERM_KEYS = {
    "kind", "term_type", "x", "x_domain", "y", "original_y", "previous_y", "controls",
    "spline_view", "ci_lower_y", "ci_upper_y", "weights", "exposure", "handle_indices", "impact",
    "n_points", "levels", "level_groups", "level_order_changed", "reference", "shape", "knots",
    "pending", "unseen", "unsmoothed", "effective_df", "edited", "x_label", "y_label", "title",
    "group_display",
}  # fmt: skip
LSS_FIELDS = {
    "parameter",
    "parameter_label",
    "parameter_role",
    "term_label",
    "display",
    "baseline",
}


def _parameter_at(case, session, term, parameter):
    frame = pd.concat([session.model.reference_row()] * len(term["x"]), ignore_index=True)
    frame["age"] = term["x"]
    return case.model.predict_parameters(frame)[parameter].to_numpy()


def test_spread_curve_is_shown_as_coefficient_of_variation():
    case = lognormal_case()
    session = open_case(case)
    term = session_payload(session)["scale:age"]
    s = _parameter_at(case, session, term, "scale")
    np.testing.assert_allclose(term["y"], np.sqrt(np.expm1(s**2)), rtol=TOLERANCE)
    assert (term["y_label"], term["y_format"]) == ("coefficient of variation", "times")


def test_gamma_scale_is_its_cv():
    case = gamma_case()
    session = open_case(case)
    term = session_payload(session)["scale:age"]
    scale = _parameter_at(case, session, term, "scale")
    np.testing.assert_allclose(term["y"], scale, rtol=TOLERANCE)
    assert (term["y_label"], term["y_format"]) == ("coefficient of variation", "times")


def test_nb_theta_is_shown_inverted():
    case = nb_case()
    session = open_case(case)
    term = session_payload(session)["theta:age"]
    theta = _parameter_at(case, session, term, "theta")
    np.testing.assert_allclose(term["y"], 1.0 / theta, rtol=TOLERANCE)
    assert (term["y_label"], term["y_format"]) == ("1/theta", "plain")
    # 1/theta falls as the effect rises, so the interval's ends are the other way round.
    assert np.all(np.asarray(term["ci_lower_y"]) <= np.asarray(term["y"]) + TOLERANCE)
    assert np.all(np.asarray(term["ci_upper_y"]) >= np.asarray(term["y"]) - TOLERANCE)


@pytest.mark.parametrize("case", [lognormal_case, gamma_case, nb_case])
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
    assert term["y_label"] == "relativity" and term["y_format"] == "relativity"
    assert term["baseline"] == 1.0
    assert term["parameter"] == "mean"


def test_lss_terms_carry_their_fields():
    case = lognormal_case()
    term = session_payload(open_case(case))["scale:region"]
    assert LSS_FIELDS <= set(term)
    assert term["parameter"] == "scale" and term["parameter_role"] == "spread"
    assert term["term_label"] == term["x_label"] == term["title"] == "region"
    display = display_for(open_case(case).model, "scale:region")
    assert term["display"] == display.to_payload()
    assert term["baseline"] == float(display.to_display(0.0))


def _transforms():
    return [
        RELATIVITY,
        DisplayTransform("lognormal_cv", -0.7, 0.05),
        DisplayTransform("gamma_cv", -0.6),
        DisplayTransform("inverse_theta", 1.0),
    ]


@pytest.mark.parametrize("transform", _transforms(), ids=lambda t: t.kind)
def test_display_round_trips(transform):
    effects = np.linspace(-2.5, 2.5, 41)
    back = transform.from_display(transform.to_display(effects))
    np.testing.assert_allclose(back, effects, rtol=1e3 * EPS, atol=1e3 * EPS)


def test_out_of_range_spread_clamps():
    lognormal = DisplayTransform("lognormal_cv", -0.7, 0.05)
    below = lognormal.from_display([0.0, 1e-9, np.sqrt(np.expm1(0.05**2)) * 0.5])
    assert np.all(np.isfinite(below))
    # The nearest reachable curve sits at the floor's own coefficient of variation.
    np.testing.assert_allclose(lognormal.to_display(below), np.sqrt(np.expm1(0.05**2)), rtol=1e-6)
    for transform in (DisplayTransform("inverse_theta", 1.0), DisplayTransform("gamma_cv", -0.6)):
        effects = transform.from_display([0.0, -3.0])
        assert np.all(np.isfinite(effects))
        assert np.all(np.isfinite(transform.to_display(effects)))


@pytest.mark.parametrize("transform", _transforms()[1:], ids=lambda t: t.kind)
def test_spread_at_or_below_zero_clamps_to_the_floor(transform):
    # A display value at or below zero is past every spread's range: it takes the effect of
    # zero, never that of its absolute value, and larger values never take smaller effects.
    floor = transform.from_display(0.0)
    np.testing.assert_array_equal(transform.from_display([-3.0, -1.0, -0.3, -1e-9]), floor)
    probes = np.array([-3.0, -1.0, 0.0, 1e-9, 0.01, 0.3, 1.0, 3.0])
    effects = transform.from_display(probes)
    direction = 1.0 if transform.kind != "inverse_theta" else -1.0
    assert np.all(np.diff(direction * effects) >= 0.0)


def test_dragged_cv_converts_back_to_the_link_scale():
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
        assert term["shape"]["available"] is False
        assert term["shape"]["reason"] == LSS_UNAVAILABLE["shape"]
        assert term["unsmoothed"] is False and term["unseen"] is None
    assert payload["mean:region"]["reference"]["level"] == "R0"
    assert payload["mean:age"]["knots"]["available"] is True
    assert payload["scale:age"]["knots"]["available"] is True
    assert payload["mean:region"]["knots"]["available"] is False


def test_glm_payload_has_no_parameter_fields(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    term = session_payload(session)["x_spline"]
    assert not (LSS_FIELDS | {"y_format"}) & set(term)
    assert set(term) == GLM_TERM_KEYS
    np.testing.assert_array_equal(term["y"], np.exp(session.terms["x_spline"].edited_log_effect))
    assert term["y_label"] == "relativity" and term["title"] == "x_spline"


def _fixture_content() -> dict:
    effects = [-6.0, -1.5, -0.25, 0.0, 0.25, 1.5, 3.0]
    out = {}
    for transform in _transforms():
        values = transform.to_display(effects)
        entry = {
            **transform.to_payload(),
            "effects": effects,
            "display": [float(v) for v in values],
        }
        if transform.kind != "relativity":
            # Values at and past what the parameter can take convert to a finite effect.
            probes = [-3.0, -1.0, -0.3, 0.0, 1e-9, 0.01, 0.5, 2.0]
            entry["probes"] = probes
            entry["effects_from_probes"] = [float(v) for v in transform.from_display(probes)]
        else:
            entry["probes"] = [-1.0, 0.0, 0.5, 1.0, 2.0]
            entry["effects_from_probes"] = [
                float(v) for v in transform.from_display(entry["probes"])
            ]
        out[transform.kind] = entry
    return out


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
    before, after, transform, weights = _shown_after(operation)
    # The curve rises and falls, so each fit has to move it.
    assert np.any(np.diff(before) < 0.0) and np.any(np.diff(before) > 0.0)
    steps = sign * np.diff(after)
    assert np.all(steps >= -1e3 * EPS * np.max(np.abs(after)))
    # 1/theta falls as the effect rises: the shown fit is the effect's fit the other way.
    effect = transform.from_display(before)
    other = "decreasing" if operation == "increasing" else "increasing"
    index = np.arange(effect.size, dtype=np.intp)
    expected = isotonic_values(effect, index, weights, other)
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


def _cv_at(model, session, term, column, parameter="scale"):
    frame = pd.concat([session.model.reference_row()] * len(term["x"]), ignore_index=True)
    frame[column] = term["x"]
    s = model.predict_parameters(frame)[parameter].to_numpy()
    return np.sqrt(np.expm1(s**2))


@pytest.mark.parametrize("level_edit", ["reference", "other"])
def test_spread_curve_follows_an_edit_to_another_term_of_its_parameter(level_edit):
    case = lognormal_case()
    session = open_case(case)
    widget = session.widget()
    try:
        before = session_payload(session)["scale:age"]
        region = session_payload(session)["scale:region"]
        reference = session.model.reference_row()["region"].iloc[0]
        index = region["levels"].index(reference) if level_edit == "reference" else 3
        assert (region["levels"][index] == reference) == (level_edit == "reference")
        widget._drag("scale:region", [index], values=[region["y"][index] * 1.3])
        edited = lss_model_with_edits(session.model, session.terms)
        after = session_payload(session)["scale:age"]
        shown = _cv_at(edited, session, after, "age")
        np.testing.assert_allclose(after["y"], shown, rtol=1e3 * EPS, atol=0.0)
        # The fit's curve stays where the fit put it.
        np.testing.assert_allclose(after["original_y"], before["original_y"], rtol=0, atol=0)
        if level_edit == "reference":
            # An edit at the reference level moves the whole spread at the reference row.
            assert np.min(np.abs(np.asarray(after["y"]) / np.asarray(before["y"]) - 1.0)) > 0.1
            assert after["display"]["reference_eta"] != before["display"]["reference_eta"]
        else:
            np.testing.assert_allclose(after["y"], before["y"], rtol=1e3 * EPS, atol=0.0)
    finally:
        widget.close()


def test_a_drag_converts_with_the_reference_in_force():
    case = lognormal_case()
    session = open_case(case)
    widget = session.widget()
    try:
        region = session_payload(session)["scale:region"]
        reference = session.model.reference_row()["region"].iloc[0]
        index = region["levels"].index(reference)
        widget._drag("scale:region", [index], values=[region["y"][index] * 1.3])
        moved = session_payload(session)["scale:age"]
        target = float(moved["y"][5]) * 1.2
        widget._drag("scale:age", [5], values=[target])
        transform = DisplayTransform(**moved["display"])
        edited = session.terms["scale:age"].edited_log_effect[5]
        np.testing.assert_allclose(transform.to_display(edited), target, rtol=1e3 * EPS)
        fitted = display_for(session.model, "scale:age")
        assert abs(float(fitted.to_display(edited)) / target - 1.0) > 0.1
        np.testing.assert_allclose(
            session_payload(session)["scale:age"]["y"][5], target, rtol=1e3 * EPS
        )
    finally:
        widget.close()
