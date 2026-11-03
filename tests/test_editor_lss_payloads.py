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
from superglm.editor.lss.availability import LSS_UNAVAILABLE
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import EPS, gamma_case, lognormal_case, nb_case, open_case
from tests.test_editor import editor_frame, editor_model  # noqa: F401

FIXTURE = Path(__file__).parent / "editor_frontend" / "fixtures" / "display_transforms.json"
TOLERANCE = 64 * EPS
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
            probes = [-1.0, 0.0, 1e-9, 0.01, 0.5, 2.0]
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
