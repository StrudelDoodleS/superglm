"""A term's display follows the link it was fitted on, whatever the model's family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from superglm import Categorical, GammaLS, GaussianLS, SuperGLM
from superglm.distributional.families import LomaxLSS
from superglm.distributional.families._links import BoundedLogitLink
from superglm.distributional.families.gaussian import LowerBoundedLogLink
from superglm.distributional.families.tweedie import BoundedPowerLink
from superglm.editor import EditorSession
from superglm.editor.cv import capture_cv_view, fold_log_curves, fold_term_items
from superglm.editor.display import (
    ADDITIVE,
    average_parameter,
    display_for,
    transform_for_link,
)
from superglm.editor.payloads import session_payload
from superglm.links import (
    CloglogLink,
    IdentityLink,
    InverseLink,
    LogitLink,
    LogLink,
    ProbitLink,
    SqrtLink,
)
from tests._lss_editor_fixtures import EPS
from tests.test_editor import _editor_features, editor_frame, editor_model  # noqa: F401


@pytest.mark.parametrize(
    ("link", "kind", "name"),
    [
        (LogLink(), "relativity", "log"),
        (LowerBoundedLogLink(0.01), "relativity", "floored_log"),
        (LogitLink(), "relativity", "logit"),
        (IdentityLink(), "additive", "identity"),
        (InverseLink(), "additive", "inverse"),
        (ProbitLink(), "additive", "probit"),
        (CloglogLink(), "additive", "cloglog"),
        (SqrtLink(), "additive", "sqrt"),
        (BoundedLogitLink(-0.9, 0.9), "additive", "bounded_logit"),
        (BoundedPowerLink(1.05, 1.95), "additive", "bounded_power"),
    ],
    ids=lambda value: type(value).__name__ if not isinstance(value, str) else value,
)
def test_transform_follows_the_link(link, kind, name):
    transform = transform_for_link(link)
    assert transform.kind == kind
    assert transform.to_payload() == {"kind": kind, "link": name}


def test_size_role_shows_the_reciprocal():
    assert transform_for_link(LogLink(), reciprocal=True).kind == "inverse_theta"


def test_additive_maps_are_identity():
    values = [-3.0, 0.0, 2.5]
    np.testing.assert_array_equal(ADDITIVE.to_display(values), values)
    np.testing.assert_array_equal(ADDITIVE.from_display(values), values)
    assert ADDITIVE.no_effect == 0.0
    assert ADDITIVE.increasing is True
    assert transform_for_link(LogLink()).no_effect == 1.0


def test_the_average_parameter_is_the_mean_else_the_location_else_the_first():
    assert average_parameter(GammaLS()) == "mean"
    assert average_parameter(GaussianLS()) == "location"
    assert average_parameter(LomaxLSS()) == "scale"


def _glm(family: str, frame, y, *, link: str | None = None) -> EditorSession:
    X, _ = frame
    model = SuperGLM(
        family=family,
        link=link,
        selection_penalty=0.0,
        spline_penalty=0.1,
        features=_editor_features(),
    )
    model.fit(X, y)
    return EditorSession.from_model(model, terms=["x_spline", "region"])


def test_glm_identity_payload_shows_the_effect(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model, terms=["x_spline", "region"])
    payload = session_payload(session)
    for name in ("x_spline", "region"):
        term = payload[name]
        np.testing.assert_allclose(
            term["y"], session.terms[name].edited_log_effect, rtol=0.0, atol=1e-12
        )
        assert term["y_label"] == "effect on the mean"
        assert term["display"] == {"kind": "additive", "link": "identity"}
        assert display_for(session.model, name) == ADDITIVE


def test_glm_logit_payload_is_an_odds_relativity(editor_frame):  # noqa: F811
    X, signal = editor_frame
    y = (signal > np.median(signal)).astype(np.float64)
    session = _glm("binomial", editor_frame, y)
    term = session_payload(session)["x_spline"]
    np.testing.assert_allclose(
        term["y"], np.exp(session.terms["x_spline"].edited_log_effect), rtol=1e-12, atol=0.0
    )
    assert term["y_label"] == "relativity of the odds"
    assert "display" not in term


def test_glm_log_payload_is_unchanged(editor_frame):  # noqa: F811
    _, signal = editor_frame
    y = np.random.default_rng(20261112).poisson(np.exp(signal - 0.6)).astype(np.float64)
    session = _glm("poisson", editor_frame, y)
    term = session_payload(session)["x_spline"]
    assert term["y_label"] == "relativity"
    assert "display" not in term
    assert display_for(session.model, "x_spline").to_payload() == {
        "kind": "relativity",
        "link": "log",
    }


def test_cv_fold_items_follow_the_glm_link(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    shown = session_payload(session)["x_spline"]
    curves = {0: fold_log_curves(editor_model, session.terms)}
    view = capture_cv_view(session, run=None, final_fit=None)
    (item,) = fold_term_items(session.terms, curves, display=view.display)
    np.testing.assert_allclose(item["fit"], shown["original_y"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(item["fit"], session.terms["x_spline"].original_log_effect)
    assert item["value_label"] == "effect on the mean"


@pytest.fixture
def thousands_session(editor_frame):  # noqa: F811
    """A Gaussian/identity SuperGLM on a response near 3000, with exposure weights."""
    X, signal = editor_frame
    rng = np.random.default_rng(20261112)
    weights = rng.uniform(0.5, 2.0, len(X))
    y = 3000.0 + 400.0 * (signal - 0.6) + rng.normal(0.0, 20.0, len(X))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features=_editor_features(),
    )
    model.fit(X, y, sample_weight=weights)
    return EditorSession.from_model(model, terms=["x_spline", "region", "band"]), X, weights


def test_additive_increase_moves_an_average_row_by_about_five_percent(thousands_session):
    session, X, weights = thousands_session
    expected = 0.05 * np.average(np.abs(session.model.predict(X)), weights=weights)
    step = session.additive_step("x_spline")
    assert step == pytest.approx(expected, rel=0.01)
    widget = session.widget()
    try:
        effect = session.terms["x_spline"].edited_log_effect
        before = effect[5]
        widget._select("x_spline", [5])
        widget._operate("shift_up", "x_spline")
        assert session.terms["x_spline"].edited_log_effect[5] - before == pytest.approx(step)
        widget._operate("shift_down", "x_spline")
        assert session.terms["x_spline"].edited_log_effect[5] == pytest.approx(before)
    finally:
        widget.close()


def test_additive_step_is_read_once_per_fit(thousands_session, monkeypatch):
    import superglm.editor.session as session_module

    session, _, _ = thousands_session
    calls = []
    read = session_module._mean_abs_eta

    def counted(model, parameter):
        calls.append(parameter)
        return read(model, parameter)

    monkeypatch.setattr(session_module, "_mean_abs_eta", counted)
    first = session.additive_step("x_spline")
    assert session.additive_step("region") == first
    assert calls == [None]


def test_log_increase_is_unchanged(editor_frame):  # noqa: F811
    _, signal = editor_frame
    y = np.random.default_rng(20261112).poisson(np.exp(signal - 0.6)).astype(np.float64)
    session = _glm("poisson", editor_frame, y)
    widget = session.widget()
    try:
        before = session.terms["x_spline"].edited_log_effect[5]
        widget._select("x_spline", [5])
        widget._operate("shift_up", "x_spline")
        moved = session.terms["x_spline"].edited_log_effect[5] - before
        assert moved == pytest.approx(np.log(1.05), rel=1e-12)
        widget._operate("shift_down", "x_spline")
        moved = session.terms["x_spline"].edited_log_effect[5] - before
        assert moved == pytest.approx(np.log(1.05) + np.log(0.95), abs=1e-12)
        assert "weighted_mean_display" not in session_payload(session)["x_spline"]["impact"]
    finally:
        widget.close()


def _shows_its_effect(session, name):
    term = session_payload(session)[name]
    effect = session.terms[name].edited_log_effect
    assert np.all(np.isfinite(effect))
    np.testing.assert_allclose(term["y"], effect, rtol=0.0, atol=1e-12)
    return term


def test_every_tool_runs_on_an_additive_curve(thousands_session):
    session, _, _ = thousands_session
    widget = session.widget()
    try:
        for name, indices in (("x_spline", [3, 4, 5, 6, 7, 8]), ("region", [0, 1, 2])):
            for operation in (
                "average",
                "linearise",
                "smooth",
                "increasing",
                "decreasing",
                "snap_highest",
                "snap_lowest",
            ):
                before = session.terms[name].edited_log_effect.copy()
                widget._select(name, indices)
                widget._operate(operation, name)
                _shows_its_effect(session, name)
                widget._operate("undo")
                np.testing.assert_array_equal(session.terms[name].edited_log_effect, before)

        reference = session.terms["x_spline"].original_log_effect
        widget._drag("x_spline", [3], values=[-50.0])
        assert session.terms["x_spline"].edited_log_effect[3] == -50.0
        term = _shows_its_effect(session, "x_spline")
        delta = session.terms["x_spline"].edited_log_effect - reference
        assert term["impact"]["weighted_mean_display"] == pytest.approx(
            np.average(delta, weights=term["weights"])
        )

        widget._control("x_spline", 1, -50.0)
        assert np.min(_shows_its_effect(session, "x_spline")["y"]) < -10.0
        widget._control("band", 1, -50.0)
        assert np.min(_shows_its_effect(session, "band")["y"]) < -10.0
        for _ in range(3):
            widget._operate("undo")
        np.testing.assert_array_equal(session.terms["x_spline"].edited_log_effect, reference)
    finally:
        widget.close()


def test_additive_step_reads_a_superlss_parameters_eta():
    from tests._lss_editor_fixtures import gamma_case, open_case

    case = gamma_case()
    session = open_case(case)
    fitted = case.model._require_fitted()
    (state,) = [p for p in fitted.layout.predictors if p.name == "scale"]
    eta = state.link.link(case.model.predict_parameters(case.X)["scale"].to_numpy())
    assert session.additive_step("scale:age") == pytest.approx(
        0.05 * np.mean(np.abs(eta)), rel=1e-6
    )


def test_an_identity_band_beyond_500_is_the_effect_plus_or_minus_z_se(editor_frame):  # noqa: F811
    X, signal = editor_frame
    noise = np.random.default_rng(20261112).normal(0.0, 100.0, len(X))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features=_editor_features(),
    )
    model.fit(X, 3000.0 + 6000.0 * (signal - 0.6) + noise)
    session = EditorSession.from_model(model, terms=["x_spline", "band"])
    payload = session_payload(session)
    beyond = False
    for name in ("x_spline", "band"):
        reference = model.term_inference(name, with_se=True, n_points=200, centering="native")
        z = norm.ppf(1.0 - reference.alpha / 2.0)
        effect, se = reference.log_relativity, reference.se_log_relativity
        lower, upper = payload[name]["ci_lower_y"], payload[name]["ci_upper_y"]
        scale = 1.0 + np.max(np.abs(effect) + z * se)
        np.testing.assert_allclose(lower, effect - z * se, rtol=0.0, atol=64 * EPS * scale)
        np.testing.assert_allclose(upper, effect + z * se, rtol=0.0, atol=64 * EPS * scale)
        beyond |= bool(np.any(np.abs(np.concatenate([lower, upper])) > 500.0))
    # The band leaves the range a relativity's exponential is clipped to.
    assert beyond


def test_a_log_link_band_wider_than_float64_holds_is_drawn_finite_and_ordered():
    """A level the rows barely fix has link-scale band ends no exponential holds.

    Its ends are held at +-500 on the link scale before the relativity is
    drawn, so the band stays finite and ordered around the curve.
    """
    X = pd.DataFrame({"level": ["A"] * 10 + ["B"] * 10})
    y = np.array([-9.0, 11.0] * 5 + [-9.999, 10.001] * 5)
    model = SuperGLM(
        family="gaussian",
        link="log",
        selection_penalty=0.0,
        tol=1e-12,
        features={"level": Categorical(base="first")},
    ).fit(X, y)
    assert model.result.converged
    term = session_payload(EditorSession.from_model(model))["level"]
    lower, curve, upper = (np.asarray(term[k]) for k in ("ci_lower_y", "y", "ci_upper_y"))
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= curve) and np.all(curve <= upper)
    assert lower[1] < upper[1]
