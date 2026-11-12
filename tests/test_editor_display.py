"""A term's display follows the link it was fitted on, whatever the model's family."""

from __future__ import annotations

import numpy as np
import pytest

from superglm import GammaLS, GaussianLS, SuperGLM
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
