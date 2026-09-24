"""Editor structural steps: one stack, set reference, transform, revert."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, OrderedCategorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.editor.payloads import session_payload
from superglm.editor.widget import EditorWidget
from tests.test_editor import _post_json


@pytest.fixture
def region_model():
    rng = np.random.default_rng(20260924)
    region = rng.choice(["A", "B", "C", "D"], 600, p=[0.3, 0.3, 0.2, 0.2])
    x = rng.uniform(0.0, 10.0, 600)
    effects = {"A": 0.0, "B": 0.15, "C": 0.2, "D": -0.1}
    y = 0.4 + np.array([effects[r] for r in region]) + 0.05 * x + rng.normal(0.0, 0.05, 600)
    X = pd.DataFrame({"region": region, "x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"region": Categorical(base="first"), "x": Spline(n_knots=6)},
    )
    model.fit(X, y)
    return model, X


def test_every_structural_step_pushes_one_restorable_entry(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    opened = session.model

    session.select_levels("region", ["B", "C"])
    collapsed = session.replace_with_collapsed_levels("region", method="fit")
    assert [s.operation for s in session.structure_history] == ["collapse_levels"]
    assert session.structure_history[-1].previous_model is opened
    assert session.structure_history[-1].label == "collapse B + C in region"

    session.select_levels("region", ["B", "C"])
    session.replace_with_ungrouped_levels("region", method="fit")
    # The ungroup removes the last group, so the pre-collapse fit is reused ...
    assert session.model is opened
    # ... but it is still a step of its own: Restore undoes it.
    assert [s.operation for s in session.structure_history] == [
        "collapse_levels",
        "ungroup_levels",
    ]
    assert session.structure_history[-1].label == "ungroup B, C in region"
    assert session.uncollapse_levels() is collapsed
    assert session.uncollapse_levels() is opened
    assert not session.can_uncollapse_levels()


def test_state_publishes_the_structure_history(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    widget = EditorWidget(session)
    try:
        assert widget._state()["structure_history"] == {"depth": 0, "last": None}
        session.select_levels("region", ["B", "C"])
        envelope = widget._collapse_levels("region", "fit")
        assert envelope["state"]["structure_history"] == {
            "depth": 1,
            "last": {
                "operation": "collapse_levels",
                "term": "region",
                "label": "collapse B + C in region",
            },
        }
        assert "last_collapse" not in envelope["state"]
        restored = widget._restore_structure()
        assert restored["timing"]["operation"] == "restore_structure"
        assert restored["state"]["structure_history"]["depth"] == 0
    finally:
        widget.close()


def test_offset_refit_summary_goes_unavailable_after_a_notebook_side_edit(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    widget = EditorWidget(session)
    try:
        session.select_indices("x", [3, 4])
        session.shift("x", 0.1)
        assert widget._refit_offset("fit")["available"] is True
        session.shift("x", 0.1)  # made in the notebook: never passes through the widget
        assert widget._summary("refit")["available"] is False
    finally:
        widget.close()


def test_set_reference_keeps_predictions_and_puts_the_level_at_one(region_model):
    model, X = region_model
    session = EditorSession.from_model(model, terms=["region"])
    before = session.model.predict(X)
    session.replace_with_reference_level("region", "C", method="fit")
    region = session.terms["region"]
    assert region.relativity[region.levels.index("C")] == 1.0
    # An unpenalized factor's reference is a reparametrisation: the optimum is
    # unchanged, so predictions agree to the refit's own convergence tolerance.
    np.testing.assert_allclose(session.model.predict(X), before, rtol=10 * model._tol)


def test_set_reference_on_an_ordered_smooth_keeps_predictions():
    rng = np.random.default_rng(20260925)
    bands = [f"B{i}" for i in range(1, 9)]
    band = rng.choice(bands, 800)
    y = 0.3 + 0.05 * np.array([bands.index(b) for b in band]) + rng.normal(0.0, 0.05, 800)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "band": OrderedCategorical(order=bands, basis=Spline(kind="ps", k=5), base="first")
        },
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["band"])
    before = session.model.predict(X)
    session.replace_with_reference_level("band", "B5", method="fit")
    # Holds because the constant lies in the smoothing penalty's null space
    # (spec §3.4). If this fails, stop and report; do not loosen the tolerance.
    np.testing.assert_allclose(session.model.predict(X), before, rtol=10 * model._tol)


def test_set_reference_changes_the_fit_under_a_selection_penalty(region_model):
    _, X = region_model
    y = region_model[0]._fit_y_ref
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.5,
        features={"region": Categorical(base="first"), "x": Numeric()},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["region"])
    # The factor must survive the penalty, or both fits agree trivially. If it
    # is zeroed, lower selection_penalty until it is active; never assert on a
    # dropped factor. The reference-dependence is first order in the penalty,
    # so a penalty much weaker than this falls under the threshold below.
    assert np.ptp(session.terms["region"].original_log_effect) > 0
    before = session.model.predict(X)
    session.replace_with_reference_level("region", "C", method="fit")
    # The hover text says the fit changes here; this pins that it is true.
    assert np.max(np.abs(session.model.predict(X) - before)) > 1e3 * model._tol


def test_set_reference_maps_a_numeric_level_label_to_its_native_value():
    rng = np.random.default_rng(20260926)
    band = rng.choice([1, 2, 3], 400)
    y = 0.5 + 0.1 * band + rng.normal(0.0, 0.05, 400)
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"band": Categorical(base="first")}
    )
    model.fit(pd.DataFrame({"band": band}), y)
    session = EditorSession.from_model(model, terms=["band"])
    session.replace_with_reference_level("band", "3", method="fit")
    assert session.model._specs["band"]._base_level == 3


def test_set_reference_on_a_grouped_member_pins_its_group(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.replace_with_reference_level("region", "B", method="fit")
    assert session.model._specs["region"]._base_level == "B+C"


def test_a_pinned_reference_survives_a_later_collapse(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    session.replace_with_reference_level("region", "C", method="fit")
    session.select_levels("region", ["A", "B"])
    session.replace_with_collapsed_levels("region", method="fit")
    assert session.model._specs["region"]._base_level == "C"


def test_set_reference_refuses_a_special_level():
    rng = np.random.default_rng(20260927)
    levels = ["0", "1", "2", "3", "4", "5"]
    band = rng.choice(levels, 600)
    y = 0.2 + 0.05 * band.astype(float) + rng.normal(0.0, 0.05, 600)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=levels, basis=Spline(kind="ps", k=5), specials=["0"])
        },
    )
    model.fit(pd.DataFrame({"band": band}), y)
    session = EditorSession.from_model(model, terms=["band"])
    with pytest.raises(EditorValueError, match="special level can't be the reference"):
        session.replace_with_reference_level("band", "0", method="fit")
    assert session.structure_history == []


def test_set_reference_refuses_a_term_used_by_an_interaction(region_model):
    _, X = region_model
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"region": Categorical(base="first"), "x": Numeric()},
        interactions=[("x", "region")],
    )
    model.fit(X, region_model[0]._fit_y_ref)
    session = EditorSession.from_model(model, terms=["region"])
    with pytest.raises(EditorValueError, match="used by interaction"):
        session.replace_with_reference_level("region", "C", method="fit")
    assert session.structure_history == []


def test_payload_reports_the_reference_and_reanchors_the_original_line(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    payload = session_payload(session)
    assert payload["region"]["reference"] == {"level": "A", "policy": "first"}
    assert payload["x"]["reference"] is None
    session.replace_with_reference_level("region", "C", method="fit")
    region = session_payload(session)["region"]
    assert region["reference"] == {"level": "C", "policy": "pinned"}
    # A pure reparametrisation: the opened model's curve, re-expressed against
    # the new reference, is the current curve.
    np.testing.assert_allclose(region["original_y"], region["y"], rtol=10 * model._tol)


def test_widget_http_set_reference_returns_transition_envelope(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    widget = session.widget()
    try:
        payload = _post_json(f"{widget.url}/set_reference", {"term": "region", "level": "C"})
        assert set(payload) == {"state", "summary", "timing"}
        assert payload["timing"]["operation"] == "set_reference"
        assert payload["state"]["terms"]["region"]["reference"] == {
            "level": "C",
            "policy": "pinned",
        }
        assert payload["state"]["structure_history"]["last"]["label"] == (
            "set reference of region to C"
        )
    finally:
        widget.close()
