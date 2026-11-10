"""SuperLSS shape rules in the editor: named on each term, and every edit kept inside them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, Poisson, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import open_case, ruled_case

# A grid curve meets its rule to the rounding of the projection's solve on unit-scale effects.
_TOL = 1e-9
_CAP = (np.log(0.8), np.log(1.25))


def _increasing(values) -> bool:
    return bool(np.min(np.diff(values)) >= -_TOL)


def _convex(values, x) -> bool:
    slopes = np.diff(values) / np.diff(x)
    return bool(np.min(np.diff(slopes)) >= -_TOL * np.max(np.abs(slopes), initial=1.0))


def _capped(session, key="scale:region") -> bool:
    term = session.terms[key]
    base = term.levels.index(str(session.model._specs[key]._base_level))
    relative = term.edited_log_effect - term.edited_log_effect[base]
    return bool(np.all(relative >= _CAP[0] - _TOL) and np.all(relative <= _CAP[1] + _TOL))


def _model_curve(model, column, parameter, x):
    case = ruled_case()
    rows = pd.concat([case.X.iloc[[0]]] * len(x), ignore_index=True)
    rows[column] = x
    return model.predict_link(rows)[parameter].to_numpy()


def test_each_term_names_its_rules_and_that_edits_keep_to_them():
    payload = session_payload(open_case(ruled_case()))

    assert payload["mean:power"]["rules"] == {"text": "increasing", "fit": True, "kept": True}
    assert payload["scale:age"]["rules"] == {"text": "convex", "fit": True, "kept": True}
    assert payload["scale:region"]["rules"] == {
        "text": "cap(0.8, 1.25)",
        "fit": True,
        "kept": True,
    }
    assert payload["mean:age"]["rules"] is None
    assert payload["mean:region"]["rules"] is None


def test_a_superglm_term_names_its_rules_without_keeping_edits_to_them():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.uniform(0.0, 10.0, 400), "z": rng.uniform(0.0, 10.0, 400)})
    y = rng.poisson(np.exp(0.1 * X["x"])).astype(float)
    model = SuperGLM(
        family=Poisson(),
        features={
            "x": Spline(kind="cr", k=6, constraint=Constraint.fit.increasing),
            "z": Spline(kind="cr", k=6, constraint=Constraint.postfit.decreasing),
        },
    ).fit(X, y)
    payload = session_payload(EditorSession.from_model(model, terms=["x", "z"]))

    assert payload["x"]["rules"] == {"text": "increasing", "fit": True, "kept": False}
    assert payload["z"]["rules"] == {"text": "decreasing", "fit": False, "kept": False}


def test_a_drag_against_an_increasing_rule_keeps_the_curve_increasing():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    tail = np.flatnonzero(term.x > 10.0)
    before = term.edited_log_effect.copy()

    session.set_values("mean:power", tail, term.edited_log_effect[tail] - 0.4)

    curve = session.terms["mean:power"].edited_log_effect
    assert _increasing(curve)
    # The edit moved the curve: the tail came down as far as the rule lets it.
    assert curve[-1] < before[-1] - 0.05
    # The model the editor writes is the curve it draws, inside the rule between grid points too.
    model = session.to_model()
    shown = _model_curve(model, "power", "mean", term.x)
    np.testing.assert_allclose(shown - shown[0], curve - curve[0], rtol=0.0, atol=1e-9)
    fine = _model_curve(model, "power", "mean", np.linspace(term.x[0], term.x[-1], 801))
    assert _increasing(fine)


def test_a_rule_kept_edit_is_one_step_undo_takes_back():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    before = term.edited_log_effect.copy()
    session.select_indices("mean:power", np.flatnonzero(term.x > 10.0))

    session.shift("mean:power", -0.4)

    record = session.history[-1]
    assert record.operation == "shift" and len(session.history) == 1
    np.testing.assert_array_equal(record.indices, np.arange(term.size))
    # History still names the stretch the user edited.
    assert record.params["lo"] > 10.0
    session.undo()
    np.testing.assert_array_equal(session.terms["mean:power"].edited_log_effect, before)
    session.redo()
    assert _increasing(session.terms["mean:power"].edited_log_effect)


def test_an_edit_the_rule_allows_is_kept_as_drawn():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    before = term.edited_log_effect.copy()
    session.select_indices("mean:power", np.arange(term.size))

    session.shift("mean:power", 0.1)

    np.testing.assert_allclose(
        session.terms["mean:power"].edited_log_effect, before + 0.1, rtol=0.0, atol=1e-9
    )


def test_an_edit_of_a_spread_curve_keeps_its_convex_rule():
    session = open_case(ruled_case())
    term = session.terms["scale:age"]
    middle = np.flatnonzero((term.x > 45.0) & (term.x < 60.0))

    session.set_values("scale:age", middle, term.edited_log_effect[middle] + 0.6)

    curve = session.terms["scale:age"].edited_log_effect
    assert _convex(curve, term.x)
    fine = np.linspace(term.x[0], term.x[-1], 801)
    assert _convex(_model_curve(session.to_model(), "age", "scale", fine), fine)


def test_a_level_set_past_its_cap_is_kept_at_the_cap():
    session = open_case(ruled_case())
    term = session.terms["scale:region"]
    base = term.levels.index(str(session.model._specs["scale:region"]._base_level))
    level = (base + 1) % term.size

    session.set_values("scale:region", [level], [term.edited_log_effect[level] + 2.0])

    assert _capped(session)
    curve = session.terms["scale:region"].edited_log_effect
    assert curve[level] - curve[base] == pytest.approx(_CAP[1], abs=1e-9)


def test_a_partial_reset_keeps_the_rule():
    session = open_case(ruled_case())
    term = session.terms["mean:power"]
    session.select_indices("mean:power", np.flatnonzero(term.x < 6.0))
    session.shift("mean:power", 0.5)
    # Putting back part of the raised start leaves a dip the rule does not allow.
    session.select_indices("mean:power", np.flatnonzero(term.x < 5.0))

    session.reset("mean:power")

    assert _increasing(session.terms["mean:power"].edited_log_effect)
    assert session.history[-1].operation == "reset"


def test_writing_a_curve_against_the_rule_projects_it_inside():
    session = open_case(ruled_case())
    term = session.terms["mean:power"].copy()
    term.edited_log_effect = term.original_log_effect.copy()
    term.edited_log_effect[term.x > 10.0] -= 0.4

    model = lss_model_with_edits(session.model, {term.name: term})

    fine = np.linspace(term.x[0], term.x[-1], 801)
    assert _increasing(_model_curve(model, "power", "mean", fine))


@pytest.mark.parametrize("key", ["mean:age", "scale:age"])
def test_shaped_ranges_are_offered_on_a_superlss_spline_without_a_rule(key):
    session = open_case(ruled_case())
    shape = session_payload(session)[key]["shape"]

    if key == "mean:age":
        assert shape["available"] is True and shape["reason"] is None
        assert shape["support"] is not None
        assert len(shape["support"]["below"]) == session.terms[key].size
    else:
        assert shape["available"] is False
        assert shape["reason"] == "Remove the term's shape constraint to add shaped ranges."


def test_a_shaped_range_on_a_superlss_term_refits_into_force():
    session = open_case(ruled_case())

    step = session.stage_structural("shape", "mean:age", {"lo": 70.0, "hi": 90.0, "degree": 1})
    session.refit_pending()

    assert step.label.startswith("Line 70")
    ranges = session_payload(session)["mean:age"]["shape"]["ranges"]
    assert [(r["degree"], r["label"]) for r in ranges] == [(1, "Line")]
    # The other terms keep their rules through the refit.
    assert _increasing(session.terms["mean:power"].edited_log_effect)
