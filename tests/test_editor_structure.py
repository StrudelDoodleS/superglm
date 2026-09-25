"""Editor structural steps: one stack, set reference, transform, revert."""

from __future__ import annotations

import io
import json
import urllib.error

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Constraint,
    LambdaPolicy,
    Numeric,
    OrderedCategorical,
    Piecewise,
    Polynomial,
    Spline,
    SuperGLM,
)
from superglm.editor import EditorSession
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.editor.payloads import session_payload
from superglm.editor.summaries import summary_payload
from superglm.editor.widget import EditorWidget
from superglm.export.summary import build_summary_export_payload
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


def test_state_says_when_the_in_force_model_is_not_the_opened_one(region_model, monkeypatch):
    model, _ = region_model

    def fit_instead_of_profiling(self, X, y, sample_weight=None, offset=None, **kwargs):
        self.fit(X, y, sample_weight=sample_weight, offset=offset)

    monkeypatch.setattr(type(model), "estimate_p", fit_instead_of_profiling)
    session = EditorSession.from_model(model, terms=["region"])
    widget = EditorWidget(session)
    try:
        assert widget._state()["in_force_is_original"] is True
        session.reprofile_distribution("tweedie_p")
        state = widget._state()
        # A re-profile replaces the in-force model and clears both histories:
        # only this fact leaves Revert something to do.
        assert state["structure_history"]["depth"] == 0 and state["history"]["active"] == []
        assert state["in_force_is_original"] is False
        assert widget._revert_to_original()["state"]["in_force_is_original"] is True
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


def test_unseen_levels_rated_at_the_reference_move_with_it():
    rng = np.random.default_rng(20260930)
    region = rng.choice(["A", "B", "C"], 600)
    y = 0.4 + 0.2 * (region == "C") + rng.normal(0.0, 0.05, 600)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"region": Categorical(base="first", unseen="base")},
    )
    model.fit(pd.DataFrame({"region": region}), y)
    session = EditorSession.from_model(model, terms=["region"])
    unseen = pd.DataFrame({"region": ["Z"]})
    with pytest.warns(UserWarning, match="unseen at fit"):
        before = session.model.predict(unseen)
    session.replace_with_reference_level("region", "C", method="fit")
    with pytest.warns(UserWarning, match="unseen at fit"):
        after = session.model.predict(unseen)
    # Disclosed in the hover, Help and tutorial (Max, 2026-09-25): an unseen
    # level is rated at the reference, so it moves from A's rate to C's.
    reference_rate = session.model.predict(pd.DataFrame({"region": ["C"]}))
    np.testing.assert_allclose(after, reference_rate, rtol=10 * model._tol)
    assert abs(after[0] - before[0]) > 1e3 * model._tol


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


@pytest.mark.parametrize(
    ("fixture", "term", "members"),
    [("region_model", "region", ["B", "C"]), ("banded", "band", ["B4", "B5"])],
    ids=["categorical", "ordered"],
)
def test_set_reference_on_a_group_label_pins_the_group(request, fixture, term, members):
    model = request.getfixturevalue(fixture)[0]
    session = EditorSession.from_model(model, terms=[term])
    session.select_levels(term, members)
    session.replace_with_collapsed_levels(term, method="fit")
    # A click on a group in the Collapsed display, or a selection of all its
    # members, sends the group's own label.
    label = "+".join(members)
    session.replace_with_reference_level(term, label, method="fit")
    assert session.model._specs[term]._base_level == label


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


@pytest.mark.parametrize("centering", ["native", "mean"])
def test_payload_reports_the_reference_and_reanchors_the_original_line(region_model, centering):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"], centering=centering)
    payload = session_payload(session)
    assert payload["region"]["reference"] == {"level": "A", "policy": "first"}
    assert payload["x"]["reference"] is None
    session.replace_with_reference_level("region", "C", method="fit")
    region = session_payload(session)["region"]
    assert region["reference"] == {"level": "C", "policy": "pinned"}
    # A pure reparametrisation: the opened model's curve, re-expressed against
    # the new reference, is the current curve.
    np.testing.assert_allclose(region["original_y"], region["y"], rtol=10 * model._tol)


def test_a_mean_centred_original_line_stays_put_for_an_untouched_term(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region", "x"], centering="mean")
    opened = session_payload(session)["region"]["original_y"]
    session.replace_with_transformed_term("x", form="polynomial", breaks=[], degree=2, method="fit")
    # A mean-centred curve has no reference to anchor at: the untouched term's
    # original line (and the impact read from it) is still the opened model's.
    np.testing.assert_allclose(
        session_payload(session)["region"]["original_y"], opened, rtol=10 * model._tol
    )


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


BANDS = [f"B{i}" for i in range(1, 9)]


@pytest.fixture
def banded():
    rng = np.random.default_rng(20260928)
    band = rng.choice(BANDS, 1200)
    x = rng.uniform(0.0, 10.0, 1200)
    kink = np.array([min(BANDS.index(b), 4) for b in band]) * 0.08
    y = 0.3 + kink + 0.04 * x + rng.normal(0.0, 0.05, 1200)
    X = pd.DataFrame({"band": band, "x": x})
    features = {
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5), base="first"),
        "x": Spline(n_knots=8),
    }
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features
    )
    model.fit(X, y)
    return model, X, y


TRANSFORM_CASES = [
    (
        "band",
        dict(form="piecewise", breaks=["B3", "B6"], degrees=[1, 2, 0]),
        lambda: OrderedCategorical(
            order=BANDS, basis=Piecewise(breaks=["B3", "B6"], degrees=[1, 2, 0]), base="first"
        ),
    ),
    (
        "band",
        dict(form="spline", breaks=["B3", "B6"]),
        lambda: OrderedCategorical(
            order=BANDS, basis=Spline(kind="ps", knots=["B3", "B6"]), base="first"
        ),
    ),
    (
        "band",
        dict(form="polynomial", breaks=[], degree=2),
        lambda: OrderedCategorical(order=BANDS, basis=Polynomial(degree=2), base="first"),
    ),
    ("x", dict(form="piecewise", breaks=[3.0, 6.5]), lambda: Piecewise(breaks=[3.0, 6.5])),
    ("x", dict(form="spline", breaks=[3.0, 6.5]), lambda: Spline(kind="ps", knots=[3.0, 6.5])),
    ("x", dict(form="polynomial", breaks=[], degree=3), lambda: Polynomial(degree=3)),
]


@pytest.mark.parametrize(("term", "request_", "expected_spec"), TRANSFORM_CASES)
def test_transformed_model_predicts_like_a_direct_fit_of_the_same_spec(
    banded, term, request_, expected_spec
):
    model, X, y = banded
    session = EditorSession.from_model(model, terms=[term])
    session.replace_with_transformed_term(term, method="fit", **request_)
    features = {
        "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5), base="first"),
        "x": Spline(n_knots=8),
    }
    features[term] = expected_spec()
    direct = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features
    )
    direct.fit(X, y)
    # Same spec, same data, same solver: agreement to the fit tolerance checks
    # the whole replacement path against an independent construction.
    np.testing.assert_allclose(session.model.predict(X), direct.predict(X), rtol=10 * model._tol)
    assert session.structure_history[-1].operation == "transform_term"


def test_transform_carries_a_monotone_constraint_to_a_spline_and_refuses_it_elsewhere(banded):
    _, X, y = banded
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"x": Spline(n_knots=8, constraint=Constraint.fit.increasing)},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["x"])
    for form in ("piecewise", "polynomial"):
        with pytest.raises(EditorValueError, match="carries the increasing constraint"):
            session.replace_with_transformed_term(
                "x", form=form, breaks=[5.0] if form == "piecewise" else [], degree=2, method="fit"
            )
    assert session.structure_history == []
    session.replace_with_transformed_term("x", form="spline", breaks=[5.0], method="fit")
    assert session.model._specs["x"].constraint_kind == "increasing"


def test_transform_keeps_the_base_specials_grouping_and_extrapolation(banded):
    _, X, y = banded
    X = X.assign(band=np.where(np.arange(len(X)) % 10 == 0, "MISSING", X["band"]))
    band_spec = OrderedCategorical(
        order=BANDS, basis=Spline(kind="ps", k=5), base="B4", specials=["MISSING"]
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"band": band_spec, "x": Piecewise(breaks=[5.0], extrapolation="extend")},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["band", "x"])
    session.select_levels("band", ["B1", "B2"])
    session.replace_with_collapsed_levels("band", method="fit")
    # A special is a free effect off the ordered axis: never a break position.
    assert session_payload(session)["band"]["transform"]["axis"] == BANDS
    session.replace_with_transformed_term(
        "band", form="piecewise", breaks=["B4", "B6"], method="fit"
    )
    session.replace_with_transformed_term("x", form="spline", breaks=[3.0, 6.5], method="fit")
    band, x = session.model._specs["band"], session.model._specs["x"]
    assert band._base_level == "B4"
    assert band._specials == ["MISSING"]
    assert band._grouping.original_to_group["B1"] == "B1+B2"
    assert x.extrapolation == "extend"


def test_transform_carries_a_numeric_sources_boundary_policy_and_pins(banded):
    _, X, y = banded

    def opened(x_spec):
        model = SuperGLM(
            family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features={"x": x_spec}
        )
        # A lambda policy is a REML setting; each refit below inherits fit_reml.
        model.fit_reml(X, y)
        return EditorSession.from_model(model, terms=["x"])

    policy = LambdaPolicy.fixed(2.0)
    session = opened(Spline(n_knots=8, boundary=(-1.0, 11.0), lambda_policy=policy))
    session.replace_with_transformed_term("x", form="spline", breaks=[3.0, 6.5])
    knotted = session.model._specs["x"]
    assert (knotted._explicit_boundary, knotted._lambda_policy) == ((-1.0, 11.0), policy)

    session = opened(Piecewise(breaks=[5.0], lower=-1.0, upper=11.0, base=5.0))
    session.replace_with_transformed_term("x", form="piecewise", breaks=[3.0, 5.0, 6.5])
    moved = session.model._specs["x"]
    assert (moved.lower, moved.upper, moved.base) == (-1.0, 11.0, 5.0)
    # With no knot left at 5.0 the pinned base has nothing to name: the default applies.
    session.replace_with_transformed_term("x", form="piecewise", breaks=[3.0, 6.5])
    assert session.model._specs["x"].base == "most_exposed"


INVALID_REQUESTS = [
    ("x", dict(form="piecewise", breaks=[11.0]), "inside the fitted range"),
    ("x", dict(form="piecewise", breaks=[5.0, 5.0]), "strictly increasing"),
    ("x", dict(form="piecewise", breaks=[6.0, 3.0]), "strictly increasing"),
    ("x", dict(form="piecewise", breaks=[]), "Add at least one break"),
    ("x", dict(form="piecewise", breaks=[5.0], degrees=[2, 1]), "straight lines"),
    ("x", dict(form="polynomial", breaks=[5.0], degree=2), "has no breaks"),
    ("x", dict(form="polynomial", breaks=[], degree=6), "degree from 1 to 5"),
    ("band", dict(form="piecewise", breaks=["B1"], degrees=[1, 1]), "interior band"),
    ("band", dict(form="piecewise", breaks=["B8"], degrees=[1, 1]), "interior band"),
    ("band", dict(form="piecewise", breaks=["B9"], degrees=[1, 1]), "interior band"),
    (
        "band",
        dict(form="piecewise", breaks=["B6", "B3"], degrees=[1, 1, 1]),
        "strictly increasing",
    ),
    ("band", dict(form="piecewise", breaks=["B3"], degrees=[1]), "one degree per segment"),
    ("band", dict(form="bumps", breaks=["B3"]), "piecewise, spline or polynomial"),
]


@pytest.mark.parametrize(("term", "request_", "message"), INVALID_REQUESTS)
def test_transform_refuses_invalid_requests_without_changing_anything(
    banded, term, request_, message
):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=[term])
    revision = session.model_revision
    with pytest.raises(EditorValueError, match=message):
        session.replace_with_transformed_term(term, method="fit", **request_)
    assert session.model is model
    assert session.model_revision == revision
    assert session.structure_history == []


def test_a_linear_term_has_no_axis_to_place_breaks_on(region_model):
    _, X = region_model
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"region": Categorical(base="first"), "x": Numeric()},
    )
    model.fit(X, region_model[0]._fit_y_ref)
    session = EditorSession.from_model(model, terms=["x"])
    # A linear term is drawn as one point, so the Breaks tool stays off for it.
    assert session_payload(session)["x"]["transform"] is None
    with pytest.raises(EditorTypeError, match="ordered or numeric axis"):
        session.replace_with_transformed_term("x", form="piecewise", breaks=[5.0], method="fit")
    assert session.model is model
    assert session.structure_history == []


@pytest.mark.parametrize("form", ["piecewise", "spline"])
def test_a_group_that_takes_in_a_break_is_refused_in_words(banded, form):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    session.replace_with_transformed_term("band", form=form, breaks=["B3", "B6"], method="fit")
    transformed = session.model
    session.select_levels("band", ["B2", "B3", "B4"])
    with pytest.raises(EditorValueError, match="break at 'B3'"):
        session.replace_with_collapsed_levels("band", method="fit")
    assert session.model is transformed
    assert len(session.structure_history) == 1
    # A group within one segment stays allowed.
    session.select_levels("band", ["B4", "B5"])
    session.replace_with_collapsed_levels("band", method="fit")
    assert len(session.structure_history) == 2


def test_a_shape_the_library_refuses_reaches_the_analyst_as_a_fixed_message(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    with pytest.raises(EditorValueError, match="could not fit this shape"):
        session.replace_with_transformed_term(
            "band", form="piecewise", breaks=["B3", "B6"], degrees=[0, 0, 0], method="fit"
        )


def test_payload_offers_the_current_breaks_for_editing(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band", "x"])
    payload = session_payload(session)
    assert payload["band"]["transform"] == {"axis": BANDS, "piecewise": None}
    assert payload["x"]["transform"] == {"axis": None, "piecewise": None}
    session.replace_with_transformed_term(
        "band", form="piecewise", breaks=["B3", "B6"], degrees=[1, 2, 0], method="fit"
    )
    session.replace_with_transformed_term("x", form="piecewise", breaks=[3.0, 6.5], method="fit")
    payload = session_payload(session)
    assert payload["band"]["transform"]["piecewise"] == {
        "breaks": ["B3", "B6"],
        "degrees": [1, 2, 0],
    }
    assert payload["x"]["transform"]["piecewise"] == {"breaks": [3.0, 6.5], "degrees": [1, 1, 1]}


def test_widget_http_transform_term_returns_transition_envelope(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["x"])
    widget = session.widget()
    try:
        payload = _post_json(
            f"{widget.url}/transform_term", {"term": "x", "form": "piecewise", "breaks": [3.0, 6.5]}
        )
        assert set(payload) == {"state", "summary", "timing"}
        assert payload["timing"]["operation"] == "transform_term"
        assert payload["state"]["structure_history"]["last"]["label"] == (
            "transform x to piecewise (2 breaks)"
        )
    finally:
        widget.close()


def test_widget_http_transform_term_refuses_breaks_that_are_not_names_or_numbers(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["x"])
    widget = session.widget()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(
                f"{widget.url}/transform_term", {"term": "x", "form": "piecewise", "breaks": [True]}
            )
        assert error.value.code == 400
        assert json.loads(error.value.read().decode("utf-8")) == {
            "error": "breaks must be a list of band names or numbers."
        }
        assert session.model is model
        assert session.structure_history == []
    finally:
        widget.close()


def test_revert_returns_the_opened_model_and_clears_every_history(region_model):
    model, X = region_model
    session = EditorSession.from_model(model, terms=["region", "x"])
    session.select_levels("region", ["B", "C"])
    session.replace_with_collapsed_levels("region", method="fit")
    session.select_indices("x", [3, 4])
    session.shift("x", 0.1)
    session.revert_to_reference_model()
    assert session.model is model
    assert session.structure_history == [] and session.history == [] and session.redo_stack == []
    np.testing.assert_array_equal(session.to_model().predict(X), model.predict(X))


def test_widget_http_revert_to_original_returns_transition_envelope(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["region"])
    widget = session.widget()
    try:
        session.select_levels("region", ["B", "C"])
        widget._collapse_levels("region", "fit")
        payload = _post_json(f"{widget.url}/revert_to_original", {})
        assert set(payload) == {"state", "summary", "timing"}
        assert payload["timing"]["operation"] == "revert_to_original"
        assert payload["state"]["structure_history"] == {"depth": 0, "last": None}
        assert session.model is model
    finally:
        widget.close()


def test_restore_walks_a_mixed_two_term_sequence_back_exactly(banded):
    model, X, _ = banded
    territory = np.random.default_rng(20260929).choice(["T1", "T2", "T3", "T4"], len(X))
    X = X.assign(territory=territory)
    y = model._fit_y_ref + 0.05 * (territory == "T3")
    two_term = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", k=5), base="first"),
            "territory": Categorical(base="first"),
        },
    )
    two_term.fit(X, y)
    session = EditorSession.from_model(two_term, terms=["band", "territory"])
    before_each_step = [session.model.predict(X)]
    session.select_levels("territory", ["T1", "T2"])
    session.replace_with_collapsed_levels("territory", method="fit")
    before_each_step.append(session.model.predict(X))
    session.replace_with_transformed_term(
        "band", form="piecewise", breaks=["B3", "B6"], degrees=[1, 1, 1], method="fit"
    )
    before_each_step.append(session.model.predict(X))
    session.replace_with_reference_level("territory", "T3", method="fit")
    before_each_step.append(session.model.predict(X))
    session.select_levels("territory", ["T1", "T2"])
    session.replace_with_ungrouped_levels("territory", method="fit")
    for expected in reversed(before_each_step):
        session.uncollapse_levels()
        np.testing.assert_array_equal(session.model.predict(X), expected)
    assert not session.can_uncollapse_levels()


def test_shape_note_reaches_every_renderer_and_survives_export(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band", "x"])
    session.replace_with_transformed_term(
        "band", form="piecewise", breaks=["B3", "B6"], degrees=[1, 2, 0], method="fit"
    )
    sentence = (
        "Shaped ranges for band were chosen in the editor from this data. "
        "Tests are conditional on them; judge them on validation deviance."
    )
    assert sentence in str(session.model.summary())
    assert sentence in session.model.summary()._repr_html_()
    assert any(sentence in note for note in build_summary_export_payload(session.model).notes)
    widget = EditorWidget(session)
    try:
        assert sentence in summary_payload(widget, "in_force")["note"]
    finally:
        widget.close()
    # A later step on another term keeps the note: the mark lives on the basis.
    session.replace_with_transformed_term("x", form="polynomial", breaks=[], degree=2, method="fit")
    edited = session.to_model()
    assert edited.summary()._info["editor_shape_terms"] == ["band"]
    buffer = io.BytesIO()
    joblib.dump(edited, buffer)
    buffer.seek(0)
    assert joblib.load(buffer).summary()._info["editor_shape_terms"] == ["band"]


@pytest.mark.parametrize(
    "step",
    [
        lambda s: s.replace_with_transformed_term(
            "band", form="polynomial", breaks=[], degree=2, method="fit"
        ),
        lambda s: s.replace_with_reference_level("band", "B3", method="fit"),
    ],
    ids=["polynomial", "set_reference"],
)
def test_no_shape_note_without_an_editor_chosen_shape(banded, step):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    step(session)
    assert "editor_shape_terms" not in session.model.summary()._info
