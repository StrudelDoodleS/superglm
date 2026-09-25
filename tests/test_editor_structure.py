"""Editor structural steps: one stack, set reference, shaped ranges, revert."""

from __future__ import annotations

import io
import json
import pickle
import re
import urllib.error

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    Constraint,
    Numeric,
    OrderedCategorical,
    Piecewise,
    Polynomial,
    Spline,
    SuperGLM,
)
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.editor.payloads import session_payload
from superglm.editor.session import _SHAPE_REFUSED, _STRETCH_REFUSED
from superglm.editor.shapes import EDITOR_CHOSEN_SHAPE_ATTRIBUTE, _numeric_edges, snap_edge
from superglm.editor.summaries import summary_payload
from superglm.editor.widget import EditorWidget
from superglm.export.summary import build_summary_export_payload
from superglm.features.spline import _SplineBase
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
    session.replace_with_shaped_range("x", lo=2.0, hi=4.0, degree=1, method="fit")
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


EPS = np.finfo(np.float64).eps


@pytest.fixture
def aged():
    # Whole-year ages: a snapped edge on the 0.1 grid of the 72-year span is
    # exact, and adjacent ages bound an interval holding exactly two values.
    rng = np.random.default_rng(20260930)
    n = 3000
    age = rng.integers(18, 91, n)
    region = rng.choice(["N", "S", "E"], n)
    y = rng.poisson(np.exp(-2.0 + 0.3 * np.sin(age / 12.0) + 0.1 * (region == "S")))
    X = pd.DataFrame({"age": age, "region": region})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=1.0,
        features={"age": Spline(kind="bs", k=12), "region": Categorical(base="first")},
    )
    model.fit(X, y)
    return model, X


def _ranges(spec):
    return [(r.lo, r.hi, r.degree) for r in spec.polynomial_ranges]


def _two_value_range(X, name):
    """Adjacent distinct training values: a range holding exactly two of them."""
    values = np.unique(X[name])
    return float(values[10]), float(values[11])


def _line_residual(x, effect):
    """Largest distance of ``effect`` from its best straight line over ``x``."""
    lo, hi = x.min(), x.max()
    vander = np.polynomial.legendre.legvander((2.0 * x - lo - hi) / (hi - lo), 1)
    fit = vander @ np.linalg.lstsq(vander, effect, rcond=None)[0]
    return float(np.max(np.abs(effect - fit)))


def _pinning_tolerance(model, name, spec, effect):
    """Round-off bound on a fitted term's distance from its pinned polynomial.

    The term is B(x) @ c with c = R_inv @ beta. Each null-space member is
    within n_basis * (degree + 1) eps of its polynomial and the weights total
    at most sqrt(n_cols) * ||c||; the editor's centring shift adds the
    round-off of one subtraction per value.
    """
    beta = np.concatenate(
        [model.result.beta[g.sl] for g in model._groups if g.feature_name == name]
    )
    coefficients = spec._R_inv @ beta
    members = np.sqrt(beta.size) * spec._n_basis * (spec.degree + 1) * EPS
    return members * np.linalg.norm(coefficients) + 4 * EPS * np.max(np.abs(effect))


def test_line_range_pins_the_curve_and_restore_undoes_it(aged):
    model, X = aged
    session = EditorSession.from_model(model, terms=["age", "region"])
    before = session.model.predict(X)
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1, method="fit")
    spec = session.model._specs["age"]
    assert _ranges(spec) == [(30.0, 45.0, 1)]
    step = session.structure_history[-1]
    assert (step.operation, step.term, step.label) == ("shape_range", "age", "Line 30–45 in age")
    # The curve the editor draws is the pinned line on the range.
    term = session.terms["age"]
    inside = (term.x >= 30.0) & (term.x <= 45.0)
    effect = term.edited_log_effect[inside]
    assert _line_residual(term.x[inside], effect) <= _pinning_tolerance(
        session.model, "age", spec, effect
    )
    session.uncollapse_levels()
    np.testing.assert_array_equal(session.model.predict(X), before)


def test_second_shape_keeps_the_free_knots(aged):
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age"])
    placed = model._specs["age"].fitted_knots
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1, method="fit")
    session.replace_with_shaped_range("age", lo=70.0, hi=80.0, degree=0, method="fit")
    spec = session.model._specs["age"]
    np.testing.assert_array_equal(spec.fitted_base_knots, placed)
    assert _ranges(spec) == [(30.0, 45.0, 1), (70.0, 80.0, 0)]
    assert spec.fitted_boundary == model._specs["age"].fitted_boundary


def test_overlapping_range_is_refused_by_name_and_same_range_replaces(aged):
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age"])
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1, method="fit")
    shaped = session.model
    with pytest.raises(
        EditorValueError,
        match="^This range overlaps the Line range 30–45. Restore it or choose a range outside it.$",
    ):
        session.replace_with_shaped_range("age", lo=40.0, hi=50.0, degree=0, method="fit")
    assert session.model is shaped and len(session.structure_history) == 1
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=2, method="fit")
    assert _ranges(session.model._specs["age"]) == [(30.0, 45.0, 2)]
    # Ranges meeting at an edge do not overlap: they join at a kink.
    session.replace_with_shaped_range("age", lo=45.0, hi=60.0, degree=0, method="fit")
    assert _ranges(session.model._specs["age"]) == [(30.0, 45.0, 2), (45.0, 60.0, 0)]


def test_single_point_selection_is_refused_before_the_library(aged):
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age"])
    with pytest.raises(EditorValueError, match="^Select at least two points to shape a range.$"):
        session.replace_with_shaped_range("age", lo=33.0, hi=33.0, degree=1, method="fit")
    assert session.model is model and session.structure_history == []


def test_library_refusal_reaches_the_browser_as_the_fixed_sentence(aged):
    model, X = aged
    session = EditorSession.from_model(model, terms=["age"])
    # Two distinct values cannot carry a quadratic: the library refuses by
    # name, and the editor turns that into its one intentional sentence.
    lo, hi = _two_value_range(X, "age")
    with pytest.raises(EditorValueError) as caught:
        session.replace_with_shaped_range("age", lo=lo, hi=hi, degree=2, method="fit")
    assert str(caught.value) == _SHAPE_REFUSED
    assert "distinct values" in str(caught.value.__cause__)
    assert session.model is model and session.structure_history == []
    # The same two values hold a line.
    session.replace_with_shaped_range("age", lo=lo, hi=hi, degree=1, method="fit")
    assert _ranges(session.model._specs["age"]) == [(lo, hi, 1)]


@pytest.mark.parametrize("discrete", [False, True])
def test_a_range_leaving_too_few_values_beside_it_says_to_widen_it(aged, discrete):
    # A third-order penalty needs two values on the free stretch past a range.
    # Stopping at 89 leaves one: even a Flat is refused, and reaching the end is not.
    model, X = aged
    third = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=1.0,
        discrete=discrete,
        features={"age": Spline(kind="bs", k=12, m=3), "region": Categorical(base="first")},
    )
    third.fit(X, model._fit_y_ref)
    session = EditorSession.from_model(third, terms=["age"])
    with pytest.raises(EditorValueError) as caught:
        session.replace_with_shaped_range("age", lo=30.0, hi=89.0, degree=0, method="fit")
    assert str(caught.value) == _STRETCH_REFUSED
    assert session.model is third and session.structure_history == []
    session.replace_with_shaped_range("age", lo=30.0, hi=90.0, degree=0, method="fit")
    assert _ranges(session.model._specs["age"]) == [(30.0, 90.0, 0)]


def test_a_binned_term_refuses_a_shape_its_bin_centres_cannot_carry():
    # The demo's settings: 512 bins, a 260-point grid and a pile of exposure
    # at the youngest age. The first two grid points snap to [18, 18.3],
    # which holds two bin centres: enough for a Line, not for a Cubic, whose
    # third free direction nothing in the binned fit would set.
    rng = np.random.default_rng(20261001)
    n = 20_000
    age = np.clip(rng.normal(48.0, 13.5, n), 18.0, 80.0)
    exposure = rng.gamma(4.0, 0.25, n) + 0.05
    y = rng.poisson(np.exp(-1.0 + 0.2 * np.sin((age - 18.0) / 6.0)) * exposure) / exposure
    X = pd.DataFrame({"age": age})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(kind="bs", k=14, knot_strategy="quantile_tempered")},
        discrete=True,
        n_bins=512,
    )
    model.fit_reml(X, y, sample_weight=exposure)
    session = EditorSession.from_model(model, terms=["age"], n_points=260)
    grid = session.terms["age"].x
    # The palette's count for a run is the one the library certifies with, so
    # its icons disable exactly the shapes the refit would refuse.
    support = session_payload(session)["age"]["shape"]["support"]
    assert support["through"][1] - support["below"][0] == 2
    assert support["through"][2] - support["below"][0] == 4
    with pytest.raises(EditorValueError) as caught:
        session.replace_with_shaped_range("age", lo=grid[0], hi=grid[1], degree=3, method="fit")
    assert str(caught.value) == _SHAPE_REFUSED
    assert "it has 2 once binned to 512" in str(caught.value.__cause__)
    assert session.model is model and session.structure_history == []
    session.replace_with_shaped_range("age", lo=grid[0], hi=grid[1], degree=1, method="fit")
    assert _ranges(session.model._specs["age"]) == [(18.0, 18.3, 1)]
    session.uncollapse_levels()
    session.replace_with_shaped_range("age", lo=grid[0], hi=grid[2], degree=3, method="fit")
    assert _ranges(session.model._specs["age"]) == [(18.0, 18.5, 3)]


def test_the_palette_counts_the_values_in_the_range_a_refit_builds(aged):
    # A run i..j holds through[j] - below[i] values: the distinct positive-
    # weight ages inside the snapped range the refit is built on. Age 30
    # carries no weight, so it counts for nothing.
    model, X = aged
    y = model._fit_y_ref
    weight = np.where(X["age"] == 30, 0.0, 1.0)
    weighted = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=1.0,
        features={"age": Spline(kind="bs", k=12), "region": Categorical(base="first")},
    )
    weighted.fit(X, y, sample_weight=weight)
    session = EditorSession.from_model(weighted, terms=["age"], train_data=(X, y, weight))
    grid = session.terms["age"].x
    support = session_payload(session)["age"]["shape"]["support"]
    counted = np.unique(X["age"][weight > 0])
    runs = [(i, i + 1) for i in range(grid.size - 1)] + [(0, grid.size - 1), (3, 40)]
    edges = np.array([_numeric_edges(weighted._specs["age"], grid[i], grid[j]) for i, j in runs])
    held = np.searchsorted(counted, edges[:, 1], side="right") - np.searchsorted(
        counted, edges[:, 0]
    )
    i, j = np.array(runs).T
    np.testing.assert_array_equal(
        np.array(support["through"])[j] - np.array(support["below"])[i], held
    )
    # The fixture exercises both: snapping moves an edge across an age, and
    # a run spans the weightless one.
    raw = np.searchsorted(counted, grid[j], side="right") - np.searchsorted(counted, grid[i])
    assert np.any(raw != held)
    assert np.any((edges[:, 0] <= 30) & (edges[:, 1] >= 30))


def test_revert_after_two_shapes_restores_the_opened_model(aged):
    model, X = aged
    session = EditorSession.from_model(model, terms=["age"])
    opened = session.to_model().predict(X)
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1, method="fit")
    session.replace_with_shaped_range("age", lo=70.0, hi=80.0, degree=0, method="fit")
    session.revert_to_reference_model()
    np.testing.assert_array_equal(session.to_model().predict(X), opened)
    assert session.structure_history == []


def test_a_selection_through_the_last_point_ends_on_the_boundary(region_model):
    model, _ = region_model
    session = EditorSession.from_model(model, terms=["x"])
    lo_b, hi_b = model._specs["x"].fitted_boundary
    span = hi_b - lo_b
    # The data ends are off the span's grid, so snapping outward passes them.
    assert snap_edge(lo_b, span, -1) < lo_b and snap_edge(hi_b, span, 1) > hi_b
    session.replace_with_shaped_range("x", lo=lo_b, hi=2.0, degree=1, method="fit")
    session.replace_with_shaped_range("x", lo=8.0, hi=hi_b, degree=0, method="fit")
    spec = session.model._specs["x"]
    # Both edges are the boundary exactly, so neither is inserted as a knot a
    # round-off away from the end.
    assert _ranges(spec) == [(lo_b, 2.0, 1), (8.0, hi_b, 0)]
    knots = spec.fitted_knots
    assert lo_b < knots.min() and knots.max() < hi_b


@pytest.mark.parametrize(
    ("value", "span", "direction", "expected"),
    [
        (30.04, 72.0, -1, 30.0),
        (30.06, 72.0, -1, 30.0),
        (44.96, 72.0, 1, 45.0),
        (44.94, 72.0, 1, 45.0),
        # 3 * 0.1 is 0.30000000000000004: an on-grid edge stays put.
        (0.3, 10.0, -1, 0.3),
        (0.3, 10.0, 1, 0.3),
        (1234.5, 5000.0, -1, 1230.0),
        (1234.5, 5000.0, 1, 1240.0),
    ],
)
def test_snap_edge_rounds_outward_on_three_figures_of_the_span(value, span, direction, expected):
    assert snap_edge(value, span, direction) == expected


def test_a_p_spline_is_shaped_as_a_b_spline_with_its_knots_degree_and_order(region_model):
    model, X = region_model
    source = model._specs["x"]
    session = EditorSession.from_model(model, terms=["x"])
    session.replace_with_shaped_range("x", lo=2.0, hi=4.0, degree=1, method="fit")
    shaped = session.model._specs["x"]
    assert type(shaped).__name__ == "BSplineSmooth" and type(source).__name__ == "PSpline"
    np.testing.assert_array_equal(shaped.fitted_base_knots, source.fitted_knots)
    assert shaped.fitted_boundary == source.fitted_boundary
    assert (shaped.degree, shaped._m_orders) == (source.degree, source._m_orders)
    assert getattr(shaped, EDITOR_CHOSEN_SHAPE_ATTRIBUTE) is True


def test_a_cubic_regression_spline_stays_one(aged):
    _, X = aged
    y = aged[0]._fit_y_ref
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=1.0,
        features={"age": Spline(kind="cr", k=10)},
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["age"])
    session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=1, method="fit")
    assert type(session.model._specs["age"]).__name__ == "CubicRegressionSpline"


UNAVAILABLE = [
    ({"x": Numeric()}, "Shapes need a spline term."),
    ({"x": Piecewise(breaks=[5.0])}, "Shapes need a spline term."),
    ({"x": Polynomial(degree=2)}, "Shapes need a spline term."),
    (
        {"x": Spline(kind="cr_cardinal", n_knots=6)},
        "Shapes are not available for cardinal cubic regression splines.",
    ),
    (
        {"x": Spline(n_knots=6, constraint=Constraint.fit.increasing)},
        "Remove the term's shape constraint to add shaped ranges.",
    ),
    (
        {"x": Spline(n_knots=6, select=True)},
        "Remove select=True from the term to add shaped ranges.",
    ),
]


@pytest.mark.parametrize(("features", "reason"), UNAVAILABLE)
def test_an_unshapeable_term_says_why_and_is_refused_unchanged(region_model, features, reason):
    _, X = region_model
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, spline_penalty=0.1, features=features
    )
    model.fit(X, region_model[0]._fit_y_ref)
    session = EditorSession.from_model(model, terms=["x"])
    assert session_payload(session)["x"]["shape"] == {
        "available": False,
        "reason": reason,
        "ranges": [],
        "support": None,
        "specials": [],
    }
    with pytest.raises(EditorValueError, match=f"^{re.escape(reason)}$"):
        session.replace_with_shaped_range("x", lo=2.0, hi=4.0, degree=1, method="fit")
    assert session.model is model and session.structure_history == []


def test_an_interaction_parent_cannot_be_reshaped(region_model):
    _, X = region_model
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={"region": Categorical(base="first"), "x": Spline(kind="bs", n_knots=6)},
        interactions=[("x", "region")],
    )
    model.fit(X, region_model[0]._fit_y_ref)
    session = EditorSession.from_model(model, terms=["x"])
    reason = "A term used by an interaction cannot be reshaped."
    assert session_payload(session)["x"]["shape"]["reason"] == reason
    with pytest.raises(EditorValueError, match=f"^{reason}$"):
        session.replace_with_shaped_range("x", lo=2.0, hi=4.0, degree=1, method="fit")


def test_categorical_and_ordered_step_terms_report_shapes_unavailable(region_model, banded):
    session = EditorSession.from_model(region_model[0], terms=["region"])
    assert session_payload(session)["region"]["shape"] == {
        "available": False,
        "reason": "Shapes need a spline term.",
        "ranges": [],
        "support": None,
        "specials": [],
    }
    _, X, y = banded
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(order=BANDS, basis=Piecewise(breaks=["B4"]), base="first")
        },
    )
    model.fit(X, y)
    ordered = session_payload(EditorSession.from_model(model, terms=["band"]))["band"]
    assert ordered["shape"]["reason"] == "Shapes need a spline term."


@pytest.mark.parametrize("degree", [-1, 4, 2.9, True])
def test_a_degree_outside_the_four_shapes_is_refused(aged, degree):
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age"])
    with pytest.raises(EditorValueError, match="Flat, Line, Quadratic or Cubic"):
        session.replace_with_shaped_range("age", lo=30.0, hi=45.0, degree=degree, method="fit")


@pytest.mark.parametrize("edge", [float("nan"), float("inf"), "30"])
def test_a_numeric_term_refuses_edges_that_are_not_finite_numbers(aged, edge):
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age"])
    with pytest.raises(EditorValueError, match="must be finite numbers"):
        session.replace_with_shaped_range("age", lo=edge, hi=45.0, degree=1, method="fit")


def test_an_edge_far_past_the_boundary_is_the_boundary(aged):
    # Snapped on the span's grid, +-1e308 would overflow; clipped first, it is the end.
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age"])
    lo_b, hi_b = model._specs["age"].fitted_boundary
    session.replace_with_shaped_range("age", lo=-1e308, hi=30.0, degree=1, method="fit")
    session.replace_with_shaped_range("age", lo=60.0, hi=1e308, degree=0, method="fit")
    assert _ranges(session.model._specs["age"]) == [(lo_b, 30.0, 1), (60.0, hi_b, 0)]


def test_a_special_level_is_refused_as_outside_the_bands(banded):
    _, X, y = banded
    band = OrderedCategorical(
        order=BANDS[:7], specials=["B8"], basis=Spline(kind="bs", k=5), base="first"
    )
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": band})
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["band"])
    assert session_payload(session)["band"]["shape"]["specials"] == ["B8"]
    with pytest.raises(
        EditorValueError,
        match="^A shaped range covers only the bands of band; "
        "leave its special levels out of the selection.$",
    ):
        session.replace_with_shaped_range("band", lo="B6", hi="B8", degree=1, method="fit")
    assert session.model is model and session.structure_history == []


def test_an_ordered_term_pins_whole_bands_and_becomes_a_b_spline(banded):
    model, X, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    placed = model._specs["band"]._basis_spline.fitted_knots
    # Band labels in either order name the same run of bands.
    session.replace_with_shaped_range("band", lo="B6", hi="B3", degree=1, method="fit")
    spec = session.model._specs["band"]
    declared = spec._spline_obj
    assert type(declared).__name__ == "BSplineSmooth"
    assert [(r.lo, r.hi, r.degree) for r in declared.polynomial_ranges] == [("B3", "B6", 1)]
    assert getattr(declared, EDITOR_CHOSEN_SHAPE_ATTRIBUTE) is True
    np.testing.assert_array_equal(spec._basis_spline.fitted_base_knots, placed)
    assert session_payload(session)["band"]["shape"]["ranges"] == [
        {"lo": "B3", "hi": "B6", "degree": 1, "label": "Line"}
    ]
    assert session.structure_history[-1].label == "Line B3–B6 in band"
    # The four pinned bands lie on one line in their axis positions.
    term = session.terms["band"]
    run = [term.levels.index(b) for b in ("B3", "B4", "B5", "B6")]
    positions = np.array([spec._level_to_value[b] for b in ("B3", "B4", "B5", "B6")])
    effect = term.edited_log_effect[run]
    assert _line_residual(positions, effect) <= _pinning_tolerance(
        session.model, "band", spec._basis_spline, effect
    )


def test_ordered_ranges_overlap_by_band_and_collapse_respects_their_edges(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    session.replace_with_shaped_range("band", lo="B3", hi="B6", degree=1, method="fit")
    with pytest.raises(EditorValueError, match="overlaps the Line range B3–B6"):
        session.replace_with_shaped_range("band", lo="B5", hi="B8", degree=0, method="fit")
    shaped = session.model
    # A group taking in an edge band is refused in words, before the library.
    session.select_levels("band", ["B2", "B3"])
    with pytest.raises(EditorValueError, match="shaped-range edge at 'B3'"):
        session.replace_with_collapsed_levels("band", method="fit")
    assert session.model is shaped
    # A group strictly inside the range is allowed, and the range survives it.
    session.select_levels("band", ["B4", "B5"])
    session.replace_with_collapsed_levels("band", method="fit")
    ranges = session.model._specs["band"]._spline_obj.polynomial_ranges
    assert [(r.lo, r.hi) for r in ranges] == [("B3", "B6")]
    # A range cannot end on a collapsed group.
    with pytest.raises(EditorValueError, match="must start and end on single bands of band"):
        session.replace_with_shaped_range("band", lo="B7", hi="B4+B5", degree=0, method="fit")
    # A range added below an existing one is reported in axis order.
    session.replace_with_shaped_range("band", lo="B1", hi="B2", degree=0, method="fit")
    ranges = session_payload(session)["band"]["shape"]["ranges"]
    assert [(r["lo"], r["hi"], r["label"]) for r in ranges] == [
        ("B1", "B2", "Flat"),
        ("B3", "B6", "Line"),
    ]


def test_widget_http_shape_range_returns_transition_envelope(aged):
    model, _ = aged
    session = EditorSession.from_model(model, terms=["age", "region"])
    widget = session.widget()
    try:
        payload = _post_json(
            f"{widget.url}/shape_range", {"term": "age", "lo": 30, "hi": 45.0, "degree": 1}
        )
        assert set(payload) == {"state", "summary", "timing"}
        assert payload["timing"]["operation"] == "shape_range"
        assert payload["state"]["structure_history"]["last"]["label"] == "Line 30–45 in age"
        shape = payload["state"]["terms"]["age"]["shape"]
        support = shape.pop("support")
        assert shape == {
            "available": True,
            "reason": None,
            "ranges": [{"lo": 30.0, "hi": 45.0, "degree": 1, "label": "Line"}],
            "specials": [],
        }
        n_points = payload["state"]["terms"]["age"]["n_points"]
        assert len(support["below"]) == len(support["through"]) == n_points
        assert "transform" not in payload["state"]["terms"]["age"]
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"term": "age", "lo": True, "hi": 45.0, "degree": 1},
            "lo and hi must be band names or numbers.",
        ),
        (
            {"term": "age", "lo": [30], "hi": 45.0, "degree": 1},
            "lo and hi must be band names or numbers.",
        ),
        ({"term": "age", "lo": 30.0, "degree": 1}, "Missing required field: hi."),
        (
            {"term": "age", "lo": 30.0, "hi": 30.0, "degree": 1},
            "Select at least two points to shape a range.",
        ),
        # The degree is never truncated: 2.9 is no shape, and neither is True.
        (
            {"term": "age", "lo": 30.0, "hi": 45.0, "degree": 2.9},
            "Choose a shape: Flat, Line, Quadratic or Cubic.",
        ),
        (
            {"term": "age", "lo": 30.0, "hi": 45.0, "degree": True},
            "Choose a shape: Flat, Line, Quadratic or Cubic.",
        ),
        ({"term": "age", "lo": 30.0, "hi": 45.0, "degree": 2}, None),
    ],
)
def test_widget_http_shape_range_answers_refusals_with_intentional_messages(aged, body, message):
    model, X = aged
    session = EditorSession.from_model(model, terms=["age"])
    widget = session.widget()
    if message is None:
        # A library refusal: the browser gets the fixed sentence, never its text.
        lo, hi = _two_value_range(X, "age")
        body = {**body, "lo": lo, "hi": hi}
        message = _SHAPE_REFUSED
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(f"{widget.url}/shape_range", body)
        assert error.value.code == 400
        assert json.loads(error.value.read().decode("utf-8")) == {"error": message}
        assert session.model is model and session.structure_history == []
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
    session.replace_with_shaped_range("band", lo="B3", hi="B6", degree=1, method="fit")
    before_each_step.append(session.model.predict(X))
    session.replace_with_reference_level("territory", "T3", method="fit")
    before_each_step.append(session.model.predict(X))
    session.select_levels("territory", ["T1", "T2"])
    session.replace_with_ungrouped_levels("territory", method="fit")
    for expected in reversed(before_each_step):
        session.uncollapse_levels()
        np.testing.assert_array_equal(session.model.predict(X), expected)
    assert not session.can_uncollapse_levels()


def test_a_model_pickled_before_shaped_ranges_opens_refits_and_takes_a_shape(banded):
    # Unpickling bypasses __init__, so every spline in a model pickled before
    # shaped ranges lacks the two attributes they added. Deleting them from a
    # live fitted model reproduces that object without a binary fixture; the
    # class-level defaults must read as no ranges, with base knots the fitted.
    model, X, y = banded
    legacy = pickle.loads(pickle.dumps(model))
    specs = legacy._specs.values()
    splines = [
        s for spec in specs for s in (spec, *vars(spec).values()) if isinstance(s, _SplineBase)
    ]
    assert len(splines) == 3  # x, and the ordered term's declared and working bases
    for spline in splines:
        spline.__dict__.pop("_polynomial_ranges")
        spline.__dict__.pop("_base_interior_knots")
    assert vars(_SplineBase)["_polynomial_ranges"] == ()  # immutable: shared by every legacy spec
    x_spec = legacy._specs["x"]
    np.testing.assert_array_equal(x_spec.fitted_base_knots, x_spec.fitted_knots)
    session = EditorSession.from_model(legacy, terms=["band", "x"])
    assert session_payload(session)["x"]["shape"]["ranges"] == []
    session.select_levels("band", ["B2", "B3"])
    session.replace_with_collapsed_levels("band", method="fit")
    session.replace_with_shaped_range("x", lo=2.0, hi=4.0, degree=1, method="fit")
    assert _ranges(session.model._specs["x"]) == [(2.0, 4.0, 1)]
    np.testing.assert_array_equal(session.model._specs["x"].fitted_base_knots, x_spec.fitted_knots)
    legacy.fit(X, y)
    np.testing.assert_array_equal(legacy._specs["x"].fitted_knots, model._specs["x"].fitted_knots)


def test_shape_note_reaches_every_renderer_and_survives_export(banded):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band", "x"])
    session.replace_with_shaped_range("band", lo="B3", hi="B6", degree=2, method="fit")
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
    # A later rebuild of the term keeps the note: the mark lives on the basis.
    session.replace_with_reference_level("band", "B4", method="fit")
    edited = session.to_model()
    assert edited.summary()._info["editor_shape_terms"] == ["band"]
    buffer = io.BytesIO()
    joblib.dump(edited, buffer)
    buffer.seek(0)
    assert joblib.load(buffer).summary()._info["editor_shape_terms"] == ["band"]


@pytest.mark.parametrize(
    "step",
    [
        lambda s: (
            s.select_levels("band", ["B1", "B2"])
            or s.replace_with_collapsed_levels("band", method="fit")
        ),
        lambda s: s.replace_with_reference_level("band", "B3", method="fit"),
    ],
    ids=["collapse", "set_reference"],
)
def test_no_shape_note_without_an_editor_chosen_shape(banded, step):
    model, _, _ = banded
    session = EditorSession.from_model(model, terms=["band"])
    step(session)
    assert "editor_shape_terms" not in session.model.summary()._info
