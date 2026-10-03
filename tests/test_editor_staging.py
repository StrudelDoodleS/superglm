"""Editor staging: waiting structural changes, one Refit, carried edits, history ids and notes."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor import session as session_module
from superglm.editor import staging as staging_module
from superglm.editor._types import new_step_id
from superglm.editor.collapse import (
    clone_with_replaced_features,
    collapsed_feature_spec,
    reference_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.payloads import undo_redo_payload
from superglm.editor.refit import fit_refit_model
from superglm.editor.shapes import shaped_feature_spec
from superglm.editor.terms import native_log_effect_values

BRANDS = ["B1", "B2", "B10", "B11", "B12"]


@pytest.fixture
def book():
    """A small motor book: a brand factor, an area factor and a driver-age spline."""
    rng = np.random.default_rng(20261003)
    n = 900
    brand = rng.choice(BRANDS, n, p=[0.3, 0.25, 0.15, 0.15, 0.15])
    area = rng.choice(["A", "B", "C", "D"], n)
    age = rng.uniform(18.0, 80.0, n)
    effects = dict(zip(BRANDS, [0.0, 0.1, 0.25, 0.22, -0.1], strict=True))
    y = (
        0.5
        + np.array([effects[b] for b in brand])
        + 0.1 * (area == "C")
        + 0.2 * np.sin(age / 15.0)
        + rng.normal(0.0, 0.05, n)
    )
    X = pd.DataFrame({"brand": brand, "area": area, "age": age})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "brand": Categorical(base="first"),
            "area": Categorical(base="first"),
            "age": Spline(n_knots=6),
        },
    )
    model.fit(X, y)
    return model, X, y


def _term(model, name):
    return EditorSession.from_model(model, terms=[name]).terms[name]


def _at(term, *labels):
    return np.array([term.levels.index(label) for label in labels], dtype=np.intp)


def _session(model, centering="native"):
    return EditorSession.from_model(model, terms=["brand", "area", "age"], centering=centering)


def _count_fits(monkeypatch) -> list[object]:
    """Every refit the session fits, recorded; each still fits."""
    fits: list[object] = []
    fit = session_module.fit_refit_model

    def counted(*args, **kwargs):
        fits.append(args[1])
        return fit(*args, **kwargs)

    monkeypatch.setattr(session_module, "fit_refit_model", counted)
    return fits


def test_a_reference_can_name_the_group_a_waiting_collapse_made(book):
    model, X, _ = book
    brand = _term(model, "brand")
    collapsed, _ = collapsed_feature_spec(model, brand, _at(brand, "B10", "B11"), X=X)
    pinned, step = reference_feature_spec(model, brand, "B10+B11", X=X, draft_spec=collapsed)
    assert (pinned.base, step["level"]) == ("B10+B11", "B10+B11")
    assert pinned._grouping.group_to_originals["B10+B11"] == ["B10", "B11"]


def test_keep_reference_keeps_the_waiting_reference_not_the_fitted_one(book):
    model, X, _ = book
    brand = _term(model, "brand")
    assert model._specs["brand"]._base_level == "B1"
    pinned, _ = reference_feature_spec(model, brand, "B2", X=X)
    collapsed, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B10", "B11"), X=X, draft_spec=pinned, keep_reference=True
    )
    assert collapsed.base == "B2"
    again, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B1", "B12"), X=X, draft_spec=collapsed, keep_reference=True
    )
    assert again.base == "B2"
    ungrouped, _ = ungrouped_feature_spec(
        model, brand, _at(brand, "B10"), X=X, draft_spec=again, keep_reference=True
    )
    assert ungrouped.base == "B2"


def test_keep_reference_after_a_waiting_change_that_let_the_policy_choose(book):
    model, X, _ = book
    brand = _term(model, "brand")
    loose, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B10", "B11"), X=X, keep_reference=False
    )
    assert loose.base == "first"
    kept, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B2", "B12"), X=X, draft_spec=loose, keep_reference=True
    )
    # The draft still has B1, the reference in force, so this step pins it.
    assert kept.base == "B1"
    # A waiting change that let the policy choose took B1 into a group, so the
    # reference in force is gone from the draft and the draft's policy stands.
    merged, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B1", "B12"), X=X, keep_reference=False
    )
    after, _ = collapsed_feature_spec(
        model, brand, _at(brand, "B10", "B11"), X=X, draft_spec=merged, keep_reference=True
    )
    assert after.base == "first"


def test_collapse_and_ungroup_keep_the_level_universe_and_the_unseen_policy():
    rng = np.random.default_rng(20261004)
    area = rng.choice(["A", "B", "C", "D"], 400)
    y = 0.5 + 0.1 * (area == "B") + rng.normal(0.0, 0.05, 400)
    X = pd.DataFrame({"area": area})
    declared = ["A", "B", "C", "D", "E"]
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"area": Categorical(base="first", levels=declared, unseen="base")},
    )
    with pytest.warns(UserWarning, match="pinned to base"):
        model.fit(X, y)
    term = _term(model, "area")
    collapsed, _ = collapsed_feature_spec(model, term, _at(term, "B", "C"), X=X)
    ungrouped, _ = ungrouped_feature_spec(
        model, term, _at(term, "B", "C"), X=X, draft_spec=collapsed
    )
    for spec in (collapsed, ungrouped):
        assert (spec._declared_levels, spec.unseen) == (declared, "base")


def test_ungrouping_to_no_groups_gives_an_integer_reference_its_native_type():
    rng = np.random.default_rng(20261005)
    code = rng.choice([1, 2, 3, 10], 400)
    y = 0.5 + 0.1 * (code == 2) - 0.1 * (code == 10) + rng.normal(0.0, 0.05, 400)
    X = pd.DataFrame({"code": code})
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"code": Categorical(base=3)}
    )
    model.fit(X, y)
    term = _term(model, "code")
    collapsed, _ = collapsed_feature_spec(model, term, _at(term, "1", "2"), X=X)
    # A grouped design speaks the grouping's labels, which are text.
    assert collapsed.base == "3"
    ungrouped, _ = ungrouped_feature_spec(
        model, term, _at(term, "1", "2"), X=X, draft_spec=collapsed
    )
    assert ungrouped._grouping is None
    assert type(ungrouped.base) is int and ungrouped.base == 3
    refit = clone_with_replaced_features(model, {"code": ungrouped})
    fit_refit_model(model, refit, method="fit", X=X, y=y)
    assert refit._specs["code"]._base_level == 3
    # The draft itself is never fitted: the clone fitted a copy.
    assert ungrouped._levels == []


def test_two_waiting_shapes_on_one_spline_compose_on_the_fitted_knots(book):
    model, X, _ = book
    fitted = model._specs["age"]
    first, _ = shaped_feature_spec(model, "age", lo=30.0, hi=45.0, degree=1, X=X)
    second, step = shaped_feature_spec(
        model, "age", lo=60.0, hi=70.0, degree=0, X=X, draft_spec=first
    )
    assert [(r.lo, r.hi, r.degree) for r in second.polynomial_ranges] == [
        (30.0, 45.0, 1),
        (60.0, 70.0, 0),
    ]
    np.testing.assert_array_equal(second._explicit_knots, fitted.fitted_base_knots)
    assert second._explicit_boundary == fitted.fitted_boundary
    assert step["label"] == "Flat 60–70 in age"
    with pytest.raises(EditorValueError, match="^This range overlaps the Line range 30–45."):
        shaped_feature_spec(model, "age", lo=40.0, hi=50.0, degree=0, X=X, draft_spec=first)


def test_step_ids_are_seven_hex_digits_unique_in_the_process():
    ids = [new_step_id() for _ in range(10_000)]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[0-9a-f]{7}", step_id) for step_id in ids)


def test_the_session_stages_through_the_staging_module(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    calls: list[tuple] = []
    monkeypatch.setattr(
        staging_module, "stage_structural", lambda *args, **kwargs: calls.append(args)
    )
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    assert calls == [(session, "collapse", "brand", {"levels": ["B10", "B11"]})]


def test_staging_waits_without_fitting_or_moving_the_model_revision(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    revision, epoch = session.model_revision, session.edit_epoch
    curves = {name: term.edited_log_effect.copy() for name, term in session.terms.items()}

    staged = session.stage_structural("collapse", "brand", {"levels": ["B11", "B10"]})

    assert fits == [] and session.model is model
    assert (session.model_revision, session.edit_epoch) == (revision, epoch)
    for name, curve in curves.items():
        np.testing.assert_array_equal(session.terms[name].edited_log_effect, curve)
    assert session.pending == [staged]
    assert (staged.operation, staged.term, staged.label) == (
        "collapse",
        "brand",
        "collapse B10 + B11 in brand",
    )
    assert staged.params == {"levels": ["B10", "B11"], "group_label": "B10+B11"}
    assert staged.history_position == 0 and re.fullmatch(r"[0-9a-f]{7}", staged.step_id)
    assert session.draft_spec("brand") is staged.draft_spec
    assert session.draft_spec("area") is model._specs["area"]
    assert undo_redo_payload(session) == {"undo": staged.label, "redo": None}


@pytest.mark.parametrize(
    ("operation", "term", "params", "error", "message"),
    [
        (
            "shape",
            "age",
            {"lo": 33.0, "hi": 33.0, "degree": 1},
            EditorValueError,
            "Select at least two points to shape a range.",
        ),
        (
            "collapse",
            "brand",
            {"levels": ["B10", "B99"]},
            EditorKeyError,
            "Unknown level(s) for term 'brand': ['B99']",
        ),
        ("set_reference", "brand", {}, EditorValueError, "Missing required field: level."),
        ("merge", "brand", {}, EditorValueError, "Unknown structural change: 'merge'"),
    ],
)
def test_a_change_the_builder_refuses_is_refused_at_staging(
    book, operation, term, params, error, message
):
    model, _, _ = book
    session = _session(model)
    with pytest.raises(error) as caught:
        session.stage_structural(operation, term, params)
    assert caught.value.public_message == message
    assert session.pending == []


def test_undo_and_redo_follow_time_across_edits_and_waiting_changes(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    session.select_indices("age", [10, 11])
    session.shift("age", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    age_edit, area_edit = session.history
    assert undo_redo_payload(session) == {"undo": "shift area", "redo": None}

    session.undo()
    assert [record.step_id for record in session.history] == [age_edit.step_id]
    assert session.pending == [staged]
    assert undo_redo_payload(session) == {"undo": staged.label, "redo": "shift area"}
    revision = session.model_revision
    session.undo()
    # A waiting change undone moves nothing: no fit, and no new model revision.
    assert session.pending == [] and session.pending_redo == [staged]
    assert session.model_revision == revision
    session.undo()
    assert session.history == [] and session.edited_terms() == []
    assert undo_redo_payload(session) == {"undo": None, "redo": "shift age"}

    session.redo()
    assert session.edited_terms() == ["age"] and session.pending == []
    revision = session.model_revision
    session.redo()
    assert session.pending == [staged] and session.model_revision == revision
    session.redo()
    assert [record.step_id for record in session.history] == [
        age_edit.step_id,
        area_edit.step_id,
    ]
    assert session.model is model and fits == []


def test_a_new_action_ends_the_future_of_an_undone_waiting_change(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    session.undo()
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    # Staging is a new action: the undone edit's future is gone.
    assert session.redo_stack == []
    session.undo()
    assert session.pending_redo == [staged]
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    assert session.pending_redo == []
    session.redo()
    assert session.pending == []


def test_a_reset_keeps_a_waiting_change_in_its_place_among_the_edits(book):
    model, _, _ = book
    session = _session(model)
    session.select_indices("age", [10])
    session.shift("age", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    session.clear_selection("age")
    session.reset("age")

    [area_edit] = session.history
    assert session.pending[0].step_id == staged.step_id
    assert session.pending[0].history_position == 0
    # The area edit came after the waiting collapse, so Undo takes it first.
    assert session.undo_target() is area_edit
    session.undo()
    assert session.undo_target().step_id == staged.step_id


def test_notes_survive_undo_and_redo_and_travel_in_the_history_records(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("area", ["C"])
    session.shift("area", 0.1)
    [edit] = session.history
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.set_step_note(staged.step_id, "  One dealer network  ")
    session.set_step_note(edit.step_id, "Area C rated by hand")
    session.undo().undo().redo().redo()
    assert session.step_notes == {
        staged.step_id: "One dealer network",
        edit.step_id: "Area C rated by hand",
    }

    session.set_step_note(edit.step_id, "")
    records = session.editor_history_records()
    assert [(r["id"], r["operation"], r["status"], r["note"]) for r in records] == [
        (edit.step_id, "shift", "edit", None),
        (staged.step_id, "collapse", "waiting", "One dealer network"),
    ]
    assert (records[1]["message"], records[1]["term"]) == (staged.label, "brand")
    assert records[1]["predictor"] is None
    assert datetime.fromisoformat(records[0]["time"]).tzinfo == UTC
    with pytest.raises(EditorKeyError) as caught:
        session.set_step_note("not-an-id", "x")
    assert caught.value.public_message == "Unknown history entry."


def test_revert_sets_the_waiting_changes_aside_and_undo_brings_them_back(book):
    model, _, _ = book
    session = _session(model)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.revert_to_reference_model()
    assert session.pending == []
    session.undo()
    assert session.pending == [staged] and session.model is model


def test_re_profiling_waits_until_nothing_is_waiting(book):
    model, _, _ = book
    session = _session(model)
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    with pytest.raises(EditorValueError) as caught:
        session.reprofile_distribution("tweedie_p")
    assert caught.value.public_message == "Refit or undo the waiting changes before re-profiling."


def test_a_term_undo_keeps_a_waiting_change_in_its_place_among_the_edits(book):
    model, _, _ = book
    session = _session(model)
    session.select_indices("age", [10])
    session.shift("age", 0.1)
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    session.undo("age")

    [area_edit] = session.history
    assert session.pending[0].history_position == 0
    # The area edit came after the waiting collapse, so Undo takes it first.
    assert session.undo_target() is area_edit
    # Redo puts the age edit back last, after both.
    session.redo()
    done, _ = session.timeline_items()
    assert [item.step_id for item, _ in done] == [
        staged.step_id,
        area_edit.step_id,
        session.history[-1].step_id,
    ]
    assert session.history[-1].term == "age"


def test_waiting_changes_on_two_terms_refit_in_one_fit_and_one_clone(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    clones: list[list[str]] = []
    clone = session_module.clone_with_replaced_features

    def spied(source, replacements, **kwargs):
        clones.append(sorted(replacements))
        return clone(source, replacements, **kwargs)

    monkeypatch.setattr(session_module, "clone_with_replaced_features", spied)
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.stage_structural("set_reference", "brand", {"level": "B10+B11"})
    session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})

    step = session.refit_pending(method="fit")

    assert len(fits) == 1 and clones == [["age", "brand"]]
    brand = session.model._specs["brand"]
    assert brand._base_level == "B10+B11"
    assert brand._grouping.group_to_originals["B10+B11"] == ["B10", "B11"]
    ranges = session.model._specs["age"].polynomial_ranges
    assert [(r.lo, r.hi, r.degree) for r in ranges] == [(30.0, 45.0, 1)]
    assert (step.operation, step.label, step.term) == ("refit_pending", "Refit · 3 changes", None)
    assert [change.operation for change in step.changes] == ["collapse", "set_reference", "shape"]
    assert session.pending == [] and session.structure_history == [step]


def test_collapse_then_partial_ungroup_keeps_the_reference_in_the_majority_group(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    session.stage_structural("collapse", "brand", {"levels": ["B1", "B10", "B11"]})
    session.stage_structural("ungroup", "brand", {"levels": ["B11"]})
    session.refit_pending(method="fit")
    spec = session.model._specs["brand"]
    assert len(fits) == 1
    assert spec._grouping.group_to_originals["B1+B10"] == ["B1", "B10"]
    assert spec._base_level == "B1+B10"


def test_a_refused_refit_leaves_the_waiting_changes_and_the_model(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    with pytest.raises(EditorValueError) as caught:
        session.refit_pending()
    assert caught.value.public_message == "No changes are waiting for a refit."
    staged = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    revision = session.model_revision

    def refused(*args, **kwargs):
        raise ValueError("the solver refused")

    monkeypatch.setattr(session_module, "fit_refit_model", refused)
    with pytest.raises(EditorValueError) as caught:
        session.refit_pending(method="fit")
    assert caught.value.public_message == (
        "The refit was refused. Undo the last waiting change and try again."
    )
    assert str(caught.value.__cause__) == "the solver refused"
    assert session.pending == [staged] and session.model is model
    assert session.model_revision == revision and session.structure_history == []


def test_undo_of_a_refit_brings_its_changes_back_waiting_and_redo_fits_nothing(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    fits = _count_fits(monkeypatch)
    first = session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    second = session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})
    session.refit_pending(method="fit")
    refit = session.model

    session.undo()
    assert session.model is model
    assert [step.step_id for step in session.pending] == [first.step_id, second.step_id]
    assert undo_redo_payload(session) == {"undo": second.label, "redo": "Refit · 2 changes"}
    session.redo()
    assert session.model is refit and session.pending == []
    assert len(fits) == 1


def test_hand_edits_on_terms_the_refit_left_alone_are_carried_over(book):
    model, _, _ = book
    session = _session(model)
    session.select_indices("age", [20, 21, 22])
    session.shift("age", 0.2)
    session.select_levels("area", ["C"])
    session.shift("area", -0.1)
    session.select_levels("brand", ["B2"])
    session.shift("brand", 0.05)
    held = {name: session.terms[name].edited_log_effect.copy() for name in ("area", "age")}
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})

    refit_step = session.refit_pending(method="fit")
    refit = session.model

    carried = session.structure_history[-1]
    assert session.structure_history == [refit_step, carried]
    assert (carried.operation, carried.label) == (
        "carry_edits",
        "Hand edits carried over: area, age",
    )
    assert session.edited_terms() == ["area", "age"]
    # Shown natively, the carried curve is the edited one, bit for bit.
    for name, curve in held.items():
        np.testing.assert_array_equal(session.terms[name].edited_log_effect, curve)
    # brand was restructured: its edit is dropped here and comes back with the refit's Undo.
    brand = session.terms["brand"]
    np.testing.assert_array_equal(brand.edited_log_effect, brand.original_log_effect)

    session.undo()
    assert session.model is refit and session.edited_terms() == []
    session.undo()
    assert session.model is model and session.edited_terms() == ["brand", "area", "age"]
    assert [step.operation for step in session.pending] == ["collapse"]


def test_a_mean_centred_carry_keeps_the_curve_the_model_scores(book):
    model, _, _ = book
    session = EditorSession.from_model(model, terms=["brand", "age"], centering="mean")
    session.select_indices("age", [20, 21, 22])
    session.shift("age", 0.2)
    before = session.terms["age"]
    held = native_log_effect_values(before)
    shift_before = before.metadata["native_original_log_effect"] - before.original_log_effect
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.refit_pending(method="fit")

    term = session.terms["age"]
    native = np.asarray(term.metadata["native_original_log_effect"])
    # The carried display is held - (native - display): one rounding each for the
    # shift and the subtraction, two more reading the curve back. Each acts on a
    # value no larger than M = |held| + |native| + |display| to first order, so
    # 4uM bounds the difference; 8uM leaves room for the second-order terms.
    u = np.finfo(np.float64).eps / 2
    bound = 8 * u * (np.abs(held) + np.abs(native) + np.abs(term.original_log_effect))
    assert np.all(np.abs(native_log_effect_values(term) - held) <= bound)
    # Copying the displayed values instead would miss by the refit's change in
    # centring constant, which is far outside that bound.
    assert np.max(np.abs((native - term.original_log_effect) - shift_before)) > bound.max()


def test_a_change_refitted_at_once_is_its_own_step_under_its_own_id(book):
    model, _, _ = book
    session = _session(model)
    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_collapsed_levels("brand", method="fit")
    [step] = session.structure_history
    [change] = step.changes
    assert (step.operation, step.label, step.step_id) == (
        "collapse_levels",
        change.label,
        change.step_id,
    )
    assert session.model._editor_step["label"] == "collapse B10 + B11 in brand"
    done, _ = session.timeline_items()
    assert [item for item, _status in done] == [step]
    session.undo()
    assert session.model is model and session.pending == []


def test_the_ungroup_shortcut_waits_for_nothing_else_to_be_waiting(book, monkeypatch):
    model, _, _ = book
    session = _session(model)
    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_collapsed_levels("brand", method="fit")
    staged = session.stage_structural("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1})
    fits = _count_fits(monkeypatch)

    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_ungrouped_levels("brand", method="fit")

    # The fit from before the collapse lacks the waiting shape, so it cannot be reused.
    assert len(fits) == 1 and session.model is not model
    assert session.model._specs["brand"]._grouping is None
    ranges = session.model._specs["age"].polynomial_ranges
    assert [(r.lo, r.hi, r.degree) for r in ranges] == [(30.0, 45.0, 1)]
    assert session.structure_history[-1].label == "Refit · 2 changes"
    # Undo takes the whole call back: the shape waits again, the ungroup is gone.
    session.undo()
    assert [step.step_id for step in session.pending] == [staged.step_id]


@pytest.mark.parametrize(
    ("also", "read", "expected"),
    [
        (
            ("shape", "age", {"lo": 30.0, "hi": 45.0, "degree": 1}),
            lambda fitted: [(r.lo, r.hi, r.degree) for r in fitted._specs["age"].polynomial_ranges],
            [(30.0, 45.0, 1)],
        ),
        (
            ("set_reference", "brand", {"level": "B2"}),
            lambda fitted: fitted._specs["brand"]._base_level,
            "B2",
        ),
    ],
)
def test_the_ungroup_shortcut_does_not_undo_the_rest_of_a_refit(
    book, monkeypatch, also, read, expected
):
    model, _, _ = book
    session = _session(model)
    session.stage_structural("collapse", "brand", {"levels": ["B10", "B11"]})
    session.stage_structural(*also)
    session.refit_pending(method="fit")
    fits = _count_fits(monkeypatch)

    session.select_levels("brand", ["B10", "B11"])
    session.replace_with_ungrouped_levels("brand", method="fit")

    # The fit from before the Refit lacks its other change, so the ungroup refits.
    assert len(fits) == 1 and session.model is not model
    assert session.model._specs["brand"]._grouping is None
    assert read(session.model) == expected
