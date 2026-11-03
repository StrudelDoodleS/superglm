"""Staging, Refit, carried edits and tool availability for SuperLSS models."""

from __future__ import annotations

import copy
import functools
import pickle

import numpy as np
import pytest

from superglm import Categorical, LogNormalLS, SuperLSS, s, ti
from superglm.distributional.revision import EditedCoefficientsWarning
from superglm.distributional.terms import _resolve_term, _term_design
from superglm.editor import EditorSession, edit
from superglm.editor import staging as staging_module
from superglm.editor._types import PendingStep
from superglm.editor.cv import capture_final_fit, final_fit_reason
from superglm.editor.errors import EditorValueError
from superglm.editor.lss import LSSModelView
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.availability import (
    LSS_UNAVAILABLE,
    NO_FIT_SETTINGS,
    OFFSETS_NOT_GIVEN,
    SESSION_TOOLS,
    TERM_TOOLS,
    WEIGHTS_DIFFER,
    WEIGHTS_NOT_GIVEN,
    lss_availability,
    session_refit_reason,
)
from superglm.editor.lss.refit import (
    FIXED_STRENGTHS_COVER_NO_NEW_PENALTY,
    lss_probe_build,
    lss_refit_with_drafts,
)
from superglm.editor.metrics import metrics_payload
from superglm.editor.payloads import session_payload
from tests._lss_editor_fixtures import (
    EPS,
    REGIONS,
    FitCase,
    gamma_case,
    lognormal_case,
    nb_case,
    open_case,
)
from tests.test_editor import editor_frame, editor_model  # noqa: F401
from tests.test_editor_lss_apply import ordered_case

KNOTS = {"count": 5, "strategy": "uniform"}


@pytest.fixture
def replays(monkeypatch):
    """Every ``SuperLSS._replay_fit`` call, recorded and still run."""
    calls: list[tuple[SuperLSS, tuple]] = []
    original = SuperLSS._replay_fit

    def spy(self, *args, **kwargs):
        calls.append((self, args))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SuperLSS, "_replay_fit", spy)
    return calls


@functools.cache
def interaction_case() -> FitCase:
    """The log-normal case with an age by power tensor in the mean only."""
    base = lognormal_case()
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(s("age", k=6), s("power", k=5), ti("age", "power", n_knots=(3, 3))),
        family.scale(s("age", k=5)),
    ).fit_reml(base.X, base.y, outer="efs+newton")
    return FitCase(model, base.X, base.y, None, None)


def _knot_count(view: LSSModelView, key: str) -> int:
    return int(np.asarray(view._specs[key].fitted_base_knots).size)


def test_refit_applies_steps_in_both_parameters_in_one_fit(replays):
    case = lognormal_case()
    session = open_case(case)
    before = session.model
    power_knots = before._specs["mean:power"].fitted_base_knots.copy()
    assert _knot_count(before, "mean:age") != KNOTS["count"]

    knots = session.stage_structural("knots", "mean:age", KNOTS)
    collapse = session.stage_structural("collapse", "scale:region", {"levels": ["R5", "R6"]})
    assert replays == []

    session.refit_pending()

    assert len(replays) == 1
    refit, (X, y, sample_weight, offsets) = replays[0]
    assert X is before.X_train and sample_weight is None and offsets is None
    np.testing.assert_array_equal(y, case.y)
    view = session.model
    assert isinstance(view, LSSModelView) and view.lss is refit and view.lss is not case.model
    assert view.X_train is before.X_train
    assert _knot_count(view, "mean:age") == knots.metadata["count"] == KNOTS["count"]
    grouping = view._specs["scale:region"]._grouping
    assert collapse.metadata["group_label"] in grouping.grouped_levels
    shown = session_payload(session)["scale:region"]["level_groups"]
    assert [(g["label"], g["levels"]) for g in shown] == [
        (collapse.metadata["group_label"], ["R5", "R6"])
    ]
    np.testing.assert_array_equal(view._specs["mean:power"].fitted_base_knots, power_knots)
    assert session.pending == []
    assert case.model._require_fitted() is before._fitted


def test_hand_edits_carry_as_absolute_curves():
    session = open_case(lognormal_case())
    edited = {}
    for key, delta in (("scale:age", 0.3), ("mean:age", -0.2)):
        session.select_indices(key, [20, 21, 22, 23])
        session.shift(key, delta)
        edited[key] = session.terms[key].edited_log_effect.copy()
    # One feature edited in both parameters, taken back and put back across them.
    session.undo()
    session.redo()
    np.testing.assert_array_equal(session.terms["scale:age"].edited_log_effect, edited["scale:age"])
    session.stage_structural("knots", "mean:power", KNOTS)

    session.refit_pending()

    assert sorted(session.edited_terms()) == ["mean:age", "scale:age"]
    for key, curve in edited.items():
        term = session.terms[key]
        np.testing.assert_array_equal(term.edited_log_effect, curve)
        assert np.abs(term.original_log_effect - curve).max() > 0.1 * 0.2
    assert session.structure_history[-1].operation == "carry_edits"


def test_refit_replays_the_fit_settings():
    case = lognormal_case()
    session = open_case(case)
    session.stage_structural("knots", "scale:age", KNOTS)

    session.refit_pending()

    fitted = session.model.lss
    assert fitted._fit_settings["outer"] == "efs+newton" == case.model._fit_settings["outer"]
    assert fitted._fit_settings["method"] == "fit_reml"
    assert fitted._editor_step["method"] == "fit_reml"
    assert fitted.family_ == case.model.family_


@pytest.mark.parametrize(
    ("make", "key", "predictor"),
    [
        (lognormal_case, "scale:age", "scale"),
        (lognormal_case, "mean:region", "mean"),
        (gamma_case, "scale:region", "scale"),
        (nb_case, "theta:age", "theta"),
    ],
)
def test_pending_step_records_its_parameter(make, key, predictor):
    session = open_case(make())
    operation, params = (
        ("collapse", {"levels": ["R5", "R6"]}) if key.endswith("region") else ("knots", KNOTS)
    )

    step = session.stage_structural(operation, key, params)

    assert session.pending[0] is step and step.predictor == predictor


def test_glm_staging_is_unchanged(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model, terms=["region"])

    step = session.stage_structural("collapse", "region", {"levels": ["A", "B"]})

    assert isinstance(step, PendingStep) and step.predictor is None


# -- which operations stage ------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "operation", "params"),
    [
        ("mean:region", "collapse", {"levels": ["R5", "R6"]}),
        ("mean:region", "set_reference", {"level": "R1"}),
        ("mean:age", "knots", KNOTS),
        ("scale:age", "basis", {"kind": "ps"}),
        ("scale:age", "basis", {"select": True}),
    ],
)
def test_supported_operations_stage_and_refit(key, operation, params):
    session = open_case(lognormal_case())
    before = session.model

    step = session.stage_structural(operation, key, params)
    assert session.model is before
    session.refit_pending()

    after = session.model
    assert after is not before and after._specs[key] is not before._specs[key]
    assert set(after._specs) == set(before._specs)
    if operation == "set_reference":
        assert str(after._specs[key]._base_level) == step.metadata["level"]
    if operation == "basis" and "kind" in params:
        assert type(after._specs[key]) is not type(before._specs[key])
    if operation == "basis" and "select" in params:
        assert after._specs[key].select is True
    assert session.pending == []


def test_a_term_in_an_interaction_is_restructured_only_in_its_own_predictor():
    session = open_case(interaction_case())
    assert "mean:age:power" in session.model._interaction_specs

    with pytest.raises(EditorValueError, match=r"used by interaction\(s\): mean:age:power"):
        session.stage_structural("knots", "mean:age", KNOTS)

    # The scale predictor has an age term and no interaction.
    session.stage_structural("knots", "scale:age", KNOTS)
    assert [step.term for step in session.pending] == ["scale:age"]


def test_collapse_then_ungroup_restores_the_levels():
    session = open_case(lognormal_case())
    session.stage_structural("collapse", "mean:region", {"levels": ["R5", "R6"]})
    session.refit_pending()
    assert session.model._specs["mean:region"]._grouping is not None
    assert sorted(session.terms["mean:region"].levels) == sorted(REGIONS)

    session.stage_structural("ungroup", "mean:region", {"levels": ["R5", "R6"]})
    session.refit_pending()

    assert session.model._specs["mean:region"]._grouping is None
    assert sorted(session.terms["mean:region"].levels) == sorted(REGIONS)


def test_a_change_refitted_at_once_is_one_step_that_undo_takes_back(replays):
    session = open_case(lognormal_case())
    before = session.model

    session.replace_with_knots("mean:age", dict(KNOTS))

    assert len(replays) == 1 and session.model is not before
    assert _knot_count(session.model, "mean:age") == KNOTS["count"]
    session.undo()
    assert session.model is before
    session.redo()
    assert len(replays) == 1 and _knot_count(session.model, "mean:age") == KNOTS["count"]


def test_undo_and_redo_of_a_refit_fit_nothing(replays):
    session = open_case(lognormal_case())
    before = session.model
    session.stage_structural("knots", "mean:age", KNOTS)
    session.refit_pending()
    refitted = session.model
    assert len(replays) == 1

    session.undo()
    assert session.model is before and len(session.pending) == 1
    session.redo()

    assert session.model is refitted and len(replays) == 1


def test_staging_builds_the_draft_on_the_refit_rows(monkeypatch, replays):
    session = open_case(lognormal_case())
    built = []
    original = staging_module.lss_probe_build

    def probe(view, key, replacement, X, sample_weight):
        built.append((view, key, X))
        return original(view, key, replacement, X, sample_weight)

    monkeypatch.setattr(staging_module, "lss_probe_build", probe)

    session.stage_structural("basis", "scale:age", {"kind": "bs"})

    assert [(view, key) for view, key, _ in built] == [(session.model, "scale:age")]
    assert built[0][2] is session.model.X_train and replays == []


def test_a_draft_the_fit_would_refuse_is_refused_when_staged(monkeypatch, replays):
    session = open_case(lognormal_case())

    def refuse(*args):
        raise ValueError("the fit's own sentence")

    monkeypatch.setattr(staging_module, "lss_probe_build", refuse)

    with pytest.raises(EditorValueError, match="cannot be fitted on the data the refit reads"):
        session.stage_structural("basis", "scale:age", {"kind": "bs"})

    assert session.pending == [] and replays == []


def test_the_probe_builds_a_term_the_way_the_fit_does():
    case = lognormal_case()
    view = open_case(case).model
    narrow = Categorical(base="R0", levels=["R0", "R1"])

    with pytest.raises(ValueError, match="R2"):
        lss_probe_build(view, "mean:region", narrow, case.X, None)

    weights = np.ones(len(case.X))
    weights[::3] = 0.0
    wide = Categorical(base="R0", levels=list(REGIONS))
    lss_probe_build(view, "mean:region", wide, case.X, weights)


def test_replace_in_force_model_wraps_a_superlss():
    case = lognormal_case()
    session = open_case(case)
    other = gamma_case().model

    session.replace_in_force_model(other)

    assert isinstance(session.model, LSSModelView) and session.model.lss is other
    assert session.model.X_train is session.reference_model.X_train
    assert set(session.terms) == set(session.model._specs)


def test_a_refit_takes_no_options_a_replay_cannot_give():
    session = open_case(lognormal_case())
    session.stage_structural("knots", "mean:age", KNOTS)

    with pytest.raises(EditorValueError, match="replays the fit"):
        session.refit_pending(lambda2=3.0)
    with pytest.raises(EditorValueError, match="replays the fit"):
        session.refit_pending(method="fit")

    assert len(session.pending) == 1


def test_lss_refit_with_drafts_leaves_its_source_alone():
    case = lognormal_case()
    session = open_case(case)
    view = session.model
    coefficients = np.array(view._fitted.coefficients)
    draft = session.stage_structural("knots", "mean:age", KNOTS).draft_spec

    refit = lss_refit_with_drafts(
        view, {"mean:age": draft}, X=case.X, y=case.y, sample_weight=None, offsets=None
    )

    assert isinstance(refit, SuperLSS) and refit is not case.model
    np.testing.assert_array_equal(
        np.asarray(case.model._require_fitted().coefficients), coefficients
    )
    assert refit._require_fitted().coefficients.shape != coefficients.shape or not np.array_equal(
        refit._require_fitted().coefficients, coefficients
    )


@pytest.mark.parametrize("refit", [False, True])
def test_a_structural_change_answers_with_the_calm_summary(refit):
    session = open_case(gamma_case())
    widget = session.widget()
    try:
        response = widget._stage("collapse", "scale:region", {"levels": ["R5", "R6"]}, refit=refit)
        assert response["summary"]["available"] is False
        assert response["state"]["model_revision"] == session.model_revision
        assert len(session.pending) == (0 if refit else 1)
    finally:
        widget.close()


# -- tool availability ------------------------------------------------------------


def _routes():
    """Each tool's call through the session or the widget route, on the log-normal case."""
    return {
        "shape": lambda s, w: w._stage("shape", "mean:age", {"lo": 30.0, "hi": 50.0, "degree": 1}),
        "special": lambda s, w: w._stage("special", "mean:region", {"levels": ["R5"]}),
        "free_levels": lambda s, w: w._free_level_comparison("mean:region"),
        "unsmoothed": lambda s, w: w._unsmoothed_line("mean:age"),
        "unseen": lambda s, w: w._set_unseen("mean:region", "base"),
        "refit_offset": lambda s, w: w._refit_offset(),
        "profile_distribution": lambda s, w: w._start_profile_distribution_job("scale"),
        "rating_table": lambda s, w: w._rating_table("mean:age"),
        "export_structure": lambda s, w: w._export_bytes("structure"),
        "export_xlsx": lambda s, w: w._export_bytes("xlsx"),
    }


def _refused(tool, session, widget):
    with pytest.raises(EditorValueError) as info:
        _routes()[tool](session, widget)
    return info.value.public_message


def _flag(tool, payload, state):
    """The reason a payload gives for ``tool``: on the term for a term tool, else on the state."""
    if tool in TERM_TOOLS:
        return payload["mean:age" if tool in {"shape", "unsmoothed"} else "mean:region"][
            "unavailable"
        ][tool]
    return state["unavailable"][tool]


@pytest.mark.parametrize(
    "tool", ["shape", "special", "free_levels", "unsmoothed", "unseen", "refit_offset"]
)
def test_unavailable_tools_refuse_with_their_reason(tool):
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        state = widget._state()
        assert _flag(tool, state["terms"], state) == LSS_UNAVAILABLE[tool]
        assert _refused(tool, session, widget) == LSS_UNAVAILABLE[tool]
        assert session.pending == [] and session.history == []
    finally:
        widget.close()


@pytest.mark.parametrize(
    "tool", ["profile_distribution", "rating_table", "export_structure", "export_xlsx"]
)
def test_the_other_session_tools_refuse_with_their_reason(tool):
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        state = widget._state()
        assert tool in SESSION_TOOLS
        assert _flag(tool, state["terms"], state) == LSS_UNAVAILABLE[tool]
        assert _refused(tool, session, widget) == LSS_UNAVAILABLE[tool]
    finally:
        widget.close()


def test_the_session_refuses_what_the_widget_refuses():
    session = open_case(lognormal_case())
    refused = {
        "refit_offset": lambda: session.refit_with_edited_offset(),
        "profile_distribution": lambda: session.reprofile_distribution("scale"),
        "unseen": lambda: session.set_unseen("mean:region", "base"),
        "shape": lambda: session.stage_structural(
            "shape", "mean:age", {"lo": 30.0, "hi": 50.0, "degree": 1}
        ),
        "special": lambda: session.stage_structural("on_curve", "mean:region", {"levels": ["R5"]}),
    }
    for tool, call in refused.items():
        with pytest.raises(EditorValueError) as info:
            call()
        assert info.value.public_message == LSS_UNAVAILABLE[tool], tool


def test_every_reason_is_a_sentence_for_the_analyst():
    assert set(LSS_UNAVAILABLE) >= set(TERM_TOOLS) | set(SESSION_TOOLS)
    assert {"metrics", "save_session", "cv"} <= set(LSS_UNAVAILABLE)
    for reason in LSS_UNAVAILABLE.values():
        assert reason[0].isupper() and reason.endswith(".")


def test_lss_availability_turns_the_term_tools_off_and_keeps_the_rest():
    payload = session_payload(open_case(lognormal_case()))
    term = payload["mean:region"]

    assert term["shape"]["available"] is False
    assert term["shape"]["reason"] == LSS_UNAVAILABLE["shape"]
    assert term["unseen"] is None and term["unsmoothed"] is False
    assert term["unavailable"] == {tool: LSS_UNAVAILABLE[tool] for tool in TERM_TOOLS}
    assert lss_availability(term) == term
    # The tools that stage structural changes stay on.
    assert term["reference"] == {"level": "R0", "policy": "most_exposed"}
    assert payload["mean:age"]["knots"]["available"] is True
    assert payload["mean:age"]["knots"]["reason"] is None
    assert payload["mean:age"]["knots"]["kind"] is not None


def test_glm_state_has_no_unavailable_keys(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model, terms=["region", "x_spline"])
    widget = session.widget()
    try:
        state = widget._state()
        assert "unavailable" not in state
        assert all("unavailable" not in term for term in state["terms"].values())
    finally:
        widget.close()


# -- hand edits after a collapse ----------------------------------------------------


def test_collapse_then_edit_a_categorical():
    case = lognormal_case()
    session = open_case(case)
    step = session.stage_structural("collapse", "mean:region", {"levels": ["R5", "R6"]})
    session.refit_pending()
    view = session.model
    term = session.terms["mean:region"]
    group = step.metadata["group_label"]
    assert list(term.levels) == list(REGIONS)
    # Both members sit at the group's effect; the edit moves them together.
    at = [list(term.levels).index(member) for member in ("R5", "R6")]
    assert term.original_log_effect[at[0]] == term.original_log_effect[at[1]]
    term.edited_log_effect = term.original_log_effect.copy()
    term.edited_log_effect[at] += 0.4

    revised = lss_model_with_edits(view, session.terms)

    fitted = view.lss
    members = case.X["region"].isin(["R5", "R6"]).to_numpy()
    after = revised.predict_link(case.X)["mean"].to_numpy()
    before = fitted.predict_link(case.X)["mean"].to_numpy()
    bound = 64 * EPS * np.abs(after).max()
    np.testing.assert_allclose(after[members] - before[members], 0.4, rtol=0.0, atol=bound)
    np.testing.assert_allclose(after[~members], before[~members], rtol=0.0, atol=bound)
    # The group is one fitted level, so both members moved by the same amount.
    assert view._specs["mean:region"]._grouping.group_to_originals[group] == ["R5", "R6"]


def test_collapse_then_edit_an_ordered_term():
    """A grouped ordered term is read at its levels through its grouping."""
    case = ordered_case()
    session = open_case(case)
    step = session.stage_structural("collapse", "scale:region", {"levels": ["R2", "R3"]})
    session.refit_pending()
    view = session.model
    key, parameter = "scale:region", "scale"
    assert step.metadata["group_label"] in view._specs[key]._grouping.grouped_levels
    assert list(session.terms[key].levels) == list(REGIONS)

    # An exactly representable edit: the same fit with its coefficients moved.
    _, state, term_slice = _resolve_term(view._fitted, parameter, "region")
    moved = np.array(view._fitted.coefficients, dtype=np.float64)
    moved[term_slice] += np.random.default_rng(20261103).normal(
        0.0, 0.05, term_slice.stop - term_slice.start
    )
    target = view.lss._with_revised_coefficients(moved, edited_terms=[key], X_train=view.X_train)
    with pytest.warns(EditedCoefficientsWarning):
        reopened = edit(target, train_data=(case.X, case.y, None, None), terms=[key])
    wanted = reopened.terms[key].original_log_effect
    term = session.terms[key]
    term.edited_log_effect = wanted

    revised = lss_model_with_edits(view, session.terms)

    design = _term_design(view._fitted, case.X, state, term_slice)
    levels, counts = np.unique(design, axis=0, return_counts=True)
    cond = np.linalg.cond(np.sqrt(counts)[:, None] * levels)
    bound = 1e3 * EPS * cond * np.abs(wanted).max()
    got = revised.predict_link(case.X)[parameter].to_numpy()
    expected = target.predict_link(case.X)[parameter].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=bound)
    assert np.abs(got - view.lss.predict_link(case.X)[parameter].to_numpy()).max() > 10 * bound


def _without_fit_settings(model: SuperLSS) -> SuperLSS:
    """``model`` as a pickle from before fit settings were recorded loads: no instance entry."""
    old = copy.copy(model)
    del old.__dict__["_fit_settings"]
    restored = pickle.loads(pickle.dumps(old))
    assert "_fit_settings" not in restored.__dict__ and restored._fit_settings is None
    return restored


def _refit_and_final_fit_refuse(session, reason):
    widget = session.widget()
    try:
        state = widget._state()
        assert state["unavailable"]["refit"] == reason
        assert state["unavailable"]["final_fit"] == reason
        assert state["final_fit_plan"] == {"available": False, "reason": reason}
        assert final_fit_reason(session) == reason
        with pytest.raises(EditorValueError) as info:
            capture_final_fit(session)
        assert info.value.public_message == reason
        session.stage_structural("knots", "mean:age", KNOTS)
        with pytest.raises(EditorValueError) as info:
            session.refit_pending()
        assert info.value.public_message == reason
        assert len(session.pending) == 1
    finally:
        widget.close()


def test_a_model_without_recorded_fit_settings_opens_but_cannot_refit(replays):
    case = lognormal_case()
    old = FitCase(_without_fit_settings(case.model), case.X, case.y, None, None)
    session = open_case(old)
    assert (
        session_payload(session)["scale:age"]["y"]
        == session_payload(open_case(case))["scale:age"]["y"]
    )
    _refit_and_final_fit_refuse(session, NO_FIT_SETTINGS)
    assert replays == []


def test_a_model_fitted_with_offsets_cannot_refit_without_them(replays):
    case = nb_case()
    session = edit(case.model, train_data=(case.X, case.y))
    _refit_and_final_fit_refuse(
        session, OFFSETS_NOT_GIVEN.format(names="'mean'", them="it", first="mean")
    )
    assert replays == []
    with_offsets = open_case(case).widget()
    try:
        assert "refit" not in with_offsets._state()["unavailable"]
    finally:
        with_offsets.close()


@functools.cache
def weighted_case() -> FitCase:
    """The gamma case refitted with row weights and the case's smoothing strengths."""
    base = gamma_case()
    weights = np.random.default_rng(20261106).uniform(0.5, 2.0, base.y.size)
    model = base.model._clone_with_replaced_terms({}).fit(
        base.X, base.y, sample_weight=weights, lambdas=dict(base.model.smoothing_parameters_)
    )
    return FitCase(model, base.X, base.y, weights, None)


def test_a_model_fitted_with_weights_cannot_refit_without_them(replays):
    case = weighted_case()
    session = edit(case.model, train_data=(case.X, case.y))
    _refit_and_final_fit_refuse(session, WEIGHTS_NOT_GIVEN)
    assert replays == []
    weighted = open_case(case).widget()
    try:
        assert "refit" not in weighted._state()["unavailable"]
    finally:
        weighted.close()


def test_the_opened_spread_curve_keeps_the_opened_models_level_after_a_refit():
    session = open_case(lognormal_case())
    opened = session.model
    before = session_payload(session)["scale:age"]["original_y"]
    session.stage_structural("collapse", "scale:region", {"levels": ["R5", "R6"]})
    session.refit_pending()
    # The refit moves the spread's level at the reference row ...
    moved = session.model.reference_eta("scale:age") - opened.reference_eta("scale:age")
    assert abs(moved) > 1e-6
    # ... and the opened curve is still the CV the opened model gives.
    payload = session_payload(session)["scale:age"]
    np.testing.assert_allclose(payload["original_y"], before, rtol=64 * EPS, atol=0.0)
    # The impact reads the curve in force against the opened one.
    ratio = np.asarray(payload["y"]) / np.asarray(payload["original_y"])
    np.testing.assert_allclose(
        payload["impact"]["weighted_mean_relativity"],
        np.average(ratio, weights=payload["weights"]),
        rtol=64 * EPS,
        atol=0.0,
    )


def test_the_opened_model_is_swept_once_after_a_refit(monkeypatch):
    session = open_case(lognormal_case())
    session.stage_structural("collapse", "scale:region", {"levels": ["R5", "R6"]})
    session.refit_pending()
    session_payload(session)
    swept = []
    sweep = SuperLSS.term_inference

    def counted(self, *args, **kwargs):
        swept.append(1)
        return sweep(self, *args, **kwargs)

    monkeypatch.setattr(SuperLSS, "term_inference", counted)
    session_payload(session)
    assert swept == []


def test_a_change_of_penalties_is_refused_when_the_fit_fixed_every_strength(replays):
    session = open_case(weighted_case())
    message = FIXED_STRENGTHS_COVER_NO_NEW_PENALTY.format(term="mean:age")
    with pytest.raises(EditorValueError) as info:
        session.stage_structural("basis", "mean:age", {"select": True})
    assert info.value.public_message == message
    assert session.pending == []
    # A change that keeps the term's penalties stages and refits on the fixed strengths.
    session.stage_structural("knots", "mean:age", KNOTS)
    session.refit_pending()
    assert len(replays) == 1


def test_switching_shrinkage_off_is_refused_when_the_fit_fixed_every_strength():
    base = gamma_case()
    lambdas = {**base.model.smoothing_parameters_, "mean:age#null": 2.0}
    shrunk = base.model._clone_with_replaced_terms({"mean:age": s("age", k=8, select=True)})
    session = open_case(
        FitCase(shrunk.fit(base.X, base.y, lambdas=lambdas), base.X, base.y, None, None)
    )
    with pytest.raises(EditorValueError) as info:
        session.stage_structural("basis", "mean:age", {"select": False})
    assert info.value.public_message == FIXED_STRENGTHS_COVER_NO_NEW_PENALTY.format(term="mean:age")
    assert session.pending == []


def test_the_state_reads_the_training_weights_once_per_model(monkeypatch):
    import superglm.editor.lss.availability as availability

    session = open_case(weighted_case())
    calls = []
    resolve = availability.resolve_likelihood_weights

    def counted(*args, **kwargs):
        calls.append(1)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(availability, "resolve_likelihood_weights", counted)
    widget = session.widget()
    try:
        widget._state()
        widget._state()
    finally:
        widget.close()
    assert len(calls) == 1


def test_a_freshly_opened_model_is_the_model_in_force(monkeypatch):
    session = open_case(gamma_case())
    swept = []
    sweep = LSSModelView.term_inference

    def counted(self, *args, **kwargs):
        swept.append(1)
        return sweep(self, *args, **kwargs)

    monkeypatch.setattr(LSSModelView, "term_inference", counted)
    session_payload(session)
    assert swept == []
    widget = session.widget()
    try:
        assert widget._state()["in_force_is_original"] is True
    finally:
        widget.close()


@pytest.mark.parametrize("retain_rows", [True, False])
def test_a_model_opened_on_the_rows_its_fit_kept_can_refit(retain_rows):
    base = gamma_case()
    weights = np.random.default_rng(20261103).uniform(0.5, 2.0, base.y.size)
    weights[::30] = 0.0
    model = base.model._clone_with_replaced_terms({}).fit(
        base.X,
        base.y,
        sample_weight=weights,
        lambdas=dict(base.model.smoothing_parameters_),
        retain_rows=retain_rows,
    )
    kept = weights > 0.0
    given = edit(model, train_data=(base.X, base.y, weights))
    retained = edit(model, train_data=(base.X[kept], base.y[kept], weights[kept]))
    assert session_refit_reason(given) is None
    assert session_refit_reason(retained) is None
    doubled = edit(model, train_data=(base.X[kept], base.y[kept], 2.0 * weights[kept]))
    assert session_refit_reason(doubled) == WEIGHTS_DIFFER
    # The fit's weights on other rows are other weights.
    moved = edit(model, train_data=(base.X, base.y, weights[::-1]))
    assert session_refit_reason(moved) == WEIGHTS_DIFFER
    if retain_rows:
        moved = edit(model, train_data=(base.X[kept], base.y[kept], weights[kept][::-1]))
        assert session_refit_reason(moved) == WEIGHTS_DIFFER


def test_a_model_opened_with_other_weights_cannot_refit(replays):
    case = weighted_case()
    session = edit(case.model, train_data=(case.X, case.y, 2.0 * case.sample_weight))
    _refit_and_final_fit_refuse(session, WEIGHTS_DIFFER)
    assert replays == []


def test_evidence_routes_answer_calmly_for_tools_a_superlss_does_not_offer():
    session = open_case(lognormal_case())
    widget = session.widget()
    try:
        state = widget._state()
        revision = session.model_revision
        summary = widget._summary("in_force", request_sequence=7)
        assert summary == {
            "available": False,
            "model_revision": revision,
            "request_sequence": 7,
            "source": "in_force",
            "level_display": "expanded",
            "error": LSS_UNAVAILABLE["summary"],
        }
        metrics = widget._metrics("deviance", request_sequence=8)
        assert metrics == {
            "available": False,
            "model_revision": revision,
            "request_sequence": 8,
            "metric": "deviance",
            "source": "in_force",
            "error": LSS_UNAVAILABLE["metrics"],
        }
        assert state["unavailable"]["summary"] == summary["error"]
        assert state["unavailable"]["metrics"] == metrics["error"]
        assert metrics_payload(session, "deviance")["error"] == LSS_UNAVAILABLE["metrics"]
    finally:
        widget.close()


@pytest.mark.parametrize("call", ["edited_offset", "edited_offset_factor"])
def test_the_edited_offset_is_refused_with_the_tools_reason(call):
    session = open_case(lognormal_case())
    with pytest.raises(EditorValueError) as info:
        getattr(session, call)()
    assert info.value.public_message == LSS_UNAVAILABLE["refit_offset"]
