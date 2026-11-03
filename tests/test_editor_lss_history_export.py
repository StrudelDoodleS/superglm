"""History tagged by parameter, and the joblib export of an edited SuperLSS."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest

from superglm import SuperLSS
from superglm.editor import EditorSession, edit
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.apply import lss_model_with_edits
from superglm.editor.lss.availability import LSS_UNAVAILABLE, SESSION_TOOLS
from superglm.editor.lss.persistence import validate_lss_artifact
from superglm.editor.payloads import pending_payload, timeline_payload
from tests._lss_editor_fixtures import EPS, lognormal_case, nb_case, open_case
from tests.test_editor import editor_frame, editor_model  # noqa: F401

KNOTS = {"count": 5, "strategy": "uniform"}


def _shift(session, key, delta, indices=(20, 21, 22, 23)):
    session.select_indices(key, list(indices))
    session.shift(key, delta)


def _edited_session():
    session = open_case(lognormal_case())
    _shift(session, "mean:age", -0.2)
    _shift(session, "scale:age", 0.3)
    return session


def test_history_rows_carry_their_parameter():
    session = _edited_session()
    session.stage_structural("knots", "mean:power", KNOTS)

    # The timeline is oldest first: the mean edit, then the scale edit, then the waiting change.
    rows = [entry for entry in timeline_payload(session) if entry.get("kind") != "marker"]
    assert [row["predictor"] for row in rows] == ["mean", "scale", "mean"]
    assert [row["predictor_label"] for row in rows] == ["Average", "Spread", "Average"]
    assert [row["term_label"] for row in rows] == ["age", "age", "power"]
    waiting = pending_payload(session)
    assert [(row["predictor"], row["predictor_label"], row["term_label"]) for row in waiting] == [
        ("mean", "Average", "power")
    ]
    records = session.editor_history_records()
    assert [record["predictor"] for record in records] == ["mean", "scale", "mean"]


def test_the_refit_step_carries_its_parameter():
    session = open_case(lognormal_case())
    session.stage_structural("knots", "scale:age", KNOTS)
    session.refit_pending()

    assert session.editor_history_records()[-1]["predictor"] == "scale"
    row = [e for e in timeline_payload(session) if e.get("kind") == "structural"][-1]
    assert (row["predictor"], row["predictor_label"]) == ("scale", "Spread")


def test_undo_redo_across_parameters():
    session = open_case(lognormal_case())
    original = {key: session.terms[key].original_log_effect.copy() for key in session.terms}
    _shift(session, "mean:age", -0.2)
    _shift(session, "scale:age", 0.3)
    edited = {key: session.terms[key].edited_log_effect.copy() for key in ("mean:age", "scale:age")}

    session.undo()
    np.testing.assert_array_equal(
        session.terms["scale:age"].edited_log_effect, original["scale:age"]
    )
    np.testing.assert_array_equal(session.terms["mean:age"].edited_log_effect, edited["mean:age"])
    session.undo()
    for key in ("mean:age", "scale:age"):
        np.testing.assert_array_equal(session.terms[key].edited_log_effect, original[key])
    assert session.editor_history_records() == []

    session.redo()
    np.testing.assert_array_equal(session.terms["mean:age"].edited_log_effect, edited["mean:age"])
    np.testing.assert_array_equal(
        session.terms["scale:age"].edited_log_effect, original["scale:age"]
    )
    session.redo()
    for key, curve in edited.items():
        np.testing.assert_array_equal(session.terms[key].edited_log_effect, curve)
    assert [r["predictor"] for r in session.editor_history_records()] == ["mean", "scale"]


def test_a_hand_edit_in_one_parameter_survives_a_refit_of_the_other():
    session = open_case(lognormal_case())
    _shift(session, "scale:age", 0.3)
    drawn = session.terms["scale:age"].edited_log_effect.copy()
    before = session.terms["mean:age"].original_log_effect.copy()
    knots_before = np.asarray(session.model._specs["mean:age"].fitted_base_knots).size
    session.stage_structural("knots", "mean:age", {"count": 5, "strategy": "uniform"})

    session.refit_pending()

    assert sorted(session.edited_terms()) == ["scale:age"]
    np.testing.assert_array_equal(session.terms["scale:age"].edited_log_effect, drawn)
    assert np.abs(session.terms["scale:age"].original_log_effect - drawn).max() > 0.1 * 0.3
    assert np.asarray(session.model._specs["mean:age"].fitted_base_knots).size != knots_before
    mean = session.terms["mean:age"]
    np.testing.assert_array_equal(mean.edited_log_effect, mean.original_log_effect)
    assert mean.original_log_effect.shape == before.shape
    assert np.abs(mean.original_log_effect - before).max() > 0.0


@pytest.mark.filterwarnings("ignore::superglm.distributional.revision.EditedCoefficientsWarning")
def test_exported_model_round_trips_with_history(tmp_path):
    case = lognormal_case()
    session = open_case(case)
    _shift(session, "mean:age", -0.2)
    _shift(session, "scale:age", 0.3)
    expected = lss_model_with_edits(session.model, session.terms)

    path = session.save_model(tmp_path / "edited.joblib")

    loaded = joblib.load(path)
    assert type(loaded) is SuperLSS
    assert loaded._training_frame is None
    got = loaded.predict_parameters(case.X)
    want = expected.predict_parameters(case.X)
    scale = float(np.abs(want.to_numpy()).max())
    np.testing.assert_allclose(got, want, rtol=0, atol=64 * EPS * scale)
    assert not np.allclose(got, case.model.predict_parameters(case.X), rtol=1e-4, atol=0.0)
    history = loaded._editor_history
    assert [record["predictor"] for record in history] == ["mean", "scale"]
    assert history[-1]["term"] == "scale:age"
    assert history[-1]["predictor"] == "scale"

    reopened = edit(loaded, train_data=(case.X, case.y, case.sample_weight, case.offsets))
    for key in ("mean:age", "scale:age"):
        parameter, term = key.split(":")
        # The reopened curve is the edited model's, the projection of the drawn one.
        effect = expected.term_inference(parameter, term, X_train=case.X).effect
        np.testing.assert_allclose(
            reopened.terms[key].original_log_effect,
            effect,
            rtol=0,
            atol=64 * EPS * (1.0 + np.abs(effect).max()),
        )
        original = session.terms[key].original_log_effect
        assert np.abs(reopened.terms[key].original_log_effect - original).max() > 0.01


def test_an_unedited_export_is_the_fitted_model(tmp_path):
    case = lognormal_case()
    session = open_case(case)

    loaded = joblib.load(session.save_model(tmp_path / "plain.joblib"))

    assert type(loaded) is SuperLSS
    assert loaded._editor_history == []
    np.testing.assert_array_equal(
        loaded.predict_parameters(case.X), case.model.predict_parameters(case.X)
    )


def test_export_with_parameter_offsets_validates_predictions(tmp_path):
    case = nb_case()
    session = open_case(case)
    _shift(session, "theta:age", 0.2)

    loaded = joblib.load(session.save_model(tmp_path / "nb.joblib"))

    want = lss_model_with_edits(session.model, session.terms).predict_parameters(
        case.X, offsets=case.offsets
    )
    got = loaded.predict_parameters(case.X, offsets=case.offsets)
    np.testing.assert_allclose(got, want, rtol=0, atol=64 * EPS * float(np.abs(want).max().max()))


def test_the_artifact_check_refuses_what_a_round_trip_changed():
    case = lognormal_case()
    other = nb_case().model
    buffer = io.BytesIO()
    joblib.dump(case.model, buffer)
    buffer.seek(0)
    validate_lss_artifact(case.model, joblib.load(buffer))
    for loaded in (object(), other, case.model.__class__.__new__(SuperLSS)):
        with pytest.raises(ValueError, match="artifact validation failed"):
            validate_lss_artifact(case.model, loaded)


def test_session_save_is_refused_for_superlss(tmp_path):
    session = _edited_session()
    path = tmp_path / "session.json"

    with pytest.raises(EditorValueError) as info:
        session.save(path)

    assert info.value.public_message == LSS_UNAVAILABLE["save_session"]
    assert "save_session" in SESSION_TOOLS
    assert not path.exists()


def test_glm_timeline_entries_unchanged(editor_model):  # noqa: F811
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_indices("region", [0])
    session.shift("region", 0.1)
    session.stage_structural("collapse", "region", {"levels": ["A", "B"]})

    entries = [e for e in timeline_payload(session) if e.get("kind") != "marker"]
    entries += pending_payload(session)

    assert len(entries) >= 2
    for entry in entries:
        assert not {"predictor", "predictor_label", "term_label"} & set(entry)


def test_the_widget_exports_an_edited_superlss_as_joblib():
    case = lognormal_case()
    session = open_case(case)
    _shift(session, "scale:age", 0.3)
    widget = session.widget()
    try:
        result = widget._export_bytes("joblib")
    finally:
        widget.close()

    loaded = joblib.load(io.BytesIO(result.data))
    assert type(loaded) is SuperLSS
    assert result.validation_scope == "artifact+predictions"
    assert [record["predictor"] for record in loaded._editor_history] == ["scale"]
