"""Edited deviance belongs to the scoring rows of the current coefficients."""

import json
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from superglm import Categorical, SuperGLM
from superglm.editor import edit
from superglm.editor.summaries import summary_payload
from superglm.model.base import predict_priced
from tests._no_mean_fixture import CELLS, CELLS_Y, TRAIN, TRAIN_Y, editor_fit


def _gamma_deviance(delta):
    # The saturated three-cell fit has means 1, 4, 4.
    eta = np.tile([1, 1, 0.25, 0.25, 0.25, 0.25], 5)
    eta[TRAIN["a"] == "1"] += delta
    ratio = TRAIN_Y * eta
    return 2 * np.sum(ratio - 1 - np.log(ratio))


def _assert_missing(model, payload, rows):
    assert np.isnan(model.result.deviance)
    assert np.isnan(model._solver_result.deviance)
    with pytest.raises(RuntimeError, match="no fit statistics") as exc:
        model.summary()
    reason = str(exc.value)
    assert rows in reason
    assert "Adjust the curve or undo the edit" in reason
    assert payload["available"] is False and payload["deviance"] is None
    assert payload["error"] == reason
    diagnostics = model.diagnostics()["_model"]
    assert np.isnan(diagnostics["deviance"])
    assert diagnostics["fit_statistics_unavailable"] == reason
    fit = model.training_telemetry()["fit"]
    assert fit["deviance"] is None
    assert fit["fit_statistics_unavailable"] == reason
    json.dumps(payload, allow_nan=False)
    json.dumps(fit, allow_nan=False)
    return reason


def test_unpriced_training_edit_round_trips_and_recovers(tmp_path):
    model = editor_fit()
    session = edit(model, train_data=(TRAIN, TRAIN_Y))
    widget = SimpleNamespace(session=session)
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    session.shift("a", -0.31)

    rows = (
        "10 of 30 rows have no quote under this model "
        "(row positions 2, 3, 8, 9, 14, 15, 20, 21, 26, 27)"
    )
    edited = session.to_model()
    payload = summary_payload(widget, "in_force")
    reason = _assert_missing(edited, payload, rows)
    loaded = joblib.load(session.save_model(tmp_path / "unavailable.joblib"))
    assert _assert_missing(loaded, payload, rows) == reason
    assert edited._fit_stats is None
    assert len(session.history) == 2

    session.undo()
    scored = session.to_model()
    assert scored.result.deviance == pytest.approx(_gamma_deviance(0.01))
    assert scored.summary()["deviance"]["deviance"] == scored.result.deviance
    assert summary_payload(widget, "in_force")["compact"]["model"]["deviance"] == pytest.approx(
        scored.result.deviance
    )
    scored_loaded = joblib.load(session.save_model(tmp_path / "scored.joblib"))
    assert scored_loaded.summary()["deviance"]["deviance"] == scored.result.deviance
    session.redo()
    assert _assert_missing(session.to_model(), summary_payload(widget, "in_force"), rows) == reason
    session.undo().undo()
    assert session.to_model().result.deviance == model.result.deviance
    assert model.result.deviance == pytest.approx(_gamma_deviance(0))
    assert summary_payload(widget, "in_force")["compact"]["model"]["deviance"] == pytest.approx(
        model.result.deviance
    )

    # Refit the unavailable artifact on different responses so a reused score would show.
    y = np.tile([0.75, 1.25, 3.0, 5.0, 3.0, 5.0], 5)
    loaded.fit(TRAIN, y)
    ratio = y * np.tile([1, 1, 0.25, 0.25, 0.25, 0.25], 5)
    assert loaded.result.deviance == pytest.approx(2 * np.sum(ratio - 1 - np.log(ratio)))
    assert loaded.summary()["deviance"]["deviance"] == loaded.result.deviance
    assert "fit_statistics_unavailable" not in loaded.training_telemetry()["fit"]


def test_default_edit_readers_agree_and_holdout_splits_score_their_own_rows(tmp_path):
    test = CELLS.iloc[[0, 2]]
    test_y = np.array([2.0, 3.0])
    session = edit(editor_fit(), validation_data=(CELLS, CELLS_Y), test_data=(test, test_y))
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    with closing(session.widget()) as widget:

        def assert_training_score(delta):
            expected = _gamma_deviance(delta)
            payload = widget._summary("in_force")
            assert payload["compact"]["model"]["deviance"] == pytest.approx(expected)
            saved = joblib.load(session.save_model(tmp_path / "edited.joblib"))
            assert saved.result.deviance == pytest.approx(expected)
            assert session.to_model().result.deviance == pytest.approx(expected)

        assert_training_score(0.01)
        validation = widget._metrics("deviance")
        assert validation["dataset"] == "validation"
        assert "1 of 4 rows have no quote" in validation["error"]
        test_metric = widget._metrics("deviance", dataset="test")
        ratio = test_y * [1.0, 0.25]
        assert test_metric["edited"] == pytest.approx(2 * np.sum(ratio - 1 - np.log(ratio)))

        session.shift("a", -0.31)
        payload = widget._summary("in_force")
        _assert_missing(session.to_model(), payload, "10 of 30 rows have no quote")
        assert len(session.history) == 2
        session.undo()
        assert_training_score(0.01)
        session.redo()
        assert widget._summary("in_force")["error"] == payload["error"]

        session.shift("a", 0.90)
        assert_training_score(0.60)
        ratio = CELLS_Y * [1.0, 0.85, 0.25, 0.10]
        restored = widget._metrics("deviance")
        assert restored["edited"] == pytest.approx(2 * np.sum(ratio - 1 - np.log(ratio)))
        assert restored["original"] is None
        final = widget._report("final")
        assert final["summary"]["compact"]["model"]["deviance"] == pytest.approx(
            _gamma_deviance(0.60)
        )


def test_rescoring_without_retained_rows_discloses_the_missing_score():
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0,
        retain_fit_state=False,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    rescored = edit(model).to_model(sample_weight=np.ones(len(TRAIN)))
    assert np.isnan(rescored.result.deviance)
    with pytest.raises(RuntimeError, match="Supply both X and y"):
        rescored.summary()


def test_pre_outcome_artifacts_load_and_keep_their_predictions(tmp_path):
    model = editor_fit()
    session = edit(model, train_data=(TRAIN, TRAIN_Y))
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    scored = session.to_model()
    session.shift("a", -0.31)
    unavailable = session.to_model()
    # Recreate the earlier serialized layout. It carried the fitted deviance in
    # both results when the edited statistics were absent and recorded no row reason.
    for revised in (scored, unavailable):
        vars(revised._fit_state).pop("edited_statistics", None)
    for result in (unavailable._result, unavailable._solver_result):
        object.__setattr__(result, "deviance", model.result.deviance)
    for name, revised in (("scored", scored), ("unavailable", unavailable)):
        path = tmp_path / f"legacy-{name}.joblib"
        joblib.dump(revised, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(
            predict_priced(loaded, TRAIN, None), predict_priced(revised, TRAIN, None)
        )
        if name == "scored":
            assert loaded.summary()["deviance"]["deviance"] == pytest.approx(_gamma_deviance(0.01))
        else:
            assert np.isnan(loaded.result.deviance)
            with pytest.raises(RuntimeError, match="saved edit did not record the scoring rows"):
                loaded.summary()


def test_every_edited_statistics_move_publishes_the_new_scoring_outcome():
    from superglm.model.fit_state import _EDITED_STATISTICS_MOVES

    model = editor_fit()
    session = edit(model, train_data=(TRAIN, TRAIN_Y))
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    scored = session.to_model()
    session.shift("a", -0.31)
    unavailable = session.to_model()
    sources = {"fitted": model, "scored": scored, "unavailable": unavailable}
    assert set(_EDITED_STATISTICS_MOVES) == {
        (source, event) for source in sources for event in ("score", "withhold")
    }
    for (source, event), destination in _EDITED_STATISTICS_MOVES.items():
        previous = sources[source]
        if event == "score":
            # Every source has the same curve on the base cells.
            base = TRAIN["a"] == "0"
            revised = edit(previous).to_model(X=TRAIN.loc[base], y=TRAIN_Y[base])
            ratio = TRAIN_Y[base] * np.tile([1, 1, 0.25, 0.25], 5)
            assert revised.result.deviance == pytest.approx(2 * np.sum(ratio - 1 - np.log(ratio)))
            assert revised._fit_state.edited_statistics.reason is None
            assert revised._fit_stats is not None
        else:
            revised = edit(previous).to_model(X=CELLS.iloc[[3]], y=CELLS_Y[[3]])
            assert np.isnan(revised.result.deviance)
            assert "1 of 1 rows have no quote" in revised._fit_state.edited_statistics.reason
        assert previous._fit_state.edited_statistics.status == source
        assert revised._fit_state.edited_statistics.status == destination


def test_only_a_validated_unavailable_revision_can_publish_missing_deviance():
    from superglm.model.fit_state import FittedStateRevision, capture_fit_state

    model = editor_fit()
    baseline = model.result
    work = FittedStateRevision.start(model)
    work.model._fit_stats = None
    with pytest.raises(RuntimeError, match="scored editor revision requires fit statistics"):
        work.with_edited_statistics().commit()

    revision = FittedStateRevision.start(model).with_edited_statistics(reason="Unpriced rows")
    # Fresh fit publication has its own finite-result contract, even for an edited input.
    with pytest.raises(RuntimeError, match="scalar results must be finite"):
        capture_fit_state(SimpleNamespace(model=revision.model), model, revision=2)
    with pytest.raises(RuntimeError, match="require a reason"):
        replace(revision, edited_statistics=replace(revision.edited_statistics, reason="")).commit()
    revision.model._fit_stats = model._fit_stats
    with pytest.raises(RuntimeError, match="require a reason and missing scores"):
        revision.commit()
    revision.model._fit_stats = None
    revision.model._solver_result.deviance = baseline.deviance
    with pytest.raises(RuntimeError, match="require a reason and missing scores"):
        revision.commit()
    revision.model._solver_result.deviance = np.nan
    revision.model._result.phi = revision.model._solver_result.phi = np.inf
    with pytest.raises(RuntimeError, match="scalar results must be finite"):
        revision.commit()
    assert model.result is baseline
