"""The editor scores the in-force coefficients on the rows each reader owns."""

import threading
from contextlib import closing

import joblib
import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm.distributions import Tweedie
from superglm.editor import edit
from superglm.editor.persistence import joblib_load_bytes
from tests._no_mean_fixture import CELLS, CELLS_Y, TRAIN, TRAIN_Y, editor_fit, negative_power_lss


@pytest.mark.parametrize(("split", "label"), [("validation", "Validation"), ("test", "Test")])
def test_holdout_fallback_is_labelled_even_when_no_row_can_be_quoted(tmp_path, split, label):
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0,
        retain_fit_state=False,
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    holdout = (CELLS.iloc[[3, 3]], CELLS_Y[[3, 3]])
    session = edit(model, **{f"{split}_data": holdout})
    with closing(session.widget()) as widget:
        widget._drag("a", [1], delta=0.01)
        summary = widget._summary("in_force")
        for revised in (
            session.to_model(),
            joblib.load(session.save_model(tmp_path / "fallback.joblib")),
        ):
            assert np.isnan(revised.result.deviance)
            with pytest.raises(RuntimeError, match=rf"{label} fallback.*2 of 2 rows") as caught:
                revised.summary()
            assert summary["error"] == str(caught.value)
        assert len(session.history) == 1
        session.shift("a", 0.60)
        priced = widget._summary("in_force")
        ratio = holdout[1] * 0.11
        expected = 2 * np.sum(ratio - 1 - np.log(ratio))
        assert priced["compact"]["model"]["deviance"] == pytest.approx(expected)
        assert f"{label} fallback" in priced["note"]
        assert f"{label} fallback" in str(session.to_model().summary())
        saved = joblib.load(session.save_model(tmp_path / "priced-fallback.joblib"))
        assert f"{label} fallback" in str(saved.summary())
        if split == "test":
            assert "Live editor metrics use test data" in widget._report("validation")["note"]


def test_zero_weight_tweedie_row_has_no_effect_on_training_statistics():
    model = SuperGLM(
        family=Tweedie(p=1.5),
        link="inverse",
        selection_penalty=0,
        weight_semantics="frequency",
        features={"a": Categorical(), "b": Categorical()},
    ).fit(TRAIN, TRAIN_Y)
    X = pd.concat([TRAIN, CELLS.iloc[[3]]], ignore_index=True)
    y = np.r_[1.1 * TRAIN_Y, 3.0]
    session = edit(
        model,
        train_data=(X, y, np.r_[np.ones(len(TRAIN)), 0.0]),
        validation_data=(CELLS, CELLS_Y),
    )
    with closing(session.widget()) as widget:
        widget._drag("a", [1], delta=0.01)
        revised = joblib_load_bytes(widget._export_bytes("joblib").data)
        expected = revised.metrics(TRAIN, y[:-1])
        assert revised.result.deviance == pytest.approx(expected.deviance, rel=1e-12)
        assert revised._fit_stats.log_likelihood == pytest.approx(
            expected.log_likelihood, rel=1e-12
        )
        assert widget._summary("in_force")["compact"]["model"]["deviance"] == pytest.approx(
            expected.deviance, rel=1e-12
        )
        assert "1 of 4 rows" in widget._metrics("deviance")["error"]


def test_default_edit_rescores_exact_offset_rows_after_an_explicit_score():
    rng = np.random.default_rng(697)
    X = pd.DataFrame({"x": rng.uniform(size=300)})
    offset = rng.uniform(-0.4, 0.5, len(X))
    weights = np.linspace(1.0, 3.0, len(X))
    y = np.sin(8.0 * X["x"].to_numpy()) + offset + rng.normal(0, 0.05, len(X))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=0.1,
        features={"x": Spline("ps", k=8)},
        discrete=True,
        n_bins=12,
    ).fit(X, y, sample_weight=weights, offset=offset)
    fitted_exact = np.dot(weights, (y - model.predict(X, offset=offset)) ** 2)
    assert abs(model.result.deviance - fitted_exact) > 0.1
    alternate, alternate_y, alternate_offset = X.iloc[:20], y[:20] + 0.5, -offset[:20]
    session = edit(model, validation_data=(alternate, alternate_y, None, alternate_offset))
    with closing(session.widget()) as widget:
        widget._drag("x", [20, 21, 22], delta=0.15)
        explicit = session.to_model(X=alternate, y=alternate_y, offset=alternate_offset)
        explicit_pred = explicit.predict(alternate, offset=alternate_offset)
        assert explicit.result.deviance == pytest.approx(np.sum((alternate_y - explicit_pred) ** 2))

        widget._drag("x", [20, 21, 22], delta=0.2)
        saved = joblib_load_bytes(widget._export_bytes("joblib").data)
        exact = saved.predict(X, offset=offset)
        expected = np.dot(weights, (y - exact) ** 2)
        assert abs(expected - np.dot(weights, (y - saved.predict(X)) ** 2)) > 0.1
        assert saved.result.deviance == pytest.approx(expected)
        assert widget._summary("in_force")["compact"]["model"]["deviance"] == pytest.approx(
            expected
        )
        assert session.to_model().result.deviance == pytest.approx(expected)
        assert explicit.result.deviance == pytest.approx(np.sum((alternate_y - explicit_pred) ** 2))


@pytest.mark.parametrize("first", ["summary", "save"])
def test_only_the_first_statistics_reader_quotes_training_rows(monkeypatch, tmp_path, first):
    from superglm.model import base

    calls = []
    predict_exact = base.predict_exact

    def counted(model, X, *args, **kwargs):
        calls.append(len(X))
        return predict_exact(model, X, *args, **kwargs)

    session = edit(
        editor_fit(),
        train_data=(TRAIN, TRAIN_Y),
        validation_data=(CELLS, CELLS_Y),
    )
    monkeypatch.setattr(base, "predict_exact", counted)
    with closing(session.widget()) as widget:
        widget._drag("a", [1], delta=0.01)
        assert calls == []
        assert "1 of 4 rows" in widget._metrics("deviance")["error"]
        assert calls == [4, 4]
        if first == "summary":
            widget._summary("in_force")
        session.save_model(tmp_path / "edited.joblib")
        widget._summary("in_force")
        widget._export_bytes("joblib")
        assert calls.count(len(TRAIN)) == 1


@pytest.mark.parametrize("reader", ["summary", "save", "widget-export"])
@pytest.mark.parametrize("source", ["retained-frame", "supplied-response"])
def test_training_statistics_readers_revalidate_cached_rows(tmp_path, reader, source):
    if source == "retained-frame":
        X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
        model = SuperGLM(family="gaussian", selection_penalty=0, features={"x": Numeric()}).fit(
            X, X.x**2
        )
        session = edit(model)
        term, index, delta = "x", 0, 0.5
    else:
        y = TRAIN_Y.copy()
        session = edit(editor_fit(), train_data=(TRAIN, y))
        term, index, delta = "a", 1, 0.01

    path = tmp_path / "edited.joblib"
    with closing(session.widget()) as widget:
        widget._drag(term, [index], delta=delta)
        before = widget._summary("in_force")["compact"]["model"]["deviance"]

        def read_deviance():
            if reader == "summary":
                return widget._summary("in_force")["compact"]["model"]["deviance"]
            if reader == "save":
                return joblib.load(session.save_model(path)).result.deviance
            return joblib_load_bytes(widget._export_bytes("joblib").data).result.deviance

        if source == "retained-frame":
            X.loc[0, "x"] = 0.25
            with pytest.raises(RuntimeError, match="retained fit data were mutated") as fresh:
                session.to_model()
            with pytest.raises(RuntimeError, match="retained fit data were mutated") as cached:
                read_deviance()
            assert str(cached.value) == str(fresh.value)
            assert not path.exists()
        else:
            y *= 2
            fresh = session.to_model()
            ratio = y / fresh.predict(TRAIN)
            expected = 2 * np.sum(ratio - 1 - np.log(ratio))
            assert before == pytest.approx(8.6460479104878)
            assert expected == pytest.approx(27.857217076783584)
            assert read_deviance() == pytest.approx(expected)


def test_late_training_summary_cannot_replace_newer_training_evidence(monkeypatch):
    from superglm.editor import apply as apply_module

    started, release = threading.Event(), threading.Event()
    replies = []
    original = apply_module._refresh_fit_statistics

    def blocked(model, **kwargs):
        started.set()
        if not release.wait(8):
            raise RuntimeError("test did not release the summary")
        return original(model, **kwargs)

    session = edit(editor_fit(), validation_data=(CELLS, CELLS_Y))
    with closing(session.widget()) as widget:
        asked = widget._drag("a", [1], delta=-0.30)
        monkeypatch.setattr(apply_module, "_refresh_fit_statistics", blocked)

        def request_summary():
            replies.append(
                widget._summary("in_force", in_force=asked["in_force"], request_sequence=7)
            )

        worker = threading.Thread(target=request_summary)
        worker.start()
        try:
            assert started.wait(5)
            restored = widget._drag("a", [1], delta=0.90)
        finally:
            release.set()
            worker.join(timeout=8)
        assert replies[0]["status"] == "superseded"
        assert replies[0]["in_force"] == asked["in_force"]
        summary = widget._summary("in_force", in_force=restored["in_force"])
        assert summary["compact"]["model"]["deviance"] == pytest.approx(32.1549535411)


def test_lss_holdout_scoring_does_not_recompute_retained_training_predictions(monkeypatch):
    from superglm.distributional import revision

    session = edit(
        negative_power_lss(),
        train_data=(TRAIN, TRAIN_Y),
        validation_data=(CELLS.iloc[:3], CELLS_Y[:3]),
    )
    calls = []
    recompute = revision._recomputed_rows

    def counted(*args, **kwargs):
        calls.append(True)
        return recompute(*args, **kwargs)

    monkeypatch.setattr(revision, "_recomputed_rows", counted)
    with closing(session.widget()) as widget:
        widget._drag("mean:a", [1], delta=0.1)
        report = widget._report("validation")
        assert calls == []
        validation = next(split for split in report["splits"] if split["name"] == "validation")
        edited = session.to_model()
        assert calls == [True]
        expected = CELLS_Y[:3].sum() / edited.predict(CELLS.iloc[:3]).sum()
        assert validation["metrics"]["edited"]["actual_expected"] == pytest.approx(expected)


def test_training_statistics_use_the_replacement_fits_retained_rows():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})

    def fit(y):
        return SuperGLM(family="gaussian", selection_penalty=0, features={"x": Numeric()}).fit(X, y)

    session = edit(fit(X.x**2))
    y = 2 * X.x + X.x**2
    session.replace_in_force_model(fit(y))
    with closing(session.widget()) as widget:
        widget._drag("x", [0], delta=0.5)
        edited = session.to_model()
        expected = np.sum((y - edited.predict(X)) ** 2)
        summary = widget._summary("in_force")
        assert summary["compact"]["model"]["deviance"] == pytest.approx(expected)


def test_notebook_save_rejects_a_superseded_serialization(tmp_path, monkeypatch):
    from superglm.editor import persistence

    session = edit(editor_fit(), validation_data=(CELLS, CELLS_Y))
    session.select_indices("a", [1])
    session.shift("a", 0.01)
    serialize = persistence.serialize_validated_model

    def edit_during_serialization(*args, **kwargs):
        session.shift("a", -0.31)
        return serialize(*args, **kwargs)

    monkeypatch.setattr(persistence, "serialize_validated_model", edit_during_serialization)
    path = tmp_path / "superseded.joblib"
    with pytest.raises(RuntimeError, match="superseded by a newer model revision"):
        session.save_model(path)
    assert not path.exists()
