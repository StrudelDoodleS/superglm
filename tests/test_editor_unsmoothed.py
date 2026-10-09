"""The Unsmoothed line in the editor: a term's curve with its smoothing switched off."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Piecewise, RandomEffect, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor import unsmoothed as unsmoothed_module
from superglm.editor.errors import EditorValueError
from superglm.editor.payloads import session_payload
from superglm.editor.unsmoothed import unsmoothed_job, unsmoothed_lambdas
from tests.test_editor import _post_json
from tests.test_editor_special_levels import (
    BANDS,
    TWELVE,
    _book,
    _declared,
    _fitted_free_models,
)


@pytest.fixture(scope="module")
def book():
    X, y, w = _book()
    return _declared().fit(X, y, sample_weight=w), X, y, w


def _spline_features() -> dict:
    return {
        "age": Spline(kind="ps", n_knots=12),
        "veh": Spline(kind="cr", n_knots=8),
        "area": Categorical(),
    }


@pytest.fixture(scope="module")
def spline_book():
    """Two splines on correlated columns, their smoothing chosen by REML.

    The columns share rows, so how smooth ``veh`` is held moves ``age``'s
    unsmoothed line: holding it ten times stiffer moves it by 0.096 on the
    log scale, and at the configured default by 0.055.
    """
    rng = np.random.default_rng(20261009)
    n = 4000
    age = rng.uniform(18, 80, n)
    veh = 0.3 * (age - 18) + rng.uniform(0, 6, n)
    area = rng.choice(["A", "B", "C"], n)
    w = rng.uniform(0.3, 1.0, n)
    eta = -1.2 + 0.25 * np.sin(age / 7) + 0.15 * np.cos(veh / 3) + 0.2 * (area == "B")
    y = rng.poisson(w * np.exp(eta)) / w
    X = pd.DataFrame({"age": age, "veh": veh, "area": area})
    model = SuperGLM(family="poisson", features=_spline_features(), tol=1e-10)
    return model.fit_reml(X, y, sample_weight=w), X, y, w


def _counted_fits(monkeypatch) -> list:
    """Record each model the Unsmoothed line fits for a spline term."""
    fitted = []
    real = unsmoothed_module.fit_refit_model

    def recorded(source, refit, **kwargs):
        fitted.append(refit)
        return real(source, refit, **kwargs)

    monkeypatch.setattr(unsmoothed_module, "fit_refit_model", recorded)
    return fitted


def _refused(url: str, term: str) -> str:
    with pytest.raises(urllib.error.HTTPError) as refused:
        _post_json(f"{url}/unsmoothed", {"term": term})
    assert refused.value.code == 400
    return json.loads(refused.value.read().decode("utf-8"))["error"]


def test_an_ordered_line_is_its_free_fit_and_one_fit_serves_free_levels_too(book, monkeypatch):
    fitted = _fitted_free_models(monkeypatch)
    model, X, y, w = book
    session = EditorSession.from_model(model, train_data=(X, y, w))
    widget = session.widget()
    try:
        line = _post_json(f"{widget.url}/unsmoothed", {"term": "band"})
        assert len(fitted) == 1
        free = fitted[0].term_inference("band", with_se=False)
        assert line["levels"] == BANDS
        assert line["gaps"] == [] and line["note"] is None
        assert line["fit_token"] == widget._state()["fit_token"]
        # The line has the free fit's shape: its level-to-level ratios.
        own = dict(zip(map(str, free.levels), free.log_relativity, strict=True))
        expected = np.array([own[level] for level in BANDS])
        np.testing.assert_allclose(
            np.log(line["y"]) - np.log(line["y"][0]),
            expected - expected[0],
            rtol=0.0,
            atol=64 * np.finfo(np.float64).eps,
        )
        # A second line and the comparison take no fit of their own, and the
        # line runs through the comparison's diamonds.
        assert _post_json(f"{widget.url}/unsmoothed", {"term": "band"}) == line
        compared = _post_json(f"{widget.url}/free_levels", {"term": "band"})
        assert compared["levels"] == BANDS
        assert compared["y"] == line["y"]
        assert len(fitted) == 1
    finally:
        widget.close()
    # The other way round, the comparison's fit draws the line.
    widget = EditorSession.from_model(model, train_data=(X, y, w)).widget()
    try:
        _post_json(f"{widget.url}/free_levels", {"term": "band"})
        assert _post_json(f"{widget.url}/unsmoothed", {"term": "band"})["y"] == line["y"]
        assert len(fitted) == 2
    finally:
        widget.close()


def test_a_level_the_free_fit_cannot_estimate_is_a_gap_its_note_names():
    """B11 has rows and no claims; B12 is declared on the curve and has no rows."""
    rng = np.random.default_rng(5)
    k = np.repeat(np.arange(12), 50)
    y = rng.poisson(np.exp(-1.0 + 0.1 * k)).astype(float)
    y[k == 11] = 0.0
    X = pd.DataFrame({"band": np.array(TWELVE)[k]})
    band = OrderedCategorical(order=[*TWELVE, "B12"], basis=Spline(kind="ps", n_knots=6))
    model = SuperGLM(family="poisson", features={"band": band}, spline_penalty=20.0).fit(X, y)
    line = unsmoothed_job(EditorSession.from_model(model, train_data=(X, y)), "band")().payload
    assert line["levels"] == [*TWELVE, "B12"]
    assert [value is None for value in line["y"]] == [False] * 11 + [True, True]
    assert line["gaps"] == ["B11", "B12"]
    assert line["note"] == (
        "The line skips B11: every response on its rows is 0, so its free value has no finite "
        "estimate. The line skips B12: the data the refit reads has no rows of positive weight "
        "for it."
    )


def test_a_spline_line_is_the_refit_with_its_smoothing_off_and_every_other_held(
    spline_book, monkeypatch
):
    fitted = _counted_fits(monkeypatch)
    model, X, y, w = spline_book
    session = EditorSession.from_model(model, train_data=(X, y, w))
    widget = session.widget()
    try:
        line = _post_json(f"{widget.url}/unsmoothed", {"term": "age"})
    finally:
        widget.close()
    assert len(fitted) == 1
    held = dict(model._reml_lambdas)
    assert set(held) == {"age", "veh"}
    direct = SuperGLM(
        family="poisson",
        features=_spline_features(),
        spline_penalty={"age": 0.0, "veh": held["veh"]},
        tol=1e-10,
    ).fit(X, y, sample_weight=w)
    # The same basis and knots, placed by the same rule on the same rows.
    np.testing.assert_array_equal(direct._specs["age"]._knots, model._specs["age"]._knots)
    inference = direct.term_inference("age", with_se=False, n_points=session.n_points)
    assert line["x"] == session.terms["age"].x.tolist() == np.asarray(inference.x).tolist()
    # Both are fits on the same rows, stopped by the same rule: each stops
    # once the deviance moves by under tol = 1e-10 relative, which bounds its
    # coefficients' error by about sqrt(tol) = 1e-5. Holding veh at the
    # default smoothing, or ten times stiffer, moves the line by 0.055 and
    # 0.096; leaving age smoothed is the curve, 0.33 away.
    np.testing.assert_allclose(np.log(line["y"]), inference.log_relativity, rtol=0, atol=1e-5)
    assert line["kind"] == "spline" and line["gaps"] == [] and line["note"] is None


def test_the_line_is_kept_through_a_hand_edit_and_dropped_by_a_refit(spline_book, monkeypatch):
    fitted = _counted_fits(monkeypatch)
    model, X, y, w = spline_book
    session = EditorSession.from_model(model, train_data=(X, y, w))
    widget = session.widget()
    try:
        first = _post_json(f"{widget.url}/unsmoothed", {"term": "age"})
        token = widget._state()["fit_token"]
        session.select_x("age", 30.0, 40.0)
        session.shift("age", 0.05)
        assert _post_json(f"{widget.url}/unsmoothed", {"term": "age"}) == first
        assert widget._state()["fit_token"] == token and len(fitted) == 1
        reference = next(
            level for level in ("A", "B", "C") if level != model._specs["area"]._base_level
        )
        _post_json(f"{widget.url}/set_reference", {"term": "area", "level": reference})
        assert session.model is not model
        again = _post_json(f"{widget.url}/unsmoothed", {"term": "age"})
        assert again["fit_token"] == widget._state()["fit_token"] != token
        assert len(fitted) == 2 and fitted[1] is not fitted[0]
    finally:
        widget.close()


def test_a_spline_its_rows_cannot_determine_unsmoothed_is_refused_once_in_a_sentence(
    monkeypatch,
):
    """No rows between 40 and 55: unpenalised, the basis functions there are free."""
    rng = np.random.default_rng(2)
    age = np.concatenate([rng.uniform(18, 40, 1500), rng.uniform(55, 80, 1500)])
    area = rng.choice(["A", "B", "C"], age.size)
    y = rng.poisson(np.exp(-1 + 0.3 * np.sin(age / 8)))
    X = pd.DataFrame({"age": age, "area": area})
    model = SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="ps", n_knots=20, knot_strategy="uniform"),
            "area": Categorical(),
        },
        spline_penalty=5.0,
    ).fit(X, y)
    fitted = _counted_fits(monkeypatch)
    widget = EditorSession.from_model(model, train_data=(X, y)).widget()
    try:
        sentence = unsmoothed_module._UNDETERMINED.format(term="age")
        assert _refused(widget.url, "age") == sentence
        assert _refused(widget.url, "age") == sentence
        assert len(fitted) == 1
    finally:
        widget.close()


def test_the_line_is_offered_for_smoothed_terms_and_refused_in_sentences_elsewhere(
    book, spline_book
):
    session = EditorSession.from_model(book[0], train_data=book[1:])
    flags = {name: term["unsmoothed"] for name, term in session_payload(session).items()}
    assert flags == {"band": True, "area": False}
    widget = session.widget()
    try:
        assert _refused(widget.url, "area") == unsmoothed_module._NOT_SMOOTHED.format(term="area")
    finally:
        widget.close()
    X, y, w = _book(n=3000)
    stepped = OrderedCategorical(order=BANDS, basis=Piecewise(breaks=["Mi030", "Mi048"]))
    piecewise = SuperGLM(family="poisson", features={"band": stepped}).fit(X, y, sample_weight=w)
    assert not session_payload(EditorSession.from_model(piecewise))["band"]["unsmoothed"]
    # A random effect is fitted by REML only, which would choose the other
    # terms' smoothing again.
    _model, X, y, w = spline_book
    X = X.assign(cell=np.random.default_rng(3).choice([f"c{i}" for i in range(8)], len(X)))
    mixed = SuperGLM(
        family="poisson", features={"age": Spline(kind="ps", n_knots=8), "cell": RandomEffect()}
    ).fit_reml(X, y, sample_weight=w)
    with pytest.raises(EditorValueError) as refused:
        unsmoothed_job(EditorSession.from_model(mixed, train_data=(X, y, w)), "age")
    assert str(refused.value) == unsmoothed_module._REML_ONLY.format(terms="'cell'")


def test_the_terms_own_smoothing_parameters_go_to_zero_and_no_others():
    """An interaction ``age:veh`` starts like a component of ``age`` and keeps its own."""
    groups = [
        SimpleNamespace(name=name, feature_name=feature)
        for name, feature in [
            ("age", "age"),
            ("veh", "veh"),
            ("age2", "age2"),
            ("age:veh", "age:veh"),
        ]
    ]
    fitted = {
        "age": 19.6,
        "veh": 1.2e8,
        "age2:null": 4.8e3,
        "age2:wiggle": 6.3e5,
        "age:veh:margin_age": 8.0e5,
        "age:veh:margin_veh": 2.6e6,
    }
    reml = SimpleNamespace(_fit_state=None, _reml_lambdas=fitted, _groups=groups)
    assert unsmoothed_lambdas(reml, "age") == {**fitted, "age": 0.0}
    assert unsmoothed_lambdas(reml, "age2") == {
        **fitted,
        "age2:null": 0.0,
        "age2:wiggle": 0.0,
        "age2": 0.0,
    }
    one = SimpleNamespace(_fit_state=None, _reml_lambdas=None, _lambda2_config=5.0, _groups=groups)
    assert unsmoothed_lambdas(one, "veh") == {"age": 5.0, "veh": 0.0, "age2": 5.0, "age:veh": 5.0}
