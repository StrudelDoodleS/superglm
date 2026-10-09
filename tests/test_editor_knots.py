"""The editor's knot adjuster: a count and rule, positions, or reset, as a waiting structural change."""

from __future__ import annotations

import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, read_structure
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.editor.payloads import session_payload
from tests.test_editor import _post_json

BANDS = [f"B{i}" for i in range(8)]
AGE_KNOTS = [20.0, 22.0, 24.0, 26.0, 35.0, 50.0, 70.0]


def _book(seed: int = 20261009, n: int = 8000):
    """A Poisson book whose age effect falls steeply in the young ages, where most rows are."""
    rng = np.random.default_rng(seed)
    age = 18.0 + np.minimum(rng.gamma(3.0, 7.0, n), 72.0)
    band = rng.choice(BANDS, n)
    area = rng.choice(["A", "B", "C"], n)
    exposure = rng.uniform(0.2, 1.0, n)
    index = np.array([BANDS.index(b) for b in band])
    eta = -1.6 + 0.8 * np.exp(-(age - 18.0) / 6.0) + 0.04 * index + 0.2 * (area == "B")
    y = rng.poisson(exposure * np.exp(eta)) / exposure
    return pd.DataFrame({"age": age, "band": band, "area": area}), y, exposure


def _declared(age=None, band=None) -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={
            "age": Spline(kind="cr", n_knots=6) if age is None else age,
            "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=3))
            if band is None
            else band,
            "area": Categorical(),
        },
        spline_penalty=10.0,
    )


@pytest.fixture(scope="module")
def book():
    X, y, w = _book()
    return _declared().fit(X, y, sample_weight=w), X, y, w


def _session(book) -> EditorSession:
    model, X, y, w = book
    return EditorSession.from_model(model, train_data=(X, y, w))


def _knots(model, term: str) -> np.ndarray:
    spec = model._specs[term]
    inner = spec._basis_spline if isinstance(spec, OrderedCategorical) else spec
    return np.asarray(inner.fitted_base_knots)


def test_a_knot_change_waits_refits_undoes_and_puts_back_the_original_fit(book):
    model, X, *_ = book
    session = _session(book)
    before = session_payload(session)["age"]["knots"]
    assert (before["count"], before["strategy"], before["from_editor"]) == (6, "uniform", False)
    assert not before["resettable"]

    step = session.stage_structural("knots", "age", {"positions": AGE_KNOTS[::-1]})
    assert step.label == "knots placed by hand in age"
    assert session_payload(session)["age"]["pending"]["knots"] == {
        "positions": AGE_KNOTS,
        "count": 7,
        "strategy": "explicit",
        "alpha": 0.2,
    }
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))

    session.refit_pending()
    np.testing.assert_array_equal(_knots(session.model, "age"), AGE_KNOTS)
    after = session_payload(session)["age"]["knots"]
    assert (after["positions"], after["strategy"], after["from_editor"]) == (
        AGE_KNOTS,
        "explicit",
        True,
    )
    assert after["resettable"]

    session.undo()
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))
    assert [step.operation for step in session.pending] == ["knots"]
    session.redo()
    np.testing.assert_array_equal(_knots(session.model, "age"), AGE_KNOTS)


@pytest.mark.parametrize(
    ("term", "params", "strategy"),
    [
        ("age", {"positions": AGE_KNOTS}, "explicit"),
        ("age", {"count": 9, "strategy": "quantile_rows"}, "quantile_rows"),
        ("band", {"positions": [1.5, 2.0, 4.3]}, "explicit"),
        ("band", {"count": 5, "strategy": "uniform"}, "uniform"),
    ],
)
def test_the_structure_export_records_the_knots_and_applies_them_to_the_declaration(
    book, term, params, strategy
):
    model, X, y, w = book
    session = _session(book)
    session.replace_with_knots(term, params)
    exported = json.loads(session.export_structure())
    entry = exported["features"][term]["knots"]
    assert entry["strategy"] == strategy
    assert entry["positions"] == _knots(session.model, term).tolist()
    applied = read_structure(exported).apply(_declared()).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(_knots(applied, term), _knots(session.model, term))
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))
    # The code's own knots are not recorded.
    assert "knots" not in json.loads(_session(book).export_structure())["features"][term]


def test_an_ordered_term_takes_at_most_one_knot_fewer_than_its_levels(book):
    """The constructor clamps a larger count; the editor refuses it and names the maximum."""
    session = _session(book)
    assert session_payload(session)["band"]["knots"]["max_count"] == 7
    sentence = "'band' has 8 levels on its curve, so it takes at most 7 knots."
    for params in (
        {"count": 8, "strategy": "uniform"},
        {"positions": [0.5 + 0.8 * i for i in range(8)]},
    ):
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural("knots", "band", params)
        assert str(refused.value) == sentence
    assert (
        session.stage_structural("knots", "band", {"count": 7, "strategy": "uniform"}).metadata[
            "count"
        ]
        == 7
    )


def test_a_knot_between_two_levels_sits_between_their_values_on_the_axis():
    """Chart positions run 0..L-1 over the smooth levels; the spline's own axis is their values."""
    values = {"B0": 0.0, "B1": 1.0, "B2": 4.0, "B3": 5.0, "B4": 9.0, "B5": 10.0}
    X, y, w = _book(n=4000)
    X = X.assign(band=X["band"].map(lambda b: f"B{int(b[1:]) % 6}"))
    band = OrderedCategorical(values=values, basis=Spline(kind="cr", n_knots=2))
    model = _declared(band=band).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    step = session.stage_structural("knots", "band", {"positions": [1.5, 3.25]})
    assert step.metadata["positions"] == [2.5, 6.0]
    session.refit_pending()
    np.testing.assert_array_equal(_knots(session.model, "band"), [2.5, 6.0])
    assert session_payload(session)["band"]["knots"]["positions"] == [1.5, 3.25]


def test_knot_changes_refuse_in_fixed_sentences(book):
    session = _session(book)
    rule = (
        "Choose how the knots are placed: even spacing, quantiles of values, quantiles of rows "
        "or tempered quantiles."
    )
    count = "The knot count must be a whole number of at least 1."
    positions = (
        "Knot positions must be numbers inside the range of 'age', at least 0.1 apart "
        "and at least 0.1 from its ends."
    )
    forms = "Give a knot count and placement rule, a list of positions, or reset."
    refusals = [
        (
            "area",
            {"count": 3, "strategy": "uniform"},
            "Knots are for spline terms and ordered terms with a spline basis.",
        ),
        ("age", {"count": 0, "strategy": "uniform"}, count),
        ("age", {"count": 2.5, "strategy": "uniform"}, count),
        ("age", {"count": True, "strategy": "uniform"}, count),
        ("age", {"count": 4, "strategy": "random"}, rule),
        (
            "age",
            {"count": 4, "strategy": "quantile_tempered", "alpha": 1.5},
            "Tempered quantiles take an alpha from 0 to 1.",
        ),
        ("age", {"positions": [10.0, 30.0]}, positions),
        ("age", {"positions": [30.0, 30.05]}, positions),
        ("age", {"positions": ["30"]}, positions),
        ("age", {"positions": [30.0, float("nan")]}, positions),
        ("age", {"count": 4}, forms),
        ("age", {"reset": False}, forms),
        ("age", {"count": 4, "strategy": "uniform", "positions": [30.0]}, forms),
    ]
    for term, params, sentence in refusals:
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural("knots", term, params)
        assert str(refused.value) == sentence
    assert session.pending == []


def test_a_level_change_waiting_on_an_ordered_term_holds_its_knots(book):
    session = _session(book)
    session.stage_structural("collapse", "band", {"levels": ["B6", "B7"]})
    with pytest.raises(EditorValueError) as refused:
        session.stage_structural("knots", "band", {"count": 2, "strategy": "uniform"})
    assert str(refused.value) == (
        "A waiting change on 'band' changes its levels; refit it before changing the knots."
    )


@pytest.mark.parametrize(
    ("age", "sentence"),
    [
        (
            Spline(kind="ns", n_knots=5),
            "'age' is a natural spline (kind=\"ns\"), whose penalty needs evenly spaced knots; "
            'change its count here, or declare it with kind="cr" or kind="ps" to place its '
            "knots freely.",
        ),
        (
            Spline(kind="ps", n_knots=5, m=4),
            "The penalty order of 'age' is above its degree, which needs evenly spaced knots; "
            "change its count here, or lower m in code to place its knots freely.",
        ),
    ],
)
def test_a_spline_whose_penalty_needs_even_knots_takes_only_a_count(age, sentence):
    X, y, w = _book(n=3000)
    model = _declared(age=age).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    for params in ({"positions": AGE_KNOTS}, {"count": 6, "strategy": "quantile"}):
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural("knots", "age", params)
        assert str(refused.value) == sentence
    session.stage_structural("knots", "age", {"count": 7, "strategy": "uniform"})
    session.refit_pending()
    assert _knots(session.model, "age").size == 7


def test_a_term_used_by_an_interaction_keeps_its_knots():
    X, y, w = _book(n=3000)
    model = SuperGLM(
        family="poisson",
        features={"age": Spline(kind="cr", n_knots=5), "area": Categorical()},
        interactions=[("age", "area")],
        spline_penalty=10.0,
    ).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    knots = session_payload(session)["age"]["knots"]
    assert (knots["available"], knots["reason"]) == (
        False,
        "A term used by an interaction keeps its knots.",
    )
    with pytest.raises(EditorValueError) as refused:
        session.stage_structural("knots", "age", {"count": 4, "strategy": "uniform"})
    assert str(refused.value).startswith(
        "Cannot change the knots for term 'age' because it is used by interaction(s): "
    )


def test_reset_puts_back_the_declared_knots_and_the_export_drops_them(book):
    model, X, *_ = book
    session = _session(book)
    session.replace_with_knots("age", {"count": 9, "strategy": "quantile"})
    assert session_payload(session)["age"]["knots"]["resettable"]
    session.stage_structural("knots", "age", {"reset": True})
    assert session_payload(session)["age"]["knots"]["resettable"] is False
    session.refit_pending()
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))
    assert "knots" not in json.loads(session.export_structure())["features"]["age"]


def test_a_shaped_range_and_a_knot_change_compose_in_either_order(book):
    """Knots keep a range in force, and a range keeps knots chosen by hand and their record."""
    model, X, y, w = book
    session = _session(book)
    session.stage_structural("shape", "age", {"lo": 40.0, "hi": 60.0, "degree": 1})
    session.stage_structural("knots", "age", {"positions": [20.0, 25.0, 30.0, 70.0]})
    session.refit_pending()
    spec = session.model._specs["age"]
    assert [(r.lo, r.hi, r.degree) for r in spec.polynomial_ranges] == [(40.0, 60.0, 1)]
    np.testing.assert_array_equal(spec.fitted_base_knots, [20.0, 25.0, 30.0, 70.0])
    session.replace_with_shaped_range("age", lo=75.0, hi=85.0, degree=0)
    exported = json.loads(session.export_structure())
    assert exported["features"]["age"]["knots"]["positions"] == [20.0, 25.0, 30.0, 70.0]
    assert len(exported["features"]["age"]["ranges"]) == 2
    applied = read_structure(exported).apply(_declared()).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))


def test_the_widget_stages_knots_and_refits_them_at_once(book):
    session = _session(book)
    widget = session.widget()
    try:
        staged = _post_json(
            f"{widget.url}/stage",
            {"operation": "knots", "term": "age", "params": {"count": 8, "strategy": "quantile"}},
        )
        assert staged["state"]["terms"]["age"]["pending"]["knots"]["count"] == 8
        session.pending.clear()
        _post_json(f"{widget.url}/knots", {"term": "band", "params": {"positions": [2.5]}})
        # Halfway between the third and fourth levels, at 2/7 and 3/7: a few roundings.
        u = np.finfo(np.float64).eps / 2
        np.testing.assert_allclose(_knots(session.model, "band"), [2.5 / 7], rtol=4 * u, atol=0)
        with pytest.raises(urllib.error.HTTPError) as refused:
            _post_json(
                f"{widget.url}/knots",
                {"term": "age", "params": {"count": 0, "strategy": "uniform"}},
            )
        assert json.loads(refused.value.read().decode("utf-8"))["error"] == (
            "The knot count must be a whole number of at least 1."
        )
    finally:
        widget.close()
    session.undo()
    assert _knots(session.model, "band").size == 3
