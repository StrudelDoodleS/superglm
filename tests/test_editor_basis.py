"""The editor's basis controls: the spline's kind and Shrink (select=True), as waiting changes."""

from __future__ import annotations

import json
import urllib.error

import numpy as np
import pytest

from superglm import Categorical, Constraint, OrderedCategorical, Spline, SuperGLM, read_structure
from superglm.editor import EditorSession
from superglm.editor.errors import EditorValueError
from superglm.editor.payloads import session_payload
from superglm.features._spline_penalties import build_general_difference_penalty
from superglm.features.ordered_categorical import _spline_kind_name
from tests.test_editor import _post_json
from tests.test_editor_knots import _book, _declared, _knots


@pytest.fixture(scope="module")
def book():
    X, y, w = _book()
    return _declared().fit(X, y, sample_weight=w), X, y, w


def _session(book) -> EditorSession:
    model, X, y, w = book
    return EditorSession.from_model(model, train_data=(X, y, w))


def _fitted(model, term: str):
    spec = model._specs[term]
    return spec._basis_spline if isinstance(spec, OrderedCategorical) else spec


def _basis(model, term: str) -> tuple[str, bool]:
    spline = _fitted(model, term)
    return _spline_kind_name(spline), bool(spline.select)


def _refused(session, term: str, params) -> str:
    with pytest.raises(EditorValueError) as refused:
        session.stage_structural("basis", term, params)
    return str(refused.value)


@pytest.mark.parametrize(
    ("term", "kind"),
    [
        ("age", "ps"),
        ("age", "bs"),
        ("age", "ns"),
        ("band", "bs"),
        ("band", "cr"),
        ("band", "ns"),
    ],
)
def test_a_kind_change_waits_refits_and_undo_puts_back_the_original_fit(book, term, kind):
    """age is declared cr and band's basis ps; each switch keeps the knots in force."""
    model, X, *_ = book
    session = _session(book)
    declared = _basis(model, term)
    assert session_payload(session)[term]["knots"]["kind"] == declared[0]

    step = session.stage_structural("basis", term, {"kind": kind})
    assert step.label == f"kind {kind} in {term}"
    assert step.params == {"kind": kind, "select": False}
    payload = session_payload(session)[term]
    assert payload["pending"]["basis"] == {"kind": kind, "select": False}
    assert payload["knots"]["kind"] == declared[0]
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))

    session.refit_pending()
    assert _basis(session.model, term) == (kind, False)
    np.testing.assert_array_equal(_knots(session.model, term), _knots(model, term))
    assert _fitted(session.model, term).degree == 3
    payload = session_payload(session)[term]
    assert (payload["knots"]["kind"], payload["pending"]["basis"]) == (kind, None)

    session.undo()
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))
    assert [step.operation for step in session.pending] == ["basis"]
    session.undo()
    assert session.pending == []
    assert _basis(session.model, term) == declared
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))


@pytest.mark.parametrize("term", ["age", "band"])
def test_shrink_turns_the_double_penalty_on_and_off_one_step_each(book, term):
    model, X, *_ = book
    session = _session(book)
    assert session_payload(session)[term]["knots"]["select_available"] is True

    session.replace_with_basis(term, {"select": True})
    assert session.structure_history[-1].label == f"shrinkage on in {term}"
    assert _basis(session.model, term) == (_basis(model, term)[0], True)
    # The penalty's null space, the straight line, is penalised as a component of its own.
    assert _fitted(session.model, term)._U_null is not None
    knots = session_payload(session)[term]["knots"]
    assert (knots["select"], knots["select_available"]) == (True, True)

    session.replace_with_basis(term, {"select": False})
    assert session.structure_history[-1].label == f"shrinkage off in {term}"
    assert _basis(session.model, term) == _basis(model, term)
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))
    session.undo()
    session.undo()
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))


def test_basis_changes_refuse_in_fixed_sentences(book):
    session = _session(book)
    kind = (
        'Choose the basis kind: P-spline ("ps"), B-spline ("bs"), cubic regression ("cr") '
        'or natural ("ns").'
    )
    forms = "Give a basis kind or a shrinkage setting, one at a time."
    refusals = [
        (
            "area",
            {"kind": "ps"},
            "The basis kind is set for spline terms and ordered terms with a spline basis.",
        ),
        ("age", {"kind": "tp"}, kind),
        ("age", {"kind": "cr_cardinal"}, kind),
        ("age", {"kind": "cr"}, "'age' is already a cubic regression spline."),
        ("age", {"select": "yes"}, "Shrinkage is on (true) or off (false)."),
        ("age", {"select": False}, "Shrinkage is already off for 'age'."),
        ("age", {}, forms),
        ("age", {"kind": "ps", "select": True}, forms),
    ]
    for term, params, sentence in refusals:
        assert _refused(session, term, params) == sentence
    assert session.pending == []


def test_a_shaped_range_holds_the_kinds_that_take_ranges_and_refuses_shrinkage(book):
    session = _session(book)
    session.stage_structural("shape", "age", {"lo": 40.0, "hi": 60.0, "degree": 1})
    for kind in ("ps", "ns"):
        assert _refused(session, "age", {"kind": kind}) == (
            "Shaped ranges need a B-spline or a cubic regression spline, and 'age' has shaped "
            "ranges; choose one of those kinds, or undo the ranges first."
        )
    assert _refused(session, "age", {"select": True}) == (
        "Shrinkage cannot be combined with shaped ranges; undo the ranges of 'age' first."
    )
    knots = session_payload(session)["age"]["knots"]
    assert (knots["select_available"], knots["select_reason"]) == (
        False,
        "Shrinkage cannot be combined with shaped ranges; undo the ranges of 'age' first.",
    )
    # A B-spline takes the range: the two changes refit as one.
    session.stage_structural("basis", "age", {"kind": "bs"})
    session.refit_pending()
    spec = session.model._specs["age"]
    assert _spline_kind_name(spec) == "bs"
    assert [(r.lo, r.hi, r.degree) for r in spec.polynomial_ranges] == [(40.0, 60.0, 1)]


def test_a_natural_spline_refuses_uneven_knots_and_shrinkage_and_takes_even_knots_only(book):
    session = _session(book)
    session.stage_structural("knots", "age", {"count": 6, "strategy": "quantile"})
    assert _refused(session, "age", {"kind": "ns"}) == (
        "A natural spline's penalty needs evenly spaced knots, and the knots of 'age' are not "
        "evenly spaced; place them by even spacing first, or choose another kind."
    )
    session.pending.clear()
    session.stage_structural("basis", "age", {"kind": "ns"})
    shrink = (
        "A natural spline cannot take shrinkage. To shrink 'age', choose another kind; to make "
        "it a natural spline, turn Shrink off first."
    )
    assert _refused(session, "age", {"select": True}) == shrink
    knots = session_payload(session)["age"]["knots"]
    assert (knots["select_available"], knots["select_reason"]) == (False, shrink)
    # The Knots tool follows the waiting kind: a natural spline takes evenly spaced knots only.
    assert knots["even_only"].startswith("'age' is a natural spline")
    session.pending.clear()
    session.replace_with_basis("age", {"select": True})
    assert _refused(session, "age", {"kind": "ns"}) == shrink


def test_a_p_spline_switched_from_uneven_knots_takes_the_general_difference_penalty(book):
    session = _session(book)
    placed = session.stage_structural("knots", "age", {"count": 6, "strategy": "quantile"})
    session.stage_structural("basis", "age", {"kind": "ps"})
    session.refit_pending()
    spec = session.model._specs["age"]
    assert (_spline_kind_name(spec), spec._knot_strategy_actual) == ("ps", "quantile")
    np.testing.assert_array_equal(spec.fitted_base_knots, placed.metadata["positions"])
    general = build_general_difference_penalty(spec._knots, spec.degree, 2)
    np.testing.assert_array_equal(spec._build_penalty(), general)


def test_a_penalty_order_above_the_degree_holds_its_kind_and_refuses_shrinkage():
    X, y, w = _book(n=3000)
    model = _declared(age=Spline(kind="ps", n_knots=5, m=4)).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    for kind, name in (("bs", "B-spline"), ("cr", "cubic regression spline")):
        assert _refused(session, "age", {"kind": kind}) == (
            f"A {name}'s penalty needs a penalty order m no higher than its degree, 3, and "
            "'age' has m=4; choose a P-spline or a natural spline, or lower m in code."
        )
    shrink = (
        "Shrinkage on a P-spline needs a penalty order m of 2 or less, and 'age' has m=4; "
        "choose a cubic regression spline, or change m in code."
    )
    assert _refused(session, "age", {"select": True}) == shrink
    assert session_payload(session)["age"]["knots"]["select_reason"] == shrink
    session.replace_with_basis("age", {"kind": "ns"})
    assert _spline_kind_name(session.model._specs["age"]) == "ns"


def test_a_shape_constraint_holds_the_natural_spline_and_shrinkage_off():
    X, y, w = _book(n=3000)
    age = Spline(kind="cr", n_knots=5, constraint=Constraint.fit.decreasing)
    model = _declared(age=age).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    assert _refused(session, "age", {"kind": "ns"}) == (
        "A natural spline takes no shape constraint, and 'age' has one; choose another kind, "
        "or remove the constraint in code."
    )
    assert _refused(session, "age", {"select": True}) == (
        "Shrinkage cannot be combined with a shape constraint the fit enforces, which 'age' "
        "has; leave Shrink off, or apply the constraint after the fit (Constraint.postfit) in "
        "code."
    )
    # Another kind keeps the constraint, mapped back to its public token.
    session.replace_with_basis("age", {"kind": "bs"})
    spec = session.model._specs["age"]
    assert (_spline_kind_name(spec), spec.constraint_kind, spec.constraint_mode) == (
        "bs",
        "decreasing",
        "fit",
    )


def test_a_level_change_waiting_on_an_ordered_term_holds_its_basis(book):
    session = _session(book)
    session.stage_structural("collapse", "band", {"levels": ["B6", "B7"]})
    sentence = "A waiting change on 'band' changes its levels; refit it before changing the basis."
    assert _refused(session, "band", {"kind": "cr"}) == sentence
    knots = session_payload(session)["band"]["knots"]
    assert (knots["select_available"], knots["select_reason"]) == (False, sentence)


def test_a_term_used_by_an_interaction_keeps_its_basis():
    X, y, w = _book(n=3000)
    model = SuperGLM(
        family="poisson",
        features={"age": Spline(kind="cr", n_knots=5), "area": Categorical()},
        interactions=[("age", "area")],
        spline_penalty=10.0,
    ).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    knots = session_payload(session)["age"]["knots"]
    assert (knots["kind"], knots["kinds"], knots["select_available"]) == (None, [], False)
    assert _refused(session, "age", {"kind": "ps"}).startswith(
        "Cannot change the basis for term 'age' because it is used by interaction(s): "
    )


def test_a_cardinal_spline_is_named_not_offered_and_still_takes_shrinkage():
    X, y, w = _book(n=3000)
    model = _declared(age=Spline(kind="cr_cardinal", n_knots=5)).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    knots = session_payload(session)["age"]["knots"]
    assert (knots["kind"], knots["kinds"]) == ("cr_cardinal", ["ps", "bs", "cr", "ns"])
    session.replace_with_basis("age", {"select": True})
    assert _basis(session.model, "age") == ("cr_cardinal", True)


@pytest.mark.parametrize(
    ("term", "params", "basis"),
    [
        ("age", {"kind": "ps"}, {"kind": "ps", "select": False}),
        ("age", {"select": True}, {"kind": "cr", "select": True}),
        ("band", {"kind": "cr"}, {"kind": "cr", "select": False}),
        ("band", {"select": True}, {"kind": "ps", "select": True}),
    ],
)
def test_the_structure_export_records_the_basis_and_applies_it_to_the_declaration(
    book, term, params, basis
):
    model, X, y, w = book
    session = _session(book)
    session.replace_with_basis(term, params)
    exported = json.loads(session.export_structure())
    assert exported["features"][term]["basis"] == basis
    applied = read_structure(exported).apply(_declared()).fit(X, y, sample_weight=w)
    assert _basis(applied, term) == (basis["kind"], basis["select"])
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))
    # The code's own basis is not recorded.
    assert "basis" not in json.loads(_session(book).export_structure())["features"][term]


def test_a_p_spline_shaped_after_its_kind_changed_keeps_the_record_of_its_basis(book):
    """A shaped range makes a P-spline a B-spline; the file must say so, or apply keeps cr."""
    model, X, y, w = book
    session = _session(book)
    session.replace_with_basis("age", {"kind": "ps"})
    session.stage_structural("shape", "age", {"lo": 40.0, "hi": 60.0, "degree": 1})
    assert session_payload(session)["age"]["pending"]["basis"] == {"kind": "bs", "select": False}
    session.refit_pending()
    exported = json.loads(session.export_structure())
    assert exported["features"]["age"]["basis"] == {"kind": "bs", "select": False}
    applied = read_structure(exported).apply(_declared(), X=X).fit(X, y, sample_weight=w)
    assert _spline_kind_name(applied._specs["age"]) == "bs"
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))


def test_a_cardinal_spline_made_cubic_regression_takes_ranges_through_a_structure_file():
    X, y, w = _book(n=3000)

    def declared():
        return _declared(age=Spline(kind="cr_cardinal", n_knots=5))

    model = declared().fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    session.replace_with_basis("age", {"kind": "cr"})
    session.replace_with_shaped_range("age", lo=40.0, hi=60.0, degree=1)
    exported = json.loads(session.export_structure())
    applied = read_structure(exported).apply(declared(), X=X).fit(X, y, sample_weight=w)
    assert _spline_kind_name(applied._specs["age"]) == "cr"
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))


def test_a_p_spline_or_b_spline_takes_the_degree_declared_in_code():
    """bs(degree=2) made cr is cubic; made ps again it is quadratic, as the file applies it."""
    X, y, w = _book(n=3000)

    def declared():
        return _declared(age=Spline(kind="bs", degree=2, n_knots=5))

    model = declared().fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    session.stage_structural("basis", "age", {"kind": "cr"})
    session.stage_structural("basis", "age", {"kind": "ps"})
    session.refit_pending()
    spec = session.model._specs["age"]
    assert (_spline_kind_name(spec), spec.degree) == ("ps", 2)
    exported = json.loads(session.export_structure())
    applied = read_structure(exported).apply(declared()).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))


def test_the_widget_stages_a_basis_change_and_refits_one_at_once(book):
    session = _session(book)
    widget = session.widget()
    try:
        staged = _post_json(
            f"{widget.url}/stage",
            {"operation": "basis", "term": "age", "params": {"kind": "bs"}},
        )
        assert staged["state"]["terms"]["age"]["pending"]["basis"] == {
            "kind": "bs",
            "select": False,
        }
        assert staged["state"]["pending"][0]["label"] == "kind bs in age"
        session.pending.clear()
        refit = _post_json(f"{widget.url}/basis", {"term": "band", "params": {"select": True}})
        assert refit["state"]["terms"]["band"]["knots"]["select"] is True
        assert _basis(session.model, "band") == ("ps", True)
        with pytest.raises(urllib.error.HTTPError) as refused:
            _post_json(f"{widget.url}/basis", {"term": "age", "params": {"kind": "cr"}})
        assert json.loads(refused.value.read().decode("utf-8"))["error"] == (
            "'age' is already a cubic regression spline."
        )
    finally:
        widget.close()
    session.undo()
    assert _basis(session.model, "band") == ("ps", False)
