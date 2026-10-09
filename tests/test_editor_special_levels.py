"""Special levels of an ordered term in the editor: Make special, Back on the curve, free levels."""

from __future__ import annotations

import json
import pickle
import urllib.error

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Piecewise, Spline, SuperGLM, collapse_levels
from superglm.editor import EditorSession
from superglm.editor import free_levels as free_levels_module
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.editor.free_levels import free_level_comparison
from superglm.editor.payloads import session_payload
from superglm.features.rebuild import (
    clone_with_replaced_features,
    freed_levels,
    full_level_order,
    rebuilt_ordered_spec,
)
from tests.test_editor import _post_json

BANDS = [f"Mi{6 * (i + 1):03d}" for i in range(12)]
BUMP = "Mi036"


def _book(seed: int = 20261009, n: int = 12000, bump: float = 0.5):
    """A Poisson book whose band effect rises steadily, with a bump at Mi036 the smooth spreads."""
    rng = np.random.default_rng(seed)
    band = rng.choice(BANDS, n)
    area = rng.choice(["A", "B", "C"], n)
    exposure = rng.uniform(0.2, 1.0, n)
    index = np.array([BANDS.index(b) for b in band])
    eta = -1.5 + 0.05 * index + bump * (band == BUMP) + 0.2 * (area == "B")
    y = rng.poisson(exposure * np.exp(eta)) / exposure
    return pd.DataFrame({"band": band, "area": area}), y, exposure


def _declared(basis=None, *, selection_penalty=None, specials=None) -> SuperGLM:
    band = OrderedCategorical(
        order=BANDS if specials is None else [*BANDS, *specials],
        basis=Spline(kind="ps", n_knots=6) if basis is None else basis,
        specials=specials,
    )
    return SuperGLM(
        family="poisson",
        features={"band": band, "area": Categorical()},
        spline_penalty=20.0,
        selection_penalty=selection_penalty,
    )


@pytest.fixture(scope="module")
def book():
    X, y, w = _book()
    return _declared().fit(X, y, sample_weight=w), X, y, w


def _session(book) -> EditorSession:
    model, X, y, w = book
    return EditorSession.from_model(model, train_data=(X, y, w))


def _rebuilt(spec, X, **changes):
    """``spec`` rebuilt with ``changes``, keeping the reference its fit resolved, or its draft names."""
    return rebuilt_ordered_spec(
        spec,
        grouping=None,
        base=spec._base_level or spec.base,
        data=X["band"].to_numpy(),
        level=True,
        **changes,
    )


@pytest.mark.parametrize(
    "basis", [Spline(kind="ps", n_knots=6), Piecewise(breaks=["Mi030", "Mi048"])]
)
def test_levels_taken_off_the_curve_go_back_to_their_places_and_the_fit_returns(basis):
    """Both freed levels go back in order, whichever goes back first, on a spline or a positional axis.

    A Piecewise axis numbers its bands 0..L-1 again on every build, so a
    freed level cannot keep its old number: it goes back after the nearest
    level before it that is on the curve.
    """
    X, y, w = _book(n=6000)
    model = _declared(basis).fit(X, y, sample_weight=w)
    spec = model._specs["band"]
    one = clone_with_replaced_features(model, {"band": _rebuilt(spec, X, freed=("Mi036",))})
    one.fit(X, y, sample_weight=w)
    two = _rebuilt(one._specs["band"], X, freed=("Mi042",))
    assert list(two._special_display) == ["Mi036", "Mi042"]
    assert full_level_order(two) == BANDS
    for first, second in (("Mi036", "Mi042"), ("Mi042", "Mi036")):
        back = _rebuilt(_rebuilt(two, X, returned=(first,)), X, returned=(second,))
        assert list(back._declared_smooth_levels) == BANDS
        assert not freed_levels(back)
        refit = clone_with_replaced_features(model, {"band": back}).fit(X, y, sample_weight=w)
        np.testing.assert_array_equal(refit.predict(X), model.predict(X))


def test_a_freed_level_keeps_its_place_through_a_fit_and_a_pickle(book):
    model, X, y, w = book
    freed = _rebuilt(model._specs["band"], X, freed=(BUMP,))
    refit = clone_with_replaced_features(model, {"band": freed}).fit(X, y, sample_weight=w)
    kept = pickle.loads(pickle.dumps(refit))._specs["band"]
    assert freed_levels(kept) == {
        BUMP: (freed_levels(freed)[BUMP][0], ("Mi030", "Mi024", "Mi018", "Mi012", "Mi006"))
    }
    assert full_level_order(kept) == BANDS


def test_make_special_waits_refits_undoes_and_puts_back_the_original_fit(book):
    model, X, *_ = book
    session = _session(book)
    session.stage_structural("special", "band", {"levels": [BUMP]})
    term = session_payload(session)["band"]
    assert term["pending"]["specials"] == [BUMP]
    assert term["shape"]["specials"] == []
    assert session.pending[0].label == f"make {BUMP} special in band"

    session.refit_pending()
    assert list(session.model._specs["band"]._special_display) == [BUMP]
    shape = session_payload(session)["band"]["shape"]
    assert (shape["specials"], shape["returnable"]) == ([BUMP], [BUMP])
    # The level keeps its own estimate, off the curve: the bump the smooth spread.
    term = session.terms["band"]
    estimate = term.original_log_effect[list(term.levels).index(BUMP)]
    neighbour = term.original_log_effect[list(term.levels).index("Mi030")]
    assert estimate - neighbour > 0.3

    session.undo()
    assert list(session.model._specs["band"]._special_display) == []
    assert [step.operation for step in session.pending] == ["special"]
    session.redo()
    assert list(session.model._specs["band"]._special_display) == [BUMP]

    session.replace_with_special_levels("band", [BUMP], special=False)
    assert list(session.model._specs["band"]._special_display) == []
    np.testing.assert_array_equal(session.model.predict(X), model.predict(X))


def test_the_structure_export_records_the_special_and_applies_it_to_the_declaration(book):
    model, X, y, w = book
    session = _session(book)
    session.replace_with_special_levels("band", [BUMP])
    entry = json.loads(session.export_structure())["features"]["band"]
    assert entry["specials"] == [BUMP]
    assert entry["levels"] == BANDS
    from superglm import read_structure

    applied = read_structure(json.loads(session.export_structure())).apply(_declared())
    applied.fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(applied.predict(X), session.model.predict(X))


def test_make_special_and_back_on_the_curve_refuse_in_fixed_sentences(book):
    session = _session(book)
    reference = str(session.model._specs["band"]._base_level)
    refusals = [
        (
            "special",
            [reference],
            f"{reference!r} is the reference of 'band', which must stay on the curve; "
            "set another reference first.",
        ),
        (
            "special",
            [level for level in BANDS if level != reference],
            "'band' needs at least two levels on its curve; make fewer levels special.",
        ),
        ("special", ["Mi999"], "'Mi999' is not a level of term 'band'."),
        ("on_curve", [BUMP], f"{BUMP!r} is on the curve of 'band' already."),
    ]
    for operation, levels, sentence in refusals:
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural(operation, "band", {"levels": levels})
        assert str(refused.value) == sentence
    # The refit's own rows decide: these hold no Mi072.
    X = book[1]
    with pytest.raises(EditorValueError) as refused:
        session.stage_structural(
            "special", "band", {"levels": ["Mi072"]}, X=X[X["band"] != "Mi072"]
        )
    assert str(refused.value) == (
        "'Mi072' has no rows in the data the refit reads, so it has nothing to estimate a free "
        "value from."
    )
    assert session.pending == []


def test_a_declared_special_cannot_go_on_the_curve_and_a_grouped_level_cannot_leave_it():
    X, y, w = _book(n=6000)
    X.loc[X.index[:300], "band"] = "MISSING"
    model = _declared(specials=["MISSING"]).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    with pytest.raises(EditorValueError) as declared:
        session.stage_structural("on_curve", "band", {"levels": ["MISSING"]})
    assert str(declared.value) == (
        "'MISSING' is declared special in 'band', so it has no place on the curve; "
        "declare it in the term's order to put it there."
    )
    session.stage_structural("collapse", "band", {"levels": ["Mi060", "Mi066"]})
    session.refit_pending()
    with pytest.raises(EditorValueError) as grouped:
        session.stage_structural("special", "band", {"levels": ["Mi060"]})
    assert str(grouped.value) == "'Mi060' is in group 'Mi060+Mi066' of 'band'; ungroup it first."
    with pytest.raises(EditorTypeError):
        session.stage_structural("special", "area", {"levels": ["A"]})


def test_a_level_returning_between_members_of_one_group_is_refused():
    """Mi036 leaves, Mi030 and Mi042 are then neighbours and are grouped: Mi036 would land inside."""
    X, y, w = _book(n=6000)
    session = EditorSession.from_model(_declared().fit(X, y, sample_weight=w), train_data=(X, y, w))
    session.replace_with_special_levels("band", [BUMP])
    session.stage_structural("collapse", "band", {"levels": ["Mi030", "Mi042"]})
    session.refit_pending()
    with pytest.raises(EditorValueError) as inside:
        session.stage_structural("on_curve", "band", {"levels": [BUMP]})
    assert str(inside.value) == (
        f"'{BUMP}' would go back between members of group 'Mi030+Mi042' of 'band'; ungroup it first."
    )


def test_free_levels_are_a_plain_categorical_fit_and_flag_the_level_the_smooth_overrides():
    """A 0.3 bump the smooth spreads is flagged; judged against the free interval alone, it is not.

    The curve has moved toward the bump's own data, so the gap is the
    smoother's residual: on this book its standardised value is 3.3 against
    the Sidak cut of 2.86, and the free estimate's own interval alone gives
    2.4, which misses it.
    """
    X, y, w = _book(bump=0.3)
    model = _declared().fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    free = free_level_comparison(session, "band")
    reference = model._specs["band"]._base_level
    direct = SuperGLM(
        family="poisson",
        features={"band": Categorical(base=reference), "area": Categorical()},
        spline_penalty=20.0,
    ).fit(X, y, sample_weight=w)
    inference = direct.term_inference("band")
    expected = dict(zip(inference.levels, np.exp(inference.log_relativity), strict=True))
    # The free levels are the plain categorical's relativities. Each fit stops
    # once its deviance moves by under tol = 1e-6 relative, which leaves the
    # coefficients settled to about sqrt(tol).
    np.testing.assert_allclose(free["y"], [expected[level] for level in free["levels"]], rtol=1e-3)
    assert free["levels"] == BANDS
    assert free["flagged"] == [BUMP]
    assert free["shrunk"] is False
    term = session.terms["band"]
    curve = np.exp(term.original_log_effect[list(term.levels).index(BUMP)])
    at = free["levels"].index(BUMP)
    assert not free["lower"][at] <= curve <= free["upper"][at]


def test_free_levels_lift_a_selection_penalty_from_the_term_only():
    X, y, w = _book(n=6000)
    plain = _declared().fit(X, y, sample_weight=w)
    selected = _declared(selection_penalty=0.5).fit(X, y, sample_weight=w)
    free_plain = free_level_comparison(
        EditorSession.from_model(plain, train_data=(X, y, w)), "band"
    )
    session = EditorSession.from_model(selected, train_data=(X, y, w))
    free_selected = free_level_comparison(session, "band")
    # Both are the same unpenalised fit, settled to about sqrt(tol) (see above).
    np.testing.assert_allclose(free_selected["y"], free_plain["y"], rtol=1e-3)
    assert selected.penalty.features is None


def test_free_levels_refuse_a_categorical_term_and_a_session_without_its_data(book):
    session = _session(book)
    with pytest.raises(EditorTypeError):
        free_level_comparison(session, "area")
    X, y, w = _book(n=3000)
    bare = SuperGLM(
        family="poisson",
        features={"band": OrderedCategorical(order=BANDS), "area": Categorical()},
        retain_fit_state=False,
    ).fit(X, y, sample_weight=w)
    with pytest.raises(EditorValueError) as missing:
        free_level_comparison(EditorSession.from_model(bare), "band")
    assert str(missing.value) == free_levels_module._NO_DATA


def test_the_widget_fits_free_levels_once_per_model_revision(book, monkeypatch):
    calls = []
    real = free_levels_module.free_level_comparison

    def counted(session, name):
        calls.append(name)
        return real(session, name)

    monkeypatch.setattr(free_levels_module, "free_level_comparison", counted)
    session = _session(book)
    widget = session.widget()
    try:
        first = _post_json(f"{widget.url}/free_levels", {"term": "band"})
        second = _post_json(f"{widget.url}/free_levels", {"term": "band"})
        assert first == second and first["model_revision"] == session.model_revision
        assert calls == ["band"]
        _post_json(f"{widget.url}/special_levels", {"term": "band", "levels": [BUMP]})
        assert list(session.model._specs["band"]._special_display) == [BUMP]
        third = _post_json(f"{widget.url}/free_levels", {"term": "band"})
        assert calls == ["band", "band"] and BUMP not in third["levels"]
        with pytest.raises(urllib.error.HTTPError) as refused:
            _post_json(
                f"{widget.url}/special_levels", {"term": "band", "levels": [BUMP], "special": "yes"}
            )
        assert (
            json.loads(refused.value.read().decode("utf-8"))["error"]
            == "special must be true or false."
        )
    finally:
        widget.close()


def test_a_grouped_term_compares_each_member_with_its_group(book):
    _model, X, y, w = book
    grouping = collapse_levels(X["band"], groups={"Mi060-66": ["Mi060", "Mi066"]}, order=BANDS)
    declared = SuperGLM(
        family="poisson",
        features={
            "band": OrderedCategorical(
                order=BANDS, basis=Spline(kind="ps", n_knots=6), grouping=grouping
            ),
            "area": Categorical(),
        },
        spline_penalty=20.0,
    ).fit(X, y, sample_weight=w)
    free = free_level_comparison(EditorSession.from_model(declared, train_data=(X, y, w)), "band")
    at = {level: k for k, level in enumerate(free["levels"])}
    assert free["y"][at["Mi060"]] == free["y"][at["Mi066"]]
    assert free["levels"] == BANDS
