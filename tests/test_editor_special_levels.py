"""Special levels of an ordered term in the editor: Make special, Back on the curve, free levels."""

from __future__ import annotations

import json
import math
import pickle
import urllib.error

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    OrderedCategorical,
    Piecewise,
    Polynomial,
    Spline,
    SuperGLM,
    collapse_levels,
)
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
    session.stage_structural("special", "band", {"levels": [BUMP]})
    for operation, levels, sentence in [
        ("special", [BUMP], f"{BUMP!r} is already a special level of 'band'."),
        ("special", [], "Select the levels to make special."),
    ]:
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural(operation, "band", {"levels": levels})
        assert str(refused.value) == sentence
    session.pending.clear()
    # The refit's own rows and weights decide: these hold no Mi072, then none
    # of positive weight, then the session's rows under the caller's weights.
    X, _y, w = book[1:]
    sentence = (
        "'Mi072' has no rows of positive weight in the data the refit reads, so it has nothing "
        "to estimate a free value from."
    )
    for rows, weights in (
        (X[X["band"] != "Mi072"], None),
        (X, np.where(X["band"] == "Mi072", 0.0, w)),
        (None, np.where(X["band"] == "Mi072", 0.0, w)),
    ):
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural(
                "special", "band", {"levels": ["Mi072"]}, X=rows, sample_weight=weights
            )
        assert str(refused.value) == sentence
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


def test_a_level_its_grouping_renames_is_refused_in_a_sentence():
    """A group of one under another name: rebuilt special, the term refused it as a 500."""
    X, y, w = _book(n=3000)
    grouping = collapse_levels(X["band"], groups={"Bumped": [BUMP]}, order=BANDS)
    band = OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=6), grouping=grouping)
    model = SuperGLM(family="poisson", features={"band": band}, spline_penalty=20.0)
    session = EditorSession.from_model(model.fit(X, y, sample_weight=w), train_data=(X, y, w))
    with pytest.raises(EditorValueError) as renamed:
        session.stage_structural("special", "band", {"levels": [BUMP]})
    assert str(renamed.value) == (
        f"{BUMP!r} is named 'Bumped' by the grouping of 'band', and a special level keeps its "
        f"own name: take {BUMP!r} out of that grouping where the model is declared first."
    )


def test_make_special_refuses_breaks_positional_breaks_and_a_basis_left_too_small():
    X, y, w = _book(n=3000)
    cases = [
        (
            Piecewise(breaks=["Mi030", "Mi048"]),
            ["Mi030"],
            "'band' has a break, knot or shaped-range edge at 'Mi030', so it can't leave the "
            "curve; move or remove it first.",
        ),
        (
            Piecewise(breaks=[4, 7]),
            [BUMP],
            "'band' states its breaks by position, which a level leaving the curve would move; "
            "state them by band name to make a level special.",
        ),
        (
            Polynomial(powers=list(range(1, 12))),
            [BUMP],
            "Taking those levels off the curve of 'band' leaves its basis too few levels for its "
            "degree; make fewer levels special, or lower the degree in code.",
        ),
        (
            Piecewise(breaks=["Mi018"], degrees=[2, 1]),
            ["Mi012"],
            "Taking those levels off the curve of 'band' leaves its basis too few levels for its "
            "degree; make fewer levels special, or lower the degree in code.",
        ),
    ]
    for basis, levels, sentence in cases:
        model = _declared(basis).fit(X, y, sample_weight=w)
        session = EditorSession.from_model(model, train_data=(X, y, w))
        with pytest.raises(EditorValueError) as refused:
            session.stage_structural("special", "band", {"levels": levels})
        assert str(refused.value) == sentence


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
    """A 0.3 bump the smooth spreads is flagged, and no other level.

    The curve has moved toward the bump's own data, so the gap is the
    smoother's residual: on this book it is 4.9 standard errors against the
    Sidak cut of 2.86.
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
    # The free levels are the plain categorical's relativities, placed against
    # the curve by the levels' mean rather than by the reference, so they agree
    # up to one factor. Each fit stops once its deviance moves by under
    # tol = 1e-6 relative. Near the optimum the deviance is quadratic in the
    # coefficients' error, so that step bounds the error before it by about
    # sqrt(tol); Newton's last step leaves it far smaller, so 1e-3 is loose.
    ratio = np.log(free["y"]) - np.log([expected[level] for level in free["levels"]])
    assert np.ptp(ratio) < 1e-3
    assert free["levels"] == BANDS
    assert free["flagged"] == [BUMP]
    assert free["shrunk"] is False
    term = session.terms["band"]
    curve = np.exp(term.original_log_effect[list(term.levels).index(BUMP)])
    at = free["levels"].index(BUMP)
    assert not free["lower"][at] <= curve <= free["upper"][at]


def _fitted_free_models(monkeypatch) -> list:
    """Record each model the comparison fits."""
    fitted = []
    real = free_levels_module.fit_refit_model

    def recorded(source, refit, **kwargs):
        fitted.append(refit)
        return real(source, refit, **kwargs)

    monkeypatch.setattr(free_levels_module, "fit_refit_model", recorded)
    return fitted


@pytest.mark.parametrize(
    ("declared_features", "kept"),
    [(None, frozenset({"area"})), (["band", "area"], frozenset({"area"})), (["band"], None)],
)
def test_free_levels_lift_a_selection_penalty_from_the_term_only(
    monkeypatch, declared_features, kept
):
    X, y, w = _book(n=6000)
    selected = SuperGLM(
        family="poisson",
        features={"band": OrderedCategorical(order=BANDS), "area": Categorical()},
        selection_penalty=0.5,
        penalty_features=declared_features,
    ).fit(X, y, sample_weight=w)
    fitted = _fitted_free_models(monkeypatch)
    free = free_level_comparison(EditorSession.from_model(selected, train_data=(X, y, w)), "band")
    penalty = fitted[0].penalty
    # The other terms keep the penalty, at its strength; with none left to
    # penalise, it is off.
    assert penalty.lambda1 == (0.5 if kept else None)
    if kept:
        assert penalty.features == kept
    assert free["shrunk"] is False
    # Left on, the penalty would have shrunk the free levels.
    monkeypatch.setattr(free_levels_module, "_lift_selection", lambda *_args: False)
    shrunk = free_level_comparison(EditorSession.from_model(selected, train_data=(X, y, w)), "band")
    assert np.max(np.abs(np.log(shrunk["y"]) - np.log(free["y"]))) > 1e-2


def test_a_penalty_that_cannot_be_restricted_is_reported_as_shrinking_the_free_levels():
    class Unrestricted:
        lambda1 = 0.5

    class Model:
        penalty = Unrestricted()

    assert free_levels_module._lift_selection(Model(), Model(), "band") is True


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


def test_the_widget_fits_free_levels_once_per_fit_in_force(book, monkeypatch):
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
        token = widget._state()["fit_token"]
        assert first["fit_token"] == token
        # A hand edit changes neither fit: the comparison and its token stand.
        session.select_levels("band", ["Mi042"])
        session.shift("band", 0.05)
        second = _post_json(f"{widget.url}/free_levels", {"term": "band"})
        assert first == second and widget._state()["fit_token"] == token
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


def test_a_level_freed_before_a_collapse_returns_to_its_place():
    """Free Mi036, collapse Mi060+Mi066, put Mi036 back: as if only the collapse were made.

    The collapse's grouping lists the special last, and the bands of a grouped
    term follow its grouping: Mi036 must go back between Mi030 and Mi042.
    """
    X, y, w = _book(n=3000)
    for basis in (Spline(kind="ps", n_knots=6), Piecewise(breaks=["Mi030", "Mi048"])):
        model = _declared(basis).fit(X, y, sample_weight=w)
        session = EditorSession.from_model(model, train_data=(X, y, w))
        session.replace_with_special_levels("band", [BUMP])
        session.stage_structural("collapse", "band", {"levels": ["Mi060", "Mi066"]})
        session.refit_pending()
        session.replace_with_special_levels("band", [BUMP], special=False)
        only = EditorSession.from_model(model, train_data=(X, y, w))
        only.stage_structural("collapse", "band", {"levels": ["Mi060", "Mi066"]})
        only.refit_pending()
        spec = session.model._specs["band"]
        assert list(spec._smooth_levels) == list(only.model._specs["band"]._smooth_levels)
        grouping = only.model._specs["band"]._grouping
        assert spec._grouping.all_original_levels == grouping.all_original_levels
        assert list(session.terms["band"].levels) == list(only.terms["band"].levels)
        np.testing.assert_array_equal(session.model.predict(X), only.model.predict(X))
        # Its neighbours are its neighbours again, for a collapse.
        session.stage_structural("collapse", "band", {"levels": [BUMP, "Mi042"]})
        session.pending.clear()
        with pytest.raises(EditorValueError, match="must be contiguous"):
            session.stage_structural("collapse", "band", {"levels": ["Mi072", BUMP]})


def test_a_curve_that_misfits_its_reference_flags_the_misfit_not_every_level():
    """A sharp first band beside the most exposed one, which is the reference.

    The smooth cannot drop that fast, so it misfits the reference itself.
    Measured from the reference, every other level's gap carries that misfit
    and nearly all were flagged; on freMTPL2's vehicle age, 19 of 20. Centred
    on the levels' mean, only the bands at the drop are.
    """
    rng = np.random.default_rng(20261009)
    n = 20000
    band = rng.choice(BANDS, n, p=[0.06, 0.3] + [0.064] * 10)
    w = rng.uniform(0.2, 1.0, n)
    index = np.array([BANDS.index(b) for b in band])
    y = rng.poisson(w * np.exp(-1.5 + 0.8 * (index == 0) - 0.03 * index)) / w
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="poisson",
        features={"band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=6))},
        spline_penalty=20.0,
    ).fit(X, y, sample_weight=w)
    assert model._specs["band"]._base_level == "Mi012"
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y, w)), "band")
    assert free["flagged"]
    assert set(free["flagged"]) <= set(BANDS[:4])


def _captured_gaps(monkeypatch) -> list:
    """Every comparison's ``_Gaps``, in order, as Free levels computes them."""
    found = []
    gaps = free_levels_module._gaps
    monkeypatch.setattr(
        free_levels_module, "_gaps", lambda *args: found.append(gaps(*args)) or found[-1]
    )
    return found


def test_both_fits_are_centred_on_one_exposure_weighted_mean(monkeypatch):
    """A declared band with no rows is on the curve but not in the free fit.

    Centring the curve over it as well moved every gap by one amount. Both
    sides are centred on the compared levels' exposure-weighted mean, so the
    gaps' weighted sum is zero.
    """
    found = _captured_gaps(monkeypatch)
    X, y, w = _book(n=6000)
    values = {band: float(i) for i, band in enumerate([*BANDS, "Mi078"])}
    model = SuperGLM(
        family="poisson",
        features={
            "band": OrderedCategorical(values=values, basis=Spline(kind="ps", n_knots=6)),
            "area": Categorical(),
        },
        spline_penalty=20.0,
    ).fit(X, y, sample_weight=w)
    session = EditorSession.from_model(model, train_data=(X, y, w))
    free = free_level_comparison(session, "band")
    assert free["levels"] == BANDS == found[-1].labels
    share = found[-1].share
    # Each level's exposure: the code sums a level's rows in one pass, then
    # the levels, then divides; the test's sums are correctly rounded.
    u = np.finfo(np.float64).eps / 2
    exposure = np.array([math.fsum(w[X["band"].to_numpy() == level]) for level in BANDS])
    np.testing.assert_allclose(share, exposure / math.fsum(exposure), rtol=(len(X) + 16) * u)
    term = session.terms["band"]
    curve = {level: term.original_log_effect[i] for i, level in enumerate(term.levels)}
    gaps = np.array([np.log(free["y"][k]) - curve[level] for k, level in enumerate(BANDS)])
    # Each gap is (f_k - share.f) - (c_k - share.c), carried through exp and log
    # on the chart's scale. Each weighted mean errs by at most gamma_L of its
    # largest value and is shared by every gap, whose shares sum to one; each
    # gap takes six more roundings, and each product of the sum one more.
    L = len(gaps)
    scale = max(
        np.max(np.abs(np.log(free["y"]))),
        np.max(np.abs(term.original_log_effect)),
        np.max(np.abs(found[-1].free)),
        np.max(np.abs(found[-1].curve)),
    )
    assert abs(math.fsum(share * gaps)) <= (2 * L + 8) * u * scale


def test_a_curve_whose_covariance_is_stale_is_taken_as_fixed(monkeypatch):
    """An export with hand edits baked in keeps the covariance of the fit before them.

    ``term_inference`` gives such a curve no errors, and Free levels takes it
    as fixed: its intervals come from the free fit alone, so two curves on the
    same data, smoothed differently, get the same widths. Fitted, they differ.
    """
    X, y, w = _book(n=6000)
    widths = {}
    for penalty in (20.0, 2.0):
        model = SuperGLM(
            family="poisson",
            features={
                "band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=6)),
                "area": Categorical(),
            },
            spline_penalty=penalty,
        ).fit(X, y, sample_weight=w)
        for stale in (False, True):
            model._editor_inference_stale = stale
            free = free_level_comparison(
                EditorSession.from_model(model, train_data=(X, y, w)), "band"
            )
            ends = np.log(np.array([free["lower"], free["upper"]]))
            widths[penalty, stale] = (ends[1] - ends[0], np.max(np.abs(ends)))
    # Each end is one exp and one log away from its log value.
    u = np.finfo(np.float64).eps / 2
    for stale, same in ((True, True), (False, False)):
        (first, scale), (second, other) = widths[20.0, stale], widths[2.0, stale]
        tolerance = 8 * u * (1.0 + max(scale, other))
        assert bool(np.max(np.abs(first - second)) <= tolerance) == same
    # A shape repair after the fit leaves its covariance as stale: the same path.
    model._editor_inference_stale = False
    monkeypatch.setattr(free_levels_module, "_shape_repaired", lambda model, name: True)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y, w)), "band")
    ends = np.log(np.array([free["lower"], free["upper"]]))
    np.testing.assert_array_equal(ends[1] - ends[0], widths[2.0, True][0])


@pytest.mark.parametrize("family", ["poisson", "gamma"])
def test_the_gap_variance_is_both_fits_linearised_on_the_same_rows(monkeypatch, family):
    """Bands whose means the curve cannot follow, beside a second term, under unequal weights.

    The free estimate's variance less the curve's is the gap's only for a
    curve fitted with the free fit's weights: on Poisson bands it put one
    gap's variance at half its value, and flagged the band. A Gamma/log fit
    moves with its observed curvature, not the expected one, which misplaced
    a gap's variance by 12%. Both are checked against central differences of
    complete refits in every band-by-area cell's weighted response total.
    """
    found = _captured_gaps(monkeypatch)
    rng = np.random.default_rng(20261009)
    bands = TWELVE[:8]
    band = np.repeat(np.arange(8), 50)
    area = np.tile(np.repeat([0, 1], 25), 8)
    cell = 2 * band + area
    w = rng.uniform(0.5, 1.5, band.size)
    mu = np.exp(np.array([0.0, 1.2, 3.0, 0.5, -0.5, -0.7, 0.8, 1.1])[band] + 0.3 * area)
    y = rng.poisson(w * mu) / w if family == "poisson" else mu * rng.gamma(2.0, 0.5, band.size)
    X = pd.DataFrame({"band": np.array(bands)[band], "area": np.where(area, "B", "A")})

    def compared(y):
        curve = OrderedCategorical(order=bands, basis=Spline(kind="ps", n_knots=4))
        model = SuperGLM(
            family=family,
            link="log",
            features={"band": curve, "area": Categorical()},
            spline_penalty=5.0,
            tol=1e-11,
        ).fit(X, y, sample_weight=w)
        free_level_comparison(EditorSession.from_model(model, train_data=(X, y, w)), "band")
        return found[-1].free - found[-1].curve

    compared(y)
    gaps = found[-1]
    totals = np.bincount(cell, weights=w * y)
    h = 1e-2 * totals.min()
    jacobian = np.empty((len(gaps.labels), 16))
    for c in range(16):
        # The fits read a cell through its weighted total alone, so the step
        # goes on its positive rows, which no step takes below zero.
        moved = (cell == c) & (y > 0.0)
        step = np.where(moved, h / np.sum(w[moved]), 0.0)
        jacobian[:, c] = (compared(y + step) - compared(y - step)) / (2 * h)
    # A cell's weighted total varies as phi V(mu) times the cell's weight, under
    # the free fit, here refitted on its own.
    free = SuperGLM(
        family=family,
        link="log",
        features={"band": Categorical(), "area": Categorical()},
        tol=1e-11,
    ).fit(X, y, sample_weight=w)
    variance = free.result.phi * np.bincount(
        cell, weights=w * free._distribution.variance(free.predict(X))
    )
    # Central differences err by (h / T)^2 / 3 = 3e-5 of each entry, and the
    # refits' stopping error at tol = 1e-11 enters near 1e-4 (tightening tol
    # from 1e-8 to 1e-13 moves it from 2e-3 to 4e-5). 1e-3 sits between that
    # and the defects: a factor 2 on Poisson, 12% on Gamma/log.
    np.testing.assert_allclose(gaps.gap_var, (jacobian**2) @ variance, rtol=1e-3)


def test_free_levels_put_both_variances_on_one_dispersion():
    """A Gaussian bump the smooth flattens: the two fits estimate very different dispersions."""
    rng = np.random.default_rng(43)
    k = np.repeat(np.arange(12), 50)
    noise = rng.normal(0.0, 1.0, k.size)
    noise -= np.bincount(k, noise)[k] / 50
    y = 10.0 + 0.05 * k + 10.0 * (k == 5) + noise
    X = pd.DataFrame({"band": np.array(BANDS)[k]})
    model = SuperGLM(
        family="gaussian",
        features={"band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=6))},
        spline_penalty=20.0,
    ).fit(X, y)
    # The free fit's dispersion is near 1, the smooth's near 6.5: on their own
    # scales the curve's variance exceeds the free level's and nothing is judged.
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert BANDS[5] in free["flagged"]


def test_free_levels_read_the_column_through_the_declaration():
    """A column of 1.0..8.0 against order=[1..8]: the levels are the declaration's."""
    order = list(range(1, 9))
    rng = np.random.default_rng(3)
    band = rng.choice(np.arange(1.0, 9.0), 2000)
    y = rng.poisson(np.exp(-1 + 0.1 * band + 0.6 * (band == 4.0))).astype(float)
    X = pd.DataFrame({"band": band})
    for grouping in (None, collapse_levels(X["band"], groups={"6-7": ["6.0", "7.0"]})):
        model = SuperGLM(
            family="poisson",
            features={"band": OrderedCategorical(order=order, grouping=grouping)},
        ).fit(X, y)
        session = EditorSession.from_model(model, train_data=(X, y))
        free = free_level_comparison(session, "band")
        assert free["levels"] == [str(level) for level in session.terms["band"].levels]
        # Make special reads the refit's rows through the declaration too.
        session.stage_structural("special", "band", {"levels": ["4"]})


def test_free_levels_keep_a_reference_named_first_as_a_level(monkeypatch):
    order = ["first", "B", "C", "D", "E", "F", "G", "H"]
    rng = np.random.default_rng(5)
    band = rng.choice(order, 2000)
    y = rng.poisson(np.exp(-1 + 0.1 * np.array([order.index(b) for b in band]))).astype(float)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="poisson", features={"band": OrderedCategorical(order=order, base="first")}
    ).fit(X, y)
    fitted = _fitted_free_models(monkeypatch)
    free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert fitted[0]._specs["band"]._base_level == "first"


def test_free_levels_compare_a_term_the_selection_penalty_removed():
    rng = np.random.default_rng(9)
    band = np.repeat(BANDS, 100)
    y = rng.poisson(0.5, band.size).astype(float)
    X = pd.DataFrame({"band": band})
    model = SuperGLM(
        family="poisson",
        features={"band": OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=6))},
        spline_penalty=20.0,
        selection_penalty=1000.0,
    ).fit(X, y)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == BANDS


def test_the_sidak_cut_counts_the_comparisons_judged():
    """Twelve bands with two groups of two: ten comparisons, one per group, not twelve."""
    from scipy.stats import norm

    X, y, w = _book(n=6000)
    groups = {"a": ["Mi012", "Mi018"], "b": ["Mi060", "Mi066"]}
    grouping = collapse_levels(X["band"], groups=groups, order=BANDS)
    band = OrderedCategorical(
        order=BANDS, grouping=grouping, base="Mi006", basis=Spline(kind="ps", n_knots=6)
    )
    model = SuperGLM(
        family="poisson", features={"band": band, "area": Categorical()}, spline_penalty=100.0
    ).fit(X, y, sample_weight=w)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y, w)), "band")
    assert free["z"] == pytest.approx(float(norm.ppf(0.5 + 0.5 * 0.95 ** (1 / 10))), rel=1e-12)


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


TWELVE = [f"B{i:02d}" for i in range(12)]


def _gaussian(seed: int, *, effect=None):
    """Twelve bands of 50 rows, the noise centred within each band."""
    k = np.repeat(np.arange(12), 50)
    noise = np.random.default_rng(seed).normal(0.0, 1.0, k.size)
    noise -= np.bincount(k, noise)[k] / 50
    y = 10.0 + 0.05 * k + (0.0 if effect is None else effect(k)) + noise
    return pd.DataFrame({"band": np.array(TWELVE)[k]}), y, k


@pytest.mark.parametrize("first", ["band", "area"])
def test_a_level_another_term_aliases_is_left_out_and_named(first):
    """``area`` is exactly B05's rows, so B05's free value is not estimable.

    Centring over it moved every gap by an undetermined amount: declared
    band first, all twelve bands were flagged, declared second only B05.
    """
    X, y, k = _gaussian(10, effect=lambda k: 3.0 * (k == 5))
    X["area"] = np.where(k == 5, "x", "y")
    features = {
        "band": OrderedCategorical(order=TWELVE, basis=Spline(kind="ps", n_knots=6)),
        "area": Categorical(),
    }
    model = SuperGLM(
        family="gaussian",
        features={name: features[name] for name in (first, *(set(features) - {first}))},
        spline_penalty=20.0,
    ).fit(X, y)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == [level for level in TWELVE if level != "B05"]
    assert free["flagged"] == []
    assert free["notice"] == (
        "No free value is drawn for B05: the model's other terms cover the same rows, "
        "so the data cannot separate its value from theirs."
    )


def test_a_response_the_curve_fits_exactly_compares_with_no_dispersion():
    X, _y, _k = _gaussian(0)
    y = np.full(len(X), 10.0)
    model = SuperGLM(
        family="gaussian",
        features={"band": OrderedCategorical(order=TWELVE, basis=Spline(kind="ps", n_knots=6))},
        spline_penalty=20.0,
    ).fit(X, y)
    # Exact on any platform: the sums of 10.0 are integers, so the centred
    # response, the band's coefficients and the residuals are all zero.
    assert model.result.phi == 0.0
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == TWELVE
    assert free["flagged"] == []
    assert free["notice"] is None


def test_a_level_with_almost_no_weight_keeps_finite_ends_and_moves_no_other_level():
    """B05 weighs 1e-8: its free interval is wider than float64 holds.

    Centred on the levels' plain mean, B05's free variance reached every other
    level's, and every interval was some 300 wide on the log scale. Centred on
    their exposure-weighted mean, the others are as if B05 had no rows.
    """
    X, y, k = _gaussian(4)
    compared = {}
    for thin in (True, False):
        w = np.where(k == 5, 1e-8, 1.0)
        rows = slice(None) if thin else k != 5
        model = SuperGLM(
            family="gaussian",
            features={"band": OrderedCategorical(order=TWELVE, basis=Spline(kind="ps", n_knots=6))},
            spline_penalty=20.0,
        ).fit(X[rows], y[rows], sample_weight=w[rows])
        session = EditorSession.from_model(model, train_data=(X[rows], y[rows], w[rows]))
        compared[thin] = free_level_comparison(session, "band")
    free = compared[True]
    json.dumps(free, allow_nan=False)
    at = free["levels"].index("B05")
    assert free["lower"][at] < free["y"][at] < free["upper"][at]
    # Each gap's standard error, out of its Sidak-widened half-width.
    sd = {
        thin: {
            level: (np.log(c["upper"][i]) - np.log(c["lower"][i])) / (2 * c["z"])
            for i, level in enumerate(c["levels"])
        }
        for thin, c in compared.items()
    }
    others = [level for level in TWELVE if level != "B05"]
    # The free fit counts B05's rows in its residual degrees of freedom, so its
    # dispersion, and with it every other standard error, moves by one factor.
    # Otherwise B05's 1e-8 of a band's weight moves the centre, the curve and
    # each contrast by a relative amount of that order; Gaussian fits are
    # direct solves, so 1e-6 leaves a factor of 100. B05's free variance in
    # every contrast made the ratios differ by half again.
    ratio = np.array([sd[True][level] / sd[False][level] for level in others])
    np.testing.assert_allclose(ratio, ratio.mean(), rtol=1e-6)


def test_a_level_whose_every_response_is_zero_is_left_out_and_named():
    """A band with exposure and no claims has no finite free value.

    Fitted free, it drifted far below the curve with a vast variance, and the
    model's own separation="error" refused the comparison outright.
    """
    rng = np.random.default_rng(5)
    k = np.repeat(np.arange(12), 50)
    y = rng.poisson(np.exp(-1.0 + 0.1 * k)).astype(float)
    y[k == 11] = 0.0
    X = pd.DataFrame({"band": np.array(TWELVE)[k]})
    band = OrderedCategorical(order=TWELVE, basis=Spline(kind="ps", n_knots=6))
    model = SuperGLM(
        family="poisson", features={"band": band}, spline_penalty=20.0, separation="error"
    ).fit(X, y)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == TWELVE[:11]
    assert free["notice"] == (
        "No free value is drawn for B11: every response on its rows is 0, so its free value "
        "has no finite estimate."
    )


def test_free_levels_refuse_a_free_fit_that_fails_and_a_single_level_in_sentences(monkeypatch):
    model, X, y = _twelve_bands(order=TWELVE)
    model.fit(X, y)
    session = EditorSession.from_model(model, train_data=(X, y))
    one = X["band"] == "B00"
    with pytest.raises(EditorValueError) as single:
        free_level_comparison(
            EditorSession.from_model(model.fit(X[one], y[one]), train_data=(X[one], y[one])),
            "band",
        )
    assert str(single.value) == (
        "Comparing 'band' with free levels needs rows of positive weight in at least two of its "
        "levels, and the data the refit reads has them in 1."
    )

    def fails(*args, **kwargs):
        raise ValueError("the solver gave up")

    monkeypatch.setattr(free_levels_module, "fit_refit_model", fails)
    with pytest.raises(EditorValueError) as failed:
        free_level_comparison(session, "band")
    assert str(failed.value) == (
        "The free levels of 'band' could not be estimated, so there is nothing to compare."
    )


def test_free_levels_on_other_rows_than_the_curve_say_so_and_draw_the_free_intervals():
    """The same rows in another order: subtracted row by row, the influences mismatched."""
    X, y, _k = _gaussian(43, effect=lambda k: 0.5 * (k == 5))
    model = SuperGLM(
        family="gaussian",
        features={"band": OrderedCategorical(order=TWELVE, basis=Spline(kind="ps", n_knots=6))},
        spline_penalty=20.0,
    ).fit(X, y)
    order = np.random.default_rng(99).permutation(len(X))
    shuffled = X.iloc[order].reset_index(drop=True), y[order]
    free = free_level_comparison(EditorSession.from_model(model, train_data=shuffled), "band")
    assert free["levels"] == TWELVE
    assert free["notice"] == (
        "Each interval is the free estimate's own, without the curve's pull toward the level: "
        "the curve was fitted on other rows than the comparison reads."
    )


def test_free_levels_hold_their_intervals_whatever_the_weights_size():
    """Gamma/log on means near 1e-30 under weights of 1e280: products of the two overflowed."""
    k = np.repeat(np.arange(12), 50)
    jitter = np.random.default_rng(8).gamma(3.0, 1.0 / 3.0, k.size)
    jitter /= np.bincount(k, jitter)[k] / 50
    y = 1e-30 * np.exp(0.05 * k + 0.8 * (k == 5)) * jitter
    X = pd.DataFrame({"band": np.array(TWELVE)[k]})
    compared = []
    for weight in (1.0, 1e280):
        w = np.full(k.size, weight)
        model = SuperGLM(
            family="gamma",
            link="log",
            features={"band": OrderedCategorical(order=TWELVE, basis=Polynomial(powers=[1]))},
            direct_solve="qr",
        ).fit(X, y, sample_weight=w)
        compared.append(
            free_level_comparison(EditorSession.from_model(model, train_data=(X, y, w)), "band")
        )
    json.dumps(compared[1], allow_nan=False)
    assert compared[1]["flagged"] == compared[0]["flagged"] == ["B05"]
    # Scaling every prior weight by one factor scales the dispersion with it,
    # and leaves each interval where it was. 1e280 is no power of two, so each
    # weight rounds once, by u; the straight-line term keeps the fits' error
    # within a small multiple of that, and 1e-9 leaves room for a condition
    # number near 1e6.
    for end in ("lower", "upper"):
        np.testing.assert_allclose(compared[1][end], compared[0][end], rtol=1e-9)


@pytest.mark.parametrize("collapse", [False, True])
def test_a_level_put_back_beside_an_equal_value_keeps_its_place(collapse):
    """B05 and B06 share a value: put back last, B05 sorted after B06.

    The two orders predict alike, but B04 and B05 were no longer adjacent to
    collapse, and the exported structure no longer matched the declaration.
    A collapse made while B05 is special lists it last in its grouping too.
    """
    X, y, _k = _gaussian(1)
    values = {level: i / 11 for i, level in enumerate(TWELVE)}
    values["B06"] = values["B05"]
    model = SuperGLM(
        family="gaussian",
        features={"band": OrderedCategorical(values=values, basis=Spline(kind="ps", n_knots=6))},
        spline_penalty=20.0,
    ).fit(X, y)
    session = EditorSession.from_model(model, train_data=(X, y))
    session.replace_with_special_levels("band", ["B05"])
    if collapse:
        session.stage_structural("collapse", "band", {"levels": ["B00", "B01"]})
        session.refit_pending()
    session.replace_with_special_levels("band", ["B05"], special=False)
    spec = session.model._specs["band"]
    assert full_level_order(spec) == TWELVE
    smooth = [str(level) for level in spec._smooth_levels]
    assert smooth[smooth.index("B05") + 1] == "B06"
    session.stage_structural("collapse", "band", {"levels": ["B04", "B05"]})


def _twelve_bands(retain_fit_state: bool = True, **declared):
    X, y, _k = _gaussian(0)
    band = OrderedCategorical(basis=Spline(kind="ps", n_knots=6), **declared)
    model = SuperGLM(
        family="gaussian",
        features={"band": band},
        spline_penalty=20.0,
        retain_fit_state=retain_fit_state,
    )
    return model, X, y


def test_free_levels_compare_a_term_whose_reference_has_no_rows():
    """The categorical took the reference B11, which the rows do not hold, and refused to fit."""
    model, X, y = _twelve_bands(order=TWELVE, base="B11")
    held = X["band"] != "B11"
    X, y = X[held], y[held.to_numpy()]
    model.fit(X, y)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == TWELVE[:11]


def test_free_levels_give_each_text_of_the_column_its_level():
    """An object column of 1 and 1.0: one value to a hash, two texts to the categorical."""
    model, X, y = _twelve_bands(order=list(range(1, 13)))
    k = np.repeat(np.arange(12), 50)
    column = [int(v + 1) if i % 2 else float(v + 1) for i, v in enumerate(k)]
    X = pd.DataFrame({"band": pd.Series(column, dtype=object)})
    assert {type(value) for value in X["band"]} == {int, float}
    model.fit(X, y)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == [str(level) for level in range(1, 13)]


def test_free_levels_without_a_fitted_design_say_their_intervals_are_the_free_estimates():
    model, X, y = _twelve_bands(retain_fit_state=False, order=TWELVE)
    model.fit(X, y)
    free = free_level_comparison(EditorSession.from_model(model, train_data=(X, y)), "band")
    assert free["levels"] == TWELVE
    assert free["notice"] == (
        "Each interval is the free estimate's own, without the curve's pull toward the level: "
        "the model keeps no fitted design to measure it. Refit it with retain_fit_state=True to "
        "include it."
    )
