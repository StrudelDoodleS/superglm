"""Editor staging: waiting structural changes, one Refit, carried edits, history ids and notes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession
from superglm.editor.collapse import (
    clone_with_replaced_features,
    collapsed_feature_spec,
    reference_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.errors import EditorValueError
from superglm.editor.refit import fit_refit_model
from superglm.editor.shapes import shaped_feature_spec

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
