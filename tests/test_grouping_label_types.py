"""One level, two spellings: the declaration's and the data column's.

`OrderedCategorical` is the only feature with two sources for a level's identity
-- the `order=`/`values=` declaration and the column. `Categorical` takes its
levels from the data alone, so nothing can disagree; it is covered here as the
control. When the two sources render a level differently (declared `9` against a
float column: "9" vs "9.0") every downstream membership test answers no, and it
has done so as a crash, a silent no-op, a zero-exposure rating row and a guard
that failed open.

The declaration is canonical. `_declared_matcher` maps the DATA onto it, once, at
the edge. These tests pin the shapes that mismatch, and the boundaries of the
matching rule -- which is where the danger is, because a reconciliation that is
too eager turns a loud "unknown level" error into a silent wrong answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM
from superglm.features.grouping import collapse_levels

SHAPES = [
    pytest.param(["1", "2", "3", "9"], np.object_, id="str-decl/str-col"),
    pytest.param([1, 2, 3, 9], np.int64, id="int-decl/int-col"),
    pytest.param([1, 2, 3, 9], np.float64, id="int-decl/float-col"),
    pytest.param([1.0, 2.0, 3.0, 9.0], np.float64, id="float-decl/float-col"),
    pytest.param([1.0, 2.0, 3.0, 9.0], np.int64, id="float-decl/int-col"),
    pytest.param(["1", "2", "3", "9"], np.float64, id="str-decl/float-col"),
]


def _fit(order, dtype, *, feature=None, n=600, seed=4):
    rng = np.random.default_rng(seed)
    codes = rng.choice(np.asarray(order, dtype=dtype), n)
    X = pd.DataFrame({"band": codes})
    y = 0.1 * codes.astype(float) + rng.normal(0.0, 0.15, n)
    spec = feature or OrderedCategorical(order=list(order), basis=Spline(kind="ps", k=5))
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    return model, X


# ── The mismatching shapes ────────────────────────────────────────────────────


@pytest.mark.parametrize(("order", "dtype"), SHAPES)
def test_every_declaration_column_shape_fits_and_predicts(order, dtype):
    # Three of these six shapes were broken: two reported levels in the declared
    # spelling while the column used another, and one raised outright.
    model, X = _fit(order, dtype)
    eta = np.asarray(model._predict_eta_exact(X))
    assert np.all(np.isfinite(eta))
    assert len(model.term_inference("band").levels) == len(order)


@pytest.mark.parametrize(("order", "dtype"), SHAPES)
def test_categorical_is_unaffected_in_every_shape(order, dtype):
    # The control. Categorical derives its levels from the data, so it has one
    # source of truth and never had this bug -- if a change here breaks it, the
    # fix has reached somewhere it should not.
    model, X = _fit(order, dtype, feature=Categorical())
    assert np.all(np.isfinite(np.asarray(model._predict_eta_exact(X))))


@pytest.mark.parametrize(("order", "dtype"), SHAPES)
def test_a_grouped_refit_survives_every_shape(order, dtype):
    from superglm.editor import EditorSession

    model, X = _fit(order, dtype)
    levels = [str(level) for level in model.term_inference("band").levels]
    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", levels[1:3])
    refit = session.replace_with_collapsed_levels("band")

    spec = refit._specs["band"]
    assert spec._grouping is not None
    merged = [m for m in spec._grouping.group_to_originals.values() if len(m) > 1]
    assert len(merged) == 1, spec._grouping.group_to_originals
    assert np.all(np.isfinite(np.asarray(refit._predict_eta_exact(X))))


# ── The boundaries of the matching rule ───────────────────────────────────────


def test_a_string_level_is_never_matched_numerically():
    # Zero-padded identifiers are real in pricing data (policy codes, vehicle
    # groups) and their leading zeros ARE their identity. An earlier attempt at
    # this fix matched on float() unconditionally, which made
    # `collapse_levels(["001","002"], groups={"g": [1]})` silently group "001"
    # instead of raising -- turning a loud error into a wrong answer, which is the
    # failure mode this whole area keeps producing.
    with pytest.raises(ValueError, match="not found in data"):
        collapse_levels(pd.Series(["001", "002"]), groups={"g": [1]})

    # And the same rule at the feature boundary: a str column keeps its identity.
    model, _ = _fit(["001", "002", "003", "009"], np.object_)
    assert [str(lv) for lv in model.term_inference("band").levels] == [
        "001",
        "002",
        "003",
        "009",
    ]


def test_an_unknown_level_is_still_rejected():
    # The reconciliation must not become a catch-all. If it silently absorbed
    # unknown levels the validation would stop meaning anything.
    model, _ = _fit([1, 2, 3, 9], np.float64)
    spec = model._specs["band"]
    with pytest.raises(ValueError, match="unseen categorical levels"):
        spec.transform(np.array([1.0, 77.0], dtype=float))


def test_a_declared_level_absent_from_training_still_predicts():
    # `order=` declares the domain; training data need not exercise all of it.
    # A level seen only at predict time must map to its declared spelling, or the
    # fit silently accepts a domain it cannot score.
    rng = np.random.default_rng(7)
    codes = rng.choice(np.array([1.0, 2.0, 3.0]), 400)  # 9.0 never observed
    X = pd.DataFrame({"band": codes})
    y = 0.1 * codes + rng.normal(0.0, 0.15, 400)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"band": OrderedCategorical(order=[1, 2, 3, 9], basis=Spline(kind="ps", k=5))},
    )
    model.fit(X, y)
    out = model.predict(pd.DataFrame({"band": np.array([9.0])}))
    assert np.all(np.isfinite(np.asarray(out)))


def test_two_spellings_of_one_value_in_the_declaration_are_ambiguous():
    # If the declaration itself contains "1" and 1.0, a column value of 1.0
    # denotes neither unambiguously. Picking by iteration order would make the
    # answer depend on declaration order; say so instead.
    # NOT raised at construction. `order=["1", "1.0", ...]` are perfectly
    # fittable levels: each is reachable by its own exact spelling, and rejecting
    # the declaration outright refuses a valid model. An earlier attempt did
    # exactly that.
    spec = OrderedCategorical(order=["1", "1.0", "2", "3", "4", "5"], basis=Spline(kind="ps", k=5))
    probe = np.array(["1", "1.0", "2", "3", "4", "5"], dtype=object)
    spec.build(probe, np.ones(len(probe)))
    out = spec.transform(probe)
    assert out.shape[0] == 6

    # Ambiguity bites only for a raw value that matches NEITHER spelling exactly
    # yet equals both numerically. Asserted on the matcher itself, because the
    # ordinary float and int spellings all hit an exact match first -- so a test
    # driving this through `transform` with 1.0 would pass without ever reaching
    # the branch it names.
    from decimal import Decimal

    from superglm.features.ordered_categorical import _declared_matcher

    match = _declared_matcher(["1", "1.0", "2"])
    assert match("1") == "1"  # exact spellings still resolve
    assert match("1.0") == "1.0"
    with pytest.raises(ValueError, match="ambiguous"):
        match(Decimal("1.000"))


def test_the_reporting_base_survives_a_spelling_mismatch():
    # `base=` is declared by the user in their own spelling and compared against
    # the level list. A mismatch here silently re-bases the whole term, changing
    # every reported relativity.
    model, _ = _fit(
        [1, 2, 3, 9],
        np.float64,
        feature=OrderedCategorical(order=[1, 2, 3, 9], base="1", basis=Spline(kind="ps", k=5)),
    )
    assert str(model._specs["band"]._base_level) == "1"


def test_a_grouping_passed_directly_works_on_a_numeric_column():
    # The non-editor path: a user builds a grouping themselves and hands it to the
    # spec. The editor path had its own reconciliation; this one had none, so it
    # was the site that stayed broken after the first fix.
    rng = np.random.default_rng(11)
    codes = rng.choice(np.array([1.0, 2.0, 3.0, 9.0]), 500)
    X = pd.DataFrame({"band": codes})
    y = 0.1 * codes + rng.normal(0.0, 0.15, 500)
    grouping = collapse_levels(X["band"], groups={"2+3": ["2.0", "3.0"]})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=[1, 2, 3, 9], grouping=grouping, basis=Spline(kind="ps", k=5)
            )
        },
    )
    model.fit(X, y)
    assert np.all(np.isfinite(np.asarray(model._predict_eta_exact(X))))


def test_a_numeric_special_survives_a_collapse_of_other_levels():
    # specials + grouping + numeric labels, the combination none of the three
    # areas covered on its own. The coverage guard compares the grouping's
    # spelling against the declared specials, so a mismatch made every collapse of
    # an unrelated level fail.
    from superglm.editor import EditorSession

    rng = np.random.default_rng(13)
    codes = rng.choice(np.array([1.0, 2.0, 3.0, 4.0, 9.0]), 700)
    X = pd.DataFrame({"band": codes})
    y = np.where(codes == 9.0, 0.9, 0.1 * codes) + rng.normal(0.0, 0.15, 700)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=[1, 2, 3, 4, 9], specials=[9], basis=Spline(kind="ps", k=5)
            )
        },
    )
    model.fit(X, y)
    levels = [str(level) for level in model.term_inference("band").levels]

    session = EditorSession.from_model(model, terms=["band"])
    session.select_levels("band", levels[1:3])
    refit = session.replace_with_collapsed_levels("band")
    assert np.all(np.isfinite(np.asarray(refit._predict_eta_exact(X))))


# ── Review round two: the boundaries of the matching rule, again ──────────────


def test_a_padded_string_is_not_claimed_by_a_numeric_declaration():
    # The regression the FIRST version of this test failed to catch. It declared
    # `order=["001", ...]`, so the exact-match branch answered before the numeric
    # one was reached -- the dangerous path (numeric declaration, padded string in
    # the data) was never exercised, and `"001"` was silently scored as level 1.
    spec = OrderedCategorical(order=[1, 2, 3, 4, 5, 6], basis=Spline(kind="ps", k=5))
    assert list(spec._canonical(np.array(["001", "2"], dtype=object))) == ["001", 2]


def test_two_levels_with_the_same_text_are_refused():
    # Everything downstream joins on str(), so `order=[1, "1"]` cannot be
    # represented: one wins every lookup and the other can never be fitted.
    with pytest.raises(ValueError, match="appear more than once"):
        OrderedCategorical(order=[1, "1", 2, 3, 4, 5], basis=Spline(kind="ps", k=5))


def test_a_group_label_colliding_with_a_respelled_level_is_refused():
    # Re-spelling can make an identity label "1.0" become "1" while the caller
    # already named a group "1". Silently, one entry is lost and every original
    # maps to the survivor -- folding level 1 into the 2+3 group.
    data = np.array([1.0, 2.0, 3.0, 9.0] * 30)
    grouping = collapse_levels(pd.Series(data), groups={"1": ["2.0", "3.0"]})
    with pytest.raises(ValueError, match="collide"):
        OrderedCategorical(order=[1, 2, 3, 9], grouping=grouping, basis=Spline(kind="ps", k=5))


def test_the_direct_grouping_path_survives_term_inference():
    # The first direct-grouping regression only exercised predict(), so it missed
    # that `_original_level_to_value` was still keyed by the declared objects
    # while the grouping had been re-spelled -- KeyError out of the public
    # inference path, which is where a user meets it.
    rng = np.random.default_rng(11)
    codes = rng.choice(np.array([1.0, 2.0, 3.0, 9.0]), 500)
    X = pd.DataFrame({"band": codes})
    y = 0.1 * codes + rng.normal(0.0, 0.15, 500)
    grouping = collapse_levels(X["band"], groups={"2+3": ["2.0", "3.0"]})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=[1, 2, 3, 9], grouping=grouping, basis=Spline(kind="ps", k=5)
            )
        },
    )
    model.fit(X, y)
    ti = model.term_inference("band")
    assert [str(level) for level in ti.levels] == ["1", "2", "3", "9"]
