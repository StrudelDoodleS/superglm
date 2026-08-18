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


def _band_frame(seed=7, n=1200):
    """Ten evenly spaced bands with a curved signal, issue #326's fixture."""
    levels = [f"Mi{index:03d}" for index in range(10)]
    rng = np.random.default_rng(seed)
    band = rng.choice(levels, n)
    position = {level: index for index, level in enumerate(levels)}
    y = np.array([0.02 * min(position[b], 4) ** 2 for b in band]) + rng.normal(0.0, 0.05, n)
    return levels, pd.DataFrame({"band": band}), y


def _grouped_fit(levels, X, y, groups, *, values=None):
    covered = {member for members in groups.values() for member in members}
    full = dict(groups)
    for level in levels:
        if level not in covered:
            full[level] = [level]
    grouping = collapse_levels(X["band"].to_numpy(dtype=object), groups=full, order=levels)
    kwargs = {"values": values} if values is not None else {"order": levels}
    spec = OrderedCategorical(basis=Spline(kind="cr", n_knots=4), grouping=grouping, **kwargs)
    model = SuperGLM(family="gaussian", selection_penalty=0.0, features={"band": spec})
    model.fit(X, y)
    return model, spec


def test_a_group_named_after_a_member_still_sits_at_the_group_mean():
    """Issue #326: the group's position is its members' mean, never the label's.

    ``{"Mi001": ["Mi001", "Mi002"]}`` -- "fold Mi002 into Mi001" -- is the
    natural spelling and keeps the surviving label stable in a rating table.
    The position map used to short-circuit on ``str(glev) in by_text``, which is
    a lookup shortcut for the singleton identity groups that make up most of any
    grouping and is harmless there (a singleton's mean IS its one member). Where
    the label collided with a member it took that member's own value instead:
    the fit landed at 0.1111 while ``_collapsed_smooth_curve`` drew the marker
    at the group mean 0.1667, half a level width apart, silently, on the panel
    ``model.plot()`` draws by default.

    Ten evenly spaced levels put the pair at 1/9 and 2/9, so the mean is exactly
    representable-free of choice: 0.5 * (1/9 + 2/9) = 1/6.
    """
    levels, X, y = _band_frame()
    _, spec = _grouped_fit(levels, X, y, {"Mi001": ["Mi001", "Mi002"]})

    declared = np.linspace(0.0, 1.0, len(levels))
    assert spec._level_to_value["Mi001"] == pytest.approx(
        float(np.mean(declared[[1, 2]])), abs=1e-15
    )
    # and NOT the named member's own value, which is where it used to land
    assert spec._level_to_value["Mi001"] != pytest.approx(float(declared[1]), abs=1e-9)


def test_the_group_label_does_not_move_the_model():
    """A label names a rating-table row; it may not change what was fitted.

    This is the argument that decides #326's convention rather than merely
    stating it. ``{"Mi001": [...]}`` and ``{"Mi001+Mi002": [...]}`` declare the
    SAME partition of the same levels, so they are the same model -- but under
    the named-member reading the first placed the group at 0.1111 and the second
    at 0.1667, which is a different point of the spline basis and therefore a
    different fit. Measured before the fix: every one of the ten reported
    log-relativities moved, by up to 1.45e-02.

    Exact equality, not a tolerance: the two fits differ only in a dict key, so
    once the positions agree every float downstream is computed from identical
    inputs in identical order.
    """
    levels, X, y = _band_frame()
    named, named_spec = _grouped_fit(levels, X, y, {"Mi001": ["Mi001", "Mi002"]})
    fresh, fresh_spec = _grouped_fit(levels, X, y, {"Mi001+Mi002": ["Mi001", "Mi002"]})

    assert named_spec._level_to_value["Mi001"] == fresh_spec._level_to_value["Mi001+Mi002"]
    left = named.term_inference("band", with_se=True)
    right = fresh.term_inference("band", with_se=True)
    assert [str(level) for level in left.levels] == [str(level) for level in right.levels]
    for field in ("log_relativity", "relativity", "se_log_relativity"):
        np.testing.assert_array_equal(
            np.asarray(getattr(left, field), dtype=np.float64),
            np.asarray(getattr(right, field), dtype=np.float64),
            err_msg=f"renaming the group moved {field}",
        )


def test_the_marker_lands_on_the_fit_for_a_group_named_after_a_member():
    """The two halves of #326 now read one definition, so they cannot disagree.

    The spec places the group and ``_collapsed_smooth_curve`` puts the collapsed
    marker back; both call ``group_axis_position``. This is the case where they
    used to differ -- and it is the only grouping shape for which the expand ->
    re-collapse round trip was not the identity.
    """
    from superglm.plotting.group_display import project_grouped_term_for_display

    levels, X, y = _band_frame()
    model, spec = _grouped_fit(levels, X, y, {"Mi001": ["Mi001", "Mi002"]})
    ti = model.term_inference("band", with_se=True)
    display = project_grouped_term_for_display(model, ti, "auto")

    assert display.collapsed is True
    np.testing.assert_array_equal(
        np.asarray(display.term.smooth_curve.level_x, dtype=np.float64),
        np.asarray(list(spec._level_to_value.values()), dtype=np.float64),
    )


def test_values_places_a_group_exactly_where_the_caller_asks():
    """The explicit route to any other position, with no special case anywhere.

    Give the members the position the group should sit at and the mean of equal
    values is that value exactly -- so a caller who really does want the group
    at ``Mi001``'s own coordinate says so in ``values=``, and gets it to the last
    bit rather than as a side effect of what they called the group.

    Not vacuous: the same grouping under the default ``order=`` spacing lands
    somewhere else, and the marker follows the fit either way.
    """
    from superglm.plotting.group_display import project_grouped_term_for_display

    levels, X, y = _band_frame()
    declared = dict(zip(levels, np.linspace(0.0, 1.0, len(levels)).tolist()))
    pinned = dict(declared)
    pinned["Mi002"] = declared["Mi001"]  # both members at Mi001's coordinate

    model, spec = _grouped_fit(levels, X, y, {"Mi001": ["Mi001", "Mi002"]}, values=pinned)
    assert spec._level_to_value["Mi001"] == declared["Mi001"]
    # the pinning is what moved it: unpinned, the same grouping sits at the mean
    _, unpinned = _grouped_fit(levels, X, y, {"Mi001": ["Mi001", "Mi002"]})
    assert unpinned._level_to_value["Mi001"] != spec._level_to_value["Mi001"]
    # and the collapsed marker follows the fit to the pinned coordinate
    ti = model.term_inference("band", with_se=True)
    display = project_grouped_term_for_display(model, ti, "auto")
    assert float(np.asarray(display.term.smooth_curve.level_x)[1]) == declared["Mi001"]


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


def test_a_large_integer_level_is_matched_exactly():
    # float() is lossy past 2**53, so two distinct integers can land on the same
    # value and one would be scored as the other. Confirming the match in the
    # numbers' own domain costs one comparison.
    from superglm.features.ordered_categorical import _declared_matcher

    match = _declared_matcher([9007199254740993, 2, 3])
    assert match(9007199254740993) == 9007199254740993
    assert match(9007199254740992) == 9007199254740992  # unmatched, not folded
    # and the ordinary case is untouched
    assert _declared_matcher([1, 2, 3])(1.0) == 1
