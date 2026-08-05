"""Specials: levels held out of the smooth and fitted as free level effects."""

import numpy as np
import pytest

from superglm import OrderedCategorical, Spline

ORDERED = [str(i) for i in range(1, 11)]
SPECIAL = "MISSING"


def _oc(**kwargs):
    params = dict(order=list(ORDERED), specials=[SPECIAL], basis=Spline(kind="ps", k=8))
    params.update(kwargs)
    return OrderedCategorical(**params)


# Today `specials` is not a parameter at all, so construction raises TypeError.
def test_specials_are_held_out_of_the_smooth_levels():
    spec = _oc()
    assert spec._specials == [SPECIAL]
    assert spec._smooth_levels == ORDERED
    assert spec._ordered_levels == ORDERED + [SPECIAL]
    assert SPECIAL not in spec._level_to_value
    assert set(spec._level_to_value) == set(ORDERED)
    assert spec._n_levels == len(ORDERED)
    assert spec.has_specials is True
    # A special is a level of the column, so predict-time validation must accept
    # it rather than reject it as unseen.
    assert SPECIAL in spec._known_levels


def test_no_specials_leaves_everything_unchanged():
    spec = OrderedCategorical(order=list(ORDERED), basis=Spline(kind="ps", k=8))
    assert spec._specials == []
    assert spec._smooth_levels == ORDERED
    assert spec._ordered_levels == ORDERED
    assert spec.has_specials is False


# Each of these raises nothing today — `specials` does not exist, and once it
# does, the naive implementation accepts all of them.
def test_label_in_both_order_and_specials_is_popped_from_order():
    spec = OrderedCategorical(
        order=[SPECIAL] + list(ORDERED), specials=[SPECIAL], basis=Spline(kind="ps", k=8)
    )
    assert spec._smooth_levels == ORDERED
    assert SPECIAL not in spec._level_to_value
    # Positions are computed over the survivors, so band 1 is at 0.0 and band 10 at 1.0.
    assert spec._level_to_value["1"] == pytest.approx(0.0)
    assert spec._level_to_value["10"] == pytest.approx(1.0)


def test_label_in_both_values_and_specials_is_popped_from_values():
    spec = OrderedCategorical(
        values={SPECIAL: -1.0, "a": 1.0, "b": 2.0, "c": 3.0},
        specials=[SPECIAL],
        basis=Spline(kind="ps", k=5),
    )
    assert spec._smooth_levels == ["a", "b", "c"]
    assert SPECIAL not in spec._level_to_value


def test_non_str_special_is_coerced_and_popped_from_order():
    # Level labels are `str` everywhere else in this file, so an int special
    # must not survive as a second, un-popped copy of the same level.
    spec = OrderedCategorical(
        order=["1", "2", "3", "4", "5", "9"], specials=[9], basis=Spline(kind="ps", k=5)
    )
    assert spec._specials == ["9"]
    assert spec._smooth_levels == ["1", "2", "3", "4", "5"]
    assert spec._ordered_levels == ["1", "2", "3", "4", "5", "9"]
    assert "9" not in spec._level_to_value


def test_special_is_popped_from_a_non_str_order_by_string_match():
    # The mirror case: `order=` holds non-str labels. Matching by `str` on both
    # sides is what stops level 9 being smoothed *and* claimed free.
    spec = OrderedCategorical(order=[1, 2, 3, 9], specials=["9"], basis=Spline(kind="ps", k=5))
    assert spec._specials == ["9"]
    assert [str(lev) for lev in spec._smooth_levels] == ["1", "2", "3"]
    assert 9 not in spec._level_to_value
    assert "9" not in spec._level_to_value


def test_special_is_popped_from_non_str_values_keys():
    # Same string match on the `values=` path, where the label is a dict key.
    spec = OrderedCategorical(
        values={1: 1.0, 2: 2.0, 3: 3.0, 9: -1.0}, specials=[9], basis=Spline(kind="ps", k=5)
    )
    assert spec._specials == ["9"]
    assert [str(lev) for lev in spec._smooth_levels] == ["1", "2", "3"]
    assert 9 not in spec._level_to_value
    assert "9" not in spec._level_to_value


def test_duplicate_special_is_rejected():
    with pytest.raises(ValueError, match="Duplicate special level"):
        _oc(specials=[SPECIAL, SPECIAL])


def test_fewer_than_two_smooth_levels_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        OrderedCategorical(order=["a", SPECIAL], specials=[SPECIAL], basis=Spline(kind="ps", k=5))


def test_specials_with_step_basis_is_rejected():
    with pytest.raises(ValueError, match="basis='step'"):
        OrderedCategorical(order=list(ORDERED), specials=[SPECIAL], basis="step")


def test_explicit_special_base_is_rejected():
    with pytest.raises(ValueError, match="reporting base"):
        _oc(base=SPECIAL)


def test_non_str_base_naming_a_special_is_rejected():
    # The base check runs against the coerced special set, so `base=9` must be
    # caught here rather than surfacing later as "Base '9' not found in levels".
    with pytest.raises(ValueError, match="reporting base"):
        OrderedCategorical(order=[1, 2, 3, 9], specials=[9], base=9, basis=Spline(kind="ps", k=5))


def test_grouping_that_merges_a_special_is_rejected():
    # The spec's validation table forbids mixing a special with ordered levels
    # in one group, but only the editor's collapse path enforces it. Built
    # directly, the special is silently smoothed inside group "6+MISSING"
    # while `_specials` still lists it as free — an inconsistent spec state
    # with no error anywhere.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"6+MISSING": ["6", SPECIAL]},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="free level"):
        _oc(grouping=grouping)


def test_grouping_that_renames_a_special_is_rejected():
    # A one-member group is still a rename: 'MISSING' becomes 'UNKNOWN', which
    # joins the grouped smooth levels with no numeric position while `_specials`
    # still names 'MISSING'. `_smooth_levels` would then hold a level absent
    # from `_level_to_value`, and `_n_levels` would count it.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"UNKNOWN": [SPECIAL]},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="free level"):
        _oc(grouping=grouping)


def test_grouping_that_collapses_every_ordered_level_is_rejected():
    # The at-least-two-smooth-levels check runs on the grouped level list, so a
    # grouping that leaves one smooth level is refused rather than reaching the
    # spline build with a single distinct position.
    from superglm.features.grouping import collapse_levels

    grouping = collapse_levels(
        np.array(ORDERED + [SPECIAL], dtype=object),
        groups={"all": list(ORDERED)},
        order=ORDERED + [SPECIAL],
    )
    with pytest.raises(ValueError, match="at least two"):
        _oc(grouping=grouping)


# Today `_choose_base` iterates `_ordered_levels`, so once specials are appended
# there, `most_exposed` picks MISSING whenever it dominates exposure — and it
# usually does on a real book.
def test_most_exposed_base_never_selects_a_special():
    spec = _oc()
    x = np.array(["1"] * 10 + [SPECIAL] * 1000, dtype=object)
    weight = np.ones(len(x))
    spec._choose_base(x, weight)
    assert spec._base_level != SPECIAL
    assert spec._base_level in ORDERED
    # The non-base list feeds relativity tables, screening and the editor, so a
    # special must not leak into it either.
    assert SPECIAL not in spec._non_base
    assert spec._non_base == [lev for lev in ORDERED if lev != spec._base_level]


def test_choose_base_reselects_when_a_special_is_already_the_base():
    # The early return must not accept a stale special as the base: a spec
    # cloned from one whose base was set before specials existed would keep it.
    spec = _oc()
    spec._base_level = SPECIAL
    x = np.array(["1"] * 10 + [SPECIAL] * 1000, dtype=object)
    spec._choose_base(x, np.ones(len(x)))
    assert spec._base_level in ORDERED
    assert SPECIAL not in spec._non_base
