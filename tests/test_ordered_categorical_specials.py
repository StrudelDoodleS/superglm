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


def test_grouping_that_collapses_every_ordered_level_is_rejected():
    # The at-least-two-smooth-levels check runs on the pre-grouping level list,
    # so a grouping that leaves one smooth level reaches the spline build with
    # a single distinct position.
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
