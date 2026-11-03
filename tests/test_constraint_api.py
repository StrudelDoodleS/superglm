"""Tests for public constraint tokens."""

import pickle

import numpy as np
import pytest

from superglm import (
    BSplineSmooth,
    Constraint,
    ConstraintRules,
    ConstraintSpec,
    PSpline,
    Spline,
    features,
)
from superglm.features.constraint import RULE_KINDS, rules_of

C = Constraint.fit

# A ConstraintSpec(mode="fit", kind="increasing") pickled (protocol 4) when the class had two fields.
_TWO_FIELD_PICKLE = (
    b"\x80\x04\x95\\\x00\x00\x00\x00\x00\x00\x00\x8c\x1csuperglm.features.constraint\x94"
    b"\x8c\x0eConstraintSpec\x94\x93\x94)\x81\x94}\x94(\x8c\x04mode\x94\x8c\x03fit\x94"
    b"\x8c\x04kind\x94\x8c\nincreasing\x94ub."
)


def test_constraint_fit_increasing_token():
    token = Constraint.fit.increasing

    assert isinstance(token, ConstraintSpec)
    assert token.mode == "fit"
    assert token.kind == "increasing"


def test_constraint_postfit_decreasing_token():
    token = Constraint.postfit.decreasing

    assert isinstance(token, ConstraintSpec)
    assert token.mode == "postfit"
    assert token.kind == "decreasing"


def test_reserved_constraint_tokens_exist():
    assert Constraint.fit.convex == ConstraintSpec(mode="fit", kind="convex")
    assert Constraint.fit.concave == ConstraintSpec(mode="fit", kind="concave")
    assert Constraint.postfit.convex == ConstraintSpec(mode="postfit", kind="convex")
    assert Constraint.postfit.concave == ConstraintSpec(mode="postfit", kind="concave")
    assert features.Constraint is Constraint


def test_pspline_constraint_token_normalizes_to_internal_monotone_fields():
    spec = PSpline(n_knots=8, constraint=Constraint.fit.increasing)

    assert spec.monotone == "increasing"
    assert spec.monotone_mode == "fit"


def test_constraint_fit_convex_token_normalizes_to_shape_fields():
    spec = BSplineSmooth(n_knots=8, constraint=Constraint.fit.convex)

    assert spec.constraint_kind == "convex"
    assert spec.constraint_mode == "fit"
    assert spec.monotone == "convex"
    assert spec.monotone_mode == "fit"


def test_constraint_postfit_concave_token_normalizes_to_shape_fields():
    spec = PSpline(n_knots=8, constraint=Constraint.postfit.concave)

    assert spec.constraint_kind == "concave"
    assert spec.constraint_mode == "postfit"
    assert spec.monotone == "concave"
    assert spec.monotone_mode == "postfit"


def test_pspline_rejects_old_public_monotone_arguments():
    with pytest.raises(TypeError):
        PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")


def test_spline_factory_rejects_old_public_monotone_arguments():
    with pytest.raises(TypeError):
        Spline(kind="ps", n_knots=8, monotone="increasing", monotone_mode="fit")


def test_and_builds_rules_in_order():
    rules = C.increasing & C.concave & C.cap(0.6, 1.8)
    assert isinstance(rules, ConstraintRules)
    assert [r.kind for r in rules] == ["increasing", "concave", "cap"]
    assert len(rules) == 3
    assert rules.describe() == "increasing & concave & cap(0.6, 1.8)"
    assert (
        repr(rules)
        == "Constraint.fit.increasing & Constraint.fit.concave & Constraint.fit.cap(0.6, 1.8)"
    )


def test_and_joins_rules_with_rules_in_either_order():
    left = C.increasing & C.concave
    right = C.max_step(0.15) & C.flat.within(25, 40)
    assert [r.kind for r in left & right] == ["increasing", "concave", "max_step", "flat"]
    assert [r.kind for r in C.cap(0.5, 2) & left] == ["cap", "increasing", "concave"]
    assert [r.kind for r in left & C.cap(0.5, 2)] == ["increasing", "concave", "cap"]


def test_single_rule_tokens_keep_their_old_identity():
    assert C.increasing == ConstraintSpec(mode="fit", kind="increasing")
    assert hash(C.increasing) == hash(ConstraintSpec(mode="fit", kind="increasing"))
    assert C.increasing.stretch == (None, None)
    assert repr(C.increasing) == "Constraint.fit.increasing"
    assert repr(Constraint.postfit.concave) == "Constraint.postfit.concave"
    assert set(RULE_KINDS) >= {"flat", "cap", "max_step"}


def test_stretches():
    r = C.decreasing.below(25)
    assert r.stretch == (None, 25) and r.describe() == "decreasing.below(25)"
    assert C.flat.within(25, 40).stretch == (25, 40)
    assert C.increasing.above(40).stretch == (40, None)
    assert C.flat.within(25, 40).describe() == "flat.within(25, 40)"
    assert C.increasing.above(40).describe() == "increasing.above(40)"


def test_label_stretches_are_kept_and_compared_only_after_resolve():
    position = {"young": 20.0, "mid": 40.0, "old": 60.0}.__getitem__
    overlapping = C.increasing.within("young", "old") & C.decreasing.above("mid")
    assert overlapping.rules[0].stretch == ("young", "old")
    overlapping.validate()  # labels are not compared before resolve
    with pytest.raises(ValueError, match=r"\[40, 60\]"):
        overlapping.resolve(position)
    disjoint = C.increasing.below("mid") & C.decreasing.above("mid")
    assert [r.stretch for r in disjoint.resolve(position)] == [(None, 40.0), (40.0, None)]


def test_resolve_refuses_labels_that_land_out_of_order():
    rules = ConstraintRules((C.increasing.within("old", "young"),))
    rules.validate()
    with pytest.raises(ValueError, match="lo < hi"):
        rules.resolve({"young": 20.0, "old": 60.0}.__getitem__)


def test_split_partitions_by_kind_preserving_order():
    rules = C.increasing & C.cap(0.5, 2) & C.concave & C.max_step(0.2)
    shape, rest = rules.split(frozenset({"increasing", "concave"}))
    assert [r.kind for r in shape] == ["increasing", "concave"]
    assert [r.kind for r in rest] == ["cap", "max_step"]
    assert isinstance(shape, ConstraintRules) and isinstance(rest, ConstraintRules)


def test_describe_cap_and_max_step():
    assert C.cap(0.6, 1.8).describe() == "cap(0.6, 1.8)"
    assert C.max_step(0.15).describe() == "max_step(0.15)"
    assert C.max_step(0.2, 0.1).describe() == "max_step(0.2, 0.1)"
    assert C.max_step(0.15).bounds == (0.15, 0.15)


@pytest.mark.parametrize(
    "bad, match",
    [
        (lambda: C.cap(1.1, 2.0), "contain 1"),
        (lambda: C.cap(0.0, 2.0), "contain 1"),
        (lambda: C.cap(0.5, 0.5), "lo < hi"),
        (lambda: C.max_step(0.1, 1.0), "down"),
        (lambda: C.max_step(0.0), "up"),
        (lambda: C.max_step(1.5), "down"),
        (
            lambda: (C.increasing.within(20, 40) & C.decreasing.within(30, 50)).validate(),
            r"\[30, 40\]",
        ),
        (lambda: (C.concave & C.convex.above(10)).validate(), "overlap"),
        (lambda: (C.cap(0.5, 2) & C.cap(0.6, 1.8)).validate(), "one cap"),
        (lambda: (C.max_step(0.1) & C.max_step(0.2)).validate(), "one max_step"),
        (lambda: (C.flat.within(10, 30) & C.flat.within(20, 40)).validate(), r"\[20, 30\]"),
        (lambda: rules_of_flat().validate(), "needs a stretch"),
        (lambda: C.flat.within(40, 25), "lo < hi"),
        (lambda: C.flat.within(25, 25), "lo < hi"),
        (lambda: C.increasing.below(5).above(1), "already has a stretch"),
        (lambda: C.cap(0.5, 2).below(1), "whole curve"),
        (lambda: C.increasing.below(float("nan")), "finite"),
        (lambda: Constraint.postfit.increasing & C.concave, "fit-time"),
        (lambda: C.concave & Constraint.postfit.increasing, "fit-time"),
        (lambda: Constraint.postfit.increasing.below(3), "fit-time"),
        (lambda: ConstraintRules((ConstraintSpec("postfit", "flat"),)).validate(), "fit-time"),
    ],
)
def test_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        bad()


def rules_of_flat():
    return C.flat & C.increasing


def test_disjoint_and_touching_stretches_are_accepted():
    (C.increasing.below(30) & C.decreasing.above(30)).validate()
    (C.increasing & C.convex & C.cap(0.5, 2) & C.max_step(0.1)).validate()
    (C.decreasing.within(0, 10) & C.decreasing.within(20, 30)).validate()


def test_rules_of_reads_legacy_single_rule():
    class Old:
        constraint_kind, constraint_mode = "increasing", "fit"

    assert [r.kind for r in rules_of(Old())] == ["increasing"]

    class Older:
        monotone, monotone_mode = "decreasing", "postfit"

    assert rules_of(Older()).rules[0] == Constraint.postfit.decreasing


def test_rules_of_prefers_constraint_rules_and_defaults_to_empty():
    class New:
        constraint_rules = C.increasing & C.concave
        constraint_kind, constraint_mode = "decreasing", "fit"

    assert [r.kind for r in rules_of(New())] == ["increasing", "concave"]
    assert len(rules_of(object())) == 0

    class Unconstrained:
        constraint_kind = None
        constraint_rules = None

    assert len(rules_of(Unconstrained())) == 0


def test_rules_of_reads_a_real_spline_spec():
    spec = PSpline(n_knots=8, constraint=Constraint.fit.convex)
    assert rules_of(spec).rules == (Constraint.fit.convex,)
    assert len(rules_of(PSpline(n_knots=8))) == 0


def test_two_field_pickle_loads_and_equals_the_new_token():
    old = pickle.loads(_TWO_FIELD_PICKLE)
    assert old == C.increasing
    assert old.stretch == (None, None) and old.bounds is None
    assert old.describe() == "increasing"
    assert pickle.loads(pickle.dumps(C.cap(0.6, 1.8))) == C.cap(0.6, 1.8)
    rules = C.increasing.below(3) & C.max_step(0.2)
    assert pickle.loads(pickle.dumps(rules)) == rules


def test_spline_stores_rules():
    spec = Spline(kind="ps", k=12, constraint=C.increasing & C.concave)
    assert spec.constraint_rules.describe() == "increasing & concave"
    assert spec.constraint_kind == "increasing & concave"
    assert spec.constraint_mode == "fit"
    assert (spec.monotone, spec.monotone_mode) == ("increasing & concave", "fit")
    assert rules_of(spec) == C.increasing & C.concave


def test_single_rule_spec_unchanged():
    spec = Spline(kind="ps", k=12, constraint=C.increasing)
    assert (spec.constraint_kind, spec.constraint_mode) == ("increasing", "fit")
    assert (spec.monotone, spec.monotone_mode) == ("increasing", "fit")
    assert rules_of(spec).rules == (C.increasing,)
    post = Spline(kind="ps", k=12, constraint=Constraint.postfit.decreasing)
    assert (post.constraint_kind, post.constraint_mode) == ("decreasing", "postfit")
    none = Spline(kind="ps", k=12)
    assert (none.constraint_kind, none.constraint_mode, len(none.constraint_rules)) == (
        None,
        "postfit",
        0,
    )


@pytest.mark.parametrize(
    "rule",
    [C.increasing.below(25), C.flat.within(20, 30), C.cap(0.6, 1.8), C.max_step(0.2)],
)
def test_a_single_stretched_or_new_rule_is_a_collection_not_dropped(rule):
    spec = Spline(kind="ps", k=12, constraint=rule)
    assert spec.constraint_rules.rules == (rule,)
    assert spec.constraint_kind == rule.describe()
    assert spec.constraint_mode == "fit"
    assert (spec.monotone, spec.monotone_mode) == (rule.describe(), "fit")


def test_a_rule_collection_never_fits_with_rules_dropped():
    # Shape collections and caps build rows; a spline has no levels to limit steps between.
    x = np.linspace(0.0, 50.0, 200)
    for rule in (C.increasing.below(25), C.increasing & C.concave, C.increasing & C.cap(0.6, 1.8)):
        assert Spline(kind="ps", k=12, constraint=rule).build(x).constraints.A.shape[0] > 0
    spec = Spline(kind="ps", k=12, constraint=C.increasing & C.max_step(0.2))
    with pytest.raises(ValueError, match="band the feature with OrderedCategorical"):
        spec.build(x)


def test_per_rule_degree_checks():
    with pytest.raises(ValueError) as single:
        Spline(kind="bs", degree=0, constraint=C.convex)
    with pytest.raises(ValueError) as combined:
        Spline(kind="bs", degree=0, constraint=C.increasing & C.convex)
    assert str(combined.value) == str(single.value)
    with pytest.raises(NotImplementedError, match="degree <= 3") as high:
        Spline(kind="bs", degree=4, constraint=C.increasing & C.concave)
    assert "Constraint.postfit" in str(high.value)
    # A shape that is not curvature has no degree limit.
    Spline(kind="bs", degree=4, constraint=C.increasing & C.max_step(0.2))


def test_ranges_with_rules_still_refused():
    from superglm import PolynomialRange

    with pytest.raises(ValueError, match="would move the pinned coefficients"):
        Spline(
            kind="cr",
            polynomial_ranges=[PolynomialRange(1, 2, 0)],
            constraint=C.increasing & C.concave,
        )


def test_constraint_input_is_checked():
    with pytest.raises(TypeError, match="Constraint"):
        Spline(kind="ps", k=12, constraint="increasing")
    with pytest.raises(ValueError, match="at most one cap"):
        Spline(kind="ps", k=12, constraint=C.cap(0.6, 1.8) & C.cap(0.5, 2.0))
    with pytest.raises(ValueError, match="both apply"):
        Spline(kind="ps", k=12, constraint=C.increasing & C.increasing.below(3))


def test_rules_of_derives_the_rule_of_a_spec_saved_before_collections():
    spec = Spline(kind="ps", k=12, constraint=C.increasing)
    del spec.__dict__["constraint_rules"]  # what an old pickle holds
    assert rules_of(spec).rules == (C.increasing,)
    assert len(rules_of(Spline(kind="ps", k=12))) == 0
