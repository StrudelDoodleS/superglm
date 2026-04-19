"""Tests for public constraint tokens."""

from superglm import Constraint, ConstraintSpec, features


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
