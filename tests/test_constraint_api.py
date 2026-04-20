"""Tests for public constraint tokens."""

import pytest

from superglm import BSplineSmooth, Constraint, ConstraintSpec, PSpline, Spline, features


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
