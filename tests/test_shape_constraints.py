import numpy as np

from superglm import BSplineSmooth, Constraint, CubicRegressionSpline


def test_bspline_fit_convex_builds_second_difference_constraints():
    x = np.linspace(0.0, 1.0, 32)
    spec = BSplineSmooth(n_knots=8, constraint=Constraint.fit.convex)
    group = spec.build(x)
    assert group.constraints is not None
    assert group.monotone_engine == "qp"
    assert group.constraints.A.shape[0] > 0


def test_cr_fit_concave_builds_second_difference_constraints():
    x = np.linspace(0.0, 1.0, 32)
    spec = CubicRegressionSpline(n_knots=8, constraint=Constraint.fit.concave)
    group = spec.build(x)
    assert group.constraints is not None
    assert group.monotone_engine == "qp"
    assert group.constraints.A.shape[0] > 0
