import numpy as np
import pandas as pd

from superglm import BSplineSmooth, Constraint, PSpline, SuperGLM


def test_qp_convex_fit_reml_auto_lambda_constrained_refit():
    x = np.linspace(0.0, 1.0, 300)
    y = (x - 0.35) ** 2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.convex)},
    ).fit_reml(df, y)

    assert model._reml_lambdas["x"] > 0.0


def test_scop_concave_fit_reml_discrete_estimates_lambda():
    x = np.linspace(0.0, 1.0, 300)
    y = 1.0 - (x - 0.4) ** 2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        discrete=True,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.concave)},
    ).fit_reml(df, y)

    assert model._reml_lambdas["x"] > 0.0
