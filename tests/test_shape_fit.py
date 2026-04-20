import numpy as np
import pandas as pd

from superglm import Constraint, PSpline, SuperGLM


def _curve_second_derivative(model, feature: str, n_points: int = 200) -> np.ndarray:
    del n_points
    curve = model.reconstruct_feature(feature)
    x = curve["x"]
    y = curve["log_relativity"]
    return np.gradient(np.gradient(y, x), x)


def test_pspline_fit_convex_discrete_reconstructs_convex_term():
    x = np.linspace(0.0, 1.0, 400)
    y = 0.5 + (x - 0.3) ** 2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.convex)},
    ).fit_reml(df, y)

    d2 = _curve_second_derivative(model, "x")
    assert d2.min() > -5e-3


def test_pspline_fit_concave_dense_reconstructs_concave_term():
    x = np.linspace(0.0, 1.0, 400)
    y = 1.0 - (x - 0.4) ** 2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=False,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.concave)},
    ).fit(df, y)

    d2 = _curve_second_derivative(model, "x")
    assert d2.max() < 5e-3
