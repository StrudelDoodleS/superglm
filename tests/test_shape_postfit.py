import numpy as np
import pandas as pd

from superglm import Constraint, PSpline, SuperGLM


def test_apply_shape_postfit_repairs_convex_term():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 200)
    y = -((x - 0.35) ** 2) + 0.05 * rng.normal(size=len(x))
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        features={"x": PSpline(n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(df, y)

    model.apply_shape_postfit(df)
    repair = model._shape_repairs["x"]

    assert repair.kind == "convex"
    assert repair.max_violation_after <= 1e-8
    assert repair.max_violation_after <= repair.max_violation_before
