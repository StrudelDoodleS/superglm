from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, Spline, SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.distributional.predictor import ShapeConstraintIgnoredWarning


def _data(n: int = 200, seed: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = -1.5 * x + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"x": x}), y


@pytest.mark.parametrize("constraint", [Constraint.fit.increasing, Constraint.postfit.increasing])
def test_a_shape_constraint_on_the_distributional_path_warns_and_fits_unconstrained(constraint):
    frame, y = _data()
    spec = Spline(kind="ps", n_knots=6, constraint=constraint)
    with pytest.warns(ShapeConstraintIgnoredWarning, match="location:x"):
        model = SuperLSS(
            family=GaussianLS(),
            predictors=(Predictor("location", {"x": spec}), Predictor("scale", {})),
        ).fit_reml(frame, y, method="efs")
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 20)})
    location = model.predict_parameters(grid)["location"].to_numpy()
    assert location[0] > location[-1], "the decreasing truth is fitted, not the requested increase"


def test_an_unconstrained_spline_does_not_warn():
    frame, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        SuperLSS(
            family=GaussianLS(),
            predictors=(
                Predictor("location", {"x": Spline(kind="ps", n_knots=6)}),
                Predictor("scale", {}),
            ),
        ).fit_reml(frame, y, method="efs")
