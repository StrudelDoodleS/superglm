import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, Spline


@pytest.mark.parametrize(
    "spec_cls",
    [NaturalSpline, CubicRegressionSpline],
    ids=["natural", "crs"],
)
def test_constrained_splines_extend_with_linear_tails(spec_cls):
    x_train = np.linspace(0.0, 1.0, 200)
    spec = spec_cls(n_knots=8, extrapolation="extend")
    spec.build(x_train)

    z = spec._Z
    assert z is not None

    rng = np.random.default_rng(42)
    for _ in range(5):
        alpha = rng.standard_normal(z.shape[1])
        beta_orig = z @ alpha

        left_grid = np.linspace(-0.5, 0.0, 5)
        right_grid = np.linspace(1.0, 1.5, 5)

        left_vals = spec.transform(left_grid) @ beta_orig
        right_vals = spec.transform(right_grid) @ beta_orig

        np.testing.assert_allclose(np.diff(left_vals, n=2), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.diff(right_vals, n=2), 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "spec_cls",
    [Spline, NaturalSpline, CubicRegressionSpline],
    ids=["pspline", "natural", "crs"],
)
def test_spline_families_recover_known_smooth_poisson_rate(spec_cls):
    rng = np.random.default_rng(123)
    n = 600

    x = rng.uniform(0.0, 1.0, n)
    eta = 0.2 + 0.6 * np.sin(2.0 * np.pi * x) - 0.2 * np.cos(4.0 * np.pi * x)
    exposure = np.full(n, 100.0)
    offset = np.log(exposure)
    y = rng.poisson(exposure * np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="poisson",
        lambda1=0.0,
        lambda2=0.03,
        features={"x": spec_cls(n_knots=12, penalty="ssp")},
    )
    model.fit(X, y, exposure=exposure, offset=offset)

    x_grid = np.linspace(0.0, 1.0, 300)
    X_grid = pd.DataFrame({"x": x_grid})
    offset_grid = np.log(np.full(len(x_grid), 100.0))
    eta_true = 0.2 + 0.6 * np.sin(2.0 * np.pi * x_grid) - 0.2 * np.cos(4.0 * np.pi * x_grid)
    eta_hat = np.log(model.predict(X_grid, offset=offset_grid)) - offset_grid

    rmse = np.sqrt(np.mean((eta_hat - eta_true) ** 2))
    corr = np.corrcoef(eta_hat, eta_true)[0, 1]

    assert rmse < 0.03
    assert corr > 0.998
