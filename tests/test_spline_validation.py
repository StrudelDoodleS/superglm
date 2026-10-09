import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.spline import CubicRegressionSpline, NaturalSpline, PSpline, Spline


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
    [PSpline, NaturalSpline, CubicRegressionSpline],
    ids=["pspline", "natural", "crs"],
)
def test_spline_families_recover_known_smooth_poisson_rate(spec_cls):
    rng = np.random.default_rng(123)
    n = 600

    x = rng.uniform(0.0, 1.0, n)
    eta = 0.2 + 0.6 * np.sin(2.0 * np.pi * x) - 0.2 * np.cos(4.0 * np.pi * x)
    sample_weight = np.full(n, 100.0)
    offset = np.log(sample_weight)
    y = rng.poisson(sample_weight * np.exp(eta)).astype(float)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.03,
        features={"x": spec_cls(n_knots=12, penalty="ssp")},
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)

    x_grid = np.linspace(0.0, 1.0, 300)
    X_grid = pd.DataFrame({"x": x_grid})
    offset_grid = np.log(np.full(len(x_grid), 100.0))
    eta_true = 0.2 + 0.6 * np.sin(2.0 * np.pi * x_grid) - 0.2 * np.cos(4.0 * np.pi * x_grid)
    eta_hat = np.log(model.predict(X_grid, offset=offset_grid)) - offset_grid

    rmse = np.sqrt(np.mean((eta_hat - eta_true) ** 2))
    corr = np.corrcoef(eta_hat, eta_true)[0, 1]

    assert rmse < 0.03
    assert corr > 0.998


@pytest.mark.parametrize("kind", ["cr", "ns"])
def test_a_natural_spline_on_a_column_with_one_value_says_so(kind):
    """The default ``cr`` cannot place knots on one value; it says why rather than SciPy's words."""
    with pytest.raises(ValueError, match="every value of this column is 3, so a natural spline"):
        Spline(kind=kind).build(np.full(300, 3.0))
    # A P-spline fits such a column, as it did when it was the default.
    Spline(kind="ps").build(np.full(300, 3.0))
