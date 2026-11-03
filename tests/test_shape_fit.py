import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Categorical,
    Constraint,
    CubicRegressionSpline,
    PSpline,
    Spline,
    SplineCategorical,
    SuperGLM,
    TensorInteraction,
)
from superglm.constraints import shape_constraint_certificate, shape_derivative_matrix
from superglm.features._spline_subclass_ops import build_shape_constraints_raw

C = Constraint.fit
U = 2.0**-53
U_RULES = C.decreasing.below(25) & C.flat.within(25, 40) & C.increasing.above(40)


def _curve_second_derivative(model, feature: str, n_points: int = 200) -> np.ndarray:
    del n_points
    curve = model.reconstruct_feature(feature)
    x = curve["x"]
    y = curve["log_relativity"]
    return np.gradient(np.gradient(y, x), x)


@pytest.mark.parametrize(
    ("basis_type", "kind", "center", "sign"),
    [
        pytest.param(PSpline, "convex", -0.2, 1.0, id="ps-convex-increasing"),
        pytest.param(PSpline, "convex", 1.2, 1.0, id="ps-convex-decreasing"),
        pytest.param(PSpline, "concave", 1.2, -1.0, id="ps-concave-increasing"),
        pytest.param(PSpline, "concave", -0.2, -1.0, id="ps-concave-decreasing"),
        pytest.param(BSplineSmooth, "convex", -0.2, 1.0, id="bs-convex-increasing"),
        pytest.param(BSplineSmooth, "convex", 1.2, 1.0, id="bs-convex-decreasing"),
        pytest.param(BSplineSmooth, "concave", 1.2, -1.0, id="bs-concave-increasing"),
        pytest.param(BSplineSmooth, "concave", -0.2, -1.0, id="bs-concave-decreasing"),
        pytest.param(
            CubicRegressionSpline,
            "convex",
            -0.2,
            1.0,
            id="cr-convex-increasing",
        ),
        pytest.param(
            CubicRegressionSpline,
            "convex",
            1.2,
            1.0,
            id="cr-convex-decreasing",
        ),
        pytest.param(
            CubicRegressionSpline,
            "concave",
            1.2,
            -1.0,
            id="cr-concave-increasing",
        ),
        pytest.param(
            CubicRegressionSpline,
            "concave",
            -0.2,
            -1.0,
            id="cr-concave-decreasing",
        ),
    ],
)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_fit_time_curvature_recovers_both_slope_orientations(
    basis_type,
    kind,
    center,
    sign,
    discrete,
):
    x = np.linspace(0.0, 1.0, 400)
    y = sign * (x - center) ** 2
    frame = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": basis_type(
                n_knots=10,
                constraint=getattr(Constraint.fit, kind),
            )
        },
    ).fit(frame, y)

    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    curve = model.reconstruct_feature("x")["log_relativity"]
    group = next(group for group in model._groups if group.name == "x")
    certificate = shape_constraint_certificate(
        model._specs["x"],
        model.result.beta[group.sl],
        kind,
    )
    d2 = _curve_second_derivative(model, "x")
    signed_d2 = d2 if kind == "convex" else -d2

    assert model.result.converged
    assert r_squared > 0.995
    assert np.ptp(curve) > 1.0
    assert signed_d2.min() > -5e-3
    assert certificate.minimum_scaled_slack >= -1e-10


@pytest.mark.parametrize(
    ("kind", "center", "sign"),
    [
        pytest.param("convex", -0.2, 1.0, id="convex-increasing"),
        pytest.param("convex", 1.2, 1.0, id="convex-decreasing"),
        pytest.param("concave", 1.2, -1.0, id="concave-increasing"),
        pytest.param("concave", -0.2, -1.0, id="concave-decreasing"),
    ],
)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_pspline_quantile_curvature_recovers_both_slope_orientations(
    kind,
    center,
    sign,
    discrete,
):
    x = np.linspace(0.0, 1.0, 400) ** 3
    y = sign * (x - center) ** 2
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": PSpline(
                n_knots=10,
                knot_strategy="quantile",
                constraint=getattr(Constraint.fit, kind),
            )
        },
    ).fit(frame, y)

    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    group = next(group for group in model._groups if group.name == "x")
    certificate = shape_constraint_certificate(
        model._specs["x"],
        model.result.beta[group.sl],
        kind,
    )

    assert model.result.converged
    assert r_squared > 0.995
    assert certificate.minimum_scaled_slack >= -1e-10


def _native_curve(model, name: str, grid: np.ndarray, order: int) -> np.ndarray:
    """The fitted term's ``order``-th derivative on ``grid``, from its exact basis."""
    group = next(group for group in model._groups if group.name == name)
    spec = model._specs[name]
    return shape_derivative_matrix(spec, grid, order) @ model.result.beta[group.sl]


def _rising_then_dipping(n: int = 20_000, seed: int = 41) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    eta = 0.8 * (1.0 - np.exp(-x / 2.0)) - 0.04 * np.maximum(x - 7.0, 0.0) ** 2
    return pd.DataFrame({"x": x}), rng.poisson(np.exp(eta))


def _driver_ages(n: int = 20_000, seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 81, n).astype(np.float64)
    eta = np.where(age < 25, 0.08 * (25 - age), np.where(age > 40, 0.02 * (age - 40), 0.0))
    return pd.DataFrame({"age": age}), rng.poisson(np.exp(eta))


@pytest.mark.parametrize("kind", ["ps", "bs", "cr"])
def test_increasing_concave_fit(kind):
    X, y = _rising_then_dipping()

    def fit(constraint):
        features = {"x": Spline(kind=kind, k=10, constraint=constraint)}
        return SuperGLM(family="poisson", selection_penalty=0.0, features=features).fit(X, y)

    model = fit(C.increasing & C.concave)
    spec = model._specs["x"]
    grid = np.linspace(spec._lo, spec._hi, 2000)
    f = _native_curve(model, "x", grid, 0)
    tol = 1e3 * U * np.max(np.abs(f))
    assert model.result.converged
    assert np.min(_native_curve(model, "x", grid, 1)) >= -tol
    assert np.max(_native_curve(model, "x", grid, 2)) <= tol
    # Mutation: the same data without the rules dips at the top.
    free = fit(None)
    assert np.min(_native_curve(free, "x", grid, 1)) < 0.0


def _assert_u_shape(model, name: str = "age") -> None:
    grid = np.linspace(17.0, 80.0, 2000)
    f = _native_curve(model, name, grid, 0)
    slope = _native_curve(model, name, grid, 1)
    tol = 1e3 * U * np.max(np.abs(f))
    below, flat, above = grid <= 25.0, (grid >= 25.0) & (grid <= 40.0), grid >= 40.0
    assert np.max(slope[below]) <= tol
    assert np.ptp(f[flat]) <= tol
    assert np.min(slope[above]) >= -tol


@pytest.mark.parametrize("kind", ["ps", "bs", "cr"])
def test_u_with_flat_bottom_fit(kind):
    X, y = _driver_ages()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(kind=kind, k=12, constraint=U_RULES)},
    )
    # 25 and 40 are not knots, so the flat holds on the whole spans it touches.
    with pytest.warns(
        UserWarning, match=r"Feature 'age': flat\.within\(25, 40\) holds on \["
    ) as seen:
        model.fit(X, y)
    assert sum("holds on" in str(w.message) for w in seen) == 1
    assert model.result.converged
    _assert_u_shape(model)
    # Mutation: without the rules the curve is not constant on [25, 40].
    free = SuperGLM(
        family="poisson", selection_penalty=0.0, features={"age": Spline(kind=kind, k=12)}
    ).fit(X, y)
    grid = np.linspace(25.0, 40.0, 500)
    f = _native_curve(free, "age", grid, 0)
    assert np.ptp(f) > 1e3 * U * np.max(np.abs(f))


def test_natural_cr_with_flat_keeps_end_curvature_rows():
    x = np.linspace(17.0, 80.0, 500)
    knots = [25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0]

    def raw_rows(rule):
        spec = CubicRegressionSpline(knots=knots, constraint=rule)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # knots at 30 and 40: no warning
            spec.build(x)
        return build_shape_constraints_raw(spec)

    convex = raw_rows(C.convex)
    combined = raw_rows(C.convex & C.flat.within(30, 40))
    # f'' is zero at the knots 30, 35 and 40; the natural ends drop f''(17) and f''(80).
    assert combined.A.shape[0] == convex.A.shape[0] - 3


def _built(spec, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return spec.build(x)


def test_flat_does_not_reach_tensor():
    rng = np.random.default_rng(8)
    age, z = rng.uniform(17.0, 80.0, 2000), rng.uniform(0.0, 1.0, 2000)
    widths = {}
    for label, rule in (("free", None), ("flat", C.flat.within(25, 40))):
        parents = {"age": PSpline(n_knots=10, constraint=rule), "z": PSpline(n_knots=6)}
        main = _built(parents["age"], age)
        parents["z"].build(z)
        tensor = TensorInteraction("age", "z")
        tensor.build(age, z, parents)
        widths[label] = (main.n_cols, tensor._marginal1.K_eff)
    assert widths["flat"][0] < widths["free"][0]  # the rule binds the main effect
    assert widths["flat"][1] == widths["free"][1]


def test_flat_does_not_reach_spline_categorical():
    rng = np.random.default_rng(9)
    age = rng.uniform(17.0, 80.0, 2000)
    region = rng.choice(["N", "S", "W"], 2000)
    widths = {}
    for label, rule in (("free", None), ("flat", C.flat.within(25, 40))):
        parents = {"age": PSpline(n_knots=10, constraint=rule), "region": Categorical()}
        main = _built(parents["age"], age)
        parents["region"].build(region)
        varying = SplineCategorical("age", "region")
        varying.build(age, region, parents)
        widths[label] = (main.n_cols, varying._projection.shape[1])
    assert widths["flat"][0] < widths["free"][0]
    assert widths["flat"][1] == widths["free"][1]
