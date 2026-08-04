import numpy as np
import pandas as pd
import pytest

from superglm import BSplineSmooth, Constraint, CubicRegressionSpline, PSpline, SuperGLM
from superglm.constraints import shape_constraint_certificate


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
