import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from superglm import BSplineSmooth, Constraint, CubicRegressionSpline, PSpline, SuperGLM
from superglm.constraints import (
    _certificate_candidates,
    _normalized_nonzero_shape_rows,
    _shape_constraint_rows,
    shape_constraint_certificate,
)
from superglm.features._spline_constraints import (
    build_curvature_difference_constraints,
    curvature_difference_operator,
)
from superglm.solvers.scop import build_scop_solver_reparam


def _clamped_irregular_knots(q: int, degree: int = 3) -> np.ndarray:
    n_interior = q - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1] ** 1.8
    return np.concatenate(
        (
            np.zeros(degree + 1),
            interior,
            np.ones(degree + 1),
        )
    )


def test_bspline_fit_convex_builds_second_difference_constraints():
    x = np.linspace(0.0, 1.0, 32)
    spec = BSplineSmooth(n_knots=8, constraint=Constraint.fit.convex)
    group = spec.build(x)
    assert group.constraints is not None
    assert group.monotone_engine == "qp"
    assert group.constraints.A.shape[0] > 0


@pytest.mark.parametrize("kind", ["convex", "concave"])
def test_cr_builds_fit_time_curvature_constraints(kind):
    x = np.linspace(0.0, 1.0, 32)
    constraint = getattr(Constraint.fit, kind)
    spec = CubicRegressionSpline(n_knots=8, constraint=constraint)

    group = spec.build(x)

    assert group.constraints is not None
    assert group.monotone_engine == "qp"
    assert group.constraints.A.shape == (spec._n_basis - 4, group.n_cols)


@pytest.mark.parametrize("kind", ["convex", "concave"])
def test_cr_natural_endpoint_equalities_do_not_create_large_scale_qp_rows(kind):
    rng = np.random.default_rng(217)
    x = np.sort(np.concatenate(([0.0, 1.0], rng.beta(0.3, 2.2, size=398))))
    sign = 1.0 if kind == "convex" else -1.0
    y = 1e6 * sign * (x - 0.4) ** 2
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": CubicRegressionSpline(
                n_knots=8,
                knot_strategy="quantile",
                constraint=getattr(Constraint.fit, kind),
            )
        },
    ).fit(frame, y)

    group = next(group for group in model._groups if group.name == "x")
    beta = model.result.beta[group.sl]
    certificate = shape_constraint_certificate(model._specs["x"], beta, kind)
    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)

    assert model.result.converged
    assert group.constraints.A.shape[0] == model._specs["x"]._n_basis - 4
    assert r_squared > 0.94
    assert certificate.minimum_scaled_slack >= -1e-10


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("kind", ["convex", "concave"])
def test_knot_aware_curvature_rows_match_exact_derivative_on_irregular_knots(
    degree,
    kind,
):
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.array([0.01, 0.07, 0.20, 0.71, 0.93]),
            np.ones(degree + 1),
        )
    )
    n_basis = len(knots) - degree - 1
    spans = knots[degree + 1 : degree + n_basis] - knots[1:n_basis]
    slopes = np.linspace(0.8, 1.0, n_basis - 1)
    coefficients = np.concatenate(([0.0], np.cumsum(slopes * spans)))
    sign = 1.0 if kind == "convex" else -1.0
    coefficients *= sign

    constraints = build_curvature_difference_constraints(knots, degree, kind)
    old_unweighted = sign * np.diff(np.eye(n_basis), n=2, axis=0)

    # This fixture is deliberately rejected by the old plain coefficient
    # differences even though its exact derivative coefficients have the
    # requested order.
    assert np.min(old_unweighted @ coefficients) < -0.1
    assert np.min(constraints.A @ coefficients) > 1e-6

    if degree == 1:
        signed_curvature = np.diff(slopes)
    else:
        grid = np.linspace(0.0, 1.0, 2001)
        signed_curvature = sign * SciPyBSpline(
            knots,
            coefficients,
            degree,
        )(grid, nu=2)
    assert np.min(signed_curvature) > 0.0


@pytest.mark.parametrize("kind", ["convex", "concave"])
def test_curvature_scop_retains_free_affine_slope(kind):
    knots = _clamped_irregular_knots(7)
    reparam = build_scop_solver_reparam(
        7,
        kind=kind,
        knots=knots,
        degree=3,
        domain=(0.0, 1.0),
    )
    beta = np.linspace(-0.7, 0.8, reparam.q)

    mapped = reparam.forward(beta)
    jacobian = reparam.jacobian_diagonal(beta)
    second_derivative = reparam.second_derivative_diagonal(beta)
    penalty = reparam.penalty_matrix()

    assert reparam.q == 6
    assert reparam.free_dim == 1
    assert mapped[0] == beta[0]
    assert jacobian[0] == 1.0
    assert second_derivative[0] == 0.0
    assert np.all(mapped[1:] > 0.0)
    np.testing.assert_allclose(jacobian[1:], mapped[1:])
    np.testing.assert_allclose(second_derivative[1:], mapped[1:])
    np.testing.assert_array_equal(penalty[0], np.zeros(reparam.q))
    np.testing.assert_array_equal(penalty[:, 0], np.zeros(reparam.q))


def test_bspline_default_curvature_qp_and_exact_certificate_agree():
    x = np.linspace(0.0, 1.0, 500)
    # The open B-spline knot vector has deliberately unequal end spans.  This
    # convex transition is accepted by the old plain coefficient-difference
    # QP while violating the exact derivative certificate near the boundary.
    y = np.logaddexp(0.0, 12.0 * (x - 0.7))
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": BSplineSmooth(
                n_knots=8,
                constraint=Constraint.fit.convex,
            )
        },
    ).fit(frame, y)

    group = next(group for group in model._groups if group.name == "x")
    beta = model.result.beta[group.sl]
    certificate = shape_constraint_certificate(model._specs["x"], beta, "convex")
    eps = np.finfo(np.float64).eps
    qp_tolerance = (
        100.0 * eps * (1.0 + np.linalg.norm(group.constraints.A, ord=np.inf) * np.linalg.norm(beta))
    )
    certificate_tolerance = 1000.0 * eps * (1.0 + beta.size)

    assert np.min(group.constraints.A @ beta) >= -qp_tolerance
    assert certificate.minimum_scaled_slack >= -certificate_tolerance
    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r_squared > 0.99


def test_bspline_curvature_rows_cover_only_the_public_fitted_domain():
    x = np.linspace(0.0, 1.0, 500)
    spec = BSplineSmooth(
        n_knots=8,
        penalty="none",
        constraint=Constraint.fit.convex,
    )
    spec.build(x)

    padded_operator = curvature_difference_operator(
        spec._knots,
        spec.degree,
        normalize=False,
    )
    padded_curvature = np.ones(padded_operator.shape[0])
    padded_curvature[0] = -1e-3
    coefficients = np.linalg.lstsq(
        padded_operator,
        padded_curvature,
        rcond=None,
    )[0]
    public_operator = curvature_difference_operator(
        spec._knots,
        spec.degree,
        domain=(spec._lo, spec._hi),
        normalize=False,
    )
    exact_curvature = SciPyBSpline(
        spec._knots,
        coefficients,
        spec.degree,
    )(x, nu=2)

    assert np.min(padded_operator @ coefficients) < 0.0
    assert np.min(public_operator @ coefficients) > 0.2
    assert np.min(spec._build_monotone_constraints_raw().A @ coefficients) > 1e-3
    assert np.min(exact_curvature) > 0.2


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_curvature_row_normalization_is_stable_at_extreme_predictor_scales(scale):
    knots = scale * _clamped_irregular_knots(8)
    operator = curvature_difference_operator(
        knots,
        3,
        domain=(0.0, scale),
    )

    assert np.all(np.isfinite(operator))
    np.testing.assert_allclose(np.linalg.norm(operator, axis=1), 1.0, rtol=2e-15)


@pytest.mark.parametrize("basis_type", [PSpline, BSplineSmooth])
def test_fit_time_curvature_rejects_degrees_above_exact_linear_cone(basis_type):
    with pytest.raises(NotImplementedError, match=r"degree <= 3"):
        basis_type(
            degree=4,
            constraint=Constraint.fit.convex,
        )


def test_shape_certificate_reports_raw_and_scaled_minima_independently():
    x = np.linspace(0.0, 1.0, 200)
    spec = BSplineSmooth(
        knots=np.array([1e-6, 1e-4, 0.01, 0.2, 0.8, 0.99, 0.9999]),
        penalty="none",
        constraint=Constraint.postfit.convex,
    )
    group = spec.build(x)
    spec._R_inv = group.projection
    beta = np.array(
        [
            -1.09405017,
            -0.64356926,
            -1.34302635,
            0.12264991,
            1.54326244,
            0.39895304,
            -0.36978445,
            0.8748653,
            -0.24319214,
            -0.04010056,
        ]
    )

    candidates = _certificate_candidates(spec, beta, "convex")
    rows = _shape_constraint_rows(spec, candidates, "convex")
    normalized, _, keep = _normalized_nonzero_shape_rows(rows)
    values = rows[keep] @ beta
    scaled = normalized @ beta
    kept_points = candidates[keep]
    raw_worst = int(np.argmin(values))
    scaled_worst = int(np.argmin(scaled))
    certificate = shape_constraint_certificate(spec, beta, "convex")

    assert raw_worst != scaled_worst
    np.testing.assert_allclose(certificate.minimum_signed_derivative, values[raw_worst])
    np.testing.assert_allclose(certificate.minimum_scaled_slack, scaled[scaled_worst])
    assert certificate.worst_x == kept_points[raw_worst]
