"""Curvature-geometry contracts for knots that land on the fitted boundary.

Data-driven knot placement puts an interior knot exactly on the fitted
boundary whenever a predictor carries a point mass at its minimum -- the
bonus-malus / age-floor shape.  The exact fit-time curvature geometry has to
keep working on that layout, in both the SCOP (PSpline) and QP
(CubicRegressionSpline) engines.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from superglm import Constraint, CubicRegressionSpline, PSpline, SuperGLM
from superglm.features._spline_constraints import curvature_difference_operator
from superglm.solvers.scop import build_scop_reparam, build_scop_solver_reparam

DEGREE = 3
# 12% of the rows sit exactly at the minimum.  With ``n_knots=10`` the
# ``quantile_rows`` probabilities start at 1 / 11, so the point mass swallows
# the first probability and ``resolve_interior_knots`` returns
# ``interior[0] == x.min()`` exactly.
POINT_MASS_FRACTION = 0.12
N_KNOTS = 10


def _point_mass_at_minimum() -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(0)
    n = 3000
    n_mass = int(POINT_MASS_FRACTION * n)
    x = np.concatenate((np.zeros(n_mass), rng.uniform(1e-9, 1.0, n - n_mass)))
    rng.shuffle(x)
    return x, rng


def _assert_first_interior_knot_is_on_the_boundary(spec_type) -> None:
    """Fail loudly if the fixture ever stops producing the hard layout."""
    x, _ = _point_mass_at_minimum()
    probe = spec_type(n_knots=N_KNOTS, knot_strategy="quantile_rows")
    probe.build(x)
    assert probe._knots[probe.degree + 1] == probe._lo


def _padded_knots_with_boundary_knot(
    interior: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    degree: int = DEGREE,
) -> np.ndarray:
    """Reproduce ``assemble_open_knot_vector`` for a chosen interior set."""
    span = hi - lo
    lo_effective = lo - 0.001 * span
    hi_effective = hi + 0.001 * span
    inner = np.concatenate(([lo_effective], interior, [hi_effective]))
    dx_lo = inner[1] - inner[0]
    dx_hi = inner[-1] - inner[-2]
    lower = lo_effective - dx_lo * np.arange(degree, 0, -1)
    upper = hi_effective + dx_hi * np.arange(1, degree + 1)
    return np.concatenate((lower, inner, upper))


def _row_normalized(rows: np.ndarray) -> np.ndarray:
    max_scaled = rows / np.max(np.abs(rows), axis=1)[:, None]
    return max_scaled / np.linalg.norm(max_scaled, axis=1)[:, None]


def _minimum_second_difference(values: np.ndarray) -> float:
    second = np.diff(values, n=2)
    return float(np.min(second) / max(1.0, float(np.max(np.abs(second)))))


# --------------------------------------------------------------------------
# SCOP engine (PSpline): a knot on the boundary must still build a square map
# --------------------------------------------------------------------------


def test_scop_curvature_map_is_built_when_a_knot_sits_on_the_fitted_boundary():
    interior = np.concatenate(([0.0], np.linspace(0.0, 1.0, 11)[1:-1]))
    knots = _padded_knots_with_boundary_knot(interior)
    q = len(knots) - DEGREE - 1
    grid = np.linspace(0.0, 1.0, 2001)
    rng = np.random.default_rng(11)

    reparam = build_scop_reparam(q, kind="convex", knots=knots, degree=DEGREE, domain=(0.0, 1.0))

    assert reparam.Sigma.shape == (q, q)
    affine = SciPyBSpline(knots, reparam.Sigma[:, :2], DEGREE)(grid, nu=2)
    np.testing.assert_allclose(affine, 0.0, atol=1e-9)
    for _ in range(5):
        gamma = reparam.forward(rng.normal(size=q))
        second_derivative = SciPyBSpline(knots, gamma, DEGREE)(grid, nu=2)
        scale = max(1.0, float(np.max(np.abs(second_derivative))))
        assert np.min(second_derivative) >= -1e-9 * scale
        assert np.max(second_derivative) > 0.0


@pytest.mark.parametrize("discrete", [False, True])
def test_fit_time_convex_pspline_fits_a_point_mass_at_the_predictor_minimum(discrete):
    _assert_first_interior_knot_is_on_the_boundary(PSpline)
    x, rng = _point_mass_at_minimum()
    y = 2.0 * x**2 + 0.5 * x + rng.normal(0.0, 0.05, x.size)
    frame = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": PSpline(
                n_knots=N_KNOTS,
                knot_strategy="quantile_rows",
                constraint=Constraint.fit.convex,
            )
        },
    ).fit(frame, y)

    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    grid = pd.DataFrame({"x": np.linspace(x.min(), x.max(), 501)})

    assert model.result.converged
    assert r_squared > 0.99
    assert _minimum_second_difference(model.predict(grid)) >= -1e-8


# --------------------------------------------------------------------------
# QP engine (CubicRegressionSpline): the clamped vector repeats the knot
# --------------------------------------------------------------------------


def test_curvature_operator_accepts_a_clamped_knot_repeated_onto_the_boundary():
    interior = np.concatenate(([0.0], np.linspace(0.0, 1.0, 11)[1:-1]))
    knots = np.concatenate((np.zeros(DEGREE + 1), interior, np.ones(DEGREE + 1)))
    n_basis = len(knots) - DEGREE - 1
    # The knot on the boundary is not an interior breakpoint of the piecewise
    # linear second derivative, so the exact probe set drops it.
    breakpoints = np.unique(np.clip(knots[DEGREE : n_basis + 1], 0.0, 1.0))

    operator = curvature_difference_operator(knots, DEGREE, domain=(0.0, 1.0))

    assert operator.shape == (n_basis - 3, n_basis)
    reference = np.asarray(
        SciPyBSpline(knots, np.eye(n_basis), DEGREE, extrapolate=False)(breakpoints, nu=2)
    )
    np.testing.assert_allclose(operator, _row_normalized(reference), atol=1e-12)


def test_fit_time_convex_cr_spline_fits_a_point_mass_at_the_predictor_minimum():
    _assert_first_interior_knot_is_on_the_boundary(CubicRegressionSpline)
    x, rng = _point_mass_at_minimum()
    y = 2.0 * x**2 + 0.5 * x + rng.normal(0.0, 0.05, x.size)
    frame = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": CubicRegressionSpline(
                n_knots=N_KNOTS,
                knot_strategy="quantile_rows",
                constraint=Constraint.fit.convex,
            )
        },
    ).fit(frame, y)

    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    grid = pd.DataFrame({"x": np.linspace(x.min(), x.max(), 501)})

    assert model.result.converged
    assert r_squared > 0.99
    assert _minimum_second_difference(model.predict(grid)) >= -1e-8


# --------------------------------------------------------------------------
# Solver-space reparameterization input contract
# --------------------------------------------------------------------------


def test_solver_space_initialize_from_gamma_rejects_a_raw_space_gamma():
    q_raw = 8
    reparam = build_scop_solver_reparam(q_raw, direction="increasing")

    assert reparam.q == q_raw - 1
    with pytest.raises(ValueError, match="solver-space"):
        reparam.initialize_from_gamma(np.ones(q_raw))


def test_solver_space_initialize_from_gamma_inverts_the_solver_forward_map():
    reparam = build_scop_solver_reparam(8, direction="increasing")
    beta_eff = np.linspace(-1.2, 0.9, reparam.q)

    recovered = reparam.initialize_from_gamma(reparam.forward(beta_eff))

    np.testing.assert_allclose(recovered, beta_eff, atol=1e-12)
