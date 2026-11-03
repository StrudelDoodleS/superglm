"""Shape rules against two independent references.

``test_matches_general_solver_at_fixed_smoothing`` rebuilds the penalised
Poisson objective the fit's active-set QP solved, from the fitted model's
design, penalty and stacked rows ``A_all beta >= b_all`` (cap rows included,
with their non-zero bounds), and solves it with scipy's general-purpose
``trust-constr`` interior-point method (Byrd, Hribar and Nocedal, "An interior
point algorithm for large-scale nonlinear programming", SIAM J. Optim. 9(4),
1999).  A flat stretch is not a row there: ``_SplineBase._apply_constraints``
absorbs it into the term's basis as an equality (Wood 2017, section 1.8.1), so
both solvers search the same flat-holding subspace, and the test checks the
flat on the fitted curve instead.

``test_scam_shapes_agree`` compares unpenalised increasing/decreasing,
convex/concave fits with scam's (Pya, N. and Wood, S. N. (2015), "Shape
constrained additive models", Statistics and Computing 25, 543-559), pinned in
``fixtures/scam_shape_rules.json`` by ``oracles/scam_shape_rules.R`` with the
data scam fitted.  scam is GPL and is run only as a black box; the test needs
the JSON alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint, minimize, nnls

import superglm.solvers.irls_direct._finalise as irls_finalise
from superglm import Constraint, Spline, SuperGLM
from superglm.constraints import shape_derivative_matrix
from superglm.types import LambdaPolicy

C = Constraint.fit
U = 2.0**-53
_SCAM_FIXTURE = Path(__file__).parent / "fixtures" / "scam_shape_rules.json"

# scam's shape name -> our rules, and the signs of f' and f'' they hold
SCAM_SHAPES = {
    "micv": (C.increasing & C.concave, 1.0, -1.0),
    "micx": (C.increasing & C.convex, 1.0, 1.0),
    "mdcv": (C.decreasing & C.concave, -1.0, -1.0),
    "mdcx": (C.decreasing & C.convex, -1.0, 1.0),
}


def _native_curve(model, name: str, grid: np.ndarray, order: int) -> np.ndarray:
    """The fitted term's ``order``-th derivative on ``grid``, from its exact basis."""
    group = next(group for group in model._groups if group.name == name)
    return shape_derivative_matrix(model._specs[name], grid, order) @ model.result.beta[group.sl]


# ── a general-purpose solver at fixed smoothing ─────────────────────────────


def _fixed_smoothing_fit(monkeypatch) -> tuple[SuperGLM, pd.DataFrame, np.ndarray, dict]:
    """A capped concave term and a U-shaped term at fixed lambda, and the system it solved.

    The rows and the penalty are read where the finaliser is handed the
    fitted state (as ``test_monotone_fit`` reads them), not rebuilt.  25 and
    40 are knots of ``z`` (range 10 to 85, nine interior knots), so its flat
    stretch holds exactly on ``[25, 40]``.  The PIRLS tolerance is tightened
    from its default ``1e-6``, at which the fit stops with a stationarity
    residual of ``4e-8`` of the gradient scale, to ``1e-10``, where it reaches
    the rounding floor (``8 u``; ``1e-12`` gives the same).  600 rows keep the
    rounding of the residual itself, at most ``gamma_(n+p)`` of the scale,
    below the ``1e3 u`` the KKT check asks for.
    """
    handed: list[dict] = []
    real_system = irls_finalise._final_centred_system

    def system(fit):
        out = real_system(fit)
        step = fit.step
        handed.append(
            {
                "A": np.array(step.A_all),
                "b": np.array(step.b_all),
                "active": list(step.active_set),
                "penalty": np.array(out[0].penalty),
            }
        )
        return out

    monkeypatch.setattr(irls_finalise, "_final_centred_system", system)
    rng = np.random.default_rng(29)
    n = 600
    x = rng.uniform(0.0, 10.0, n)
    z = rng.integers(10, 86, n).astype(np.float64)
    eta = (
        -1.0
        + 1.5 * (1.0 - np.exp(-x / 2.0))
        + np.where(z < 25, 0.08 * (25 - z), np.where(z > 40, 0.02 * (z - 40), 0.0))
    )
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    X = pd.DataFrame({"x": x, "z": z})
    fixed = LambdaPolicy.fixed(1.0)
    rules_x = C.increasing & C.concave & C.cap(0.6, 1.8)
    rules_z = C.decreasing.below(25) & C.flat.within(25, 40) & C.increasing.above(40)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "x": Spline(kind="ps", k=10, constraint=rules_x, lambda_policy=fixed),
            "z": Spline(kind="ps", n_knots=9, constraint=rules_z, lambda_policy=fixed),
        },
    )
    model.fit_reml(X, y, pirls_tol=1e-10, max_pirls_iter=200)
    return model, X, y, handed[-1]


def _augmented(design: np.ndarray, penalty: np.ndarray, A: np.ndarray):
    """The design, penalty and rows with the unpenalised intercept as column 0."""
    n, p = design.shape
    design_z = np.hstack([np.ones((n, 1)), design])
    penalty_z = np.zeros((p + 1, p + 1))
    penalty_z[1:, 1:] = penalty
    return design_z, penalty_z, np.hstack([np.zeros((A.shape[0], 1)), A])


def _gradient_terms(design_z, penalty_z, y, z):
    """The Poisson penalised objective's gradient and the sum of its absolute terms."""
    residual = np.exp(design_z @ z) - y
    gradient = design_z.T @ residual + penalty_z @ z
    scale = np.abs(design_z).T @ np.abs(residual) + np.abs(penalty_z) @ np.abs(z)
    return gradient, scale


def _general_solver(design, y, penalty, A, b) -> np.ndarray:
    """Minimise ``sum(mu - y eta) + beta' S beta / 2`` subject to ``A beta >= b`` with trust-constr.

    Returns ``(intercept, beta)``.  It starts from the flat curve at the mean
    count, which every row admits.  Its tolerances are tightened and its
    barrier started small: at ``gtol=1e-12, xtol=1e-14, barrier_tol=1e-14``
    it stopped with rows still ``1e-9`` slack, and agreed with our fit to
    ``2e-7`` on 2000 rows against ``2e-9`` with these settings.
    """
    design_z, penalty_z, A_z = _augmented(design, penalty, A)

    def objective(z):
        eta = design_z @ z
        return float(np.sum(np.exp(eta) - y * eta) + 0.5 * z @ penalty_z @ z)

    def jacobian(z):
        return _gradient_terms(design_z, penalty_z, y, z)[0]

    def hessian(z):
        return (design_z.T * np.exp(design_z @ z)) @ design_z + penalty_z

    start = np.zeros(design_z.shape[1])
    start[0] = np.log(np.mean(y))
    result = minimize(
        objective,
        start,
        jac=jacobian,
        hess=hessian,
        method="trust-constr",
        constraints=[LinearConstraint(A_z, b, np.inf)],
        options={
            "gtol": 1e-14,
            "xtol": 1e-18,
            "barrier_tol": 1e-18,
            "initial_barrier_parameter": 1e-6,
            "initial_barrier_tolerance": 1e-6,
            "maxiter": 20000,
        },
    )
    assert result.status in (1, 2), result.message
    return result.x


def _kkt_residual(design, y, penalty, A, b, active, z) -> tuple[float, float]:
    """The KKT residuals of ``z = (intercept, beta)`` on the penalised objective.

    Multipliers are the non-negative least-squares fit of the gradient on the
    QP's active rows.  Returns the stationarity residual relative to the
    gradient scale (the sum of the absolute terms of the gradient and of
    ``A' lambda``, per coordinate) and the worst feasibility or
    complementarity slack relative to its row's scale ``|A||beta| + |b|``.
    """
    design_z, penalty_z, A_z = _augmented(design, penalty, A)
    gradient, scale = _gradient_terms(design_z, penalty_z, y, z)
    held = A_z[active]
    multipliers, _ = nnls(held.T, gradient)
    stationarity = gradient - held.T @ multipliers
    scale = scale + np.abs(held).T @ multipliers
    slack = (A_z @ z - b) / (np.abs(A_z) @ np.abs(z) + np.abs(b))
    on_rows = max(-float(np.min(slack)), float(np.max(np.abs(slack[active]))))
    return float(np.max(np.abs(stationarity) / scale)), on_rows


def test_matches_general_solver_at_fixed_smoothing(monkeypatch):
    model, X, y, system = _fixed_smoothing_fit(monkeypatch)
    assert model.result.converged
    design = model._dm.toarray()
    z = np.concatenate([[model.result.intercept], model.result.beta])
    A, b = system["A"], system["b"]
    assert np.count_nonzero(b) == 4  # the cap's two ends, each bounded twice

    # the cap binds the printed curve, so its rows shape the optimum
    printed = model.term_inference("x", with_se=False, n_points=2000, centering="native")
    assert np.min(printed.relativity) == pytest.approx(0.6, abs=1e-3)

    reference = _general_solver(design, y, system["penalty"], A, b)
    mu_reference = np.exp(_augmented(design, system["penalty"], A)[0] @ reference)
    np.testing.assert_allclose(model.predict(X), mu_reference, rtol=1e-6, atol=0.0)

    stationarity, on_rows = _kkt_residual(design, y, system["penalty"], A, b, system["active"], z)
    assert stationarity <= 1e3 * U
    assert on_rows <= 1e3 * U

    # the flat stretch is absorbed into the basis and holds exactly on [25, 40]
    grid = np.linspace(25.0, 40.0, 500)
    scale = np.max(np.abs(_native_curve(model, "z", np.linspace(10.0, 85.0, 2000), 0)))
    assert np.max(np.abs(_native_curve(model, "z", grid, 1))) <= 1e3 * U * scale


# ── scam's shape-constrained P-splines ──────────────────────────────────────


@pytest.fixture(scope="module")
def scam_fixture() -> dict:
    return json.loads(_SCAM_FIXTURE.read_text(encoding="utf-8"))


def _assert_shape_at_points(x, eta, slope_sign, curvature_sign) -> None:
    """``eta`` has the shape at the distinct points ``x``: signed differences and slope changes.

    Each slope carries the rounding of its two values over its spacing, so
    the slope test's tolerance scales with the inverse spacings beside it.
    """
    x, first = np.unique(x, return_index=True)
    eta = np.asarray(eta)[first]
    tol = 1e3 * U * np.max(np.abs(eta))
    dx = np.diff(x)
    assert np.min(slope_sign * np.diff(eta)) >= -tol
    slope_change = curvature_sign * np.diff(np.diff(eta) / dx)
    assert np.all(slope_change >= -2.0 * tol * (1.0 / dx[:-1] + 1.0 / dx[1:]))


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    ratio = np.where(y > 0.0, y / mu, 1.0)
    return float(2.0 * np.sum(y * np.log(ratio) - (y - mu)))


def _scam_case(scam_fixture, shape):
    """The case's data, scam's outputs, and our unpenalised fit on scam's knots.

    scam's cubic basis has ``k + 4`` evenly spaced knots, its fourth and
    fourth-last at the data range; ours takes the six inside as its interior
    knots.  Its open knot vector pads the range by 0.001 of it where scam's
    starts at the data, so the two bases differ outside the knots inside.
    """
    case = scam_fixture[shape]
    rules = SCAM_SHAPES[shape][0]
    x = np.asarray(case["data"]["x"], dtype=np.float64)
    y = np.asarray(case["data"]["y"], dtype=np.float64)
    knots = np.asarray(case["knots"], dtype=np.float64)
    assert case["converged"]
    assert knots.size == 14
    assert (knots[3], knots[-4]) == (x.min(), x.max())
    X = pd.DataFrame({"x": x})
    spline = Spline(kind="ps", knots=knots[4:-4], constraint=rules)
    model = SuperGLM(
        family="poisson", selection_penalty=0.0, spline_penalty=0.0, features={"x": spline}
    ).fit(X, y)
    assert model.result.converged
    return case, x, y, model


@pytest.mark.parametrize("shape", sorted(SCAM_SHAPES))
def test_scam_shapes_hold(scam_fixture, shape):
    """Our unpenalised fit has scam's shape on a dense grid, and fits no worse than scam's.

    scam holds the shape through signed differences of reparameterised
    coefficients, our rows the curvature exactly at the knots and, with it,
    the slope at the ends, on a basis padded beyond the data; the feasible
    sets differ, and on these data our fit is never the worse of the two.
    """
    case, x, y, model = _scam_case(scam_fixture, shape)
    _, slope_sign, curvature_sign = SCAM_SHAPES[shape]
    spec = model._specs["x"]
    grid = np.linspace(spec._lo, spec._hi, 2000)
    tol = 1e3 * U * np.max(np.abs(_native_curve(model, "x", grid, 0)))
    assert np.min(slope_sign * _native_curve(model, "x", grid, 1)) >= -tol
    assert np.min(curvature_sign * _native_curve(model, "x", grid, 2)) >= -tol
    # the oracle's own values have the shape at the data (checks the fixture, not our code)
    _assert_shape_at_points(x, case["linear_predictor"], slope_sign, curvature_sign)
    ours = _poisson_deviance(y, model.predict(pd.DataFrame({"x": x})))
    scam = _poisson_deviance(y, np.asarray(case["fitted_values"]))
    assert ours <= scam * (1.0 + 1e3 * U)


# scam 1.2.23 at sp=0 stops above the optimum of its own problem on three of
# the four datasets: solving its stated conditions (signed first and second
# differences of the coefficients, on its knots) with trust-constr gives a
# lower deviance than scam's (micv and mdcv: ours, to 1e-8; micx: 428.59894,
# ours 428.59884 with the end-slope row), and tightening scam.control's
# devtol.fit and steptol.fit to 1e-13 leaves scam's deviance unchanged.
# Measured: max |mu / mu_scam - 1| and the deviances, ours against scam's.
_SCAM_STALLS = {
    "micv": "max rel 1.38e-1 at x = 0.000665; deviance 429.8287 against 430.4398",
    "micx": "max rel 4.08e-2 at x = 0.000966; deviance 428.5988 against 428.6233",
    "mdcv": "max rel 5.96e-2 at x = 0.995959; deviance 421.8743 against 422.0504",
}


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            shape,
            marks=pytest.mark.xfail(
                shape in _SCAM_STALLS,
                reason=f"scam stops short of its optimum: {_SCAM_STALLS.get(shape)}",
                raises=AssertionError,
                strict=True,
            ),
        )
        for shape in sorted(SCAM_SHAPES)
    ],
)
def test_scam_shapes_agree(scam_fixture, shape):
    """Our fitted means are scam's to 2e-2 relative.

    Loose by design: the two hold the shape through different sufficient
    conditions, and the bases differ outside the shared knots.
    """
    case, x, _, model = _scam_case(scam_fixture, shape)
    np.testing.assert_allclose(
        model.predict(pd.DataFrame({"x": x})), case["fitted_values"], rtol=2e-2, atol=0.0
    )
