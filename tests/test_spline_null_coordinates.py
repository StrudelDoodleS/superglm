"""A spline's penalty null space is exact in its term coordinates and reaches REML unrounded.

On quantile knots over five decades an integrated-derivative penalty's nonzero
spectrum spans 10 to 14 decades, so a formed matrix moves its null space off the
polynomials it annihilates by ``u ||Omega|| / mu_min`` (Davis-Kahan), about
1e-4, differently on each BLAS kernel: at a fixed lambda the fit leaks onto the
line by 8e-5, and REML's lambda moves by percents.  Each kind states its null
basis; the term holds it on leading coordinates whose penalty rows are exact
zeros, kept through the SSP whitening into REML.

The references are an independent mpmath natural cubic smoothing spline on the
fit's own knots (Green and Silverman, *Nonparametric Regression and Generalized
Linear Models*, 1994, ch. 2): ``cr`` and ``cr_cardinal`` span the natural cubic
splines on those knots and penalise ``int f''(u)^2 du`` over ``u = (x - x_0) /
hbar``, ``hbar`` the mean knot interval, with the intercept unpenalised.  Its
fitted values at a fixed lambda, and its Gaussian REML optimum with the scale
profiled (Wood 2011, JRSSB 73, sec. 2), the root of the criterion's derivative
in ``log lambda``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM

U = 2.0**-53
FIXED_LAMBDA = 3000.0
# _exact_reml_optimum on the fixture, stored because the root takes seconds;
# test_the_stored_reml_optima_are_the_oracles recomputes them.
LAMBDA_STAR = {0.08: 2964.1283569199913, 0.09: 1027.718621024655}


def _fixture(amplitude: float) -> tuple[pd.DataFrame, np.ndarray]:
    """A shallow REML basin: a line plus a small wiggle, on five decades (EDF near 2)."""
    x = np.geomspace(1.0, 1e5, 200)
    noise = np.random.default_rng(74).normal(0.0, 0.15, x.size)
    return pd.DataFrame({"x": x}), 1.0 + x / 1e5 + amplitude * np.sin(np.log(x)) + noise


def _model(kind: str, discrete: bool = True, **settings) -> SuperGLM:
    spline = Spline(kind=kind, n_knots=10, knot_strategy="quantile_rows")
    return SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={"x": spline},
        **settings,
    )


def _knots(model: SuperGLM) -> tuple[float, ...]:
    """The fit's distinct knots, boundaries included."""
    spec = model._specs["x"]
    knots = spec._cr_knots if hasattr(spec, "_cr_knots") else np.unique(spec._knots)
    return tuple(float(k) for k in knots)


def _natural_smoother(knots, x, dps):
    """``(C, S)``: the map from knot values to fitted values at ``x``, and the penalty on them."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = dps
    k = [mp.mpf(v) for v in knots]
    K = len(k)
    hbar = (k[-1] - k[0]) / (K - 1)
    u = [(v - k[0]) / hbar for v in k]
    h = [u[i + 1] - u[i] for i in range(K - 1)]
    Q, R = mp.zeros(K, K - 2), mp.zeros(K - 2, K - 2)
    for j in range(K - 2):
        Q[j, j], Q[j + 1, j], Q[j + 2, j] = 1 / h[j], -1 / h[j] - 1 / h[j + 1], 1 / h[j + 1]
        R[j, j] = (h[j] + h[j + 1]) / 3
        if j < K - 3:
            R[j, j + 1] = R[j + 1, j] = h[j + 1] / 6
    G = mp.inverse(R) * Q.T  # knot values -> second derivatives at the interior knots
    C = mp.zeros(len(x), K)
    for r, value in enumerate(x):
        t = (mp.mpf(value) - k[0]) / hbar
        i = next(i for i in range(K - 1) if t <= u[i + 1] or i == K - 2)
        a, b = t - u[i], u[i + 1] - t
        C[r, i] += b / h[i]
        C[r, i + 1] += a / h[i]
        for j in range(K):
            lower = G[i - 1, j] if i >= 1 else 0
            upper = G[i, j] if i <= K - 3 else 0
            C[r, j] -= a * b * ((1 + b / h[i]) * lower + (1 + a / h[i]) * upper) / 6
    return C, Q * G


def _exact_fit(knots, x, y, lam, dps=40) -> np.ndarray:
    mp = pytest.importorskip("mpmath")
    C, S = _natural_smoother(knots, x, dps)
    response = mp.matrix([mp.mpf(v) for v in y])
    values = mp.lu_solve(C.T * C + mp.mpf(lam) * S, C.T * response)
    fitted = C * values
    return np.array([float(fitted[i]) for i in range(len(x))])


def _exact_reml_optimum(knots, x, y, dps=40) -> float:
    """The root in ``rho = log lambda`` of ``V = (n - 2)/2 log(RSS + pen) + log|C'C + lambda S|/2 - (K - 2) rho / 2``."""
    mp = pytest.importorskip("mpmath")
    C, S = _natural_smoother(knots, x, dps)
    n, K = C.rows, C.cols
    response = mp.matrix([mp.mpf(v) for v in y])
    gram, rhs, total = C.T * C, C.T * response, (response.T * response)[0]

    def slope(rho):
        lam = mp.e**rho
        inverse = mp.inverse(gram + lam * S)
        values = inverse * rhs
        residual = total - (rhs.T * values)[0]
        quadratic = (values.T * S * values)[0]
        trace = sum((inverse * S)[j, j] for j in range(K))
        return (n - 2) * lam * quadratic / (2 * residual) + lam * trace / 2 - mp.mpf(K - 2) / 2

    root = mp.findroot(
        slope, (mp.log(100), mp.log(1e6)), solver="illinois", tol=1e-30, maxsteps=200
    )
    return float(mp.e**root)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("kind", ["cr", "cr_cardinal", "bs", "ps"])
def test_the_null_rows_are_exact_zeros_through_to_reml(kind, discrete):
    """The term penalty and REML's solver penalty have exactly zero rows on the line.

    The centred line is the one null direction (a second-order penalty, less
    the intercept), and the solver coordinate whose rows are zero carries it:
    on the basis it is affine in ``x`` to the basis's rounding.
    """
    frame, y = _fixture(0.08)
    model = _model(kind, discrete)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    group = model._dm.group_matrices[0]
    (penalty,) = model._reml_penalties
    assert penalty.omega_ssp.shape[0] - int(penalty.rank) == 1
    for matrix in (group.omega, penalty.omega_ssp):
        assert np.flatnonzero(~np.any(matrix != 0.0, axis=1)).size == 1
    (null,) = np.flatnonzero(~np.any(penalty.omega_ssp != 0.0, axis=1))
    grid = np.geomspace(1.0, 1e5, 50)
    curve = model._specs["x"]._basis_matrix(grid) @ group.R_inv[:, null]
    line = np.column_stack((np.ones_like(grid), grid))
    residual = curve - line @ np.linalg.lstsq(line, curve, rcond=None)[0]
    assert np.max(np.abs(residual)) <= 2**8 * U * np.max(np.abs(curve))


@pytest.mark.parametrize("kind", ["cr", "cr_cardinal"])
def test_a_fixed_lambda_fits_the_exact_smoother(kind):
    """At a fixed lambda the fit is the exact natural smoothing spline's.

    The range block is still the formed penalty turned into the term's
    coordinates, whose smallest eigenvalue keeps a relative error of about
    ``u kappa_+`` (``kappa_+`` near 2e12 here), which moves the fit by about
    1e-6; a penalty leaking onto the line moves it by 8e-5.
    """
    frame, y = _fixture(0.08)
    model = _model(kind, spline_penalty=FIXED_LAMBDA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
    exact = _exact_fit(_knots(model), frame["x"], y, FIXED_LAMBDA)
    assert np.max(np.abs(model.predict(frame) - exact) / np.abs(exact)) <= 1e-5


@pytest.mark.parametrize("amplitude", [0.08, 0.09])
@pytest.mark.parametrize("kind", ["cr", "cr_cardinal"])
def test_reml_finds_the_exact_optimum(kind, amplitude):
    """REML's lambda is the exact smoother's REML optimum to the stop rule's resolution.

    These basins are shallow (EDF near 2), so a leak onto the line of
    ``u ||Omega||`` tilts the criterion enough to move lambda by 2 to 7 percent;
    with the null rows exact the stop leaves it within about 5e-4.
    """
    frame, y = _fixture(amplitude)
    model = _model(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    estimate = model._reml_result.lambdas["x"]
    assert abs(estimate / LAMBDA_STAR[amplitude] - 1.0) <= 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("amplitude", [0.08, 0.09])
@pytest.mark.parametrize("kind", ["cr", "cr_cardinal"])
def test_the_stored_reml_optima_are_the_oracles(kind, amplitude):
    """Both kinds place the same knots here, so they share the optimum."""
    frame, y = _fixture(amplitude)
    model = _model(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    optimum = _exact_reml_optimum(_knots(model), frame["x"], y)
    assert optimum == pytest.approx(LAMBDA_STAR[amplitude], rel=1e-12)


@pytest.mark.parametrize("kind", ["cr", "bs", "ps"])
def test_a_null_space_its_rows_cannot_see_stays_unfitted(kind):
    """A spline on one observed value predicts that value's fit everywhere on its range.

    Its null directions are unpenalised and its rows cannot see them, so their
    columns are rounding; the term then declares no null basis and keeps its
    penalty as built, and the curve stays flat rather than following that
    rounding (by 1e2 here).
    """
    z = np.tile(np.linspace(0.0, 1.0, 31), 4)
    y = 2.0 + np.sin(8.0 * z) + 0.1 * np.random.default_rng(2026).normal(size=z.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=0.1,
        features={
            "x": Spline(kind=kind, n_knots=3, boundary=(0.0, 1.0)),
            "z": Spline(kind="ps", n_knots=4),
        },
    )
    model.fit(pd.DataFrame({"x": np.full(z.size, 0.5), "z": z}), y)
    predictions = model.predict(pd.DataFrame({"x": [0.2, 0.5, 0.8], "z": [0.3, 0.3, 0.3]}))
    assert np.ptp(predictions) <= 2**10 * U * np.max(np.abs(y))
