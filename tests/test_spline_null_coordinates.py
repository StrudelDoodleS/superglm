"""A spline's penalty null space is exact in its term coordinates and reaches REML unrounded.

On quantile knots over five decades an integrated-derivative penalty's nonzero
spectrum spans 10 to 14 decades, so a formed matrix moves its null space off the
polynomials it annihilates by ``u ||Omega|| / mu_min`` (Davis-Kahan), about
1e-4, differently on each BLAS kernel: at a fixed lambda the fit leaks onto the
line by 8e-5, and REML's lambda moves by percents.  Each kind states its null
basis; the term holds it on leading coordinates whose penalty rows are exact
zeros, kept through the SSP whitening into REML. Its range block is the Gram of
the penalty's square root (the quadrature root, the difference matrix or the
cardinal ``L^{-1} B``), whose rows keep the knot spacing's grading, rather
than the formed penalty turned into those coordinates. A multi-order term's
null spaces nest (the polynomials of lower degree inside those of higher), so
its penalties share the coordinates, each with its own exact null block.

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

from superglm import FactorSmooth, LambdaPolicy, Numeric, Spline, SuperGLM
from superglm.features._spline_penalties import _polynomial_coefficients
from superglm.features.spline import PSpline
from superglm.reml.scale import profile_gaussian_reml_scale

U = 2.0**-53
FIXED_LAMBDA = 3000.0
# A d1 penalty off leaves m=(1, 2) the m=2 spline (test_large_lambda_null_space).
D1_OFF = {"d1": LambdaPolicy.off()}
# _exact_reml_optimum on the fixture, stored because the root takes seconds;
# test_the_stored_reml_optima_are_the_oracles recomputes them.
LAMBDA_STAR = {0.08: 2964.1283569199913, 0.09: 1027.718621024655}


def _fixture(amplitude: float) -> tuple[pd.DataFrame, np.ndarray]:
    """A shallow REML basin: a line plus a small wiggle, on five decades (EDF near 2)."""
    x = np.geomspace(1.0, 1e5, 200)
    noise = np.random.default_rng(74).normal(0.0, 0.15, x.size)
    return pd.DataFrame({"x": x}), 1.0 + x / 1e5 + amplitude * np.sin(np.log(x)) + noise


def _model(kind: str, discrete: bool = True, m=2, policy=None, **settings) -> SuperGLM:
    spline = Spline(kind=kind, n_knots=10, m=m, knot_strategy="quantile_rows", lambda_policy=policy)
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


@pytest.mark.parametrize("kind", ["cr_cardinal", "cr"], ids=["cardinal-geometric", "cr-lognormal"])
@pytest.mark.parametrize("fit", ["fit", "fit_reml"])
def test_graded_penalty_range_matches_the_natural_smoother(kind, fit):
    """A genuine weak range direction still smooths at fixed lambda, in predictions and EDF."""
    mp = pytest.importorskip("mpmath")
    if kind == "cr_cardinal":
        x = np.geomspace(1, 1e5, 200)
    else:
        x = np.sort(np.random.default_rng(42).lognormal(0, 3, 200))
    t = x / x[-1]
    y = 1 + t + np.sin(4 * t)
    frame = pd.DataFrame({"x": x})
    lam = 1e14
    # A fixed REML weight also exercises the component's rank and log-determinant.
    # For the integrated spline its nested first-derivative component is off.
    m = (1, 2) if fit == "fit_reml" and kind == "cr" else 2
    policy = (
        {"d1": LambdaPolicy.off(), "d2": LambdaPolicy.fixed(lam)}
        if m != 2
        else LambdaPolicy.fixed(lam)
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=lam,
        discrete=False,
        features={
            "x": Spline(
                kind=kind,
                n_knots=20,
                knot_strategy="quantile_rows",
                m=m,
                lambda_policy=policy if fit == "fit_reml" else None,
            )
        },
    )
    getattr(model, fit)(frame, y)
    with mp.workdps(100):
        basis, penalty = _natural_smoother(_knots(model), x, 100)
        gram = basis.T * basis
        inverse = (gram + lam * penalty) ** -1
        fitted = basis * (inverse * (basis.T * mp.matrix(y)))
        influence = inverse * gram
        edf = sum(influence[i, i] for i in range(influence.rows))
    # The fitted values and hat trace are both order one. Losing a range
    # direction changes them by order one even at this smoothing weight.
    np.testing.assert_allclose(model.predict(frame), [float(v) for v in fitted], atol=1e-8, rtol=0)
    assert model.result.effective_df == pytest.approx(float(edf), abs=1e-8)
    if fit == "fit_reml":
        from superglm.reml import build_penalty_caches, cached_logdet_s_plus
        from superglm.reml.penalty_algebra import compute_penalty_nullity

        active = model._reml_penalties[-1]
        rank = len(_knots(model)) - 2
        assert active.rank == rank
        weights = {pc.name: float(pc is active) for pc in model._reml_penalties}
        width = model._dm.p
        assert (
            compute_penalty_nullity(
                hessian_rank=width + 1, penalties=model._reml_penalties, lambdas=weights
            )
            == width + 1 - rank
        )
        caches = build_penalty_caches(model._dm.group_matrices, list(enumerate(model._groups)))
        # Scaling just the active component by two adds rank * log(2) to
        # its log pseudodeterminant, including through the compatibility cache.
        twice = {key: 2 * value for key, value in weights.items()}
        change = cached_logdet_s_plus(twice, caches) - cached_logdet_s_plus(weights, caches)
        assert change == pytest.approx(rank * np.log(2.0), abs=1e-10)


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


@pytest.mark.parametrize(
    ("kind", "m", "tolerance"),
    [("cr", 2, 2**4 * U * 1e8), ("cr", (1, 2), 2**4 * U * 1e8), ("cr_cardinal", 2, 4e-6)],
    ids=["cr", "cr m=(1,2) d1 off", "cr_cardinal"],
)
def test_a_fixed_lambda_fits_the_exact_smoother(kind, m, tolerance):
    """At a fixed lambda the fit is the exact natural smoothing spline's.

    The penalty's rounding no longer limits it. The range block is the Gram
    of the quadrature root, whose rows each carry one knot interval's scale:
    scaled to unit rows its condition is 8 here, against the formed
    penalty's ``kappa_+`` of 2e12, whose turned smallest eigenvalue kept a
    relative error near ``u kappa_+`` and moved the fit by 1e-6 (a penalty
    leaking onto the line, by 8e-5). What remains is the solve in the SSP
    whitened coordinates, whose map has condition 7e7 at this lambda: about
    ``u cond`` relative, 1e-8 (5e-9 to 1.2e-8 measured on three kernels), so
    the bound is ``16 u 1e8``. ``m=(1, 2)`` with ``d1`` off holds its ``d2``
    penalty in the same coordinates as ``m=2`` and fits the same smoother.

    ``cr_cardinal`` sits at 5e-7 to 1.3e-6 on three kernels however its
    penalty is held (its formed matrix turned, its root, or the correctly
    rounded exact penalty, 8e-8 to 2e-7), above its whitening's ``u cond``
    of 2e-8, so that floor is not the penalty's; 4e-6 still separates the
    1e-5 to 2e-5 a leaking null space costs it.
    """
    frame, y = _fixture(0.08)
    fixed = LambdaPolicy.fixed(FIXED_LAMBDA)
    model = _model(kind, m=m, policy=fixed if m == 2 else {**D1_OFF, "d2": fixed})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    exact = _exact_fit(_knots(model), frame["x"], y, FIXED_LAMBDA)
    assert np.max(np.abs(model.predict(frame) - exact) / np.abs(exact)) <= tolerance


@pytest.mark.parametrize("amplitude", [0.08, 0.09])
@pytest.mark.parametrize(
    ("kind", "m"),
    [("cr", 2), ("cr", (1, 2)), ("cr_cardinal", 2)],
    ids=["cr", "cr m=(1,2) d1 off", "cr_cardinal"],
)
def test_reml_finds_the_exact_optimum(kind, m, amplitude):
    """REML's lambda is the exact smoother's REML optimum to the stop rule's resolution.

    These basins are shallow (EDF near 2), so a leak onto the line of
    ``u ||Omega||`` tilts the criterion enough to move lambda by 2 to 7 percent;
    with the null rows exact the stop leaves it within about 5e-4. ``m=(1,
    2)`` with ``d1`` off estimates its ``d2`` lambda on the same criterion.
    """
    frame, y = _fixture(amplitude)
    model = _model(kind, m=m, policy=None if m == 2 else D1_OFF)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    lambdas = model._reml_result.lambdas
    estimate = lambdas["x"] if m == 2 else lambdas["x:d2"]
    assert abs(estimate / LAMBDA_STAR[amplitude] - 1.0) <= 1e-3


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("kind", ["bs", "ps"])
def test_nested_penalties_share_the_null_coordinates(kind, discrete):
    """``m=(2, 3)``: the line is ``d2``'s null coordinate, the line and the quadratic ``d3``'s.

    Each component's term penalty is exactly zero on its own leading null
    coordinates, and the line, which neither penalises, has exactly zero
    rows in every component's REML penalty; on the basis those coordinates
    are the polynomials, to the basis's rounding.
    """
    frame, y = _fixture(0.08)
    model = _model(kind, discrete, m=(2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    group = model._dm.group_matrices[0]
    zero_rows = {
        suffix: np.flatnonzero(~np.any(omega != 0.0, axis=1)).tolist()
        for suffix, omega in group.omega_components
    }
    assert zero_rows == {"d2": [0], "d3": [0, 1]}
    for penalty in model._reml_penalties:
        assert not np.any(penalty.omega_ssp[0]) and not np.any(penalty.omega_ssp[:, 0])
    grid = np.geomspace(1.0, 1e5, 50)
    curves = model._specs["x"]._basis_matrix(grid) @ group.R_inv[:, :2]
    for degree in (1, 2):
        polynomials = np.vander(grid / 1e5, degree + 1)
        curve = curves[:, degree - 1]
        residual = curve - polynomials @ np.linalg.lstsq(polynomials, curve, rcond=None)[0]
        assert np.max(np.abs(residual)) <= 2**8 * U * np.max(np.abs(curve))


@pytest.mark.pins_behaviour
def test_penalties_whose_null_spaces_do_not_nest_keep_their_own():
    """``ps`` with ``m=(2, 4)`` fits its two penalties as built.

    Pins the fallback to data coordinates when component null spaces do not
    nest. On quantile knots ``d2`` is the projected difference penalty, whose
    null space is the polynomials, while ``d4`` (above the degree) keeps the
    standard one, whose null space is the integer sequences ``j**p``: the
    line is not in it, so no coordinates hold both null spaces on leading
    axes and the term keeps the data's. The reference is the dense penalised
    least squares on the term's basis and its two formed penalties, which a
    well conditioned solve meets to rounding (2e-13 measured). Mutation:
    nesting them anyway drops part of ``d4`` and moves the fit by 1.5e-3.
    """
    x = np.geomspace(1.0, 1e3, 300)
    y = np.log(x) + np.sin(np.log(x)) + 0.1 * np.random.default_rng(5).normal(size=x.size)
    lam = 0.3
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=lam,
        features={"x": Spline(kind="ps", n_knots=12, m=(2, 4), knot_strategy="quantile_rows")},
    ).fit(pd.DataFrame({"x": x}), y)
    spec, group = model._specs["x"], model._dm.group_matrices[0]
    assert spec._difference_penalty == {2: "projected", 4: "standard"}
    P = group.projection
    columns = np.column_stack([np.ones_like(x), spec._basis_matrix(x).toarray() @ P])
    raw = spec._build_penalty_for_order(2) + spec._build_penalty_for_order(4)
    penalty = np.zeros((columns.shape[1], columns.shape[1]))
    penalty[1:, 1:] = lam * P.T @ raw @ P
    values, vectors = np.linalg.eigh(penalty)
    root = np.sqrt(np.maximum(values, 0.0))[:, None] * vectors.T
    stacked = np.vstack([columns, root])
    coef = np.linalg.lstsq(stacked, np.r_[y, np.zeros(len(root))], rcond=None)[0]
    exact = columns @ coef
    fitted = model.predict(pd.DataFrame({"x": x}))
    assert np.max(np.abs(fitted - exact) / np.abs(exact)) <= 2**16 * U


@pytest.mark.pins_behaviour
@pytest.mark.slow
@pytest.mark.parametrize("amplitude", [0.08, 0.09])
@pytest.mark.parametrize("kind", ["cr", "cr_cardinal"])
def test_the_stored_reml_optima_are_the_oracles(kind, amplitude):
    """Pins that ``LAMBDA_STAR`` matches the independent mpmath REML optimum on both kinds."""
    frame, y = _fixture(amplitude)
    model = _model(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    optimum = _exact_reml_optimum(_knots(model), frame["x"], y)
    assert optimum == pytest.approx(LAMBDA_STAR[amplitude], rel=1e-12)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("kind", ["cr", "bs", "ps"])
def test_a_null_space_its_rows_cannot_see_stays_unfitted(kind):
    """Pins the distinct-value guard: a one-value spline keeps data coordinates and stays flat.

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


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("kind", ["bs", "ps"])
def test_a_basis_function_no_row_reaches_keeps_its_exact_zero_column(kind):
    """Pins the empty-support guard: unreached basis functions keep exact zero solver columns.

    A gap in ``x`` empties six basis functions, whose solver columns stay
    exactly zero. Unpenalised, they are what the fit's rank leaves out, counted
    exactly. A rotation onto the null basis would mix them into every column at
    rounding (the polynomials reach inside and outside the gap), so the term
    keeps the data's coordinates. Mutation: declaring the null basis anyway
    leaves no zero column, and the constrained QP of ``test_constrained_qp``'s
    gap fixture then counts rank 14 of 19 for 13.
    """
    x = np.concatenate([np.linspace(0.0, 0.2, 60), np.linspace(0.8, 1.0, 60)])
    y = np.exp(0.5 + 2.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=0.0,
        features={"x": Spline(kind=kind, n_knots=16)},
    ).fit(pd.DataFrame({"x": x}), y)
    group = model._dm.group_matrices[0]
    columns = model._specs["x"]._basis_matrix(x) @ group.R_inv
    assert np.count_nonzero(~np.any(columns != 0.0, axis=0)) == 6


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("kind", ["bs", "ps"])
def test_a_null_direction_centring_cancels_keeps_the_datas_coordinates(kind):
    """Pins the centring-cancellation guard: a nearly-constant line beside ``u = x`` keeps rank 3.

    On ``x`` within 1e-8 of 0.5, beside ``u = x``, the centred line's column is
    1.8e-9 of the basis's, under ``sqrt(u)``, so the basis's rounding is
    ``6.6e-8`` of it, above the fit's ``sqrt(eps)`` rank cut: on the line's own
    coordinate the fit counted it apart from ``u`` (which it equals exactly),
    rank 4 of 3, with coefficients near 1e13 and fitted values 4e-3 off the
    line. The term declares no null basis and keeps the data's coordinates,
    where the rank is 3 and the fit is the line to 2e-9. Mutation: no
    cancellation gate in ``declared_null_coordinates``.
    """
    a = np.linspace(-1.0, 1.0, 80)
    frame = pd.DataFrame({"x": 0.5 + 1e-8 * a, "u": 0.5 + 1e-8 * a})
    y = 1.0 + 0.4 * a + 0.1 * np.cos(np.arange(80))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e-6,
        features={
            "x": Spline(kind=kind, n_knots=0, degree=2, boundary=(-1.0, 2.0)),
            "u": Numeric(),
        },
    ).fit(frame, y)
    line = np.column_stack((np.ones_like(a), a))
    exact = line @ np.linalg.lstsq(line, y, rcond=None)[0]
    assert model.result.reml_hessian_rank == model._dm.p
    assert np.max(np.abs(model.predict(frame) - exact) / np.abs(exact)) <= 1e-6


@pytest.mark.pins_behaviour
def test_one_occupied_bin_cannot_identify_a_spline_slope():
    """Pin the constant prediction across a spline represented by a single bin."""
    z = np.tile(np.linspace(0, 1, 31), 4)
    y = 2 + np.sin(8 * z) + 0.1 * np.random.default_rng(2026).normal(size=z.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=0.1,
        features={
            "x": Spline(kind="bs", n_knots=3, n_bins=1, discrete=True),
            "z": Spline(kind="ps", n_knots=4),
        },
    ).fit(pd.DataFrame({"x": z, "z": z}), y)
    grid = pd.DataFrame({"x": [0, 0.2, 0.5, 0.8, 1], "z": 0.3})
    assert np.ptp(model.predict(grid)) <= 2**10 * U * np.max(np.abs(y))


@pytest.mark.pins_behaviour
def test_discrete_spline_does_not_decompose_a_training_row_matrix(monkeypatch):
    """Pin the discrete build's work bound on the number of rows in spectral norms."""
    spectral_norm = np.linalg.norm
    row_counts = []

    def counted_norm(a, ord=None, *args, **kwargs):
        if ord == 2 and a.ndim == 2:
            row_counts.append(a.shape[0])
        return spectral_norm(a, ord, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", counted_norm)
    x = np.linspace(0, 1, 2000)
    SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1,
        discrete=True,
        n_bins=64,
        features={"x": Spline(kind="cr", n_knots=12)},
    ).fit(pd.DataFrame({"x": x}), np.cos(2 * x))
    assert max(row_counts, default=0) <= 64


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("reml", [False, True])
def test_cardinal_smooth_reproduces_lines_far_from_the_origin(reml):
    """Pin linear reproduction under a large second-derivative penalty after translating x."""
    x = 1e14 + np.linspace(0, 1, 200)
    frame = pd.DataFrame({"x": x})
    y = 1 + 2 * (x - x[0])
    penalty = {"lambda_policy": LambdaPolicy.fixed(1e3)} if reml else {}
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e4,
        discrete=False,
        features={"x": Spline(kind="cr_cardinal", n_knots=6 if reml else 8, **penalty)},
    )
    (model.fit_reml if reml else model.fit)(frame, y)
    np.testing.assert_allclose(model.predict(frame), y, rtol=0, atol=1e-11)


@pytest.mark.pins_behaviour
def test_finite_extreme_knots_can_fit_an_integrated_spline():
    """Pin finite fitted predictions on a large but representable knot interval."""
    x = np.linspace(1e308, 1.05e308, 120)
    t = (x - x[0]) / (x[-1] - x[0])
    y = 1 + t + 0.05 * np.sin(7 * t)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=2,
        discrete=False,
        features={"x": Spline(kind="bs", n_knots=5)},
    ).fit(frame, y)
    np.testing.assert_allclose(model.predict(frame), y, atol=0.05, rtol=0)


@pytest.mark.pins_behaviour
def test_elementwise_selection_keeps_the_declared_spline_coordinates():
    """Pin the fitted curve for L1 acting on the original constrained columns."""
    from superglm.features.spline import CubicRegressionSpline
    from superglm.penalties import SparseGroupLasso

    class DeclaredBasis(CubicRegressionSpline):
        """Use the ordinary declared columns with no analytic null-space rotation."""

    x = np.linspace(0, 1, 200)
    frame = pd.DataFrame({"x": x})
    y = 1 + np.sin(6 * x) + 0.1 * np.random.default_rng(8).normal(size=x.size)

    def predict(spec):
        model = SuperGLM(
            family="gaussian",
            penalty=SparseGroupLasso(lambda1=0.1, alpha=0.8),
            spline_penalty=0.1,
            discrete=False,
            features={"x": spec},
        ).fit(frame, y)
        return model.predict(frame)

    options = {"n_knots": 8, "penalty": "none"}
    np.testing.assert_allclose(
        predict(Spline(kind="cr", **options)), predict(DeclaredBasis(**options)), atol=1e-10, rtol=0
    )


@pytest.mark.pins_behaviour
def test_precision_weights_decide_whether_a_null_direction_is_visible():
    """Pin the rank and weighted least-squares line with negligible-weight endpoints."""
    a = np.linspace(-1, 1, 80)
    x = np.r_[0.5 + 1e-9 * a, 0.0, 1.0]
    frame = pd.DataFrame({"x": x, "u": x})
    y = np.r_[1 + 0.4 * a + 0.1 * np.cos(np.arange(a.size)), 0, 2]
    weights = np.r_[np.ones_like(a), 1e-23, 1e-23]
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e-6,
        discrete=False,
        features={"x": Spline(kind="bs", n_knots=0, degree=2, boundary=(-1, 2)), "u": Numeric()},
    ).fit(frame, y, sample_weight=weights)
    design = np.column_stack([np.ones_like(x), (x - 0.5) / 1e-9])
    coef = np.linalg.lstsq(np.sqrt(weights)[:, None] * design, np.sqrt(weights) * y, rcond=None)[0]
    np.testing.assert_allclose(model.predict(frame)[:80], (design @ coef)[:80], rtol=1e-6)
    assert model.result.reml_hessian_rank == 3


@pytest.mark.pins_behaviour
def test_high_order_difference_preserves_a_curve_in_its_null_space():
    """Pin the exact-fit curve whose coefficient vector is orthogonal to the difference row."""
    x = np.linspace(0, 1, 300)
    frame = pd.DataFrame({"x": x})
    spec = Spline(kind="ps", n_knots=21, m=24)
    spec.build(x)
    root = np.diff(np.eye(25), n=24, axis=0).ravel()
    root /= np.linalg.norm(root)
    coef = np.random.default_rng(2026).normal(size=root.size)
    y = spec._basis_matrix(x) @ (coef - root * (root @ coef))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=1e-6,
        discrete=False,
        features={"x": spec},
    ).fit(frame, y)
    np.testing.assert_allclose(model.predict(frame), y, atol=1e-11, rtol=0)


def _closed_form_large_lambda_reml_objective(spec, x: np.ndarray, y: np.ndarray) -> float:
    """Exact Gaussian REML objective in the ``lambda -> inf`` limit of SSP coordinates.

    With an exact zero null row in ``omega``, the leading SSP column is
    ``B u_null / sqrt(G_00 + 1e-8)`` independent of ``lambda``, where ``u_null``
    is the unit-norm raw B-spline coefficient vector of the centred linear
    function, and the range block's ``log|H_rr + lambda S_rr| - log|lambda S_rr|``
    vanishes as ``O(1 / lambda)``.
    """
    n = x.size
    basis = spec._basis_matrix(x).toarray()
    poly = _polynomial_coefficients(spec._knots, spec.degree, 2)
    s_col = basis @ poly[:, 1]
    c_line = poly[:, 1] - float(s_col.mean()) * poly[:, 0]
    u_null = c_line / np.linalg.norm(c_line)
    col = basis @ u_null
    g00 = float(col @ col) / n
    logdet_null = np.log(n) + np.log(n * g00 / (g00 + 1e-8))
    design = np.column_stack((np.ones(n), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    rss = float(np.sum((y - design @ beta) ** 2))
    scale = profile_gaussian_reml_scale(rss, float(n), 2.0, saturated_log_weight=0.0)
    return float(0.5 * logdet_null + scale.criterion)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
def test_fixed_1e14_skew_matches_linear_limit_and_exact_zero_null(discrete):
    """At ``lambda = 1e14`` on skewed quantile knots, EDF, predictions and REML objective match the line.

    On unfixed master, ``P' Omega P`` carries a negative null eigenvalue near
    ``-1e-6`` that scales by ``1e14`` to ``-1e8``, corrupting ``omega_ssp``:
    SkylakeX publishes term EDF 2.0000, REML objective -117.36 (20.82 off
    -96.54) and 3.0% prediction error vs OLS; Haswell publishes REML objective
    -99.78 (3.25 off -96.54) and 9.1e-5 prediction error. Both kernels fail on
    unfixed master.
    """
    x = np.geomspace(1.0, 1e5, 200)
    y = 1.0 + x / 1e5 + np.random.default_rng(0).normal(0.0, 0.15, size=x.size)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=discrete,
        features={
            "x": Spline(
                kind="cr",
                n_knots=10,
                knot_strategy="quantile",
                penalty="ssp",
                lambda_policy=LambdaPolicy.fixed(1e14),
            )
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)

    (penalty,) = model._reml_penalties
    assert np.flatnonzero(~np.any(penalty.omega_ssp != 0.0, axis=1)).tolist() == [0]
    assert model.term_inference("x", with_se=False).edf == pytest.approx(1.0, abs=1e-6)

    design = np.column_stack((np.ones_like(x), x))
    ols = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    pred = np.asarray(model.predict(frame))
    assert np.max(np.abs(pred - ols) / np.abs(ols)) <= 2**16 * U

    exact_obj = _closed_form_large_lambda_reml_objective(model._specs["x"], x, y)
    assert model._reml_result.objective == pytest.approx(exact_obj, abs=1e-5)


def test_fixed_1e14_uniform_matches_exact_reml_objective():
    """At ``lambda = 1e14`` on uniform knots, the REML objective matches the exact linear limit.

    On unfixed master, null-space rounding in ``M = G + lambda Omega + 1e-8 I``
    shifts the null whitening factor ``R_00`` differently on each kernel:
    SkylakeX publishes 347.769 (0.022 above 347.748) and Haswell publishes
    347.484 (0.264 below 347.748), both failing against the exact limit.
    """
    x = np.linspace(0.0, 1.0, 500)
    y = 1.0 + 0.5 * x + np.random.default_rng(42).normal(0.0, 0.5, size=x.size)
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        discrete=False,
        features={
            "x": Spline(
                kind="cr",
                n_knots=10,
                penalty="ssp",
                lambda_policy=LambdaPolicy.fixed(1e14),
            )
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)

    (penalty,) = model._reml_penalties
    assert np.flatnonzero(~np.any(penalty.omega_ssp != 0.0, axis=1)).tolist() == [0]
    assert model.term_inference("x", with_se=False).edf == pytest.approx(1.0, abs=1e-6)

    exact_obj = _closed_form_large_lambda_reml_objective(model._specs["x"], x, y)
    assert model._reml_result.objective == pytest.approx(exact_obj, abs=1e-6)


@pytest.mark.parametrize("penalty", ["none", "ssp"])
def test_stiff_population_smooth_keeps_its_line_beside_a_thin_level(penalty):
    """Even with a one-point level, a dominant penalty must annihilate the population line.

    Paired observations at each point average to the line, which has zero
    penalty and minimizes the residual sum of squares on every observed row.
    """
    x = np.geomspace(1.0, 1e5, 100)
    frame = pd.DataFrame({"x": np.tile(x, 4), "g": np.repeat(list("abcd"), x.size)})
    frame = frame.loc[frame.index.repeat(2)].reset_index(drop=True)
    weights = np.asarray((frame.g != "a") | (frame.x == x[0]), dtype=float)
    line = 1.0 + 0.4 * frame.x.to_numpy() / x[-1]
    y = line + np.tile([-0.02, 0.02], len(frame) // 2)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        features={
            "x": Spline(
                kind="cr",
                n_knots=10,
                knot_strategy="quantile",
                penalty=penalty,
                lambda_policy=LambdaPolicy.fixed(1e14),
            )
        },
        interactions=[
            FactorSmooth("x", group="g", basis="sz", k=5, lambda_policy=LambdaPolicy.fixed(0.7))
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=weights)
    main = next(pc for pc in model._reml_penalties if pc.group_name == "x")
    assert np.flatnonzero(~np.any(main.omega_ssp, axis=1)).tolist() == [0]
    live = weights > 0
    np.testing.assert_allclose(model.predict(frame)[live], line[live], atol=2e-12, rtol=0)


@pytest.mark.pins_behaviour
def test_spline_subclass_with_custom_penalty_keeps_its_own_penalty():
    """Pins the built-in type guard: a spline subclass keeps its custom penalty, not the parent's root.

    A subclass of ``PSpline`` overriding ``_build_penalty_for_order`` with a
    permuted difference penalty has the same rank as ``PSpline`` but a different
    null space. Without the ``type(spec) not in _BUILT_IN_SPLINES`` guard,
    ``declared_null_coordinates`` would inherit the parent's null basis and root
    and silently replace the subclass's penalty.
    """

    class _PermutedPSpline(PSpline):
        def _build_penalty_for_order(self, order: int) -> np.ndarray:
            base = super()._build_penalty_for_order(order)
            perm = np.arange(base.shape[0])
            perm[[0, 2, 4, 6]] = perm[[6, 0, 2, 4]]
            return base[np.ix_(perm, perm)]

    x = np.linspace(0.0, 1.0, 160)
    y = 1.0 + np.sin(4.0 * np.pi * x) + 0.1 * np.random.default_rng(19).normal(size=x.size)
    lam = 5.0
    spec = _PermutedPSpline(n_knots=6, penalty="ssp")
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0,
        spline_penalty=lam,
        features={"x": spec},
    ).fit(pd.DataFrame({"x": x}), y)

    fitted_spec, group = model._specs["x"], model._dm.group_matrices[0]
    assert getattr(group, "R_inv_local", None) is None
    P = group.projection
    columns = np.column_stack([np.ones_like(x), fitted_spec._basis_matrix(x).toarray() @ P])
    raw = fitted_spec._build_penalty_for_order(2)
    penalty = np.zeros((columns.shape[1], columns.shape[1]))
    penalty[1:, 1:] = lam * P.T @ raw @ P
    values, vectors = np.linalg.eigh(penalty)
    root = np.sqrt(np.maximum(values, 0.0))[:, None] * vectors.T
    coef = np.linalg.lstsq(np.vstack([columns, root]), np.r_[y, np.zeros(len(root))], rcond=None)[0]
    exact = columns @ coef
    fitted = np.asarray(model.predict(pd.DataFrame({"x": x})))
    assert np.max(np.abs(fitted - exact) / np.abs(exact)) <= 2**16 * U
