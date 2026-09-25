"""Polynomial ranges on B-splines: geometry, pinning rows and their null space.

Bounds lean on two facts. The columns of an orthonormal Z have entries of
magnitude <= 1 and B-splines are a partition of unity, so every null-space
member satisfies ``|f| <= 1`` and an absolute tolerance in units of eps is
already relative. Householder QR is backward stable with an error growing
with the dimension, and a B-spline value sums ``DEGREE + 1`` terms.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline

import superglm
from superglm import Constraint, LambdaPolicy, Numeric, Spline, SuperGLM
from superglm.export._ppform import extract_ppform
from superglm.features._spline_penalties import build_integrated_derivative_penalty
from superglm.features._spline_ranges import (
    PolynomialRange,
    constraint_null_space,
    derivative_design,
    merged_interior_knots,
    pinning_rows,
    validate_ranges,
)

LO, HI, DEGREE = 0.0, 10.0, 3
EPS = np.finfo(np.float64).eps


def _clamped(interior, lo=LO, hi=HI):
    return np.concatenate([[lo] * (DEGREE + 1), interior, [hi] * (DEGREE + 1)])


def _null_space_members(ranges, base, lo=LO, hi=HI):
    """Every basis member of the pinned space, as one vector-valued spline."""
    ranges = validate_ranges(ranges, DEGREE, lo, hi)
    knots = _clamped(merged_interior_knots(base, ranges, DEGREE, lo, hi), lo, hi)
    Z = constraint_null_space(pinning_rows(knots, DEGREE, ranges))
    return BSpline(knots, Z, DEGREE), knots


def _polynomial_residual(spline, lo, hi, degree):
    """Largest distance of each member from its best degree-``degree`` fit on [lo, hi]."""
    grid = np.linspace(lo, hi, 41)
    values = spline(grid)
    # Legendre basis on the range mapped to [-1, 1]: a well-conditioned fit.
    vander = np.polynomial.legendre.legvander((2.0 * grid - lo - hi) / (hi - lo), degree)
    coefficients = np.linalg.lstsq(vander, values, rcond=None)[0]
    return float(np.max(np.abs(values - vander @ coefficients)))


def _left_limit(spline, x, nu):
    """nu-th derivative from the left: the mirrored spline's right limit."""
    mirrored = BSpline(-spline.t[::-1], spline.c[::-1], spline.k)
    return (-1) ** nu * mirrored(-x, nu=nu)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("join", ["kink", "smooth"])
def test_every_member_is_a_polynomial_of_the_range_degree(degree, join):
    spline, knots = _null_space_members(
        [PolynomialRange(3.0, 6.0, degree, join)], np.linspace(1, 9, 9)
    )
    n_basis = len(knots) - DEGREE - 1
    # Pinning one piece of degree DEGREE to degree d removes DEGREE - d
    # directions and nothing else.
    assert spline.c.shape[1] == n_basis - (DEGREE - degree)
    tolerance = n_basis * (DEGREE + 1) * EPS
    assert _polynomial_residual(spline, 3.0, 6.0, degree) <= tolerance


def test_pinning_is_independent_of_the_feature_units():
    """Rows of different derivative orders must not be refused for their scale.

    On a 1e9-wide axis a Flat range's f' rows and a Quadratic range's f''' rows
    differ in size by (knot spacing)^2 = 1e16, below the rank threshold unless
    the rows are scaled before the rank decision.
    """
    scale = 1e9
    ranges = [PolynomialRange(2 * scale, 3 * scale, 0), PolynomialRange(6 * scale, 8 * scale, 2)]
    spline, knots = _null_space_members(ranges, np.linspace(1, 9, 9) * scale, lo=0.0, hi=10 * scale)
    n_basis = len(knots) - DEGREE - 1
    assert spline.c.shape[1] == n_basis - DEGREE - (DEGREE - 2)
    tolerance = n_basis * (DEGREE + 1) * EPS
    assert _polynomial_residual(spline, 2 * scale, 3 * scale, 0) <= tolerance
    assert _polynomial_residual(spline, 6 * scale, 8 * scale, 2) <= tolerance


def test_knots_inside_a_range_are_dropped_and_kink_edges_repeat():
    ranges = validate_ranges([PolynomialRange(3.0, 6.0, 1)], DEGREE, LO, HI)
    interior = merged_interior_knots(np.array([2.0, 3.0, 4.0, 5.0, 7.0]), ranges, DEGREE, LO, HI)
    np.testing.assert_array_equal(interior, [2.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 7.0])


def test_boundary_edges_are_not_inserted():
    ranges = validate_ranges([PolynomialRange(LO, 4.0, 0)], DEGREE, LO, HI)
    interior = merged_interior_knots(np.array([2.0, 5.0, 8.0]), ranges, DEGREE, LO, HI)
    np.testing.assert_array_equal(interior, [4.0, 4.0, 4.0, 5.0, 8.0])


def test_a_shared_edge_is_inserted_once_at_its_multiplicity():
    ranges = validate_ranges(
        [PolynomialRange(2.0, 5.0, 1), PolynomialRange(5.0, 7.0, 0)], DEGREE, LO, HI
    )
    interior = merged_interior_knots(np.array([1.0, 4.0, 6.0, 8.0]), ranges, DEGREE, LO, HI)
    np.testing.assert_array_equal(interior, [1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 8.0])


def test_kink_join_is_continuous_and_leaves_the_slope_free():
    spline, _ = _null_space_members([PolynomialRange(3.0, 6.0, 1)], np.linspace(1, 9, 9))
    edges = np.array([3.0, 6.0])
    value_jump = np.abs(_left_limit(spline, edges, 0) - spline(edges))
    assert np.max(value_jump) <= 2 * (DEGREE + 1) * EPS
    # At each edge some member bends by far more than round-off: the kink is
    # available to the fit, not forced away.
    slope_jump = np.abs(_left_limit(spline, edges, 1) - spline(edges, nu=1))
    slope_scale = np.max(np.abs(spline(np.linspace(LO, HI, 201), nu=1)))
    assert np.all(np.max(slope_jump, axis=1) > np.sqrt(EPS) * slope_scale)


@pytest.mark.parametrize("nu", range(DEGREE))
def test_smooth_join_keeps_the_splines_own_continuity(nu):
    spline, _ = _null_space_members([PolynomialRange(3.0, 6.0, 1, "smooth")], np.linspace(1, 9, 9))
    edges = np.array([3.0, 6.0])
    jump = np.abs(_left_limit(spline, edges, nu) - spline(edges, nu=nu))
    scale = np.max(np.abs(spline(np.linspace(LO, HI, 201), nu=nu)))
    assert np.max(jump) <= 2 * (DEGREE + 1) * EPS * scale


@pytest.mark.parametrize(
    ("ranges", "degree", "message"),
    [
        ([(4.0, 2.0, 1)], DEGREE, "lo must be below hi"),
        ([(-1.0, 2.0, 1)], DEGREE, "inside the fitted range"),
        ([(8.0, 11.0, 1)], DEGREE, "inside the fitted range"),
        ([(2.0, 5.0, 1), (4.0, 7.0, 0)], DEGREE, "overlap"),
        ([(2.0, 5.0, 1, "smooth"), (5.0, 7.0, 0)], DEGREE, "meet at a kink"),
        ([(2.0, 5.0, 3)], 2, "exceeds the spline degree 2"),
    ],
)
def test_invalid_ranges_are_refused_by_name(ranges, degree, message):
    ranges = [PolynomialRange(*fields) for fields in ranges]
    with pytest.raises(ValueError, match=message):
        validate_ranges(ranges, degree, LO, HI)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ((2.0, 5.0, 4), "degree must be 0-3"),
        ((2.0, 5.0, -1), "degree must be 0-3"),
        ((2.0, 5.0, True), "degree must be an integer"),
        ((2.0, 5.0, 1.0), "degree must be an integer"),
        ((2.0, 5.0, 1, "round"), "join must be one of"),
    ],
)
def test_malformed_ranges_are_refused_at_construction(fields, message):
    with pytest.raises(ValueError, match=message):
        PolynomialRange(*fields)


def test_dependent_rows_are_refused_not_silently_truncated():
    row = np.arange(1.0, 6.0)
    with pytest.raises(ValueError, match="dependent"):
        constraint_null_space(np.vstack([row, 2.0 * row]))


def test_close_smooth_ranges_are_refused_as_dependent():
    """Two Flat ranges one knot interval apart with smooth joins share
    coefficients: their six f' rows act on a six-coefficient stretch whose
    constants they all annihilate, so they cannot be independent."""
    ranges = validate_ranges(
        [PolynomialRange(2.0, 3.0, 0, "smooth"), PolynomialRange(4.0, 5.0, 0, "smooth")],
        DEGREE,
        LO,
        HI,
    )
    knots = _clamped(merged_interior_knots(np.linspace(1, 9, 9), ranges, DEGREE, LO, HI))
    with pytest.raises(ValueError, match="meet at a kink"):
        constraint_null_space(pinning_rows(knots, DEGREE, ranges))


# ── Spline(polynomial_ranges=...) on fitted models ────────────────────────


def _book(n=6_000, seed=5):
    """A Poisson frequency book on an 18-80 age axis with gamma exposures."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18.0, 80.0, n)
    exposure = rng.gamma(4.0, 0.25, n) + 0.05
    eta = 0.1 + 0.3 * np.sin((age - 18.0) / 9.0)
    y = rng.poisson(np.exp(eta) * exposure) / exposure
    return pd.DataFrame({"age": age}), y, exposure


def _fit(kind, ranges, *, discrete=False, book=None):
    X, y, w = _book() if book is None else book
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Spline(kind=kind, k=12, polynomial_ranges=ranges)},
        discrete=discrete,
        n_bins=512,
    )
    model.fit_reml(X, y, sample_weight=w)
    return model


def _term(model):
    """The fitted age term on the link scale, read through the exact predictor."""

    def curve(grid):
        eta = model._predict_eta_exact(pd.DataFrame({"age": grid}))
        return np.asarray(eta, dtype=np.float64) - model.result.intercept

    return curve


def _pinning_tolerance(model):
    """Round-off bound on a fitted term's distance from its pinned polynomial.

    The term is B(x) @ c with c = P w and P the orthonormal constraint and
    centering projection, so it combines unit-norm null-space members (each
    within n_basis * (DEGREE + 1) eps of its polynomial, as above) with weights
    of total size ||w||_1 <= sqrt(n_cols) ||c||. Reading the term back off the
    predictor adds the round-off of adding and removing the intercept.
    """
    spec = model._specs["age"]
    beta = model.result.beta
    coefficients = spec._R_inv @ beta
    members = np.sqrt(beta.size) * spec._n_basis * (DEGREE + 1) * EPS
    return members * np.linalg.norm(coefficients) + 4 * EPS * abs(model.result.intercept)


def test_polynomial_range_is_exported_from_superglm():
    assert superglm.PolynomialRange is PolynomialRange


@pytest.mark.parametrize("kind", ["bs", "cr"])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_fitted_curve_is_the_pinned_polynomial_on_the_range(kind, discrete, degree):
    # Weighted, and binned when discrete: the pinning lives in the basis, so
    # neither the weights nor the bins can bend the piece.
    model = _fit(kind, [PolynomialRange(30.0, 45.0, degree)], discrete=discrete)
    residual = _polynomial_residual(_term(model), 30.0, 45.0, degree)
    assert residual <= _pinning_tolerance(model)


def test_curve_outside_the_range_stays_smooth_and_penalised():
    model = _fit("bs", [PolynomialRange(30.0, 45.0, 1)])
    assert 0.0 < model._reml_result.lambdas["age"] < np.inf
    # Outside the range the term is the smooth, not a line: its bend is far
    # above round-off.
    assert _polynomial_residual(_term(model), 50.0, 75.0, 1) > 1e6 * _pinning_tolerance(model)


def test_restricted_penalty_is_the_curvature_integral_over_free_intervals_only():
    """beta' S beta is the integral of f''^2 over the knot intervals outside the range."""
    ranges = validate_ranges([PolynomialRange(3.0, 6.0, 2)], DEGREE, LO, HI)
    knots = _clamped(merged_interior_knots(np.linspace(1, 9, 9), ranges, DEGREE, LO, HI))
    Z = constraint_null_space(pinning_rows(knots, DEGREE, ranges))
    omega = build_integrated_derivative_penalty(knots, DEGREE, 2, excluded=[(3.0, 6.0)])
    beta = Z @ np.random.default_rng(1).normal(size=Z.shape[1])
    breaks = np.unique(knots)
    a, b = breaks[:-1], breaks[1:]
    free = ~((a >= 3.0) & (b <= 6.0))
    a, b = a[free], b[free]
    # f'' of a cubic is linear on each interval, so f''^2 is quadratic and a
    # two-point Gauss rule -- not the builder's three -- integrates it exactly.
    nodes, weights = np.polynomial.legendre.leggauss(2)
    points = 0.5 * (b - a)[:, None] * nodes + 0.5 * (a + b)[:, None]
    curvature = (derivative_design(knots, DEGREE, points.ravel(), 2) @ beta).reshape(points.shape)
    integral = float(np.sum(0.5 * (b - a)[:, None] * weights * curvature**2))
    penalty = float(beta @ omega @ beta)
    # Both sides sum O(n_basis^2) products bounded by |beta|' |omega| |beta|.
    n_basis = len(beta)
    scale = float(np.abs(beta) @ np.abs(omega) @ np.abs(beta))
    assert abs(penalty - integral) <= n_basis**2 * EPS * scale


@pytest.mark.parametrize("kind", ["bs", "cr"])
def test_pinned_quadratic_is_not_shrunk_by_the_penalty(kind):
    """A quadratic bump confined to the range costs nothing under the penalty.

    The bump (x - 30)(45 - x) on [30, 45], zero elsewhere, lies in the ranged
    basis (the kink edges leave it C0). Its curvature integral, 4 * 15 = 60, is
    all inside the range, so the term's penalty must charge none of it and no
    smoothing parameter can shrink it.
    """
    x = _book()[0]["age"].to_numpy()
    spec = Spline(kind=kind, k=12, polynomial_ranges=[PolynomialRange(30.0, 45.0, 2)])
    spec._place_knots(x)
    grid = np.linspace(spec._lo, spec._hi, 2001)
    bump = np.where((grid >= 30.0) & (grid <= 45.0), (grid - 30.0) * (45.0 - grid), 0.0)
    design = BSpline.design_matrix(grid, spec._knots, spec.degree).toarray()
    coefficients = np.linalg.lstsq(design, bump, rcond=None)[0]
    in_span = spec._n_basis * (DEGREE + 1) * EPS * np.linalg.cond(design) * np.max(bump)
    assert np.max(np.abs(design @ coefficients - bump)) <= in_span
    unrestricted = build_integrated_derivative_penalty(spec._knots, spec.degree, 2)
    scale = float(np.abs(coefficients) @ np.abs(unrestricted) @ np.abs(coefficients))
    tolerance = spec._n_basis**2 * EPS * scale
    assert abs(coefficients @ unrestricted @ coefficients - 60.0) <= tolerance
    assert abs(coefficients @ spec._build_penalty() @ coefficients) <= tolerance


@pytest.mark.parametrize("kind", ["bs", "cr"])
def test_whole_axis_range_equals_an_unpenalised_polynomial_fit(kind):
    X, y, w = _book()
    lo, hi = float(X["age"].min()), float(X["age"].max())
    ranged = _fit(kind, [PolynomialRange(lo, hi, 2)], book=(X, y, w))
    Xq = pd.DataFrame({"age": X["age"], "age2": X["age"] ** 2})
    poly = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"age": Numeric(), "age2": Numeric()},
    )
    poly.fit(Xq, y, sample_weight=w)
    # Agreement is bounded by the two fits' IRLS convergence tolerance, not round-off.
    np.testing.assert_allclose(ranged.predict(X), poly.predict(Xq), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("lambda_policy", [None, LambdaPolicy.fixed(1.0)])
@pytest.mark.parametrize("discrete", [False, True])
def test_a_fully_pinned_term_carries_no_smoothing_parameter(discrete, lambda_policy):
    """Ranges over every knot interval leave an identically zero penalty.

    Its REML gradient is identically zero, so a smoothing parameter for it
    would be reported "estimated" at whatever value it started from. The term
    is an unpenalised polynomial instead, and only the other smooth has one --
    even when the term states a lambda policy, which has nothing to govern.
    """
    X, y, w = _book()
    X["other"] = np.random.default_rng(11).uniform(0.0, 1.0, len(X))
    lo, hi = float(X["age"].min()), float(X["age"].max())
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "age": Spline(
                kind="bs",
                k=12,
                lambda_policy=lambda_policy,
                polynomial_ranges=[PolynomialRange(lo, hi, 2)],
            ),
            "other": Spline(kind="bs", k=8),
        },
        discrete=discrete,
    )
    model.fit_reml(X, y, sample_weight=w)
    assert set(model._reml_result.lambdas) == {"other"}


def test_polynomial_ranges_are_reported_sorted_with_float_edges():
    x = _book()[0]["age"].to_numpy()
    spec = Spline(
        kind="cr", k=12, polynomial_ranges=[PolynomialRange(60, 70, 0), PolynomialRange(30, 45, 1)]
    )
    spec._place_knots(x)
    reported = [(r.lo, r.hi, r.degree, r.label) for r in spec.polynomial_ranges]
    assert reported == [(30.0, 45.0, 1, "Line"), (60.0, 70.0, 0, "Flat")]
    edges = [r.lo for r in spec.polynomial_ranges] + [r.hi for r in spec.polynomial_ranges]
    assert {type(edge) for edge in edges} == {float}


def test_ranges_at_both_ends_of_a_cr_fit_without_dependent_rows():
    """A Flat or Line range at an end already makes f'' vanish there, so that
    end's natural row would be a dependent duplicate of the pinning rows."""
    X, y, w = _book()
    lo, hi = float(X["age"].min()), float(X["age"].max())
    model = _fit("cr", [PolynomialRange(lo, 25.0, 1), PolynomialRange(70.0, hi, 0)], book=(X, y, w))
    # A boundary edge is already a knot of the clamped vector: never inserted again.
    assert not np.isin([lo, hi], model._specs["age"].fitted_knots).any()
    tolerance = _pinning_tolerance(model)
    assert _polynomial_residual(_term(model), lo, 25.0, 1) <= tolerance
    assert _polynomial_residual(_term(model), 70.0, hi, 0) <= tolerance


def test_a_quadratic_range_at_a_cr_end_is_a_full_quadratic():
    """The range sets the curve's shape up to the end, so that end's natural row
    f''(end) = 0 is dropped: kept, it would flatten the quadratic to a line."""
    x = _book()[0]["age"].to_numpy()
    hi = float(x.max())
    spec = Spline(kind="cr", k=12, polynomial_ranges=[PolynomialRange(70.0, hi, 2)])
    spec._place_knots(x)
    Z = spec._apply_constraints(None, spec._build_penalty())[3]
    grid = np.linspace(70.0, hi, 41)
    members = BSpline(spec._knots, Z, spec.degree)(grid)
    vander = np.polynomial.legendre.legvander((2.0 * grid - 70.0 - hi) / (hi - 70.0), 2)
    quadratic_part = np.linalg.lstsq(vander, members, rcond=None)[0][2]
    assert np.max(np.abs(quadratic_part)) > np.sqrt(EPS) * np.max(np.abs(members))


def test_fitted_knots_report_edges_and_base_knots_reproduce_placement():
    x = _book()[0]["age"].to_numpy()
    ranges = [PolynomialRange(30.0, 45.0, 1)]
    spec = Spline(kind="bs", k=12, polynomial_ranges=ranges)
    spec._place_knots(x)
    knots, base = spec.fitted_knots, spec.fitted_base_knots
    assert np.count_nonzero(knots == 30.0) == 3
    assert np.count_nonzero(knots == 45.0) == 3
    assert not np.any((knots > 30.0) & (knots < 45.0))
    # The base knots are the placement before the ranges: the ones inside
    # [30, 45] were dropped from the fitted vector, not never placed.
    assert np.any((base > 30.0) & (base < 45.0))
    assert not np.isin([30.0, 45.0], base).any()
    again = Spline(kind="bs", knots=base, boundary=spec.fitted_boundary, polynomial_ranges=ranges)
    again._place_knots(x)
    np.testing.assert_array_equal(again.fitted_knots, knots)
    np.testing.assert_array_equal(again.fitted_base_knots, base)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "ps"}, "kind='bs' or kind='cr'"),
        ({"kind": "ns"}, "kind='bs' or kind='cr'"),
        ({"kind": "cr_cardinal"}, "kind='bs' or kind='cr'"),
        ({"kind": "bs", "select": True}, "select=True"),
        ({"kind": "cr", "constraint": Constraint.fit.increasing}, "shape constraint"),
        ({"kind": "bs", "constraint": Constraint.postfit.convex}, "shape constraint"),
    ],
)
def test_unsupported_combinations_are_refused_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Spline(k=12, polynomial_ranges=[PolynomialRange(30.0, 45.0, 1)], **kwargs)


@pytest.mark.parametrize("discrete", [False, True])
def test_too_few_distinct_values_in_a_range_are_refused_by_name(discrete):
    X, y, w = _book()
    X.loc[(X["age"] > 50.0) & (X["age"] < 52.0), "age"] = 51.0
    with pytest.raises(ValueError, match="needs at least 2 distinct values .* it has 1"):
        _fit("bs", [PolynomialRange(50.5, 51.5, 1)], discrete=discrete, book=(X, y, w))
    # One value is enough for a Flat range: the bound is degree + 1, no more.
    _fit("bs", [PolynomialRange(50.5, 51.5, 0)], discrete=discrete, book=(X, y, w))


def test_ppform_export_reproduces_a_kinked_ranged_spline():
    model = _fit("bs", [PolynomialRange(30.0, 45.0, 1)])
    block = extract_ppform(model, "age")
    lo, hi = model._specs["age"].fitted_boundary
    grid = np.linspace(lo, hi, 301)
    # Ten times the export's own exactness certificate (1e-11 on its read grid).
    np.testing.assert_allclose(block.evaluate(grid), _term(model)(grid), rtol=0.0, atol=1e-10)
    assert np.isin([30.0, 45.0], block.breaks).all()
