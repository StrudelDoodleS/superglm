"""Polynomial ranges on B-splines: geometry, pinning rows and their null space.

Bounds lean on two facts. The columns of an orthonormal Z have entries of
magnitude <= 1 and B-splines are a partition of unity, so every null-space
member satisfies ``|f| <= 1`` and an absolute tolerance in units of eps is
already relative. Householder QR is backward stable with an error growing
with the dimension, and a B-spline value sums ``DEGREE + 1`` terms.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from superglm.features._spline_ranges import (
    PolynomialRange,
    constraint_null_space,
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
