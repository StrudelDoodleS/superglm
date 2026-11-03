"""Rows for shape rules confined to a stretch of a B-spline curve."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize

from superglm import Constraint
from superglm.features._shape_rule_rows import (
    derivative_value_row,
    refined_derivative_rows,
    shape_pieces,
    shape_rows,
)

C = Constraint.fit
U = 2.0**-53
LO, HI, K, DEGREE = 17.0, 80.0, 14, 3


def _open_knots() -> np.ndarray:
    """The padded open cubic knot vector a ps/bs term builds on [17, 80], 14 coefficients."""
    pad = 0.001 * (HI - LO)
    inner = np.linspace(LO - pad, HI + pad, K - DEGREE + 1)
    dx = inner[1] - inner[0]
    steps = dx * np.arange(1, DEGREE + 1)
    return np.concatenate([inner[0] - steps[::-1], inner, inner[-1] + steps])


def _clamped_knots() -> np.ndarray:
    """The clamped cubic knot vector a natural (cr) term builds on [17, 80]."""
    interior = np.linspace(LO, HI, K - DEGREE + 1)[1:-1]
    return np.concatenate([np.full(DEGREE + 1, LO), interior, np.full(DEGREE + 1, HI)])


def _refined_knots(knots: np.ndarray, order: int, a: float, b: float) -> np.ndarray:
    """The derivative spline's knots on [a, b], clamped at a and b."""
    t_r = knots[order : len(knots) - order]
    p = DEGREE - order
    inner = t_r[(t_r > a) & (t_r < b)]
    return np.concatenate([np.full(p + 1, a), inner, np.full(p + 1, b)])


def _coefficients_of(knots: np.ndarray, f) -> np.ndarray:
    """Coefficients of a polynomial of degree <= 3, which the spline space holds exactly."""
    x = np.linspace(knots[DEGREE], knots[-DEGREE - 1], 400)
    B = BSpline.design_matrix(x, knots, DEGREE).toarray()
    return np.linalg.lstsq(B, f(x), rcond=None)[0]


def _slack(A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-row rounding allowance for ``A @ c``."""
    return 1e3 * U * (np.abs(A) @ np.abs(c))


@pytest.mark.parametrize("knots", [_open_knots(), _clamped_knots()], ids=["open", "clamped"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("stretch", [(LO, HI), (25.0, 40.0), (17.0, 31.4)])
def test_refined_rows_reproduce_derivative(knots, order, stretch):
    a, b = stretch
    c = np.random.default_rng(0).normal(size=K)
    R = refined_derivative_rows(knots, DEGREE, order, a, b)
    x = np.linspace(a, b, 1000)
    exact = BSpline(knots, c, DEGREE)(x, nu=order)
    refined = BSpline(_refined_knots(knots, order, a, b), R @ c, DEGREE - order)(x)
    np.testing.assert_allclose(refined, exact, rtol=0, atol=1e3 * U * np.max(np.abs(exact)))


def test_rows_confine_to_stretch():
    knots = _open_knots()
    R = refined_derivative_rows(knots, DEGREE, 1, 25.0, 40.0)
    outside = (knots[DEGREE + 1 : DEGREE + 1 + K] <= 25.0) | (knots[:K] >= 40.0)
    assert outside.any() and not outside.all()
    assert np.max(np.abs(R[:, outside])) <= 1e3 * U * np.max(np.abs(R))


def test_sign_rows_are_sufficient():
    knots = _open_knots()
    R = refined_derivative_rows(knots, DEGREE, 1, 25.0, 40.0)
    x = np.linspace(25.0, 40.0, 2000)
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = rng.normal(size=K)
        projected = minimize(
            lambda z, c=c: 0.5 * np.sum((z - c) ** 2),
            c,
            jac=lambda z, c=c: z - c,
            hess=lambda z: np.eye(K),
            method="trust-constr",
            constraints=[LinearConstraint(R, 0.0, np.inf)],
        ).x
        slope = BSpline(knots, projected, DEGREE)(x, nu=1)
        # Partition of unity: f' is a convex combination of the refined
        # coefficients, so it is no lower than their least value.
        floor = min(0.0, float(np.min(R @ projected)))
        assert np.min(slope) >= floor - 1e3 * U * np.max(np.abs(slope))
        assert floor >= -1e-6 * np.max(np.abs(slope))


def test_u_turn_slope_zero_only_at_turn():
    knots = _open_knots()
    c = _coefficients_of(knots, lambda x: (x - 25.0) ** 2)
    pieces = [("decreasing", LO, 25.0), ("increasing", 25.0, HI)]
    rows, flat = shape_rows(knots, DEGREE, pieces, natural_ends=False)
    assert flat.shape == (0, K)
    assert np.all(rows.A @ c >= -_slack(rows.A, c))
    slope = BSpline(knots, c, DEGREE)(np.array([24.0, 26.0]), nu=1)
    assert slope[0] < 0.0 < slope[1]


@pytest.mark.parametrize(
    ("direction", "curvature", "feasible", "infeasible"),
    [
        ("increasing", "concave", lambda x: -((x - 90.0) ** 2), lambda x: -((x - 50.0) ** 2)),
        ("increasing", "convex", lambda x: (x - 10.0) ** 2, lambda x: (x - 50.0) ** 2),
        ("decreasing", "convex", lambda x: (x - 90.0) ** 2, lambda x: (x - 50.0) ** 2),
        ("decreasing", "concave", lambda x: -((x - 10.0) ** 2), lambda x: -((x - 50.0) ** 2)),
    ],
)
def test_end_slope_pairing(direction, curvature, feasible, infeasible):
    knots = _open_knots()
    rules = getattr(C, direction) & getattr(C, curvature)
    rows, _ = shape_rows(knots, DEGREE, shape_pieces(rules, LO, HI), natural_ends=False)
    assert rows.A.shape == (K - 2 + 1, K)
    # Mutation check: the direction rows stacked in full give 2k - 3 rows.
    stacked = sum(
        shape_rows(knots, DEGREE, [(kind, LO, HI)], natural_ends=False)[0].A.shape[0]
        for kind in (direction, curvature)
    )
    assert stacked == 2 * K - 3
    good = _coefficients_of(knots, feasible)
    assert np.all(rows.A @ good >= -_slack(rows.A, good))
    # Both quadratics have the right curvature; only the end-slope row can
    # refuse the wrong one, so this pins which end the row sits at.
    bad = _coefficients_of(knots, infeasible)
    assert np.min(rows.A @ bad + _slack(rows.A, bad)) < 0.0


def test_end_slope_row_is_the_slope_at_the_end():
    knots = _open_knots()
    rows, _ = shape_rows(
        knots, DEGREE, [("increasing", 25.0, 40.0), ("concave", 25.0, 40.0)], natural_ends=False
    )
    slope_row = derivative_value_row(knots, DEGREE, 40.0, 1)
    matches = [
        np.allclose(r, slope_row / np.max(np.abs(slope_row)), rtol=0, atol=1e3 * U) for r in rows.A
    ]
    assert sum(matches) == 1


def test_natural_ends_drop_the_pinned_curvature_rows():
    knots = _clamped_knots()
    natural = derivative_value_row(knots, DEGREE, LO, 2), derivative_value_row(knots, DEGREE, HI, 2)
    Q, _ = np.linalg.qr(np.vstack(natural).T, mode="complete")
    Z = Q[:, 2:]  # coefficients that keep f''(17) = f''(80) = 0
    for rule, dropped in [(C.convex, 2), (C.convex.below(40), 1), (C.concave.within(25, 40), 0)]:
        pieces = shape_pieces(rule, LO, HI)
        plain, _ = shape_rows(knots, DEGREE, pieces, natural_ends=False)
        rows, _ = shape_rows(knots, DEGREE, pieces, natural_ends=True)
        assert rows.A.shape[0] == plain.A.shape[0] - dropped
        # The rows kept survive the natural projection; the dropped ones vanish there.
        kept = np.linalg.norm(rows.A @ Z, axis=1)
        assert np.min(kept) > 1e-3
        gone = [r for r in plain.A if not any(np.array_equal(r, k) for k in rows.A)]
        assert len(gone) == dropped
        for r in gone:
            assert np.linalg.norm(r @ Z) <= 1e3 * U * np.linalg.norm(r)


@pytest.mark.parametrize(
    "rules",
    [
        C.increasing & C.convex & C.flat.within(25, 40),
        C.decreasing.below(25) & C.concave & C.flat.within(25, 40),
        C.increasing & C.flat.within(30, 31),
    ],
    ids=["paired", "one-side", "narrow"],
)
def test_flat_rows_hold_the_slope_at_zero_and_drop_held_rows(rules):
    knots = _open_knots()
    pieces = shape_pieces(rules, LO, HI)
    (a, b) = next((a, b) for kind, a, b in pieces if kind == "flat")
    rows, flat = shape_rows(knots, DEGREE, pieces, natural_ends=False)
    np.testing.assert_array_equal(flat, refined_derivative_rows(knots, DEGREE, 1, a, b))
    Q, _ = np.linalg.qr(flat.T, mode="complete")
    Z = Q[:, flat.shape[0] :]  # coefficients the flat equalities allow
    x = np.linspace(a, b, 500)
    for c in np.random.default_rng(2).normal(size=(5, Z.shape[1])):
        slope = BSpline(knots, Z @ c, DEGREE)(x, nu=1)
        scale = np.max(np.abs(BSpline(knots, Z @ c, DEGREE)(np.linspace(LO, HI, 500), nu=1)))
        assert np.max(np.abs(slope)) <= 1e3 * U * scale
    # Mutation check: the same pieces without the flat equalities keep every
    # row; the rows dropped are exactly those the equalities hold at zero.
    others = [piece for piece in pieces if piece[0] != "flat"]
    plain, _ = shape_rows(knots, DEGREE, others, natural_ends=False)
    vanishing = np.linalg.norm(plain.A @ Z, axis=1) <= 1e3 * U * np.linalg.norm(plain.A, axis=1)
    assert vanishing.any()
    kept = [r for r in plain.A if any(np.array_equal(r, k) for k in rows.A)]
    assert len(kept) == rows.A.shape[0] == np.count_nonzero(~vanishing)
    assert np.min(np.linalg.norm(rows.A @ Z, axis=1)) > 1e-3


def test_pieces_split_around_flat():
    pieces = shape_pieces(C.increasing & C.flat.within(25, 40), LO, HI)
    assert sorted(pieces) == sorted(
        [("increasing", 17.0, 25.0), ("increasing", 40.0, 80.0), ("flat", 25.0, 40.0)]
    )


def test_pieces_ignore_caps_and_open_ends_take_the_range():
    pieces = shape_pieces(C.decreasing.below(30) & C.cap(0.5, 2.0) & C.convex, LO, HI)
    assert sorted(pieces) == sorted([("decreasing", 17.0, 30.0), ("convex", 17.0, 80.0)])


def test_stretch_clipped_and_refused():
    with pytest.raises(ValueError, match=r"increasing\.above\(90\)"):
        shape_pieces(C.increasing.above(90), LO, HI)
    with pytest.warns(UserWarning, match=r"increasing\.within\(70, 90\)") as record:
        pieces = shape_pieces(C.increasing.within(70, 90), LO, HI)
    assert len(record) == 1
    assert pieces == [("increasing", 70.0, 80.0)]


def test_label_ends_refused():
    with pytest.raises(ValueError, match="label"):
        shape_pieces(C.increasing.above("B"), LO, HI)


def test_low_degree_refused():
    knots = np.concatenate([np.full(3, LO), np.linspace(LO, HI, 8)[1:-1], np.full(3, HI)])
    with pytest.raises(ValueError, match="degree 2"):
        refined_derivative_rows(knots, 2, 2, 25.0, 40.0)
