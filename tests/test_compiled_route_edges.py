"""Every compiled route against its array route and mpmath at the edges of its inputs.

``tests/_route_edges.py`` holds the sweep and each route's grid, each distilled from a search on
ten times as fine a walk.  Each test here forces its event: a limit test asks for a row
past the limit, and a mutation test puts an entry of a compiled row off by ``1e-7`` near one edge,
which the sweep must report.
"""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("mpmath")

from superglm.distributional.kernels._declared_bounds import gpd_window_jets  # noqa: E402
from tests import _route_edges as edges  # noqa: E402

BY_NAME = {route.name: route for route in edges.ROUTES}
# What mpmath may spend per route: these laws cost most per reference.
REFERENCES = {
    "two_piece_mean": 30,
    "generalized_gamma_location": 20,
    "generalized_gamma_mean": 30,
    "tweedie_prior": 12,
    "tweedie_frequency": 12,
    "tweedie_predictor_prior": 12,
    "tweedie_predictor_frequency": 12,
}
MUTATED = 1.0 + 1e-7


@pytest.mark.parametrize("route", edges.ROUTES, ids=lambda route: route.name)
def test_a_compiled_row_reads_as_well_as_its_array_row(route):
    """The rule of ``tests/_route_edges.py``, over the route's grid: a row that fails names itself."""
    columns = route.grid(False)
    failures, counts = edges.sweep(
        route, columns, max_references=REFERENCES.get(route.name, 40), audit=10
    )
    assert not failures, failures[:5]
    assert counts["compiled"] >= 0.3 * counts["rows"], counts


def _served(route_name: str, *row: float) -> bool:
    route = BY_NAME[route_name]
    columns = tuple(np.array([value]) for value in row)
    k = len(route.units(columns))
    return bool(edges.compiled_rows(route, columns, k, k * (k + 1) // 2)[1][0])


@pytest.mark.parametrize(
    ("skew", "compiled"),
    [
        (0.0, True),
        (0.99, True),
        (-0.99, True),
        (np.nextafter(0.99, 1.0), False),
        (np.nextafter(-0.99, -1.0), False),
        (1.0 - 1e-12, False),
        (-1.0 + 1e-12, False),
    ],
)
def test_a_two_piece_mean_row_past_its_skew_limit_is_the_array_routes(skew, compiled):
    assert _served("two_piece_mean", 1.0, 1.0, 0.5, skew, 1.0) is compiled


@pytest.mark.parametrize(
    ("shape", "compiled"),
    [(0.0, True), (0.5, True), (1.0, True), (-1e-12, False), (-0.5, False), (-1.0, False)],
)
def test_a_pareto_row_below_its_familys_shape_domain_is_the_array_routes(shape, compiled):
    assert _served("generalized_pareto", 0.3, 1.2, shape, 1.0) is compiled


@pytest.mark.parametrize(
    ("scale", "compiled"),
    [(1.0, True), (1e-4, True), (np.nextafter(1e-4, 0.0), False), (1e-8, False), (1e-18, False)],
)
def test_a_two_piece_mean_row_below_its_scale_limit_is_the_array_routes(scale, compiled):
    assert _served("two_piece_mean", 1.0, 1.0, scale, -0.6, 1.0) is compiled


def test_a_row_at_a_tiny_scale_reads_the_same_beside_any_other_row():
    """The reviewed row: at ``scale = 1e-18`` the compiled value read ``-9588``, and ``41.4``, the
    array route's, once an unrelated row past the range gate sent its chunk to the array route."""

    def rows(*columns):
        y, mean, scale, skew = (np.array(column) for column in columns)
        return edges.two_piece_kernel.mean_rows(
            y, mean, scale, skew, np.ones_like(y), derivative_order=2
        )

    alone = rows([1.0], [1.0], [1e-18], [-0.6])
    beside = rows([1.0, 2.0**201], [1.0, 2.0**201], [1e-18, 0.1], [-0.6, 0.0])
    assert alone.optimizing_log_likelihood[0] == beside.optimizing_log_likelihood[0]
    assert np.array_equal(alone.score[0], beside.score[0])
    assert np.array_equal(alone.hessian_packed[0], beside.hessian_packed[0])


def test_a_pareto_row_next_to_a_negative_shape_endpoint_keeps_its_scale_entries():
    """The reviewed rows: at ``shape = -1`` the exact scale score is ``-1/scale`` and the exact
    scale curvature ``1/scale^2``, whatever the response.  The compiled route read ``0.7458`` for
    ``0.6944`` at a gap of ``1e-5``, and ``30.0`` for ``-0.8333`` at a gap of ``1e-9``."""
    scale = 1.2

    def row(gap):
        return BY_NAME["generalized_pareto"].call(
            tuple(np.array([v]) for v in (scale * (1.0 - gap), scale, -1.0, 1.0))
        )

    assert row(1e-5).hessian_packed[0, 0] == pytest.approx(1.0 / scale**2, rel=1e-5)
    assert row(1e-9).score[0, 0] == pytest.approx(-1.0 / scale, rel=1e-6)


@pytest.mark.parametrize(
    ("eta_mean", "eta_phi", "compiled"),
    [
        (100.0, -100.0, True),
        (-100.0, 100.0, True),
        (float(np.nextafter(100.0, 101.0)), 0.0, False),
        (0.0, float(np.nextafter(-100.0, -101.0)), False),
    ],
)
def test_a_tweedie_zero_row_past_its_window_is_the_natural_routes(eta_mean, eta_phi, compiled):
    assert _served("tweedie_predictor_prior", 0.0, eta_mean, eta_phi, 0.0, 1.0) is compiled


@pytest.mark.parametrize(
    "eta_power", [37.0, -38.0, 40.0, -40.0, float("inf"), float("-inf"), float("nan")]
)
@pytest.mark.parametrize("y", [0.0, 2.0], ids=("zero", "positive"))
def test_a_tweedie_row_with_its_power_on_a_wall_is_the_natural_routes_to_refuse(y, eta_power):
    """The family's power is strictly inside its walls: the natural route refuses this row, so a
    trial reaching it is rejected, and a compiled number there would accept it."""
    route = BY_NAME["tweedie_predictor_prior"]
    columns = tuple(np.array([v]) for v in (y, 0.5, 0.2, eta_power, 1.0))
    with np.errstate(all="ignore"):
        assert not _served("tweedie_predictor_prior", y, 0.5, 0.2, eta_power, 1.0)
        assert not edges.array_rows(route, columns, 3, 6)[1][0]


def test_a_tweedie_row_whose_weighted_value_overflows_is_the_natural_routes_to_refuse():
    """The series finishes at about ``-1e304``, and a frequency of ``2^53`` takes it past float64."""
    row = (5e-324, 0.0, -700.0, -10.0)

    def value_trial(weight):
        y, weights = np.array([row[0]]), np.array([weight])
        rows = edges.tweedie_kernel.tweedie_predictor_rows(
            y, weights, "frequency", edges._POWER_LINK, False
        )
        stack = np.empty((9, 1))
        stack[:3, 0] = row[1:]
        return rows.evaluate(0, 0, 1, stack, 0)

    finished, value = value_trial(1.0)
    assert finished and -1e305 < value < -1e303
    assert value_trial(2.0**53) == (False, 0.0)
    with pytest.raises(edges.tweedie_kernel.TweedieNumericalRefusal):
        edges.tweedie_kernel.evaluate_tweedie_rows(
            np.array([row[0]]),
            np.ones(1),
            np.exp(np.array([row[2]])),
            edges._POWER_LINK.inverse(np.array([row[3]])),
            np.array([2.0**53]),
            "frequency",
            derivative_order=0,
        )


def test_positive_shape_rows_are_compiled_up_to_their_finite_limits():
    """The limit takes no row of the family's domain: a shape of zero or one, ties and far tails."""
    for shape in (0.0, 1e-50, 1e-8, 0.5, 1.0):
        for y in (0.0, 1e-30, 1.0, 1e30):
            assert _served("generalized_pareto", y, 1.0, shape, 1.0)


def test_a_pareto_shape_past_the_family_domain_takes_the_array_route():
    """The reviewed row: ``scale/y = 1e-120`` underflows the tail engine's square of it before a
    shape of ``1e50`` scales it back, which read a scale curvature of ``-2e70`` for ``-1e70``."""
    assert not _served("generalized_pareto", 1e60, 1e-60, 1e50, 1.0)
    assert not _served("generalized_pareto", 1.0, 1.0, float(np.nextafter(1.0, 2.0)), 1.0)
    row = BY_NAME["generalized_pareto"].call(tuple(np.array([v]) for v in (1e60, 1e-60, 1e50, 1.0)))
    assert row.hessian_packed[0, 0] == pytest.approx(-1e70, rel=1e-3)


@pytest.mark.parametrize("shape", [-0.2, 2.0])
def test_a_narrow_pareto_window_outside_the_family_domain_reads_its_tails(shape):
    """The density kernel marks such a shape, and the window must not read that mark as its value:
    ``log(S(lower) - S(upper))`` against mpmath, on a window the quadrature would otherwise take."""
    mp = pytest.importorskip("mpmath")
    lower, upper, scale = 1.0, 1.0 + 1e-4, 2.0
    value = gpd_window_jets(
        np.array([lower]),
        np.array([upper]),
        np.array([scale]),
        np.array([shape]),
        order=2,
    )[0][0]
    with mp.workdps(60):
        survival = [
            (1 + mp.mpf(shape) * mp.mpf(y) / mp.mpf(scale)) ** (-1 / mp.mpf(shape))
            for y in (lower, upper)
        ]
        exact = float(mp.log(survival[0] - survival[1]))
    assert value == pytest.approx(exact, rel=1e-9)


def _edge_rows(name: str):
    """The rows of one route's grid at one edge, where its conditioning leaves ``MUTATED`` visible."""
    route = BY_NAME[name]
    columns = route.grid(False)
    inside = _EDGES[name][0](columns) & (route.conditioning(columns) <= 1e3)
    return route, tuple(c[inside] for c in columns)


def _off_jets(original, edge_of):
    def density_rows(*args, **kwargs):
        jets = original(*args, **kwargs)
        if jets is None:
            return None
        inside = edge_of(args[2], *args[3])
        jets[2][inside, :] *= MUTATED
        return jets

    return density_rows


def _off_evaluation(original, edge_of):
    def evaluated(*args, **kwargs):
        rows = original(*args, **kwargs)
        if rows is None:
            return None
        hessian = np.array(rows.hessian_packed)
        hessian[edge_of(*args)] *= MUTATED
        return replace(rows, hessian_packed=hessian)

    return evaluated


def _off_negative_binomial(original, edge_of):
    def compiled_rows(counts, continuous, prior, order):
        rows = original(counts, continuous, prior, order)
        if rows is None:
            return None
        hessian = np.array(rows.hessian_packed)
        hessian[edge_of(counts, *continuous)] *= MUTATED
        return replace(rows, hessian_packed=hessian)

    return compiled_rows


def _off_predictor_rows(original, edge_of):
    """The row kernel takes its rows through ``stack``, so the off entries are made there."""

    def rows(y, weight, mode, row_start, first, last, stack, *args):
        inside = edge_of(y[row_start + first], *stack[:3, first], weight[row_start + first])
        result = original(y, weight, mode, row_start, first, last, stack, *args)
        if inside:
            stack[:6, first:last] *= MUTATED
        return result

    return rows


def _column(index, test):
    return lambda columns: test(columns[index])


# Per route: the grid's edge, the attribute to replace and how, and the edge in that callee's terms.
_EDGES = {
    "gamma_prior": (
        _column(2, lambda cv: cv <= 1e-10),
        (edges.gamma_kernel, "density_rows", _off_jets, lambda y, mean, cv: cv <= 1e-10),
    ),
    "gamma_frequency": (
        _column(2, lambda cv: cv >= 1e10),
        (edges.gamma_kernel, "density_rows", _off_jets, lambda y, mean, cv: cv >= 1e10),
    ),
    "generalized_pareto": (
        _column(2, lambda xi: xi == 0.0),
        (edges.pareto_kernel, "density_rows", _off_jets, lambda y, psi, xi: xi == 0.0),
    ),
    "log_normal_mean": (
        _column(2, lambda s: s >= 1e5),
        (edges.log_normal_kernel, "density_rows", _off_jets, lambda y, m, s: s >= 1e5),
    ),
    "two_piece_location": (
        _column(3, lambda e: np.abs(e) >= 0.999),
        (edges.two_piece_kernel, "density_rows", _off_jets, lambda y, mu, s, e: np.abs(e) >= 0.999),
    ),
    "two_piece_mean": (
        _column(3, lambda e: np.abs(e) >= 0.9),
        (edges.two_piece_kernel, "density_rows", _off_jets, lambda y, m, s, e: np.abs(e) >= 0.9),
    ),
    "negative_binomial_prior": (
        _column(0, lambda c: c >= 256.0),
        (edges.nb_kernel, "_compiled_rows", _off_negative_binomial, lambda c, m, t, w: c >= 256),
    ),
    "negative_binomial_frequency": (
        _column(2, lambda t: t >= 1e30),
        (edges.nb_kernel, "_compiled_rows", _off_negative_binomial, lambda c, m, t, w: t >= 1e30),
    ),
    "generalized_gamma_location": (
        _column(3, lambda q: np.abs(q) <= 1e-12),
        (
            edges.gg_kernel,
            "_compiled_rows",
            _off_evaluation,
            lambda y, first, s, q, m, **_: np.abs(q) <= 1e-12,
        ),
    ),
    "generalized_gamma_mean": (
        _column(3, lambda q: q <= -3.0),
        (edges.gg_kernel, "_compiled_rows", _off_evaluation, lambda y, m, s, q, w, **_: q <= -3.0),
    ),
    "tweedie_prior": (
        _column(3, lambda p: p >= 1.95 - 1e-9),
        (
            edges.tweedie_kernel,
            "evaluate_tweedie_rows",
            _off_evaluation,
            lambda y, m, d, p, *_, **__: p >= 1.95 - 1e-9,
        ),
    ),
    "tweedie_frequency": (
        _column(3, lambda p: p <= 1.05 + 1e-9),
        (
            edges.tweedie_kernel,
            "evaluate_tweedie_rows",
            _off_evaluation,
            lambda y, m, d, p, *_, **__: p <= 1.05 + 1e-9,
        ),
    ),
    "tweedie_predictor_prior": (
        _column(3, lambda eta_power: eta_power <= -10.0),
        (
            edges.tweedie_kernel._compiled,
            "_tweedie_predictor_rows",
            _off_predictor_rows,
            lambda y, eta_mean, eta_phi, eta_power, w: eta_power <= -10.0,
        ),
    ),
    "tweedie_predictor_frequency": (
        _column(1, lambda eta_mean: np.abs(eta_mean) >= 50.0),
        (
            edges.tweedie_kernel._compiled,
            "_tweedie_predictor_rows",
            _off_predictor_rows,
            lambda y, eta_mean, eta_phi, eta_power, w: abs(eta_mean) >= 50.0,
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(_EDGES))
def test_the_sweep_fails_a_compiled_entry_off_by_a_small_factor_near_an_edge(name):
    route, columns = _edge_rows(name)
    assert len(columns[0]) > 0
    module, attribute, wrap, edge_of = _EDGES[name][1]
    with mock.patch.object(module, attribute, wrap(getattr(module, attribute), edge_of)):
        failures, _ = edges.sweep(route, columns, max_references=6)
    assert failures, "a Hessian off by 1e-7 near this edge went unreported"
