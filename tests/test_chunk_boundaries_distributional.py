"""Distributional chunked loops give the same numbers when the data spans several chunks.

Each loop runs with its chunk shrunk through the knob it reads at call time (a module
constant, or the row chunk of the predictor matrices it is handed) or, where the step is
compiled in, on data longer than one step, so the data spans several chunks with a ragged
last one. The result is compared with the same quantity in one chunk, or with a reference
computed here: exactly where every cell is the same arithmetic in either case, and within
the rounding of a reordered sum where the chunks reorder a sum. The chunked call runs
first, so no freed one-chunk result can stand in for a row the loop never wrote.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from superglm import GeneralizedParetoLSS, TweedieLSS
from superglm._frame import as_eager_frame
from superglm.distributional._row_design import bounded_predictor_matrices
from superglm.distributional.checks import compare, qq
from superglm.distributional.families import gaussian
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import _curvature_direction
from superglm.distributional.kernels import negative_binomial as nb_kernel
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.residuals import compute_residuals
from superglm.distributional.smoothing import endpoint_laml
from superglm.distributional.solver._joint_tables import _MAX_TABLE_CHANNELS, joint_table_plan
from superglm.distributional.solver._likelihood_cache import build_likelihood_cache
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Categorical, Spline
from tests.bound_predictor_fixtures import model_from_templates

U = 2.0**-53


def _gamma(k: int) -> float:
    return k * U / (1.0 - k * U)


def _unit_weights(n: int):
    return resolve_likelihood_weights(None, n_observations=n, contract=WeightContract("prior"))


def _discrete_layout(n: int):
    """Two Gaussian predictors sharing one binned spline column, the first with a level term."""
    rng = np.random.default_rng(12)
    frame = as_eager_frame(
        pd.DataFrame(
            {"x": rng.choice(np.linspace(-1.0, 1.0, 9), n), "g": rng.choice(["a", "b", "c"], n)}
        )
    )
    family = GaussianLS()
    compiled = compile_predictors(
        frame,
        _unit_weights(n),
        family.parameters,
        (
            Predictor(
                "location",
                {"x": Spline(kind="cr", n_knots=5, discrete=True), "g": Categorical()},
            ),
            Predictor("scale", {"x": Spline(kind="cr", n_knots=4, discrete=True)}),
        ),
        model_discrete=True,
        n_bins_config=9,
    )
    return build_stacked_layout(compiled)


def test_murphy_curves_are_the_same_in_threshold_blocks(monkeypatch) -> None:
    """Blocks of three thresholds over eleven (a ragged last block) give the one-block curves.

    Each threshold's mean score is its own mass-weighted sum over the ``n`` rows, so blocking
    changes only the order BLAS gives that sum: each order is within ``gamma_{n + 1}`` (the
    sum and its division by the count) of the row terms' magnitude, which ``a + b`` bounds
    for both curves and their difference. The standard error is the root of a quotient of a
    sum of nonnegative squares, so it agrees to relative ``2 gamma_{n + 4}``.
    """
    rng = np.random.default_rng(268)
    n = 53
    y = rng.gamma(2.0, 1.0, n)
    a = y + rng.normal(scale=0.5, size=n)
    b = 0.8 * y + 0.3
    thresholds = np.linspace(-0.5, 5.0, 11)
    monkeypatch.setattr(compare, "_MURPHY_BLOCK_CELLS", 3 * n)
    blocked = compare.murphy_diagram(a, b, y, level=0.3, thresholds=thresholds)
    monkeypatch.undo()
    whole = compare.murphy_diagram(a, b, y, level=0.3, thresholds=thresholds)
    tolerance = 2.0 * _gamma(n + 1) * (whole.a + whole.b)
    for name in ("a", "b", "difference"):
        assert np.all(np.abs(getattr(blocked, name) - getattr(whole, name)) <= tolerance), name
    np.testing.assert_allclose(blocked.difference_se, whole.difference_se, rtol=2.0 * _gamma(n + 4))


def test_qq_envelope_is_the_same_in_simulation_blocks(monkeypatch) -> None:
    """Blocks of three simulated vectors over ten (ragged last) give the one-block envelope.

    Tweedie has an atom at zero, so every simulated cell takes a uniform from one stream for
    its randomised PIT: blocks must take the stream's draws in order, each once. Every cell
    is then the same elementwise transform of the same draw and uniform in either case, so
    the envelopes are equal exactly.
    """
    rng = np.random.default_rng(138)
    n = 40
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, n)})
    jumps = rng.poisson(1.5, n)
    y = np.array([rng.gamma(2.0 * count, 0.5) if count else 0.0 for count in jumps])
    assert np.any(y == 0.0)
    model = model_from_templates(
        family=TweedieLSS(),
        predictors=[Predictor(name, {}) for name in ("mean", "dispersion", "power")],
    ).fit(frame, y)
    fitted = model._require_fitted()
    residuals = compute_residuals(fitted, frame, y)
    monkeypatch.setattr(qq, "_BLOCK_CELLS", 3 * n)
    blocked = qq.qq_payload(fitted, residuals, n_sim=10, X=frame, seed=3)
    monkeypatch.undo()
    whole = qq.qq_payload(fitted, residuals, n_sim=10, X=frame, seed=3)
    np.testing.assert_array_equal(blocked.envelope_lower, whole.envelope_lower)
    np.testing.assert_array_equal(blocked.envelope_upper, whole.envelope_upper)


def test_gaussian_carrier_formed_in_weight_blocks_is_the_eager_carrier(monkeypatch) -> None:
    """A chunk-prepared plan's carrier, formed in blocks of eight rows over 37, is the eager one.

    ``0.5 log w`` is the same elementwise arithmetic in a block, so the carrier, its rounding
    scale and its digest (streamed in the same blocks) are equal exactly.
    """
    n = 37
    weights = resolve_likelihood_weights(
        np.linspace(0.25, 4.0, n), n_observations=n, contract=WeightContract("prior")
    )
    response = np.linspace(-0.5, 0.75, n)
    family = GaussianLS()
    monkeypatch.setattr(gaussian, "_CARRIER_BLOCK_ROWS", 8)
    root = family.bind_chunked_likelihood(response, weights, COMPLETE_OBSERVATION)
    assert root.parameter_independent_carrier is None
    blocked = root.carrier_rounding()
    eager = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    np.testing.assert_array_equal(blocked.carrier, eager.parameter_independent_carrier)
    np.testing.assert_array_equal(blocked.scale, eager.carrier_rounding().scale)
    assert root.carrier_digest == eager.carrier_digest


def test_generalized_pareto_third_derivatives_are_the_same_in_row_blocks(monkeypatch) -> None:
    """Second curvature directions whose third derivatives are formed in blocks of seven rows
    over 37 (ragged last) are the one-block directions.

    Each row's third derivatives come from that row's ``dH(e_k)`` alone, the same arithmetic
    in any block, so the directions agree exactly: stacked axis pairs (the LAML Hessian's
    call) and one direction pair per row.
    """
    family = GeneralizedParetoLSS()
    links = tuple(spec.default_link for spec in family.parameters)
    n = 37
    y = np.linspace(0.05, 6.0, n)
    eta = np.column_stack(
        [links[0].link(np.linspace(0.5, 3.0, n)), links[1].link(np.linspace(0.05, 0.6, n))]
    )
    plan = family.bind_likelihood(y, _unit_weights(n), COMPLETE_OBSERVATION)
    axes = [(0, 0), (0, 1), (1, 1)]
    first, second = np.zeros((2, len(axes), 1, 2))
    for index, (q, r) in enumerate(axes):
        first[index, :, q] = second[index, :, r] = 1.0
    per_row = np.random.default_rng(287).normal(size=(2, n, 2))
    pairs = ((first, second), (per_row[0], per_row[1]))

    def directions():
        return [
            family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
            for a, b in pairs
        ]

    monkeypatch.setattr(_curvature_direction, "STACKED_ROW_CHUNK", 7)
    blocked = directions()
    monkeypatch.undo()
    for got, want in zip(blocked, directions(), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("route", ["compiled", "array"])
def test_negative_binomial_recurrence_sums_every_term_past_its_block(route, monkeypatch) -> None:
    """Counts either side of the 256-term recurrence block, up to 1,000, give the row's value.

    The value's count-dependent part is ``sum_{j < y} log((theta + j) / (mean + theta))``.
    Both routes sum it in blocks of 256 terms (the compiled kernel within its row, the array
    route in tiles of terms), a block size the compiled kernel fixes, so the counts past 256
    span several blocks. The reference sums the same terms with ``math.fsum``; a blocked sum
    of ``y`` terms, its blocks folded exactly, and the few roundings that add ``y log(mean) -
    theta log1p(mean / theta)`` stay within ``gamma_{y + 8}`` of the terms' magnitude.
    """
    counts = np.array([0, 3, 255, 256, 257, 600, 1000])
    mean = np.array([1.0, 2.0, 200.0, 250.0, 300.0, 300.0, 950.0])
    theta = np.array([2.0, 1.5, 3.0, 2.0, 2.5, 2.0, 4.0])
    if route == "array":
        monkeypatch.setattr(nb_kernel, "_compiled_rows", lambda *args, **kwargs: None)
    got = nb_kernel.evaluate_negative_binomial_rows(
        counts, mean, theta, np.ones(len(counts)), "prior", derivative_order=2
    ).optimizing_log_likelihood
    for row, (count, row_mean, row_theta) in enumerate(zip(counts, mean, theta, strict=True)):
        offsets = np.arange(count, dtype=np.float64)
        total = row_mean + row_theta
        argument = (offsets - row_mean) / total
        terms = np.where(argument < -0.5, np.log((row_theta + offsets) / total), np.log1p(argument))
        tail = [count * math.log(row_mean), -row_theta * math.log1p(row_mean / row_theta)]
        expected = math.fsum([*terms, *tail])
        magnitude = math.fsum(np.abs(terms)) + abs(tail[0]) + abs(tail[1])
        assert abs(got[row] - expected) <= _gamma(int(count) + 8) * magnitude, int(count)


def test_endpoint_row_certificate_sums_every_row_chunk(monkeypatch) -> None:
    """Row chunks of seven over 50 (ragged last) give the one-chunk drift-certificate bound.

    The bound is a sum of nonnegative row terms inflated for the rounding of one chunk, so
    the chunks move it by that inflation and a reordered sum, both below 1e-12 relative;
    a dropped or repeated chunk moves it by a whole chunk's share.
    """
    n = 50
    layout = _discrete_layout(n)
    p = layout.n_coefficients
    rng = np.random.default_rng(375)
    certificate = rng.uniform(0.5, 2.0, size=(n, 3))
    basis = np.linalg.qr(rng.normal(size=(p, p - 2)))[0]
    root = rng.normal(size=(p - 2, p - 2))
    inverse = root @ root.T + (p - 2) * np.eye(p - 2)
    monkeypatch.setattr(
        endpoint_laml,
        "bounded_predictor_matrices",
        lambda layout: bounded_predictor_matrices(layout, chunk_size=7),
    )
    blocked = endpoint_laml._row_certificate_error(layout, certificate, basis, inverse)
    monkeypatch.undo()
    whole = endpoint_laml._row_certificate_error(layout, certificate, basis, inverse)
    assert blocked == pytest.approx(whole, rel=1e-10)


def test_likelihood_cache_compares_every_index_block_before_reusing_a_child() -> None:
    """A cached child is reused only for exactly its range of rows, checked in 8,192-index blocks.

    Two rows swapped inside any one of the three blocks of 16,461 indices (two full blocks
    and a ragged final block of 77) are not the range, so the cache takes a fresh child with
    those rows' weights rather than the child it holds; each block is checked on its own.
    """
    n = 2 * 8192 + 77
    values = np.linspace(0.5, 1.5, n)
    weights = resolve_likelihood_weights(values, n_observations=n, contract=WeightContract("prior"))
    family = GaussianLS()
    plan = family.bind_likelihood(np.linspace(-1.0, 1.0, n), weights, COMPLETE_OBSERVATION)
    cache = build_likelihood_cache(family, plan)
    assert cache is not None
    indices = np.arange(n)
    held = cache.take(plan, indices, start=0, stop=n)
    assert cache.take(plan, indices, start=0, stop=n) is held
    for rows in ([20, 21], [8200, 8201], [n - 2, n - 1]):
        swapped = indices.copy()
        swapped[rows] = swapped[rows[::-1]]
        child = cache.take(plan, swapped, start=0, stop=n)
        assert child is not held
        np.testing.assert_array_equal(child.weights.values, values[swapped])


def test_joint_tables_hold_each_channel_of_a_column_pair_once() -> None:
    """A column pair's channels past ``_MAX_TABLE_CHANNELS`` go to further tables, each once.

    The spline column both predictors share carries every input channel (three pair
    channels, two scores, two row weights), so its level table splits into tables of four
    and three. Each table accumulates every channel it holds over the rows, so a pair's
    tables hold its channels once each, in order, and their widths sum to the entries the
    plan reads.
    """
    plan = joint_table_plan(_discrete_layout(50), 1 << 30)
    assert plan is not None
    held: dict[tuple[int, int | None], list[int]] = {}
    for table in plan.tables:
        assert len(table.picks) <= _MAX_TABLE_CHANNELS
        held.setdefault((table.left, table.right), []).extend(table.picks)
    widest = max(held.values(), key=len)
    assert widest == list(range(plan.stack_rows))
    assert len(widest) > _MAX_TABLE_CHANNELS
    for channels in held.values():
        assert channels == sorted(set(channels))
    assert sum(map(len, held.values())) == len(plan.slot)
