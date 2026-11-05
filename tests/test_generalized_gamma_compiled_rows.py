"""The compiled generalized gamma rows against the array rows they were ported from.

The array code is the oracle. ``valid`` and every refusal agree exactly. Values
agree to a few dozen roundings of the magnitude of a channel's terms, plus a
multiple of what the channel moves when one input moves by ``256 u``. That spread
is how a row's conditioning reaches the comparison: near ``log y = mu`` or at a
tiny ``sigma`` a rounding in ``log`` moves a channel by far more than ``u`` times
its value, in both codes.
"""

from __future__ import annotations

import itertools
import math
import threading

import numpy as np
import pytest
from scipy import special

from superglm._parallel import parallel_config
from superglm.distributional.kernels import _generalized_gamma_numba as compiled
from superglm.distributional.kernels import generalized_gamma as gg
from superglm.distributional.kernels._common import PARALLEL_MIN_ROWS

U = 2.0**-53
ROUNDINGS = 64.0 * U
PERTURBATION = 256.0 * U
SPREADS = 32.0


def _refused_by_the_array(sigma, q):
    """Rows whose mean exists but whose loading the array code refuses, per ``(sigma, Q)``."""

    def refuses(s, shape):
        pair = np.array([s]), np.array([shape])
        if not gg.mean_exists(*pair)[0]:
            return False
        try:
            gg.log_mean_loading(*pair)
        except gg.GeneralizedGammaDomainError:
            return True
        return False

    pairs, inverse = np.unique(np.column_stack([sigma, q]), axis=0, return_inverse=True)
    with np.errstate(all="ignore"):
        refused = np.array([refuses(*pair) for pair in pairs])
    return refused[inverse.ravel()]


def _chain(me, loading, score, hessian, *, terms):
    """The mean-form channels from the location channels; ``terms`` sums their magnitudes."""
    _, c_s, c_q, c_ss, c_sq, c_qq = loading
    at, sign = (np.abs, 1.0) if terms else (lambda value: value, -1.0)
    s_mu, s_sig, s_q = score.T
    chained_score = np.column_stack(
        [at(s_mu) / me, at(s_sig) + sign * at(s_mu * c_s), at(s_q) + sign * at(s_mu * c_q)]
    )
    if hessian is None:
        return chained_score, None
    h_mm, h_ms, h_mq, h_ss, h_sq, h_qq = hessian.T
    chained_hessian = np.column_stack(
        [
            (at(h_mm) + sign * at(s_mu)) / (me * me),
            (at(h_ms) + sign * at(h_mm * c_s)) / me,
            (at(h_mq) + sign * at(h_mm * c_q)) / me,
            at(h_ss) + sign * at(2.0 * h_ms * c_s) + at(h_mm * c_s * c_s) + sign * at(s_mu * c_ss),
            at(h_sq)
            + sign * at(h_ms * c_q)
            + sign * at(h_mq * c_s)
            + at(h_mm * c_s * c_q)
            + sign * at(s_mu * c_sq),
            at(h_qq) + sign * at(2.0 * h_mq * c_q) + at(h_mm * c_q * c_q) + sign * at(s_mu * c_qq),
        ]
    )
    return chained_score, chained_hessian


def _array_mean_rows(y, mean, sigma, q, weight, order):
    """The array mean rows, ``mu = log m - log C`` chained onto ``gg._location_rows``, and the
    magnitude of each channel's terms, which a cancelling channel rounds at.

    Refused rows take zero placeholders here: a caller compares them separately.
    """
    exists = gg.mean_exists(sigma, q) & ~_refused_by_the_array(sigma, q)
    n = len(y)
    optimizing = np.zeros(n)
    score = np.zeros((n, 3)) if order >= 1 else None
    hessian = np.zeros((n, 6)) if order == 2 else None
    score_terms = np.zeros_like(score) if order >= 1 else None
    hessian_terms = np.zeros_like(hessian) if order == 2 else None
    valid = np.array(exists, dtype=bool)
    if np.any(exists):
        loading, stirling = gg._log_mean_loading_and_stirling(sigma[exists], q[exists])
        me = mean[exists]
        base = gg._location_rows(
            y[exists],
            np.log(me) - loading[0],
            sigma[exists],
            q[exists],
            weight[exists],
            derivative_order=order,
            stirling=stirling,
        )
        optimizing[exists] = base.optimizing_log_likelihood
        valid[exists] = base.valid
        if order >= 1:
            channels = me, loading, base.score, base.hessian_packed
            score[exists], chained_hessian = _chain(*channels, terms=False)
            score_terms[exists], chained_terms = _chain(*channels, terms=True)
            if order == 2:
                hessian[exists] = chained_hessian
                hessian_terms[exists] = chained_terms
    evaluated = gg._placeholders_where_invalid(optimizing, score, hessian, valid)
    terms = [c for c in (np.abs(optimizing)[:, None], score_terms, hessian_terms) if c is not None]
    return evaluated, terms


def _array_rows(form, inputs, order):
    """The array evaluation and the magnitude of each channel's terms."""
    if form == "mean":
        return _array_mean_rows(*inputs, order)
    evaluated = gg._location_rows(*inputs, derivative_order=order)
    return evaluated, [np.abs(channel) for channel in _channels(evaluated)]


def _compiled_rows(form, inputs, order):
    rows = gg.mean_rows if form == "mean" else gg.location_rows
    return rows(*inputs, derivative_order=order)


def _channels(evaluated):
    channels = [evaluated.optimizing_log_likelihood[:, None]]
    channels += [c for c in (evaluated.score, evaluated.hessian_packed) if c is not None]
    return channels


def _grid(form):
    """Rows across every switch: ``Q`` at zero, at ``k = 8``; ``Q w`` at 1; ``sigma Q`` at
    1/4, 0.8 and the mean's wall; responses, scales and shapes at the ends of float64."""
    switches = [1.0 / math.sqrt(k) for k in (8.0 * (1 - 1e-15), 8.0, 8.0 * (1 + 1e-15), 7.5, 1.0)]
    shapes = [0.0, 1e-300, 5e-9, 9.999999e-9, 1.0000001e-8, 2e-8, 1e-6, 1e-4, 0.01, 0.1, 0.3]
    shapes += [0.5, 0.9, 1.5, 2.0, 5.0, 30.0, 1e4, 1e100, 1e150, 1.3e154, 1e200, *switches]
    shapes = [*shapes, *(-q for q in shapes)]
    scales = [1e-300, 1e-6, 0.01, 0.3, 1.0, 1.9, 5.0, 1e3, 1e8, 1e300]
    responses = [1e-300, 1e-10, 0.1, 1.0, 3.7, 1e3, 1e10, 1e300]
    firsts = (
        [1e-200, 1e-5, 1.0, 3.0, 1e5, 1e200]
        if form == "mean"
        else [-500.0, -5.0, 0.0, 1.3, 5.0, 500.0]
    )
    rows = list(itertools.product(responses, firsts, scales, shapes))
    # sigma Q across its switches and the wall at -1, on both signs of Q.
    walls = [0.2499, 0.25, 0.2501, 0.7999, 0.8, 0.8001, 0.999, 1.0 - 1e-12, 1.0, 1.1, 2.0]
    for y, first, q, v in itertools.product(
        responses[1:-1], firsts[1:-1], [0.2, 0.5, 1.0, 2.0, 1e-3, 1e-7], walls
    ):
        rows += [(y, first, v / q, q), (y, first, v / q, -q)]
    # Q w across 1 at location zero; for the mean form the same responses at a mean of one.
    for q, ratio, side in itertools.product(
        [0.3, 0.5, 1.0, 2.0], [0.999999, 1.0, 1.000001], [1.0, -1.0]
    ):
        rows.append((math.exp(side * ratio / q), 1.0 if form == "mean" else 0.0, 1.0, q))
        rows.append((math.exp(side * ratio / q), 1.0 if form == "mean" else 0.0, 1.0, -q))
    y, first, sigma, q = (np.array(column) for column in zip(*rows, strict=True))
    multiplier = np.resize(np.array([1.0, 0.5, 3.0e4, 1e-300, 1e300, 7.0]), len(y))
    return y, first, sigma, q, multiplier


def _without_refused(form, inputs):
    if form == "location":
        return inputs
    keep = ~_refused_by_the_array(inputs[2], inputs[3])
    return tuple(values[keep] for values in inputs)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("form", ["mean", "location"])
def test_compiled_rows_agree_with_the_array_rows(form, order):
    inputs = _without_refused(form, _grid(form))
    moved_by = [
        tuple(
            values * (1.0 + sign * PERTURBATION) if index == moved else values
            for index, values in enumerate(inputs)
        )
        for moved, sign in itertools.product(range(4), (1.0, -1.0))
    ]
    with np.errstate(all="ignore"):
        array, terms = _array_rows(form, inputs, order)
        moved = [_channels(_array_rows(form, near, order)[0]) for near in moved_by]
    rows = _compiled_rows(form, inputs, order)
    assert np.array_equal(rows.valid, array.valid)
    assert 0.3 < np.mean(array.valid) < 0.9  # both outcomes are exercised
    for channel, (reference, computed) in enumerate(
        zip(_channels(array), _channels(rows), strict=True)
    ):
        spread = np.max([np.abs(near[channel] - reference) for near in moved], axis=0)
        tolerance = ROUNDINGS * terms[channel] + SPREADS * spread
        far = np.abs(computed - reference) > tolerance
        assert not np.any(far), [
            tuple(values[row] for values in inputs) for row in np.flatnonzero(far.any(axis=1))[:5]
        ]
        assert np.all(computed[~array.valid] == 0.0)


def test_the_comparison_is_tight_on_ordinary_rows():
    """The tolerance stays near rounding where nothing is ill-conditioned."""
    inputs = tuple(
        np.array(values)
        for values in (
            [2.0, 0.4, 12.0, 3.7],
            [1.5, 0.9, 7.0, 2.0],
            [0.7, 0.3, 1.2, 0.9],
            [0.6, -0.4, 1.5, 1.0e-3],
            [1.0, 2.0, 0.5, 3.0],
        )
    )
    array, _ = _array_mean_rows(*inputs, 2)
    rows = _compiled_rows("mean", inputs, 2)
    assert np.all(array.valid) and np.all(rows.valid)
    for reference, computed in zip(_channels(array), _channels(rows), strict=True):
        np.testing.assert_allclose(computed, reference, rtol=1e-11, atol=1e-11)


def test_the_refusals_agree_on_every_scale_and_shape():
    inputs = _grid("mean")
    pairs = np.unique(np.column_stack(inputs[2:4]), axis=0)
    one = np.ones(1)
    refused_by_array = _refused_by_the_array(pairs[:, 0], pairs[:, 1])
    assert 0 < refused_by_array.sum() < len(pairs)
    for (sigma, q), refused in zip(pairs, refused_by_array, strict=True):
        row = (one, one, np.array([sigma]), np.array([q]), one)
        if refused:
            with pytest.raises(gg.GeneralizedGammaDomainError, match="mean loading"):
                gg.mean_rows(*row, derivative_order=0)
        else:
            gg.mean_rows(*row, derivative_order=0)


@pytest.mark.parametrize(
    "x",
    [1e-100, 1e-20, 1e-3, 0.5, 1.0, 1.4616321449683623, 2.0, 7.999, 8.0, 8.001, 40.0, 1e6, 1e16],
)
def test_the_stirling_triplet_matches_scipy(x):
    s0, s1, s2 = compiled._stirling_triplet(x)
    point = np.array([x])
    terms = abs(special.gammaln(x)) + abs((x - 0.5) * math.log(x)) + x + 1.0 if x < 8.0 else 0.0
    assert abs(s0 - gg.stirling_remainder(point)[0]) <= ROUNDINGS * (abs(s0) + terms)
    d_terms = abs(special.digamma(x)) + abs(math.log(x)) + 0.5 / x if x < 8.0 else 0.0
    assert abs(s1 - gg.stirling_remainder_d1(point)[0]) <= ROUNDINGS * (abs(s1) + d_terms)
    assert abs(s2 - gg.stirling_remainder_d2(point)[0]) <= ROUNDINGS * abs(s2)
    if x < 8.0:  # the array code's S'' against scipy's trigamma, the identity it walks
        identity = special.polygamma(1, x) - 1.0 / x - 0.5 / (x * x)
        scale = special.polygamma(1, x) + 1.0 / x + 0.5 / (x * x)
        assert abs(s2 - identity) <= ROUNDINGS * scale


def test_rows_that_repeat_a_shape_or_scale_keep_the_bits_of_the_row_alone():
    """Mutation check: terms kept past a change of ``Q`` or ``sigma`` read another row's."""
    sigma = np.array([0.7, 0.7, 0.9, 0.9, 5.0, 0.9, 0.9, 0.9, 1e300, 0.9])
    q = np.array([0.5, 0.5, 0.5, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 1.5])
    y = 1.0 + np.arange(len(q)) / 3.0
    first = 2.0 + np.arange(len(q)) / 7.0
    multiplier = np.ones(len(q))
    keep = np.flatnonzero(sigma < 1e300)  # the row a mean refuses goes alone, below
    for form, order in itertools.product(("mean", "location"), (0, 1, 2)):
        inputs = tuple(v[keep] for v in (y, first, sigma, q, multiplier))
        rows = _compiled_rows(form, inputs, order)
        for index in range(len(keep)):
            alone = _compiled_rows(form, tuple(v[index : index + 1] for v in inputs), order)
            for together, single in zip(_channels(rows), _channels(alone), strict=True):
                assert np.array_equal(together[index], single[0])
            assert rows.valid[index] == alone.valid[0]
    assert not gg.mean_exists(sigma[4:5], q[4:5])[0]  # a row with no mean sits between repeats
    with pytest.raises(gg.GeneralizedGammaDomainError, match="mean loading"):
        gg.mean_rows(y, first, sigma, q, multiplier, derivative_order=0)


def _pool_rows():
    index = np.arange(2.0 * PARALLEL_MIN_ROWS)
    y = 0.2 + (index % 97.0) / 11.0
    first = 1.0 + (index % 31.0) / 5.0
    sigma = 0.3 + (index % 13.0) / 17.0
    # Zero, both signs, the k = 8 switch and rows with no mean (sigma |Q| >= 1).
    q = np.array([0.0, 0.3, -0.3, 1.0 / math.sqrt(8.0), 1.2, -1.5, 5e-9])[(index % 7).astype(int)]
    multiplier = 1.0 + (index % 3.0)
    return y, first, sigma, q, multiplier


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("form", ["mean", "location"])
def test_rows_on_the_pool_keep_their_bits_at_every_worker_count(monkeypatch, form, order):
    """Two blocks must be inside the kernel at once. Mutation check: a route that never pools
    breaks the barrier."""
    inputs = _pool_rows()
    assert not np.all(gg.mean_exists(inputs[2], inputs[3]))  # some rows have no mean
    with parallel_config(n_jobs=1):
        serial = _compiled_rows(form, inputs, order)
    kernel = compiled._evaluate_rows
    barrier = threading.Barrier(2)
    meetings = [barrier, barrier]

    def recorded(*args):
        if meetings:
            meetings.pop().wait(timeout=30)
        return kernel(*args)

    monkeypatch.setattr(compiled, "_evaluate_rows", recorded)
    with parallel_config(n_jobs=4):
        pooled = _compiled_rows(form, inputs, order)
    assert not meetings
    assert np.array_equal(serial.valid, pooled.valid)
    assert np.any(pooled.valid) and (form == "location" or not np.all(pooled.valid))
    for left, right in zip(_channels(serial), _channels(pooled), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_a_refusal_in_a_late_block_refuses_the_call(n_jobs):
    y, first, sigma, q, multiplier = _pool_rows()
    sigma[-1], q[-1] = 1e300, 0.0
    with (
        parallel_config(n_jobs=n_jobs),
        pytest.raises(gg.GeneralizedGammaDomainError, match="mean loading"),
    ):
        gg.mean_rows(y, first, sigma, q, multiplier, derivative_order=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (0, 0.0),  # response
        (2, 0.0),  # sigma
        (3, np.inf),  # shape
        (4, 0.0),  # multiplier
    ],
)
@pytest.mark.parametrize("form", ["mean", "location"])
def test_input_domain_refusals_reach_both_forms(form, field, value):
    inputs = [np.ones(2) for _ in range(5)]
    inputs[field][1] = value
    with pytest.raises(gg.GeneralizedGammaDomainError):
        _compiled_rows(form, tuple(inputs), 0)


def test_readonly_and_strided_inputs_reach_the_kernel():
    theta = np.array([[1.5, 0.7, 0.5], [2.5, 0.9, -0.4]])
    y = np.array([1.0, 2.0])
    y.setflags(write=False)
    strided = gg.mean_rows(y, theta[:, 0], theta[:, 1], theta[:, 2], np.ones(2), derivative_order=2)
    plain = gg.mean_rows(
        y.copy(),
        theta[:, 0].copy(),
        theta[:, 1].copy(),
        theta[:, 2].copy(),
        np.ones(2),
        derivative_order=2,
    )
    np.testing.assert_array_equal(strided.hessian_packed, plain.hessian_packed)
