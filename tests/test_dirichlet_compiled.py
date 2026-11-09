"""Dispatch/work regressions; numerical accuracy has independent oracles below."""

import math
from fractions import Fraction

import numpy as np
import pytest

from superglm.distributional.kernels import _dirichlet_numba as compiled
from superglm.distributional.kernels import dirichlet

from ._dirichlet_oracle import mp_curvature_direction, mp_row


@pytest.mark.parametrize("k", [2, 3, 5, 17])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_rows_do_not_dispatch_scalar_python_work(monkeypatch, k, order):
    alpha = np.tile(np.linspace(0.3, 12, k), (4, 1))
    y = alpha / alpha.sum(axis=1, keepdims=True)
    logs = dirichlet.prepare_log_shares(y)
    calls = []
    original = dirichlet._row_state

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(dirichlet, "_row_state", counted)
    result = dirichlet.evaluate_dirichlet_rows(
        y, alpha, np.ones(4), derivative_order=order, log_shares=logs
    )
    assert result.valid.all()
    assert result.derivative_order == order
    assert not calls, "a whole batch must execute without Python per-row dispatch"


@pytest.mark.parametrize("double", [False, True])
def test_curvature_directions_do_not_dispatch_scalar_python_work(monkeypatch, double):
    alpha = np.array([[0.2, 2.0, 15.0], [1e12, 2e12, 3e12]])
    y = alpha / alpha.sum(axis=1, keepdims=True)
    first = np.full_like(alpha, 0.2)
    second = -first if double else None
    calls = []
    original = dirichlet._row_state

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(dirichlet, "_row_state", counted)
    original_direction = dirichlet._curvature_python

    def counted_direction(*args):
        calls.append(1)
        return original_direction(*args)

    monkeypatch.setattr(dirichlet, "_curvature_python", counted_direction)
    result = dirichlet.dirichlet_curvature_direction(
        y,
        alpha,
        np.ones(2),
        [alpha] * 4,
        first,
        second,
        log_shares=dirichlet.prepare_log_shares(y),
    )
    assert np.isfinite(result).all()
    assert not calls, "curvature directions must also leave the Python row loop"


def test_compiled_sum_keeps_cancellation_subnormals_and_half_even_rounding():
    tiny = np.nextafter(0.0, 1.0)
    cases = [
        [1.0, 2.0**-53, tiny],
        [1e-16, 1.0, 1e16],
        [2.0**900, tiny, -(2.0**900)],
        [1.0, tiny, -1.0, tiny],
        [1.0, 2.0**-53, -tiny],
    ]
    rng = np.random.default_rng(2011)
    for size in (3, 17, 64, 257):
        values = np.ldexp(rng.uniform(-1, 1, size), rng.integers(-1000, 1000, size))
        cases.append(np.r_[values, -values, 2.0**-1000])
    for values in cases:
        values = np.asarray(values, dtype=float)
        # An independent exact-input oracle belongs in tests, never production.
        expected = float(sum(map(Fraction.from_float, values), Fraction()))
        assert compiled._fsum(values) == expected
        assert compiled._fsum(values[::-1]) == expected
    assert sum(cases[0]) != float(sum(map(Fraction.from_float, cases[0]), Fraction()))


@pytest.mark.parametrize("k", [2, 7, 17])
def test_compiled_rows_against_independent_oracle_at_series_and_routing_edges(k):
    shapes = np.resize(np.array([0.25, 10.0, 0.2, 0.7, 25.0]), k)
    alphas = [
        shapes,
        np.nextafter(shapes, 0),
        np.nextafter(shapes, np.inf),
        np.linspace(1, 2, k) * 1e12,
        np.r_[2.0**64, np.full(k - 1, 2.0**-64)],
    ]
    for alpha in alphas:
        y = alpha / math.fsum(alpha)
        y[0] = np.nextafter(y[0], 1.0)
        value, score, hessian = mp_row(y, alpha)
        full = dirichlet.evaluate_dirichlet_rows(y[None], alpha[None], np.array([3.0]))
        assert full.valid[0], full.failure_reasons
        np.testing.assert_allclose(
            full.reported_log_likelihood[0], 3 * value, rtol=3e-12, atol=3e-12
        )
        np.testing.assert_allclose(full.score[0] * alpha, 3 * score * alpha, rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(full.hessian_packed[0], 3 * hessian, rtol=3e-12)
        for order in (0, 1):
            partial = dirichlet.evaluate_dirichlet_rows(
                y[None], alpha[None], np.array([3.0]), derivative_order=order
            )
            assert partial.valid[0]
            np.testing.assert_array_equal(
                partial.reported_log_likelihood, full.reported_log_likelihood
            )
        for array in (full.score, full.hessian_packed, full.reported_log_likelihood, full.valid):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flat[0] = 0


def test_arbitrary_component_count_curvature_matches_independent_derivatives():
    k = 7
    eta = np.linspace(-1.7, 3.2, k)[None]
    alpha = np.exp(eta)
    y = alpha / alpha.sum(axis=1, keepdims=True)
    first = np.linspace(-0.5, 0.3, k)[None]
    second = np.linspace(0.2, -0.7, k)[None]
    for other in (None, second):
        got = dirichlet.dirichlet_curvature_direction(
            y,
            alpha,
            np.array([3.0]),
            [alpha] * 4,
            first,
            other,
            log_shares=dirichlet.prepare_log_shares(y),
        )
        expected = mp_curvature_direction(
            y[0], eta[0], first[0], None if other is None else other[0]
        )
        np.testing.assert_allclose(got[0], 3 * expected, rtol=3e-11, atol=1e-12)


def test_routing_envelope_is_not_an_admission_limit_and_refusals_stay_row_local():
    alpha = np.array([[1e-50, 0.2, 2.0], [1e-200, 1.0, 2.0], [1e308, 1e308, 1e308]])
    y = np.tile([0.2, 0.3, 0.5], (3, 1))
    full = dirichlet.evaluate_dirichlet_rows(y, alpha, np.ones(3))
    np.testing.assert_array_equal(full.valid, [True, False, False])
    value, score, hessian = mp_row(y[0], alpha[0])
    np.testing.assert_allclose(full.reported_log_likelihood[0], value, rtol=3e-12)
    np.testing.assert_allclose(full.score[0], score, rtol=3e-12)
    np.testing.assert_allclose(full.hessian_packed[0], hessian, rtol=3e-12)
    for array in (full.score, full.hessian_packed, full.reported_log_likelihood):
        np.testing.assert_array_equal(array[1:], 0)
    assert all(full.failure_reasons[i] for i in (1, 2))
    value_only = dirichlet.evaluate_dirichlet_rows(y, alpha, np.ones(3), derivative_order=0)
    np.testing.assert_array_equal(value_only.valid, [True, True, False])


def test_entry_points_are_nopython_after_real_evaluations():
    alpha = np.array([[2.0, 3.0, 5.0]])
    y = np.array([[0.2, 0.3, 0.5]])
    dirichlet.evaluate_dirichlet_rows(y, alpha, np.ones(1))
    dirichlet.dirichlet_curvature_direction(
        y, alpha, np.ones(1), [alpha] * 4, y, y, log_shares=dirichlet.prepare_log_shares(y)
    )
    for dispatcher in (compiled._evaluate_batch, compiled._curvature_batch):
        assert dispatcher.nopython_signatures
        assert len(dispatcher.nopython_signatures) == len(dispatcher.signatures)


def test_unselected_direction_channel_keeps_intermediate_overflow_refusal():
    alpha = np.array([[2.0, 3.0, 5.0]])
    y = np.array([[0.2, 0.3, 0.5]])
    derivatives = [alpha, alpha, alpha, np.full_like(alpha, 1e308)]
    # The first-direction answer alone could be finite, but the existing
    # four-channel computation refuses overflow before multiplying by zero.
    with pytest.raises(FloatingPointError):
        dirichlet.dirichlet_curvature_direction(
            y,
            alpha,
            np.ones(1),
            derivatives,
            np.full_like(alpha, 2.0),
            log_shares=dirichlet.prepare_log_shares(y),
        )
