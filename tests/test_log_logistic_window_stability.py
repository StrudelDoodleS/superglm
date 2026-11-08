"""Independent regressions for endpoint-gap loss and rounded quadrature nodes."""

from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np
import pytest

from superglm import LogLogisticLSS
from superglm.distributional.family import bounded_observation
from superglm.distributional.kernels import log_logistic, lomax
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import UnsupportedLikelihoodContractError
from tests._declared_scalar_oracles import jets, log_density, log_window
from tests.test_declared_scalar_adapter import weights


@pytest.mark.parametrize("shape", [1e-12, 1e-16])
def test_tiny_shape_positive_window_retains_the_endpoint_gap(shape):
    theta = (1.0, shape)
    actual, valid = log_logistic.window_jets(
        np.array([0.1]), np.array([0.2]), np.array([theta]), order=2
    )
    reference = jets(
        lambda s, a: log_window("log_logistic", mp.mpf(0.1), mp.mpf(0.2), s, a), theta, 2
    )
    # The value remains available, but derivative use as a truncation
    # normalizer is deliberately declined in this small-shape domain.
    value, value_valid = log_logistic.window_jets(
        np.array([0.1]), np.array([0.2]), np.array([theta]), order=0
    )
    assert value_valid.all() and not valid.any()
    np.testing.assert_array_equal(value[0], actual[0])
    # These channels have no root. A few hundred roundings of relative
    # error cover the elementary operations without hiding a lost gap.
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(np.ravel(got), want, rtol=1024 * 2.0**-53, atol=0)


@pytest.mark.parametrize("lo", [1e-20, 1e-30])
def test_far_left_window_has_representable_log_probability_and_jets(lo):
    theta = (1.0, 12.0)
    reference = jets(
        lambda s, a: log_window("log_logistic", mp.mpf(lo), mp.mpf(2 * lo), s, a), theta, 2
    )
    actual, valid = log_logistic.window_jets(
        np.array([lo]), np.array([2 * lo]), np.array([theta]), order=2
    )
    assert valid.all()
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(np.ravel(got), want, rtol=1024 * 2.0**-53, atol=0)


def test_tiny_shape_normalized_derivatives_are_explicitly_declined():
    family = LogLogisticLSS()
    y = np.array([0.15])
    theta = (1.0, 1e-16)
    plan = family.bind_likelihood(
        y,
        weights([1]),
        bounded_observation(
            ResponseBounds.from_arrays(np.array([0.1]), np.array([0.2]), np.zeros(1, int))
        ),
    )
    reference = jets(
        lambda s, a: (
            log_density("log_logistic", mp.mpf(0.15), s, a)
            - log_window("log_logistic", mp.mpf(0.1), mp.mpf(0.2), s, a)
        ),
        theta,
        2,
    )
    out = family.evaluate_natural(y, np.array([theta]), plan)
    assert not out.valid[0]
    assert np.all(np.isfinite(out.score)) and np.all(np.isfinite(out.hessian_packed))
    # Value queries still work; no arbitrary response epsilon or shape clip.
    value = family.evaluate_natural(y, np.array([theta]), plan, derivative_order=0)
    assert value.valid[0]
    np.testing.assert_allclose(
        value.reported_log_likelihood, reference[0], rtol=1024 * 2.0**-53, atol=0
    )


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("shape", [1e10, 1e13, 1e16])
def test_adjacent_response_window_core_derivatives(scale, shape):
    lo, hi = scale, np.nextafter(scale, np.inf)
    theta = (scale, shape)
    reference = jets(
        lambda s, a: log_window("log_logistic", mp.mpf(lo), mp.mpf(hi), s, a), theta, 2
    )
    actual, valid = log_logistic.window_jets(
        np.array([lo]), np.array([hi]), np.array([theta]), order=2
    )
    assert valid.all()
    # Per-channel relative checks are essential: a single maximum over the
    # Hessian hides the mixed derivative behind its enormous scale curvature.
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(np.ravel(got), want, rtol=2048 * 2.0**-53, atol=0)


@pytest.mark.parametrize(
    "scale,shape,lo_ratio,hi_ratio",
    [
        (1e-6, 0.6, 0.1, 2.0),
        (1.0, 12.0, 1e-30, 2e-30),
        (1e6, 4.0, 1e20, 2e20),
        (1.0, 0.25, 0.9, 1.2),
        (1.0, 1e13, 1.0, np.nextafter(1.0, np.inf)),
        (1e6, 1e13, 0.999999999999, 1.000000000002),
    ],
)
def test_interval_all_natural_and_log_coordinate_partials(scale, shape, lo_ratio, hi_ratio):
    lo, hi = scale * lo_ratio, scale * hi_ratio
    theta = (scale, shape)
    actual, valid = log_logistic.window_jets(
        np.array([lo]), np.array([hi]), np.array([theta]), order=4
    )
    assert valid.all()

    def function(s, a):
        return log_window("log_logistic", mp.mpf(lo), mp.mpf(hi), s, a)

    reference = jets(function, theta, 4)
    for got, want in zip(actual, reference, strict=True):
        # These fixtures avoid exact roots; compare each channel separately.
        np.testing.assert_allclose(np.ravel(got), want, rtol=8192 * 2.0**-53, atol=0)
    log_reference = jets(
        lambda u, v: function(mp.mpf(scale) * mp.exp(u), mp.mpf(shape) * mp.exp(v)), (0, 0), 4
    )
    # (s*d/ds)^p = sum_k S(p,k)*s^k*d^k/ds^k; the Stirling
    # coefficients below are independent of the production falling-factorial
    # conversion. Bound cancellation by the absolute sum of its terms.
    stirling = ((1,), (0, 1), (0, 1, 1), (0, 1, 3, 1), (0, 1, 7, 6, 1))
    for degree in range(1, 5):
        for index, packed in enumerate(combinations_with_replacement(range(2), degree)):
            p, q = packed.count(0), packed.count(1)
            terms = [
                stirling[p][i] * stirling[q][j] * scale**i * shape**j * actual[i + j][0, j]
                for i in range(p + 1)
                for j in range(q + 1)
                if i + j and stirling[p][i] * stirling[q][j]
            ]
            expected = log_reference[degree][index]
            bound = 8192 * 2.0**-53 * (max(1, abs(expected)) + sum(abs(x) for x in terms))
            assert abs(sum(terms) - expected) <= bound


@pytest.mark.parametrize("factor", [1 - 1e-6, 1 + 1e-6])
def test_window_derivatives_on_both_sides_of_former_quadrature_switch(factor):
    scale, shape = 3.0, 2.0
    lo, hi = 2.0, 2.0 * np.exp(factor * 0.05 / (6 * shape))
    reference = jets(
        lambda s, a: log_window("log_logistic", mp.mpf(lo), mp.mpf(hi), s, a),
        (scale, shape),
        4,
    )
    actual, valid = log_logistic.window_jets(
        np.array([lo]), np.array([hi]), np.array([[scale, shape]]), order=4
    )
    assert valid[0]
    for got, want in zip(actual, reference, strict=True):
        np.testing.assert_allclose(np.ravel(got), want, rtol=8192 * 2.0**-53, atol=0)


@pytest.mark.parametrize("order,threshold", [(1, 2.0**-8), (2, 2.0**-8), (4, 0.125)])
def test_small_shape_admission_switch_is_explicit(order, threshold):
    for direction, expected in ((0.0, False), (np.inf, True)):
        a = np.nextafter(threshold, direction)
        _, valid = log_logistic.window_jets(
            np.array([0.9]), np.array([1.2]), np.array([[1.0, a]]), order=order
        )
        assert bool(valid[0]) is expected


def test_small_shape_optional_directions_decline_without_invalidating_core():
    family = LogLogisticLSS()
    y, theta = np.array([0.15]), np.array([[1.0, 0.02]])
    plan = family.bind_likelihood(
        y,
        weights([1]),
        bounded_observation(
            ResponseBounds.from_arrays(np.array([0.1]), np.array([0.2]), np.zeros(1, int))
        ),
    )
    assert family.evaluate_natural(y, theta, plan).valid[0]
    with pytest.raises(UnsupportedLikelihoodContractError, match="analytic higher"):
        family._higher(y, theta, plan, 4)


@pytest.mark.parametrize("shape", [np.nextafter(2.0**-8, np.inf), 0.02, 0.2, 1.5])
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_admitted_normalized_channels_against_independent_derivatives(shape, side):
    family = LogLogisticLSS()
    lo, hi, y, mass = 0.9, 1.2, 1.1, 3
    theta = (1.0, shape)
    plan = family.bind_likelihood(
        np.array([y]),
        weights([mass]),
        bounded_observation(
            ResponseBounds.from_arrays(np.array([lo]), np.array([hi]), np.array([side]))
        ),
    )

    def observed(s, a):
        if not side:
            return log_density("log_logistic", mp.mpf(y), s, a)
        return log_window(
            "log_logistic", mp.mpf(y if side > 0 else lo), mp.mpf(hi if side > 0 else y), s, a
        )

    def normalizer(s, a):
        return log_window("log_logistic", mp.mpf(lo), mp.mpf(hi), s, a)

    reference = jets(lambda s, a: mass * (observed(s, a) - normalizer(s, a)), theta, 2)
    numerator, denominator = jets(observed, theta, 2), jets(normalizer, theta, 2)
    actual = family.evaluate_natural(np.array([y]), np.array([theta]), plan)
    assert actual.valid[0]
    for got, want, num, den in zip(
        (actual.reported_log_likelihood, actual.score, actual.hessian_packed),
        reference,
        numerator,
        denominator,
        strict=True,
    ):
        # Subtraction of two independently evaluated jets has absolute error
        # O(u*(|numerator|+|denominator|)). This exposes natural shape
        # curvature errors near the admission boundary without demanding
        # relative accuracy at a root or treating a log value as a jet oracle.
        bound = 128 * 2.0**-53 * mass * (abs(num) + abs(den))
        bound += 2048 * 2.0**-53 * np.maximum(1.0, abs(want))
        assert np.all(abs(np.ravel(got) - want) <= bound)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("shape", [0.25, 2.0, 1e13])
def test_lomax_adjacent_body_window_is_accurate_at_rounding_scale(scale, shape):
    lo = scale / shape
    hi = np.nextafter(lo, np.inf)
    theta = (scale, shape)
    actual, valid = lomax.window_jets(np.array([lo]), np.array([hi]), np.array([theta]), order=2)
    reference = jets(lambda s, a: log_window("lomax", mp.mpf(lo), mp.mpf(hi), s, a), theta, 2)
    assert valid[0]
    for degree, (got, want) in enumerate(zip(actual, reference, strict=True)):
        units = np.array(
            [
                np.prod([theta[j] for j in indices])
                for indices in combinations_with_replacement(range(2), degree)
            ]
        )
        # Here relative node error is O(u) and density variation over the
        # window is O(u): (a+1)*span/(s+lo) ~ span/lo. Scores can be at a
        # root, so an O(u) absolute bound in parameter units is appropriate.
        bound = 2048 * 2.0**-53 * np.maximum(1.0, abs(want * units))
        assert np.all(abs((np.ravel(got) - want) * units) <= bound)
    assert not lomax.analytic_window_admitted(np.array([lo]), np.array([hi]), np.array([theta]))[0]
