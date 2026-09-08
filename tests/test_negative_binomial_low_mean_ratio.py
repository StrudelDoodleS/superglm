"""Finite NB2 arithmetic below the former mean/theta ratio boundary."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels.negative_binomial import (
    NegativeBinomialNumericalDomainError,
    NegativeBinomialPoissonBoundaryError,
    evaluate_negative_binomial_rows,
)
from superglm.distributional.smoothing.endpoint_direction import (
    _curvature_packed,
    finite_difference_curvature_direction,
    finite_difference_curvature_second_direction,
)
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.links import LogLink


def _decimal_count_oracle(count, mean, theta, weight, semantics):
    """Independent finite gamma recurrence, differentiated in natural coordinates."""
    with localcontext() as context:
        context.prec = 220
        mean, theta, weight = map(Decimal.from_float, map(float, (mean, theta, weight)))
        scale = weight if semantics == "prior" else Decimal(1)
        multiplier = Decimal(1) if semantics == "prior" else weight
        mu, size = scale * mean, scale * theta
        total = mu + size
        n = Decimal(count)
        value = sum(((1 + Decimal(j) / size).ln() for j in range(count)), Decimal(0))
        value += n * mu.ln() - (n + size) * (1 + mu / size).ln()
        score_mu = n / mu - (n + size) / total
        score_theta = sum((1 / (size + j) for j in range(count)), Decimal(0))
        score_theta += size.ln() - total.ln() + 1 - (size + n) / total
        h_mu = -n / mu**2 + (size + n) / total**2
        h_cross = (n - mu) / total**2
        h_theta = -sum((1 / (size + j) ** 2 for j in range(count)), Decimal(0))
        h_theta += 1 / size - 1 / total + (n - mu) / total**2
        score = [multiplier * scale * value for value in (score_mu, score_theta)]
        hessian = [multiplier * scale**2 * value for value in (h_mu, h_cross, h_theta)]
        log_score = [score[0] * mean, score[1] * theta]
        log_hessian = [
            hessian[0] * mean**2 + log_score[0],
            hessian[1] * mean * theta,
            hessian[2] * theta**2 + log_score[1],
        ]
        return (
            float(multiplier * value),
            np.array(score, dtype=float),
            np.array(hessian, dtype=float),
            np.array(log_score, dtype=float),
            np.array(log_hessian, dtype=float),
        )


@pytest.mark.parametrize("count", [0, 1, 7])
@pytest.mark.parametrize("mean", [0.0020498669992843285, 1.0, 2.0])
@pytest.mark.parametrize("ratio_exponent", [30, 40, 52])
@pytest.mark.parametrize("semantics,weight", [("prior", 0.25), ("frequency", 4.0)])
def test_low_mean_ratio_keeps_exact_finite_count_native_and_log_derivatives(
    count, mean, ratio_exponent, semantics, weight
):
    theta = math.ldexp(mean, ratio_exponent)
    expected_value, expected_score, expected_hessian, expected_log_score, expected_log_hessian = (
        _decimal_count_oracle(count, mean, theta, weight, semantics)
    )
    # These fixtures keep the derivative channels well conditioned. The budget
    # covers recurrence length, order-12 series evaluation, and link products.
    relative_bound = 4096 * np.finfo(float).eps * (1 + count)
    for order in (0, 1, 2):
        result = evaluate_negative_binomial_rows(
            np.array([count]),
            np.array([mean]),
            np.array([theta]),
            np.array([weight]),
            semantics,
            derivative_order=order,
        )
        np.testing.assert_allclose(
            result.optimizing_log_likelihood, [expected_value], rtol=relative_bound, atol=0.0
        )
        if order == 0:
            continue
        np.testing.assert_allclose(result.score[0], expected_score, rtol=relative_bound, atol=0.0)
        log_score = result.score[0] * [mean, theta]
        np.testing.assert_allclose(log_score, expected_log_score, rtol=relative_bound, atol=0.0)
        if order == 2:
            np.testing.assert_allclose(
                result.hessian_packed[0], expected_hessian, rtol=relative_bound, atol=0.0
            )
            hessian = result.hessian_packed[0]
            log_hessian = np.array(
                [
                    (hessian[0] * mean) * mean + log_score[0],
                    (hessian[1] * mean) * theta,
                    (hessian[2] * theta) * theta + log_score[1],
                ]
            )
            np.testing.assert_allclose(
                log_hessian, expected_log_hessian, rtol=relative_bound, atol=0.0
            )


def test_low_exposure_zero_count_direction_can_cross_former_ratio_boundary():
    mean, theta = 0.0020498669992843285, 137564.24548107304
    # This is the real-book guard row. Its finite change in theta is part of
    # an improving full-book Newton step, but the former domain refused it.
    values = []
    for size in (theta, theta * 1.01):
        evaluation = evaluate_negative_binomial_rows(
            np.zeros(1, dtype=int),
            np.array([mean]),
            np.array([size]),
            np.ones(1),
            "prior",
            derivative_order=2,
        )
        expected = _decimal_count_oracle(0, mean, size, 1.0, "prior")
        assert evaluation.score[0, 1] < 0.0
        np.testing.assert_allclose(
            evaluation.optimizing_log_likelihood,
            [expected[0]],
            rtol=32 * np.finfo(float).eps,
            atol=0.0,
        )
        values.append(evaluation.optimizing_log_likelihood[0])
    assert values[1] < values[0]


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize(
    "semantics,theta_exponent,weight_exponent",
    [
        ("prior", 450, -450),
        ("prior", -398, 450),
        ("frequency", 450, 0),
        ("frequency", 450, 52),
        ("frequency", -398, 0),
        ("frequency", -398, 52),
    ],
)
def test_low_ratio_native_channels_remain_normal_at_weight_corners(
    count, semantics, theta_exponent, weight_exponent
):
    theta, weight = math.ldexp(1.0, theta_exponent), math.ldexp(1.0, weight_exponent)
    mean = math.ldexp(theta, -52)
    expected = _decimal_count_oracle(count, mean, theta, weight, semantics)
    result = evaluate_negative_binomial_rows(
        np.array([count]),
        np.array([mean]),
        np.array([theta]),
        np.array([weight]),
        semantics,
        derivative_order=2,
    )
    # The zero-count leading theta Hessian scales as (weight/theta)*ratio**2.
    # Bounds +/-450 and ratio>=2^-52 leave its worst exponent at -1004,
    # above binary64's normal floor -1022; unbounded ratios lose this channel.
    nonzero = expected[2] != 0.0
    assert np.all(np.abs(expected[2][nonzero]) >= np.finfo(float).tiny)
    bound = 4096 * np.finfo(float).eps * (1 + count)
    np.testing.assert_allclose(result.score[0], expected[1], rtol=bound, atol=0.0)
    np.testing.assert_allclose(result.hessian_packed[0], expected[2], rtol=bound, atol=0.0)


def _decimal_zero_count_curvature(eta):
    mean, theta = (value.exp() for value in eta)
    ratio = mean / theta
    denominator = 1 + ratio
    return (
        mean / denominator**2,
        theta * ratio**2 / denominator**2,
        theta * (denominator.ln() - ratio / denominator - ratio**2 / denominator**2),
    )


@pytest.mark.parametrize("theta_multiplier", [1.01, 1.0e4])
@pytest.mark.parametrize("axis", [0, 1])
def test_low_ratio_curvature_directions_match_independent_high_order_oracle(theta_multiplier, axis):
    family = NegativeBinomialLS()
    y = np.zeros(1)
    weights = resolve_likelihood_weights(None, n_observations=1, contract=WeightContract("prior"))
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    links = (LogLink(), LogLink())
    eta = np.log([[0.0020498669992843285, 137564.24548107304 * theta_multiplier]])
    direction = np.zeros((1, 2))
    direction[:, axis] = 1.0
    step = 1.0e-3
    first = finite_difference_curvature_direction(family, y, eta, direction, links, plan, step=step)
    second = finite_difference_curvature_second_direction(
        family, y, eta, direction, direction, links, plan, step=step
    )
    with localcontext() as context:
        context.prec = 100
        center = [Decimal.from_float(value) for value in eta[0]]
        decimal_step = Decimal("1e-15")
        plus, minus = center.copy(), center.copy()
        plus[axis] += decimal_step
        minus[axis] -= decimal_step
        c, p, m = map(_decimal_zero_count_curvature, (center, plus, minus))
        expected_first = np.array([float((a - b) / (2 * decimal_step)) for a, b in zip(p, m)])
        expected_second = np.array(
            [float((a - 2 * b + c_) / decimal_step**2) for a, b, c_ in zip(p, c, m)]
        )
    # The first-difference receipt measures truncation only. Its Richardson
    # weights have total magnitude 3/h; add the separate binary64 evaluation
    # bound (192 operations covers the zero-count row/link arithmetic).
    stencil_scale = np.max(
        np.abs(
            [
                _curvature_packed(family, y, eta + shift * direction, links, plan)[0]
                for shift in (-step, -0.5 * step, 0.0, 0.5 * step, step)
            ]
        ),
        axis=0,
    )
    gamma = 192 * np.finfo(float).eps / (1 - 192 * np.finfo(float).eps)
    rounding = 3 * gamma * stencil_scale / step
    assert np.all(np.abs(first.values[0] - expected_first) <= first.certificate[0] + rounding)
    second_rounding = (64.0 / 3.0) * gamma * stencil_scale / step**2
    assert np.all(
        np.abs(second.values[0] - expected_second) <= second.certificate[0] + second_rounding
    )


@pytest.mark.parametrize(
    "mean,theta,error",
    [
        (1.0, np.nextafter(2.0**52, np.inf), NegativeBinomialPoissonBoundaryError),
        (np.nextafter(2.0**26, np.inf), 1.0, NegativeBinomialNumericalDomainError),
        (2.0**-450, 2.0**450, NegativeBinomialPoissonBoundaryError),
        (np.nextafter(2.0**-450, 0.0), 1.0, NegativeBinomialNumericalDomainError),
        (1.0, np.nextafter(2.0**450, np.inf), NegativeBinomialNumericalDomainError),
    ],
)
def test_low_ratio_extension_keeps_bounded_underflow_and_reciprocal_domain_refusals(
    mean, theta, error
):
    with pytest.raises(error):
        evaluate_negative_binomial_rows(
            np.zeros(1, dtype=int),
            np.array([mean]),
            np.array([theta]),
            np.ones(1),
            "prior",
            derivative_order=2,
        )
