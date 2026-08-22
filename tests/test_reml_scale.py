"""Independent checks for family-correct REML scale profiling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import digamma

from superglm.distributions import Gamma
from superglm.reml.scale import (
    GammaScaleProfileData,
    _gamma_saturated_normalizer,
    _shape_times_log_minus_digamma,
    _trigamma_minus_inverse,
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
    profile_gaussian_reml_scale,
)


def _brute_gamma_profile(
    y: np.ndarray,
    weights: np.ndarray,
    penalized_deviance: float,
    penalty_nullity: float,
) -> tuple[float, float]:
    distribution = Gamma()

    def criterion(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        return float(
            penalized_deviance / (2.0 * phi)
            - distribution.log_likelihood(y, y, weights, phi=phi)
            - 0.5 * penalty_nullity * np.log(2.0 * np.pi * phi)
        )

    result = minimize_scalar(
        criterion,
        bounds=(-12.0, 8.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    assert result.success
    return float(np.exp(result.x)), float(result.fun)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (_shape_times_log_minus_digamma, 0.5008333250003967837377323792877688335),
        (_gamma_saturated_normalizer, 1.382813229233738027554280476585931908),
        (_trigamma_minus_inverse, 0.00005016666333357139524566846570142254),
    ],
)
def test_gamma_asymptotics_match_independent_high_precision_values(function, expected) -> None:
    """Hard-coded 100-digit reference values guard cancellation thresholds."""
    assert function(100.0) == pytest.approx(expected, rel=3.0e-15, abs=3.0e-15)


@pytest.mark.parametrize(
    "function",
    [
        _shape_times_log_minus_digamma,
        _gamma_saturated_normalizer,
        _trigamma_minus_inverse,
    ],
)
def test_gamma_asymptotic_switch_is_continuous_across_adjacent_floats(function) -> None:
    below = function(float(np.nextafter(100.0, 0.0)))
    above = function(float(np.nextafter(100.0, np.inf)))
    assert below == pytest.approx(above, rel=2.0e-13, abs=2.0e-14)


def test_gamma_scale_profile_matches_direct_wood_criterion() -> None:
    y = np.array([0.35, 0.7, 1.1, 1.8, 2.6, 3.2])
    weights = np.ones_like(y)
    expected_phi, expected_criterion = _brute_gamma_profile(y, weights, 3.7, 2.0)

    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")
    actual = profile_gamma_reml_scale(profile_data, 3.7, 2.0)

    assert actual.phi == pytest.approx(expected_phi, rel=2.0e-8)
    assert actual.criterion == pytest.approx(expected_criterion, rel=2.0e-10, abs=2.0e-10)

    step = 1.0e-5
    lower = profile_gamma_reml_scale(profile_data, 3.7 - step, 2.0)
    upper = profile_gamma_reml_scale(profile_data, 3.7 + step, 2.0)
    finite_difference = ((1.0 / upper.phi) - (1.0 / lower.phi)) / (2.0 * step)
    assert actual.inverse_phi == pytest.approx(1.0 / actual.phi, rel=2.0e-15)
    assert actual.d_inverse_phi_d_penalized_deviance == pytest.approx(
        finite_difference,
        rel=2.0e-7,
    )


def test_gamma_frequency_weights_match_expanded_rows() -> None:
    y = np.array([0.4, 0.8, 1.7, 3.1])
    weights = np.array([1.0, 3.0, 2.0, 4.0])
    repeated_y = np.repeat(y, weights.astype(int))

    weighted = profile_gamma_reml_scale(
        prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency"), 5.2, 1.0
    )
    expanded = profile_gamma_reml_scale(
        prepare_gamma_reml_scale_data(
            repeated_y, np.ones_like(repeated_y), weight_semantics="frequency"
        ),
        5.2,
        1.0,
    )

    assert weighted.phi == pytest.approx(expanded.phi, rel=2.0e-13)
    assert weighted.criterion == pytest.approx(expanded.criterion, rel=2.0e-13)


def test_gamma_scale_profile_rejects_nonpositive_effective_likelihood_size() -> None:
    with pytest.raises(ValueError, match="no finite interior optimum"):
        profile_gamma_reml_scale(
            prepare_gamma_reml_scale_data(
                np.array([1.0, 2.0]),
                np.array([0.1, 0.1]),
                weight_semantics="frequency",
            ),
            1.0,
            penalty_nullity=1.0,
        )


@pytest.mark.parametrize(
    ("sum_weight", "sum_weight_log_y"),
    [
        (np.inf, 0.0),
        (1.0, np.nan),
    ],
)
def test_gamma_scale_profile_rejects_invalid_reduced_statistics(
    sum_weight,
    sum_weight_log_y,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        GammaScaleProfileData(
            sum_weight=sum_weight,
            sum_weight_log_y=sum_weight_log_y,
        )


@pytest.mark.parametrize(
    ("y", "weights"),
    [
        (np.ones(2), np.full(2, np.finfo(np.float64).max)),
        (np.array([np.finfo(np.float64).max]), np.array([1.0e307])),
    ],
)
def test_gamma_scale_preparation_rejects_overflowed_reductions(y, weights) -> None:
    with pytest.raises(ValueError, match="finite"):
        prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")


def test_gaussian_profile_uses_frequency_weight_likelihood_size() -> None:
    profile = profile_gaussian_reml_scale(
        penalized_deviance=8.5,
        likelihood_size=17.0,
        penalty_nullity=3.0,
    )

    residual_size = 14.0
    assert profile.phi == pytest.approx(8.5 / 14.0)
    assert profile.inverse_phi == pytest.approx(14.0 / 8.5)
    assert profile.criterion == pytest.approx(
        0.5 * residual_size * (1.0 + np.log(2.0 * np.pi * 8.5 / residual_size))
    )
    assert profile.d_inverse_phi_d_penalized_deviance == pytest.approx(-14.0 / 8.5**2)


@pytest.mark.parametrize(
    ("penalized_deviance", "likelihood_size"),
    [
        (1.0e-300, 1.0e100),  # phi underflows while 1 / phi overflows
        (1.0e-309, 1.0e-309),  # phi is finite but its Dp derivative overflows
    ],
)
def test_gaussian_profile_rejects_unrepresentable_outputs(
    penalized_deviance: float,
    likelihood_size: float,
) -> None:
    with pytest.raises(FloatingPointError, match="representable"):
        profile_gaussian_reml_scale(
            penalized_deviance=penalized_deviance,
            likelihood_size=likelihood_size,
            penalty_nullity=0.0,
        )


def _gamma_shape_score(
    shape: float,
    penalized_deviance: float,
    sum_weight: float,
    penalty_nullity: float,
) -> float:
    if shape < 30.0:
        shape_log_minus_digamma = shape * (np.log(shape) - digamma(shape))
    else:
        inverse = 1.0 / shape
        shape_log_minus_digamma = (
            0.5
            + inverse / 12.0
            - inverse**3 / 120.0
            + inverse**5 / 252.0
            - inverse**7 / 240.0
            + inverse**9 / 132.0
            - 691.0 * inverse**11 / 32760.0
        )
    return float(
        0.5 * penalized_deviance * shape
        - sum_weight * shape_log_minus_digamma
        + 0.5 * penalty_nullity
    )


@pytest.mark.parametrize(
    ("penalized_deviance", "penalty_nullity", "inverse_phi_bound", "comparison"),
    [
        (1.0e-20, 0.0, np.exp(30.0), "greater"),
        (1.0, 2.0 - 1.0e-14, np.exp(-30.0), "less"),
    ],
)
def test_gamma_scale_profile_adaptively_brackets_extreme_valid_optima(
    penalized_deviance: float,
    penalty_nullity: float,
    inverse_phi_bound: float,
    comparison: str,
) -> None:
    profile_data = prepare_gamma_reml_scale_data(
        np.array([1.0]), np.array([1.0]), weight_semantics="frequency"
    )

    profile = profile_gamma_reml_scale(
        profile_data,
        penalized_deviance,
        penalty_nullity,
    )

    if comparison == "greater":
        assert profile.inverse_phi > inverse_phi_bound
    else:
        assert profile.inverse_phi < inverse_phi_bound
    score = _gamma_shape_score(
        profile.inverse_phi,
        penalized_deviance,
        profile_data.sum_weight,
        penalty_nullity,
    )
    assert score == pytest.approx(0.0, abs=2.0e-14)


def test_gamma_scale_profile_handles_curvature_below_square_underflow() -> None:
    """A valid tiny shape must not fail after the root has been found."""
    profile_data = prepare_gamma_reml_scale_data(
        np.array([1.0]), np.array([1.0]), weight_semantics="frequency"
    )

    profile = profile_gamma_reml_scale(
        profile_data,
        penalized_deviance=1.0e200,
        penalty_nullity=0.0,
    )

    assert profile.inverse_phi == pytest.approx(2.0e-200, rel=2.0e-12)
    assert profile.phi == pytest.approx(5.0e199, rel=2.0e-12)
    assert profile.d_inverse_phi_d_penalized_deviance == 0.0
    assert np.signbit(profile.d_inverse_phi_d_penalized_deviance)


def test_gamma_scale_profile_accepts_maximum_finite_numpy_deviance_without_warning() -> None:
    profile = profile_gamma_reml_scale(
        GammaScaleProfileData(sum_weight=1.0, sum_weight_log_y=0.0),
        penalized_deviance=np.float64(np.finfo(np.float64).max),
        penalty_nullity=0.0,
    )

    assert 0.0 < profile.inverse_phi < np.finfo(np.float64).tiny
    assert np.isfinite(profile.phi)
    assert np.isfinite(profile.criterion)


@pytest.mark.parametrize(
    ("sum_weight", "penalized_deviance", "expected_shape", "expected_derivative", "atol"),
    [
        (1.0, 1.0e160, 2.0e-160, -2.0e-320, 5.0e-323),
        (1.0e-200, 1.0, 2.0e-200, -2.0e-200, 1.0e-212),
    ],
)
def test_gamma_scale_profile_retains_representable_tiny_shape_derivatives(
    sum_weight: float,
    penalized_deviance: float,
    expected_shape: float,
    expected_derivative: float,
    atol: float,
) -> None:
    profile = profile_gamma_reml_scale(
        GammaScaleProfileData(sum_weight=sum_weight, sum_weight_log_y=0.0),
        penalized_deviance=penalized_deviance,
        penalty_nullity=0.0,
    )

    assert profile.inverse_phi == pytest.approx(expected_shape, rel=2.0e-12)
    assert profile.d_inverse_phi_d_penalized_deviance == pytest.approx(
        expected_derivative,
        rel=2.0e-12,
        abs=atol,
    )


@pytest.mark.parametrize("penalized_deviance", np.logspace(-18.0, 18.0, 13))
def test_gamma_scale_profile_is_stationary_across_wide_deviance_sweep(
    penalized_deviance: float,
) -> None:
    y = np.array([0.4, 0.8, 1.7, 3.1])
    weights = np.array([0.5, 1.25, 2.0, 1.75])
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="frequency")

    profile = profile_gamma_reml_scale(
        profile_data,
        penalized_deviance,
        penalty_nullity=2.0,
    )

    assert profile.phi > 0.0
    assert profile.inverse_phi > 0.0
    assert profile.phi * profile.inverse_phi == pytest.approx(1.0, rel=2.0e-15)
    assert profile.d_inverse_phi_d_penalized_deviance < 0.0
    score = _gamma_shape_score(
        profile.inverse_phi,
        penalized_deviance,
        profile_data.sum_weight,
        2.0,
    )
    assert score == pytest.approx(0.0, abs=1.0e-11)
