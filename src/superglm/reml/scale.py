"""Family-correct profiling of REML dispersion terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, polygamma

_GAMMA_ASYMPTOTIC_SHAPE = 100.0


@dataclass(frozen=True)
class ProfiledScaleTerm:
    """Minimized scale-dependent part of Wood's REML criterion."""

    phi: float
    inverse_phi: float
    criterion: float
    d_inverse_phi_d_penalized_deviance: float


@dataclass(frozen=True)
class GammaScaleProfileData:
    """Fit-invariant sufficient statistics for saturated Gamma likelihoods."""

    sum_weight: float
    sum_weight_log_y: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.sum_weight, self.sum_weight_log_y],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or self.sum_weight <= 0.0:
            raise ValueError(
                "Gamma scale sufficient statistics must be finite with positive weight"
            )
        object.__setattr__(self, "sum_weight", float(self.sum_weight))
        object.__setattr__(self, "sum_weight_log_y", float(self.sum_weight_log_y))


def prepare_gamma_reml_scale_data(
    y: NDArray,
    sample_weight: NDArray,
) -> GammaScaleProfileData:
    """Validate rows once and reduce them to Gamma saturated-likelihood statistics."""
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if y.ndim != 1 or sample_weight.shape != y.shape:
        raise ValueError("y and sample_weight must be one-dimensional with matching shape")
    if (
        not np.all(np.isfinite(y))
        or np.any(y <= 0.0)
        or not np.all(np.isfinite(sample_weight))
        or np.any(sample_weight < 0.0)
    ):
        raise ValueError("Gamma scale profiling requires positive y and non-negative weights")
    with np.errstate(over="ignore", invalid="ignore"):
        sum_weight = float(np.sum(sample_weight, dtype=np.float64))
        sum_weight_log_y = float(np.sum(sample_weight * np.log(y), dtype=np.float64))
    if sum_weight <= 0.0:
        raise ValueError("Gamma scale profiling requires positive total weight")
    return GammaScaleProfileData(
        sum_weight=sum_weight,
        # Elementwise reduction avoids BLAS thread-launch overhead for this
        # one-vector sufficient statistic (materially slower in measured fits).
        sum_weight_log_y=sum_weight_log_y,
    )


def profile_gaussian_reml_scale(
    penalized_deviance: float,
    likelihood_size: float,
    penalty_nullity: float,
) -> ProfiledScaleTerm:
    """Return the full closed-form Gaussian scale term from Wood's criterion."""
    penalized_deviance = float(penalized_deviance)
    likelihood_size = float(likelihood_size)
    penalty_nullity = float(penalty_nullity)
    values = np.asarray(
        [penalized_deviance, likelihood_size, penalty_nullity],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or penalized_deviance <= 0.0:
        raise ValueError("Gaussian profile inputs must be finite with positive deviance")
    residual_size = likelihood_size - penalty_nullity
    if penalty_nullity < 0.0 or residual_size <= 0.0:
        raise ValueError("Gaussian REML profile requires positive residual likelihood size")
    log_phi = float(np.log(penalized_deviance) - np.log(residual_size))
    log_float_max = float(np.log(np.finfo(np.float64).max))
    log_derivative_magnitude = float(np.log(residual_size) - 2.0 * np.log(penalized_deviance))
    if abs(log_phi) > log_float_max or log_derivative_magnitude > log_float_max:
        raise FloatingPointError("Gaussian REML scale profile is not representable")
    phi = float(np.exp(log_phi))
    inverse_phi = float(np.exp(-log_phi))
    log_smallest = float(np.log(np.nextafter(0.0, 1.0)))
    derivative = (
        -0.0
        if log_derivative_magnitude < log_smallest
        else float(-np.exp(log_derivative_magnitude))
    )
    criterion = float(0.5 * residual_size * (1.0 + np.log(2.0 * np.pi) + log_phi))
    if not np.isfinite(criterion):
        raise FloatingPointError("Gaussian REML scale criterion is not representable")
    return ProfiledScaleTerm(
        phi=phi,
        inverse_phi=inverse_phi,
        criterion=criterion,
        d_inverse_phi_d_penalized_deviance=derivative,
    )


def _log_minus_digamma(shape: float) -> float:
    """Evaluate ``log(shape) - digamma(shape)`` without large-shape cancellation."""
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(np.log(shape) - digamma(shape))
    inverse = 1.0 / shape
    inverse2 = inverse * inverse
    return float(
        0.5 * inverse
        + inverse2 / 12.0
        - inverse2 * inverse2 / 120.0
        + inverse2 * inverse2 * inverse2 / 252.0
    )


def _shape_times_log_minus_digamma(shape: float) -> float:
    """Evaluate ``shape * (log(shape) - digamma(shape))`` stably."""
    if shape < 1.0e-4:
        euler_gamma = 0.5772156649015329
        zeta_2 = np.pi**2 / 6.0
        zeta_3 = 1.2020569031595942
        zeta_4 = np.pi**4 / 90.0
        return float(
            1.0
            + shape * (np.log(shape) + euler_gamma)
            - zeta_2 * shape**2
            + zeta_3 * shape**3
            - zeta_4 * shape**4
        )
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(shape * _log_minus_digamma(shape))
    inverse = 1.0 / shape
    return float(0.5 + inverse / 12.0 - inverse**3 / 120.0 + inverse**5 / 252.0)


def _gamma_saturated_normalizer(shape: float) -> float:
    """Return ``k log(k) - k - log Gamma(k)`` stably."""
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(shape * np.log(shape) - shape - gammaln(shape))
    inverse = 1.0 / shape
    return float(
        0.5 * (np.log(shape) - np.log(2.0 * np.pi))
        - inverse / 12.0
        + inverse**3 / 360.0
        - inverse**5 / 1260.0
    )


def _trigamma_minus_inverse(shape: float) -> float:
    """Evaluate ``trigamma(shape) - 1 / shape`` stably."""
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(polygamma(1, shape) - 1.0 / shape)
    inverse = 1.0 / shape
    inverse2 = inverse * inverse
    return float(
        0.5 * inverse2
        + inverse2 * inverse / 6.0
        - inverse2 * inverse**3 / 30.0
        + inverse2 * inverse**5 / 42.0
    )


def _gamma_inverse_shape_derivative(
    shape: float,
    sum_weight: float,
    penalty_nullity: float,
) -> float:
    """Return ``d(shape) / d(Dp)`` without squaring extreme shapes.

    The profile curvature can overflow when ``shape**2`` underflows, even
    though its reciprocal derivative is representable as signed zero.  Work
    instead with ``shape**2 * curvature`` and evaluate the final ratio in log
    space.
    """
    if shape < 1.0e-4:
        zeta_2 = np.pi**2 / 6.0
        zeta_3 = 1.2020569031595942
        zeta_4 = np.pi**4 / 90.0
        scaled_curvature = float(
            sum_weight
            - 0.5 * penalty_nullity
            - sum_weight * shape
            + sum_weight * (zeta_2 * shape**2 - 2.0 * zeta_3 * shape**3 + 3.0 * zeta_4 * shape**4)
        )
    elif shape < _GAMMA_ASYMPTOTIC_SHAPE:
        scaled_curvature = float(
            sum_weight * shape**2 * _trigamma_minus_inverse(shape) - 0.5 * penalty_nullity
        )
    else:
        inverse = 1.0 / shape
        scaled_curvature = float(
            0.5 * (sum_weight - penalty_nullity)
            + sum_weight * (inverse / 6.0 - inverse**3 / 30.0 + inverse**5 / 42.0)
        )
    if not np.isfinite(scaled_curvature) or scaled_curvature <= 0.0:
        raise FloatingPointError("Gamma REML scale profile has non-positive curvature")

    log_magnitude = float(np.log(0.5) + 2.0 * np.log(shape) - np.log(scaled_curvature))
    if log_magnitude < np.log(np.nextafter(0.0, 1.0)):
        return -0.0
    if log_magnitude > np.log(np.finfo(np.float64).max):
        return float("-inf")
    return float(-np.exp(log_magnitude))


def profile_gamma_reml_scale(
    profile_data: GammaScaleProfileData,
    penalized_deviance: float,
    penalty_nullity: float,
) -> ProfiledScaleTerm:
    """Profile Gamma dispersion while retaining Wood's saturated likelihood.

    The non-Tweedie weight contract is frequency weighting, so ``sum(weights)``
    is the likelihood observation count.  The calculation uses Gamma's scalar
    sufficient statistics and therefore does not rescan rows during root finding.
    """
    if not isinstance(profile_data, GammaScaleProfileData):
        raise TypeError("profile_data must be GammaScaleProfileData")
    penalized_deviance = float(penalized_deviance)
    penalty_nullity = float(penalty_nullity)
    if not np.isfinite(penalized_deviance) or penalized_deviance <= 0.0:
        raise ValueError("penalized_deviance must be positive and finite")
    if not np.isfinite(penalty_nullity) or penalty_nullity < 0.0:
        raise ValueError("penalty_nullity must be finite and non-negative")

    sum_weight = profile_data.sum_weight
    if 2.0 * sum_weight <= penalty_nullity:
        raise ValueError("Gamma REML scale profile has no finite interior optimum")

    def shape_score(log_shape: float) -> float:
        shape = float(np.exp(log_shape))
        return float(
            0.5 * penalized_deviance * shape
            - sum_weight * _shape_times_log_minus_digamma(shape)
            + 0.5 * penalty_nullity
        )

    log_shape_lo = -30.0
    log_shape_hi = 30.0
    score_lo = shape_score(log_shape_lo)
    score_hi = shape_score(log_shape_hi)
    log_shape_step = 30.0
    log_shape_min = float(np.log(np.nextafter(0.0, 1.0)))
    log_shape_max = float(np.log(np.finfo(np.float64).max))
    while score_lo >= 0.0 and log_shape_lo > log_shape_min:
        log_shape_lo = max(log_shape_lo - log_shape_step, log_shape_min)
        score_lo = shape_score(log_shape_lo)
    while score_hi <= 0.0 and log_shape_hi < log_shape_max:
        log_shape_hi = min(log_shape_hi + log_shape_step, log_shape_max)
        score_hi = shape_score(log_shape_hi)
    if not score_lo < 0.0 or not score_hi > 0.0:
        raise ValueError("Gamma REML scale profile could not bracket a finite optimum")
    log_shape = float(
        brentq(
            shape_score,
            log_shape_lo,
            log_shape_hi,
            xtol=1.0e-12,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=100,
        )
    )
    shape = float(np.exp(log_shape))
    phi = 1.0 / shape
    saturated_log_likelihood = (
        sum_weight * _gamma_saturated_normalizer(shape) - profile_data.sum_weight_log_y
    )
    criterion = (
        0.5 * penalized_deviance * shape
        - saturated_log_likelihood
        + 0.5 * penalty_nullity * log_shape
        - 0.5 * penalty_nullity * np.log(2.0 * np.pi)
    )
    if not np.isfinite(phi) or not np.isfinite(criterion):
        raise FloatingPointError("Gamma REML scale profile produced a non-finite result")
    d_inverse_phi_d_penalized_deviance = _gamma_inverse_shape_derivative(
        shape,
        sum_weight,
        penalty_nullity,
    )
    if not np.isfinite(d_inverse_phi_d_penalized_deviance):
        raise FloatingPointError("Gamma REML scale derivative is not representable")
    return ProfiledScaleTerm(
        phi=phi,
        inverse_phi=shape,
        criterion=float(criterion),
        d_inverse_phi_d_penalized_deviance=float(d_inverse_phi_d_penalized_deviance),
    )


__all__ = [
    "GammaScaleProfileData",
    "ProfiledScaleTerm",
    "prepare_gamma_reml_scale_data",
    "profile_gamma_reml_scale",
    "profile_gaussian_reml_scale",
]
