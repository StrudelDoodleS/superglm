"""Family-correct profiling of REML dispersion terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar
from scipy.special import digamma, gammaln, i0e, i1e, polygamma

from superglm.profiling.tweedie import _P15_BESSEL_ASYMPTOTIC_MIN_ARGUMENT

_GAMMA_ASYMPTOTIC_SHAPE = 100.0

_LOG_TWO_PI = float(np.log(2.0 * np.pi))

# Initial log-phi search window for the Tweedie profile, expanded on demand up
# to the hard limit. The window is a numerical bracket, not a model bound: a
# profile optimum outside the representable window raises rather than clamps.
_TWEEDIE_LOG_PHI_WINDOW = 12.0
_TWEEDIE_LOG_PHI_STEP = 15.0
_TWEEDIE_LOG_PHI_LIMIT = 45.0
_TWEEDIE_LOG_PHI_XATOL = 1e-9
_TWEEDIE_LOG_PHI_EDGE_TOL = 1e-6
# Central-difference step (in log phi) for the profile curvature. The
# preferred arm differences the family's ANALYTIC log-phi score, so
# evaluation noise eps enters the curvature at O(eps/step); the value-form
# fallback (analytic scores unavailable on some saddlepoint rows) pays
# O(eps/step^2) and uses the smaller step to bound its truncation instead.
_TWEEDIE_SCORE_CURVATURE_STEP = 1e-3


class _ScoreUnavailableInBracketError(Exception):
    """An analytic log-phi score branch dropped out inside a polish bracket."""


_TWEEDIE_CURVATURE_STEP = 1e-4


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


@dataclass(frozen=True)
class TweedieScaleProfileData:
    """Fit-invariant state for exact Tweedie saturated log-likelihoods.

    The Tweedie saturated log-likelihood decomposes exactly over the
    zero/positive split of the response (Joergensen 1997): a zero row's
    saturated contribution is the log of an atom probability at ``mu = y = 0``,
    which is exactly ``0`` for every dispersion, while a positive row
    contributes the log-density normalizer evaluated at ``mu = y`` (its unit
    deviance vanishes there). Only the positive rows therefore vary with
    ``phi``, and their prepared density state (validation, base measure) is
    hoisted here once per fit so each profile evaluation is a single series
    pass over positive rows.
    """

    power: float
    n_positive: int
    prepared_positive: Any = field(repr=False)
    # Closed-form p = 1.5 state, or None at every other power.
    #
    # At p = 1.5 the Wright parameter a = (2-p)/(p-1) is exactly 1, and DLMF
    # 10.46.2 -- I_nu(z) = (z/2)^nu phi(1, nu+1; z^2/4) -- collapses Wright's
    # function to a modified Bessel function:
    #
    #     Phi(1, 2; t) = I_1(2 sqrt(t)) / sqrt(t),   Phi(1, 1; t) = I_0(2 sqrt(t))
    #
    # This is the case Dunn & Smyth (2005) single out as Siegel's (1979)
    # noncentral chi-squared with zero degrees of freedom, and whose Bessel form
    # their Fourier-inversion companion (2008, Table 2) uses as its reference
    # truth.  The saturated profile is simpler still: this state is prepared
    # with mu = y, so the unit deviance is identically zero and t = (K/(2 phi))^2
    # with K = 4 w sqrt(y) fit-invariant.  The whole positive-row saturated
    # log-likelihood is then
    #
    #     l_sat(phi) = C0 - n_positive log phi + sum_i log i1e(K_i / phi)
    #     T(phi)     = sum_i z_i (i0e(z_i) - i1e(z_i)) / i1e(z_i),  z = K / phi
    #
    # with C0 = sum_i (log 2 + log w_i - log(y_i)/2).  ``i1e``/``i0e`` are the
    # d->d Cephes Chebyshev evaluators, not the dd->d AMOS ``ive`` the density
    # evaluator's overflow fallback reaches for.
    bessel_scale: NDArray | None = field(default=None, repr=False, compare=False)
    bessel_log_constant: float = field(default=0.0, repr=False, compare=False)
    _saturated_cache: dict[float, float] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _saturated_score_cache: dict[float, float] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def _bessel_saturated_log_likelihood(self, phi: float) -> float | None:
        """Closed-form saturated log-likelihood at p = 1.5, or None.

        Returns None whenever the closed form is unavailable (any other power)
        or unrepresentable at this ``phi``, in which case the caller keeps the
        general Wright-Bessel route and its ``FloatingPointError`` contract.
        """
        scale = self.bessel_scale
        if scale is None:
            return None
        with np.errstate(all="ignore"):
            argument = scale / phi
            scaled_bessel_one = i1e(argument)
            log_scaled = np.log(scaled_bessel_one)
            if not np.all(np.isfinite(log_scaled)):
                return None
            value = float(
                self.bessel_log_constant
                - self.n_positive * float(np.log(phi))
                + np.sum(log_scaled, dtype=np.float64)
            )
        return value if np.isfinite(value) else None

    def _bessel_saturated_score(self, phi: float) -> tuple[float, float] | None:
        """Closed-form ``(T(phi), l_sat(phi))`` at p = 1.5, or None.

        ``T = d(-l_sat)/d log phi``.  Both come out of one ``i1e`` pass, so the
        value is returned alongside and cross-fills the value cache for free.
        The large-argument rule is the density evaluator's, verbatim, so this
        path introduces no score behaviour of its own.
        """
        scale = self.bessel_scale
        if scale is None:
            return None
        with np.errstate(all="ignore"):
            argument = scale / phi
            scaled_bessel_one = i1e(argument)
            log_scaled = np.log(scaled_bessel_one)
            if not np.all(np.isfinite(log_scaled)):
                return None
            scaled_bessel_zero = i0e(argument)
            score_component = (
                argument * (scaled_bessel_zero - scaled_bessel_one) / scaled_bessel_one
            )
            large_argument = np.isfinite(argument) & (
                argument >= _P15_BESSEL_ASYMPTOTIC_MIN_ARGUMENT
            )
            asymptotic_score = ~np.isfinite(score_component) & large_argument
            if np.any(asymptotic_score):
                inverse_z = 1.0 / argument[asymptotic_score]
                score_component[asymptotic_score] = (
                    0.5
                    + 3.0 * inverse_z / 8.0
                    + 3.0 * np.square(inverse_z) / 8.0
                    + 63.0 * np.power(inverse_z, 3) / 128.0
                )
            if not np.all(np.isfinite(score_component)):
                return None
            score = float(np.sum(score_component, dtype=np.float64))
            value = float(
                self.bessel_log_constant
                - self.n_positive * float(np.log(phi))
                + np.sum(log_scaled, dtype=np.float64)
            )
        if not np.isfinite(score):
            return None
        return score, value

    def saturated_log_likelihood(self, phi: float) -> float:
        """Exact saturated log-likelihood at dispersion ``phi``.

        Zero rows contribute exactly zero and are not evaluated; the returned
        value is the positive-row sum through the adaptive Wright-Bessel density
        evaluation the Tweedie likelihood uses everywhere else (Dunn & Smyth
        2005; Wood, Pya & Saefken 2016, supplementary App. J), or -- at p = 1.5,
        where Wright's function reduces to a modified Bessel function in closed
        form -- through that reduction.  The two agree to 4e-13 relative across
        eleven decades of phi, and mpmath at 45 digits puts the Bessel form on
        the accurate side wherever they differ.
        """
        key = float(phi)
        cached = self._saturated_cache.get(key)
        if cached is not None:
            return cached
        value = self._bessel_saturated_log_likelihood(key)
        if value is None:
            from superglm.profiling.tweedie import _evaluate_tweedie_density

            evaluation = _evaluate_tweedie_density(self.prepared_positive, key)
            value = float(np.sum(evaluation.logpdf, dtype=np.float64))
        if not np.isfinite(value):
            raise FloatingPointError(
                f"Tweedie saturated log-likelihood is not finite at phi={key:g}"
            )
        self._saturated_cache[key] = value
        return value

    def saturated_nll_log_phi_score(self, phi: float) -> float | None:
        """Analytic d(-l_sat)/d(log phi) at ``phi``, or None if unavailable.

        The density evaluator's per-row log-phi score is the closed-form
        derivative of the negative log-density (it agrees with numerical
        differentiation of the log-density to ~1e-10 relative); it is
        unavailable only when a row's evaluation lands on a branch without
        an analytic score, in which case the caller falls back to
        differencing the criterion itself.
        """
        key = float(phi)
        cached = self._saturated_score_cache.get(key)
        if cached is not None:
            return cached if np.isfinite(cached) else None
        bessel = self._bessel_saturated_score(key)
        if bessel is not None:
            score_value, saturated_value = bessel
            if np.isfinite(saturated_value):
                self._saturated_cache.setdefault(key, saturated_value)
            self._saturated_score_cache[key] = score_value
            return score_value if np.isfinite(score_value) else None
        from superglm.profiling.tweedie import _evaluate_tweedie_density

        evaluation = _evaluate_tweedie_density(
            self.prepared_positive,
            key,
            compute_score=True,
        )
        # The score pass fills ``logpdf`` too and ``compute_score`` does not
        # touch it, so the saturated VALUE at this phi is already computed and
        # would otherwise be thrown away.  brentq returns its last evaluated
        # point, so the criterion evaluation at the polished optimum lands on
        # exactly one of these keys: cross-filling turns it into a cache hit
        # for the cost of one log-sum.  Only finite values are stored, so
        # ``saturated_log_likelihood``'s FloatingPointError contract is intact.
        saturated_value = float(np.sum(evaluation.logpdf, dtype=np.float64))
        if np.isfinite(saturated_value):
            self._saturated_cache.setdefault(key, saturated_value)
        if not evaluation.score_valid or evaluation.log_phi_score is None:
            self._saturated_score_cache[key] = float("nan")
            return None
        value = float(np.sum(evaluation.log_phi_score, dtype=np.float64))
        if not np.isfinite(value):
            self._saturated_score_cache[key] = float("nan")
            return None
        self._saturated_score_cache[key] = value
        return value


def prepare_tweedie_reml_scale_data(
    y: NDArray,
    sample_weight: NDArray,
    power: float,
) -> TweedieScaleProfileData:
    """Validate rows once and hoist the phi-invariant Tweedie density state.

    ``sample_weight`` follows the Tweedie EDM prior-weight contract
    (observation-specific dispersion ``phi / w``); the prepared state applies
    it inside the density evaluation exactly as the fitted likelihood does.

    At ``power == 1.5`` the closed-form Bessel state is prepared alongside; see
    ``TweedieScaleProfileData``.  Every other power carries ``bessel_scale =
    None`` and is evaluated exactly as before.
    """
    from superglm.profiling.tweedie import _prepare_tweedie_density

    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if y.ndim != 1 or sample_weight.shape != y.shape or y.size == 0:
        raise ValueError("y and sample_weight must be one-dimensional with matching shape")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Tweedie scale profiling requires finite non-negative y")
    if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0.0):
        raise ValueError("Tweedie scale profiling requires strictly positive prior weights")
    positive = y > 0.0
    n_positive = int(np.count_nonzero(positive))
    if n_positive == 0:
        raise ValueError(
            "Tweedie scale profiling requires at least one positive response; "
            "an all-zero response has no estimable dispersion"
        )
    y_positive = y[positive]
    weights_positive = sample_weight[positive]
    prepared = _prepare_tweedie_density(
        y_positive,
        y_positive,
        float(power),
        weights=weights_positive,
    )
    bessel_scale: NDArray | None = None
    bessel_log_constant = 0.0
    if float(power) == 1.5:
        # Associated exactly as the density evaluator's own p = 1.5 branch
        # associates its Bessel argument: (4 w) * sqrt(y), then / phi.
        bessel_scale = (4.0 * weights_positive) * np.sqrt(y_positive)
        bessel_scale.setflags(write=False)
        bessel_log_constant = float(
            np.sum(
                np.log(2.0) + np.log(weights_positive) - 0.5 * np.log(y_positive),
                dtype=np.float64,
            )
        )
        if not np.isfinite(bessel_log_constant) or not np.all(np.isfinite(bessel_scale)):
            bessel_scale = None
            bessel_log_constant = 0.0
    return TweedieScaleProfileData(
        power=float(power),
        n_positive=n_positive,
        prepared_positive=prepared,
        bessel_scale=bessel_scale,
        bessel_log_constant=bessel_log_constant,
    )


def profile_tweedie_reml_scale(
    profile_data: TweedieScaleProfileData,
    penalized_deviance: float,
    penalty_nullity: float,
) -> ProfiledScaleTerm:
    """Profile Tweedie dispersion while retaining Wood's saturated likelihood.

    Minimizes the exact scale-dependent part of the Wood (2011) Eq. (4) /
    Wood, Pya & Saefken (2016) Sec. 3.3 criterion over ``log(phi)``:

        Q(phi) = Dp / (2 phi) - l_sat(phi) - (Mp / 2) log(2 pi phi)

    with ``l_sat`` the exact compound Poisson-gamma saturated log-likelihood
    (zero rows are an atom and contribute a phi-free 0; positive rows carry
    the Dunn-Smyth series normalizer). This replaces the Gaussian-shaped
    substitution ``0.5 (n - Mp) log(Dp)``, which charges every zero row a
    ``log phi`` the exact saturated likelihood does not contain and thereby
    overweights the deviance arm in proportion to the zero fraction.

    The solve is a bounded scalar minimization in ``log(phi)`` with an
    expanding bracket; the ``d(1/phi)/d(Dp)`` contract required by the outer
    REML Newton follows from implicit differentiation of the profile score,
    with the log-phi curvature measured by a central difference of ``Q``
    around the optimum (the same quantity the Gaussian and Gamma profilers
    obtain in closed form).
    """
    if not isinstance(profile_data, TweedieScaleProfileData):
        raise TypeError("profile_data must be TweedieScaleProfileData")
    penalized_deviance = float(penalized_deviance)
    penalty_nullity = float(penalty_nullity)
    if not np.isfinite(penalized_deviance) or penalized_deviance <= 0.0:
        raise ValueError("penalized_deviance must be positive and finite")
    if not np.isfinite(penalty_nullity) or penalty_nullity < 0.0:
        raise ValueError("penalty_nullity must be finite and non-negative")
    # Each positive row's saturated density decays like phi**(-1/(p-1)) as
    # phi grows (the Dunn-Smyth series is dominated by its single-event term,
    # whose weight carries phi**(-(alpha+1)) with alpha+1 = 1/(p-1); verified
    # numerically at p in {1.2, 1.5, 1.8} to 1e-6), NOT like 1/phi - assuming
    # the Gaussian-shaped 1/phi tail here is the same substitution this
    # profiler exists to remove, one level down. Q's upper-tail slope in
    # log(phi) is therefore n_positive/(p-1) - Mp/2, and a finite interior
    # optimum needs that positive.
    if 2.0 * profile_data.n_positive <= (profile_data.power - 1.0) * penalty_nullity:
        raise ValueError("Tweedie REML scale profile has no finite interior optimum")

    def criterion(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        return float(
            0.5 * penalized_deviance / phi
            - profile_data.saturated_log_likelihood(phi)
            - 0.5 * penalty_nullity * (_LOG_TWO_PI + log_phi)
        )

    log_phi_lo = -_TWEEDIE_LOG_PHI_WINDOW
    log_phi_hi = _TWEEDIE_LOG_PHI_WINDOW
    while True:
        solution = minimize_scalar(
            criterion,
            bounds=(log_phi_lo, log_phi_hi),
            method="bounded",
            options={"xatol": _TWEEDIE_LOG_PHI_XATOL},
        )
        log_phi = float(solution.x)
        if (
            log_phi - log_phi_lo < _TWEEDIE_LOG_PHI_EDGE_TOL
            and log_phi_lo > -_TWEEDIE_LOG_PHI_LIMIT
        ):
            log_phi_lo = max(log_phi_lo - _TWEEDIE_LOG_PHI_STEP, -_TWEEDIE_LOG_PHI_LIMIT)
            continue
        if log_phi_hi - log_phi < _TWEEDIE_LOG_PHI_EDGE_TOL and log_phi_hi < _TWEEDIE_LOG_PHI_LIMIT:
            log_phi_hi = min(log_phi_hi + _TWEEDIE_LOG_PHI_STEP, _TWEEDIE_LOG_PHI_LIMIT)
            continue
        break
    if (
        log_phi - log_phi_lo < _TWEEDIE_LOG_PHI_EDGE_TOL
        or log_phi_hi - log_phi < _TWEEDIE_LOG_PHI_EDGE_TOL
    ):
        raise FloatingPointError("Tweedie REML scale profile is not representable")

    # Polish the bounded minimizer to a root of the ANALYTIC profile score
    # S(u) = -Dp e^{-u}/2 + T(u) - Mp/2 (T = d(-l_sat)/d log phi, closed
    # form). Bounded Brent leaves O(xatol) placement freedom in WHERE inside
    # its final bracket it stops, and which side it stops on is decided by
    # late golden-section comparisons that can flip on machine-classed
    # summation rounding: measured placement scatter across trivially
    # equivalent solver configurations is ~2e-8 in log phi around a score
    # residual of 1e-13. Downstream consumers difference gradients built on
    # this optimum at O(1e-4) steps, amplifying that freedom ~2500x into
    # their comparisons. A root of the analytic score has no placement
    # freedom: cross-machine variation reduces to the score evaluation's
    # own rounding divided by the score slope. When the analytic score is
    # unavailable (saddlepoint rows without scores) the bounded minimizer
    # stands, with its documented tolerance.
    def profile_score(u: float) -> float | None:
        t_value = profile_data.saturated_nll_log_phi_score(float(np.exp(u)))
        if t_value is None:
            return None
        return -0.5 * penalized_deviance * float(np.exp(-u)) + t_value - 0.5 * penalty_nullity

    polish_window = 64.0 * _TWEEDIE_LOG_PHI_XATOL
    while polish_window <= 1e-2:
        bracket_lo = max(log_phi - polish_window, log_phi_lo)
        bracket_hi = min(log_phi + polish_window, log_phi_hi)
        score_bracket_lo = profile_score(bracket_lo)
        score_bracket_hi = profile_score(bracket_hi)
        if score_bracket_lo is None or score_bracket_hi is None:
            break
        if score_bracket_lo < 0.0 < score_bracket_hi:

            def bracketed_score(u: float) -> float:
                value = profile_score(u)
                if value is None:  # pragma: no cover - branch flip inside a tiny bracket
                    raise _ScoreUnavailableInBracketError
                return value

            try:
                log_phi = float(
                    brentq(
                        bracketed_score,
                        bracket_lo,
                        bracket_hi,
                        xtol=1e-15,
                        rtol=4.0 * np.finfo(np.float64).eps,
                        maxiter=100,
                    )
                )
            except _ScoreUnavailableInBracketError:  # pragma: no cover - see above
                pass
            break
        polish_window *= 8.0

    criterion_value = float(criterion(log_phi))
    # Profile curvature in log(phi). The deviance arm Dp/(2 phi) is analytic;
    # the saturated arm is a central difference of the family's ANALYTIC
    # log-phi score T(u) = sum d(-log f)/d(log phi), so d2(l_sat)/du2 = -T'(u)
    # and Q''(u) = Dp e^{-u}/2 + T'(u). Differencing an analytic first
    # derivative keeps evaluation noise eps at O(eps/step) in the curvature;
    # differencing the criterion value amplifies it by O(eps/step^2), which
    # is stack-sensitive (an older special-function stack's eps reached the
    # published d(1/phi)/d(Dp) at test-visible size). Truncation is
    # O(step^2 * T'''/6), a curvature relative error around 1e-6 at the 1e-3
    # step. The value-difference fallback below runs only when a saddlepoint
    # row carries no analytic score.
    score_step = _TWEEDIE_SCORE_CURVATURE_STEP
    score_hi = profile_data.saturated_nll_log_phi_score(float(np.exp(log_phi + score_step)))
    score_lo = profile_data.saturated_nll_log_phi_score(float(np.exp(log_phi - score_step)))
    if score_hi is not None and score_lo is not None:
        log_phi_curvature = 0.5 * penalized_deviance * float(np.exp(-log_phi)) + (
            score_hi - score_lo
        ) / (2.0 * score_step)
    else:
        step = _TWEEDIE_CURVATURE_STEP
        log_phi_curvature = (
            criterion(log_phi - step) - 2.0 * criterion_value + criterion(log_phi + step)
        ) / (step * step)
    if not np.isfinite(log_phi_curvature) or log_phi_curvature <= 0.0:
        raise FloatingPointError("Tweedie REML scale profile has non-positive curvature")

    phi = float(np.exp(log_phi))
    inverse_phi = float(np.exp(-log_phi))
    # At the optimum Q'(xi) = 0, so d2Q/d(log phi)2 = xi^2 * Q''(xi) with
    # xi = 1/phi, and implicit differentiation of the profile score gives
    # d(xi)/d(Dp) = -1/2 / Q''(xi). (For the Gaussian profile this reduces to
    # the closed form -(n - Mp)/Dp^2 published by profile_gaussian_reml_scale.)
    log_derivative_magnitude = float(np.log(0.5) - 2.0 * log_phi - np.log(log_phi_curvature))
    log_smallest = float(np.log(np.nextafter(0.0, 1.0)))
    if log_derivative_magnitude > float(np.log(np.finfo(np.float64).max)):
        raise FloatingPointError("Tweedie REML scale derivative is not representable")
    derivative = (
        -0.0
        if log_derivative_magnitude < log_smallest
        else float(-np.exp(log_derivative_magnitude))
    )
    if not np.isfinite(phi) or not np.isfinite(criterion_value):
        raise FloatingPointError("Tweedie REML scale profile produced a non-finite result")
    return ProfiledScaleTerm(
        phi=phi,
        inverse_phi=inverse_phi,
        criterion=criterion_value,
        d_inverse_phi_d_penalized_deviance=derivative,
    )


__all__ = [
    "GammaScaleProfileData",
    "ProfiledScaleTerm",
    "TweedieScaleProfileData",
    "prepare_gamma_reml_scale_data",
    "prepare_tweedie_reml_scale_data",
    "profile_gamma_reml_scale",
    "profile_gaussian_reml_scale",
    "profile_tweedie_reml_scale",
]
