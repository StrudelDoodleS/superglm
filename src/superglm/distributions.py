"""Exponential dispersion family distributions.

Each distribution provides V(mu) (variance function) and d(y, mu)
(unit deviance) needed by the PIRLS solver.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, gammaln

# ── Numerical guard constants for positive-mean families ─────────
_POSITIVE_INIT_MIN = 1e-12  # floor for initial_mean (replaces 0.1 pseudo-response)
_POSITIVE_MU_MIN = 1e-50  # clip_mu lower bound (log → eta ≈ -115)
_POSITIVE_MU_MAX = 1e50  # clip_mu upper bound (log → eta ≈ +115)
_VARIANCE_FLOOR = 1e-100  # V(mu) floor for IRLS working weights
_FLOAT64_MIN_NORMAL = np.finfo(np.float64).tiny


def _poisson_half_deviance(y: NDArray, mu: NDArray) -> NDArray:
    """Return y*log(y/mu)-y+mu, centering its near-equality remainder."""
    result = np.array(mu, dtype=np.float64, copy=True)
    positive = y > 0.0
    response, mean = y[positive], mu[positive]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        delta = (response - mean) / mean
        ratio = response / mean
        log_ratio = np.log(ratio)
        exceptional = ~np.isfinite(log_ratio)
        log_ratio[exceptional] = np.log(response[exceptional]) - np.log(mean[exceptional])
        value = response * (log_ratio - 1.0) + mean
    close = np.abs(delta) <= 0.125
    if np.any(close):
        t = delta[close]
        # (1+t)*log1p(t)-t = sum_{n>=2} (-t)^n/[n(n-1)].
        # Through n=25 the absolute tail is <= |t|^26/[650(1-|t|)],
        # below binary64 roundoff relative to this positive remainder.
        polynomial = np.zeros_like(t)
        for n in range(25, 1, -1):
            polynomial = polynomial * t + (-1.0 if n % 2 else 1.0) / (n * (n - 1))
        value[close] = (mean[close] * t) * t * polynomial
    result[positive] = value
    return result


def _poisson_log_density(y: NDArray, mu: NDArray) -> NDArray:
    """Retain the scalar mean floor and center large-count log densities."""
    with np.errstate(over="ignore", invalid="ignore"):
        density = y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1.0)
    centered = (y >= 16.0) & (mu >= 1e-300) & np.isfinite(y) & np.isfinite(mu)
    if np.any(centered):
        from superglm.distributional.kernels.gamma import _gamma_log_normalizer

        count = y[centered]
        density[centered] = (
            _gamma_log_normalizer(count)
            - np.log(count)
            - _poisson_half_deviance(count, mu[centered])
        )
    return density


def _gamma_log_density(y: NDArray, mu: NDArray, shape: NDArray | float) -> NDArray:
    """Use the existing GammaLS normalizer and centered scaled deviance."""
    from superglm.distributional.kernels.gamma import _gamma_log_normalizer, _scaled_ratio_terms

    response, mean, size = np.broadcast_arrays(y, mu, shape)
    density = np.empty_like(response, dtype=np.float64)
    stable = (
        np.isfinite(response)
        & (response > 0.0)
        & np.isfinite(mean)
        & (mean > 0.0)
        & np.isfinite(size)
        & (size > 0.0)
    )
    fallback = ~stable
    if np.any(stable):
        normalizer = _gamma_log_normalizer(size[stable]) - np.log(response[stable])
        try:
            _, _, deviance = _scaled_ratio_terms(
                response[stable], mean[stable], size[stable], derivative_order=0
            )
            density[stable] = normalizer - deviance
        except ValueError:
            # Preserve the scalar API's previous nonfinite disposition for
            # out-of-domain or unrepresentable rows; one such row must not
            # prevent stable evaluation of the other carried rows.
            for local, index in enumerate(np.flatnonzero(stable)):
                try:
                    _, _, deviance = _scaled_ratio_terms(
                        response[index : index + 1],
                        mean[index : index + 1],
                        size[index : index + 1],
                        derivative_order=0,
                    )
                    density[index] = normalizer[local] - deviance[0]
                except ValueError:
                    fallback[index] = True
    if np.any(fallback):
        a, value, location = size[fallback], response[fallback], mean[fallback]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            density[fallback] = (
                a * np.log(a * value / location) - a * value / location - np.log(value) - gammaln(a)
            )
    return density


def _standardized_residual(y: NDArray, mu: NDArray, scale: float) -> NDArray:
    with np.errstate(over="ignore", invalid="ignore"):
        residual = y - mu
        result = residual / scale
    overflowed = ~np.isfinite(residual) & np.isfinite(y) & np.isfinite(mu)
    if np.any(overflowed):
        result[overflowed] = y[overflowed] / scale - mu[overflowed] / scale
    return result


def _weighted_residual_square(
    y: NDArray, mu: NDArray, weights: NDArray, phi: float | NDArray
) -> NDArray:
    """Return w*(y-mu)^2/phi, recovering only unsafe intermediate ranges."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        residual = y - mu
        first = weights * residual
        second = first * residual
        result = second / phi
    zero = (weights == 0.0) | (y == mu)
    unsafe = ~zero & (
        ~np.isfinite(residual)
        | ~np.isfinite(first)
        | ~np.isfinite(second)
        | (np.abs(first) < _FLOAT64_MIN_NORMAL)
        | (np.abs(second) < _FLOAT64_MIN_NORMAL)
    )
    if np.any(unsafe):
        from superglm.distributional.kernels.gamma import _binary_product_divide

        denominator = np.broadcast_to(phi, result.shape)
        for index in np.flatnonzero(unsafe):
            difference = float(residual[index])
            if np.isfinite(difference):
                factors = (float(weights[index]), difference, difference)
            else:
                # An overflowing difference of finite inputs has opposite
                # signs, so this scaled subtraction has no cancellation.
                scale = max(abs(float(y[index])), abs(float(mu[index])))
                difference = float(y[index]) / scale - float(mu[index]) / scale
                factors = (float(weights[index]), difference, difference, scale, scale)
            try:
                result[index] = _binary_product_divide(factors, (float(denominator[index]),))
            except ValueError:
                # Positive finite inputs reached the binary primitive: a
                # final overflow is a truly unrepresentable quadratic.
                result[index] = np.inf
    result[zero] = 0.0
    return result


@runtime_checkable
class Distribution(Protocol):
    """Protocol for exponential dispersion family distributions.

    Required: scale_known, default_link, variance, deviance_unit,
    log_likelihood.

    Optional: variance_derivative (V'(μ), used by REML W(ρ) correction;
    if absent, the correction is skipped for custom distribution objects).
    variance_second_derivative (V''(μ), used by second-order W(ρ)
    correction; Wood 2011, Appendix D).
    """

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion parameter φ is known (True) or estimated (False)."""
        ...

    @property
    def default_link(self) -> str:
        """Name of the canonical/default link function."""
        ...

    def variance(self, mu: NDArray) -> NDArray:
        """V(mu) — variance as a function of the mean."""
        ...

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Per-observation unit deviance d(y, mu)."""
        ...

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Sum of weighted log-likelihood contributions."""
        ...


class Poisson:
    """Poisson distribution. V(mu) = mu."""

    @property
    def scale_known(self) -> bool:
        return True

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ."""
        return mu.copy()

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 1."""
        return np.ones_like(mu)

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 0. Wood (2011) Appendix D."""
        return np.zeros_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Unit deviance: 2[y log(y/μ) - (y - μ)]."""
        return 2.0 * _poisson_half_deviance(y, mu)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Poisson log-likelihood (φ fixed at 1)."""
        carried = weights != 0.0
        return float(np.sum(weights[carried] * _poisson_log_density(y[carried], mu[carried])))


class Gaussian:
    """Gaussian distribution. V(mu) = 1."""

    @property
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "identity"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = 1."""
        return np.ones_like(mu)

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 0."""
        return np.zeros_like(mu)

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 0. Wood (2011) Appendix D."""
        return np.zeros_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Gaussian unit deviance: (y - μ)^2."""
        return (y - mu) ** 2

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Gaussian log-likelihood with dispersion φ = σ²."""
        phi_safe = max(phi, 1e-300)
        carried = weights != 0.0
        response, mean, mass = y[carried], mu[carried], weights[carried]
        residual = _standardized_residual(response, mean, np.sqrt(phi_safe))
        normalizer = np.log(2 * np.pi) + np.log(phi_safe)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            square = residual * residual
            ll = mass * (-0.5 * (normalizer + square))
        unsafe = ~np.isfinite(square) | ((square < _FLOAT64_MIN_NORMAL) & (response != mean))
        if np.any(unsafe):
            quadratic = _weighted_residual_square(
                response[unsafe], mean[unsafe], mass[unsafe], phi_safe
            )
            ll[unsafe] = (-0.5 * mass[unsafe]) * normalizer - 0.5 * quadratic
        return float(np.sum(ll))


class Gamma:
    """Gamma distribution. V(mu) = mu^2."""

    @property
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ²."""
        return mu**2

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 2μ."""
        return 2.0 * mu

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 2. Wood (2011) Appendix D."""
        return 2.0 * np.ones_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Unit deviance: 2[-log(y/μ) + (y - μ)/μ]."""
        from superglm.distributional.kernels.gamma import _vector_deviance_from_t

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            delta = (y - mu) / mu
            log_ratio = np.log(y / mu)
            exceptional = ~np.isfinite(log_ratio) & (y > 0.0) & (mu > 0.0)
            log_ratio[exceptional] = np.log(y[exceptional]) - np.log(mu[exceptional])
            deviance = delta - log_ratio
        close = np.abs(delta) <= 0.125
        if np.any(close):
            deviance[close] = _vector_deviance_from_t(delta[close])
        return 2.0 * deviance

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Gamma log-likelihood. Shape k = 1/φ."""
        k = 1.0 / phi
        carried = weights != 0.0
        return float(np.sum(weights[carried] * _gamma_log_density(y[carried], mu[carried], k)))


def _log1p_ratio(numerator: NDArray, denominator: NDArray) -> NDArray:
    """Return log(1 + numerator / denominator) without an overflowing ratio."""
    top, bottom = np.broadcast_arrays(numerator, denominator)
    small = top <= bottom
    result = np.empty_like(top, dtype=np.float64)
    result[small] = np.log1p(top[small] / bottom[small])
    result[~small] = (
        np.log(top[~small]) - np.log(bottom[~small]) + np.log1p(bottom[~small] / top[~small])
    )
    return result


def _negative_binomial_log_density(y: NDArray, mu: NDArray, theta: NDArray | float) -> NDArray:
    """Finite-theta NB density, including the existing fractional-count extension.

    The beta-function identity avoids subtracting two large log Gamma values.
    Near the Poisson limit, bounded integer counts and mu <= theta use the
    Gamma recurrence as a finite sum of log1p(j/theta). This removes the
    cancelling y*log(theta) terms before rounding. The size/count gates
    select the numerical algorithm, not support or the probability law;
    ordinary sizes retain the vector beta-function evaluation.
    """
    count, mean, size = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
    )
    mean_is_smaller = mean <= size
    log_mean = np.log(mean)
    log_mean_ratio = _log1p_ratio(mean, size)
    log_count_probability = -_log1p_ratio(size, mean)
    density = np.empty_like(mean)
    ratio = mean[mean_is_smaller] / size[mean_is_smaller]
    log_ratio_over_ratio = np.divide(
        log_mean_ratio[mean_is_smaller],
        ratio,
        out=np.ones_like(ratio),
        where=ratio > 0.0,
    )
    # size*log1p(mean/size) = mean*log1p(r)/r. Its r=0 limiting
    # factor is one even when the represented ratio itself underflows.
    density[mean_is_smaller] = -mean[mean_is_smaller] * log_ratio_over_ratio
    density[~mean_is_smaller] = -size[~mean_is_smaller] * log_mean_ratio[~mean_is_smaller]

    positive = count > 0.0
    recurrence = (
        positive
        & mean_is_smaller
        & (size >= 1.0 / np.sqrt(np.finfo(np.float64).eps))
        & (count <= 64.0)
        & (count == np.rint(count))
    )
    ordinary = positive & ~recurrence
    if np.any(ordinary):
        density[ordinary] += (
            -betaln(size[ordinary], count[ordinary])
            - np.log(count[ordinary])
            + count[ordinary] * log_count_probability[ordinary]
        )
    if np.any(recurrence):
        counts = count[recurrence]
        sizes = size[recurrence]
        rising = np.zeros_like(counts)
        for step in range(1, int(np.max(counts))):
            active = counts > step
            rising[active] += _log1p_ratio(np.full(np.count_nonzero(active), step), sizes[active])
        density[recurrence] += (
            rising
            - gammaln(counts + 1.0)
            + counts * log_mean[recurrence]
            - counts * log_mean_ratio[recurrence]
        )
    return density


class NegativeBinomial:
    """Negative binomial (NB2) family with overdispersion controlled by ``theta``.

    Parameters
    ----------
    theta : float or "auto"
        Overdispersion parameter (>0). Larger theta = less overdispersion.
        As theta -> inf, approaches Poisson. Pass ``"auto"`` to estimate
        theta via profile likelihood during ``fit()``.
    """

    def __init__(self, theta: float | str):
        if theta != "auto":
            if theta <= 0:
                raise ValueError(f"NB theta must be > 0, got {theta}")
        self.theta = theta

    @property
    def scale_known(self) -> bool:
        return True  # NB2 uses theta for overdispersion, so phi stays fixed at 1

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ + μ²/θ."""
        return mu + mu**2 / self.theta

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 1 + 2μ/θ."""
        return 1.0 + 2.0 * mu / self.theta

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = 2/θ. Wood (2011) Appendix D."""
        return (2.0 / self.theta) * np.ones_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """NB2 unit deviance."""
        theta = self.theta
        d = np.where(
            y > 0,
            2
            * (
                y * np.log(np.maximum(y, 1e-300) / mu)
                - (y + theta) * np.log((y + theta) / (mu + theta))
            ),
            2 * theta * np.log((mu + theta) / theta),
        )
        return d

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """NB2 log-likelihood: Σ w[log Γ(y+θ) - log Γ(θ) - log Γ(y+1) + θ log(θ/(μ+θ)) + y log(μ/(μ+θ))]."""
        carried = np.asarray(weights) != 0.0
        ll = _negative_binomial_log_density(y[carried], mu[carried], self.theta)
        return float(np.sum(weights[carried] * ll))


class Binomial:
    """Binomial (Bernoulli) distribution. V(mu) = mu * (1 - mu).

    For use with binary y in {0, 1}.  This is a Bernoulli GLM (n_trials=1);
    sample_weight is case/frequency weight, not binomial trials.
    """

    @property
    def scale_known(self) -> bool:
        return True

    @property
    def default_link(self) -> str:
        return "logit"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μ(1 − μ)."""
        return mu * (1 - mu)

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = 1 − 2μ."""
        return 1.0 - 2.0 * mu

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = -2. Wood (2011) Appendix D."""
        return -2.0 * np.ones_like(mu)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Bernoulli unit deviance: 2[y log(y/μ) + (1-y) log((1-y)/(1-μ))]."""
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        # For Bernoulli y in {0,1}: d = -2[y·log(μ) + (1-y)·log(1-μ)]
        return -2 * (y * np.log(mu_safe) + (1 - y) * np.log(1 - mu_safe))

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Bernoulli log-likelihood."""
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        ll = y * np.log(mu_safe) + (1 - y) * np.log(1 - mu_safe)
        return float(np.sum(weights * ll))


class Tweedie:
    """Tweedie distribution. V(mu) = mu^p, with p in (1, 2).

    Parameters
    ----------
    p : float
        Power parameter. Must be in (1, 2).
        p → 1 approaches Poisson, p → 2 approaches Gamma.
    """

    def __init__(self, p: float):
        if not 1 < p < 2:
            raise ValueError(f"Tweedie p must be in (1, 2), got {p}")
        self.p = p

    @property
    def scale_known(self) -> bool:
        return False

    @property
    def default_link(self) -> str:
        return "log"

    def variance(self, mu: NDArray) -> NDArray:
        """V(μ) = μᵖ."""
        return np.power(mu, self.p)

    def variance_derivative(self, mu: NDArray) -> NDArray:
        """V'(μ) = p·μᵖ⁻¹."""
        return self.p * np.power(mu, self.p - 1)

    def variance_second_derivative(self, mu: NDArray) -> NDArray:
        """V''(μ) = p(p-1)·μᵖ⁻². Wood (2011) Appendix D."""
        return self.p * (self.p - 1) * np.power(mu, self.p - 2)

    def deviance_unit(self, y: NDArray, mu: NDArray) -> NDArray:
        """Tweedie unit deviance evaluated without close-mean cancellation."""
        from superglm.profiling.tweedie import _tweedie_positive_unit_deviance

        return _tweedie_positive_unit_deviance(y, mu, self.p)

    def log_likelihood(self, y: NDArray, mu: NDArray, weights: NDArray, phi: float = 1.0) -> float:
        """Tweedie log-likelihood via exact Wright-Bessel evaluation."""
        from superglm.profiling.tweedie import tweedie_logpdf

        logpdf = tweedie_logpdf(y, mu, phi, self.p, weights=weights)
        return float(np.sum(logpdf))


DISTRIBUTION_SHORTCUTS: dict[str, type] = {
    "poisson": Poisson,
    "gaussian": Gaussian,
    "gamma": Gamma,
    "binomial": Binomial,
}


def resolve_distribution(
    family: str | Distribution,
) -> Distribution:
    """Convert string shorthand to distribution object, or pass through.

    Parameter-free families can be specified as strings (``"poisson"``,
    ``"gaussian"``, ``"gamma"``, ``"binomial"``).  Parameterized families
    (Tweedie, NB2) must be passed as Distribution objects::

        from superglm import families
        resolve_distribution(families.tweedie(p=1.5))
    """
    if not isinstance(family, str):
        return family
    if family in DISTRIBUTION_SHORTCUTS:
        return DISTRIBUTION_SHORTCUTS[family]()
    if family in ("tweedie", "negative_binomial"):
        raise ValueError(
            f"'{family}' requires parameters.  "
            f"Use families.tweedie(p=...) or families.nb2(theta=...) instead of a string."
        )
    raise ValueError(
        f"Unknown distribution '{family}'. Use 'poisson', 'gaussian', 'gamma', 'binomial', "
        f"or pass a Distribution object (e.g. families.tweedie(p=1.5))."
    )


# ── Family-aware helpers ───────────────────────────────────────────


def validate_response(y: NDArray, family: Distribution) -> None:
    """Validate the response vector for the given family.

    Raises ValueError for invalid responses (e.g. non-binary for binomial,
    negative for Poisson/Gamma).
    """
    if not np.all(np.isfinite(y)):
        raise ValueError("Response y must contain only finite values")
    if isinstance(family, Binomial):
        bad = ~np.isin(y, [0, 1])
        if np.any(bad):
            n_bad = int(np.sum(bad))
            vals = np.unique(y[bad])[:5]
            raise ValueError(
                f"Binomial family requires y in {{0, 1}}, "
                f"but found {n_bad} invalid values (e.g. {vals})."
            )
    elif isinstance(family, Poisson):
        if np.any(y < 0.0):
            raise ValueError("Poisson family requires nonnegative y")
    elif isinstance(family, NegativeBinomial):
        if np.any(y < 0.0):
            raise ValueError("NegativeBinomial family requires nonnegative y")
    elif isinstance(family, Gamma):
        if np.any(y <= 0.0):
            raise ValueError("Gamma family requires strictly positive y")
    elif isinstance(family, Tweedie):
        if np.any(y < 0.0):
            raise ValueError("Tweedie family requires nonnegative y")

    custom_hook = getattr(family, "validate_response", None)
    hook_function = getattr(custom_hook, "__func__", custom_hook)
    if callable(custom_hook) and hook_function is not validate_response:
        custom_hook(y)


def initial_mean(y: NDArray, weights: NDArray, family: Distribution) -> float:
    """Weighted mean of y, clipped to the valid range for the family.

    For positive-response families (Poisson, Gamma, NB, Tweedie), use the raw
    weighted mean with only a small positive floor so sparse or near-separated
    fits are not biased upward by an arbitrary pseudo-response.
    For binomial, the raw weighted mean is clipped to (eps, 1-eps).
    For Gaussian, use the raw weighted mean with no positivity clipping.
    """
    response = np.asarray(y, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        products = mass * response
        total = float(np.sum(mass))
        numerator = float(np.sum(products))
        if total == 0.0:
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        y_bar = float(np.divide(numerator, total))
    unsafe_products = (mass != 0.0) & (response != 0.0) & (np.abs(products) < _FLOAT64_MIN_NORMAL)
    if not np.isfinite(total) or not np.isfinite(y_bar) or np.any(unsafe_products):
        from fractions import Fraction

        # Global rescaling can discard a small weight before its large
        # response rescues the product. Sum the original binary inputs
        # exactly on this exceptional path and round only the final ratio.
        exact_total = Fraction(0)
        exact_numerator = Fraction(0)
        for value, weight in zip(response, mass):
            exact_weight = Fraction.from_float(float(weight))
            exact_total += exact_weight
            exact_numerator += exact_weight * Fraction.from_float(float(value))
        y_bar = float(exact_numerator / exact_total)
    if isinstance(family, Binomial):
        return np.clip(y_bar, 1e-3, 1 - 1e-3)
    if isinstance(family, Gaussian):
        return y_bar
    return max(y_bar, _POSITIVE_INIT_MIN)


def clip_mu(mu: NDArray, family: Distribution) -> NDArray:
    """Clip predicted means to a valid range for the family.

    For positive-mean families, the bounds must be wide enough that the
    IRLS can converge for near-separated categorical levels.
    """
    if isinstance(family, Binomial):
        return np.clip(mu, 1e-7, 1 - 1e-7)
    if isinstance(family, Gaussian):
        return mu
    return np.clip(mu, _POSITIVE_MU_MIN, _POSITIVE_MU_MAX)


def prior_weight_log_density(
    family: Distribution,
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    phi: float,
) -> NDArray | None:
    """Return the per-row EDM prior-weight log density, or None for a custom family.

    Each shipped family is closed under the EDM prior-weight construction --
    ``w Y`` is a member of the same family at parameters scaled by ``w`` -- so
    every form below is exact rather than a quasi-likelihood, and each reduces
    to the family's own ``log_likelihood`` at ``w == 1``.  A zero-weight row
    contributes exactly zero: it carries no information, and several of the
    forms below are not finite at ``w = 0`` because their normalizer sits at a
    gamma-function pole.

    "Exact" is unqualified for Gaussian, Gamma and Tweedie, whose supports are
    continuous, and for Binomial, where ``validate_response`` pins
    ``y in {0, 1}`` and the scaled coefficient is then exactly 1.  For Poisson
    and the negative binomial it holds **on the family's lattice**, where
    ``w * y`` is a non-negative integer -- which is the canonical case here,
    ``y = count / exposure`` with ``w = exposure``.  Off the lattice
    ``validate_response`` requires only ``y >= 0``, and the ``gammaln`` factor
    is then a Gamma-function interpolation of the counting density rather than
    a density.  For Poisson the interpolated part depends only on ``(y, w)``,
    so it moves the reported log-likelihood, AIC and BIC and nothing else; for
    the negative binomial the ``Gamma(w y + w theta) / Gamma(w theta)`` factor
    is theta-dependent, so it reaches ``theta_hat`` too.

    The mean-dependent part is *identical* to the frequency form in every case:
    prior weighting scales the same sufficient statistic, which is why the two
    contracts share a score equation and a ``beta_hat``.  What differs is the
    normalizer, and -- for the estimated-scale families -- how ``phi`` enters
    it.  That is the whole of the reported log-likelihood's dependence on the
    contract, and hence of AIC's and BIC's.
    """
    w = np.asarray(weights, dtype=np.float64)
    if isinstance(family, Gaussian):
        # N(mu, phi / w): the residual arm w r^2 / (2 phi) is already the
        # frequency arm, and only the per-row normalizer moves.
        phi_safe = max(phi, 1e-300)
        carried = w > 0.0
        contribution = np.zeros_like(w)
        quadratic = _weighted_residual_square(y[carried], mu[carried], w[carried], phi_safe)
        contribution[carried] = (
            0.5 * (np.log(np.maximum(w[carried], 1e-300)) - np.log(2 * np.pi) - np.log(phi_safe))
            - 0.5 * quadratic
        )
        return contribution
    if isinstance(family, Gamma):
        # Shape w/phi, scale mu phi/w. The shape enters lgamma per row, so
        # this is genuinely a row scan rather than sum(w) times a scalar.
        carried = w > 0.0
        contribution = np.zeros_like(w)
        with np.errstate(over="ignore"):
            shape = w[carried] / max(phi, 1e-300)
        contribution[carried] = _gamma_log_density(y[carried], mu[carried], shape)
        return contribution
    if isinstance(family, Poisson):
        # w Y ~ Poisson(w mu) on the lattice w^-1 Z.
        wy = w * y
        with np.errstate(divide="ignore", invalid="ignore"):
            contribution = (
                wy * np.log(np.maximum(mu, 1e-300))
                + wy * np.log(np.maximum(w, 1e-300))
                - w * mu
                - gammaln(wy + 1.0)
            )
        # Preserve the two existing mean/weight log floors. Where neither
        # is active, the exact scaled Poisson law admits the centered form.
        centered = (w >= 1e-300) & (mu >= 1e-300) & (wy >= 16.0)
        if np.any(centered):
            contribution[centered] = _poisson_log_density(wy[centered], (w * mu)[centered])
        return np.where(w > 0.0, contribution, 0.0)
    if isinstance(family, NegativeBinomial):
        # w Y ~ NB2(w mu, w theta): the negative binomial is infinitely
        # divisible, so this extends to fractional w.
        # ``theta`` is declared ``float | str`` because the family accepts
        # "auto"; by the time a likelihood is evaluated the profile has
        # resolved it to a number.
        theta = float(family.theta)
        carried = w > 0.0
        contribution = np.zeros_like(w)
        contribution[carried] = _negative_binomial_log_density(
            w[carried] * y[carried], w[carried] * mu[carried], w[carried] * theta
        )
        return contribution
    if isinstance(family, Binomial):
        # w is the trial count and y the success proportion, which is R's
        # documented binomial convention.  On this family's own domain
        # y in {0, 1} the binomial coefficient is exactly 1, so the prior and
        # frequency forms coincide; the general expression is kept because it
        # is what makes that agreement a derivation rather than a coincidence.
        mu_safe = np.clip(mu, 1e-15, 1 - 1e-15)
        wy = w * y
        contribution = (
            gammaln(w + 1.0)
            - gammaln(wy + 1.0)
            - gammaln(w - wy + 1.0)
            + wy * np.log(mu_safe)
            + (w - wy) * np.log(1 - mu_safe)
        )
        return np.where(w > 0.0, contribution, 0.0)
    if isinstance(family, Tweedie):
        # Already the prior form: the compound-Poisson density evaluator takes
        # the weight into its own normalizer.
        from superglm.profiling.tweedie import tweedie_logpdf

        return np.asarray(tweedie_logpdf(y, mu, phi, family.p, weights=w), dtype=np.float64)
    return None


def _frequency_weight_log_likelihood(
    family: Distribution,
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    phi: float,
) -> float:
    """Return ``sum(w_i * log f(y_i; mu_i, phi))``."""
    if isinstance(family, Tweedie):
        # The family's own method applies w as a prior weight, so the
        # replication form has to be assembled from unit-weight rows.
        from superglm.profiling.tweedie import tweedie_logpdf

        w = np.asarray(weights, dtype=np.float64)
        unit = np.ones_like(w)
        logpdf = tweedie_logpdf(y, mu, phi, family.p, weights=unit)
        return float(np.sum(w * logpdf))
    return float(family.log_likelihood(y, mu, weights, phi))


def weighted_log_likelihood(
    family: Distribution,
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    phi: float = 1.0,
    *,
    weight_semantics: str,
) -> float:
    """Return the log-likelihood the declared weight contract defines.

    A custom distribution that SuperGLM does not ship owns its own weight
    contract, because only it knows its normalizer; its ``log_likelihood`` is
    used as written and a mismatch with a declared ``"prior"`` contract is
    reported rather than silently substituted.
    """
    if weight_semantics == "frequency":
        return _frequency_weight_log_likelihood(family, y, mu, weights, phi)
    if weight_semantics != "prior":
        raise ValueError(
            f"weight_semantics must be 'prior' or 'frequency', got {weight_semantics!r}",
        )
    density = prior_weight_log_density(family, y, mu, weights, phi)
    if density is not None:
        return float(np.sum(density, dtype=np.float64))
    # At unit weight there is no mismatch to warn about: the two contracts are
    # the same likelihood, so the family's own `log_likelihood` IS the prior
    # form rather than a substitute for it. Warning here fired on every
    # ordinary unweighted custom-family fit -- twice, since the null likelihood
    # is computed too -- and turned into a hard failure under `-W error`.
    # Zero-weight rows are carried through the same test because they
    # contribute nothing under either reading.
    carried = np.asarray(weights, dtype=np.float64)
    if np.all((carried == 0.0) | (carried == 1.0)):
        return float(family.log_likelihood(y, mu, weights, phi))
    import warnings

    warnings.warn(
        f"{type(family).__name__} is not a SuperGLM-shipped family, so its EDM "
        "prior-weight normalizer cannot be derived here. Its own log_likelihood "
        "is being used unchanged, which reports sum(w * log f(y; mu, phi)) -- "
        "the frequency form -- while the rest of the fit follows "
        "weight_semantics='prior'. Implement the prior-weight normalizer on the "
        "family, or fit with weight_semantics='frequency'.",
        UserWarning,
        stacklevel=2,
    )
    return float(family.log_likelihood(y, mu, weights, phi))
