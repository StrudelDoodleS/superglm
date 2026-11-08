"""Lomax numerics in natural ``(scale, shape)`` coordinates, location zero."""

from math import factorial

import numpy as np

from superglm.distributional.kernels._common import readonly
from superglm.distributional.kernels._declared import lomax as generated
from superglm.distributional.kernels._declared_adapter import (
    ScalarJetResult,
    finite_positive_exp,
    generated_density,
    log_expm1,
    narrow_windows,
    positive_parameters,
    positive_window_jets,
    probabilities,
)

PARAMETERS = generated.PARAMETERS
COORDINATES = generated.COORDINATES


def carrier(y, multiplier):
    """The full Lomax log density has no parameter-independent carrier."""
    return np.zeros(len(y))


def density_jets(y, theta, multiplier, *, order: int) -> ScalarJetResult:
    """Optimizing log density, carrier and packed partials, only through ``order``.

    At zero the log density is exactly log(shape)-log(scale). Its pure
    d-th partials are signed (d-1)!/parameter**d and mixed partials vanish.
    This avoids log(0), with no epsilon response or altered law. The ordinary
    generated path uses log ratios, so y/scale need not be representable.
    """
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    zero = y == 0
    result = generated_density(generated, np.where(zero, 1.0, y), theta, multiplier, order=order)
    valid = result.valid.copy()
    if zero.any():
        s, a = theta[zero].T
        mass = np.asarray(multiplier)[zero]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result.jets[0][zero] = mass * (np.log(a) - np.log(s))
            for d, values in enumerate(result.jets[1:], 1):
                values[zero] = 0.0
                values[zero, 0] = mass * ((-1) ** d * factorial(d - 1)) / s**d
                values[zero, -1] = mass * ((-1) ** (d - 1) * factorial(d - 1)) / a**d
        valid[zero] = True
    valid &= np.isfinite(y) & (y >= 0) & np.all(np.isfinite(theta) & (theta > 0), axis=1)
    # Powers through four and their reciprocals remain normal in this
    # admitted parameter range (2**+-800). Unlike a response appearing only
    # inside logratio, an unrestricted parameter can silently overflow an
    # intermediate power and leave a finite, incorrect zero derivative.
    # This is numerical admission, not a mathematical shape restriction.
    if order:
        valid &= np.all((theta >= 2.0**-200) & (theta <= 2.0**200), axis=1)
    for values in result.jets:
        valid &= np.isfinite(values) if values.ndim == 1 else np.all(np.isfinite(values), axis=1)
    return ScalarJetResult(result.jets, result.carrier, valid)


def _window_routes(lower, upper, theta):
    s, a = np.asarray(theta).T
    lo, hi = np.maximum(lower, 0.0), np.asarray(upper)
    # On log y, l'=1-(a+1)*expit(log(y/s)); higher log-density
    # derivatives are bounded by polynomials in that expit. a+6 bounds
    # the scale of derivatives needed by the three-point rule.
    rate = a + 6.0
    # In y, |f^(j)/f| = (a+1)...(a+j)/(s+y)^j <= ((a+6)/(s+lo))^j,
    # j<=6. Three-node Gauss relative error <= reach^6/2016000,
    # hence reach<=.01 gives <5e-19 for density; mixed jet outcomes are
    # independently tested. We do NOT claim a directional certificate for
    # the quadrature approximation; family higher directions decline it.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reach = ((hi - lo) / (s + lo)) * (a + 6.0)
    return rate, np.isfinite(hi) & (reach <= 0.01)


def window_jets(lower, upper, theta, *, order):
    rate, linear = _window_routes(lower, upper, theta)
    return positive_window_jets(
        generated,
        lower,
        upper,
        theta,
        order=order,
        rate=rate,
        linear_narrow=linear,
        density=density_jets,
    )


def analytic_window_admitted(lower, upper, theta):
    rate, linear = _window_routes(lower, upper, theta)
    return ~(linear | narrow_windows(lower, upper, rate))


def cdf(y, theta):
    theta = positive_parameters(theta)
    s, a = theta.T
    y = np.broadcast_to(np.asarray(y, dtype=float), (len(theta),))
    if np.isnan(y).any():
        raise ValueError("CDF response must not be NaN")
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        z = np.log(np.maximum(y, 0.0)) - np.log(s)
        # log(softplus(z)) ~ z below -36; relative correction < 2**-53.
        # Keeping the product with shape in log space avoids premature
        # underflow when shape rescales a tiny y/scale back into range.
        log_t = np.where(z < -36.0, z, np.log(np.logaddexp(0.0, z))) + np.log(a)
        result = -np.expm1(-np.exp(log_t))
    return readonly(result)


def quantile(p, theta):
    theta = positive_parameters(theta)
    p = probabilities(p, len(theta))
    s, a = theta.T
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        t = -np.log1p(-p) / a
    small = t < 2.0**-53
    log_loading = np.empty_like(t)
    # expm1(t)/t = 1+t/2+O(t^2); below u, replacing this factor by
    # one costs less than u. Keep log(t) before division can underflow,
    # since monetary scale can restore a representable final quantile.
    log_loading[small] = np.log(-np.log1p(-p[small])) - np.log(a[small])
    log_loading[~small] = log_expm1(t[~small])
    return finite_positive_exp(np.log(s) + log_loading, "quantile")


def positive_moment_exists(theta, r):
    if not np.isfinite(r) or r <= 0:
        raise ValueError("moment order must be finite and positive")
    return positive_parameters(theta)[:, 1] > r


def mean(theta):
    theta = positive_parameters(theta)
    finite = positive_moment_exists(theta, 1.0)
    result = np.full(len(theta), np.inf)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        values = theta[finite, 0] / (theta[finite, 1] - 1.0)
    if np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError("finite mean is unresolved in the numerical range")
    result[finite] = values
    return readonly(result)


def variance(theta):
    theta = positive_parameters(theta)
    finite = positive_moment_exists(theta, 2.0)
    s, a = theta[finite].T
    result = np.full(len(theta), np.inf)
    result[finite] = finite_positive_exp(
        2 * np.log(s) + np.log(a) - 2 * np.log(a - 1) - np.log(a - 2), "variance"
    )
    return readonly(result)


def expected_information(theta, multiplier):
    s, a = positive_parameters(theta).T
    # Complete-row information only. This never replaces an observed Hessian.
    information = np.column_stack(
        (
            np.exp(np.log(a) - 2 * np.log(s) - np.log(a + 2)),
            -np.exp(-np.log(s) - np.log(a + 1)),
            np.exp(-2 * np.log(a)),
        )
    )
    information *= np.asarray(multiplier)[:, None]
    if not np.isfinite(information).all():
        raise ValueError("Fisher information is unresolved in the numerical range")
    return readonly(information)


def expected_shortfall(p, theta):
    theta = positive_parameters(theta)
    p = probabilities(p, len(theta))
    finite = positive_moment_exists(theta, 1.0)
    s, a = theta[finite].T
    log_s = np.log(s)
    log_q = log_s + log_expm1(-np.log1p(-p[finite]) / a)
    # q + (s+q)/(a-1), evaluated without overflowing an intermediate s+q.
    log_es = np.logaddexp(log_q, np.logaddexp(log_s, log_q) - np.log(a - 1))
    result = np.full(len(theta), np.inf)
    result[finite] = finite_positive_exp(log_es, "expected shortfall")
    return readonly(result)
