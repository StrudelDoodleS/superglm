"""An exact common unit for an IRLS objective outside binary64's normal range.

``W`` and ``S`` are both multiplied by ``2**K`` before forming products. The
minimizer and the equilibrated rank problem are unchanged. Retained geometry
uses this solver unit; published scalar deviances and dispersion use the
caller's unit. An even ``K`` also permits an exact change of unit after a
square root, when a variance overflows but its standard error is finite.
"""

from __future__ import annotations

import math

import numpy as np


def objective_weight_exponent(w_max, group_matrices, groups, lambda2, reml_penalties):
    """Choose an even lift, leaving headroom for the unchanged penalty target.

    This is a range test, not a numerical tolerance. Ordinary fits do not
    inspect the penalty, copy arrays, or change any arithmetic. Like LAPACK's
    DLASCL, scaling precedes the products that would lose exponent range.
    """
    if w_max >= 2.0**-512:
        return 0
    from superglm.inference.covariance import _active_penalty_components

    exponent = 1 - math.frexp(w_max)[1]
    exponent += exponent & 1
    powers = []
    for _, lam, omega in _active_penalty_components(
        group_matrices, groups, groups, lambda2, reml_penalties=reml_penalties
    ):
        magnitude = 1.0 if omega is None else float(np.max(np.abs(omega), initial=0.0))
        if magnitude and lam:
            # Only penalties the objective actually uses constrain its unit.
            # A configured spline lambda is unused in an all-Numeric model.
            exponent = min(exponent, 960 - math.frexp(abs(lam))[1])
            powers.append(math.frexp(abs(lam))[1] + math.frexp(magnitude)[1])
    if powers:
        # Each product is below 2**power; allow for their sum without forming
        # a possibly subnormal lambda * Omega first.
        exponent = min(exponent, 960 - max(powers) - (len(powers) - 1).bit_length())
    return max(0, exponent - (exponent & 1))


def scale_penalties(lambda2, exponent):
    """Lift lambda before assembly, after checking its active components.

    An unused configured lambda may not fit the chosen unit; it contributes
    zero. The range chooser guarantees every nonzero component fits, with
    headroom. At K == 0 preserve object identity and all arithmetic.
    """
    if not exponent:
        return lambda2

    def lift(value):
        if value and math.frexp(abs(value))[1] + exponent > 1023:
            return 0.0
        return float(np.ldexp(value, exponent))

    if isinstance(lambda2, dict):
        return {name: lift(value) for name, value in lambda2.items()}
    return lift(lambda2)


def solver_weights(weights, result):
    """Read physical prior weights in the retained geometry's objective unit."""
    exponent = result.weight_exponent
    return np.ldexp(weights, exponent) if exponent else weights


def solver_dispersion(result):
    """The dispersion paired with a retained inverse, before root scaling."""
    return result.phi if result.solver_phi is None else result.solver_phi


def covariance_exponent(result):
    """An estimated dispersion cancels the unit; a known dispersion does not."""
    geometry = getattr(result, "_geometry_weight_exponent", result.weight_exponent)
    return geometry if result.solver_phi is None else geometry - result.weight_exponent


def coefficient_covariance(inverse, result, *, known_scale=False):
    """Pair the inverse and dispersion before changing their common unit."""
    # Legacy fit_path results carry Pearson phi even for a known-scale family;
    # its fitted-family tables explicitly choose the known dispersion instead.
    scale = 1.0 if known_scale else solver_dispersion(result)
    exponent = (
        getattr(result, "_geometry_weight_exponent", result.weight_exponent)
        if known_scale
        else covariance_exponent(result)
    )
    if result.weight_exponent and scale > 0.0:
        # Leave the even part of dispersion's exponent out of the matrix too:
        # phi' * H'^-1 can overflow while its standard errors still fit.
        scale_exponent = math.frexp(scale)[1]
        scale_exponent -= scale_exponent & 1
        scale = math.ldexp(scale, -scale_exponent)
        exponent += scale_exponent
    covariance = inverse.scaled(scale) if hasattr(inverse, "scaled") else scale * inverse
    if exponent:
        from superglm.inference.covariance import BinaryScaledCovariance

        return BinaryScaledCovariance(covariance, exponent)
    return covariance


def standard_errors(variance, result, *, raw=False, known_scale=False):
    """Change units after the square root, avoiding an overflowing variance."""
    if raw or known_scale:
        scale = 1.0
        exponent = getattr(result, "_geometry_weight_exponent", result.weight_exponent)
    else:
        scale, exponent = solver_dispersion(result), covariance_exponent(result)
    if result.weight_exponent:
        return scaled_standard_errors(variance, scale, exponent)
    errors = np.sqrt(np.maximum(scale * variance, 0.0))
    return np.ldexp(errors, exponent // 2) if exponent else errors


def scaled_standard_errors(variance, scale, exponent):
    """A root with both exponents carried past its normal mantissa product."""
    if scale > 0.0:
        scale_exponent = math.frexp(scale)[1]
        scale_exponent -= scale_exponent & 1
        scale = math.ldexp(scale, -scale_exponent)
        exponent += scale_exponent
    errors = np.sqrt(max(scale, 0.0)) * np.sqrt(np.maximum(variance, 0.0))
    with np.errstate(over="ignore"):
        return np.ldexp(errors, exponent // 2) if exponent else errors


def physical_variance(variance, result):
    """Publish variance entries; their standard errors can have greater range."""
    scale = solver_dispersion(result)
    exponent = covariance_exponent(result)
    if result.weight_exponent:
        mantissa, powers = np.frexp(variance)
        phi_mantissa, phi_power = math.frexp(scale)
        with np.errstate(over="ignore"):
            return np.ldexp(phi_mantissa * mantissa, powers + phi_power + exponent)
    value = scale * variance
    with np.errstate(over="ignore"):
        return np.ldexp(value, exponent) if exponent else value


def unit_prior_log_density(family, y, mu, weights, result):
    """The row's density, retaining dispersion digits after scalar underflow."""
    from superglm.distributions import Gamma, Gaussian, Tweedie, prior_weight_log_density

    phi = solver_dispersion(result)
    exponent = result.weight_exponent if result.solver_phi is not None else 0
    if not exponent:
        return prior_weight_log_density(family, y, mu, weights, phi)
    if phi <= 0.0:
        # Preserve the density's existing convention for an exact zero scale.
        return prior_weight_log_density(family, y, mu, weights, result.phi)
    if np.max(weights, initial=0.0) <= np.ldexp(2.0**960, -exponent):
        return prior_weight_log_density(family, y, mu, np.ldexp(weights, exponent), phi)
    # Evaluation weights can be many orders above the fitting weights. Their
    # common lift would overflow, although a log density or a root can fit.
    carried = weights > 0.0
    density = np.zeros_like(weights)
    log_phi = math.log(phi) - exponent * math.log(2.0)
    log_precision = np.log(weights[carried]) - log_phi
    response, mean = y[carried], mu[carried]
    if isinstance(family, Gaussian):
        with np.errstate(over="ignore"):
            residual = np.ldexp(
                (response - mean) * np.sqrt(weights[carried]) / np.sqrt(phi), exponent // 2
            )
            density[carried] = 0.5 * (log_precision - math.log(2 * math.pi) - residual**2)
        return density
    if isinstance(family, Gamma):
        from superglm.distributions import _gamma_log_density

        large = log_precision > 512 * math.log(2.0)
        values = np.empty_like(response)
        values[~large] = _gamma_log_density(
            response[~large], mean[~large], np.exp(log_precision[~large])
        )
        with np.errstate(divide="ignore", over="ignore"):
            half_deviance = 0.5 * family.deviance_unit(response[large], mean[large])
            values[large] = (
                0.5 * (log_precision[large] - math.log(2 * math.pi))
                - np.log(response[large])
                - np.exp(log_precision[large] + np.log(half_deviance))
            )
        density[carried] = values
        return density
    if isinstance(family, Tweedie):
        from superglm._tweedie import TweedieRows, _log_unit_deviance

        rows = TweedieRows.prepare(response, weights[carried], family.p)
        saturated = np.zeros_like(response)
        saturated[response > 0.0] = rows.row_saturated(phi, _log_phi=log_phi)[0]
        with np.errstate(over="ignore"):
            density[carried] = saturated - np.exp(
                _log_unit_deviance(response, mean, family.p) + log_precision - math.log(2.0)
            )
        return density
    return prior_weight_log_density(family, y, mu, weights, result.phi)


def unit_log_likelihood(family, y, mu, weights, result, *, weight_semantics, report_contract=True):
    """Evaluate an estimated-scale likelihood without rounding phi to zero.

    Prior densities depend on W / phi. Frequency densities keep their physical
    counts: their saturated normalizer is evaluated at log(phi), and their
    deviance ratio uses the solver unit. The ordinary path is unchanged.
    """
    from superglm.distributions import Gamma, Gaussian, Tweedie, weighted_log_likelihood

    if not result.weight_exponent or result.solver_phi is None:
        return weighted_log_likelihood(
            family,
            y,
            mu,
            weights,
            result.phi,
            weight_semantics=weight_semantics,
            report_contract=report_contract,
        )
    phi = result.solver_phi
    if weight_semantics == "prior":
        density = unit_prior_log_density(family, y, mu, weights, result)
        if density is not None:
            return float(np.sum(density))
    lifted = solver_weights(weights, result)
    if phi > 0.0 and isinstance(family, Gaussian | Gamma | Tweedie):
        from superglm._utils import weighted_row_sum

        log_phi = math.log(phi) - result.weight_exponent * math.log(2.0)
        positive = weights > 0.0
        mass = weights[positive]
        if isinstance(family, Gaussian):
            saturated = float(np.sum(mass)) * (-0.5 * (math.log(2 * math.pi) + log_phi))
        elif isinstance(family, Gamma):
            from superglm.distributional.kernels.gamma import _gamma_log_normalizer

            if log_phi < -512 * math.log(2.0):
                # Stirling's first omitted term is phi/12, far below u here.
                normalizer = -0.5 * (math.log(2 * math.pi) + log_phi)
            else:
                normalizer = float(_gamma_log_normalizer(np.array([math.exp(-log_phi)]))[0])
            saturated = float(np.sum(mass * (normalizer - np.log(y[positive]))))
        else:
            from superglm._tweedie import TweedieRows

            rows = TweedieRows.prepare(y, weights, family.p, frequency=True)
            saturated = float(rows.count @ rows.row_saturated(phi, _log_phi=log_phi)[0])
        deviance = weighted_row_sum(lifted, family.deviance_unit, y, mu)
        return saturated - deviance / (2.0 * phi)
    return weighted_log_likelihood(
        family,
        y,
        mu,
        weights,
        result.phi,
        weight_semantics=weight_semantics,
        report_contract=report_contract,
    )
