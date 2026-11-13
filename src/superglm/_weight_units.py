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


def objective_weight_exponent(
    w_max, group_matrices, groups, lambda2, reml_penalties, *, response=None, offset=None
):
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
    if exponent > 0:
        # Bound centred data moments by 4*n*w_max*max(1, |X|, |y|, |offset|)^2.
        # Work with exponents: neither a large square nor its tiny weight is
        # formed first. This scan belongs only to the exceptional lift path.
        magnitude_power = _data_magnitude_power(group_matrices, response, offset)
        n = group_matrices[0].shape[0] if group_matrices else len(response)
        exponent = min(exponent, _moment_exponent_limit(w_max, n, magnitude_power))
    return max(0, exponent - (exponent & 1))


def _moment_exponent_limit(w_max, n, magnitude_power):
    """Keep 4*n*w_max*max_magnitude**2 below the moment range ceiling."""
    return 960 - math.frexp(w_max)[1] - 2 * magnitude_power - (max(n, 1) - 1).bit_length() - 2


def _data_magnitude_power(group_matrices, response, offset, *, mean=None):
    """A power bounding represented columns, without a dense full design."""
    from scipy import sparse

    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        RandomEffectGroupMatrix,
        SparseGroupMatrix,
    )

    def power(values):
        if values is None:
            return 0
        if sparse.issparse(values):
            values = values.data
        maximum = max(float(np.max(values, initial=0.0)), -float(np.min(values, initial=0.0)))
        # An unsafe unweighted factor evaluation declines the lift, retaining
        # the original solve. It need not reject an otherwise valid fit.
        return math.frexp(maximum)[1] if np.isfinite(maximum) else 2048

    largest = max(1, power(response), power(offset), power(mean))
    for matrix in group_matrices:
        if type(matrix) in (CategoricalGroupMatrix, RandomEffectGroupMatrix):
            continue
        if type(matrix) in (DenseGroupMatrix, SparseGroupMatrix):
            largest = max(largest, power(matrix.M))
        elif hasattr(matrix, "iter_dense_chunks"):
            for _, _, block in matrix.iter_dense_chunks():
                largest = max(largest, power(block))
        elif hasattr(matrix, "R_inv") and (hasattr(matrix, "B_unique") or hasattr(matrix, "B")):
            basis = matrix.B_unique if hasattr(matrix, "B_unique") else matrix.B
            # |B R|_max <= q |B|_max |R|_max; inspect stored factors only.
            largest = max(
                largest,
                power(basis) + power(matrix.R_inv) + (max(basis.shape[1], 1) - 1).bit_length(),
            )
        else:
            # The generic bounded-memory path mirrors the matrix algebra's
            # column fallback and retains no generated column or new cache.
            unit = np.zeros(matrix.shape[1])
            with np.errstate(over="ignore", invalid="ignore"):
                for column in range(matrix.shape[1]):
                    unit[column] = 1.0
                    largest = max(largest, power(matrix.matvec(unit)))
                    unit[column] = 0.0
    return largest


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
    exponent = getattr(result, "weight_exponent", 0)
    return np.ldexp(weights, exponent) if exponent else weights


def evaluation_weight_exponent(
    weights,
    result,
    *,
    group_matrices=(),
    response=None,
    offset=None,
    mean=None,
    moment_weights=None,
):
    """An evaluation's moment unit, independent of the retained fit geometry."""
    exponent = getattr(result, "weight_exponent", 0)
    if not exponent:
        return 0
    w_max = float(np.max(weights, initial=0.0))
    if w_max == 0.0 or w_max >= 2.0**-512:
        return 0
    # Never lift evaluation weights above their own normalized unit. For
    # tiny evaluation weights, apply the same centred-data headroom cap as
    # the fit, on these rows rather than on the training design.
    normalized = 1 - math.frexp(w_max)[1]
    normalized += normalized & 1
    moment_max = w_max
    if moment_weights is not None:
        working_max = float(np.max(moment_weights, initial=0.0))
        if not np.isfinite(working_max):
            return 0
        moment_max = max(moment_max, working_max)
    magnitude_power = _data_magnitude_power(group_matrices, response, offset, mean=mean)
    exponent = min(
        exponent, normalized, _moment_exponent_limit(moment_max, len(weights), magnitude_power)
    )
    return max(0, exponent - (exponent & 1))


def solver_dispersion(result):
    """The dispersion paired with a retained inverse, before root scaling."""
    phi = getattr(result, "solver_phi", None)
    return result.phi if phi is None else phi


def covariance_exponent(result):
    """An estimated dispersion cancels the unit; a known dispersion does not."""
    exponent = getattr(result, "weight_exponent", 0)
    geometry = getattr(result, "_geometry_weight_exponent", exponent)
    return geometry if getattr(result, "solver_phi", None) is None else geometry - exponent


def coefficient_covariance(inverse, result, *, known_scale=False):
    """Pair the inverse and dispersion before changing their common unit."""
    # Legacy fit_path results carry Pearson phi even for a known-scale family;
    # its fitted-family tables explicitly choose the known dispersion instead.
    scale = 1.0 if known_scale else solver_dispersion(result)
    exponent = (
        getattr(result, "_geometry_weight_exponent", getattr(result, "weight_exponent", 0))
        if known_scale
        else covariance_exponent(result)
    )
    if getattr(result, "weight_exponent", 0) and scale > 0.0:
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
        exponent = getattr(
            result, "_geometry_weight_exponent", getattr(result, "weight_exponent", 0)
        )
    else:
        scale, exponent = solver_dispersion(result), covariance_exponent(result)
    if getattr(result, "weight_exponent", 0):
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
    if getattr(result, "weight_exponent", 0):
        mantissa, powers = np.frexp(variance)
        phi_mantissa, phi_power = math.frexp(scale)
        with np.errstate(over="ignore"):
            return np.ldexp(phi_mantissa * mantissa, powers + phi_power + exponent)
    value = scale * variance
    with np.errstate(over="ignore"):
        return np.ldexp(value, exponent) if exponent else value


def dispersion_standard_deviation(weights, result):
    """sqrt(phi / w), without forming either an overflowing w lift or phi/w."""
    phi_mantissa, phi_power = math.frexp(solver_dispersion(result))
    weight_mantissa, weight_power = np.frexp(weights)
    exponent = phi_power - weight_power
    if getattr(result, "solver_phi", None) is not None:
        exponent = exponent - result.weight_exponent
    mantissa = np.divide(
        phi_mantissa,
        weight_mantissa,
        out=np.full_like(weight_mantissa, np.inf),
        where=weight_mantissa > 0.0,
    )
    odd = exponent & 1
    with np.errstate(over="ignore"):
        return np.ldexp(np.sqrt(np.ldexp(mantissa, odd)), (exponent - odd) // 2)


def unit_prior_log_density(family, y, mu, weights, result):
    """The row's density, retaining dispersion digits after scalar underflow."""
    from superglm.distributions import Gamma, Gaussian, Tweedie, prior_weight_log_density

    phi = solver_dispersion(result)
    exponent = (
        getattr(result, "weight_exponent", 0)
        if getattr(result, "solver_phi", None) is not None
        else 0
    )
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

    if not getattr(result, "weight_exponent", 0) or getattr(result, "solver_phi", None) is None:
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
    if phi > 0.0 and isinstance(family, Gaussian | Gamma | Tweedie):
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
        # Each physical w*d/(2*phi) can fit even when lifting w alone would
        # overflow. Combine normal mantissas before applying the net exponent.
        weight_mantissa, weight_power = np.frexp(mass)
        deviance_mantissa, deviance_power = np.frexp(
            family.deviance_unit(y[positive], mu[positive])
        )
        phi_mantissa, phi_power = math.frexp(phi)
        mantissa = weight_mantissa * deviance_mantissa / phi_mantissa
        exponent = weight_power + deviance_power - phi_power + result.weight_exponent - 1
        with np.errstate(over="ignore"):
            half_deviance = float(np.sum(np.ldexp(mantissa, exponent)))
        return saturated - half_deviance
    return weighted_log_likelihood(
        family,
        y,
        mu,
        weights,
        result.phi,
        weight_semantics=weight_semantics,
        report_contract=report_contract,
    )
