"""P-values for weighted chi-squared mixtures Q = sum(w_j * chi²(d_j)).

Uses the Imhof (1961) characteristic function inversion, integrated
via scipy.integrate.quad. For typical smooth terms (5-20 eigenvalues),
this runs in sub-millisecond time.

Satterthwaite moment-matching fallback when numerical integration fails.

References:
    Imhof, J.P. (1961). Computing the distribution of quadratic forms in
    normal variables. Biometrika, 48(3/4), 419-426.

    Davies, R.B. (1980). Algorithm AS 155: The distribution of a linear
    combination of chi-squared random variables. JRSS C, 29(3), 323-333.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def psum_chisq(
    q: float,
    weights: NDArray,
    df: NDArray | None = None,
    sigma: float = 0.0,
    lim: int = 10000,
    acc: float = 1e-4,
) -> tuple[float, int]:
    """P[sum(w_j * chi²(d_j)) + sigma * N(0,1) > q] via Imhof (1961).

    Parameters
    ----------
    q : float
        Test statistic value.
    weights : array of float
        Weights w_j in the linear combination. Must be non-zero.
    df : array of float, optional
        Degrees of freedom for each chi² term. Default is all 1.
    sigma : float
        Standard deviation of additional normal component.
    lim : int
        Maximum number of integration subdivisions for quad.
    acc : float
        Required accuracy.

    Returns
    -------
    p_value : float
        Upper tail probability P[Q > q].
    ifault : int
        0 = success, 1 = non-convergence, 4 = invalid input.
    """
    import warnings

    from scipy.integrate import IntegrationWarning, quad
    from scipy.stats import chi2 as chi2_dist
    from scipy.stats import f as f_dist
    from scipy.stats import norm

    weights = np.asarray(weights, dtype=np.float64).ravel()
    r = len(weights)
    float_info = np.finfo(np.float64)

    if df is None:
        n = np.ones(r, dtype=np.float64)
    else:
        n = np.asarray(df, dtype=np.float64).ravel()

    valid_lim = isinstance(lim, int | np.integer) and not isinstance(lim, bool) and lim >= 3
    if len(n) != r:
        return np.nan, 4
    if (
        np.isnan(q)
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(n))
        or np.any(n <= 0.0)
        or not np.isfinite(sigma)
        or sigma < 0.0
        or not valid_lim
        or not np.isfinite(acc)
        or acc <= 0.0
    ):
        return np.nan, 4
    if q == np.inf:
        return 0.0, 0
    if q == -np.inf:
        return 1.0, 0

    # Remove zero weights
    mask = weights != 0.0
    if np.any(mask):
        lb = weights[mask]
        n = n[mask]
    else:
        lb = np.empty(0, dtype=np.float64)
        n = np.empty(0, dtype=np.float64)

    if len(lb) == 0:
        if sigma > 0.0:
            return float(norm.sf(q / sigma)), 0
        return (1.0, 0) if q < 0 else (0.0, 0)

    # Respect one-sided support before invoking a numerical inversion.
    if sigma == 0.0 and np.all(lb > 0.0) and q <= 0.0:
        return 1.0, 0
    if sigma == 0.0 and np.all(lb < 0.0) and q >= 0.0:
        return 0.0, 0

    # Equal weights have an exact scaled-chi-square distribution. Besides
    # avoiding unnecessary quadrature, this supplies a certified far-tail
    # path where Fourier inversion would otherwise lose the tail to
    # cancellation against 1/2.
    if sigma == 0.0 and np.all(lb == lb[0]):
        total_df = float(np.sum(n))
        scaled_q = q / float(lb[0])
        if lb[0] > 0.0:
            return float(chi2_dist.sf(scaled_q, total_df)), 0
        return float(chi2_dist.cdf(scaled_q, total_df)), 0

    # A positive and a negative chi-square(2) term are exponentials whose
    # difference has an exact asymmetric-Laplace tail.
    opposite_signs = len(lb) == 2 and ((lb[0] > 0.0) != (lb[1] > 0.0))
    if sigma == 0.0 and opposite_signs and np.all(n == 2.0):
        positive_index = int(np.argmax(lb))
        negative_index = 1 - positive_index
        positive_weight = float(lb[positive_index])
        negative_weight = abs(float(lb[negative_index]))
        share_scale = max(positive_weight, negative_weight)
        scaled_positive = positive_weight / share_scale
        scaled_negative = negative_weight / share_scale
        positive_share = scaled_positive / (scaled_positive + scaled_negative)
        if q < 0.0:
            return (
                float(1.0 - (1.0 - positive_share) * math.exp(0.5 * (float(q) / negative_weight))),
                0,
            )
        return (
            float(positive_share * math.exp(-0.5 * (float(q) / positive_weight))),
            0,
        )

    # At the zero threshold, a difference of exactly two scaled chi-squares
    # is an F ratio. This exact route is especially important when the scales
    # are far apart: direct inversion can otherwise skip the smaller
    # component while still receiving an optimistic quadrature error.
    if sigma == 0.0 and q == 0.0 and opposite_signs:
        positive_index = int(np.argmax(lb))
        negative_index = 1 - positive_index
        positive_weight = float(lb[positive_index])
        negative_weight = abs(float(lb[negative_index]))
        positive_df = float(n[positive_index])
        negative_df = float(n[negative_index])
        log_threshold = (
            math.log(negative_weight)
            + math.log(negative_df)
            - math.log(positive_weight)
            - math.log(positive_df)
        )
        if log_threshold >= math.log(float_info.max):
            threshold = np.inf
        elif log_threshold <= math.log(np.nextafter(0.0, 1.0)):
            threshold = 0.0
        else:
            threshold = math.exp(log_threshold)
        return float(f_dist.sf(threshold, positive_df, negative_df)), 0

    sigsq = sigma * sigma
    square_safe_limit = math.sqrt(float_info.max)
    normal_decay_limit = math.sqrt(8.0 * 745.0)
    weight_scale = max(float(np.max(np.abs(lb))), sigma, float_info.tiny)
    mean = float(weight_scale * np.dot(lb / weight_scale, n))
    if not np.isfinite(mean):
        return np.nan, 1
    scaled_variance = float(2.0 * np.dot((lb / weight_scale) ** 2, n) + (sigma / weight_scale) ** 2)
    if not np.isfinite(scaled_variance) or scaled_variance <= 0.0:
        return np.nan, 1
    smallest_component_scale = min(
        float(np.min(np.abs(lb))),
        sigma if sigma > 0.0 else weight_scale,
    )
    component_scale_ratio = weight_scale / smallest_component_scale

    # Imhof (1961) formula:
    # P[Q > q] = 0.5 + (1/pi) * integral_0^inf f(u) du
    #
    # where f(u) = sin(theta(u)) / (u * rho(u))
    # theta(u) = 0.5 * sum_j [n_j * atan(lambda_j * u)] - 0.5 * q * u
    # rho(u) = prod_j [(1 + lambda_j^2 * u^2)^(n_j/4)] * exp(sigma^2 * u^2 / 8)
    #
    # The factors above use u = 2t relative to the usual characteristic-
    # function coordinate t.  Consequently the normal factor
    # exp(-sigma^2*t^2/2) contributes exp(+sigma^2*u^2/8) to rho.
    #
    # For numerical stability, compute log(rho) and use exp.

    def _phase_and_log_rho(u: float) -> tuple[float, float]:
        phase = 0.0
        log_rho = 0.0
        for lj, nj in zip(lb, n, strict=True):
            lu = float(lj) * u
            phase += 0.5 * float(nj) * math.atan(lu)
            abs_lu = abs(lu)
            if abs_lu < square_safe_limit:
                log_one_plus_square = math.log1p(lu * lu)
            else:
                log_one_plus_square = 2.0 * math.log(abs_lu)
            log_rho += 0.25 * float(nj) * log_one_plus_square

        if sigsq > 0.0:
            sigma_u = sigma * u
            if abs(sigma_u) >= normal_decay_limit:
                return phase, math.inf
            log_rho += 0.125 * sigma_u * sigma_u
        return phase, log_rho

    def _integrand(u: float) -> float:
        if u == 0.0:
            return 0.5 * (mean - q)

        phase, log_rho = _phase_and_log_rho(u)
        if log_rho > 745.0:
            return 0.0

        theta = phase - 0.5 * q * u
        return math.sin(theta) * math.exp(-log_rho) / u

    # The default public tolerance predates the far-tail contract and is too
    # loose to distinguish a small probability from quadrature noise. Work to
    # a tighter absolute tolerance, while still honouring stricter requests.
    epsilon = float_info.eps
    integration_acc = max(min(float(acc), 1e-10), 50.0 * epsilon)

    def _quad_checked(*args, **kwargs) -> tuple[float, float, bool]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            output = quad(
                *args,
                full_output=1,
                epsabs=kwargs.pop("epsabs", integration_acc),
                epsrel=integration_acc,
                **kwargs,
            )
        value, error = float(output[0]), float(output[1])
        converged = len(output) == 3 and not any(
            issubclass(item.category, IntegrationWarning) for item in caught
        )
        return value, error, converged

    try:
        # QUADPACK's Fourier driver is ill-conditioned as its frequency tends
        # to zero.  In that regime the characteristic-function amplitude
        # decays before even one q-driven oscillation, so ordinary adaptive
        # quadrature is both the stable and the natural representation.
        #
        # A single direct integral can still skip a small-weight component:
        # its transition occurs around u=1/abs(weight), potentially decades
        # after the largest component has decayed. Partitioning at every such
        # reciprocal scale makes those transitions explicit to QUADPACK.
        omega = 0.5 * abs(float(q))
        used_direct_inversion = omega <= epsilon**0.25 * weight_scale
        if used_direct_inversion:
            component_scales = np.abs(lb)
            if sigma > 0.0:
                component_scales = np.append(component_scales, sigma)
            with np.errstate(over="ignore"):
                reciprocal_scales = 1.0 / component_scales
            breakpoints = np.unique(
                reciprocal_scales[np.isfinite(reciprocal_scales) & (reciprocal_scales > 0.0)]
            )
            bounds = np.concatenate(
                [
                    np.array([0.0], dtype=np.float64),
                    breakpoints,
                    np.array([np.inf], dtype=np.float64),
                ]
            )
            component_acc = integration_acc / max(len(bounds) - 1, 1)
            result = 0.0
            abserr = 0.0
            converged = True
            for lower, upper in zip(bounds[:-1], bounds[1:], strict=True):
                value, error, interval_ok = _quad_checked(
                    _integrand,
                    float(lower),
                    float(upper),
                    limit=lim,
                    epsabs=component_acc,
                )
                result += value
                abserr += error
                converged = converged and interval_ok
        else:
            # Resolve the non-oscillatory origin separately, then let QUADPACK's
            # Fourier routines integrate the remaining cos(q*u/2) and
            # sin(q*u/2) components cycle by cycle. A single adaptive integral
            # over (0, inf) can entirely miss those cycles for large |q|.
            split = min(1.0 / weight_scale, math.pi / omega)
            component_acc = integration_acc / 3.0
            near, near_err, near_ok = _quad_checked(
                _integrand,
                0.0,
                split,
                limit=lim,
                epsabs=component_acc,
            )

            def _cos_amplitude(u: float) -> float:
                phase, log_rho = _phase_and_log_rho(u)
                if log_rho > 745.0:
                    return 0.0
                return math.sin(phase) * math.exp(-log_rho) / u

            def _sin_amplitude(u: float) -> float:
                phase, log_rho = _phase_and_log_rho(u)
                if log_rho > 745.0:
                    return 0.0
                return math.cos(phase) * math.exp(-log_rho) / u

            cos_tail, cos_err, cos_ok = _quad_checked(
                _cos_amplitude,
                split,
                np.inf,
                weight="cos",
                wvar=omega,
                limit=lim,
                limlst=lim,
                maxp1=100,
                epsabs=component_acc,
            )
            sin_tail, sin_err, sin_ok = _quad_checked(
                _sin_amplitude,
                split,
                np.inf,
                weight="sin",
                wvar=omega,
                limit=lim,
                limlst=lim,
                maxp1=100,
                epsabs=component_acc,
            )
            result = near + cos_tail - math.copysign(1.0, q) * sin_tail
            abserr = near_err + cos_err + sin_err
            converged = near_ok and cos_ok and sin_ok
    except (ArithmeticError, ValueError):
        return np.nan, 1

    raw_p = 0.5 + result / math.pi
    p_val = max(0.0, min(1.0, raw_p))
    probability_error = abserr / math.pi
    resolution = max(
        probability_error,
        64.0 * epsilon,
        4096.0 * integration_acc,
    )

    # QUADPACK's absolute error can be small even when cancellation has erased
    # either tail. Such a value is useful only with a non-convergence flag,
    # allowing callers to choose an explicit approximation. The conservative
    # floor also covers QUADPACK's optimistic error estimates near a one-sided
    # support boundary.
    resolved_tail = min(p_val, 1.0 - p_val) > 8.0 * resolution
    in_probability_range = -resolution <= raw_p <= 1.0 + resolution
    scaled_distance = (float(q) - mean) / weight_scale
    if abs(scaled_distance) >= square_safe_limit:
        cantelli_bound = 0.0
    else:
        cantelli_bound = scaled_variance / (scaled_variance + scaled_distance**2)
    if q > mean:
        moment_bound_consistent = p_val <= cantelli_bound + 8.0 * resolution
    elif q < mean:
        moment_bound_consistent = 1.0 - p_val <= cantelli_bound + 8.0 * resolution
    else:
        moment_bound_consistent = True
    one_sided_boundary_resolved = not (
        sigma == 0.0
        and (
            (np.all(lb > 0.0) and 0.0 < q < 0.01 * float(np.min(lb)))
            or (np.all(lb < 0.0) and -0.01 * float(np.min(np.abs(lb))) < q < 0.0)
        )
    )
    direct_frequency_resolved = (
        not used_direct_inversion
        or q == 0.0
        or abs(float(q)) <= math.sqrt(integration_acc) * smallest_component_scale
    )
    ifault = (
        0
        if (
            converged
            and np.isfinite(raw_p)
            and np.isfinite(abserr)
            and abserr <= 10.0 * integration_acc
            and in_probability_range
            and resolved_tail
            and moment_bound_consistent
            and component_scale_ratio <= 1.0 / math.sqrt(epsilon)
            and one_sided_boundary_resolved
            and direct_frequency_resolved
        )
        else 1
    )
    return p_val, ifault


# ── Satterthwaite fallback ───────────────────────────────────────


def satterthwaite(
    q: float,
    weights: NDArray,
    df: NDArray | None = None,
) -> tuple[float, float, float]:
    """Satterthwaite approximation: match first 2 moments to c * chi²(d).

    Parameters
    ----------
    q : float
        Test statistic.
    weights : array
        Weights of the chi² mixture.
    df : array, optional
        Degrees of freedom per term (default all 1).

    Returns
    -------
    p_value : float
        Upper tail probability under c * chi²(d).
    c : float
        Scale parameter.
    d : float
        Effective degrees of freedom.
    """
    from scipy.stats import chi2 as chi2_dist

    weights = np.asarray(weights, dtype=np.float64).ravel()
    if df is None:
        df = np.ones(len(weights), dtype=np.float64)
    else:
        df = np.asarray(df, dtype=np.float64).ravel()

    # E[Q] = sum(w_j * d_j)
    # Var[Q] = sum(2 * w_j^2 * d_j)
    mean = float(np.sum(weights * df))
    var = float(np.sum(2.0 * weights**2 * df))

    if var <= 0 or mean <= 0:
        return (1.0 if q <= 0 else 0.0), 1.0, 1.0

    # Match: c * chi²(d) has mean c*d, var 2*c²*d
    # => c = var / (2 * mean), d = 2 * mean² / var
    c = var / (2.0 * mean)
    d = 2.0 * mean**2 / var

    p_val = float(chi2_dist.sf(q / c, d))
    return p_val, c, d
