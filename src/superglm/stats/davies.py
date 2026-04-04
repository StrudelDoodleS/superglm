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
    df : array of int, optional
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
        0 = success, 1 = non-convergence.
    """
    import warnings

    from scipy.integrate import quad

    weights = np.asarray(weights, dtype=np.float64).ravel()
    r = len(weights)

    if r == 0:
        return (1.0, 0) if q <= 0 else (0.0, 0)

    if df is None:
        n = np.ones(r, dtype=np.float64)
    else:
        n = np.asarray(df, dtype=np.float64).ravel()

    if len(n) != r:
        return np.nan, 4

    # Remove zero weights
    mask = weights != 0.0
    if not np.any(mask):
        return (1.0, 0) if q <= 0 else (0.0, 0)
    lb = weights[mask]
    n = n[mask]
    r = len(lb)

    sigsq = sigma * sigma

    # Imhof (1961) formula:
    # P[Q > q] = 0.5 + (1/pi) * integral_0^inf f(u) du
    #
    # where f(u) = sin(theta(u)) / (u * rho(u))
    # theta(u) = 0.5 * sum_j [n_j * atan(lambda_j * u)] - 0.5 * q * u
    # rho(u) = prod_j [(1 + lambda_j^2 * u^2)^(n_j/4)] * exp(sigma^2 * u^2 / 2)
    #
    # For numerical stability, compute log(rho) and use exp.

    def _integrand(u: float) -> float:
        if u == 0.0:
            return 0.0

        theta = 0.0
        log_rho = 0.0
        for i in range(r):
            lj = lb[i]
            nj = n[i]
            lu = lj * u
            theta += 0.5 * nj * math.atan(lu)
            log_rho += 0.25 * nj * math.log(1.0 + lu * lu)

        theta -= 0.5 * q * u

        if sigsq > 0:
            log_rho += 0.5 * sigsq * u * u

        if log_rho > 500:  # avoid overflow in exp
            return 0.0

        return math.sin(theta) / (u * math.exp(log_rho))

    # Integrate from 0 to infinity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result, abserr = quad(_integrand, 0, np.inf, limit=lim, epsabs=acc, epsrel=acc)

    p_val = 0.5 + result / math.pi

    # Clamp to [0, 1]
    p_val = max(0.0, min(1.0, p_val))

    # Check convergence
    ifault = 0 if abserr < acc * 10 else 1

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

    p_val = float(1.0 - chi2_dist.cdf(q / c, d))
    return p_val, c, d
