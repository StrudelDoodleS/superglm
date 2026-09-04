"""Independent references for the generalized Pareto distribution of excesses.

Nothing here shares code with the kernel.  ``textbook_log_density`` and
``naive_score_and_hessian`` are written in their naive ``1/xi`` forms on
purpose: they are the thing the kernel's series branch replaces, and one test
pins how badly they fail so the branch cannot be quietly deleted.

``fit_scale_regression`` and ``profile_shape`` are independent reference
implementations reduced to what these tests need.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize

PACKED = ((0, 0), (0, 1), (1, 1))
_PARAMETER_BOUND = 30.0


def textbook_log_density(y, psi, xi):
    """The naive ``-log psi - (1 + 1/xi) log1p(xi y / psi)``."""
    return -np.log(psi) - (1.0 + 1.0 / xi) * np.log1p(xi * y / psi)


def naive_score_and_hessian(y, psi, xi):
    """Naive ``(s_psi, s_xi)`` and ``(H_psipsi, H_psixi, H_xixi)`` with the ``1/xi`` terms kept."""
    t = y / psi
    z = xi * t
    s = 1.0 + z
    s_psi = -1.0 / psi + (1.0 + 1.0 / xi) * z / (psi * s)
    s_xi = np.log1p(z) / xi**2 - (1.0 + 1.0 / xi) * t / s
    h_psipsi = (1.0 - t * (2.0 + z)) / (psi * psi * s * s)
    h_psixi = -t * (t - 1.0) / (psi * s * s)
    h_xixi = -2.0 * np.log1p(z) / xi**3 + 2.0 * t / (xi**2 * s) + (1.0 + 1.0 / xi) * t * t / (s * s)
    return (s_psi, s_xi), (h_psipsi, h_psixi, h_xixi)


def mp_log_density(y, psi, xi, dps=60):
    """The log density in exact arithmetic, at ``dps`` digits *or the ambient precision*.

    ``mp.diff`` differentiates by evaluating at ``x + h`` with ``h`` sized for a
    working precision it raises internally.  A nested ``workdps(60)`` here would
    clamp that back down and round ``h`` away, so every reference derivative
    would come back as exactly zero; re-coercing an argument that is already an
    ``mpf`` rounds it for the same reason.  Both are avoided: the context can
    only raise the precision, and existing ``mpf`` arguments pass through.
    """
    import mpmath as mp

    target = int(dps * 3.3219281) + 10
    with mp.workprec(max(mp.mp.prec, target)):
        y, psi, xi = (a if isinstance(a, mp.mpf) else mp.mpf(a) for a in (y, psi, xi))
        return -mp.log(psi) - (1 + 1 / xi) * mp.log1p(xi * y / psi)


def mp_derivatives(function, point, orders, dps=60):
    """mpmath partial derivatives of ``function`` at ``point`` for each order tuple."""
    import mpmath as mp

    with mp.workdps(dps):
        pt = tuple(mp.mpf(p) for p in point)
        return [float(mp.diff(function, pt, order)) for order in orders]


def _loglikelihood_and_score(excess, design, parameters):
    """Summed log likelihood and its gradient for log-linear scale with constant shape."""
    alpha = parameters[:-1]
    xi = float(parameters[-1])
    eta = design @ alpha
    if not np.all(np.isfinite(eta)) or np.any(np.abs(eta) > _PARAMETER_BOUND):
        return -math.inf, np.full_like(parameters, np.nan)
    sigma = np.exp(eta)
    ratio = excess / sigma
    support = 1.0 + xi * ratio
    if np.min(support) <= 0.0:
        return -math.inf, np.full_like(parameters, np.nan)
    log_support = np.log1p(xi * ratio)
    rows = -eta - (1.0 + 1.0 / xi) * log_support
    xi_rows = log_support / xi**2 - ((xi + 1.0) / xi) * ratio / support
    eta_rows = -1.0 + (xi + 1.0) * ratio / support
    score = np.concatenate((design.T @ eta_rows, np.array([float(np.sum(xi_rows))])))
    return float(np.sum(rows)), score


def _objective(excess, design, parameters):
    value, score = _loglikelihood_and_score(excess, design, parameters)
    if not math.isfinite(value) or not np.all(np.isfinite(score)):
        return 1.0e20, np.zeros_like(parameters)
    return -value, -score


def fit_scale_regression(excess, design, *, xi_bounds=(1.0e-6, 0.999)):
    """Maximum-likelihood log-linear scale with a constant shape, by L-BFGS-B."""
    excess = np.asarray(excess, dtype=float)
    design = np.asarray(design, dtype=float)
    mean = float(np.mean(excess))
    variance = float(np.var(excess))
    start_xi = float(np.clip(0.5 * (1.0 - mean**2 / variance), xi_bounds[0], xi_bounds[1]))
    best = None
    for candidate in (start_xi, 0.1, 0.3):
        start = np.zeros(design.shape[1] + 1)
        start[0] = math.log(max(mean * (1.0 - candidate), 1.0e-8))
        start[-1] = float(np.clip(candidate, xi_bounds[0] + 1e-6, xi_bounds[1] - 1e-6))
        result = optimize.minimize(
            lambda p: _objective(excess, design, p),
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(None, None)] * design.shape[1] + [xi_bounds],
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
        )
        if best is None or result.fun < best.fun:
            best = result
    return best.x[:-1], float(best.x[-1]), float(-best.fun)


def profile_shape(excess, design, grid):
    """Profile log likelihood over a shape grid, re-optimising the scale coefficients."""
    excess = np.asarray(excess, dtype=float)
    design = np.asarray(design, dtype=float)
    values = []
    for xi in np.asarray(grid, dtype=float):
        start = np.zeros(design.shape[1])
        start[0] = math.log(max(float(np.mean(excess)) * (1.0 - xi), 1.0e-8))

        def objective(alpha, xi=xi):
            value, score = _objective(excess, design, np.append(alpha, xi))
            return value, score[:-1]

        result = optimize.minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
        )
        values.append(float(-result.fun))
    return np.array(values)
