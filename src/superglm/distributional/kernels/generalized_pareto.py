"""Primitive row numerics for the generalized Pareto distribution of excesses.

Coordinates: ``(psi, xi)`` = (scale, shape), response the excess ``y >= 0`` over
a threshold chosen outside the family (the splice recipe lives in
``docs/models/distributional.md``):

    log f = -log psi - (1 + 1/xi) log1p(xi y / psi)

with the exponential as the ``xi -> 0`` limit.  Every formula is written without
catastrophic cancellation, using ``t = y/psi``, ``z = xi t``, ``s = 1 + z`` and
the identity ``z / xi = t``:

    log f    = -log psi - t L1(z) - log1p(z)
    s_psi    = (t - 1)/(psi s)
    s_xi     = t^2 D(z) - t/s
    H_psipsi = (1 - t (2 + z))/(psi^2 s^2)
    H_psixi  = -t (t - 1)/(psi s^2)
    H_xixi   = t^3 E(z) + t^2/s^2

    L1(z) = log1p(z)/z, D(z) = (log1p(z) - z/(1+z))/z^2, E(z) = (1/(1+z)^2 - 2 D(z))/z

``D`` and ``E`` lose everything in their direct form as ``z -> 0`` (``E`` is a
total loss by ``z = 1e-10``), so both switch to an alternating power series
inside ``|z| < 0.5``.  The switch and truncation are checked against
high-precision references in the family kernel tests.

Imports only numpy, scipy and the standard library (primitive-kernel rule).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._common import readonly, readonly_bool

_FLOAT = np.float64
_Z_SERIES_RADIUS = 0.5
_Z_SERIES_TERMS = 70
_MOMENT_WALL_MARGIN = 0.02
_FALLBACK_SHAPE = 0.25


class GeneralizedParetoDomainError(ValueError):
    """Raised when a generalized Pareto row is outside the numerical domain."""


class GeneralizedParetoInitializationWarning(UserWarning):
    """The method-of-moments start was not usable inside the configured shape walls."""


def _vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=_FLOAT)
    if array.ndim != 1 or array.size == 0:
        raise GeneralizedParetoDomainError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise GeneralizedParetoDomainError(f"{name} must be finite")
    return array


def _positive_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array <= 0.0):
        raise GeneralizedParetoDomainError(f"{name} must be strictly positive")
    return array


def _nonnegative_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array < 0.0):
        raise GeneralizedParetoDomainError(f"{name} must be non-negative")
    return array


def _matching(reference: NDArray[np.float64], *others: NDArray[np.float64]) -> None:
    if any(other.shape != reference.shape for other in others):
        raise GeneralizedParetoDomainError("generalized Pareto row arrays must have the same shape")


# ---------------------------------------------------------------- series functions of z


def _power_series(arg: NDArray[np.float64], coefficient, terms: int) -> NDArray[np.float64]:
    acc = np.zeros_like(arg)
    power = np.ones_like(arg)
    for j in range(terms):
        acc = acc + coefficient(j) * power
        power = power * arg
    return acc


def _piecewise(
    arg: NDArray,
    coefficient,
    direct,
    *,
    radius: float = _Z_SERIES_RADIUS,
    terms: int = _Z_SERIES_TERMS,
) -> NDArray[np.float64]:
    values = np.asarray(arg, dtype=_FLOAT)
    out = np.empty_like(values)
    inner = np.abs(values) < radius
    if np.any(inner):
        out[inner] = _power_series(values[inner], coefficient, terms)
    outer = ~inner
    if np.any(outer):
        out[outer] = direct(values[outer])
    return out


def series_l1(z: NDArray) -> NDArray[np.float64]:
    """``log1p(z)/z``, ``1`` at zero."""
    return _piecewise(z, lambda j: (-1.0) ** j / (j + 1), lambda a: np.log1p(a) / a)


def series_d(z: NDArray) -> NDArray[np.float64]:
    """``(log1p(z) - z/(1 + z))/z^2``, ``1/2`` at zero."""
    return _piecewise(
        z,
        lambda j: (-1.0) ** j * (j + 1) / (j + 2),
        lambda a: (np.log1p(a) - a / (1.0 + a)) / (a * a),
    )


def series_e(z: NDArray) -> NDArray[np.float64]:
    """``(1/(1 + z)^2 - 2 D(z))/z``, ``-2/3`` at zero."""

    def direct(a):
        d = (np.log1p(a) - a / (1.0 + a)) / (a * a)
        return (1.0 / ((1.0 + a) * (1.0 + a)) - 2.0 * d) / a

    return _piecewise(z, lambda j: (-1.0) ** (j + 1) * (j + 2) * (j + 1) / (j + 3), direct)


def series_e1(a: NDArray) -> NDArray[np.float64]:
    """``expm1(a)/a``, ``1`` at zero; used by the quantile."""
    return _piecewise(a, lambda j: 1.0 / math.factorial(j + 1), lambda b: np.expm1(b) / b)


@dataclass(frozen=True)
class GeneralizedParetoKernelEvaluation:
    """Immutable row evaluation with read-only outputs."""

    optimizing_log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    valid: NDArray[np.bool_]


def scale_rows(
    response: NDArray,
    scale: NDArray,
    shape: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> GeneralizedParetoKernelEvaluation:
    """Generalized Pareto log density, score and packed Hessian in ``(psi, xi)`` per row."""
    y = _nonnegative_vector(response, name="response")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    _matching(y, psi, xi, weight)
    if derivative_order not in (0, 1, 2):
        raise GeneralizedParetoDomainError("derivative_order must be an integer from zero to two")

    with np.errstate(over="ignore", invalid="ignore"):
        t = y / psi
        z = xi * t
    support = 1.0 + z
    inside = np.isfinite(support) & (support > 0.0)
    n = len(y)
    optimizing = np.zeros(n, dtype=_FLOAT)
    score = np.zeros((n, 2), dtype=_FLOAT) if derivative_order >= 1 else None
    hessian = np.zeros((n, 3), dtype=_FLOAT) if derivative_order == 2 else None
    if np.any(inside):
        ti, zi, si = t[inside], z[inside], support[inside]
        pi, wi = psi[inside], weight[inside]
        optimizing[inside] = wi * (-np.log(pi) - ti * series_l1(zi) - np.log1p(zi))
        if score is not None:
            score[inside, 0] = wi * (ti - 1.0) / (pi * si)
            score[inside, 1] = wi * (ti * ti * series_d(zi) - ti / si)
        if hessian is not None:
            s2 = si * si
            hessian[inside, 0] = wi * (1.0 - ti * (2.0 + zi)) / (pi * pi * s2)
            hessian[inside, 1] = -wi * ti * (ti - 1.0) / (pi * s2)
            hessian[inside, 2] = wi * (ti**3 * series_e(zi) + ti * ti / s2)
    for name, values in (("likelihood", optimizing), ("score", score), ("Hessian", hessian)):
        if values is not None and not np.all(np.isfinite(values)):
            raise GeneralizedParetoDomainError(f"generalized Pareto {name} is not representable")
    return GeneralizedParetoKernelEvaluation(
        readonly(optimizing),
        None if score is None else readonly(score),
        None if hessian is None else readonly(hessian),
        readonly_bool(inside),
    )


def expected_information(
    scale: NDArray,
    shape: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed ``-E[Hessian]`` in ``(psi, xi)`` per row (channel order as the Hessian).

    Yoshida (arXiv:2303.02402, Appendix B) in ``(log psi, xi)``, converted by
    ``I_psi. = I_eta. / psi``.  Finite and positive definite for ``xi > -1/2``
    (Smith 1985).
    """
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    _matching(xi, psi, weight)
    if np.any(2.0 * xi + 1.0 <= 0.0):
        raise GeneralizedParetoDomainError(
            "the generalized Pareto expected information exists only for shape > -1/2"
        )
    information = np.empty((len(xi), 3), dtype=_FLOAT)
    information[:, 0] = 1.0 / (psi * psi * (2.0 * xi + 1.0))
    information[:, 1] = 1.0 / (psi * (2.0 * xi + 1.0) * (xi + 1.0))
    information[:, 2] = 2.0 / ((2.0 * xi + 1.0) * (xi + 1.0))
    information = information * weight[:, None]
    if not np.all(np.isfinite(information)):
        raise GeneralizedParetoDomainError(
            "generalized Pareto expected information is not representable"
        )
    return readonly(information)


def generalized_pareto_cdf(y: NDArray, scale: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """``P(Y <= y)`` for an excess: zero at ``y <= 0``, one past a finite upper endpoint."""
    values = _vector(y, name="y")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(values, psi, xi)
    out = np.zeros(len(values), dtype=_FLOAT)
    interior = values > 0.0
    if np.any(interior):
        with np.errstate(over="ignore", invalid="ignore"):
            t = values[interior] / psi[interior]
            z = xi[interior] * t
        interior_cdf = np.ones(len(t), dtype=_FLOAT)
        below_upper_endpoint = np.isfinite(z) & (1.0 + z > 0.0)
        if np.any(below_upper_endpoint):
            interior_cdf[below_upper_endpoint] = -np.expm1(
                -t[below_upper_endpoint] * series_l1(z[below_upper_endpoint])
            )
        out[interior] = interior_cdf
    return readonly(np.clip(out, 0.0, 1.0))


def generalized_pareto_quantile(p: NDArray, scale: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """Quantile of the excess for ``p`` strictly inside ``(0, 1)``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise GeneralizedParetoDomainError("quantile probabilities must lie strictly inside (0, 1)")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(probabilities, psi, xi)
    log_survival = np.log1p(-probabilities)
    return readonly(psi * (-log_survival) * series_e1(-xi * log_survival))


def generalized_pareto_expected_shortfall(
    p: NDArray, scale: NDArray, shape: NDArray
) -> NDArray[np.float64]:
    """``E[Y | Y > q_p] = (q_p + psi) / (1 - xi)``; infinity for ``xi >= 1``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(probabilities, psi, xi)
    quantile = generalized_pareto_quantile(probabilities, psi, xi)
    result = np.full(psi.shape, np.inf, dtype=_FLOAT)
    finite = xi < 1.0
    if np.any(finite):
        with np.errstate(over="ignore", invalid="ignore"):
            result[finite] = (quantile[finite] + psi[finite]) / (1.0 - xi[finite])
    return readonly(result)


def generalized_pareto_mean(scale: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """``E[Y] = psi/(1 - xi)``; ``inf`` for ``xi >= 1``."""
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(psi, xi)
    result = np.full(psi.shape, np.inf, dtype=_FLOAT)
    finite = xi < 1.0
    if np.any(finite):
        result[finite] = psi[finite] / (1.0 - xi[finite])
    return readonly(result)


def initialize_generalized_pareto(
    response: NDArray,
    mass: NDArray,
    *,
    shape_lower: float,
    shape_upper: float,
) -> NDArray[np.float64]:
    """Constant start from the weighted first two moments of the excesses.

    ``E[Y] = psi/(1 - xi)`` and ``Var[Y] = psi^2/((1 - xi)^2 (1 - 2 xi))`` give
    ``ybar^2 / s2 = 1 - 2 xi``, so ``xi0 = (1 - ybar^2/s2)/2`` and
    ``psi0 = ybar (1 - xi0)``.  Falls back to a quarter shape, clamped inside the
    walls, when the moments are unusable or the estimate leaves the walls.
    """
    y = _nonnegative_vector(response, name="response")
    weights = _positive_vector(mass, name="mass")
    _matching(y, weights)
    if not (
        math.isfinite(shape_lower) and math.isfinite(shape_upper) and shape_lower < shape_upper
    ):
        raise GeneralizedParetoDomainError("shape walls must be finite and strictly ordered")
    margin = _MOMENT_WALL_MARGIN * (shape_upper - shape_lower)
    fallback = min(max(_FALLBACK_SHAPE, shape_lower + margin), shape_upper - margin)
    total = float(np.sum(weights, dtype=_FLOAT))
    mean = float(np.sum(weights * y, dtype=_FLOAT) / total)
    centred = y - mean
    variance = float(np.sum(weights * centred * centred, dtype=_FLOAT) / total)
    usable = math.isfinite(mean) and mean > 0.0 and math.isfinite(variance) and variance > 0.0
    estimate = 0.5 * (1.0 - mean * mean / variance) if usable else math.nan
    if (
        not usable
        or not math.isfinite(estimate)
        or estimate <= shape_lower + margin
        or estimate >= shape_upper - margin
    ):
        warnings.warn(
            "the method-of-moments generalized Pareto start is not usable inside the shape walls "
            f"({shape_lower!r}, {shape_upper!r}); starting from shape {fallback:.4g}",
            GeneralizedParetoInitializationWarning,
            stacklevel=2,
        )
        estimate = fallback
    start_scale = mean * (1.0 - estimate) if (math.isfinite(mean) and mean > 0.0) else 1.0
    if not math.isfinite(start_scale) or start_scale <= 0.0:
        start_scale = 1.0
    theta = np.column_stack(
        (np.full(len(y), start_scale, dtype=_FLOAT), np.full(len(y), estimate, dtype=_FLOAT))
    )
    return readonly(theta)
