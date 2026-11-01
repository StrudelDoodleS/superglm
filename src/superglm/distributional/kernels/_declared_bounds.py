"""Window jets of declared severity laws, for truncated and censored rows.

:func:`declared_window_jets` gives ``log P(lower < Y <= upper)`` and its packed
derivatives to order 4, unit mass, in a generated module's own coordinates
(natural or log), from that module's log survival, log CDF and log density
kernels.  A window is read from the lower tail, ``log F``, where the caller
asks and always when its lower end is the support's start, otherwise from the
upper tail by reflection; :func:`log_window_jets` combines the two ends.

A window with two finite ends whose log-width ``w = log(upper/lower)`` times
the row's ``rate`` is at most :data:`NARROW_WINDOW` is integrated instead by
:func:`quadrature_window_jets` on ``s = log y``, where ``log Y`` has density
``f(y) y``; its nodes are ``y = lower e^(fraction w)``, formed as ``lower +
lower expm1(fraction w)`` so they keep their place in the window.  ``rate`` is
the scale of the density of ``log Y`` across the window and is at least one,
so a narrow window is also narrow in relative width: with ``l(s) = log f_s``,
``w max(1, sqrt(|l''|), |l'|) <= NARROW_WINDOW`` holds every term of the
three-point rule's sixth-derivative error (Abramowitz and Stegun 25.4.30) to
about ``5e-7 NARROW_WINDOW^6`` relative, as for the normal window.

:func:`gamma_window_jets` is ``GammaLS``'s window.  On ``s = log y`` the gamma
of shape ``a`` and mean ``m`` has ``l(s) = a s - a e^s/m + const``, so ``l' =
a (1 - r)`` and every higher derivative is ``-a r``, ``r = y/m``; its ``rate``
is ``max(1, sqrt(a r_upper), a max(|1 - r_lower|, |1 - r_upper|))``, the
largest of each over the window.  It reads a two-ended window from the lower
tail when the window's midpoint is below the mean.
"""

from __future__ import annotations

from math import comb
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._declared import gamma, gamma_log, law_jets
from superglm.distributional.kernels._response_tails import (
    NARROW_WINDOW,
    Jet,
    log_window_jets,
    quadrature_window_jets,
)


def declared_window_jets(
    module: ModuleType,
    lower: NDArray,
    upper: NDArray,
    parameters: NDArray,
    data: tuple[NDArray, ...],
    *,
    order: int,
    from_lower: NDArray,
    rate: NDArray,
) -> Jet:
    """``log P(lower < Y <= upper)`` and its packed jet to ``order`` under ``module``'s law.

    ``lower`` at or below zero is the support's start and ``upper = inf`` an
    open end, not both on one row; ``parameters`` are ``(n, K)`` in the
    module's coordinates and ``data`` its per-row data columns.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    theta = np.asarray(parameters, dtype=np.float64)
    columns = tuple(np.asarray(column, dtype=np.float64) for column in data)
    n_rows = len(lo)
    open_below = ~(lo > 0.0)
    open_above = ~(hi < np.inf)
    lower_tail = open_below | (np.asarray(from_lower, dtype=bool) & ~open_above)
    near_point = np.where(lower_tail, hi, lo)
    far_point = np.where(lower_tail, lo, hi)
    far_present = np.where(lower_tail, ~open_below, ~open_above)

    near = [np.zeros(n_rows)]
    far = [np.full(n_rows, -np.inf)]
    for level in range(1, order + 1):
        width = comb(theta.shape[1] + level - 1, level)
        near.append(np.zeros((n_rows, width)))
        far.append(np.zeros((n_rows, width)))
    for law, tail in (("logcdf", lower_tail), ("logsf", ~lower_tail)):
        near_rows = np.flatnonzero(tail)
        far_rows = np.flatnonzero(tail & far_present)
        rows = np.concatenate((near_rows, far_rows))
        if len(rows) == 0:
            continue
        points = np.concatenate((near_point[near_rows], far_point[far_rows]))
        jets, _ = law_jets(module, law, order, points, theta[rows], tuple(c[rows] for c in columns))
        split = len(near_rows)
        for level, values in enumerate(jets):
            near[level][near_rows] = values[:split]
            far[level][far_rows] = values[split:]
    result = log_window_jets(tuple(near), tuple(far), order=order)

    two_ended = np.flatnonzero(~open_below & ~open_above)
    if len(two_ended) == 0:
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        relative = (hi[two_ended] - lo[two_ended]) / lo[two_ended]
        width = np.where(
            relative <= 1.0,
            np.log1p(np.minimum(relative, 1.0)),
            np.log(hi[two_ended]) - np.log(lo[two_ended]),
        )
        narrow = width * np.asarray(rate, dtype=np.float64)[two_ended] <= NARROW_WINDOW
    rows, width = two_ended[narrow], width[narrow]
    if len(rows) == 0:
        return result
    start, row_theta = lo[rows], theta[rows]
    row_data = tuple(c[rows] for c in columns)
    ones = np.ones(len(rows))

    def log_density_jets(fraction: float) -> Jet:
        y = start + start * np.expm1(fraction * width)
        jets, _ = law_jets(module, "density", order, y, row_theta, row_data)
        carrier = np.empty(len(rows))
        module.carrier(y, *(np.ascontiguousarray(c) for c in row_data), ones, carrier)
        return (jets[0] + carrier + np.log(y), *jets[1:])

    quadrature = quadrature_window_jets(0.5 * width, log_density_jets, order=order)
    for target, source in zip(result, quadrature, strict=True):
        target[rows] = source
    return result


def gamma_window_jets(
    lower: NDArray,
    upper: NDArray,
    parameters: NDArray,
    weight: NDArray,
    *,
    order: int,
    coordinates: str = "natural",
) -> Jet:
    """``GammaLS``'s window jets: shape ``weight/cv^2``, mean ``mean``.

    ``parameters`` are ``(mean, cv)``, or ``(log mean, log cv)`` with
    ``coordinates="log"``, the predictor scale of its log links.
    """
    theta = np.asarray(parameters, dtype=np.float64)
    natural = np.exp(theta) if coordinates == "log" else theta
    mean, cv = natural[:, 0], natural[:, 1]
    lo = np.maximum(np.asarray(lower, dtype=np.float64), 0.0)
    hi = np.asarray(upper, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        shape = np.asarray(weight, dtype=np.float64) / (cv * cv)
        r_lower, r_upper = lo / mean, hi / mean
        rate = np.maximum(
            np.maximum(1.0, np.sqrt(shape * r_upper)),
            shape * np.maximum(np.abs(1.0 - r_lower), np.abs(1.0 - r_upper)),
        )
        from_lower = lo + hi < 2.0 * mean
    return declared_window_jets(
        gamma_log if coordinates == "log" else gamma,
        lower,
        upper,
        theta,
        (weight,),
        order=order,
        from_lower=from_lower,
        rate=rate,
    )


def _warmup_declared_bounds() -> None:
    """Compile the window kernels a bounded ``GammaLS`` fit calls.

    Orders 0-2 in ``(mean, cv)`` for the fit and 3-4 in ``(log mean, log cv)``
    for its smoothing-parameter derivatives, each with both tails, the
    narrow-window density and the carrier, and the incomplete-gamma engine
    under them.
    """
    lower = np.array([1.0, 0.0, 1.0])
    upper = np.array([np.inf, 1.0, 1.0 + 1e-6])
    weight = np.ones(3)
    for order in (0, 1, 2):
        gamma_window_jets(lower, upper, np.ones((3, 2)), weight, order=order)
    for order in (3, 4):
        gamma_window_jets(lower, upper, np.zeros((3, 2)), weight, order=order, coordinates="log")


__all__ = ["declared_window_jets", "gamma_window_jets"]
