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

from collections.abc import Callable
from math import comb
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._curvature_direction import compose_derivatives
from superglm.distributional.kernels._declared import gamma, gamma_log, law_jets
from superglm.distributional.kernels._gg_log_tail import gg_log_tail_jets
from superglm.distributional.kernels._response_tails import (
    NARROW_WINDOW,
    Jet,
    log_window_jets,
    quadrature_window_jets,
)
from superglm.distributional.kernels.generalized_gamma import (
    location_rows,
    location_third_fourth,
    mean_exists,
    mean_location_map,
    series_e1,
)

_HALF_LOG_TWO_PI = 0.5 * np.log(2.0 * np.pi)


def window_jets(
    lower: NDArray,
    upper: NDArray,
    n_parameters: int,
    *,
    order: int,
    from_lower: NDArray,
    rate: NDArray,
    tail_jets: Callable[[str, NDArray, NDArray], Jet],
    log_density_jets: Callable[[NDArray, NDArray], Jet],
) -> Jet:
    """``log P(lower < Y <= upper)`` and its packed jet to ``order`` from a law's tails.

    ``lower`` at or below zero is the support's start and ``upper = inf`` an
    open end, not both on one row.  ``tail_jets(law, points, rows)`` is the
    ``"logcdf"`` or ``"logsf"`` jet at ``points`` under those rows' laws, and
    ``log_density_jets(y, rows)`` the jet of the density of ``log Y`` there.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
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
        width = comb(n_parameters + level - 1, level)
        near.append(np.zeros((n_rows, width)))
        far.append(np.zeros((n_rows, width)))
    for law, tail in (("logcdf", lower_tail), ("logsf", ~lower_tail)):
        near_rows = np.flatnonzero(tail)
        far_rows = np.flatnonzero(tail & far_present)
        rows = np.concatenate((near_rows, far_rows))
        if len(rows) == 0:
            continue
        points = np.concatenate((near_point[near_rows], far_point[far_rows]))
        jets = tail_jets(law, points, rows)
        split = len(near_rows)
        for level, values in enumerate(jets[: order + 1]):
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
    start = lo[rows]

    def node_jets(fraction: float) -> Jet:
        return log_density_jets(start + start * np.expm1(fraction * width), rows)

    quadrature = quadrature_window_jets(0.5 * width, node_jets, order=order)
    for target, source in zip(result, quadrature, strict=True):
        target[rows] = source
    return result


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
    """:func:`window_jets` under a generated module's law, in its own coordinates.

    ``parameters`` are ``(n, K)`` in the module's coordinates and ``data`` its
    per-row data columns.
    """
    theta = np.asarray(parameters, dtype=np.float64)
    columns = tuple(np.asarray(column, dtype=np.float64) for column in data)

    def tail_jets(law: str, points: NDArray, rows: NDArray) -> Jet:
        return law_jets(module, law, order, points, theta[rows], tuple(c[rows] for c in columns))[0]

    def log_density_jets(y: NDArray, rows: NDArray) -> Jet:
        row_data = tuple(np.ascontiguousarray(c[rows]) for c in columns)
        jets, _ = law_jets(module, "density", order, y, theta[rows], row_data)
        carrier = np.empty(len(rows))
        module.carrier(y, *row_data, np.ones(len(rows)), carrier)
        return (jets[0] + carrier + np.log(y), *jets[1:])

    return window_jets(
        lower,
        upper,
        theta.shape[1],
        order=order,
        from_lower=from_lower,
        rate=rate,
        tail_jets=tail_jets,
        log_density_jets=log_density_jets,
    )


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


def _through_w(jets: Jet, w: NDArray, sigma: NDArray, order: int) -> Jet:
    """``(w, Q)`` jets to ``(mu, sigma, Q)`` through ``w = (log y - mu)/sigma``.

    ``d^b w/d sigma^b = (-1)^b b! w/sigma^b`` and ``d^(1+b) w/(d mu d sigma^b) =
    (-1)^(b+1) b!/sigma^(b+1)``; every other derivative of ``w`` is zero.
    """
    inner: dict = {1: {(1, (2,)): 1.0}}
    factorial = 1.0
    for b in range(order + 1):
        sign = -1.0 if b % 2 else 1.0
        if b >= 1:
            factorial *= b
            inner.setdefault(b, {})[(0, (1,) * b)] = sign * factorial * w / sigma**b
        if b + 1 <= order:
            inner.setdefault(b + 1, {})[(0, (0,) + (1,) * b)] = -sign * factorial / sigma ** (b + 1)
    outer = dict(enumerate(jets[1 : order + 1], start=1))
    return (
        jets[0],
        *(compose_derivatives(outer, inner, level, 3, components=2) for level in outer),
    )


def gg_window_jets(
    lower: NDArray,
    upper: NDArray,
    parameters: NDArray,
    *,
    order: int,
    mean_form: bool,
) -> Jet:
    """``GeneralizedGammaLSS``'s window jets in ``(mean | mu, sigma, Q)``.

    The tails are :func:`gg_log_tail_jets` at ``w = (log y - mu)/sigma``,
    composed through ``w``; a two-ended window is read from the lower tail
    when its midpoint lies below ``w = -Q/3``, the median of ``W`` to first
    order in ``Q``.  On ``s = log y`` the law has ``l(s) = -log sigma + G(w,
    Q)``, so ``l' = -expm1(Q w)/(Q sigma)`` and ``l^(k) = -Q^(k-2) e^(Q
    w)/sigma^k`` for ``k >= 2``, each ``|l^(k)|^(1/k)`` at most ``max(|Q|,
    e^(Q w/2))/sigma``; the narrow-window ``rate`` is ``max(1, |l'|, max(|Q|,
    e^(Q w/2))/sigma)``, largest at an end.  The mean form composes the
    location form's jets through ``mu = log m - log C(sigma, Q)``; rows whose
    mean does not exist are NaN.
    """
    theta = np.asarray(parameters, dtype=np.float64)
    n_rows = len(theta)
    sigma, q = theta[:, 1], theta[:, 2]
    rows = np.flatnonzero(mean_exists(sigma, q)) if mean_form else np.arange(n_rows)
    result: list[NDArray] = [np.full(n_rows, np.nan)]
    result += [np.full((n_rows, (k + 1) * (k + 2) // 2), np.nan) for k in range(1, order + 1)]
    if len(rows) == 0:
        return tuple(result)
    sigma, q = sigma[rows], q[rows]
    if mean_form:
        mu, mean_inner, _ = mean_location_map(theta[rows, 0], sigma, q, max(order, 1))
    else:
        mu = theta[rows, 0]
    lo = np.asarray(lower, dtype=np.float64)[rows]
    hi = np.asarray(upper, dtype=np.float64)[rows]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w_lo = (np.log(lo) - mu) / sigma
        w_hi = (np.log(hi) - mu) / sigma
        ends = np.column_stack((w_lo, w_hi))
        slope = np.abs(ends * series_e1(q[:, None] * ends))
        curvature = np.exp(0.5 * np.max(q[:, None] * ends, axis=1))
        steepest = np.maximum(np.max(slope, axis=1), np.maximum(np.abs(q), curvature))
        rate = np.maximum(1.0, steepest / sigma)

    def tail_jets(law: str, points: NDArray, at: NDArray) -> Jet:
        w = (np.log(points) - mu[at]) / sigma[at]
        jets = gg_log_tail_jets(w, q[at], upper=law == "logsf", order=order)
        return _through_w(jets, w, sigma[at], order)

    def log_density_jets(y: NDArray, at: NDArray) -> Jet:
        unit = np.ones(len(at))
        args = (y, mu[at], sigma[at], q[at], unit)
        rows_2 = location_rows(*args, derivative_order=min(order, 2))
        jets = [rows_2.optimizing_log_likelihood - _HALF_LOG_TWO_PI, rows_2.score]
        jets = [*jets, rows_2.hessian_packed][: order + 1]
        if order >= 3:
            jets += list(location_third_fourth(*args))[: order - 2]
        return tuple(jets)

    location = window_jets(
        lo,
        hi,
        3,
        order=order,
        from_lower=0.5 * (w_lo + w_hi) < -q / 3.0,
        rate=rate,
        tail_jets=tail_jets,
        log_density_jets=log_density_jets,
    )
    if mean_form:
        outer = dict(enumerate(location[1:], start=1))
        location = (
            location[0],
            *(compose_derivatives(outer, mean_inner, level, 3) for level in outer),
        )
    for target, source in zip(result, location, strict=True):
        target[rows] = source
    return tuple(result)


def _warmup_declared_bounds() -> None:
    """Compile the window kernels a bounded ``GammaLS`` or ``GeneralizedGammaLSS`` fit calls.

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
    # GeneralizedGammaLSS: one row kernel for every order, on both tail routes.
    theta = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, -0.5]])
    for mean_form in (False, True):
        gg_window_jets(lower, upper, theta, order=4, mean_form=mean_form)


__all__ = ["declared_window_jets", "gamma_window_jets", "gg_window_jets", "window_jets"]
