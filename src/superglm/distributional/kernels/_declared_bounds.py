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
tail when the window's midpoint is below the mean.  A tail point of a row whose
shape exceeds :data:`UNIFORM_SHAPE` (``1e4``, the top of the incomplete-gamma
engine's measured envelope, ``tests/test_incomplete_gamma_engine.py``) and
lies in Temme's uniform domain is read from the declaration's ``*_uniform``
laws: that expansion costs the same at every shape and is analytic through
the log-normal limit (``kernels/_gg_uniform_tail.py``), where the engine's
series and continued fraction take ``O(sqrt(a))`` terms near ``y = mean``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._curvature_direction import compose_derivatives
from superglm.distributional.kernels._declared import (
    gamma,
    gamma_log,
    generalized_pareto,
    law_jets,
    normal,
)
from superglm.distributional.kernels._gg_log_tail import gg_log_tail_jets
from superglm.distributional.kernels._gg_uniform_tail import gg_uniform_domain
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

UNIFORM_SHAPE = 1e4


def _tail_jets(module, law, order, points, theta, columns, uniform):
    """``law``'s jets at ``points``; the entries ``uniform`` flags read ``<law>_uniform``."""
    flagged = (
        np.zeros(len(points), dtype=bool) if uniform is None else uniform(points, theta, columns)
    )
    if not flagged.any():
        return law_jets(module, law, order, points, theta, columns)[0]
    jets = (np.empty(len(points)),) + tuple(
        np.empty((len(points), comb(theta.shape[1] + level - 1, level)))
        for level in range(1, order + 1)
    )
    routed = np.flatnonzero(flagged)
    sub, valid = law_jets(
        module,
        f"{law}_uniform",
        order,
        points[routed],
        theta[routed],
        tuple(c[routed] for c in columns),
    )
    for target, source in zip(jets, sub, strict=True):
        target[routed] = source
    # A point the kernel's own domain test rejects is read by the engine.
    flagged[routed[~valid]] = False
    rest = np.flatnonzero(~flagged)
    if len(rest):
        sub, _ = law_jets(
            module, law, order, points[rest], theta[rest], tuple(c[rest] for c in columns)
        )
        for target, source in zip(jets, sub, strict=True):
            target[rest] = source
    return jets


def tail_window_jets(
    lower: NDArray,
    upper: NDArray,
    n_parameters: int,
    *,
    order: int,
    from_lower: NDArray,
    tail_jets: Callable[[str, NDArray, NDArray], Jet],
    support_start: float = 0.0,
) -> Jet:
    """``log P(lower < Y <= upper)`` and its packed jet from the two ends' log tails.

    ``lower`` at or below ``support_start`` is the support's start and
    ``upper = inf`` an open end, not both on one row.  A window open below is
    read from the lower tail, one open above from the upper tail, and a
    two-ended one from the lower tail where ``from_lower``.
    ``tail_jets(law, points, rows)`` is the ``"logcdf"`` or ``"logsf"`` jet at
    ``points`` under those rows' laws.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    n_rows = len(lo)
    open_below = ~(lo > support_start)
    open_above = ~(hi < np.inf)
    lower_tail = open_below | (np.asarray(from_lower, dtype=bool) & ~open_above)
    near_point = np.where(lower_tail, hi, lo)
    far_point = np.where(lower_tail, lo, hi)
    far_present = np.where(lower_tail, ~open_below, ~open_above)
    one_sided = not far_present.any()
    laws = [("logcdf", np.flatnonzero(lower_tail)), ("logsf", np.flatnonzero(~lower_tail))]
    if one_sided and n_rows and len(laws[1][1]) in (0, n_rows):
        # Every row is one tail under one law: the window is that tail's jet.
        law, rows = laws[0] if len(laws[1][1]) == 0 else laws[1]
        return tuple(tail_jets(law, near_point, rows)[: order + 1])

    widths = [comb(n_parameters + level - 1, level) for level in range(1, order + 1)]
    near = [np.zeros(n_rows), *(np.zeros((n_rows, width)) for width in widths)]
    far = None if one_sided else [np.full(n_rows, -np.inf), *(np.zeros_like(a) for a in near[1:])]
    for law, near_rows in laws:
        far_rows = near_rows[far_present[near_rows]]
        rows = np.concatenate((near_rows, far_rows))
        if len(rows) == 0:
            continue
        points = np.concatenate((near_point[near_rows], far_point[far_rows]))
        jets = tail_jets(law, points, rows)
        split = len(near_rows)
        for level, values in enumerate(jets[: order + 1]):
            near[level][near_rows] = values[:split]
            if far is not None:
                far[level][far_rows] = values[split:]
    if far is None:
        return tuple(near)
    return log_window_jets(tuple(near), tuple(far), order=order)


def _integrated(result: Jet, rows: NDArray, half_width: NDArray, node_jets, order: int) -> Jet:
    """``result`` with ``rows`` replaced by the narrow-window quadrature of ``node_jets``."""
    quadrature = quadrature_window_jets(half_width, node_jets, order=order)
    for target, source in zip(result, quadrature, strict=True):
        target[rows] = source
    return result


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
    """:func:`tail_window_jets` on positive support, narrow windows integrated on ``log y``.

    ``log_density_jets(y, rows)`` is the jet of the density of ``log Y`` at
    ``y`` under those rows' laws.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    result = tail_window_jets(
        lo, hi, n_parameters, order=order, from_lower=from_lower, tail_jets=tail_jets
    )
    two_ended = np.flatnonzero((lo > 0.0) & (hi < np.inf))
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

    return _integrated(result, rows, 0.5 * width, node_jets, order)


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
    uniform=None,
) -> Jet:
    """:func:`window_jets` under a generated module's law, in its own coordinates.

    ``parameters`` are ``(n, K)`` in the module's coordinates and ``data`` its
    per-row data columns.  ``uniform``, given, maps ``(points, parameters,
    data)`` to the tail points read from the module's ``<law>_uniform`` laws.
    """
    theta = np.asarray(parameters, dtype=np.float64)
    columns = tuple(np.asarray(column, dtype=np.float64) for column in data)

    def tail_jets(law: str, points: NDArray, rows: NDArray) -> Jet:
        row_data = tuple(c[rows] for c in columns)
        return _tail_jets(module, law, order, points, theta[rows], row_data, uniform)

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
    log_coordinates = coordinates == "log"

    def uniform(points, row_theta, columns):
        row_mean = np.exp(row_theta[:, 0]) if log_coordinates else row_theta[:, 0]
        row_cv = np.exp(row_theta[:, 1]) if log_coordinates else row_theta[:, 1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            routed = columns[0] / (row_cv * row_cv) > UNIFORM_SHAPE
            if routed.any():
                rows = np.flatnonzero(routed)
                q = row_cv[rows] / np.sqrt(columns[0][rows])
                # The kernels' centred log(y/mean), so the route sees their w.
                y, mean = points[rows], row_mean[rows]
                near = (0.5 * mean <= y) & (y <= 2.0 * mean)
                log_ratio = np.where(near, np.log1p((y - mean) / mean), np.log(y) - np.log(mean))
                routed[rows] = gg_uniform_domain(log_ratio / q, q)
        return routed

    return declared_window_jets(
        gamma_log if log_coordinates else gamma,
        lower,
        upper,
        theta,
        (weight,),
        order=order,
        from_lower=from_lower,
        rate=rate,
        uniform=uniform,
    )


def normal_window_jets(
    lower: NDArray,
    upper: NDArray,
    location: NDArray,
    scale: NDArray,
    root: NDArray,
    *,
    order: int,
    width: NDArray | None = None,
) -> Jet:
    """``log P(lower < X <= upper)`` for ``X ~ N(location, (scale/root)^2)``, in ``(location, scale)``.

    ``lower`` may be ``-inf`` and ``upper`` ``+inf``, not both.  The tails are
    the ``normal`` declaration's at the end's residual ``c - location`` with
    location zero, the law being translation invariant, so a large common
    offset of the window and the location cancels before the kernels see it.
    A two-ended window is read from the lower tail when its midpoint is below
    the location, so neither end's tail rounds to one.  A window whose
    standardised width, times ``max(1, |midpoint|)``, is at most
    :data:`NARROW_WINDOW` is integrated by quadrature on nodes placed in
    standardised coordinates.  ``width`` is ``upper - lower`` when the caller
    has it more accurately than that difference.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    mu = np.asarray(location, dtype=np.float64)
    sigma = np.asarray(scale, dtype=np.float64)
    k = np.asarray(root, dtype=np.float64)
    inverse = 1.0 / sigma
    with np.errstate(invalid="ignore", over="ignore"):
        z_lo = (lo - mu) * k * inverse
        z_hi = (hi - mu) * k * inverse
        from_lower = z_lo + z_hi < 0.0

    def law(name: str, residual: NDArray, rows: NDArray) -> Jet:
        theta = np.column_stack((np.zeros(len(rows)), sigma[rows]))
        return law_jets(normal, name, order, residual, theta, (k[rows],))[0]

    def tail_jets(name: str, points: NDArray, rows: NDArray) -> Jet:
        jets = law(name, points - mu[rows], rows)
        # Where Phi rounds to one the tail is flat in float64 and its
        # derivatives are zero, though the point's own derivatives may overflow.
        flat = jets[1][:, 0] == 0.0 if order >= 1 else ()
        for level in jets[1:]:
            level[flat] = 0.0
        return jets

    result = tail_window_jets(
        lo,
        hi,
        2,
        order=order,
        from_lower=from_lower,
        tail_jets=tail_jets,
        support_start=-np.inf,
    )
    two_ended = np.flatnonzero(np.isfinite(z_lo) & np.isfinite(z_hi))
    if len(two_ended) == 0:
        return result
    span = (hi - lo if width is None else np.asarray(width, dtype=np.float64))[two_ended]
    with np.errstate(invalid="ignore", over="ignore"):
        extent = span * (k * inverse)[two_ended]
        reach = extent * np.maximum(1.0, 0.5 * np.abs(z_lo + z_hi)[two_ended])
    narrow = reach <= NARROW_WINDOW
    rows, span, extent = two_ended[narrow], span[narrow], extent[narrow]
    if len(rows) == 0:
        return result
    start, spread = z_lo[rows], sigma[rows] / k[rows]
    carrier = np.empty(len(rows))
    normal.carrier(np.zeros(len(rows)), np.ascontiguousarray(k[rows]), np.ones(len(rows)), carrier)
    carrier -= _HALF_LOG_TWO_PI

    def node_jets(fraction: float) -> Jet:
        jets = law("density", (start + fraction * extent) * spread, rows)
        return (jets[0] + carrier, *jets[1:])

    return _integrated(result, rows, 0.5 * span, node_jets, order)


def log_normal_window_jets(
    lower: NDArray,
    upper: NDArray,
    parameters: NDArray,
    *,
    order: int,
    mean_form: bool,
) -> Jet:
    """``LogNormalLS``'s window jets: the normal window on ``log y``.

    ``parameters`` are ``(mu, sigma)``, or ``(m, sigma)`` with ``mean_form``,
    composed through ``mu = log m - sigma^2/2`` to order two; higher orders are
    composed in predictor coordinates by :func:`normal_predictor_jets`.  A lower
    end at or below zero is the support's start.  A two-ended window's log-scale width is
    ``log1p`` of its relative width where ``log(upper) - log(lower)`` would
    cancel, and that difference where the ratio is large.
    """
    if mean_form and order > 2:
        raise ValueError(
            "mean-form window jets are formed to order two; compose higher orders "
            "from the location form with normal_predictor_jets"
        )
    theta = np.asarray(parameters, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    sigma = theta[:, 1]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        log_lower = np.log(np.maximum(lo, 0.0))
        log_upper = np.log(hi)
        relative = (hi - lo) / lo
        width = np.where(
            relative <= 1.0, np.log1p(np.minimum(relative, 1.0)), log_upper - log_lower
        )
        mu = np.log(theta[:, 0]) - 0.5 * sigma * sigma if mean_form else theta[:, 0]
    location = normal_window_jets(
        log_lower, log_upper, mu, sigma, np.ones(len(mu)), order=order, width=width
    )
    if not mean_form or order == 0:
        return location
    # mu_m = 1/m, mu_s = -sigma, mu_mm = -1/m^2, mu_ss = -1, mu_ms = 0.
    inverse = 1.0 / theta[:, 0]
    l_mu, l_sigma = location[1][:, 0], location[1][:, 1]
    score = np.column_stack((l_mu * inverse, l_sigma - sigma * l_mu))
    if order == 1:
        return location[0], score
    h_mm, h_ms, h_ss = location[2][:, 0], location[2][:, 1], location[2][:, 2]
    hessian = np.column_stack(
        (
            (h_mm - l_mu) * inverse * inverse,
            (h_ms - sigma * h_mm) * inverse,
            h_ss - 2.0 * sigma * h_ms + sigma * sigma * h_mm - l_mu,
        )
    )
    return location[0], score, hessian


def normal_predictor_jets(
    jets: Jet, scale: NDArray, increment: NDArray, *, mean_form: bool = False
) -> Jet:
    """``(mu, sigma)`` jets to the predictors ``(a, b)``, ``sigma = floor + kappa``, ``kappa = e^b``.

    ``mu = a``, or ``mu = a - sigma^2/2`` in the log-normal mean form (``a =
    log m``).  Every derivative of ``sigma`` in ``b`` is ``kappa`` and ``d^j
    mu/d b^j = -((2^(j-1) - 1) kappa^2 + sigma kappa)``, sums of positive
    terms, so the inner map rounds each factor once.
    """
    sigma = np.asarray(scale, dtype=np.float64)
    kappa = np.asarray(increment, dtype=np.float64)
    order = len(jets) - 1
    inner: dict = {level: {(1, (1,) * level): kappa} for level in range(1, order + 1)}
    inner.setdefault(1, {})[(0, (0,))] = 1.0
    if mean_form:
        for level in range(1, order + 1):
            inner[level][(0, (1,) * level)] = -(
                (2 ** (level - 1) - 1) * kappa * kappa + sigma * kappa
            )
    outer = dict(enumerate(jets[1:], start=1))
    return (jets[0], *(compose_derivatives(outer, inner, level, 2) for level in outer))


def gpd_window_jets(
    lower: NDArray,
    upper: NDArray,
    scale: NDArray,
    shape: NDArray,
    *,
    order: int,
) -> Jet:
    """``log(S(lower) - S(upper))`` of ``GeneralizedParetoLSS``'s law, in ``(scale, shape)``.

    ``lower`` at or below zero means the support's start (``S = 1``) and
    ``upper = +inf`` an open window.  Both ends come from the survival
    function (the ``generalized_pareto`` declaration's ``logsf``), which no
    window on ``y >= 0`` rounds to one at its far end.  The log density varies
    on the scale ``(psi + xi y)/(1 + |xi|)``; a window at most
    :data:`NARROW_WINDOW` of that scale wide at its steeper end is integrated
    by quadrature on nodes in ``y``.
    """
    lo = np.maximum(np.asarray(lower, dtype=np.float64), 0.0)
    hi = np.asarray(upper, dtype=np.float64)
    psi = np.asarray(scale, dtype=np.float64)
    xi = np.asarray(shape, dtype=np.float64)
    theta = np.column_stack((psi, xi))

    def tail_jets(name: str, points: NDArray, rows: NDArray) -> Jet:
        return law_jets(generalized_pareto, name, order, points, theta[rows])[0]

    result = tail_window_jets(
        lo,
        hi,
        2,
        order=order,
        from_lower=np.zeros(len(lo), dtype=bool),
        tail_jets=tail_jets,
        support_start=-np.inf,
    )
    span = hi - lo
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        support = psi + xi * np.where(xi < 0.0, hi, lo)
        reach = span * (1.0 + np.abs(xi)) / support
    rows = np.flatnonzero((support > 0.0) & (reach <= NARROW_WINDOW))
    if len(rows) == 0:
        return result

    def node_jets(fraction: float) -> Jet:
        points = lo[rows] + fraction * span[rows]
        return law_jets(generalized_pareto, "density", order, points, theta[rows])[0]

    return _integrated(result, rows, 0.5 * span[rows], node_jets, order)


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
    """Compile the window kernels a bounded fit of a declared-bounds family calls.

    ``GammaLS``: orders 0-2 in ``(mean, cv)`` for the fit and 3-4 in ``(log
    mean, log cv)`` for its smoothing-parameter derivatives, each with both
    tails, the narrow-window density and the carrier, and the incomplete-gamma
    engine under them.  ``GeneralizedGammaLSS``, ``GaussianLS``,
    ``LogNormalLS`` and ``GeneralizedParetoLSS``: each kernel once, the order
    being a run-time argument, on both tail routes and a narrow window.
    """
    lower = np.array([1.0, 0.0, 1.0])
    upper = np.array([np.inf, 1.0, 1.0 + 1e-6])
    weight = np.ones(3)
    for order in (0, 1, 2):
        gamma_window_jets(lower, upper, np.ones((3, 2)), weight, order=order)
    for order in (3, 4):
        gamma_window_jets(lower, upper, np.zeros((3, 2)), weight, order=order, coordinates="log")
    # GeneralizedGammaLSS: one row kernel for every order, on both tail routes.
    # A positive first column is a valid mean and a valid location.
    theta = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, -0.5]])
    for mean_form in (False, True):
        gg_window_jets(lower, upper, theta, order=4, mean_form=mean_form)
    ones = np.ones(3)
    normal_window_jets(
        np.array([-np.inf, 1.0, 0.0]),
        np.array([0.0, np.inf, 1e-6]),
        np.zeros(3),
        ones,
        ones,
        order=4,
    )
    gpd_window_jets(lower, upper, ones, np.array([0.0, 0.5, 0.2]), order=4)


__all__ = [
    "declared_window_jets",
    "gamma_window_jets",
    "gg_window_jets",
    "gpd_window_jets",
    "log_normal_window_jets",
    "normal_predictor_jets",
    "normal_window_jets",
    "tail_window_jets",
    "window_jets",
]
