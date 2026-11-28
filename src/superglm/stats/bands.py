"""Simultaneous bands: a check that flags a plot when any one point leaves its band.

A band drawn point by point at level ``alpha`` is left somewhere far more often
than ``alpha`` when there are many points. The two bands here hold for every
point at once.

``equal_local_levels_band`` is the band for a Q-Q or P-P plot of ``n`` values.
On the probability scale the ``i``-th of ``n`` sorted iid uniform values is
``Beta(i, n + 1 - i)``. Each order statistic is tested at one local level
``eta``, between its ``eta / 2`` and ``1 - eta / 2`` quantiles, and ``eta`` is
set so that the chance that any of the ``n`` leaves its interval is the global
level ``alpha``. This is the equal local levels test (Berk and Jones 1979,
*Z. Wahrscheinlichkeitstheorie verw. Gebiete* 47(1)), the "tail-sensitive"
band of Aldor-Noiman, Brown, Buja, Rolke and Stine (2013), *The American
Statistician* 67(4), 249-260, and the two-sided band of Weine, McPeek and
Abney (2023), *Journal of Statistical Software* 106(10), sections 2.2 to 2.4.
Unlike a Kolmogorov-Smirnov band it is as sensitive in the tails as in the
middle. The local level depends on ``n`` and ``alpha`` only:

* one point is tested at the global level, ``eta = alpha``;
* from 2 to 249 points, ``eta`` is found by searching for the local level whose
  exact global level is ``alpha`` (Weine et al., section 2.3), the global level
  of a band coming from the recursion of their appendix A. The band's level is
  ``alpha`` to a relative ``1e-7``; measured, it is within ``3e-10`` at every
  ``n`` from 2 to 249 and each of the three levels;
* from 250 points, the asymptotic local level of Gontscharuk and Finner
  (2017), *Communications in Statistics - Theory and Methods* 46(5),
  2332-2342, with its second-order correction. Gontscharuk and Finner chose its
  constant as 1.6, 1.3 and 1.1 at ``alpha`` = 0.01, 0.05 and 0.1; Weine et al.
  (section 2.4) replaced 1.6 by 1.591 and confirmed 1.3. The band takes 1.591,
  1.3 and 1.1, and accepts only those three levels. Its global level runs about
  1.6% of ``alpha`` high at 250 points (4.5% at 0.01) and falls toward
  ``alpha`` as ``n`` grows.

``bonferroni_binomial_band`` bounds the counts of ``m`` cells, each
``Binomial(trials, probability)``, by the ``alpha / (2 m)`` and
``1 - alpha / (2 m)`` quantiles of its own law. By Bonferroni's inequality
(Bonferroni 1936, *Pubbl. R. Ist. Sup. Sci. Econ. Comm. Firenze* 8, 1-62) the
chance that any cell leaves is at most ``alpha`` whatever the dependence
between cells, so the band holds for the multinomial bins of a histogram too.

Both assume iid continuous values on the probability scale from a law whose
parameters are known, and that is where their level holds. With parameters
fitted to the same rows the fit absorbs part of the noise, so the bands are
conservative (Durbin 1973, *Ann. Statist.* 1(2), 279-290): for a normal Q-Q
band at 5%, Weine et al. (section 2.6, table 1) measure 0.11%, 0.35% and
1.11% at 100, 500 and 10,000 rows with the sample mean and standard
deviation. On rows held out from the fit the fitting noise is added to every
row at once, so the bands are anti-conservative, by an amount set by the ratio
of held-out to fitted rows rather than by either count (West 1996,
*Econometrica* 64(5), 1067-1084).

There the level is calibrated by simulation: replicate the held-out rows from
the fit's posterior, score each replicate as the rows are scored, and read the
level off the replicates' statistics by Barnard's Monte Carlo rule (Barnard
1963, as Myllymaki, Mrkvicka, Grabarnik, Seijo and Hahn 2017, *JRSS B* 79(2),
section 4, describe it): with ``B`` replicates and ``k = floor(alpha (B + 1))``,
an observed statistic beyond the ``k``-th most extreme replicate has chance
``k / (B + 1)`` under the null. This is the holdout predictive check of Moran,
Blei and Ranganath (2024), *JRSS B*, also the single split predictive check of
Li and Huggins (2024). Each band has one statistic for it:

* ``order_statistic_level``, the least two-sided Beta tail over the order
  statistics, which falls below ``eta`` exactly when the sample leaves the
  equal-local-levels band at ``eta``; ``band_at_level`` draws that band;
* ``binomial_cell_level``, the least one-sided binomial tail over the cells,
  which falls below ``cut`` exactly when a count leaves
  ``bonferroni_binomial_band(..., cut=cut)``;

and ``monte_carlo_level`` reads the level from replicates of either.

With many rows far from their law these tails fall below float64's range, about ``1e-308``,
and SciPy returns 0 for them, which would tie every such replicate at a level no point can
fall below. So each statistic and band also takes ``log=True`` and works with natural logs.
A tail SciPy returns below the least normal float is read in logs: the prefix
``x^a (1 - x)^b / B(a, b)`` in logs over the continued fraction of DiDonato and Morris (1992,
*ACM TOMS* 18(3), Algorithm 708), which converges within a few terms that far below the mean.
A band edge at such a level comes from Newton's method on the concave log tail (Bagnoli and
Bergstrom 2005, *Economic Theory* 26, theorem 1).
"""

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: The global levels whose large-sample correction constant ``c`` is published.
_CORRECTION = {0.01: 1.591, 0.05: 1.3, 0.1: 1.1}
#: From this many points on, the local level is the asymptotic approximation: below it the exact
#: search's first call costs at most about 0.2 s on one core, and its cost grows faster than n.
_ASYMPTOTIC_FROM = 250
#: The exact search stops once ``eta`` is known to this relative tolerance.
_SEARCH_RTOL = 1e-9
#: The least global level ``bonferroni_binomial_band`` and ``monte_carlo_level`` accept.
_LEAST_ALPHA = 1e-12


@dataclass(frozen=True)
class UniformBand:
    """A ``1 - alpha`` band for the ``n`` sorted values of a uniform sample, held at once.

    ``lower``, ``upper`` and ``median`` are read-only arrays of length ``n`` on the
    probability scale, ascending: the ``local_level / 2``, ``1 - local_level / 2`` and
    ``1 / 2`` quantiles of ``Beta(i, n + 1 - i)``, the law of the ``i``-th order statistic.
    Map them through a law's quantile function to draw the band on that law's scale.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    median: NDArray[np.float64]
    local_level: float
    alpha: float


def _bounds(n: int, eta: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The ``eta / 2`` and ``1 - eta / 2`` quantiles of ``Beta(i, n + 1 - i)``, ``i = 1..n``.

    SciPy's inverse can return NaN deep in a tail at a level float64 still holds. Such an
    edge is found in logs by ``_log_beta_quantile``, an upper one as one minus the lower edge
    of ``Beta(n + 1 - i, i)``; below ``1 / e`` every such quantile lies below its mean.
    """
    import scipy.linalg
    from scipy import special

    i = np.arange(1, n + 1, dtype=np.float64)
    lower = special.betaincinv(i, n + 1 - i, eta / 2.0)
    upper = special.betainccinv(i, n + 1 - i, eta / 2.0)
    if 0.0 < eta / 2.0 < math.exp(-1.0):
        log_p, low, high = math.log(eta / 2.0), ~np.isfinite(lower), ~np.isfinite(upper)
        lower[low] = np.exp(_log_beta_quantile(i[low], n + 1.0 - i[low], log_p))
        upper[high] = -np.expm1(_log_beta_quantile(n + 1.0 - i[high], i[high], log_p))
    return lower, upper


def _global_level(lower: NDArray[np.float64], upper: NDArray[np.float64]) -> float:
    """The chance that some order statistic of ``n`` iid uniforms leaves ``(lower_i, upper_i)``.

    Weine et al. (2023), appendix A. With ``N(t)`` the count of values at or below ``t``, the
    ``i``-th order statistic lies above ``lower_i`` when ``N(lower_i) <= i - 1`` and below
    ``upper_i`` when ``N(upper_i) >= i``. Given ``N(s) = m`` the other ``n - m`` values are iid
    uniform on ``(s, 1)``, so at the next bound ``t`` the count is ``m`` plus a
    ``Binomial(n - m, (t - s) / (1 - s))``. Walking the ``2 n`` bounds in order carries the law of
    ``N`` over the counts between ``#{upper_i <= t}`` and ``#{lower_i < t}``: a count outside
    breaks a bound already passed or, since ``N`` only grows, one still ahead.
    """
    import scipy.linalg
    from scipy import special

    n = lower.size
    t = np.sort(np.concatenate([lower, upper]))
    least = np.searchsorted(upper, t, side="right")
    most = np.searchsorted(lower, t, side="left")
    if np.any(least > most):
        return 1.0
    log_factorial = special.gammaln(np.arange(n + 1, dtype=np.float64) + 1.0)
    law, first, s = np.ones(1), 0, 0.0
    for k in range(2 * n):
        q = min(1.0, (t[k] - s) / (1.0 - s))
        m = np.arange(first, first + law.size)[:, None]
        j = np.arange(least[k], most[k] + 1)[None, :]
        step = j - m
        reachable = step >= 0
        step = np.where(reachable, step, 0)
        log_pmf = (
            log_factorial[n - m]
            - log_factorial[step]
            - log_factorial[n - j]
            + special.xlogy(step, q)
            + special.xlog1py(n - j, -q)
        )
        law = law @ np.where(reachable, np.exp(log_pmf), 0.0)
        first, s = int(least[k]), float(t[k])
    return 1.0 - float(law.sum())


def _local_level(n: int, alpha: float) -> float:
    """The two-sided level ``eta`` at which each of ``n`` order statistics is tested."""
    if n == 1:
        return alpha
    if n >= _ASYMPTOTIC_FROM:
        log_n = math.log(n)
        loglog_n = math.log(log_n)
        leading = -math.log1p(-alpha) / (2.0 * loglog_n * log_n)
        return leading * (1.0 - _CORRECTION[alpha] * math.log(loglog_n) / loglog_n)
    from scipy import optimize

    # The global level rises with eta, from at most alpha at the Bonferroni level alpha / n to at
    # least alpha at alpha, where the first order statistic alone leaves with chance alpha.
    return float(
        optimize.brentq(
            lambda eta: _global_level(*_bounds(n, eta)) - alpha,
            alpha / n,
            alpha,
            xtol=1e-3 * _SEARCH_RTOL * alpha / n,
            rtol=_SEARCH_RTOL,
        )
    )


# The cache belongs to this module and lives for the process. Its key is the validated
# ``(points, level)``, the whole input: no model state enters the band, so nothing invalidates an
# entry. It holds bytes, which no caller can reach or change; each call reads new arrays over
# them. 32 bands at the residual sampler's cap of 100,000 rows hold under 80 MB.
@functools.lru_cache(maxsize=32)
def _stored_band(points: int, level: float) -> tuple[bytes, bytes, bytes, float]:
    """The band's lower and upper bounds and medians as bytes, and its local level."""
    eta = float(_local_level(points, level))
    lower, upper = _bounds(points, eta)
    return lower.tobytes(), upper.tobytes(), _stored_median(points), eta


# Owned and keyed like ``_stored_band``: the medians depend on the number of points alone.
@functools.lru_cache(maxsize=32)
def _stored_median(points: int) -> bytes:
    """The medians of ``Beta(i, points + 1 - i)``, ``i = 1..points``, as bytes."""
    import scipy.linalg
    from scipy import special

    i = np.arange(1, points + 1, dtype=np.float64)
    return special.betaincinv(i, points + 1 - i, 0.5).tobytes()


def _frozen(stored: bytes) -> NDArray[np.float64]:
    """A new array over immutable bytes: read-only, and no caller can make it writeable."""
    return np.frombuffer(stored, dtype=np.float64)


def equal_local_levels_band(n: int, *, alpha: float = 0.05) -> UniformBand:
    """Return the equal-local-levels band for ``n`` order statistics at global level ``alpha``.

    Under iid uniform values the chance that any order statistic leaves its interval is
    ``alpha``: to a relative ``1e-7`` below 250 points, and from 250 to the accuracy of the
    asymptotic local level (about 1.6% of ``alpha`` high at 250 points, 4.5% at 0.01). The
    band depends on ``n`` and ``alpha`` alone and is cached for the process, so a redraw
    costs nothing. Its arrays are read-only and the caller's own.

    Raises
    ------
    ValueError
        If ``n`` is below one, or ``alpha`` is not 0.01, 0.05 or 0.1.
    """
    points = operator.index(n)
    if points < 1:
        raise ValueError(f"equal_local_levels_band needs at least one point; got n={points}")
    level = float(alpha)
    if level not in _CORRECTION:
        raise ValueError(
            "equal_local_levels_band: alpha must be 0.01, 0.05 or 0.1, the levels whose "
            f"large-sample local level is published; got alpha={alpha!r}"
        )
    lower, upper, median, eta = _stored_band(points, level)
    return UniformBand(
        lower=_frozen(lower),
        upper=_frozen(upper),
        median=_frozen(median),
        local_level=eta,
        alpha=level,
    )


def band_at_level(
    n: int, local_level: float, *, alpha: float = 0.05, log: bool = False
) -> UniformBand:
    """Return the equal-local-levels band for ``n`` order statistics at local level ``local_level``.

    Each order statistic lies between the ``local_level / 2`` and ``1 - local_level / 2``
    quantiles of ``Beta(i, n + 1 - i)``. ``equal_local_levels_band`` sets the local level from
    ``n`` and ``alpha`` for a law with known parameters; this band takes it as given, as
    ``monte_carlo_level`` reads it from replicates, and records ``alpha`` as the global level it
    stands for. At ``equal_local_levels_band(n, alpha=alpha).local_level`` it is that band.

    With ``log``, ``local_level`` is the natural log of the local level, as Barnard's rule
    reads it from ``order_statistic_level(..., log=True)``. Below twice the least normal float
    each edge is found in logs, by Newton's method on the log tail. The band's arrays and its
    ``local_level`` are then float64's roundings: an edge closer to 0 or 1 than float64 resolves
    reads 0 or 1, and a level below its range reads 0.

    Raises
    ------
    ValueError
        If ``n`` is below one, ``local_level`` lies outside ``[0, 1]`` (with ``log``, outside
        ``[-inf, 0]``), or ``alpha`` outside ``(0, 1)``.
    """
    points = operator.index(n)
    if points < 1:
        raise ValueError(f"band_at_level needs at least one point; got n={points}")
    eta, level = float(local_level), float(alpha)
    if log and not -math.inf <= eta <= 0.0:
        raise ValueError(
            f"band_at_level: a log local_level must lie in [-inf, 0]; got {local_level!r}"
        )
    if not log and not 0.0 <= eta <= 1.0:
        raise ValueError(f"band_at_level: local_level must lie in [0, 1]; got {local_level!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"band_at_level: alpha must lie in (0, 1); got {alpha!r}")
    if log and eta < math.log(_LOG_BELOW):
        log_lower = _log_band_lower(points, eta)
        lower, upper, eta = np.exp(log_lower), -np.expm1(log_lower[::-1]), math.exp(eta)
    else:
        eta = math.exp(eta) if log else eta
        lower, upper = _bounds(points, eta)
    lower.setflags(write=False)
    upper.setflags(write=False)
    return UniformBand(
        lower=lower,
        upper=upper,
        median=_frozen(_stored_median(points)),
        local_level=eta,
        alpha=level,
    )


def _beta_tail_levels(u: NDArray[np.float64], i: NDArray[np.float64], n: int) -> NDArray:
    """``2 min(I_u(i, n + 1 - i), 1 - I_u(i, n + 1 - i))``, each tail read from its own side."""
    import scipy.linalg
    from scipy import special

    b = n + 1.0 - i
    return 2.0 * np.minimum(special.betainc(i, b, u), special.betaincc(i, b, u))


def _edge_levels(
    lower: NDArray[np.float64], upper: NDArray[np.float64], i: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """``2 min(I_lower(i, n + 1 - i), 1 - I_upper(i, n + 1 - i))``, the least two-sided tail
    of any value between a band's edges, each tail read at its own edge."""
    import scipy.linalg
    from scipy import special

    b = n + 1.0 - i
    return 2.0 * np.minimum(special.betainc(i, b, lower), special.betaincc(i, b, upper))


#: Order statistics per replicate whose tails an upper bound on its level reads.
_SCREEN_POINTS = 8
#: Replicates screened at once, so the standardised deviations stay a small block.
_SCREEN_BLOCK = 16
#: Trust an edge whose level is at least this times the target: half the factor two, in logs.
_EDGE_MARGIN = math.sqrt(2.0)
#: The least positive normal float64. Below it SciPy's tails carry an absolute error.
_LEAST_NORMAL = float(np.finfo(np.float64).tiny)
#: A tail or level below this is read and carried in logs: half of it, the tail a Q-Q band's
#: edge sits at, is still a normal float.
_LOG_BELOW = 2.0 * _LEAST_NORMAL
_EPS = float(np.finfo(np.float64).eps)
#: Terms of the continued fraction at most. Below the mean it converges within about 80 terms
#: in float64 (Boost.Math's measurement); below float64's range, within a handful.
_FRACTION_TERMS = 100
#: Newton steps of a log band edge at most: measured, at most 8 up to a million order statistics.
_NEWTON_STEPS = 16


def _log_fraction(
    a: NDArray, b: NDArray, x: NDArray, y: NDArray, log_x: NDArray, log_y: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``log I_x(a, b)`` for whole ``a, b >= 1`` and ``x`` below the mean ``a / (a + b)``, with
    ``y = 1 - x`` and both logs given, and the continued fraction's value ``F``.

    ``I_x(a, b) = x^a y^b / (B(a, b) F)`` with ``F = beta_1 + alpha_2 / (beta_2 + ...)``, the
    fraction of DiDonato and Morris (1992, ACM TOMS 18(3), Algorithm 708) as Boost.Math's
    ``ibeta`` uses it, evaluated by the modified Lentz method (Thompson and Barnett 1986). Below
    the mean every ``alpha`` is nonnegative and every ``beta`` positive, so no partial
    denominator vanishes, and the fraction ends at ``b`` terms. The prefix stays in logs, where
    it cannot underflow: its rounding is an absolute error in the log of at most a few ``u``
    times ``a |log x| + b |log y| + |log B(a, b)|``, plus SciPy's ``betaln``, about
    ``u (a + b) log(a + b)``; that is the relative error of the tail.
    """
    import scipy.linalg
    from scipy import special

    f = a * (a * y - b * x + 1.0) / (a + 1.0)
    c, d = f.copy(), np.zeros_like(f)
    live = np.arange(f.size)
    for m in range(1, _FRACTION_TERMS + 1):
        am, bm, xm = a[live], b[live], x[live]
        denom = am + 2.0 * m - 1.0
        alpha = (m * (am + m - 1.0) / denom) * ((am + bm + m - 1.0) / denom) * (bm - m) * xm * xm
        beta = (
            m
            + m * (bm - m) * xm / denom
            + (am + m) * (am * y[live] - bm * xm + 1.0 + m * (2.0 - xm)) / (am + 2.0 * m + 1.0)
        )
        d[live] = 1.0 / (beta + alpha * d[live])
        c[live] = beta + alpha / c[live]
        step = c[live] * d[live]
        f[live] *= step
        live = live[np.abs(step - 1.0) > _EPS]
        if not live.size:
            break
    with np.errstate(invalid="ignore"):
        return a * log_x + b * log_y - special.betaln(a, b) - np.log(f), f


def _logs(tails: NDArray, deep_log: Callable[[NDArray[np.bool_]], NDArray]) -> NDArray:
    """The natural log of ``tails``; each below ``_LOG_BELOW`` is ``deep_log`` of its mask."""
    tails = np.asarray(tails, dtype=np.float64)
    logs = np.empty_like(tails)
    with np.errstate(divide="ignore"):
        np.log(tails, out=logs)
    deep = tails < _LOG_BELOW
    if np.any(deep):
        logs[deep] = deep_log(deep)
    return logs


def _log_beta_tail_levels(u: NDArray[np.float64], i: NDArray[np.float64], n: int) -> NDArray:
    """The natural log of ``_beta_tail_levels``.

    A level below ``_LOG_BELOW`` is read again by ``_log_fraction``. Its lesser tail lies on
    ``u``'s side of the mean ``i / (n + 1)``: ``Beta(i, n + 1 - i)`` is log-concave, so each
    side of its mean has chance at least ``1 / e`` (Grünbaum 1960).
    """
    levels = _beta_tail_levels(u, i, n)
    u, i = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(i, dtype=np.float64))

    def deep_log(deep: NDArray[np.bool_]) -> NDArray:
        ud, id_ = u[deep], i[deep]
        below = ud <= id_ / (n + 1.0)
        with np.errstate(divide="ignore"):
            log_u, log1m_u = np.log(ud), np.log1p(-ud)
        side = (
            np.where(below, id_, n + 1.0 - id_),
            np.where(below, n + 1.0 - id_, id_),
            np.where(below, ud, 1.0 - ud),
            np.where(below, 1.0 - ud, ud),
            np.where(below, log_u, log1m_u),
            np.where(below, log1m_u, log_u),
        )
        return math.log(2.0) + _log_fraction(*side)[0]

    return _logs(levels, deep_log)


def _log_beta_quantile(a: NDArray, b: NDArray, log_p: float) -> NDArray[np.float64]:
    """``log x`` with ``log I_x(a, b) = log_p``, for a quantile below the mean of ``Beta(a, b)``.

    Newton's method in ``t = log x``. ``log X`` has a log-concave density, so ``log I_{e^t}`` is
    concave (Bagnoli and Bergstrom 2005, theorem 1), with slope ``F / y``, ``F`` the continued
    fraction. ``I_x(a, b) <= x^a / (a B(a, b))``, so the first step starts at or below the root
    and every later one rises to it. A step below ``sqrt(eps)`` relative leaves an error of
    about its square.
    """
    import scipy.linalg
    from scipy import special

    t = (log_p + np.log(a) + special.betaln(a, b)) / a
    live = np.flatnonzero(np.isfinite(t))
    for _ in range(_NEWTON_STEPS):
        if not live.size:
            break
        start = t[live]
        y = -np.expm1(start)
        log_tail, fraction = _log_fraction(a[live], b[live], np.exp(start), y, start, np.log(y))
        step = (log_p - log_tail) * y / fraction
        t[live] = start + step
        live = live[np.abs(step) > math.sqrt(_EPS) * np.maximum(1.0, np.abs(start))]
    return t


def _log_band_lower(n: int, log_level: float) -> NDArray[np.float64]:
    """The natural log of each lower edge of the band at local level ``exp(log_level)``: the
    ``exp(log_level) / 2`` quantile of ``Beta(i, n + 1 - i)``, ``i = 1..n``. The upper edges are
    one minus the lower ones reversed, ``1 - lower_{n + 1 - i}``.
    """
    i = np.arange(1, n + 1, dtype=np.float64)
    return _log_beta_quantile(i, n + 1.0 - i, log_level - math.log(2.0))


def _order_statistic_levels(
    u_sorted: NDArray[np.float64], *, resolve: int, log: bool = False
) -> NDArray:
    """Each ``(replicates, n)`` ascending row's level, exact for its ``resolve`` smallest; in
    logs with ``log``, exact below float64's range too.

    A screen of each row's end points and ``_SCREEN_POINTS`` largest standardised deviations
    bounds its level; ``t`` is the ``resolve``-th smallest bound. Every point outside the band
    at ``2 t`` is read, with every point whose edge level is below ``_EDGE_MARGIN t`` or not
    finite, so a row with level at most ``t`` is exact and any other returns the next float
    above ``t``, at most its level. A target below the least normal float reads every point;
    with ``log`` it reads those outside the band at ``2 t`` found in logs, whose Newton edges
    are good to ``1e-10``.
    """
    count, n = u_sorted.shape
    i = np.arange(1, n + 1, dtype=np.float64)
    tails = _log_beta_tail_levels if log else _beta_tail_levels
    if n <= 2 + _SCREEN_POINTS:
        return np.min(tails(u_sorted, i, n), axis=1)
    mean = i / (n + 1.0)
    sd = np.sqrt(i * (n + 1.0 - i) / ((n + 1.0) ** 2 * (n + 2.0)))
    bounds = np.empty(count)
    for start in range(0, count, _SCREEN_BLOCK):
        block = u_sorted[start : start + _SCREEN_BLOCK]
        deviation = np.abs(block - mean) / sd
        largest = np.argpartition(-deviation, _SCREEN_POINTS - 1, axis=1)[:, :_SCREEN_POINTS]
        ends = np.broadcast_to([0, n - 1], (len(block), 2))
        points = np.concatenate([ends, largest], axis=1)
        values = np.take_along_axis(block, points, axis=1)
        bounds[start : start + len(block)] = np.min(tails(values, i[points], n), axis=1)
    target = float(np.partition(bounds, min(resolve, count) - 1)[min(resolve, count) - 1])
    if not (target > -math.inf if log else target >= _LEAST_NORMAL):
        return np.concatenate(
            [
                np.min(tails(u_sorted[start : start + _SCREEN_BLOCK], i, n), axis=1)
                for start in range(0, count, _SCREEN_BLOCK)
            ]
        )
    if log and target < math.log(_LEAST_NORMAL):
        log_lower = _log_band_lower(n, target + math.log(2.0))

        def inside(block: NDArray[np.float64]) -> NDArray[np.bool_]:
            with np.errstate(divide="ignore"):
                return (np.log(block) >= log_lower) & (np.log1p(-block) >= log_lower[::-1])

    else:
        cut = math.exp(target) if log else target
        lower, upper = _bounds(n, min(1.0, 2.0 * cut))
        trusted = _edge_levels(lower, upper, i, n) >= _EDGE_MARGIN * cut

        def inside(block: NDArray[np.float64]) -> NDArray[np.bool_]:
            return (block >= lower) & (block <= upper) & trusted

    levels = np.full(count, np.nextafter(target, np.inf))
    for start in range(0, count, _SCREEN_BLOCK):
        block = u_sorted[start : start + _SCREEN_BLOCK]
        rows, columns = np.nonzero(~inside(block))
        least = np.full(len(block), np.inf)
        np.minimum.at(least, rows, tails(block[rows, columns], i[columns], n))
        exact = least <= target
        levels[start : start + len(block)][exact] = least[exact]
    return levels


def order_statistic_level(u_sorted: ArrayLike, *, log: bool = False) -> float:
    """Return the equal-local-levels statistic of ``n`` values sorted ascending.

    It is ``min_i 2 min(I_u(i, n + 1 - i), 1 - I_u(i, n + 1 - i))`` with ``I`` the
    regularised incomplete beta function: the least two-sided tail of any order statistic
    under its ``Beta(i, n + 1 - i)`` law. A sample leaves ``band_at_level(n, eta)`` exactly
    when this is below ``eta``, the decision Berk and Jones (1979) and Aldor-Noiman et al.
    (2013) make point by point. It is exact: every point whose tail could be the least is read
    (``_order_statistic_levels``).

    With ``log`` it is the statistic's natural log. That stays exact where the statistic itself
    falls below float64's range and returns 0, as it does for a few thousand values far from
    their law. A tail below the least normal float is then read in logs, through DiDonato and
    Morris's continued fraction (1992, Algorithm 708).

    Raises
    ------
    ValueError
        If the values are not one or more finite numbers in ``[0, 1]`` sorted ascending.
    """
    values = np.asarray(u_sorted, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("order_statistic_level needs a one-dimensional sample of one or more")
    if not np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)):
        raise ValueError("order_statistic_level: each value must be a probability in [0, 1]")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("order_statistic_level: the values must be sorted ascending")
    return float(_order_statistic_levels(values[None, :], resolve=1, log=log)[0])


def _monte_carlo_rank(replicates: int, alpha: float) -> int:
    """Barnard's ``k = floor(alpha (B + 1))``; ``alpha (B + 1)`` one rounding below a whole
    number still counts as that number."""
    return math.floor(alpha * (replicates + 1) * (1.0 + 4.0 * 2.0**-53))


def monte_carlo_level(
    null_levels: ArrayLike, *, alpha: float, larger_is_extreme: bool = False
) -> float:
    """Return the level of a Monte Carlo test from ``B`` replicate statistics of its null.

    With ``k = floor(alpha (B + 1))`` it is the ``k``-th smallest replicate statistic, or the
    ``k``-th largest when ``larger_is_extreme``. An observed statistic exchangeable with the
    replicates falls strictly beyond it (below, or above) with chance ``k / (B + 1)``, at most
    ``alpha`` and exactly ``alpha`` when ``alpha (B + 1)`` is a whole number and the statistic
    has no ties (Barnard 1963). Tied with it, an observed statistic is not beyond it, so ties
    only make the test conservative. It reads ranks alone, so the statistics may be the natural
    logs ``log=True`` gives, which keep their order below float64's range; the level is then
    the log of the level. At 99 replicates and 5% it is the 5th. ``alpha`` lies in
    ``[1e-12, 0.5]``: from the least level ``bonferroni_binomial_band`` takes to one half, so
    the rank is at most ``(B + 1) / 2`` and names a replicate.

    Raises
    ------
    ValueError
        If the statistics are not a one-dimensional array without NaN, ``alpha`` lies outside
        ``[1e-12, 0.5]``, or there are too few replicates for any rank at ``alpha``.
    """
    values = np.asarray(null_levels, dtype=np.float64)
    level = float(alpha)
    if values.ndim != 1 or np.any(np.isnan(values)):
        raise ValueError("monte_carlo_level needs a one-dimensional array of statistics")
    if not _LEAST_ALPHA <= level <= 0.5:
        raise ValueError(f"monte_carlo_level: alpha must lie in [1e-12, 0.5]; got {alpha!r}")
    rank = _monte_carlo_rank(values.size, level)
    if rank < 1:
        least = max(1, math.ceil(1.0 / level) - 1)
        while _monte_carlo_rank(least, level) < 1:
            least += 1
        raise ValueError(
            f"monte_carlo_level at alpha={level!r} needs at least {least} replicates; "
            f"got {values.size}"
        )
    ordered = np.sort(values)
    return float(ordered[-rank] if larger_is_extreme else ordered[rank - 1])


def _monte_carlo_order_statistic_level(
    u_sorted: NDArray[np.float64], *, alpha: float, log: bool = False
) -> float:
    """``monte_carlo_level`` of the equal-local-levels statistics of a ``(replicates, n)`` block,
    each row one replicate's values sorted ascending; with ``log``, of their natural logs.

    It is exact: ``_order_statistic_levels`` resolves the rank Barnard's rule reads, and each
    row it leaves at a floor stands above every resolved one.
    """
    rank = _monte_carlo_rank(len(u_sorted), alpha)
    levels = _order_statistic_levels(u_sorted, resolve=max(rank, 1), log=log)
    return monte_carlo_level(levels, alpha=alpha)


def _least_count(
    trials: NDArray[np.float64], holds: Callable[[NDArray[np.int64]], NDArray[np.bool_]]
) -> NDArray[np.float64]:
    """The least count ``k`` in ``[0, trials]`` at which ``holds(k)``, cell by cell.

    ``holds`` is monotone in ``k`` and true at ``trials``. The bisection runs in integers: its
    midpoint ``low + (high - low) // 2`` lies below ``high`` while the interval is open, so
    every step shrinks it, and at most ``log2(trials) + 1`` steps end it, one probability per
    cell each.
    """
    low = np.zeros(trials.shape, dtype=np.int64)
    high = trials.astype(np.int64)
    while np.any(low < high):
        middle = low + (high - low) // 2
        met = holds(middle)
        high = np.where(met, middle, high)
        low = np.where(met, low, middle + 1)
    return low.astype(np.float64)


def _binomial_law(
    caller: str, trials: ArrayLike, probability: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``trials`` and ``probability`` broadcast together as float64, once they name binomial laws.

    Raises ``ValueError``, naming ``caller``, if a count of trials is not a whole number from 0
    to ``2**53 - 1`` or a probability lies outside ``[0, 1]``.
    """
    n_trials, p = np.broadcast_arrays(
        np.asarray(trials, dtype=np.float64), np.asarray(probability, dtype=np.float64)
    )
    whole = np.isfinite(n_trials) & (n_trials == np.floor(n_trials))
    if not np.all(whole & (n_trials >= 0) & (n_trials < 2.0**53)):
        raise ValueError(f"{caller}: trials must be whole numbers from 0 to 2**53 - 1")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError(f"{caller}: each probability must lie in [0, 1]")
    return n_trials, p


def bonferroni_binomial_band(
    trials: ArrayLike,
    probability: ArrayLike,
    *,
    alpha: float = 0.05,
    cut: float | None = None,
    log: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return count bounds for ``m`` binomial cells that hold for all cells at once.

    Cell ``k``'s count is ``Binomial(trials[k], probability[k])``, and with
    ``cut = alpha / (2 m)`` its bounds are the ``cut`` and ``1 - cut`` quantiles of that law,
    each decided in its own tail: ``lower`` is the least count with ``P(X <= lower) >= cut``,
    so ``P(X < lower) < cut``, and ``upper`` the least with ``P(X > upper) <= cut``. So any
    cell leaves with chance at most ``alpha``, whatever the dependence between cells. Each
    tail is computed and compared with the cut in float64, so a tail equal to the cut to the
    last bit is decided in float64. ``trials`` and ``probability`` broadcast together, and the
    bounds take their shape.

    A given ``cut``, such as ``monte_carlo_level`` reads from replicates of
    ``binomial_cell_level``, replaces ``alpha / (2 m)``, and the band holds exactly the counts
    whose level reaches it: ``P(X <= k) >= cut`` and ``P(X >= k) >= cut``. So ``upper`` is
    then the least count with ``P(X > upper) < cut``, and a count leaves the band exactly when
    its ``binomial_cell_level`` is below ``cut``.

    With ``log``, ``cut`` is the natural log of the cut, as Barnard's rule reads it from
    ``binomial_cell_level(..., log=True)``, and each tail is compared with it in logs, so a cut
    below float64's range still decides every count.

    Raises
    ------
    ValueError
        If a count of trials is not a whole number from 0 to ``2**53 - 1``, a probability
        lies outside ``[0, 1]``, ``alpha`` is below ``1e-12`` or not below 1, ``cut`` lies
        outside ``[0, 1]`` (with ``log``, outside ``[-inf, 0]``), or ``log`` has no ``cut``.
    """
    import scipy.spatial
    from scipy import stats

    level = float(alpha)
    if not _LEAST_ALPHA <= level < 1.0:
        raise ValueError(
            f"bonferroni_binomial_band: alpha must be at least 1e-12 and below 1; got {alpha!r}"
        )
    n_trials, p = _binomial_law("bonferroni_binomial_band", trials, probability)
    if log and (cut is None or not -math.inf <= float(cut) <= 0.0):
        raise ValueError(
            f"bonferroni_binomial_band: a log cut must lie in [-inf, 0]; got cut={cut!r}"
        )
    if not log and cut is not None and not 0.0 <= float(cut) <= 1.0:
        raise ValueError(f"bonferroni_binomial_band: cut must lie in [0, 1]; got {cut!r}")
    if n_trials.size == 0:
        return np.zeros(n_trials.shape), np.zeros(n_trials.shape)
    if cut is None:
        tail = level / (2.0 * n_trials.size)
        lower = _least_count(n_trials, lambda k: stats.binom.cdf(k, n_trials, p) >= tail)
        upper = _least_count(n_trials, lambda k: stats.binom.sf(k, n_trials, p) <= tail)
        return lower, upper
    given = float(cut)
    at_most, above = (
        (_log_binomial_at_most, _log_binomial_above) if log else (stats.binom.cdf, stats.binom.sf)
    )
    lower = _least_count(n_trials, lambda k: at_most(k, n_trials, p) >= given)
    # ``P(X > k) < cut`` holds at ``trials`` for any positive cut; at cut 0 every count is in.
    upper = _least_count(n_trials, lambda k: (above(k, n_trials, p) < given) | (k >= n_trials))
    return lower, upper


def _log_binomial_at_most(k: ArrayLike, trials: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """``log P(X <= k)`` for ``X ~ Binomial(trials, p)``: SciPy's tail, and below
    ``_LOG_BELOW`` the continued fraction of ``I_{1-p}(trials - k, k + 1)``, its lower tail."""
    import scipy.spatial
    from scipy import stats

    k, trials, p = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (k, trials, p)))

    def deep_log(deep: NDArray[np.bool_]) -> NDArray:
        kd, nd, pd = k[deep], trials[deep], p[deep]
        with np.errstate(divide="ignore"):
            return _log_fraction(nd - kd, kd + 1.0, 1.0 - pd, pd, np.log1p(-pd), np.log(pd))[0]

    return _logs(stats.binom.cdf(k, trials, p), deep_log)


def _log_binomial_at_least(k: ArrayLike, trials: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """``log P(X >= k)`` for ``X ~ Binomial(trials, p)``: SciPy's tail, and below
    ``_LOG_BELOW`` the continued fraction of ``I_p(k, trials - k + 1)``; above ``trials`` the
    chance is 0."""
    import scipy.spatial
    from scipy import stats

    k, trials, p = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (k, trials, p)))

    def deep_log(deep: NDArray[np.bool_]) -> NDArray:
        nd, pd = trials[deep], p[deep]
        kd = np.minimum(k[deep], nd)
        with np.errstate(divide="ignore"):
            logs = _log_fraction(kd, nd - kd + 1.0, pd, 1.0 - pd, np.log(pd), np.log1p(-pd))[0]
        return np.where(k[deep] > nd, -np.inf, logs)

    return _logs(stats.binom.sf(k - 1.0, trials, p), deep_log)


def _log_binomial_above(k: ArrayLike, trials: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    """``log P(X > k) = log P(X >= k + 1)``."""
    return _log_binomial_at_least(np.asarray(k, dtype=np.float64) + 1.0, trials, p)


def _binomial_cell_levels(
    counts: ArrayLike, trials: ArrayLike, probability: ArrayLike, *, log: bool = False
) -> NDArray[np.float64]:
    """``min(P(X <= k), P(X >= k))`` of every count, its least over the last axis; with
    ``log``, its natural log, exact below float64's range too."""
    import scipy.spatial
    from scipy import stats

    k = np.asarray(counts, dtype=np.float64)
    if log:
        tails = np.minimum(
            _log_binomial_at_most(k, trials, probability),
            _log_binomial_at_least(k, trials, probability),
        )
    else:
        tails = np.minimum(
            stats.binom.cdf(k, trials, probability), stats.binom.sf(k - 1.0, trials, probability)
        )
    return np.min(tails, axis=-1)


def binomial_cell_level(
    counts: ArrayLike, trials: ArrayLike, probability: ArrayLike, *, log: bool = False
) -> float:
    """Return the least one-sided binomial tail of ``m`` cell counts.

    Cell ``k``'s count is read against ``Binomial(trials[k], probability[k])`` by the lesser of
    ``P(X <= count)`` and ``P(X >= count)``; the statistic is the least over cells. A count
    leaves ``bonferroni_binomial_band(trials, probability, cut=c)`` exactly when its level is
    below ``c``, and with ``c = alpha / (2 m)`` that is the Bonferroni band's level.

    With ``log`` it is the statistic's natural log. That stays exact where the statistic itself
    falls below float64's range and returns 0, as it does for counts of a few thousand far from
    their law, and it reaches ``bonferroni_binomial_band(..., cut=c, log=True)`` the same way.

    Raises
    ------
    ValueError
        If there is no cell, a count of trials or a probability is one
        ``bonferroni_binomial_band`` refuses, or a count is not a whole number from 0 to its
        trials.
    """
    n_trials, p = _binomial_law("binomial_cell_level", trials, probability)
    k, n_trials, p = np.broadcast_arrays(np.asarray(counts, dtype=np.float64), n_trials, p)
    if k.size < 1:
        raise ValueError("binomial_cell_level needs at least one cell")
    if not np.all(np.isfinite(k) & (k == np.floor(k)) & (k >= 0.0) & (k <= n_trials)):
        raise ValueError(
            "binomial_cell_level: each count must be a whole number from 0 to its trials"
        )
    return float(np.min(_binomial_cell_levels(k.ravel(), n_trials.ravel(), p.ravel(), log=log)))


__all__ = [
    "UniformBand",
    "band_at_level",
    "binomial_cell_level",
    "bonferroni_binomial_band",
    "equal_local_levels_band",
    "monte_carlo_level",
    "order_statistic_level",
]
