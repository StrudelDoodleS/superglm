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
* from 2 to 99 points, ``eta`` is calibrated by simulation, the procedure of
  Aldor-Noiman et al. (calibration for simultaneity: Buja and Rolke,
  *Calibration for simultaneity*, Wharton technical report). Each of 4,000
  uniform samples gives the smallest two-sided local level at which it leaves
  the band, and ``eta`` is the ``alpha``-quantile of those levels;
* from 100 points, the asymptotic local level of Gontscharuk and Finner
  (2017), *Communications in Statistics - Theory and Methods* 46(5),
  2332-2342, with its second-order correction, at the constants Weine et al.
  (section 2.4) checked: 1.591, 1.3 and 1.1 at ``alpha`` = 0.01, 0.05 and
  0.1. Those are the levels the band accepts.

The global level is ``alpha`` up to the error of each rule. At ``alpha = 0.05``
the calibration's Monte Carlo standard error is about 7% of ``alpha`` (16% at
0.01), and the asymptotic level runs about 5% of ``alpha`` high at 100 points
(10% at 0.01), within 1% by 500 points (3% at 0.01).

``bonferroni_binomial_band`` bounds the counts of ``m`` cells, each
``Binomial(trials, probability)``, by the ``alpha / (2 m)`` and
``1 - alpha / (2 m)`` quantiles of its own law. By Bonferroni's inequality
(Bonferroni 1936, *Pubbl. R. Ist. Sup. Sci. Econ. Comm. Firenze* 8, 1-62) the
chance that any cell leaves is at most ``alpha`` whatever the dependence
between cells, so the band holds for the multinomial bins of a histogram too.

Both assume iid continuous values on the probability scale from a law whose
parameters are known. Parameters estimated from the same rows pull the values
toward the law, so a check on in-sample rows is slightly conservative (Weine
et al., sections 2.6 and 4).
"""

from __future__ import annotations

import functools
import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: The global levels whose large-sample correction constant ``c`` is published.
_CORRECTION = {0.01: 1.591, 0.05: 1.3, 0.1: 1.1}
#: From this many points on, the local level is the asymptotic approximation.
_ASYMPTOTIC_FROM = 100
#: Uniform samples drawn to calibrate the local level below ``_ASYMPTOTIC_FROM`` points.
_CALIBRATION_SAMPLES = 4000
_CALIBRATION_SEED = 20131119


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


def _local_level(n: int, alpha: float) -> float:
    """The two-sided level ``eta`` at which each of ``n`` order statistics is tested."""
    if n == 1:
        return alpha
    if n >= _ASYMPTOTIC_FROM:
        log_n = math.log(n)
        loglog_n = math.log(log_n)
        leading = -math.log1p(-alpha) / (2.0 * loglog_n * log_n)
        return leading * (1.0 - _CORRECTION[alpha] * math.log(loglog_n) / loglog_n)
    from scipy import special

    rng = np.random.default_rng(np.random.SeedSequence([n, _CALIBRATION_SEED]))
    u = np.sort(rng.uniform(size=(_CALIBRATION_SAMPLES, n)), axis=1)
    i = np.arange(1, n + 1, dtype=np.float64)
    tail = np.minimum(special.betainc(i, n + 1 - i, u), special.betaincc(i, n + 1 - i, u))
    escape_level = 2.0 * tail.min(axis=1)
    return float(np.quantile(escape_level, alpha))


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values.setflags(write=False)
    return values


# The cache belongs to this module and lives for the process. Its key, ``(n, alpha)``, is the
# whole input: no model state enters the band, so nothing invalidates an entry. Its arrays are
# read-only, so every caller shares them safely; 32 bands at the residual sampler's cap of
# 100,000 rows hold under 80 MB.
@functools.lru_cache(maxsize=32)
def equal_local_levels_band(n: int, *, alpha: float = 0.05) -> UniformBand:
    """Return the equal-local-levels band for ``n`` order statistics at global level ``alpha``.

    Under iid uniform values the chance that any order statistic leaves its interval is
    ``alpha``. The result depends on ``n`` and ``alpha`` alone and is cached for the
    process, so a redraw costs nothing.

    Raises
    ------
    ValueError
        If ``n`` is below one, or ``alpha`` is not 0.01, 0.05 or 0.1.
    """
    from scipy import special

    points = operator.index(n)
    if points < 1:
        raise ValueError(f"equal_local_levels_band needs at least one point; got n={points}")
    level = float(alpha)
    if level not in _CORRECTION:
        raise ValueError(
            "equal_local_levels_band: alpha must be 0.01, 0.05 or 0.1, the levels whose "
            f"large-sample local level is published; got alpha={alpha!r}"
        )
    eta = float(_local_level(points, level))
    i = np.arange(1, points + 1, dtype=np.float64)
    j = points + 1 - i
    return UniformBand(
        lower=_readonly(special.betaincinv(i, j, eta / 2.0)),
        upper=_readonly(special.betainccinv(i, j, eta / 2.0)),
        median=_readonly(special.betaincinv(i, j, 0.5)),
        local_level=eta,
        alpha=level,
    )


def bonferroni_binomial_band(
    trials: ArrayLike, probability: ArrayLike, *, alpha: float = 0.05
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return count bounds for ``m`` binomial cells that hold for all cells at once.

    Cell ``k``'s count is ``Binomial(trials[k], probability[k])``, and its bounds are the
    ``alpha / (2 m)`` and ``1 - alpha / (2 m)`` quantiles of that law: the chance that a
    cell's count falls below its ``lower`` is under ``alpha / (2 m)``, and above its
    ``upper`` at most ``alpha / (2 m)``. So any cell leaves with chance at most ``alpha``,
    whatever the dependence between cells. ``trials`` and ``probability`` broadcast
    together, and the bounds take their shape.

    Raises
    ------
    ValueError
        If a count of trials is negative or not whole, a probability lies outside
        ``[0, 1]``, or ``alpha`` outside ``(0, 1)``.
    """
    from scipy import stats

    n_trials, p = np.broadcast_arrays(
        np.asarray(trials, dtype=np.float64), np.asarray(probability, dtype=np.float64)
    )
    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValueError(f"bonferroni_binomial_band: alpha must lie in (0, 1); got {alpha!r}")
    if not np.all(np.isfinite(n_trials) & (n_trials >= 0) & (n_trials == np.floor(n_trials))):
        raise ValueError("bonferroni_binomial_band: trials must be whole numbers, zero or more")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("bonferroni_binomial_band: each probability must lie in [0, 1]")
    if n_trials.size == 0:
        return np.zeros(n_trials.shape), np.zeros(n_trials.shape)
    cut = level / (2.0 * n_trials.size)
    lower = np.asarray(stats.binom.ppf(cut, n_trials, p), dtype=np.float64)
    upper = np.asarray(stats.binom.isf(cut, n_trials, p), dtype=np.float64)
    return lower, upper


__all__ = ["UniformBand", "bonferroni_binomial_band", "equal_local_levels_band"]
