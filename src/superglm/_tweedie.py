"""Tweedie (1 < p < 2) density, dispersion profile and simulation on one series.

Every EDM satisfies log f(y; mu, phi/w) = log f(y; y, phi/w) - w d(y, mu) / (2 phi).
The saturated term depends on (y, w, p) and phi only, and a zero response
contributes nothing to it, so one prepared set of positive rows serves the
density, the fitted/null pair and the dispersion profile.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, lambertw, polygamma

from superglm._tweedie_series import _LOG_CUTOFF, log_peak_index, series_moments

_LOG_TWO_PI = math.log(2.0 * math.pi)
_EPS = float(np.finfo(np.float64).eps)
_TINY = float(np.finfo(np.float64).tiny)
# The series' float64 error in l_sat stays below _SERIES_ERROR eps (a + 1) j log j
# against the 50-digit oracle (benchmarks/tweedie_series_oracle.py) over
# p in [1.001, 1.99] and peak indices j from 1e2 to 1e6: log W ~ (a + 1) j
# cancels against the canonical term, both built from terms of that size.
_SERIES_ERROR = 2.5
# The corrected saddlepoint's remainder is B3 e^3, e = 1/((2 - p) j): |B3| = 1/360
# at p -> 1 and p -> 2 (Stirling's series), smaller between on the same oracle,
# and exactly 21/8192 at p = 1.5 (DLMF 10.40.1 for I_1).
_SADDLE_REMAINDER = 1.0 / 360.0
_NEWTON_MAX_STEPS = 60
_NEWTON_MAX_STEP = 2.0
_NEWTON_STEP_TOL = 1e-8
# Row passes one lattice comparison may make, about five seconds of series work.
# Its grid grows like 1 / (p - 1); past this bound the power is refused, not
# searched. A pass costs about 35 us plus 0.1 us per row (measured from 6 to
# 30,000 rows at p from 1.00001 to 1.5), so each is charged 500 rows more.
_SCAN_WORK_LIMIT = float(1 << 26)
_PASS_OVERHEAD_ROWS = 500
_POISSON_LAM_MAX = float(np.iinfo(np.int64).max) - 10.0 * np.sqrt(float(np.iinfo(np.int64).max))


@dataclass(frozen=True)
class TweedieRows:
    """Phi-invariant saturated state of the positive responses at one power."""

    p: float
    a: float
    log_t_unit_phi: NDArray
    log_y: NDArray
    saturated_canonical: NDArray
    count: NDArray | None
    # solve_log_phi results by (D, M). A REML optimizer re-profiles its accepted
    # line-search point with identical (Dp, Mp) at the next iteration's start; the
    # solve is a pure function of these immutable rows and the two floats, so it
    # lives as long as the rows do.
    phi_solves: dict[tuple[float, float], PhiSolve] = field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def prepare(
        cls, y: NDArray, weights: NDArray, p: float, *, frequency: bool = False
    ) -> TweedieRows:
        """Hoist the positive rows; prior weights enter the density, frequency counts multiply it."""
        positive = (y > 0.0) & (weights > 0.0) if frequency else y > 0.0
        y_positive = y[positive]
        a = (2.0 - p) / (p - 1.0)
        log_y = np.log(y_positive)
        log_t = a * (log_y - math.log(p - 1.0)) - math.log(2.0 - p)
        # c w may overflow; only rows past the switch reach that size at a
        # representable phi, and those take its logarithm from log t instead.
        with np.errstate(over="ignore"):
            canonical = np.power(y_positive, 2.0 - p) / ((1.0 - p) * (2.0 - p))
            if frequency:
                return cls(p, a, log_t, log_y, canonical, weights[positive])
            w = weights[positive]
            return cls(p, a, log_t + (a + 1.0) * np.log(w), log_y, canonical * w, None)

    @classmethod
    def profile(
        cls,
        y: NDArray,
        weights: NDArray,
        p: float,
        *,
        frequency: bool = False,
        grouping: tuple[NDArray, NDArray] | None | bool = True,
    ) -> TweedieRows:
        """Rows for the dispersion profile, with repeated positive rows collapsed into counts.

        Equal (y, w) rows, or equal y under frequency counts, give equal row
        terms, so every series pass costs O(distinct rows). A book in which no
        row repeats keeps its rows, and its profile, bitwise unchanged.
        ``grouping`` is `profile_grouping`'s result for these (y, w), which does not
        depend on p; True computes it here.
        """
        rows = cls.prepare(y, weights, p, frequency=frequency)
        if grouping is True:
            grouping = cls.profile_grouping(y, weights, frequency=frequency)
        if grouping is None or grouping is False:
            return rows
        first, group = grouping
        count = np.bincount(group, weights=rows.count, minlength=first.size)
        canonical = rows.saturated_canonical[first]
        return cls(p, rows.a, rows.log_t_unit_phi[first], rows.log_y[first], canonical, count)

    @staticmethod
    def profile_grouping(
        y: NDArray, weights: NDArray, *, frequency: bool = False
    ) -> tuple[NDArray, NDArray] | None:
        """First index and group of each positive row's repeat class, or None if none repeats."""
        positive = (y > 0.0) & (weights > 0.0) if frequency else y > 0.0
        y_positive = y[positive]
        if np.unique(y_positive).size == y_positive.size:
            return None
        key = y_positive[:, None] if frequency else np.column_stack([y_positive, weights[positive]])
        _, first, group = np.unique(key, axis=0, return_index=True, return_inverse=True)
        if first.size == y_positive.size:
            # Repeated y with distinct w: nothing merges, so keep the rows as they are.
            return None
        return first, group.reshape(-1)

    @property
    def size(self) -> float:
        """Likelihood size of the positive rows: their number, or their total count."""
        return float(self.log_y.size) if self.count is None else float(np.sum(self.count))

    def row_saturated(self, phi: float) -> tuple[NDArray, NDArray, NDArray]:
        """Per row: l_sat, T = d(-l_sat)/d log phi and dT/d log phi at ``phi``."""
        log_t = self.log_t_unit_phi - (self.a + 1.0) * math.log(phi)
        ok, log_w, mean_j, var_j = series_moments(
            log_t, self.a, max_mode=saddlepoint_switch(self.p)
        )
        # |c w / phi| = (a + 1) j is at most (a + 1) times the switch on summed
        # rows; past it the quotient may overflow and the saddlepoint takes its log.
        with np.errstate(over="ignore"):
            canonical = self.saturated_canonical / phi
        unrepresented = np.flatnonzero(ok & ~np.isfinite(canonical))
        canonical[unrepresented] = -np.exp(_log_negative_canonical(self.p, log_t[unrepresented]))
        inverse_r = self.a + 1.0
        value = log_w - self.log_y + canonical
        score = mean_j * inverse_r + canonical
        slope = -var_j * inverse_r**2 - canonical
        # The closed forms take log |c w / phi| of the quotient where that is a
        # normal number, and from log t, whose rounding grows with |log w| and
        # |log phi|, only where it overflowed.
        past = np.flatnonzero(~ok)
        negative = -canonical[past]
        normal = np.isfinite(negative) & (negative >= _TINY)
        log_negative = np.empty_like(negative)
        log_negative[normal] = np.log(negative[normal])
        log_negative[~normal] = _log_negative_canonical(self.p, log_t[past][~normal])
        # Below the switch the series refuses a row only at its work bound, which
        # binds there only for 2 - p < 1.2e-9: those rows take the p -> 2 limit.
        # The kernel's own peak index decides which side of the switch a row is.
        switch = math.log(saddlepoint_switch(self.p))
        capped = log_peak_index(log_t[past], self.a) <= switch
        # An empty arm is skipped: its scipy calls cost as much as a small book's pass.
        for arm, taken in ((_corrected_saddlepoint, ~capped), (_gamma_limit, capped)):
            if not taken.any():
                continue
            rows = past[taken]
            value[rows], score[rows], slope[rows] = arm(
                self.p, log_negative[taken], self.log_y[rows]
            )
        return value, score, slope

    @cached_property
    def peak_groups(self) -> tuple[NDArray, NDArray]:
        """Distinct log peak indices at phi = 1, and the likelihood size sharing each.

        Rows with one peak index (equal y and w, or a counted row) carry one
        lattice phase at every phi, so their lattice terms bend Q together.
        """
        log_peak = math.log(self.p - 1.0) + _log_negative_canonical(self.p, self.log_t_unit_phi)
        count = np.ones_like(log_peak) if self.count is None else self.count
        distinct, group = np.unique(log_peak, return_inverse=True)
        return distinct, np.bincount(group, weights=count, minlength=distinct.size)

    def saturated(self, phi: float) -> tuple[float, float, float]:
        value, score, slope = self.row_saturated(phi)
        if self.count is None:
            return float(np.sum(value)), float(np.sum(score)), float(np.sum(slope))
        return float(self.count @ value), float(self.count @ score), float(self.count @ slope)


def saddlepoint_switch(p: float) -> float:
    """Peak index above which the corrected saddlepoint beats the series in float64.

    Solves _SERIES_ERROR eps (a + 1) j log j = _SADDLE_REMAINDER / ((2 - p) j)^3,
    that is j^4 log j = K, taking log j ~ log(K) / 4 inside the slowly varying
    factor (about a 2% shift in j). Below 37 (a + 1) / (2 pi^2), Var J ~ j / (a + 1) is
    small enough that the sum over integer j departs from the saddlepoint's
    integral by the lattice term 2 exp(-2 pi^2 Var J) (Poisson summation); that
    floor binds as p -> 1, where the density becomes the Poisson lattice.
    """
    a_plus_one = 1.0 / (p - 1.0)
    log_k = math.log(_SADDLE_REMAINDER / (_SERIES_ERROR * _EPS * a_plus_one * (2.0 - p) ** 3))
    # K < e^4 only for p < 1 + 1.1e-11, where the lattice floor is far larger.
    switch = math.exp(0.25 * (log_k - math.log(max(1.0, 0.25 * log_k))))
    return max(switch, _LOG_CUTOFF * a_plus_one / (2.0 * math.pi**2))


def _log_negative_canonical(p: float, log_t: NDArray) -> NDArray:
    """log |c w / phi| from the series' log t, finite wherever log t is.

    With 2 - p = a / (a + 1), log t / (a + 1) differs from
    log |c w / phi| = (2 - p) log y + log w - log phi - log((p - 1)(2 - p)) only
    by the constant (p - 1) log(p - 1) + (2 - p) log(2 - p).
    """
    return (p - 1.0) * log_t - (p - 1.0) * math.log(p - 1.0) - (2.0 - p) * math.log(2.0 - p)


def _corrected_saddlepoint(
    p: float, log_negative_canonical: NDArray, log_y: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """l_sat, T and T' of rows past the switch: the saddlepoint with two corrections.

    The small-dispersion saddlepoint (Jorgensen 1997, Ch. 3) is
    -(1/2) log(2 pi phi y^p / w) = (1/2) log((p-1)(2-p) |c w / phi| / (2 pi)) - log y.
    Daniels' (1954) expansion adds A e + B e^2, e = 1/((2-p) j) at the peak index
    j = w y^(2-p) / ((2-p) phi) = (p-1) |c w / phi|. The Tweedie standardized
    cumulants are rho_r = e^(r/2 - 1) prod_{k=1}^{r-2} (k p - k + 1), so
    A = rho4/8 - 5 rho3^2/24 = p (p-3)/24, and the second-order term (Kato, Sekine
    & Yoshikawa 2014, Prop. 16) less A^2/2 is B = -p (p-1)(p-2)(p-3)/48.
    e is proportional to phi, so T = 1/2 - A e - 2 B e^2 and T' = -A e - 4 B e^2.
    Taking log |c w / phi| keeps rows finite where phi and w leave the float range.
    """
    dispersion = np.exp(-log_negative_canonical) / ((p - 1.0) * (2.0 - p))
    first = p * (p - 3.0) / 24.0 * dispersion
    second = -p * (p - 1.0) * (p - 2.0) * (p - 3.0) / 48.0 * dispersion**2
    log_saddle_scale = math.log((p - 1.0) * (2.0 - p) / (2.0 * math.pi))
    value = 0.5 * (log_saddle_scale + log_negative_canonical) - log_y + first + second
    return value, 0.5 - first - 2.0 * second, -first - 4.0 * second


def _gamma_limit(
    p: float, log_negative_canonical: NDArray, log_y: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """l_sat, T and T' at the p -> 2 limit, for rows the series' work bound refused.

    Given N = n jumps, y is Gamma(n a, gamma) with N ~ Poisson(lambda), and at
    mu = y, y / gamma = lambda a = s = (2 - p) |c w / phi|. As p -> 2 the shape
    n a concentrates at s with variance s a, so l_sat is the Gamma(s) log density
    at its mean, s log s - s - log Gamma(s) - log y, short by a s h(s) / 2 + O(a^2),
    h = (log s - psi(s))^2 - psi'(s): relatively O(a (1 + |log s|)). Where the
    work bound binds below the switch, a < 1.2e-9 and s < 4, and the corrected
    saddlepoint's e = 1/((2 - p) j) is order one. s is proportional to 1/phi, so
    T = s (log s - psi(s)) and T' = -s dT/ds.
    """
    log_shape = math.log(2.0 - p) + log_negative_canonical
    shape = np.exp(log_shape)
    excess = log_shape - digamma(shape)
    value = shape * (log_shape - 1.0) - gammaln(shape) - log_y
    slope = -shape * (excess + 1.0 - shape * polygamma(1, shape))
    return value, shape * excess, slope


@dataclass(frozen=True)
class PhiSolve:
    """Profiled dispersion: phi, Q at the optimum, Q'' in log phi and the series passes used.

    ``curvature`` is Q'' at the iterate before the last move, which is at most
    the step tolerance from log phi. ``n_passes`` includes any lattice scan.
    """

    phi: float
    criterion: float
    curvature: float
    n_passes: int


def solve_log_phi(rows: TweedieRows, deviance: float, nullity: float = 0.0) -> PhiSolve:
    """Minimise Q(u) = D e^-u / 2 - l_sat(e^u) - (M / 2)(log 2 pi + u) over u = log phi.

    M = 0 gives the maximum-likelihood dispersion at a fitted mean; D = Dp and
    M = Mp give Wood (2011) Eq. 4's REML scale term. Newton on the analytic
    score Q'(u) = -D e^-u / 2 + T(u) - M / 2, one series pass per step,
    safeguarded by bisection inside the sign-change bracket (Press et al.,
    rtsafe). Q is not convex near p = 1, where the density approaches the
    Poisson lattice (Dunn & Smyth 2005, Sec. 8), so where a group of rows
    sharing a peak index can bend Q the Newton root is compared with every other
    local minimum (`_lattice_scan`). Its model: rows share a phase only within an
    exact group (near-harmonic groups, such as integer responses, are seen where
    any is banded), the bending is measured against the curvature at the Newton
    root, and Q' is monotone between banded stretches. A comparison whose grid
    would take more than _SCAN_WORK_LIMIT row passes raises
    NearPoissonDispersionError, which a power search scores infeasible and a
    fixed-power fit raises; polishing its brackets adds a bracketed Newton each.
    The bound is per solve: a REML fit solves again at every new (Dp, Mp).
    """
    key = (deviance, nullity)
    if key not in rows.phi_solves:
        rows.phi_solves[key] = _global_log_phi(rows, deviance, nullity)
    return rows.phi_solves[key]


class NearPoissonDispersionError(FloatingPointError):
    """The Tweedie dispersion profile is too close to the Poisson limit to search globally.

    Near p = 1 the density approaches the Poisson lattice and the dispersion
    profile has many local minima; the global search over them grows like
    1 / (p - 1). It is raised where one search would exceed its bound (about
    five seconds of series work). ``estimate_p`` skips such a power as
    infeasible and records it; ``fit_reml`` at that fixed power raises it
    (``fit()`` does not profile the dispersion). Fit a power further from 1, or
    a Poisson family.
    """


def _global_log_phi(rows: TweedieRows, deviance: float, nullity: float) -> PhiSolve:
    size = rows.size
    if not (math.isfinite(deviance) and deviance > 0.0):
        raise ValueError("Tweedie dispersion needs a positive finite deviance")
    # l_sat decays like phi^(-1/(p-1)) per positive row, so Q has an interior
    # minimum only if the upper-tail slope N/(p-1) - M/2 is positive.
    if 2.0 * size <= (rows.p - 1.0) * nullity:
        raise ValueError("Tweedie dispersion profile has no finite interior optimum")
    # The saddlepoint density's root, where every positive row adds 1/2 to T.
    local = _newton_log_phi(
        rows, deviance, nullity, math.log(deviance / max(size - nullity, 0.5 * size))
    )
    # The window's density bound needs gamma jumps of shape a >= 1 (p <= 1.5).
    if rows.a < 1.0:
        return local
    return _lattice_scan(rows, deviance, nullity, local)


def _lattice_bands(p: float, curvature: float, sizes: NDArray) -> tuple[NDArray, NDArray]:
    """Peak indices [low, high] over which K rows sharing a phase can bend Q; high = 0 if none.

    A row's density carries the Poisson-summation lattice term (as in
    `saddlepoint_switch`) 2 exp(-2 pi^2 Var J) cos(2 pi j), Var J ~ j / (a + 1),
    whose phase turns at rate 2 pi j per unit log phi, j = |c w| / ((a + 1) phi).
    It bends Q by at most rho(j) = 8 pi^2 j^2 exp(-k j), k = 2 pi^2 / (a + 1),
    and only for j >= 1: below one term of the series dominates and the phase
    does not turn. K rows of one phase bend Q by K rho(j). The band is where
    that reaches a quarter of ``curvature``, the deviance term's D / (2 phi) at
    the Newton root ((N - M) / 2 at the saddlepoint's minimum), a margin for the
    leading-term model: the roots of j exp(-k j / 2) = c, c^2 = curvature /
    (32 pi^2 K), are j = -(2 / k) W(-k c / 2) on the two real branches of Lambert's W.
    """
    k = 2.0 * math.pi**2 * (p - 1.0)
    argument = -0.5 * k * np.sqrt(curvature / (32.0 * math.pi**2 * sizes))
    real = argument > -1.0 / math.e
    low, high = np.ones_like(argument), np.zeros_like(argument)
    high[real] = -2.0 / k * lambertw(argument[real], -1).real
    low[real] = np.maximum(1.0, -2.0 / k * lambertw(argument[real], 0).real)
    high[high < 1.0] = 0.0
    return low, high


def _lattice_scan(rows: TweedieRows, deviance: float, nullity: float, local: PhiSolve) -> PhiSolve:
    """The lowest local minimum of Q inside the window that must hold the global one.

    A group of rows sharing a peak index J lies in its band over one interval of
    u, where j = J e^-u. Q' is sampled four times per lattice period 1 / j there,
    at the finest period of the groups overlapping; between those stretches no
    group bends Q, Q' is monotone and its endpoints decide the sign change. Each
    - to + change is a bracket the Newton polishes.
    """
    curvature = 0.5 * deviance / local.phi
    # No group is larger than the book: if even that cannot bend Q, none can.
    if _lattice_bands(rows.p, curvature, np.array([rows.size]))[1][0] == 0.0:
        return local
    log_peak, sizes = rows.peak_groups
    distinct_sizes, of_size = np.unique(sizes, return_inverse=True)
    low, high = _lattice_bands(rows.p, curvature, distinct_sizes)
    banded = np.flatnonzero(high[of_size] > 0.0)
    if banded.size == 0:
        return local
    lower, upper, rising_edge = _scan_window(rows, deviance, nullity, local)
    band_low, band_high = low[of_size][banded], high[of_size][banded]
    starts = log_peak[banded] - np.log(band_high)
    stops = log_peak[banded] - np.log(band_low)
    inside = np.flatnonzero((stops > lower) & (starts < upper))
    if inside.size == 0:
        # No group bends Q anywhere in the window: the Newton root is its minimum.
        return local
    order = inside[np.argsort(starts[inside])]
    starts = np.maximum(starts[order], lower)
    reach = np.maximum.accumulate(np.minimum(stops[order], upper))
    # A start past every earlier stop opens a new piece of the union.
    opens = np.flatnonzero(np.r_[True, starts[1:] > reach[:-1]])
    closes = np.r_[opens[1:] - 1, starts.size - 1]
    steps = np.minimum.reduceat(0.25 / band_high[order], opens)
    counts = np.ceil((reach[closes] - starts[opens]) / steps).astype(np.int64) + 1
    work = float(np.sum(counts) + 2) * (rows.log_y.size + _PASS_OVERHEAD_ROWS)
    if work > _SCAN_WORK_LIMIT:
        raise NearPoissonDispersionError(
            f"Tweedie p={rows.p:.10g} is too close to 1 to profile the dispersion globally: "
            f"its lattice comparison needs {work:.3g} row passes, over {_SCAN_WORK_LIMIT:.3g}. "
            "Fit a power further from 1 or a Poisson family; a power search skips such "
            "a power as infeasible."
        )
    points = [np.array([lower, upper])]
    for start, stop, count in zip(starts[opens], reach[closes], counts, strict=True):
        points.append(np.linspace(start, stop, count))
    grid = np.unique(np.concatenate(points))
    u_local = math.log(local.phi)
    best, n_passes = local, local.n_passes
    previous_u, previous_score = -math.inf, 0.0
    for u in grid:
        half_deviance = 0.5 * deviance * math.exp(-u)
        score = rows.saturated(math.exp(u))[1] - half_deviance - 0.5 * nullity
        n_passes += 1
        if previous_score < 0.0 <= score and not previous_u <= u_local <= u:
            polished = _newton_log_phi(
                rows, deviance, nullity, 0.5 * (previous_u + u), previous_u, u
            )
            n_passes += polished.n_passes
            if polished.criterion < best.criterion:
                best = polished
        previous_u, previous_score = u, score
    if rising_edge and previous_score < 0.0:
        # Past `rising` Q' > 0, so a score still negative at the window's end puts
        # a minimum within round-off of it: once every row is down to one jump,
        # E J = 1 holds exactly and that edge is itself a root of Q'.
        polished = _newton_log_phi(rows, deviance, nullity, previous_u, previous_u, math.inf)
        n_passes += polished.n_passes
        if polished.criterion < best.criterion:
            best = polished
    return replace(best, n_passes=n_passes)


def _scan_window(
    rows: TweedieRows, deviance: float, nullity: float, local: PhiSolve
) -> tuple[float, float, bool]:
    """An interval of u outside which Q exceeds the Newton minimum or rises, and whether
    its upper end is the E J >= 1 edge rather than the Q-bound crossing.

    Given J = j >= 1 jumps, y is Gamma(j a, gamma), gamma = phi (p - 1) y^(p-1) / w
    at mu = y, whose peak density is largest at j = 1 for a >= 1, so
    l_sat <= -log gamma + (a - 1) log(a - 1) - (a - 1) - log Gamma(a). Summed, Q is
    at least the convex D e^-u / 2 + (N - M / 2) u - B - (M / 2) log 2 pi. And
    E J >= 1 gives T >= (a + 1) N - e^-u sum |c w|, so Q' > 0 past
    log((D / 2 + sum |c w|) / ((a + 1) N - M / 2)).
    """
    p, a = rows.p, rows.a
    count = np.ones_like(rows.log_y) if rows.count is None else rows.count
    # log_t_unit_phi carries (a + 1) log w for prior weights and nothing for counts.
    log_weight = (
        rows.log_t_unit_phi - a * (rows.log_y - math.log(p - 1.0)) + math.log(2.0 - p)
    ) / (a + 1.0)
    peak = (a - 1.0) * math.log(a - 1.0) if a > 1.0 else 0.0
    peak -= (a - 1.0) + math.lgamma(a)
    bound = (
        float(count @ (log_weight - math.log(p - 1.0) - (p - 1.0) * rows.log_y)) + rows.size * peak
    )
    slope = rows.size - 0.5 * nullity

    def excess(u: float) -> float:
        floor = 0.5 * deviance * math.exp(-u) + slope * u - bound
        return floor - 0.5 * nullity * _LOG_TWO_PI - local.criterion

    rising = math.log(
        (0.5 * deviance - float(count @ rows.saturated_canonical))
        / ((a + 1.0) * rows.size - 0.5 * nullity)
    )
    u_local = math.log(local.phi)
    lower = _bound_crossing(excess, u_local, -1.0)
    upper = rising if slope <= 0.0 else min(rising, _bound_crossing(excess, u_local, 1.0))
    return lower, max(upper, lower), upper == rising


def _bound_crossing(excess: Callable[[float], float], start: float, direction: float) -> float:
    """Where the convex ``excess`` turns positive walking from ``start``, to 1e-6 in u."""
    inside, width = start, 1.0
    while excess(start + direction * width) <= 0.0:
        inside, width = start + direction * width, 2.0 * width
    outside = start + direction * width
    while abs(outside - inside) > 1e-6:
        middle = 0.5 * (inside + outside)
        inside, outside = (middle, outside) if excess(middle) <= 0.0 else (inside, middle)
    return outside


def _newton_log_phi(
    rows: TweedieRows,
    deviance: float,
    nullity: float,
    u: float,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> PhiSolve:
    # A bracket end is a point where the score's sign was seen. While one side
    # has none, every proposal lies on that side of u (a Newton step toward the
    # root or a 2-unit walk), so bisection never lands on an unevaluated limit.
    for n_passes in range(1, _NEWTON_MAX_STEPS + 1):
        saturated, saturated_score, saturated_slope = rows.saturated(math.exp(u))
        half_deviance = 0.5 * deviance * math.exp(-u)
        score = saturated_score - half_deviance - 0.5 * nullity
        curvature = saturated_slope + half_deviance
        if score > 0.0:
            upper = u
        else:
            lower = u
        # A zero score made u the lower end, so the walk from it goes up.
        walk = -_NEWTON_MAX_STEP if score > 0.0 else _NEWTON_MAX_STEP
        step = -score / curvature if curvature > 0.0 else walk
        proposal = u + max(-_NEWTON_MAX_STEP, min(step, _NEWTON_MAX_STEP))
        # rtsafe tests the move actually taken, Newton or bisection: at large
        # modes the score's round-off keeps Newton steps above the tolerance
        # after the sign-change bracket has already collapsed. A proposal on a
        # bracket end bisects instead: near p = 1, a clipped Newton step and the
        # walk from a point of negative curvature can land on each other's
        # iterate and cycle. A zero score leaves u, its own bracket end, as the root.
        inside = lower < proposal < upper or proposal == u
        move = (proposal if inside else 0.5 * (lower + upper)) - u
        if curvature > 0.0 and abs(move) <= _NEWTON_STEP_TOL:
            # A converged Newton step or a collapsed bracket puts u + move within
            # the tolerance of the root; l_sat moves by -T per unit u, so carrying
            # it keeps Q to O(move^2).
            u += move
            criterion = (
                half_deviance * math.exp(-move)
                - (saturated - saturated_score * move)
                - 0.5 * nullity * (_LOG_TWO_PI + u)
            )
            return PhiSolve(math.exp(u), criterion, curvature, n_passes)
        u += move
    bracketed = math.isfinite(lower) and math.isfinite(upper)
    raise FloatingPointError(
        f"Tweedie dispersion Newton did not settle in {_NEWTON_MAX_STEPS} steps at p={rows.p:.6g}"
        + ("" if bracketed else "; the score kept one sign from the saddlepoint start")
    )


def _density_arrays(y, mu, phi, p, weights):
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or mu.shape != y.shape or weights.shape != y.shape:
        raise ValueError("y, mu and weights must be one-dimensional with the same shape")
    if not (np.all(np.isfinite(y)) and np.all(y >= 0.0)):
        raise ValueError("y must be finite and non-negative")
    if not (np.all(np.isfinite(mu)) and np.all(mu > 0.0)):
        raise ValueError("mu must be finite and strictly positive")
    if not (np.all(np.isfinite(weights)) and np.all(weights > 0.0)):
        raise ValueError("weights must be finite and strictly positive")
    if not (math.isfinite(phi) and phi > 0.0):
        raise ValueError("phi must be finite and strictly positive")
    if not 1.0 < p < 2.0:
        raise ValueError("p must be in the open interval (1, 2)")
    return y, mu, float(phi), float(p), weights


def _saturated_rows(y, weights, phi, p) -> NDArray:
    """Per-row saturated log density, with zero rows at their exact 0."""
    saturated = np.zeros_like(y)
    positive = y > 0.0
    saturated[positive] = TweedieRows.prepare(y, weights, p).row_saturated(phi)[0]
    return saturated


def _scaled_deviance(y, mu, p, weights, phi) -> NDArray:
    """w d(y, mu) / (2 phi), through logs where a factor or the product leaves the float range.

    The density depends on phi / w alone, and d may overflow or underflow where
    w d / phi does not; those rows, and rows whose w / phi leaves the normal
    range, take the exponential of the term's logarithm.
    """
    deviance = tweedie_unit_deviance(y, mu, p)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = weights / phi
        scaled = deviance * (0.5 * ratio)
    # A positive d below the normal range has lost digits, or underflowed to 0,
    # though w d / (2 phi) may be representable.
    underflowed = (deviance < _TINY) & (y != mu)
    extreme = np.flatnonzero(~np.isfinite(scaled) | ~(ratio >= _TINY) | underflowed)
    if extreme.size:
        # A normal d keeps its own accuracy; an overflowed or underflowed one needs log d.
        with np.errstate(divide="ignore"):
            log_deviance = np.log(deviance[extreme])
        overflowed = np.flatnonzero(~np.isfinite(deviance[extreme]) | underflowed[extreme])
        log_deviance[overflowed] = _log_unit_deviance(
            y[extreme][overflowed], mu[extreme][overflowed], p
        )
        with np.errstate(over="ignore"):
            scaled[extreme] = np.exp(
                log_deviance + np.log(weights[extreme]) - math.log(2.0) - math.log(phi)
            )
    return scaled


def _log_unit_deviance(y: NDArray, mu: NDArray, p: float) -> NDArray:
    """log d(y, mu), finite where d itself over- or underflows.

    d is homogeneous of degree 2 - p. With mu = m 2^e, m in [1/2, 1), scaling
    both arguments by 2^-e is exact, subnormals included, so
    d(y, mu) = 2^(e (2 - p)) d(y 2^-e, m) keeps y - mu exact, where a rounded
    y / mu would not. Only where y 2^-e or that d overflows, y / mu nears
    overflow, and the half-deviance factors as A (1 - B/A + C/A),
    A = y mu^(1-p) / (p - 1), with B/A = (mu / y)^(p-1) / (2 - p) and
    C/A = (p - 1)(mu / y) / (2 - p) far below one.
    """
    log_mu = np.log(mu)
    _, exponent = np.frexp(mu)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        scaled = tweedie_unit_deviance(np.ldexp(y, -exponent), np.ldexp(mu, -exponent), p)
        log_deviance = exponent * ((2.0 - p) * math.log(2.0)) + np.log(scaled)
    huge = np.flatnonzero(np.isposinf(log_deviance))
    if huge.size:
        log_y = np.log(y[huge])
        log_mu_over_y = log_mu[huge] - log_y
        correction = -np.expm1((p - 1.0) * log_mu_over_y - math.log(2.0 - p)) + np.exp(
            math.log(p - 1.0) - math.log(2.0 - p) + log_mu_over_y
        )
        log_deviance[huge] = (
            math.log(2.0)
            + log_y
            + (1.0 - p) * log_mu[huge]
            - math.log(p - 1.0)
            + np.log(correction)
        )
    return log_deviance


def weighted_deviance(y, mu, p, weights) -> float:
    """sum w d(y, mu), finite wherever every term is."""
    return 2.0 * float(np.sum(_scaled_deviance(y, mu, p, weights, 1.0)))


def tweedie_logpdf(y, mu, phi, p, weights=None) -> NDArray:
    """Row log densities of Tweedie(mu, phi / w, p), 1 < p < 2 (Dunn & Smyth 2005)."""
    y, mu, phi, p, weights = _density_arrays(y, mu, phi, p, weights)
    saturated = _saturated_rows(y, weights, phi, p)
    return saturated - _scaled_deviance(y, mu, p, weights, phi)


def tweedie_logpdf_pair(y, mu, null_mu, phi, p, *, weights=None) -> tuple[NDArray, NDArray]:
    """Fitted and null row log densities from one saturated pass."""
    y, mu, phi, p, weights = _density_arrays(y, mu, phi, p, weights)
    null_mu = np.asarray(null_mu, dtype=np.float64)
    if null_mu.shape != y.shape or not (np.all(np.isfinite(null_mu)) and np.all(null_mu > 0.0)):
        raise ValueError("null_mu must match mu and be finite and strictly positive")
    saturated = _saturated_rows(y, weights, phi, p)
    return (
        saturated - _scaled_deviance(y, mu, p, weights, phi),
        saturated - _scaled_deviance(y, null_mu, p, weights, phi),
    )


_TWEEDIE_DEVIANCE_SERIES_THRESHOLD = 1e-3
_TWEEDIE_DEVIANCE_SERIES_TERMS = 8


def tweedie_unit_deviance(y: NDArray, mu: NDArray, p: float) -> NDArray:
    """Compute positive-response unit deviance without close-mean cancellation."""
    y_array, mu_array = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
    )
    zero_mask = y_array == 0.0
    # Split on integer positions, not on the boolean mask. Every gather and
    # every scatter below then indexes directly instead of re-walking the mask,
    # and the two `np.any` scans collapse into the one `flatnonzero` that has
    # to run anyway. Measured 2.5-2.7x on the whole call at n = 67,000 across
    # every branch, power and zero fraction; the selected elements, their
    # order and their values are unchanged, so the result is bitwise identical.
    zero_indices = np.flatnonzero(zero_mask)
    if zero_indices.size:
        # A zero response has the closed form 2 mu**(2-p) / (2-p). The y < mu/2
        # branch below reproduces it through log(0) = -inf at several
        # transcendental evaluations per row; on zero-inflated fits those rows
        # are the bulk of every deviance evaluation. The two routes are bitwise
        # identical: that branch's g reduces to exactly -0.0 - (-1.0)/(2.0 - p)
        # and both multiply the same power the same way.
        deviance = np.empty_like(mu_array)
        with np.errstate(over="ignore"):
            deviance[zero_indices] = (
                2.0 * np.power(mu_array[zero_indices], 2.0 - p) * (1.0 / (2.0 - p))
            )
        if zero_indices.size != y_array.size:
            positive_indices = np.flatnonzero(~zero_mask)
            deviance[positive_indices] = tweedie_unit_deviance(
                y_array[positive_indices],
                mu_array[positive_indices],
                p,
            )
        return deviance
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        delta = (y_array - mu_array) / mu_array
    extreme_positive = np.isposinf(delta)
    near = np.abs(delta) <= _TWEEDIE_DEVIANCE_SERIES_THRESHOLD
    g = np.empty_like(delta)

    if np.any(near):
        delta_near = delta[near]
        term = np.full_like(delta_near, 0.5)
        series = term.copy()
        # The recurrence adds k=1,...,8 from the integral expansion. At the
        # 1e-3 threshold, the first omitted term is O(1e-27), below binary64
        # rounding even for the largest coefficient over 1 < p < 2.
        for k in range(_TWEEDIE_DEVIANCE_SERIES_TERMS):
            term *= -delta_near * (p + k) / (k + 3.0)
            series += term
        g[near] = delta_near**2 * series

    # Sterbenz: y - mu is exact only for mu/2 <= y <= 2 mu. Below mu/2 its
    # rounding leaves 1 + delta an absolute error of order eps, so y / mu, and
    # the deviance through it, carry a relative error of order eps * mu / y. At
    # y = 4e-9 mu that moved one row's deviance by 9e-9, ten times a PIRLS line
    # search's round-off allowance for a whole 20,000-row fit. Those rows take
    # the log-ratio branch below.
    regular = ~near & ~extreme_positive & (y_array >= 0.5 * mu_array)
    if np.any(regular):
        delta_regular = delta[regular]
        with np.errstate(all="ignore"):
            log_ratio = np.log1p(delta_regular)
            first = (1.0 + delta_regular) * np.expm1((1.0 - p) * log_ratio) / (1.0 - p)
            second = np.expm1((2.0 - p) * log_ratio) / (2.0 - p)
        g[regular] = first - second

    # y < mu/2, down to a delta that rounds to exactly -1: take log(y / mu)
    # from the original values. A normal quotient carries one rounding, so its
    # log errs by about eps (1 + 2 |log(y / mu)|) at any scale, where
    # log y - log mu errs by ulps of max(|log y|, |log mu|); only a subnormal or
    # underflowed quotient needs the difference. With log_ratio <= -log 2 both
    # expm1 factors are accurate, and g = first - second cancels by at most a
    # factor of 6.2 (reached at y = mu/2 as p -> 2).
    below = ~near & ~regular & ~extreme_positive
    if np.any(below):
        y_below, mu_below = y_array[below], mu_array[below]
        with np.errstate(all="ignore"):
            ratio = y_below / mu_below
            log_ratio = np.log(ratio)
            subnormal = ratio < np.finfo(np.float64).tiny
            log_ratio[subnormal] = np.log(y_below[subnormal]) - np.log(mu_below[subnormal])
            first = np.exp((2.0 - p) * log_ratio) * np.expm1((p - 1.0) * log_ratio) / (p - 1.0)
            second = np.expm1((2.0 - p) * log_ratio) / (2.0 - p)
        g[below] = first - second

    deviance = np.empty_like(delta)
    ordinary_ratio = ~extreme_positive
    with np.errstate(all="ignore"):
        deviance[ordinary_ratio] = (
            2.0 * np.power(mu_array[ordinary_ratio], 2.0 - p) * g[ordinary_ratio]
        )

    if np.any(extreme_positive):
        # For y >> mu, factor the expanded positive half-deviance by
        # A = y * mu**(1-p) / (p-1). The remaining ratios use mu/y and
        # cannot create inf-inf or 0*inf. -expm1 keeps 1 - B/A accurate
        # when p is itself very close to one.
        with np.errstate(all="ignore"):
            log_y = np.log(y_array[extreme_positive])
            log_mu = np.log(mu_array[extreme_positive])
            log_mu_over_y = log_mu - log_y
            log_b_over_a = (p - 1.0) * log_mu_over_y - np.log(2.0 - p)
            log_c_over_a = np.log(p - 1.0) - np.log(2.0 - p) + log_mu_over_y
            correction = -np.expm1(log_b_over_a) + np.exp(log_c_over_a)
            log_deviance = (
                np.log(2.0) + log_y + (1.0 - p) * log_mu - np.log(p - 1.0) + np.log(correction)
            )
            deviance[extreme_positive] = np.exp(log_deviance)
    negative_roundoff = np.isfinite(deviance) & (deviance < 0.0)
    if np.any(negative_roundoff):
        deviance = deviance.copy()
        deviance[negative_roundoff] = 0.0
    return deviance


def generate_tweedie_cpg(n: int, mu, phi, p: float, rng=None) -> NDArray:
    """Simulate Tweedie(mu, phi, p) as compound Poisson-gamma: N ~ Poisson, Y | N ~ Gamma.

    ``rng`` is a numpy Generator, or any object whose ``poisson(lam)`` and
    ``gamma(shape, scale=...)`` return one draw per element as numpy's do; the
    draws are checked, since fractional counts or one draw broadcast over every
    row would sample a different distribution.
    """
    if isinstance(n, bool | np.bool_):
        raise TypeError("n must be a non-negative integer")
    n = operator.index(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    p = float(p)
    if not 1.0 < p < 2.0:
        raise ValueError("p must be in the open interval (1, 2)")
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,))
    phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), (n,))
    if not (np.all(np.isfinite(mu)) and np.all(mu > 0.0)):
        raise ValueError("mu must be finite and strictly positive")
    if not (np.all(np.isfinite(phi)) and np.all(phi > 0.0)):
        raise ValueError("phi must be finite and strictly positive")
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        rate = np.power(mu, 2.0 - p) / ((2.0 - p) * phi)
        scale = phi * (p - 1.0) * np.power(mu, p - 1.0)
    if not (np.all(rate > 0.0) and np.all(rate <= _POISSON_LAM_MAX)):
        raise ValueError("the Poisson rate is not representable for these mu, phi and p")
    if not (np.all(np.isfinite(scale)) and np.all(scale > 0.0)):
        raise ValueError("the Gamma scale is not representable for these mu, phi and p")
    rng = np.random.default_rng() if rng is None else rng
    counts = np.asarray(rng.poisson(rate))
    if counts.shape != (n,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise RuntimeError("rng.poisson must return one non-negative integer count per row")
    y = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    if positive.any():
        raw = np.asarray(rng.gamma((2.0 - p) / (p - 1.0) * counts[positive], scale=scale[positive]))
        if raw.shape != (int(np.count_nonzero(positive)),) or raw.dtype.kind not in "iuf":
            raise RuntimeError("rng.gamma must return one real draw per positive count")
        draws = raw.astype(np.float64)
        # A draw that is not a representable positive number would read as a
        # structural zero or an infinite claim; both corrupt the density's split.
        if not (np.all(np.isfinite(draws)) and np.all(draws > 0.0)):
            raise ValueError("a positive event's Gamma draw underflowed to zero or overflowed")
        y[positive] = draws
    return y
