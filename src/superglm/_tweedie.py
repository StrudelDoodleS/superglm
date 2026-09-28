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

import numpy as np
from numpy.typing import NDArray
from scipy.special import lambertw

from superglm._tweedie_series import _LOG_CUTOFF, series_moments

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
        past = np.flatnonzero(~ok)
        value[past], score[past], slope[past] = _corrected_saddlepoint(
            self.p, _log_negative_canonical(self.p, log_t[past]), self.log_y[past]
        )
        return value, score, slope

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
    Poisson lattice (Dunn & Smyth 2005, Sec. 8), so where a row's lattice
    term can bend Q the Newton root is compared with every other local
    minimum (`_lattice_scan`).
    """
    key = (deviance, nullity)
    if key not in rows.phi_solves:
        rows.phi_solves[key] = _global_log_phi(rows, deviance, nullity)
    return rows.phi_solves[key]


def _global_log_phi(rows: TweedieRows, deviance: float, nullity: float) -> PhiSolve:
    size = rows.size
    if not (math.isfinite(deviance) and deviance > 0.0):
        raise ValueError("Tweedie dispersion needs a positive finite deviance")
    # l_sat decays like phi^(-1/(p-1)) per positive row, so Q has an interior
    # minimum only if the upper-tail slope N/(p-1) - M/2 is positive.
    if 2.0 * size <= (rows.p - 1.0) * nullity:
        raise ValueError("Tweedie dispersion profile has no finite interior optimum")
    # The saddlepoint density's root, where every positive row adds 1/2 to T.
    smooth_size = max(size - nullity, 0.5 * size)
    local = _newton_log_phi(rows, deviance, nullity, math.log(deviance / smooth_size))
    band = _lattice_band(rows.p, smooth_size)
    # The window's density bound needs gamma jumps of shape a >= 1 (p <= 1.5).
    if band is None or rows.a < 1.0:
        return local
    return _lattice_scan(rows, deviance, nullity, local, band)


def _lattice_band(p: float, smooth_size: float) -> tuple[float, float] | None:
    """Peak indices whose lattice term can bend Q, or None where no row's can.

    A row's density carries the Poisson-summation lattice term (as in
    `saddlepoint_switch`) 2 exp(-2 pi^2 Var J) cos(2 pi j), Var J ~ j / (a + 1),
    whose phase turns at rate 2 pi j per unit log phi, j = |c w| / ((a + 1) phi).
    It bends Q by at most rho(j) = 8 pi^2 j^2 exp(-k j), k = 2 pi^2 / (a + 1),
    and only for j >= 1: below one term of the series dominates and the phase
    does not turn. The band is where rho reaches a quarter of the smooth
    profile's curvature at its minimum, (N - M) / 2, a margin for the leading-
    term model: the roots of j exp(-k j / 2) = c, c^2 = (N - M) / (64 pi^2),
    are j = -(2 / k) W(-k c / 2) on the two real branches of Lambert's W.
    """
    k = 2.0 * math.pi**2 * (p - 1.0)
    argument = -0.5 * k * math.sqrt(smooth_size / (64.0 * math.pi**2))
    if argument <= -1.0 / math.e:
        return None
    top = -2.0 / k * float(lambertw(argument, -1).real)
    if top < 1.0:
        return None
    return max(1.0, -2.0 / k * float(lambertw(argument, 0).real)), top


def _lattice_scan(
    rows: TweedieRows,
    deviance: float,
    nullity: float,
    local: PhiSolve,
    band: tuple[float, float],
) -> PhiSolve:
    """The lowest local minimum of Q inside the window that must hold the global one.

    Q' is sampled four times per lattice period 1 / j wherever some row's peak
    index lies in the band; between those stretches no row bends Q, Q' is
    monotone and its endpoints decide the sign change. Each - to + change is a
    bracket the Newton polishes.
    """
    lower, upper = _scan_window(rows, deviance, nullity, local)
    band_low, band_high = band
    # Row i's peak index is J_i e^-u; it lies in the band over this u-interval.
    log_peak = math.log(rows.p - 1.0) + _log_negative_canonical(rows.p, rows.log_t_unit_phi)
    starts, stops = log_peak - math.log(band_high), log_peak - math.log(band_low)
    inside = (stops > lower) & (starts < upper)
    if not np.any(inside):
        # No row bends Q anywhere in the window: the Newton root is its minimum.
        return local
    order = np.argsort(starts[inside])
    starts = np.maximum(starts[inside][order], lower)
    reach = np.maximum.accumulate(np.minimum(stops[inside][order], upper))
    # A start past every earlier stop opens a new piece of the union.
    opens = np.r_[True, starts[1:] > reach[:-1]]
    closes = np.flatnonzero(np.r_[opens[1:], True])
    step = 0.25 / band_high
    points = [np.array([lower, upper])]
    for start, stop in zip(starts[opens], reach[closes], strict=True):
        points.append(np.linspace(start, stop, int(math.ceil((stop - start) / step)) + 1))
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
    return replace(best, n_passes=n_passes)


def _scan_window(
    rows: TweedieRows, deviance: float, nullity: float, local: PhiSolve
) -> tuple[float, float]:
    """An interval of u outside which Q exceeds the Newton minimum or rises.

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
    return lower, max(upper, lower)


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
    """w d(y, mu) / (2 phi), through logs where a factor or the product overflows.

    The density depends on phi / w alone, and d may overflow where w d / phi
    does not; those rows, and rows whose w / phi leaves the normal range, take
    the exponential of the term's logarithm.
    """
    ratio = weights / phi
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = tweedie_unit_deviance(y, mu, p) * (0.5 * ratio)
    extreme = np.flatnonzero(~np.isfinite(scaled) | ~(ratio >= _TINY))
    if extreme.size:
        log_deviance = _log_unit_deviance(y[extreme], mu[extreme], p)
        with np.errstate(over="ignore"):
            scaled[extreme] = np.exp(
                log_deviance + np.log(weights[extreme]) - math.log(2.0) - math.log(phi)
            )
    return scaled


def _log_unit_deviance(y: NDArray, mu: NDArray, p: float) -> NDArray:
    """log d(y, mu), finite where d itself overflows.

    d is homogeneous of degree 2 - p, d(y, mu) = mu^(2-p) d(y / mu, 1), and
    d(r, 1) is finite unless r nears overflow. There the half-deviance factors
    as A (1 - B/A + C/A), A = y mu^(1-p) / (p - 1), with
    B/A = (mu / y)^(p-1) / (2 - p) and C/A = (p - 1)(mu / y) / (2 - p) far below one.
    """
    log_mu = np.log(mu)
    with np.errstate(over="ignore", divide="ignore"):
        ratio = y / mu
        log_deviance = (2.0 - p) * log_mu + np.log(
            tweedie_unit_deviance(ratio, np.ones_like(ratio), p)
        )
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
    """Simulate Tweedie(mu, phi, p) as compound Poisson-gamma: N ~ Poisson, Y | N ~ Gamma."""
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
    counts = rng.poisson(rate)
    y = np.zeros(n, dtype=np.float64)
    positive = counts > 0
    if positive.any():
        draws = rng.gamma((2.0 - p) / (p - 1.0) * counts[positive], scale=scale[positive])
        # A draw that is not a representable positive number would read as a
        # structural zero or an infinite claim; both corrupt the density's split.
        if not (np.all(np.isfinite(draws)) and np.all(draws > 0.0)):
            raise ValueError("a positive event's Gamma draw underflowed to zero or overflowed")
        y[positive] = draws
    return y
