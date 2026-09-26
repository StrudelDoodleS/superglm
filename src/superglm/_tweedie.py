"""Tweedie (1 < p < 2) density, dispersion profile and simulation on one series.

Every EDM satisfies log f(y; mu, phi/w) = log f(y; y, phi/w) - w d(y, mu) / (2 phi).
The saturated term depends on (y, w, p) and phi only, and a zero response
contributes nothing to it, so one prepared set of positive rows serves the
density, the fitted/null pair and the dispersion profile.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm._tweedie_series import series_moments

_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_PHI_LIMIT = 45.0
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
        ok, log_w, mean_j, var_j = series_moments(
            self.log_t_unit_phi - (self.a + 1.0) * math.log(phi), self.a
        )
        if not ok.all():
            raise FloatingPointError(
                f"Tweedie series cannot evaluate {int(np.count_nonzero(~ok))} of {ok.size} "
                f"positive rows at p={self.p:.6g}, phi={phi:.6g}"
            )
        canonical = self.saturated_canonical / phi
        inverse_r = self.a + 1.0
        return (
            log_w - self.log_y + canonical,
            mean_j * inverse_r + canonical,
            -var_j * inverse_r**2 - canonical,
        )

    def saturated(self, phi: float) -> tuple[float, float, float]:
        value, score, slope = self.row_saturated(phi)
        if self.count is None:
            return float(np.sum(value)), float(np.sum(score)), float(np.sum(slope))
        return float(self.count @ value), float(self.count @ score), float(self.count @ slope)


@dataclass(frozen=True)
class PhiSolve:
    """Profiled dispersion: phi, Q at the optimum, Q'' in log phi and the series passes used."""

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
    rtsafe): no concavity result is known for the Tweedie dispersion profile.
    """
    key = (deviance, nullity)
    if key not in rows.phi_solves:
        rows.phi_solves[key] = _newton_log_phi(rows, deviance, nullity)
    return rows.phi_solves[key]


def _newton_log_phi(rows: TweedieRows, deviance: float, nullity: float) -> PhiSolve:
    size = rows.size
    if not (math.isfinite(deviance) and deviance > 0.0):
        raise ValueError("Tweedie dispersion needs a positive finite deviance")
    # l_sat decays like phi^(-1/(p-1)) per positive row, so Q has an interior
    # minimum only if the upper-tail slope N/(p-1) - M/2 is positive.
    if 2.0 * size <= (rows.p - 1.0) * nullity:
        raise ValueError("Tweedie dispersion profile has no finite interior optimum")
    lower, upper = -_LOG_PHI_LIMIT, _LOG_PHI_LIMIT
    # The saddlepoint density's root, where every positive row adds 1/2 to T.
    u = min(max(math.log(deviance / max(size - nullity, 0.5 * size)), lower + 1.0), upper - 1.0)
    for n_passes in range(1, _NEWTON_MAX_STEPS + 1):
        saturated, saturated_score, saturated_slope = rows.saturated(math.exp(u))
        half_deviance = 0.5 * deviance * math.exp(-u)
        score = saturated_score - half_deviance - 0.5 * nullity
        curvature = saturated_slope + half_deviance
        if score > 0.0:
            upper = u
        else:
            lower = u
        step = -score / curvature if curvature > 0.0 else math.copysign(_NEWTON_MAX_STEP, -score)
        proposal = u + max(-_NEWTON_MAX_STEP, min(step, _NEWTON_MAX_STEP))
        # rtsafe tests the move actually taken, Newton or bisection: at large
        # modes the score's round-off keeps Newton steps above the tolerance
        # after the sign-change bracket has already collapsed. The bracket test
        # is inclusive because a zero score leaves u itself as the bracket end.
        move = (proposal if lower <= proposal <= upper else 0.5 * (lower + upper)) - u
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
    raise FloatingPointError(
        f"Tweedie dispersion Newton did not settle in {_NEWTON_MAX_STEPS} steps at p={rows.p:.6g}"
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


def _log_density(saturated, weights, phi, deviance) -> NDArray:
    return saturated - weights * deviance / (2.0 * phi)


def tweedie_logpdf(y, mu, phi, p, weights=None) -> NDArray:
    """Row log densities of Tweedie(mu, phi / w, p), 1 < p < 2 (Dunn & Smyth 2005)."""
    y, mu, phi, p, weights = _density_arrays(y, mu, phi, p, weights)
    saturated = _saturated_rows(y, weights, phi, p)
    return _log_density(saturated, weights, phi, tweedie_unit_deviance(y, mu, p))


def tweedie_logpdf_pair(y, mu, null_mu, phi, p, *, weights=None) -> tuple[NDArray, NDArray]:
    """Fitted and null row log densities from one saturated pass."""
    y, mu, phi, p, weights = _density_arrays(y, mu, phi, p, weights)
    null_mu = np.asarray(null_mu, dtype=np.float64)
    if null_mu.shape != y.shape or not (np.all(np.isfinite(null_mu)) and np.all(null_mu > 0.0)):
        raise ValueError("null_mu must match mu and be finite and strictly positive")
    saturated = _saturated_rows(y, weights, phi, p)
    return (
        _log_density(saturated, weights, phi, tweedie_unit_deviance(y, mu, p)),
        _log_density(saturated, weights, phi, tweedie_unit_deviance(y, null_mu, p)),
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
        # A zero response has the closed form 2 mu**(2-p) / (2-p). The general
        # machinery below reproduces it through its delta == -1 recovery
        # branch (log(0), exp) at several transcendental evaluations per row;
        # on zero-inflated fits those rows are the bulk of every deviance
        # evaluation. The two routes are bitwise identical: the recovery
        # branch's g reduces to exactly 0.0 - (0.0 - 1.0)/(2.0 - p) and both
        # multiply the same power the same way.
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

    regular = ~near & ~extreme_positive & (delta > -1.0)
    if np.any(regular):
        delta_regular = delta[regular]
        with np.errstate(all="ignore"):
            log_ratio = np.log1p(delta_regular)
            first = (1.0 + delta_regular) * np.expm1((1.0 - p) * log_ratio) / (1.0 - p)
            second = np.expm1((2.0 - p) * log_ratio) / (2.0 - p)
        g[regular] = first - second

    # For a positive y many orders below mu, delta can round to exactly -1.
    # Recover log(y / mu) from the original values; this branch is far from
    # the cancellation region, so the power-scale formula is well-conditioned.
    rounded_to_minus_one = ~near & ~regular & ~extreme_positive
    if np.any(rounded_to_minus_one):
        with np.errstate(all="ignore"):
            log_ratio = np.log(y_array[rounded_to_minus_one]) - np.log(
                mu_array[rounded_to_minus_one]
            )
            ratio = np.exp(log_ratio)
            ratio_two_minus_p = np.exp((2.0 - p) * log_ratio)
            first = (ratio_two_minus_p - ratio) / (1.0 - p)
            second = (ratio_two_minus_p - 1.0) / (2.0 - p)
        g[rounded_to_minus_one] = first - second

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
