"""One-parameter profile likelihood: recorded bounded search and likelihood-ratio interval."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

# An infeasible point's likelihood-ratio excess. Finite so brentq's
# interpolation stays finite; any excess above 1 at a root marks the side
# censored (see _interval_side).
_BARRIER = 1e6
# Stand-in objective for an infeasible point inside Brent: finite because the
# parabolic step differences objective values (inf - inf is NaN); its square
# stays far below overflow.
_INFEASIBLE = 1e50


class RecordedObjective:
    """Cache an objective by exact argument and keep every evaluation in call order."""

    def __init__(self, objective: Callable[[float], float]):
        self._objective = objective
        self.values: dict[float, float] = {}

    def __call__(self, x: float) -> float:
        key = float(x)
        if key not in self.values:
            self.values[key] = float(self._objective(key))
        return self.values[key]

    def best(self) -> tuple[float, float]:
        """The evaluated point with the smallest finite objective."""
        finite = {x: v for x, v in self.values.items() if math.isfinite(v)}
        if not finite:
            raise RuntimeError("no evaluated point has a finite profile objective")
        x_best = min(finite, key=finite.__getitem__)
        return x_best, finite[x_best]


def minimize_profile(
    objective: RecordedObjective, bounds: tuple[float, float], *, xatol: float, maxiter: int
) -> bool:
    """Bounded Brent (Brent 1973) after evaluating both bounds; returns whether it converged.

    Brent's bounded method never evaluates the bounds themselves, so they are
    recorded first and a boundary optimum is found by `objective.best()`.
    Infeasible points (inf) are replaced by a finite barrier inside the search
    so the parabolic steps stay finite; they never win `best()`.
    """
    objective(bounds[0])
    objective(bounds[1])

    def finite_objective(x: float) -> float:
        value = objective(x)
        return value if math.isfinite(value) else _INFEASIBLE

    result = minimize_scalar(
        finite_objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    return bool(result.success)


@dataclass(frozen=True)
class Interval:
    """Likelihood-ratio interval; a censored side stops at a bound or an infeasible point."""

    lower: float
    upper: float
    lower_censored: bool
    upper_censored: bool


def likelihood_ratio_interval(
    objective: Callable[[float], float],
    x_hat: float,
    nll_hat: float,
    bounds: tuple[float, float],
    *,
    alpha: float,
    scale: float,
    rtol: float,
) -> Interval:
    """{x : 2 scale (nll(x) - nll_hat) <= chi2_1(1 - alpha)} around x_hat (Venzon & Moolgavkar 1988)."""
    cutoff = float(chi2.ppf(1.0 - alpha, 1))

    def excess(x: float) -> float:
        value = objective(x)
        return 2.0 * scale * (value - nll_hat) - cutoff if math.isfinite(value) else _BARRIER

    lower, lower_censored = _interval_side(excess, bounds[0], x_hat, rtol)
    upper, upper_censored = _interval_side(excess, bounds[1], x_hat, rtol)
    return Interval(lower, upper, lower_censored, upper_censored)


def _interval_side(excess, bound: float, x_hat: float, rtol: float) -> tuple[float, bool]:
    """Root of the excess between x_hat and the bound, or the bound when none exists.

    A genuine likelihood-ratio crossing has |excess| <= |slope| * xtol at the
    root, well under 1 for any realistic n (slope ~ 2 sqrt(chi2 n curvature),
    about 4e3 at n = 1e6, times the ~1.5e-6 root tolerance). A root that brentq
    places at a jump into the infeasible barrier does not, and is reported
    censored.
    """
    if excess(bound) <= 0.0:
        return bound, True
    root = float(brentq(excess, min(bound, x_hat), max(bound, x_hat), xtol=1e-12, rtol=rtol))
    return root, abs(excess(root)) > 1.0


def profile_plot(values, x_hat, nll_hat, *, scale, alpha, interval, label, ax=None):
    """Likelihood-ratio statistic against the parameter, with the cutoff and the interval."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    points = sorted((x, v) for x, v in values.items() if math.isfinite(v))
    ax.plot([x for x, _ in points], [2.0 * scale * (v - nll_hat) for _, v in points], "o-", ms=3)
    ax.axhline(chi2.ppf(1.0 - alpha, 1), ls="--", c="grey", lw=1)
    ax.axvline(x_hat, c="k", lw=1)
    if interval is not None:
        ax.axvspan(interval.lower, interval.upper, alpha=0.15)
    ax.set_xlabel(label)
    ax.set_ylabel("likelihood-ratio statistic")
    return ax
