"""One-parameter profile likelihood: recorded bounded search and likelihood-ratio interval."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

# An infeasible point's likelihood-ratio excess. Finite so brentq's
# interpolation stays finite; a root whose bracket partner is infeasible marks
# the side censored (see _infeasible_beyond).
_BARRIER = 1e6
# Stand-in objective for an infeasible point inside Brent: finite because the
# parabolic step differences objective values (inf - inf is NaN); its square
# stays far below overflow.
_INFEASIBLE = 1e50
# Frames under this directory are superglm's own; a warning skips them to land
# on the caller's line however deep the call (Python 3.12, skip_file_prefixes).
_PACKAGE_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def warn_caller(message: str, category: type[Warning] = UserWarning) -> None:
    """Warn at the first frame outside superglm."""
    warnings.warn(message, category, skip_file_prefixes=(_PACKAGE_PREFIX,))


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

    The method is local (a local minimiser in the interval, as SciPy's
    fminbound documents it): on a profile with more than one interior minimum
    it can settle in either, and nothing here detects the other. Only an
    optimum on a bound is caught, by `best()`.
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


def censoring_warnings(
    interval: Interval, alpha: float, label: str, where: Callable[[float], str]
) -> list[str]:
    """One warning per censored side, naming where the interval stopped (``where(end)``)."""
    sides = (
        ("lower", interval.lower, interval.lower_censored),
        ("upper", interval.upper, interval.upper_censored),
    )
    return [
        f"the {100.0 * (1.0 - alpha):g}% interval for {label} is censored at its {side} end "
        f"{label}={end:.6g}, {where(end)}: the likelihood-ratio statistic does not reach its "
        "cutoff there, so the interval may extend beyond it."
        for side, end, censored in sides
        if censored
    ]


def likelihood_ratio_interval(
    objective: RecordedObjective,
    x_hat: float,
    nll_hat: float,
    bounds: tuple[float, float],
    *,
    alpha: float,
    scale: float,
    xtol: float,
) -> Interval:
    """{x : 2 scale (nll(x) - nll_hat) <= chi2_1(1 - alpha)} around x_hat (Venzon & Moolgavkar 1988).

    Each side is one bracketed root of the excess over the cutoff, found to
    ``xtol``; a side with no crossing before its bound, or whose crossing is a
    jump into infeasible points, is censored where it stopped.
    """
    cutoff = float(chi2.ppf(1.0 - alpha, 1))

    def excess(x: float) -> float:
        value = objective(x)
        return 2.0 * scale * (value - nll_hat) - cutoff if math.isfinite(value) else _BARRIER

    recorded = objective.values
    lower, lower_censored = _interval_side(excess, recorded, x_hat, bounds[0], cutoff, xtol)
    upper, upper_censored = _interval_side(excess, recorded, x_hat, bounds[1], cutoff, xtol)
    return Interval(lower, upper, lower_censored, upper_censored)


def _interval_side(excess, recorded, x_hat, bound, cutoff, xtol) -> tuple[float, bool]:
    """Root of the excess between x_hat and the bound, or the bound when none exists.

    Each evaluation is a model fit, so the walk starts where a quadratic
    profile predicts the crossing and doubles its step toward the bound until
    the excess turns positive: a near-quadratic profile is bracketed at the
    first or second point, and only a censored side reaches the bound.
    """
    step = _predicted_crossing(excess, recorded, x_hat, bound, cutoff)
    inner, outer = x_hat, _toward(x_hat, bound, step)
    while excess(outer) <= 0.0:
        if outer == bound:
            return bound, True
        step *= 2.0
        inner, outer = outer, _toward(x_hat, bound, step)
    root = float(brentq(excess, min(inner, outer), max(inner, outer), xtol=xtol))
    return root, excess(root) < 0.0 and _infeasible_beyond(recorded, root, bound)


def _predicted_crossing(excess, recorded, x_hat, bound, cutoff) -> float:
    """Distance from x_hat to the crossing of a quadratic profile, or to the bound.

    The quadratic passes through x_hat and the recorded point on this side
    whose statistic is nearest the cutoff, the point whose curvature matters
    most for this crossing. With no such point the walk starts at the bound.
    """
    side = [
        x
        for x, value in recorded.items()
        if math.isfinite(value) and (x - x_hat) * (bound - x_hat) > 0.0 and excess(x) > -cutoff
    ]
    if not side:
        return abs(bound - x_hat)
    nearest = min(side, key=lambda x: abs(excess(x)))
    return abs(nearest - x_hat) * math.sqrt(cutoff / (excess(nearest) + cutoff))


def _toward(x_hat: float, bound: float, step: float) -> float:
    """The point ``step`` from x_hat toward the bound, the bound itself once the step reaches it."""
    return bound if step >= abs(bound - x_hat) else x_hat + math.copysign(step, bound - x_hat)


def _infeasible_beyond(recorded, root: float, bound: float) -> bool:
    """Whether the evaluated point next beyond the root is infeasible.

    brentq keeps a sign-change bracket, so that point is the root's partner:
    a likelihood-ratio crossing has a feasible partner, a jump into the
    infeasible region an infeasible one, however small the bracket.
    """
    beyond = [x for x in recorded if (x - root) * (bound - root) > 0.0]
    partner = min(beyond, key=lambda x: abs(x - root))
    return not math.isfinite(recorded[partner])


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
