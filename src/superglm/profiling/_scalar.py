"""One-parameter profile likelihood: recorded bounded search and likelihood-ratio interval."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import scipy.spatial
from scipy.optimize import brentq, minimize_scalar

from superglm._warn import warn_caller as warn_caller

# An infeasible point's likelihood-ratio excess. Finite so brentq's
# interpolation stays finite; a root whose bracket partner is infeasible marks
# the side censored (see _infeasible_beyond).
_BARRIER = 1e6
# Stand-in objective for an infeasible point inside Brent: finite because the
# parabolic step differences objective values (inf - inf is NaN); its square
# stays far below overflow.
_INFEASIBLE = 1e50
# Times one interval re-centres on a lower point, a budget like maxiter: each
# pass inverts both sides again, a model fit per evaluation.
_MAX_RECENTRINGS = 2


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

    The method is local (as SciPy's fminbound documents): with more than one
    interior minimum it can settle in either. Only an optimum on a bound is
    caught, by `best()`.
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


@dataclass(frozen=True)
class LowerPoint:
    """A recorded point below the centre on the recorded curve.

    ``resolved``: its values' errors, or refined values, show it lower; otherwise
    even values refined to the caller's floor cannot tell which point is lower,
    and ``settled`` says whether those values carried bounds: False when a
    refit did not converge.
    """

    x: float
    nll: float
    resolved: bool
    settled: bool = True


def credible_lower_point(
    objective: RecordedObjective,
    x_hat: float,
    nll_hat: float,
    error: Callable[[float], float],
    refine: Callable[[float, float], tuple[float, float]] | None = None,
) -> LowerPoint | None:
    """A recorded point shown to lie below x_hat, else one that refined values cannot
    place, else None.

    ``error(x)`` bounds the error of the value recorded at x. Values known only
    to a tolerance are told apart only beyond their noise (Moré & Wild 2011);
    the two bounds summed are the 2 eps_f of Shi, Xie, Byrd & Nocedal (2022).
    A drop D within that noise is either noise or a real drop, which no
    threshold on D can tell apart. Without ``refine`` it is taken as noise, and
    only the lowest recorded point is compared. ``refine(x, target)`` returns
    the value at x to within ``target``, or as close as its caller's floor
    allows, with its bound. Both points are refined to within D / 4, so each
    value errs by a fraction of the drop it is to certify, an analogy to the
    dynamic-accuracy condition of trust-region methods (Conn, Gould & Toint
    2000, section 10.6; Cartis, Gould & Toint 2020), which measure accuracy
    against the model's predicted decrease rather than an observed drop. The
    refined pair is compared by the same test, and a pair it leaves undecided
    is refined again at the floor. A point the refined values put above x_hat
    passes the comparison to the next lowest recorded point below x_hat, until
    one is shown lower or none is left. The refined values decide only which
    point is lower: a lower point found is returned with its recorded value,
    the curve the interval inverts.
    """
    values = objective.values
    # Lowest first; equal values in the order recorded, as `RecordedObjective.best`.
    below = sorted((x for x, nll in values.items() if nll < nll_hat), key=values.__getitem__)
    undecided = None
    for x in below:
        drop = nll_hat - values[x]
        if drop > error(x_hat) + error(x):
            return LowerPoint(x, values[x], resolved=True)
        if refine is None:
            return None
        lower, settled = _refined_order(refine, x_hat, x, drop)
        if lower:
            return LowerPoint(x, values[x], resolved=True)
        if lower is None and undecided is None:
            undecided = LowerPoint(x, values[x], resolved=False, settled=settled)
    return undecided


def _refined_order(refine, x_hat, x, drop) -> tuple[bool | None, bool]:
    """Whether x lies below x_hat on values refined to within drop / 4, or failing that at
    the caller's floor (None when neither tells), and whether the last values had bounds."""
    for target in (drop / 4.0, 0.0):
        value_hat, error_hat = refine(x_hat, target)
        value_x, error_x = refine(x, target)
        # A refined value without a bound (inf) or a value (nan) decides nothing.
        if abs(value_hat - value_x) > error_hat + error_x:
            return value_x < value_hat, True
    return None, math.isfinite(error_hat + error_x)


def likelihood_ratio_interval(
    objective: RecordedObjective,
    x_hat: float,
    nll_hat: float,
    bounds: tuple[float, float],
    *,
    alpha: float,
    scale: float,
    xtol: float,
    error: Callable[[float], float],
    refine: Callable[[float, float], tuple[float, float]] | None = None,
) -> tuple[Interval, float, float]:
    """{x : 2 scale (nll(x) - nll_min) <= chi2_1(1 - alpha)} (Venzon & Moolgavkar 1988),
    with the minimum it is measured from.

    Each side is one bracketed root of the excess over the cutoff, found to
    ``xtol``; a side with no crossing before its bound, or whose crossing is a
    jump into infeasible points, is censored where it stopped. A point either
    side evaluates below ``nll_hat`` by more than the two values' ``error``, or
    by less but confirmed by values ``refine`` recomputes (`credible_lower_point`),
    shows x_hat is not the minimum, so the interval is inverted again from that
    point, at most ``_MAX_RECENTRINGS`` times; a drop neither
    resolves leaves x_hat the centre. Like the search, a side assumes one
    crossing: a profile that dips back under the cutoff within one step of its
    outward walk joins that dip to the interval.
    """
    # The upper tail directly: 1 - alpha rounds to 1 for alpha below 1.1e-16.
    from scipy.stats import chi2

    cutoff = float(chi2.isf(alpha, 1))
    recorded = objective.values
    recentrings = 0
    while True:

        def excess(x: float, nll_min: float = nll_hat) -> float:
            value = objective(x)
            return 2.0 * scale * (value - nll_min) - cutoff if math.isfinite(value) else _BARRIER

        lower, lower_censored = _interval_side(excess, recorded, x_hat, bounds[0], cutoff, xtol)
        upper, upper_censored = _interval_side(excess, recorded, x_hat, bounds[1], cutoff, xtol)
        below = credible_lower_point(objective, x_hat, nll_hat, error, refine)
        if below is None or not below.resolved or recentrings == _MAX_RECENTRINGS:
            return Interval(lower, upper, lower_censored, upper_censored), x_hat, nll_hat
        x_hat, nll_hat, recentrings = below.x, below.nll, recentrings + 1


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
    # A cutoff inside the profile's round-off can leave the inner end, x_hat itself, at or
    # above it: the crossing is then the inner end, to within that round-off.
    if excess(inner) >= 0.0:
        return inner, False
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
    import html

    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    points = sorted((x, v) for x, v in values.items() if math.isfinite(v))
    ax.plot([x for x, _ in points], [2.0 * scale * (v - nll_hat) for _, v in points], "o-", ms=3)
    ax.axhline(chi2.isf(alpha, 1), ls="--", c="grey", lw=1)
    ax.axvline(x_hat, c="k", lw=1)
    if interval is not None:
        ax.axvspan(interval.lower, interval.upper, alpha=0.15)
    ax.set_xlabel(label)
    ax.set_ylabel("likelihood-ratio statistic")
    return ax
