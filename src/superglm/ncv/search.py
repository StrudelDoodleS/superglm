"""The search for the smoothing that minimises the neighbourhood cross-validation criterion.

The criterion ``V(rho)`` of :mod:`superglm.ncv.engine`, ``rho = log lambda`` of the estimated
penalty components, is smooth, not convex, flat at both ends of every ``rho_j``, and comes with
its gradient.  It is minimised inside the fit's smoothing box by L-BFGS-B (Byrd, Lu, Nocedal and
Zhu, "A limited memory algorithm for bound constrained optimization", SIAM J. Sci. Comput.
16(5), 1995; ``scipy.optimize.minimize``).  This module owns everything around the optimiser:
where it starts, the units it works in, what it is handed at an infeasible trial, when it stops
and what the stop is called.  The optimiser's own success flag is never read.

**What is promised.**  A converged search is a first-order stop of a local numerical search:
at the returned smoothing no slope of the criterion exceeds the tolerance, and every
criterion it evaluated came from a certified inner fit.  It is not a proof of a minimum, and
no curvature is examined.

**Units.**  ``V`` carries the units of the loss and ``rho`` none, so every test is relative to
``V`` or to its slope and the chosen smoothing does not depend on the units of the response:

- the first-order test is ``max_j |g_j| <= tol |V|`` over the slopes ``g = dV/d rho`` that are
  not projected out (below).  A criterion of exactly zero has zero slopes and is its least
  possible value, a sum of deviances;
- the optimiser works on ``V / s``, ``s`` the largest such slope at its start.  L-BFGS-B's
  first step in a box is the negative gradient itself, so this moves the steepest component by
  one unit of ``log lambda``;
- the optimiser's own stop is its relative-reduction test on ``V / s`` at ``tol**2`` (SciPy:
  ``(f_k - f_(k+1)) / max(|f_k|, |f_(k+1)|, 1) <= ftol``).  Where the first-order test just
  fails, a Newton step gains about ``(tol V)^2 / (2 V'')``, so a step that gains less than
  ``tol^2 V`` has stalled unless the curvature ``V''`` in ``log lambda`` exceeds half the
  criterion.  It ends a search that has stopped moving; such a search has not converged.  The
  optimiser's projected-gradient test is switched off.

**Bounds.**  A component sits at a bound by the box's own test (``LambdaBox.side``).  Its slope
is projected out of the first-order test only there, and only when it points out of the box:
a positive slope at the floor, a negative one at the ceiling.  A component just inside a bound
keeps its slope.  Whether a component is at a bound and whether the bound holds it back (an
outward slope of at least the stopping bar) are reported separately.  Fixed components are not
searched and enter neither.

**Starts.**  The default start is the half-EDF point of :func:`edf_share_start`, a per-penalty
heuristic read from the provisional fit's data Gram, with a caller's start for any estimated
component it names.  The criterion is flat where every penalty is negligible or dominant, and
can be stationary at a maximum, so a start at which the search takes no step is not taken as
the answer while another start remains: the half-EDF point, then the configured smoothing
(where the provisional fit was usable), are searched in turn until one moves, and the
lowest end is returned.  A start with no finite criterion passes to the next in the same way.

**Infeasible trials.**  A trial with no usable inner mode, or no finite criterion and
gradient, is never handed to the optimiser as it is: it reads as far above the point the search
stands on as the step to it promised to go below, with a zero slope, which a line search
rejects.  The search stands only on feasible points, each lower than the last, so a search
that fails returns the lowest point it reached with the stop that ended it, and that zero slope
enters no test.  Every infeasible trial is counted under its reason
(:data:`~superglm.ncv.result.TRIAL_FAILURES`), so a search that fails for want of a feasible
trial says why.  Rows that do not identify the model once a neighbourhood is removed are the
engine's refusal (:class:`~superglm.ncv.engine.LeaveOutRankError`): true at every smoothing,
it is raised when it comes before any criterion has been found, and later it is an infeasible
trial.  A trial the engine could not afford (more kept-rows fits than ``engine.KEPT_FITS``) or
could not resolve at its smoothing is infeasible and never a refusal.

**Fits at the limit of float64.**  An inner fit whose solve the solver ended as stagnated,
short of its bar, stands for the mode (``engine.ModeFit.usable``): its trial is a point like
any other, it warms the next fit, and the search goes on from it.  The search's stop is
then what its test found on those values, and ``termination_reason`` says so, but a search
that took any criterion from such a fit is not converged: the count is
``work["stagnated_fits"]``, and the fit is disclosed where it is published.  A fit stopped by
its iteration limit is no such fit, and its trial is infeasible.

**Rows read.**  The search asks for no held-out pairs, so a Gaussian model whose neighbourhoods
are formed and factored reads its rows at the first evaluation only; the pairs are taken once,
from the fit that is published.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.optimize import OptimizeResult, minimize

from superglm._lambda_box import LambdaBox, glm_lambda_box
from superglm._roundoff import EPS
from superglm.ncv.engine import (
    DEFAULT_NCV_TOL,
    Evaluation,
    LeaveOutRankError,
    ModeFit,
    NCVProblem,
    Route,
    WorkCounts,
    evaluate,
    solve_mode,
)
from superglm.ncv.neighbourhoods import Neighbourhoods
from superglm.ncv.result import NCVResult, StopReason, TrialFailure

DEFAULT_MAX_NCV_ITER = 100

START_SHARE = 0.5
"""The share of a penalty's range the default start leaves unpenalised."""

MEMORY = 10
"""L-BFGS-B's correction pairs."""

LINE_SEARCH_TRIALS = 20
"""L-BFGS-B's trials per line search."""

_HALVINGS = 60
"""Bisection steps for a start: they take a bracket no wider than ``log(1 / EPS)`` below the
spacing of float64 there."""

_FLOAT_MAX = float(np.finfo(np.float64).max)

Criterion = Callable[[NDArray[np.float64], "ModeFit | None"], Evaluation]
"""The criterion at ``rho``, its inner fit started from the fit given."""


@dataclass(frozen=True)
class Point:
    """A smoothing with a finite criterion and gradient: what a search may stand on."""

    rho: NDArray[np.float64]
    value: float
    gradient: NDArray[np.float64]


@dataclass(frozen=True)
class FirstOrder:
    """A point's first-order state in the box.

    ``sides`` is each component's bound (``"floor"``, ``"ceiling"`` or None), ``outward`` marks
    a slope that points out of the box at one, ``projected`` is the gradient with those zeroed,
    and ``bar`` is ``tol |V|``.
    """

    sides: tuple[str | None, ...]
    outward: NDArray[np.bool_]
    gradient: NDArray[np.float64]
    bar: float

    @property
    def projected(self) -> NDArray[np.float64]:
        """The gradient without the slopes a bound blocks."""
        return np.where(self.outward, 0.0, self.gradient)

    @property
    def met(self) -> bool:
        """Whether no free slope exceeds the bar."""
        return bool(np.max(np.abs(self.projected), initial=0.0) <= self.bar)

    def at_bound(self, names: Sequence[str]) -> tuple[tuple[str, str], ...]:
        """``(name, side)`` of each component at a bound."""
        return tuple(
            (name, side) for name, side in zip(names, self.sides, strict=True) if side is not None
        )

    def pressure(self, names: Sequence[str]) -> tuple[tuple[str, str], ...]:
        """``(name, side)`` of each component a bound holds back by the bar or more."""
        held = self.outward & (np.abs(self.gradient) >= self.bar)
        return tuple(
            (name, side)
            for name, side, on in zip(names, self.sides, held.tolist(), strict=True)
            if on and side is not None
        )


def first_order(
    rho: NDArray[np.float64],
    gradient: NDArray[np.float64],
    value: float,
    box: LambdaBox,
    tol: float,
) -> FirstOrder:
    """The first-order state of the criterion ``value`` with slopes ``gradient`` at ``rho``."""
    sides = tuple(box.side(float(lam)) for lam in np.exp(rho))
    outward = np.array(
        [
            (side == "floor" and slope > 0.0) or (side == "ceiling" and slope < 0.0)
            for side, slope in zip(sides, gradient.tolist(), strict=True)
        ],
        dtype=bool,
    )
    return FirstOrder(sides, outward, gradient, tol * abs(value))


@dataclass
class Descent:
    """Searches of one criterion from a sequence of starts, and what they found.

    ``ends`` holds, for every start with a finite criterion, the point its search ended on and
    why; ``path`` each such start and each accepted step.  ``warm`` is the last usable inner
    fit, which starts the next one and carries the bar it runs to (``engine.first_bar``).
    ``reasons`` counts the infeasible trials by reason.
    """

    criterion: Criterion
    box: LambdaBox
    tol: float = DEFAULT_NCV_TOL
    max_iter: int = DEFAULT_MAX_NCV_ITER
    warm: ModeFit | None = None
    ends: list[tuple[Point, StopReason]] = field(default_factory=list)
    path: list[Point] = field(default_factory=list)
    steps: int = 0
    evaluations: int = 0
    infeasible: int = 0
    reasons: dict[TrialFailure, int] = field(default_factory=dict)
    work: WorkCounts = field(default_factory=WorkCounts)
    _seen: dict[bytes, Point | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.tol) and self.tol > 0.0):
            raise ValueError(
                f"The search tolerance must be a positive number, got {self.tol!r}; "
                f"the default is {DEFAULT_NCV_TOL:g}."
            )
        if self.max_iter < 1:
            raise ValueError(
                f"The search needs at least one step, got a limit of {self.max_iter!r}; "
                f"the default is {DEFAULT_MAX_NCV_ITER}."
            )

    def at(self, rho: NDArray[np.float64]) -> Point | None:
        """The criterion at ``rho``, evaluated once; None where it is infeasible."""
        key = rho.tobytes()
        if key not in self._seen:
            self._seen[key] = self._evaluate(np.array(rho, dtype=np.float64))
        return self._seen[key]

    def _evaluate(self, rho: NDArray[np.float64]) -> Point | None:
        self.evaluations += 1
        try:
            evaluation = self.criterion(rho, self.warm)
        except LeaveOutRankError:
            if not any(point is not None for point in self._seen.values()):
                raise
            return self._infeasible("leave_out_refused")
        for count in fields(WorkCounts):
            spent = getattr(self.work, count.name) + getattr(evaluation.work, count.name)
            setattr(self.work, count.name, spent)
        if evaluation.fit is not None:
            self.warm = evaluation.fit
        slopes = evaluation.gradient
        finite = (
            slopes is not None and np.isfinite(evaluation.value) and np.all(np.isfinite(slopes))
        )
        if not (evaluation.feasible and finite):
            return self._infeasible(_trial_failure(evaluation))
        return Point(rho, float(evaluation.value), np.asarray(slopes, dtype=np.float64))

    def _infeasible(self, reason: TrialFailure) -> None:
        """Count a trial that gave no point, under ``reason``."""
        self.infeasible += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def first_order(self, point: Point) -> FirstOrder:
        """``point``'s first-order state at this search's tolerance."""
        return first_order(point.rho, point.gradient, point.value, self.box, self.tol)

    def descend(self, starts: Sequence[NDArray[np.float64]]) -> None:
        """Search from each start in turn until a search accepts a step."""
        for start in starts:
            if self.run(start):
                return

    def run(self, start: NDArray[np.float64]) -> bool:
        """Search from ``start``, a point of the box; whether a step was accepted."""
        here = self.at(start)
        if here is None:
            return False
        self.path.append(here)
        order = self.first_order(here)
        if order.met:
            self.ends.append((here, "gradient"))
            return False
        run = _Run(self, here, float(np.max(np.abs(order.projected))))
        halted = minimize(
            run.objective,
            here.rho,
            jac=True,
            method="L-BFGS-B",
            bounds=[(self.box.log_floor, self.box.log_ceiling)] * len(here.rho),
            callback=run.accept,
            options={
                "gtol": 0.0,
                "ftol": self.tol * self.tol,
                "maxiter": self.max_iter,
                # More than max_iter steps can ask for (a line search and one restarted line
                # search each), so the step limit is the only budget.
                "maxfun": 2 * LINE_SEARCH_TRIALS * self.max_iter + 1,
                "maxcor": MEMORY,
                "maxls": LINE_SEARCH_TRIALS,
            },
        )
        self.steps += run.steps
        self.ends.append((run.here, run.stop or _optimiser_stop(halted, self.max_iter)))
        return run.steps > 0

    def outcome(self) -> tuple[Point | None, StopReason]:
        """The lowest end of the searches and its stop; no point when none had a finite criterion."""
        if not self.ends:
            return None, "no_finite_criterion"
        lowest = int(np.argmin([point.value for point, _ in self.ends]))
        return self.ends[lowest]


def _trial_failure(evaluation: Evaluation) -> TrialFailure:
    """Why ``evaluation`` gave the search no point.

    The engine's own reason, except that its ``"leave_out_rank"`` at the limit on kept-rows
    fits is ``"kept_rows_limit"``: a trial it could not afford, not one it could not resolve.
    An evaluation the engine calls feasible with a value or slope that is not finite is
    ``"non_finite"``.
    """
    reason = evaluation.infeasible_reason
    if reason is None:
        return "non_finite"
    if reason == "leave_out_rank" and evaluation.past_kept_fits:
        return "kept_rows_limit"
    return reason


def _optimiser_stop(halted: OptimizeResult, max_iter: int) -> StopReason:
    """Why L-BFGS-B ended a search by itself.

    With its projected-gradient test off, its only convergence is the relative-reduction test;
    anything that is neither that nor the step limit is a line search that found no step.
    """
    if halted.nit >= max_iter:
        return "max_iter"
    return "criterion_settled" if halted.status == 0 else "line_search"


@dataclass
class _Run:
    """One L-BFGS-B search: the point it stands on, in the optimiser's units ``V / scale``."""

    descent: Descent
    here: Point
    scale: float
    steps: int = 0
    stop: StopReason | None = None

    def scaled(self, point: Point | None) -> tuple[float, NDArray[np.float64]] | None:
        """``point`` in the optimiser's units; None when it is infeasible or not finite there."""
        if point is None:
            return None
        value, slopes = point.value / self.scale, point.gradient / self.scale
        return (value, slopes) if np.isfinite(value) and np.all(np.isfinite(slopes)) else None

    def objective(self, rho: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """What the optimiser is handed at ``rho``: never a number that is not finite.

        An infeasible trial reads as far above the point the search stands on as the step to
        it promised, to first order, to go below (at least the next float64 above), with a zero
        slope.  The line search rejects it and shortens the step by a share that does not depend
        on the criterion's scale along the step.
        """
        handed = self.scaled(self.descent.at(rho))
        if handed is not None:
            return handed
        level = self.here.value / self.scale
        promised = abs(float(self.here.gradient @ (rho - self.here.rho))) / self.scale
        worse = max(level + promised, float(np.nextafter(level, np.inf)))
        return min(worse, _FLOAT_MAX), np.zeros(len(rho))

    def accept(self, rho: NDArray[np.float64]) -> None:
        """Stand on the optimiser's new iterate, and stop it when the first-order test is met.

        An iterate that is infeasible, or no lower than the point stood on, is a line search
        that ended without a step: the search stops where it stands.
        """
        point = self.descent.at(rho)
        if point is None or self.scaled(point) is None or not point.value < self.here.value:
            self.stop = "line_search"
            raise StopIteration
        self.here = point
        self.steps += 1
        self.descent.path.append(point)
        if self.descent.first_order(point).met:
            self.stop = "gradient"
            raise StopIteration


@dataclass(frozen=True)
class SearchOutcome:
    """A finished search.

    ``fit`` is the last usable inner fit, at or near the returned smoothing: the state to
    start the terminal fit from.  It lives in the search's basis and is not part of ``result``.
    """

    result: NCVResult
    fit: ModeFit | None


def search(
    problem: NCVProblem,
    nb: Neighbourhoods,
    *,
    configured: Mapping[str, float],
    start: Mapping[str, float] | None = None,
    box: LambdaBox | None = None,
    tol: float = DEFAULT_NCV_TOL,
    max_iter: int = DEFAULT_MAX_NCV_ITER,
    route: Route | None = None,
) -> SearchOutcome:
    """Choose the estimated smoothing parameters of ``problem`` by its criterion over ``nb``.

    ``configured`` gives every estimated component the smoothing the model was declared with:
    the provisional fit is solved there, and it is the last start tried.  ``start`` names
    estimated components by component name and the smoothing to start each from; a name that is
    not an estimated component of ``problem`` is not read, so a fixed component keeps its value
    in every inner fit.  ``box`` defaults to the fit's own (``glm_lambda_box``).  ``tol`` is the
    first-order tolerance relative to the criterion and ``max_iter`` the limit on accepted
    steps.

    Every evaluation is made without held-out pairs: the result holds none, and they are read
    from the fit published at the returned smoothing (``evaluate_fit``).  A model with nothing
    to estimate is evaluated once and the optimiser is not called.
    """
    box = _fit_box(problem) if box is None else box
    descent = Descent(
        criterion=lambda rho, warm: evaluate(
            problem, nb, rho, warm=warm, heldout=False, route=route
        ),
        box=box,
        tol=float(tol),
        max_iter=int(max_iter),
    )
    if not problem.names:
        only = descent.at(np.zeros(0))
        if only is not None:
            descent.path.append(only)
            descent.ends.append((only, "no_smoothing_parameters"))
        return _outcome(problem, descent, np.zeros(0))
    declared = np.array([_log_lambda(box, name, configured[name]) for name in problem.names])
    asked = {
        name: _log_lambda(box, name, lam)
        for name, lam in (start or {}).items()
        if name in problem.names
    }
    provisional = solve_mode(problem, declared, work=descent.work)
    default = declared
    if provisional.usable:
        descent.warm = provisional
        default = edf_share_start(problem, provisional, box, declared)
    first = np.array(
        [asked.get(name, rho) for name, rho in zip(problem.names, default.tolist(), strict=True)]
    )
    descent.descend(_distinct([first, default, declared]))
    return _outcome(problem, descent, first)


def _fit_box(problem: NCVProblem) -> LambdaBox:
    return glm_lambda_box(
        distribution=problem.distribution,
        link=problem.link,
        y=problem.y,
        sample_weight=problem.weights,
        offset=problem.offset,
        weight_semantics=problem.weight_semantics,
    )


def _log_lambda(box: LambdaBox, name: str, value: float) -> float:
    """``log`` of a smoothing parameter given for ``name``, inside the box."""
    lam = float(value)
    if not (np.isfinite(lam) and lam > 0.0):
        raise ValueError(
            f"The smoothing parameter given for {name!r} must be a positive number, got "
            f"{lam!r}; give one, or leave it out to use the default start."
        )
    return float(np.log(box.clip(lam)))


def _distinct(starts: Sequence[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    """``starts`` without a later copy of an earlier one."""
    kept: list[NDArray[np.float64]] = []
    for start in starts:
        if not any(np.array_equal(start, other) for other in kept):
            kept.append(start)
    return kept


def _outcome(problem: NCVProblem, descent: Descent, first: NDArray[np.float64]) -> SearchOutcome:
    """The result of ``descent``, whose first start was ``first``.

    With no finite criterion the result is at that start, with no criterion and no gradient.
    A search is converged when its stop met the first-order test and no criterion it
    evaluated came from a fit at the limit of float64 (``work["stagnated_fits"]``).
    """
    point, reason = descent.outcome()
    rho = first if point is None else point.rho
    slopes = np.zeros(len(rho)) if point is None else point.gradient
    order = first_order(
        rho, slopes, 0.0 if point is None else point.value, descent.box, descent.tol
    )
    result = NCVResult(
        lambdas=_lambdas(problem, rho),
        pirls_result=None,
        n_reml_iter=descent.steps,
        converged=reason in ("gradient", "no_smoothing_parameters")
        and descent.work.stagnated_fits == 0,
        lambda_history=[_lambdas(problem, visited.rho) for visited in descent.path],
        objective=None if point is None else point.value,
        reml_penalties=None,
        objective_history=[visited.value for visited in descent.path],
        termination_reason=reason,
        lambda_box=descent.box,
        bound_pressure=() if point is None else order.pressure(problem.names),
        n_evaluations=descent.evaluations,
        infeasible_trials=descent.infeasible,
        infeasible_reasons=dict(descent.reasons),
        gradient=None if point is None else point.gradient.copy(),
        start_lambdas={
            name: lam for name, lam in _lambdas(problem, first).items() if name in problem.names
        },
        at_bound=order.at_bound(problem.names),
        work=asdict(descent.work),
    )
    return SearchOutcome(result, descent.warm)


def _lambdas(problem: NCVProblem, rho: NDArray[np.float64]) -> dict[str, float]:
    """Every component's smoothing parameter at ``rho``, in the penalties' order."""
    every = problem.lambdas(rho)
    return {component.name: every[component.name] for component in problem.penalties}


def edf_share_start(
    problem: NCVProblem,
    fit: ModeFit,
    box: LambdaBox,
    fallback: NDArray[np.float64],
    share: float = START_SHARE,
) -> NDArray[np.float64]:
    """``log lambda`` at which each estimated penalty, taken alone, leaves ``share`` of its range.

    A heuristic for a start, not an estimate.  For one component with penalty ``S`` on its
    term's columns and the data's Gram ``D`` on them (``fit.data_gram``, the solver's own),
    the eigenvalues ``g_i > 0`` of ``S v = g D v`` give the degrees of freedom its range keeps,
    ``sum_i 1 / (1 + lambda g_i)``, were it the only penalty on that term; the start is the
    ``lambda`` at which that is ``share`` of the range's dimension.  mgcv documents the same
    aim for its starting values (``?initial.sp``).  Each component is treated on its own:
    penalties that share a term, and correlated terms, each leave less than ``share`` once
    fitted together.

    Both are read in the design's own columns, on the whole of the term, also where the fit is
    rank deficient: the pair on one term's columns does not depend on the coordinates the fit
    identified, so a column aliased with another term's leaves a term's start where it is
    without the alias.

    Where ``D`` is not positive definite the single ratio ``tr(S) / tr(D)`` stands for the
    eigenvalues.  A component whose penalty or data Gram has no positive trace has no range
    this can read, and starts at ``fallback``; so does every component of a fit that carries
    no data Gram.
    """
    rho = np.array(fallback, dtype=np.float64)
    data = fit.data_gram
    if data is None:
        return np.clip(rho, box.log_floor, box.log_ceiling)
    for j, (columns, block) in enumerate(problem.blocks):
        ratios = _penalty_over_data(block, data[columns, columns])
        if ratios is not None:
            rho[j] = _share_point(ratios, share)
    return np.clip(rho, box.log_floor, box.log_ceiling)


def _penalty_over_data(
    S: NDArray[np.float64], D: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """The positive eigenvalues of ``S v = g D v``; None when there is no range to read."""
    if not (np.trace(S) > 0.0 and np.trace(D) > 0.0):
        return None
    try:
        ratios = eigh(S, D, eigvals_only=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.array([np.trace(S) / np.trace(D)])
    ratios = ratios[ratios > len(ratios) * EPS * max(float(ratios.max()), 0.0)]
    return ratios if len(ratios) else None


def _share_point(ratios: NDArray[np.float64], share: float) -> float:
    """``log lambda`` at which ``sum_i 1 / (1 + lambda g_i)`` is ``share`` of the count of ``g``.

    The sum falls as ``lambda`` grows, and the root lies between the points at which the
    largest and the smallest ``g`` alone reach ``share``; bisection from that bracket.
    """
    odds = 1.0 / share - 1.0
    low, high = float(np.log(odds / ratios.max())), float(np.log(odds / ratios.min()))
    for _ in range(_HALVINGS):
        middle = 0.5 * (low + high)
        if np.sum(1.0 / (1.0 + np.exp(middle) * ratios)) > share * len(ratios):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


search_smoothing = search
"""``search`` under the name :mod:`superglm.ncv` exports: the package's ``search`` is this module."""

__all__ = [
    "DEFAULT_MAX_NCV_ITER",
    "Descent",
    "FirstOrder",
    "Point",
    "SearchOutcome",
    "edf_share_start",
    "first_order",
    "search",
    "search_smoothing",
]
