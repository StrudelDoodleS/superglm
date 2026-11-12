"""``Constraint.postfit`` rules on SuperLSS terms, repaired after the fit as SuperGLM repairs them.

A post-fit rule asks for the fitted curve to be repaired, not for the fit to be constrained.
``SuperGLM.apply_shape_postfit`` projects each such term's fitted coefficients onto its shape
cone in the term's own fitted basis, in the term's penalized Hessian at the fit
(``MonotoneRepairer`` and ``CurvatureRepairer`` in ``superglm.constraints``), re-estimates the
intercept with the terms held, and withholds the repaired term's standard errors, bands and test:
the projection is a constrained estimator whose reference distribution is not the unconstrained
fit's (Meyer and Woodroofe, *Ann. Statist.* 28(4), 2000; Meyer, *Ann. Appl. Statist.* 2(3),
2008), while the point estimate needs no such caveat (Chernozhukov, Fernandez-Val and Galichon,
*Biometrika* 96(3), 2009, Proposition 1).

SuperLSS runs the same repairer on the same fitted basis, for a term of any parameter, on that
parameter's link scale, in the same metric: the penalized curvature at the coefficients being
repaired, the fit's own at the fit, on the term, with that parameter's intercept profiled
(``_term_precision``, ``_curvature_at``). The basis carries the fit's centring, so a repair keeps
the term's training mean on that scale; the parameter's intercept is then re-estimated on the
joint likelihood of the retained rows with every term and every other parameter held. A
projection the dropped term beats is continued as SuperGLM continues one
(``_continued_repair``), and where no curve the rule allows beats it, the term is published
flat, with a warning. The result is published as a revision of the fit (``revision.py``):
predictions, curves and rating tables follow it, while ``result_`` and the covariance keep
describing the fit.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.results.shape import ShapeRepairRecord
from superglm.distributional.solver.packing import packed_pairs
from superglm.features.constraint import rules_of
from superglm.links import _is_shifted_log_link

REPAIR_NEEDS_THE_ROWS = (
    "A post-fit shape repair weights its projection by the training rows and re-estimates the "
    "intercept on them; refit with retain_rows=True before calling apply_shape_postfit()."
)
# The search for the re-estimated intercept runs in units of the parameter's own scale on its
# link (``_intercept_scale``), so that the response's units do not change it. It is Newton's
# method on the likelihood's slope in the shift, kept inside a bracket (``_newton_minimum``).
# Where the family or link gives no derivatives there, a derivative-free search decides
# (``_derivative_free_minimum``): bracket, then Brent (Brent, *Algorithms for Minimization
# without Derivatives*, 1973, ch. 5; Press et al., *Numerical Recipes*, 3rd ed., section 10.1).
# Brent first runs within ``_INTERCEPT_SEARCH`` of those units of the held intercept. Where its
# result is no lower than an end, the bracket walks downhill from that end, each step growing
# by ``_INTERCEPT_GROWTH``, until the objective rises, a row turns invalid, or the shift leaves
# float64's range. No count limits it, so an optimum any number of units away is bracketed in
# steps logarithmic in its distance.
_INTERCEPT_SEARCH = 4.0
# Each downhill step is the golden ratio times the last, ``mnbrak``'s default growth.
_INTERCEPT_GROWTH = (1.0 + np.sqrt(5.0)) / 2.0
_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0
# A smooth minimum is located to about sqrt(u) of its scale, since the objective is flat to
# rounding within that distance of it (Press et al., section 10.1); Brent stops there, and
# Newton's method once its step is that short.
_INTERCEPT_XATOL = float(np.sqrt(_UNIT_ROUNDOFF))
# Newton's method converges quadratically to a minimum of positive curvature, and its bracket
# halves at least every other step, so this many steps hold any search that is not walking out
# toward a distant minimum; beyond them the derivative-free search decides.
_NEWTON_STEPS = 100
# Where the held intercept leaves rows invalid, the search starts from the nearest valid shift
# among doublings both ways, from 2**-60 of ``_INTERCEPT_WIDEST`` units up to it.
_INTERCEPT_WIDEST = 2.0**20
_INTERCEPT_PROBES = 61
# Where no probe is valid, the gaps between them are bisected (``_between_probes``) with at
# most this many evaluations. Where each midpoint's invalid rows say which side the valid
# shifts lie on, the gap holding them is bisected to adjacent floats, about 53 of them. Where
# a midpoint's rows cannot say (an evaluation refused as a whole has no row-wise answer),
# both halves are searched, breadth first over every gap that may hold the band: with ``G``
# such gaps, a band wider than 2**-(d+1) of its gap is found once G * (2**(d+1) - 1) <= 256.
_BETWEEN_PROBES = 256


class PostfitState(Enum):
    """Where one term's ``Constraint.postfit`` rule stands on the model that predicts.

    The stored state is one repair record per term (``HandEdit.repairs``, written only by
    ``revision.revised_model``); this is that record read against the curve the model predicts
    with now, by ``postfit_status`` only.
    """

    OBEYED = "obeyed"  # not repaired, and the fitted curve obeys the rule
    PENDING = "pending"  # not repaired, and the fitted curve breaks the rule
    REPAIRED = "repaired"  # repaired, and the curve obeys the rule
    BROKEN = "broken"  # repaired, then revised so that the curve breaks the rule again


_UNREPAIRED = frozenset({PostfitState.OBEYED, PostfitState.PENDING})
_REPAIRED = frozenset({PostfitState.REPAIRED, PostfitState.BROKEN})
#: The moves of one term's state, by event. ``repair`` is ``apply_shape_postfit``; ``edit`` is a
#: coefficient revision (a hand edit, the editor's ``to_model``), which keeps every repair
#: record; ``refit`` is a fit, which starts from no revision.
POSTFIT_MOVES = MappingProxyType(
    {
        (PostfitState.OBEYED, "repair"): frozenset({PostfitState.OBEYED}),
        (PostfitState.PENDING, "repair"): frozenset({PostfitState.REPAIRED}),
        (PostfitState.REPAIRED, "repair"): frozenset({PostfitState.REPAIRED}),
        (PostfitState.BROKEN, "repair"): frozenset({PostfitState.REPAIRED}),
        **{(state, "edit"): _UNREPAIRED for state in _UNREPAIRED},
        **{(state, "edit"): _REPAIRED for state in _REPAIRED},
        **{(state, "refit"): _UNREPAIRED for state in PostfitState},
    }
)


@dataclass(frozen=True)
class PostfitTerm:
    """One term that carries a ``Constraint.postfit`` rule, in the fitted model's coordinates.

    ``curve_spec`` is the spline whose coefficient space the repair works in (the term's own,
    or an ordered term's inner spline), ``None`` for a basis that states no curve.
    """

    term: str
    parameter: str
    label: Hashable
    kind: str
    spec: Any
    curve_spec: Any
    columns: slice


def _curve_spec_or_none(term: str, spec: Any) -> Any:
    from superglm.model.shape_ops import _curve_spec

    try:
        return _curve_spec(term, spec)
    except NotImplementedError:
        return None


def postfit_terms(fitted: Any) -> tuple[PostfitTerm, ...]:
    """Every term of every parameter whose declaration carries a ``Constraint.postfit`` rule."""

    state = fitted.fit_state
    found: list[PostfitTerm] = []
    for template, compiled in zip(
        state.predictor_templates, state.compiled_predictors, strict=True
    ):
        for label, declared in template.features.items():
            # A post-fit rule is always one whole-curve shape: ``Constraint.postfit`` refuses
            # combinations, and a term carries one ``constraint=``.
            rule = next((r for r in rules_of(declared) if r.mode == "postfit"), None)
            if rule is None:
                continue
            term = f"{template.name}:{label}"
            spec = compiled.compiled.specs[label]
            found.append(
                PostfitTerm(
                    term=term,
                    parameter=template.name,
                    label=label,
                    kind=rule.kind,
                    spec=spec,
                    curve_spec=_curve_spec_or_none(term, spec),
                    columns=state.layout.term_slices[term],
                )
            )
    return tuple(found)


def postfit_term_feasible(term: PostfitTerm, coefficients: NDArray[np.float64]) -> bool:
    """Whether the term's curve at ``coefficients`` already obeys its rule, to rounding.

    A basis that states no curve has nothing to check; ``repaired_model`` refuses it.
    """
    from superglm.constraints import shape_constraint_is_roundoff_feasible

    if term.curve_spec is None:
        return False

    return bool(
        shape_constraint_is_roundoff_feasible(
            term.curve_spec, np.asarray(coefficients[term.columns]), term.kind
        )
    )


@dataclass(frozen=True)
class PostfitStatus:
    """One term's post-fit rule, its state, and its repair record if it has one."""

    term: PostfitTerm
    state: PostfitState
    record: ShapeRepairRecord | None


def postfit_status(fitted: Any) -> tuple[PostfitStatus, ...]:
    """The state of every ``Constraint.postfit`` rule of the model, read on its current curve."""
    edit = fitted.fit_state.hand_edit
    records = {} if edit is None else {record.term: record for record in edit.repairs}
    coefficients = np.asarray(fitted.coefficients, dtype=np.float64)
    found: list[PostfitStatus] = []
    for term in postfit_terms(fitted):
        record = records.get(term.term)
        obeyed = postfit_term_feasible(term, coefficients)
        if record is None:
            state = PostfitState.OBEYED if obeyed else PostfitState.PENDING
        else:
            state = PostfitState.REPAIRED if obeyed else PostfitState.BROKEN
        found.append(PostfitStatus(term, state, record))
    return tuple(found)


def _axis(term: PostfitTerm, frame: EagerFrame) -> tuple[NDArray, NDArray[np.bool_]]:
    """The axis the term's shape lives on, and the rows that sit on it (SuperGLM's reading)."""
    from superglm.model.shape_ops import _repair_axis_column

    return _repair_axis_column(SimpleNamespace(spec=term.spec, name=term.label), frame)


def _row_log_likelihoods(evaluation: Any) -> NDArray[np.float64]:
    """Each row's log-likelihood, ``-inf`` where the family marks it invalid or it is not finite."""
    values = np.array(evaluation.optimizing_log_likelihood, dtype=np.float64, copy=True)
    if evaluation.valid is not None:
        values[~np.asarray(evaluation.valid, dtype=np.bool_)] = -np.inf
    values[~np.isfinite(values)] = -np.inf
    return values


def _profiled_merit(
    fitted: Any,
    rows: Any,
    frame: EagerFrame,
    coefficients: NDArray[np.float64],
    parameter: str,
    reach: float,
    dropped: Hashable | None = None,
    *,
    eta_base: NDArray[np.float64] | None = None,
) -> tuple[float, float, int | None]:
    """The intercept shift that maximizes the joint log-likelihood, and the penalized merit there.

    Every term and other parameter is held at ``coefficients``; the merit is the fit's own
    negative log-likelihood on the retained rows plus half its penalty quadratic. A shift no
    better than rounding, or for a parameter with no intercept, is not taken. The merit is
    infinite where a retained row has no valid parameters; the third value counts those rows
    (``None`` where the parameters or the likelihood cannot be evaluated at all). ``dropped``
    names a term of ``parameter`` whose coefficients are zero, so that its basis is not scored.

    Raises ``RuntimeError`` when the likelihood still improves where the shift leaves
    float64's range, so that no intercept could be re-estimated.
    """
    family = fitted.family
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, rows.observation)
    state = fitted.layout.predictor(parameter)
    link = state.link
    column = [predictor.name for predictor in fitted.layout.predictors].index(parameter)
    eta = (
        fitted._predict_eta_resolved(
            coefficients,
            frame,
            rows.offsets,
            dropped=None if dropped is None else (parameter, dropped),
        )
        if eta_base is None
        else eta_base
    )
    penalty = 0.5 * float(coefficients @ np.asarray(fitted.result.penalty) @ coefficients)
    unit = _intercept_scale(eta[:, column], reach)
    # A probe moves only this parameter's column of eta, so every other parameter is formed
    # once, from the same column by the same operations as ``parameters_from_eta``; one with
    # no finite valid value refuses every probe, as it would refuse each.
    fixed = np.empty_like(eta)
    try:
        for index in range(eta.shape[1]):
            if index != column:
                fixed[:, index] = fitted.parameter_from_eta(eta, index)
        steady = bool(np.all(np.isfinite(np.delete(fixed, column, axis=1))))
    except (ValueError, FloatingPointError, OverflowError):
        steady = False
    # Newton's method reads the inverse link's second derivative, which a link may lack.
    newton = state.intercept_index is not None and callable(getattr(link, "deriv2_inverse", None))
    diagonal = packed_pairs(eta.shape[1]).index((column, column))

    def slope_at(evaluation: Any, moved: NDArray[np.float64]) -> tuple[float, float] | None:
        """The log-likelihood's slope and curvature in the shift, where both are finite: the
        family's natural derivatives through the inverse link, as
        ``transform_natural_derivatives`` forms them, on this parameter's column."""
        with np.errstate(over="ignore", invalid="ignore"):
            first = np.asarray(link.deriv_inverse(moved), dtype=np.float64)
            second = np.asarray(link.deriv2_inverse(moved), dtype=np.float64)
            score = np.asarray(evaluation.score, dtype=np.float64)[:, column]
            hessian = np.asarray(evaluation.hessian_packed, dtype=np.float64)[:, diagonal]
            slope = float(np.sum(score * first, dtype=np.float64))
            curvature = float(np.sum(hessian * first * first + score * second, dtype=np.float64))
        return (slope, curvature) if np.isfinite(slope) and np.isfinite(curvature) else None

    def rows_at(
        shift: float, derivatives: bool
    ) -> tuple[NDArray[np.float64] | None, tuple[float, float] | None]:
        """Each row's log-likelihood, ``-inf`` where its parameters are not valid, and where
        ``derivatives`` asks and every row is valid, the slope and curvature in ``shift``."""
        if not steady:
            return None, None
        shifted = np.array(eta, dtype=np.float64, copy=True)
        shifted[:, column] += shift
        # A probe is a trial of the fit's likelihood, refused as the solver refuses one:
        # parameters or a likelihood with no float64 value make the probe invalid.
        try:
            theta = fixed.copy()
            theta[:, column] = fitted.parameter_from_eta(shifted, column)
            if not np.all(np.isfinite(theta[:, column])):
                return None, None
            theta.setflags(write=False)
            if derivatives:
                try:
                    full = family.evaluate_natural(rows.response, theta, plan, derivative_order=2)
                except (ValueError, FloatingPointError, OverflowError):
                    full = None
                if full is not None and full.hessian_packed is not None:
                    values = _row_log_likelihoods(full)
                    if np.all(np.isfinite(values)):
                        return values, slope_at(full, shifted[:, column])
            # The value-only evaluation says which rows are valid: it refuses no row whose
            # derivatives are valid, and may keep one whose derivatives are not.
            evaluation = family.evaluate_natural(rows.response, theta, plan, derivative_order=0)
        except (ValueError, FloatingPointError, OverflowError):
            return None, None
        return _row_log_likelihoods(evaluation), None

    values: dict[float, float] = {}
    invalid: dict[float, NDArray[np.bool_] | None] = {}
    # Each shift's slope and curvature in the search's units, and the sum of its rows' absolute
    # log-likelihoods (the objective's resolution), where they were asked for.
    slopes: dict[float, tuple[float, float, float] | None] = {}

    def probe(t: float, derivatives: bool) -> float:
        # Memoised: the bracket and the walls' bisection revisit their end points. A shift read
        # for its value alone is read again only where its derivatives are asked for.
        if t not in values or (derivatives and t not in slopes):
            found, local = rows_at(unit * t, derivatives)
            failed = None if found is None else ~np.isfinite(found)
            invalid[t] = failed
            whole = found is not None and failed is not None and not failed.any()
            values[t] = -float(np.sum(found, dtype=np.float64)) if whole else np.inf
            if derivatives:
                slopes[t] = None
                if found is not None and local is not None:
                    slope, curvature = -unit * local[0], -unit * (unit * local[1])
                    if np.isfinite(slope) and np.isfinite(curvature):
                        size = float(np.sum(np.abs(found), dtype=np.float64))
                        slopes[t] = (slope, curvature, size)
        return values[t]

    def objective(t: float) -> float:
        return probe(t, False)

    def invalid_at(t: float) -> NDArray[np.bool_] | None:
        objective(t)
        return invalid[t]

    def derivatives_at(t: float) -> tuple[float, tuple[float, float, float] | None]:
        return probe(t, True), slopes[t]

    held = probe(0.0, newton)
    zero_failed = invalid[0.0]
    zero_invalid = None if zero_failed is None else int(np.count_nonzero(zero_failed))
    if state.intercept_index is None:
        return 0.0, held + penalty, zero_invalid
    start = 0.0 if np.isfinite(held) else _finite_start(objective, invalid_at)
    if start is None:
        return 0.0, held + penalty, zero_invalid
    best = _newton_minimum(derivatives_at, start, unit, len(rows.response)) if newton else None
    if best is None:
        best = _derivative_free_minimum(objective, start, unit)
    if best is None:
        from superglm.constraints import _InterceptSearchError

        raise _InterceptSearchError(
            f"Shape repair rejected before publication: the {parameter} intercept could "
            "not be re-estimated, since the likelihood still improves where the shift "
            "leaves float64's range on its link scale. Check the model's specification, "
            "or fit the rule instead (Constraint.fit)."
        )
    best_value = objective(best)
    # Where the held intercept leaves rows invalid, any finite optimum is the re-estimate.
    # Otherwise two sums of one size: the rounding either can carry (Higham, section 3.1).
    tolerance = 4.0 * len(rows.response) * _UNIT_ROUNDOFF * abs(held)
    if np.isfinite(held) and not best_value < held - tolerance:
        return 0.0, held + penalty, 0
    return unit * best, best_value + penalty, 0


def _newton_minimum(
    probe: Callable[[float], tuple[float, tuple[float, float, float] | None]],
    start: float,
    unit: float,
    rows: int,
) -> float | None:
    """A minimum of the objective from ``start`` by Newton's method on its slope, kept inside a
    bracket of the minimum (Press et al., section 9.4, ``rtsafe``; Hager, *Comput. Math. Appl.*
    18(9), 1989); ``None`` where it cannot say, and the derivative-free search decides.

    ``probe(t)`` is the objective at ``t``, infinite where a row is invalid, and where every row
    is valid its slope, curvature and the sum of its ``rows`` terms' magnitudes (else ``None``).
    ``None`` comes back where a point it moves to has no slope, where its slope is zero and its
    curvature is not positive, where the open bracket would leave float64's range in ``unit``s,
    or after ``_NEWTON_STEPS``. Every other stop is a minimum: a positive curvature with a
    decrement within rounding, or a bracket closed on the best point.
    """
    x = start
    value, derivatives = probe(x)
    # The minimum lies above a point of negative slope, below one of positive slope, and on the
    # best point's side of a point no lower than it or invalid (a wall).
    lower, upper = -np.inf, np.inf
    before = last = np.inf  # the last two step lengths, as rtsafe keeps them
    for _ in range(_NEWTON_STEPS):
        if derivatives is None:
            return None
        slope, curvature, size = derivatives
        # A zero slope is a minimum only where the curvature is positive. Elsewhere it is a
        # maximum or an inflection (a square-root link's, where the mean is zero on every row).
        if slope == 0.0:
            return x if curvature > 0.0 else None
        if slope < 0.0:
            lower = x
        else:
            upper = x
        far = upper if slope < 0.0 else lower
        tolerance = _INTERCEPT_XATOL * (1.0 + abs(x - start))
        step = -slope / curvature if curvature > 0.0 else np.nan
        # Stop where the decrease Newton's model predicts (the Newton decrement: Boyd and
        # Vandenberghe, Convex Optimization, 2004, section 9.5) is below the objective's last
        # place, or where the step is within the tolerance and that decrease within the sum's
        # rounding: a short step alone says nothing beside a wall, where the curvature can be
        # vast. Or where the bracket has closed on a wall.
        decrease = -0.5 * slope * step
        rounding = 4.0 * rows * _UNIT_ROUNDOFF * size
        if decrease <= _UNIT_ROUNDOFF * size or (abs(step) <= tolerance and decrease <= rounding):
            return x
        if abs(far - x) <= tolerance:
            return x
        trial = x + step
        # No step (NaN), one that overflows, or one too short to move x is not taken. One that
        # leaves the bracket, or is not half the step two before it, gives way to bisection, or
        # while the bracket is open downhill to a step growing as ``_bracket``'s walk does.
        newton = bool(np.isfinite(unit * trial)) and trial != x
        if not (newton and lower < trial < upper and abs(step) <= 0.5 * before):
            if np.isfinite(far):
                trial = 0.5 * (x + far)
            else:
                grown = max(_INTERCEPT_SEARCH, _INTERCEPT_GROWTH * last if last < np.inf else 0.0)
                trial = x + float(np.copysign(max(grown, abs(step)) if newton else grown, -slope))
                if not np.isfinite(unit * trial):
                    return None
        before, last = last, abs(trial - x)
        trial_value, trial_derivatives = probe(trial)
        if trial_value < value:
            x, value, derivatives = trial, trial_value, trial_derivatives
        elif trial > x:
            upper = trial
        else:
            lower = trial
    return None


def _derivative_free_minimum(
    objective: Callable[[float], float], start: float, unit: float
) -> float | None:
    """A minimum of ``objective`` from ``start`` by bracketing and Brent alone, or ``None``
    when the objective still falls where ``unit`` times the shift overflows."""
    lower = _finite_reach(objective, start, start - _INTERCEPT_SEARCH)
    upper = _finite_reach(objective, start, start + _INTERCEPT_SEARCH)
    best = _brent(objective, lower, upper, centre=start)
    # Brent's result is bracketed when it is lower than each end that is not a wall of
    # invalid rows. Otherwise the optimum may lie beyond that end: walk out from it.
    edges = ((upper, start + _INTERCEPT_SEARCH), (lower, start - _INTERCEPT_SEARCH))
    open_end = next(
        (end for end, edge in edges if end == edge and objective(end) <= objective(best)), None
    )
    if open_end is None:
        return best
    bracket = _bracket(objective, start, open_end, unit)
    if bracket is None:
        return None
    best = _brent(objective, *bracket, centre=start)
    # Brent's tolerance grows with the distance from ``centre``, so the optimum is
    # polished in coordinates centred on it. SciPy's bounded Brent stops with the
    # minimum within 4 * (sqrt(eps) |d| + xatol / 3) of its result.
    polish = 4.0 * (np.sqrt(np.finfo(np.float64).eps) * abs(best - start) + _INTERCEPT_XATOL)
    lower = _finite_reach(objective, best, best - polish)
    upper = _finite_reach(objective, best, best + polish)
    return _brent(objective, lower, upper, centre=best)


def _finite_start(
    objective: Callable[[float], float],
    invalid_at: Callable[[float], NDArray[np.bool_] | None],
) -> float | None:
    """The nearest shift to zero at which ``objective`` is finite, probed by doublings both
    ways out to ``_INTERCEPT_WIDEST``; else a finite shift between two probes
    (``_between_probes``), or ``None`` when neither finds one.

    Nearest first: the probe found lies within a factor of 2 of the wall of invalid rows.
    """
    probes = [0.0]
    for doublings in range(_INTERCEPT_PROBES):
        reach = _INTERCEPT_WIDEST * 0.5 ** (_INTERCEPT_PROBES - 1 - doublings)
        for shift in (reach, -reach):
            if np.isfinite(objective(shift)):
                return shift
            probes.append(shift)
    return _between_probes(objective, invalid_at, sorted(probes))


def _within(rows: NDArray[np.bool_] | None, others: NDArray[np.bool_] | None) -> bool:
    """Whether the invalid ``rows`` are among ``others``; an unknown set (``None``) may be."""
    return rows is None or others is None or not np.any(rows & ~others)


def _between_probes(
    objective: Callable[[float], float],
    invalid_at: Callable[[float], NDArray[np.bool_] | None],
    probes: list[float],
) -> float | None:
    """A finite shift between two consecutive ``probes``, each invalid, or ``None``.

    Each row is taken to be valid on an interval of shifts, as ``_finite_reach`` takes it: a
    domain wall or an overflow at each end. All rows are valid on the intersection ``I``. Left
    of ``I`` a shift's invalid rows are those whose interval starts above it, so they shrink
    toward ``I``; right of it they grow. Gaps where that can hold are bisected, nearest zero
    first: a midpoint lies left of ``I`` only if its invalid rows are among its left end's,
    and right only if among its right end's. Where both may hold, both halves are searched,
    breadth first. At most ``_BETWEEN_PROBES`` evaluations are spent (see there).
    """
    rows = [invalid_at(probe) for probe in probes]
    falls = [True]  # each probe's invalid rows shrink from the leftmost probe up to it
    for j in range(1, len(probes)):
        falls.append(falls[-1] and _within(rows[j], rows[j - 1]))
    rises = [True]  # and grow from it to the rightmost probe
    for j in range(len(probes) - 2, -1, -1):
        rises.append(rises[-1] and _within(rows[j], rows[j + 1]))
    rises.reverse()
    gaps = [j for j in range(len(probes) - 1) if falls[j] and rises[j + 1]]
    gaps.sort(key=lambda j: min(abs(probes[j]), abs(probes[j + 1])))
    queue = deque((probes[j], probes[j + 1], rows[j], rows[j + 1]) for j in gaps)
    for _ in range(_BETWEEN_PROBES):
        if not queue:
            return None
        left, right, left_rows, right_rows = queue.popleft()
        middle = 0.5 * (left + right)
        if middle in (left, right):  # adjacent floats: no shift lies between
            continue
        if np.isfinite(objective(middle)):
            return middle
        middle_rows = invalid_at(middle)
        if _within(middle_rows, left_rows):
            queue.append((middle, right, middle_rows, right_rows))
        if _within(middle_rows, right_rows):
            queue.append((left, middle, left_rows, middle_rows))
    return None


def _bracket(
    objective: Callable[[float], float], behind: float, near: float, unit: float
) -> tuple[float, float] | None:
    """An interval of finite values holding a minimum of ``objective``, walked downhill from
    ``near``, away from ``behind``, or ``None`` when the objective still falls where ``unit``
    times the shift overflows.

    Each step grows by ``_INTERCEPT_GROWTH`` (Press et al., section 10.1, ``mnbrak``) until the
    objective rises, or a wall of invalid rows is met, where the finite optimum is taken.
    """
    while True:
        far = near + _INTERCEPT_GROWTH * (near - behind)
        if not np.isfinite(unit * far):
            return None
        reached = _finite_reach(objective, near, far)
        if reached != far or objective(reached) >= objective(near):
            # The objective rose, or the valid interval ends: the minimum lies in between.
            return min(behind, reached), max(behind, reached)
        behind, near = near, reached


def _brent(objective: Callable[[float], float], lower: float, upper: float, *, centre: float):
    """Brent's bounded minimum of ``objective`` on ``[lower, upper]``, run in coordinates
    centred on ``centre``, or whichever of ``centre`` and the end points is lower."""
    candidates = [centre, lower, upper]
    if upper > lower:
        # A row could still turn invalid inside, where Brent's parabolic step forms
        # inf - inf before it falls back to a golden-section step.
        with np.errstate(invalid="ignore"):
            search = minimize_scalar(
                lambda d: objective(centre + d),
                bounds=(lower - centre, upper - centre),
                method="bounded",
                options={"xatol": _INTERCEPT_XATOL},
            )
        candidates.insert(1, centre + float(search.x))
    return min(candidates, key=objective)


def _finite_reach(objective: Callable[[float], float], start: float, edge: float) -> float:
    """How far from ``start`` toward ``edge`` the objective stays finite: ``edge`` itself,
    or a finite point within ``_INTERCEPT_XATOL`` of the first row to turn invalid.

    The rows' valid shifts form an interval around ``start``, so bisection finds its end.
    """
    if np.isfinite(objective(edge)):
        return edge
    inside, outside = start, edge
    while abs(outside - inside) > _INTERCEPT_XATOL:
        middle = 0.5 * (inside + outside)
        if middle in (inside, outside):  # adjacent floats, far out: the wall is found
            break
        if np.isfinite(objective(middle)):
            inside = middle
        else:
            outside = middle
    return inside


def _intercept_scale(eta: NDArray[np.float64], reach: float) -> float:
    """The parameter's own scale on its link: how far its linear predictor ranges over the
    rows, or the repair's largest change, whichever is wider.

    Both are differences on the link, so a log link's scale does not depend on the
    response's units, and an identity link's scales with them. A parameter that is the
    same on every row, with no change, falls back to the size of its value, then to 1.
    """
    finite = eta[np.isfinite(eta)]
    spread = float(np.ptp(finite)) if finite.size else 0.0
    for scale in (max(spread, reach), float(np.max(np.abs(finite), initial=0.0))):
        if np.isfinite(scale) and scale > 0.0:
            return scale
    return 1.0


def _zero_to_rounding(coefficients: NDArray, basis: Callable[[], NDArray], apex: float) -> bool:
    """Whether a repair's result is the zero curve to ``apex``, the rounding it is certified
    at: its coefficients' norm, or its curve's largest value on the repair's grid (``basis()``,
    the fitted basis there, formed only where the norm is not), within it.

    A projection meets the cone's apex only to rounding, there in either measure; the zero
    curve is the exact point it approximates. Read as anything else, its decision against the
    dropped term would be that rounding's: beside the apex a move inside the cone costs merit
    at first order, by far more than the merit's own rounding where the merit is steep.
    """
    if float(np.linalg.norm(coefficients)) <= apex:
        return True
    values = np.asarray(basis(), dtype=np.float64) @ np.asarray(coefficients, dtype=np.float64)
    return values.size > 0 and float(np.max(np.abs(values))) <= apex


def _settled(
    term: PostfitTerm,
    repair: Any,
    fitted_coefficients: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
    certified_scale: float | None = None,
) -> NDArray[np.float64]:
    """The repaired coefficients, certified as ``postfit_status`` reads them, or a refusal.

    The repair certifies its result at ``certified_scale`` (a continued repair's), by default
    the scale of the curve it repaired; the rule's state is read at the repaired curve's own.
    A result zero to the rounding of the fitted term, or of a continued repair's scale, in its
    coefficients or along its curve (``_zero_to_rounding``), is the zero curve, and is
    published as one. A result that breaks its rule at the second scale by
    no more than the rounding certified at the first is moved strictly inside the cone by that
    rounding, and published. Any other result that breaks it is refused.
    """
    from superglm.constraints import (
        _inside_ray,
        shape_constraint_certificate,
        shape_constraint_is_roundoff_feasible,
        shape_slack_tolerance,
    )

    spec, kind = term.curve_spec, term.kind
    repaired = np.array(repair.repaired_beta_reparam[term.columns], dtype=np.float64)
    zero_scale = float(np.linalg.norm(fitted_coefficients))
    if certified_scale is not None:
        zero_scale = max(zero_scale, certified_scale)
    apex = shape_slack_tolerance(repaired, zero_scale)
    certificate = repair.certificate
    if _zero_to_rounding(repaired, lambda: spec.transform(repair.grid), apex):
        repaired, certificate = np.zeros_like(repaired), None
    # ``postfit_term_feasible``'s own check, on the term's block: the repair's certificate of
    # these coefficients where it formed one, formed here only for coefficients moved here.
    if certificate is None:
        certificate = shape_constraint_certificate(spec, repaired, kind)
    slack = certificate.minimum_scaled_slack
    obeyed = slack >= -shape_slack_tolerance(repaired)
    if not obeyed:
        violation = -slack
        grid = np.asarray(repair.grid, dtype=np.float64)
        share = np.asarray(grid_weights, dtype=np.float64) / float(np.sum(grid_weights))
        if certified_scale is None:
            # The scale the projection certified at: the weighted RMS of the curve it repaired.
            target = np.asarray(repair.original_log_effect, dtype=np.float64)
            largest = float(np.max(np.abs(target), initial=0.0))
            certified_scale = (
                largest * float(np.sqrt(share @ (target / largest) ** 2)) if largest else 0.0
            )
        if violation <= shape_slack_tolerance(repaired, certified_scale):
            # Along the repairer's own inside ray, by twice the violation: the least slack,
            # a unit row times the coefficients minimised over the range, is superadditive
            # in them, so every slack is then at least the violation.
            ray = _inside_ray(np.asarray(spec.transform(grid)), grid, share, kind)
            ray_slack = shape_constraint_certificate(spec, ray, kind).minimum_scaled_slack
            if ray_slack > 0.0:
                repaired = repaired + (2.0 * violation / ray_slack) * ray
                obeyed = shape_constraint_is_roundoff_feasible(spec, repaired, kind)
    if not obeyed:
        raise RuntimeError(
            f"Shape repair of {term.term} rejected before publication: the projection could "
            "not be certified to obey the rule at its own scale. Fit the rule instead "
            f"(Constraint.fit.{term.kind}), or drop the term."
        )
    return repaired


def _profiled_curvature(
    fitted: Any,
    term: PostfitTerm,
    curvature: NDArray[np.float64],
    *,
    coefficients: NDArray[np.float64] | None = None,
    source: str | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """``curvature`` on the term's coefficients with its parameter's intercept profiled, and
    the intercept's coupling over its own curvature (``None`` where nothing is profiled), which
    profiles a gradient in the same way: ``g_t - ratio * g_a``."""
    # Minimising a semidefinite quadratic over the intercept leaves the generalized Schur
    # complement H_tt - h h' H_aa^+ (Boyd and Vandenberghe, Convex Optimization, 2004,
    # appendix A.5.5; Gallier, The Schur complement and symmetric positive semidefinite
    # matrices, 2019, theorem 4.3). An intercept curvature at the block's rounding is zero
    # there, as a pseudo-inverse takes it, and semidefiniteness then makes h zero too.
    columns = np.arange(term.columns.start, term.columns.stop)
    block = np.array(curvature[np.ix_(columns, columns)], dtype=np.float64)
    ratio = None
    intercept = fitted.layout.predictor(term.parameter).intercept_index
    if intercept is not None:
        coupling = np.asarray(curvature[columns, intercept], dtype=np.float64)
        own = float(curvature[intercept, intercept])
        scale = max(float(np.max(np.abs(np.diag(block)), initial=0.0)), abs(own))
        if own > (columns.size + 1) * _UNIT_ROUNDOFF * scale:
            ratio = coupling / own
            if _profile_cancels(block, coupling, own):
                # Raw second moments can agree in nearly every digit when one
                # observation carries the information. Center before the product,
                # as for stable variances (Chan, Golub and LeVeque 1983).
                try:
                    block = _centered_term_curvature(fitted, term, ratio, coefficients, source)
                    if not np.all(np.isfinite(block)):
                        raise FloatingPointError("nonfinite centered repair metric")
                except (
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                    MemoryError,
                    np.linalg.LinAlgError,
                ):
                    block = np.array(curvature[np.ix_(columns, columns)], dtype=np.float64)
                    block -= np.outer(coupling, coupling) / own
                    warnings.warn(
                        "The repair's centered curvature could not be evaluated; intercept "
                        "profiling loses precision in the stored curvature, so the closest "
                        "curve is not certified. The available repair is published.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
            else:
                block -= np.outer(coupling, coupling) / own
    return 0.5 * (block + block.T), ratio


def _profile_cancels(block, coupling, own) -> bool:
    """Whether the rank-one intercept subtraction loses half the float64 digits.

    With ``v = h / sqrt(H_aa)``, ``H_tt - vv'`` has a small generalized
    eigenvalue exactly when ``v' H_tt^-1 v`` approaches one. Read in Jacobi
    coordinates so coefficient units do not choose this exceptional row pass.
    This chooses a stable formation, not a rank or a convergence decision.
    """
    from scipy.linalg import solve_triangular

    if not np.any(coupling):
        return False
    diagonal = np.diag(block)
    if np.any(diagonal <= 0.0):
        return True
    scale = np.sqrt(diagonal)
    try:
        root = np.linalg.cholesky(block / scale[:, None] / scale[None, :])
        projected = solve_triangular(
            root, coupling / np.sqrt(own) / scale, lower=True, check_finite=False
        )
    except np.linalg.LinAlgError:
        return True
    return 1.0 - float(projected @ projected) <= np.sqrt(_UNIT_ROUNDOFF)


def _centered_term_curvature(fitted, term, ratio, coefficients, source):
    """``B_c' D B_c + S`` from the fit's actual curvature source, in bounded chunks.

    Only the cancellation path replays likelihood derivatives. No row matrix or
    new cache survives this call. Each chunk uses the fitted design, including
    its identifiability transformations, before subtracting the intercept mean.
    """
    from superglm.distributional.solver import chunks

    rows = fitted.fit_state.retained_rows
    beta = fitted.coefficients if coefficients is None else coefficients
    actual = fitted.result.terminal_curvature.actual_source if source is None else source
    plan = fitted.family.bind_likelihood(rows.response, rows.likelihood_weights, rows.observation)
    state = fitted.layout.predictor(term.parameter)
    start = (
        term.columns.start - state.coefficient_slice.start - int(state.intercept_index is not None)
    )
    stop = start + term.columns.stop - term.columns.start
    selected, offset = [], 0
    for group in state.design.group_matrices:
        end = offset + group.shape[1]
        if start < end and stop > offset:
            selected.append(
                (group, slice(max(start - offset, 0), min(stop - offset, group.shape[1])))
            )
        offset = end
    diagonal = packed_pairs(len(fitted.layout.predictors)).index(
        (state.parameter_index, state.parameter_index)
    )
    block = np.array(fitted.result.penalty[term.columns, term.columns], copy=True)
    for chunk in chunks.iter_likelihood_chunks(
        fitted.family,
        fitted.layout,
        rows.response,
        plan,
        beta,
        chunk_size=fitted.result.resolved_chunk_size or 2048,
        curvature_source=actual,
    ):
        basis = np.hstack(
            [
                group.row_subset(chunk.rows.indices).toarray()[:, columns]
                for group, columns in selected
            ]
        )
        basis -= ratio
        block += basis.T @ (chunk.curvature_packed[:, diagonal, None] * basis)
    return block


def _term_precision(
    fitted: Any,
    term: PostfitTerm,
    curvature: NDArray[np.float64] | None = None,
    *,
    source: str | None = None,
) -> NDArray[np.float64]:
    """The metric the term's repair projects in: the penalized curvature at the coefficients
    being repaired on the term's coefficients, every other coefficient held and the term's
    parameter's intercept profiled; ``curvature`` is the fit's terminal one by default.

    It is the second-order model, there, of the merit the publication rule judges
    (``_profiled_merit``), as SuperGLM's ``X'WX + S_lambda`` at a revision's working weights is
    (``model.shape_ops``): the curvature the fit accepts, ``-H(l) + S``, never a covariance
    inverted back.
    """
    coefficients = None
    if curvature is None:
        curvature = fitted.result.terminal_penalized_curvature
        coefficients = getattr(fitted.result, "coefficients", None)
    return _profiled_curvature(fitted, term, curvature, coefficients=coefficients, source=source)[0]


def _curvature_at(
    fitted: Any, rows: Any
) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64], str] | None]:
    """The penalized score and curvature at given coefficients, as the fit forms its terminal
    ones (``solver.solver``), on the fit's backend and smoothing parameters: the observed
    curvature unless it is materially indefinite, then the expected (Fisher), then the nearest
    semidefinite (``resolve_curvature``). The fit's retry, a further Newton run, would move the
    coefficients, so it is not taken. ``None`` where they cannot be evaluated or no curvature
    is accepted. The context, and its design matrices, are formed on the first call.
    """
    from superglm.distributional.solver import solver as _solver
    from superglm.distributional.solver.curvature import resolve_curvature

    context: list[Any] = []

    def at(
        coefficients: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], str] | None:
        if not context:
            plan = fitted.family.bind_likelihood(
                rows.response, rows.likelihood_weights, rows.observation
            )
            context.append(
                _solver._validated_context(
                    fitted.family,
                    fitted.layout,
                    rows.response,
                    plan,
                    fitted.result.penalty,
                    coefficient_curvature="observed",
                    chunk_size=fitted.result.resolved_chunk_size,
                    coefficient_face=fitted.result.coefficient_face,
                )
            )
        (solver_context,) = context
        try:
            state = _solver._evaluate_state(solver_context, coefficients)
            if state is None:
                return None
            observed = _solver._measured_geometry(solver_context, state, "observed", None)
            decision = resolve_curvature(
                "observed",
                _solver._policy_curvature(solver_context, observed.penalized_curvature),
                matrix_kind="penalized",
            )
            accepted = observed
            if decision.retry_required:
                found: dict[str, Any] = {}

                def measured(source: Any) -> Any:
                    found[source] = _solver._measured_geometry(solver_context, state, source, None)
                    return _solver._policy_curvature(
                        solver_context, found[source].penalized_curvature
                    )

                fisher = None if solver_context.fisher_family is None else measured("fisher")
                decision = resolve_curvature(
                    "observed",
                    _solver._policy_curvature(solver_context, observed.penalized_curvature),
                    fisher_matrix=fisher,
                    semidefinite_fallback=lambda: measured("nearest_semidefinite"),
                    state=decision.state,
                    matrix_kind="penalized",
                )
                accepted = found[decision.telemetry.actual_source]
        # A curvature no policy accepts is a ValueError, as a family's refusal to evaluate is.
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return None
        return (
            np.asarray(observed.score_penalized, dtype=np.float64),
            np.asarray(accepted.penalized_curvature, dtype=np.float64),
            decision.telemetry.actual_source,
        )

    return at


def _candidate(
    fitted: Any,
    rows: Any,
    frame: EagerFrame,
    coefficients: NDArray[np.float64],
    term: PostfitTerm,
    block: NDArray[np.float64],
    reach: float,
) -> tuple[NDArray[np.float64], float, float]:
    """``coefficients`` with the term's block replaced, its intercept shift and penalized
    merit (``_profiled_merit``); refused where a retained row has no valid parameters."""
    candidate = coefficients.copy()
    candidate[term.columns] = block
    shift, merit, invalid = _profiled_merit(fitted, rows, frame, candidate, term.parameter, reach)
    _require_valid(term, rows, merit, invalid)
    return candidate, shift, merit


def _require_valid(
    term: PostfitTerm, rows: Any, merit: float, invalid: int | None, *, dropped: bool = False
) -> None:
    """Refuse to publish the repaired curve, or with ``dropped`` the term dropped, where a
    retained row has no valid parameters with it (``_profiled_merit``'s ``merit``, ``invalid``)."""
    if np.isfinite(merit):
        return
    where = (
        "the model's parameters or likelihood cannot be evaluated on the training rows"
        if invalid is None
        else f"{invalid} of the {len(rows.response)} training rows have no valid "
        "parameters there (the family's likelihood is not finite)"
    )
    raise RuntimeError(
        f"Unsafe shape repair of {term.term} rejected before publication: with "
        f"{'the term dropped' if dropped else 'the repaired curve'}, {where}. Fit the rule "
        f"instead (Constraint.fit.{term.kind}), which keeps every row valid"
        f"{'' if dropped else ', or drop the term'}."
    )


# A repair's projection is the first step of the constrained penalized fit at the fit's
# smoothing parameters, exact only where the merit is quadratic. A projection the zero term
# beats is continued as SuperGLM continues one (``model.shape_ops._continued_repair``), by
# iteratively re-weighted cone projection (Liao & Meyer, J. Stat. Softw. 89(5), 2019, section
# 1.2), the SQP of the merit over a polyhedral cone (Masselot et al., arXiv:2509.18406, 2025,
# supplement B): at each feasible iterate the merit's second-order model in the penalized
# curvature there (``_curvature_at``), intercept profiled, is minimised over the cone by the
# same repairer, and the next iterate lies on the segment back to the current one, which the
# cone holds, at Armijo's sufficient decrease by halving (Nocedal & Wright, Numerical
# Optimization, 2nd ed., 2006, section 3.1), each trial's intercept re-estimated. The zero term
# is feasible, so where the merit is convex the iterates approach a curve no worse than it.
def _continued_repair(
    fitted: Any,
    rows: Any,
    frame: EagerFrame,
    term: PostfitTerm,
    repair: Any,
    start: NDArray[np.float64],
    merit: float,
    curvature_at: Callable[..., Any],
    grid_weights: NDArray[np.float64],
    n_grid: int,
) -> Any:
    """The refused repair continued towards the constrained penalized fit, and the scale it is
    certified at (``model.shape_ops._continued_scale``); or ``None``.

    ``start`` is the refused candidate with its intercept re-estimated, ``merit`` its merit.
    Stops where the model's predicted decrease or a step's actual decrease is within the
    merit's rounding, or a step finds no decrease; ``None`` when no step is taken. The
    publication rule judges the result as it judged the projection, at the scale returned.
    """
    from dataclasses import replace

    from superglm.constraints import shape_constraint_certificate, shape_sum_tolerance
    from superglm.model.shape_ops import (
        _ARMIJO,
        _CONTINUED_STEPS,
        _STEP_HALVINGS,
        _certified_curve_scale,
        _continued_scale,
        _repairer,
    )

    columns, spec = term.columns, term.curve_spec
    intercept = fitted.layout.predictor(term.parameter).intercept_index
    groups = [SimpleNamespace(sl=columns, size=columns.stop - columns.start)]
    reach = float(repair.projection_residual)
    repairer = _repairer(term.kind)
    beta, moved = start, False
    certified = _certified_curve_scale(repair.original_log_effect, grid_weights)
    for _ in range(_CONTINUED_STEPS):
        tolerance = shape_sum_tolerance(abs(merit), len(rows.response))
        local = curvature_at(beta)
        if local is None:
            break
        score, curvature, source = local
        H, ratio = _profiled_curvature(fitted, term, curvature, coefficients=beta, source=source)
        # The merit's gradient is the penalized score's negative, profiled as its curvature is.
        gradient = -score[columns] if ratio is None else ratio * score[intercept] - score[columns]
        current = beta[columns]
        target = beta.copy()
        try:
            target[columns] = current - np.linalg.solve(H, gradient)
            projected = repairer.repair(
                spec, target, groups, weights=grid_weights, n_grid=n_grid, metric=H
            )
        except (np.linalg.LinAlgError, RuntimeError):
            break
        cone_point = np.asarray(projected.repaired_beta_reparam, dtype=np.float64)[columns]
        step = cone_point - current
        slope = float(gradient @ step)
        if not slope < 0.0 or -(slope + 0.5 * float(step @ H @ step)) <= tolerance:
            break
        length = 1.0
        for _ in range(_STEP_HALVINGS):
            trial = beta.copy()
            # On the segment, exact at its ends.
            trial[columns] = (1.0 - length) * current + length * cone_point
            try:
                shift, trial_merit, _ = _profiled_merit(
                    fitted, rows, frame, trial, term.parameter, reach
                )
            except RuntimeError:
                trial_merit = np.inf
            if trial_merit <= merit + _ARMIJO * length * slope:
                break
            length *= 0.5
        else:
            break
        if intercept is not None:
            trial[intercept] += shift
        decrease = merit - trial_merit
        certified = _continued_scale(certified, current, cone_point, projected, grid_weights)
        beta, merit, moved = trial, trial_merit, True
        if decrease <= tolerance:
            break
    if not moved:
        return None
    block = beta[columns]
    certificate = shape_constraint_certificate(spec, block, term.kind)
    full = np.array(repair.repaired_beta_reparam, dtype=np.float64, copy=True)
    full[columns] = block
    curve = np.asarray(spec.transform(repair.grid), dtype=np.float64) @ block
    continued = replace(
        repair,
        repaired_beta_reparam=full,
        repaired_log_effect=curve,
        projection_residual=float(np.max(np.abs(curve - repair.original_log_effect))),
        max_violation_after=certificate.violation,
        certificate=certificate,
    )
    return continued, certified


def repaired_model(
    fitted: Any,
    X_train: FrameLike | EagerFrame | None,
    *,
    n_grid: int = 500,
) -> Any:
    """The fitted model with every post-fit rule its curve breaks repaired, or ``fitted``.

    A term whose curve already obeys its rule is left exactly as it is, and a model with no
    term to repair is returned unchanged, so a second call repairs nothing. As in SuperGLM, a
    projection whose penalized likelihood is worse than dropping the term altogether is
    continued towards the constrained fit; where that is still worse, or no projection or
    intercept can be formed, the term is published flat, the term dropped, with a
    ``RuntimeWarning``. A repair is refused, and nothing published, only where a retained row
    would have no valid parameters, or the dropped term's own intercept cannot be re-estimated.
    """
    from superglm.constraints import (
        _flat_repair,
        _IndefiniteMetricError,
        _InterceptSearchError,
        shape_constraint_certificate,
        shape_slack_tolerance,
        shape_sum_tolerance,
    )
    from superglm.distributional.revision import NO_TRAINING_FRAME, _rows_of_the_fit, revised_model
    from superglm.model.shape_ops import (
        _curve_spec,
        _grid_weights,
        _repairer,
        flat_publication_message,
    )

    terms = postfit_terms(fitted)
    for term in terms:
        if term.curve_spec is None:
            _curve_spec(term.term, term.spec)  # refuses by name, as SuperGLM's repair does
    coefficients = np.array(fitted.coefficients, dtype=np.float64, copy=True)
    # ``postfit_term_feasible``'s check, keeping each certificate for the term's repair.
    certificates = {
        term.term: shape_constraint_certificate(
            term.curve_spec, coefficients[term.columns], term.kind
        )
        for term in terms
    }
    pending = [
        term
        for term in terms
        if certificates[term.term].minimum_scaled_slack
        < -shape_slack_tolerance(coefficients[term.columns])
    ]
    if not pending:
        return fitted
    rows = fitted.fit_state.retained_rows
    if rows is None:
        raise RuntimeError(REPAIR_NEEDS_THE_ROWS)
    if X_train is None:
        raise ValueError(NO_TRAINING_FRAME)
    # The frame given to the fit, or the rows it kept: weights and likelihood read the kept rows.
    frame = _rows_of_the_fit(as_eager_frame(X_train), rows)
    weights = np.asarray(rows.likelihood_weights.values, dtype=np.float64)
    curvature_at = _curvature_at(fitted, rows)
    # The metric follows a revision, as SuperGLM's follows its working weights: the fit's own
    # terminal curvature at the fit's coefficients, else the curvature at the coefficients being
    # repaired, formed as the fit forms it, or the fit's where none is accepted there.
    curvature, curvature_source = None, None
    if not np.array_equal(coefficients, fitted.result.coefficients):
        local = curvature_at(coefficients)
        curvature = None if local is None else local[1]
        curvature_source = None if local is None else local[2]

    records: list[ShapeRepairRecord] = []
    flat: list[str] = []  # one warning per term published flat, once the revision is formed
    for term in pending:
        x, on_axis = _axis(term, frame)
        width = term.columns.stop - term.columns.start
        block = SimpleNamespace(sl=term.columns, size=width)
        grid_weights = _grid_weights(term.curve_spec, x, weights[on_axis], n_grid)
        # The repaired coefficients other than the zero curve, where they are formed.
        candidate, shift, merit = None, 0.0, np.inf
        try:
            repair = _repairer(term.kind).repair(
                term.curve_spec,
                coefficients,
                [block],
                weights=grid_weights,
                n_grid=n_grid,
                metric=(
                    _term_precision(fitted, term)
                    if curvature is None
                    else _term_precision(fitted, term, curvature, source=curvature_source)
                ),
                certificate=certificates[term.term],
            )
            repaired = _settled(term, repair, coefficients[term.columns], grid_weights)
        except _IndefiniteMetricError:
            repair = _flat_repair(
                term.kind,
                term.curve_spec,
                coefficients,
                [block],
                n_grid,
                certificate=certificates[term.term],
            )
            repaired = np.zeros(width)
        if np.array_equal(repaired, coefficients[term.columns]):
            continue
        reach = float(repair.projection_residual)
        if not repair.flat:
            try:
                candidate, shift, merit = _candidate(
                    fitted, rows, frame, coefficients, term, repaired, reach
                )
            except _InterceptSearchError:
                pass  # the dropped term's own profile decides
        fallback = coefficients.copy()
        fallback[term.columns] = 0.0
        if candidate is not None and not np.any(repaired):
            # The candidate is already the dropped term. Its valid profile has
            # just been computed; scoring the zero contribution again changes
            # neither the predictor nor its objective.
            fallback_shift, fallback_merit, fallback_invalid = shift, merit, 0
        else:
            fallback_shift, fallback_merit, fallback_invalid = _profiled_merit(
                fitted, rows, frame, fallback, term.parameter, reach, dropped=term.label
            )
        bar = fallback_merit + shape_sum_tolerance(abs(fallback_merit), len(rows.response))
        state = fitted.layout.predictor(term.parameter)
        if not repair.flat and (candidate is None or merit > bar):
            # A candidate whose intercept has no minimum does not rule out the
            # cone's other curves. Its apex has just been profiled successfully.
            start = fallback.copy() if candidate is None else candidate.copy()
            start_shift = fallback_shift if candidate is None else shift
            start_merit = fallback_merit if candidate is None else merit
            if state.intercept_index is not None:
                start[state.intercept_index] += start_shift
            continued = _continued_repair(
                fitted,
                rows,
                frame,
                term,
                repair,
                start,
                start_merit,
                curvature_at,
                grid_weights,
                n_grid,
            )
            if continued is not None:
                repair, certified = continued
                repaired = _settled(
                    term, repair, coefficients[term.columns], grid_weights, certified
                )
                reach = float(repair.projection_residual)
                # At the zero curve the dropped term is published, below.
                candidate = None
                if np.any(repaired):
                    try:
                        candidate, shift, merit = _candidate(
                            fitted, rows, frame, coefficients, term, repaired, reach
                        )
                    except _InterceptSearchError:
                        pass
        if candidate is None or merit > bar:
            # No curve the rule allows beats the dropped term, so the term is published flat,
            # with the dropped term's intercept profiled above: a profile from another scale
            # could end elsewhere where the likelihood is flat in the intercept (a saturated
            # link's). It is refused where a retained row has no valid parameters there.
            _require_valid(term, rows, fallback_merit, fallback_invalid, dropped=True)
            candidate, shift = fallback.copy(), fallback_shift
        if state.intercept_index is not None:
            candidate[state.intercept_index] += shift
        coefficients = candidate
        published_flat = not np.any(coefficients[term.columns])
        if published_flat:
            flat.append(flat_publication_message(repr(term.term), term.kind))
        records.append(
            ShapeRepairRecord(
                term=term.term,
                kind=term.kind,
                violation_before=float(repair.max_violation_before),
                violation_after=0.0 if published_flat else float(repair.max_violation_after),
                largest_change=(
                    float(np.max(np.abs(repair.original_log_effect), initial=0.0))
                    if published_flat
                    else reach
                ),
                log_scale=_is_shifted_log_link(state.link),
                intercept_shift=shift,
                flat=published_flat,
            )
        )
    if not records:
        return fitted
    repaired_terms = {record.term for record in records}
    parameters = tuple(
        dict.fromkeys(
            term.parameter
            for term in pending
            if term.term in repaired_terms
            and fitted.layout.predictor(term.parameter).intercept_index is not None
        )
    )
    if len(parameters) > 1:
        from dataclasses import replace

        before_settling = coefficients
        coefficients = _settled_intercepts(fitted, rows, frame, coefficients, parameters)
        parameter_of = {term.term: term.parameter for term in pending}
        last_record = {parameter_of[record.term]: i for i, record in enumerate(records)}
        for parameter, i in last_record.items():
            index = fitted.layout.predictor(parameter).intercept_index
            if index is not None:
                # Keep each repair's own step; charge the joint settling step
                # once, to this parameter's last record, so their shifts add up.
                records[i] = replace(
                    records[i],
                    intercept_shift=records[i].intercept_shift
                    + float(coefficients[index] - before_settling[index]),
                )
    revised = revised_model(fitted, coefficients, (), frame, repairs=records)
    for message in flat:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return revised


def _settled_intercepts(fitted, rows, frame, coefficients, parameters):
    """Settle the repaired parameters' coupled intercepts with all term curves held.

    One sweep leaves earlier conditional optima stale after a later parameter
    moves. Cyclic conditional minimization revisits them until none improves
    beyond the likelihood's rounding (Tseng 2001, section 2, cyclic rule).
    Single-parameter repairs keep their existing one-profile path.
    """
    from superglm.constraints import _InterceptSearchError

    beta = coefficients.copy()
    # Curves, rows and offsets are held throughout this numerical iteration.
    # Keep their predictor for this call only, and update precisely the column
    # whose intercept changed, rather than rebuilding every term each profile.
    eta = np.array(fitted._predict_eta_resolved(beta, frame, rows.offsets), copy=True)
    for _ in range(_NEWTON_STEPS):
        changed = False
        for parameter in parameters:
            index = fitted.layout.predictor(parameter).intercept_index
            if index is None:
                continue
            try:
                shift, _, _ = _profiled_merit(
                    fitted, rows, frame, beta, parameter, 0.0, eta_base=eta
                )
            except _InterceptSearchError:
                break
            if shift != 0.0:
                before = beta[index]
                beta[index] += shift
                actual_shift = beta[index] - before
                column = fitted.layout.predictor(parameter).parameter_index
                eta[:, column] += actual_shift
                changed |= actual_shift != 0.0
        else:
            if not changed:
                return beta
            continue
        break
    warnings.warn(
        "The repaired parameters' intercepts did not settle jointly; the last valid "
        "repair is published with its conditional intercept estimates.",
        RuntimeWarning,
        stacklevel=3,
    )
    return beta


__all__ = [
    "POSTFIT_MOVES",
    "REPAIR_NEEDS_THE_ROWS",
    "PostfitState",
    "PostfitStatus",
    "PostfitTerm",
    "postfit_status",
    "postfit_term_feasible",
    "postfit_terms",
    "repaired_model",
]
