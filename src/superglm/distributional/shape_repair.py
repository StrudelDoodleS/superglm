"""``Constraint.postfit`` rules on SuperLSS terms, repaired after the fit as SuperGLM repairs them.

A post-fit rule asks for the fitted curve to be repaired, not for the fit to be constrained.
``SuperGLM.apply_shape_postfit`` projects each such term's curve onto its shape cone in the
term's own fitted basis, weighting the grid by where the training rows sit along the axis
(``MonotoneRepairer`` and ``CurvatureRepairer`` in ``superglm.constraints``), re-estimates the
intercept with the terms held, and withholds the repaired term's standard errors, bands and test:
the projection is a constrained estimator whose reference distribution is not the unconstrained
fit's (Meyer and Woodroofe, *Ann. Statist.* 28(4), 2000; Meyer, *Ann. Appl. Statist.* 2(3),
2008), while the point estimate needs no such caveat (Chernozhukov, Fernandez-Val and Galichon,
*Biometrika* 96(3), 2009, Proposition 1).

SuperLSS runs the same repairer on the same fitted basis, for a term of any parameter, on that
parameter's link scale. The basis carries the fit's centring, so a repair keeps the term's
training mean on that scale; the parameter's intercept is then re-estimated on the joint
likelihood of the retained rows with every term and every other parameter held. The result is
published as a revision of the fit (``revision.py``): predictions, curves and rating tables
follow it, while ``result_`` and the covariance keep describing the fit.
"""

from __future__ import annotations

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
from superglm.features.constraint import rules_of
from superglm.links import _is_shifted_log_link

REPAIR_NEEDS_THE_ROWS = (
    "A post-fit shape repair weights its projection by the training rows and re-estimates the "
    "intercept on them; refit with retain_rows=True before calling apply_shape_postfit()."
)
# The search for the re-estimated intercept runs in units of the parameter's own scale on its
# link (``_intercept_scale``), so that the response's units do not change it: bracket, then
# Brent (Brent, *Algorithms for Minimization without Derivatives*, 1973, ch. 5; Press et al.,
# *Numerical Recipes*, 3rd ed., section 10.1). Brent first runs within ``_INTERCEPT_SEARCH``
# of those units of the held intercept. Where its result is no lower than an end, the bracket
# walks downhill from that end, each step growing by ``_INTERCEPT_GROWTH``, until the
# objective rises, a row turns invalid, or the shift leaves float64's range. No count limits
# it, so an optimum any number of units away is bracketed in steps logarithmic in its distance.
_INTERCEPT_SEARCH = 4.0
# Each downhill step is the golden ratio times the last, ``mnbrak``'s default growth.
_INTERCEPT_GROWTH = (1.0 + np.sqrt(5.0)) / 2.0
_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0
# A smooth minimum is located to about sqrt(u) of its scale, since the objective is flat to
# rounding within that distance of it (Press et al., section 10.1); Brent stops there.
_INTERCEPT_XATOL = float(np.sqrt(_UNIT_ROUNDOFF))
# Where the held intercept leaves rows invalid, the search starts from the nearest valid shift
# among doublings both ways, from 2**-60 of ``_INTERCEPT_WIDEST`` units up to it.
_INTERCEPT_WIDEST = 2.0**20
_INTERCEPT_PROBES = 61


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


def _profiled_merit(
    fitted: Any,
    rows: Any,
    frame: EagerFrame,
    coefficients: NDArray[np.float64],
    parameter: str,
    reach: float,
) -> tuple[float, float, int | None]:
    """The intercept shift that maximizes the joint log-likelihood, and the penalized merit there.

    Every term and other parameter is held at ``coefficients``; the merit is the fit's own
    negative log-likelihood on the retained rows plus half its penalty quadratic. A shift no
    better than rounding, or for a parameter with no intercept, is not taken. The merit is
    infinite where a retained row has no valid parameters; the third value counts those rows
    (``None`` where the parameters or the likelihood cannot be evaluated at all).

    Raises ``RuntimeError`` when the likelihood still improves where the shift leaves
    float64's range, so that no intercept could be re-estimated.
    """
    family = fitted.family
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, rows.observation)
    state = fitted.layout.predictor(parameter)
    column = [predictor.name for predictor in fitted.layout.predictors].index(parameter)
    eta = fitted._predict_eta_resolved(coefficients, frame, rows.offsets)
    penalty = 0.5 * float(coefficients @ np.asarray(fitted.result.penalty) @ coefficients)

    def rows_at(shift: float) -> NDArray[np.float64] | None:
        """Each row's log-likelihood, ``-inf`` where its parameters are not valid."""
        shifted = np.array(eta, dtype=np.float64, copy=True)
        shifted[:, column] += shift
        # A probe is a value-only trial of the fit's likelihood, refused as the solver refuses
        # one: parameters or a likelihood with no float64 value make the probe invalid.
        try:
            theta = fitted.parameters_from_eta(shifted)
            evaluation = family.evaluate_natural(rows.response, theta, plan, derivative_order=0)
        except (ValueError, FloatingPointError, OverflowError):
            return None
        values = np.array(evaluation.optimizing_log_likelihood, dtype=np.float64, copy=True)
        if evaluation.valid is not None:
            values[~np.asarray(evaluation.valid, dtype=np.bool_)] = -np.inf
        values[~np.isfinite(values)] = -np.inf
        return values

    def negative_log_likelihood(shift: float) -> float:
        values = rows_at(shift)
        if values is None or not np.all(np.isfinite(values)):
            return np.inf
        return -float(np.sum(values, dtype=np.float64))

    def invalid_rows(shift: float) -> int | None:
        values = rows_at(shift)
        return None if values is None else int(np.count_nonzero(~np.isfinite(values)))

    held = negative_log_likelihood(0.0)
    if state.intercept_index is None:
        return 0.0, held + penalty, invalid_rows(0.0)
    unit = _intercept_scale(eta[:, column], reach)

    values: dict[float, float] = {}

    def objective(t: float) -> float:
        # Memoised: the bracket and the walls' bisection revisit their end points.
        if t not in values:
            values[t] = negative_log_likelihood(unit * t)
        return values[t]

    start = 0.0 if np.isfinite(held) else _finite_start(objective)
    if start is None:
        return 0.0, held + penalty, invalid_rows(0.0)
    lower = _finite_reach(objective, start, start - _INTERCEPT_SEARCH)
    upper = _finite_reach(objective, start, start + _INTERCEPT_SEARCH)
    best = _brent(objective, lower, upper, centre=start)
    # Brent's result is bracketed when it is lower than each end that is not a wall of
    # invalid rows. Otherwise the optimum may lie beyond that end: walk out from it.
    edges = ((upper, start + _INTERCEPT_SEARCH), (lower, start - _INTERCEPT_SEARCH))
    open_end = next(
        (end for end, edge in edges if end == edge and objective(end) <= objective(best)), None
    )
    if open_end is not None:
        bracket = _bracket(objective, start, open_end, unit)
        if bracket is None:
            raise RuntimeError(
                f"Shape repair rejected before publication: the {parameter} intercept could "
                "not be re-estimated, since the likelihood still improves where the shift "
                "leaves float64's range on its link scale. Check the model's specification, "
                "or fit the rule instead (Constraint.fit)."
            )
        best = _brent(objective, *bracket, centre=start)
        # Brent's tolerance grows with the distance from ``centre``, so the optimum is
        # polished in coordinates centred on it. SciPy's bounded Brent stops with the
        # minimum within 4 * (sqrt(eps) |d| + xatol / 3) of its result.
        polish = 4.0 * (np.sqrt(np.finfo(np.float64).eps) * abs(best - start) + _INTERCEPT_XATOL)
        lower = _finite_reach(objective, best, best - polish)
        upper = _finite_reach(objective, best, best + polish)
        best = _brent(objective, lower, upper, centre=best)
    best_value = objective(best)
    # Where the held intercept leaves rows invalid, any finite optimum is the re-estimate.
    # Otherwise two sums of one size: the rounding either can carry (Higham, section 3.1).
    tolerance = 4.0 * len(rows.response) * _UNIT_ROUNDOFF * abs(held)
    if np.isfinite(held) and not best_value < held - tolerance:
        return 0.0, held + penalty, 0
    return unit * best, best_value + penalty, 0


def _finite_start(objective: Callable[[float], float]) -> float | None:
    """The nearest shift to zero at which ``objective`` is finite, probed by doublings both
    ways out to ``_INTERCEPT_WIDEST``, or ``None`` when every probe leaves a row invalid.

    Nearest first: the probe found lies within a factor of 2 of the wall of invalid rows.
    """
    for doublings in range(_INTERCEPT_PROBES):
        reach = _INTERCEPT_WIDEST * 0.5 ** (_INTERCEPT_PROBES - 1 - doublings)
        for shift in (reach, -reach):
            if np.isfinite(objective(shift)):
                return shift
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


def _settled(
    term: PostfitTerm, repair: Any, fitted_coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The repaired coefficients, certified as ``postfit_status`` reads them, or a refusal.

    The projection certifies its result at the scale of the curve it repaired, while the
    state of the rule is read at the scale of the repaired curve alone. A projection onto the
    cone's apex can come back as coefficients that are zero only to the first scale's rounding,
    and would read as broken; it is the zero curve, which obeys every rule, so it is published
    as one. Any other result that does not obey its rule at the second scale is refused.
    """
    from superglm.constraints import shape_constraint_is_roundoff_feasible, shape_slack_tolerance

    repaired = np.array(repair.repaired_beta_reparam[term.columns], dtype=np.float64)
    apex = shape_slack_tolerance(repaired, float(np.linalg.norm(fitted_coefficients)))
    if float(np.linalg.norm(repaired)) <= apex:
        repaired = np.zeros_like(repaired)
    # ``postfit_term_feasible``'s own check, on the term's block.
    if not shape_constraint_is_roundoff_feasible(term.curve_spec, repaired, term.kind):
        raise RuntimeError(
            f"Shape repair of {term.term} rejected before publication: the projection could "
            "not be certified to obey the rule at its own scale. Fit the rule instead "
            f"(Constraint.fit.{term.kind}), or drop the term."
        )
    return repaired


def repaired_model(
    fitted: Any,
    X_train: FrameLike | EagerFrame | None,
    *,
    n_grid: int = 500,
) -> Any:
    """The fitted model with every post-fit rule its curve breaks repaired, or ``fitted``.

    A term whose curve already obeys its rule is left exactly as it is, and a model with no
    term to repair is returned unchanged, so a second call repairs nothing. As in SuperGLM, a
    repair whose penalized likelihood is worse than dropping the term altogether is refused
    before anything is published.
    """
    from superglm.constraints import shape_sum_tolerance
    from superglm.distributional.revision import NO_TRAINING_FRAME, _rows_of_the_fit, revised_model
    from superglm.model.shape_ops import _curve_spec, _grid_weights, _repairer

    terms = postfit_terms(fitted)
    for term in terms:
        if term.curve_spec is None:
            _curve_spec(term.term, term.spec)  # refuses by name, as SuperGLM's repair does
    coefficients = np.array(fitted.coefficients, dtype=np.float64, copy=True)
    pending = [term for term in terms if not postfit_term_feasible(term, coefficients)]
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

    records: list[ShapeRepairRecord] = []
    for term in pending:
        x, on_axis = _axis(term, frame)
        width = term.columns.stop - term.columns.start
        block = SimpleNamespace(sl=term.columns, size=width)
        repair = _repairer(term.kind).repair(
            term.curve_spec,
            coefficients,
            [block],
            weights=_grid_weights(term.curve_spec, x, weights[on_axis], n_grid),
            n_grid=n_grid,
        )
        repaired = _settled(term, repair, coefficients[term.columns])
        if np.array_equal(repaired, coefficients[term.columns]):
            continue
        candidate = coefficients.copy()
        candidate[term.columns] = repaired
        fallback = coefficients.copy()
        fallback[term.columns] = 0.0
        reach = float(repair.projection_residual)
        shift, merit, invalid = _profiled_merit(
            fitted, rows, frame, candidate, term.parameter, reach
        )
        if not np.isfinite(merit):
            where = (
                "the model's parameters or likelihood cannot be evaluated on the training rows"
                if invalid is None
                else f"{invalid} of the {len(rows.response)} training rows have no valid "
                "parameters there (the family's likelihood is not finite)"
            )
            raise RuntimeError(
                f"Unsafe shape repair of {term.term} rejected before publication: with the "
                f"repaired curve, {where}. Fit the rule instead (Constraint.fit.{term.kind}), "
                "which keeps every row valid, or drop the term."
            )
        _, fallback_merit, _ = _profiled_merit(fitted, rows, frame, fallback, term.parameter, reach)
        if merit > fallback_merit + shape_sum_tolerance(abs(fallback_merit), len(rows.response)):
            raise RuntimeError(
                f"Unsafe shape repair of {term.term} rejected before publication: its penalized "
                "objective, the negative log-likelihood plus the fit's smoothing penalty at the "
                f"fit's smoothing parameters, is {merit:.6g} against {fallback_merit:.6g} with "
                "the term dropped. Fit the rule instead "
                f"(Constraint.fit.{term.kind}), which chooses the smoothing under the rule, or "
                "drop the term."
            )
        state = fitted.layout.predictor(term.parameter)
        if state.intercept_index is not None:
            candidate[state.intercept_index] += shift
        coefficients = candidate
        records.append(
            ShapeRepairRecord(
                term=term.term,
                kind=term.kind,
                violation_before=float(repair.max_violation_before),
                violation_after=float(repair.max_violation_after),
                largest_change=reach,
                log_scale=_is_shifted_log_link(state.link),
                intercept_shift=shift,
            )
        )
    if not records:
        return fitted
    return revised_model(fitted, coefficients, (), frame, repairs=records)


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
