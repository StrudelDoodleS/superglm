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

from collections.abc import Hashable
from dataclasses import dataclass
from types import SimpleNamespace
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
# The search for the re-estimated intercept stays within this many link-scale units of the
# held one, or within four times the largest change the repairs made, whichever is wider.
_INTERCEPT_SEARCH = 1.0


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
) -> tuple[float, float]:
    """The intercept shift that maximizes the joint log-likelihood, and the penalized merit there.

    Every term and every other parameter is held at ``coefficients``; the likelihood is the
    fit's own, bound to the retained rows, and the merit is its negative plus half the fit's
    penalty quadratic, the objective the fit minimized. A shift that does not improve on the
    held intercept by more than rounding is not taken, nor is one for a parameter with no
    intercept.
    """
    family = fitted.family
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, rows.observation)
    state = fitted.layout.predictor(parameter)
    column = [predictor.name for predictor in fitted.layout.predictors].index(parameter)
    eta = fitted._predict_eta_resolved(coefficients, frame, rows.offsets)
    penalty = 0.5 * float(coefficients @ np.asarray(fitted.result.penalty) @ coefficients)

    def negative_log_likelihood(shift: float) -> float:
        shifted = np.array(eta, dtype=np.float64, copy=True)
        shifted[:, column] += shift
        try:
            theta = fitted.parameters_from_eta(shifted)
        except ValueError:
            return np.inf
        evaluation = family.evaluate_natural(rows.response, theta, plan, derivative_order=0)
        if evaluation.valid is not None and not np.all(evaluation.valid):
            return np.inf
        value = -float(np.sum(evaluation.optimizing_log_likelihood, dtype=np.float64))
        return value if np.isfinite(value) else np.inf

    held = negative_log_likelihood(0.0)
    if state.intercept_index is None:
        return 0.0, held + penalty
    bound = max(_INTERCEPT_SEARCH, 4.0 * reach)
    search = minimize_scalar(
        negative_log_likelihood,
        bounds=(-bound, bound),
        method="bounded",
        options={"xatol": 1e-10},
    )
    # Two sums of one size: the rounding either can carry (Higham, section 3.1).
    tolerance = 2.0 * len(rows.response) * np.finfo(np.float64).eps * abs(held)
    if not np.isfinite(search.fun) or not search.fun < held - tolerance:
        return 0.0, held + penalty
    return float(search.x), float(search.fun) + penalty


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
    from superglm.distributional.revision import NO_TRAINING_FRAME, revised_model
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
    frame = as_eager_frame(X_train)
    if len(frame) != len(rows.response):
        raise ValueError(
            f"X_train has {len(frame)} rows, but the fit kept {len(rows.response)}. "
            "Pass the frame the model was fitted on."
        )
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
        repaired = np.asarray(repair.repaired_beta_reparam[term.columns], dtype=np.float64)
        if np.array_equal(repaired, coefficients[term.columns]):
            continue
        candidate = coefficients.copy()
        candidate[term.columns] = repaired
        fallback = coefficients.copy()
        fallback[term.columns] = 0.0
        reach = float(repair.projection_residual)
        shift, merit = _profiled_merit(fitted, rows, frame, candidate, term.parameter, reach)
        _, fallback_merit = _profiled_merit(fitted, rows, frame, fallback, term.parameter, reach)
        if merit > fallback_merit + shape_sum_tolerance(abs(fallback_merit), len(rows.response)):
            raise RuntimeError(
                f"Unsafe shape repair of {term.term} rejected before publication: the repaired "
                "curve fits worse than dropping the term. Fit the rule instead "
                f"(Constraint.fit.{term.kind}), or drop the term."
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
    "REPAIR_NEEDS_THE_ROWS",
    "PostfitTerm",
    "postfit_term_feasible",
    "postfit_terms",
    "repaired_model",
]
