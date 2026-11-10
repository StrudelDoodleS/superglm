"""Shape-constrained coefficients for SuperLSS: the rows, a sequential QP, and the face it ends on.

A ``Constraint.fit`` rule reaches a compiled design as linear rows ``A b >= c`` on its term's
solver coefficients (``GroupSlice.constraints``, the SuperGLM spline builders' rows). SuperLSS
chooses its smoothing without the rows, then refits the coefficients at that smoothing under
them, as ``SuperGLM`` documents for its own constrained terms. The refit is sequential quadratic
programming from a feasible start (Wood 2026, "Shape constrained additive smooth models",
Statistical Modelling; Nocedal and Wright 2006, ch. 18): each step solves the convex QP
``min  d' H d / 2 - s' d   s.t.  A (b + d) >= c`` over the solver's own penalized Newton model
(``s`` the penalized score, ``H`` the positive-definite penalized curvature), written for the
new point ``b + d`` as the SuperGLM QP route writes it so every row is read at the scale of its
own action, with the repository's active-set QP (``solvers/constrained_qp.py``), then backtracks along ``d`` on the
penalized log-likelihood. Every iterate is feasible, since the rows are linear and both ends of
a step are, so that likelihood is the merit function.

The rows binding at the end hold the fit on an affine face (``ShapeFace``). The solver's own
face fit then certifies it, and the covariance and degrees of freedom of the fit are those of the
fit on that face, ``Z (Z' H Z)^+ Z'``, as the SuperGLM QP route reads its own.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import gamma
from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import CompiledPredictor
from superglm.distributional.results.shape import ShapeFace, ShapeFit, ShapeTermRecord
from superglm.distributional.results.solver import DenseSolverConfig, DenseSolverResult
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver import chunks as chunking
from superglm.distributional.solver import solver as _solver
from superglm.distributional.timing import FitPhaseRecorder
from superglm.features.constraint import rules_of
from superglm.links import LogLink
from superglm.solvers.constrained_qp import (
    QPResult,
    _constraint_structure,
    _ConstraintStructure,
    _feasibility_slack,
    _is_feasible,
)
from superglm.solvers.irls_direct._finalise import _binding_rank
from superglm.solvers.irls_direct._qp import (
    _QP_FEASIBILITY_TOL,
    _binding_rows,
    _solve_constrained_qp_with_cold_retry,
)

# Projecting onto the rows is one QP when the curvature is a good model of the likelihood;
# a few more allow for a step the predictor cap shortened.
_MAX_PROJECTIONS = 8


@dataclass(frozen=True)
class ShapeRows:
    """Every term's fit-time rows as ``A b >= c`` on the stacked coefficients.

    ``terms`` lists, per constrained term, its qualified name, the rules as written, its rows
    ``rows`` in ``A`` and its coefficients ``columns`` in the stack.
    """

    A: NDArray[np.float64]
    b: NDArray[np.float64]
    terms: tuple[tuple[str, str, slice, slice], ...]


def shape_rows(layout: StackedLayout, compiled: Sequence[CompiledPredictor]) -> ShapeRows | None:
    """The stacked rows of every compiled group that carries fit-time rules, or ``None``."""
    blocks: list[NDArray] = []
    bounds: list[NDArray] = []
    terms: list[tuple[str, str, slice, slice]] = []
    width = layout.n_coefficients
    row = 0
    for state, built in zip(layout.predictors, compiled, strict=True):
        slope_start = state.coefficient_slice.start + int(state.intercept_index is not None)
        for group in state.groups:
            if group.constraints is None:
                continue
            local = np.asarray(group.constraints.A, dtype=np.float64)
            block = np.zeros((local.shape[0], width), dtype=np.float64)
            columns = slice(slope_start + group.start, slope_start + group.end)
            block[:, columns] = local
            blocks.append(block)
            bounds.append(np.asarray(group.constraints.b, dtype=np.float64))
            rules = rules_of(built.compiled.specs.get(group.feature_name))
            text = rules.describe() if rules else "shape rule"
            terms.append(
                (f"{state.name}:{group.name}", text, slice(row, row + local.shape[0]), columns)
            )
            row += local.shape[0]
    if not blocks:
        return None
    return ShapeRows(np.vstack(blocks), np.concatenate(bounds), tuple(terms))


def _qp_step(
    curvature: NDArray,
    target: NDArray,
    anchor: NDArray,
    A: NDArray,
    rhs: NDArray,
    active: list[int] | None,
    structure: _ConstraintStructure,
    config: DenseSolverConfig,
) -> QPResult | None:
    """The QP's new point ``x`` for ``curvature @ x = target`` under the rows, damping as
    ``_solve_direction`` does if the curvature is indefinite.

    ``target`` is ``curvature @ anchor + score``, so the unconstrained point is the Newton step
    from ``anchor``. A damped curvature ``curvature + shift * D`` takes ``shift * D @ anchor`` into
    its target too: the step from ``anchor`` is then the damped Newton step, not one pulled toward
    zero.
    """
    diagonal = np.maximum(np.abs(np.diag(curvature)), np.finfo(np.float64).tiny)
    shifts = [0.0]
    shifts.extend(
        config.initial_levenberg_shift * config.levenberg_growth**attempt
        for attempt in range(config.max_levenberg_attempts)
    )
    for shift in shifts:
        system = curvature if shift == 0.0 else curvature + shift * np.diag(diagonal)
        shifted = target if shift == 0.0 else target + shift * diagonal * anchor
        try:
            result = _solve_constrained_qp_with_cold_retry(
                system, shifted, A, rhs, active, structure=structure
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(result.beta)):
            return result
    return None


def _face_of(
    held: NDArray, anchor: NDArray, base: PenaltyFace | None, names: tuple[str, ...]
) -> ShapeFace | None:
    """The affine face through ``anchor`` that the binding rows ``held`` hold, inside ``base``."""
    width = held.shape[1]
    inside = np.eye(width) if base is None else np.asarray(base.null_basis, dtype=np.float64)
    reduced = held @ inside
    norms = np.linalg.norm(reduced, axis=1)
    kept = norms > np.finfo(np.float64).eps * max(1.0, float(np.max(norms, initial=0.0)))
    if not np.any(kept):
        return None
    touched, vt, rank = _binding_rank(reduced[kept])
    if rank == 0:
        return None
    row_space = np.zeros((inside.shape[1], rank))
    row_space[touched] = vt[:rank].T
    free = np.zeros((inside.shape[1], inside.shape[1] - rank))
    untouched = np.setdiff1d(np.arange(inside.shape[1]), touched)
    free[untouched, np.arange(untouched.size)] = 1.0
    free[np.ix_(touched, np.arange(untouched.size, free.shape[1]))] = vt[rank:].T
    constraint_basis = inside @ row_space
    constraint = constraint_basis @ constraint_basis.T
    if base is not None:
        constraint = constraint + np.asarray(base.constraint_matrix, dtype=np.float64)
        constraint_basis = np.column_stack((base.constraint_basis, constraint_basis))
    return ShapeFace(
        component_names=() if base is None else base.component_names,
        coefficient_names=names,
        constraint_matrix=0.5 * (constraint + constraint.T),
        null_basis=inside @ free,
        constraint_basis=constraint_basis,
        constraint_rank=constraint_basis.shape[1],
        anchor=anchor,
        base=base,
    )


def _term_changes(
    layout: StackedLayout,
    rows: ShapeRows,
    weights: NDArray,
    before: NDArray,
    after: NDArray,
    binding: NDArray,
) -> tuple[ShapeTermRecord, ...]:
    """Per term: rows, rows binding, and the largest change of its centred effect on the data."""
    records: list[ShapeTermRecord] = []
    total = float(np.sum(weights))
    for name, text, row_slice, columns in rows.terms:
        state = layout.predictor(name.partition(":")[0])
        slope_start = state.coefficient_slice.start + int(state.intercept_index is not None)
        change = np.zeros(state.design.p)
        change[columns.start - slope_start : columns.stop - slope_start] = (
            after[columns] - before[columns]
        )
        # a fit the rules left alone moved nothing: no row pass for its record
        effect = (
            np.asarray(state.design.matvec(change), dtype=np.float64)
            if np.any(change)
            else np.zeros(state.design.n)
        )
        centred = effect - float(weights @ effect) / total
        records.append(
            ShapeTermRecord(
                term=name,
                rules=text,
                n_rows=row_slice.stop - row_slice.start,
                n_binding=int(np.count_nonzero(binding[row_slice])),
                largest_change=float(np.max(np.abs(centred), initial=0.0)),
                log_scale=isinstance(state.link, LogLink),
            )
        )
    return tuple(records)


def refit_with_shape_rules(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    penalty: NDArray,
    unconstrained: DenseSolverResult,
    rows: ShapeRows,
    *,
    config: DenseSolverConfig | None = None,
    chunk_size: chunking.ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
) -> tuple[DenseSolverResult, ShapeFit]:
    """Refit ``unconstrained``'s coefficients at its smoothing under ``rows``.

    Returns ``unconstrained`` itself, with a record that nothing was applied, when it already
    meets every row. Otherwise returns the fit on the face of the rows that bind, certified by
    the solver's face fit, so its covariance and degrees of freedom treat those rows as fixed.
    A refit that does not converge, or finds no feasible point, is returned and disclosed
    (``converged`` False with the reason), never refused.
    """
    started = time.perf_counter()
    solver_config = DenseSolverConfig() if config is None else config
    A, b = rows.A, rows.b
    structure = _constraint_structure(A)
    base = unconstrained.coefficient_face
    beta0 = np.asarray(unconstrained.coefficients, dtype=np.float64)
    weights = np.asarray(likelihood_plan.weights.values, dtype=np.float64)

    def record(
        result: DenseSolverResult,
        *,
        applied: bool,
        converged: bool,
        reason: str,
        iterations: int = 0,
        backtracks: int = 0,
        binding: NDArray | None = None,
    ) -> tuple[DenseSolverResult, ShapeFit]:
        beta = np.asarray(result.coefficients, dtype=np.float64)
        flags = np.zeros(A.shape[0], dtype=bool) if binding is None else binding
        slack = _feasibility_slack(A, beta, b, abs_A=structure.abs_entries)
        unruled = applied and result is unconstrained
        # A refit that stopped short is a fit that did not converge, whatever the solver said
        # of the fit it started from.
        if applied and not converged:
            solver_reason = "line_search_failed" if reason == "no_feasible_start" else reason
            result = replace(result, converged=False, convergence_reason=solver_reason)
        return result, ShapeFit(
            unruled=unruled,
            terms=_term_changes(layout, rows, weights, beta0, beta, flags),
            applied=applied,
            converged=converged,
            reason=reason,
            iterations=iterations,
            backtracks=backtracks,
            n_rows=A.shape[0],
            n_binding=int(np.count_nonzero(flags)),
            max_violation=float(max(0.0, -np.min(slack, initial=0.0))),
            unconstrained_coefficients=beta0,
            unconstrained_penalized_log_likelihood=float(unconstrained.penalized_log_likelihood),
            penalized_log_likelihood=float(result.penalized_log_likelihood),
            seconds=time.perf_counter() - started,
        )

    if _is_feasible(A, beta0, b, _QP_FEASIBILITY_TOL, abs_A=structure.abs_entries):
        return record(unconstrained, applied=False, converged=True, reason="not_violated")

    session = _solver._DenseObservedReuseSession()
    memoised = (
        session.dense_matrices(layout, phase_recorder=phase_recorder)
        if chunk_size is None
        else None
    )
    context = _solver._validated_context(
        family,
        layout,
        y,
        likelihood_plan,
        penalty,
        coefficient_curvature=solver_config.coefficient_curvature,
        chunk_size=chunk_size,
        coefficient_face=base,
        dense_matrices=memoised,
        _reuse_session=session,
    )
    inside = None if base is None else np.asarray(base.null_basis, dtype=np.float64)
    A_inside = A if inside is None else A @ inside

    def reduce(vector: NDArray) -> NDArray:
        return vector if inside is None else inside.T @ vector

    def lift(vector: NDArray) -> NDArray:
        return vector if inside is None else inside @ vector

    def measure(state: _solver._AcceptedState):
        return _solver._measured_geometry(
            context, state, solver_config.coefficient_curvature, phase_recorder
        )

    def reduced_curvature(geometry) -> NDArray:
        matrix = np.asarray(geometry.penalized_curvature, dtype=np.float64)
        return matrix if inside is None else inside.T @ matrix @ inside

    state = _solver._evaluate_state(context, beta0, phase_recorder=phase_recorder)
    if state is None:
        return record(unconstrained, applied=True, converged=False, reason="no_ascent_direction")
    geometry = measure(state)
    beta = beta0
    active: list[int] | None = None

    inner = structure if inside is None else _constraint_structure(A_inside)

    def propose(geometry, beta: NDArray, *, with_score: bool) -> tuple[QPResult, NDArray] | None:
        """The QP's new point for the model at ``beta`` and its step, or None when the QP finds
        none; the model's pure projection without the score."""
        matrix = reduced_curvature(geometry)
        here = reduce(beta)
        score = reduce(np.asarray(geometry.score_penalized, dtype=np.float64))
        target = matrix @ here + (score if with_score else 0.0)
        found = _qp_step(matrix, target, here, A_inside, b, active, inner, solver_config)
        return None if found is None else (found, lift(found.beta) - beta)

    # A feasible start: the unconstrained fit projected onto the rows in the curvature's metric,
    # which is the first QP, since the score vanishes there.
    for _ in range(_MAX_PROJECTIONS):
        if _is_feasible(A, beta, b, _QP_FEASIBILITY_TOL, abs_A=structure.abs_entries):
            break
        proposal = propose(geometry, beta, with_score=False)
        if proposal is None:
            break
        step, direction = proposal
        # The projection lands on the rows whatever the response's units; only a move the link
        # cannot evaluate is taken in the solver's capped steps instead.
        candidate = beta + direction
        moved = _solver._evaluate_state(context, candidate, phase_recorder=phase_recorder)
        if moved is None:
            capped, _ = _solver._cap_predictor_step(
                context, state, direction, solver_config.max_predictor_step
            )
            candidate = beta + capped
            moved = _solver._evaluate_state(context, candidate, phase_recorder=phase_recorder)
        if moved is None:
            break
        beta, state = candidate, moved
        geometry = measure(state)
        active = step.active_set
    if not _is_feasible(A, beta, b, _QP_FEASIBILITY_TOL, abs_A=structure.abs_entries):
        return record(unconstrained, applied=True, converged=False, reason="no_feasible_start")

    iterations = backtracks = 0
    converged, reason = False, "max_iterations"
    last_active: list[int] = [] if active is None else list(active)
    tolerance = solver_config.tolerance
    for iterations in range(1, solver_config.max_iterations + 1):
        proposal = propose(geometry, beta, with_score=True)
        if proposal is None:
            converged, reason = False, "no_ascent_direction"
            break
        step, direction = proposal
        active = step.active_set
        last_active = list(active)
        step_relative = float(
            np.linalg.norm(direction, ord=np.inf) / (1.0 + np.linalg.norm(beta, ord=np.inf))
        )
        if step.converged and step_relative <= tolerance:
            converged, reason = True, "kkt"
            break
        capped, _ = _solver._cap_predictor_step(
            context, state, direction, solver_config.max_predictor_step
        )
        slope = float(np.asarray(geometry.score_penalized, dtype=np.float64) @ capped)
        accepted = None
        alpha = 1.0
        for attempt in range(solver_config.max_backtracks + 1):
            candidate = beta + alpha * capped
            trial = _solver._evaluate_state(
                context,
                candidate,
                phase_recorder=phase_recorder,
                derivative_order=2 if attempt == 0 else 0,
            )
            required = (
                state.penalized_optimizing_log_likelihood
                + solver_config.armijo_constant * alpha * slope
            )
            if trial is not None and trial.penalized_optimizing_log_likelihood >= required:
                if attempt:
                    trial = _solver._evaluate_state(
                        context, candidate, phase_recorder=phase_recorder
                    )
                accepted = (candidate, trial) if trial is not None else None
                break
            alpha *= solver_config.backtrack_factor
            backtracks += 1
        if accepted is None:
            converged, reason = False, "line_search_failed"
            break
        beta, state = accepted
        geometry = measure(state)

    # The fit is refined on the face of the rows that bind, where the standard errors hold them.
    # The other rows must still hold there: a refinement that crosses one steps back to where it
    # meets that row, which is then held too, and refines again (a primal active-set step).
    scale = np.sqrt(np.maximum(np.diag(np.asarray(geometry.data_curvature)), 0.0))
    bar = _QP_FEASIBILITY_TOL + gamma(A.shape[1] + 2)
    held_rows = list(last_active)
    anchor = beta
    for _ in range(A.shape[0] + 1):
        rows_held = np.asarray(held_rows, dtype=np.intp)
        held = (
            _binding_rows(A[rows_held], b[rows_held], anchor, scale)
            if rows_held.size
            else np.empty((0, A.shape[1]))
        )
        face = _face_of(held, anchor, base, layout.coefficient_names) if held.shape[0] else None
        polished = _solver.fit_dense_fixed_lambda(
            family,
            layout,
            y,
            likelihood_plan,
            penalty,
            initial=anchor,
            config=solver_config,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
            coefficient_face=face if face is not None else base,
            _reuse_session=session,
        )
        moved = np.asarray(polished.coefficients, dtype=np.float64)
        crossed = np.flatnonzero(
            _feasibility_slack(A, moved, b, abs_A=structure.abs_entries) < -bar
        )
        if not crossed.size:
            break
        before, after = A[crossed] @ anchor - b[crossed], A[crossed] @ moved - b[crossed]
        reach = np.clip(before / (before - after), 0.0, 1.0)
        anchor = anchor + float(np.min(reach)) * (moved - anchor)
        held_rows.append(int(crossed[np.argmin(reach)]))
    else:
        converged, reason = False, "line_search_failed"
    # A row binds by the reading that built the face, so the rows Diagnose names are the rows
    # the standard errors hold.
    final = np.asarray(polished.coefficients, dtype=np.float64)
    binding = np.array(
        [_binding_rows(A[[i]], b[[i]], final, scale).shape[0] == 1 for i in range(A.shape[0])],
        dtype=bool,
    )
    if not converged:
        polished = replace(polished, converged=False, convergence_reason=reason)
    elif not polished.converged:
        converged, reason = False, "line_search_failed"
    return record(
        polished,
        applied=True,
        converged=converged,
        reason=reason,
        iterations=iterations,
        backtracks=backtracks,
        binding=binding,
    )
