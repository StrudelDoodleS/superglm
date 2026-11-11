"""Smoothing chosen with a model's shape rules active: Fellner--Schall on the rules' face.

A rule fit is locally the fit with its binding rows held as equalities, ``b = b* + Z a`` for
``Z`` the face's orthonormal null basis (``results.shape.ShapeFace``). On that face the model
is an ordinary penalized model in ``a``: penalty ``Z' S Z``, penalized curvature ``Z' H Z``,
and the covariance the fit already reports, ``Z (Z' H Z)^+ Z'``. Its Laplace marginal
likelihood ``l(b) - b' S b / 2 + log|Z' S Z|_+ / 2 - log|Z' H Z| / 2`` is maximised over the
smoothing by the generalized Fellner--Schall update (Wood and Fasiolo 2017, eq. 5), read on the
face:

``lambda_j <- (lambda_j tr((Z' S Z)^+ Z' S_j Z) - lambda_j tr(Z (Z' H Z)^+ Z' S_j)) / b' S_j b``,

alternated with the sequential-QP refit at each new smoothing, which may change the face.
Wood (2026, "Shape constrained additive smooth models", Statistical Modelling) chooses the
smoothing of a shape-constrained model by an adaptation of this update; holding the binding
rows as equalities is the reading under which the update is the ordinary one, the conditional
reading the covariance and degrees of freedom of the fit already make. Its assumptions are the
update's own on the face: a positive semidefinite data curvature there, which keeps the
numerator positive (Wood and Fasiolo 2017, Theorem 1, with ``B = Z' A Z`` and the null space of
``Z' S_lambda Z`` independent of ``lambda``), and a Hessian whose change with the smoothing is
neglected, as every Fellner--Schall step neglects it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.shape import ShapeFit, ShapeSmoothing
from superglm.distributional.results.solver import DenseSolverConfig, DenseSolverResult
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import _component_states
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver import chunks as chunking
from superglm.distributional.solver.shape import ShapeRows, refit_with_shape_rules
from superglm.distributional.solver.solver import _DenseObservedReuseSession
from superglm.distributional.timing import FitPhaseRecorder
from superglm.reml.efs_update import EFSComponentState, EFSUpdateResult, wood_fasiolo_update


def face_component_states(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
    base: PenaltyFace | None,
) -> tuple[EFSComponentState, ...]:
    """Each finite penalty component with its effective rank on ``face``.

    A block the shape rows do not cut keeps the rank the smoothing search gives it (on the
    exact penalty face ``base`` when there is one). A block they cut has
    ``r_j = lambda_j tr((U' S U)^+ U' S_j U)`` for ``U`` an orthonormal basis of the face within
    the block: the face is block-diagonal, since every row and every exact penalty face is
    local to one term, so ``Z_b Z_b'`` is a projector and ``U`` its unit singular vectors.
    """
    states = (
        _component_states(layout, lambdas)
        if base is None
        else projected_component_states(layout=layout, lambdas=lambdas, face=base)
    )
    if face is None or face is base:
        return states
    basis = np.asarray(face.null_basis, dtype=np.float64)
    reference = None if base is None else np.asarray(base.null_basis, dtype=np.float64)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, state in enumerate(states):
        block = state.coefficient_slice
        groups.setdefault((block.start, block.stop), []).append(index)
    ranks = [state.rank for state in states]
    for (start, stop), members in groups.items():
        local = basis[start:stop]
        whole = (
            float(stop - start) if reference is None else float(np.sum(reference[start:stop] ** 2))
        )
        if float(np.sum(local**2)) > whole - 0.5:
            continue  # no shape row cuts this block
        left, singular, _ = np.linalg.svd(local, full_matrices=False)
        within = left[:, singular > 0.5]
        penalties = [
            within.T @ np.asarray(states[i].penalty, dtype=np.float64) @ within for i in members
        ]
        total = np.zeros((within.shape[1], within.shape[1]))
        for i, penalty in zip(members, penalties, strict=True):
            total += states[i].lambda_value * penalty
        values, vectors = np.linalg.eigh(0.5 * (total + total.T))
        # numerical rank by the LAPACK / NumPy convention, width times eps times the largest
        kept = values > max(within.shape[1], 1) * np.finfo(np.float64).eps * max(
            float(np.max(values, initial=0.0)), 0.0
        )
        inverse = (vectors[:, kept] / values[kept]) @ vectors[:, kept].T
        for i, penalty in zip(members, penalties):
            ranks[i] = float(states[i].lambda_value * np.sum(inverse * penalty))
    return tuple(
        state if rank == state.rank else replace(state, rank=rank)
        for state, rank in zip(states, ranks)
    )


def _update(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    fit: DenseSolverResult,
    base: PenaltyFace | None,
    config: DistributionalEFSConfig,
) -> EFSUpdateResult:
    """The Fellner--Schall update at ``fit``, read on its face."""
    return wood_fasiolo_update(
        face_component_states(layout, lambdas, fit.coefficient_face, base),
        fit.coefficients,
        fit.terminal_pseudo_inverse(),
        max_log_step=config.max_log_step,
        minimum_lambda=config.minimum_lambda,
        maximum_lambda=config.maximum_lambda,
    )


def _ruled_components(
    layout: StackedLayout, rows: ShapeRows, base: PenaltyFace | None
) -> tuple[str, ...]:
    """The estimated finite penalties on a term that carries rules."""
    exact = () if base is None else base.component_names
    names = []
    for component in layout.penalties:
        policy = component.lambda_policy
        if component.name in exact or (policy is not None and policy.mode != "estimate"):
            continue
        block = component.group_sl
        if any(
            block.start < columns.stop and columns.start < block.stop for *_, columns in rows.terms
        ):
            names.append(component.name)
    return tuple(names)


def choose_smoothing_with_shape_rules(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    unconstrained: DenseSolverResult,
    fit: DenseSolverResult,
    shape_fit: ShapeFit,
    rows: ShapeRows,
    efs_config: DistributionalEFSConfig,
    config: DenseSolverConfig | None = None,
    chunk_size: chunking.ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
) -> tuple[DenseSolverResult, ShapeFit, dict[str, float]]:
    """Re-choose the smoothing of the ruled terms of a rule fit with its rules held.

    ``fit`` is the rule fit at the smoothing ``lambdas`` the search chose without the rules,
    whose terminal fit is ``unconstrained``. The rules change the model only in the blocks of
    the terms that carry them, so only those terms' penalties are re-chosen; every other
    penalty keeps the search's value. The search may stop at the exact gradient rather than at
    the Fellner--Schall fixed point, which neglects the curvature's change with the smoothing,
    so the update is read against its own value at the search's fit: the step re-chosen is
    ``s_face(rho) - s_free``, which is zero wherever the face is the whole space. The steps are
    the update's own, unaccelerated: the face's likelihood is smooth only while the rows that
    bind stay the same, so it can hold a fixed point on more than one face, and plain steps
    from the search's smoothing reach the nearest one, the same one on every platform, where
    an extrapolated step can jump to another.

    Returns the rule fit where that step vanishes to the search's tolerance, its record, and
    its smoothing. A fit the rules leave in place, or a refit that did not converge, is
    returned as it came. A search that stops short keeps its last converged rule fit and says
    so in ``shape_fit.smoothing``, which the fit's warnings disclose.
    """
    if not shape_fit.applied or not shape_fit.converged or shape_fit.unruled:
        return fit, shape_fit, dict(lambdas)
    started = time.perf_counter()
    base = unconstrained.coefficient_face
    names = _ruled_components(layout, rows, base)
    if not names:
        return fit, shape_fit, dict(lambdas)
    free = _update(layout, lambdas, unconstrained, base, efs_config)
    offset = np.array(
        [
            float(free.raw_log_steps[name]) if free.proposal_kinds[name] == "gfs" else 0.0
            for name in names
        ]
    )
    exact = () if base is None else tuple(base.component_names)
    lower, upper = math.log(efs_config.minimum_lambda), math.log(efs_config.maximum_lambda)
    session = _DenseObservedReuseSession()

    def step_at(fit: DenseSolverResult, current: Mapping[str, float]) -> NDArray:
        """The face's step for the ruled penalties, less the search's own, clipped to the box."""
        update = _update(layout, current, fit, base, efs_config)
        raw = np.array([float(update.raw_log_steps[name]) for name in names]) - offset
        here = np.log([current[name] for name in names])
        return np.clip(raw, lower - here, upper - here)

    def refit(trial: dict[str, float]):
        penalty = layout.penalty_matrix({**trial, **dict.fromkeys(exact, 0.0)})
        return refit_with_shape_rules(
            family,
            layout,
            y,
            likelihood_plan,
            penalty,
            unconstrained,
            rows,
            config=config,
            chunk_size=chunk_size,
            phase_recorder=phase_recorder,
            start=fit,
            session=session,
        )

    def face_of(record: ShapeFit) -> tuple[int, ...]:
        return tuple(term.n_binding for term in record.terms)

    current = dict(lambdas)
    record = shape_fit
    converged, reason = False, "max_iterations"
    refits = 0
    step = step_at(fit, current)
    # A penalty whose step reverses moves at most half its last move next, so steps that cross
    # the smoothing at which a row starts to bind bisect onto it: below it the row binds and
    # the face's update rises, above it the row is free and the search's update falls. A fixed
    # point of the update on either side, if there is one, is reached inside the bracket.
    reach = np.full(len(names), float(efs_config.max_log_step))
    crossed = False
    held = (fit, record, current, step)  # the last iterate whose rows bind
    while True:
        residual = float(np.max(np.abs(step), initial=0.0))
        if residual <= efs_config.tolerance:
            converged, reason = True, "stationary"
            break
        moving = np.abs(step) > efs_config.tolerance
        if crossed and np.all(reach[moving] <= efs_config.practical_parameter_tolerance):
            converged, reason = True, "face_boundary"
            # at the boundary the side where the row binds is reported, so the record, the
            # standard errors and Diagnose name the row the smoothing sits on
            fit, record, current, step = held
            residual = float(np.max(np.abs(step), initial=0.0))
            break
        if refits >= efs_config.max_iterations:
            break
        here = np.log([current[name] for name in names])
        target = here + np.clip(step, -reach, reach)
        trial = {**current, **dict(zip(names, np.exp(target).tolist(), strict=True))}
        refitted, refit_record = refit(trial)
        refits += 1
        if not refit_record.converged:
            converged, reason = False, "refit_not_converged"
            break
        face = face_of(record)
        fit, record, current, step = refitted, refit_record, trial, step_at(refitted, trial)
        move = np.log([current[name] for name in names]) - here
        reversed_ = np.sign(step) * np.sign(move) < 0
        reach = np.where(reversed_, 0.5 * np.abs(move), reach)
        crossed = crossed or (bool(np.any(reversed_)) and face != face_of(record))
        if record.binds:
            held = (fit, record, current, step)
    if refits == 0:
        return fit, shape_fit, dict(lambdas)
    smoothing = ShapeSmoothing(
        lambdas_without_rules=dict(lambdas),
        estimated=names,
        iterations=refits,
        converged=converged,
        reason=reason,
        residual=residual,
        seconds=time.perf_counter() - started,
    )
    return fit, replace(record, smoothing=smoothing), current


__all__ = ["choose_smoothing_with_shape_rules", "face_component_states"]
