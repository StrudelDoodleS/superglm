"""Information-scaled starting penalties for distributional smoothing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import DenseSolverConfig, DistributionalEFSConfig
from superglm.distributional.smoothing.objective import initialize_distributional_lambdas
from superglm.distributional.solver.assembly import _evaluate_predictors_from_matrices
from superglm.distributional.solver.chunks import ChunkSize, _predictor_values, iter_row_chunks
from superglm.distributional.solver.derivatives import transform_natural_information
from superglm.distributional.solver.packing import packed_pairs
from superglm.distributional.solver.solver import (
    _DenseObservedReuseSession,
    _initial_coefficients,
    _validated_context,
)
from superglm.distributional.timing import FitPhaseRecorder
from superglm.reml.penalty_algebra import penalty_component_dense_matrix


def _information_scaled_lambda(fisher: NDArray, penalty: NDArray, rank: float) -> float | None:
    """Calibrate one penalty after profiling its unpenalized directions.

    Unsupported Fisher directions contribute no EDF. A penalty whose declared
    rank cannot be resolved at working precision has no certified start.
    The largest supported generalized eigenvalue makes every resolved mode's
    starting EDF at most one half. Components sharing coefficients are
    calibrated separately; this is not a constraint on their joint EDF.
    """
    width = len(penalty)
    if width == 0 or rank <= 0 or rank != int(rank) or rank > width:
        return None
    if not np.all(np.isfinite(fisher)) or not np.all(np.isfinite(penalty)):
        return None
    # Symmetric eigensolves and matrix products have normwise backward errors
    # proportional to dimension * epsilon. Account for the two decompositions
    # and the whitening/projection products, in their respective matrix units.
    arithmetic = (8 * width + 4) * np.finfo(float).eps
    s, u = np.linalg.eigh((penalty + penalty.T) * 0.5)
    s_error = arithmetic * np.max(np.abs(s))
    nullity = width - int(rank)
    if s[nullity] <= s_error or np.any(np.abs(s[:nullity]) > s_error):
        return None
    r = u[:, nullity:] / np.sqrt(s[nullity:])
    n = u[:, :nullity]
    f, v = np.linalg.eigh((fisher + fisher.T) * 0.5)
    f_error = arithmetic * np.max(np.abs(f))
    if np.min(f) < -f_error:
        return None
    supported = f > f_error
    if not np.any(supported):
        return None
    factor = np.sqrt(f[supported])[:, None] * v[:, supported].T
    a = factor @ r
    b = factor @ n
    a_norm = np.linalg.norm(a, 2)
    lifted = r
    projection_margin = 1.0
    if nullity:
        q, singular, vt = np.linalg.svd(b, full_matrices=False)
        # Uncertainty in a Gram factor is bounded on the square-root scale.
        b_error = np.sqrt(f_error) * np.linalg.norm(n, 2)
        active = singular > b_error
        b_roundoff = arithmetic * np.linalg.norm(factor, 2) * np.linalg.norm(n, 2)
        if np.any((singular > b_roundoff) & ~active):
            # A small but unresolved nuisance direction can still span all of
            # A. Discarding it would invent supported penalized information.
            return None
        q = q[:, active]
        if np.any(active):
            nuisance = (vt[active].T / singular[active]) @ (q.T @ a)
            lifted = r - n @ nuisance
            projection_margin = 1.0 - f_error / np.min(singular[active]) ** 2
        a = a - q @ (q.T @ a)
    singular = np.linalg.svd(a, compute_uv=False)
    d = singular**2
    # The profiled quadratic is evaluated on R - N B^+ A. Its sensitivity
    # includes the conditioning of the supported nuisance solve, not merely R.
    # Penalty eigenspace/whitening error scales with its certified positive gap;
    # the final term covers cancellation in the projection and residual SVD.
    d_error = (
        f_error * np.linalg.norm(lifted, 2) ** 2 / projection_margin
        + s_error / (s[nullity] - s_error) * a_norm**2
        + arithmetic * (a_norm + np.linalg.norm(factor, 2) * np.linalg.norm(lifted, 2)) ** 2
    )
    d = d[d > d_error]
    if len(d) == 0:
        return None
    return float(np.max(d))


def prepare_distributional_initialization(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    supplied: Mapping[str, float] | None,
    config: DistributionalEFSConfig,
    solver_config: DenseSolverConfig,
    initial: NDArray | None,
    chunk_size: ChunkSize | None,
    reuse_session: _DenseObservedReuseSession,
    phase_recorder: FitPhaseRecorder | None,
) -> tuple[dict[str, float], NDArray | None]:
    """Resolve overrides, then calibrate missing starts at admitted coefficients.

    Families without expected information and blocks with unresolved numerical
    rank retain the previous 0.1 start, clipped to the requested bounds.
    """
    if config.initial_lambda is not None:
        return initialize_distributional_lambdas(layout, supplied, config), initial
    fallback = float(np.clip(0.1, config.minimum_lambda, config.maximum_lambda))
    resolved = initialize_distributional_lambdas(
        layout, supplied, replace(config, initial_lambda=fallback)
    )
    automatic = {
        component.name
        for component in layout.penalties
        if (supplied is None or component.name not in supplied)
        and (component.lambda_policy is None or component.lambda_policy.mode != "fixed")
    }
    if not automatic or not isinstance(family, ExpectedInformationFamily):
        return resolved, initial
    matrices = (
        reuse_session.dense_matrices(layout, phase_recorder=phase_recorder)
        if chunk_size is None
        else None
    )
    context = _validated_context(
        family,
        layout,
        y,
        likelihood_plan,
        layout.penalty_matrix(resolved),
        coefficient_curvature=solver_config.coefficient_curvature,
        chunk_size=chunk_size,
        coefficient_face=None,
        dense_matrices=matrices,
        _reuse_session=reuse_session,
    )
    coefficients = (
        _initial_coefficients(context) if initial is None else np.asarray(initial, dtype=np.float64)
    )
    if coefficients.shape != (layout.n_coefficients,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("initial must be a finite vector with the global layout shape")
    if context.chunk_size is None:
        assert context.dense_matrices is not None
        eta = _evaluate_predictors_from_matrices(layout, coefficients, context.dense_matrices)
    else:
        eta = np.empty((len(context.response), len(layout.predictors)), dtype=np.float64)
        for rows in iter_row_chunks(len(context.response), context.chunk_size):
            eta[rows.start : rows.stop] = _predictor_values(
                layout, coefficients, rows, include_offsets=True
            )
    theta = np.empty_like(eta)
    for state in layout.predictors:
        theta[:, state.parameter_index] = state.link.inverse(eta[:, state.parameter_index])
    information = transform_natural_information(
        family.expected_information_natural(theta, context.likelihood_plan), eta, context.links
    )
    diagonal = {
        left: packed_index
        for packed_index, (left, right) in enumerate(packed_pairs(len(layout.predictors)))
        if left == right
    }
    for state in layout.predictors:
        weights = information[:, diagonal[state.parameter_index]]
        grams: dict[int, NDArray] = {}
        for component in state.penalties:
            if component.name not in automatic:
                continue
            if component.group_index not in grams:
                grams[component.group_index] = state.design.group_matrices[
                    component.group_index
                ].gram(weights)
            value = _information_scaled_lambda(
                grams[component.group_index],
                penalty_component_dense_matrix(component),
                component.rank,
            )
            if value is not None:
                resolved[component.name] = float(
                    np.clip(value, config.minimum_lambda, config.maximum_lambda)
                )
    return resolved, coefficients
