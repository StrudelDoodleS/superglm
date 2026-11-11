"""The joint Laplace objective and the isolated Wood-Fasiolo update it drives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import UNIT_ROUNDOFF, gamma_up
from superglm.distributional.layout import StackedLayout
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.solver import DenseSolverResult
from superglm.distributional.smoothing.endpoint_laml import evaluate_endpoint_laplace
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.smoothing.penalty_geometry import (
    _admit_penalty_logdet,
    _PenaltyEvaluationMemo,
)
from superglm.distributional.smoothing.proposals import _scaled_proposal
from superglm.reml.efs_update import EFSComponentState, EFSUpdateResult
from superglm.reml.penalty_algebra import (
    _compute_penalty_logdet_evaluation,
    _PenaltyLogdetEvaluation,
    penalty_component_dense_matrix,
)
from superglm.solvers.rank import RankDecomposition, _eigensolver_relative_bar
from superglm.types import LambdaPolicy


def joint_laplace_objective(
    result: DenseSolverResult,
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    objective_tolerance: float = DistributionalEFSConfig.objective_tolerance,
) -> float:
    """Return the generic joint negative Laplace criterion, up to constants.

    This evaluates the objective used to safeguard EFS fixed-point proposals;
    it does not introduce a Newton/LAML smoothing optimizer and it does not
    profile a scalar dispersion.  A modeled scale remains part of ``result``'s
    observation-level joint likelihood.
    """
    if not isinstance(result, DenseSolverResult):
        raise TypeError("result must be a DenseSolverResult")
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    if not isinstance(lambdas, Mapping):
        raise TypeError("lambdas must be a mapping")
    expected_penalty = layout.penalty_matrix(lambdas)
    if not np.array_equal(result.penalty, expected_penalty):
        raise ValueError("result penalty does not match layout and lambdas")
    penalty_evaluation = _compute_penalty_logdet_evaluation(
        dict(lambdas),
        list(layout.penalties),
    )
    assert result.penalized_optimizing_log_likelihood is not None
    objective = -result.penalized_optimizing_log_likelihood + 0.5 * (
        result.terminal_rank.log_pdet - penalty_evaluation.logdet
    )
    if not math.isfinite(objective):
        raise ValueError("joint Laplace objective is non-finite")
    _admit_penalty_logdet(penalty_evaluation, objective, objective_tolerance)
    return float(objective)


def _discarded_space_error(rank: RankDecomposition) -> float:
    """Bound on the sine between a computed discarded subspace and the exact one.

    Measured in the decomposition's equilibrated coordinates, where ``eigh``
    is backward stable: the computed eigenvectors are exact for ``G + E``
    with ``||E||_2 <= p(n) u ||G||_2`` (*LAPACK Users' Guide*, 3rd ed., sec.
    4.7, whose ``epsilon`` is ``u``). ``cutoff`` is
    ``_eigensolver_relative_bar(n) * max|w|``, i.e. ``2 n u ||G||``, so it
    bounds ``||E||`` with ``p(n) = 2n``. By Weyl the exact retained
    eigenvalues are at least ``min(retained) - cutoff`` and the computed
    discarded ones are at most ``cutoff``, so the sin theta theorem (Davis
    and Kahan, SIAM J. Numer. Anal. 7, 1970; Stewart and Sun, *Matrix
    Perturbation Theory*, 1990, ch. V) gives
    ``cutoff / (min(retained) - 2 cutoff)``; ``inf`` when that separation is
    not positive. Structural (zero-diagonal) columns are discarded as exact
    unit vectors and add nothing.
    """
    if rank.rank == 0 or int(np.count_nonzero(rank.column_scale)) == rank.rank:
        return 0.0
    gap = float(np.min(cast(NDArray, rank.retained_values))) - 2.0 * rank.cutoff
    return rank.cutoff / gap if gap > 0.0 else math.inf


def _restricted_objective(
    result: DenseSolverResult,
    objective: float,
    other: RankDecomposition,
) -> float | None:
    """``objective`` with its log-determinant taken off the other fit's discarded span.

    The Laplace log-determinant is the pseudo-determinant of the penalized
    curvature ``C`` over the subspace it retains. Restricting it to the
    orthogonal complement of the other fit's discarded directions ``N`` puts
    both fits on the same retained subspace when this fit's own discarded
    directions lie in ``span(N)``. Otherwise (fits that discard different
    directions) ``None``: compare as computed.

    Containment is tested to the accuracy the two decompositions determine
    their discarded spaces. In this fit's equilibrated coordinates its own
    space is within ``e_own`` of exact and the other's within ``k e_other``
    (:func:`_discarded_space_error`; ``k`` is the condition number of the
    diagonal map between the two fits' equilibrations, which stretches a
    sine by at most ``k``). Exactly nested spaces then have principal sines
    at most ``e_own + k e_other``, plus ``2 n u`` for forming the sines; a
    larger sine shows the spaces are not nested. Near a vanishing curvature
    the gap is small and the bound wide, so a pair is restricted unless the
    decompositions resolve that it is not nested. Measured in parameter
    coordinates the sines mislead: on the 20k freMTPL NB2 sample a pair at
    5.1e-4 there is at 3e-10 in either fit's equilibrated coordinates.

    For nested spaces Jacobi's complementary-minor identity (Brualdi and
    Schneider, Linear Algebra Appl. 52/53, 1983), in its pseudo-determinant
    form, gives the restriction without rotating ``C``:
    ``log det(Q'CQ) = log pdet(C) + log det(B'C^+B)``, with ``B`` an
    orthonormal basis of the part of ``span(N)`` orthogonal to this fit's own
    null space. ``C^+`` comes from the fit's own equilibrated decomposition,
    and ``B`` is formed from the columns of ``N``, so a coefficient whose
    curvature is ``1e-30`` beside ``1e9`` (a size near its Poisson limit) keeps
    the exact zeros the rotated ``Q'CQ`` would fill with rounding.

    Within the tolerance ``own`` may still be turned out of ``span(N)``, by
    principal angles ``theta_i`` in parameter coordinates (the coordinates
    of ``log pdet``). Write its principal vectors as
    ``u_i = cos theta_i n_i + sin theta_i r_i``, with ``n_i`` in ``span(N)``
    and ``r_i`` in ``R = span(N)^perp``. The identity covers
    ``(own + B)^perp``, which is ``R`` with each ``r_i`` replaced by
    ``w_i = r_i / cos theta_i - tan theta_i u_i``; ``C u_i = 0`` gives
    ``C w_i = C r_i / cos theta_i``, so its log-determinant exceeds the one
    over ``R`` by ``-2 sum log cos theta_i``. Adding ``sum log cos theta_i``
    to the half log-determinant makes the value exact at any admitted angle,
    including those a vacuous bound admits. The cosines are the singular
    values of ``Q_own' Q_N`` (Bjorck and Golub, Math. Comp. 27, 1973); their
    absolute error of order ``u`` is one of ``u / cos theta_i`` in the log.
    """
    null = np.asarray(other.parameter_null_basis, dtype=np.float64)
    rank = result.terminal_rank
    own = rank.parameter_null_basis
    alignment = 0.0
    if own is not None and own.shape[1]:
        own = np.asarray(own, dtype=np.float64)
        scale, other_scale = rank.column_scale, other.column_scale
        weight = np.where(scale > 0.0, scale, np.where(other_scale > 0.0, other_scale, 1.0))
        stretch = weight[other_scale > 0.0] / other_scale[other_scale > 0.0]
        other_error = _discarded_space_error(other)
        tolerance = _discarded_space_error(rank) + _eigensolver_relative_bar(rank.width)
        if other_error:
            tolerance += float(stretch.max() / stretch.min()) * other_error
        basis = np.linalg.qr(weight[:, None] * null)[0]
        own_basis = np.linalg.qr(weight[:, None] * own)[0]
        sines = np.linalg.svd(own_basis - basis @ (basis.T @ own_basis), compute_uv=False)
        if sines.max() > tolerance:
            return None
        cosines = np.linalg.svd(np.linalg.qr(own)[0].T @ np.linalg.qr(null)[0], compute_uv=False)
        # A vacuous bound admits any angle, and the correction falls without
        # limit as one nears 90 degrees: a pair closer to orthogonal than to
        # nested compares as computed, which caps the correction at log(2)/2
        # per discarded direction.
        if float(np.min(cosines)) ** 2 <= 0.5:
            return None
        alignment = float(np.sum(np.log(cosines)))
        null = null @ np.linalg.svd(own.T @ null)[2][own.shape[1] :].T
    inverse = np.column_stack([rank.solve(column) for column in null.T])
    sign, log_det = np.linalg.slogdet(null.T @ inverse)
    _, log_gram = np.linalg.slogdet(null.T @ null)
    value = objective + 0.5 * (log_det - log_gram) + alignment
    return value if sign > 0 and math.isfinite(value) else None


def rank_matched_objectives(
    current: DenseSolverResult,
    current_objective: float,
    trial: DenseSolverResult,
    trial_objective: float,
) -> tuple[float, float]:
    """Two fits' Laplace objectives over the same retained curvature subspace.

    A coefficient direction whose curvature vanishes (a size heading to its
    Poisson limit, a level whose rows all sit on the response boundary) is
    retained while its eigenvalue is above the rank cutoff and discarded once
    below it; the pseudo-determinant then drops a log-eigenvalue near the
    cutoff's, a jump the smoothing parameters did not cause. Wood, Pya and
    Saefken (JASA 111, 2016, sec. 3.1.2) drop such unidentifiable
    coefficients and evaluate the criterion in the reduced space; two fits
    are compared there, the higher-rank one restricted to the complement of
    the lower-rank one's discarded directions. Equal ranks, active faces, and
    fits whose discarded spaces are not nested compare as computed.
    """

    current_rank, trial_rank = current.terminal_rank, trial.terminal_rank
    if (
        current_rank.rank == trial_rank.rank
        or current.coefficient_face is not None
        or trial.coefficient_face is not None
    ):
        return current_objective, trial_objective
    if trial_rank.rank < current_rank.rank:
        if trial_rank.parameter_null_basis is None:
            return current_objective, trial_objective
        restricted = _restricted_objective(current, current_objective, trial_rank)
        return (current_objective if restricted is None else restricted), trial_objective
    if current_rank.parameter_null_basis is None:
        return current_objective, trial_objective
    restricted = _restricted_objective(trial, trial_objective, current_rank)
    return current_objective, (trial_objective if restricted is None else restricted)


def _laplace_objective(
    result: DenseSolverResult,
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
    objective_tolerance: float,
    penalty_memo: _PenaltyEvaluationMemo | None = None,
) -> float:
    if face is None:
        return joint_laplace_objective(
            result, layout=layout, lambdas=lambdas, objective_tolerance=objective_tolerance
        )
    return evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas=lambdas,
        face=face,
        penalty_memo=penalty_memo,
        objective_tolerance=objective_tolerance,
    ).objective


def _penalty_lambdas(
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
) -> dict[str, float]:
    values = dict(lambdas)
    if face is not None:
        for name in face.component_names:
            values[name] = 0.0
    return values


def initialize_distributional_lambdas(
    layout: StackedLayout,
    supplied: Mapping[str, float] | None,
    config: DistributionalEFSConfig,
) -> dict[str, float]:
    """Resolve qualified initial values while keeping ``LambdaPolicy`` authoritative."""
    values: Mapping[str, float] = {} if supplied is None else supplied
    if not isinstance(values, Mapping):
        raise TypeError("lambdas must be a qualified penalty-name mapping")
    unknown = set(values) - set(layout.penalty_names)
    if unknown:
        raise ValueError(f"unknown penalty lambda names: {sorted(unknown)}")

    resolved: dict[str, float] = {}
    for component in layout.penalties:
        policy = component.lambda_policy
        if policy is not None and policy.mode == "fixed":
            fixed_value = policy.value
            if fixed_value is None:  # Defensive against malformed deserialized policies.
                raise ValueError(f"fixed policy for {component.name!r} has no value")
            value = float(fixed_value)
        else:
            value = float(values.get(component.name, config.initial_lambda))
            if not config.minimum_lambda <= value <= config.maximum_lambda:
                raise ValueError(
                    f"initial lambda for {component.name!r} lies outside configured bounds"
                )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"lambda for {component.name!r} must be finite and non-negative")
        resolved[component.name] = value
    return resolved


def _component_states(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    *,
    evaluation: _PenaltyLogdetEvaluation | None = None,
) -> tuple[EFSComponentState, ...]:
    if evaluation is None:
        evaluation = _compute_penalty_logdet_evaluation(dict(lambdas), list(layout.penalties))
    effective_ranks = evaluation.gradient
    states: list[EFSComponentState] = []
    for component in layout.penalties:
        # ``EFSComponentState`` demands a dense (width, width) block, but a
        # PenaltyComponent stores a compact representation whose expansion
        # depends on ``penalty_kind``: ``identity`` carries no array at all,
        # ``repeated`` a single diagonal block, ``sum_to_zero`` a raw-level
        # block needing the contrast.  Reading ``omega_ssp``/``omega_raw``
        # directly gets three of the four kinds wrong, so delegate to the one
        # expander that owns that algebra rather than restating it here.
        penalty = penalty_component_dense_matrix(component)
        rank = float(effective_ranks[component.name])
        policy = component.lambda_policy
        if policy is None:
            policy = LambdaPolicy.estimate()
        states.append(
            EFSComponentState(
                name=component.name,
                coefficient_slice=component.group_sl,
                penalty=penalty,
                rank=rank,
                lambda_value=lambdas[component.name],
                policy=policy,
            )
        )
    return tuple(states)


def _estimated_names(components: tuple[EFSComponentState, ...]) -> tuple[str, ...]:
    return tuple(component.name for component in components if component.policy.mode == "estimate")


def _slices_overlap(left: slice, right: slice) -> bool:
    return left.start < right.stop and right.start < left.stop


def _stable_isolated_gfs_update(
    components: tuple[EFSComponentState, ...],
    fit: DenseSolverResult,
    inverse: NDArray[np.float64],
    update: EFSUpdateResult,
    config: DistributionalEFSConfig,
) -> tuple[EFSUpdateResult, frozenset[str]]:
    """Replace cancellation-prone GFS quotients where an exact identity applies.

    For one isolated penalty S with range projector P and penalized curvature
    H = A + λS, ``rank(S) - λ tr(H⁻¹S) = tr(P H⁻¹ A)``; the right side avoids
    the left's cancellation between two numbers near ``rank(S)`` as λ grows.
    It is used only for a full-rank fit and a penalty whose slice overlaps no
    other; shared-penalty blocks keep the generic update.  A rank decided from
    the rows evaluates it against the matrix it inverts
    (:func:`_representative_numerator`).
    """
    rank = fit.terminal_rank
    if fit.coefficient_face is not None or rank.rank != len(fit.coefficients):
        return update, frozenset()

    grey = rank.grey_columns if rank.row_decided else None
    epsilon = np.finfo(np.float64).eps
    stable_raw_steps: dict[str, float] = {}
    stable_quadratics: dict[str, float] = {}
    stable_traces: dict[str, float] = {}
    stable_stationarity: dict[str, float] = {}
    for component in components:
        name = component.name
        if update.proposal_kinds[name] == "inactive":
            continue
        if component.rank <= 0.0:
            continue
        # This is an algebra-selection gate, not an endpoint classifier.  The
        # direct numerator is well behaved until its trace term has absorbed
        # nearly the whole effective rank; changing formulas earlier would
        # perturb ordinary EFS trajectories for no numerical benefit.
        saturation = component.lambda_value * float(update.trace_terms[name]) / component.rank
        block = component.coefficient_slice
        trace_scale = float(np.sum(np.abs(inverse[block, block]) * np.abs(component.penalty.T)))
        trace_error = (2 * len(component.penalty) + 4) * epsilon
        trace_bound = trace_error * trace_scale / (1.0 - trace_error)
        saturation_bound = component.lambda_value * trace_bound / component.rank
        if saturation + saturation_bound < 1.0 - math.sqrt(epsilon):
            continue
        if any(
            other.name != name
            and _slices_overlap(component.coefficient_slice, other.coefficient_slice)
            for other in components
        ):
            continue

        penalty = component.penalty
        try:
            values, vectors = np.linalg.eigh(penalty)
        except np.linalg.LinAlgError:
            continue
        penalty_scale = float(np.max(np.abs(values), initial=0.0))
        if penalty_scale == 0.0:
            continue
        width = len(values)
        cutoff = 64.0 * max(width, 1) * epsilon * penalty_scale
        active = values > cutoff
        numerical_rank = int(np.count_nonzero(active))
        rank_tolerance = (
            256.0
            * max(width, 1)
            * epsilon
            * max(
                1.0,
                abs(component.rank),
            )
        )
        if numerical_rank == 0 or abs(component.rank - numerical_rank) > rank_tolerance:
            continue
        resolved = (
            _formed_numerator(component, fit, inverse, vectors[:, active])
            if grey is None
            else _representative_numerator(component, fit, grey, vectors[:, active], values[active])
        )
        if resolved is None:
            continue
        numerator, numerator_bound = resolved
        # Squaring in the penalty range avoids cancellation against its null space.
        range_beta = vectors[:, active].T @ fit.coefficients[block]
        quadratic = float(np.dot(values[active], range_beta * range_beta))
        if (
            not math.isfinite(numerator_bound)
            or numerator <= numerator_bound
            or numerator >= component.rank
            or quadratic <= 0.0
        ):
            continue
        delta = (numerator - component.lambda_value * quadratic) / component.rank
        if not math.isfinite(delta) or delta >= 1.0:
            continue
        raw_step = math.log(numerator) - math.log(quadratic) - math.log(component.lambda_value)
        trace = (component.rank - numerator) / component.lambda_value
        if math.isfinite(raw_step) and math.isfinite(trace):
            stable_raw_steps[name] = raw_step
            stable_quadratics[name] = quadratic
            stable_traces[name] = trace
            stable_stationarity[name] = -math.log1p(-delta)

    if not stable_raw_steps:
        return update, frozenset()

    raw_steps = dict(update.raw_log_steps)
    raw_steps.update(stable_raw_steps)
    quadratics = dict(update.quadratic_forms)
    quadratics.update(stable_quadratics)
    traces = dict(update.trace_terms)
    traces.update(stable_traces)
    proposal_kinds = dict(update.proposal_kinds)
    stationarity_residuals = dict(update.stationarity_log_residuals)
    stationarity_residuals.update(stable_stationarity)
    for name in stable_raw_steps:
        proposal_kinds[name] = "gfs"
    bounded_steps = {
        name: float(np.clip(raw_steps[name], -config.max_log_step, config.max_log_step))
        for name in raw_steps
    }
    proposed_lambdas, log_steps = _scaled_proposal(
        {component.name: component.lambda_value for component in components},
        bounded_steps,
        tuple(component.name for component in components if component.policy.mode == "estimate"),
        1.0,
        config,
    )
    return (
        replace(
            update,
            lambdas=proposed_lambdas,
            log_steps=log_steps,
            raw_log_steps=raw_steps,
            quadratic_forms=quadratics,
            trace_terms=traces,
            proposal_kinds=proposal_kinds,
            stationarity_log_residuals=stationarity_residuals,
        ),
        frozenset(stable_raw_steps),
    )


def _formed_numerator(
    component: EFSComponentState,
    fit: DenseSolverResult,
    inverse: NDArray[np.float64],
    range_basis: NDArray[np.float64],
) -> tuple[float, float] | None:
    """``tr(P H⁻¹ A)`` from the formed curvatures and its bound; ``None`` if unauthenticated."""
    block = component.coefficient_slice
    margin = 1024.0 * max(len(fit.coefficients), 1) * np.finfo(np.float64).eps
    projector = range_basis @ range_basis.T
    # Authenticate the projector identity before using its non-cancelling
    # form.  A truncated inverse or unresolved solve is not silently
    # promoted into convergence authority.
    penalized_curvature = fit.terminal_penalized_curvature
    identity_product = inverse[block, :] @ penalized_curvature[:, block]
    identity_trace = float(np.trace(projector @ identity_product))
    absolute_product = np.abs(inverse[block, :]) @ np.abs(penalized_curvature[:, block])
    identity_scale = float(np.sum(np.abs(projector) * absolute_product.T))
    identity_bound = margin * max(1.0, abs(component.rank), identity_scale)
    if (
        not math.isfinite(identity_trace)
        or not math.isfinite(identity_bound)
        or abs(identity_trace - component.rank) > identity_bound
    ):
        return None

    data_curvature = fit.terminal_data_curvature
    residual_product = inverse[block, :] @ data_curvature[:, block]
    numerator = float(np.trace(projector @ residual_product))
    absolute_data_product = np.abs(inverse[block, :]) @ np.abs(data_curvature[:, block])
    numerator_scale = float(np.sum(np.abs(projector) * absolute_data_product.T))
    return numerator, margin * max(numerator_scale, np.finfo(np.float64).tiny)


def _representative_numerator(
    component: EFSComponentState,
    fit: DenseSolverResult,
    grey: NDArray[np.bool_],
    range_basis: NDArray[np.float64],
    range_values: NDArray[np.float64],
) -> tuple[float, float]:
    """``tr(P G⁻¹ A)`` for the matrix a row-decided rank inverts, and its bound.

    With dual bases ``W'E = I`` and the penalty's range ``U diag(s) U'``,
    ``tr(P G⁻¹ A) = Σ_j α_j' U'(A w_j) / r_j``, ``α_j = U'w_j``.  The grey
    columns, evaluated from the rows, take the representative's own
    ``A w_j = r_j e_j - λ S w_j``, so their share is ``α_j'ε_j - λ α_j'
    diag(s) α_j / r_j`` with ``ε_j = U'e_j``: squaring in the range keeps
    ``S``'s null space out of the subtraction (Wood, Pya and Säfken 2016,
    Suppl. C).  The bound is written in ``u`` in the comment below.
    """
    # Bound, in u (Higham 2002, eq. 3.5 per dot product). Other columns: 2048 p u
    # times the absolute products, which covers the eigenvector residual, plus the
    # grey block's coupling V_g Z', at most ||ε_g||_F ||Z||_F ||α_j / r_j||_F.
    # A grey column, with a = |U|'|w_j|, b = |U|'|e_j|, α~ = |α| + γ_m a and
    # ε~ = |ε| + γ_m b (m the block width): γ_m (α~'b + a'ε~ + 2 λ s'(a α~) / r_j)
    # + γ_(k+3) (α~'ε~ + λ s'α~² / r_j), plus λ α~'|R|'|w_j| / r_j for the
    # eigensolver residual R = S U - U diag(s), within γ_(m+1) (|S||U| + |U|
    # diag(s)). Every term is a product with α or ε, so a grey column in S's null
    # space keeps a bound at rounding level. At full rank W E' = I, so
    # rank(S) = tr(P W E') needs no authentication.
    rank = fit.terminal_rank
    block = component.coefficient_slice
    solution = np.asarray(rank.solution_basis, dtype=np.float64)
    retained = np.asarray(rank.retained_values, dtype=np.float64)
    top = ~grey
    width, count = range_basis.shape
    absolute_basis = np.abs(range_basis)
    alpha = range_basis.T @ solution[block, :]

    top_solution = solution[:, top]
    data_rows = fit.terminal_data_curvature[block, :]
    data_top = range_basis.T @ (data_rows @ top_solution)
    top_share = float(np.sum(np.sum(alpha[:, top] * data_top, axis=0) / retained[top]))
    top_size = (absolute_basis.T @ np.abs(solution[block][:, top])) * (
        absolute_basis.T @ (np.abs(data_rows) @ np.abs(top_solution))
    )
    top_scale = float(np.sum(np.sum(top_size, axis=0) / retained[top]))

    lam = component.lambda_value
    grey_solution = solution[block][:, grey]
    grey_factor = np.asarray(rank.estimable_functional_basis, dtype=np.float64)[block][:, grey]
    grey_alpha = alpha[:, grey]
    grey_epsilon = range_basis.T @ grey_factor
    weight = lam / retained[grey]
    grey_share = float(
        np.sum(
            np.sum(grey_alpha * grey_epsilon, axis=0)
            - weight * (range_values @ (grey_alpha * grey_alpha))
        )
    )

    gamma_m = gamma_up(width)
    penalty = component.penalty
    residual_size = np.abs(penalty @ range_basis - range_basis * range_values) + gamma_up(
        width + 1
    ) * (np.abs(penalty) @ absolute_basis + absolute_basis * range_values)
    a = absolute_basis.T @ np.abs(grey_solution)
    b = absolute_basis.T @ np.abs(grey_factor)
    alpha_size = np.abs(grey_alpha) + gamma_m * a
    epsilon_size = np.abs(grey_epsilon) + gamma_m * b
    grey_bound = (
        gamma_m
        * (
            np.sum(alpha_size * b + a * epsilon_size, axis=0)
            + 2.0 * weight * (range_values @ (a * alpha_size))
        )
        + gamma_up(count + 3)
        * (
            np.sum(alpha_size * epsilon_size, axis=0)
            + weight * (range_values @ (alpha_size * alpha_size))
        )
        + weight * np.sum(alpha_size * (residual_size.T @ np.abs(grey_solution)), axis=0)
    )
    # ``coupling`` is ``||Z||_F`` relative to the largest formed eigenvalue.
    coupling = float((rank.grey_subspace or {}).get("coupling", math.inf))
    coupling *= float(np.max(retained[top], initial=0.0))
    coupling_bound = (
        float(np.linalg.norm(epsilon_size))
        * coupling
        * float(np.linalg.norm(alpha[:, top] / retained[top]))
    )
    bound = (
        2048.0 * len(fit.coefficients) * UNIT_ROUNDOFF * max(top_scale, np.finfo(np.float64).tiny)
        + coupling_bound
        + float(np.sum(grey_bound))
    ) * (1.0 + gamma_up(width + count + 16))
    return top_share + grey_share, bound


def _maximum_step(update: EFSUpdateResult, estimated_names: tuple[str, ...]) -> float:
    return max((abs(update.log_steps[name]) for name in estimated_names), default=0.0)


def _complete_mapping(
    names: Mapping[str, float],
    values: Mapping[str, float],
    *,
    missing: float | Mapping[str, float],
) -> dict[str, float]:
    defaults = missing if isinstance(missing, Mapping) else None
    completed: dict[str, float] = {}
    for name in names:
        if name in values:
            value = values[name]
        elif defaults is not None:
            value = defaults[name]
        else:
            value = cast(float, missing)
        completed[name] = float(value)
    return completed
