"""Exact gradient and Hessian of the negative LAML in log smoothing parameters.

Wood, Pya and Säfken (2016) §3.1.3–3.2 with the third and fourth derivatives of
the row log-likelihood supplied by finite differences of the packed row
curvature along the K parameter directions (``endpoint_direction``), recombined
by linearity.  Every quantity is for ``F(rho) = joint_laplace_objective`` at the
fit's own penalised curvature ``H``, and the packed curvature is the negated
row Hessian in the linear predictors.

The dense coefficient solver publishes the observed penalised Hessian as its
terminal curvature whatever step policy it ran (``coefficient_curvature`` only
chooses the inner iteration's geometry), so ``H = -grad^2 l(beta_hat) + S`` is
the matrix inside ``F`` and the one the implicit derivative of the mode goes
through; a fit whose terminal curvature fell back to Fisher had a materially
indefinite observed Hessian, where that derivative does not exist, and is
refused.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    DistributionalFamily,
    FamilyLikelihoodPlan,
    PredictorCurvatureDirectionalFamily,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import ANALYTIC_DIRECTION_AUTHORITY, DenseSolverResult
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    FINITE_DIFFERENCE_AUTHORITY,
    _curvature_packed,
    _first_difference,
    _second_difference,
    _unit_directions,
)
from superglm.distributional.smoothing.endpoint_laml import (
    _projected_finite_penalty_inputs,
    _projected_penalty_group_indices,
)
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import _component_states, _estimated_names
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver.packing import packed_pairs
from superglm.distributional.weights import UnsupportedLikelihoodContractError
from superglm.reml.efs_update import EFSComponentState
from superglm.reml.multi_penalty import logdet_s_hessian, similarity_transform_logdet
from superglm.reml.penalty_algebra import compute_logdet_s_derivatives

_PackedRows = Callable[[NDArray[np.float64]], NDArray[np.float64]]
_Stencil = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
#: Rows per chunk of the row-chunked assemblies (scratch of ``chunk * m * P_r`` doubles).
_ROW_CHUNK = 4096


class LamlDerivativeError(RuntimeError):
    """The derivatives cannot be formed at this fit."""


def _readonly(values: NDArray, *, name: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, not {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LamlDerivatives:
    """Gradient and Hessian of the negative LAML in log lambda at one fit.

    ``hessian`` and ``hessian_certificate`` are ``None`` when the caller asked
    for the gradient only.  ``fs_gradient`` holds the three Fellner–Schall
    terms alone, ``gradient`` minus the trace of ``H^-1`` against the
    third-derivative correction that the FS update omits.  ``penalty_hessian``
    is the Hessian with the likelihood's third and fourth derivatives dropped:
    every term of ``h_kl`` that the fit's factor and the penalties alone
    determine (exact for a Gaussian identity link), which a gradient-only pass
    freezes and judges its stopping test with; it vanishes at a working
    infinity as the exact Hessian does, where the Fellner–Schall diagonal
    ``1/2 lambda_k beta' S_k beta + 1/2 lambda_k tr(H^-1 S_k)`` tends to half
    the penalty rank.  ``evaluations`` counts the packed-curvature evaluations
    this call made (a Hessian pass reusing a gradient pass's stencils counts
    only its own).
    """

    names: tuple[str, ...]
    gradient: NDArray[np.float64]
    hessian: NDArray[np.float64] | None
    gradient_certificate: NDArray[np.float64]
    hessian_certificate: NDArray[np.float64] | None
    fs_gradient: NDArray[np.float64]
    penalty_hessian: NDArray[np.float64]
    third_derivative_authority: str
    evaluations: int
    provenance: tuple[object, ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("names must be a non-empty tuple of component names")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")
        count = len(names)
        gradient = _readonly(self.gradient, name="gradient", shape=(count,))
        fs_gradient = _readonly(self.fs_gradient, name="fs_gradient", shape=(count,))
        penalty_hessian = _readonly(
            self.penalty_hessian, name="penalty_hessian", shape=(count, count)
        )
        if not np.array_equal(penalty_hessian, penalty_hessian.T):
            raise ValueError("penalty_hessian must be symmetric")
        gradient_certificate = _readonly(
            self.gradient_certificate, name="gradient_certificate", shape=(count,)
        )
        if np.any(gradient_certificate < 0.0):
            raise ValueError("gradient_certificate must be non-negative")
        if (self.hessian is None) != (self.hessian_certificate is None):
            raise ValueError("hessian and hessian_certificate must be present together")
        hessian = None
        hessian_certificate = None
        if self.hessian is not None:
            hessian = _readonly(self.hessian, name="hessian", shape=(count, count))
            if not np.array_equal(hessian, hessian.T):
                raise ValueError("hessian must be symmetric")
            assert self.hessian_certificate is not None
            hessian_certificate = _readonly(
                self.hessian_certificate, name="hessian_certificate", shape=(count, count)
            )
            if np.any(hessian_certificate < 0.0):
                raise ValueError("hessian_certificate must be non-negative")
        if self.third_derivative_authority not in (
            ANALYTIC_DIRECTION_AUTHORITY,
            FINITE_DIFFERENCE_AUTHORITY,
        ):
            raise ValueError("unknown third-derivative authority")
        if (
            isinstance(self.evaluations, bool)
            or not isinstance(self.evaluations, int)
            or self.evaluations < 0
        ):
            raise ValueError("evaluations must be a non-negative integer")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "fs_gradient", fs_gradient)
        object.__setattr__(self, "penalty_hessian", penalty_hessian)
        object.__setattr__(self, "gradient_certificate", gradient_certificate)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "hessian_certificate", hessian_certificate)
        object.__setattr__(self, "provenance", tuple(self.provenance))


def _validated_matrices(
    layout: StackedLayout,
    dense_matrices: Sequence[NDArray],
    n_observations: int,
) -> tuple[NDArray[np.float64], ...]:
    matrices = tuple(np.asarray(matrix, dtype=np.float64) for matrix in dense_matrices)
    if len(matrices) != len(layout.predictors):
        raise ValueError("one dense predictor matrix per predictor is required")
    for state, matrix in zip(layout.predictors, matrices, strict=True):
        width = state.coefficient_slice.stop - state.coefficient_slice.start
        if matrix.shape != (n_observations, width) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"dense matrix for predictor {state.name!r} has invalid shape")
    return matrices


def _stencil(
    rows: _PackedRows,
    eta: NDArray[np.float64],
    unit: NDArray[np.float64],
    step: float,
) -> _Stencil:
    """Packed rows at ``eta +- step*u`` and ``eta +- step/2*u`` (four evaluations)."""
    plus = rows(eta + step * unit)
    minus = rows(eta - step * unit)
    half = 0.5 * step
    return plus, minus, rows(eta + half * unit), rows(eta - half * unit)


def _first_derivatives(
    rows: _PackedRows,
    eta: NDArray[np.float64],
    step: float,
    *,
    analytic_first: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[_Stencil | None], int]:
    """First derivatives of the packed rows along the parameter axes.

    Returns ``(D1, C1, stencils, evaluations)``: ``D1[q]`` is ``d rows / d eta_q``
    with certificate ``C1[q]`` (zero for an analytic derivative) and
    ``stencils[q]`` the four-point axis stencil it was differenced on, which
    the second-derivative stage shares (``None`` for an analytic derivative,
    which evaluates nothing).
    """
    n_observations, k_parameters = eta.shape
    first: list[NDArray[np.float64]] = []
    first_certificate: list[NDArray[np.float64]] = []
    stencils: list[_Stencil | None] = []
    evaluations = 0
    for q in range(k_parameters):
        unit = np.zeros((n_observations, k_parameters), dtype=np.float64)
        unit[:, q] = 1.0
        stencil: _Stencil | None = None
        if analytic_first is not None:
            values = analytic_first(unit)
            certificate = np.zeros_like(values)
        else:
            stencil = _stencil(rows, eta, unit, step)
            evaluations += 4
            values, certificate = _first_difference(stencil, step)
        first.append(values)
        first_certificate.append(certificate)
        stencils.append(stencil)
    for values in first:
        if not np.all(np.isfinite(values)):
            raise LamlDerivativeError("a derivative of the packed row curvature is not finite")
    return first, first_certificate, stencils, evaluations


def _second_derivatives(
    rows: _PackedRows,
    eta: NDArray[np.float64],
    step: float,
    stencils: Sequence[_Stencil | None],
) -> tuple[
    dict[tuple[int, int], NDArray[np.float64]],
    dict[tuple[int, int], NDArray[np.float64]],
    int,
]:
    """Second derivatives of the packed rows along the axes and the mixed pairs.

    Returns ``(D2, C2, evaluations)``: ``D2[(q, r)]`` for ``q <= r`` is
    ``d^2 rows / d eta_q d eta_r`` with certificate ``C2[(q, r)]``.  A diagonal
    pair is the five-point stencil on the axis stencil (evaluated here when
    the first-derivative stage was analytic and left none) and the centre
    value, which every pair shares; a mixed pair is the single-direction
    polarisation ``D4[e_q, e_r] = (D4[e_q + e_r, e_q + e_r] - D4[e_q, e_q]
    - D4[e_r, e_r]) / 2`` on one extra stencil along ``(e_q + e_r) / sqrt 2``,
    reusing the diagonal second differences, with half the sum of the three
    certificates as its own: ``1 + 4K + 4K(K - 1)/2`` evaluations in all
    (25 at K = 3), the axis stencils reused from the gradient pass when it
    differenced them.
    """
    n_observations, k_parameters = eta.shape
    center = rows(eta)
    evaluations = 1
    second: dict[tuple[int, int], NDArray[np.float64]] = {}
    second_certificate: dict[tuple[int, int], NDArray[np.float64]] = {}
    for q in range(k_parameters):
        stencil = stencils[q]
        if stencil is None:
            unit = np.zeros((n_observations, k_parameters), dtype=np.float64)
            unit[:, q] = 1.0
            stencil = _stencil(rows, eta, unit, step)
            evaluations += 4
        second[(q, q)], second_certificate[(q, q)] = _second_difference(center, stencil, step)
    for q in range(k_parameters):
        for r in range(q + 1, k_parameters):
            direction = np.zeros((n_observations, k_parameters), dtype=np.float64)
            direction[:, q] = 1.0
            direction[:, r] = 1.0
            norms, unit = _unit_directions(direction)
            stencil = _stencil(rows, eta, unit, step)
            evaluations += 4
            values, certificate = _second_difference(center, stencil, step)
            scale = (norms * norms)[:, None]
            second[(q, r)] = (values * scale - second[(q, q)] - second[(r, r)]) / 2.0
            second_certificate[(q, r)] = (
                certificate * scale + second_certificate[(q, q)] + second_certificate[(r, r)]
            ) / 2.0
    for values in second.values():
        if not np.all(np.isfinite(values)):
            raise LamlDerivativeError("a derivative of the packed row curvature is not finite")
    return second, second_certificate, evaluations


def _predictor_directions(
    matrices: tuple[NDArray[np.float64], ...],
    slices: tuple[slice, ...],
    vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """eta direction ``X v`` as an ``(n, K)`` array."""
    return np.column_stack(
        [matrix @ vector[block] for matrix, block in zip(matrices, slices, strict=True)]
    )


def _cross_all(
    matrices: tuple[NDArray[np.float64], ...],
    slices: tuple[slice, ...],
    packed: NDArray[np.float64],
    width: int,
    *,
    chunk: int = _ROW_CHUNK,
) -> NDArray[np.float64]:
    """``X^T diag-blocks(packed[:, k, :]) X`` for every ``k`` at once, as ``(m, P, P)``.

    ``packed`` is ``(n, m, K(K+1)/2)``, the packed row weights of ``m``
    components.  For each packed channel ``(l, r)`` the ``m`` products
    ``X_l^T diag(w_k) X_r`` are one GEMM per chunk of rows,
    ``X_l[c]^T @ (w[c, :, channel][:, :, None] * X_r[c][:, None, :]).reshape(b, m P_r)``,
    with at most ``chunk * m * P_r`` doubles of scratch (15 MB at the retained
    cell's ``m = 15``, ``P_r = 31`` and the default chunk).
    """
    n_observations, count, channels = packed.shape
    pairs = packed_pairs(len(matrices))
    if channels != len(pairs):
        raise ValueError("packed weights must carry one channel per parameter pair")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    result = np.zeros((count, width, width), dtype=np.float64)
    for channel, (left, right) in enumerate(pairs):
        left_matrix = matrices[left]
        right_matrix = matrices[right]
        width_left = left_matrix.shape[1]
        width_right = right_matrix.shape[1]
        accumulator = np.zeros((width_left, count * width_right), dtype=np.float64)
        for start in range(0, n_observations, chunk):
            rows = slice(start, min(start + chunk, n_observations))
            weighted = (
                packed[rows, :, channel][:, :, None] * right_matrix[rows][:, None, :]
            ).reshape(rows.stop - rows.start, count * width_right)
            accumulator += left_matrix[rows].T @ weighted
        block = np.moveaxis(accumulator.reshape(width_left, count, width_right), 1, 0)
        result[:, slices[left], slices[right]] += block
        if left != right:
            result[:, slices[right], slices[left]] += np.transpose(block, (0, 2, 1))
    return 0.5 * (result + np.transpose(result, (0, 2, 1)))


def _leverage_blocks(
    matrices: tuple[NDArray[np.float64], ...],
    slices: tuple[slice, ...],
    inverse: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``G_i = X_i H^-1 X_i^T`` for every row, packed ``(n, K(K+1)/2)``."""
    k_parameters = len(matrices)
    pairs = packed_pairs(k_parameters)
    blocks = np.empty((matrices[0].shape[0], len(pairs)), dtype=np.float64)
    for channel, (left, right) in enumerate(pairs):
        product = matrices[left] @ inverse[slices[left], slices[right]]
        blocks[:, channel] = np.einsum("ij,ij->i", product, matrices[right])
    return blocks


def _packed_weights(k_parameters: int) -> NDArray[np.float64]:
    return np.array(
        [1.0 if left == right else 2.0 for left, right in packed_pairs(k_parameters)],
        dtype=np.float64,
    )


def _rank_hessian(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
) -> dict[tuple[str, str], float]:
    """``d^2 log|S_lambda|_+ / d rho_i d rho_j`` for every pair in one penalty group."""
    if face is None:
        _ranks, hessian = compute_logdet_s_derivatives(dict(lambdas), list(layout.penalties))
        return {key: float(value) for key, value in hessian.items()}
    components, projected, values = _projected_finite_penalty_inputs(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )
    result: dict[tuple[str, str], float] = {}
    if not components or not projected:
        return result
    for indices in _projected_penalty_group_indices(components):
        group_projected = [projected[index] for index in indices]
        group_values = values[list(indices)]
        decomposition = similarity_transform_logdet(group_projected, group_values)
        hessian = np.asarray(
            logdet_s_hessian(decomposition, group_projected, group_values),
            dtype=np.float64,
        )
        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                result[(components[global_i].name, components[global_j].name)] = float(
                    hessian[local_i, local_j]
                )
    return result


def _penalty_apply(
    component: EFSComponentState, vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``S_k v`` in full coordinates: the component's block applied to its slice."""
    result = np.zeros_like(vector)
    block = component.coefficient_slice
    result[block] = component.penalty @ vector[block]
    return result


def _analytic_first_derivative(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray[np.float64],
    links: tuple,
    likelihood_plan: FamilyLikelihoodPlan,
    n_observations: int,
    k_parameters: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]] | None:
    """The family's analytic directional derivative of the observed curvature, if usable.

    ``None`` when the family does not implement the protocol or refuses these
    links (GammaLS accepts only its log links); the caller then differences.
    """
    if not (
        isinstance(family, PredictorCurvatureDirectionalFamily)
        and callable(getattr(family, "predictor_curvature_directional_derivative", None))
    ):
        return None
    expected = (n_observations, k_parameters * (k_parameters + 1) // 2)

    def derivative(unit: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(
            family.predictor_curvature_directional_derivative(y, eta, unit, links, likelihood_plan),
            dtype=np.float64,
        )
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise LamlDerivativeError("family returned an invalid curvature direction")
        return values

    probe = np.zeros((n_observations, k_parameters), dtype=np.float64)
    probe[:, 0] = 1.0
    try:
        derivative(probe)
    except (
        TypeError,
        ValueError,
        FloatingPointError,
        OverflowError,
        UnsupportedLikelihoodContractError,
        LamlDerivativeError,
    ):
        return None
    return derivative


@dataclass
class _GradientPass:
    """Everything one gradient pass computed, for a Hessian pass at the same fit.

    With ``G_i`` the row leverage blocks and ``w`` the packed weights (two on
    an off-diagonal channel): ``direction_stack[:, q, k]`` is ``eta_k[:, q]``,
    ``coefficient_directions[:, k]`` is ``beta_k``, ``weighted_blocks`` is
    ``G_i w`` row by row, ``third_vector`` is ``sum_q X_q' P_q`` with
    ``P_q[i] = <D1_q[i], G_i>_w``, ``certificate_leverage[:, q]`` is
    ``<C1_q[i], |G_i|>_w`` and ``leverage_trace[i]`` is ``tr(G_i)``, the
    squared Frobenius norm of ``X_i H^-1/2`` over the ``K`` design rows of
    observation ``i``.
    """

    family: DistributionalFamily
    likelihood_plan: FamilyLikelihoodPlan
    fit: DenseSolverResult
    lambdas: dict[str, float]
    step: float
    source_matrices: tuple[object, ...]
    estimated: tuple[EFSComponentState, ...]
    rank_hessian: dict[tuple[str, str], float]
    rows: _PackedRows
    eta: NDArray[np.float64]
    matrices: tuple[NDArray[np.float64], ...]
    slices: tuple[slice, ...]
    width: int
    beta: NDArray[np.float64]
    inverse: NDArray[np.float64]
    first: list[NDArray[np.float64]]
    first_certificate: list[NDArray[np.float64]]
    stencils: list[_Stencil | None]
    lambda_values: NDArray[np.float64]
    penalty_beta: list[NDArray[np.float64]]
    quadratic: NDArray[np.float64]
    trace: NDArray[np.float64]
    beta_k: list[NDArray[np.float64]]
    coefficient_directions: NDArray[np.float64]
    direction_stack: NDArray[np.float64]
    weights: NDArray[np.float64]
    weighted_blocks: NDArray[np.float64]
    absolute_weighted_blocks: NDArray[np.float64]
    third_vector: NDArray[np.float64]
    certificate_leverage: NDArray[np.float64]
    leverage_trace: NDArray[np.float64]
    result: LamlDerivatives

    def matches(
        self,
        family: DistributionalFamily,
        likelihood_plan: FamilyLikelihoodPlan,
        fit: DenseSolverResult,
        lambdas: Mapping[str, float],
        step: float,
        dense_matrices: Sequence[NDArray],
    ) -> bool:
        return (
            self.family is family
            and self.likelihood_plan is likelihood_plan
            and self.fit is fit
            and self.lambdas == dict(lambdas)
            and self.step == step
            and len(self.source_matrices) == len(dense_matrices)
            and all(
                held is given
                for held, given in zip(self.source_matrices, dense_matrices, strict=True)
            )
        )


class LamlDerivativeWorkspace:
    """Carries a gradient pass's stencils and leverage blocks to a Hessian pass at the same fit.

    Pass one instance to ``laml_derivatives(..., want_hessian=False, reuse=ws)``
    and then to ``laml_derivatives(..., want_hessian=True, reuse=ws)`` at the
    same fit, lambdas, step and dense matrices: the Hessian pass then adds
    only the centre and mixed-pair evaluations to the gradient pass's axis
    stencils.  A pass at anything else is ignored and recomputed.  The held
    arrays are ``O(n K^2)`` each; ``clear`` releases them.
    """

    __slots__ = ("_pass",)

    def __init__(self) -> None:
        self._pass: _GradientPass | None = None

    def clear(self) -> None:
        self._pass = None


def _gradient_pass(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    fit: DenseSolverResult,
    dense_matrices: Sequence[NDArray],
    step: float,
) -> _GradientPass:
    source = fit.terminal_curvature.actual_source
    face = fit.coefficient_face
    components = (
        _component_states(layout, lambdas)
        if face is None
        else projected_component_states(layout=layout, lambdas=lambdas, face=face)
    )
    names = _estimated_names(components)
    if not names:
        raise LamlDerivativeError("no estimated smoothing component to differentiate")
    estimated = tuple(component for component in components if component.name in set(names))
    rank_hessian = _rank_hessian(layout, lambdas, face)

    eta = np.asarray(fit.eta, dtype=np.float64)
    n_observations, k_parameters = eta.shape
    matrices = _validated_matrices(layout, dense_matrices, n_observations)
    slices = tuple(state.coefficient_slice for state in layout.predictors)
    links = tuple(state.link for state in layout.predictors)
    width = layout.n_coefficients
    beta = np.asarray(fit.coefficients, dtype=np.float64)
    if beta.shape != (width,):
        raise ValueError("fit coefficients do not match the layout width")
    inverse = np.asarray(fit.terminal_pseudo_inverse(), dtype=np.float64)

    def observed_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _curvature_packed(family, y, values, links, likelihood_plan)

    analytic_first = _analytic_first_derivative(
        family, y, eta, links, likelihood_plan, n_observations, k_parameters
    )
    try:
        first, first_certificate, stencils, evaluations = _first_derivatives(
            observed_rows, eta, step, analytic_first=analytic_first
        )
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise LamlDerivativeError("the row curvature could not be differentiated") from exc
    authority = (
        ANALYTIC_DIRECTION_AUTHORITY if analytic_first is not None else FINITE_DIFFERENCE_AUTHORITY
    )

    count = len(estimated)
    lambda_values = np.array([component.lambda_value for component in estimated])
    ranks = np.array([component.rank for component in estimated])
    penalty_beta = [_penalty_apply(component, beta) for component in estimated]
    quadratic = np.array(
        [
            lambda_values[k]
            * float(
                beta[component.coefficient_slice] @ penalty_beta[k][component.coefficient_slice]
            )
            for k, component in enumerate(estimated)
        ]
    )
    trace = np.array(
        [
            lambda_values[k]
            * float(
                np.sum(
                    inverse[component.coefficient_slice, component.coefficient_slice]
                    * component.penalty
                )
            )
            for k, component in enumerate(estimated)
        ]
    )
    beta_k = [-(inverse @ (lambda_values[k] * penalty_beta[k])) for k in range(count)]
    coefficient_directions = np.column_stack(beta_k)
    direction_stack = np.stack(
        [matrices[q] @ coefficient_directions[slices[q]] for q in range(k_parameters)], axis=1
    )
    weights = _packed_weights(k_parameters)
    weighted_blocks = _leverage_blocks(matrices, slices, inverse) * weights
    absolute_weighted_blocks = np.abs(weighted_blocks)
    diagonal_channels = [
        channel for channel, (left, right) in enumerate(packed_pairs(k_parameters)) if left == right
    ]
    leverage_trace = np.sum(weighted_blocks[:, diagonal_channels], axis=1)
    # The third-derivative term of g_k is sum_i <D3_i[eta_k[i]], G_i>_w
    # = sum_q sum_i eta_k[i, q] P_q[i] = beta_k . v with v = sum_q X_q' P_q, and
    # its certificate sum_q sum_i |eta_k[i, q]| N_q[i] with N_q = <C1_q, |G|>_w:
    # no per-component row array is ever formed.
    first_leverage = np.column_stack(
        [np.sum(weighted_blocks * first[q], axis=1) for q in range(k_parameters)]
    )
    certificate_leverage = np.column_stack(
        [
            np.sum(absolute_weighted_blocks * first_certificate[q], axis=1)
            for q in range(k_parameters)
        ]
    )
    third_vector = np.zeros(width, dtype=np.float64)
    for q in range(k_parameters):
        third_vector[slices[q]] = matrices[q].T @ first_leverage[:, q]
    fs_gradient = 0.5 * quadratic + 0.5 * trace - 0.5 * ranks
    gradient = fs_gradient + 0.5 * (third_vector @ coefficient_directions)
    gradient_certificate = np.zeros(count, dtype=np.float64)
    for q in range(k_parameters):
        gradient_certificate += np.abs(direction_stack[:, q, :]).T @ certificate_leverage[:, q]
    gradient_certificate *= 0.5
    # Every term of h_kl that needs no third or fourth derivative of the
    # likelihood, from the factor and the penalties alone:
    #   1/2 delta_kl lambda_k beta'S_k beta + lambda_k beta'S_k beta_l
    #   + 1/2 delta_kl lambda_k tr(H^-1 S_k) - 1/2 lambda_k lambda_l tr(H^-1 S_l H^-1 S_k)
    #   - 1/2 d^2 log|S|_+ / d rho_k d rho_l
    # with tr(H^-1 S_l H^-1 S_k) over the (k, l) block pair only, since S_k
    # lives on block k.
    penalty_hessian = np.empty((count, count))
    for k, component_k in enumerate(estimated):
        block_k = component_k.coefficient_slice
        for ell, component_l in enumerate(estimated):
            block_l = component_l.coefficient_slice
            same = k == ell
            left = inverse[block_k, block_l] @ component_l.penalty
            right = inverse[block_l, block_k] @ component_k.penalty
            product_trace = float(np.sum(left * right.T, dtype=np.float64))
            penalty_hessian[k, ell] = (
                (0.5 * quadratic[k] if same else 0.0)
                + lambda_values[k] * float(penalty_beta[k] @ beta_k[ell])
                + (0.5 * trace[k] if same else 0.0)
                - 0.5 * lambda_values[k] * lambda_values[ell] * product_trace
                - 0.5 * rank_hessian.get((component_k.name, component_l.name), 0.0)
            )
    penalty_hessian = 0.5 * (penalty_hessian + penalty_hessian.T)

    provenance: tuple[object, ...] = (
        fit.terminal_rank.method,
        int(fit.terminal_rank.rank),
        source,
        () if face is None else tuple(face.component_names),
    )
    result = LamlDerivatives(
        names=names,
        gradient=gradient,
        hessian=None,
        gradient_certificate=gradient_certificate,
        hessian_certificate=None,
        fs_gradient=fs_gradient,
        penalty_hessian=penalty_hessian,
        third_derivative_authority=authority,
        evaluations=evaluations,
        provenance=provenance,
    )
    return _GradientPass(
        family=family,
        likelihood_plan=likelihood_plan,
        fit=fit,
        lambdas=dict(lambdas),
        step=step,
        source_matrices=tuple(dense_matrices),
        estimated=estimated,
        rank_hessian=rank_hessian,
        rows=observed_rows,
        eta=eta,
        matrices=matrices,
        slices=slices,
        width=width,
        beta=beta,
        inverse=inverse,
        first=first,
        first_certificate=first_certificate,
        stencils=stencils,
        lambda_values=lambda_values,
        penalty_beta=penalty_beta,
        quadratic=quadratic,
        trace=trace,
        beta_k=beta_k,
        coefficient_directions=coefficient_directions,
        direction_stack=direction_stack,
        weights=weights,
        weighted_blocks=weighted_blocks,
        absolute_weighted_blocks=absolute_weighted_blocks,
        third_vector=third_vector,
        certificate_leverage=certificate_leverage,
        leverage_trace=leverage_trace,
        result=result,
    )


def _hessian_pass(gradient_pass: _GradientPass, *, reused: bool) -> LamlDerivatives:
    """The Hessian on top of a gradient pass; ``reused`` means its stencils came from an earlier call.

    The certificate is a norm bound from scalars, taken in the ``H`` metric so
    the row leverage survives: with ``Y_i = X_i H^-1/2`` (``||Y_i||_F^2 =
    tr(G_i)``, the leverage blocks' diagonal) and ``C_k[i]`` the
    finite-difference certificate of ``V_k[i]``, ``e_k = sum_i tr(G_i)
    <C_k[i], w>`` bounds ``||H^-1/2 delta H_k H^-1/2||`` for ``delta H_k =
    X' diag-blocks(delta V_k) X``; then

    * ``|delta tr(H^-1 H_l H^-1 H_k)| <= ||H^-1/2 H_l H^-1/2||_2 e_k + ||H^-1/2 H_k H^-1/2||_2 e_l``
      (``tr(H^-1 H_l H^-1 delta H_k) = sum_i <Y_i M_l Y_i', delta V_k[i]>``),
    * ``||H^1/2 delta beta_kl|| <= e_l ||H^1/2 beta_k||`` through the solve
      against ``H_l beta_k``, reaching ``eta_kl`` by ``sqrt(tr G_i)`` and
      ``V_kl`` through ``||D1[i, c, :]||_2``, so its contraction against
      ``|G|`` is that scalar times one row sum formed once,
    * the fourth-derivative certificate contraction ``sum_i <|eta_k| C2 |eta_l|, |G|>_w``
      and the ``eta_kl`` contraction against ``C1`` are the exact entrywise
      sums, as before, but for every pair at once (``K^2`` GEMMs of the
      ``(n, m)`` direction matrices, and one row-chunked GEMM of the pair
      directions).

    The Euclidean form of the same bound (``||X_i||^2`` for ``tr(G_i)``,
    ``||H^-1||_2`` and ``||H^-1 H_l H^-1||_2`` for the relative norms) was
    2.5e5 times the entrywise certificate on the Gamma finite-difference
    fixture and distrusted the Hessian at the default step.  Every quantity is
    one pass over the rows or ``O(m P^3)`` once.
    """
    p = gradient_pass
    try:
        second, second_certificate, evaluations = _second_derivatives(
            p.rows, p.eta, p.step, p.stencils
        )
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise LamlDerivativeError("the row curvature could not be differentiated") from exc
    if not reused:
        evaluations += p.result.evaluations
    n_observations, k_parameters = p.eta.shape
    estimated = p.estimated
    count = len(estimated)
    matrices, slices, width, inverse, beta = p.matrices, p.slices, p.width, p.inverse, p.beta
    lambda_values, penalty_beta, quadratic, trace = (
        p.lambda_values,
        p.penalty_beta,
        p.quadratic,
        p.trace,
    )
    beta_k, direction_stack, weights = p.beta_k, p.direction_stack, p.weights
    weighted_blocks, absolute_weighted_blocks = p.weighted_blocks, p.absolute_weighted_blocks
    first, first_certificate = p.first, p.first_certificate
    third_vector, certificate_leverage, leverage_trace = (
        p.third_vector,
        p.certificate_leverage,
        p.leverage_trace,
    )
    channels = weighted_blocks.shape[1]

    # H_k = lambda_k S_k + X' V_k X with V_k[i] = D3_i[eta_k[i]] = sum_q eta_k[i, q] D1_q[i].
    embedded = []
    for k, component in enumerate(estimated):
        matrix = np.zeros((width, width), dtype=np.float64)
        block = component.coefficient_slice
        matrix[block, block] = lambda_values[k] * component.penalty
        embedded.append(matrix)
    packed = np.zeros((n_observations, count, channels), dtype=np.float64)
    for q in range(k_parameters):
        packed += direction_stack[:, q, :, None] * first[q][:, None, :]
    crosses = _cross_all(matrices, slices, packed, width)
    del packed
    hessians = [embedded[k] + crosses[k] for k in range(count)]
    products = [inverse @ hessians[k] for k in range(count)]

    # The scalar certificate ingredients, in the H metric so the row leverage
    # survives.  With Y_i = X_i H^-1/2 (||Y_i||_F^2 = tr(G_i)) and C_k[i] the
    # certificate of V_k[i]:
    #   e_k = sum_i tr(G_i) <C_k[i], w>  >=  ||H^-1/2 delta H_k H^-1/2||_2,
    # since delta H_k = sum_i X_i' delta V_k[i] X_i and ||Y_i' M Y_i|| <= ||Y_i||_F^2 ||M||_F.
    certificate_weight = leverage_trace[:, None] * np.column_stack(
        [first_certificate[q] @ weights for q in range(k_parameters)]
    )
    hessian_error_norm = np.zeros(count, dtype=np.float64)
    for q in range(k_parameters):
        hessian_error_norm += np.abs(direction_stack[:, q, :]).T @ certificate_weight[:, q]
    # ||H^-1/2 H_k H^-1/2||_2 from the symmetric square root of the pseudo-inverse:
    # tr(H^-1 H_l H^-1 delta H_k) = sum_i <Y_i (H^-1/2 H_l H^-1/2) Y_i', delta V_k[i]>
    # is bounded by that norm times e_k.
    values, vectors = np.linalg.eigh(inverse)
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    relative_norm = np.array(
        [
            float(np.max(np.abs(np.linalg.eigvalsh(root @ matrix @ root)), initial=0.0))
            for matrix in hessians
        ]
    )
    # eta_kl's drift through H_l's error: ||X_i delta beta_kl|| <= sqrt(tr G_i)
    # ||H^1/2 delta beta_kl|| <= sqrt(tr G_i) e_l ||H^1/2 beta_k||, and
    # ||H^1/2 beta_k||^2 = -lambda_k (S_k beta) . beta_k; against |D1| and |G|
    # that is the scalar times one row sum.
    first_norm = np.sqrt(sum(first[q] * first[q] for q in range(k_parameters)))
    drift_base = float(
        np.sum(
            np.sqrt(leverage_trace)[:, None] * absolute_weighted_blocks * first_norm,
            dtype=np.float64,
        )
    )
    direction_h_norm = np.sqrt(
        np.maximum(
            [-lambda_values[k] * float(penalty_beta[k] @ beta_k[k]) for k in range(count)], 0.0
        )
    )

    # The value and certificate contractions of the fourth-derivative term for
    # every pair at once: sum_i <V_kl[i], G_i>_w splits into
    #   sum_q sum_i eta_kl[i, q] P_q[i]  +  sum_{q, r} sum_i eta_k[i, q] Q_qr[i] eta_l[i, r]
    # with P_q[i] = <D1_q[i], G_i>_w and Q_qr[i] = <D2_qr[i], G_i>_w; the first is
    # beta_kl . v for v = sum_q X_q' P_q, the second K^2 GEMMs of the (n, m)
    # direction matrices; the certificate follows the same split with absolute
    # values, so no pair touches the rows on its own.
    fourth_trace = np.zeros((count, count), dtype=np.float64)
    fourth_certificate = np.zeros((count, count), dtype=np.float64)
    absolute_directions = np.abs(direction_stack)
    for q in range(k_parameters):
        for r in range(k_parameters):
            pair = (q, r) if q <= r else (r, q)
            second_leverage = np.sum(weighted_blocks * second[pair], axis=1)
            second_certificate_leverage = np.sum(
                absolute_weighted_blocks * second_certificate[pair], axis=1
            )
            fourth_trace += (direction_stack[:, q, :] * second_leverage[:, None]).T @ (
                direction_stack[:, r, :]
            )
            fourth_certificate += (
                absolute_directions[:, q, :] * second_certificate_leverage[:, None]
            ).T @ absolute_directions[:, r, :]
    flat = np.stack([product.ravel() for product in products])
    flat_transposed = np.stack([product.T.ravel() for product in products])
    product_trace = flat_transposed @ flat.T

    # beta_kl for k <= ell (symmetric in the pair: one solve each, mirrored), then
    # the exact contraction of |eta_kl| against C1 in row chunks.
    pairs = [(k, ell) for k in range(count) for ell in range(k, count)]
    right_hand_sides = np.empty((width, len(pairs)), dtype=np.float64)
    for index, (k, ell) in enumerate(pairs):
        component = estimated[k]
        right_hand_side = hessians[ell] @ beta_k[k] + lambda_values[k] * _penalty_apply(
            component, beta_k[ell]
        )
        if k == ell:
            right_hand_side = right_hand_side + lambda_values[k] * penalty_beta[k]
        right_hand_sides[:, index] = right_hand_side
    pair_directions = -(inverse @ right_hand_sides)
    third_values = third_vector @ pair_directions
    third_certificate = np.zeros(len(pairs), dtype=np.float64)
    for start in range(0, n_observations, _ROW_CHUNK):
        rows = slice(start, min(start + _ROW_CHUNK, n_observations))
        for q in range(k_parameters):
            third_certificate += (
                np.abs(matrices[q][rows] @ pair_directions[slices[q]]).T
                @ certificate_leverage[rows, q]
            )
    hessian = np.empty((count, count))
    hessian_certificate = np.empty((count, count))
    for index, (k, ell) in enumerate(pairs):
        component = estimated[k]
        block = component.coefficient_slice
        same = k == ell
        value = (
            (0.5 * quadratic[k] if same else 0.0)
            + lambda_values[k] * float(beta[block] @ (component.penalty @ beta_k[ell][block]))
            + 0.5
            * ((trace[k] if same else 0.0) + float(third_values[index]) + fourth_trace[k, ell])
            - 0.5 * product_trace[ell, k]
            - 0.5 * p.rank_hessian.get((component.name, estimated[ell].name), 0.0)
        )
        # eta_kl inherits H_l's finite-difference error through beta_kl.
        drift = hessian_error_norm[ell] * direction_h_norm[k] * drift_base
        certificate = 0.5 * (
            float(third_certificate[index]) + fourth_certificate[k, ell] + drift
        ) + 0.5 * (
            relative_norm[ell] * hessian_error_norm[k] + relative_norm[k] * hessian_error_norm[ell]
        )
        hessian[k, ell] = hessian[ell, k] = value
        hessian_certificate[k, ell] = hessian_certificate[ell, k] = certificate
    base = p.result
    return LamlDerivatives(
        names=base.names,
        gradient=base.gradient,
        hessian=hessian,
        gradient_certificate=base.gradient_certificate,
        hessian_certificate=hessian_certificate,
        fs_gradient=base.fs_gradient,
        penalty_hessian=base.penalty_hessian,
        third_derivative_authority=base.third_derivative_authority,
        evaluations=evaluations,
        provenance=base.provenance,
    )


def laml_derivatives(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    lambdas: Mapping[str, float],
    fit: DenseSolverResult,
    dense_matrices: Sequence[NDArray],
    step: float = DEFAULT_STEP,
    want_hessian: bool = True,
    reuse: LamlDerivativeWorkspace | None = None,
) -> LamlDerivatives:
    """Exact gradient (and Hessian) of the negative LAML in log lambda at ``fit``.

    ``fit`` must be the converged penalised-likelihood fit at ``lambdas`` whose
    published curvature defines ``joint_laplace_objective``; ``dense_matrices``
    are the layout's dense predictor matrices (the reuse session's).  With a
    ``reuse`` workspace, a gradient-only call stores its stencils and leverage
    blocks there and a later Hessian call at the same fit adds only the centre
    and mixed-pair evaluations.
    """
    if not isinstance(fit, DenseSolverResult):
        raise TypeError("fit must be a DenseSolverResult")
    if not fit.converged:
        raise LamlDerivativeError("derivatives require a converged coefficient fit")
    if isinstance(step, bool) or not isinstance(step, int | float) or not math.isfinite(step):
        raise ValueError("step must be a finite positive float")
    if step <= 0.0:
        raise ValueError("step must be a finite positive float")
    if not isinstance(want_hessian, bool):
        raise TypeError("want_hessian must be bool")
    if reuse is not None and not isinstance(reuse, LamlDerivativeWorkspace):
        raise TypeError("reuse must be a LamlDerivativeWorkspace")
    source = fit.terminal_curvature.actual_source
    if source != "observed":
        raise LamlDerivativeError(
            "derivatives require observed terminal curvature; the fit published "
            f"{source!r}, so its observed Hessian was materially indefinite"
        )
    gradient_pass: _GradientPass | None = None
    if reuse is not None and reuse._pass is not None:
        held = reuse._pass
        if held.matches(family, likelihood_plan, fit, lambdas, float(step), dense_matrices):
            gradient_pass = held
    reused = gradient_pass is not None
    if gradient_pass is None:
        gradient_pass = _gradient_pass(
            family,
            layout,
            y,
            likelihood_plan,
            lambdas=lambdas,
            fit=fit,
            dense_matrices=dense_matrices,
            step=float(step),
        )
        if reuse is not None and not want_hessian:
            reuse._pass = gradient_pass
    if not want_hessian:
        return gradient_pass.result
    return _hessian_pass(gradient_pass, reused=reused)


__all__ = [
    "LamlDerivativeError",
    "LamlDerivativeWorkspace",
    "LamlDerivatives",
    "laml_derivatives",
]
