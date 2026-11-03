"""Exact gradient and Hessian of the negative LAML in log smoothing parameters.

Wood, Pya and Säfken (2016) §3.1.3–3.2, with the third and fourth derivatives
of the row log-likelihood along the K parameter directions taken from the
family's analytic curvature directions where it has them, otherwise by finite
differences of the packed row curvature (``endpoint_direction``). ``F(rho)``
is ``joint_laplace_objective`` at the fit's penalised curvature
``H = -grad^2 l(beta_hat) + S``, through which the implicit derivative of the
mode goes; a fit whose terminal curvature fell back to Fisher is refused by
default. ``curvature="terminal"`` differentiates the accepted substitute
instead (expected information, or each row block projected onto the
semidefinite cone; Wood, Pya and Säfken 2016 §3.3.1, Wood and Fasiolo 2017 §3),
a first-order approximation to the true derivative.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.kernels._curvature_direction as curvature_kernels
from superglm.distributional._row_design import (
    BoundedPredictorMatrix,
    PredictorMatrix,
    matrix_row_chunk,
)
from superglm.distributional.family import (
    CertifiedCurvatureDirectionalFamily,
    CertifiedCurvatureSecondDirectionalFamily,
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    PredictorCurvatureDirectionalFamily,
    PredictorCurvatureSecondDirectionalFamily,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.results.endpoint_evidence import ANALYTIC_DIRECTION_AUTHORITY
from superglm.distributional.results.solver import DenseSolverResult
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    FINITE_DIFFERENCE_AUTHORITY,
    _curvature_packed,
    _first_difference,
    _second_difference,
    _theta_from_eta,
    _unit_directions,
    analytic_curvature_direction,
    analytic_curvature_second_direction,
)
from superglm.distributional.smoothing.endpoint_laml import (
    EndpointLaplaceError,
    _projected_finite_penalty_evaluation,
)
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import _component_states, _estimated_names
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver.derivatives import (
    nearest_semidefinite_curvature,
    transform_natural_information,
)
from superglm.distributional.solver.packing import packed_pairs
from superglm.distributional.weights import UnsupportedLikelihoodContractError
from superglm.reml.efs_update import EFSComponentState
from superglm.reml.penalty_algebra import (
    _compute_penalty_logdet_evaluation,
    _PenaltyLogdetEvaluation,
)

_PackedRows = Callable[[NDArray[np.float64]], NDArray[np.float64]]
_Stencil = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
#: Rows per chunk of the row-chunked assemblies (scratch of ``chunk * m * P_r`` doubles).
_ROW_CHUNK = 4096
#: Which terminal curvature ``laml_derivatives`` differentiates: always the
#: observed one, or whatever the fit accepted (see the module docstring).
LamlCurvature = Literal["observed", "terminal"]
#: Accepted terminal curvatures other than the observed one whose rows can be
#: re-evaluated at a perturbed predictor state.
_SUBSTITUTE_SOURCES = ("fisher", "nearest_semidefinite")


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

    ``hessian`` and ``hessian_certificate`` are ``None`` for a gradient-only
    call. ``fs_gradient`` holds the three Fellner–Schall terms alone.
    ``penalty_hessian`` drops the likelihood's third and fourth derivatives
    (exact for a Gaussian identity link); a gradient-only pass judges its stop
    with it. ``evaluations`` counts this call's packed-curvature evaluations.
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
    dense_matrices: Sequence[PredictorMatrix],
    n_observations: int,
) -> tuple[PredictorMatrix, ...]:
    matrices = tuple(
        matrix
        if isinstance(matrix, BoundedPredictorMatrix)
        else np.asarray(matrix, dtype=np.float64)
        for matrix in dense_matrices
    )
    if len(matrices) != len(layout.predictors):
        raise ValueError("one dense predictor matrix per predictor is required")
    for state, matrix in zip(layout.predictors, matrices, strict=True):
        width = state.coefficient_slice.stop - state.coefficient_slice.start
        if matrix.shape != (n_observations, width) or (
            not isinstance(matrix, BoundedPredictorMatrix) and not np.all(np.isfinite(matrix))
        ):
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
    analytic_first: _AnalyticFirst | None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], list[_Stencil | None], int]:
    """First derivatives of the packed rows along the parameter axes.

    Returns ``(D1, C1, stencils, evaluations)``; ``stencils[q]`` is the
    four-point axis stencil the second-derivative stage reuses (``None`` for
    an analytic derivative, whose certificate is the family's).
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
            values, certificate = analytic_first.axes[q]
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

    Returns ``(D2, C2, evaluations)`` for pairs ``q <= r``. Diagonal pairs use
    the five-point axis stencil; a mixed pair uses polarisation
    ``(D[e_q + e_r] - D[e_q] - D[e_r]) / 2`` on one extra stencil, with half
    the three certificates' sum: ``1 + 4K + 4K(K - 1)/2`` evaluations in all.
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


def _cross_all(
    matrices: tuple[PredictorMatrix, ...],
    slices: tuple[slice, ...],
    packed: NDArray[np.float64],
    width: int,
    *,
    chunk: int = _ROW_CHUNK,
) -> NDArray[np.float64]:
    """``X^T diag-blocks(packed[:, k, :]) X`` for every ``k`` at once, as ``(m, P, P)``.

    ``packed`` is ``(n, m, K(K+1)/2)``. Each channel is one GEMM per row chunk,
    with at most ``chunk * m * P_r`` doubles of scratch.
    """
    n_observations, count, channels = packed.shape
    pairs = packed_pairs(len(matrices))
    if channels != len(pairs):
        raise ValueError("packed weights must carry one channel per parameter pair")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    chunk = matrix_row_chunk(matrices, chunk)
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
    matrices: tuple[PredictorMatrix, ...],
    slices: tuple[slice, ...],
    inverse: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``G_i = X_i H^-1 X_i^T`` for every row, packed ``(n, K(K+1)/2)``."""
    k_parameters = len(matrices)
    pairs = packed_pairs(k_parameters)
    blocks = np.empty((matrices[0].shape[0], len(pairs)), dtype=np.float64)
    chunk = matrix_row_chunk(matrices, _ROW_CHUNK)
    for start in range(0, len(blocks), chunk):
        rows = slice(start, min(start + chunk, len(blocks)))
        local = tuple(matrix[rows] for matrix in matrices)
        for channel, (left, right) in enumerate(pairs):
            product = local[left] @ inverse[slices[left], slices[right]]
            blocks[rows, channel] = np.einsum("ij,ij->i", product, local[right])
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
    *,
    evaluation: _PenaltyLogdetEvaluation | None = None,
) -> dict[tuple[str, str], float]:
    """``d^2 log|S_lambda|_+ / d rho_i d rho_j`` for every pair in one penalty group."""
    if evaluation is None:
        evaluation = (
            _compute_penalty_logdet_evaluation(dict(lambdas), list(layout.penalties))
            if face is None
            else _projected_finite_penalty_evaluation(layout=layout, lambdas=lambdas, face=face)
        )
    return evaluation.hessian


def _penalty_apply(
    component: EFSComponentState, vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``S_k v`` in full coordinates: the component's block applied to its slice."""
    result = np.zeros_like(vector)
    block = component.coefficient_slice
    result[block] = component.penalty @ vector[block]
    return result


_CertifiedRows = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class _AnalyticFirst:
    """A family's analytic derivatives along every parameter axis.

    ``axes[q]`` is ``(values, certificate)`` along ``e_q``: the slice
    ``-l[:, :, q]`` of the row's third-derivative tensor in the predictors.
    All axes are evaluated first, so one refused axis sends the whole pass to
    differences.
    """

    axes: tuple[_CertifiedRows, ...]


def _analytic_first_derivative(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray[np.float64],
    links: tuple,
    likelihood_plan: FamilyLikelihoodPlan,
    n_observations: int,
    k_parameters: int,
) -> _AnalyticFirst | None:
    """The family's analytic directional derivative of the observed curvature, if usable.

    ``None`` when the family does not implement the protocol, refuses these
    links (NB2 and GammaLS accept only their log links) or refuses a row on any
    axis (a two-piece row on its join); the caller then differences.  A family
    with ``stacked_curvature_directions`` takes the ``K`` axes stacked, so its
    state is evaluated once per pass rather than once per axis, over blocks of
    ``STACKED_ROW_CHUNK`` rows so that its ``K``-fold scratch is bounded by the
    block, not the row count.
    """
    if not isinstance(
        family, (PredictorCurvatureDirectionalFamily, CertifiedCurvatureDirectionalFamily)
    ):
        return None
    channels = k_parameters * (k_parameters + 1) // 2

    def derivative(
        y_rows: NDArray,
        eta_rows: NDArray[np.float64],
        unit: NDArray[np.float64],
        plan_rows: FamilyLikelihoodPlan,
    ) -> _CertifiedRows:
        analytic = analytic_curvature_direction(family, y_rows, eta_rows, unit, links, plan_rows)
        if analytic is None:
            raise UnsupportedLikelihoodContractError("the family refused these links")
        expected = (*unit.shape[:-1], channels)
        values, certificate = analytic
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise LamlDerivativeError("family returned an invalid curvature direction")
        if certificate is None:
            return values, np.zeros_like(values)
        if (
            certificate.shape != expected
            or not np.all(np.isfinite(certificate))
            or np.any(certificate < 0.0)
        ):
            raise LamlDerivativeError("family returned an invalid curvature-direction certificate")
        return values, certificate

    def axis_units(rows: int) -> NDArray[np.float64]:
        units = np.zeros((k_parameters, rows, k_parameters), dtype=np.float64)
        for q in range(k_parameters):
            units[q, :, q] = 1.0
        return units

    try:
        if getattr(family, "stacked_curvature_directions", False) is True:
            chunk = min(curvature_kernels.STACKED_ROW_CHUNK, n_observations)
            if chunk == n_observations:
                values, certificate = derivative(y, eta, axis_units(chunk), likelihood_plan)
            else:
                values = np.empty((k_parameters, n_observations, channels), dtype=np.float64)
                certificate = np.empty_like(values)
                units = axis_units(chunk)
                for start in range(0, n_observations, chunk):
                    stop = min(start + chunk, n_observations)
                    if stop - start != chunk:
                        units = axis_units(stop - start)
                    plan_rows = likelihood_plan.take(np.arange(start, stop, dtype=np.intp))
                    values[:, start:stop], certificate[:, start:stop] = derivative(
                        y[start:stop], eta[start:stop], units, plan_rows
                    )
            axes = [(values[q], certificate[q]) for q in range(k_parameters)]
        else:
            axes = []
            for q in range(k_parameters):
                unit = np.zeros((n_observations, k_parameters), dtype=np.float64)
                unit[:, q] = 1.0
                axes.append(derivative(y, eta, unit, likelihood_plan))
    except (
        TypeError,
        ValueError,
        FloatingPointError,
        OverflowError,
        UnsupportedLikelihoodContractError,
        LamlDerivativeError,
    ):
        return None
    return _AnalyticFirst(axes=tuple(axes))


def _analytic_second_derivatives(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray[np.float64],
    links: tuple,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    step: float,
) -> (
    tuple[
        dict[tuple[int, int], NDArray[np.float64]],
        dict[tuple[int, int], NDArray[np.float64]],
    ]
    | None
):
    """The family's analytic second derivatives of the curvature along every axis pair.

    One stacked call over the ``K(K+1)/2`` pairs; ``step`` is the differenced
    pass's step, used as a near-kink certificate's reach. Returns
    ``(D2, C2)`` like :func:`_second_derivatives`, or ``None`` (the caller
    then differences) when the family cannot supply them.
    """
    if not isinstance(
        family,
        (PredictorCurvatureSecondDirectionalFamily, CertifiedCurvatureSecondDirectionalFamily),
    ):
        return None
    n_observations, k_parameters = eta.shape
    pairs = [(q, r) for q in range(k_parameters) for r in range(q, k_parameters)]
    # Axis pairs are the same on every row, so they broadcast over the rows.
    first = np.zeros((len(pairs), 1, k_parameters), dtype=np.float64)
    second = np.zeros_like(first)
    for index, (q, r) in enumerate(pairs):
        first[index, 0, q] = 1.0
        second[index, 0, r] = 1.0
    try:
        analytic = analytic_curvature_second_direction(
            family, y, eta, first, second, links, likelihood_plan, step=step
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return None
    if analytic is None:
        return None
    values, certificate = analytic
    expected = (len(pairs), n_observations, k_parameters * (k_parameters + 1) // 2)
    if values.shape != expected or not np.all(np.isfinite(values)):
        return None
    if certificate is None:
        certificate = np.zeros_like(values)
    elif (
        certificate.shape != expected
        or not np.all(np.isfinite(certificate))
        or np.any(certificate < 0.0)
    ):
        return None
    return (
        {pair: values[index] for index, pair in enumerate(pairs)},
        {pair: certificate[index] for index, pair in enumerate(pairs)},
    )


def _substitute_rows(
    family: DistributionalFamily,
    y: NDArray,
    links: tuple,
    likelihood_plan: FamilyLikelihoodPlan,
    source: str,
) -> _PackedRows:
    """Packed rows of an accepted substitute curvature, as a function of ``eta``.

    The same rows the coefficient solver assembled at its terminal state:
    the expected information through the inverse links, or each observed row
    block projected onto the semidefinite cone.
    """
    if source == "nearest_semidefinite":

        def projected(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return nearest_semidefinite_curvature(
                _curvature_packed(family, y, values, links, likelihood_plan)
            )

        return projected
    fisher_family = cast(ExpectedInformationFamily, family)

    def information(values: NDArray[np.float64]) -> NDArray[np.float64]:
        natural = fisher_family.expected_information_natural(
            _theta_from_eta(values, links), likelihood_plan
        )
        return np.asarray(transform_natural_information(natural, values, links), dtype=np.float64)

    return information


@dataclass
class _GradientPass:
    """Everything one gradient pass computed, for a Hessian pass at the same fit.

    With ``G_i`` the row leverage blocks and ``w`` the packed weights:
    ``weighted_blocks`` is ``G_i w``, ``third_vector`` is ``sum_q X_q' P_q``
    (``P_q[i] = <D1_q[i], G_i>_w``), ``certificate_leverage[:, q]`` is
    ``<C1_q[i], |G_i|>_w`` and ``leverage_trace[i]`` is ``tr(G_i)``.
    """

    family: DistributionalFamily
    likelihood_plan: FamilyLikelihoodPlan
    fit: DenseSolverResult
    y: NDArray
    links: tuple
    lambdas: dict[str, float]
    step: float
    source_matrices: tuple[object, ...]
    estimated: tuple[EFSComponentState, ...]
    rank_hessian: dict[tuple[str, str], float]
    penalty_hessian_error: NDArray[np.float64]
    rows: _PackedRows
    # The rows are a substitute curvature, which no family derivative describes.
    substitute: bool
    eta: NDArray[np.float64]
    matrices: tuple[PredictorMatrix, ...]
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
        dense_matrices: Sequence[PredictorMatrix],
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

    Pass the same instance as ``reuse=`` to a gradient-only and then a Hessian
    call at the same fit, lambdas, step and matrices; anything else is
    recomputed. Held arrays are ``O(n K^2)``; ``clear`` releases them.
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
    dense_matrices: Sequence[PredictorMatrix],
    step: float,
) -> _GradientPass:
    from superglm.reml.penalty_support import PenaltyNumericalError

    source = fit.terminal_curvature.actual_source
    substitute = source != "observed"
    face = fit.coefficient_face
    try:
        penalty_evaluation = (
            _compute_penalty_logdet_evaluation(dict(lambdas), list(layout.penalties))
            if face is None
            else _projected_finite_penalty_evaluation(layout=layout, lambdas=lambdas, face=face)
        )
    except PenaltyNumericalError as exc:
        raise LamlDerivativeError("finite penalty geometry could not be certified") from exc
    except EndpointLaplaceError as exc:
        if not isinstance(exc.__cause__, PenaltyNumericalError):
            raise
        raise LamlDerivativeError(
            "finite penalty geometry could not be certified on the face"
        ) from exc
    components = (
        _component_states(layout, lambdas, evaluation=penalty_evaluation)
        if face is None
        else projected_component_states(
            layout=layout, lambdas=lambdas, face=face, evaluation=penalty_evaluation
        )
    )
    names = _estimated_names(components)
    if not names:
        raise LamlDerivativeError("no estimated smoothing component to differentiate")
    estimated = tuple(component for component in components if component.name in set(names))
    rank_hessian = _rank_hessian(layout, lambdas, face, evaluation=penalty_evaluation)
    penalty_hessian_error = np.array(
        [
            [penalty_evaluation.hessian_error.get((left, right), 0.0) for right in names]
            for left in names
        ]
    )

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

    curvature_rows = (
        _substitute_rows(family, y, links, likelihood_plan, source) if substitute else observed_rows
    )
    # A family's analytic directions differentiate its observed curvature only.
    analytic_first = (
        None
        if substitute
        else _analytic_first_derivative(
            family, y, eta, links, likelihood_plan, n_observations, k_parameters
        )
    )
    try:
        first, first_certificate, stencils, evaluations = _first_derivatives(
            curvature_rows, eta, step, analytic_first=analytic_first
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
    weights = _packed_weights(k_parameters)
    weighted_blocks = _leverage_blocks(matrices, slices, inverse) * weights
    absolute_weighted_blocks = np.abs(weighted_blocks)
    diagonal_channels = [
        channel for channel, (left, right) in enumerate(packed_pairs(k_parameters)) if left == right
    ]
    leverage_trace = np.sum(weighted_blocks[:, diagonal_channels], axis=1)
    # The third-derivative term of g_k is beta_k . v with v = sum_q X_q' P_q;
    # its certificate is sum_q sum_i |eta_k[i, q]| N_q[i], N_q = <C1_q, |G|>_w.
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
    gradient_certificate = np.zeros(count, dtype=np.float64)
    chunk = matrix_row_chunk(matrices, _ROW_CHUNK)
    for start in range(0, n_observations, chunk):
        rows = slice(start, min(start + chunk, n_observations))
        for q in range(k_parameters):
            matrix = matrices[q][rows]
            third_vector[slices[q]] += matrix.T @ first_leverage[rows, q]
            directions = matrix @ coefficient_directions[slices[q]]
            gradient_certificate += np.abs(directions).T @ certificate_leverage[rows, q]
    fs_gradient = 0.5 * quadratic + 0.5 * trace - 0.5 * ranks
    gradient = fs_gradient + 0.5 * (third_vector @ coefficient_directions)
    gradient_certificate *= 0.5
    gradient_certificate += 0.5 * np.asarray(
        [penalty_evaluation.gradient_error[name] for name in names]
    )
    # Every term of h_kl that needs no third or fourth likelihood derivative:
    #   1/2 delta_kl lambda_k (beta'S_k beta + tr(H^-1 S_k)) + lambda_k beta'S_k beta_l
    #   - 1/2 lambda_k lambda_l tr(H^-1 S_l H^-1 S_k) - 1/2 d^2 log|S|_+ / d rho_k d rho_l
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
        y=y,
        links=links,
        lambdas=dict(lambdas),
        step=step,
        source_matrices=tuple(dense_matrices),
        estimated=estimated,
        rank_hessian=rank_hessian,
        penalty_hessian_error=penalty_hessian_error,
        rows=curvature_rows,
        substitute=substitute,
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

    Fourth derivatives come from the family's analytic second curvature
    direction when available, else from second differences. The certificate
    is a norm bound in the ``H`` metric so row leverage survives (derivation
    in the comments below); every quantity is one row pass or ``O(m P^3)``.
    """
    p = gradient_pass
    analytic_second = (
        None
        if p.substitute
        else _analytic_second_derivatives(
            p.family, p.y, p.eta, p.links, p.likelihood_plan, step=p.step
        )
    )
    if analytic_second is not None:
        second, second_certificate = analytic_second
        evaluations = 0
    else:
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
    beta_k, weights = p.beta_k, p.weights
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
    # Predictor directions and derivative row weights depend on the number of
    # smoothing components. Keep both local to a bounded row chunk.
    crosses = np.zeros((count, width, width), dtype=np.float64)
    hessian_error_norm = np.zeros(count, dtype=np.float64)
    fourth_trace = np.zeros((count, count), dtype=np.float64)
    fourth_certificate = np.zeros((count, count), dtype=np.float64)
    chunk = matrix_row_chunk(matrices, _ROW_CHUNK)
    for start in range(0, n_observations, chunk):
        rows = slice(start, min(start + chunk, n_observations))
        local = tuple(matrix[rows] for matrix in matrices)
        directions = [local[q] @ p.coefficient_directions[slices[q]] for q in range(k_parameters)]
        absolute_directions = [np.abs(direction) for direction in directions]
        packed = np.zeros((rows.stop - rows.start, count, channels), dtype=np.float64)
        for q in range(k_parameters):
            packed += directions[q][:, :, None] * first[q][rows, None, :]
            certificate_weight = (first_certificate[q][rows] @ weights) * leverage_trace[rows]
            hessian_error_norm += absolute_directions[q].T @ certificate_weight
            for r in range(k_parameters):
                pair = (q, r) if q <= r else (r, q)
                second_leverage = np.sum(weighted_blocks[rows] * second[pair][rows], axis=1)
                second_certificate_leverage = np.sum(
                    absolute_weighted_blocks[rows] * second_certificate[pair][rows], axis=1
                )
                fourth_trace += (directions[q] * second_leverage[:, None]).T @ directions[r]
                fourth_certificate += (
                    absolute_directions[q] * second_certificate_leverage[:, None]
                ).T @ absolute_directions[r]
        crosses += _cross_all(local, slices, packed, width)
    hessians = [embedded[k] + crosses[k] for k in range(count)]
    products = [inverse @ hessians[k] for k in range(count)]

    # Certificate in the H metric (Y_i = X_i H^-1/2, ||Y_i||_F^2 = tr G_i):
    # e_k = sum_i tr(G_i) <C_k[i], w> >= ||H^-1/2 delta H_k H^-1/2||_2, and
    # |tr(H^-1 H_l H^-1 delta H_k)| <= ||H^-1/2 H_l H^-1/2||_2 e_k.
    values, vectors = np.linalg.eigh(inverse)
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    relative_norm = np.array(
        [
            float(np.max(np.abs(np.linalg.eigvalsh(root @ matrix @ root)), initial=0.0))
            for matrix in hessians
        ]
    )
    # eta_kl drift through H_l's error: ||X_i delta beta_kl|| <= sqrt(tr G_i) e_l
    # ||H^1/2 beta_k||, with ||H^1/2 beta_k||^2 = -lambda_k (S_k beta) . beta_k.
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

    # Fourth-derivative contractions for every pair at once: sum_i <V_kl[i], G_i>_w
    # = beta_kl . v (v = sum_q X_q' P_q) + K^2 GEMMs of eta_k Q_qr eta_l, with
    # Q_qr[i] = <D2_qr[i], G_i>_w; the certificate splits the same way in |.|.
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
    for start in range(0, n_observations, chunk):
        rows = slice(start, min(start + chunk, n_observations))
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
        certificate += 0.5 * p.penalty_hessian_error[k, ell]
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
    dense_matrices: Sequence[PredictorMatrix],
    step: float = DEFAULT_STEP,
    want_hessian: bool = True,
    reuse: LamlDerivativeWorkspace | None = None,
    curvature: LamlCurvature = "observed",
) -> LamlDerivatives:
    """Exact gradient (and Hessian) of the negative LAML in log lambda at ``fit``.

    ``fit`` must be the converged penalised-likelihood fit at ``lambdas`` whose
    published curvature defines ``joint_laplace_objective``; ``dense_matrices``
    are the layout's dense predictor matrices or bounded grouped row adapters.
    With a ``reuse`` workspace, a gradient-only call stores its stencils and
    leverage blocks there and a later Hessian call at the same fit adds only
    the centre and mixed-pair evaluations. Row-sized work is chunked.

    ``curvature="observed"`` refuses a fit whose terminal curvature fell back;
    ``curvature="terminal"`` differentiates the accepted substitute (see the
    module docstring).
    """
    if not isinstance(fit, DenseSolverResult):
        raise TypeError("fit must be a DenseSolverResult")
    if not fit.converged:
        raise LamlDerivativeError("derivatives require a converged coefficient fit")
    if fit.eta is None:
        raise LamlDerivativeError("derivatives require retained predictor rows")
    if isinstance(step, bool) or not isinstance(step, int | float) or not math.isfinite(step):
        raise ValueError("step must be a finite positive float")
    if step <= 0.0:
        raise ValueError("step must be a finite positive float")
    if not isinstance(want_hessian, bool):
        raise TypeError("want_hessian must be bool")
    if reuse is not None and not isinstance(reuse, LamlDerivativeWorkspace):
        raise TypeError("reuse must be a LamlDerivativeWorkspace")
    source = fit.terminal_curvature.actual_source
    if source != "observed" and not (curvature == "terminal" and source in _SUBSTITUTE_SOURCES):
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
