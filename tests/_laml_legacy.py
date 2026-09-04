"""The LAML derivative assembly as it stood before the derivative-pass cost work.

A frozen copy of the previous implementation: two-direction polarisation for
every mixed second difference, one ``X' diag-blocks(V) X`` assembly per
component, and the entrywise absolute-valued certificate machinery. It remains
test-only as an independent parity oracle for the optimized production
assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.smoothing.derivatives import (
    LamlDerivativeError,
    _analytic_first_derivative,
    _leverage_blocks,
    _packed_weights,
    _penalty_apply,
    _predictor_directions,
    _rank_hessian,
    _stencil,
    _validated_matrices,
)
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    _curvature_packed,
    _first_difference,
    _second_difference,
    _unit_directions,
)
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import _component_states, _estimated_names
from superglm.distributional.solver.packing import packed_pairs


@dataclass(frozen=True)
class LegacyDerivatives:
    names: tuple[str, ...]
    gradient: NDArray[np.float64]
    hessian: NDArray[np.float64] | None
    gradient_certificate: NDArray[np.float64]
    hessian_certificate: NDArray[np.float64] | None
    fs_gradient: NDArray[np.float64]
    evaluations: int


def _legacy_axis_derivatives(rows, eta, step, *, want_second, analytic_first):
    n_observations, k_parameters = eta.shape
    first = []
    first_certificate = []
    second = {}
    second_certificate = {}
    evaluations = 0
    center = None
    if want_second:
        center = rows(eta)
        evaluations += 1
    for q in range(k_parameters):
        unit = np.zeros((n_observations, k_parameters), dtype=np.float64)
        unit[:, q] = 1.0
        stencil = None
        if analytic_first is None or want_second:
            stencil = _stencil(rows, eta, unit, step)
            evaluations += 4
        if analytic_first is not None:
            values = analytic_first(unit)
            certificate = np.zeros_like(values)
        else:
            values, certificate = _first_difference(stencil, step)
        first.append(values)
        first_certificate.append(certificate)
        if want_second:
            second[(q, q)], second_certificate[(q, q)] = _second_difference(center, stencil, step)
    if want_second:
        for q in range(k_parameters):
            for r in range(q + 1, k_parameters):
                polarised = []
                polarised_certificate = []
                for sign in (1.0, -1.0):
                    direction = np.zeros((n_observations, k_parameters), dtype=np.float64)
                    direction[:, q] = 1.0
                    direction[:, r] = sign
                    norms, unit = _unit_directions(direction)
                    stencil = _stencil(rows, eta, unit, step)
                    evaluations += 4
                    values, certificate = _second_difference(center, stencil, step)
                    scale = (norms * norms)[:, None]
                    polarised.append(values * scale)
                    polarised_certificate.append(certificate * scale)
                second[(q, r)] = (polarised[0] - polarised[1]) / 4.0
                second_certificate[(q, r)] = (
                    polarised_certificate[0] + polarised_certificate[1]
                ) / 4.0
    for values in (*first, *second.values()):
        if not np.all(np.isfinite(values)):
            raise LamlDerivativeError("a derivative of the packed row curvature is not finite")
    return first, first_certificate, second, second_certificate, evaluations


def legacy_cross(matrices, slices, packed, width, *, absolute=False):
    """``X^T diag-blocks(packed) X`` in the global layout, one component at a time."""
    result = np.zeros((width, width), dtype=np.float64)
    k_parameters = len(matrices)
    for channel, (left, right) in enumerate(packed_pairs(k_parameters)):
        left_matrix = np.abs(matrices[left]) if absolute else matrices[left]
        right_matrix = np.abs(matrices[right]) if absolute else matrices[right]
        block = left_matrix.T @ (packed[:, channel, None] * right_matrix)
        result[slices[left], slices[right]] += block
        if left != right:
            result[slices[right], slices[left]] += block.T
    return 0.5 * (result + result.T)


def legacy_laml_derivatives(
    family,
    layout,
    y,
    likelihood_plan,
    *,
    lambdas,
    fit,
    dense_matrices,
    step=DEFAULT_STEP,
    want_hessian=True,
):
    """Evaluate the frozen reference assembly without argument validation."""
    face = fit.coefficient_face
    components = (
        _component_states(layout, lambdas)
        if face is None
        else projected_component_states(layout=layout, lambdas=lambdas, face=face)
    )
    names = _estimated_names(components)
    estimated = tuple(component for component in components if component.name in set(names))
    rank_hessian = _rank_hessian(layout, lambdas, face)

    eta = np.asarray(fit.eta, dtype=np.float64)
    n_observations, k_parameters = eta.shape
    matrices = _validated_matrices(layout, dense_matrices, n_observations)
    slices = tuple(state.coefficient_slice for state in layout.predictors)
    links = tuple(state.link for state in layout.predictors)
    width = layout.n_coefficients
    beta = np.asarray(fit.coefficients, dtype=np.float64)
    inverse = np.asarray(fit.terminal_pseudo_inverse(), dtype=np.float64)

    def observed_rows(values):
        return _curvature_packed(family, y, values, links, likelihood_plan)

    analytic_first = _analytic_first_derivative(
        family, y, eta, links, likelihood_plan, n_observations, k_parameters
    )
    first, first_certificate, second, second_certificate, evaluations = _legacy_axis_derivatives(
        observed_rows,
        eta,
        float(step),
        want_second=want_hessian,
        analytic_first=analytic_first,
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
    eta_k = [_predictor_directions(matrices, slices, vector) for vector in beta_k]
    blocks = _leverage_blocks(matrices, slices, inverse)
    absolute_blocks = np.abs(blocks)
    weights = _packed_weights(k_parameters)

    def contract(rows_values, leverage):
        return float(np.sum(rows_values * leverage * weights, dtype=np.float64))

    def combine(direction, parts, *, absolute=False):
        total = np.zeros_like(parts[0])
        for q in range(k_parameters):
            column = np.abs(direction[:, q]) if absolute else direction[:, q]
            total += column[:, None] * parts[q]
        return total

    third = [combine(eta_k[k], first) for k in range(count)]
    third_certificate = [combine(eta_k[k], first_certificate, absolute=True) for k in range(count)]
    gradient = np.empty(count)
    fs_gradient = np.empty(count)
    gradient_certificate = np.empty(count)
    for k in range(count):
        fs_gradient[k] = 0.5 * quadratic[k] + 0.5 * trace[k] - 0.5 * ranks[k]
        gradient[k] = fs_gradient[k] + 0.5 * contract(third[k], blocks)
        gradient_certificate[k] = 0.5 * contract(third_certificate[k], absolute_blocks)
    if not want_hessian:
        return LegacyDerivatives(
            names=names,
            gradient=gradient,
            hessian=None,
            gradient_certificate=gradient_certificate,
            hessian_certificate=None,
            fs_gradient=fs_gradient,
            evaluations=evaluations,
        )

    embedded = []
    for k, component in enumerate(estimated):
        matrix = np.zeros((width, width), dtype=np.float64)
        block = component.coefficient_slice
        matrix[block, block] = lambda_values[k] * component.penalty
        embedded.append(matrix)
    hessians = [embedded[k] + legacy_cross(matrices, slices, third[k], width) for k in range(count)]
    hessian_error = [
        legacy_cross(matrices, slices, third_certificate[k], width, absolute=True)
        for k in range(count)
    ]
    products = [inverse @ hessians[k] for k in range(count)]
    sandwiches = [np.abs(products[k] @ inverse) for k in range(count)]
    absolute_leverage = [
        np.abs(matrix) @ np.abs(inverse[block, :])
        for matrix, block in zip(matrices, slices, strict=True)
    ]
    absolute_first = [np.abs(values) for values in first]
    hessian = np.empty((count, count))
    hessian_certificate = np.empty((count, count))
    for k, component in enumerate(estimated):
        block = component.coefficient_slice
        for ell in range(count):
            same = k == ell
            right_hand_side = hessians[ell] @ beta_k[k] + lambda_values[k] * _penalty_apply(
                component, beta_k[ell]
            )
            if same:
                right_hand_side = right_hand_side + lambda_values[k] * penalty_beta[k]
            beta_kl = -(inverse @ right_hand_side)
            eta_kl = _predictor_directions(matrices, slices, beta_kl)
            fourth = combine(eta_kl, first)
            fourth_certificate = combine(eta_kl, first_certificate, absolute=True)
            for q in range(k_parameters):
                for r in range(k_parameters):
                    pair = (q, r) if q <= r else (r, q)
                    product = eta_k[k][:, q] * eta_k[ell][:, r]
                    fourth += product[:, None] * second[pair]
                    fourth_certificate += np.abs(product)[:, None] * second_certificate[pair]
            bound = hessian_error[ell] @ np.abs(beta_k[k])
            for q in range(k_parameters):
                drift = absolute_leverage[q] @ bound
                fourth_certificate += drift[:, None] * absolute_first[q]
            value = (
                (0.5 * quadratic[k] if same else 0.0)
                + lambda_values[k] * float(beta[block] @ (component.penalty @ beta_k[ell][block]))
                + 0.5 * ((trace[k] if same else 0.0) + contract(fourth, blocks))
                - 0.5 * float(np.sum(products[ell].T * products[k], dtype=np.float64))
                - 0.5 * rank_hessian.get((component.name, estimated[ell].name), 0.0)
            )
            certificate = 0.5 * contract(fourth_certificate, absolute_blocks) + 0.5 * (
                float(np.sum(sandwiches[ell] * hessian_error[k], dtype=np.float64))
                + float(np.sum(sandwiches[k] * hessian_error[ell], dtype=np.float64))
            )
            hessian[k, ell] = value
            hessian_certificate[k, ell] = certificate
    symmetric = 0.5 * (hessian + hessian.T)
    asymmetry = 0.5 * np.abs(hessian - hessian.T)
    hessian_certificate = 0.5 * (hessian_certificate + hessian_certificate.T) + asymmetry
    return LegacyDerivatives(
        names=names,
        gradient=gradient,
        hessian=symmetric,
        gradient_certificate=gradient_certificate,
        hessian_certificate=hessian_certificate,
        fs_gradient=fs_gradient,
        evaluations=evaluations,
    )
