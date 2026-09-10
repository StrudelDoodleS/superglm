"""Fixed-target penalty geometry shared by endpoint objectives and result assessment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.reml.penalty_algebra import _PenaltyLogdetEvaluation, penalty_component_dense_matrix
from superglm.types import PenaltyComponent


class EndpointLaplaceError(ValueError):
    """Raised when a fit cannot prove the supplied endpoint provenance."""


def _finite_penalty_evaluation_from_components(
    *, components: tuple[PenaltyComponent, ...], resolved: Mapping[str, float], face: PenaltyFace
) -> _PenaltyLogdetEvaluation:
    """Evaluate inputs whose names, weights and face were authenticated by the caller."""
    from superglm.reml.multi_penalty import (
        _evaluate_penalty_summary,
        _matmul_enclosed,
    )
    from superglm.reml.penalty_algebra import _context_geometry
    from superglm.reml.penalty_support import (
        PenaltyNumericalError,
        _component_root,
        _penalty_support_from_roots,
    )

    finite_components = tuple(
        component for component in components if component.name not in face.component_names
    )
    groups = _projected_penalty_group_indices(finite_components)
    if any(
        _context_geometry([finite_components[index] for index in indices]) is not None
        for indices in groups
    ):
        return _retained_finite_penalty_evaluation(
            finite_components, groups=groups, resolved=resolved, face=face
        )
    names = tuple(component.name for component in finite_components)
    gradient = dict.fromkeys(names, 0.0)
    gradient_error = dict.fromkeys(names, 0.0)
    hessian: dict[tuple[str, str], float] = {}
    hessian_error: dict[tuple[str, str], float] = {}
    rank = 0
    log_terms = []
    log_errors = []
    unit = np.finfo(float).eps / 2
    for indices in groups:
        components = tuple(finite_components[index] for index in indices)
        roots = []
        resolution_flags = []
        projection_bounds = []
        for component in components:
            local = np.asarray(penalty_component_dense_matrix(component), dtype=float)
            try:
                np.asarray_chkfinite(local)
            except ValueError as evidence:
                raise EndpointLaplaceError(
                    f"retained penalty {component.name!r} contains non-finite local values"
                ) from evidence
            try:
                root, resolution_limited, _reconstruction = _component_root(local)
            except np.linalg.LinAlgError as exc:
                raise EndpointLaplaceError(
                    f"retained penalty {component.name!r} could not be decomposed"
                ) from exc
            except ValueError as exc:
                if "positive semidefinite" not in str(exc) and "symmetric" not in str(exc):
                    raise
                raise EndpointLaplaceError(
                    f"retained penalty {component.name!r} has invalid PSD geometry: {exc}"
                ) from exc
            projection = face.null_basis[component.group_sl, :]
            try:
                projected, bound = _matmul_enclosed(root, projection)
            except PenaltyNumericalError as evidence:
                raise EndpointLaplaceError(
                    f"projection of retained penalty {component.name!r} could not be certified"
                ) from evidence
            # The root action and its bound share the kernel's precision,
            # nonnegative-dot enclosure, cast error and subnormal allowance.
            roots.append(projected)
            projection_bounds.append(bound)
            resolution_flags.append(resolution_limited)
        group_values = np.asarray([resolved[component.name] for component in components])
        try:
            support = _penalty_support_from_roots(
                roots, resolution_limited=resolution_flags, input_error_bounds=projection_bounds
            )
            decomposition = _evaluate_penalty_summary(support, group_values)
        except PenaltyNumericalError as exc:
            raise EndpointLaplaceError(
                "finite penalty geometry could not be certified on this face"
            ) from exc
        group_rank = decomposition.rank
        group_log_pdet = decomposition.logdet_s_plus
        if (
            isinstance(group_rank, bool)
            or not isinstance(group_rank, Integral)
            or group_rank < 0
            or group_rank > face.reduced_width
        ):
            raise ValueError("projected penalty decomposition returned an invalid rank")
        if not isinstance(group_log_pdet, Real) or not np.isfinite(float(group_log_pdet)):
            raise ValueError(
                "projected penalty decomposition returned a non-finite log determinant"
            )
        if group_rank == 0 and float(group_log_pdet) != 0.0:
            raise ValueError(
                "projected penalty decomposition returned a nonzero log determinant for zero rank"
            )
        certificate = decomposition._certificate
        if certificate is None:
            raise EndpointLaplaceError("projected penalty decomposition has no arithmetic evidence")
        group_gradient = decomposition.gradient
        group_hessian = decomposition.hessian
        rank += int(group_rank)
        log_terms.append(float(group_log_pdet))
        log_errors.append(certificate.logdet_error)
        for i, component_i in enumerate(components):
            gradient[component_i.name] = float(group_gradient[i])
            gradient_error[component_i.name] = float(certificate.gradient_error[i])
            for j, component_j in enumerate(components):
                key = (component_i.name, component_j.name)
                hessian[key] = float(group_hessian[i, j])
                hessian_error[key] = float(certificate.hessian_error[i, j])
    if rank > face.reduced_width:
        raise ValueError("projected penalty decomposition returned an invalid rank")
    logdet = math.fsum(log_terms)
    if not np.isfinite(logdet):
        raise ValueError("projected penalty decomposition returned a non-finite log determinant")
    return _PenaltyLogdetEvaluation(
        rank=rank,
        logdet=logdet,
        gradient=gradient,
        hessian=hessian,
        logdet_error=math.fsum(log_errors) + 2 * unit * abs(logdet),
        gradient_error=gradient_error,
        hessian_error=hessian_error,
    )


def _retained_finite_penalty_evaluation(
    components: tuple[PenaltyComponent, ...],
    *,
    groups: tuple[tuple[int, ...], ...],
    resolved: Mapping[str, float],
    face: PenaltyFace,
) -> _PenaltyLogdetEvaluation:
    """Keep every local target and bound the one common finite-block map.

    Complete retained families keep their original support, SSP map and error
    ledgers. A manual family selects its PSD roots and common support in its
    own coefficient block; that support's selected representatives and local
    projection evidence define its target. Both then use the same face map.
    Wholly manual layouts retain the legacy projected-root entry above.
    """
    from superglm.reml.multi_penalty import _evaluate_penalty_summary
    from superglm.reml.penalty_algebra import (
        _compute_penalty_logdet_evaluation,
        _context_geometry,
        _enclosed_bound_sum,
        _joint_near_isometry_volume_error,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support

    evaluations = []
    rows = []
    root_rows = 0
    try:
        for indices in groups:
            local = [components[index] for index in indices]
            geometry = _context_geometry(local)
            if geometry is not None:
                evaluation = _compute_penalty_logdet_evaluation(dict(resolved), local)
                root_rows += geometry.repeat * sum(
                    len(root) for root in geometry.get_support().component_roots
                )
            else:
                # No retained target exists for this manual family. Select it
                # once locally, keeping the support's own root/projection and
                # reconstruction evidence; never re-root an attached target.
                matrices = [penalty_component_dense_matrix(item) for item in local]
                support = _penalty_support(matrices)
                summary = _evaluate_penalty_summary(
                    support, np.array([resolved[item.name] for item in local])
                )
                certificate = summary._certificate
                root_rows += sum(len(root) for root in support.component_roots)
                evaluation = _PenaltyLogdetEvaluation(
                    rank=summary.rank,
                    logdet=summary.logdet_s_plus,
                    logdet_error=certificate.logdet_error,
                    gradient={
                        item.name: float(summary.gradient[i]) for i, item in enumerate(local)
                    },
                    gradient_error={
                        item.name: float(certificate.gradient_error[i])
                        for i, item in enumerate(local)
                    },
                    hessian={
                        (left.name, right.name): float(summary.hessian[i, j])
                        for i, left in enumerate(local)
                        for j, right in enumerate(local)
                    },
                    hessian_error={
                        (left.name, right.name): float(certificate.hessian_error[i, j])
                        for i, left in enumerate(local)
                        for j, right in enumerate(local)
                    },
                )
            evaluations.append(evaluation)
            rows.extend(range(local[0].group_sl.start, local[0].group_sl.stop))
        rank = sum(item.rank for item in evaluations)
        volume_error = _joint_near_isometry_volume_error(
            face.null_basis[rows],
            rank=rank,
            root_rows=root_rows,
            components=len(components),
        )
    except PenaltyNumericalError as evidence:
        raise EndpointLaplaceError(
            "finite penalty geometry could not be certified on this face"
        ) from evidence
    names = tuple(item.name for item in components)
    gradient = dict.fromkeys(names, 0.0)
    gradient_error = dict.fromkeys(names, 0.0)
    hessian = dict.fromkeys(((left, right) for left in names for right in names), 0.0)
    hessian_error = hessian.copy()
    for evaluation in evaluations:
        gradient.update(evaluation.gradient)
        gradient_error.update(evaluation.gradient_error)
        hessian.update(evaluation.hessian)
        hessian_error.update(evaluation.hessian_error)
    logdet = math.fsum(item.logdet for item in evaluations)
    if not math.isfinite(logdet):
        raise ValueError("projected penalty decomposition returned a non-finite log determinant")
    return _PenaltyLogdetEvaluation(
        rank=rank,
        logdet=logdet,
        gradient=gradient,
        hessian=hessian,
        logdet_error=float(
            _enclosed_bound_sum(
                volume_error,
                *(item.logdet_error for item in evaluations),
                np.finfo(float).eps * abs(logdet),
            )
        ),
        gradient_error=gradient_error,
        hessian_error=hessian_error,
    )


def _projected_penalty_group_indices(
    components: tuple[PenaltyComponent, ...],
) -> tuple[tuple[int, ...], ...]:
    """Keep projected components partitioned by their original coefficient block."""
    grouped: dict[str, list[int]] = {}
    group_blocks: dict[str, tuple[int, int, int]] = {}
    group_index_owners: dict[int, str] = {}
    for index, component in enumerate(components):
        block = component.group_sl
        group_index = component.group_index
        if (
            not isinstance(component.group_name, str)
            or not component.group_name
            or isinstance(group_index, bool)
            or not isinstance(group_index, Integral)
            or not isinstance(block, slice)
            or block.step not in (None, 1)
            or not isinstance(block.start, int)
            or not isinstance(block.stop, int)
            or block.start < 0
            or block.stop <= block.start
        ):
            raise EndpointLaplaceError(
                "retained penalty group metadata has invalid coefficient blocks"
            )
        identity = (int(group_index), block.start, block.stop)
        previous = group_blocks.get(component.group_name)
        if previous is not None and previous != identity:
            raise EndpointLaplaceError(
                "retained penalty group metadata has inconsistent coefficient blocks"
            )
        owner = group_index_owners.get(int(group_index))
        if owner is not None and owner != component.group_name:
            raise EndpointLaplaceError(
                "retained penalty group metadata has inconsistent coefficient blocks"
            )
        for other_name, (_, other_start, other_stop) in group_blocks.items():
            if other_name == component.group_name:
                continue
            if max(block.start, other_start) < min(block.stop, other_stop):
                raise EndpointLaplaceError(
                    "retained penalty group metadata has overlapping coefficient blocks"
                )
        group_blocks.setdefault(component.group_name, identity)
        group_index_owners.setdefault(int(group_index), component.group_name)
        grouped.setdefault(component.group_name, []).append(index)
    return tuple(tuple(indices) for indices in grouped.values())
