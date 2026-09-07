"""EFS penalty geometry after restricting coefficients to an exact face."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from superglm.distributional.layout import StackedLayout
from superglm.distributional.smoothing.endpoint_laml import (
    _projected_finite_penalty_inputs,
    _projected_penalty_group_indices,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.reml.efs_update import EFSComponentState
from superglm.reml.multi_penalty import logdet_s_gradient, similarity_transform_logdet
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.types import LambdaPolicy


def _bounded_effective_rank(value: float, *, width: int, problem_width: int) -> float:
    rank = float(value)
    tolerance = (
        256.0
        * max(width, problem_width, 1)
        * np.finfo(np.float64).eps
        * max(abs(rank), float(width), 1.0)
    )
    if not np.isfinite(rank) or rank < -tolerance or rank > width + tolerance:
        raise ValueError("projected effective rank lies outside its coefficient block")
    return float(np.clip(rank, 0.0, width))


def projected_component_states(
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace,
) -> tuple[EFSComponentState, ...]:
    """Build finite-component EFS states using ranks on the face."""
    components, projected, values = _projected_finite_penalty_inputs(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )
    if not components:
        return ()
    effective_ranks = [0.0] * len(components)
    for indices in _projected_penalty_group_indices(components):
        group_projected = [projected[index] for index in indices]
        group_values = values[list(indices)]
        decomposition = similarity_transform_logdet(group_projected, group_values)
        group_ranks = logdet_s_gradient(decomposition, group_projected, group_values)
        for index, effective_rank in zip(indices, group_ranks, strict=True):
            effective_ranks[index] = float(effective_rank)
    return tuple(
        EFSComponentState(
            name=component.name,
            coefficient_slice=component.group_sl,
            penalty=penalty_component_dense_matrix(component),
            rank=_bounded_effective_rank(
                effective_rank,
                width=component.group_sl.stop - component.group_sl.start,
                problem_width=face.reduced_width,
            ),
            lambda_value=float(lambdas[component.name]),
            policy=component.lambda_policy or LambdaPolicy.estimate(),
        )
        for component, effective_rank in zip(components, effective_ranks, strict=True)
    )


__all__ = ["projected_component_states"]
