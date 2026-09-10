"""EFS penalty geometry after restricting coefficients to an exact face."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from superglm.distributional.layout import StackedLayout
from superglm.distributional.smoothing.endpoint_laml import (
    _projected_finite_penalty_evaluation,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.reml.efs_update import EFSComponentState
from superglm.reml.penalty_algebra import _PenaltyLogdetEvaluation, penalty_component_dense_matrix
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
    evaluation: _PenaltyLogdetEvaluation | None = None,
) -> tuple[EFSComponentState, ...]:
    """Build finite-component EFS states using ranks on the face."""
    if evaluation is None:
        evaluation = _projected_finite_penalty_evaluation(layout=layout, lambdas=lambdas, face=face)
    components = tuple(
        component for component in layout.penalties if component.name not in face.component_names
    )
    if not components:
        return ()
    return tuple(
        EFSComponentState(
            name=component.name,
            coefficient_slice=component.group_sl,
            penalty=penalty_component_dense_matrix(component),
            rank=_bounded_effective_rank(
                evaluation.gradient[component.name],
                width=component.group_sl.stop - component.group_sl.start,
                problem_width=face.reduced_width,
            ),
            lambda_value=float(lambdas[component.name]),
            policy=component.lambda_policy or LambdaPolicy.estimate(),
        )
        for component in components
    )


__all__ = ["projected_component_states"]
