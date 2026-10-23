"""Summary of the projected finite-penalty evaluation used by the endpoint tests."""

from __future__ import annotations

from collections.abc import Mapping

from superglm.distributional.layout import StackedLayout
from superglm.distributional.smoothing.endpoint_laml import (
    ProjectedPenaltyLogDet,
    _projected_finite_penalty_evaluation,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace


def projected_finite_penalty_logdet(
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace,
) -> ProjectedPenaltyLogDet:
    """Return ``log|sum(lambda_j Q.T S_j Q)|+`` off the selected face."""
    evaluation = _projected_finite_penalty_evaluation(layout=layout, lambdas=lambdas, face=face)
    return ProjectedPenaltyLogDet(
        component_names=tuple(evaluation.gradient), rank=evaluation.rank, log_pdet=evaluation.logdet
    )
