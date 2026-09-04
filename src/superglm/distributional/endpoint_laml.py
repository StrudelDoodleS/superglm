"""Compatibility shim: this module lives in ``superglm.distributional.smoothing.endpoint_laml``."""

# ruff: noqa: F401

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    DistributionalFamily,
    FamilyLikelihoodPlan,
    PredictorCurvatureDirectionalFamily,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    ANALYTIC_DIRECTION_AUTHORITY,
    DIRECTION_AUTHORITIES,
    FINITE_DIFFERENCE_DIRECTION_AUTHORITY,
    DenseSolverResult,
    EndpointDirectionDecision,
    EndpointDirectionEvidence,
)
from superglm.distributional.smoothing.endpoint_direction import (
    FiniteDifferenceDirection,
    finite_difference_curvature_direction,
)
from superglm.distributional.smoothing.endpoint_laml import (
    EndpointLaplaceDerivative,
    EndpointLaplaceError,
    EndpointLaplaceEvaluation,
    ProjectedPenaltyLogDet,
    _predictor_direction,
    _projected_finite_penalty_inputs,
    _projected_penalty_group_indices,
    _selected_whitened_basis,
    _spectral_norm,
    _validate_reduced_terminal_provenance,
    _validated_complete_lambdas,
    evaluate_endpoint_laplace,
    evaluate_endpoint_laplace_derivative,
    projected_finite_penalty_logdet,
    resolve_endpoint_direction,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver.assembly import assemble_grouped_geometry
from superglm.reml.multi_penalty import similarity_transform_logdet
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.solvers.rank import RankDecomposition, decompose_gram
from superglm.types import PenaltyComponent
