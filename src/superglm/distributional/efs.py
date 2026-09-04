"""Compatibility shim: the EFS implementation lives in ``superglm.distributional.smoothing``."""

# ruff: noqa: F401

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    ANALYTIC_DIRECTION_AUTHORITY,
    JOINT_ANALYTIC_DIRECTION_AUTHORITY,
    JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY,
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    DistributionalEFSResult,
    EFSConvergenceReason,
    EndpointAssessmentFailureReason,
    JointEndpointDirectionEvidence,
    _assessment_is_numerically_stationary,
    _dense_penalty_fingerprint,
    _maximum_relative_natural_parameter_change,
)
from superglm.distributional.smoothing.acceleration import WindowedTypeIIAnderson
from superglm.distributional.smoothing.authority import (
    _endpoint_candidate_refit_bound,
    _endpoint_objective_accumulation_bound,
    _endpoint_polish_provenance_matches,
    _endpoint_positive_dot,
    _endpoint_retained_curvature,
    _endpoint_retained_kkt_relative,
    _endpoint_retained_rank,
    _endpoint_retained_rank_provenance,
    _endpoint_retained_score,
    _endpoint_shared_provenance,
    _face_authority_config,
    _fit_endpoint_authority_stationary,
    _fit_fixed_state,
    _is_sole_cap_outside_face,
    _optional_penalty_face,
)
from superglm.distributional.smoothing.endpoint_laml import (
    EndpointDirectionEvidence,
    EndpointLaplaceError,
    evaluate_endpoint_laplace,
    evaluate_endpoint_laplace_derivative,
    resolve_endpoint_direction,
)
from superglm.distributional.smoothing.evidence import (
    _fresh_raw_evidence,
    _FreshRawEvidence,
    _lower_bound_pressure,
    _saturated_names,
)
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.faces import (
    _assess_joint_face_directions,
    _check_face_direction,
    _face_assessment_refusal_iteration,
    _face_retraction_iteration,
    _face_revalidation_iteration,
    _face_transition_iteration,
    _FaceDirectionAttempt,
    _FaceDirectionCheck,
    _FacePromotion,
    _FaceRecheck,
    _FaceRetraction,
    _indexed_direction,
    _indexed_joint_direction,
    _joint_direction_is_endpoint,
    _joint_face_revalidation_iteration,
    _JointFaceDirectionAttempt,
    _recheck_exact_face,
    _try_exact_face,
    _try_joint_exact_face,
)
from superglm.distributional.smoothing.loop import _efs_result, fit_distributional_efs
from superglm.distributional.smoothing.objective import (
    _complete_mapping,
    _component_states,
    _estimated_names,
    _laplace_objective,
    _maximum_step,
    _penalty_lambdas,
    _slices_overlap,
    _stable_isolated_gfs_update,
    initialize_distributional_lambdas,
    joint_laplace_objective,
)
from superglm.distributional.smoothing.penalty_face import (
    PenaltyFace,
    PenaltyFaceError,
    build_penalty_face,
)
from superglm.distributional.smoothing.proposals import (
    _accelerated_proposal,
    _acceleration_provenance,
    _named_steps,
    _ordered_log_lambdas,
    _ordered_steps,
    _scaled_proposal,
    _snap_to_bounds,
)
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import (
    DenseSolverError,
    _DenseObservedReuseSession,
    _fit_dense_fixed_lambda_score_only,
    fit_dense_fixed_lambda,
)
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.reml.efs_update import EFSComponentState, EFSUpdateResult, wood_fasiolo_update
from superglm.reml.penalty_algebra import (
    compute_logdet_s_derivatives,
    compute_logdet_s_plus,
    penalty_component_dense_matrix,
)
from superglm.solvers.rank import RankDecomposition
from superglm.types import LambdaPolicy

__all__ = [
    "DistributionalEFSConfig",
    "DistributionalEFSIteration",
    "DistributionalEFSResult",
    "fit_distributional_efs",
    "initialize_distributional_lambdas",
    "joint_laplace_objective",
]
