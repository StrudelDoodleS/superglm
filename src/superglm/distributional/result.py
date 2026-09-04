"""Compatibility shim: the result types live in ``superglm.distributional.results``."""

# ruff: noqa: F401

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.results.endpoint_evidence import (
    ANALYTIC_DIRECTION_AUTHORITY,
    CHUNKED_EXECUTION_BACKEND_IDENTIFIER,
    DENSE_EXECUTION_BACKEND_IDENTIFIER,
    DIRECTION_AUTHORITIES,
    FINITE_DIFFERENCE_DIRECTION_AUTHORITY,
    JOINT_ANALYTIC_DIRECTION_AUTHORITY,
    JOINT_DIRECTION_AUTHORITIES,
    JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY,
    CoefficientCurvature,
    ConvergenceReason,
    EFSAccelerationOutcome,
    EFSConvergenceReason,
    EndpointAssessmentFailureReason,
    EndpointDirectionDecision,
    EndpointDirectionEvidence,
    ExecutionBackendIdentifier,
    JointEndpointDirectionEvidence,
    _finite_nonnegative,
    _finite_positive,
)
from superglm.distributional.results.fit import DistributionalFitResult, _frozen_array_mapping
from superglm.distributional.results.iteration import (
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    _maximum_relative_natural_parameter_change,
)
from superglm.distributional.results.smoothing import (
    DistributionalEFSResult,
    _revalidated_endpoint_directions,
)
from superglm.distributional.results.solver import (
    DenseSolverConfig,
    DenseSolverResult,
    SolverIteration,
    _assessment_coefficient_refit_bound,
    _assessment_complete_face_penalty_matches,
    _assessment_exact_face_objective,
    _assessment_face_geometry_matches,
    _assessment_finite_objective,
    _assessment_is_numerically_stationary,
    _assessment_penalty_direction_matches,
    _assessment_retained_kkt_ratio,
    _assessment_scalar_error_bound,
    _assessment_unpenalized_logdet_term,
    _dense_penalty_fingerprint,
    _frozen_endpoint_mapping,
    _frozen_float_mapping,
    _readonly_finite,
    _resolution_limited_decrement_is_within_objective_ulp,
    _validate_newton_decrement_certificate,
    _validate_resolution_limited_stationarity,
    validate_solver_likelihood_decomposition,
)
from superglm.distributional.smoothing.acceleration import AccelerationRefusalReason
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.solvers.rank import RankDecomposition, decompose_gram
