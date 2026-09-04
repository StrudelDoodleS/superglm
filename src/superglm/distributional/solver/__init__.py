"""Compatibility shim: the coefficient solver lives in ``superglm.distributional.solver.solver``."""

# ruff: noqa: F401

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.solver.chunks as chunking
from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.result import (
    CHUNKED_EXECUTION_BACKEND_IDENTIFIER,
    DENSE_EXECUTION_BACKEND_IDENTIFIER,
    CoefficientCurvature,
    ConvergenceReason,
    DenseSolverConfig,
    DenseSolverResult,
    ExecutionBackendIdentifier,
    SolverIteration,
    _resolution_limited_decrement_is_within_objective_ulp,
    _validate_resolution_limited_stationarity,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver.assembly import (
    DenseJointGeometry,
    _assemble_dense_geometry_from_matrices,
    _evaluate_predictors_from_matrices,
    dense_predictor_matrices,
    validated_dense_penalty,
)
from superglm.distributional.solver.curvature import (
    CurvatureDecision,
    CurvaturePolicyState,
    resolve_curvature,
)
from superglm.distributional.solver.derivatives import (
    PredictorLikelihoodEvaluation,
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.solver.solver import (
    DenseSolverError,
    _AcceptedState,
    _cap_predictor_step,
    _DenseObservedReuseOwner,
    _DenseObservedReuseSession,
    _Direction,
    _evaluate_state,
    _evaluate_state_unmeasured,
    _fit_dense_fixed_lambda_core,
    _fit_dense_fixed_lambda_score_only,
    _geometry,
    _initial_coefficients,
    _measured_geometry,
    _optimization_score,
    _optimization_score_relative,
    _OptimizationRun,
    _policy_curvature,
    _readonly,
    _relative_score,
    _reuse_observed_initial_result,
    _run_iterations,
    _solve_coefficient_direction,
    _solve_direction,
    _SolverContext,
    _StopPolicy,
    _theta_from_eta,
    _validated_context,
    fit_dense_fixed_lambda,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.distributional.weights import UnsupportedLikelihoodContractError
from superglm.links import Link
from superglm.solvers.rank import RankDecomposition, decompose_gram, try_decompose_verified_spd_gram

__all__ = ["DenseSolverConfig", "DenseSolverError", "DenseSolverResult", "fit_dense_fixed_lambda"]
