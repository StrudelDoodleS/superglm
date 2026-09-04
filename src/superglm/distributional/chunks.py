"""Compatibility shim: this module lives in ``superglm.distributional.solver.chunks``."""

# ruff: noqa: F401

import operator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    LikelihoodPlanValidatingFamily,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver.assembly import DenseJointGeometry, GroupedGeometryAccumulator
from superglm.distributional.solver.chunks import (
    AUTO_CHUNK_MEMORY_BYTES,
    AUTO_CHUNK_SELECTOR,
    ChunkedLikelihoodSums,
    ChunkSize,
    CurvatureSource,
    LikelihoodChunk,
    RowChunk,
    _immutable_response,
    _positive_integer,
    _predictor_chunk,
    _theta_chunk,
    _validate_bound_likelihood,
    _validate_chunk_inputs,
    _validated_coefficients,
    assemble_chunked_geometry,
    evaluate_chunked_log_likelihood,
    iter_likelihood_chunks,
    iter_row_chunks,
    materialize_terminal_predictions,
    maximum_chunked_predictor_change,
    resolve_chunk_size,
)
from superglm.distributional.solver.derivatives import (
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)

__all__ = [
    "AUTO_CHUNK_MEMORY_BYTES",
    "ChunkSize",
    "ChunkedLikelihoodSums",
    "LikelihoodChunk",
    "RowChunk",
    "assemble_chunked_geometry",
    "evaluate_chunked_log_likelihood",
    "iter_likelihood_chunks",
    "iter_row_chunks",
    "materialize_terminal_predictions",
    "maximum_chunked_predictor_change",
    "resolve_chunk_size",
]
