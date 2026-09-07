"""Compatibility shim: this module lives in ``superglm.distributional.solver.assembly``."""

# ruff: noqa: F401

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver.assembly import (
    DenseJointGeometry,
    GroupedGeometryAccumulator,
    _assemble_dense_geometry_from_matrices,
    _evaluate_predictors_from_matrices,
    _grouped_predictor_plans,
    _readonly,
    _validated_channels,
    _validated_coefficients,
    assemble_grouped_geometry,
    dense_predictor_matrices,
    evaluate_predictors_dense,
    validated_dense_penalty,
)
from superglm.distributional.solver.packing import packed_pairs
