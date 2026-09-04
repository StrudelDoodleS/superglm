"""Compatibility shim: this module lives in ``superglm.distributional.solver.curvature``."""

# ruff: noqa: F401

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import CurvatureKind
from superglm.distributional.solver.curvature import (
    CurvatureDecision,
    CurvaturePolicyError,
    CurvaturePolicyState,
    RepeatedCurvatureIndefinitenessError,
    _analyze_curvature,
    _CurvatureAnalysis,
    _telemetry,
    resolve_curvature,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.solvers.rank import SHARED_RANK_POLICY, RankDecomposition, RankPolicy, decompose_gram
