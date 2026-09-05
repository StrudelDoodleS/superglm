"""Compatibility shim: this module lives in ``superglm.distributional.smoothing.penalty_face``."""

# ruff: noqa: F401

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.distributional.layout import StackedLayout
from superglm.distributional.smoothing.penalty_face import (
    PenaltyFace,
    PenaltyFaceError,
    _component_slice,
    _declared_rank,
    _expanded_components,
    _readonly,
    _readonly_typed,
    _spectral_norm,
    build_penalty_face,
)
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.solvers.rank import RankDecomposition
from superglm.types import PenaltyComponent

__all__ = ["PenaltyFace", "PenaltyFaceError", "build_penalty_face"]
