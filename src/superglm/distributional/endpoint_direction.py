"""Compatibility shim: this module lives in ``superglm.distributional.smoothing.endpoint_direction``."""

# ruff: noqa: F401

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.smoothing.endpoint_direction import (
    DEFAULT_STEP,
    FINITE_DIFFERENCE_AUTHORITY,
    FiniteDifferenceDirection,
    _central,
    _curvature_packed,
    _theta_from_eta,
    finite_difference_curvature_direction,
)
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.links import Link

__all__ = [
    "DEFAULT_STEP",
    "FINITE_DIFFERENCE_AUTHORITY",
    "FiniteDifferenceDirection",
    "finite_difference_curvature_direction",
]
