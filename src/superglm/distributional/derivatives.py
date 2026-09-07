"""Compatibility shim: this module lives in ``superglm.distributional.solver.derivatives``."""

# ruff: noqa: F401

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import NaturalLikelihoodEvaluation
from superglm.distributional.solver.derivatives import (
    PredictorLikelihoodEvaluation,
    _inverse_link_derivatives,
    _readonly_float_array,
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.solver.packing import packed_pairs
from superglm.links import Link
