"""Compatibility shim: this module lives in ``superglm.distributional.smoothing.acceleration``."""

# ruff: noqa: F401

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.smoothing.acceleration import (
    AccelerationRefusalReason,
    MultisecantDecision,
    MultisecantProposal,
    WindowedTypeIIAnderson,
    _AcceptedPair,
    _common_scaled_step,
    _model_reduction_bound,
    _nonnegative_product,
    _nonnegative_sum,
    _NumericalProposalError,
    _provenance_equal,
    _readonly_copy,
    _scaled_norm,
    _truncated_svd_solution,
    _validated_float_vector,
    _validated_provenance,
)
