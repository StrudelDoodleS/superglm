"""Compatibility shim: this module lives in ``superglm.distributional.smoothing.face_efs``."""

# ruff: noqa: F401

from collections.abc import Mapping

import numpy as np

from superglm.distributional.layout import StackedLayout
from superglm.distributional.smoothing.endpoint_laml import (
    _projected_finite_penalty_inputs,
    _projected_penalty_group_indices,
)
from superglm.distributional.smoothing.face_efs import (
    _bounded_effective_rank,
    projected_component_states,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.reml.efs_update import EFSComponentState
from superglm.reml.multi_penalty import logdet_s_gradient, similarity_transform_logdet
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.types import LambdaPolicy

__all__ = ["projected_component_states"]
