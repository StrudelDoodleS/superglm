"""Saved-state classes of the retired SCOP shape-constrained engine.

SCOP fitted shape-constrained P-splines up to 0.41.0. A model it fitted keeps
these objects on its result (``scop_geometry``, ``scop_inference``), and
inference reads the stored ``SCOPInferenceInfo`` arrays, so the classes stay
for such a model to load, predict and summarize. Nothing builds them now: a
refit goes to the QP engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SCOPCurvatureSource = Literal["observed", "fisher", "mixed"]


@dataclass(frozen=True)
class SCOPJointGeometry:
    """One coherent latent Hessian/decomposition for SCOP LAML consumers."""

    centered_hessian: NDArray
    hessian_inverse: NDArray
    transformed_intercept_cross: NDArray
    sum_w: float
    log_det_H: float  # noqa: N815
    hessian_rank: int
    curvature_source: SCOPCurvatureSource
    transformed_mean_x: NDArray | None = None


@dataclass(frozen=True)
class SCOPInferenceInfo:
    """Pya--Wood post-fit inference in public mapped-coefficient coordinates."""

    coefficient_inverse: NDArray
    augmented_inverse: NDArray
    intercept_edf: float
    feature_edf: NDArray
    feature_edf1: NDArray
    group_edf: Mapping[str, float]
    curvature_source: SCOPCurvatureSource

    @property
    def total_edf(self) -> float:
        return self.intercept_edf + float(np.sum(self.feature_edf))
