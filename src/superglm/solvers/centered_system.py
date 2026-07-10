"""Stable intercept-profiled systems shared by fitting and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import centered_gram_rhs
from superglm.group_matrix import DesignMatrix


def _freeze(values: NDArray) -> NDArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CenteredSystem:
    """Complete weighted system after profiling the intercept."""

    sum_w: float
    mean_x: NDArray
    mean_z: float
    data_gram: NDArray
    rhs: NDArray
    penalty: NDArray
    hessian: NDArray


def build_centered_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
    penalty: NDArray,
) -> CenteredSystem:
    """Build a stably centered data Gram, RHS, and penalized Hessian."""
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    z_off = np.asarray(z_off, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    if W.shape != (n,) or z_off.shape != (n,):
        raise ValueError("W and z_off must match the design row count")
    if penalty.shape != (p, p):
        raise ValueError("penalty must have shape (p, p)")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("working weights must be finite and non-negative")

    sum_w = float(np.sum(W, dtype=np.float64))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ValueError("working weights must have a positive finite sum")
    mean_x = dm.rmatvec(W) / sum_w
    mean_z = float(np.dot(W, z_off) / sum_w)
    data_gram, rhs = centered_gram_rhs(
        dm=dm,
        W=W,
        mean_x=mean_x,
        z_centered=z_off - mean_z,
    )
    penalty_symmetric = 0.5 * (penalty + penalty.T)
    hessian = data_gram + penalty_symmetric
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=_freeze(mean_x),
        mean_z=mean_z,
        data_gram=_freeze(data_gram),
        rhs=_freeze(rhs),
        penalty=_freeze(penalty_symmetric),
        hessian=_freeze(hessian),
    )
