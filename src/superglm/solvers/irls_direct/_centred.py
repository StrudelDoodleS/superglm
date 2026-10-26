"""Helpers on the centred working system that more than one of the fit's steps reads."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DesignMatrix
from superglm.solvers.centered_system import CenteredSystem
from superglm.solvers.mode_score import centre_offset_mean, dense_columns


def _system_offset_mean(
    dm: DesignMatrix, W: NDArray, system: CenteredSystem, state_center: NDArray
) -> NDArray:
    """``mean_x - c``, the working mean's offset from the state's centre, read from the system's pair.

    A dense column's working mean is the system's exact pair ``(hi, lo)``
    (``centered_system.dense_mean_pair``, ``_attach_dense_split``), so its
    offset from ``c`` is ``(hi - c) + lo``: ``hi - c`` is exact where the two
    lie within a factor two (Sterbenz), as they do at a column's offset, and
    otherwise rounds at ``u |hi - c|``, inside ``centre_offset_mean``'s
    ``gamma_n sum W |x - c| / sum W``.  No pass.  A system without a pair has
    no dense column and reads ``mean_x - c``.
    """
    if system.mean_hi is None or system.mean_lo is None:
        return centre_offset_mean(dm, W, float(system.sum_w), state_center, system.mean_x)
    offset_mean = np.asarray(system.mean_x, dtype=np.float64) - state_center
    dense = dense_columns(dm)
    offset_mean[dense] = (system.mean_hi[dense] - state_center[dense]) + system.mean_lo[dense]
    return offset_mean
