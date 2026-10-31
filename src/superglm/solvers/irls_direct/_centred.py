"""Helpers on the centred working system that more than one of the fit's steps reads."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import DesignMatrix
from superglm.solvers.centered_system import CenteredSystem
from superglm.solvers.mode_score import centre_offset_mean, dense_columns
from superglm.solvers.rank import RankDecomposition

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit


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


def _commit_centred_slopes(
    fit: _IRLSDirectFit, centered: CenteredSystem, beta: NDArray, rank: RankDecomposition, t0: float
) -> None:
    """Write a centred step's slopes, profiled intercept and rank state to the fit; time its solve from ``t0``.

    With a state centre, the intercept is formed about it from the working
    mean's offset to it, on centred rows (``_system_offset_mean``): no
    raw-scale cancellation.
    """
    W, state_center = fit.W, fit._state_center
    proposal_centred_intercept = fit.proposal_centred_intercept
    intercept = centered.mean_z - float(centered.mean_x @ beta)
    offset_mean = None
    if state_center is not None:
        offset_mean = _system_offset_mean(fit.dm, W, centered, state_center)
        proposal_centred_intercept = centered.mean_z - math.fsum(offset_mean * beta)
        intercept = proposal_centred_intercept - math.fsum(state_center * beta)
    fit._cond_est = rank.pre_truncation_condition
    fit._used_svd = rank.used_svd_fallback
    fit.rank_truncated = rank.rank_truncated
    if fit.convergence == "mode_score":
        fit._score_centre = (
            centered.mean_x,
            centered.sum_w,
            np.diag(centered.data_gram),
            (),
            offset_mean,
        )
    fit.recorder.t_solve += time.perf_counter() - t0
    fit.beta, fit.intercept = beta, intercept
    fit._last_working_offset_mean = offset_mean
    fit.proposal_centred_intercept = proposal_centred_intercept
