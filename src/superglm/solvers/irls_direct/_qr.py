"""The QR step: least squares on the weighted, centred design and the penalty's square root."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
)
from superglm.solvers.centered_system import CenteredSystem
from superglm.solvers.irls_direct._centred import _commit_centred_slopes
from superglm.solvers.irls_direct._finalise import _finalise_centred
from superglm.solvers.irls_direct._step import _IRLSDirectStep
from superglm.solvers.rank import RankDecomposition, decompose_factor

if TYPE_CHECKING:
    from superglm.solvers.irls_direct._fit import _IRLSDirectFit

# One logger for the whole package, so every submodule's records share its name.
logger = logging.getLogger(__package__)


def _sqrt_penalty_augmented(S: NDArray, p: int) -> NDArray:
    """Build (p+1, p+1) augmented sqrt-penalty for QR solver.

    Returns L_aug where L_aug.T @ L_aug has S in the [1:, 1:] block and
    zeros in the intercept row/column.
    """
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 0.0)
    L_aug = np.zeros((p + 1, p + 1))
    L_aug[1:, 1:] = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    return L_aug


def _centred_rows(X: NDArray, system: CenteredSystem) -> NDArray:
    """``X`` centred about the system's exact centre pair, ``(X - hi) - lo``.

    Centring about the rounded ``mean_x`` alone would leave ``sqrt(W) d`` per
    row in the QR's data block, ``d`` that rounding (issue #430).
    """
    centre, centre_lo = system.centre_pair()
    rows = X - centre
    return rows if centre_lo is None else rows - centre_lo


class _IRLSDirectQRStep(_IRLSDirectStep):
    """The fit's step under ``direct_solve="qr"``: a QR factor of ``[sqrt(W) X~; sqrt(S)]``.

    For fits with no linear constraints.  Materialises the dense ``(n, p)``
    design and the penalty's square root once, at setup; each iteration
    profiles the intercept out and solves the slopes from the factor.  Holds
    the iteration's decomposition for the true-score certificate, and no
    reference to the fit.
    """

    __slots__ = ("_X_full", "_L_aug", "_held_rank")
    backend = "qr"
    finalise = _finalise_centred

    def __init__(self, gms: list[GroupMatrix], S: NDArray, p: int) -> None:
        has_disc = any(
            isinstance(gm, DiscretizedSSPGroupMatrix | DiscretizedSplineCategoricalGroupMatrix)
            for gm in gms
        )
        if has_disc:
            logger.warning(
                "direct_solve='qr' with discretized groups materialises the full "
                "(n, p) design matrix, defeating the O(n_bins) discretization "
                "benefit.  Consider direct_solve='auto' for large-n discrete fits."
            )
        self._X_full = np.hstack([gm.toarray() for gm in gms])  # (n, p)
        self._L_aug = _sqrt_penalty_augmented(S, p)  # (p+1, p+1)
        self._held_rank: RankDecomposition | None = None

    def propose(self, fit: _IRLSDirectFit, it: int) -> None:
        W = fit.W
        # Profile the intercept, then apply the shared factor-space rule.
        _t0 = time.perf_counter()
        sqrtW = np.sqrt(W)
        z_off = fit.z - fit.offset
        centered = fit.get_centered_system(W, z_off)
        fit._last_working_centered = centered
        A_data = sqrtW[:, None] * _centred_rows(self._X_full, centered)
        A = np.vstack([A_data, self._L_aug[1:, 1:]])
        rhs_qr = np.concatenate([sqrtW * (z_off - centered.mean_z), np.zeros(fit.p)])
        iteration_rank = decompose_factor(A, retain_factor_solve=True)
        beta = iteration_rank.solve_factor_rhs(rhs_qr)
        _commit_centred_slopes(fit, centered, beta, iteration_rank, _t0)
        self._held_rank = iteration_rank

    def certificate_inputs(self) -> tuple:
        """The iteration's decomposition: the certificate judges its truncated directions."""
        return self._held_rank, None

    def data_rank(self, centered_final: CenteredSystem, W: NDArray) -> RankDecomposition:
        """The retained fit's data rank, from the factor of its weighted, centred dense rows."""
        sqrtW = np.sqrt(W)
        A_data_final = sqrtW[:, None] * _centred_rows(self._X_full, centered_final)
        return decompose_factor(A_data_final)
