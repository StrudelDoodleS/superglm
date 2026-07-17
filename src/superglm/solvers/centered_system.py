"""Stable intercept-profiled systems shared by fitting and inference."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_centered import (
    _try_tabmat_centering,
    centered_gram_rhs,
    centered_rhs,
    packed_centered_gram_rhs,
)
from superglm.group_matrix import DesignMatrix

_FACTOR_CHUNK_BYTES = 16 * 1024 * 1024
_FACTOR_CHUNK_ROWS = 8192


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

    def raw_weighted_moments(self) -> tuple[NDArray, NDArray, NDArray, float]:
        """Recover raw Gram/RHS moments from the stable centered system."""
        xtw1 = self.sum_w * self.mean_x
        sum_wz = self.sum_w * self.mean_z
        gram = self.data_gram + self.sum_w * np.outer(self.mean_x, self.mean_x)
        xtwz = self.rhs + self.mean_x * sum_wz
        return gram, xtw1, xtwz, sum_wz


@dataclass
class TabmatCenteringState:
    """Fit-local safety decision for raw-moment Tabmat centering."""

    eligible: bool | None = None


def iter_grouped_design_chunks(dm: DesignMatrix) -> Iterator[tuple[int, int, NDArray]]:
    """Yield bounded dense rows only for rare factor-rank certification."""
    bytes_per_row = 3 * np.dtype(np.float64).itemsize * max(dm.p, 1)
    chunk_rows = max(1, min(_FACTOR_CHUNK_ROWS, _FACTOR_CHUNK_BYTES // bytes_per_row))
    for start in range(0, dm.n, chunk_rows):
        stop = min(start + chunk_rows, dm.n)
        rows = np.arange(start, stop, dtype=np.intp)
        yield start, stop, np.asarray(dm.row_subset(rows).toarray(), dtype=np.float64)


def grouped_weighted_factor(
    dm: DesignMatrix,
    W: NDArray,
    *,
    center: NDArray | None = None,
) -> NDArray:
    """Return a streaming weighted QR factor without retaining all design rows."""
    from superglm.solvers.rank import streamed_weighted_factor

    return streamed_weighted_factor(iter_grouped_design_chunks(dm), W, center=center)


def penalty_factor(penalty: NDArray) -> NDArray:
    """Return a square factor whose cross-product is a PSD penalty matrix."""
    if penalty.shape == (0, 0) or not np.any(penalty):
        return np.empty((0, penalty.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (penalty + penalty.T))
    positive = eigenvalues > 0.0
    return np.sqrt(eigenvalues[positive])[:, None] * eigenvectors[:, positive].T


def grouped_augmented_factor(
    dm: DesignMatrix,
    W: NDArray,
    penalty: NDArray,
    *,
    center: NDArray | None = None,
) -> NDArray:
    """Return the bounded weighted-design factor augmented by ``sqrt(S)``."""
    data_factor = grouped_weighted_factor(dm, W, center=center)
    smooth_factor = penalty_factor(penalty)
    return data_factor if smooth_factor.shape[0] == 0 else np.vstack((data_factor, smooth_factor))


def refresh_centered_rhs(
    *,
    system: CenteredSystem,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
) -> CenteredSystem:
    """Reuse an invariant centered Gram while refreshing its working RHS."""
    mean_z = float(np.dot(W, z_off) / system.sum_w)
    z_centered = z_off - mean_z
    centered_scale = np.sqrt(np.maximum(np.diag(system.data_gram), 0.0) / system.sum_w)
    max_fast_ratio = np.finfo(float).eps ** -0.25
    well_scaled = np.all(
        (np.abs(system.mean_x) <= max_fast_ratio * centered_scale)
        | ((system.mean_x == 0.0) & (centered_scale == 0.0))
    )
    if well_scaled:
        weighted_z = W * z_centered
        rhs = dm.rmatvec(weighted_z) - system.mean_x * float(np.sum(weighted_z))
    else:
        rhs = centered_rhs(
            dm=dm,
            W=W,
            mean_x=system.mean_x,
            z_centered=z_centered,
        )
    return CenteredSystem(
        sum_w=system.sum_w,
        mean_x=system.mean_x,
        mean_z=mean_z,
        data_gram=system.data_gram,
        rhs=_freeze(rhs),
        penalty=system.penalty,
        hessian=system.hessian,
    )


def build_centered_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
    penalty: NDArray,
    tabmat_split=None,
    tabmat_state: TabmatCenteringState | None = None,
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
    mean_z = float(np.dot(W, z_off) / sum_w)
    z_centered = z_off - mean_z
    packed = packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered)
    if (
        packed is None
        and tabmat_split is not None
        and (tabmat_state is None or tabmat_state.eligible is not False)
    ):
        packed = _try_tabmat_centering(
            tabmat_split=tabmat_split,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            preflight=tabmat_state is None or tabmat_state.eligible is None,
        )
        if tabmat_state is not None:
            # A rejection is permanent for this fit.  Later IRLS weights can
            # change the centering ratio, but the stable path remains correct
            # and avoids repeating rejected raw work.
            tabmat_state.eligible = packed is not None
    if packed is None:
        mean_x = dm.rmatvec(W) / sum_w
        data_gram, rhs = centered_gram_rhs(
            dm=dm,
            W=W,
            mean_x=mean_x,
            z_centered=z_centered,
        )
    else:
        mean_x, data_gram, rhs = packed
    penalty_symmetric = 0.5 * (penalty + penalty.T)
    hessian = data_gram + penalty_symmetric
    # Both terms are mathematically PSD. Degenerate spline
    # reparameterizations can introduce visible negative round-off, so project
    # only this declared-PSD system back onto its valid cone before rank work.
    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError:
        hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(hessian)
        if hessian_eigenvalues.size and hessian_eigenvalues[0] < 0.0:
            hessian = (
                hessian_eigenvectors * np.maximum(hessian_eigenvalues, 0.0)[None, :]
            ) @ hessian_eigenvectors.T
            hessian = 0.5 * (hessian + hessian.T)
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=_freeze(mean_x),
        mean_z=mean_z,
        data_gram=_freeze(data_gram),
        rhs=_freeze(rhs),
        penalty=_freeze(penalty_symmetric),
        hessian=_freeze(hessian),
    )
