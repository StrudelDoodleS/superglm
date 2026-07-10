"""Stable centered weighted products for grouped design matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_kernels import _disc_disc_2d_hist, _fused_bincount_2

_MAX_PACKED_HIST_CELLS = 5_000_000


@dataclass(frozen=True)
class _CenteredSupport:
    values: NDArray
    codes: NDArray
    mean: NDArray
    mass: NDArray
    weighted_z: NDArray


def _anchor_center_support(
    *,
    values: NDArray,
    codes: NDArray,
    W: NDArray,
    Wz: NDArray,
    sum_w: float,
    transform: NDArray | None = None,
) -> _CenteredSupport:
    """Center compact support rows before any weighted cross-products."""
    mass, weighted_z = _fused_bincount_2(codes, W, Wz, len(values))
    anchor = int(np.argmax(mass))
    differences = values - values[anchor]
    mean_difference = mass @ differences / sum_w
    centered = differences - mean_difference
    mean = values[anchor] + mean_difference
    if transform is not None:
        centered = centered @ transform
        mean = mean @ transform
    return _CenteredSupport(
        values=centered,
        codes=codes,
        mean=mean,
        mass=mass,
        weighted_z=weighted_z,
    )


def packed_centered_gram_rhs(
    *,
    dm,
    W: NDArray,
    z_centered: NDArray,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Build centered products from indexed supports when every group is eligible."""
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    )

    supports: list[_CenteredSupport] = []
    widths: list[int] = []
    weighted_z = W * z_centered
    sum_w = float(np.sum(W, dtype=np.float64))
    for gm in dm.group_matrices:
        if type(gm) in (DiscretizedSSPGroupMatrix, DiscretizedTensorGroupMatrix):
            supports.append(
                _anchor_center_support(
                    values=gm.B_unique,
                    codes=gm.bin_idx,
                    W=W,
                    Wz=weighted_z,
                    sum_w=sum_w,
                    transform=gm.R_inv,
                )
            )
        elif isinstance(gm, CategoricalGroupMatrix):
            values = np.zeros((gm.n_levels + 1, gm.n_levels), dtype=float)
            values[np.arange(gm.n_levels), np.arange(gm.n_levels)] = 1.0
            supports.append(
                _anchor_center_support(
                    values=values,
                    codes=gm.codes,
                    W=W,
                    Wz=weighted_z,
                    sum_w=sum_w,
                )
            )
        else:
            return None
        widths.append(gm.shape[1])

    if any(
        len(left.values) * len(right.values) > _MAX_PACKED_HIST_CELLS
        for i, left in enumerate(supports)
        for right in supports[i + 1 :]
    ):
        return None

    p = dm.p
    gram = np.zeros((p, p), dtype=float)
    rhs = np.zeros(p, dtype=float)
    mean_x = (
        np.concatenate([support.mean for support in supports])
        if supports
        else np.zeros(0, dtype=float)
    )
    starts = np.cumsum([0, *widths])

    for i, support_i in enumerate(supports):
        sl_i = slice(starts[i], starts[i + 1])
        gram[sl_i, sl_i] = support_i.values.T @ (support_i.mass[:, None] * support_i.values)
        rhs[sl_i] = support_i.values.T @ support_i.weighted_z

        for j in range(i + 1, len(supports)):
            support_j = supports[j]
            sl_j = slice(starts[j], starts[j + 1])
            n_j = len(support_j.values)
            joint_mass = _disc_disc_2d_hist(
                support_i.codes,
                support_j.codes,
                W,
                len(support_i.values),
                n_j,
            )
            cross = support_i.values.T @ joint_mass @ support_j.values
            gram[sl_i, sl_j] = cross
            gram[sl_j, sl_i] = cross.T

    return mean_x, gram, rhs


def _compensated_add(total: NDArray, compensation: NDArray, value: NDArray) -> None:
    corrected = value - compensation
    updated = total + corrected
    compensation[...] = (updated - total) - corrected
    total[...] = updated


def centered_gram_rhs(
    *,
    dm,
    W: NDArray,
    mean_x: NDArray,
    z_centered: NDArray,
    chunk_size: int = 8192,
) -> tuple[NDArray, NDArray]:
    """Return centered ``X'WX`` and ``X'Wz`` without raw-moment subtraction.

    Rows are materialized only in bounded chunks. Centering happens before
    multiplication, so large feature offsets cannot cancel two raw moments.
    Group-specific ``row_subset`` implementations preserve sparse/discretized
    storage and avoid materializing the full training design.
    """
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    mean_x = np.asarray(mean_x, dtype=float)
    z_centered = np.asarray(z_centered, dtype=float)
    if W.shape != (n,) or z_centered.shape != (n,):
        raise ValueError("W and z_centered must match the design row count")
    if mean_x.shape != (p,):
        raise ValueError("mean_x must match the design column count")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if p == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    gram = np.zeros((p, p), dtype=float)
    gram_compensation = np.zeros_like(gram)
    rhs = np.zeros(p, dtype=float)
    rhs_compensation = np.zeros_like(rhs)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        W_block = W[start:stop]
        gram_block = block.T @ (W_block[:, None] * block)
        rhs_block = block.T @ (W_block * z_centered[start:stop])
        _compensated_add(gram, gram_compensation, gram_block)
        _compensated_add(rhs, rhs_compensation, rhs_block)

    gram = 0.5 * (gram + gram.T)
    return gram, rhs


def centered_rhs(
    *,
    dm,
    W: NDArray,
    mean_x: NDArray,
    z_centered: NDArray,
    chunk_size: int = 8192,
) -> NDArray:
    """Return ``(X - mean_x)' W z_centered`` without rebuilding the Gram."""
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    mean_x = np.asarray(mean_x, dtype=float)
    z_centered = np.asarray(z_centered, dtype=float)
    if W.shape != (n,) or z_centered.shape != (n,):
        raise ValueError("W and z_centered must match the design row count")
    if mean_x.shape != (p,):
        raise ValueError("mean_x must match the design column count")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if p == 0:
        return np.zeros(0, dtype=float)

    rhs = np.zeros(p, dtype=float)
    compensation = np.zeros_like(rhs)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        rhs_block = block.T @ (W[start:stop] * z_centered[start:stop])
        _compensated_add(rhs, compensation, rhs_block)
    return rhs
