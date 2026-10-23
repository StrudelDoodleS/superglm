"""Leaf systems built from explicit rows: a test fixture of the structured factor tests.

``leaf_system_from_rows`` folds hand-made rows with the engine's own leaf kernel
(``block_leaves._leaf_segments``) and assembles them as the design's leaf pass
does (``block_leaves._assemble_compact``), without a design matrix or layout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.solvers._structured.block_leaves import (
    _assemble_compact,
    _close_levels,
    _leaf_segments,
    _LeafParts,
    _signed_parts,
)
from superglm.solvers._structured.border import BorderGenerators


def leaf_system_from_rows(
    basis_rows: NDArray,
    border_rows: NDArray,
    levels: NDArray,
    W: NDArray,
    Wz: NDArray,
    *,
    n_levels: int,
    small_indices: NDArray,
    structured_indices: NDArray,
    center: NDArray | None = None,
    generators: BorderGenerators | None = None,
    error: NDArray | None = None,
    signed: bool = False,
    name: str = "fs",
    basis: str = "fs",
):
    """An ``fs`` (or ``sz``) leaf system from explicit rows, free of any design or layout.

    ``basis_rows`` ``(n, k)`` in the natural basis, ``border_rows`` ``(n, q)``
    raw, ``levels`` ``(n,)``; ``center`` ``(q,)`` (default 0) is subtracted from
    the border rows.  The rows are folded in stable level order in one chunk,
    by the same kernel as the design's leaf pass.  ``basis="sz"`` takes
    ``n_levels`` levels behind ``structured_indices`` ``(n_levels - 1, k)``.
    """
    basis_rows = np.asarray(basis_rows, dtype=np.float64)
    border_rows = np.asarray(border_rows, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.intp)
    weights = np.asarray(W, dtype=np.float64)
    weighted_rhs = np.asarray(Wz, dtype=np.float64)
    n, k = basis_rows.shape
    q = border_rows.shape[1]
    center = np.zeros(q) if center is None else np.asarray(center, dtype=np.float64)
    if not signed and np.any(weights < 0.0):
        raise ValueError("Fisher rows must have non-negative weights; signed rows declare it.")
    order = np.argsort(levels, kind="stable")
    p = k + q + 2
    rows = np.empty((n, p))
    rows[:, :k] = basis_rows[order]
    rows[:, k] = 1.0
    rows[:, k + 1 : k + 1 + q] = border_rows[order] - center
    w = weights[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        rows[:, p - 1] = np.where(w != 0.0, weighted_rhs[order] / np.where(w != 0.0, w, 1.0), 0.0)
    if not np.all(np.isfinite(rows)):
        raise np.linalg.LinAlgError(f"FactorSmooth term {name!r} has non-finite leaf rows.")
    R_acc = np.zeros((n_levels, p, p))
    M_acc = np.zeros((n_levels, p, p)) if signed else np.zeros((1, 1, 1))
    E_acc = np.zeros((n_levels, p - 1, p - 1)) if signed else np.zeros((1, 1, 1))
    started = np.zeros(n_levels, dtype=np.bool_)
    counts = np.zeros(n_levels, dtype=np.int64)
    merges = np.zeros(n_levels, dtype=np.int64)
    errors = np.abs(weights) if error is None else np.asarray(error, dtype=np.float64)
    _leaf_segments(
        rows,
        np.ascontiguousarray(w),
        np.ascontiguousarray(errors[order]),
        np.ascontiguousarray(levels[order]),
        R_acc,
        M_acc,
        E_acc,
        started,
        counts,
        merges,
        signed,
    )
    return _assemble_system(
        R_acc,
        M_acc if signed else None,
        E_acc if signed else None,
        counts,
        merges,
        center=center,
        generators=generators,
        signed=signed,
        block_size=k,
        weights=weights,
        weighted_rhs=weighted_rhs,
        small_indices=np.asarray(small_indices, dtype=np.intp),
        structured_indices=np.asarray(structured_indices, dtype=np.intp),
        group_index=0,
        group_name=name,
        basis=basis,
    )


def _assemble_system(
    R_acc: NDArray,
    M_acc: NDArray | None,
    E_acc: NDArray | None,
    counts: NDArray,
    merges: NDArray,
    *,
    center: NDArray,
    generators: BorderGenerators | None,
    signed: bool,
    block_size: int,
    weights: NDArray,
    weighted_rhs: NDArray,
    small_indices: NDArray,
    structured_indices: NDArray,
    group_index: int,
    group_name: str,
    basis: str = "fs",
    thin_levels: tuple = (),
    thin_counts: tuple[NDArray, NDArray] | None = None,
):
    """The leaf data (pseudo-rows or within Gram) and the raw moment operator by products.

    The whole triangles ``R_acc`` are closed level by level into the compact
    leaf data (``_close_levels``, the arithmetic of the compact pass) and the
    system is assembled from it (``_assemble_compact``); signed rows read
    their triangles and middle factors once more there, and an ``sz`` leaf
    keeps the triangles.
    """
    k = block_size
    K, p = R_acc.shape[0], R_acc.shape[2]
    top = np.zeros((K, k, p))
    tail_norms2 = np.zeros((K, p - k))
    tail_gram = np.zeros((p - k, p - k))
    _close_levels(R_acc, k, top, tail_norms2, tail_gram, not signed)
    parts = _LeafParts(top, tail_norms2, None if signed else tail_gram, counts, merges)
    if signed:
        assert M_acc is not None
        pseudo_rows, signature, level_rows, border = _signed_parts(R_acc, M_acc, k)
        assert E_acc is not None
        parts = parts._replace(
            pseudo_rows=pseudo_rows,
            signature=signature,
            error_rows=np.ascontiguousarray(E_acc[:, :k, :]),
            error_diagonal=np.diagonal(E_acc, axis1=1, axis2=2).copy(),
            level_rows=level_rows,
            border=border,
        )
    return _assemble_compact(
        parts._replace(triangles=R_acc),
        center=center,
        generators=generators,
        signed=signed,
        block_size=block_size,
        weights=weights,
        weighted_rhs=weighted_rhs,
        small_indices=small_indices,
        structured_indices=structured_indices,
        group_index=group_index,
        group_name=group_name,
        basis=basis,
        thin_levels=thin_levels,
        thin_counts=thin_counts,
    )
