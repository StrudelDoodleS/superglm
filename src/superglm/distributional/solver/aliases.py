"""Which directions a SuperLSS terminal rank drops are aliases of the design, and where they sit.

The stacked curvature is ``H = sum_r X_r' C_r X_r + S``, block ``b`` the design ``[1, X_b]`` of
predictor ``b`` (with its intercept column, if it has one).  Where every row's channel matrix
``C_r`` is positive definite, ``H v = 0`` exactly when ``[1, X_b] v_b = 0`` for every block and
``S v = 0``, so the design's null space is the product of its blocks' (Wood, Pya & Saefken 2016,
JASA 111, sec. 3.1.2, for one block).  The decision SuperGLM's fit takes,
``identifiability._design_null`` at unit weight on the rows with positive prior weight and with
the actual penalty, is therefore taken block by block on the span the terminal rank drops: a
direction of it is an alias when each of its block parts is, and the rest are identified by the
design and carry no curvature the fit resolves (a working weight carries no information about
identifiability, only about its precision).  An alias's coordinates are dropped where SuperGLM's
gauge holds them (``identifiability._construction_gauge``: the latest construction column each
direction reaches in term order, a predictor's intercept first), so the two fits blank the same
coefficients on the same design.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import SQRT_EPS
from superglm.distributional.layout import StackedLayout
from superglm.group_matrix import DesignMatrix
from superglm.solvers.identifiability import (
    ConstructionColumns,
    TruncatedDecomposition,
    _construction_gauge,
    _design_null,
    _RowFacts,
    penalty_null_space,
)
from superglm.types import GroupSlice

# The owner's share floor (``identifiability._SHARE_FLOOR``): a block part of a unit direction
# below it, in Jacobi-scaled coordinates, carries no information about that block.
_SHARE_FLOOR = SQRT_EPS


def _jacobi(column_scale: NDArray) -> NDArray:
    scale = np.asarray(column_scale, dtype=np.float64)
    return np.where(scale > 0.0, scale, 1.0)


def _orthonormal(vectors: NDArray) -> NDArray:
    """An orthonormal basis of ``vectors`` (already Jacobi-scaled), each normalised first."""
    norms = np.linalg.norm(vectors, axis=0)
    vectors = vectors[:, norms > 0.0] / norms[norms > 0.0]
    if vectors.shape[1] == 0:
        return vectors
    left, singular, _ = np.linalg.svd(vectors, full_matrices=False)
    return left[:, singular > _SHARE_FLOOR * singular[0]]


def _block_design(state) -> tuple[DesignMatrix, int]:
    """Predictor ``state``'s design with its intercept as a column, and that column's count."""
    from superglm._group_matrix._group_matrix_core import CategoricalGroupMatrix

    design = state.design
    if state.intercept_index is None:
        return design, 0
    ones = CategoricalGroupMatrix(np.zeros(design.n, dtype=np.intp), 1)
    return DesignMatrix([ones, *design.group_matrices], design.n, design.p + 1), 1


def design_split(
    layout: StackedLayout,
    span: NDArray,
    column_scale: NDArray,
    penalty: NDArray,
    weights: NDArray,
) -> tuple[NDArray, NDArray]:
    """``span`` (``width x k``) split into the design's aliases and the directions it identifies.

    Each block's part of the span is read by ``_design_null`` on that block's rows with positive
    prior weight in ``weights``, uncentred on ``[1, X_b]``, with the block of ``penalty``; a
    block part under the share floor of its direction is rounding.  The aliases are the
    directions every block part of which is null, the rest their complement in the span; both
    are returned in coefficient coordinates, orthonormal in the Jacobi scale ``column_scale``.
    """
    scale = _jacobi(column_scale)
    basis = _orthonormal(np.asarray(span, dtype=np.float64) * scale[:, None])
    count = basis.shape[1]
    if count == 0:
        return basis, basis
    prior = np.asarray(weights, dtype=np.float64)
    constraints = []
    for state in layout.predictors:
        block = basis[state.coefficient_slice]
        left, singular, _ = np.linalg.svd(block, full_matrices=False)
        local = left[:, singular > _SHARE_FLOOR]
        if local.shape[1] == 0:
            continue
        design, _ = _block_design(state)
        support = (
            (prior > 0.0).astype(np.float64) if prior.shape == (design.n,) else np.ones(design.n)
        )
        own = scale[state.coefficient_slice]
        S = np.asarray(penalty, dtype=np.float64)[state.coefficient_slice, state.coefficient_slice]
        decomposition = TruncatedDecomposition(
            null_vectors=np.zeros((own.size, 0)),
            rank=own.size,
            column_scale=own,
            pre_truncation_condition=1.0,
        )
        _, rest = _design_null(
            decomposition,
            local / own[:, None],
            [None] * len(design.group_matrices),
            design,
            support,
            lambda T, S=S: (S @ T, np.abs(S) @ np.abs(T)),
            centred=False,
        )
        if rest.shape[1]:
            constraints.append((rest * own[:, None]).T @ block)
    if not constraints:
        return basis / scale[:, None], np.zeros((basis.shape[0], 0))
    _, singular, right = np.linalg.svd(np.vstack(constraints))
    identified = np.zeros(count, dtype=bool)
    identified[: singular.size] = singular > _SHARE_FLOOR
    rotated = basis @ right.T
    return rotated[:, ~identified] / scale[:, None], rotated[:, identified] / scale[:, None]


def _stacked_groups(layout: StackedLayout) -> tuple[list[GroupSlice], list]:
    """Every term of every predictor in stacked coordinates, each predictor's intercept first."""
    groups: list[GroupSlice] = []
    matrices: list = []
    for state in layout.predictors:
        start = state.coefficient_slice.start
        if state.intercept_index is not None:
            groups.append(GroupSlice(f"{state.name}:(intercept)", start, start + 1))
            matrices.append(None)
            start += 1
        for group, matrix in zip(state.groups, state.design.group_matrices, strict=True):
            name = f"{state.name}:{group.name}"
            groups.append(
                replace(group, name=name, start=group.start + start, end=group.end + start)
            )
            matrices.append(matrix)
    return groups, matrices


def held_coordinates(
    layout: StackedLayout,
    aliases: NDArray,
    column_scale: NDArray,
    penalty: NDArray,
    data_diagonal: NDArray,
) -> NDArray | None:
    """The coordinates SuperGLM's gauge holds at zero on the alias span ``aliases``.

    Each term's construction columns are its actual penalty's null space in its raw basis
    (``state_ops._raw_construction``), a predictor's intercept and an unpenalised term's its
    coordinates; ``_construction_gauge`` holds the latest each direction reaches in term order.
    ``None`` when a held column is a smooth's unpenalised part, which no coordinate represents.
    """
    from superglm.model.state_ops import _raw_construction

    groups, matrices = _stacked_groups(layout)
    width = layout.n_coefficients
    owner = np.empty(width, dtype=np.intp)
    for index, group in enumerate(groups):
        owner[group.sl] = index
    full = np.asarray(penalty, dtype=np.float64)
    built: dict[int, ConstructionColumns] = {}

    def construction(index: int) -> ConstructionColumns:
        if index not in built:
            group, matrix = groups[index], matrices[index]
            null = penalty_null_space([full[group.sl, group.sl]], group.size)
            built[index] = _raw_construction(null, 1, group, matrix)
        return built[index]

    decomposition = TruncatedDecomposition(
        null_vectors=np.asarray(aliases, dtype=np.float64),
        rank=width - np.shape(aliases)[1],
        column_scale=np.asarray(column_scale, dtype=np.float64),
        pre_truncation_condition=1.0,
    )
    gauge, _ = _construction_gauge(
        np.asarray(aliases, dtype=np.float64),
        np.zeros(np.shape(aliases)[1], dtype=bool),
        decomposition,
        np.arange(width),
        width,
        groups,
        owner,
        construction,
        np.asarray(data_diagonal, dtype=np.float64),
        None,
        None,
        _RowFacts(None, None),
        {},
    )
    if gauge is None:
        return None
    functionals = gauge.functionals
    if np.any(np.count_nonzero(functionals, axis=0) != 1):
        return None
    held = np.zeros(width, dtype=bool)
    held[np.argmax(np.abs(functionals), axis=0)] = True
    return held


def reached(directions: NDArray, column_scale: NDArray) -> NDArray:
    """The coordinates ``directions`` reach: leverage above the share floor in the Jacobi scale."""
    scale = _jacobi(column_scale)
    basis = _orthonormal(np.asarray(directions, dtype=np.float64) * scale[:, None])
    return np.einsum("ij,ij->i", basis, basis) > _SHARE_FLOOR**2


__all__ = ["design_split", "held_coordinates", "reached"]
