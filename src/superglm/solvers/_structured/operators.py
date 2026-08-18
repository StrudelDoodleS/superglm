"""Compact structured operators and low-rank algebra."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import (
    adjoint_sum_to_zero_blocks,
    expand_sum_to_zero_blocks,
)


@dataclass(frozen=True)
class SymmetricBlockOperator:
    """Symmetric matrix represented by dense-small, cross, and diagonal blocks."""

    A: NDArray
    C: NDArray
    d: NDArray
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        for name, dtype in (
            ("A", np.float64),
            ("C", np.float64),
            ("d", np.float64),
            ("small_indices", np.intp),
            ("structured_indices", np.intp),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        q = len(self.small_indices)
        k = len(self.structured_indices)
        if self.A.shape != (q, q):
            raise ValueError(f"A shape {self.A.shape} does not match ({q}, {q}).")
        if self.C.shape != (k, q):
            raise ValueError(f"C shape {self.C.shape} does not match ({k}, {q}).")
        if self.d.shape != (k,):
            raise ValueError(f"d shape {self.d.shape} does not match ({k},).")
        all_indices = np.concatenate([self.small_indices, self.structured_indices])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")
        object.__setattr__(self, "shape", (len(all_indices), len(all_indices)))

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply the compact symmetric operator to one or many RHS columns."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        small_rhs = values[self.small_indices]
        structured_rhs = values[self.structured_indices]
        result = np.empty_like(values)
        result[self.small_indices] = self.A @ small_rhs + self.C.T @ structured_rhs
        result[self.structured_indices] = self.C @ small_rhs + self.d[:, None] * structured_rhs
        return result[:, 0] if vector_rhs else result


@dataclass(frozen=True)
class BlockSymmetricOperator:
    """Symmetric matrix with a dense-small block and repeated dense local blocks."""

    A: NDArray
    C: NDArray
    D: NDArray
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        for name, dtype in (
            ("A", np.float64),
            ("C", np.float64),
            ("D", np.float64),
            ("small_indices", np.intp),
            ("structured_indices", np.intp),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        if self.C.ndim != 3:
            raise ValueError("C must have shape (n_levels, block_size, small_size).")
        n_levels, block_size, small_size = self.C.shape
        if self.A.shape != (small_size, small_size):
            raise ValueError(f"A shape {self.A.shape} does not match ({small_size}, {small_size}).")
        if self.D.shape != (n_levels, block_size, block_size):
            raise ValueError(
                f"D shape {self.D.shape} does not match ({n_levels}, {block_size}, {block_size})."
            )
        if self.small_indices.shape != (small_size,):
            raise ValueError("small_indices width does not match A.")
        if self.structured_indices.shape != (n_levels, block_size):
            raise ValueError("structured_indices shape does not match C and D.")
        if not np.allclose(self.D, self.D.transpose(0, 2, 1), rtol=0.0, atol=1e-13):
            raise ValueError("Every local D block must be symmetric.")
        all_indices = np.concatenate([self.small_indices, self.structured_indices.ravel()])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("small_indices and structured_indices must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("Structured index partitions must cover every coefficient once.")
        object.__setattr__(self, "shape", (len(all_indices), len(all_indices)))

    @property
    def n_levels(self) -> int:
        return int(self.C.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.C.shape[1])

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply the compact block operator to one or many RHS columns."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        small_rhs = values[self.small_indices]
        structured_rhs = values[self.structured_indices]
        result = np.empty_like(values)
        result[self.small_indices] = self.A @ small_rhs + np.einsum(
            "kiq,kim->qm",
            self.C,
            structured_rhs,
            optimize=True,
        )
        result[self.structured_indices] = np.einsum(
            "kiq,qm->kim", self.C, small_rhs, optimize=True
        ) + np.einsum("kij,kjm->kim", self.D, structured_rhs, optimize=True)
        return result[:, 0] if vector_rhs else result


@dataclass(frozen=True)
class SumToZeroBlockOperator:
    """Raw all-level blocks exposed through ``K - 1`` sum-to-zero coordinates."""

    A: NDArray
    C: NDArray
    D: NDArray
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        for name, dtype in (
            ("A", np.float64),
            ("C", np.float64),
            ("D", np.float64),
            ("small_indices", np.intp),
            ("structured_indices", np.intp),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        if self.C.ndim != 3:
            raise ValueError("C must have shape (K, k, q).")
        n_levels, block_size, small_size = self.C.shape
        if n_levels < 2 or self.D.shape != (n_levels, block_size, block_size):
            raise ValueError("SZ raw blocks must have shapes (K, k, q) and (K, k, k).")
        if self.A.shape != (small_size, small_size):
            raise ValueError("SZ ordinary block has the wrong shape.")
        if self.small_indices.shape != (small_size,):
            raise ValueError("SZ small_indices width does not match A.")
        if self.structured_indices.shape != (n_levels - 1, block_size):
            raise ValueError("SZ public indices must have shape (K - 1, k).")
        if not all(np.all(np.isfinite(values)) for values in (self.A, self.C, self.D)):
            # Non-finite blocks are what THIS iterate's weights produced, not a
            # malformed call, and callers separate the two by type: the
            # observed-geometry build scores a LinAlgError as a point with no
            # usable penalized mode and routes around it, while a ValueError
            # stops the fit. The shape and partition checks around this one stay
            # ValueError for exactly that reason -- no iterate can cause them.
            raise np.linalg.LinAlgError("SZ operator blocks must be finite.")
        if not np.allclose(self.A, self.A.T, rtol=0.0, atol=1e-13):
            raise ValueError("SZ ordinary block must be symmetric.")
        if not np.allclose(self.D, self.D.transpose(0, 2, 1), rtol=0.0, atol=1e-13):
            raise ValueError("Every SZ local block must be symmetric.")
        all_indices = np.concatenate((self.small_indices, self.structured_indices.ravel()))
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("SZ index partitions must be disjoint.")
        if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
            raise ValueError("SZ index partitions must cover every coefficient once.")
        object.__setattr__(self, "shape", (len(all_indices), len(all_indices)))

    @property
    def n_levels(self) -> int:
        return int(self.C.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.C.shape[1])

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply raw block geometry through the public sum-to-zero contrast."""
        values = np.asarray(rhs, dtype=np.float64)
        vector_rhs = values.ndim == 1
        if vector_rhs:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        small = values[self.small_indices]
        free = values[self.structured_indices]
        raw = expand_sum_to_zero_blocks(free)
        raw_result = np.einsum(
            "kiq,qm->kim",
            self.C,
            small,
            optimize=True,
        ) + np.einsum("kij,kjm->kim", self.D, raw, optimize=True)
        result = np.empty_like(values)
        result[self.small_indices] = self.A @ small + np.einsum(
            "kiq,kim->qm",
            self.C,
            raw,
            optimize=True,
        )
        result[self.structured_indices] = adjoint_sum_to_zero_blocks(raw_result)
        return result[:, 0] if vector_rhs else result


@dataclass(frozen=True)
class CenteredBlockOperator:
    """A block operator centered around a fixed weighted design mean."""

    raw: SymmetricBlockOperator | BlockSymmetricOperator | SumToZeroBlockOperator
    cross: NDArray
    total: float
    center: NDArray
    raw_structured_cross: NDArray | None = None
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        p = self.raw.shape[0]
        cross = np.array(self.cross, dtype=np.float64, copy=True)
        center = np.array(self.center, dtype=np.float64, copy=True)
        if cross.shape != (p,) or center.shape != (p,):
            raise ValueError("Centered operator vectors must match its coefficient width.")
        cross.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "cross", cross)
        object.__setattr__(self, "center", center)
        if self.raw_structured_cross is not None:
            raw_structured_cross = np.array(
                self.raw_structured_cross,
                dtype=np.float64,
                copy=True,
            )
            if not isinstance(self.raw, SumToZeroBlockOperator) or raw_structured_cross.shape != (
                self.raw.n_levels,
                self.raw.block_size,
            ):
                raise ValueError(
                    "raw_structured_cross is only valid for all-level sum-to-zero geometry"
                )
            raw_structured_cross.setflags(write=False)
            object.__setattr__(self, "raw_structured_cross", raw_structured_cross)
        object.__setattr__(self, "total", float(self.total))
        object.__setattr__(self, "shape", self.raw.shape)

    def matvec(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        result = self.raw.matvec(values)
        if values.ndim == 1:
            center_projection = float(self.center @ values)
            cross_projection = float(self.cross @ values)
            return (
                result
                - self.cross * center_projection
                - self.center * cross_projection
                + self.total * self.center * center_projection
            )
        center_projection = self.center @ values
        cross_projection = self.cross @ values
        return (
            result
            - self.cross[:, None] * center_projection
            - self.center[:, None] * cross_projection
            + self.total * self.center[:, None] * center_projection
        )


@dataclass(frozen=True)
class LowRankSymmetricOperator:
    """A symmetric low-rank update ``U R U.T``."""

    basis: NDArray
    core: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        basis = np.array(self.basis, dtype=np.float64, copy=True)
        core = np.array(self.core, dtype=np.float64, copy=True)
        if basis.ndim != 2 or core.shape != (basis.shape[1], basis.shape[1]):
            raise ValueError("Low-rank operator basis and core shapes are inconsistent.")
        if not np.allclose(core, core.T, rtol=0.0, atol=1e-14):
            raise ValueError("Low-rank operator core must be symmetric.")
        basis.setflags(write=False)
        core.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "core", core)
        object.__setattr__(self, "shape", (basis.shape[0], basis.shape[0]))

    def matvec(self, rhs: NDArray) -> NDArray:
        values = np.asarray(rhs, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.shape[0]:
            raise ValueError("rhs does not match the low-rank operator width.")
        return self.basis @ (self.core @ (self.basis.T @ values))


@dataclass(frozen=True)
class SumBlockOperator:
    """A small sum of compact symmetric operators."""

    operators: tuple[
        SymmetricBlockOperator
        | BlockSymmetricOperator
        | SumToZeroBlockOperator
        | CenteredBlockOperator
        | LowRankSymmetricOperator,
        ...,
    ]
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self):
        if not self.operators:
            raise ValueError("A compact operator sum cannot be empty.")
        shape = self.operators[0].shape
        if any(operator.shape != shape for operator in self.operators[1:]):
            raise ValueError("All compact operators in a sum must have the same shape.")
        object.__setattr__(self, "shape", shape)

    def matvec(self, rhs: NDArray) -> NDArray:
        return sum(
            (operator.matvec(rhs) for operator in self.operators),
            start=np.zeros_like(np.asarray(rhs, dtype=np.float64)),
        )


CompactSymmetricOperator = (
    SymmetricBlockOperator
    | BlockSymmetricOperator
    | SumToZeroBlockOperator
    | CenteredBlockOperator
    | LowRankSymmetricOperator
    | SumBlockOperator
)


@dataclass(frozen=True)
class _DiagonalLowRank:
    """Internal exact ``diag(d) + U R U.T`` representation."""

    diagonal: NDArray
    basis: NDArray
    core: NDArray


@dataclass(frozen=True)
class _GeneralDiagonalLowRank:
    """Internal exact ``diag(d) + L M R.T`` representation."""

    diagonal: NDArray
    left: NDArray
    core: NDArray
    right: NDArray


def _block_operator_dlr(operator: SymmetricBlockOperator) -> _DiagonalLowRank:
    p = operator.shape[0]
    q = len(operator.small_indices)
    diagonal = np.zeros(p, dtype=np.float64)
    diagonal[operator.structured_indices] = operator.d
    if q == 0:
        return _DiagonalLowRank(
            diagonal=diagonal,
            basis=np.empty((p, 0)),
            core=np.empty((0, 0)),
        )
    small_basis = np.zeros((p, q), dtype=np.float64)
    small_basis[operator.small_indices] = np.eye(q)
    cross_basis = np.zeros((p, q), dtype=np.float64)
    cross_basis[operator.structured_indices] = operator.C
    basis = np.column_stack((small_basis, cross_basis))
    core = np.block(
        [
            [operator.A, np.eye(q)],
            [np.eye(q), np.zeros((q, q))],
        ]
    )
    return _DiagonalLowRank(diagonal=diagonal, basis=basis, core=core)


def _merge_dlr(parts: tuple[_DiagonalLowRank, ...]) -> _DiagonalLowRank:
    if not parts:
        raise ValueError("At least one diagonal-low-rank part is required.")
    diagonal = sum(
        (part.diagonal for part in parts),
        start=np.zeros_like(parts[0].diagonal),
    )
    ranks = [part.core.shape[0] for part in parts]
    if not any(ranks):
        return _DiagonalLowRank(
            diagonal=diagonal,
            basis=np.empty((len(diagonal), 0)),
            core=np.empty((0, 0)),
        )
    basis = np.column_stack([part.basis for part in parts if part.core.shape[0]])
    core = scipy.linalg.block_diag(*[part.core for part in parts if part.core.shape[0]])
    return _DiagonalLowRank(diagonal=diagonal, basis=basis, core=core)


def _operator_dlr(operator: CompactSymmetricOperator) -> _DiagonalLowRank:
    if isinstance(operator, SumBlockOperator):
        return _merge_dlr(tuple(_operator_dlr(item) for item in operator.operators))
    if isinstance(operator, LowRankSymmetricOperator):
        return _DiagonalLowRank(
            diagonal=np.zeros(operator.shape[0]),
            basis=operator.basis,
            core=operator.core,
        )
    base = _block_operator_dlr(
        operator.raw if isinstance(operator, CenteredBlockOperator) else operator
    )
    if not isinstance(operator, CenteredBlockOperator):
        return base
    update_basis = np.column_stack((operator.cross, operator.center))
    update_core = np.array(
        [
            [0.0, -1.0],
            [-1.0, operator.total],
        ]
    )
    return _merge_dlr(
        (
            base,
            _DiagonalLowRank(
                diagonal=np.zeros(operator.shape[0]),
                basis=update_basis,
                core=update_core,
            ),
        )
    )


def _trace_symmetric_dlr(left: _DiagonalLowRank, right: _DiagonalLowRank) -> float:
    value = float(left.diagonal @ right.diagonal)
    if right.core.size:
        value += float(
            np.trace(right.core @ (right.basis.T @ (left.diagonal[:, None] * right.basis)))
        )
    if left.core.size:
        value += float(
            np.trace(left.core @ (left.basis.T @ (right.diagonal[:, None] * left.basis)))
        )
    if left.core.size and right.core.size:
        overlap = left.basis.T @ right.basis
        value += float(np.trace(left.core @ overlap @ right.core @ overlap.T))
    return value


def _multiply_symmetric_dlr(
    left: _DiagonalLowRank,
    right: _DiagonalLowRank,
) -> _GeneralDiagonalLowRank:
    diagonal = left.diagonal * right.diagonal
    left_parts: list[NDArray] = []
    core_parts: list[NDArray] = []
    right_parts: list[NDArray] = []
    if right.core.size:
        left_parts.append(left.diagonal[:, None] * right.basis)
        core_parts.append(right.core)
        right_parts.append(right.basis)
    if left.core.size:
        left_parts.append(left.basis)
        core_parts.append(left.core)
        right_parts.append(right.diagonal[:, None] * left.basis)
    if left.core.size and right.core.size:
        left_parts.append(left.basis)
        core_parts.append(left.core @ (left.basis.T @ right.basis) @ right.core)
        right_parts.append(right.basis)
    if not core_parts:
        empty = np.empty((len(diagonal), 0))
        return _GeneralDiagonalLowRank(
            diagonal=diagonal,
            left=empty,
            core=np.empty((0, 0)),
            right=empty,
        )
    return _GeneralDiagonalLowRank(
        diagonal=diagonal,
        left=np.column_stack(left_parts),
        core=scipy.linalg.block_diag(*core_parts),
        right=np.column_stack(right_parts),
    )


def _general_dlr_diagonal(operator: _GeneralDiagonalLowRank) -> NDArray:
    """Return the diagonal of a general diagonal-plus-low-rank operator."""
    diagonal = np.array(operator.diagonal, dtype=np.float64, copy=True)
    if operator.core.size:
        diagonal += np.sum((operator.left @ operator.core) * operator.right, axis=1)
    return diagonal


def _general_dlr_square_diagonal(operator: _GeneralDiagonalLowRank) -> NDArray:
    """Return the diagonal of the square of a general DLR operator."""
    diagonal = np.square(operator.diagonal)
    if not operator.core.size:
        return diagonal
    low_diagonal = np.sum((operator.left @ operator.core) * operator.right, axis=1)
    diagonal += 2.0 * operator.diagonal * low_diagonal
    square_left = operator.left @ operator.core @ (operator.right.T @ operator.left) @ operator.core
    diagonal += np.sum(square_left * operator.right, axis=1)
    return diagonal


def _trace_general_product(
    left: _GeneralDiagonalLowRank,
    right: _GeneralDiagonalLowRank,
) -> float:
    value = float(left.diagonal @ right.diagonal)
    if right.core.size:
        value += float(
            np.trace(right.core @ (right.right.T @ (left.diagonal[:, None] * right.left)))
        )
    if left.core.size:
        value += float(np.trace(left.core @ (left.right.T @ (right.diagonal[:, None] * left.left))))
    if left.core.size and right.core.size:
        value += float(
            np.trace(
                left.core @ (left.right.T @ right.left) @ right.core @ (right.right.T @ left.left)
            )
        )
    return value


@dataclass(frozen=True)
class _BlockDiagonalLowRank:
    """Exact ``blockdiag(B_k) + U R U.T`` representation."""

    blocks: NDArray
    structured_indices: NDArray
    basis: NDArray
    core: NDArray
    shape: tuple[int, int]


@dataclass(frozen=True)
class _GeneralBlockDiagonalLowRank:
    """Exact ``blockdiag(B_k) + L M R.T`` representation."""

    blocks: NDArray
    structured_indices: NDArray
    left: NDArray
    core: NDArray
    right: NDArray
    shape: tuple[int, int]


def _apply_local_blocks(
    blocks: NDArray,
    structured_indices: NDArray,
    values: NDArray,
    *,
    transpose: bool = False,
) -> NDArray:
    """Apply local blocks to global vectors, leaving the small rows zero."""
    result = np.zeros_like(values, dtype=np.float64)
    local_values = values[structured_indices]
    local_blocks = blocks.transpose(0, 2, 1) if transpose else blocks
    result[structured_indices] = np.einsum(
        "kij,kjr->kir",
        local_blocks,
        local_values,
        optimize=True,
    )
    return result


def _block_operator_bdlr(operator: BlockSymmetricOperator) -> _BlockDiagonalLowRank:
    p = operator.shape[0]
    q = len(operator.small_indices)
    has_small = bool(np.any(operator.A))
    has_cross = bool(np.any(operator.C))
    if q == 0 or (not has_small and not has_cross):
        return _BlockDiagonalLowRank(
            blocks=operator.D,
            structured_indices=operator.structured_indices,
            basis=np.empty((p, 0)),
            core=np.empty((0, 0)),
            shape=operator.shape,
        )
    small_basis = np.zeros((p, q), dtype=np.float64)
    small_basis[operator.small_indices] = np.eye(q)
    if not has_cross:
        return _BlockDiagonalLowRank(
            blocks=operator.D,
            structured_indices=operator.structured_indices,
            basis=small_basis,
            core=operator.A,
            shape=operator.shape,
        )
    cross_basis = np.zeros((p, q), dtype=np.float64)
    cross_basis[operator.structured_indices] = operator.C
    basis = np.column_stack((small_basis, cross_basis))
    small_core = operator.A if has_small else np.zeros_like(operator.A)
    core = np.block(
        [
            [small_core, np.eye(q)],
            [np.eye(q), np.zeros((q, q))],
        ]
    )
    return _BlockDiagonalLowRank(
        blocks=operator.D,
        structured_indices=operator.structured_indices,
        basis=basis,
        core=core,
        shape=operator.shape,
    )


def _sum_to_zero_operator_bdlr(
    operator: SumToZeroBlockOperator,
) -> _BlockDiagonalLowRank:
    """Convert raw constrained blocks to public block-diagonal-plus-low-rank form."""
    base = BlockSymmetricOperator(
        A=operator.A,
        C=operator.C[:-1] - operator.C[-1:],
        D=operator.D[:-1],
        small_indices=operator.small_indices,
        structured_indices=operator.structured_indices,
    )
    last_basis = np.zeros((operator.shape[0], operator.block_size))
    for indices in operator.structured_indices:
        last_basis[indices] = np.eye(operator.block_size)
    last = _BlockDiagonalLowRank(
        blocks=np.zeros_like(operator.D[:-1]),
        structured_indices=operator.structured_indices,
        basis=last_basis,
        core=operator.D[-1],
        shape=operator.shape,
    )
    return _merge_bdlr((_block_operator_bdlr(base), last))


def _empty_block_part(
    shape: tuple[int, int],
    structured_indices: NDArray,
) -> _BlockDiagonalLowRank:
    n_levels, block_size = structured_indices.shape
    return _BlockDiagonalLowRank(
        blocks=np.zeros((n_levels, block_size, block_size)),
        structured_indices=structured_indices,
        basis=np.empty((shape[0], 0)),
        core=np.empty((0, 0)),
        shape=shape,
    )


def _merge_bdlr(parts: tuple[_BlockDiagonalLowRank, ...]) -> _BlockDiagonalLowRank:
    if not parts:
        raise ValueError("At least one block-diagonal-low-rank part is required.")
    reference = parts[0]
    if any(
        part.shape != reference.shape
        or not np.array_equal(part.structured_indices, reference.structured_indices)
        for part in parts[1:]
    ):
        raise ValueError("Block-diagonal-low-rank parts must share one coefficient layout.")
    blocks = sum((part.blocks for part in parts), start=np.zeros_like(reference.blocks))
    active = [part for part in parts if part.core.size]
    if not active:
        return _BlockDiagonalLowRank(
            blocks=blocks,
            structured_indices=reference.structured_indices,
            basis=np.empty((reference.shape[0], 0)),
            core=np.empty((0, 0)),
            shape=reference.shape,
        )
    return _BlockDiagonalLowRank(
        blocks=blocks,
        structured_indices=reference.structured_indices,
        basis=np.column_stack([part.basis for part in active]),
        core=scipy.linalg.block_diag(*[part.core for part in active]),
        shape=reference.shape,
    )


def _operator_bdlr(
    operator: CompactSymmetricOperator,
    structured_indices: NDArray,
) -> _BlockDiagonalLowRank:
    """Convert a compact operator to matching block-diagonal-plus-low-rank form."""
    if isinstance(operator, SumBlockOperator):
        return _merge_bdlr(
            tuple(_operator_bdlr(item, structured_indices) for item in operator.operators)
        )
    if isinstance(operator, LowRankSymmetricOperator):
        empty = _empty_block_part(operator.shape, structured_indices)
        return _BlockDiagonalLowRank(
            blocks=empty.blocks,
            structured_indices=structured_indices,
            basis=operator.basis,
            core=operator.core,
            shape=operator.shape,
        )
    raw = operator.raw if isinstance(operator, CenteredBlockOperator) else operator
    if not isinstance(raw, BlockSymmetricOperator | SumToZeroBlockOperator):
        raise TypeError("BlockSchurFactor requires block-compatible compact operators.")
    if not np.array_equal(raw.structured_indices, structured_indices):
        raise ValueError("Compact operator has a different structured block layout.")
    base = (
        _sum_to_zero_operator_bdlr(raw)
        if isinstance(raw, SumToZeroBlockOperator)
        else _block_operator_bdlr(raw)
    )
    if not isinstance(operator, CenteredBlockOperator):
        return base
    update = _empty_block_part(operator.shape, structured_indices)
    return _merge_bdlr(
        (
            base,
            _BlockDiagonalLowRank(
                blocks=update.blocks,
                structured_indices=structured_indices,
                basis=np.column_stack((operator.cross, operator.center)),
                core=np.array(
                    [
                        [0.0, -1.0],
                        [-1.0, operator.total],
                    ]
                ),
                shape=operator.shape,
            ),
        )
    )


def _trace_symmetric_bdlr(
    left: _BlockDiagonalLowRank,
    right: _BlockDiagonalLowRank,
) -> float:
    if left.shape != right.shape or not np.array_equal(
        left.structured_indices,
        right.structured_indices,
    ):
        raise ValueError("Block-diagonal-low-rank layouts must match.")
    value = float(np.einsum("kij,kji->", left.blocks, right.blocks, optimize=True))
    if right.core.size:
        left_applied = _apply_local_blocks(
            left.blocks,
            left.structured_indices,
            right.basis,
        )
        value += float(np.trace(right.core @ (right.basis.T @ left_applied)))
    if left.core.size:
        right_applied = _apply_local_blocks(
            right.blocks,
            right.structured_indices,
            left.basis,
        )
        value += float(np.trace(left.core @ (left.basis.T @ right_applied)))
    if left.core.size and right.core.size:
        overlap = left.basis.T @ right.basis
        value += float(np.trace(left.core @ overlap @ right.core @ overlap.T))
    return value


def _multiply_symmetric_bdlr(
    left: _BlockDiagonalLowRank,
    right: _BlockDiagonalLowRank,
) -> _GeneralBlockDiagonalLowRank:
    if left.shape != right.shape or not np.array_equal(
        left.structured_indices,
        right.structured_indices,
    ):
        raise ValueError("Block-diagonal-low-rank layouts must match.")
    blocks = np.einsum("kij,kjl->kil", left.blocks, right.blocks, optimize=True)
    left_parts: list[NDArray] = []
    core_parts: list[NDArray] = []
    right_parts: list[NDArray] = []
    if right.core.size:
        left_parts.append(_apply_local_blocks(left.blocks, left.structured_indices, right.basis))
        core_parts.append(right.core)
        right_parts.append(right.basis)
    if left.core.size:
        left_parts.append(left.basis)
        core_parts.append(left.core)
        right_parts.append(
            _apply_local_blocks(
                right.blocks,
                right.structured_indices,
                left.basis,
                transpose=True,
            )
        )
    if left.core.size and right.core.size:
        left_parts.append(left.basis)
        core_parts.append(left.core @ (left.basis.T @ right.basis) @ right.core)
        right_parts.append(right.basis)
    if not core_parts:
        empty = np.empty((left.shape[0], 0))
        return _GeneralBlockDiagonalLowRank(
            blocks=blocks,
            structured_indices=left.structured_indices,
            left=empty,
            core=np.empty((0, 0)),
            right=empty,
            shape=left.shape,
        )
    return _GeneralBlockDiagonalLowRank(
        blocks=blocks,
        structured_indices=left.structured_indices,
        left=np.column_stack(left_parts),
        core=scipy.linalg.block_diag(*core_parts),
        right=np.column_stack(right_parts),
        shape=left.shape,
    )


def _multiply_symmetric_bdlr_coalesced(
    left: _BlockDiagonalLowRank,
    right: _BlockDiagonalLowRank,
) -> _GeneralBlockDiagonalLowRank:
    """Multiply BDLR operators while coalescing repeated low-rank bases."""
    if left.shape != right.shape or not np.array_equal(
        left.structured_indices,
        right.structured_indices,
    ):
        raise ValueError("Block-diagonal-low-rank layouts must match.")
    blocks = np.einsum("kij,kjl->kil", left.blocks, right.blocks, optimize=True)
    if not left.core.size and not right.core.size:
        empty = np.empty((left.shape[0], 0))
        return _GeneralBlockDiagonalLowRank(
            blocks=blocks,
            structured_indices=left.structured_indices,
            left=empty,
            core=np.empty((0, 0)),
            right=empty,
            shape=left.shape,
        )
    if not left.core.size:
        return _GeneralBlockDiagonalLowRank(
            blocks=blocks,
            structured_indices=left.structured_indices,
            left=_apply_local_blocks(
                left.blocks,
                left.structured_indices,
                right.basis,
            ),
            core=right.core,
            right=right.basis,
            shape=left.shape,
        )
    right_local_left = _apply_local_blocks(
        right.blocks,
        right.structured_indices,
        left.basis,
        transpose=True,
    )
    if not right.core.size:
        return _GeneralBlockDiagonalLowRank(
            blocks=blocks,
            structured_indices=left.structured_indices,
            left=left.basis,
            core=left.core,
            right=right_local_left,
            shape=left.shape,
        )

    # In (B + U R U') (D + V S V'), coalesce the two occurrences
    # of U and V instead of representing the three updates independently.
    left_local_right = _apply_local_blocks(
        left.blocks,
        left.structured_indices,
        right.basis,
    )
    left_width = left.core.shape[0]
    right_width = right.core.shape[0]
    core = np.zeros(
        (right_width + left_width, right_width + left_width),
        dtype=np.float64,
    )
    core[:right_width, :right_width] = right.core
    core[right_width:, :right_width] = left.core @ (left.basis.T @ right.basis) @ right.core
    core[right_width:, right_width:] = left.core
    return _GeneralBlockDiagonalLowRank(
        blocks=blocks,
        structured_indices=left.structured_indices,
        left=np.column_stack((left_local_right, left.basis)),
        core=core,
        right=np.column_stack((right.basis, right_local_left)),
        shape=left.shape,
    )


def _general_bdlr_diagonal(operator: _GeneralBlockDiagonalLowRank) -> NDArray:
    diagonal = np.zeros(operator.shape[0], dtype=np.float64)
    diagonal[operator.structured_indices] = np.diagonal(
        operator.blocks,
        axis1=1,
        axis2=2,
    )
    if operator.core.size:
        diagonal += np.sum((operator.left @ operator.core) * operator.right, axis=1)
    return diagonal


def _general_bdlr_square_diagonal(operator: _GeneralBlockDiagonalLowRank) -> NDArray:
    square_blocks = np.einsum(
        "kij,kjl->kil",
        operator.blocks,
        operator.blocks,
        optimize=True,
    )
    diagonal = np.zeros(operator.shape[0], dtype=np.float64)
    diagonal[operator.structured_indices] = np.diagonal(
        square_blocks,
        axis1=1,
        axis2=2,
    )
    if not operator.core.size:
        return diagonal
    block_left = _apply_local_blocks(
        operator.blocks,
        operator.structured_indices,
        operator.left,
    )
    block_transpose_right = _apply_local_blocks(
        operator.blocks,
        operator.structured_indices,
        operator.right,
        transpose=True,
    )
    diagonal += np.sum((block_left @ operator.core) * operator.right, axis=1)
    diagonal += np.sum((operator.left @ operator.core) * block_transpose_right, axis=1)
    low_square_left = (
        operator.left @ operator.core @ (operator.right.T @ operator.left) @ operator.core
    )
    diagonal += np.sum(low_square_left * operator.right, axis=1)
    return diagonal


def _trace_general_bdlr_product(
    left: _GeneralBlockDiagonalLowRank,
    right: _GeneralBlockDiagonalLowRank,
) -> float:
    if left.shape != right.shape or not np.array_equal(
        left.structured_indices,
        right.structured_indices,
    ):
        raise ValueError("General block-diagonal-low-rank layouts must match.")
    value = float(np.einsum("kij,kji->", left.blocks, right.blocks, optimize=True))
    if right.core.size:
        left_applied = _apply_local_blocks(
            left.blocks,
            left.structured_indices,
            right.left,
        )
        value += float(np.trace(right.core @ (right.right.T @ left_applied)))
    if left.core.size:
        right_applied = _apply_local_blocks(
            right.blocks,
            right.structured_indices,
            left.left,
        )
        value += float(np.trace(left.core @ (left.right.T @ right_applied)))
    if left.core.size and right.core.size:
        value += float(
            np.trace(
                left.core @ (left.right.T @ right.left) @ right.core @ (right.right.T @ left.left)
            )
        )
    return value


def materialize_compact_operator(operator: CompactSymmetricOperator) -> NDArray:
    """Materialize a compact operator for dense-reference paths only."""
    return operator.matvec(np.eye(operator.shape[0]))


def compact_operator_diagonal(
    operator: CompactSymmetricOperator,
) -> NDArray:
    """Return an exact compact-operator diagonal in O(Kq + q²) memory."""
    if isinstance(operator, SumBlockOperator):
        return sum(
            (compact_operator_diagonal(item) for item in operator.operators),
            start=np.zeros(operator.shape[0]),
        )
    if isinstance(operator, LowRankSymmetricOperator):
        return np.sum((operator.basis @ operator.core) * operator.basis, axis=1)
    raw = operator.raw if isinstance(operator, CenteredBlockOperator) else operator
    diagonal = np.empty(raw.shape[0], dtype=np.float64)
    diagonal[raw.small_indices] = np.diag(raw.A)
    if isinstance(raw, BlockSymmetricOperator):
        diagonal[raw.structured_indices] = np.diagonal(raw.D, axis1=1, axis2=2)
    elif isinstance(raw, SumToZeroBlockOperator):
        diagonal[raw.structured_indices] = np.diagonal(
            raw.D[:-1] + raw.D[-1:],
            axis1=1,
            axis2=2,
        )
    else:
        diagonal[raw.structured_indices] = raw.d
    if isinstance(operator, CenteredBlockOperator):
        diagonal = (
            diagonal - 2.0 * operator.cross * operator.center + operator.total * operator.center**2
        )
    return diagonal
