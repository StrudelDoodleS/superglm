"""Compact structured operators and low-rank algebra."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import (
    adjoint_sum_to_zero_blocks,
    expand_sum_to_zero_blocks,
)
from superglm.solvers._structured.retired import module_getattr

if TYPE_CHECKING:
    from superglm.solvers._structured.nested import NestedDataOperator


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
        if not all(np.all(np.isfinite(values)) for values in (self.A, self.C, self.D)):
            # Above the symmetry check for the reason SumToZeroBlockOperator and
            # both Schur factors put it there: a NaN fails `np.allclose` against
            # its own transpose, so without this a NaN local block is refused as
            # an ASYMMETRIC one -- a structural verdict for a condition the
            # iterate's weights caused. This is the `fs` operator, and `fs` is
            # the default basis, so it is the path most likely to reach here.
            raise np.linalg.LinAlgError("Block operator blocks must be finite.")
        if not np.allclose(self.D, self.D.transpose(0, 2, 1), rtol=0.0, atol=1e-13):
            raise ValueError("Every local D block must be symmetric.")
        all_indices = np.concatenate([self.small_indices, self.structured_indices.ravel()])
        size = len(all_indices)
        # O(p) partition test (perf F6), in the order and with the verdicts of a
        # unique count then a sort: duplicates first, then coverage of 0..p-1.
        if size:
            low = int(all_indices.min())
            if np.any(np.bincount(all_indices - low) > 1):
                raise ValueError("small_indices and structured_indices must be disjoint.")
            if low != 0 or int(all_indices.max()) != size - 1:
                raise ValueError("Structured index partitions must cover every coefficient once.")
        object.__setattr__(self, "shape", (size, size))

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

    raw: (
        SymmetricBlockOperator
        | BlockSymmetricOperator
        | SumToZeroBlockOperator
        | NestedDataOperator
    )
    cross: NDArray
    total: float
    center: NDArray
    raw_structured_cross: NDArray | None = None
    # Centred column norms taken from the rows where the moments cancel below
    # round-off, NaN elsewhere (geometry.cancelled_column_row_norms).
    row_column_norm: NDArray | None = None
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
        if self.row_column_norm is not None:
            row_column_norm = np.array(self.row_column_norm, dtype=np.float64, copy=True)
            if row_column_norm.shape != (p,):
                raise ValueError("row_column_norm must match the coefficient width.")
            row_column_norm.setflags(write=False)
            object.__setattr__(self, "row_column_norm", row_column_norm)
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
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(core)):
            # Above the symmetry check, as everywhere else here, and the basis
            # is the half that carries it. At this class's one construction site
            # (`reml_w_correction`) the basis is column_stack(dmean_i, dmean_j)
            # and had no check of any kind -- not finiteness, not symmetry. The
            # core there is [[0, -sum_w], [-sum_w, 0]], symmetric by
            # construction and built from a sum_w already certified finite and
            # positive upstream, so it is close to unreachable; a NaN in it
            # would have failed the check below under the wrong name, and an inf
            # would have passed it and been accepted outright.
            raise np.linalg.LinAlgError("Low-rank operator basis and core must be finite.")
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
if TYPE_CHECKING:
    CompactSymmetricOperator = CompactSymmetricOperator | NestedDataOperator


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


def _block_operator_bdlr(
    operator: BlockSymmetricOperator,
    *,
    local_only: bool = False,
) -> _BlockDiagonalLowRank:
    p = operator.shape[0]
    q = len(operator.small_indices)
    has_small = bool(np.any(operator.A))
    has_cross = bool(np.any(operator.C))
    if q == 0 or local_only or (not has_small and not has_cross):
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
    """Convert raw constrained blocks to public block-diagonal-plus-low-rank form.

    This constructs a fresh ``BlockSymmetricOperator``, so its finiteness guard
    can raise from here -- lazily, on a path (``_operator_bdlr`` ->
    ``trace_inverse_operator``) that no seam covers. It is a backstop rather
    than a live refusal: ``operator`` passed its own finiteness guard at
    construction, ``D[:-1]`` is a slice of already-finite blocks, and
    ``C[:-1] - C[-1:]`` is a difference of finite quantities, so reaching it
    needs that subtraction to overflow. Written down rather than seamed,
    because a speculative catch around the Hessian assembly would be wide
    enough to swallow the structural refusals that path is supposed to surface.
    """
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
    *,
    local_only: bool = False,
) -> _BlockDiagonalLowRank:
    """Convert a compact operator to matching block-diagonal-plus-low-rank form.

    ``local_only`` drops a block operator's dense-small and cross blocks,
    which never reach the structured-by-structured block, so the result agrees
    with ``operator`` on that block alone -- all ``_bdlr_cross_traces`` reads;
    sum-to-zero operators keep their exact full form.
    """
    if isinstance(operator, SumBlockOperator):
        return _merge_bdlr(
            tuple(
                _operator_bdlr(item, structured_indices, local_only=local_only)
                for item in operator.operators
            )
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
        else _block_operator_bdlr(raw, local_only=local_only)
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
        value += float(np.sum(right.core * (right.basis.T @ left_applied).T))
    if left.core.size:
        right_applied = _apply_local_blocks(
            right.blocks,
            right.structured_indices,
            left.basis,
        )
        value += float(np.sum(left.core * (left.basis.T @ right_applied).T))
    if left.core.size and right.core.size:
        overlap = left.basis.T @ right.basis
        value += float(np.sum((left.core @ overlap) * (right.core @ overlap.T).T))
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
        value += float(np.sum(right.core * (right.right.T @ left_applied).T))
    if left.core.size:
        right_applied = _apply_local_blocks(
            right.blocks,
            right.structured_indices,
            left.left,
        )
        value += float(np.sum(left.core * (left.right.T @ right_applied).T))
    if left.core.size and right.core.size:
        # tr(AB) = sum(A * B.T): never form a square product to read its trace.
        value += float(
            np.sum(
                (left.core @ (left.right.T @ right.left))
                * (right.core @ (right.right.T @ left.left)).T
            )
        )
    return value


def _direction_stacks(
    basis: NDArray,
    core: NDArray,
    structured_indices: NDArray,
    directions: Iterable[tuple[NDArray, _BlockDiagonalLowRank]],
) -> tuple[NDArray, NDArray, list]:
    """Keep each ``W = O U`` on the structured rows, ``R U' W``, and O's local part."""
    borders, cores, local_parts = [], [], []
    for product, local in directions:
        borders.append(product[structured_indices])
        cores.append(core @ (basis.T @ product))
        local_parts.append(local)
    return np.stack(borders), np.stack(cores), local_parts


def _pairwise_traces(parts: Sequence, trace: Callable[[Any, Any], float]) -> NDArray:
    traces = np.empty((len(parts), len(parts)))
    for i, j in itertools.combinations_with_replacement(range(len(parts)), 2):
        traces[i, j] = traces[j, i] = trace(parts[i], parts[j])
    return traces


def _schur_cross_traces(
    borders: NDArray,
    weight: Callable[[NDArray], NDArray],
    cores: NDArray,
    local_traces: NDArray,
) -> NDArray:
    """Assemble every ``trace(Z O_i Z O_j)`` for ``Z = Z_local + U R U'``.

    With ``W_i = O_i U`` and ``T_i = U' W_i`` each trace splits exactly into
    ``tr(Z_local O_i Z_local O_j) + tr(R W_i' Z_local W_j) + tr(R W_j' Z_local W_i)
    + tr(R T_i R T_j)``.  ``Z_local`` lives on the structured rows only, so the
    middle pair needs just ``W`` there (``borders``) and ``Z_local W R``,
    which ``weight`` forms one direction at a time so that only one is alive
    beside the stack; ``cores`` holds ``R T_i``.  Every pair is then two inner
    products, O(Kq + q^2), where re-forming both ``H^-1 O`` products per pair
    (diagonal plus low rank of width 4q-5q) cost O(p q^2) Grams (Wood 2008,
    Appendix C: store per-parameter products, read pairwise traces off them
    with ``tr(AB) = sum(A * B')``).
    """
    m = len(cores)
    flat = borders.reshape(m, -1)
    border_traces = np.stack([flat @ weight(border).ravel() for border in borders])
    core_traces = cores.reshape(m, -1) @ cores.transpose(0, 2, 1).reshape(m, -1).T
    traces = local_traces + border_traces + border_traces.T + core_traces
    return 0.5 * (traces + traces.T)


def _bdlr_cross_traces(
    inverse: _BlockDiagonalLowRank,
    directions: Iterable[tuple[NDArray, _BlockDiagonalLowRank]],
) -> NDArray:
    """Return ``trace(Z O_i Z O_j)`` for every pair, ``Z = blockdiag(Z_k) + U R U'``.

    ``directions`` yields ``(O_i U, O_i's local part)`` once per direction;
    ``Z_local`` has dense local blocks on the structured rows.
    """
    structured = inverse.structured_indices
    borders, cores, local_parts = _direction_stacks(
        inverse.basis, inverse.core, structured, directions
    )
    local = [
        _GeneralBlockDiagonalLowRank(
            blocks=inverse.blocks @ part.blocks,
            structured_indices=structured,
            left=_apply_local_blocks(inverse.blocks, structured, part.basis),
            core=part.core,
            right=part.basis,
            shape=inverse.shape,
        )
        for part in local_parts
    ]
    return _schur_cross_traces(
        borders,
        lambda border: inverse.blocks @ (border @ inverse.core),
        cores,
        _pairwise_traces(local, _trace_general_bdlr_product),
    )


def materialize_compact_operator(operator: CompactSymmetricOperator) -> NDArray:
    """Materialize a compact operator for dense-reference paths only."""
    return operator.matvec(np.eye(operator.shape[0]))


def compact_operator_diagonal(
    operator: CompactSymmetricOperator,
) -> NDArray:
    """Return an exact compact-operator diagonal in O(Kq + q²) memory."""
    from superglm.solvers._structured.nested import NestedDataOperator

    if isinstance(operator, SumBlockOperator):
        return sum(
            (compact_operator_diagonal(item) for item in operator.operators),
            start=np.zeros(operator.shape[0]),
        )
    if isinstance(operator, LowRankSymmetricOperator):
        return np.sum((operator.basis @ operator.core) * operator.basis, axis=1)
    raw = operator.raw if isinstance(operator, CenteredBlockOperator) else operator
    diagonal = np.empty(raw.shape[0], dtype=np.float64)
    if isinstance(raw, NestedDataOperator) and isinstance(operator, CenteredBlockOperator):
        # the border entries come from the leaf statistics below; never form A
        diagonal[raw.small_indices] = 0.0
    else:
        diagonal[raw.small_indices] = np.diag(raw.A)
    if isinstance(raw, BlockSymmetricOperator):
        diagonal[raw.structured_indices] = np.diagonal(raw.D, axis1=1, axis2=2)
    elif isinstance(raw, SumToZeroBlockOperator):
        diagonal[raw.structured_indices] = np.diagonal(
            raw.D[:-1] + raw.D[-1:],
            axis1=1,
            axis2=2,
        )
    elif isinstance(raw, NestedDataOperator):
        diagonal[raw.structured_indices] = np.concatenate(raw.tree.subtree_sum(raw.leaf.weight))
    else:
        diagonal[raw.structured_indices] = raw.d
    if isinstance(operator, CenteredBlockOperator):
        centred = (
            diagonal - 2.0 * operator.cross * operator.center + operator.total * operator.center**2
        )
        if isinstance(raw, NestedDataOperator):
            # The border rows' own centre c (design §3.2) keeps the offsets out:
            # sum a (x - m)^2 = sum a (x - c)^2 - 2 (m - c) sum a (x - c) + (m - c)^2 sum a,
            # every term free of the columns' offsets, where the raw identity
            # above subtracts moments of size |x|^2.
            leaf = raw.leaf
            about = np.diag(leaf.within) + leaf.weight @ leaf.mean**2
            sums = leaf.weight @ leaf.mean
            if leaf.deviation is not None:
                about = about + 2.0 * np.sum(leaf.deviation * leaf.mean, axis=0)
                sums = sums + np.sum(leaf.deviation, axis=0)
            shift = operator.center[raw.small_indices] - leaf.center
            centred[raw.small_indices] = about - 2.0 * shift * sums + operator.total * shift**2
        diagonal = centred
    return diagonal


# Retired by the one engine (design §3.12): the scalar factor's diagonal-plus-
# low-rank inverse, which a model saved by v0.35.0 pickles.  Importable at its
# pickled path as an inert stand-in; release 0.37.0 may drop it.
__getattr__ = module_getattr(__name__, frozenset({"_DiagonalLowRank"}))
