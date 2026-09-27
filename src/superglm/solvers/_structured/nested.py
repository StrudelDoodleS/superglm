"""Nested random-effect chain elimination (fix D).

The mathematics, accuracy bounds and refusal rules are in
``notes/research/2026-09-26-nested-random-effect-elimination.md`` (cited below
as §N) and the exact contracts in
``notes/research/lean-gaussian-certificate/NestedElimination.lean``.

Geometry.  A chain is ``L >= 2`` random-effect terms, each strictly nested in
the next coarser one.  Level 0 is the coarsest (its nodes are the roots) and
level ``L - 1`` the finest (the leaves, the only level whose codes touch
rows).  Every tree node carries one coefficient.  Node order is level-major,
coarsest first: node ``tree.offsets[j] + u`` is local node ``u`` of level
``j``, and ``structured_indices[tree.offsets[j] + u]`` is its global
coefficient index.  The border ("small") block holds every other coefficient,
crossed random effects included; in augmented coordinates the intercept is
border column 0 and global index 0.

Every operator here is leaf-form (§3.6).  With ``M`` the leaf-to-node
incidence (``M[l, u] = 1`` iff ``u`` is ``l`` or an ancestor of ``l``), row
weights ``a_r`` and border rows ``x_r``,

    O = [[M' diag(a_leaf) M, M' C_leaf], [C_leaf' M, A]],
    a_leaf[l] = sum_{r in l} a_r,  C_leaf[l] = sum_{r in l} a_r x_r,  A = X_b' diag(a) X_b,

and the penalized Hessian is ``H = O_w + blockdiag(diag(lambda_node), S_b)``.

Refusals follow the structured-factor convention.  ``np.linalg.LinAlgError``
reports what an iterate's numbers caused (non-finite statistics, a tree pivot
at or below its cancellation floor, material negative Schur curvature, a
coupled Schur null space, exact intercept aliasing); callers such as the
observed-geometry build rely on that type.  ``ValueError`` reports a
malformed call (shapes, partitions, coordinates, an operator built about
other leaf means).  ``TypeError`` reports an operator kind the nested factor
does not represent (for example a single-level ``SymmetricBlockOperator``).

The factors are verified against exact rational references in
``tests/test_nested_schur_factor.py``; ``tests/test_nested_structured_fit.py``
runs complete fits against the dense backend.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.typing import NDArray

from superglm.solvers._structured.factors import _penalty_product
from superglm.solvers._structured.operators import (
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    SumBlockOperator,
)
from superglm.solvers.hessian_factor import _component_indices, _component_omega
from superglm.types import PenaltyComponent

if TYPE_CHECKING:
    from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan
    from superglm.group_matrix import GroupMatrix
    from superglm.solvers._structured.factors import DerivativeDirection
    from superglm.solvers._structured.operators import CompactSymmetricOperator
    from superglm.types import GroupSlice

# §3.7: a tree pivot refuses when D_u <= gamma_u with
# gamma_u = 10 eps (fan_u + 2) (|w_u| + sum_{c in ch(u)} |s_c| + lambda_u).
_TREE_PIVOT_FLOOR_FACTOR = 10.0
# §3.7: the Cholesky of the Jacobi-scaled Schur complement Q_s falls back to
# its SVD when a scaled pivot^2 <= 20 q eps or the probe residual is too large.
_SCALED_PIVOT_FLOOR_FACTOR = 20.0
_SCALED_PROBE_RESIDUAL = 1e-6
# §3.7: the SVD fallback on Q_s truncates below 1e-10 sigma_max(Q_s).
_SCALED_SVD_RCOND = 1e-10


def _frozen(values, dtype) -> NDArray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class NestedTree:
    """Forest of a nested chain: level sizes and child-to-parent pointers.

    ``sizes[j]`` is ``K_j``, the coefficient count of level ``j`` (observed and
    unobserved levels alike).  ``parent[j]`` is an ``intp`` array of length
    ``K_j``: for ``j >= 1`` the local index at level ``j - 1`` of each node's
    parent, in ``[0, K_{j-1})``; ``parent[0]`` is all ``-1`` (roots).  A node
    with no weighted row below it may point at any parent (the layout uses 0):
    its accumulated weight is exactly zero, so ``sigma = omega / D = 0`` severs
    it (§3.7, §10).  ``offsets[j]`` is the first node-order position of level
    ``j`` and ``offsets[L] == n_nodes``.

    ``subtree_sum`` is ``M'`` extended to every level and ``path_sum`` is ``M``
    (§3.1); both accept one value or a row of values per node.  Equality is
    identity; two trees describe the same forest when their sizes and parent
    arrays are equal.
    """

    sizes: tuple[int, ...]
    parent: tuple[NDArray, ...]
    offsets: NDArray = field(init=False)
    n_nodes: int = field(init=False)

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.sizes)
        if len(sizes) < 2 or min(sizes) < 1:
            raise ValueError("A nested chain needs at least two non-empty levels.")
        if len(self.parent) != len(sizes):
            raise ValueError("parent must hold one array per level.")
        parent = tuple(_frozen(values, np.intp) for values in self.parent)
        if parent[0].shape != (sizes[0],) or np.any(parent[0] != -1):
            raise ValueError("parent[0] must mark every root with -1.")
        for j in range(1, len(sizes)):
            if parent[j].shape != (sizes[j],):
                raise ValueError(f"parent[{j}] must have shape ({sizes[j]},).")
            if np.any((parent[j] < 0) | (parent[j] >= sizes[j - 1])):
                raise ValueError(f"parent[{j}] points outside level {j - 1}.")
        offsets = _frozen(np.concatenate(([0], np.cumsum(sizes))), np.intp)
        object.__setattr__(self, "sizes", sizes)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "n_nodes", int(offsets[-1]))

    @property
    def depth(self) -> int:
        """Number of chain levels ``L``."""
        return len(self.sizes)

    @cached_property
    def _child_sums(self) -> tuple[scipy.sparse.csr_array, ...]:
        """Per level ``j >= 1`` the ``K_{j-1} x K_j`` child-to-parent incidence.

        Cache contract: owned by this immutable tree and living exactly as long
        as it; the parent arrays never change, so nothing invalidates it.
        """
        return tuple(
            scipy.sparse.csr_array(
                (np.ones(size), (parent, np.arange(size))),
                shape=(previous, size),
            )
            for previous, size, parent in zip(
                self.sizes[:-1], self.sizes[1:], self.parent[1:], strict=True
            )
        )

    def subtree_sum(self, leaf_values: NDArray) -> tuple[NDArray, ...]:
        """Return per level the sums of ``leaf_values`` over each node's leaves.

        ``leaf_values`` has shape ``(K_{L-1},)`` or ``(K_{L-1}, m)``; level ``j``
        of the result has shape ``(K_j,)`` or ``(K_j, m)``.  The leaf level is
        returned as given (converted to float64).
        """
        values = np.asarray(leaf_values, dtype=np.float64)
        if values.shape[:1] != (self.sizes[-1],):
            raise ValueError(f"leaf_values must have {self.sizes[-1]} rows.")
        sums = [values]
        for child_sum in reversed(self._child_sums):
            sums.append(child_sum @ sums[-1])
        return tuple(reversed(sums))

    def path_sum(self, per_level: Sequence[NDArray]) -> NDArray:
        """Return per leaf the sum of node values over its ancestors-or-self.

        ``per_level[j]`` has shape ``(K_j,)`` or ``(K_j, m)``; the result has the
        leaf level's shape.
        """
        if len(per_level) != self.depth:
            raise ValueError("per_level must hold one array per level.")
        total = np.asarray(per_level[0], dtype=np.float64)
        for parent, values in zip(self.parent[1:], per_level[1:], strict=True):
            total = total[parent] + np.asarray(values, dtype=np.float64)
        return total

    def split(self, node_values: NDArray) -> tuple[NDArray, ...]:
        """Split node-order values ``(n_nodes, ...)`` into per-level views."""
        values = np.asarray(node_values)
        if values.shape[:1] != (self.n_nodes,):
            raise ValueError(f"node_values must have {self.n_nodes} rows.")
        return tuple(values[start:stop] for start, stop in zip(self.offsets[:-1], self.offsets[1:]))


@dataclass(frozen=True, eq=False)
class NestedStructuredLayout:
    """Design partition for one nested chain beside a dense border (§4).

    Built once per design and chain by ``layout.build_nested_structured_layout``
    and cached on the ``DesignMatrix`` like the single-level layouts.  Chain
    fields run coarsest to finest; ``chain_group_indices[-1]`` is the leaf
    group, the one whose ``RandomEffectGroupMatrix.codes`` are the leaf codes.
    ``level_indices[j]`` holds the global design coefficient indices of level
    ``j`` (its ``GroupSlice`` range) and ``structured_indices`` their
    concatenation in node order.  ``leaf_order`` holds the rows sorted by leaf
    code (stable) and ``leaf_starts`` ``(K + 1,)`` each leaf's start in it, so
    ``leaf_order[leaf_starts[l]:leaf_starts[l + 1]]`` are the rows of leaf
    ``l`` in row order; both are read-only and shared with the nesting cache
    entry that holds the tree.  The border fields
    mean exactly what they mean on ``ScalarStructuredLayout``: every group
    outside the chain, crossed random effects included.

    There is deliberately no ``dominant_group_index``: code written for one
    dominant group fails on this layout instead of treating the leaf as the
    only structured level.
    """

    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    tree: NestedTree
    level_indices: tuple[NDArray, ...]
    leaf_order: NDArray
    leaf_starts: NDArray
    small_group_indices: tuple[int, ...]
    small_matrices: tuple[GroupMatrix, ...]
    local_groups: tuple[GroupSlice, ...]
    small_indices: NDArray
    dense_small_matrix: NDArray | None
    small_execution_plan: MatrixExecutionPlan | None
    structured_indices: NDArray = field(init=False)

    def __post_init__(self) -> None:
        depth = self.tree.depth
        if len(self.chain_group_indices) != depth or len(self.chain_group_names) != depth:
            raise ValueError("Chain group indices and names must match the tree depth.")
        levels = tuple(_frozen(values, np.intp) for values in self.level_indices)
        if tuple(len(values) for values in levels) != self.tree.sizes:
            raise ValueError("level_indices must match the tree level sizes.")
        object.__setattr__(self, "level_indices", levels)
        object.__setattr__(self, "small_indices", _frozen(self.small_indices, np.intp))
        object.__setattr__(self, "structured_indices", _frozen(np.concatenate(levels), np.intp))
        if self.dense_small_matrix is not None:
            object.__setattr__(
                self, "dense_small_matrix", _frozen(self.dense_small_matrix, np.float64)
            )

    @property
    def leaf_group_index(self) -> int:
        return self.chain_group_indices[-1]

    @property
    def leaf_group_name(self) -> str:
        return self.chain_group_names[-1]

    @cached_property
    def border_center(self) -> NDArray:
        """The global centre ``c`` ``(q,)``: the row mean of each border column whose
        offset exceeds its spread, 0 elsewhere.

        The row pass works on ``x - c``, so every leaf statistic rounds at
        ``|x - c|`` rather than ``|x|`` (§3.4); any fixed ``c`` is exact algebra.
        Centring pays off when ``|mean| > sd``; it would cost a sparse column
        (an indicator's 0 becomes ``-p``) and move its exact zeros, which keep
        a rounding-weight null direction on its own coordinate (§3.7), onto a
        combination with the intercept.  Cache contract: owned by this
        immutable layout and living as long as it; the border matrices never
        change, so nothing invalidates it.  Each block gives its own column
        sums and Gram diagonal; no cross-block product is formed.
        """
        if not self.small_matrices:
            return _frozen(np.zeros(0), np.float64)
        rows = self.small_matrices[0].shape[0]
        ones = np.ones(rows)
        mean = np.concatenate([matrix.rmatvec(ones) for matrix in self.small_matrices]) / rows
        squares = np.concatenate([np.diag(matrix.gram(ones)) for matrix in self.small_matrices])
        spread = np.sqrt(np.maximum(squares / rows - mean**2, 0.0))
        return _frozen(np.where(np.abs(mean) > spread, mean, 0.0), np.float64)


@dataclass(frozen=True, eq=False)
class NestedLeafStatistics:
    """Per-leaf statistics of one row-weight vector ``a`` (§3.4, §3.6, §4).

    Produced by one row pass (``moments.build_nested_leaf_statistics``) in the
    border coordinates of the operator that carries them (``q`` columns), on
    the rows ``x - center`` so that no statistic carries a column's offset:

    - ``weight`` ``(K,)``: ``a_leaf[l] = sum_{r in l} a_r``.
    - ``mean`` ``(K, q)``: the DATA leaf means ``m_l - center`` of the factor
      these statistics belong to, in the shifted form ``x_ref(l) - c + sum_{r
      in l} w_r ((x_r - c) - (x_ref(l) - c)) / w_l`` with the data weights
      ``w`` (never the operator's ``a``; a leaf cut by the row pass's chunks
      combines its pieces' means the same way about the heaviest piece), and
      0 for a leaf with ``w_l == 0``.
      A column that is constant within a leaf gives that constant (less
      ``c``) exactly.  Signed operators carry the same array their factor was
      built from.
    - ``within`` ``(q, q)``: ``sum_r a_r (x_r - m_l)(x_r - m_l)'`` by the centred
      row pass, never ``X'aX - sum_l ...`` (decision 1); exactly symmetric (the
      builder canonicalises ``0.5 (W + W')``).  A column constant within every
      leaf has an exactly zero row and column.
    - ``absolute`` ``(q,)``: ``sum_r e_r (x_rj - m_lj)^2``, the within-leaf
      curvature the rounding of the row weights can move, with ``e_r = a_r``
      when no weight is negative (componentwise accurate) and ``e_r = max |a|``
      on each weighted row of a vector with a rounding-negative row; the
      factor's floor for a border pivot (§3.7).
    - ``center`` ``(q,)``: the global centre ``c`` of the rows (the layout's
      ``border_center``; 0 on an intercept column).
    - ``deviation`` ``(K, q)`` or ``None``: ``dev_l = sum_{r in l} a_r (x_r -
      m_l)``; ``None`` means exactly zero, which is the data pass itself.

    ``cross`` is the raw ``C_leaf[l] = sum_{r in l} a_r x_r = a_l (m_l + c) +
    dev_l``, derived for the raw-coordinate readers.  Non-finite values raise
    ``np.linalg.LinAlgError`` (the iterate's weights caused them);
    inconsistent shapes or an asymmetric ``within`` raise ``ValueError``.
    """

    weight: NDArray
    mean: NDArray
    within: NDArray
    absolute: NDArray
    center: NDArray
    deviation: NDArray | None = None

    def __post_init__(self) -> None:
        weight = _frozen(self.weight, np.float64)
        mean = _frozen(self.mean, np.float64)
        within = _frozen(self.within, np.float64)
        absolute = _frozen(self.absolute, np.float64)
        center = _frozen(self.center, np.float64)
        deviation = None if self.deviation is None else _frozen(self.deviation, np.float64)
        if weight.ndim != 1 or mean.ndim != 2 or mean.shape[0] != len(weight):
            raise ValueError("Leaf weight and mean must have shapes (K,) and (K, q).")
        q = mean.shape[1]
        if within.shape != (q, q) or absolute.shape != (q,) or center.shape != (q,):
            raise ValueError("Within-leaf scatter, absolute mass and centre must match q.")
        if deviation is not None and deviation.shape != mean.shape:
            raise ValueError("Leaf deviation must match the mean shape (K, q).")
        arrays = [weight, mean, within, absolute, center] + (
            [] if deviation is None else [deviation]
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise np.linalg.LinAlgError("Nested leaf statistics must be finite.")
        if not np.array_equal(within, within.T):
            raise ValueError("Within-leaf scatter must be exactly symmetric.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "within", within)
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "deviation", deviation)

    @cached_property
    def cross(self) -> NDArray:
        """Raw ``C_leaf`` ``(K, q)``: ``a_l (m_l + c) + dev_l``; cached for this immutable object."""
        cross = self.weight[:, None] * (self.mean + self.center)
        return cross if self.deviation is None else cross + self.deviation

    @property
    def width(self) -> int:
        """Border width ``q``."""
        return int(self.mean.shape[1])


def _check_partition(small_indices: NDArray, structured_indices: NDArray) -> int:
    all_indices = np.concatenate((small_indices, structured_indices))
    if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
        raise ValueError("Structured index partitions must cover every coefficient once.")
    return len(all_indices)


@dataclass(frozen=True, eq=False)
class NestedDataOperator:
    """Leaf-form symmetric operator ``O = X' diag(a) X`` of a nested design (§3.6).

    ``leaf`` carries ``a_leaf``, the factor's data leaf means about the global
    centre and the row-pass scatter; ``small_indices`` ``(q,)`` and
    ``structured_indices`` ``(n_nodes,)`` (node order) partition ``0..p-1``.
    The data operator of a fit (``a = w``) is ``NestedStructuredSystem.operator``;
    signed W-derivative operators use the same class, built about the same
    leaf means.  It is one member of ``CompactSymmetricOperator`` and may be
    wrapped by ``CenteredBlockOperator`` exactly like the single-level raw
    operators.
    """

    tree: NestedTree
    leaf: NestedLeafStatistics
    small_indices: NDArray
    structured_indices: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        small_indices = _frozen(self.small_indices, np.intp)
        structured_indices = _frozen(self.structured_indices, np.intp)
        q = len(small_indices)
        if self.leaf.weight.shape != (self.tree.sizes[-1],) or self.leaf.width != q:
            raise ValueError("Leaf statistics do not match the tree leaves and border width.")
        if structured_indices.shape != (self.tree.n_nodes,):
            raise ValueError("Nested operator blocks do not match the tree and border.")
        p = _check_partition(small_indices, structured_indices)
        object.__setattr__(self, "small_indices", small_indices)
        object.__setattr__(self, "structured_indices", structured_indices)
        object.__setattr__(self, "shape", (p, p))

    @cached_property
    def A(self) -> NDArray:
        """Raw border block ``X_b' diag(a) X_b`` ``(q, q)``, exactly symmetric.

        ``within + sum_l [dev_l m_l' + m_l dev_l' + a_l m_l m_l']`` with the raw
        leaf means ``m = mean + center``: the chain path never forms it from
        rows (decision 1); only the dense-reference, diagonal and estimability
        readers ask for it.  Cached for the life of this immutable operator.
        """
        leaf = self.leaf
        mean = leaf.mean + leaf.center
        A = leaf.within + mean.T @ (leaf.weight[:, None] * mean)
        if leaf.deviation is not None:
            A = A + leaf.deviation.T @ mean + mean.T @ leaf.deviation
        return _frozen(0.5 * (A + A.T), np.float64)

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply ``O`` to ``(p,)`` or ``(p, m)`` in O(n_nodes m + q^2 m).

        Tree rows: ``subtree_sum(a_leaf * (M x_tree) + C_leaf x_b)``; border
        rows: ``C_leaf' (M x_tree) + A x_b``.  Dense-reference and test use only;
        no factor method materializes ``O``.
        """
        values = np.asarray(rhs, dtype=np.float64)
        columns = values[:, None] if values.ndim == 1 else values
        if columns.ndim != 2 or columns.shape[0] != self.shape[0]:
            raise ValueError(f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m).")
        border = columns[self.small_indices]
        path = self.tree.path_sum(self.tree.split(columns[self.structured_indices]))
        result = np.empty_like(columns)
        result[self.structured_indices] = np.concatenate(
            self.tree.subtree_sum(self.leaf.weight[:, None] * path + self.leaf.cross @ border)
        )
        result[self.small_indices] = self.leaf.cross.T @ path + self.A @ border
        return result[:, 0] if values.ndim == 1 else result

    def diagonal(self) -> NDArray:
        """Return ``diag(O)`` ``(p,)``: subtree sums of ``a_leaf`` on the tree, ``diag(A)`` on the border.

        ``operators.compact_operator_diagonal`` delegates here for a nested raw
        operator.
        """
        diagonal = np.empty(self.shape[0])
        diagonal[self.structured_indices] = np.concatenate(self.tree.subtree_sum(self.leaf.weight))
        diagonal[self.small_indices] = np.diag(self.A)
        return diagonal

    def augmented(self) -> NestedDataOperator:
        """Return the same operator on the coordinates ``[1, X]``, intercept first.

        The result has ``weight`` unchanged, ``mean = [1 | mean]`` (1 for every
        leaf), ``center = [0 | center]``, ``within = [[0, 0], [0, within]]``,
        ``absolute = [0 | absolute]``, ``deviation = None`` or ``[0 |
        deviation]``, ``small_indices = [0, small_indices + 1]`` and
        ``structured_indices + 1``, and shares ``tree``.  Every new entry is a
        copy or an exact zero or one: the intercept column of an operator whose
        rows all carry ``x_0 = 1`` has zero within-leaf deviation.
        """
        leaf = self.leaf
        q, leaves = leaf.width, len(leaf.weight)
        within = np.zeros((q + 1, q + 1))
        within[1:, 1:] = leaf.within
        deviation = leaf.deviation
        if deviation is not None:
            deviation = np.column_stack((np.zeros(leaves), deviation))
        return NestedDataOperator(
            tree=self.tree,
            leaf=NestedLeafStatistics(
                weight=leaf.weight,
                mean=np.column_stack((np.ones(leaves), leaf.mean)),
                within=within,
                absolute=np.concatenate(([0.0], leaf.absolute)),
                center=np.concatenate(([0.0], leaf.center)),
                deviation=deviation,
            ),
            small_indices=np.concatenate(([0], self.small_indices + 1)),
            structured_indices=self.structured_indices + 1,
        )


@dataclass(frozen=True, eq=False)
class NestedPenalizedOperator:
    """``H = O_w + blockdiag(diag(lambda_node), S_b)`` for one nested system.

    ``node_penalty[j]`` ``(K_j,)`` holds ``lambda_u`` for every node of level
    ``j`` (per-node values admit an authoritative ``S_override`` diagonal,
    §3.7); every entry is finite and non-negative.  ``border_penalty``
    ``(q, q)`` is ``S_b`` in the border coordinates of ``data``, exactly
    symmetric.  Built by ``assembly.build_penalized_nested_operator``; the
    factor never adds ``S_b`` to ``A`` (it enters ``Q`` as its own PSD term,
    §3.4).  Negative, non-finite or misshapen penalties raise ``ValueError``.
    """

    data: NestedDataOperator
    node_penalty: tuple[NDArray, ...]
    border_penalty: NDArray
    shape: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        tree = self.data.tree
        node_penalty = tuple(_frozen(values, np.float64) for values in self.node_penalty)
        border_penalty = _frozen(self.border_penalty, np.float64)
        q = len(self.data.small_indices)
        if tuple(values.shape for values in node_penalty) != tuple((k,) for k in tree.sizes):
            raise ValueError("node_penalty must hold one (K_j,) array per level.")
        if border_penalty.shape != (q, q) or not np.array_equal(border_penalty, border_penalty.T):
            raise ValueError("border_penalty must be an exactly symmetric (q, q) matrix.")
        penalties = [*node_penalty, border_penalty]
        if not all(np.all(np.isfinite(values)) for values in penalties):
            raise ValueError("Nested penalties must be finite.")
        if any(np.any(values < 0.0) for values in node_penalty):
            raise ValueError("Nested node penalties must be non-negative.")
        object.__setattr__(self, "node_penalty", node_penalty)
        object.__setattr__(self, "border_penalty", border_penalty)
        object.__setattr__(self, "shape", self.data.shape)

    @property
    def tree(self) -> NestedTree:
        return self.data.tree

    @property
    def small_indices(self) -> NDArray:
        return self.data.small_indices

    @property
    def structured_indices(self) -> NDArray:
        return self.data.structured_indices

    def matvec(self, rhs: NDArray) -> NDArray:
        """Apply ``H`` to ``(p,)`` or ``(p, m)``: ``data.matvec`` plus the penalty."""
        values = np.asarray(rhs, dtype=np.float64)
        result = self.data.matvec(values)
        structured = self.structured_indices
        result[structured] += (np.concatenate(self.node_penalty) * values[structured].T).T
        result[self.small_indices] += self.border_penalty @ values[self.small_indices]
        return result

    def augmented(self) -> NestedPenalizedOperator:
        """Return ``H`` on ``[1, X]``: ``data.augmented()``, ``node_penalty``
        unchanged and ``border_penalty`` padded with an exactly zero intercept
        row and column."""
        q = len(self.small_indices)
        border_penalty = np.zeros((q + 1, q + 1))
        border_penalty[1:, 1:] = self.border_penalty
        return NestedPenalizedOperator(
            data=self.data.augmented(),
            node_penalty=self.node_penalty,
            border_penalty=border_penalty,
        )


def _to_parent(tree: NestedTree, level: int, values: NDArray) -> NDArray:
    """Sum the node values (rows) of ``level`` into their parents at ``level - 1``."""
    return tree._child_sums[level - 1] @ values


def _divide_rows(values: NDArray, weight: NDArray) -> NDArray:
    """Divide the rows of ``values`` ``(K, q)`` by ``weight`` ``(K,)``, exactly zero where it is zero."""
    return np.divide(values.T, weight, out=np.zeros(values.T.shape), where=weight != 0.0).T


def _descendant_products(rows: NDArray) -> NDArray:
    """Row ``i`` of the result is the product of the rows strictly after ``i`` (1 for the last)."""
    products = np.ones_like(rows)
    if len(rows) > 1:
        products[:-1] = np.cumprod(rows[:0:-1], axis=0)[::-1]
    return products


def _low_rank_trace(solve: Callable[[NDArray], NDArray], piece: LowRankSymmetricOperator) -> float:
    """Return ``tr(H^-1 U R U')`` through the solve ``H^-1 U``."""
    return float(np.sum(solve(piece.basis) * (piece.basis @ piece.core)))


def _low_rank_diagonal(
    solve: Callable[[NDArray], NDArray], piece: LowRankSymmetricOperator
) -> NDArray:
    """Return ``diag(H^-1 U R U')`` through the solve ``H^-1 U``."""
    return np.sum(solve(piece.basis) * (piece.basis @ piece.core), axis=1)


def _square_diagonal_by_columns(
    solve: Callable[[NDArray], NDArray],
    matvec: Callable[[NDArray], NDArray],
    width: int,
    block: int = 256,
) -> NDArray:
    """Return ``diag((H^-1 O)^2)`` from explicit columns, ``sum_v (O H^-1)_vi (H^-1 O)_vi``.

    O(p (n_nodes q + q^2)) in all: the route for an operator other than the
    factor's own data (decision 3), which no production caller passes; its
    accuracy is the forward error of the solves.
    """
    diagonal = np.empty(width)
    for start in range(0, width, block):
        stop = min(start + block, width)
        unit = np.zeros((width, stop - start))
        unit[np.arange(start, stop), np.arange(stop - start)] = 1.0
        diagonal[start:stop] = np.sum(solve(matvec(unit)) * matvec(solve(unit)), axis=0)
    return diagonal


def _reject_coupled_null_space(
    F: NDArray, null: NDArray, allowance: NDArray, *, term_name: str
) -> None:
    """Refuse a Schur null space that the tree elimination couples to (§3.7).

    With ``Z`` the orthonormal unscaled null basis of ``Q``, the factor omits
    the log volume ``logdet(I + (F Z)'(F Z)) <= ||F Z||_F^2`` and its
    generalized inverse is Moore-Penrose only when ``F Z = 0``; a nullity-``r``
    block permits ``r eps`` of omitted volume.  ``Z`` is known only through the
    scaled null vectors, so ``allowance`` (per tree node) is the part of
    ``|F Z|`` their own uncertainty explains and is not charged; the product
    rounding ``gamma |F| |Z|`` is.
    """
    eps = np.finfo(np.float64).eps
    coupling = np.abs(F @ null) + F.shape[1] * eps * (np.abs(F) @ np.abs(null))
    certified = np.maximum(coupling - allowance[:, None], 0.0)
    tolerance = float(np.sqrt(eps * null.shape[1]))
    maximum = float(np.max(certified, initial=0.0))
    coupled = not np.isfinite(maximum) or maximum > tolerance
    if not coupled and maximum:
        coupled = maximum * float(np.linalg.norm(certified / maximum)) > tolerance
    if coupled:
        raise np.linalg.LinAlgError(
            f"Nested chain {term_name!r} has a coupled rank-deficient Schur null space."
        )


@dataclass(frozen=True)
class _Piece:
    """Row-pass quantities of a leaf-form operator about the factor's leaf means (§3.6).

    ``a`` ``(K,)`` are the leaf weights, ``V = dev + a (.) e`` ``(K, q)`` and
    ``UOU = W_a + dev'e + e'dev + sum_l a_l e_l e_l'`` ``(q, q)``, exactly
    symmetric.  Pieces of one direction add.
    """

    a: NDArray
    V: NDArray
    UOU: NDArray

    def __add__(self, other: _Piece) -> _Piece:
        return _Piece(self.a + other.a, self.V + other.V, self.UOU + other.UOU)


@dataclass(frozen=True)
class _Prepared:
    """Per-direction products every pair reads (§3.6).

    ``zhat = M T^-1 M'V`` ``(K, q)``, ``VQ = V Q^-1``, ``P = Q^-1 (U'OU) Q^-1``,
    ``weight`` the ``A_x`` accumulation ``A_x = sum_c rho_c^2 A_c`` per level
    with ``a`` at the leaves, and ``level[I] = tr(H^-1 E_I H^-1 O)`` for every
    chain level, formed once from ``W = T^-1 M'V`` so that ``W`` need not stay.
    """

    zhat: NDArray
    VQ: NDArray
    P: NDArray
    weight: tuple[NDArray, ...]
    level: NDArray


@dataclass(frozen=True)
class _Direction:
    """One direction ``scale Omega + dH`` split into what each route reads.

    ``kind`` is ``"level"`` (``key`` the chain level), ``"border"`` (``key``
    the penalty as a ``(q, q)`` matrix in border coordinates) or ``None``;
    ``piece`` merges every leaf-form part of ``dH``; ``operators`` keeps those
    parts for products with low-rank columns and ``low_rank`` the low-rank
    parts, which go through solves.
    """

    component: PenaltyComponent | None
    scale: float
    kind: str | None
    key: int | NDArray | None
    piece: _Piece | None
    operators: tuple
    low_rank: tuple[LowRankSymmetricOperator, ...]

    def apply(self, values: NDArray, *, full: bool) -> NDArray:
        """Return ``(scale Omega + dH) values`` ``(p, r)``, the low-rank parts only when ``full``."""
        result = np.zeros_like(values)
        if self.component is not None:
            result += _penalty_product(self.component, self.scale, values)
        for operator in self.operators + (self.low_rank if full else ()):
            result += operator.matvec(values)
        return result


def _low_rank_terms(
    solve: Callable[[NDArray], NDArray], left: _Direction, right: _Direction
) -> float:
    """Return ``tr(H^-1 L_l H^-1 O_r) + tr(H^-1 X_l H^-1 L_r)`` for the low-rank parts ``L``.

    ``O = X + L`` with ``X`` the penalty and leaf-form parts.  The pair
    ``L_l, L_r`` is counted once, inside the first term, so with ``left is
    right`` the two loops give ``2 tr(H^-1 X H^-1 L) + tr(H^-1 L H^-1 L)``.
    """
    value = 0.0
    for piece in left.low_rank:
        columns = solve(piece.basis)
        value += float(np.sum((columns @ piece.core) * right.apply(columns, full=True)))
    for piece in right.low_rank:
        columns = solve(piece.basis)
        value += float(np.sum((columns @ piece.core) * left.apply(columns, full=False)))
    return value


def _low_rank_matrix(solve: Callable[[NDArray], NDArray], records: Sequence[_Direction]) -> NDArray:
    """Return the ``(m, m)`` low-rank cross terms of ``records``, zero without low-rank parts."""
    traces = np.zeros((len(records), len(records)))
    if any(record.low_rank for record in records):
        for i, j in itertools.combinations_with_replacement(range(len(records)), 2):
            traces[i, j] = traces[j, i] = _low_rank_terms(solve, records[i], records[j])
    return traces


class NestedSchurFactor:
    """Factorization of a nested chain beside a dense border (§3.2-§3.7).

    ``H = [[T, C], [C', A]]`` with the tree block eliminated leaves first by the
    closed-form pivot recursion (no fill; Lean ``chain_ldl``, ``chain_det``) and
    the border Schur complement assembled as the PSD sum

        Q = S_b + W_in + sum_{u non-root} s_u d_u d_u' + sum_{u root} s_u m_u m_u'

    from shifted means (§3.4).  ``sigma = omega / D`` (never ``1 - rho``); ``F``
    and ``M F`` by the top-down ``g/e`` recursion (never ``c/D - sigma A``);
    ``H^-1`` on the pattern from the Takahashi scalars ``t, Z_uu, v, kappa``
    (§3.5).  Every rank decision is taken on the Jacobi-scaled
    ``Q_s = D_s Q D_s``, ``D_s = diag(Q)^(-1/2)`` (§3.7): the Cholesky probe
    residual and pivot floor ``20 q eps``, the eigenvalue fallback at ``1e-10
    lambda_max(Q_s)``, the negative-curvature test, and the coupled-null test
    on ``Z = orth(D_s Z_s)`` against ``F = T^-1 C``, charged the noise ``D_s``
    amplifies in the scaled null vectors ``Z_s``.  A border column whose
    ``|Q_jj|`` is within its floor ``gamma_Q (absolute_j + sum_u |s_u|
    d_uj^2)`` (the row pass's ``absolute``: the curvature the rounding of the
    row weights can move) is an exact null direction, its row and column
    zeroed; one below minus its floor is refused as materially negative.
    Non-negative weights are charged componentwise, so only an exactly zero
    pivot is null and a tiny positive weight keeps its column, as in the
    dense Jacobi-scaled convention; a vector with a rounding-negative row is
    charged at ``max |w|``, so under it weights of ``+eps``, ``0`` and
    ``-eps`` give the same factor.  On a truncated factor every retained-subspace
    quantity is formed in scaled coordinates, the convention of the dense
    ``gram_eigh`` decomposition: the generalized inverse is ``Q^+ = D_s Q_s^+
    D_s`` (Moore-Penrose in the scaled metric, equal to the unscaled one when
    ``Q_jj`` is constant on the null support, as for duplicated columns) and
    ``log pdet(Q)`` is the unscaled pseudo-determinant ``log pdet(Q_s) + sum
    log Q_jj + log det(Z_s' D_s^2 Z_s)``.  The identity routes then read
    ``diag(Q^+ Q) = 1 - ||Z_s[j]||^2`` on the border in place of 1.

    Construction takes a ``NestedPenalizedOperator``.  ``intercept=True`` says
    border column 0 is the unpenalized all-ones intercept (the augmented
    system of ``assembly.build_augmented_nested_factor``); the factor then
    refuses ``Q_00`` within its floor, which for ``w >= 0`` is ``Q_00 == 0``
    exactly, the exact singularity of ``H`` (§3.7), with an intercept-aliasing
    ``LinAlgError``, and it works in the centred coordinates ``[1, X - 1 c']``
    of its leaf statistics (``c = leaf.center``, 0 on the intercept): ``Q``,
    ``F``, ``e`` and every closed form are free of the columns' offsets.  The
    change of basis ``R = I - e_0 c'`` touches only the intercept row, so the
    public methods return raw-coordinate values: ``solve`` maps ``R' r`` in
    and ``R x`` out, the intercept entry of ``diag(H^-1)`` is ``(e_0 - c)'
    Q^-1 (e_0 - c)``, ``inverse_operator_diagonal``, ``diag(Q^+ Q)`` and
    ``coefficient_estimable`` add their intercept-column terms, and traces and
    a full-rank ``logdet`` are invariant.  A pseudo-determinant is not: the
    ``logdet`` of a truncated factor is mapped to the dense backend's
    weighted-mean-centred coordinates, where the intercept is H-orthogonal,
    by ``log det(N_1' N_1)`` with ``N_1`` the orthonormal null basis of ``Q``
    without its intercept row, so it does not depend on which columns ``c``
    centres.  ``intercept=False`` is the unaugmented coefficient
    factor that ``reml_finalize`` retains for the ``(X'WX + S)^-1``
    covariance view; with no intercept to absorb the centre it works on the
    raw means ``mean + center``.

    Refusals at construction: ``LinAlgError`` when a tree pivot ``D_u <=
    gamma_u`` (§3.7; it never fires for ``w >= 0`` since then ``D_u >=
    lambda_u``, and it is the backstop for signed observed rows), for a node
    with ``lambda_u = 0`` and ``omega_u = 0``, material negative curvature of
    ``Q_s`` (a border pivot below minus its floor, or an eigenvalue below ``-q
    gamma_Q`` with ``gamma_Q = (n_nodes + q + 10) eps``, the componentwise
    floor of the factor's own PSD-sum accumulation; the row pass is the
    builder's), a coupled Schur null space, or intercept aliasing.  On the fallback ``Q_s^+`` and ``log
    pdet(Q_s)`` come from the Cholesky of the deflated ``Q_s + Z_s Z_s'``.

    Attributes read by callers (all set at construction):
    ``shape``, ``backend = "structured"``, ``rank`` (``n_nodes + rank(Q)``),
    ``rank_truncated``, ``used_dense_fallback`` (the scaled SVD fallback ran),
    ``fallback_reason``, ``schur_condition_estimate`` (of ``Q_s``: squared
    Cholesky diagonal ratio, or ``sigma_max / sigma_min`` on the fallback,
    ``inf`` when truncated), ``minimum_local_diagonal`` (``min_u D_u``),
    ``small_indices``, ``structured_indices`` (node order),
    ``max_structured_inverse_block``, ``chain_group_indices``,
    ``chain_group_names``, ``dominant_group_name`` (the leaf name, for
    reporting only: never use it to find a layout), ``intercept``,
    ``operator`` (the ``NestedPenalizedOperator`` it factors).

    Operators accepted by the operator methods, in this factor's coordinates:
    ``NestedDataOperator`` on the same forest (the same tree object, or equal
    sizes and parents), the same partitions and bitwise the same
    ``leaf.mean`` and ``leaf.center`` (else ``ValueError``);
    ``LowRankSymmetricOperator`` (handled
    by solves); and ``SumBlockOperator`` of those.  Anything else, including
    ``CenteredBlockOperator`` and the single-level operators, raises
    ``TypeError``.  ``operator.data`` itself is recognised by identity as the
    factor's own data operator (``H - S``).

    Caches (owner: the factor; lifetime: the factor; invalidation: none,
    because weights, penalties, basis, target and precision are all fixed at
    construction and any change builds a new factor): ``Q^-1``, ``diag(H^-1)``,
    the scaled Schur eigenvalues and, per chain level on first use, ``G_I =
    F_I' F_I``, ``h^I`` and the level-pair traces ``t1 + t2 + t3``.  ``T^-1 E_I
    F`` is recomputed per pair (one O(n_nodes q) tree solve).  They must not
    change a rank decision or a bound.
    """

    backend = "structured"
    shape: tuple[int, int]
    rank: int
    rank_truncated: bool
    used_dense_fallback: bool
    fallback_reason: str | None
    schur_condition_estimate: float
    minimum_local_diagonal: float
    small_indices: NDArray
    structured_indices: NDArray
    max_structured_inverse_block: int
    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    dominant_group_name: str
    intercept: bool
    operator: NestedPenalizedOperator

    def __init__(
        self,
        operator: NestedPenalizedOperator,
        *,
        chain_group_names: tuple[str, ...],
        chain_group_indices: tuple[int, ...],
        intercept: bool,
        max_structured_inverse_block: int = 256,
    ):
        data = operator.data
        tree, leaf = data.tree, data.leaf
        depth, q = tree.depth, leaf.width
        names = tuple(chain_group_names)
        if len(names) != depth or len(chain_group_indices) != depth:
            raise ValueError("Chain group names and indices must match the tree depth.")
        if intercept and q == 0:
            raise ValueError("An intercept factor needs border column 0.")
        self.operator = operator
        self.chain_group_names = names
        self.chain_group_indices = tuple(int(index) for index in chain_group_indices)
        self.dominant_group_name = names[-1]
        self.intercept = bool(intercept)
        self.max_structured_inverse_block = int(max_structured_inverse_block)
        self.shape = operator.shape
        self.small_indices = data.small_indices
        self.structured_indices = data.structured_indices
        self._tree, self._leaf, self._penalty = tree, leaf, operator.node_penalty
        p = self.shape[0]
        self._small_position = np.full(p, -1, dtype=np.intp)
        self._small_position[self.small_indices] = np.arange(q)
        self._node_position = np.full(p, -1, dtype=np.intp)
        self._node_position[self.structured_indices] = np.arange(tree.n_nodes)
        eps = np.finfo(np.float64).eps
        # The factor's coordinates: centred about leaf.center with an intercept
        # to absorb the centre, raw without one.  R = I - e_0 c' maps them.
        self._mean = leaf.mean if self.intercept else leaf.mean + leaf.center
        self._center = leaf.center if self.intercept else np.zeros(q)

        # Up pass, leaves first (§3.3): pivots, multipliers, shrunk weights and
        # the shifted node means (§3.4).
        slots: list[list[Any]] = [[None] * depth for _ in range(7)]
        omegas, means, pivots, rhos, sigmas, shrunk, deltas = slots
        omega, mean = leaf.weight, self._mean
        mass, fan = np.abs(leaf.weight), np.zeros(tree.sizes[-1])
        for level in reversed(range(depth)):
            lam = self._penalty[level]
            pivot = omega + lam
            if np.any((lam == 0.0) & (omega == 0.0)):
                raise np.linalg.LinAlgError(
                    f"Nested chain level {names[level]!r} has a node with zero penalty and "
                    "zero accumulated weight, a singular pivot."
                )
            floor = _TREE_PIVOT_FLOOR_FACTOR * eps * (fan + 2.0) * (mass + lam)
            if np.any(pivot <= floor):
                worst = int(np.argmin(pivot - floor))
                raise np.linalg.LinAlgError(
                    f"Nested chain level {names[level]!r} has a tree pivot {pivot[worst]:.6g} "
                    f"at or below its cancellation floor {floor[worst]:.3g}."
                )
            rho = lam / pivot
            sigma = omega / pivot
            s = omega * rho
            omegas[level], means[level], pivots[level] = omega, mean, pivot
            rhos[level], sigmas[level], shrunk[level] = rho, sigma, s
            if level == 0:
                deltas[0] = mean
                break
            parent = tree.parent[level]
            omega_parent = _to_parent(tree, level, s)
            # Shifted parent means about the child of largest s: a column that
            # is constant within the parent gives d = m_c - m_p = 0 exactly.
            # A refinement no bound needs: a plain mean leaves the same eps |m|
            # in d_c for every child of one parent, which sum_c s_c d_c = 0 per
            # parent cancels in the between-child scatter to second order
            # (measured: no Q_s entry moves by more than 5e-16 on the fixtures
            # of tests/test_nested_schur_factor.py).
            order = np.lexsort((-s, parent))
            head = order[np.r_[True, parent[order][1:] != parent[order][:-1]]]
            reference = np.zeros((tree.sizes[level - 1], q))
            reference[parent[head]] = mean[head]
            shift = _to_parent(tree, level, s[:, None] * (mean - reference[parent]))
            mean_parent = np.where(
                omega_parent[:, None] != 0.0, reference + _divide_rows(shift, omega_parent), 0.0
            )
            deltas[level] = mean - mean_parent[parent]
            mass = _to_parent(tree, level, np.abs(s))
            fan = _to_parent(tree, level, np.ones(tree.sizes[level]))
            omega, mean = omega_parent, mean_parent
        self._pivots, self._rho, self._sigma = tuple(pivots), tuple(rhos), tuple(sigmas)
        self.minimum_local_diagonal = float(min(np.min(pivot) for pivot in pivots))

        # F = T^-1 C and M F by the top-down g/e recursion (§3.3).
        F: list[Any] = [None] * depth
        e: list[Any] = [None] * depth
        growth = deltas[0]
        F[0], e[0] = sigmas[0][:, None] * growth, rhos[0][:, None] * growth
        for level in range(1, depth):
            growth = deltas[level] + e[level - 1][tree.parent[level]]
            F[level], e[level] = sigmas[level][:, None] * growth, rhos[level][:, None] * growth
        self._F = tuple(F)
        self._e_leaf = e[-1]

        # Takahashi scalars (§3.5), the ancestor tables and the rho path products.
        t = [np.zeros(tree.sizes[0])]
        for level in range(1, depth):
            variance = 1.0 / pivots[level - 1] + rhos[level - 1] ** 2 * t[level - 1]
            t.append(variance[tree.parent[level]])
        self._t = tuple(t)
        self._Z_diag = tuple(1.0 / D + sigma**2 * t_ for D, sigma, t_ in zip(pivots, sigmas, t))
        self._v = tuple(1.0 / D + rho**2 * t_ for D, rho, t_ in zip(pivots, rhos, t))
        self._kappa = tuple(
            1.0 / D - rho * sigma * t_ for D, rho, sigma, t_ in zip(pivots, rhos, sigmas, t)
        )
        # Node-order copies of the per-level scalars and, per level j, the
        # (j + 1, K_j) table of node-order indices of each node's ancestors (row
        # i the level-i ancestor, row j the node itself): every ancestor gather
        # is then one fancy index and the path products one cumprod.
        self._rho_all, self._sigma_all = np.concatenate(rhos), np.concatenate(sigmas)
        self._kappa_all, self._v_all = np.concatenate(self._kappa), np.concatenate(self._v)
        tables = [tree.offsets[0] + np.arange(tree.sizes[0])[None, :]]
        for level in range(1, depth):
            own = tree.offsets[level] + np.arange(tree.sizes[level])[None, :]
            tables.append(np.vstack((tables[-1][:, tree.parent[level]], own)))
        self._ancestor = tuple(tables)
        # _path[j][i]: pi_{p(u)}(z), the product of rho over levels i+1..j-1 for
        # the level-i ancestor z of a level-j node u; _pi[i]: pi_l(z) for the
        # leaves, which includes the leaf's own rho (1 at the leaf).
        self._path = tuple(_descendant_products(self._rho_all[table[:-1]]) for table in tables)
        self._pi = _descendant_products(self._rho_all[tables[-1]])

        # The border Schur complement as a PSD sum (§3.4; deltas[0] are the
        # root means), then every rank decision on the Jacobi-scaled Q_s (§3.7).
        between = sum(
            ((shrunk[level][:, None] * deltas[level]).T @ deltas[level] for level in range(depth)),
            start=np.zeros((q, q)),
        )
        Q = operator.border_penalty + leaf.within + between
        Q = 0.5 * (Q + Q.T)
        q_diag = np.diag(Q)
        # A border pivot within the curvature the rounding of its own terms and
        # of the row weights can move is an exact null.
        gamma_Q = (tree.n_nodes + q + 10) * eps
        floor = gamma_Q * (
            leaf.absolute
            + sum(np.abs(shrunk[level]) @ deltas[level] ** 2 for level in range(depth))
        )
        if np.any(q_diag < -floor):
            worst = int(np.argmin(q_diag + floor))
            raise np.linalg.LinAlgError(
                f"Nested chain {names[-1]!r} border column {worst} has materially negative "
                f"Schur curvature {q_diag[worst]:.6g}, below minus its floor {floor[worst]:.3g}."
            )
        null_column = q_diag <= floor
        if self.intercept and null_column[0]:
            raise np.linalg.LinAlgError(
                f"Nested chain {names!r} is aliased with the fitted intercept: the border "
                "Schur complement has a zero intercept pivot."
            )
        Q[null_column, :] = Q[:, null_column] = 0.0
        q_diag = np.where(null_column, 0.0, q_diag)
        scale = 1.0 / np.sqrt(np.where(null_column, 1.0, q_diag))
        Q_scaled = scale[:, None] * Q * scale[None, :]
        self._Q_scaled, self._scale = Q_scaled, scale
        self._scaled_eigenvalues_cache: NDArray | None = None
        self.used_dense_fallback = False
        self.fallback_reason: str | None = None
        self._null = self._null_scaled = np.zeros((q, 0))
        if q == 0:
            self._Q_inverse = np.zeros((0, 0))
            self.schur_condition_estimate = 1.0
            logdet_Q = 0.0
        else:
            try:
                cholesky = scipy.linalg.cholesky(Q_scaled, lower=True, check_finite=False)
                probe = np.zeros(q)
                probe[0] = 1.0
                solution = scipy.linalg.cho_solve((cholesky, True), probe, check_finite=False)
                residual = float(np.linalg.norm(Q_scaled @ solution - probe))
                if not np.isfinite(residual) or residual >= _SCALED_PROBE_RESIDUAL:
                    raise np.linalg.LinAlgError(
                        f"scaled Schur Cholesky residual {residual:.3g} exceeds "
                        f"{_SCALED_PROBE_RESIDUAL:g}"
                    )
                squares = np.diag(cholesky) ** 2
                if np.any(squares <= _SCALED_PIVOT_FLOOR_FACTOR * q * eps):
                    raise np.linalg.LinAlgError(
                        "scaled Schur Cholesky pivot is below its cancellation floor"
                    )
                inverse = scipy.linalg.cho_solve((cholesky, True), np.eye(q), check_finite=False)
                self._Q_inverse = scale[:, None] * (0.5 * (inverse + inverse.T)) * scale[None, :]
                self.schur_condition_estimate = float(squares.max() / squares.min())
                logdet_Q = float(2.0 * np.sum(np.log(np.diag(cholesky))) + np.sum(np.log(q_diag)))
            except (np.linalg.LinAlgError, ValueError) as error:
                self.used_dense_fallback = True
                self.fallback_reason = f"Schur Cholesky fallback: {error}"
                eigenvalues, vectors = np.linalg.eigh(Q_scaled)
                self._scaled_eigenvalues_cache = eigenvalues
                curvature_floor = q * gamma_Q
                if eigenvalues[0] < -curvature_floor:
                    raise np.linalg.LinAlgError(
                        f"Nested chain {names[-1]!r} has materially negative Schur curvature."
                    ) from error
                threshold = max(
                    _SCALED_SVD_RCOND * float(np.max(np.abs(eigenvalues))), curvature_floor
                )
                positive = eigenvalues > threshold
                null_scaled = vectors[:, ~positive]
                lifted = scale[:, None] * null_scaled
                null = np.linalg.qr(lifted)[0]
                if null.shape[1]:
                    # Z_s is exact to the Davis-Kahan angle ||E|| / gap, ||E|| the
                    # eigh backward error plus the componentwise uncertainty of
                    # Q_s; D_s amplifies that noise where Q_jj is small, so the
                    # coupling test is not charged the part of |F Z| it explains.
                    gap = float(eigenvalues[positive].min()) if np.any(positive) else np.inf
                    angle = q * ((q + 3.0) * eps + 2.0 * gamma_Q) / gap
                    smallest = float(np.linalg.svd(lifted, compute_uv=False)[-1])
                    F_all = np.concatenate(F)
                    allowance = (4.0 * angle * np.sqrt(null.shape[1]) / smallest) * (
                        np.abs(F_all) @ scale
                    )
                    _reject_coupled_null_space(F_all, null, allowance, term_name=names[-1])
                # Deflating the null gives Q_s + Z_s Z_s' the retained spectrum and
                # unit eigenvalues on the null: one Cholesky yields log pdet(Q_s)
                # and (Q_s + Z_s Z_s')^-1 - Z_s Z_s' = Q_s^+, the retained inverse
                # in scaled coordinates.
                try:
                    cholesky = scipy.linalg.cholesky(
                        Q_scaled + null_scaled @ null_scaled.T, lower=True, check_finite=False
                    )
                except np.linalg.LinAlgError as retained_error:
                    raise np.linalg.LinAlgError(
                        f"Nested chain {names[-1]!r} retained Schur block is not positive "
                        f"definite: {retained_error}"
                    ) from retained_error
                inverse = scipy.linalg.cho_solve((cholesky, True), np.eye(q), check_finite=False)
                inverse = 0.5 * (inverse + inverse.T) - null_scaled @ null_scaled.T
                self._Q_inverse = scale[:, None] * inverse * scale[None, :]
                # log pdet(Q) by Jacobi's complementary minor over the retained
                # scaled eigenvectors V: det(V' D_s^-2 V) = det(D_s^-2) det(Z_s' D_s^2 Z_s).
                null_gram = null_scaled.T @ (scale[:, None] ** 2 * null_scaled)
                logdet_Q = float(
                    2.0 * np.sum(np.log(np.diag(cholesky)))
                    + np.sum(np.log(q_diag[q_diag > 0.0]))
                    + np.linalg.slogdet(null_gram)[1]
                )
                if self.intercept:
                    # To the dense convention, whatever centre c the threshold
                    # chose: in weighted-mean-centred coordinates the intercept is
                    # H-orthogonal, so a change of centre R = I + e_0 r' maps the
                    # null basis N to R^-1 N = N with row 0 set to 0, and
                    # pdet(R'QR) = pdet(Q) det(N'R^-T R^-1 N) for orthonormal N.
                    logdet_Q += float(np.linalg.slogdet(null[1:].T @ null[1:])[1])
                self.schur_condition_estimate = (
                    float("inf") if null.shape[1] else float(eigenvalues[-1] / eigenvalues[0])
                )
                self._null, self._null_scaled = null, null_scaled
        # diag(Q^+ Q) = 1 - ||Z_s[j]||^2: what the identity routes read as
        # diag(H^+ H) on the border of a truncated factor, mapped by R through
        # the column Q^+ Q e_0 = e_0 - D_s Z_s Z_s[0]' / D_s[0].
        self._retained_diagonal = 1.0 - np.sum(self._null_scaled**2, axis=1)
        if self.intercept:
            column = -scale * (self._null_scaled @ self._null_scaled[0]) / scale[0]
            column[0] += 1.0
            self._retained_diagonal += self._center * column
            self._retained_diagonal[0] -= self._center @ column
        self._logdet = float(sum(np.sum(np.log(pivot)) for pivot in pivots) + logdet_Q)
        self.rank = int(tree.n_nodes + q - self._null.shape[1])
        self.rank_truncated = self.rank < p
        self._diagonal_cache: NDArray | None = None
        self._level_gram: dict[int, NDArray] = {}
        self._level_h: dict[int, NDArray] = {}
        self._level_pairs: dict[tuple[int, int], float] = {}

    # -- tree kernels -------------------------------------------------------
    def _tree_solve(self, rhs: Sequence[NDArray]) -> list[NDArray]:
        """Return ``T^-1 rhs`` for per-level ``(K_j,)`` or ``(K_j, m)`` right-hand sides.

        ``T = L D L'`` with ``L_{a,u} = sigma_u`` for ``a`` a strict ancestor of
        ``u`` (Lean ``chain_ldl``): a forward pass leaves to roots, the pivot
        division, and a backward pass roots to leaves.
        """
        tree, depth = self._tree, self._tree.depth
        y: list[Any] = [None] * depth
        carried = np.zeros_like(np.asarray(rhs[-1], dtype=np.float64))
        for level in reversed(range(depth)):
            y[level] = rhs[level] - carried
            if level:
                carried = _to_parent(tree, level, (self._sigma[level] * y[level].T).T + carried)
        x: list[Any] = [None] * depth
        above = np.zeros_like(y[0])
        for level in range(depth):
            if level:
                above = (above + x[level - 1])[tree.parent[level]]
            x[level] = ((y[level].T / self._pivots[level]) - self._sigma[level] * above.T).T
        return x

    def _selector_h(self, weights: Sequence) -> tuple[NDArray, ...]:
        """Return ``h_a = sum_{c in ch(a)} (weight_c sigma_c^2 + rho_c^2 h_c)`` bottom-up.

        ``weights[j]`` is a scalar or ``(K_j,)`` per level: the level selector
        ``1_I`` for ``penalty_cross_trace``, the node penalties for the penalty
        sandwich.
        """
        tree = self._tree
        h: list[Any] = [None] * tree.depth
        h[-1] = np.zeros(tree.sizes[-1])
        for level in reversed(range(1, tree.depth)):
            summand = weights[level] * self._sigma[level] ** 2 + self._rho[level] ** 2 * h[level]
            h[level - 1] = _to_parent(tree, level, summand)
        return tuple(h)

    def _level_selector_h(self, level: int) -> NDArray:
        """``h^I`` of the level-``level`` selector in node order, cached per level."""
        cached = self._level_h.get(level)
        if cached is None:
            selector = [float(j == level) for j in range(self._tree.depth)]
            cached = self._level_h[level] = np.concatenate(self._selector_h(selector))
        return cached

    def _gram(self, level: int) -> NDArray:
        """``G_I = F_I' F_I``, cached per level for the life of the factor."""
        cached = self._level_gram.get(level)
        if cached is None:
            cached = self._level_gram[level] = self._F[level].T @ self._F[level]
        return cached

    def _inverse_diagonal(self) -> NDArray:
        """``diag(H^-1)``: ``Z_uu + F_u Q^-1 F_u'`` on the tree, ``diag(Q^-1)`` on the border; cached."""
        if self._diagonal_cache is None:
            Q_inverse = self._Q_inverse
            diagonal = np.empty(self.shape[0])
            diagonal[self.structured_indices] = np.concatenate(
                [
                    Z + np.sum((F @ Q_inverse) * F, axis=1)
                    for Z, F in zip(self._Z_diag, self._F, strict=True)
                ]
            )
            diagonal[self.small_indices] = np.diag(Q_inverse)
            # the intercept entry of R Q^-1 R': (e_0 - c)' Q^-1 (e_0 - c)
            intercept = -self._center
            intercept[:1] += 1.0
            diagonal[self.small_indices[:1]] = intercept @ Q_inverse @ intercept
            self._diagonal_cache = diagonal
        return self._diagonal_cache

    # -- protocol: solves, determinant, selected inverse ------------------
    def solve(self, rhs: NDArray) -> NDArray:
        """Return ``H^-1 rhs`` for ``(p,)`` or ``(p, m)`` (§3.6).

        ``u = T^-1 r_t`` by the tree forward, diagonal and backward passes,
        ``x_b = Q^+ (r_b - C_leaf' (M u)_leaf)`` with ``C_leaf = w m`` in the
        factor's coordinates, ``x_t = u - F x_b``, between ``R'`` in and ``R``
        out; ``Q^+`` is the retained-subspace inverse on a truncated factor.
        O(n_nodes q + q^2) per column.  A non-finite solution raises
        ``LinAlgError``.
        """
        values = np.asarray(rhs, dtype=np.float64)
        columns = values[:, None] if values.ndim == 1 else values
        if columns.ndim != 2 or columns.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {values.shape}."
            )
        tree = self._tree
        u = self._tree_solve(tree.split(columns[self.structured_indices]))
        border_rhs = columns[self.small_indices]
        # R' r: the centred border rows lose c times the intercept row.
        border_rhs = border_rhs - self._center[:, None] * border_rhs[:1]
        path = self._leaf.weight[:, None] * tree.path_sum(u)
        border = self._Q_inverse @ (border_rhs - self._mean.T @ path)
        solution = np.empty_like(columns)
        solution[self.structured_indices] = np.concatenate(
            [u_level - F @ border for u_level, F in zip(u, self._F, strict=True)]
        )
        # R x: only the intercept entry moves back to raw coordinates.
        border[:1] -= self._center @ border
        solution[self.small_indices] = border
        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                f"Nested chain {self.dominant_group_name!r} solve is not representable."
            )
        return solution[:, 0] if values.ndim == 1 else solution

    def logdet(self) -> float:
        """Return ``sum_u log D_u + log|Q|``.

        On a truncated factor ``log|Q|`` is ``log det(B' Q B)`` with ``B`` an
        orthonormal basis of the complement of the certified null space, the
        pseudo-determinant the single-level factor reports; with an intercept
        it is taken in the dense backend's weighted-mean-centred coordinates.
        """
        return self._logdet

    def _validate_selected_indices(self, indices: NDArray) -> NDArray:
        selected = np.asarray(indices, dtype=np.intp)
        if selected.ndim != 1:
            raise ValueError("Selected inverse indices must be one-dimensional.")
        if np.any((selected < 0) | (selected >= self.shape[0])):
            raise IndexError("Selected inverse index is outside the factor dimensions.")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Selected inverse indices must be unique.")
        return selected

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        """Return ``diag(H^-1)[indices]`` in the requested order.

        Tree node ``u``: ``Z_uu + F_u Q^-1 F_u'``; border: ``diag(Q^-1)``.  The
        full diagonal is formed once, O(n_nodes q^2), and cached.  Indices must
        be one-dimensional, unique and in range (``ValueError``/``IndexError``).
        """
        diagonal = self._inverse_diagonal()[self._validate_selected_indices(indices)]
        if not np.all(np.isfinite(diagonal)):
            raise np.linalg.LinAlgError(
                f"Nested chain {self.dominant_group_name!r} selected inverse is not representable."
            )
        return diagonal

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        """Return the principal block ``H^-1[indices][:, indices]`` by solves.

        Refuses with ``ValueError`` when more than ``max_structured_inverse_block``
        of the requested indices are tree nodes (any chain level), with the
        single-level factor's "request its diagonal instead" message.
        """
        selected = self._validate_selected_indices(indices)
        nodes = int(np.count_nonzero(self._node_position[selected] >= 0))
        if nodes > self.max_structured_inverse_block:
            raise ValueError(
                f"Refusing to materialize a {nodes} x {nodes} inverse block for structured "
                f"term {self.dominant_group_name!r}; request its diagonal instead."
            )
        unit = np.zeros((self.shape[0], len(selected)))
        unit[selected, np.arange(len(selected))] = 1.0
        block = self.solve(unit)[selected]
        return 0.5 * (block + block.T)

    def row_quadratic_forms(self, rows: NDArray) -> NDArray:
        """Return ``x_i' H^+ x_i`` for each row of ``rows`` ``(m, p)``, forming no ``K x K`` block.

        In the factor's coordinates ``H^+ = [[T^-1 + F Q^+ F', -F Q^+], [-Q^+ F',
        Q^+]]``, so a row with tree part ``b`` and border part ``a`` (``R'``
        applied) gives ``||D^-1/2 L^-1 b||^2 + y' Q^+ y`` with ``y = a - F' b``:
        the random-effect and fixed-effect halves of the hat diagonal of Bates et
        al. (2015, eqs. 63-65).  ``L^-1 e_u`` lives on the reach of ``u`` (Gilbert
        and Peierls 1988), its ancestors-or-self: 1 at ``u`` and ``-sigma_u`` times
        the ``rho`` product strictly between at each ancestor (``_path``).  Each
        nonzero of ``b`` adds at most ``depth`` terms, summed per row and node
        before squaring, so a row whose levels are not one root-to-leaf path is
        exact too.  O(nnz(b) depth + m q^2).
        """
        values = scipy.sparse.csr_array(rows, dtype=np.float64)
        if values.shape[1] != self.shape[0]:
            raise ValueError(f"rows must have shape (m, {self.shape[0]}), got {values.shape}.")
        tree = self._tree
        border = values[:, self.small_indices].toarray()
        # R' x: the centred border columns lose c times the intercept entry.
        border -= border[:, :1] * self._center
        keys, terms = [], []
        for level, (start, stop) in enumerate(zip(tree.offsets[:-1], tree.offsets[1:])):
            level_rows = values[:, self.structured_indices[start:stop]].tocoo()
            border -= level_rows @ self._F[level]
            row, local = (np.asarray(index, dtype=np.intp) for index in level_rows.coords)
            reach = np.vstack(
                (-self._sigma[level][local] * self._path[level][:, local], np.ones(len(local)))
            )
            keys.append((row * tree.n_nodes + self._ancestor[level][:, local]).ravel())
            terms.append((level_rows.data * reach).ravel())
        reached, slot = np.unique(np.concatenate(keys), return_inverse=True)
        solved = np.bincount(slot, weights=np.concatenate(terms))
        pivots = np.concatenate(self._pivots)[reached % tree.n_nodes]
        tree_part = np.bincount(
            reached // tree.n_nodes, weights=solved * solved / pivots, minlength=values.shape[0]
        )
        return tree_part + np.sum((border @ self._Q_inverse) * border, axis=1)

    # -- penalty components --------------------------------------------------
    def _classify(self, component: PenaltyComponent) -> tuple[str, int | NDArray]:
        """Return ``("level", I)`` for an identity penalty on exactly level ``I``, else ``("border", Omega)``.

        ``Omega`` is the component's penalty embedded in a ``(q, q)`` matrix in
        border coordinates.  Anything else (a dense penalty on the chain, part
        of a level, or a component straddling chain and border) is ``ValueError``.
        """
        indices = _component_indices(component, self.shape[0])
        nodes = self._node_position[indices]
        if np.all(nodes >= 0):
            offsets = self._tree.offsets
            level = int(np.searchsorted(offsets, nodes.min(), side="right") - 1)
            whole_level = np.arange(offsets[level], offsets[level + 1])
            if component.penalty_kind != "identity" or not np.array_equal(
                np.sort(nodes), whole_level
            ):
                raise ValueError(
                    f"Penalty component {component.name!r} must be an identity penalty on "
                    "exactly one nested chain level."
                )
            return "level", level
        positions = self._small_position[indices]
        if np.any(positions < 0):
            raise ValueError(
                f"Penalty component {component.name!r} straddles the nested chain and the border."
            )
        omega = np.zeros((len(self.small_indices),) * 2)
        if component.penalty_kind == "identity":
            omega[positions, positions] = 1.0
        else:
            omega[np.ix_(positions, positions)] = _component_omega(component, self.shape[0])
        return "border", omega

    def _level_pair(self, left_level: int, right_level: int) -> float:
        """``||(H^-1)_IJ||_F^2 = t1 + t2 + t3`` for two chain levels (§3.6)."""
        depth = self._tree.depth
        h_I = self._tree.split(self._level_selector_h(left_level))
        h_J = self._tree.split(self._level_selector_h(right_level))
        t1 = 0.0
        for level in range(depth):
            D, rho, sigma, t = (
                self._pivots[level],
                self._rho[level],
                self._sigma[level],
                self._t[level],
            )
            diag_I, diag_J = (
                h_I[level] + float(level == left_level),
                h_J[level] + float(level == right_level),
            )
            off_I = rho * h_I[level] - float(level == left_level) * sigma
            off_J = rho * h_J[level] - float(level == right_level) * sigma
            t1 += float(np.sum(diag_I * diag_J / D**2) + 2.0 * np.sum(off_I * off_J * t / D))
        selected = [
            self._F[left_level] if level == left_level else np.zeros_like(F)
            for level, F in enumerate(self._F)
        ]
        Y = self._tree_solve(selected)
        Q_inverse = self._Q_inverse
        t2 = 2.0 * float(np.sum(Q_inverse * (self._F[right_level].T @ Y[right_level])))
        t3 = float(
            np.sum((Q_inverse @ self._gram(right_level)) * (Q_inverse @ self._gram(left_level)).T)
        )
        return t1 + t2 + t3

    def _penalty_pair(self, left: tuple, right: tuple) -> float:
        """Unscaled ``tr(H^-1 Omega_l H^-1 Omega_r)`` for two classified components."""
        Q_inverse = self._Q_inverse
        if left[0] == "level" and right[0] == "level":
            # coarser level first: the order the §8 verification used, and
            # bitwise symmetric under the cancellation of t1 + t2 + t3
            pair = (min(left[1], right[1]), max(left[1], right[1]))
            if pair not in self._level_pairs:
                self._level_pairs[pair] = self._level_pair(*pair)
            return self._level_pairs[pair]
        if left[0] == "level":
            return float(np.sum((Q_inverse @ self._gram(left[1]) @ Q_inverse) * right[1]))
        if right[0] == "level":
            return float(np.sum((Q_inverse @ self._gram(right[1]) @ Q_inverse) * left[1]))
        return float(np.sum((Q_inverse @ left[1]) * (Q_inverse @ right[1]).T))

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        """Return ``tr(H^-1 Omega)`` for one penalty component.

        An identity component whose indices are exactly one chain level ``I``
        gives ``sum_{u in I} diag(H^-1)_u``; a component wholly in the border
        (identity or dense) gives ``tr(Q^-1_JJ Omega)``.  A component that
        straddles the chain and the border, or covers part of a level, raises
        ``ValueError``.
        """
        kind, key = self._classify(component)
        if kind == "level":
            offsets = self._tree.offsets
            nodes = self.structured_indices[offsets[key] : offsets[key + 1]]
            return float(np.sum(self._inverse_diagonal()[nodes]))
        return float(np.sum(self._Q_inverse * key))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        """Return ``left_scale right_scale tr(H^-1 Omega_l H^-1 Omega_r)`` (§3.6).

        Two levels: ``t1 + t2 + t3`` with ``t1`` the O(n_nodes) ``h/e`` closed
        form, ``t2 = 2 tr(Q^-1 F_J' (T^-1 E_I F)_J)`` and ``t3 = tr(Q^-1 G_J Q^-1
        G_I)``.  Level and border: ``tr((Q^-1 G_I Q^-1)_JJ Omega_J)``.  Border
        pair: ``Q^-1`` blocks.  Component rules as in ``trace_inverse_penalty``.
        """
        pair = self._penalty_pair(self._classify(left), self._classify(right))
        return float(left_scale * right_scale * pair)

    # -- operators ------------------------------------------------------------
    def _check_operator(self, operator: NestedDataOperator) -> None:
        tree = self._tree
        same_forest = operator.tree is tree or (
            operator.tree.sizes == tree.sizes
            and all(
                np.array_equal(a, b) for a, b in zip(operator.tree.parent, tree.parent, strict=True)
            )
        )
        if not same_forest:
            raise ValueError("Nested operator describes another forest than the factor.")
        if not (
            np.array_equal(operator.small_indices, self.small_indices)
            and np.array_equal(operator.structured_indices, self.structured_indices)
        ):
            raise ValueError("Nested operator partitions differ from the factor's.")
        if not (
            np.array_equal(operator.leaf.mean, self._leaf.mean)
            and np.array_equal(operator.leaf.center, self._leaf.center)
        ):
            raise ValueError("Nested operator was built about other leaf means than the factor's.")

    def _pieces(self, operator) -> tuple[tuple, tuple[LowRankSymmetricOperator, ...]]:
        """Split an accepted operator into leaf-form and low-rank parts (``TypeError`` otherwise)."""
        if operator is None:
            return (), ()
        if isinstance(operator, SumBlockOperator):
            parts = [self._pieces(item) for item in operator.operators]
            return (
                tuple(itertools.chain.from_iterable(leaf for leaf, _ in parts)),
                tuple(itertools.chain.from_iterable(low for _, low in parts)),
            )
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, LowRankSymmetricOperator):
            return (), (operator,)
        if isinstance(operator, NestedDataOperator):
            self._check_operator(operator)
            return (operator,), ()
        raise TypeError(
            f"NestedSchurFactor does not represent {type(operator).__name__} operators."
        )

    def _piece(self, operator: NestedDataOperator) -> _Piece:
        """Row-pass quantities of a leaf-form operator about the factor's leaf means (§3.6)."""
        leaf, e = operator.leaf, self._e_leaf
        weighted = leaf.weight[:, None] * e
        V = weighted if leaf.deviation is None else leaf.deviation + weighted
        UOU = leaf.within + weighted.T @ e
        if leaf.deviation is not None:
            UOU = UOU + leaf.deviation.T @ e + e.T @ leaf.deviation
        return _Piece(leaf.weight, V, 0.5 * (UOU + UOU.T))

    def _direction(
        self,
        component: PenaltyComponent | None,
        scale: float,
        operators: tuple,
        low_rank: tuple[LowRankSymmetricOperator, ...],
    ) -> _Direction:
        kind, key = (None, None) if component is None else self._classify(component)
        piece = self._merged_piece(operators) if operators else None
        return _Direction(component, float(scale), kind, key, piece, operators, low_rank)

    def _merged_piece(self, operators: Sequence[NestedDataOperator]) -> _Piece:
        """The summed row-pass quantities of one or more accepted leaf-form operators."""
        return sum((self._piece(item) for item in operators[1:]), start=self._piece(operators[0]))

    def _direction_of(self, component, scale, operator) -> _Direction:
        return self._direction(component, scale, *self._pieces(operator))

    def _weight_delta(self, weights: NDArray) -> list[NDArray]:
        """``delta_l = a_l``, ``delta_u = sum_c rho_c delta_c`` per level, for ``(K,)`` or ``(K, n)``."""
        tree, depth = self._tree, self._tree.depth
        delta: list[Any] = [None] * depth
        delta[-1] = weights
        for level in reversed(range(1, depth)):
            delta[level - 1] = _to_parent(tree, level, (self._rho[level] * delta[level].T).T)
        return delta

    def _tree_weight_solve(self, weights: NDArray) -> list[NDArray]:
        """Return ``T^-1 M'a`` per level from the tree covariances, never by the tree solve.

        ``(T^-1 M'a)_u = sum_l Cov(beta_u, S_l) a_l = kappa_u delta_u - sigma_u
        sum_{x strict ancestor} v_x pi_{p(u)}(x) (delta_x - rho_c delta_c)``, ``c``
        the child of ``x`` toward ``u``.  With ``lambda << omega`` the forward
        pass of the tree solve cancels ``M'a`` against the eliminated mass at
        the internal nodes; this form does not.
        """
        delta = self._weight_delta(weights)
        flat = np.concatenate(delta).reshape(self._tree.n_nodes, -1)
        values = []
        for level, table in enumerate(self._ancestor):
            above, below, own = table[:-1], table[1:], table[-1]
            mass = flat[above] - self._rho_all[below][..., None] * flat[below]
            weight = (self._v_all[above] * self._path[level])[..., None]
            ancestors = self._sigma_all[own][:, None] * np.sum(weight * mass, axis=0)
            value = self._kappa_all[own][:, None] * flat[own] - ancestors
            values.append(value.reshape(delta[level].shape))
        return values

    def _prepare(self, piece: _Piece) -> _Prepared:
        """Form the per-direction products, with ``tr(H^-1 E_I H^-1 O)`` per level.

        The level values are ``sum_{i in I} [(Z M'diag(a) M Z)_ii - 2 W_i Q^-1 F_i'
        + F_i P F_i']`` with the tree-covariance closed form for the first term
        (never explicit inverse columns) and ``W = T^-1 M'V``.
        """
        tree, depth = self._tree, self._tree.depth
        Q_inverse = self._Q_inverse
        tree_solve = self._tree_solve(tree.subtree_sum(piece.V))
        P = Q_inverse @ piece.UOU @ Q_inverse
        table = self._ancestor[-1]
        values = np.empty(depth)
        for level in range(depth):
            h = self._level_selector_h(level)
            above, below = table[:level], table[1 : level + 1]
            mass = h[above] - self._rho_all[below] ** 2 * h[below]
            if level:
                mass[-1] -= self._sigma_all[below[-1]] ** 2
            covariance = self._kappa_all[table[level]] ** 2 * self._pi[level] ** 2 + np.sum(
                self._v_all[above] ** 2 * self._pi[:level] ** 2 * mass, axis=0
            )
            F = self._F[level]
            values[level] = (
                piece.a @ covariance
                - 2.0 * np.sum((tree_solve[level] @ Q_inverse) * F)
                + np.sum((F @ P) * F)
            )
        weight = self._weight_delta(piece.a)
        for level in reversed(range(1, depth)):
            weight[level - 1] = _to_parent(tree, level, self._rho[level] ** 2 * weight[level])
        return _Prepared(
            zhat=tree.path_sum(tree_solve),
            VQ=piece.V @ Q_inverse,
            P=P,
            weight=tuple(weight),
            level=values,
        )

    def _operator_pair(
        self, left: _Piece, left_prep: _Prepared, right: _Piece, right_prep: _Prepared
    ) -> float:
        """``tr(H^-1 O_l H^-1 O_r)`` for two leaf-form pieces (§3.6)."""
        tree, depth = self._tree, self._tree.depth
        zz = 0.0
        for level in range(depth):
            inner = left_prep.weight[level] * right_prep.weight[level]
            if level < depth - 1:
                below = (
                    self._rho[level + 1] ** 4
                    * left_prep.weight[level + 1]
                    * right_prep.weight[level + 1]
                )
                inner = inner - _to_parent(tree, level + 1, below)
            zz += float(np.sum(self._v[level] ** 2 * inner))
        return (
            zz
            + float(np.sum(right_prep.VQ * left_prep.zhat))
            + float(np.sum(left_prep.VQ * right_prep.zhat))
            + float(np.sum(left_prep.P * right.UOU))
        )

    def _penalty_operator(self, kind: str, key, prepared: _Prepared) -> float:
        if kind == "level":
            return float(prepared.level[key])
        return float(np.sum(prepared.P * key))

    def _pair(self, left: _Direction, left_prep, right: _Direction, right_prep) -> float:
        """The closed-form part of ``tr(H^-1 O_l H^-1 O_r)`` (penalty and leaf-form parts)."""
        value = 0.0
        if left.kind and right.kind:
            value += (
                left.scale
                * right.scale
                * self._penalty_pair((left.kind, left.key), (right.kind, right.key))
            )
        if left.kind and right.piece is not None:
            value += left.scale * self._penalty_operator(left.kind, left.key, right_prep)
        if right.kind and left.piece is not None:
            value += right.scale * self._penalty_operator(right.kind, right.key, left_prep)
        if left.piece is not None and right.piece is not None:
            value += self._operator_pair(left.piece, left_prep, right.piece, right_prep)
        return value

    def _cross_matrix(self, records: Sequence[_Direction]) -> NDArray:
        """The closed-form ``(m, m)`` matrix over ``records`` (low-rank parts excluded)."""
        prepared = [
            None if record.piece is None else self._prepare(record.piece) for record in records
        ]
        traces = np.empty((len(records), len(records)))
        for i, j in itertools.combinations_with_replacement(range(len(records)), 2):
            traces[i, j] = traces[j, i] = self._pair(
                records[i], prepared[i], records[j], prepared[j]
            )
        return traces

    def _intercept_column_solve(self, piece: _Piece) -> NDArray:
        """Return ``H^-1 O e_0`` in the factor's coordinates, border column 0 the intercept.

        ``O e_0 = [M'a; A_{:,0}]`` and ``U'(O e_0) = sum_l V_l =: g`` from the
        row pass (never ``A_{:,0} - (MF)'a``): tree rows ``T^-1 M'a - F Q^-1 g``,
        border rows ``Q^-1 g``.
        """
        g = self._Q_inverse @ piece.V.sum(axis=0)
        nodes = self._tree_weight_solve(piece.a)
        column = np.empty(self.shape[0])
        column[self.structured_indices] = np.concatenate(
            [level - F @ g for level, F in zip(nodes, self._F, strict=True)]
        )
        column[self.small_indices] = g
        return column

    def _intercept_column(self, piece: _Piece) -> NDArray:
        """``O e_0 = [M'a; A_{:,0}]`` in the factor's coordinates, ``A_{:,0} = sum_l V_l + (MF)'a``."""
        column = np.empty(self.shape[0])
        column[self.structured_indices] = np.concatenate(self._tree.subtree_sum(piece.a))
        column[self.small_indices] = piece.V.sum(axis=0) + (self._mean - self._e_leaf).T @ piece.a
        return column

    def _piece_trace(self, piece: _Piece) -> float:
        return float(piece.a @ self._v[-1] + np.sum(self._Q_inverse * piece.UOU))

    def _piece_diagonal(self, piece: _Piece) -> NDArray:
        """``diag(H^-1 O)`` by the row-pass closed form (§3.6)."""
        tree, depth = self._tree, self._tree.depth
        delta: list[Any] = [None] * depth
        delta[-1] = piece.a
        for level in reversed(range(1, depth)):
            delta[level - 1] = _to_parent(tree, level, self._rho[level] * delta[level])
        MV = tree.subtree_sum(piece.V)
        Q_inverse = self._Q_inverse
        diagonal = np.empty(self.shape[0])
        diagonal[self.structured_indices] = np.concatenate(
            [
                kappa * d - np.sum((F @ Q_inverse) * MV_level, axis=1)
                for kappa, d, F, MV_level in zip(self._kappa, delta, self._F, MV, strict=True)
            ]
        )
        diagonal[self.small_indices] = np.einsum(
            "ij,ji->i", Q_inverse, piece.UOU + piece.V.T @ (self._mean - self._e_leaf)
        )
        return diagonal

    def _identity_diagonal(self) -> NDArray:
        """``diag(H^+ (H - S)) = diag(H^+ H) - diag(H^+ S)`` for the factor's own data operator.

        ``diag(H^+ H)`` is 1 on the tree and ``_retained_diagonal`` on the
        border (1 unless truncated).
        """
        diagonal = self._inverse_diagonal()
        edf = np.empty(self.shape[0])
        edf[self.structured_indices] = (
            1.0 - np.concatenate(self._penalty) * diagonal[self.structured_indices]
        )
        edf[self.small_indices] = self._retained_diagonal - np.sum(
            self._Q_inverse * self.operator.border_penalty, axis=1
        )
        return edf

    def _identity_square_diagonal(self) -> NDArray:
        """``diag((H^+ (H - S))^2)`` through the penalty sandwich ``H^+ S H^+`` (§3.6).

        ``(P - H^+ S)^2 = P - 2 H^+ S + H^+ S H^+ S`` with the projector ``P =
        H^+ H`` (``H^+ S P = H^+ S`` because a truncated direction is
        unpenalized), so the border reads ``_retained_diagonal`` in place of 1.
        Tree entries ``(H^-1 S H^-1)_uu = N_u + 2 W_u Q^-1 F_u' + F_u P F_u'`` with
        ``N_u = sum_v lambda_v Z_uv^2`` from the tree covariances (own,
        descendant, ancestor and incomparable nodes, O(n_nodes depth)), ``W =
        T^-1 Lambda F`` and ``P = Q^-1 (F'Lambda F + S_b) Q^-1``; border entries
        are ``P``.
        """
        lam = self._penalty
        h, lam_all = np.concatenate(self._selector_h(lam)), np.concatenate(lam)
        Q_inverse = self._Q_inverse
        weighted = self._tree_solve(
            [lam_level[:, None] * F for lam_level, F in zip(lam, self._F, strict=True)]
        )
        core = self.operator.border_penalty + sum(
            (F.T @ (lam_level[:, None] * F) for lam_level, F in zip(lam, self._F, strict=True)),
            start=np.zeros_like(self.operator.border_penalty),
        )
        P = Q_inverse @ core @ Q_inverse
        sandwich = []
        for level, table in enumerate(self._ancestor):
            above, below = table[:-1], table[1:]
            mass = (
                h[above]
                - lam_all[below] * self._sigma_all[below] ** 2
                - self._rho_all[below] ** 2 * h[below]
            )
            ancestors = np.sum(
                self._path[level] ** 2
                * (lam_all[above] * self._kappa_all[above] ** 2 + self._v_all[above] ** 2 * mass),
                axis=0,
            )
            F = self._F[level]
            sandwich.append(
                lam[level] * self._Z_diag[level] ** 2
                + self._kappa[level] ** 2 * h[table[-1]]
                + self._sigma[level] ** 2 * ancestors
                + 2.0 * np.sum((weighted[level] @ Q_inverse) * F, axis=1)
                + np.sum((F @ P) * F, axis=1)
            )
        diagonal = self._inverse_diagonal()[self.structured_indices]
        S_b = self.operator.border_penalty
        result = np.empty(self.shape[0])
        result[self.structured_indices] = (
            1.0 - 2.0 * lam_all * diagonal + lam_all * np.concatenate(sandwich)
        )
        result[self.small_indices] = (
            self._retained_diagonal
            - 2.0 * np.sum(Q_inverse * S_b, axis=1)
            + np.sum(P * S_b, axis=1)
        )
        return result

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        """Return ``tr(H^-1 O)``: ``sum_l a_l v_l + <Q^-1, U'OU>`` (§3.6).

        ``U'OU = W_a + dev'e + e'dev + sum_l a_l e_l e_l'`` from the row-pass
        quantities about the factor's own leaf means, never ``A_O`` minus
        path-sum products.  For ``operator.data`` it returns ``p - tr(H^-1 S)``,
        the sum of the identity-route ``inverse_operator_diagonal``.
        """
        if operator is self.operator.data:
            return float(np.sum(self._identity_diagonal()))
        record = self._direction_of(None, 0.0, operator)
        value = 0.0 if record.piece is None else self._piece_trace(record.piece)
        return value + sum(_low_rank_trace(self.solve, piece) for piece in record.low_rank)

    def inverse_operator_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag(H^-1 O)`` ``(p,)`` (§3.6, decision 3).

        The factor's own data operator takes the identity route: ``1 -
        lambda_u (H^-1)_uu`` on the tree, ``1 - diag(Q^-1 S_b)`` on the border.
        Other leaf-form operators take the row-pass closed form: tree ``u``:
        ``kappa_u delta_u - F_u Q^-1 (M'V)_u'`` with ``V = dev + a (.) e``,
        ``delta_l = a_l``, ``delta_u = sum_c rho_c delta_c``; border
        ``diag(Q^-1 (U'OU + V'(MF)))``.  With an intercept these are centred
        values ``N = diag(H_c^-1 O_c)``; ``R N R^-1`` adds ``c_j N_j0`` to border
        entry ``j`` and ``-c' N_b0`` to the intercept.
        """
        if operator is self.operator.data:
            return self._identity_diagonal()
        record = self._direction_of(None, 0.0, operator)
        diagonal = np.zeros(self.shape[0])
        if record.piece is not None:
            diagonal = self._piece_diagonal(record.piece)
            if self.intercept:
                column = self._intercept_column_solve(record.piece)[self.small_indices]
                diagonal[self.small_indices] += self._center * column
                diagonal[self.small_indices[0]] -= self._center @ column
        for piece in record.low_rank:
            diagonal += _low_rank_diagonal(self.solve, piece)
        return diagonal

    def inverse_operator_square_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag((H^-1 O)^2)`` ``(p,)`` (§3.6, decision 3).

        Own data operator: ``1 - 2 lambda_u Z^H_uu + lambda_u (H^-1 S H^-1)_uu``
        on the tree and ``1 - 2 (Q^-1 S_b)_jj + ((H^-1 S H^-1)_bb S_b)_jj`` on the
        border, from the penalty sandwich; O(n_nodes d^2 + n_nodes q^2).  Other
        operators take ``sum_v (O H^-1)_vi (H^-1 O)_vi`` from explicit columns
        (``_square_diagonal_by_columns``), O(p (n_nodes q + q^2)) and limited to
        the forward error of the solves; the exact-arithmetic decomposition
        ``sum_v (H^-1 O H^-1)_uv O_vu`` it evaluates has terms of size
        ``1/lambda`` that cancel to a result of size ``1/w`` when ``lambda <<
        omega`` (§10, measured 1.9e-3).  No production caller passes another
        operator.
        """
        if operator is self.operator.data:
            return self._identity_square_diagonal()
        self._pieces(operator)
        return _square_diagonal_by_columns(self.solve, operator.matvec, self.shape[0])

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H^-1 O_l H^-1 O_r)`` (§3.6).

        ``tr(Zhat diag(a_l) Zhat diag(a_r)) + <Q^-1, V_r' Zhat V_l> + <Q^-1, V_l'
        Zhat V_r> + <Q^-1 (U'O_lU) Q^-1, U'O_rU>`` with ``Zhat = M Z M'``, the
        first term by the O(n_nodes) ``A_x`` recursion and ``Zhat V`` by one
        tree solve with ``q`` right-hand sides.
        """
        records = [self._direction_of(None, 0.0, left), self._direction_of(None, 0.0, right)]
        return float((self._cross_matrix(records) + _low_rank_matrix(self.solve, records))[0, 1])

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H^-1 (scale Omega) H^-1 O)`` (§3.6).

        Level ``I``: ``scale sum_{i in I} [(Z M'diag(a)M Z)_ii - 2 W_i Q^-1 F_i' +
        F_i P F_i']`` with the tree-covariance closed form for the first term
        (never explicit inverse columns).  Border: ``scale <Q^-1 Omega Q^-1,
        U'OU>``.
        """
        records = [
            self._direction(component, scale, (), ()),
            self._direction_of(None, 0.0, operator),
        ]
        return float((self._cross_matrix(records) + _low_rank_matrix(self.solve, records))[0, 1])

    def derivative_cross_traces(self, directions: Sequence[DerivativeDirection]) -> NDArray:
        """Return the symmetric ``(m, m)`` matrix ``tr(H^-1 O_i H^-1 O_j)``.

        ``O_i = scale_i Omega_i + dH_i``: ``Omega_i`` an identity component on
        exactly one chain level or a component wholly in the border, ``dH_i``
        ``None`` or an accepted operator.  By bilinearity each entry is a sum of
        the four method formulas above; per direction ``Zhat V_i``, ``V_i
        Q^-1``, ``P_i = Q^-1 (U'O_iU) Q^-1``, ``W_i = T^-1 M'V_i`` and the
        ``A_x`` accumulations are formed once, and per level ``G_I``,
        ``T^-1 E_I F`` and ``h^I``.  O(m (K q^2 + q^3) + m^2 (K q + q^2 +
        n_nodes)).  This is the ``DerivativeCrossTraceFactor`` method that
        ``reml_direct_hessian`` dispatches on.
        """
        records = [
            self._direction_of(component, scale, operator)
            for component, scale, operator in directions
        ]
        return self._cross_matrix(records) + _low_rank_matrix(self.solve, records)

    def coefficient_estimable(self) -> NDArray:
        """Return per-coefficient estimability ``(p,)`` after Schur truncation.

        All ``True`` unless truncated; otherwise the null basis of the scaled
        ``H_s = diag(I, D_s) H diag(I, D_s)``, ``[-F D_s Z_s; Z_s]``, passed to
        ``geometry._coefficient_estimable_from_null_basis``: its border rows
        carry the scaled null vectors' eps-level noise, not the ``D_s``-amplified
        noise of the unscaled ``Z``, and a coordinate touches the null space in
        either scaling.
        """
        from superglm.solvers._structured.geometry import _coefficient_estimable_from_null_basis

        null = self._null_scaled
        if not null.shape[1]:
            return np.ones(self.shape[0], dtype=bool)
        null_basis = np.zeros((self.shape[0], null.shape[1]))
        null_basis[self.small_indices] = null
        null_basis[self.small_indices[0]] -= (self._center * self._scale) @ null / self._scale[0]
        null_basis[self.structured_indices] = -np.concatenate(self._F) @ (
            self._scale[:, None] * null
        )
        return _coefficient_estimable_from_null_basis(self.shape[0], null_basis)

    def scaled_schur_eigenvalues(self) -> NDArray:
        """Return the ascending eigenvalues of ``Q_s = D_s Q D_s`` ``(q,)``, cached.

        The public curvature check the observed-geometry build applies in place
        of reading a private ``Q`` (§6): it refuses the iterate when the smallest
        is below ``-1e-10 max(|eigenvalues|)``.
        """
        if self._scaled_eigenvalues_cache is None:
            self._scaled_eigenvalues_cache = np.linalg.eigvalsh(self._Q_scaled)
        return self._scaled_eigenvalues_cache


class ProfiledNestedSchurFactor:
    """Slope inverse of a nested fit with the intercept profiled out (§3.6, §6).

    ``augmented_factor`` factors ``H_aug`` on ``[1, X]`` (``intercept=True``);
    this adapter exposes ``H_c^-1``, the slope block of ``H_aug^-1``, through the
    Hessian-factor protocol.  With ``e0`` the intercept coordinate, ``u = (1,
    mean_x)`` and ``P = [-mean_x'; I]`` these identities hold exactly and are
    the whole adapter:

        H_aug^-1 = e0 e0' / sum_w + P H_c^-1 P',   H_aug^-1 u = e0 / sum_w,
        G := H_aug^-1 - e0 e0' / sum_w = P H_c^-1 P',   H_c^-1 P' = E' G,
        O_c = P' O_aug P for an operator centred on mean_x.

    So for a slope penalty ``Omega`` (embedded with a zero intercept row) and
    centred operators ``O_c = P' O_aug P`` with ``y = O_aug e0`` and ``t =
    (O_aug)_00``:

        tr(H_c^-1 Omega) and all penalty pairs: the augmented values (index shift);
        tr(H_c^-1 O_c) = tr(H_aug^-1 O_aug) - t / sum_w;
        tr(H_c^-1 Omega H_c^-1 O_c) = tr(H_aug^-1 Omega H_aug^-1 O_aug);
        tr(H_c^-1 O_c H_c^-1 O'_c) = tr(H_aug^-1 O_aug H_aug^-1 O'_aug)
                                      - 2 y' H_aug^-1 y' / sum_w + t t' / sum_w^2;
        diag(H_c^-1 O_c)_j = (H_aug^-1 O_aug)_(j+1, j+1) - mean_x[j] (H_aug^-1 y)_(j+1).

    ``O_aug`` is ``raw.augmented()`` for a ``CenteredBlockOperator`` whose
    ``raw`` is a ``NestedDataOperator`` in slope coordinates.  The augmented
    factor evaluates every piece in its centred coordinates (centre ``c``,
    0 on the tree), where the last identity reads ``N_(j+1, j+1) - (mean_x[j]
    - c[j]) N_(j+1, 0)`` with ``N = H_aug^-1 O_aug`` centred: both terms are
    free of the columns' offsets.  The corrections carry no ``1/lambda`` term: the near-null
    direction of ``H_aug`` (intercept against the roots) is annihilated by
    ``O_aug`` and orthogonal to ``y``.  ``tests/test_nested_schur_factor.py``
    verifies these routes against an exact centred reference (the §8
    verification checked the augmented factor only).

    Accepted operators, in slope coordinates: a ``CenteredBlockOperator`` whose
    ``center`` is bitwise ``mean_x`` and whose ``raw`` is a ``NestedDataOperator``
    accepted by the augmented factor after augmentation; ``data_operator``
    itself passed raw, which is wrapped as ``CenteredBlockOperator(raw,
    cross=xtw, total=sum_w, center=mean_x)``, the implicit centring
    ``ProfiledScalarSchurFactor`` applies and ``irls_direct``'s ``p_eff`` relies
    on; ``LowRankSymmetricOperator`` pieces through profiled solves; and
    ``SumBlockOperator`` of those.  Any other raw ``NestedDataOperator``, or a
    centred operator about any other center, raises ``ValueError`` (its
    correction would carry ``1/lambda`` terms); other kinds raise
    ``TypeError``.  The centred ``data_operator``
    (``StructuredLinearSystemState.centered_data_operator``, the ``edf`` and
    ``edf1`` caller in ``state_ops``) takes the identity routes, recognised by
    ``raw is data_operator`` with ``center``, ``cross`` and ``total`` equal to
    ``mean_x``, ``xtw`` and ``sum_w``.

    Attributes (the named contracts of §6): ``augmented_factor``, ``sum_w``,
    ``xtw`` ``(p,)``, ``mean_x = xtw / sum_w`` (``w_derivatives`` reads
    ``mean_x`` and ``sum_w`` by ``getattr`` and silently uncentres without
    them), ``data_operator``, ``shape = (p, p)``, ``backend = "structured"``,
    ``rank = max(augmented rank - 1, 0)``, ``rank_truncated``,
    ``used_dense_fallback``, ``fallback_reason``, ``schur_condition_estimate``,
    ``minimum_local_diagonal``, ``dominant_group_name``,
    ``chain_group_indices``, ``chain_group_names``, ``small_indices`` and
    ``structured_indices`` (slope coordinates, ``augmented - 1`` without the
    intercept), ``max_structured_inverse_block``.  ``logdet()`` is
    ``augmented logdet - log(sum_w)``.  ``ValueError`` when ``sum_w`` is not
    positive and finite or the widths disagree.
    """

    backend = "structured"
    shape: tuple[int, int]
    augmented_factor: NestedSchurFactor
    sum_w: float
    xtw: NDArray
    mean_x: NDArray
    data_operator: NestedDataOperator
    rank: int
    rank_truncated: bool
    used_dense_fallback: bool
    fallback_reason: str | None
    schur_condition_estimate: float
    minimum_local_diagonal: float
    dominant_group_name: str
    chain_group_indices: tuple[int, ...]
    chain_group_names: tuple[str, ...]
    small_indices: NDArray
    structured_indices: NDArray
    max_structured_inverse_block: int

    def __init__(
        self,
        *,
        augmented_factor: NestedSchurFactor,
        sum_w: float,
        xtw: NDArray,
        data_operator: NestedDataOperator,
    ):
        self.augmented_factor = augmented_factor
        self.sum_w = float(sum_w)
        self.xtw = _frozen(xtw, np.float64)
        if not np.isfinite(self.sum_w) or self.sum_w <= 0.0:
            raise ValueError("sum_w must be positive and finite.")
        p = len(self.xtw)
        if augmented_factor.shape != (p + 1, p + 1) or data_operator.shape != (p, p):
            raise ValueError("Augmented factor, data operator and xtw widths disagree.")
        if not augmented_factor.intercept:
            raise ValueError("The augmented factor must carry the intercept in border column 0.")
        self.data_operator = data_operator
        self.shape = (p, p)
        self.mean_x = _frozen(self.xtw / self.sum_w, np.float64)
        self.rank = max(int(augmented_factor.rank) - 1, 0)
        self.rank_truncated = self.rank < p
        self.used_dense_fallback = augmented_factor.used_dense_fallback
        self.fallback_reason = augmented_factor.fallback_reason
        self.schur_condition_estimate = augmented_factor.schur_condition_estimate
        self.minimum_local_diagonal = augmented_factor.minimum_local_diagonal
        self.dominant_group_name = augmented_factor.dominant_group_name
        self.chain_group_indices = augmented_factor.chain_group_indices
        self.chain_group_names = augmented_factor.chain_group_names
        self.small_indices = augmented_factor.small_indices[1:] - 1
        self.structured_indices = augmented_factor.structured_indices - 1
        self.max_structured_inverse_block = augmented_factor.max_structured_inverse_block
        self._centered_data = CenteredBlockOperator(
            raw=data_operator, cross=self.xtw, total=self.sum_w, center=self.mean_x
        )
        # the augmented factor's centre in slope coordinates (0 on the tree)
        self._center = np.zeros(p)
        self._center[self.small_indices] = augmented_factor._center[1:]

    @staticmethod
    def _shift_component(component: PenaltyComponent) -> PenaltyComponent:
        start, stop = component.group_sl.start, component.group_sl.stop
        if start is None or stop is None:
            raise ValueError("Penalty component slices must have explicit bounds.")
        return replace(component, group_sl=slice(start + 1, stop + 1, component.group_sl.step))

    def _augment(self, operator: CenteredBlockOperator) -> NestedDataOperator:
        """``O_aug`` of a centred slope operator, checked like any augmented-factor operator.

        The augmented factor's forest, partitions and bitwise leaf means must
        match (``ValueError`` otherwise): a raw operator built about other leaf
        means would give silently wrong row-pass quantities.
        """
        augmented = cast(NestedDataOperator, operator.raw).augmented()
        self.augmented_factor._check_operator(augmented)
        return augmented

    def _pieces(self, operator) -> tuple[tuple[CenteredBlockOperator, ...], tuple]:
        """Split a slope operator into centred leaf-form and low-rank parts."""
        if operator is None:
            return (), ()
        if isinstance(operator, SumBlockOperator):
            parts = [self._pieces(item) for item in operator.operators]
            return (
                tuple(itertools.chain.from_iterable(leaf for leaf, _ in parts)),
                tuple(itertools.chain.from_iterable(low for _, low in parts)),
            )
        if operator.shape != self.shape:
            raise ValueError("Operator and factor dimensions must match.")
        if isinstance(operator, LowRankSymmetricOperator):
            return (), (operator,)
        if isinstance(operator, NestedDataOperator):
            if operator is not self.data_operator:
                raise ValueError(
                    "Only the factor's own data operator may be passed raw; centre other "
                    "nested operators on mean_x."
                )
            return (self._centered_data,), ()
        if isinstance(operator, CenteredBlockOperator) and isinstance(
            operator.raw, NestedDataOperator
        ):
            if not np.array_equal(operator.center, self.mean_x):
                raise ValueError("Centred nested operator must be centred on the factor's mean_x.")
            return (operator,), ()
        raise TypeError(
            f"ProfiledNestedSchurFactor does not represent {type(operator).__name__} operators."
        )

    def _is_centered_data(self, centered: tuple, low_rank: tuple) -> bool:
        if len(centered) != 1 or low_rank:
            return False
        operator = centered[0]
        return (
            operator.raw is self.data_operator
            and operator.total == self.sum_w
            and np.array_equal(operator.cross, self.xtw)
        )

    def _direction(self, component, scale, operator) -> tuple[_Direction, _Direction, float]:
        """Return the augmented record, the slope record and ``t = sum (O_aug)_00``."""
        centered, low_rank = self._pieces(operator)
        shifted = None if component is None else self._shift_component(component)
        augmented = tuple(self._augment(item) for item in centered)
        record = self.augmented_factor._direction(shifted, scale, augmented, ())
        slope = _Direction(component, float(scale), None, None, None, centered, low_rank)
        return record, slope, float(sum(item.total for item in centered))

    def _cross_matrix(self, records: Sequence[tuple]) -> NDArray:
        """``tr(H_c^-1 O_i H_c^-1 O_j)`` for parsed directions by the class-docstring identity."""
        augmented = [record for record, _, _ in records]
        slopes = [slope for _, slope, _ in records]
        totals = np.array([total for _, _, total in records])
        factor = self.augmented_factor
        traces = factor._cross_matrix(augmented)
        zero = np.zeros(factor.shape[0])
        pieces = [record.piece for record in augmented]
        Y = np.column_stack([zero if p is None else factor._intercept_column(p) for p in pieces])
        HY = np.column_stack(
            [zero if p is None else factor._intercept_column_solve(p) for p in pieces]
        )
        quadratic = Y.T @ HY
        traces = traces - (quadratic + quadratic.T) / self.sum_w
        traces = traces + np.outer(totals, totals) / self.sum_w**2
        return 0.5 * (traces + traces.T) + _low_rank_matrix(self.solve, slopes)

    def solve(self, rhs: NDArray) -> NDArray:
        """Return ``H_c^-1 rhs``: the augmented solve of ``[0; rhs]``, intercept row dropped."""
        values = np.asarray(rhs, dtype=np.float64)
        columns = values[:, None] if values.ndim == 1 else values
        if columns.ndim != 2 or columns.shape[0] != self.shape[0]:
            raise ValueError(
                f"rhs must have shape ({self.shape[0]},) or ({self.shape[0]}, m), "
                f"got {values.shape}."
            )
        augmented = np.zeros((self.shape[0] + 1, columns.shape[1]))
        augmented[1:] = columns
        solution = self.augmented_factor.solve(augmented)[1:]
        return solution[:, 0] if values.ndim == 1 else solution

    def logdet(self) -> float:
        """Return ``augmented_factor.logdet() - log(sum_w)``."""
        return float(self.augmented_factor.logdet() - np.log(self.sum_w))

    def selected_inverse_diagonal(self, indices: NDArray) -> NDArray:
        """Augmented ``selected_inverse_diagonal`` at ``indices + 1``."""
        return self.augmented_factor.selected_inverse_diagonal(
            np.asarray(indices, dtype=np.intp) + 1
        )

    def selected_inverse_block(self, indices: NDArray) -> NDArray:
        """Augmented ``selected_inverse_block`` at ``indices + 1`` (same cap)."""
        return self.augmented_factor.selected_inverse_block(np.asarray(indices, dtype=np.intp) + 1)

    def trace_inverse_penalty(self, component: PenaltyComponent) -> float:
        """Augmented ``trace_inverse_penalty`` of the component shifted by one."""
        return self.augmented_factor.trace_inverse_penalty(self._shift_component(component))

    def penalty_cross_trace(
        self,
        left: PenaltyComponent,
        right: PenaltyComponent,
        left_scale: float,
        right_scale: float,
    ) -> float:
        """Augmented ``penalty_cross_trace`` of the shifted components."""
        return self.augmented_factor.penalty_cross_trace(
            self._shift_component(left), self._shift_component(right), left_scale, right_scale
        )

    def trace_inverse_operator(self, operator: CompactSymmetricOperator) -> float:
        """Return ``tr(H_c^-1 O_c)`` by the class-docstring identity."""
        centered, low_rank = self._pieces(operator)
        if self._is_centered_data(centered, low_rank):
            return float(np.sum(self.augmented_factor._identity_diagonal()[1:]))
        value = sum(_low_rank_trace(self.solve, piece) for piece in low_rank)
        if centered:
            piece = self.augmented_factor._merged_piece([self._augment(item) for item in centered])
            total = sum(item.total for item in centered)
            value += self.augmented_factor._piece_trace(piece) - total / self.sum_w
        return float(value)

    def inverse_operator_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag(H_c^-1 O_c)`` ``(p,)``; identity route for the centred data operator."""
        centered, low_rank = self._pieces(operator)
        if self._is_centered_data(centered, low_rank):
            return self.augmented_factor._identity_diagonal()[1:]
        diagonal = np.zeros(self.shape[0])
        for piece in low_rank:
            diagonal += _low_rank_diagonal(self.solve, piece)
        if centered:
            factor = self.augmented_factor
            piece = factor._merged_piece([self._augment(item) for item in centered])
            diagonal += factor._piece_diagonal(piece)[1:]
            diagonal -= (self.mean_x - self._center) * factor._intercept_column_solve(piece)[1:]
        return diagonal

    def inverse_operator_square_diagonal(self, operator: CompactSymmetricOperator) -> NDArray:
        """Return ``diag((H_c^-1 O_c)^2)`` ``(p,)``.

        The centred data operator takes the augmented identity route restricted
        to the slopes (``H_c^-1 S H_c^-1 S = E' H_aug^-1 S H_aug^-1 S E``); other
        operators the explicit-column route with its documented caveat.
        """
        centered, low_rank = self._pieces(operator)
        if self._is_centered_data(centered, low_rank):
            return self.augmented_factor._identity_square_diagonal()[1:]
        if isinstance(operator, NestedDataOperator):
            operator = self._centered_data
        return _square_diagonal_by_columns(self.solve, operator.matvec, self.shape[0])

    def operator_cross_trace(
        self,
        left: CompactSymmetricOperator,
        right: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H_c^-1 O_l H_c^-1 O_r)`` by the class-docstring identity."""
        records = [self._direction(None, 0.0, left), self._direction(None, 0.0, right)]
        return float(self._cross_matrix(records)[0, 1])

    def penalty_operator_cross_trace(
        self,
        component: PenaltyComponent,
        scale: float,
        operator: CompactSymmetricOperator,
    ) -> float:
        """Return ``tr(H_c^-1 (scale Omega) H_c^-1 O_c)``: the augmented value."""
        records = [self._direction(component, scale, None), self._direction(None, 0.0, operator)]
        return float(self._cross_matrix(records)[0, 1])

    def derivative_cross_traces(self, directions: Sequence[DerivativeDirection]) -> NDArray:
        """Return ``tr(H_c^-1 O_i H_c^-1 O_j)`` for every pair of slope directions.

        The augmented factor's matrix for the augmented directions, minus
        ``2 Y' H_aug^-1 Y / sum_w`` and plus ``t t' / sum_w^2``, with ``Y`` the
        columns ``y_i = O_aug,i e0`` (zero for penalty-only directions) and
        ``t_i`` their intercept entries; ``H_aug^-1 y_i`` is the closed-form
        column of ``_intercept_column_solve``, no solve.
        """
        records = [
            self._direction(component, scale, operator) for component, scale, operator in directions
        ]
        return self._cross_matrix(records)

    def coefficient_estimable(self) -> NDArray:
        """Augmented ``coefficient_estimable`` without the intercept entry."""
        return self.augmented_factor.coefficient_estimable()[1:]

    def scaled_schur_eigenvalues(self) -> NDArray:
        """Augmented ``scaled_schur_eigenvalues``."""
        return self.augmented_factor.scaled_schur_eigenvalues()
