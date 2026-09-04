"""Per-group matrix wrappers for sparse/dense BCD operations.

Five wrapper types with the same interface:
- DenseGroupMatrix: numeric features (single column) or dense fallback
- SparseGroupMatrix: categoricals, non-SSP splines
- SparseSSPGroupMatrix: SSP splines (factored: sparse B + dense R_inv)
- DiscretizedSSPGroupMatrix: discretized SSP splines (binned B_unique + index)
- DiscretizedSCOPGroupMatrix: discretized SCOP monotone splines (centered design at bin centers)

DesignMatrix holds the list and provides full-matrix matvec/rmatvec.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._group_matrix import _group_matrix_algebra
from ._group_matrix._cross_matrix_execution import (
    CrossMatrixExecutionPlan as CrossMatrixExecutionPlan,
)
from ._group_matrix._group_matrix_bin_space import (
    MixedBinSpaceCenteringPlan,
    build_mixed_bin_space_centering_plan,
)
from ._group_matrix._group_matrix_bins import discretize_column
from ._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from ._group_matrix._group_matrix_discretized import (
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from ._group_matrix._group_matrix_execution import MatrixExecutionPlan
from ._group_matrix._group_matrix_kernels import (
    _disc_disc_2d_hist as _kernel_disc_disc_2d_hist,
)
from ._group_matrix._group_matrix_tabmat import (
    RawSplineTabmatPlan,
    _build_raw_spline_tabmat_plan,
    _build_tabmat_split,
    _is_retained_tabmat_vector_candidate,
    _is_tabmat_centering_candidate,
    _tabmat_vector,
)

DenseGroupMatrix.__module__ = __name__
SparseGroupMatrix.__module__ = __name__
CategoricalGroupMatrix.__module__ = __name__
RandomEffectGroupMatrix.__module__ = __name__
FactorSmoothGroupMatrix.__module__ = __name__
SparseSSPGroupMatrix.__module__ = __name__
SplineCategoricalGroupMatrix.__module__ = __name__
DiscretizedSSPGroupMatrix.__module__ = __name__
SupportCompressedSSPGroupMatrix.__module__ = __name__
DiscretizedSCOPGroupMatrix.__module__ = __name__
DiscretizedSplineCategoricalGroupMatrix.__module__ = __name__
SupportCompressedSplineCategoricalGroupMatrix.__module__ = __name__
DiscretizedTensorGroupMatrix.__module__ = __name__


def _discretize_column(x: NDArray, n_bins: int = 256) -> tuple[NDArray, NDArray]:
    """Compatibility wrapper for the private discretization helper."""
    return cast(tuple[NDArray, NDArray], discretize_column(x, n_bins))


GroupMatrix = (
    DenseGroupMatrix
    | SparseGroupMatrix
    | CategoricalGroupMatrix
    | RandomEffectGroupMatrix
    | FactorSmoothGroupMatrix
    | SparseSSPGroupMatrix
    | SplineCategoricalGroupMatrix
    | DiscretizedSSPGroupMatrix
    | DiscretizedSCOPGroupMatrix
    | DiscretizedSplineCategoricalGroupMatrix
    | DiscretizedTensorGroupMatrix
)

# The histogram cell caps live in _group_matrix_algebra, which is where they are
# read.  Duplicating them here was dead weight of an actively misleading kind:
# they are the names a reviewer greps first, and patching them did nothing --
# the same failure mode as a threshold frozen into a default argument.


def _agg_by_bin(gm: GroupMatrix, bin_idx: NDArray, W: NDArray, n_bins: int) -> NDArray:
    """Compatibility wrapper for the private algebra helper."""
    return _group_matrix_algebra._agg_by_bin(gm, bin_idx, W, n_bins)


def _cross_gram_tensor_tensor(gm_i, gm_j, W: NDArray) -> NDArray:
    """Compatibility wrapper for tensor×tensor cross-gram helper."""
    return _group_matrix_algebra._cross_gram_tensor_tensor(gm_i, gm_j, W)


def _disc_disc_2d_hist(
    bin_idx_i: NDArray, bin_idx_j: NDArray, W: NDArray, n_bins_i: int, n_bins_j: int
) -> NDArray:
    """Compatibility wrapper for the fused discretized 2D histogram kernel."""
    return cast(
        NDArray,
        _kernel_disc_disc_2d_hist(bin_idx_i, bin_idx_j, W, n_bins_i, n_bins_j),
    )


def _cross_gram_tensor_main(gm_tensor, gm_main, W: NDArray) -> NDArray:
    """Compatibility wrapper for tensor×main-effect cross-gram helper."""
    return _group_matrix_algebra._cross_gram_tensor_main(gm_tensor, gm_main, W)


def _cross_gram(
    gm_i: GroupMatrix,
    gm_j: GroupMatrix,
    W: NDArray,
    profile: dict | None = None,
) -> NDArray:
    """Compatibility wrapper for the private cross-gram helper."""
    return _group_matrix_algebra._cross_gram(gm_i, gm_j, W, profile=profile)


def _gram_any_sign(gm: GroupMatrix, W: NDArray) -> NDArray:
    """Compatibility wrapper for any-sign diagonal gram helper."""
    return _group_matrix_algebra._gram_any_sign(gm, W)


def _block_xtwx(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict | None = None,
) -> NDArray:
    """Compatibility wrapper for block XtWX assembly."""
    return _group_matrix_algebra._block_xtwx(
        gms,
        groups,
        W,
        tabmat_split=tabmat_split,
        profile=profile,
    )


def _block_xtwx_rhs(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    Wz: NDArray,
    *,
    tabmat_split=None,
    profile: dict | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compatibility wrapper for block XtWX/XtW/XtWz assembly."""
    return cast(
        tuple[NDArray, NDArray, NDArray],
        _group_matrix_algebra._block_xtwx_rhs(
            gms,
            groups,
            W,
            Wz,
            tabmat_split=tabmat_split,
            profile=profile,
        ),
    )


def _block_xtwx_signed(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict | None = None,
) -> NDArray:
    """Compatibility wrapper for arbitrary-sign block XtWX assembly."""
    return _group_matrix_algebra._block_xtwx_signed(
        gms,
        groups,
        W,
        tabmat_split=tabmat_split,
        profile=profile,
    )


class _LazyTabmatSplit:
    """One shared lazy split without a back-reference to its DesignMatrix."""

    __slots__ = ("built", "group_matrices", "split")

    def __init__(self, group_matrices) -> None:
        self.group_matrices = group_matrices
        self.split = None
        self.built = False

    def get(self):
        if not self.built:
            self.split = _build_tabmat_split(self.group_matrices)
            self.built = True
        return self.split


class _LazyRawSplineTabmatPlan:
    """One releasable DesignMatrix-owned raw-spline acceleration plan."""

    __slots__ = ("built", "group_matrices", "n", "plan")

    def __init__(self, group_matrices, *, n: int) -> None:
        self.group_matrices = group_matrices
        self.n = n
        self.plan: RawSplineTabmatPlan | None = None
        self.built = False

    def get(self) -> tuple[RawSplineTabmatPlan | None, bool, float]:
        if self.built:
            return self.plan, False, 0.0
        started = perf_counter()
        self.plan = _build_raw_spline_tabmat_plan(self.group_matrices, n=self.n)
        elapsed = perf_counter() - started
        self.built = True
        return self.plan, True, elapsed

    def clear(self) -> None:
        self.plan = None
        self.built = False


class DesignMatrix:
    """Container for per-group matrices. Provides full-matrix operations."""

    def __init__(self, group_matrices: list[GroupMatrix], n: int, p: int):
        matrices = tuple(group_matrices)
        grouped_shape = (
            n if not matrices else matrices[0].shape[0],
            sum(matrix.shape[1] for matrix in matrices),
        )
        rows_match = all(matrix.shape[0] == n for matrix in matrices)
        if not rows_match or grouped_shape != (n, p):
            actual_rows: int | str = grouped_shape[0] if rows_match else "inconsistent"
            raise ValueError(
                f"declared design shape {(n, p)} does not match grouped shape "
                f"{(actual_rows, grouped_shape[1])}"
            )
        self.group_matrices = matrices
        self.n = n
        self.p = p
        self.shape = (n, p)
        self._tabmat_holder = _LazyTabmatSplit(self.group_matrices)
        self._raw_spline_tabmat_holder = _LazyRawSplineTabmatPlan(self.group_matrices, n=n)
        self._tabmat_centering_candidate = None
        self._tabmat_vector_candidate = _is_retained_tabmat_vector_candidate(
            self.group_matrices,
            n=n,
        )
        self._execution_plan: MatrixExecutionPlan | None = None
        self._mixed_bin_space_centering_plan: MixedBinSpaceCenteringPlan | None = None
        self._mixed_bin_space_centering_plan_attempted = False
        self._centered_pattern_plan = None
        self._centered_solver_supports = None
        self._scalar_structured_layout_cache: dict[object, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Serialize durable matrix state without the rebuildable execution plan."""
        state = self.__dict__.copy()
        state["_execution_plan"] = None
        state["_raw_spline_tabmat_holder"] = _LazyRawSplineTabmatPlan(
            self.group_matrices,
            n=self.n,
        )
        state.pop("_mixed_centering_execution_plan", None)
        state["_mixed_bin_space_centering_plan"] = None
        state["_mixed_bin_space_centering_plan_attempted"] = False
        state["_scalar_structured_layout_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore current and pre-shared-Tabmat-holder pickle state."""
        state = state.copy()
        legacy_split = state.pop("_tabmat_split", None)
        legacy_built = bool(state.pop("_tabmat_built", False))
        group_matrices = tuple(state["group_matrices"])
        state["group_matrices"] = group_matrices

        holder = state.get("_tabmat_holder")
        if not isinstance(holder, _LazyTabmatSplit):
            holder = _LazyTabmatSplit(group_matrices)
            holder.split = legacy_split
            holder.built = legacy_built
        else:
            holder.group_matrices = group_matrices
        state["_tabmat_holder"] = holder
        state["_raw_spline_tabmat_holder"] = _LazyRawSplineTabmatPlan(
            group_matrices,
            n=int(state["n"]),
        )

        state["_execution_plan"] = None
        state.pop("_mixed_centering_execution_plan", None)
        state["_mixed_bin_space_centering_plan"] = None
        state["_mixed_bin_space_centering_plan_attempted"] = False
        state["_scalar_structured_layout_cache"] = {}
        state.setdefault("_tabmat_centering_candidate", None)
        state["_tabmat_vector_candidate"] = _is_retained_tabmat_vector_candidate(
            group_matrices,
            n=int(state["n"]),
        )
        state.setdefault("_centered_pattern_plan", None)
        state.setdefault("_centered_solver_supports", None)
        self.__dict__.update(state)

    def _get_or_build_tabmat_split(self):
        """Return the single DesignMatrix-owned Tabmat split, building it once."""
        return self._tabmat_holder.get()

    @property
    def _tabmat_split(self):
        """Compatibility view of the shared lazy split."""
        return self._tabmat_holder.split

    @property
    def _tabmat_built(self) -> bool:
        """Compatibility view of whether split construction was attempted."""
        return self._tabmat_holder.built

    @property
    def tabmat_split(self):
        """Lazily build a tabmat SplitMatrix for non-discrete paths."""
        return self._get_or_build_tabmat_split()

    @property
    def tabmat_centering_split(self):
        """Build Tabmat only when the centered solver can dispatch to it."""
        if self._tabmat_centering_candidate is None:
            self._tabmat_centering_candidate = _is_tabmat_centering_candidate(self.group_matrices)
        if not self._tabmat_centering_candidate:
            return None
        return self.tabmat_split

    @property
    def raw_spline_tabmat_plan_built(self) -> bool:
        """Return whether raw-spline plan construction has been attempted."""
        return self._raw_spline_tabmat_holder.built

    def get_raw_spline_tabmat_centering_plan(
        self,
        *,
        profile: dict | None = None,
    ) -> RawSplineTabmatPlan | None:
        """Return the lazy raw-spline plan and record its one-time policy decision."""
        plan, newly_built, elapsed = self._raw_spline_tabmat_holder.get()
        if profile is not None and newly_built:
            if plan is None:
                profile["centered_spline_tabmat_policy_rejections"] = (
                    profile.get("centered_spline_tabmat_policy_rejections", 0) + 1
                )
            else:
                profile["centered_spline_tabmat_builds"] = (
                    profile.get("centered_spline_tabmat_builds", 0) + 1
                )
                profile["centered_spline_tabmat_build_s"] = (
                    profile.get("centered_spline_tabmat_build_s", 0.0) + elapsed
                )
        if profile is not None and plan is not None:
            profile["centered_spline_tabmat_retained_bytes"] = max(
                profile.get("centered_spline_tabmat_retained_bytes", 0),
                plan.retained_bytes,
            )
        return plan

    def release_raw_spline_tabmat_plan(self) -> None:
        """Release the optional CSC/CSR acceleration cache after fit publication."""
        self._raw_spline_tabmat_holder.clear()

    @property
    def execution_plan(self) -> MatrixExecutionPlan:
        """Return the cached backend-neutral matrix execution plan."""
        plan = self._execution_plan
        if plan is None:
            plan = MatrixExecutionPlan(
                self.group_matrices,
                n=self.n,
                ordinary_split_factory=self._tabmat_holder.get,
            )
            if plan.shape != self.shape:
                raise ValueError(
                    f"declared design shape {self.shape} does not match grouped shape {plan.shape}"
                )
            self._execution_plan = plan
        return plan

    @property
    def mixed_bin_space_centering_plan(self) -> MixedBinSpaceCenteringPlan | None:
        """Return the one cached augmented Tabmat plan for supported mixed layouts."""
        plan = self._mixed_bin_space_centering_plan
        if plan is None and not self._mixed_bin_space_centering_plan_attempted:
            plan = build_mixed_bin_space_centering_plan(
                self.group_matrices,
                n=self.n,
                p=self.p,
            )
            self._mixed_bin_space_centering_plan = plan
            self._mixed_bin_space_centering_plan_attempted = True
        if plan is not None and plan.shape != self.shape:
            raise ValueError(
                f"cached bin-space plan shape {plan.shape} does not match "
                f"declared design shape {self.shape}"
            )
        return plan

    def matvec(self, beta: NDArray) -> NDArray:
        """X @ beta via per-group matvecs."""
        holder = self._tabmat_holder
        if self._tabmat_vector_candidate and holder.split is not None:
            return np.asarray(holder.split.matvec(_tabmat_vector(beta)), dtype=np.float64)
        result = np.zeros(self.n)
        col = 0
        for gm in self.group_matrices:
            p_g = gm.shape[1]
            result += gm.matvec(beta[col : col + p_g])
            col += p_g
        return result

    def rmatvec(self, w: NDArray) -> NDArray:
        """X.T @ w via per-group rmatvecs."""
        holder = self._tabmat_holder
        if self._tabmat_vector_candidate and holder.split is not None:
            return np.asarray(holder.split.transpose_matvec(_tabmat_vector(w)), dtype=np.float64)
        result = np.zeros(self.p)
        col = 0
        for gm in self.group_matrices:
            p_g = gm.shape[1]
            result[col : col + p_g] = gm.rmatvec(w)
            col += p_g
        return result

    def toarray(self) -> NDArray:
        """Concatenate per-group arrays into full (n, p) dense matrix."""
        return np.hstack([gm.toarray() for gm in self.group_matrices])

    def row_subset(self, idx: NDArray) -> DesignMatrix:
        """Return a new DesignMatrix with only the rows at idx."""
        return DesignMatrix(
            [gm.row_subset(idx) for gm in self.group_matrices],
            len(idx),
            self.p,
        )
