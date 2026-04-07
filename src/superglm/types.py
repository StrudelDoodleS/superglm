"""
Core types and protocols.

Every feature spec (Spline, Categorical, Numeric) implements FeatureSpec.
The solver never sees feature types — only GroupInfo objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.solvers.scop import SCOPSolverReparam


# ── Linear inequality constraints ──────────────────────────────
@dataclass
class LinearConstraintSet:
    """Linear inequality constraints: A @ theta >= b.

    Generic constraint representation, not spline-specific. The QP solver
    operates on these without knowing which basis produced them.

    Parameters
    ----------
    A : NDArray
        Constraint matrix, shape (n_constraints, n_params).
    b : NDArray
        Right-hand side, shape (n_constraints,).
    """

    A: NDArray
    b: NDArray

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def n_params(self) -> int:
        return self.A.shape[1]

    def is_feasible(self, theta: NDArray, tol: float = -1e-10) -> bool:
        """Check whether theta satisfies all constraints (A @ theta >= b)."""
        return bool(np.all(self.A @ theta - self.b >= tol))

    def compose(self, P: NDArray) -> LinearConstraintSet:
        """Map constraints through a projection: A_new = A @ P, b unchanged."""
        return LinearConstraintSet(A=self.A @ P, b=self.b.copy())


# ── Per-component lambda control ────────────────────────────────
@dataclass(frozen=True)
class LambdaPolicy:
    """Per-penalty-component lambda control for REML.

    Two modes:
    - ``estimate``: lambda optimised by REML, initialised from ``lambda2_init``.
    - ``fixed(value)``: lambda held at ``value`` throughout fitting.

    ``off()`` is sugar for ``fixed(0.0)`` — the penalty component contributes
    nothing to the penalty matrix.
    """

    mode: Literal["estimate", "fixed"]
    value: float | None = None

    def __post_init__(self):
        if self.mode == "fixed" and self.value is None:
            raise ValueError("LambdaPolicy(mode='fixed') requires a value")
        if self.mode not in ("estimate", "fixed"):
            raise ValueError(f"Invalid LambdaPolicy mode: {self.mode!r}")

    @classmethod
    def estimate(cls) -> LambdaPolicy:
        return cls(mode="estimate")

    @classmethod
    def fixed(cls, value: float) -> LambdaPolicy:
        return cls(mode="fixed", value=float(value))

    @classmethod
    def off(cls) -> LambdaPolicy:
        return cls.fixed(0.0)


# ── What a feature spec must provide ────────────────────────────
@runtime_checkable
class FeatureSpec(Protocol):
    """Protocol for feature specifications.

    build()       -> fit-time: learn parameters and return columns + metadata
    transform()   -> predict-time: apply stored parameters to new data
    reconstruct() -> turn fitted coefficients back into interpretable output
    """

    def build(
        self,
        x: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo | list[GroupInfo]: ...

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.floating]: ...

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]: ...


# ── What the solver actually sees ───────────────────────────────
@dataclass
class GroupInfo:
    """Uniform interface between feature specs and the solver.

    The solver receives a list of these. It doesn't know or care
    whether the columns came from a spline, a one-hot encoding,
    or a single numeric feature.
    """

    columns: NDArray[np.floating] | sp.spmatrix | None  # (n, p_g) design sub-matrix
    n_cols: int  # p_g — group size
    penalty_matrix: NDArray | None = None  # (p_g, p_g) for SSP, else None
    reparametrize: bool = False  # whether to apply SSP transform
    penalized: bool = True  # whether this group is subject to the penalty
    cat_codes: NDArray | None = None  # (n,) integer codes for categorical features
    # select=True subgroup support for double-penalty decomposition
    subgroup_name: str | None = None  # "linear" or "spline"
    projection: NDArray | None = None  # (K, n_cols) projection from B-spline basis
    # Multi-penalty: list of (suffix, omega) tuples for shared-block penalties.
    # Each omega is in the same basis as penalty_matrix (projected when projection present).
    # penalty_matrix should equal the sum of the component omegas.
    penalty_components: list[tuple[str, NDArray]] | None = None
    # Optional mapping suffix → component_type for penalty_components.
    # E.g. {"null": "selection"} marks null-space penalty as selection penalty.
    component_types: dict[str, str] | None = None
    # Optional per-component lambda control; keys match penalty_components suffixes.
    lambda_policies: dict[str, LambdaPolicy] | None = None
    # Monotone constraint metadata (Phase 2 QP engine).
    # constraints are in post-identifiability space after build(); they become
    # solver-ready only after the DM builder composes with R_inv.
    constraints: LinearConstraintSet | None = None
    monotone_engine: str | None = None
    raw_to_solver_map: NDArray | None = None
    # SCOP reparameterization for monotone P-splines (Phase 3).
    # Solver-space (q_eff = q_raw - 1 dimensional) — the solver calls
    # forward(), jacobian(), penalty_matrix() directly without dimension translation.
    scop_reparameterization: SCOPSolverReparam | None = None

    def __post_init__(self):
        if self.columns is None:
            # Discretized path: columns not needed, skip shape validation
            pass
        elif self.projection is not None:
            # select=True subgroup: columns is the full B-spline basis, projection maps to subspace
            if self.projection.shape[1] != self.n_cols:
                raise ValueError(
                    f"projection has {self.projection.shape[1]} cols but n_cols={self.n_cols}"
                )
            if self.projection.shape[0] != self.columns.shape[1]:
                raise ValueError(
                    f"projection has {self.projection.shape[0]} rows but "
                    f"columns has {self.columns.shape[1]} cols"
                )
        elif self.columns.shape[1] != self.n_cols:
            raise ValueError(f"columns has {self.columns.shape[1]} cols but n_cols={self.n_cols}")
        if self.penalty_matrix is not None and self.penalty_matrix.shape != (
            self.n_cols,
            self.n_cols,
        ):
            raise ValueError(
                f"penalty_matrix shape {self.penalty_matrix.shape} != ({self.n_cols}, {self.n_cols})"
            )
        if self.penalty_components is not None:
            pm_dim = (
                self.penalty_matrix.shape[0] if self.penalty_matrix is not None else self.n_cols
            )
            comp_sum = np.zeros((pm_dim, pm_dim))
            for suffix, omega in self.penalty_components:
                if omega.shape != (pm_dim, pm_dim):
                    raise ValueError(
                        f"penalty_component '{suffix}' shape {omega.shape} != ({pm_dim}, {pm_dim})"
                    )
                comp_sum += omega
            if self.penalty_matrix is not None and not np.allclose(
                comp_sum, self.penalty_matrix, atol=1e-12
            ):
                raise ValueError("penalty_components sum does not match penalty_matrix")


# ── Group bookkeeping for the solver ────────────────────────────
@dataclass
class GroupSlice:
    """Tracks where each group lives in the full coefficient vector."""

    name: str
    start: int
    end: int
    weight: float = 1.0  # sqrt(p_g) by default, or adaptive weight
    penalized: bool = True  # whether this group is subject to the penalty
    feature_name: str = ""  # parent feature name, defaults to name
    subgroup_type: str | None = None  # "linear", "spline", or None
    constraints: LinearConstraintSet | None = None
    monotone_engine: str | None = None
    scop_reparameterization: SCOPSolverReparam | None = None

    def __post_init__(self):
        if not self.feature_name:
            self.feature_name = self.name

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def sl(self) -> slice:
        return slice(self.start, self.end)


# ── Penalty components for multi-penalty REML ─────────────────
@dataclass
class PenaltyComponent:
    """One (omega, lambda) pair within a term's penalty structure.

    REML operates over a list of these — one per smoothing parameter.
    A single-penalty term has one PenaltyComponent; a multi-penalty
    term (e.g. tensor product, adaptive smooth) has several, each
    with its own lambda optimized by REML.

    The key separation: GroupSlice defines column blocks (term
    structure, used by PIRLS). PenaltyComponent defines smoothing
    parameters (penalty structure, used by REML derivatives).
    """

    name: str  # stable lambda key, e.g. "age:wiggle", "age:null"
    group_name: str  # parent GroupSlice name
    group_index: int  # index into groups list
    group_sl: slice  # coefficient slice from parent GroupSlice
    omega_raw: NDArray  # (K, K) penalty in B-spline / raw basis space
    omega_ssp: NDArray | None = None  # (p_g, p_g) in SSP coordinates
    rank: float = 0.0
    log_det_omega_plus: float = 0.0
    eigvals_omega: NDArray | None = None  # positive eigenvalues of omega_ssp
    component_type: str | None = None  # "selection" for null-space select penalty
    lambda_policy: LambdaPolicy | None = None  # per-component lambda control


# ── Tensor marginal ingredients ────────────────────────────────
@dataclass
class TensorMarginalInfo:
    """Pre-computed marginal ingredients for tensor product interactions.

    Produced by ``_SplineBase.tensor_marginal_ingredients()`` so that
    ``TensorInteraction`` can consume parent spline geometry (knot
    vectors, penalties, constraints) without rebuilding from scratch.
    """

    basis: NDArray  # (n, K_eff) centered + constrained basis
    penalty: NDArray  # (K_eff, K_eff) penalty in centered space
    knots: NDArray  # full knot vector (for evaluating at new points)
    lo: float  # training range lower bound
    hi: float  # training range upper bound
    projection: NDArray  # (K_raw, K_eff) raw→centered+constrained projection
    K_eff: int  # effective column count
    degree: int  # B-spline degree (for basis eval at new points)


# ── Discretized tensor build result ─────────────────────────────
@dataclass
class DiscreteTensorBuildResult:
    """Return type for TensorInteraction.build_discrete().

    Carries the materialized Kronecker product (for fallback compatibility)
    alongside the factored marginal bases and bin indices (for the fast
    DiscretizedTensorGroupMatrix path).
    """

    infos: GroupInfo | list[GroupInfo]
    B_joint: NDArray  # (n_pairs, K1*K2) materialized Kronecker
    pair_idx: NDArray  # (n,) observation → pair mapping
    B1_unique: NDArray  # (n_bins1, K1) first marginal at bin centers
    B2_unique: NDArray  # (n_bins2, K2) second marginal at bin centers
    idx1: NDArray  # (n,) first marginal bin index
    idx2: NDArray  # (n,) second marginal bin index


# ── Fit statistics (summary without sample arrays) ────────────
@dataclass(frozen=True)
class FitStats:
    """Scalar fit statistics computed at end of fit()/fit_reml().

    Stores everything ``summary()`` needs so that the model no longer
    has to cache training arrays (``_train_y``, ``_train_mu``).
    """

    log_likelihood: float
    null_log_likelihood: float
    null_deviance: float
    explained_deviance: float
    pearson_chi2: float
    n_obs: int
