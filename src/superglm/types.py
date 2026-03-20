"""
Core types and protocols.

Every feature spec (Spline, Categorical, Numeric) implements FeatureSpec.
The solver never sees feature types — only GroupInfo objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


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
    # select=True subgroup support (mgcv-style double penalty)
    subgroup_name: str | None = None  # "linear" or "spline"
    projection: NDArray | None = None  # (K, n_cols) projection from B-spline basis

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

    def __post_init__(self):
        if not self.feature_name:
            self.feature_name = self.name

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def sl(self) -> slice:
        return slice(self.start, self.end)


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
