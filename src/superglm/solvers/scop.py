"""SCOP (Shape Constrained P-spline) reparameterization.

Implements the raw-space and solver-space SCOP maps used for fit-time
shape-constrained P-splines. Monotone constraints use a one-dimensional
constant null space; curvature constraints use a two-dimensional affine
null space.

Two levels:

- ``SCOPReparameterization``: raw-space SCOP map (q-dimensional).
- ``SCOPSolverReparam``: solver-space wrapper (q_eff = q_raw - null_dim)
  after identifiability drops the null-space block and centers the
  remaining shape columns.

References
----------
Pya & Wood 2015, "Shape constrained additive models", Statistics and
Computing, 25, 543-559.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.solvers.constrained_qp import solve_constrained_qp

logger = logging.getLogger(__name__)

SCOPKind = str
_VALID_KINDS = {"increasing", "decreasing", "convex", "concave"}


def _warn_qp_initialization_fallback(space: str) -> None:
    """Warn that a ``qp_initialize`` QP did not converge.

    ``space`` is ``"raw-space"`` or ``"solver-space"``: the two initialization
    paths must stay distinguishable in a log, which is the only thing that
    differs between them.  Interpolated lazily so the rendered text is
    byte-identical to the two hand-written messages this replaced.
    """
    logger.warning(
        "SCOP %s QP initialization did not converge; falling back "
        "to an approximate shape-constrained starting point.",
        space,
    )


def _resolve_kind(kind: SCOPKind | None, direction: str | None) -> SCOPKind:
    resolved = kind or direction or "increasing"
    if kind is not None and direction is not None and kind != direction:
        raise ValueError(f"Conflicting SCOP kind={kind!r} and direction={direction!r}")
    if resolved not in _VALID_KINDS:
        raise ValueError(f"SCOP kind must be one of {sorted(_VALID_KINDS)}, got {resolved!r}")
    return resolved


def _constant_null_basis(q: int) -> NDArray:
    return np.ones((q, 1), dtype=np.float64)


def _monotone_sigma(q: int) -> NDArray:
    return np.tril(np.ones((q, q - 1), dtype=np.float64), k=-1)


def _second_order_sigma(q: int) -> NDArray:
    if q <= 2:
        return np.zeros((q, 0), dtype=np.float64)
    # Shift by two rows so the shape block contributes no affine component.
    base = np.tril(np.ones((q, q - 2), dtype=np.float64), k=-2)
    return np.cumsum(base, axis=0)


def _affine_null_basis(q: int) -> NDArray:
    idx = np.arange(q, dtype=np.float64)
    return np.column_stack([np.ones(q, dtype=np.float64), idx])


@dataclass
class SCOPReparameterization:
    """Raw-space SCOP reparameterization for one shape-constrained smooth."""

    q: int
    kind: str
    null_dim: int
    null_basis: NDArray
    sigma_shape: NDArray
    shape_sign: float

    @property
    def Sigma(self) -> NDArray:
        """Legacy full beta_tilde -> gamma map."""
        return np.concatenate([self.null_basis, self.shape_matrix], axis=1)

    @property
    def shape_dim(self) -> int:
        return self.q - self.null_dim

    @property
    def shape_matrix(self) -> NDArray:
        return self.shape_sign * self.sigma_shape

    @property
    def direction(self) -> str:
        """Legacy alias retained for monotone call sites."""
        return self.kind

    def beta_tilde(self, beta: NDArray) -> NDArray:
        """Positivity-transformed vector in null-plus-shape coordinates."""
        bt = np.array(beta, copy=True)
        if self.shape_dim > 0:
            bt[self.null_dim :] = np.exp(np.clip(bt[self.null_dim :], -500, 500))
        return bt

    def forward(self, beta: NDArray) -> NDArray:
        """Forward map: beta -> gamma."""
        beta_null = beta[: self.null_dim]
        beta_shape = np.exp(np.clip(beta[self.null_dim :], -500, 500))
        return self.null_basis @ beta_null + self.shape_matrix @ beta_shape

    def jacobian(self, beta: NDArray) -> NDArray:
        """Jacobian d(gamma)/d(beta), shape (q, q)."""
        J = np.zeros((self.q, self.q), dtype=np.float64)
        J[:, : self.null_dim] = self.null_basis
        for i in range(self.shape_dim):
            j = self.null_dim + i
            J[:, j] = self.shape_matrix[:, i] * np.exp(np.clip(beta[j], -500, 500))
        return J

    def penalty_matrix(self) -> NDArray:
        """SCOP penalty matrix on the shape block."""
        S = np.zeros((self.q, self.q), dtype=np.float64)
        if self.shape_dim <= 1:
            return S
        D = np.diff(np.eye(self.shape_dim), axis=0)
        S[self.null_dim :, self.null_dim :] = D.T @ D
        return S

    def initialize_from_gamma(self, gamma: NDArray, floor: float = 1e-6) -> NDArray:
        """Recover beta from gamma, clamping infeasible shape differences."""
        beta = np.empty(self.q, dtype=np.float64)
        if self.kind in {"increasing", "decreasing"}:
            diffs = np.diff(gamma)
            if self.kind == "decreasing":
                diffs = -diffs
            diffs = np.maximum(diffs, floor)
            beta[0] = gamma[0]
            beta[1:] = np.log(diffs)
            return beta

        beta[0] = gamma[0]
        beta[1] = gamma[1] - gamma[0]
        curv = np.diff(gamma, n=2)
        if self.kind == "concave":
            curv = -curv
        curv = np.maximum(curv, floor)
        beta[2:] = np.log(curv)
        return beta

    def qp_initialize(
        self,
        B: NDArray,
        y: NDArray,
        lambda_penalty: float = 0.01,
        weights: NDArray | None = None,
    ) -> NDArray:
        """SCAM-style QP initialization in beta_tilde space."""
        X = B @ self.Sigma
        if weights is not None:
            W = np.sqrt(weights)
            X_w = X * W[:, None]
            y_w = y * W
        else:
            X_w = X
            y_w = y

        H = X_w.T @ X_w + lambda_penalty * self.penalty_matrix()
        H += 1e-8 * np.eye(self.q)
        g = X_w.T @ y_w

        A = np.zeros((self.shape_dim, self.q), dtype=np.float64)
        for i in range(self.shape_dim):
            A[i, self.null_dim + i] = 1.0
        b = np.zeros(self.shape_dim, dtype=np.float64)

        result = solve_constrained_qp(H, g, A, b)
        if not result.converged:
            _warn_qp_initialization_fallback("raw-space")
        beta_tilde_init = result.beta

        beta = np.array(beta_tilde_init, copy=True)
        if self.shape_dim > 0:
            beta[self.null_dim :] = np.log(np.maximum(beta_tilde_init[self.null_dim :], 1e-8))
        return beta


def build_scop_reparam(
    q: int,
    kind: SCOPKind | None = None,
    *,
    direction: str | None = None,
) -> SCOPReparameterization:
    """Build a raw SCOP reparameterization."""
    resolved_kind = _resolve_kind(kind, direction)
    if resolved_kind in {"increasing", "decreasing"}:
        null_dim = 1
        null_basis = _constant_null_basis(q)
        sigma_shape = _monotone_sigma(q)
        shape_sign = 1.0 if resolved_kind == "increasing" else -1.0
    else:
        null_dim = 2
        null_basis = _affine_null_basis(q)
        sigma_shape = _second_order_sigma(q)
        shape_sign = 1.0 if resolved_kind == "convex" else -1.0

    if q < null_dim:
        raise ValueError(
            f"SCOP reparameterization for kind={resolved_kind!r} requires q >= {null_dim}, got {q}"
        )

    return SCOPReparameterization(
        q=q,
        kind=resolved_kind,
        null_dim=null_dim,
        null_basis=null_basis,
        sigma_shape=sigma_shape,
        shape_sign=shape_sign,
    )


@dataclass
class SCOPSolverReparam:
    """Solver-space SCOP reparameterization on the shape block only."""

    q: int
    raw_reparam: SCOPReparameterization

    @property
    def null_dim(self) -> int:
        return self.raw_reparam.null_dim

    @property
    def kind(self) -> str:
        return self.raw_reparam.kind

    @property
    def direction(self) -> str:
        """Legacy alias retained for monotone call sites."""
        return self.raw_reparam.direction

    def _embed(self, beta_eff: NDArray) -> NDArray:
        """Embed solver-space beta_eff into raw-space beta."""
        beta_raw = np.zeros(self.raw_reparam.q, dtype=np.float64)
        beta_raw[self.null_dim :] = beta_eff
        return beta_raw

    def forward(self, beta_eff: NDArray) -> NDArray:
        """Forward map in solver space: beta_eff -> beta_tilde_shape."""
        return np.exp(np.clip(beta_eff, -500, 500))

    def beta_tilde_eff(self, beta_eff: NDArray) -> NDArray:
        """Effective positivity-transformed vector: exp(beta_eff)."""
        return self.forward(beta_eff)

    def jacobian(self, beta_eff: NDArray) -> NDArray:
        """Jacobian d(forward)/d(beta_eff), shape (q_eff, q_eff)."""
        return np.diag(np.exp(np.clip(beta_eff, -500, 500)))

    def penalty_matrix(self) -> NDArray:
        """Penalty matrix in solver space on the shape block."""
        if self.q <= 1:
            return np.zeros((self.q, self.q), dtype=np.float64)
        D = np.diff(np.eye(self.q), axis=0)
        return D.T @ D

    def initialize_from_gamma(self, gamma: NDArray, floor: float = 1e-6) -> NDArray:
        """Recover solver-space beta_eff from gamma."""
        beta_raw = self.raw_reparam.initialize_from_gamma(gamma, floor=floor)
        return beta_raw[self.null_dim :]

    def qp_initialize(self, B_centered: NDArray, y: NDArray, **kwargs) -> NDArray:
        """QP initialization in solver space."""
        weights = kwargs.get("weights", None)
        lambda_penalty = kwargs.get("lambda_penalty", 0.01)

        if weights is not None:
            W = np.sqrt(weights)
            X_w = B_centered * W[:, None]
            y_w = y * W
        else:
            X_w = B_centered
            y_w = y

        H = X_w.T @ X_w + lambda_penalty * self.penalty_matrix()
        H += 1e-8 * np.eye(self.q)
        g = X_w.T @ y_w

        A = np.eye(self.q, dtype=np.float64)
        b = np.zeros(self.q, dtype=np.float64)

        result = solve_constrained_qp(H, g, A, b)
        if not result.converged:
            _warn_qp_initialization_fallback("solver-space")
        beta_tilde_eff = result.beta
        return np.log(np.maximum(beta_tilde_eff, 1e-8))


def build_scop_solver_reparam(
    q_raw: int,
    kind: SCOPKind | None = None,
    *,
    direction: str | None = None,
) -> SCOPSolverReparam:
    """Build a solver-space SCOP reparameterization."""
    raw = build_scop_reparam(q_raw, kind=kind, direction=direction)
    return SCOPSolverReparam(q=q_raw - raw.null_dim, raw_reparam=raw)
