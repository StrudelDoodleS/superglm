"""SCOP (Shape Constrained P-spline) reparameterization.

Implements the raw-space and solver-space SCOP maps used for fit-time
shape-constrained P-splines. Monotone constraints use a one-dimensional
constant null space; curvature constraints use a two-dimensional affine
null space.

Two levels:

- ``SCOPReparameterization``: raw-space SCOP map (q-dimensional).
- ``SCOPSolverReparam``: solver-space wrapper after identifiability drops
  the constant direction and centers the remaining columns. Curvature
  constraints retain an identity-mapped affine slope coordinate; only the
  curvature coordinates are positivity transformed.

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
    paths must stay distinguishable in a log.
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


def _curvature_scop_basis(
    q: int,
    *,
    knots: NDArray | None,
    degree: int | None,
    domain: tuple[float, float] | None,
) -> tuple[NDArray, NDArray]:
    """Build the exact affine null and positive-curvature right-inverse."""
    if knots is None or degree is None:
        raise ValueError("Curvature SCOP reparameterization requires the spline knots and degree")
    knots_arr = np.asarray(knots, dtype=np.float64)
    if knots_arr.ndim != 1 or len(knots_arr) != q + degree + 1:
        raise ValueError("Curvature SCOP knots do not match q and degree")

    # Local import avoids making the feature package part of the solver's
    # import-time dependency graph.
    from superglm.features._spline_constraints import curvature_difference_operator

    curvature = curvature_difference_operator(
        knots_arr,
        degree,
        domain=domain,
    )
    expected_shape = (max(q - 2, 0), q)
    if curvature.shape != expected_shape:
        raise ValueError(
            "Curvature SCOP domain must retain one independent row per "
            "non-affine coefficient direction"
        )

    active_lo = float(knots_arr[degree])
    active_hi = float(knots_arr[q])
    normalized_knots = (knots_arr - active_lo) / (active_hi - active_lo)
    greville = np.asarray(
        [np.mean(normalized_knots[index + 1 : index + degree + 1]) for index in range(q)],
        dtype=np.float64,
    )
    null_basis = np.column_stack((np.ones(q, dtype=np.float64), greville))

    if q <= 2:
        return null_basis, np.zeros((q, 0), dtype=np.float64)

    sigma_shape, _, rank, _ = np.linalg.lstsq(
        curvature,
        np.eye(q - 2, dtype=np.float64),
        rcond=None,
    )
    if rank != q - 2 or not np.all(np.isfinite(sigma_shape)):
        raise ValueError("Curvature SCOP constraint geometry is numerically rank-deficient")

    identity_error = curvature @ sigma_shape - np.eye(q - 2, dtype=np.float64)
    null_error = curvature @ null_basis
    geometry_scale = max(
        1.0,
        float(np.linalg.norm(curvature, ord=np.inf))
        * float(np.linalg.norm(sigma_shape, ord=np.inf)),
    )
    tolerance = 4096.0 * np.finfo(np.float64).eps * geometry_scale
    if (
        float(np.linalg.norm(identity_error, ord=np.inf)) > tolerance
        or float(np.linalg.norm(null_error, ord=np.inf)) > tolerance
    ):
        raise ValueError("Curvature SCOP reparameterization failed its geometry certificate")
    return null_basis, sigma_shape


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
        beta = np.linalg.solve(self.Sigma, np.asarray(gamma, dtype=np.float64))
        if self.shape_dim:
            beta[self.null_dim :] = np.log(np.maximum(beta[self.null_dim :], floor))
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
    knots: NDArray | None = None,
    degree: int | None = None,
    domain: tuple[float, float] | None = None,
) -> SCOPReparameterization:
    """Build a raw SCOP reparameterization.

    Curvature constraints require the fitted knot vector and degree so the
    free affine direction and positive-curvature block live in function space,
    rather than in a generally non-uniform coefficient-index geometry.
    """
    resolved_kind = _resolve_kind(kind, direction)
    if resolved_kind in {"increasing", "decreasing"}:
        null_dim = 1
        null_basis = _constant_null_basis(q)
        sigma_shape = _monotone_sigma(q)
        shape_sign = 1.0 if resolved_kind == "increasing" else -1.0
    else:
        null_dim = 2
        null_basis, sigma_shape = _curvature_scop_basis(
            q,
            knots=knots,
            degree=degree,
            domain=domain,
        )
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
    """Solver-space SCOP reparameterization after constant removal."""

    q: int
    raw_reparam: SCOPReparameterization

    @property
    def null_dim(self) -> int:
        return self.raw_reparam.null_dim

    @property
    def free_dim(self) -> int:
        """Number of identity-mapped coordinates retained in solver space."""
        return self.raw_reparam.null_dim - 1

    @property
    def shape_dim(self) -> int:
        """Number of positivity-mapped shape coordinates."""
        return self.q - self.free_dim

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
        beta_raw[1:] = beta_eff
        return beta_raw

    def forward(self, beta_eff: NDArray) -> NDArray:
        """Map latent coordinates to one free block plus positive shape block."""
        mapped = np.array(beta_eff, dtype=np.float64, copy=True)
        if self.shape_dim:
            mapped[self.free_dim :] = np.exp(np.clip(mapped[self.free_dim :], -500, 500))
        return mapped

    def beta_tilde_eff(self, beta_eff: NDArray) -> NDArray:
        """Effective mixed identity/positivity-transformed vector."""
        return self.forward(beta_eff)

    def jacobian_diagonal(self, beta_eff: NDArray) -> NDArray:
        """Diagonal of ``d forward(beta_eff) / d beta_eff``."""
        diagonal = np.ones(self.q, dtype=np.float64)
        if self.shape_dim:
            diagonal[self.free_dim :] = np.exp(np.clip(beta_eff[self.free_dim :], -500, 500))
        return diagonal

    def second_derivative_diagonal(self, beta_eff: NDArray) -> NDArray:
        """Elementwise second derivatives of the solver-space map."""
        diagonal = np.zeros(self.q, dtype=np.float64)
        if self.shape_dim:
            diagonal[self.free_dim :] = np.exp(np.clip(beta_eff[self.free_dim :], -500, 500))
        return diagonal

    def jacobian(self, beta_eff: NDArray) -> NDArray:
        """Jacobian d(forward)/d(beta_eff), shape (q_eff, q_eff)."""
        return np.diag(self.jacobian_diagonal(beta_eff))

    def penalty_matrix(self) -> NDArray:
        """Penalty matrix in solver space on the shape block."""
        S = np.zeros((self.q, self.q), dtype=np.float64)
        if self.shape_dim <= 1:
            return S
        D = np.diff(np.eye(self.shape_dim), axis=0)
        S[self.free_dim :, self.free_dim :] = D.T @ D
        return S

    def initialize_from_gamma(self, gamma: NDArray, floor: float = 1e-6) -> NDArray:
        """Recover solver-space beta_eff from gamma."""
        beta = np.array(gamma, dtype=np.float64, copy=True)
        if self.shape_dim:
            beta[self.free_dim :] = np.log(np.maximum(beta[self.free_dim :], floor))
        return beta

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

        A = np.zeros((self.shape_dim, self.q), dtype=np.float64)
        if self.shape_dim:
            rows = np.arange(self.shape_dim)
            A[rows, self.free_dim + rows] = 1.0
        b = np.zeros(self.shape_dim, dtype=np.float64)

        result = solve_constrained_qp(H, g, A, b)
        if not result.converged:
            _warn_qp_initialization_fallback("solver-space")
        beta_tilde_eff = result.beta
        beta = np.array(beta_tilde_eff, copy=True)
        if self.shape_dim:
            beta[self.free_dim :] = np.log(np.maximum(beta_tilde_eff[self.free_dim :], 1e-8))
        return beta


def build_scop_solver_reparam(
    q_raw: int,
    kind: SCOPKind | None = None,
    *,
    direction: str | None = None,
    knots: NDArray | None = None,
    degree: int | None = None,
    domain: tuple[float, float] | None = None,
) -> SCOPSolverReparam:
    """Build a solver-space SCOP reparameterization."""
    raw = build_scop_reparam(
        q_raw,
        kind=kind,
        direction=direction,
        knots=knots,
        degree=degree,
        domain=domain,
    )
    return SCOPSolverReparam(q=q_raw - 1, raw_reparam=raw)
