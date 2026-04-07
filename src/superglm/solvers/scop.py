"""SCOP (Shape Constrained P-spline) reparameterization.

Implements the beta -> beta_tilde -> gamma chain from Pya & Wood (2015)
for structural monotonicity in P-splines.

Notation
--------
beta       = unconstrained working parameters (Newton operates here)
beta_tilde = (beta_1, exp(beta_2), ..., exp(beta_q)) -- positivity-transformed
gamma      = Sigma @ beta_tilde -- basis coefficients (monotone by construction)
Sigma      = lower-triangular cumulative-sum matrix
J          = d(gamma)/d(beta) -- lower-triangular cumulative Jacobian

Two levels:

- ``SCOPReparameterization``: raw-space SCOP map (q-dimensional).
- ``SCOPSolverReparam``: solver-space wrapper (q_eff = q_raw - 1) after
  SCAM-style identifiability (drop constant column, center).

References
----------
Pya & Wood 2015, "Shape constrained additive models", Statistics and
Computing, 25, 543-559.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SCOPReparameterization:
    """Raw-space SCOP reparameterization for a monotone smooth.

    Carries forward map, Jacobian, penalty, and initialization
    logic for one monotone PSpline term.
    """

    q: int
    direction: str  # "increasing" or "decreasing"
    Sigma: NDArray  # (q, q) cumulative-sum matrix

    def beta_tilde(self, beta: NDArray) -> NDArray:
        """Positivity-transformed vector: (beta_1, exp(beta_2), ...)."""
        bt = np.empty_like(beta)
        bt[0] = beta[0]
        bt[1:] = np.exp(np.clip(beta[1:], -500, 500))
        return bt

    def forward(self, beta: NDArray) -> NDArray:
        """Forward map: beta -> gamma = Sigma @ beta_tilde(beta)."""
        return self.Sigma @ self.beta_tilde(beta)

    def jacobian(self, beta: NDArray) -> NDArray:
        """Jacobian d(gamma)/d(beta), shape (q, q).

        Column 0 (beta_1): Sigma[:, 0] = all ones (increasing) or mixed.
        Column i (i >= 1): Sigma[:, i] * exp(beta_i) in rows j >= i.
        Lower-triangular because gamma_j depends only on beta_1..beta_j.
        """
        q = self.q
        J = np.zeros((q, q))
        # Column 0: d(gamma)/d(beta_1) = Sigma[:, 0] (no exp transform)
        J[:, 0] = self.Sigma[:, 0]
        # Columns 1..q-1: d(gamma)/d(beta_i) = Sigma[:, i] * exp(beta_i)
        for i in range(1, q):
            J[:, i] = self.Sigma[:, i] * np.exp(np.clip(beta[i], -500, 500))
        return J

    def penalty_matrix(self) -> NDArray:
        """SCOP penalty matrix (q x q).

        First-difference penalty on (beta_2, ..., beta_q).
        S[i,j] = (D^T D) where D is the first-difference matrix on
        components 2..q. beta_1 is unpenalized (row/col 0 is zero).
        """
        q = self.q
        S = np.zeros((q, q))
        if q <= 2:
            return S
        # D is (q-2, q-1) first-difference matrix on beta[1:]
        D = np.diff(np.eye(q - 1), axis=0)
        S[1:, 1:] = D.T @ D
        return S

    def initialize_from_gamma(self, gamma: NDArray, floor: float = 1e-6) -> NDArray:
        """Recover beta from gamma.

        Clamps non-monotone differences to a small positive floor
        so that log() is well-defined.
        """
        q = self.q
        beta = np.empty(q)
        if self.direction == "increasing":
            diffs = np.diff(gamma)
            diffs = np.maximum(diffs, floor)
            beta[0] = gamma[0]
            beta[1:] = np.log(diffs)
        else:
            diffs = -np.diff(gamma)
            diffs = np.maximum(diffs, floor)
            beta[0] = gamma[0]
            beta[1:] = np.log(diffs)
        return beta

    def qp_initialize(
        self,
        B: NDArray,
        y: NDArray,
        lambda_penalty: float = 0.01,
        weights: NDArray | None = None,
    ) -> NDArray:
        """SCAM-style QP initialization in beta_tilde space.

        Minimize ||y - B @ Sigma @ beta_tilde||^2 + lambda * ||D @ beta_tilde[1:]||^2
        subject to beta_tilde[i] >= 0 for i >= 1.

        The objective is quadratic in beta_tilde, so the QP solver can
        find the global optimum directly.
        """
        from superglm.solvers.constrained_qp import solve_constrained_qp

        q = self.q
        # Design in beta_tilde space: X = B @ Sigma
        X = B @ self.Sigma
        if weights is not None:
            W = np.sqrt(weights)
            X_w = X * W[:, None]
            y_w = y * W
        else:
            X_w = X
            y_w = y

        H = X_w.T @ X_w

        # Penalty on beta_tilde[1:] (first-difference)
        if q > 2:
            D = np.diff(np.eye(q - 1), axis=0)
            S_tilde = np.zeros((q, q))
            S_tilde[1:, 1:] = D.T @ D
            H += lambda_penalty * S_tilde
        H += 1e-8 * np.eye(q)

        g = X_w.T @ y_w

        # Constraints: beta_tilde[i] >= 0 for i >= 1 (components 2..q)
        # A @ beta_tilde >= 0 where A selects components 1..q-1
        A = np.zeros((q - 1, q))
        for i in range(q - 1):
            A[i, i + 1] = 1.0
        b = np.zeros(q - 1)

        result = solve_constrained_qp(H, g, A, b)
        beta_tilde_init = result.beta

        # Convert beta_tilde -> beta: beta_i = log(beta_tilde_i) for i >= 1
        beta = np.empty(q)
        beta[0] = beta_tilde_init[0]
        beta[1:] = np.log(np.maximum(beta_tilde_init[1:], 1e-8))

        return beta


def build_scop_reparam(q: int, direction: str = "increasing") -> SCOPReparameterization:
    """Build a raw SCOP reparameterization.

    Parameters
    ----------
    q : int
        Number of basis functions (raw dimension).
    direction : str
        "increasing" or "decreasing".
    """
    if direction == "increasing":
        Sigma = np.tril(np.ones((q, q)))
    elif direction == "decreasing":
        Sigma = np.tril(np.ones((q, q)))
        Sigma[:, 1:] = -Sigma[:, 1:]
    else:
        raise ValueError(f"direction must be 'increasing' or 'decreasing', got '{direction}'")

    return SCOPReparameterization(q=q, direction=direction, Sigma=Sigma)


@dataclass
class SCOPSolverReparam:
    """Solver-space SCOP reparameterization (q_eff = q_raw - 1 dimensional).

    After SCAM-style identifiability (drop constant column, center), the
    solver works with beta_eff (length q_eff), which corresponds to the
    positive-increment parameters (beta_2, ..., beta_q) from the raw SCOP
    map. beta_1 is absorbed into the intercept.

    This wrapper embeds solver-space into raw-space via:
        beta_raw = (0, beta_eff_1, ..., beta_eff_{q_eff})
    then delegates to the raw SCOPReparameterization.

    All solver-facing APIs operate in q_eff dimensions.
    """

    q: int  # q_eff = solver dimension
    raw_reparam: SCOPReparameterization  # q_raw = q_eff + 1

    def _embed(self, beta_eff: NDArray) -> NDArray:
        """Embed solver-space beta_eff into raw-space beta."""
        beta_raw = np.zeros(self.q + 1)
        beta_raw[1:] = beta_eff
        return beta_raw

    def forward(self, beta_eff: NDArray) -> NDArray:
        """Forward map: beta_eff -> coefficients that multiply B_centered.

        Since B_centered = (B @ Sigma)[:, 1:] - col_means, the Sigma
        cumulative sum is already baked into the design matrix. The
        coefficients are therefore beta_tilde_eff = exp(beta_eff),
        the positive increment block — NOT gamma = Sigma @ beta_tilde
        (which would apply the cumulative sum twice).

        Returns (q_eff,) vector: eta_scop = B_centered @ forward(beta_eff).
        """
        return np.exp(np.clip(beta_eff, -500, 500))

    def beta_tilde_eff(self, beta_eff: NDArray) -> NDArray:
        """Effective positivity-transformed vector: exp(beta_eff).

        Same as forward() — kept for semantic clarity.
        """
        return self.forward(beta_eff)

    def jacobian(self, beta_eff: NDArray) -> NDArray:
        """Jacobian d(forward)/d(beta_eff), shape (q_eff, q_eff).

        Since forward(beta_eff) = exp(beta_eff) element-wise, the
        Jacobian is diagonal: diag(exp(beta_eff)).
        """
        return np.diag(np.exp(np.clip(beta_eff, -500, 500)))

    def penalty_matrix(self) -> NDArray:
        """Penalty matrix in solver space (q_eff x q_eff).

        First-difference penalty on the effective parameters.
        All components are exp-mapped increments, so all are penalized.
        """
        q_eff = self.q
        if q_eff <= 1:
            return np.zeros((q_eff, q_eff))
        D = np.diff(np.eye(q_eff), axis=0)
        return D.T @ D

    def initialize_from_gamma(self, gamma: NDArray, floor: float = 1e-6) -> NDArray:
        """Recover solver-space beta_eff from gamma.

        Delegates to raw reparam then drops beta_1.
        """
        beta_raw = self.raw_reparam.initialize_from_gamma(gamma, floor=floor)
        return beta_raw[1:]  # drop beta_1

    def qp_initialize(self, B_centered: NDArray, y: NDArray, **kwargs) -> NDArray:
        """QP initialization in solver space.

        All q_eff components are exp-mapped, so all are constrained >= 0.
        """
        from superglm.solvers.constrained_qp import solve_constrained_qp

        q_eff = self.q
        X = B_centered

        weights = kwargs.get("weights", None)
        lambda_penalty = kwargs.get("lambda_penalty", 0.01)

        if weights is not None:
            W = np.sqrt(weights)
            X_w = X * W[:, None]
            y_w = y * W
        else:
            X_w = X
            y_w = y

        H = X_w.T @ X_w + lambda_penalty * self.penalty_matrix()
        H += 1e-8 * np.eye(q_eff)
        g = X_w.T @ y_w

        # All q_eff components constrained >= 0
        A = np.eye(q_eff)
        b = np.zeros(q_eff)

        result = solve_constrained_qp(H, g, A, b)
        beta_tilde_eff = result.beta

        # Convert: beta_eff_i = log(beta_tilde_eff_i)
        return np.log(np.maximum(beta_tilde_eff, 1e-8))

    @property
    def direction(self) -> str:
        """Monotone direction."""
        return self.raw_reparam.direction


def build_scop_solver_reparam(q_raw: int, direction: str = "increasing") -> SCOPSolverReparam:
    """Build a solver-space SCOP reparameterization.

    Parameters
    ----------
    q_raw : int
        Number of raw basis functions.
    direction : str
        "increasing" or "decreasing".

    Returns
    -------
    SCOPSolverReparam
        Solver-space wrapper with q = q_raw - 1 effective parameters.
    """
    raw = build_scop_reparam(q_raw, direction=direction)
    return SCOPSolverReparam(q=q_raw - 1, raw_reparam=raw)
