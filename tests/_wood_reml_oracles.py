"""Independent dense REML/LAML oracles from Wood (2011, 2016).

These helpers deliberately use only NumPy/SciPy and distribution likelihoods.
They must not call SuperGLM's REML, PIRLS, determinant, or rank helpers: their
purpose is to give the production implementation an independent reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


class _DistributionWithLikelihood(Protocol):
    def log_likelihood(
        self,
        y: NDArray,
        mu: NDArray,
        weights: NDArray,
        phi: float = 1.0,
    ) -> float: ...


@dataclass(frozen=True)
class DenseWoodState:
    """Dense penalized state in Wood's full coefficient space."""

    beta: NDArray
    intercept: float
    deviance: float
    penalty_quad: float
    slope_xtwx: NDArray
    full_hessian: NDArray
    logdet_s_plus: float
    penalty_nullity: int

    @property
    def penalized_deviance(self) -> float:
        return self.deviance + self.penalty_quad


def _as_symmetric(matrix: NDArray) -> NDArray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def full_logdet(matrix: NDArray) -> float:
    """Return log determinant of a numerically positive-definite matrix."""
    eigvals = np.linalg.eigvalsh(_as_symmetric(matrix))
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    if np.any(eigvals <= 1e-12 * scale):
        raise AssertionError(f"oracle expected full rank, got eigenvalues {eigvals!r}")
    return float(np.sum(np.log(eigvals)))


def logdet_plus_and_nullity(penalty: NDArray) -> tuple[float, int]:
    """Return Wood's generalized log determinant and null-space dimension."""
    eigvals = np.linalg.eigvalsh(_as_symmetric(penalty))
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    positive = eigvals > 1e-12 * scale
    if np.any(eigvals < -1e-12 * scale):
        raise AssertionError(f"oracle penalty must be positive semidefinite: {eigvals!r}")
    return float(np.sum(np.log(eigvals[positive]))), int(np.sum(~positive))


def augmented_penalty(slope_penalty: NDArray) -> NDArray:
    """Embed a slope penalty in the full ``(intercept, slopes)`` space."""
    slope_penalty = np.asarray(slope_penalty, dtype=np.float64)
    result = np.zeros((slope_penalty.shape[0] + 1, slope_penalty.shape[1] + 1))
    result[1:, 1:] = slope_penalty
    return result


def solve_gaussian_state(X: NDArray, y: NDArray, slope_penalty: NDArray) -> DenseWoodState:
    """Solve the Gaussian penalized normal equations in full coefficient space."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack((np.ones(X.shape[0]), X))
    S_aug = augmented_penalty(slope_penalty)
    hessian = A.T @ A + S_aug
    coef = np.linalg.solve(hessian, A.T @ y)
    residual = y - A @ coef
    beta = coef[1:]
    logdet_s_plus, penalty_nullity = logdet_plus_and_nullity(S_aug)
    return DenseWoodState(
        beta=beta,
        intercept=float(coef[0]),
        deviance=float(residual @ residual),
        penalty_quad=float(beta @ slope_penalty @ beta),
        slope_xtwx=X.T @ X,
        full_hessian=hessian,
        logdet_s_plus=logdet_s_plus,
        penalty_nullity=penalty_nullity,
    )


def gaussian_profiled_reml_reduced(state: DenseWoodState, n_obs: int) -> float:
    """Wood (2011), Eq. (4), Gaussian phi profiled out, modulo constants.

    For Gaussian data, ``l_s(phi) = -n log(2 pi phi) / 2``. Profiling
    ``phi`` therefore leaves ``(n-Mp) log(Dp) / 2 + K`` up to terms that
    cannot depend on smoothing parameters. ``Mp`` is the nullity of the
    *full* penalty, including the unpenalized intercept.
    """
    residual_df = n_obs - state.penalty_nullity
    if residual_df <= 0:
        raise AssertionError("oracle requires positive residual degrees of freedom")
    determinant_term = 0.5 * (full_logdet(state.full_hessian) - state.logdet_s_plus)
    return float(0.5 * residual_df * np.log(state.penalized_deviance) + determinant_term)


def solve_poisson_log_state(X: NDArray, y: NDArray, slope_penalty: NDArray) -> DenseWoodState:
    """Solve a canonical Poisson penalized likelihood by dense Newton iteration."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.column_stack((np.ones(X.shape[0]), X))
    S_aug = augmented_penalty(slope_penalty)
    coef = np.zeros(A.shape[1], dtype=np.float64)
    coef[0] = np.log(max(float(np.mean(y)), 1e-4))

    def objective(candidate: NDArray) -> float:
        eta = A @ candidate
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta) + 0.5 * candidate @ S_aug @ candidate)

    for _ in range(100):
        eta = A @ coef
        mu = np.exp(eta)
        gradient = A.T @ (mu - y) + S_aug @ coef
        hessian = A.T @ (mu[:, None] * A) + S_aug
        step = np.linalg.solve(hessian, gradient)
        if float(np.max(np.abs(step))) < 1e-13:
            break
        current = objective(coef)
        alpha = 1.0
        while alpha >= 2.0**-30:
            trial = coef - alpha * step
            if objective(trial) < current:
                coef = trial
                break
            alpha *= 0.5
        else:
            # Near the optimum, the Newton decrement can be smaller than the
            # rounding error in the summed objective even while the score is
            # still large enough to benefit from one final full Newton step.
            # Accept only that roundoff-limited case; the score check below
            # remains the authoritative convergence witness.
            trial = coef - step
            trial_objective = objective(trial)
            objective_scale = max(abs(current), abs(trial_objective), 1.0)
            objective_resolution = 64.0 * np.finfo(np.float64).eps * objective_scale
            newton_decrement = float(gradient @ step)
            if (
                np.isfinite(trial_objective)
                and 0.0 <= newton_decrement <= objective_resolution
                and trial_objective <= current + objective_resolution
            ):
                coef = trial
                continue
            raise AssertionError("dense Poisson oracle line search failed")
    else:
        raise AssertionError("dense Poisson oracle did not converge")

    eta = A @ coef
    mu = np.exp(eta)
    gradient = A.T @ (mu - y) + S_aug @ coef
    if float(np.max(np.abs(gradient))) > 1e-9:
        raise AssertionError(f"dense Poisson oracle score did not converge: {gradient!r}")

    hessian = A.T @ (mu[:, None] * A) + S_aug
    beta = coef[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        y_log_ratio = np.where(y > 0.0, y * np.log(y / mu), 0.0)
    deviance = float(2.0 * np.sum(y_log_ratio - (y - mu)))
    logdet_s_plus, penalty_nullity = logdet_plus_and_nullity(S_aug)
    return DenseWoodState(
        beta=beta,
        intercept=float(coef[0]),
        deviance=deviance,
        penalty_quad=float(beta @ slope_penalty @ beta),
        slope_xtwx=X.T @ (mu[:, None] * X),
        full_hessian=hessian,
        logdet_s_plus=logdet_s_plus,
        penalty_nullity=penalty_nullity,
    )


def poisson_laml_reduced(state: DenseWoodState) -> float:
    """Known-scale canonical Poisson LAML, modulo the saturated constant."""
    determinant_term = 0.5 * (full_logdet(state.full_hessian) - state.logdet_s_plus)
    return float(0.5 * (state.deviance + state.penalty_quad) + determinant_term)


def profile_edf_scale_term(
    distribution: _DistributionWithLikelihood,
    y: NDArray,
    penalized_deviance: float,
    penalty_nullity: int,
) -> tuple[float, float]:
    """Profile Wood's Eq. (4) scale-dependent terms over ``log(phi)``.

    Unlike the Gaussian shortcut, this retains the family-specific saturated
    log likelihood ``l_s(phi)``. The returned criterion includes constants;
    callers should compare differences between smoothing states.
    """
    y = np.asarray(y, dtype=np.float64)
    weights = np.ones_like(y)

    def criterion(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        saturated = distribution.log_likelihood(y, y, weights, phi=phi)
        return float(
            penalized_deviance / (2.0 * phi)
            - saturated
            - 0.5 * penalty_nullity * np.log(2.0 * np.pi * phi)
        )

    result = minimize_scalar(
        criterion,
        bounds=(-12.0, 8.0),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise AssertionError(f"scale-profile oracle failed: {result!r}")
    if min(result.x + 12.0, 8.0 - result.x) < 1e-4:
        raise AssertionError(f"scale-profile optimum hit an oracle bound: {result!r}")
    return float(result.fun), float(np.exp(result.x))
