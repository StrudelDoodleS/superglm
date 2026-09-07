"""Independent Gaussian location-scale laws for distributional fit tests.

This module deliberately depends only on NumPy and test-side containers.  In
particular, it does not import the production Gaussian family, link derivative,
coefficient assembly, inference, rank, or LAML implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

WeightLaw = Literal["prior", "frequency"]

_FLOAT = np.float64
_EPS = np.finfo(_FLOAT).eps
_LOG_2PI = np.log(2.0 * np.pi)


def gamma(operation_count: int) -> float:
    """Return Higham's γ(k) floating-point accumulation factor."""

    if isinstance(operation_count, bool) or not isinstance(operation_count, int):
        raise TypeError("operation_count must be an integer")
    if operation_count < 0 or operation_count * _EPS >= 1.0:
        raise ValueError("operation_count is outside the usable γ(k) range")
    return float(operation_count * _EPS / (1.0 - operation_count * _EPS))


def _vector(values: NDArray, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=_FLOAT)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector")
    return result


def _matrix(values: NDArray, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=_FLOAT)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


@dataclass(frozen=True)
class GaussianRowOracle:
    """Literal row values and derivative channels under one Gaussian law."""

    optimizing_log_likelihood: NDArray[np.float64]
    parameter_independent_carrier: NDArray[np.float64]
    reported_log_likelihood: NDArray[np.float64]
    natural_score: NDArray[np.float64]
    natural_hessian_packed: NDArray[np.float64]
    natural_fisher_packed: NDArray[np.float64]
    link_score: NDArray[np.float64]
    observed_link_curvature_packed: NDArray[np.float64]
    fisher_link_curvature_packed: NDArray[np.float64]


def gaussian_row_oracle(
    y: NDArray,
    mu: NDArray,
    sigma: NDArray,
    weights: NDArray,
    *,
    semantics: WeightLaw,
    scale_floor: float = 0.0,
) -> GaussianRowOracle:
    """Evaluate literal Gaussian prior or exact-replication row mathematics."""

    response = _vector(y, name="y")
    location = _vector(mu, name="mu")
    scale = _vector(sigma, name="sigma")
    mass = _vector(weights, name="weights")
    if (
        location.shape != response.shape
        or scale.shape != response.shape
        or mass.shape != response.shape
    ):
        raise ValueError("Gaussian row inputs must share one shape")
    if np.any(mass <= 0.0) or np.any(scale <= scale_floor):
        raise ValueError("weights and scales must be strictly inside their supports")
    if semantics == "frequency" and np.any(mass != np.floor(mass)):
        raise ValueError("frequency weights must be exact positive integers")
    if semantics not in ("prior", "frequency"):
        raise ValueError("semantics must be 'prior' or 'frequency'")

    residual = response - location
    residual2 = residual * residual
    inverse_sigma = 1.0 / scale
    inverse_sigma2 = inverse_sigma * inverse_sigma
    ordinary = -np.log(scale) - 0.5 * _LOG_2PI - 0.5 * residual2 * inverse_sigma2

    if semantics == "prior":
        optimizing = -np.log(scale) - 0.5 * _LOG_2PI - 0.5 * mass * residual2 * inverse_sigma2
        carrier = 0.5 * np.log(mass)
        score = np.column_stack(
            (
                mass * residual * inverse_sigma2,
                -inverse_sigma + mass * residual2 * inverse_sigma**3,
            )
        )
        hessian = np.column_stack(
            (
                -mass * inverse_sigma2,
                -2.0 * mass * residual * inverse_sigma**3,
                inverse_sigma2 - 3.0 * mass * residual2 * inverse_sigma**4,
            )
        )
        fisher = np.column_stack((mass * inverse_sigma2, np.zeros(len(mass)), 2.0 * inverse_sigma2))
    else:
        optimizing = mass * ordinary
        carrier = np.zeros(len(mass), dtype=_FLOAT)
        score = mass[:, None] * np.column_stack(
            (
                residual * inverse_sigma2,
                -inverse_sigma + residual2 * inverse_sigma**3,
            )
        )
        hessian = mass[:, None] * np.column_stack(
            (
                -inverse_sigma2,
                -2.0 * residual * inverse_sigma**3,
                inverse_sigma2 - 3.0 * residual2 * inverse_sigma**4,
            )
        )
        fisher = mass[:, None] * np.column_stack(
            (inverse_sigma2, np.zeros(len(mass)), 2.0 * inverse_sigma2)
        )

    scale_derivative = scale - float(scale_floor)
    link_score = np.column_stack((score[:, 0], scale_derivative * score[:, 1]))
    observed_link = np.column_stack(
        (
            -hessian[:, 0],
            -scale_derivative * hessian[:, 1],
            -(scale_derivative**2 * hessian[:, 2] + scale_derivative * score[:, 1]),
        )
    )
    fisher_link = np.column_stack(
        (
            fisher[:, 0],
            scale_derivative * fisher[:, 1],
            scale_derivative**2 * fisher[:, 2],
        )
    )
    return GaussianRowOracle(
        optimizing_log_likelihood=optimizing,
        parameter_independent_carrier=carrier,
        reported_log_likelihood=optimizing + carrier,
        natural_score=score,
        natural_hessian_packed=hessian,
        natural_fisher_packed=fisher,
        link_score=link_score,
        observed_link_curvature_packed=observed_link,
        fisher_link_curvature_packed=fisher_link,
    )


@dataclass(frozen=True)
class SpectralGap:
    """A full-rank decision separated from the eigensolver resolution bar."""

    smallest_eigenvalue: float
    largest_eigenvalue: float
    resolution_bar: float
    condition: float


@dataclass(frozen=True)
class GaussianCoefficientOracle:
    """Literal likelihood, score, curvature, and inference at one coefficient vector."""

    semantics: WeightLaw
    response: NDArray[np.float64]
    weights: NDArray[np.float64]
    location_design: NDArray[np.float64]
    scale_design: NDArray[np.float64]
    location_offset: NDArray[np.float64]
    scale_offset: NDArray[np.float64]
    scale_floor: float
    coefficients: NDArray[np.float64]
    eta: NDArray[np.float64]
    theta: NDArray[np.float64]
    rows: GaussianRowOracle
    penalty: NDArray[np.float64]
    penalty_value: float
    score_data: NDArray[np.float64]
    score_penalized: NDArray[np.float64]
    data_curvature: NDArray[np.float64]
    penalized_curvature: NDArray[np.float64]
    optimizing_log_likelihood: float
    parameter_independent_carrier: float
    reported_log_likelihood: float
    penalized_optimizing_log_likelihood: float
    penalized_reported_log_likelihood: float
    objective: float
    covariance: NDArray[np.float64]
    influence: NDArray[np.float64]
    effective_degrees_freedom: float
    spectral_gap: SpectralGap


def _coefficient_curvature(
    location_design: NDArray[np.float64],
    scale_design: NDArray[np.float64],
    packed: NDArray[np.float64],
) -> NDArray[np.float64]:
    p_location = location_design.shape[1]
    p_scale = scale_design.shape[1]
    result = np.zeros((p_location + p_scale, p_location + p_scale), dtype=_FLOAT)
    location_slice = slice(0, p_location)
    scale_slice = slice(p_location, p_location + p_scale)
    result[location_slice, location_slice] = location_design.T @ (
        packed[:, 0, None] * location_design
    )
    cross = location_design.T @ (packed[:, 1, None] * scale_design)
    result[location_slice, scale_slice] = cross
    result[scale_slice, location_slice] = cross.T
    result[scale_slice, scale_slice] = scale_design.T @ (packed[:, 2, None] * scale_design)
    return 0.5 * (result + result.T)


def coefficient_oracle(
    y: NDArray,
    weights: NDArray,
    *,
    semantics: WeightLaw,
    location_design: NDArray,
    scale_design: NDArray,
    coefficients: NDArray,
    penalty: NDArray,
    location_offset: NDArray | None = None,
    scale_offset: NDArray | None = None,
    scale_floor: float = 0.0,
) -> GaussianCoefficientOracle:
    """Reconstruct the complete fixed-λ coefficient state from literal matrices."""

    response = _vector(y, name="y")
    mass = _vector(weights, name="weights")
    x_location = _matrix(location_design, name="location_design")
    x_scale = _matrix(scale_design, name="scale_design")
    beta = _vector(coefficients, name="coefficients")
    penalty_matrix = _matrix(penalty, name="penalty")
    n_rows = len(response)
    if mass.shape != response.shape or x_location.shape[0] != n_rows or x_scale.shape[0] != n_rows:
        raise ValueError("response, weights, and designs must share a row count")
    p_location = x_location.shape[1]
    p_scale = x_scale.shape[1]
    if beta.shape != (p_location + p_scale,):
        raise ValueError("coefficient vector does not match the two design blocks")
    if penalty_matrix.shape != (len(beta), len(beta)):
        raise ValueError("penalty matrix does not match the coefficient vector")
    if not np.array_equal(penalty_matrix, penalty_matrix.T):
        raise ValueError("penalty matrix must be exactly symmetric")

    location_shift = (
        np.zeros(n_rows, dtype=_FLOAT)
        if location_offset is None
        else _vector(location_offset, name="location_offset")
    )
    scale_shift = (
        np.zeros(n_rows, dtype=_FLOAT)
        if scale_offset is None
        else _vector(scale_offset, name="scale_offset")
    )
    if location_shift.shape != response.shape or scale_shift.shape != response.shape:
        raise ValueError("offsets must match the response rows")

    eta_location = x_location @ beta[:p_location] + location_shift
    eta_scale = x_scale @ beta[p_location:] + scale_shift
    sigma = float(scale_floor) + np.exp(eta_scale)
    theta = np.column_stack((eta_location, sigma))
    rows = gaussian_row_oracle(
        response,
        eta_location,
        sigma,
        mass,
        semantics=semantics,
        scale_floor=scale_floor,
    )
    score_data = np.concatenate(
        (
            x_location.T @ rows.link_score[:, 0],
            x_scale.T @ rows.link_score[:, 1],
        )
    )
    score_penalized = score_data - penalty_matrix @ beta
    data_curvature = _coefficient_curvature(
        x_location,
        x_scale,
        rows.observed_link_curvature_packed,
    )
    penalized_curvature = data_curvature + penalty_matrix
    eigenvalues = np.linalg.eigvalsh(penalized_curvature)
    largest = float(np.max(np.abs(eigenvalues), initial=0.0))
    smallest = float(np.min(eigenvalues, initial=np.inf))
    resolution_bar = gamma(max(128 * len(beta), 1)) * max(1.0, largest)
    if not smallest > resolution_bar:
        raise AssertionError(
            "oracle fixture lacks a certified positive full-rank gap: "
            f"smallest={smallest}, bar={resolution_bar}"
        )
    covariance = np.linalg.solve(penalized_curvature, np.eye(len(beta), dtype=_FLOAT))
    influence = covariance @ data_curvature
    optimizing = float(np.sum(rows.optimizing_log_likelihood, dtype=_FLOAT))
    carrier = float(np.sum(rows.parameter_independent_carrier, dtype=_FLOAT))
    reported = float(optimizing + carrier)
    penalty_value = 0.5 * float(beta @ penalty_matrix @ beta)
    penalized_optimizing = optimizing - penalty_value
    penalized_reported = reported - penalty_value
    return GaussianCoefficientOracle(
        semantics=semantics,
        response=response,
        weights=mass,
        location_design=x_location,
        scale_design=x_scale,
        location_offset=location_shift,
        scale_offset=scale_shift,
        scale_floor=float(scale_floor),
        coefficients=beta,
        eta=np.column_stack((eta_location, eta_scale)),
        theta=theta,
        rows=rows,
        penalty=penalty_matrix,
        penalty_value=penalty_value,
        score_data=score_data,
        score_penalized=score_penalized,
        data_curvature=data_curvature,
        penalized_curvature=penalized_curvature,
        optimizing_log_likelihood=optimizing,
        parameter_independent_carrier=carrier,
        reported_log_likelihood=reported,
        penalized_optimizing_log_likelihood=penalized_optimizing,
        penalized_reported_log_likelihood=penalized_reported,
        objective=-penalized_optimizing,
        covariance=covariance,
        influence=influence,
        effective_degrees_freedom=float(np.trace(influence)),
        spectral_gap=SpectralGap(
            smallest_eigenvalue=smallest,
            largest_eigenvalue=largest,
            resolution_bar=resolution_bar,
            condition=largest / smallest,
        ),
    )


@dataclass(frozen=True)
class OracleBounds:
    """Dimension-, norm-, and conditioning-derived arithmetic bounds."""

    row_component: float
    likelihood_sum: float
    score_roundoff: float
    curvature: float
    covariance_backward: float
    covariance_forward: float
    edf: float
    eta_evaluation: NDArray[np.float64]
    theta_evaluation: NDArray[np.float64]


def _maximum_absolute(values: NDArray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _assert_componentwise_close(
    actual: NDArray,
    expected: NDArray,
    allowance: NDArray,
    *,
    label: str,
) -> None:
    """Assert an array-valued absolute budget without scalarizing it."""

    observed = np.asarray(actual, dtype=_FLOAT)
    target = np.asarray(expected, dtype=_FLOAT)
    if observed.shape != target.shape:
        raise AssertionError(
            f"{label} shapes differ: actual={observed.shape}, expected={target.shape}"
        )
    budget = np.broadcast_to(np.asarray(allowance, dtype=_FLOAT), observed.shape)
    difference = np.abs(observed - target)
    failing = difference > budget
    if np.any(failing):
        excess = np.where(failing, difference - budget, -np.inf)
        index = np.unravel_index(int(np.argmax(excess)), observed.shape)
        raise AssertionError(
            f"{label} exceeds its componentwise arithmetic budget at {index}: "
            f"difference={difference[index]}, allowance={budget[index]}, "
            f"actual={observed[index]}, expected={target[index]}"
        )


def oracle_bounds(
    oracle: GaussianCoefficientOracle,
) -> OracleBounds:
    """Construct same-point arithmetic budgets without solver-exit implications."""

    n_rows = oracle.eta.shape[0]
    width = len(oracle.coefficients)
    row_scale = max(
        1.0,
        _maximum_absolute(oracle.rows.optimizing_log_likelihood),
        _maximum_absolute(oracle.rows.parameter_independent_carrier),
        _maximum_absolute(oracle.rows.link_score),
        _maximum_absolute(oracle.rows.observed_link_curvature_packed),
    )
    row_component = gamma(32) * row_scale
    likelihood_scale = max(
        1.0,
        float(np.sum(np.abs(oracle.rows.optimizing_log_likelihood), dtype=_FLOAT)),
        float(np.sum(np.abs(oracle.rows.parameter_independent_carrier), dtype=_FLOAT)),
        abs(oracle.reported_log_likelihood),
    )
    likelihood_sum = gamma(max(64 * n_rows, 1)) * likelihood_scale

    x_location = np.abs(oracle.location_design)
    x_scale = np.abs(oracle.scale_design)
    absolute_score = np.concatenate(
        (
            x_location.T @ np.abs(oracle.rows.link_score[:, 0]),
            x_scale.T @ np.abs(oracle.rows.link_score[:, 1]),
        )
    ) + np.abs(oracle.penalty) @ np.abs(oracle.coefficients)
    score_scale = max(1.0, _maximum_absolute(absolute_score))
    score_roundoff = gamma(max(64 * n_rows + 32 * width, 1)) * score_scale

    packed = np.abs(oracle.rows.observed_link_curvature_packed)
    absolute_curvature = _coefficient_curvature(x_location, x_scale, packed)
    curvature_scale = max(
        1.0,
        float(np.linalg.norm(absolute_curvature + np.abs(oracle.penalty), ord=np.inf)),
    )
    curvature = gamma(max(96 * n_rows + 32 * width, 1)) * curvature_scale
    product_scale = max(
        1.0,
        float(
            np.linalg.norm(
                np.abs(oracle.penalized_curvature) @ np.abs(oracle.covariance),
                ord=np.inf,
            )
        ),
    )
    covariance_backward = gamma(max(64 * width, 1)) * product_scale + curvature * float(
        np.linalg.norm(oracle.covariance, ord=np.inf)
    )
    covariance_forward = (
        2.0 * float(np.linalg.norm(oracle.covariance, ord=np.inf)) * covariance_backward
    )
    edf_scale = max(
        1.0,
        float(np.sum(np.abs(np.diag(oracle.influence)), dtype=_FLOAT)),
        abs(oracle.effective_degrees_freedom),
    )
    edf = gamma(max(64 * width * width, 1)) * edf_scale + width * (
        covariance_forward * float(np.linalg.norm(oracle.data_curvature, ord=np.inf))
        + float(np.linalg.norm(oracle.covariance, ord=np.inf)) * curvature
    )
    location_eta_evaluation = gamma(
        max(8 * oracle.location_design.shape[1] + 4, 1)
    ) * np.maximum.reduce(
        (
            np.ones(n_rows, dtype=_FLOAT),
            np.abs(oracle.location_design)
            @ np.abs(oracle.coefficients[: oracle.location_design.shape[1]]),
            np.abs(oracle.location_offset),
        )
    )
    scale_eta_evaluation = gamma(max(8 * oracle.scale_design.shape[1] + 4, 1)) * np.maximum.reduce(
        (
            np.ones(n_rows, dtype=_FLOAT),
            np.abs(oracle.scale_design)
            @ np.abs(oracle.coefficients[oracle.location_design.shape[1] :]),
            np.abs(oracle.scale_offset),
        )
    )
    eta_evaluation = np.column_stack((location_eta_evaluation, scale_eta_evaluation))
    theta_scale_evaluation = oracle.theta[:, 1] * (np.expm1(scale_eta_evaluation) + gamma(8))
    theta_evaluation = np.column_stack((location_eta_evaluation, theta_scale_evaluation))
    return OracleBounds(
        row_component=row_component,
        likelihood_sum=likelihood_sum,
        score_roundoff=score_roundoff,
        curvature=curvature,
        covariance_backward=covariance_backward,
        covariance_forward=covariance_forward,
        edf=edf,
        eta_evaluation=eta_evaluation,
        theta_evaluation=theta_evaluation,
    )


def covariance_backward_error(
    curvature: NDArray,
    covariance: NDArray,
) -> float:
    """Return ||H V - I||∞ for a proposed covariance action."""

    hessian = _matrix(curvature, name="curvature")
    candidate = _matrix(covariance, name="covariance")
    if hessian.shape != candidate.shape or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("curvature and covariance must be matching square matrices")
    identity = np.eye(hessian.shape[0], dtype=_FLOAT)
    return float(np.linalg.norm(hessian @ candidate - identity, ord=np.inf))


@dataclass(frozen=True)
class LocalRootCertificate:
    """A posteriori certificate for one strict stationary mode in a local ball."""

    center: NDArray[np.float64]
    radius: float
    lambda_lower_at_center: float
    hessian_drift: float
    strong_convexity: float
    center_gradient_upper: float
    center_error: float
    candidate_errors: NDArray[np.float64]
    candidate_kappa: NDArray[np.float64]
    direct_kkt_limit: float


@dataclass(frozen=True)
class GaussianFitCertificate:
    """Independent same-point and local-root evidence for one fitted result."""

    oracle: GaussianCoefficientOracle
    bounds: OracleBounds
    local_root: LocalRootCertificate
    covariance: NDArray[np.float64]
    total_edf: float
    rank: int


def _oracle_at(
    reference: GaussianCoefficientOracle,
    coefficients: NDArray,
) -> GaussianCoefficientOracle:
    return coefficient_oracle(
        reference.response,
        reference.weights,
        semantics=reference.semantics,
        location_design=reference.location_design,
        scale_design=reference.scale_design,
        coefficients=coefficients,
        penalty=reference.penalty,
        location_offset=reference.location_offset,
        scale_offset=reference.scale_offset,
        scale_floor=reference.scale_floor,
    )


def _gradient_roundoff(
    oracle: GaussianCoefficientOracle,
) -> tuple[float, float, float]:
    n_rows = len(oracle.response)
    width = len(oracle.coefficients)
    absolute_score = np.concatenate(
        (
            np.abs(oracle.location_design).T @ np.abs(oracle.rows.link_score[:, 0]),
            np.abs(oracle.scale_design).T @ np.abs(oracle.rows.link_score[:, 1]),
        )
    ) + np.abs(oracle.penalty) @ np.abs(oracle.coefficients)
    scale = max(1.0, _maximum_absolute(absolute_score))
    operation_count = max(64 * n_rows + 32 * width, 1)
    roundoff = gamma(operation_count) * scale
    kappa = (float(np.linalg.norm(oracle.score_penalized, ord=np.inf)) + roundoff) / scale
    return scale, roundoff, kappa


def _polished_center(reference: GaussianCoefficientOracle) -> GaussianCoefficientOracle:
    center = np.array(reference.coefficients, dtype=_FLOAT, copy=True)
    operation_count = max(64 * len(reference.response) + 32 * len(center), 1)
    polish_limit = gamma(operation_count)
    direct_limit = float(np.sqrt(polish_limit))
    for _ in range(12):
        current = _oracle_at(reference, center)
        _, _, kappa = _gradient_roundoff(current)
        if kappa <= polish_limit:
            return current
        direction = np.linalg.solve(
            current.penalized_curvature,
            current.score_penalized,
        )
        directional_derivative = float(current.score_penalized @ direction)
        if not np.isfinite(directional_derivative) or directional_derivative <= 0.0:
            raise AssertionError("independent Newton polish lacks an ascent direction")
        accepted = None
        alpha = 1.0
        for _ in range(32):
            proposal = _oracle_at(reference, center + alpha * direction)
            required = current.penalized_optimizing_log_likelihood + (
                1.0e-4 * alpha * directional_derivative
            )
            if proposal.penalized_optimizing_log_likelihood >= required:
                accepted = proposal
                break
            alpha *= 0.5
        if accepted is None:
            if kappa <= direct_limit:
                return current
            raise AssertionError("independent Newton polish line search failed")
        center = np.array(accepted.coefficients, copy=True)
    final = _oracle_at(reference, center)
    if _gradient_roundoff(final)[2] > direct_limit:
        raise AssertionError("independent Newton polish missed its direct KKT target")
    return final


def _hessian_drift_bound(
    center: GaussianCoefficientOracle,
    radius: float,
) -> float:
    if center.scale_floor != 0.0:
        raise AssertionError("local Gaussian root certification requires zero scale floor")
    if not np.isfinite(radius) or radius <= 0.0:
        raise AssertionError("local Gaussian root radius must be finite and positive")
    u_norm = np.linalg.norm(center.location_design, axis=1)
    v_norm = np.linalg.norm(center.scale_design, axis=1)
    residual = center.response - center.theta[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        t0 = center.weights * np.exp(-2.0 * center.eta[:, 1])
        response_drift = u_norm * radius
        maximum_t = t0 * np.exp(2.0 * v_norm * radius)
        delta_t = t0 * np.expm1(2.0 * v_norm * radius)
        delta_tr = delta_t * np.abs(residual) + maximum_t * response_drift
        delta_tr2 = delta_t * residual**2 + maximum_t * (
            2.0 * np.abs(residual) * response_drift + response_drift**2
        )
        summands = (
            delta_t * u_norm**2 + 4.0 * delta_tr * u_norm * v_norm + 2.0 * delta_tr2 * v_norm**2
        )
    if not np.all(np.isfinite(summands)):
        raise AssertionError("local Gaussian Hessian drift overflowed")
    raw = float(np.sum(summands, dtype=_FLOAT))
    operation_count = max(256 * len(summands) + 64 * len(center.coefficients), 1)
    upward = raw + gamma(operation_count) * max(
        1.0,
        float(np.sum(np.abs(summands), dtype=_FLOAT)),
    )
    return float(np.nextafter(upward, np.inf))


def local_root_certificate(
    reference: GaussianCoefficientOracle,
    candidate_coefficients: NDArray,
) -> LocalRootCertificate:
    """Certify direct KKT residuals and one unique strict local Gaussian mode."""

    if reference.scale_floor != 0.0:
        raise AssertionError("local Gaussian root certification requires zero scale floor")
    candidates = np.asarray(candidate_coefficients, dtype=_FLOAT)
    width = len(reference.coefficients)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != width
        or len(candidates) == 0
        or not np.all(np.isfinite(candidates))
    ):
        raise ValueError("candidate coefficients must be a finite nonempty matrix")
    operation_count = max(64 * len(reference.response) + 32 * width, 1)
    direct_limit = float(np.sqrt(gamma(operation_count)))
    candidate_oracles = tuple(_oracle_at(reference, value) for value in candidates)
    candidate_kappa = np.array(
        [_gradient_roundoff(oracle)[2] for oracle in candidate_oracles],
        dtype=_FLOAT,
    )
    if np.any(candidate_kappa > direct_limit):
        raise AssertionError(
            "direct dimensionless KKT residual exceeds sqrt(gamma(k)): "
            f"maximum={float(np.max(candidate_kappa))}, limit={direct_limit}"
        )

    center = _polished_center(reference)
    center_bounds = oracle_bounds(center)
    lambda_lower = (
        center.spectral_gap.smallest_eigenvalue
        - center.spectral_gap.resolution_bar
        - center_bounds.curvature
    )
    if not lambda_lower > 0.0:
        raise AssertionError("center curvature has no certified positive lower bound")
    _, center_roundoff, _ = _gradient_roundoff(center)
    center_gradient_upper = (
        float(np.linalg.norm(center.score_penalized, ord=2))
        + float(np.sqrt(width)) * center_roundoff
    )
    candidate_distances = np.linalg.norm(candidates - center.coefficients, axis=1)
    radius_seed = max(
        np.sqrt(_EPS) * (1.0 + float(np.linalg.norm(center.coefficients, ord=2))),
        center_gradient_upper / lambda_lower,
        float(np.max(candidate_distances, initial=0.0)),
    )
    radius = float(np.nextafter(2.0 * radius_seed, np.inf))
    hessian_drift = _hessian_drift_bound(center, radius)
    strong_convexity = lambda_lower - hessian_drift
    if not strong_convexity > 0.0:
        raise AssertionError("local Gaussian ball lacks certified strong convexity")
    if not center_gradient_upper < strong_convexity * radius:
        raise AssertionError("local Gaussian ball does not contain a certified stationary mode")
    if np.any(candidate_distances >= radius):
        raise AssertionError("candidate lies outside the certified local Gaussian ball")

    candidate_errors = np.empty(len(candidate_oracles), dtype=_FLOAT)
    for index, oracle in enumerate(candidate_oracles):
        _, roundoff, _ = _gradient_roundoff(oracle)
        q2 = float(np.linalg.norm(oracle.score_penalized, ord=2)) + float(np.sqrt(width)) * roundoff
        candidate_errors[index] = q2 / strong_convexity
    center_error = center_gradient_upper / strong_convexity
    return LocalRootCertificate(
        center=np.array(center.coefficients, copy=True),
        radius=radius,
        lambda_lower_at_center=lambda_lower,
        hessian_drift=hessian_drift,
        strong_convexity=strong_convexity,
        center_gradient_upper=center_gradient_upper,
        center_error=center_error,
        candidate_errors=candidate_errors,
        candidate_kappa=candidate_kappa,
        direct_kkt_limit=direct_limit,
    )


def explicit_design_blocks(layout) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract two explicit predictor matrices without production math helpers."""

    blocks: list[NDArray[np.float64]] = []
    for state in layout.predictors:
        slopes = (
            np.zeros((state.design.n, 0), dtype=_FLOAT)
            if state.design.p == 0
            else np.asarray(state.design.toarray(), dtype=_FLOAT)
        )
        block = (
            np.column_stack((np.ones(state.design.n, dtype=_FLOAT), slopes))
            if state.intercept_index is not None
            else slopes
        )
        if block.shape[1] != state.coefficient_slice.stop - state.coefficient_slice.start:
            raise AssertionError("explicit design block disagrees with coefficient slice")
        blocks.append(block)
    if len(blocks) != 2:
        raise AssertionError("Gaussian location-scale certification requires two predictors")
    return blocks[0], blocks[1]


def certify_gaussian_result(
    layout,
    result,
    response: NDArray,
    weights: NDArray,
    *,
    semantics: WeightLaw,
    covariance: NDArray,
    total_edf: float,
    inference_rank: int,
    scale_floor: float = 0.0,
    prediction_parameters: NDArray | None = None,
    default_prediction: NDArray | None = None,
) -> GaussianFitCertificate:
    """Certify one production result at its coefficients and in a local root ball."""

    location_design, scale_design = explicit_design_blocks(layout)
    oracle = coefficient_oracle(
        response,
        weights,
        semantics=semantics,
        location_design=location_design,
        scale_design=scale_design,
        coefficients=result.coefficients,
        penalty=result.penalty,
        location_offset=layout.predictors[0].offset,
        scale_offset=layout.predictors[1].offset,
        scale_floor=scale_floor,
    )
    bounds = oracle_bounds(oracle)
    candidate_covariance = _matrix(covariance, name="covariance")
    if not result.converged:
        raise AssertionError("Gaussian fit did not converge")
    if result.terminal_curvature.actual_source != "observed":
        raise AssertionError("Gaussian semantic certification requires observed curvature")
    if not oracle.spectral_gap.smallest_eigenvalue > oracle.spectral_gap.resolution_bar:
        raise AssertionError("Gaussian fit lacks a certified spectral gap")
    _assert_componentwise_close(
        result.eta,
        oracle.eta,
        bounds.eta_evaluation,
        label="production eta",
    )
    _assert_componentwise_close(
        result.theta,
        oracle.theta,
        bounds.theta_evaluation,
        label="production theta",
    )
    for actual, expected in (
        (result.optimizing_log_likelihood, oracle.optimizing_log_likelihood),
        (result.parameter_independent_carrier, oracle.parameter_independent_carrier),
        (result.log_likelihood, oracle.reported_log_likelihood),
        (result.penalty_value, oracle.penalty_value),
        (
            result.penalized_optimizing_log_likelihood,
            oracle.penalized_optimizing_log_likelihood,
        ),
        (result.penalized_log_likelihood, oracle.penalized_reported_log_likelihood),
        (result.objective, oracle.objective),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bounds.likelihood_sum)
    np.testing.assert_allclose(
        result.terminal_score,
        oracle.score_penalized,
        rtol=0.0,
        atol=bounds.score_roundoff,
    )
    np.testing.assert_allclose(
        result.terminal_data_curvature,
        oracle.data_curvature,
        rtol=0.0,
        atol=bounds.curvature,
    )
    np.testing.assert_allclose(
        result.terminal_penalized_curvature,
        oracle.penalized_curvature,
        rtol=0.0,
        atol=bounds.curvature,
    )
    backward = covariance_backward_error(oracle.penalized_curvature, candidate_covariance)
    if backward > bounds.covariance_backward:
        raise AssertionError("production covariance exceeds its backward-error budget")
    np.testing.assert_allclose(
        candidate_covariance,
        oracle.covariance,
        rtol=0.0,
        atol=bounds.covariance_forward,
    )
    np.testing.assert_allclose(
        total_edf,
        oracle.effective_degrees_freedom,
        rtol=0.0,
        atol=bounds.edf,
    )
    if result.terminal_rank.rank != inference_rank or inference_rank != len(result.coefficients):
        raise AssertionError("production rank disagrees with the certified full-rank oracle")
    if prediction_parameters is not None:
        _assert_componentwise_close(
            prediction_parameters,
            oracle.theta,
            bounds.theta_evaluation,
            label="production parameter prediction",
        )
    if default_prediction is not None:
        _assert_componentwise_close(
            default_prediction,
            oracle.theta[:, 0],
            bounds.theta_evaluation[:, 0],
            label="production default prediction",
        )
    local = local_root_certificate(oracle, result.coefficients[None, :])
    return GaussianFitCertificate(
        oracle=oracle,
        bounds=bounds,
        local_root=local,
        covariance=np.array(candidate_covariance, copy=True),
        total_edf=float(total_edf),
        rank=int(inference_rank),
    )


def _prediction_theta(
    reference: GaussianCoefficientOracle,
    coefficients: NDArray,
) -> NDArray[np.float64]:
    p_location = reference.location_design.shape[1]
    eta_location = reference.location_design @ coefficients[:p_location] + reference.location_offset
    eta_scale = reference.scale_design @ coefficients[p_location:] + reference.scale_offset
    return np.column_stack((eta_location, reference.scale_floor + np.exp(eta_scale)))


def _root_prediction_bound(
    reference: GaussianCoefficientOracle,
    common: LocalRootCertificate,
    left_coefficients: NDArray,
    right_coefficients: NDArray,
) -> NDArray[np.float64]:
    combined_error = float(np.sum(common.candidate_errors, dtype=_FLOAT))
    location_eta = np.linalg.norm(reference.location_design, axis=1) * combined_error
    scale_eta = np.linalg.norm(reference.scale_design, axis=1) * combined_error
    candidate_scales = np.vstack(
        (
            _prediction_theta(reference, left_coefficients)[:, 1],
            _prediction_theta(reference, right_coefficients)[:, 1],
        )
    )
    scale_theta = np.max(candidate_scales, axis=0) * np.expm1(scale_eta)
    left_evaluation = oracle_bounds(_oracle_at(reference, left_coefficients)).theta_evaluation
    right_evaluation = oracle_bounds(_oracle_at(reference, right_coefficients)).theta_evaluation
    return np.column_stack((location_eta, scale_theta)) + left_evaluation + right_evaluation


def _root_eta_bound(
    reference: GaussianCoefficientOracle,
    common: LocalRootCertificate,
    left_coefficients: NDArray,
    right_coefficients: NDArray,
) -> NDArray[np.float64]:
    combined_error = float(np.sum(common.candidate_errors, dtype=_FLOAT))
    movement = np.column_stack(
        (
            np.linalg.norm(reference.location_design, axis=1) * combined_error,
            np.linalg.norm(reference.scale_design, axis=1) * combined_error,
        )
    )
    left_evaluation = oracle_bounds(_oracle_at(reference, left_coefficients)).eta_evaluation
    right_evaluation = oracle_bounds(_oracle_at(reference, right_coefficients)).eta_evaluation
    return movement + left_evaluation + right_evaluation


def assert_gaussian_fit_parity(
    left_result,
    right_result,
    left: GaussianFitCertificate,
    right: GaussianFitCertificate,
    *,
    expected_optimizing_difference: float = 0.0,
    expected_carrier_difference: float = 0.0,
    expected_reported_difference: float = 0.0,
    probe: NDArray | None = None,
    left_eta: NDArray | None = None,
    right_eta: NDArray | None = None,
    left_prediction: NDArray | None = None,
    right_prediction: NDArray | None = None,
) -> LocalRootCertificate:
    """Compare two certified routes through one common local-root certificate."""

    candidates = np.vstack((left_result.coefficients, right_result.coefficients))
    common = local_root_certificate(left.oracle, candidates)
    mode_bound = float(np.sum(common.candidate_errors, dtype=_FLOAT))
    np.testing.assert_allclose(
        left_result.coefficients,
        right_result.coefficients,
        rtol=0.0,
        atol=mode_bound,
    )

    left_center = _oracle_at(left.oracle, common.center)
    right_center = _oracle_at(right.oracle, common.center)
    left_center_bounds = oracle_bounds(left_center)
    right_center_bounds = oracle_bounds(right_center)
    center_likelihood_bound = left_center_bounds.likelihood_sum + right_center_bounds.likelihood_sum
    likelihood_channels = (
        (
            "optimizing_log_likelihood",
            "optimizing_log_likelihood",
            expected_optimizing_difference,
        ),
        (
            "parameter_independent_carrier",
            "parameter_independent_carrier",
            expected_carrier_difference,
        ),
        ("reported_log_likelihood", "log_likelihood", expected_reported_difference),
        ("penalty_value", "penalty_value", 0.0),
        (
            "penalized_optimizing_log_likelihood",
            "penalized_optimizing_log_likelihood",
            expected_optimizing_difference,
        ),
        (
            "penalized_reported_log_likelihood",
            "penalized_log_likelihood",
            expected_reported_difference,
        ),
        ("objective", "objective", -expected_optimizing_difference),
    )
    for oracle_name, result_name, expected_difference in likelihood_channels:
        left_center_value = float(getattr(left_center, oracle_name))
        right_center_value = float(getattr(right_center, oracle_name))
        np.testing.assert_allclose(
            left_center_value - right_center_value,
            expected_difference,
            rtol=0.0,
            atol=center_likelihood_bound,
        )
        movement_bound = abs(float(getattr(left.oracle, oracle_name)) - left_center_value) + abs(
            float(getattr(right.oracle, oracle_name)) - right_center_value
        )
        likelihood_bound = (
            left.bounds.likelihood_sum
            + right.bounds.likelihood_sum
            + center_likelihood_bound
            + movement_bound
        )
        np.testing.assert_allclose(
            float(getattr(left_result, result_name)) - float(getattr(right_result, result_name)),
            expected_difference,
            rtol=0.0,
            atol=likelihood_bound,
        )

    center_curvature_bound = left_center_bounds.curvature + right_center_bounds.curvature
    np.testing.assert_allclose(
        left_center.penalized_curvature,
        right_center.penalized_curvature,
        rtol=0.0,
        atol=center_curvature_bound,
    )
    curvature_bound = (
        left.bounds.curvature
        + right.bounds.curvature
        + center_curvature_bound
        + _maximum_absolute(left.oracle.penalized_curvature - left_center.penalized_curvature)
        + _maximum_absolute(right.oracle.penalized_curvature - right_center.penalized_curvature)
    )
    np.testing.assert_allclose(
        left_result.terminal_penalized_curvature,
        right_result.terminal_penalized_curvature,
        rtol=0.0,
        atol=curvature_bound,
    )

    action = (
        np.linspace(0.5, 1.5, len(left_result.coefficients))
        if probe is None
        else _vector(probe, name="probe")
    )
    center_action_bound = (
        left_center_bounds.covariance_forward + right_center_bounds.covariance_forward
    ) * float(np.linalg.norm(action, ord=1))
    np.testing.assert_allclose(
        left_center.covariance @ action,
        right_center.covariance @ action,
        rtol=0.0,
        atol=center_action_bound,
    )
    covariance_action_bound = (
        left.bounds.covariance_forward + right.bounds.covariance_forward
    ) * float(np.linalg.norm(action, ord=1))
    covariance_action_bound += center_action_bound
    covariance_action_bound += float(
        np.linalg.norm((left.oracle.covariance - left_center.covariance) @ action, ord=np.inf)
        + np.linalg.norm(
            (right.oracle.covariance - right_center.covariance) @ action,
            ord=np.inf,
        )
    )
    np.testing.assert_allclose(
        left.covariance @ action,
        right.covariance @ action,
        rtol=0.0,
        atol=covariance_action_bound,
    )

    center_edf_bound = left_center_bounds.edf + right_center_bounds.edf
    np.testing.assert_allclose(
        left_center.effective_degrees_freedom,
        right_center.effective_degrees_freedom,
        rtol=0.0,
        atol=center_edf_bound,
    )
    edf_bound = (
        left.bounds.edf
        + right.bounds.edf
        + center_edf_bound
        + abs(left.oracle.effective_degrees_freedom - left_center.effective_degrees_freedom)
        + abs(right.oracle.effective_degrees_freedom - right_center.effective_degrees_freedom)
    )
    np.testing.assert_allclose(left.total_edf, right.total_edf, rtol=0.0, atol=edf_bound)
    if left.rank != right.rank:
        raise AssertionError("certified route ranks differ")

    if (left_eta is None) != (right_eta is None):
        raise ValueError("both route link predictions must be supplied together")
    if left_eta is not None and right_eta is not None:
        _assert_componentwise_close(
            left_eta,
            right_eta,
            _root_eta_bound(
                left.oracle,
                common,
                left_result.coefficients,
                right_result.coefficients,
            ),
            label="route link predictions",
        )
    if (left_prediction is None) != (right_prediction is None):
        raise ValueError("both route predictions must be supplied together")
    if left_prediction is not None and right_prediction is not None:
        prediction_bound = _root_prediction_bound(
            left.oracle,
            common,
            left_result.coefficients,
            right_result.coefficients,
        )
        _assert_componentwise_close(
            left_prediction,
            right_prediction,
            prediction_bound,
            label="route parameter predictions",
        )
    return common


@dataclass(frozen=True)
class FixedRouteFixture:
    """Small finite-support fixture shared by fixed-fit and EFS certification."""

    x: NDArray[np.float64]
    z: NDArray[np.float64]
    y: NDArray[np.float64]
    prior_weights: NDArray[np.float64]
    frequency_counts: NDArray[np.float64]


def fixed_route_fixture() -> FixedRouteFixture:
    """Return deterministic, well-conditioned rows with five predictor supports."""

    x = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 4)
    z = np.tile(np.array([-1.0, 0.0, 1.0, 0.5, -0.5]), 4)
    error = np.array(
        [
            -0.7,
            0.2,
            1.1,
            -0.4,
            0.6,
            0.5,
            -1.0,
            0.3,
            0.8,
            -0.2,
            -0.3,
            0.9,
            -0.6,
            0.1,
            1.2,
            0.4,
            -0.8,
            0.7,
            -0.1,
            -0.5,
        ]
    )
    sigma = np.exp(-0.45 + 0.18 * z)
    y = 0.4 + 0.65 * x - 0.1 * x**2 + sigma * error
    prior_weights = np.array(
        [
            0.35,
            1.7,
            0.8,
            2.2,
            1.1,
            0.6,
            1.4,
            0.45,
            2.6,
            0.95,
            1.8,
            0.55,
            1.25,
            0.75,
            2.0,
            0.4,
            1.6,
            0.7,
            2.35,
            1.05,
        ]
    )
    frequency_counts = np.array(
        [1, 3, 2, 1, 4, 2, 1, 3, 1, 2, 4, 1, 2, 3, 1, 2, 1, 4, 2, 1],
        dtype=_FLOAT,
    )
    return FixedRouteFixture(
        x=x,
        z=z,
        y=y,
        prior_weights=prior_weights,
        frequency_counts=frequency_counts,
    )


def quantile_route_fixture() -> FixedRouteFixture:
    """Return exact counts whose expanded row quantiles differ from physical rows."""

    fixture = fixed_route_fixture()
    return FixedRouteFixture(
        x=fixture.x,
        z=fixture.z,
        y=fixture.y,
        prior_weights=fixture.prior_weights,
        frequency_counts=np.tile(np.array([1.0, 4.0, 1.0, 2.0, 1.0]), 4),
    )


def intercept_fixture() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the primary six-row unequal-prior closed-form fixture."""

    return (
        np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0]),
        np.array([0.3, 0.7, 1.0, 1.5, 2.0, 4.0]),
    )


def integer_scale_design_fixture() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return the legal no-scale-intercept frequency equivalence boundary."""

    return (
        np.array([-1.1, 0.2, 1.7, -0.4, 2.3, 0.8]),
        np.array([1.0, 2.0, 3.0, 1.0, 2.0, 4.0]),
        np.array([-1.0, -2.0, 1.0, 2.0, 3.0, -1.0]),
        np.array([-1.0, -2.0, 1.0, 2.0, 3.0, 1.0]),
    )


__all__ = [
    "FixedRouteFixture",
    "GaussianCoefficientOracle",
    "GaussianFitCertificate",
    "GaussianRowOracle",
    "LocalRootCertificate",
    "OracleBounds",
    "assert_gaussian_fit_parity",
    "certify_gaussian_result",
    "coefficient_oracle",
    "covariance_backward_error",
    "explicit_design_blocks",
    "fixed_route_fixture",
    "gamma",
    "gaussian_row_oracle",
    "integer_scale_design_fixture",
    "intercept_fixture",
    "local_root_certificate",
    "oracle_bounds",
    "quantile_route_fixture",
]
