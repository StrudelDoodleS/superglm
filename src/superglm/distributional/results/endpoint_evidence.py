"""Endpoint direction evidence and the authority vocabulary every result shares."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

CoefficientCurvature = Literal["fisher", "observed"]


ExecutionBackendIdentifier = Literal[
    "distributional-dense-v1",
    "distributional-chunked-v1",
]


DENSE_EXECUTION_BACKEND_IDENTIFIER: ExecutionBackendIdentifier = "distributional-dense-v1"


CHUNKED_EXECUTION_BACKEND_IDENTIFIER: ExecutionBackendIdentifier = "distributional-chunked-v1"


ConvergenceReason = Literal[
    "score",
    "objective_and_step",
    "objective_and_score",
    "resolution_limited_stationarity",
    "max_iterations",
    "line_search_failed",
    # No damping of the coefficient curvature gave a certified ascent direction.
    "no_ascent_direction",
]


EFSConvergenceReason = Literal[
    "fixed_only",
    "lambda_change",
    "objective_plateau",
    "practical_plateau",
    "lambda_cap_unresolved",
    "endpoint_revalidation_failed",
    "max_iterations",
    "objective_rejected",
    "coefficient_not_converged",
    "stationary",
    "gradient_unresolved",
    "derivative_unavailable",
]


EFSAccelerationOutcome = Literal[
    "disabled",
    "warming",
    "refused",
    "accepted",
    "rejected",
]


EndpointDirectionDecision = Literal["endpoint", "finite", "unresolved"]


ANALYTIC_DIRECTION_AUTHORITY = "analytic-observed-curvature-direction/v1"


FINITE_DIFFERENCE_DIRECTION_AUTHORITY = "finite-difference-curvature-direction/v1"


DIRECTION_AUTHORITIES = frozenset(
    {ANALYTIC_DIRECTION_AUTHORITY, FINITE_DIFFERENCE_DIRECTION_AUTHORITY}
)


JOINT_ANALYTIC_DIRECTION_AUTHORITY = "joint-analytic-observed-curvature-direction/v1"


JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY = "joint-finite-difference-curvature-direction/v1"


JOINT_DIRECTION_AUTHORITIES = frozenset(
    {JOINT_ANALYTIC_DIRECTION_AUTHORITY, JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY}
)


EndpointAssessmentFailureReason = Literal[
    "cap_not_converged",
    "cap_not_stationary",
    "endpoint_not_converged",
    "endpoint_not_stationary",
    "endpoint_state_changed",
    "provenance_changed",
    # Diagnostic only: no curvature direction derivative could be evaluated,
    # analytic or finite-difference; two fitted states cannot prove why.
    "analytic_unavailable",
    "joint_endpoint_not_converged",
    "joint_endpoint_not_stationary",
    # Diagnostic only: no curvature direction derivative could be evaluated for
    # the joint face, analytic or finite-difference.
    "joint_analytic_unavailable",
    "joint_objective_rejected",
]


@dataclass(frozen=True)
class EndpointDirectionEvidence:
    """Analytic one-sided endpoint direction and its fitted-state references."""

    authority_identifier: str
    decision: EndpointDirectionDecision
    endpoint_objective: float
    analytic_derivative: float
    profile_score_term: float
    curvature_schur_term: float
    curvature_drift_term: float
    numerical_error: float
    lower_bound: float
    upper_bound: float
    fit_indices: tuple[int, ...] = ()
    coefficient_tolerance: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "endpoint_objective",
            "analytic_derivative",
            "profile_score_term",
            "curvature_schur_term",
            "curvature_drift_term",
            "numerical_error",
            "lower_bound",
            "upper_bound",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "fit_indices", tuple(self.fit_indices))
        if self.coefficient_tolerance is not None:
            tolerance = _finite_positive(self.coefficient_tolerance, name="coefficient_tolerance")
            object.__setattr__(self, "coefficient_tolerance", tolerance)

    def _derived_authority_matches(self) -> bool:
        if self.authority_identifier not in DIRECTION_AUTHORITIES:
            return False
        expected_derivative = 0.5 * (
            self.profile_score_term + self.curvature_schur_term + self.curvature_drift_term
        )
        if self.analytic_derivative != expected_derivative:
            return False
        expected_lower = self.analytic_derivative - self.numerical_error
        expected_upper = self.analytic_derivative + self.numerical_error
        if expected_lower > 0.0:
            expected_decision: EndpointDirectionDecision = "endpoint"
        elif expected_upper < 0.0:
            expected_decision = "finite"
        else:
            expected_decision = "unresolved"
        return bool(
            self.numerical_error >= 0.0
            and self.lower_bound == expected_lower
            and self.upper_bound == expected_upper
            and self.decision == expected_decision
        )


@dataclass(frozen=True)
class JointEndpointDirectionEvidence:
    """Named analytic directions evaluated from one exact joint-face fit."""

    authority_identifier: str
    component_directions: tuple[tuple[str, EndpointDirectionEvidence], ...]
    endpoint_fit_index: int | None = None
    coefficient_tolerance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_directions",
            tuple((name, direction) for name, direction in self.component_directions),
        )
        if self.coefficient_tolerance is not None:
            tolerance = _finite_positive(self.coefficient_tolerance, name="coefficient_tolerance")
            object.__setattr__(self, "coefficient_tolerance", tolerance)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _direction in self.component_directions)

    @property
    def endpoint_objective(self) -> float:
        return self.component_directions[0][1].endpoint_objective

    @property
    def fit_indices(self) -> tuple[int, ...]:
        if self.endpoint_fit_index is None:
            return ()
        return (self.endpoint_fit_index,)

    @property
    def decision(self) -> EndpointDirectionDecision:
        decisions = tuple(direction.decision for _name, direction in self.component_directions)
        if "finite" in decisions:
            return "finite"
        if "unresolved" in decisions:
            return "unresolved"
        return "endpoint"

    def _derived_authority_matches(self) -> bool:
        if (
            self.authority_identifier not in JOINT_DIRECTION_AUTHORITIES
            or len(self.component_directions) < 2
        ):
            return False
        names = self.component_names
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
            or any(
                not isinstance(direction, EndpointDirectionEvidence)
                or direction.fit_indices
                or direction.coefficient_tolerance is not None
                or not direction._derived_authority_matches()
                for _name, direction in self.component_directions
            )
        ):
            return False
        objective_bits = float(self.endpoint_objective).hex()
        if any(
            float(direction.endpoint_objective).hex() != objective_bits
            for _name, direction in self.component_directions[1:]
        ):
            return False
        if self.endpoint_fit_index is None:
            return self.coefficient_tolerance is None
        return bool(
            not isinstance(self.endpoint_fit_index, bool)
            and isinstance(self.endpoint_fit_index, int)
            and self.endpoint_fit_index >= 0
            and self.coefficient_tolerance is not None
            and math.isfinite(self.coefficient_tolerance)
            and self.coefficient_tolerance > 0.0
        )


def _finite_positive(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)
