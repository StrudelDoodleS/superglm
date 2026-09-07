"""Endpoint direction evidence and the authority vocabulary every result shares."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

CoefficientCurvature = Literal["fisher", "observed"]


ExecutionBackendIdentifier = Literal[
    "distributional-dense-v1",
    "distributional-chunked-v1",
]


DENSE_EXECUTION_BACKEND_IDENTIFIER: ExecutionBackendIdentifier = "distributional-dense-v1"


CHUNKED_EXECUTION_BACKEND_IDENTIFIER: ExecutionBackendIdentifier = "distributional-chunked-v1"


ConvergenceReason = Literal[
    "score",
    "newton_decrement",
    "objective_and_step",
    "objective_and_score",
    "resolution_limited_stationarity",
    "max_iterations",
    "line_search_failed",
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
        if self.authority_identifier not in DIRECTION_AUTHORITIES:
            raise ValueError("unknown endpoint direction authority")
        if self.decision not in {"endpoint", "finite", "unresolved"}:
            raise ValueError("invalid endpoint direction decision")
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
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        fit_indices = tuple(self.fit_indices)
        if fit_indices:
            if (
                len(fit_indices) != 2
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in fit_indices
                )
                or tuple(sorted(fit_indices)) != fit_indices
                or len(set(fit_indices)) != len(fit_indices)
            ):
                raise ValueError("endpoint evidence must index its cap and endpoint fits")
            if self.coefficient_tolerance is None:
                raise ValueError("indexed endpoint evidence requires a coefficient tolerance")
            coefficient_tolerance = _finite_positive(
                self.coefficient_tolerance,
                name="coefficient_tolerance",
            )
        else:
            if self.coefficient_tolerance is not None:
                raise ValueError("unindexed endpoint evidence cannot carry a tolerance")
            coefficient_tolerance = None
        if not self._derived_authority_matches():
            raise ValueError("derived endpoint direction evidence is internally inconsistent")
        object.__setattr__(self, "fit_indices", fit_indices)
        object.__setattr__(self, "coefficient_tolerance", coefficient_tolerance)

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
        if self.authority_identifier not in JOINT_DIRECTION_AUTHORITIES:
            raise ValueError("unknown joint endpoint direction authority")
        if not isinstance(self.component_directions, Sequence) or isinstance(
            self.component_directions,
            str | bytes,
        ):
            raise TypeError("component_directions must be an ordered sequence")
        raw_directions = tuple(self.component_directions)
        component_directions: list[tuple[str, EndpointDirectionEvidence]] = []
        for item in raw_directions:
            if not isinstance(item, tuple | list) or len(item) != 2:
                raise TypeError("each joint endpoint direction must be a name-evidence pair")
            name, direction = item
            if not isinstance(name, str) or not name:
                raise ValueError("joint endpoint component names must be non-empty strings")
            if not isinstance(direction, EndpointDirectionEvidence):
                raise TypeError(
                    "joint endpoint directions must contain EndpointDirectionEvidence values"
                )
            component_directions.append((name, direction))
        normalized = tuple(component_directions)
        if len(normalized) < 2:
            raise ValueError("joint endpoint evidence requires at least two components")
        component_names = tuple(name for name, _direction in normalized)
        if len(set(component_names)) != len(component_names):
            raise ValueError("joint endpoint component names must be unique")
        if any(
            direction.fit_indices or direction.coefficient_tolerance is not None
            for _name, direction in normalized
        ):
            raise ValueError("joint endpoint inner directions must remain unindexed")
        objective_bits = float(normalized[0][1].endpoint_objective).hex()
        if any(
            float(direction.endpoint_objective).hex() != objective_bits
            for _name, direction in normalized[1:]
        ):
            raise ValueError("joint directions must share the same endpoint objective")
        expected_joint = (
            JOINT_ANALYTIC_DIRECTION_AUTHORITY
            if all(
                direction.authority_identifier == ANALYTIC_DIRECTION_AUTHORITY
                for _name, direction in normalized
            )
            else JOINT_FINITE_DIFFERENCE_DIRECTION_AUTHORITY
        )
        if self.authority_identifier != expected_joint:
            raise ValueError("joint endpoint direction authority does not match its components")

        endpoint_fit_index = self.endpoint_fit_index
        coefficient_tolerance = self.coefficient_tolerance
        if endpoint_fit_index is None:
            if coefficient_tolerance is not None:
                raise ValueError("unindexed joint endpoint evidence cannot carry a tolerance")
            coefficient_tolerance = None
        else:
            if (
                isinstance(endpoint_fit_index, bool)
                or not isinstance(endpoint_fit_index, int)
                or endpoint_fit_index < 0
            ):
                raise ValueError("endpoint_fit_index must be a non-negative integer")
            if coefficient_tolerance is None:
                raise ValueError("indexed joint endpoint evidence requires a coefficient tolerance")
            coefficient_tolerance = _finite_positive(
                coefficient_tolerance,
                name="coefficient_tolerance",
            )

        object.__setattr__(self, "component_directions", normalized)
        object.__setattr__(self, "endpoint_fit_index", endpoint_fit_index)
        object.__setattr__(self, "coefficient_tolerance", coefficient_tolerance)
        if not self._derived_authority_matches():
            raise ValueError("derived joint endpoint evidence is internally inconsistent")

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
