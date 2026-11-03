"""Automatic smoothing configuration and the record of one EFS iteration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm._lambda_box import LAMBDA_CEILING, LAMBDA_FLOOR
from superglm.distributional._common import frozen_float_mapping
from superglm.distributional.results.endpoint_evidence import (
    EFSAccelerationOutcome,
    EndpointAssessmentFailureReason,
    EndpointDirectionEvidence,
    JointEndpointDirectionEvidence,
    _finite_nonnegative,
    _finite_positive,
)
from superglm.distributional.smoothing.acceleration import AccelerationRefusalReason
from superglm.distributional.telemetry import CurvatureTelemetry


def _maximum_relative_natural_parameter_change(
    source: NDArray | None,
    candidate: NDArray | None,
) -> float:
    """Largest fitted-parameter movement on a stable relative scale."""
    if source is None or candidate is None:
        raise ValueError("natural-parameter rows are required for plateau replay")
    source_array = np.asarray(source, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if source_array.shape != candidate_array.shape or source_array.ndim != 2:
        raise ValueError("natural-parameter states must have one matching two-dimensional shape")
    difference = np.abs(candidate_array - source_array)
    scale = 1.0 + np.maximum(np.abs(source_array), np.abs(candidate_array))
    with np.errstate(over="ignore", invalid="ignore"):
        relative = difference / scale
    maximum = float(np.max(relative, initial=0.0))
    if not math.isfinite(maximum):
        raise ValueError("natural-parameter movement must be finite")
    return maximum


#: The default conditioning limit of a component released beyond the cap.
DEFAULT_LAMBDA_CONDITIONING = 1.0e14


@dataclass(frozen=True)
class DistributionalEFSConfig:
    """Outer-loop policy for the algorithm-matched EFS path."""

    max_iterations: int = 50
    tolerance: float = 1.0e-6
    max_log_step: float = 5.0
    minimum_lambda: float = LAMBDA_FLOOR
    maximum_lambda: float = LAMBDA_CEILING
    max_backtracks: int = 8
    backtrack_factor: float = 0.5
    objective_tolerance: float = 1.0e-9
    initial_lambda: float | None = 0.1
    plateau_tolerance: float = 1.0e-7
    plateau_iterations: int = 3
    practical_convergence: bool = False
    practical_parameter_tolerance: float = 1.0e-3
    acceleration: Literal["none", "multisecant"] = "none"
    acceleration_history: int = 5
    acceleration_max_amplification: float = 8.0
    # The outer method. ``"efs"`` is the Fellner--Schall fixed point;
    # ``"efs+newton"`` adds a Newton endgame after the warm-up.
    outer: Literal["efs", "efs+newton"] = "efs"
    handoff_step: float = 0.5
    handoff_iterations: int = 10
    # An optional cap on the endgame's own iterations (``None``: what remains of
    # ``max_iterations``, which bounds warm-up and endgame together), the
    # finite-difference step of the row-curvature derivatives (link scale), the
    # fraction of the smallest active Hessian diagonal that its certificate may
    # reach before the iteration steps by damped BFGS instead, and the
    # conditioning limit a component released beyond ``maximum_lambda`` by the
    # bracketed search may reach.
    max_newton_iterations: int | None = None
    derivative_step: float = 1.0e-3  # endpoint_direction.DEFAULT_STEP (pinned by test)
    hessian_certificate_fraction: float = 0.1
    # ``None`` resolves to ``max(1e14, maximum_lambda)``: a cap above 1e14 is
    # itself the conditioning limit rather than a configuration error.
    maximum_lambda_conditioning: float | None = None
    retain_history_rows: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.retain_history_rows, bool):
            raise TypeError("retain_history_rows must be bool")
        for name in (
            "max_iterations",
            "max_backtracks",
            "plateau_iterations",
            "acceleration_history",
            "handoff_iterations",
        ):
            value = getattr(self, name)
            # Keep iteration/history windows positive.  A zero plateau streak
            # would qualify immediately, although fresh evidence still
            # independently controls convergence.
            minimum = 0 if name == "max_backtracks" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualification = "positive" if minimum else "non-negative"
                raise ValueError(f"{name} must be a {qualification} integer")
        for name in (
            "tolerance",
            "max_log_step",
            "minimum_lambda",
            "maximum_lambda",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        if self.initial_lambda is not None:
            object.__setattr__(
                self, "initial_lambda", _finite_positive(self.initial_lambda, name="initial_lambda")
            )
        object.__setattr__(
            self,
            "objective_tolerance",
            _finite_nonnegative(self.objective_tolerance, name="objective_tolerance"),
        )
        object.__setattr__(
            self,
            "plateau_tolerance",
            _finite_nonnegative(self.plateau_tolerance, name="plateau_tolerance"),
        )
        if not isinstance(self.practical_convergence, bool):
            raise TypeError("practical_convergence must be bool")
        object.__setattr__(
            self,
            "practical_parameter_tolerance",
            _finite_nonnegative(
                self.practical_parameter_tolerance,
                name="practical_parameter_tolerance",
            ),
        )
        if self.maximum_lambda < self.minimum_lambda:
            raise ValueError("maximum_lambda must not be smaller than minimum_lambda")
        if self.initial_lambda is not None and not (
            self.minimum_lambda <= self.initial_lambda <= self.maximum_lambda
        ):
            raise ValueError("initial_lambda must lie inside the configured lambda bounds")
        if not math.isfinite(self.backtrack_factor) or not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie strictly between zero and one")
        if self.acceleration not in ("none", "multisecant"):
            raise ValueError("acceleration must be 'none' or 'multisecant'")
        if self.outer not in ("efs", "efs+newton"):
            raise ValueError("outer must be 'efs' or 'efs+newton'")
        newton_cap = self.max_newton_iterations
        if newton_cap is not None and (
            isinstance(newton_cap, bool) or not isinstance(newton_cap, int) or newton_cap < 1
        ):
            raise ValueError("max_newton_iterations must be None or a positive integer")
        for name in ("handoff_step", "derivative_step"):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
        conditioning = self.maximum_lambda_conditioning
        if conditioning is None:
            conditioning = max(DEFAULT_LAMBDA_CONDITIONING, self.maximum_lambda)
        conditioning = _finite_positive(conditioning, name="maximum_lambda_conditioning")
        object.__setattr__(
            self,
            "maximum_lambda_conditioning",
            conditioning,
        )
        if (
            isinstance(self.hessian_certificate_fraction, bool)
            or not math.isfinite(self.hessian_certificate_fraction)
            or not 0.0 < self.hessian_certificate_fraction <= 1.0
        ):
            raise ValueError("hessian_certificate_fraction must lie in (0, 1]")
        object.__setattr__(
            self, "hessian_certificate_fraction", float(self.hessian_certificate_fraction)
        )
        if conditioning < self.maximum_lambda:
            raise ValueError("maximum_lambda_conditioning must not be smaller than maximum_lambda")
        object.__setattr__(
            self,
            "acceleration_max_amplification",
            _finite_positive(
                self.acceleration_max_amplification,
                name="acceleration_max_amplification",
            ),
        )


@dataclass(frozen=True)
class DistributionalEFSIteration:
    """One proposed smoothing update plus all coefficient refits used to assess it."""

    iteration: int
    source_fit_index: int
    lambdas_before: Mapping[str, float]
    proposed_lambdas: Mapping[str, float]
    lambdas_after: Mapping[str, float]
    proposed_log_steps: Mapping[str, float]
    accepted_log_steps: Mapping[str, float]
    quadratic_forms: Mapping[str, float]
    trace_terms: Mapping[str, float]
    objective_before: float
    objective_after: float
    objective_relative_change: float
    max_proposed_log_step: float
    max_accepted_log_step: float
    accepted: bool
    acceleration_outcome: EFSAccelerationOutcome
    acceleration_refusal_reason: AccelerationRefusalReason | None
    accelerated_fit_index: int | None
    backtracks: int
    raw_backtracks: int
    coefficient_fit_indices: tuple[int, ...]
    accepted_fit_index: int | None
    coefficient_tolerances: tuple[float, ...]
    update_curvature: CurvatureTelemetry
    accepted_curvature: CurvatureTelemetry | None
    activated_face_components: tuple[str, ...] = ()
    deactivated_face_components: tuple[str, ...] = ()
    revalidated_face_components: tuple[str, ...] = ()
    refused_face_components: tuple[str, ...] = ()
    endpoint_direction_evidence: (
        EndpointDirectionEvidence | JointEndpointDirectionEvidence | None
    ) = None
    endpoint_assessment_failure_reason: EndpointAssessmentFailureReason | None = None
    # The Newton endgame's record.  ``stage="newton"`` iterations carry the
    # exact LAML gradient (with its finite-difference certificate) that
    # produced their step, the largest Hessian certificate on the active set
    # (``None`` on a pass that formed no exact Hessian: a quasi-Newton step
    # from the reused memory or a bracket step), the
    # projected-gradient norm judged for convergence, and the ridge that
    # made the active Hessian positive definite (``None`` for a BFGS or
    # bracket step).  A convergence check that makes no trial fit is not an
    # iteration and leaves no record.
    stage: Literal["efs", "newton"] = "efs"
    step_source: Literal["efs", "newton", "bfgs", "bracket"] = "efs"
    gradient: Mapping[str, float] | None = None
    gradient_certificate: Mapping[str, float] | None = None
    hessian_certificate: float | None = None
    projected_gradient_norm: float | None = None
    newton_ridge: float | None = None

    def __post_init__(self) -> None:
        for name in ("lambdas_before", "proposed_lambdas", "lambdas_after"):
            mapping = frozen_float_mapping(getattr(self, name), name=name, nonnegative=True)
            object.__setattr__(self, name, mapping)
        for name in ("proposed_log_steps", "accepted_log_steps", "quadratic_forms", "trace_terms"):
            object.__setattr__(self, name, frozen_float_mapping(getattr(self, name), name=name))
        for name in ("objective_before", "objective_after"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in (
            "objective_relative_change",
            "max_proposed_log_step",
            "max_accepted_log_step",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name=name))
        object.__setattr__(self, "coefficient_fit_indices", tuple(self.coefficient_fit_indices))
        object.__setattr__(
            self,
            "coefficient_tolerances",
            tuple(
                _finite_positive(value, name="coefficient tolerance")
                for value in self.coefficient_tolerances
            ),
        )
        for name in (
            "activated_face_components",
            "deactivated_face_components",
            "revalidated_face_components",
            "refused_face_components",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.gradient is not None:
            object.__setattr__(
                self, "gradient", frozen_float_mapping(self.gradient, name="gradient")
            )
        if self.gradient_certificate is not None:
            certificate = frozen_float_mapping(
                self.gradient_certificate, name="gradient_certificate", nonnegative=True
            )
            object.__setattr__(self, "gradient_certificate", certificate)
        for name in ("hessian_certificate", "projected_gradient_norm", "newton_ridge"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_nonnegative(value, name=name))
