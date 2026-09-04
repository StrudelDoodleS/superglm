"""Automatic smoothing configuration and the record of one EFS iteration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.results.endpoint_evidence import (
    EFSAccelerationOutcome,
    EndpointAssessmentFailureReason,
    EndpointDirectionEvidence,
    JointEndpointDirectionEvidence,
    _finite_nonnegative,
    _finite_positive,
)
from superglm.distributional.results.solver import _frozen_float_mapping
from superglm.distributional.smoothing.acceleration import AccelerationRefusalReason
from superglm.distributional.telemetry import CurvatureTelemetry


def _maximum_relative_natural_parameter_change(
    source: NDArray,
    candidate: NDArray,
) -> float:
    """Largest fitted-parameter movement on a stable relative scale."""
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
    minimum_lambda: float = 1.0e-6
    maximum_lambda: float = 1.0e10
    max_backtracks: int = 8
    backtrack_factor: float = 0.5
    objective_tolerance: float = 1.0e-9
    initial_lambda: float = 0.1
    plateau_tolerance: float = 1.0e-7
    plateau_iterations: int = 3
    practical_convergence: bool = False
    practical_parameter_tolerance: float = 1.0e-3
    # OPT-IN telemetry, disabled by default.  A component whose true effect
    # lies in its own penalty's null space has an optimal lambda of +infinity,
    # which the finite log-step test can never reach: the rearranged fixed
    # point advances such a lambda by a constant additive increment, so
    # |dlog lambda| decays as 1/iteration.
    #
    # The classifier is lambda_j tr(H^-1 S_j) / r_j, which rises to one when
    # the penalty has absorbed the component's whole block of the curvature --
    # but that is a statement about the CURRENT lambda, not about where the
    # optimum is.  A fit started above a finite optimum shows the same
    # saturation while walking down toward it, so nomination additionally
    # requires the component to still be climbing (a positive proposed log
    # step, the signature of boundary drift) for the whole streak.  No
    # threshold is sound enough to remove the coordinate or stop the fit: a
    # finite optimum with a barely-significant effect can sit at saturation
    # arbitrarily close to one.  The nomination is telemetry only.  The
    # default of 1.0 keeps it off.
    boundary_saturation: float = 1.0
    boundary_iterations: int = 3
    acceleration: Literal["none", "multisecant"] = "none"
    acceleration_history: int = 5
    acceleration_max_amplification: float = 8.0
    # The outer method. ``"efs"`` is the Fellner--Schall fixed point;
    # ``"efs+newton"`` opts into a Newton endgame after the warm-up.
    outer: Literal["efs", "efs+newton"] = "efs"
    handoff_step: float = 0.5
    handoff_iterations: int = 10
    # The endgame's own budget, the finite-difference step of the row-curvature
    # derivatives (link scale), the fraction of the smallest active Hessian
    # diagonal that its certificate may reach before the iteration steps by
    # damped BFGS instead, and the conditioning limit a component released
    # beyond ``maximum_lambda`` by the bracketed search may reach.
    max_newton_iterations: int = 20
    derivative_step: float = 1.0e-3  # endpoint_direction.DEFAULT_STEP (pinned by test)
    hessian_certificate_fraction: float = 0.1
    # ``None`` resolves to ``max(1e14, maximum_lambda)``: a cap above 1e14 is
    # itself the conditioning limit rather than a configuration error.
    maximum_lambda_conditioning: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_iterations",
            "max_backtracks",
            "plateau_iterations",
            "boundary_iterations",
            "acceleration_history",
            "handoff_iterations",
            "max_newton_iterations",
        ):
            value = getattr(self, name)
            # Keep iteration/history windows positive.  A zero nomination or
            # plateau streak would qualify immediately, although fresh
            # evidence still independently controls convergence.
            minimum = 0 if name == "max_backtracks" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualification = "positive" if minimum else "non-negative"
                raise ValueError(f"{name} must be a {qualification} integer")
        for name in (
            "tolerance",
            "max_log_step",
            "minimum_lambda",
            "maximum_lambda",
            "initial_lambda",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name=name))
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
        if not self.minimum_lambda <= self.initial_lambda <= self.maximum_lambda:
            raise ValueError("initial_lambda must lie inside the configured lambda bounds")
        if not math.isfinite(self.boundary_saturation) or not 0.0 < self.boundary_saturation <= 1.0:
            raise ValueError("boundary_saturation must lie in (0, 1]")
        if not math.isfinite(self.backtrack_factor) or not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie strictly between zero and one")
        if self.acceleration not in ("none", "multisecant"):
            raise ValueError("acceleration must be 'none' or 'multisecant'")
        if self.outer not in ("efs", "efs+newton"):
            raise ValueError("outer must be 'efs' or 'efs+newton'")
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
    boundary_nominations: tuple[str, ...]
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
    joint_rollback_penalty_fingerprint: str | None = None
    # The Newton endgame's record.  ``stage="newton"`` iterations carry the
    # exact LAML gradient (with its finite-difference certificate) that
    # produced their step, the largest Hessian certificate on the active set
    # (``None`` on a pass that formed no exact Hessian: a quasi-Newton step
    # from the reused memory, a halved step or a bracket step), the
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
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or self.iteration < 1
        ):
            raise ValueError("iteration must be a positive integer")
        if (
            isinstance(self.source_fit_index, bool)
            or not isinstance(self.source_fit_index, int)
            or self.source_fit_index < 0
        ):
            raise ValueError("source_fit_index must be a non-negative integer")
        before = _frozen_float_mapping(
            self.lambdas_before,
            name="lambdas_before",
            nonnegative=True,
        )
        names = tuple(before)
        mappings = {
            "proposed_lambdas": _frozen_float_mapping(
                self.proposed_lambdas,
                name="proposed_lambdas",
                nonnegative=True,
            ),
            "lambdas_after": _frozen_float_mapping(
                self.lambdas_after,
                name="lambdas_after",
                nonnegative=True,
            ),
            "proposed_log_steps": _frozen_float_mapping(
                self.proposed_log_steps,
                name="proposed_log_steps",
            ),
            "accepted_log_steps": _frozen_float_mapping(
                self.accepted_log_steps,
                name="accepted_log_steps",
            ),
            "quadratic_forms": _frozen_float_mapping(
                self.quadratic_forms,
                name="quadratic_forms",
            ),
            "trace_terms": _frozen_float_mapping(self.trace_terms, name="trace_terms"),
        }
        if any(tuple(mapping) != names for mapping in mappings.values()):
            raise ValueError("EFS iteration mappings must share one deterministic key order")
        for field_name, mapping in mappings.items():
            object.__setattr__(self, field_name, mapping)
        object.__setattr__(self, "lambdas_before", before)
        for name in (
            "objective_before",
            "objective_after",
            "objective_relative_change",
            "max_proposed_log_step",
            "max_accepted_log_step",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name)
                if name not in ("objective_before", "objective_after")
                else float(getattr(self, name)),
            )
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        derived_objective_relative_change = abs(self.objective_after - self.objective_before) / (
            1.0 + abs(self.objective_before)
        )
        if self.objective_relative_change != derived_objective_relative_change:
            raise ValueError(
                "objective_relative_change must match objective_before and objective_after"
            )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        if isinstance(self.backtracks, bool) or not isinstance(self.backtracks, int):
            raise ValueError("backtracks must be a non-negative integer")
        if (
            isinstance(self.raw_backtracks, bool)
            or not isinstance(self.raw_backtracks, int)
            or self.raw_backtracks < 0
            or self.raw_backtracks > self.backtracks
        ):
            raise ValueError("raw_backtracks must be between zero and total backtracks")
        fit_indices = tuple(self.coefficient_fit_indices)
        if (
            not fit_indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in fit_indices
            )
            or tuple(sorted(fit_indices)) != fit_indices
            or len(set(fit_indices)) != len(fit_indices)
        ):
            raise ValueError(
                "coefficient_fit_indices must be unique increasing non-negative integers"
            )
        face_assessment = bool(
            self.activated_face_components
            or self.deactivated_face_components
            or self.revalidated_face_components
            or self.refused_face_components
        )
        if face_assessment:
            if self.backtracks != 0 or self.raw_backtracks != 0:
                raise ValueError("exact-face assessments do not represent backtracking trials")
        elif self.backtracks != len(fit_indices) - 1:
            raise ValueError("backtracks must match the number of attempted coefficient fits")
        tolerances = tuple(
            _finite_positive(value, name="coefficient tolerance")
            for value in self.coefficient_tolerances
        )
        if len(tolerances) != len(fit_indices):
            raise ValueError("one coefficient tolerance is required per attempted fit")
        if self.accepted:
            if self.accepted_fit_index != fit_indices[-1] or self.accepted_curvature is None:
                raise ValueError("accepted EFS iterations must identify the accepted terminal fit")
        elif self.accepted_fit_index is not None or self.accepted_curvature is not None:
            raise ValueError("rejected EFS iterations cannot publish an accepted fit")
        valid_outcomes = {"disabled", "warming", "refused", "accepted", "rejected"}
        if self.acceleration_outcome not in valid_outcomes:
            raise ValueError("invalid EFS acceleration outcome")
        refusal_reason = self.acceleration_refusal_reason
        if self.acceleration_outcome == "refused":
            valid_refusal_reasons = {
                "zero_raw_residual",
                "zero_history_rank",
                "no_model_reduction",
                "current_duplicate",
                "raw_duplicate",
                "box_blocked",
                "nonfinite",
                "non_gfs_proposal",
            }
            if refusal_reason not in valid_refusal_reasons:
                raise ValueError("refused acceleration must retain its refusal reason")
        elif refusal_reason is not None:
            raise ValueError("only refused acceleration may retain a refusal reason")
        accelerated_fit_index = self.accelerated_fit_index
        if accelerated_fit_index is not None and (
            isinstance(accelerated_fit_index, bool)
            or not isinstance(accelerated_fit_index, int)
            or accelerated_fit_index < 0
        ):
            raise ValueError("accelerated_fit_index must be a non-negative integer or None")
        if self.acceleration_outcome == "accepted":
            if (
                not self.accepted
                or accelerated_fit_index != self.accepted_fit_index
                or fit_indices != (accelerated_fit_index,)
                or self.raw_backtracks != 0
            ):
                raise ValueError("accepted acceleration must be the sole accepted trial")
        elif self.acceleration_outcome == "rejected":
            if (
                accelerated_fit_index != fit_indices[0]
                or accelerated_fit_index == self.accepted_fit_index
                or len(fit_indices) < 2
                or self.raw_backtracks != self.backtracks - 1
            ):
                raise ValueError("rejected acceleration must precede the raw fallback trials")
        elif accelerated_fit_index is not None:
            raise ValueError("unattempted acceleration cannot identify a coefficient fit")
        elif self.raw_backtracks != self.backtracks:
            raise ValueError("raw-only trials must account for every backtrack")
        nominations = tuple(self.boundary_nominations)
        if (
            any(not isinstance(name, str) or name not in names for name in nominations)
            or len(set(nominations)) != len(nominations)
            or tuple(name for name in names if name in nominations) != nominations
        ):
            raise ValueError("boundary_nominations must be unique and follow lambda order")
        activated = tuple(self.activated_face_components)
        if (
            any(not isinstance(name, str) or name not in names for name in activated)
            or len(set(activated)) != len(activated)
            or tuple(name for name in names if name in set(activated)) != activated
        ):
            raise ValueError(
                "activated_face_components must contain distinct names in lambda order"
            )
        deactivated = tuple(self.deactivated_face_components)
        if (
            any(not isinstance(name, str) or name not in names for name in deactivated)
            or len(set(deactivated)) != len(deactivated)
            or tuple(name for name in names if name in set(deactivated)) != deactivated
            or (activated and deactivated)
        ):
            raise ValueError(
                "deactivated_face_components must contain distinct names in lambda order"
            )
        rollback_fingerprint = self.joint_rollback_penalty_fingerprint
        if rollback_fingerprint is not None:
            if (
                not isinstance(rollback_fingerprint, str)
                or len(rollback_fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in rollback_fingerprint)
            ):
                raise ValueError(
                    "joint rollback penalty fingerprint must be lowercase SHA-256 hexadecimal"
                )
            if len(deactivated) < 2:
                raise ValueError("only a joint endpoint rollback may retain a penalty fingerprint")
        revalidated = tuple(self.revalidated_face_components)
        if (
            any(not isinstance(name, str) or name not in names for name in revalidated)
            or len(set(revalidated)) != len(revalidated)
            or tuple(name for name in names if name in set(revalidated)) != revalidated
            or (revalidated and (activated or deactivated))
        ):
            raise ValueError(
                "revalidated_face_components must contain distinct names in lambda order"
            )
        refused = tuple(self.refused_face_components)
        if (
            any(not isinstance(name, str) or name not in names for name in refused)
            or len(set(refused)) != len(refused)
            or tuple(name for name in names if name in set(refused)) != refused
            or (refused and (activated or deactivated or revalidated or self.accepted))
        ):
            raise ValueError(
                "refused_face_components must identify unaccepted components in lambda order"
            )
        direction = self.endpoint_direction_evidence
        assessment_failure = self.endpoint_assessment_failure_reason
        scalar_assessment_failures = {
            "cap_not_converged",
            "cap_not_stationary",
            "endpoint_not_converged",
            "endpoint_not_stationary",
            "provenance_changed",
            "analytic_unavailable",
        }
        joint_assessment_failures = {
            "joint_endpoint_not_converged",
            "joint_endpoint_not_stationary",
            "joint_analytic_unavailable",
            "joint_objective_rejected",
        }
        valid_assessment_failures = scalar_assessment_failures | joint_assessment_failures
        if assessment_failure is not None and assessment_failure not in valid_assessment_failures:
            raise ValueError("invalid endpoint assessment failure reason")
        if assessment_failure is not None:
            failure_names = refused or deactivated
            if not failure_names or direction is not None:
                raise ValueError(
                    "an endpoint assessment failure reason requires an event without direction "
                    "evidence"
                )
            if deactivated and (
                len(deactivated) < 2 or assessment_failure not in joint_assessment_failures
            ):
                raise ValueError(
                    "only a joint deactivation may retain a joint assessment failure reason"
                )
        if (activated or deactivated or revalidated) and not self.accepted:
            raise ValueError("an exact-face transition must publish its accepted coefficient fit")

        def validate_joint_direction(
            event_names: tuple[str, ...],
            *,
            allows_rollback_fit: bool = False,
        ) -> None:
            if not isinstance(direction, JointEndpointDirectionEvidence):
                raise TypeError(
                    "a multi-component exact-face event requires joint endpoint direction evidence"
                )
            if direction.component_names != event_names:
                raise ValueError(
                    "joint endpoint direction names must exactly match the event components"
                )
            expected_direction_indices = fit_indices[:1] if allows_rollback_fit else fit_indices
            if (
                direction.fit_indices != expected_direction_indices
                or len(direction.fit_indices) != 1
                or (allows_rollback_fit and len(fit_indices) != 2)
                or (not allows_rollback_fit and len(fit_indices) != 1)
            ):
                raise ValueError(
                    "joint endpoint direction evidence requires one indexed strict endpoint fit"
                )
            if direction.coefficient_tolerance != tolerances[0]:
                raise ValueError(
                    "joint endpoint direction evidence requires one indexed strict endpoint fit"
                )
            if tolerances[0] > 1.0e-12:
                raise ValueError(
                    "joint endpoint direction evidence requires a strict coefficient tolerance"
                )

        if activated:
            if len(activated) > 1:
                validate_joint_direction(activated)
            elif not isinstance(direction, EndpointDirectionEvidence):
                raise TypeError("an exact-face activation requires endpoint direction evidence")
            assert direction is not None
            if direction.decision != "endpoint":
                raise ValueError("an exact-face activation requires resolved endpoint evidence")
        elif deactivated:
            if len(deactivated) > 1:
                if direction is not None:
                    validate_joint_direction(deactivated, allows_rollback_fit=True)
                elif assessment_failure is not None:
                    if assessment_failure not in joint_assessment_failures or len(fit_indices) != 2:
                        raise ValueError(
                            "a fitted joint endpoint failure requires its common assessment and "
                            "rollback fits"
                        )
                elif len(fit_indices) != 1:
                    raise ValueError(
                        "a joint endpoint preflight failure requires only its rollback fit"
                    )
            elif direction is not None and not isinstance(direction, EndpointDirectionEvidence):
                raise TypeError("exact-face deactivation evidence has the wrong type")
            if direction is not None and direction.decision == "endpoint":
                raise ValueError("an exact face can be deactivated only by non-endpoint evidence")
        elif revalidated:
            if len(revalidated) > 1:
                validate_joint_direction(revalidated)
            elif not isinstance(direction, EndpointDirectionEvidence):
                raise TypeError("an exact-face revalidation requires endpoint direction evidence")
            assert direction is not None
            if direction.decision != "endpoint":
                raise ValueError("an exact-face revalidation requires resolved endpoint evidence")
        elif refused:
            if (direction is None) == (assessment_failure is None):
                raise ValueError(
                    "an exact-face refusal requires direction evidence or a failure reason"
                )
            if len(refused) > 1:
                if direction is not None:
                    validate_joint_direction(refused)
                    if direction.decision == "endpoint":
                        raise ValueError(
                            "a joint exact-face refusal requires at least one non-endpoint direction"
                        )
                elif assessment_failure not in joint_assessment_failures:
                    raise ValueError(
                        "a joint exact-face refusal requires a typed joint failure reason"
                    )
            else:
                if direction is not None and not isinstance(direction, EndpointDirectionEvidence):
                    raise TypeError("exact-face refusal evidence has the wrong type")
                if assessment_failure in joint_assessment_failures:
                    raise ValueError("a scalar exact-face refusal requires a scalar failure reason")
        elif direction is not None:
            raise ValueError("only an exact-face transition may carry endpoint direction evidence")
        elif assessment_failure is not None:
            raise ValueError("only an exact-face refusal may carry an assessment failure reason")
        if not isinstance(self.update_curvature, CurvatureTelemetry):
            raise TypeError("update_curvature must be CurvatureTelemetry")
        if self.accepted_curvature is not None and not isinstance(
            self.accepted_curvature, CurvatureTelemetry
        ):
            raise TypeError("accepted_curvature must be CurvatureTelemetry when present")
        if self.stage not in ("efs", "newton"):
            raise ValueError("stage must be 'efs' or 'newton'")
        if self.step_source not in ("efs", "newton", "bfgs", "bracket"):
            raise ValueError("invalid step_source")
        newton_fields = (
            self.gradient,
            self.gradient_certificate,
            self.hessian_certificate,
            self.projected_gradient_norm,
            self.newton_ridge,
        )
        if self.stage == "efs":
            if self.step_source != "efs" or any(value is not None for value in newton_fields):
                raise ValueError("an EFS iteration carries no Newton endgame record")
        else:
            if self.step_source == "efs":
                raise ValueError("a Newton iteration must name its step source")
            if face_assessment:
                raise ValueError("an exact-face assessment is not a Newton iteration")
            if self.gradient is None or self.projected_gradient_norm is None:
                raise ValueError(
                    "a Newton iteration must carry its gradient and projected gradient norm"
                )
            gradient = _frozen_float_mapping(self.gradient, name="gradient")
            gradient_names = tuple(gradient)
            if (
                any(name not in names for name in gradient_names)
                or tuple(name for name in names if name in gradient_names) != gradient_names
            ):
                raise ValueError("gradient names must follow the iteration's lambda order")
            object.__setattr__(self, "gradient", gradient)
            if self.gradient_certificate is not None:
                certificate = _frozen_float_mapping(
                    self.gradient_certificate, name="gradient_certificate", nonnegative=True
                )
                if tuple(certificate) != gradient_names:
                    raise ValueError("gradient_certificate must cover exactly the gradient names")
                object.__setattr__(self, "gradient_certificate", certificate)
            for name in ("hessian_certificate", "projected_gradient_norm", "newton_ridge"):
                value = getattr(self, name)
                if value is not None:
                    object.__setattr__(self, name, _finite_nonnegative(value, name=name))
        object.__setattr__(self, "coefficient_fit_indices", fit_indices)
        object.__setattr__(self, "coefficient_tolerances", tolerances)
        object.__setattr__(self, "boundary_nominations", nominations)
        object.__setattr__(self, "activated_face_components", activated)
        object.__setattr__(self, "deactivated_face_components", deactivated)
        object.__setattr__(self, "revalidated_face_components", revalidated)
        object.__setattr__(self, "refused_face_components", refused)
