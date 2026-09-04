"""The certified result of automatic smoothing-parameter selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.results.endpoint_evidence import (
    EFSConvergenceReason,
    EndpointDirectionEvidence,
    JointEndpointDirectionEvidence,
    _finite_nonnegative,
)
from superglm.distributional.results.iteration import (
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    _maximum_relative_natural_parameter_change,
)
from superglm.distributional.results.solver import (
    DenseSolverResult,
    _assessment_coefficient_refit_bound,
    _assessment_complete_face_penalty_matches,
    _assessment_exact_face_objective,
    _assessment_face_geometry_matches,
    _assessment_finite_objective,
    _assessment_is_numerically_stationary,
    _assessment_penalty_direction_matches,
    _assessment_retained_kkt_ratio,
    _assessment_scalar_error_bound,
    _assessment_unpenalized_logdet_term,
    _dense_penalty_fingerprint,
    _frozen_endpoint_mapping,
    _frozen_float_mapping,
)


def _revalidated_endpoint_directions(
    item: DistributionalEFSIteration,
) -> tuple[tuple[str, EndpointDirectionEvidence], ...]:
    names = item.revalidated_face_components
    direction = item.endpoint_direction_evidence
    if not names or direction is None:
        return ()
    if isinstance(direction, JointEndpointDirectionEvidence):
        return direction.component_directions
    if isinstance(direction, EndpointDirectionEvidence) and len(names) == 1:
        return ((names[0], direction),)
    return ()


@dataclass(frozen=True)
class DistributionalEFSResult:
    """Complete inner and outer histories for one safeguarded EFS fit.

    ``terminal_raw_max_log_step`` retains its serialized compatibility name,
    but stores the fresh convergence authority: normalized stationarity for a
    finite target, augmented only by a direction that is actionable at the
    configured upper cap or that identifies working infinity.

    A ``stationary`` result is the Newton endgame's: its authority is the
    exact LAML gradient in log lambda, ``terminal_gradient`` with its
    finite-difference certificate, projected onto the box
    (``terminal_projected_gradient_norm`` within ``tolerance * (1 + |F|)``);
    the Fellner--Schall residual is no longer the certificate there and may
    exceed the tolerance.  ``smoothing_hessian`` is the exact Hessian at that
    point over ``terminal_gradient``'s names.  ``beyond_cap_components``
    names components the bracketed search released above ``maximum_lambda``
    (never above ``maximum_lambda_conditioning``).
    """

    config: DistributionalEFSConfig
    initial_lambdas: Mapping[str, float]
    lambdas: Mapping[str, float]
    initial_objective: float
    objective: float
    converged: bool
    convergence_reason: EFSConvergenceReason
    terminal_raw_max_log_step: float
    unresolved_upper_bound: tuple[str, ...]
    iterations: int
    history: tuple[DistributionalEFSIteration, ...]
    coefficient_fits: tuple[DenseSolverResult, ...]
    terminal_fit_index: int
    terminal_endpoint_directions: Mapping[str, EndpointDirectionEvidence] = field(
        default_factory=dict
    )
    terminal_gradient: Mapping[str, float] | None = None
    terminal_gradient_certificate: Mapping[str, float] | None = None
    terminal_projected_gradient_norm: float | None = None
    smoothing_hessian: NDArray | None = None
    smoothing_hessian_certificate: NDArray | None = None
    newton_iterations: int = 0
    bfgs_fallback_iterations: int = 0
    beyond_cap_components: tuple[str, ...] = ()

    @property
    def stationarity_bar(self) -> float:
        """The projected-gradient bar of the Newton endgame at this objective."""

        return self.config.tolerance * (1.0 + abs(self.objective))

    def __post_init__(self) -> None:
        if not isinstance(self.config, DistributionalEFSConfig):
            raise TypeError("config must be DistributionalEFSConfig")
        initial = _frozen_float_mapping(
            self.initial_lambdas,
            name="initial_lambdas",
            nonnegative=True,
        )
        terminal = _frozen_float_mapping(self.lambdas, name="lambdas", nonnegative=True)
        endpoint_directions = _frozen_endpoint_mapping(self.terminal_endpoint_directions)
        if tuple(initial) != tuple(terminal):
            raise ValueError("initial and terminal lambdas must share deterministic key order")
        if any(name not in terminal for name in endpoint_directions) or tuple(
            name for name in terminal if name in endpoint_directions
        ) != tuple(endpoint_directions):
            raise ValueError("terminal endpoint directions must follow terminal lambda order")
        for name in ("initial_objective", "objective"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        history = tuple(self.history)
        fits = tuple(self.coefficient_fits)
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations != len(history)
        ):
            raise ValueError("iterations must equal the smoothing-history length")
        valid_reasons = {
            "fixed_only",
            "lambda_change",
            "objective_plateau",
            "practical_plateau",
            "stationary",
            "lambda_cap_unresolved",
            "endpoint_revalidation_failed",
            "max_iterations",
            "objective_rejected",
            "coefficient_not_converged",
            "gradient_unresolved",
        }
        if self.convergence_reason not in valid_reasons:
            raise ValueError(f"invalid EFS convergence reason: {self.convergence_reason!r}")
        converged_reasons = {
            "fixed_only",
            "lambda_change",
            "objective_plateau",
            "practical_plateau",
            "stationary",
        }
        if self.converged != (self.convergence_reason in converged_reasons):
            raise ValueError("EFS convergence flag and reason disagree")
        terminal_raw_max_log_step = _finite_nonnegative(
            self.terminal_raw_max_log_step,
            name="terminal_raw_max_log_step",
        )
        unresolved_upper_bound = tuple(self.unresolved_upper_bound)
        unresolved_set = set(unresolved_upper_bound)
        if (
            any(
                not isinstance(name, str) or name not in terminal for name in unresolved_upper_bound
            )
            or len(unresolved_set) != len(unresolved_upper_bound)
            or tuple(name for name in terminal if name in unresolved_set) != unresolved_upper_bound
        ):
            raise ValueError(
                "unresolved_upper_bound must be unique and follow terminal lambda order"
            )
        if any(terminal[name] != self.config.maximum_lambda for name in unresolved_upper_bound):
            raise ValueError(
                "every unresolved upper-bound lambda must equal the exact configured maximum"
            )
        if unresolved_upper_bound and terminal_raw_max_log_step == 0.0:
            raise ValueError(
                "unresolved upper pressure requires positive fresh convergence evidence"
            )
        if self.converged and unresolved_upper_bound:
            raise ValueError("converged EFS results require an empty unresolved upper bound")
        if (
            self.converged
            and terminal_raw_max_log_step > self.config.tolerance
            and self.convergence_reason not in ("practical_plateau", "stationary")
        ):
            raise ValueError(
                "a converged non-stationary EFS result requires fresh convergence evidence"
            )
        self._validate_endgame_fields(terminal)
        if self.convergence_reason == "fixed_only" and terminal_raw_max_log_step != 0.0:
            raise ValueError("fixed-only EFS evidence must have zero fresh convergence residual")
        if self.convergence_reason == "lambda_cap_unresolved" and not unresolved_upper_bound:
            raise ValueError("lambda_cap_unresolved requires a nonempty unresolved upper bound")
        if unresolved_upper_bound and self.convergence_reason not in {
            "lambda_cap_unresolved",
            "coefficient_not_converged",
            "endpoint_revalidation_failed",
            # The Newton endgame's own failures end the fit where they stand;
            # a Fellner--Schall nomination at the cap is then diagnostic.
            "max_iterations",
            "objective_rejected",
            "gradient_unresolved",
        }:
            raise ValueError("unresolved upper pressure must control the terminal reason")
        if not fits or any(not isinstance(fit, DenseSolverResult) for fit in fits):
            raise ValueError("coefficient_fits must contain at least one dense solver result")
        newton_count = sum(item.stage == "newton" for item in history)
        bfgs_count = sum(item.step_source == "bfgs" for item in history)
        for name, expected in (
            ("newton_iterations", newton_count),
            ("bfgs_fallback_iterations", bfgs_count),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"{name} must count the smoothing history's iterations")
        plan_identifiers = {fit.family_likelihood_plan_identifier for fit in fits}
        if len(plan_identifiers) != 1:
            raise ValueError("EFS coefficient fits must share one family likelihood plan")
        if (
            isinstance(self.terminal_fit_index, bool)
            or not isinstance(self.terminal_fit_index, int)
            or not 0 <= self.terminal_fit_index < len(fits)
        ):
            raise ValueError("terminal_fit_index lies outside coefficient_fits")
        expected_fit_index = 0
        next_trial_fit_index = 1
        expected_lambdas: Mapping[str, float] = initial
        expected_objective = self.initial_objective
        active_face_components: set[str] = set()

        def coefficient_face_names(index: int) -> tuple[str, ...]:
            face = fits[index].coefficient_face
            return () if face is None else face.component_names

        def assessment_shared_signature(fit: DenseSolverResult) -> tuple[object, ...]:
            rank = fit.terminal_rank if fit.coefficient_face is None else fit.terminal_reduced_rank
            if rank is None:
                raise ValueError("endpoint assessment face requires reduced-rank provenance")
            return (
                fit.family_likelihood_plan_identifier,
                fit.execution_backend_identifier,
                fit.terminal_curvature.requested_source,
                fit.terminal_curvature.actual_source,
                rank.policy_version,
                fit.resolved_chunk_size,
            )

        def validate_sole_capped_component(
            item: DistributionalEFSIteration,
            *,
            finite_face: tuple[str, ...],
            assessed_name: str,
        ) -> None:
            capped_outside = tuple(
                name
                for name in terminal
                if name not in finite_face
                and item.lambdas_before[name] == self.config.maximum_lambda
            )
            if capped_outside != (assessed_name,):
                raise ValueError("scalar face evidence requires one sole capped component")

        def validate_joint_endpoint_assessment(
            item: DistributionalEFSIteration,
            *,
            source_face: tuple[str, ...],
            accepted_face: tuple[str, ...],
            event_names: tuple[str, ...],
        ) -> None:
            direction = item.endpoint_direction_evidence
            is_retraction = bool(item.deactivated_face_components)
            expected_fit_count = 2 if is_retraction else 1
            if len(item.coefficient_fit_indices) != expected_fit_count:
                raise ValueError("joint endpoint assessment requires one strict common fit")
            endpoint_index = item.coefficient_fit_indices[0]
            endpoint_fit = fits[endpoint_index]
            tolerance = item.coefficient_tolerances[0]
            if (
                tolerance > 1.0e-12
                or endpoint_fit.config.tolerance != tolerance
                or endpoint_fit.config.coefficient_curvature != "observed"
                or endpoint_fit.terminal_curvature.requested_source != "observed"
            ):
                raise ValueError("joint endpoint assessment changed its strict fit policy")

            source_signature = assessment_shared_signature(fits[item.source_fit_index])
            endpoint_signature = assessment_shared_signature(endpoint_fit)
            if (
                endpoint_signature[0] != source_signature[0]
                or endpoint_signature[1] != source_signature[1]
                or endpoint_signature[4] != source_signature[4]
                or endpoint_signature[5] != source_signature[5]
            ):
                raise ValueError("joint endpoint fit changed shared solver provenance")

            if item.activated_face_components:
                expected_endpoint_face = accepted_face
            elif item.revalidated_face_components or item.deactivated_face_components:
                expected_endpoint_face = source_face
            else:
                selected = {*source_face, *event_names}
                expected_endpoint_face = tuple(name for name in terminal if name in selected)
            if coefficient_face_names(endpoint_index) != expected_endpoint_face:
                raise ValueError("joint endpoint assessment used the wrong exact coefficient face")
            if item.revalidated_face_components or item.deactivated_face_components:
                source_fit = fits[item.source_fit_index]
                if event_names != source_face:
                    raise ValueError(
                        "joint endpoint revalidation has incomplete component coverage"
                    )
                source_geometry = source_fit.coefficient_face
                endpoint_geometry = endpoint_fit.coefficient_face
                if (
                    source_geometry is None
                    or endpoint_geometry is None
                    or not _assessment_face_geometry_matches(
                        source_geometry,
                        endpoint_geometry,
                    )
                ):
                    raise ValueError("joint endpoint revalidation changed its exact-face geometry")
            else:
                added_names = tuple(
                    name for name in expected_endpoint_face if name not in source_face
                )
                if added_names != event_names:
                    raise ValueError(
                        "joint endpoint face difference has incomplete component coverage"
                    )
            if any(item.lambdas_before[name] != self.config.maximum_lambda for name in event_names):
                raise ValueError("joint endpoint components must be assessed at the exact cap")

            reason = item.endpoint_assessment_failure_reason
            if direction is not None:
                if not isinstance(direction, JointEndpointDirectionEvidence):
                    raise TypeError("joint endpoint assessment requires joint direction evidence")
                if (
                    direction.component_names != event_names
                    or direction.fit_indices != (endpoint_index,)
                    or direction.coefficient_tolerance != tolerance
                ):
                    raise ValueError(
                        "joint endpoint evidence does not cover its one strict common fit"
                    )
            elif reason not in {
                "joint_endpoint_not_converged",
                "joint_endpoint_not_stationary",
                "joint_analytic_unavailable",
                "joint_objective_rejected",
            }:
                raise ValueError("joint endpoint refusal requires a typed joint failure reason")

            if reason == "joint_endpoint_not_converged":
                if endpoint_fit.converged:
                    raise ValueError("joint endpoint convergence failure disagrees with its fit")
                return
            if reason == "joint_endpoint_not_stationary":
                if not endpoint_fit.converged or _assessment_is_numerically_stationary(
                    endpoint_fit,
                    tolerance,
                ):
                    raise ValueError("joint endpoint stationarity failure disagrees with its fit")
                return
            if not endpoint_fit.converged or not _assessment_is_numerically_stationary(
                endpoint_fit,
                tolerance,
            ):
                raise ValueError("joint endpoint evidence requires a stationary common fit")

            calculated_objective, penalty_logdet = _assessment_exact_face_objective(endpoint_fit)
            current_ceiling = item.objective_before + self.config.objective_tolerance * (
                1.0 + abs(item.objective_before)
            )
            if reason == "joint_objective_rejected":
                if calculated_objective <= current_ceiling:
                    raise ValueError("joint objective failure disagrees with its common fit")
                return
            if reason == "joint_analytic_unavailable":
                return
            assert isinstance(direction, JointEndpointDirectionEvidence)
            if (
                endpoint_fit.terminal_curvature.actual_source != "observed"
                or endpoint_fit.terminal_curvature.fallback_count != 0
            ):
                raise ValueError(
                    "joint endpoint evidence requires unfallbacked observed-curvature authority"
                )
            objective_error = _assessment_scalar_error_bound(
                calculated_objective,
                direction.endpoint_objective,
                width=endpoint_fit.penalty.shape[0],
                calculation_scale=abs(penalty_logdet),
            )
            if abs(calculated_objective - direction.endpoint_objective) > objective_error:
                raise ValueError("joint endpoint directions do not share the fitted objective")
            if direction.endpoint_objective > current_ceiling:
                raise ValueError("joint endpoint evidence does not improve its source state")
            if item.activated_face_components or item.revalidated_face_components:
                if any(
                    component_direction.lower_bound <= 0.0
                    for _name, component_direction in direction.component_directions
                ):
                    raise ValueError(
                        "joint exact-face activation requires every positive direction bound"
                    )

        def validate_joint_rollback(
            item: DistributionalEFSIteration,
            *,
            source_face: tuple[str, ...],
            accepted_face: tuple[str, ...],
            has_common_assessment: bool,
        ) -> None:
            if item.deactivated_face_components != source_face:
                raise ValueError("joint endpoint rollback has incomplete component coverage")
            if any(item.lambdas_before[name] != self.config.maximum_lambda for name in source_face):
                raise ValueError("joint endpoint rollback components must remain at the exact cap")
            indices = item.coefficient_fit_indices
            expected_count = 2 if has_common_assessment else 1
            if len(indices) != expected_count:
                raise ValueError("joint endpoint rollback retained the wrong assessment fits")
            rollback_index = indices[-1]
            rollback_fit = fits[rollback_index]
            rollback_tolerance = item.coefficient_tolerances[-1]
            if (
                item.accepted_fit_index != rollback_index
                or coefficient_face_names(rollback_index) != accepted_face
                or rollback_tolerance > 1.0e-12
                or rollback_fit.config.tolerance != rollback_tolerance
                or rollback_fit.config.coefficient_curvature != "observed"
                or rollback_fit.terminal_curvature.requested_source != "observed"
                or rollback_fit.terminal_curvature.actual_source != "observed"
                or rollback_fit.terminal_curvature.fallback_count != 0
                or rollback_fit.config.max_iterations < 150
                or rollback_fit.config.newton_decrement_tolerance is not None
                or not rollback_fit.converged
                or not _assessment_is_numerically_stationary(
                    rollback_fit,
                    rollback_tolerance,
                )
                or item.accepted_curvature != rollback_fit.terminal_curvature
            ):
                raise ValueError("joint endpoint rollback used the wrong finite fit")

            source_fit = fits[item.source_fit_index]
            source_geometry = source_fit.coefficient_face
            if source_geometry is None or source_geometry.component_names != source_face:
                raise ValueError("joint endpoint rollback lost its source-face geometry")
            reference_fit = fits[indices[0]] if has_common_assessment else source_fit
            if has_common_assessment and rollback_fit.config != reference_fit.config:
                raise ValueError("joint endpoint rollback changed its strict authority config")
            reference_geometry = reference_fit.coefficient_face
            if (
                reference_geometry is None
                or not _assessment_face_geometry_matches(source_geometry, reference_geometry)
                or not np.array_equal(reference_fit.penalty, source_fit.penalty)
            ):
                raise ValueError("joint endpoint rollback retained the wrong common fit")
            reference_signature = assessment_shared_signature(reference_fit)
            rollback_signature = assessment_shared_signature(rollback_fit)
            if any(
                rollback_signature[index] != reference_signature[index] for index in (0, 1, 4, 5)
            ):
                raise ValueError("joint endpoint rollback changed shared solver provenance")
            if not _assessment_complete_face_penalty_matches(
                face=source_geometry,
                endpoint_penalty=reference_fit.penalty,
                finite_penalty=rollback_fit.penalty,
            ):
                raise ValueError("joint endpoint rollback has invalid finite penalty geometry")
            rollback_fingerprint = item.joint_rollback_penalty_fingerprint
            if rollback_fingerprint is not None and rollback_fingerprint != (
                _dense_penalty_fingerprint(rollback_fit.penalty)
            ):
                raise ValueError(
                    "joint endpoint rollback penalty fingerprint disagrees with terminal penalty"
                )

            calculated_objective, penalty_logdet = _assessment_finite_objective(rollback_fit)
            objective_error = _assessment_scalar_error_bound(
                calculated_objective,
                item.objective_after,
                width=rollback_fit.penalty.shape[0],
                calculation_scale=abs(penalty_logdet),
            )
            if abs(calculated_objective - item.objective_after) > objective_error:
                raise ValueError("joint endpoint rollback objective disagrees with its finite fit")

        def validate_endpoint_assessment(
            item: DistributionalEFSIteration,
            *,
            source_face: tuple[str, ...],
            accepted_face: tuple[str, ...],
        ) -> None:
            direction = item.endpoint_direction_evidence
            joint_event_names = (
                item.activated_face_components
                or item.revalidated_face_components
                or item.refused_face_components
                or item.deactivated_face_components
            )
            if len(joint_event_names) > 1:
                if item.deactivated_face_components:
                    has_common_assessment = bool(
                        direction is not None or item.endpoint_assessment_failure_reason is not None
                    )
                    if has_common_assessment:
                        validate_joint_endpoint_assessment(
                            item,
                            source_face=source_face,
                            accepted_face=accepted_face,
                            event_names=joint_event_names,
                        )
                    validate_joint_rollback(
                        item,
                        source_face=source_face,
                        accepted_face=accepted_face,
                        has_common_assessment=has_common_assessment,
                    )
                    return
                validate_joint_endpoint_assessment(
                    item,
                    source_face=source_face,
                    accepted_face=accepted_face,
                    event_names=joint_event_names,
                )
                return
            if direction is None:
                if not item.refused_face_components:
                    return
                reason = item.endpoint_assessment_failure_reason
                if reason is None:
                    raise ValueError(
                        "an endpoint refusal without evidence requires a failure reason"
                    )
                assessment_fits = tuple(fits[index] for index in item.coefficient_fit_indices)
                expected_count = 1 if reason in {"cap_not_converged", "cap_not_stationary"} else 2
                if len(assessment_fits) != expected_count:
                    raise ValueError("endpoint assessment failure reason disagrees with its fits")
                if any(
                    tolerance != item.coefficient_tolerances[0] or fit.config.tolerance != tolerance
                    for fit, tolerance in zip(
                        assessment_fits,
                        item.coefficient_tolerances,
                        strict=True,
                    )
                ):
                    raise ValueError("endpoint assessment failure changed its tight tolerance")
                cap_fit = assessment_fits[0]
                if coefficient_face_names(item.coefficient_fit_indices[0]) != source_face:
                    raise ValueError("endpoint assessment cap used the wrong coefficient face")
                if reason == "cap_not_converged":
                    if cap_fit.converged:
                        raise ValueError("endpoint cap failure reason disagrees with its fit")
                    return
                if reason == "cap_not_stationary":
                    if not cap_fit.converged or _assessment_is_numerically_stationary(
                        cap_fit,
                        item.coefficient_tolerances[0],
                    ):
                        raise ValueError(
                            "endpoint cap stationarity failure reason disagrees with its fit"
                        )
                    return
                endpoint_fit = assessment_fits[1]
                endpoint_names = {*source_face, item.refused_face_components[0]}
                expected_endpoint_face = tuple(name for name in terminal if name in endpoint_names)
                if (
                    coefficient_face_names(item.coefficient_fit_indices[1])
                    != expected_endpoint_face
                ):
                    raise ValueError("endpoint assessment used the wrong exact coefficient face")
                tolerance = item.coefficient_tolerances[0]
                cap_stationary = _assessment_is_numerically_stationary(
                    cap_fit,
                    tolerance,
                )
                if not cap_stationary:
                    validate_sole_capped_component(
                        item,
                        finite_face=source_face,
                        assessed_name=item.refused_face_components[0],
                    )
                if reason == "endpoint_not_converged":
                    if not cap_fit.converged or endpoint_fit.converged:
                        raise ValueError("endpoint fit failure reason disagrees with its fits")
                    return
                if reason == "endpoint_not_stationary":
                    if not cap_fit.converged or not cap_stationary:
                        raise ValueError(
                            "endpoint stationarity failure requires a stationary endpoint cap"
                        )
                    if not endpoint_fit.converged or _assessment_is_numerically_stationary(
                        endpoint_fit,
                        tolerance,
                    ):
                        raise ValueError(
                            "endpoint stationarity failure reason disagrees with its fit"
                        )
                    return
                if not cap_fit.converged or not endpoint_fit.converged:
                    raise ValueError("endpoint analytic failure requires converged assessment fits")
                provenance_changed = assessment_shared_signature(endpoint_fit) != (
                    assessment_shared_signature(cap_fit)
                )
                if (reason == "provenance_changed") != provenance_changed:
                    raise ValueError("endpoint provenance failure reason disagrees with its fits")
                return
            if not isinstance(direction, EndpointDirectionEvidence):
                raise TypeError("single-component endpoint evidence has the wrong type")
            indices = direction.fit_indices
            indices_match_iteration = indices == item.coefficient_fit_indices
            if item.deactivated_face_components:
                indices_match_iteration = item.coefficient_fit_indices[: len(indices)] == indices
            if (
                len(indices) != 2
                or not indices_match_iteration
                or any(index not in item.coefficient_fit_indices for index in indices)
            ):
                raise ValueError("endpoint evidence must index its assessment coefficient fits")
            assessment_fits = tuple(fits[index] for index in indices)
            if any(not fit.converged for fit in assessment_fits):
                raise ValueError("endpoint assessment fits must converge")
            if any(
                fit.config.tolerance != direction.coefficient_tolerance for fit in assessment_fits
            ):
                raise ValueError("endpoint assessment fits must retain their tight tolerance")
            if any(
                fit.config.coefficient_curvature != "observed"
                or fit.terminal_curvature.requested_source != "observed"
                or fit.terminal_curvature.actual_source != "observed"
                or fit.terminal_curvature.fallback_count != 0
                for fit in assessment_fits
            ):
                raise ValueError(
                    "endpoint analytic evidence requires unfallbacked observed-curvature authority"
                )
            cap_fit = assessment_fits[0]
            endpoint_fit = assessment_fits[-1]
            finite_face = coefficient_face_names(indices[0])
            if assessment_shared_signature(endpoint_fit) != assessment_shared_signature(cap_fit):
                raise ValueError("endpoint fit changed shared solver provenance")
            if item.activated_face_components:
                expected_finite_face = source_face
                expected_endpoint_face = accepted_face
            elif item.revalidated_face_components:
                assessed_name = item.revalidated_face_components[0]
                expected_finite_face = tuple(name for name in source_face if name != assessed_name)
                expected_endpoint_face = source_face
            elif item.refused_face_components:
                assessed_name = item.refused_face_components[0]
                expected_finite_face = source_face
                endpoint_names = {*source_face, assessed_name}
                expected_endpoint_face = tuple(name for name in terminal if name in endpoint_names)
            else:
                if not set(finite_face).issubset(source_face) or len(finite_face) != max(
                    len(source_face) - 1,
                    0,
                ):
                    raise ValueError("endpoint retraction finite face is inconsistent")
                expected_finite_face = finite_face
                expected_endpoint_face = source_face
            if finite_face != expected_finite_face:
                raise ValueError("endpoint finite reference used the wrong coefficient face")
            if coefficient_face_names(indices[-1]) != expected_endpoint_face:
                raise ValueError("endpoint assessment used the wrong exact coefficient face")
            assessed_components = tuple(
                name for name in expected_endpoint_face if name not in expected_finite_face
            )
            if len(assessed_components) != 1:
                raise ValueError("endpoint assessment must add exactly one face component")
            assessed_name = assessed_components[0]
            finite_rank = (
                0 if cap_fit.coefficient_face is None else cap_fit.coefficient_face.constraint_rank
            )
            endpoint_face = endpoint_fit.coefficient_face
            if endpoint_face is None:
                raise ValueError("endpoint assessment did not fit an exact coefficient face")
            tolerance = direction.coefficient_tolerance
            assert tolerance is not None
            if not _assessment_is_numerically_stationary(endpoint_fit, tolerance):
                raise ValueError("endpoint analytic evidence requires a stationary endpoint fit")
            direct_nonstationary_cap = _assessment_retained_kkt_ratio(cap_fit) > tolerance
            cap_numerically_stationary = _assessment_is_numerically_stationary(
                cap_fit,
                tolerance,
            )
            if item.activated_face_components or item.revalidated_face_components:
                validate_sole_capped_component(
                    item,
                    finite_face=expected_finite_face,
                    assessed_name=assessed_name,
                )
            if direct_nonstationary_cap:
                endpoint_rank = endpoint_fit.terminal_reduced_rank
                if endpoint_rank is None or endpoint_rank.rank != endpoint_face.reduced_width:
                    raise ValueError(
                        "nonstationary-cap exact-face authority requires full retained rank"
                    )
            if item.revalidated_face_components and not np.array_equal(
                endpoint_fit.coefficients,
                fits[item.source_fit_index].coefficients,
            ):
                source_coefficients = fits[item.source_fit_index].coefficients
                movement_bound = _assessment_coefficient_refit_bound(
                    source_coefficients,
                    endpoint_fit.coefficients,
                    tolerance=tolerance,
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    movement = float(
                        np.max(
                            np.abs(endpoint_fit.coefficients - source_coefficients),
                            initial=0.0,
                        )
                    )
                if (
                    cap_numerically_stationary
                    or not direct_nonstationary_cap
                    or movement_bound is None
                    or not math.isfinite(movement)
                    or movement > movement_bound
                ):
                    raise ValueError("endpoint revalidation changed the canonical endpoint state")
            selected_rank = endpoint_face.constraint_rank - finite_rank
            selected_penalty = _assessment_penalty_direction_matches(
                cap_fit=cap_fit,
                endpoint_fit=endpoint_fit,
                cap_lambda=float(item.lambdas_before[assessed_name]),
                selected_rank=selected_rank,
            )
            if selected_penalty is None:
                raise ValueError("endpoint assessment does not match its fitted penalty")
            endpoint_term = _assessment_unpenalized_logdet_term(endpoint_fit)
            if item.activated_face_components or item.revalidated_face_components:
                cap_tau = 1.0 / float(item.lambdas_before[assessed_name])
                cap_term = _assessment_unpenalized_logdet_term(cap_fit)
                cap_penalty_logdet = selected_penalty.log_pdet - selected_penalty.rank * math.log(
                    cap_tau
                )
                cap_difference = cap_term - endpoint_term - 0.5 * cap_penalty_logdet
                cap_objective = direction.endpoint_objective + cap_difference
                cap_roundoff = _assessment_scalar_error_bound(
                    cap_term,
                    endpoint_term,
                    width=cap_fit.penalty.shape[0],
                    calculation_scale=abs(cap_penalty_logdet),
                )
                cap_ceiling = (
                    cap_objective
                    + self.config.objective_tolerance * (1.0 + abs(cap_objective))
                    + cap_roundoff
                )
                current_ceiling = item.objective_before + self.config.objective_tolerance * (
                    1.0 + abs(item.objective_before)
                )
                if direction.endpoint_objective > current_ceiling or (
                    cap_numerically_stationary and direction.endpoint_objective > cap_ceiling
                ):
                    raise ValueError(
                        "endpoint assessment does not improve its tightened finite reference"
                    )
                if direction.lower_bound <= 0.0:
                    raise ValueError(
                        "exact-face authority requires a strictly positive direction bound"
                    )
            if not direction._derived_authority_matches():
                raise ValueError("endpoint assessment direction is internally inconsistent")

        final_revalidations: dict[str, EndpointDirectionEvidence] = {}

        for expected_iteration, item in enumerate(history, start=1):
            if not isinstance(item, DistributionalEFSIteration):
                raise TypeError("history must contain DistributionalEFSIteration values")
            if item.iteration != expected_iteration:
                raise ValueError("EFS history iteration numbers must be contiguous")
            if item.source_fit_index != expected_fit_index:
                raise ValueError("EFS history source fit does not match accepted state")
            expected_source_face = tuple(
                name for name in terminal if name in active_face_components
            )
            if coefficient_face_names(item.source_fit_index) != expected_source_face:
                raise ValueError("EFS history source coefficient face is not authoritative")
            if any(index >= len(fits) for index in item.coefficient_fit_indices):
                raise ValueError("EFS history references a coefficient fit outside the result")
            expected_trial_indices = tuple(
                range(
                    next_trial_fit_index,
                    next_trial_fit_index + len(item.coefficient_fit_indices),
                )
            )
            if item.coefficient_fit_indices != expected_trial_indices:
                raise ValueError("EFS coefficient-fit chronology must be contiguous")
            next_trial_fit_index += len(item.coefficient_fit_indices)
            if dict(item.lambdas_before) != dict(expected_lambdas):
                raise ValueError("EFS history lambda states are not contiguous")
            if item.objective_before != expected_objective:
                raise ValueError("EFS history objectives are not contiguous")
            refused_names = set(item.refused_face_components)
            refusal_has_upper_pressure = (
                bool(refused_names & unresolved_set)
                if len(refused_names) > 1
                else refused_names.issubset(unresolved_set)
            )
            if item.refused_face_components and (
                expected_iteration != len(history)
                or self.converged
                or self.convergence_reason != "lambda_cap_unresolved"
                or not refusal_has_upper_pressure
            ):
                raise ValueError(
                    "a refused endpoint assessment must be the terminal lambda-cap failure"
                )
            expected_lambdas = item.lambdas_after
            expected_objective = item.objective_after
            if item.accepted:
                assert item.accepted_fit_index is not None
                expected_fit_index = item.accepted_fit_index
            for name in item.revalidated_face_components:
                if name not in active_face_components:
                    raise ValueError("an exact face cannot revalidate an inactive component")
            for name in item.activated_face_components:
                if name in active_face_components:
                    raise ValueError("an exact face cannot activate the same component twice")
                active_face_components.add(name)
            for name in item.deactivated_face_components:
                if name not in active_face_components:
                    raise ValueError("an exact face cannot deactivate an inactive component")
                active_face_components.remove(name)
            expected_accepted_face = tuple(
                name for name in terminal if name in active_face_components
            )
            if (
                item.accepted
                and coefficient_face_names(expected_fit_index) != expected_accepted_face
            ):
                raise ValueError("EFS accepted coefficient face does not match its transition")
            direction = item.endpoint_direction_evidence
            if item.activated_face_components or item.revalidated_face_components:
                assert direction is not None
                if (
                    direction.fit_indices != item.coefficient_fit_indices
                    or direction.fit_indices[-1] != item.accepted_fit_index
                    or direction.endpoint_objective != item.objective_after
                ):
                    raise ValueError(
                        "endpoint direction evidence does not authenticate its accepted fits"
                    )
            elif item.refused_face_components and direction is not None:
                if direction.fit_indices != item.coefficient_fit_indices:
                    raise ValueError(
                        "endpoint refusal evidence does not authenticate its assessment fits"
                    )
            elif item.deactivated_face_components and direction is not None:
                if item.coefficient_fit_indices[: len(direction.fit_indices)] != (
                    direction.fit_indices
                ):
                    raise ValueError(
                        "endpoint retraction evidence does not authenticate its assessment fits"
                    )
            validate_endpoint_assessment(
                item,
                source_face=expected_source_face,
                accepted_face=expected_accepted_face,
            )
            for name, scalar_direction in _revalidated_endpoint_directions(item):
                final_revalidations[name] = scalar_direction
        if next_trial_fit_index != len(fits):
            raise ValueError("EFS coefficient-fit chronology cannot contain orphan fits")
        deactivation_indices = tuple(
            index for index, item in enumerate(history) if item.deactivated_face_components
        )
        if self.convergence_reason == "endpoint_revalidation_failed":
            if deactivation_indices != (len(history) - 1,):
                raise ValueError(
                    "endpoint_revalidation_failed requires a terminal exact-face deactivation"
                )
        elif deactivation_indices:
            raise ValueError(
                "a terminal exact-face deactivation requires endpoint_revalidation_failed"
            )
        if expected_fit_index != self.terminal_fit_index:
            raise ValueError("terminal_fit_index does not match the accepted EFS history")
        if dict(expected_lambdas) != dict(terminal) or expected_objective != self.objective:
            raise ValueError("terminal EFS state does not match the smoothing history")
        if self.convergence_reason == "practical_plateau":
            if not self.config.practical_convergence:
                raise ValueError("practical_plateau requires practical convergence to be enabled")
            if terminal_raw_max_log_step <= self.config.tolerance:
                raise ValueError("practical_plateau must remain distinct from strict stationarity")
            practical_tail = history[-self.config.plateau_iterations :]
            if len(practical_tail) != self.config.plateau_iterations:
                raise ValueError("practical_plateau requires a complete plateau window")
            for item in practical_tail:
                if (
                    not item.accepted
                    or item.activated_face_components
                    or item.deactivated_face_components
                    or item.revalidated_face_components
                    or item.refused_face_components
                    or item.objective_relative_change > self.config.plateau_tolerance
                ):
                    raise ValueError("practical_plateau history does not satisfy its plateau gate")
                assert item.accepted_fit_index is not None
                parameter_change = _maximum_relative_natural_parameter_change(
                    fits[item.source_fit_index].theta,
                    fits[item.accepted_fit_index].theta,
                )
                if parameter_change > self.config.practical_parameter_tolerance:
                    raise ValueError(
                        "practical_plateau fitted parameters exceed the configured tolerance"
                    )
            accepted_steps = [item.max_accepted_log_step for item in practical_tail]
            if any(later > earlier for earlier, later in zip(accepted_steps, accepted_steps[1:])):
                raise ValueError("practical_plateau history does not satisfy its plateau gate")
        terminal_face = fits[self.terminal_fit_index].coefficient_face
        if self.convergence_reason == "practical_plateau" and terminal_face is not None:
            raise ValueError("practical_plateau cannot terminate on an exact coefficient face")
        terminal_face_names = () if terminal_face is None else terminal_face.component_names
        ordered_active = tuple(name for name in terminal if name in active_face_components)
        if ordered_active != terminal_face_names:
            raise ValueError("terminal coefficient face does not match EFS face chronology")
        if endpoint_directions:
            if tuple(endpoint_directions) != terminal_face_names:
                raise ValueError("terminal endpoint directions must cover the terminal exact face")
            if any(
                evidence.decision != "endpoint" or evidence.endpoint_objective != self.objective
                for evidence in endpoint_directions.values()
            ):
                raise ValueError(
                    "terminal endpoint directions must resolve the terminal objective toward infinity"
                )
            ordered_revalidations = tuple(
                name for name in terminal_face_names if name in final_revalidations
            )
            if ordered_revalidations != terminal_face_names or dict(endpoint_directions) != {
                name: final_revalidations[name] for name in terminal_face_names
            }:
                raise ValueError(
                    "terminal endpoint directions must match the final face revalidation"
                )
        if self.converged and terminal_face is not None and not endpoint_directions:
            raise ValueError("a converged exact face requires fresh terminal endpoint directions")
        if terminal_face is None and endpoint_directions:
            raise ValueError("a finite terminal fit cannot retain exact-face endpoint directions")
        gradient_names = () if self.terminal_gradient is None else tuple(self.terminal_gradient)
        if any(name in terminal_face_names for name in gradient_names):
            raise ValueError("the terminal gradient cannot name an exact-face component")
        object.__setattr__(self, "initial_lambdas", initial)
        object.__setattr__(self, "lambdas", terminal)
        object.__setattr__(self, "terminal_raw_max_log_step", terminal_raw_max_log_step)
        object.__setattr__(self, "unresolved_upper_bound", unresolved_upper_bound)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "coefficient_fits", fits)
        object.__setattr__(self, "terminal_endpoint_directions", endpoint_directions)

    def _validate_endgame_fields(self, terminal: Mapping[str, float]) -> None:
        """Validate the Newton endgame's terminal record against the reason."""

        gradient = self.terminal_gradient
        certificate = self.terminal_gradient_certificate
        if (gradient is None) != (certificate is None):
            raise ValueError("terminal gradient and certificate must be present together")
        names: tuple[str, ...] = ()
        if gradient is not None:
            frozen = _frozen_float_mapping(gradient, name="terminal_gradient")
            names = tuple(frozen)
            if (
                not names
                or any(name not in terminal for name in names)
                or tuple(name for name in terminal if name in names) != names
            ):
                raise ValueError("terminal gradient names must follow terminal lambda order")
            assert certificate is not None
            frozen_certificate = _frozen_float_mapping(
                certificate, name="terminal_gradient_certificate", nonnegative=True
            )
            if tuple(frozen_certificate) != names:
                raise ValueError("terminal gradient certificate must cover the gradient names")
            object.__setattr__(self, "terminal_gradient", frozen)
            object.__setattr__(self, "terminal_gradient_certificate", frozen_certificate)
        norm = self.terminal_projected_gradient_norm
        if norm is not None:
            norm = _finite_nonnegative(norm, name="terminal_projected_gradient_norm")
            object.__setattr__(self, "terminal_projected_gradient_norm", norm)
        if (self.smoothing_hessian is None) != (self.smoothing_hessian_certificate is None):
            raise ValueError("smoothing Hessian and certificate must be present together")
        if self.smoothing_hessian is not None:
            if gradient is None:
                raise ValueError("a smoothing Hessian requires the terminal gradient's names")
            hessian = np.array(self.smoothing_hessian, dtype=np.float64, copy=True)
            hessian_certificate = np.array(
                self.smoothing_hessian_certificate, dtype=np.float64, copy=True
            )
            shape = (len(names), len(names))
            if hessian.shape != shape or hessian_certificate.shape != shape:
                raise ValueError("smoothing Hessian must be square over the terminal gradient")
            if not np.all(np.isfinite(hessian)) or not np.array_equal(hessian, hessian.T):
                raise ValueError("smoothing Hessian must be finite and symmetric")
            if not np.all(np.isfinite(hessian_certificate)) or np.any(hessian_certificate < 0.0):
                raise ValueError("smoothing Hessian certificate must be finite and non-negative")
            hessian.setflags(write=False)
            hessian_certificate.setflags(write=False)
            object.__setattr__(self, "smoothing_hessian", hessian)
            object.__setattr__(self, "smoothing_hessian_certificate", hessian_certificate)
        if self.convergence_reason == "stationary":
            if gradient is None or norm is None:
                raise ValueError("a stationary result requires its terminal projected gradient")
            if norm > self.stationarity_bar:
                raise ValueError(
                    "a stationary result requires its projected gradient within tolerance"
                )
        if self.convergence_reason == "gradient_unresolved" and gradient is None:
            raise ValueError("gradient_unresolved requires the terminal gradient and certificate")
        beyond = tuple(self.beyond_cap_components)
        beyond_set = set(beyond)
        if (
            any(not isinstance(name, str) or name not in terminal for name in beyond)
            or len(beyond_set) != len(beyond)
            or tuple(name for name in terminal if name in beyond_set) != beyond
        ):
            raise ValueError(
                "beyond_cap_components must be unique and follow terminal lambda order"
            )
        for name, value in terminal.items():
            if name in beyond_set:
                maximum_lambda_conditioning = self.config.maximum_lambda_conditioning
                assert maximum_lambda_conditioning is not None
                if not self.config.maximum_lambda < value <= maximum_lambda_conditioning:
                    raise ValueError(
                        "a component released beyond the cap must lie above maximum_lambda and "
                        "within maximum_lambda_conditioning"
                    )
            elif value > self.config.maximum_lambda:
                raise ValueError("a lambda above maximum_lambda must be a released component")
        object.__setattr__(self, "beyond_cap_components", beyond)

    @property
    def terminal_evidence_fresh(self) -> bool:
        """Whether the terminal state carries its own convergence authority.

        The Fellner--Schall residual within tolerance, or -- for a stationary
        result -- the projected exact gradient within its bar with every
        component's certificate below that bar.
        """

        if self.convergence_reason == "stationary":
            norm = self.terminal_projected_gradient_norm
            certificate = self.terminal_gradient_certificate
            return bool(
                norm is not None
                and norm <= self.stationarity_bar
                and certificate is not None
                and all(value <= self.stationarity_bar for value in certificate.values())
            )
        return self.terminal_raw_max_log_step <= self.config.tolerance

    @property
    def terminal_fit(self) -> DenseSolverResult:
        return self.coefficient_fits[self.terminal_fit_index]

    @property
    def terminal_convergence_max_log_residual(self) -> float:
        """Fresh maximum residual used by outer convergence."""

        return self.terminal_raw_max_log_step

    @property
    def coefficient_converged(self) -> bool:
        return self.terminal_fit.converged

    @property
    def fallback_count(self) -> int:
        return sum(fit.terminal_curvature.fallback_count for fit in self.coefficient_fits)

    @property
    def accelerated_trial_count(self) -> int:
        return sum(item.acceleration_outcome in ("accepted", "rejected") for item in self.history)

    @property
    def accelerated_accept_count(self) -> int:
        return sum(item.acceleration_outcome == "accepted" for item in self.history)

    @property
    def raw_fallback_count(self) -> int:
        return sum(item.acceleration_outcome == "rejected" for item in self.history)

    @property
    def matched_certified(self) -> bool:
        face = self.terminal_fit.coefficient_face
        face_names = () if face is None else face.component_names
        direction_evidence = tuple(
            item.endpoint_direction_evidence
            for item in self.history
            if item.endpoint_direction_evidence is not None
        )
        indexed_assessments_valid = all(
            evidence.fit_indices
            and all(
                index < len(self.coefficient_fits)
                and self.coefficient_fits[index].converged
                and self.coefficient_fits[index].config.tolerance == evidence.coefficient_tolerance
                for index in evidence.fit_indices
            )
            for evidence in direction_evidence
        )
        final_revalidations = {
            name: direction
            for item in self.history
            for name, direction in _revalidated_endpoint_directions(item)
        }
        return bool(
            self.converged
            and self.convergence_reason != "practical_plateau"
            and self.coefficient_converged
            and face is None
            and self.terminal_evidence_fresh
            and not self.unresolved_upper_bound
            and not any(
                item.refused_face_components
                or item.deactivated_face_components
                or item.endpoint_assessment_failure_reason is not None
                for item in self.history
            )
            and self.fallback_count == 0
            # The cap never certifies under the Newton authority: a stationary
            # outcome voids the Fellner--Schall nomination that would have
            # produced an endpoint decision, so a component resting at the configured
            # maximum is stationary at a box bound, not a certified infinity.  The
            # Fellner--Schall path assesses every cap-resting nomination itself and keeps
            # its contract for a finite optimum that sits exactly at the cap.
            and not (
                self.convergence_reason == "stationary"
                and any(
                    value == self.config.maximum_lambda
                    for name, value in self.lambdas.items()
                    if name not in face_names
                )
            )
            and tuple(self.terminal_endpoint_directions) == face_names
            and indexed_assessments_valid
            and dict(self.terminal_endpoint_directions) == final_revalidations
            and all(
                evidence._derived_authority_matches()
                and evidence.authority_identifier == "analytic-observed-curvature-direction/v1"
                for evidence in self.terminal_endpoint_directions.values()
            )
            and all(
                fit.terminal_curvature.actual_source == fit.terminal_curvature.requested_source
                for fit in self.coefficient_fits
            )
        )

    def assert_matched_certified(self) -> None:
        if self.convergence_reason == "practical_plateau":
            raise RuntimeError(
                "algorithm-matched certification requires strict smoothing convergence"
            )
        face = self.terminal_fit.coefficient_face
        if any(item.deactivated_face_components for item in self.history):
            raise RuntimeError(
                "algorithm-matched certification forbids a terminal exact-face revalidation failure"
            )
        if face is not None:
            raise RuntimeError(
                "the exact coefficient face is numerically supported but not certified"
            )
        if any(
            item.refused_face_components or item.endpoint_assessment_failure_reason is not None
            for item in self.history
        ):
            raise RuntimeError(
                "algorithm-matched certification forbids a refused endpoint assessment"
            )
        if any(
            not evidence._derived_authority_matches()
            or evidence.authority_identifier != "analytic-observed-curvature-direction/v1"
            for evidence in self.terminal_endpoint_directions.values()
        ):
            raise RuntimeError(
                "algorithm-matched certification requires self-consistent endpoint evidence"
            )
        for item in self.history:
            evidence = item.endpoint_direction_evidence
            if evidence is None:
                continue
            if not evidence.fit_indices or any(
                index >= len(self.coefficient_fits)
                or not self.coefficient_fits[index].converged
                or self.coefficient_fits[index].config.tolerance != evidence.coefficient_tolerance
                for index in evidence.fit_indices
            ):
                raise RuntimeError(
                    "algorithm-matched certification requires converged endpoint assessment fits"
                )
        final_revalidations = {
            name: direction
            for item in self.history
            for name, direction in _revalidated_endpoint_directions(item)
        }
        if dict(self.terminal_endpoint_directions) != final_revalidations:
            raise RuntimeError(
                "algorithm-matched certification requires the final face revalidation"
            )
        if self.fallback_count or any(
            fit.terminal_curvature.actual_source != fit.terminal_curvature.requested_source
            for fit in self.coefficient_fits
        ):
            raise RuntimeError("algorithm-matched certification forbids curvature fallback")
        if self.unresolved_upper_bound:
            raise RuntimeError("algorithm-matched certification forbids an unresolved upper bound")
        if not self.terminal_evidence_fresh:
            raise RuntimeError(
                "algorithm-matched certification requires fresh convergence evidence"
            )
        if not self.converged or not self.coefficient_converged:
            raise RuntimeError(
                "algorithm-matched certification requires converged inner and EFS fits"
            )
