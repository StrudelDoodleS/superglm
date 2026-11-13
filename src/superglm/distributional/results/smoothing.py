"""The certified result of automatic smoothing-parameter selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from superglm.distributional._common import frozen_float_mapping
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
    _frozen_endpoint_mapping,
)


def _practical_outward_window(
    *,
    history: Sequence[DistributionalEFSIteration],
    coefficient_fits: Sequence[DenseSolverResult],
    terminal_raw_log_steps: Mapping[str, float] | None,
    config: DistributionalEFSConfig,
) -> bool:
    """Replay substantial outward probes on a stable finite profile.

    Increasing log steps and upper-bound pressure can be practically harmless
    only after actual accepted movement has tested a factor-e penalty increase
    in every such coordinate. This is a finite-fit stability policy, not an
    assertion about the sign of a derivative at the infinite-penalty endpoint.
    """
    if not config.practical_convergence or not terminal_raw_log_steps:
        return False
    window_size = config.plateau_iterations
    window = history[-window_size:]
    if len(window) != window_size:
        return False
    for item in window:
        if (
            item.stage != "efs"
            or not item.accepted
            or item.activated_face_components
            or item.deactivated_face_components
            or item.revalidated_face_components
            or item.refused_face_components
            or item.objective_relative_change > config.plateau_tolerance
        ):
            return False
        assert item.accepted_fit_index is not None
        if (
            _maximum_relative_natural_parameter_change(
                coefficient_fits[item.source_fit_index].theta,
                coefficient_fits[item.accepted_fit_index].theta,
            )
            > config.practical_parameter_tolerance
        ):
            return False
    first, last = window[0], window[-1]
    if tuple(terminal_raw_log_steps) != tuple(last.lambdas_after):
        return False
    assert last.accepted_fit_index is not None
    if (
        abs(last.objective_after - first.objective_before) / (1.0 + abs(first.objective_before))
        > config.plateau_tolerance
        or _maximum_relative_natural_parameter_change(
            coefficient_fits[first.source_fit_index].theta,
            coefficient_fits[last.accepted_fit_index].theta,
        )
        > config.practical_parameter_tolerance
    ):
        return False
    exempted = False
    for name, raw_step in terminal_raw_log_steps.items():
        terminal_lambda = last.lambdas_after[name]
        if raw_step == 0.0 and all(
            item.lambdas_after[name] == item.lambdas_before[name] for item in window
        ):
            # Fixed penalties have no raw pressure, including fixed zero and
            # cap-valued penalties. They neither supply nor obstruct a probe.
            continue
        if terminal_lambda == config.lower(name) and raw_step < -config.tolerance:
            return False
        # Reconstruct movement from accepted penalties, never from proposed or
        # clipped raw steps. A duplicate or reversal cannot count as a probe.
        steps = [math.log(item.lambdas_after[name] / item.lambdas_before[name]) for item in window]
        shrinking = all(abs(after) <= abs(before) for before, after in zip(steps, steps[1:]))
        at_upper = terminal_lambda == config.upper(name)
        if not at_upper and (shrinking or abs(raw_step) <= config.tolerance):
            continue
        if raw_step <= 0.0 or any(step <= 0.0 for step in steps):
            return False
        if math.log(terminal_lambda / first.lambdas_before[name]) < 1.0:
            return False
        exempted = True
    return exempted


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

    ``terminal_raw_max_log_step`` keeps its serialized name but stores the
    convergence authority: normalized stationarity for a finite target,
    augmented only by a direction actionable at the upper cap or identifying
    working infinity.

    A ``stationary`` result is the Newton endgame's: its authority is the
    exact LAML gradient in log lambda, ``terminal_gradient``, projected onto
    the box (``terminal_projected_gradient_norm`` within
    ``tolerance * (1 + |F|)``); the Fellner--Schall residual may then exceed
    the tolerance. ``smoothing_hessian`` is the exact Hessian at that point
    over ``terminal_gradient``'s names. ``beyond_cap_components`` names
    components released above ``maximum_lambda`` (never above
    ``maximum_lambda_conditioning``).
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
    terminal_raw_log_steps: Mapping[str, float] | None = None
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

    def bound_pressure(self) -> tuple[tuple[str, str], ...]:
        """``(name, "floor" | "ceiling")`` for each component a bound held back.

        At the ceiling: an unresolved cap, or a cap the endgame's gradient
        still pushes outward. At the floor: the endgame's gradient, or else the
        terminal Fellner--Schall step, pointing below it by more than the bar.
        """
        held: list[tuple[str, str]] = []
        gradient = self.terminal_gradient or {}
        raw_steps = self.terminal_raw_log_steps or {}
        bar = self.stationarity_bar
        for name, value in self.lambdas.items():
            if name in self.unresolved_upper_bound or (
                value == self.config.upper(name) and gradient.get(name, 0.0) < -bar
            ):
                held.append((name, "ceiling"))
            elif value == self.config.lower(name) and (
                gradient[name] > bar
                if name in gradient
                else raw_steps.get(name, 0.0) < -self.config.tolerance
            ):
                held.append((name, "floor"))
        return tuple(held)

    def __post_init__(self) -> None:
        for name in ("initial_objective", "objective"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(
            self,
            "initial_lambdas",
            frozen_float_mapping(self.initial_lambdas, name="initial_lambdas", nonnegative=True),
        )
        object.__setattr__(
            self, "lambdas", frozen_float_mapping(self.lambdas, name="lambdas", nonnegative=True)
        )
        object.__setattr__(
            self,
            "terminal_raw_max_log_step",
            _finite_nonnegative(self.terminal_raw_max_log_step, name="terminal_raw_max_log_step"),
        )
        object.__setattr__(self, "unresolved_upper_bound", tuple(self.unresolved_upper_bound))
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "coefficient_fits", tuple(self.coefficient_fits))
        object.__setattr__(
            self,
            "terminal_endpoint_directions",
            _frozen_endpoint_mapping(self.terminal_endpoint_directions),
        )
        if self.terminal_raw_log_steps is not None:
            raw_steps = frozen_float_mapping(
                self.terminal_raw_log_steps, name="terminal_raw_log_steps"
            )
            object.__setattr__(self, "terminal_raw_log_steps", raw_steps)
        if self.terminal_gradient is not None:
            gradient = frozen_float_mapping(self.terminal_gradient, name="terminal_gradient")
            object.__setattr__(self, "terminal_gradient", gradient)
        if self.terminal_gradient_certificate is not None:
            certificate = frozen_float_mapping(
                self.terminal_gradient_certificate,
                name="terminal_gradient_certificate",
                nonnegative=True,
            )
            object.__setattr__(self, "terminal_gradient_certificate", certificate)
        if self.terminal_projected_gradient_norm is not None:
            norm = _finite_nonnegative(
                self.terminal_projected_gradient_norm, name="terminal_projected_gradient_norm"
            )
            object.__setattr__(self, "terminal_projected_gradient_norm", norm)
        for name in ("smoothing_hessian", "smoothing_hessian_certificate"):
            if getattr(self, name) is not None:
                values = np.array(getattr(self, name), dtype=np.float64, copy=True)
                values.setflags(write=False)
                object.__setattr__(self, name, values)
        object.__setattr__(self, "beyond_cap_components", tuple(self.beyond_cap_components))

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
            # The cap never certifies under the Newton authority: a component
            # resting at the configured maximum is stationary at a box bound.
            # The Fellner--Schall path assesses cap-resting nominations itself.
            and not (
                self.convergence_reason == "stationary"
                and any(
                    value == self.config.upper(name)
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
