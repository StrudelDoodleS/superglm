"""Safeguarded generalized Fellner--Schall updates for smoothing penalties.

For a component penalty ``S_j``, let ``r_j(lambda)`` denote its effective
log-determinant derivative

``r_j(lambda) = lambda_j tr(S_lambda^+ S_j)``.

For an isolated block with declared positive rank ``r_j``, this reduces to
``r_j(lambda) = r_j`` because

``tr(S_lambda^+ S_j) = r_j / lambda_j``.

At a coefficient mode, the Wood--Fasiolo approximate smoothing-score
stationarity equation is

``q_j / phi = r_j(lambda) / lambda_j - tr(H^-1 S_j)``,

where ``q_j = beta_j.T @ S_j @ beta_j`` and ``H`` is penalized coefficient
curvature. The generalized Fellner--Schall proposal applies the current
residual effective degrees of freedom directly:

``lambda_j_new = (r_j(lambda) - lambda_j tr(H^-1 S_j)) / (q_j / phi)``.

The caller supplies ``r_j(lambda)`` so shared-block penalties use the joint
penalty geometry while isolated blocks retain the rank shortcut. This module
owns only that penalty-side algebra. It has no family, design, PIRLS, cache,
or objective dependency. Callers remain responsible for coefficient refits
and acceptance safeguards after a proposed update.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.types import LambdaPolicy

EFSProposalKind = Literal[
    "inactive",
    "gfs",
    "fixed_point_fallback",
    "working_infinity",
]
_PROPOSAL_KINDS = frozenset({"inactive", "gfs", "fixed_point_fallback", "working_infinity"})
_SMALLEST_POSITIVE_FLOAT = float(np.nextafter(0.0, math.inf))


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class EFSComponentState:
    """One penalty component in a caller-owned coefficient coordinate system."""

    name: str
    coefficient_slice: slice
    penalty: NDArray[np.float64]
    rank: float
    lambda_value: float
    policy: LambdaPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("component name must be a non-empty string")
        coefficient_slice = self.coefficient_slice
        if not isinstance(coefficient_slice, slice) or coefficient_slice.step not in (None, 1):
            raise ValueError("coefficient_slice must be a contiguous slice")
        start = coefficient_slice.start
        stop = coefficient_slice.stop
        if not isinstance(start, int) or not isinstance(stop, int) or start < 0 or stop <= start:
            raise ValueError("coefficient_slice must have finite increasing integer bounds")
        width = stop - start
        try:
            penalty = np.asarray(self.penalty, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("penalty must be a finite symmetric matrix") from exc
        if penalty.shape != (width, width):
            raise ValueError(
                f"penalty shape {penalty.shape} does not match coefficient slice width {width}"
            )
        if not np.all(np.isfinite(penalty)):
            raise ValueError("penalty must be a finite symmetric matrix")
        scale = max(1.0, float(np.linalg.norm(penalty, ord=np.inf)))
        tolerance = 32.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(penalty, penalty.T, rtol=0.0, atol=tolerance):
            raise ValueError("penalty must be symmetric")
        penalty = 0.5 * (penalty + penalty.T)
        if float(np.min(np.linalg.eigvalsh(penalty))) < -tolerance:
            raise ValueError("penalty must be positive semidefinite")

        rank = _finite_float(self.rank, name="rank")
        if rank < 0.0 or rank > width:
            raise ValueError("rank must lie between zero and the component width")
        lambda_value = _finite_float(self.lambda_value, name="lambda_value")
        if not isinstance(self.policy, LambdaPolicy):
            raise TypeError("policy must be a LambdaPolicy")
        if self.policy.mode == "estimate":
            if lambda_value <= 0.0:
                raise ValueError("estimated lambda_value must be positive")
        else:
            fixed_value = self.policy.value
            if fixed_value is None:  # Defensive against malformed deserialized policies.
                raise ValueError("fixed policy must define a value")
            policy_value = _finite_float(fixed_value, name="fixed policy value")
            if policy_value < 0.0:
                raise ValueError("fixed policy value must be non-negative")
            if lambda_value != policy_value:
                raise ValueError("lambda_value must equal the fixed policy value")

        owned_penalty = np.array(penalty, dtype=np.float64, copy=True)
        owned_penalty.setflags(write=False)
        object.__setattr__(self, "coefficient_slice", slice(start, stop))
        object.__setattr__(self, "penalty", owned_penalty)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "lambda_value", lambda_value)


def _frozen_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({name: float(value) for name, value in values.items()})


def _frozen_proposal_mapping(
    values: Mapping[str, EFSProposalKind],
) -> Mapping[str, EFSProposalKind]:
    return MappingProxyType(dict(values))


def _bounded_lambda_from_raw_log_step(
    current: float,
    raw_log_step: float,
    *,
    max_log_step: float | None,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Apply the configured trust step and lambda bounds without overflow."""

    bounded_log_step = (
        raw_log_step
        if max_log_step is None
        else float(np.clip(raw_log_step, -max_log_step, max_log_step))
    )
    if bounded_log_step == 0.0:
        return current, 0.0
    current_log = math.log(current)
    proposed_log = current_log + bounded_log_step
    if proposed_log <= math.log(lower):
        proposed = lower
    elif proposed_log >= math.log(upper):
        proposed = upper
    else:
        proposed = math.exp(proposed_log)
    return proposed, math.log(proposed) - current_log


def _quadratic_psd_error_bound(beta: NDArray, penalty: NDArray) -> float:
    """Bound admitted PSD and floating-point error in ``beta.T @ S @ beta``."""

    width = len(beta)
    epsilon = np.finfo(np.float64).eps
    operations = 2 * width + 2
    product = operations * epsilon
    if product >= 1.0:
        return math.inf
    gamma = product / (1.0 - product)
    absolute_sum = float(np.abs(beta) @ np.abs(penalty) @ np.abs(beta))
    penalty_scale = max(1.0, float(np.linalg.norm(penalty, ord=np.inf)))
    admission = 32.0 * epsilon * penalty_scale * float(beta @ beta)
    bound = gamma * absolute_sum + admission
    return float(np.nextafter(bound, math.inf)) if math.isfinite(bound) else math.inf


@dataclass(frozen=True)
class EFSUpdateResult:
    """One deterministic penalty update and its complete scalar diagnostics.

    ``raw_log_steps`` are the chosen unbounded proposal steps.
    ``proposal_kinds`` distinguishes generalized Fellner--Schall, safeguarded
    fixed-point fallback, working infinity, and inactive components.
    ``stationarity_log_residuals`` are normalized residuals of the shared
    finite fixed-point equation. They have the same interior roots as GFS, but
    GFS is amplified near a saturated penalty and is therefore not itself a
    convergence certificate.
    """

    lambdas: Mapping[str, float]
    log_steps: Mapping[str, float]
    raw_log_steps: Mapping[str, float]
    stationarity_log_residuals: Mapping[str, float]
    proposal_kinds: Mapping[str, EFSProposalKind]
    quadratic_forms: Mapping[str, float]
    trace_terms: Mapping[str, float]

    def __post_init__(self) -> None:
        names = tuple(self.lambdas)
        if (
            tuple(self.log_steps) != names
            or tuple(self.raw_log_steps) != names
            or tuple(self.stationarity_log_residuals) != names
            or tuple(self.proposal_kinds) != names
            or tuple(self.quadratic_forms) != names
            or tuple(self.trace_terms) != names
        ):
            raise ValueError("EFS result mappings must have identical deterministic key order")
        for name in names:
            values = (
                self.lambdas[name],
                self.log_steps[name],
                self.raw_log_steps[name],
                self.stationarity_log_residuals[name],
                self.quadratic_forms[name],
                self.trace_terms[name],
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"EFS diagnostics for {name!r} must be finite")
            if self.lambdas[name] < 0.0:
                raise ValueError(f"lambda for {name!r} must be non-negative")
            if self.proposal_kinds[name] not in _PROPOSAL_KINDS:
                raise ValueError(f"invalid EFS proposal kind for {name!r}")
        object.__setattr__(self, "lambdas", _frozen_mapping(self.lambdas))
        object.__setattr__(self, "log_steps", _frozen_mapping(self.log_steps))
        object.__setattr__(self, "raw_log_steps", _frozen_mapping(self.raw_log_steps))
        object.__setattr__(
            self,
            "stationarity_log_residuals",
            _frozen_mapping(self.stationarity_log_residuals),
        )
        object.__setattr__(
            self,
            "proposal_kinds",
            _frozen_proposal_mapping(self.proposal_kinds),
        )
        object.__setattr__(self, "quadratic_forms", _frozen_mapping(self.quadratic_forms))
        object.__setattr__(self, "trace_terms", _frozen_mapping(self.trace_terms))


def wood_fasiolo_update(
    components: Sequence[EFSComponentState],
    coefficients: NDArray,
    penalized_inverse: NDArray,
    *,
    inverse_scale: float = 1.0,
    max_log_step: float | None = 5.0,
    minimum_lambda: float = 1.0e-6,
    maximum_lambda: float = 1.0e10,
) -> EFSUpdateResult:
    """Return bounded generalized Fellner--Schall proposals."""
    component_tuple = tuple(components)
    if not component_tuple:
        raise ValueError("at least one EFS component is required")
    if not all(isinstance(component, EFSComponentState) for component in component_tuple):
        raise TypeError("components must contain only EFSComponentState values")
    names = tuple(component.name for component in component_tuple)
    if len(set(names)) != len(names):
        raise ValueError("EFS component names must be unique")

    try:
        beta = np.asarray(coefficients, dtype=np.float64)
        inverse = np.asarray(penalized_inverse, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("coefficients and penalized_inverse must be finite arrays") from exc
    if beta.ndim != 1 or not np.all(np.isfinite(beta)):
        raise ValueError("coefficients must be a finite vector")
    if inverse.shape != (len(beta), len(beta)) or not np.all(np.isfinite(inverse)):
        raise ValueError("penalized_inverse must be a finite square coefficient matrix")
    inverse_tolerance = (
        32.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.linalg.norm(inverse, ord=np.inf)),
        )
    )
    if not np.allclose(inverse, inverse.T, rtol=0.0, atol=inverse_tolerance):
        raise ValueError("penalized_inverse must be symmetric")
    inverse = 0.5 * (inverse + inverse.T)
    for component in component_tuple:
        if component.coefficient_slice.stop > len(beta):
            raise ValueError(f"component {component.name!r} lies outside coefficients")

    scale_weight = _finite_float(inverse_scale, name="inverse_scale")
    if scale_weight <= 0.0:
        raise ValueError("inverse_scale must be positive")
    lower = _finite_float(minimum_lambda, name="minimum_lambda")
    upper = _finite_float(maximum_lambda, name="maximum_lambda")
    if lower <= 0.0 or upper < lower:
        raise ValueError("lambda bounds must be positive and increasing")
    if max_log_step is not None:
        step_bound = _finite_float(max_log_step, name="max_log_step")
        if step_bound <= 0.0:
            raise ValueError("max_log_step must be positive")
    else:
        step_bound = None
    lambdas: dict[str, float] = {}
    log_steps: dict[str, float] = {}
    raw_log_steps: dict[str, float] = {}
    stationarity_log_residuals: dict[str, float] = {}
    proposal_kinds: dict[str, EFSProposalKind] = {}
    quadratic_forms: dict[str, float] = {}
    trace_terms: dict[str, float] = {}
    for component in component_tuple:
        name = component.name
        local_beta = beta[component.coefficient_slice]
        inverse_block = inverse[component.coefficient_slice, component.coefficient_slice]
        quadratic = float(local_beta @ component.penalty @ local_beta)
        trace_term = float(np.trace(inverse_block @ component.penalty))
        if not math.isfinite(quadratic) or not math.isfinite(trace_term):
            raise ValueError(f"EFS diagnostics for {name!r} are non-finite")
        if quadratic < 0.0:
            quadratic_error = _quadratic_psd_error_bound(local_beta, component.penalty)
            if not math.isfinite(quadratic_error) or -quadratic > quadratic_error:
                raise ValueError(f"EFS quadratic form for {name!r} is materially negative")
            quadratic = 0.0
        quadratic_forms[name] = quadratic
        trace_terms[name] = trace_term

        proposed = component.lambda_value
        log_step = 0.0
        raw_log_step = 0.0
        stationarity_log_residual = 0.0
        proposal_kind: EFSProposalKind = "inactive"
        estimated = component.policy.mode == "estimate"
        fixed_point_denominator = scale_weight * quadratic + trace_term
        if estimated and component.rank > 0.0:
            if fixed_point_denominator > 0.0:
                stationarity_log_residual = (
                    math.log(component.rank)
                    - math.log(fixed_point_denominator)
                    - math.log(component.lambda_value)
                )
            else:
                # No finite fixed point is visible at this numerical resolution.
                stationarity_log_residual = math.log(np.finfo(np.float64).max) - math.log(
                    component.lambda_value
                )

            numerator = component.rank - component.lambda_value * trace_term
            denominator = scale_weight * quadratic
            if numerator > 0.0 and denominator > 0.0:
                proposal_kind = "gfs"
                raw_log_step = (
                    math.log(numerator) - math.log(denominator) - math.log(component.lambda_value)
                )
            elif numerator > 0.0:
                proposal_kind = "working_infinity"
                # q≈0 is the GFS infinity update. Keep a finite, positive raw
                # diagnostic while the configured upper bound limits the trial.
                raw_log_step = max(
                    _SMALLEST_POSITIVE_FLOAT,
                    math.log(np.finfo(np.float64).max) - math.log(component.lambda_value),
                )
            else:
                proposal_kind = "fixed_point_fallback"
                # The GFS quotient is not positive, which falls outside its
                # positive-curvature assumptions. The normalized fixed-point
                # map remains a safe directional fallback under backtracking.
                raw_log_step = stationarity_log_residual

            proposed, log_step = _bounded_lambda_from_raw_log_step(
                component.lambda_value,
                raw_log_step,
                max_log_step=step_bound,
                lower=lower,
                upper=upper,
            )

        lambdas[name] = proposed
        log_steps[name] = log_step
        raw_log_steps[name] = raw_log_step
        stationarity_log_residuals[name] = stationarity_log_residual
        proposal_kinds[name] = proposal_kind

    return EFSUpdateResult(
        lambdas=lambdas,
        log_steps=log_steps,
        raw_log_steps=raw_log_steps,
        stationarity_log_residuals=stationarity_log_residuals,
        proposal_kinds=proposal_kinds,
        quadratic_forms=quadratic_forms,
        trace_terms=trace_terms,
    )


__all__ = ["EFSComponentState", "EFSUpdateResult", "wood_fasiolo_update"]
