"""Finite-difference directional derivative of predictor-scale curvature.

Every family provides ``evaluate_natural`` at derivative order two and the
inverse-link chain rule in :func:`transform_natural_derivatives`.  The
endpoint LAML derivative needs one more order along one direction; this
module supplies it by differencing the engine's own ``curvature_packed``
channel, which is an analytic second derivative, so the difference has an
eps^(4/5) floor after Richardson extrapolation rather than the eps^(1/2)
floor of a differenced first derivative.  The exact LAML Hessian needs one
order more still -- the second directional difference of the same channel,
bilinear in two directions -- which
:func:`finite_difference_curvature_second_direction` takes on the same
five-point stencil.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.links import Link

FINITE_DIFFERENCE_AUTHORITY = "finite-difference-curvature-direction/v1"
DEFAULT_STEP = 1.0e-3


@dataclass(frozen=True)
class FiniteDifferenceDirection:
    """Directional derivative of packed predictor curvature with an error estimate."""

    values: NDArray[np.float64]
    certificate: NDArray[np.float64]
    step: float
    evaluations: int

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        certificate = np.array(self.certificate, dtype=np.float64, copy=True)
        if values.ndim != 2 or certificate.shape != values.shape:
            raise ValueError("values and certificate must be matching two-dimensional arrays")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(certificate)):
            raise ValueError("finite-difference direction must be finite")
        if np.any(certificate < 0.0):
            raise ValueError("certificate must be non-negative")
        if not np.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("step must be a finite positive float")
        if isinstance(self.evaluations, bool) or self.evaluations < 1:
            raise ValueError("evaluations must be a positive integer")
        values.setflags(write=False)
        certificate.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "certificate", certificate)


def _theta_from_eta(eta: NDArray[np.float64], links: Sequence[Link]) -> NDArray[np.float64]:
    theta = np.empty_like(eta)
    for index, link in enumerate(links):
        column = np.asarray(link.inverse(eta[:, index]), dtype=np.float64)
        if column.shape != (eta.shape[0],) or not np.all(np.isfinite(column)):
            raise ValueError(f"inverse link {index} produced an invalid natural parameter")
        theta[:, index] = column
    return theta


def _curvature_packed(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray[np.float64],
    links: Sequence[Link],
    plan: FamilyLikelihoodPlan,
) -> NDArray[np.float64]:
    theta = _theta_from_eta(eta, links)
    natural = family.evaluate_natural(y, theta, plan, derivative_order=2)
    if natural.derivative_order != 2:
        raise ValueError("family must return derivative order two for a curvature direction")
    if natural.valid is not None and not bool(np.all(natural.valid)):
        raise ValueError("family flagged invalid rows at a perturbed predictor state")
    transformed = transform_natural_derivatives(natural, eta, links)
    return np.asarray(transformed.curvature_packed, dtype=np.float64)


def _central(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray[np.float64],
    unit_direction: NDArray[np.float64],
    links: Sequence[Link],
    plan: FamilyLikelihoodPlan,
    step: float,
) -> NDArray[np.float64]:
    plus = _curvature_packed(family, y, eta + step * unit_direction, links, plan)
    minus = _curvature_packed(family, y, eta - step * unit_direction, links, plan)
    return (plus - minus) / (2.0 * step)


def _axis_stencil(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray[np.float64],
    unit_direction: NDArray[np.float64],
    links: Sequence[Link],
    plan: FamilyLikelihoodPlan,
    step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Packed curvature at ``eta +- step*u`` and ``eta +- step/2*u``.

    These four points serve both public routines: the first derivative is the
    Richardson combination of the two central differences they form, and the
    second derivative is the five-point stencil they form with the centre
    value.  Returned in the order ``(plus, minus, plus_half, minus_half)``.
    """
    plus = _curvature_packed(family, y, eta + step * unit_direction, links, plan)
    minus = _curvature_packed(family, y, eta - step * unit_direction, links, plan)
    half = 0.5 * step
    plus_half = _curvature_packed(family, y, eta + half * unit_direction, links, plan)
    minus_half = _curvature_packed(family, y, eta - half * unit_direction, links, plan)
    return plus, minus, plus_half, minus_half


def _first_difference(
    stencil: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ],
    step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Order-4 first directional difference and its ``|D4 - D2(step/2)|`` certificate."""
    plus, minus, plus_half, minus_half = stencil
    coarse = (plus - minus) / (2.0 * step)
    fine = (plus_half - minus_half) / (2.0 * (0.5 * step))
    extrapolated = (4.0 * fine - coarse) / 3.0
    return extrapolated, np.abs(extrapolated - fine)


def _second_difference(
    center: NDArray[np.float64],
    stencil: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ],
    step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Order-4 second directional difference on the shared five-point stencil.

    With spacing ``d = step / 2`` the stencil holds ``f(0), f(+-d), f(+-2d)``;
    the order-2 estimate is ``(f(d) - 2 f(0) + f(-d)) / d^2`` and the order-4
    Richardson combination ``(-f(2d) + 16 f(d) - 30 f(0) + 16 f(-d) - f(-2d))
    / (12 d^2)``.  The certificate is their difference (the order-2
    truncation, which bounds the order-4 term) plus a round-off floor.

    The floor is at the ROW's curvature scale, ``64 eps max_row|f| / d^2``,
    not the entry's: a packed channel can be the difference of larger
    intermediates (the Gaussian ``(s, s)`` channel is ``(1 - 3 d^2/sigma^2)
    + (-1 + d^2/sigma^2)``), so its round-off follows the row, exactly as the
    first-derivative certificate test documents.  Measured on the Gaussian
    closed form over ten seeds, the worst error is 15.0 in units of
    ``eps max_row|f| / d^2`` (the stencil's coefficient sum is 64/12 = 5.3 per
    unit of evaluation error); 64 keeps a four-fold margin and is the
    constant that test already uses.
    """
    plus, minus, plus_half, minus_half = stencil
    spacing = 0.5 * step
    spacing_2 = spacing * spacing
    order2 = (plus_half - 2.0 * center + minus_half) / spacing_2
    order4 = (-plus + 16.0 * plus_half - 30.0 * center + 16.0 * minus_half - minus) / (
        12.0 * spacing_2
    )
    row_scale = np.max(
        np.abs(np.stack([center, plus, minus, plus_half, minus_half])),
        axis=(0, 2),
    )
    floor = 64.0 * np.finfo(np.float64).eps * row_scale[:, None] / spacing_2
    return order4, np.abs(order4 - order2) + floor


def _unit_directions(
    direction: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Row norms and the row-wise unit directions (zero rows stay zero)."""
    norms = np.linalg.norm(direction, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return norms, direction / safe[:, None]


def finite_difference_curvature_direction(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray,
    eta_direction: NDArray,
    links: Sequence[Link],
    plan: FamilyLikelihoodPlan,
    *,
    step: float = DEFAULT_STEP,
) -> FiniteDifferenceDirection:
    """Richardson-extrapolated central difference of packed curvature along a direction.

    The derivative is linear in the direction, so each row is differenced
    along its unit direction with an absolute link-scale step and rescaled by
    the row's direction norm; this keeps the step independent of the
    direction's arbitrary scale.  The certificate is ``|D4 - D2(step/2)|``,
    the difference between the extrapolated value and the finer central
    difference, an estimate of the remaining truncation and round-off error.
    """
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive float")
    eta_array = np.asarray(eta, dtype=np.float64)
    direction = np.asarray(eta_direction, dtype=np.float64)
    if eta_array.ndim != 2 or direction.shape != eta_array.shape:
        raise ValueError("eta and eta_direction must be matching n-by-k arrays")
    if not np.all(np.isfinite(eta_array)) or not np.all(np.isfinite(direction)):
        raise ValueError("eta and eta_direction must be finite")
    norms, unit = _unit_directions(direction)
    stencil = _axis_stencil(family, y, eta_array, unit, links, plan, step)
    extrapolated, certificate = _first_difference(stencil, step)
    if not np.all(np.isfinite(extrapolated)):
        raise ValueError("finite-difference curvature direction is not finite")
    return FiniteDifferenceDirection(
        values=extrapolated * norms[:, None],
        certificate=certificate * norms[:, None],
        step=float(step),
        evaluations=4,
    )


def finite_difference_curvature_second_direction(
    family: DistributionalFamily,
    y: NDArray,
    eta: NDArray,
    direction_a: NDArray,
    direction_b: NDArray,
    links: Sequence[Link],
    plan: FamilyLikelihoodPlan,
    *,
    step: float = DEFAULT_STEP,
) -> FiniteDifferenceDirection:
    """Richardson-extrapolated second directional difference of packed curvature.

    Bilinear in ``(direction_a, direction_b)``; a mixed pair is evaluated by
    polarisation, ``D4[a, b] = (D4[a+b, a+b] - D4[a-b, a-b]) / 4``, so only
    axis second differences are ever taken.  Each axis direction is
    differenced row-wise along its unit vector with the absolute link-scale
    step and rescaled by the squared row norm.  The centre value is shared by
    every axis, so a diagonal pair costs five packed-curvature evaluations and
    a mixed pair nine.
    """
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive float")
    eta_array = np.asarray(eta, dtype=np.float64)
    a = np.asarray(direction_a, dtype=np.float64)
    b = np.asarray(direction_b, dtype=np.float64)
    if eta_array.ndim != 2 or a.shape != eta_array.shape or b.shape != eta_array.shape:
        raise ValueError("directions must match eta's n-by-k shape")
    if not (np.all(np.isfinite(eta_array)) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("eta and directions must be finite")
    center = _curvature_packed(family, y, eta_array, links, plan)
    evaluations = 1

    def axis_second(
        direction: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        nonlocal evaluations
        norms, unit = _unit_directions(direction)
        stencil = _axis_stencil(family, y, eta_array, unit, links, plan, step)
        evaluations += 4
        values, certificate = _second_difference(center, stencil, step)
        scale = (norms * norms)[:, None]
        return values * scale, certificate * scale

    if np.array_equal(a, b):
        values, certificate = axis_second(a)
    else:
        plus, certificate_plus = axis_second(a + b)
        minus, certificate_minus = axis_second(a - b)
        values = (plus - minus) / 4.0
        certificate = (certificate_plus + certificate_minus) / 4.0
    if not np.all(np.isfinite(values)):
        raise ValueError("finite-difference curvature second direction is not finite")
    return FiniteDifferenceDirection(
        values=values,
        certificate=certificate,
        step=float(step),
        evaluations=evaluations,
    )


__all__ = [
    "DEFAULT_STEP",
    "FINITE_DIFFERENCE_AUTHORITY",
    "FiniteDifferenceDirection",
    "finite_difference_curvature_direction",
    "finite_difference_curvature_second_direction",
]
