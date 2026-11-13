"""The box the REML engines search smoothing parameters in, in the fit's own units.

A penalty ``lambda S`` is weighed against the data curvature ``X' W X``, and
only their ratio decides a fit. Rescaling every prior weight by ``c`` (and,
in families whose variance is a power of the mean, the response by ``a``)
rescales ``W`` and the optimal ``lambda`` together, while the predictions and
EDF stay put (Wood, Pya & Saefken 2016, JASA 111, sec. 3.1; Reiss & Ogden
2009, JRSSB 71, sec. 2). An absolute box breaks that equivariance, so the box
is ``[LAMBDA_FLOOR, LAMBDA_CEILING]`` times the fit's *information unit*: the
mean expected (Fisher) information per likelihood row at the intercept-only
fit, rounded to a power of two. The rounding makes a rescaling by ``2**k``
move the bounds exactly, as LAPACK's balancing (``xGEBAL``; Parlett &
Reinsch 1969) scales by powers of the radix, and keeps the old box exactly for
data whose unit rounds to one (an unweighted Gaussian or log-link gamma fit).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: The smallest and largest smoothing parameter an engine proposes, in units of
#: the information one likelihood row carries.
LAMBDA_FLOOR = 1.0e-6
LAMBDA_CEILING = 1.0e10

# The unit's binary exponent stays where every bound, and the largest
# conditioning limit (1e14) times it, is a finite normal float64.
_UNIT_EXPONENT_RANGE = (-950, 950)
# A value within this relative distance of a bound sits on it.
_ON_BOUND = 1.0e-8


def information_unit(information: NDArray, size: float) -> float:
    """``2**k`` nearest the mean of ``information`` over ``size`` likelihood rows.

    One when the mean is not a positive finite number.
    """
    # A scaling by 2**k commutes with every rounding of the sum (Higham 2002, sec. 2.2).
    total = float(np.sum(information, dtype=np.float64))
    if not (size > 0.0 and math.isfinite(size) and math.isfinite(total) and total > 0.0):
        return 1.0
    mean = total / size
    if not (math.isfinite(mean) and mean > 0.0):
        return 1.0
    exponent = round(math.log2(mean))
    low, high = _UNIT_EXPONENT_RANGE
    return math.ldexp(1.0, min(max(exponent, low), high))


@dataclass(frozen=True)
class LambdaBox:
    """``[LAMBDA_FLOOR, LAMBDA_CEILING] * unit``, the search box of one fit."""

    unit: float = 1.0

    @property
    def floor(self) -> float:
        return LAMBDA_FLOOR * self.unit

    @property
    def ceiling(self) -> float:
        return LAMBDA_CEILING * self.unit

    @property
    def log_floor(self) -> float:
        return float(np.log(self.floor))

    @property
    def log_ceiling(self) -> float:
        return float(np.log(self.ceiling))

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.floor, self.ceiling))

    def side(self, value: float) -> str | None:
        """``"floor"`` or ``"ceiling"`` when ``value`` sits on that bound, else None."""
        if value <= self.floor * (1.0 + _ON_BOUND):
            return "floor"
        if value >= self.ceiling * (1.0 - _ON_BOUND):
            return "ceiling"
        return None


def fitted_lambda_box(model) -> LambdaBox:
    """The box a fitted SuperGLM's smoothing parameters were searched in.

    A model saved before the box was relative, or never searched, reads the
    unit box its search used.
    """
    box = getattr(getattr(model, "_reml_result", None), "lambda_box", None)
    return box if isinstance(box, LambdaBox) else LambdaBox()


def bound_pressure(
    gradient: NDArray,
    rho: NDArray,
    estimated_mask: NDArray,
    names: Sequence[str],
    box: LambdaBox,
    *,
    bar: float,
    bound_window: float = 0.01,
) -> tuple[tuple[str, str], ...]:
    """``(name, side)`` of each estimated component a bound holds back.

    ``gradient`` is that of the minimised criterion in ``rho = log lambda``.
    A component within ``bound_window`` (``project_reml_gradient``'s) of a
    bound is held back when its gradient points out of the box by at least
    ``bar``, the stop rule's gradient bar: the search would have moved it on.
    """
    held = []
    for name, score, log_lambda, free in zip(names, gradient, rho, estimated_mask, strict=True):
        if not free or not abs(float(score)) >= bar:
            continue
        if log_lambda >= box.log_ceiling - bound_window and score < 0.0:
            held.append((name, "ceiling"))
        elif log_lambda <= box.log_floor + bound_window and score > 0.0:
            held.append((name, "floor"))
    return tuple(held)


def glm_lambda_box(
    *,
    distribution,
    link,
    y: NDArray,
    sample_weight: NDArray,
    offset: NDArray,
    weight_semantics: str,
) -> LambdaBox:
    """The box of a SuperGLM fit: its Fisher working weights at the null fit.

    Where the intercept-only fit has no valid mode the prior weights stand in.
    """
    from superglm.distributions import clip_mu
    from superglm.solvers.dispersion import dispersion_likelihood_size
    from superglm.solvers.working_rows import fisher_working_weights, null_fit_eta

    weights = np.asarray(sample_weight, dtype=np.float64)
    size = dispersion_likelihood_size(weights, weight_semantics=weight_semantics)
    eta = null_fit_eta(
        distribution=distribution,
        link=link,
        y=y,
        sample_weight=weights,
        offset=offset,
        weight_semantics=weight_semantics,
    )
    if eta is None:
        return LambdaBox(information_unit(weights, size))
    with np.errstate(all="ignore"):
        information = fisher_working_weights(
            distribution=distribution,
            link=link,
            mu=clip_mu(link.inverse(eta), distribution),
            eta=eta,
            sample_weight=weights,
        )
    return LambdaBox(information_unit(information, size))


def bound_warning_message(hits: Sequence[tuple[str, str, float, float]]) -> str | None:
    """One warning naming each term whose smoothing a bound held back.

    ``hits`` holds ``(term, side, lambda, unit)``, ``side`` being ``"floor"``
    or ``"ceiling"``: terms whose REML score still pointed past the bound when
    the search stopped.
    """
    if not hits:
        return None
    parts = []
    for term, side, value, unit in hits:
        relative = LAMBDA_FLOOR if side == "floor" else LAMBDA_CEILING
        which = "smallest" if side == "floor" else "largest"
        parts.append(
            f"{term!r} at the {which} smoothing parameter the search allows "
            f"(lambda = {value:.3g}, {relative:.0e} times this fit's information "
            f"per row, {unit:.3g})"
        )
    advice = []
    if any(side == "floor" for _, side, _, _ in hits):
        advice.append(
            "A term at the smallest value is nearly unpenalized and can follow noise: "
            "give it fewer basis functions, or fix its smoothing parameter."
        )
    if any(side == "ceiling" for _, side, _, _ in hits):
        advice.append(
            "A term at the largest value is close to its unpenalized part (a straight "
            "line for a spline): if that shape is expected, use that simpler term; "
            "otherwise check the term's data."
        )
    return (
        "Smoothing-parameter selection stopped with the REML score still asking for "
        "more than its limits allow on " + "; ".join(parts) + ". " + " ".join(advice)
    )
