"""Shape-constraint specifications attached to a feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The values ``ConstraintSpec`` takes; spline construction validates against them.
CONSTRAINT_MODES = ("fit", "postfit")
CONSTRAINT_KINDS = ("increasing", "decreasing", "convex", "concave")


@dataclass(frozen=True)
class ConstraintSpec:
    """A single shape constraint: when it is applied, and what it requires."""

    mode: Literal["fit", "postfit"]
    kind: Literal["increasing", "decreasing", "convex", "concave"]


class _ConstraintKindNamespace:
    def __init__(self, mode: Literal["fit", "postfit"]):
        self.increasing = ConstraintSpec(mode=mode, kind="increasing")
        self.decreasing = ConstraintSpec(mode=mode, kind="decreasing")
        self.convex = ConstraintSpec(mode=mode, kind="convex")
        self.concave = ConstraintSpec(mode=mode, kind="concave")


class _ConstraintNamespace:
    """The shape constraints a feature can carry, reached through ``Constraint``.

    ``Constraint.fit`` enforces the shape inside the fit, so the fitted coefficients satisfy it;
    ``Constraint.postfit`` repairs the shape after an unconstrained fit. Each offers
    ``increasing``, ``decreasing``, ``convex`` and ``concave``, each a :class:`ConstraintSpec`.

    Examples
    --------
    >>> from superglm import Constraint, Spline
    >>> Spline(kind="ps", k=10, constraint=Constraint.fit.increasing)  # doctest: +SKIP
    """

    fit = _ConstraintKindNamespace("fit")
    postfit = _ConstraintKindNamespace("postfit")


Constraint = _ConstraintNamespace()


__all__ = ["Constraint", "ConstraintSpec"]
