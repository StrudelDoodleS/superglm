from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintSpec:
    mode: str
    kind: str


class _ConstraintKindNamespace:
    def __init__(self, mode: str):
        self.increasing = ConstraintSpec(mode=mode, kind="increasing")
        self.decreasing = ConstraintSpec(mode=mode, kind="decreasing")
        self.convex = ConstraintSpec(mode=mode, kind="convex")
        self.concave = ConstraintSpec(mode=mode, kind="concave")


class _ConstraintNamespace:
    fit = _ConstraintKindNamespace("fit")
    postfit = _ConstraintKindNamespace("postfit")


Constraint = _ConstraintNamespace()


__all__ = ["Constraint", "ConstraintSpec"]
