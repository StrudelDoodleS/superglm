"""Shape-constraint specifications attached to a feature."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

# The values ``ConstraintSpec`` takes; spline construction validates against them.
CONSTRAINT_MODES = ("fit", "postfit")
CONSTRAINT_KINDS = ("increasing", "decreasing", "convex", "concave")
# Every rule a fit-time combination may hold: the four shapes plus flat, cap and max_step.
RULE_KINDS = CONSTRAINT_KINDS + ("flat", "cap", "max_step")

_DIRECTION_KINDS = ("increasing", "decreasing")
_CURVATURE_KINDS = ("convex", "concave")
_STRETCH_KINDS = _DIRECTION_KINDS + _CURVATURE_KINDS + ("flat",)
_FIT_TIME_ONLY = (
    "{what} is fit-time only: write it with Constraint.fit, because Constraint.postfit "
    "repairs one monotone or curvature shape after the fit and cannot combine rules"
)


def _fmt(x: float | str) -> str:
    if isinstance(x, str):
        return repr(x)
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def _check_end(x: float | str | None) -> None:
    if x is None or isinstance(x, str):
        return
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"a stretch end must be a finite number or a band label, got {x!r}")


@dataclass(frozen=True, repr=False)
class ConstraintSpec:
    """
    A single shape rule: when it is applied, and what it requires.

    ``lo`` and ``hi`` bound the stretch of the feature the rule applies to
    (``None`` is unbounded; an end is a number or an ordered-categorical band
    label). ``bounds`` carries the two numbers of a ``cap`` (lo, hi) or a
    ``max_step`` (up, down). Rules join with ``&`` into :class:`ConstraintRules`.
    """

    mode: Literal["fit", "postfit"]
    kind: Literal["increasing", "decreasing", "convex", "concave", "flat", "cap", "max_step"]
    lo: float | str | None = None
    hi: float | str | None = None
    bounds: tuple[float, float] | None = None

    @property
    def stretch(self) -> tuple[float | str | None, float | str | None]:
        return (self.lo, self.hi)

    def _with_stretch(self, lo: float | str | None, hi: float | str | None) -> ConstraintSpec:
        if self.mode != "fit":
            raise ValueError(_FIT_TIME_ONLY.format(what="a stretch"))
        if self.kind not in _STRETCH_KINDS:
            raise ValueError(f"{self.describe()} applies to the whole curve and takes no stretch")
        if self.stretch != (None, None):
            raise ValueError(f"{self.describe()} already has a stretch; give one stretch per rule")
        _check_end(lo)
        _check_end(hi)
        if lo is not None and hi is not None:
            numeric = not isinstance(lo, str) and not isinstance(hi, str)
            if numeric and not lo < hi:
                raise ValueError(
                    f"a stretch needs lo < hi, got within({_fmt(lo)}, {_fmt(hi)}); "
                    "swap the ends or use a single below/above"
                )
        return ConstraintSpec(self.mode, self.kind, lo, hi, self.bounds)

    def below(self, x: float | str) -> ConstraintSpec:
        """Apply the rule up to ``x`` only."""
        return self._with_stretch(None, x)

    def above(self, x: float | str) -> ConstraintSpec:
        """Apply the rule from ``x`` upward only."""
        return self._with_stretch(x, None)

    def within(self, a: float | str, b: float | str) -> ConstraintSpec:
        """Apply the rule between ``a`` and ``b`` only."""
        return self._with_stretch(a, b)

    def describe(self) -> str:
        if self.kind in ("cap", "max_step"):
            up, down = self.bounds if self.bounds is not None else (None, None)
            if self.kind == "max_step" and up == down:
                return f"max_step({_fmt(up)})"
            return f"{self.kind}({_fmt(up)}, {_fmt(down)})"
        if self.lo is None and self.hi is None:
            return self.kind
        if self.lo is None:
            return f"{self.kind}.below({_fmt(self.hi)})"
        if self.hi is None:
            return f"{self.kind}.above({_fmt(self.lo)})"
        return f"{self.kind}.within({_fmt(self.lo)}, {_fmt(self.hi)})"

    def __repr__(self) -> str:
        return f"Constraint.{self.mode}.{self.describe()}"

    def __and__(self, other: ConstraintSpec | ConstraintRules) -> ConstraintRules:
        return ConstraintRules((self,)) & other


@dataclass(frozen=True, repr=False)
class ConstraintRules:
    """
    An ordered set of fit-time shape rules, built with ``&``.

    Examples
    --------
    >>> from superglm import Constraint
    >>> C = Constraint.fit
    >>> C.increasing & C.concave & C.cap(0.6, 1.8)
    Constraint.fit.increasing & Constraint.fit.concave & Constraint.fit.cap(0.6, 1.8)
    """

    rules: tuple[ConstraintSpec, ...] = ()

    def __post_init__(self) -> None:
        if len(self.rules) > 1 and any(r.mode != "fit" for r in self.rules):
            raise ValueError(_FIT_TIME_ONLY.format(what="combining rules with &"))

    def __iter__(self) -> Iterator[ConstraintSpec]:
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def __and__(self, other: ConstraintSpec | ConstraintRules) -> ConstraintRules:
        if isinstance(other, ConstraintSpec):
            return ConstraintRules(self.rules + (other,))
        if isinstance(other, ConstraintRules):
            return ConstraintRules(self.rules + other.rules)
        return NotImplemented

    def describe(self) -> str:
        return " & ".join(r.describe() for r in self.rules)

    def __repr__(self) -> str:
        return " & ".join(repr(r) for r in self.rules)

    def split(self, kinds: frozenset[str]) -> tuple[ConstraintRules, ConstraintRules]:
        """The rules whose kind is in ``kinds``, then the rest."""
        chosen = tuple(r for r in self.rules if r.kind in kinds)
        rest = tuple(r for r in self.rules if r.kind not in kinds)
        return ConstraintRules(chosen), ConstraintRules(rest)

    def resolve(self, position: Callable[[float | str], float]) -> ConstraintRules:
        """Replace every stretch end by ``position(end)``, then re-check the rules."""

        def place(x: float | str | None) -> float | None:
            return None if x is None else position(x)

        resolved = ConstraintRules(
            tuple(
                ConstraintSpec(r.mode, r.kind, place(r.lo), place(r.hi), r.bounds)
                for r in self.rules
            )
        )
        resolved.validate()
        return resolved

    def validate(self) -> None:
        """Refuse rule sets that contradict themselves or cannot be fitted."""
        for r in self.rules:
            self._validate_one(r)
        for kind in ("cap", "max_step"):
            if sum(r.kind == kind for r in self.rules) > 1:
                raise ValueError(f"give at most one {kind} rule per term; merge them into one")
        for group in (_DIRECTION_KINDS, _CURVATURE_KINDS, ("flat",)):
            members = [r for r in self.rules if r.kind in group]
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    self._refuse_overlap(a, b)

    @staticmethod
    def _validate_one(r: ConstraintSpec) -> None:
        if r.mode != "fit" and (r.kind not in CONSTRAINT_KINDS or r.stretch != (None, None)):
            raise ValueError(_FIT_TIME_ONLY.format(what=r.describe()))
        if r.kind == "flat" and r.stretch == (None, None):
            raise ValueError(
                "a flat rule needs a stretch such as Constraint.fit.flat.within(25, 40); "
                "a term with no effect should be removed"
            )
        lo, hi = r.stretch
        labelled = isinstance(lo, str) or isinstance(hi, str)
        if lo is not None and hi is not None and not labelled and not lo < hi:
            raise ValueError(
                f"{r.describe()} needs lo < hi; swap the ends or use a single below/above"
            )

    @staticmethod
    def _refuse_overlap(a: ConstraintSpec, b: ConstraintSpec) -> None:
        ends = (a.lo, a.hi, b.lo, b.hi)
        if any(isinstance(x, str) for x in ends):
            return  # band labels are compared once resolved to positions
        start = max(-math.inf if a.lo is None else a.lo, -math.inf if b.lo is None else b.lo)
        stop = min(math.inf if a.hi is None else a.hi, math.inf if b.hi is None else b.hi)
        if start < stop:
            raise ValueError(
                f"{a.describe()} and {b.describe()} both apply on [{_fmt(start)}, {_fmt(stop)}]; "
                "give them stretches that do not overlap"
            )


def _cap(lo: float, hi: float) -> ConstraintSpec:
    if not lo < hi:
        raise ValueError(f"a cap needs lo < hi, got cap({_fmt(lo)}, {_fmt(hi)})")
    if not (0 < lo <= 1 <= hi) or not math.isfinite(hi):
        raise ValueError(
            f"a cap must contain 1 and stay positive (0 < lo <= 1 <= hi), got "
            f"cap({_fmt(lo)}, {_fmt(hi)}); relativities are measured against the base level at 1"
        )
    return ConstraintSpec("fit", "cap", bounds=(float(lo), float(hi)))


def _max_step(up: float, down: float | None = None) -> ConstraintSpec:
    down = up if down is None else down
    if not (0 < up < math.inf):
        raise ValueError(f"max_step needs up > 0, got up={_fmt(up)}")
    if not 0 < down < 1:
        raise ValueError(
            f"max_step needs 0 < down < 1 (adjacent ratios stay in [1 - down, 1 + up]), "
            f"got down={_fmt(down)}; give down explicitly when up is 1 or more"
        )
    return ConstraintSpec("fit", "max_step", bounds=(float(up), float(down)))


class _ConstraintKindNamespace:
    def __init__(self, mode: Literal["fit", "postfit"]):
        self.increasing = ConstraintSpec(mode=mode, kind="increasing")
        self.decreasing = ConstraintSpec(mode=mode, kind="decreasing")
        self.convex = ConstraintSpec(mode=mode, kind="convex")
        self.concave = ConstraintSpec(mode=mode, kind="concave")


class _FitKindNamespace(_ConstraintKindNamespace):
    def __init__(self) -> None:
        super().__init__("fit")
        self.flat = ConstraintSpec(mode="fit", kind="flat")
        self.cap = _cap
        self.max_step = _max_step


class _ConstraintNamespace:
    """The shape constraints a feature can carry, reached through ``Constraint``.

    ``Constraint.fit`` enforces the shape inside the fit, so the fitted coefficients satisfy it;
    ``Constraint.postfit`` repairs the shape after an unconstrained fit. Each offers
    ``increasing``, ``decreasing``, ``convex`` and ``concave``, each a :class:`ConstraintSpec`.
    ``Constraint.fit`` also offers ``flat``, ``cap(lo, hi)`` and ``max_step(up, down=None)``;
    its rules combine with ``&`` into :class:`ConstraintRules`, and ``.below(x)``, ``.above(x)``
    and ``.within(a, b)`` restrict a rule to a stretch of the feature.

    Examples
    --------
    >>> from superglm import Constraint, Spline
    >>> Spline(kind="ps", k=10, constraint=Constraint.fit.increasing)  # doctest: +SKIP
    """

    fit = _FitKindNamespace()
    postfit = _ConstraintKindNamespace("postfit")


Constraint = _ConstraintNamespace()


def rules_of(spec: Any) -> ConstraintRules:
    """The rules a spline spec carries, reading current fields then the legacy ones."""
    rules = getattr(spec, "constraint_rules", None)
    if rules is not None:
        return rules
    kind = getattr(spec, "constraint_kind", getattr(spec, "monotone", None))
    if kind is None:
        return ConstraintRules(())
    mode = getattr(spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit"))
    return ConstraintRules((ConstraintSpec(mode, kind),))


__all__ = ["Constraint", "ConstraintRules", "ConstraintSpec", "RULE_KINDS", "rules_of"]
