"""Column-bound feature declarations, independent of likelihood families."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import ArrayLike

from superglm.features import Categorical, LevelGrouping, Numeric, RandomEffect, Spline
from superglm.features.constraint import ConstraintSpec
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.spline import _SplineBase
from superglm.types import FeatureSpec, LambdaPolicy

type InteractionSpec = (
    CategoricalInteraction
    | NumericCategorical
    | NumericInteraction
    | PolynomialCategorical
    | PolynomialInteraction
    | SplineCategorical
    | TensorInteraction
    | FactorSmooth
)

_INTERACTION_TYPES = (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
    FactorSmooth,
)


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Term and column names must be non-empty strings")


@dataclass(frozen=True)
class BoundTerm:
    """An owned feature configuration attached to one source column."""

    column: str
    _spec: FeatureSpec = field(repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.column)
        if isinstance(self._spec, _INTERACTION_TYPES):
            raise TypeError("Use interaction(spec) to bind an interaction specification")
        if not isinstance(self._spec, FeatureSpec):
            raise TypeError("term() requires a FeatureSpec")
        object.__setattr__(self, "_spec", deepcopy(self._spec))

    @property
    def spec(self) -> FeatureSpec:
        """Return an independent copy of the feature configuration."""
        return deepcopy(self._spec)


@dataclass(frozen=True)
class BoundInteraction:
    """An owned explicit interaction with a stable public name."""

    name: str
    _spec: InteractionSpec = field(repr=False)
    _requires_spline_parents: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not isinstance(self._spec, _INTERACTION_TYPES):
            raise TypeError("interaction() requires a supported explicit interaction specification")
        for parent in self._spec.parent_names:
            _validate_name(parent)
        object.__setattr__(self, "_spec", deepcopy(self._spec))

    @property
    def spec(self) -> InteractionSpec:
        """Return an independent copy of the interaction configuration."""
        return deepcopy(self._spec)


type TermInput = str | BoundTerm | BoundInteraction


@dataclass(frozen=True)
class NormalizedTerms:
    """Owned compiler inputs in main-effect and interaction declaration order."""

    features: dict[str, FeatureSpec]
    interaction_specs: dict[str, InteractionSpec]
    interaction_order: tuple[str, ...]


def term(column: str, spec: FeatureSpec) -> BoundTerm:
    """Bind an existing feature specification without building a design."""
    return BoundTerm(column, spec)


def s(
    column: str,
    *,
    kind: str = "ps",
    k: int | None = None,
    n_knots: int | None = None,
    degree: int = 3,
    knot_strategy: str = "uniform",
    penalty: str = "ssp",
    select: bool = False,
    knots: ArrayLike | None = None,
    discrete: bool | None = None,
    n_bins: int | None = None,
    extrapolation: str = "clip",
    boundary: tuple[float, float] | None = None,
    knot_alpha: float = 0.2,
    constraint: ConstraintSpec | None = None,
    m: int | tuple[int, ...] = 2,
    lambda_policy: LambdaPolicy | dict[str, LambdaPolicy] | None = None,
) -> BoundTerm:
    """Bind a spline with the existing Spline factory's parameter meanings."""
    return term(
        column,
        Spline(
            kind=kind,
            k=k,
            n_knots=n_knots,
            degree=degree,
            knot_strategy=knot_strategy,
            penalty=penalty,
            select=select,
            knots=knots,
            discrete=discrete,
            n_bins=n_bins,
            extrapolation=extrapolation,
            boundary=boundary,
            knot_alpha=knot_alpha,
            constraint=constraint,
            m=m,
            lambda_policy=lambda_policy,
        ),
    )


def cat(
    column: str,
    *,
    base: str = "most_exposed",
    grouping: LevelGrouping | None = None,
    levels: Any = None,
    unseen: Literal["error", "base"] = "error",
) -> BoundTerm:
    """Bind explicit categorical encoding, without dtype inference."""
    return term(column, Categorical(base=base, grouping=grouping, levels=levels, unseen=unseen))


def re(
    column: str,
    *,
    levels: Any = None,
    unseen: Literal["population", "error"] = "population",
    missing: Literal["error"] = "error",
    lambda_policy: LambdaPolicy | None = None,
) -> BoundTerm:
    """Bind an all-level random effect."""
    return term(
        column,
        RandomEffect(
            levels=levels,
            unseen=unseen,
            missing=missing,
            lambda_policy=lambda_policy,
        ),
    )


def interaction(spec: InteractionSpec, *, name: str | None = None) -> BoundInteraction:
    """Bind an existing explicit interaction, retaining its parent orientation."""
    if name is None and isinstance(spec, FactorSmooth):
        name = spec.name
    if name is None:
        parents = getattr(spec, "parent_names", None)
        if not isinstance(parents, tuple) or len(parents) != 2:
            raise TypeError("interaction() requires a supported explicit interaction specification")
        for parent in parents:
            _validate_name(parent)
        name = ":".join(parents)
    return BoundInteraction(name, spec)


def ti(
    left: str,
    right: str,
    *,
    n_knots: tuple[int, int] | None = None,
    decompose: bool = False,
) -> BoundInteraction:
    """Bind an interaction-only tensor of two declared spline parents."""
    _validate_name(left)
    _validate_name(right)
    return BoundInteraction(
        f"{left}:{right}",
        TensorInteraction(left, right, n_knots=n_knots, decompose=decompose),
        _requires_spline_parents=True,
    )


def normalize_terms(terms: Sequence[TermInput]) -> NormalizedTerms:
    """Resolve declarations without reading data or constructing row designs."""
    if isinstance(terms, str | bytes) or not isinstance(terms, Sequence):
        raise TypeError("terms must be a sequence of term declarations")
    features: dict[str, FeatureSpec] = {}
    interactions: dict[str, InteractionSpec] = {}
    spline_parent_interactions: set[str] = set()
    # Copy the complete graph once, retaining aliases within the owned graph.
    for declaration in deepcopy(tuple(terms)):
        if isinstance(declaration, str):
            _validate_name(declaration)
            name, spec = declaration, Numeric()
        elif isinstance(declaration, BoundTerm):
            name, spec = declaration.column, declaration._spec
        elif isinstance(declaration, BoundInteraction):
            if declaration.name in interactions:
                raise ValueError(f"Duplicate interaction declaration: {declaration.name!r}")
            interactions[declaration.name] = declaration._spec
            if declaration._requires_spline_parents:
                spline_parent_interactions.add(declaration.name)
            continue
        else:
            raise TypeError("Expected a numeric column string or a bound term declaration")
        if name in features:
            raise ValueError(f"Duplicate main term declaration for column {name!r}")
        features[name] = spec

    for name, ispec in interactions.items():
        if name in features:
            raise ValueError(
                f"Term name {name!r} identifies both a main feature and an interaction"
            )
        for parent in ispec.parent_names:
            if parent not in features:
                raise ValueError(f"Interaction {name!r} requires declared parent {parent!r}")
            if name in spline_parent_interactions and not isinstance(features[parent], _SplineBase):
                raise TypeError(
                    f"ti parent {parent!r} must be a declared spline; use s({parent!r})"
                )
    return NormalizedTerms(features, interactions, tuple(interactions))
