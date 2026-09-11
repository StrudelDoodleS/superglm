"""Owned predictor declarations bound to the originating family instance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MethodType

from superglm.distributional.family import DistributionalFamily, validate_family
from superglm.distributional.predictor import Predictor
from superglm.links import Link
from superglm.terms import TermInput, normalize_terms


def _copy_template(template: Predictor) -> Predictor:
    features, interactions, link = deepcopy(
        (dict(template.features), dict(template.interaction_specs), template.link)
    )
    return Predictor(
        template.name,
        features,
        link=link,
        intercept=template.intercept,
        interactions=template.interactions,
        interaction_specs=interactions,
        interaction_order=template.interaction_order,
    )


@dataclass(frozen=True, eq=False)
class BoundPredictor:
    """A family-owned declaration with defensive access to its normalized template."""

    family: DistributionalFamily = field(repr=False)
    _template: Predictor = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._template, Predictor):
            raise TypeError("template must be a Predictor")
        names = tuple(parameter.name for parameter in validate_family(self.family))
        if self._template.name not in names:
            raise ValueError(
                f"Unknown predictor {self._template.name!r} for {type(self.family).__name__}; "
                f"valid predictors: {', '.join(names)}"
            )
        object.__setattr__(self, "_template", _copy_template(self._template))

    @property
    def name(self) -> str:
        return self._template.name

    @property
    def template(self) -> Predictor:
        """Return an independent normalized configuration."""
        return _copy_template(self._template)


def bind_predictor(
    family: DistributionalFamily,
    name: str,
    *terms: TermInput,
    intercept: bool = True,
    link: str | Link | None = None,
) -> BoundPredictor:
    """Bind terms to one canonical parameter of a built-in or custom family."""
    normalized = normalize_terms(terms)
    return BoundPredictor(
        family,
        Predictor(
            name,
            normalized.features,
            intercept=intercept,
            link=link,
            interaction_specs=normalized.interaction_specs,
            interaction_order=normalized.interaction_order,
        ),
    )


def _bind_predictor_template(family: DistributionalFamily, template: Predictor) -> BoundPredictor:
    """Bind an internal template without dropping persisted controls or interactions."""
    return BoundPredictor(family, template)


def _helper_names(family: DistributionalFamily) -> dict[str, str]:
    """Collect explicit helper metadata across the small parameter mixins."""
    helpers: dict[str, str] = {}
    for cls in reversed(type(family).__mro__):
        declared = vars(cls).get("_predictor_helpers", {})
        if isinstance(declared, Mapping):
            for name, helper in declared.items():
                if isinstance(name, str) and isinstance(helper, str) and helper.isidentifier():
                    helpers[name] = helper
    return helpers


def _missing_message(
    family: DistributionalFamily, names: tuple[str, ...], missing: tuple[str, ...]
) -> str:
    helpers = _helper_names(family)
    lines = [
        f"{type(family).__name__} is missing a predictor for {', '.join(missing)}.",
        "",
        "SuperLSS(",
        "    family,",
    ]
    for name in names:
        helper = helpers.get(name)
        declaration = (
            f"family.{helper}(...)"
            if helper is not None
            else f'bind_predictor(family, "{name}", ...)'
        )
        marker = "  # <--- missing predictor; add this" if name in missing else ""
        lines.append(f"    {declaration},{marker}")
    lines.append(")")
    return "\n".join(lines)


def resolve_predictors(
    family: DistributionalFamily, predictors: Sequence[BoundPredictor]
) -> tuple[DistributionalFamily, tuple[Predictor, ...]]:
    """Validate origin identity, snapshot the family, and order complete templates."""
    validate_family(family)
    if isinstance(predictors, str | bytes) or not isinstance(predictors, Sequence):
        raise TypeError("predictors must be a sequence of BoundPredictor values")
    values = tuple(predictors)
    for position, predictor in enumerate(values, start=1):
        if not isinstance(predictor, BoundPredictor):
            if isinstance(predictor, MethodType):
                raise TypeError(
                    f"Predictor argument {position} is a bare method; "
                    f"call family.{predictor.__name__}(...) to produce a BoundPredictor"
                )
            raise TypeError(f"Predictor argument {position} must be a BoundPredictor")
    for predictor in values:
        if predictor.family is not family:
            raise ValueError(
                f"Predictor {predictor.name!r} belongs to a different family instance; "
                "use helpers on the family instance passed to SuperLSS"
            )
    try:
        owned_family = deepcopy(family)
    except Exception as exc:
        raise TypeError("family configuration could not be independently snapshotted") from exc
    if owned_family is family:
        raise TypeError("family configuration must support an independent snapshot")
    names = tuple(parameter.name for parameter in validate_family(owned_family))
    by_name: dict[str, BoundPredictor] = {}
    for predictor in values:
        if predictor.name in by_name:
            raise ValueError(f"Duplicate predictor name: {predictor.name}")
        if predictor.name not in names:
            raise ValueError(f"Unknown predictor name: {predictor.name}")
        by_name[predictor.name] = predictor
    missing = tuple(name for name in names if name not in by_name)
    if missing:
        raise ValueError(_missing_message(owned_family, names, missing))
    return owned_family, tuple(by_name[name].template for name in names)
