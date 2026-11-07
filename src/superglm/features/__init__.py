import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.features.categorical import Categorical
    from superglm.features.constraint import Constraint, ConstraintRules, ConstraintSpec
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.grouping import LevelGrouping, collapse_levels
    from superglm.features.interaction import (
        CategoricalInteraction,
        NumericCategorical,
        NumericInteraction,
        PolynomialCategorical,
        PolynomialInteraction,
        SplineCategorical,
    )
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import (
        BSplineSmooth,
        CubicRegressionSpline,
        NaturalSpline,
        PSpline,
        Spline,
    )

# Each name is read from its module on first access (``superglm._lazy``), so importing the
# package, or one of its modules, loads only what that module imports.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.features.categorical": ("Categorical",),
    "superglm.features.constraint": ("Constraint", "ConstraintRules", "ConstraintSpec"),
    "superglm.features.factor_smooth": ("FactorSmooth",),
    "superglm.features.grouping": ("LevelGrouping", "collapse_levels"),
    "superglm.features.interaction": (
        "CategoricalInteraction",
        "NumericCategorical",
        "NumericInteraction",
        "PolynomialCategorical",
        "PolynomialInteraction",
        "SplineCategorical",
    ),
    "superglm.features.numeric": ("Numeric",),
    "superglm.features.ordered_categorical": ("OrderedCategorical",),
    "superglm.features.random_effect": ("RandomEffect",),
    "superglm.features.spline": (
        "BSplineSmooth",
        "CubicRegressionSpline",
        "NaturalSpline",
        "PSpline",
        "Spline",
    ),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = [
    "Spline",
    "PSpline",
    "BSplineSmooth",
    "NaturalSpline",
    "CubicRegressionSpline",
    "Categorical",
    "OrderedCategorical",
    "Constraint",
    "ConstraintRules",
    "ConstraintSpec",
    "FactorSmooth",
    "RandomEffect",
    "LevelGrouping",
    "collapse_levels",
    "Numeric",
    "SplineCategorical",
    "PolynomialCategorical",
    "NumericCategorical",
    "CategoricalInteraction",
    "NumericInteraction",
    "PolynomialInteraction",
]
