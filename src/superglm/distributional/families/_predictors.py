"""Statically declared family helpers; binding imports remain lazy."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from superglm.distributional.binding import BoundPredictor
    from superglm.distributional.family import DistributionalFamily
    from superglm.links import Link
    from superglm.terms import TermInput


class TweediePredictors:
    """Predictors for the Tweedie mean, dispersion and power."""

    _predictor_helpers = {"mean": "mu", "dispersion": "phi", "power": "p"}

    def mu(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the mean predictor."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "mean", *terms, intercept=intercept, link=link
        )

    def phi(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the dispersion predictor."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "dispersion", *terms, intercept=intercept, link=link
        )

    def p(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the power predictor."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "power", *terms, intercept=intercept, link=link
        )


class MeanPredictor:
    """Typed helper for a family's mean predictor."""

    _predictor_helpers = {"mean": "mean"}

    def mean(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the mean predictor when present in this family's parameterization."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "mean", *terms, intercept=intercept, link=link
        )


class LocationPredictor:
    """Typed helper for a family's location predictor."""

    _predictor_helpers = {"location": "location"}

    def location(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the location predictor when present in this family's parameterization."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "location", *terms, intercept=intercept, link=link
        )


class ScalePredictor:
    """Typed helper for a family's scale predictor."""

    _predictor_helpers = {"scale": "scale"}

    def scale(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the scale predictor when present in this family's parameterization."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "scale", *terms, intercept=intercept, link=link
        )


class ThetaPredictor:
    """Typed helper for a family's theta predictor."""

    _predictor_helpers = {"theta": "theta"}

    def theta(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the theta predictor when present in this family's parameterization."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "theta", *terms, intercept=intercept, link=link
        )


class ShapePredictor:
    """Typed helper for a family's shape predictor."""

    _predictor_helpers = {"shape": "shape"}

    def shape(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the shape predictor when present in this family's parameterization."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "shape", *terms, intercept=intercept, link=link
        )


class SkewPredictor:
    """Typed helper for a family's skew predictor."""

    _predictor_helpers = {"skew": "skew"}

    def skew(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the skew predictor when present in this family's parameterization."""
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "skew", *terms, intercept=intercept, link=link
        )
