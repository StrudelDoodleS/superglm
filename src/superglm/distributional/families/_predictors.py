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
        """Declare the Tweedie conditional-mean predictor.

        Pass numeric column names and terms such as ``s("age")`` or
        ``cat("region")``. The default link is log, so term effects multiply
        the mean after exponentiation. Results and offsets use ``"mean"``.

        Calling ``mu()`` with no terms declares an intercept-only predictor.
        Set ``intercept=False`` to omit its intercept, or ``link=`` to supply
        a link supported by the family. Pass the declaration to ``SuperLSS``
        with this same family instance, plus ``phi(...)`` and ``p(...)``.
        """
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "mean", *terms, intercept=intercept, link=link
        )

    def phi(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the Tweedie dispersion predictor.

        At unit prior weight, variance is ``phi * mean**power``. The default
        link is log. Results and offsets use the name ``"dispersion"``.

        Pass numeric column names and bound terms. Calling ``phi()`` with
        no terms declares an intercept-only predictor. ``intercept=False``
        omits its intercept; ``link=`` overrides the family's default link
        where supported. This declaration is required alongside ``mu`` and
        ``p`` when constructing ``SuperLSS``.
        """
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "dispersion", *terms, intercept=intercept, link=link
        )

    def p(
        self, *terms: TermInput, intercept: bool = True, link: str | Link | None = None
    ) -> BoundPredictor:
        """Declare the Tweedie variance-power predictor.

        The family's default link keeps power inside the configured
        ``power_lower`` and ``power_upper`` bounds. Results and offsets use
        the name ``"power"``.

        Pass numeric column names and bound terms, or call ``p()`` for an
        intercept-only predictor. The declaration must be present alongside
        ``mu`` and ``phi``. ``intercept=False`` omits its intercept; a custom
        ``link`` must preserve the family's permitted power range.
        """
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
        """Declare a predictor for the conditional response mean.

        Pass numeric column names and bound terms such as ``s("age")`` and
        ``cat("region")``. The family supplies the default link. An empty
        call declares an intercept-only predictor; ``intercept=False`` omits
        the intercept. ``link=`` overrides the default where supported.

        This helper is valid when ``mean`` is a parameter of the selected
        family parametrization. Pass its result to ``SuperLSS`` with the
        same family instance and a declaration for every other parameter.
        """
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
        """Declare the family's location predictor.

        Gaussian location is the response mean. In lognormal, generalized
        gamma and two-piece lognormal location forms, it is a location on
        the log-response scale. Check the family definition before treating
        location as a mean or median.

        Pass numeric column names and bound terms, or call ``location()``
        for an intercept-only predictor. ``intercept=False`` omits the
        intercept; ``link=`` overrides the default where supported. Families
        with a ``parametrisation`` option require its ``"location"`` form
        for this helper.
        """
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
        """Declare the family's scale predictor.

        Scale is the standard deviation for ``GaussianLS``, the coefficient
        of variation for ``GammaLS``, and the standard deviation of the log
        response for ``LogNormalLS``. Other families define their own scale
        parameter; it need not be a standard deviation.

        Pass numeric column names and bound terms, or call ``scale()`` for
        an intercept-only predictor. ``intercept=False`` omits its intercept;
        ``link=`` overrides the default where supported. The family's link
        and support determine the permitted scale values.
        """
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
        """Declare the negative-binomial size predictor.

        For NB2, variance is ``mean + mean**2 / theta``. Larger ``theta``
        means less overdispersion. The default link is log, and results and
        offsets use the name ``"theta"``.

        Pass numeric column names and bound terms, or call ``theta()`` for
        an intercept-only size predictor. ``intercept=False`` omits its
        intercept; ``link=`` overrides the default where supported.
        """
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
        """Declare the family's shape predictor.

        For generalized Pareto this is the tail-shape parameter xi; for
        generalized gamma it is Prentice's Q. Their domains and default links
        differ, so the selected family owns those choices.

        Pass numeric column names and bound terms, or call ``shape()`` for
        an intercept-only predictor. ``intercept=False`` omits its intercept;
        a custom ``link`` must respect the family's parameter support.
        """
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
        """Declare the two-piece family's asymmetry predictor.

        This is the epsilon parameter controlling the two piece widths,
        rather than the standardized third moment. Positive values make
        the right piece wider, on the log-response scale for
        ``TwoPieceLogNormalLSS`` and on the response scale for
        ``TwoPieceNormalLSS``.

        Pass numeric column names and bound terms, or call ``skew()`` for an
        intercept-only predictor. ``intercept=False`` omits its intercept.
        The default link keeps values within the family's skew bounds;
        a custom ``link`` must respect that support.
        """
        from superglm.distributional.binding import bind_predictor

        return bind_predictor(
            cast("DistributionalFamily", self), "skew", *terms, intercept=intercept, link=link
        )
