"""Custom splines whose interaction margins read as master 5f28e8e80 read them, and their models.

A subclass of a superglm spline, or an evaluator it hands a tensor, keeps master's margins
whatever its extrapolation policy. ``scripts/make_saved_interaction_margin_fixtures.py`` fits
these cases with master and stores its predictions, and its saved model pickles
``ContinuingPSpline`` under this module's name, so the names here stay as they are.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

import superglm as sg


class ContinuingPSpline(sg.PSpline):
    """A raw basis that continues the end polynomials past the range under every policy."""

    def _raw_basis_matrix(self, x):
        return BSpline.design_matrix(x, self._knots, self.degree, extrapolate=True).toarray()


class _ContinuingRows:
    """A callable evaluator of ``ContinuingPSpline``'s basis, declaring the spline's policy."""

    def __init__(self, spline):
        self.knots = spline._knots.copy()
        self.degree = spline.degree
        self.extrapolation = spline.extrapolation

    def __call__(self, x):
        return BSpline.design_matrix(x, self.knots, self.degree, extrapolate=True).toarray()


class CallableMarginPSpline(ContinuingPSpline):
    """Hands the tensor a callable object in place of its bound raw basis."""

    def tensor_marginal_ingredients(self, x, **kwargs):
        info = super().tensor_marginal_ingredients(x, **kwargs)
        return replace(info, raw_basis_eval=_ContinuingRows(self))


class WrappedPSpline(sg.PSpline):
    """Hands the tensor the stock raw basis wrapped in ``functools.partial``."""

    def tensor_marginal_ingredients(self, x, **kwargs):
        info = super().tensor_marginal_ingredients(x, **kwargs)
        return replace(info, raw_basis_eval=partial(info.raw_basis_eval))


class ConstructorTailPSpline(sg.PSpline):
    """Configures its raw basis from the policy its constructor is given."""

    def __init__(self, extrapolation="clip", **kwargs):
        super().__init__(extrapolation=extrapolation, **kwargs)
        self._extrapolates = extrapolation == "extend"

    def _raw_basis_matrix(self, x):
        if not self._extrapolates:
            x = np.clip(x, self._lo, self._hi)
        design = BSpline.design_matrix(x, self._knots, self.degree, extrapolate=self._extrapolates)
        return design.toarray()


class InRangeRaw(sg.PSpline):
    """A raw basis on the base interval only, refusing a point past it as SciPy's does."""

    def _raw_basis_matrix(self, x):
        return BSpline.design_matrix(x, self._knots, self.degree).toarray()


class InRangeIngredients(sg.PSpline):
    """Hands the tensor an evaluator on the base interval only."""

    def _interior_rows(self, x):
        return BSpline.design_matrix(x, self._knots, self.degree).toarray()

    def tensor_marginal_ingredients(self, x, **kwargs):
        info = super().tensor_marginal_ingredients(x, **kwargs)
        return replace(info, raw_basis_eval=self._interior_rows)


class OneArgumentTail(ContinuingPSpline):
    """The one-argument ``tensor_marginal_ingredients`` signature."""

    def tensor_marginal_ingredients(self, x):
        return super().tensor_marginal_ingredients(x)


def rating_cells() -> tuple[pd.DataFrame, np.ndarray]:
    """Equal-exposure Poisson frequencies on an age x bonus x region grid, ages 25 to 75."""
    age, bonus, region = (
        grid.ravel()
        for grid in np.meshgrid(
            np.linspace(25, 75, 21), np.linspace(50, 100, 11), ["A", "B"], indexing="ij"
        )
    )
    a, b = (age - 50) / 25, (bonus - 75) / 25
    eta = -2.2 + 0.35 * a + 0.14 * b + 0.51 * a * b + (region == "B") * (0.08 + 0.37 * a)
    return pd.DataFrame({"age": age, "bonus": bonus, "region": region}), np.exp(eta)


def quotes() -> pd.DataFrame:
    """Quotes inside the boundary, past it and past the data, at two bonuses and both regions."""
    age, bonus, region = (
        grid.ravel()
        for grid in np.meshgrid(
            [20.0, 25.0, 30.0, 47.5, 65.0, 70.0, 75.0, 80.0, 90.0],
            [55.0, 95.0],
            ["A", "B"],
            indexing="ij",
        )
    )
    return pd.DataFrame({"age": age, "bonus": bonus, "region": region})


_BOUNDARY = (30.0, 65.0)
# case: (age spline, age options, tensor knot counts, () for the parents', or None for a
# region curve). A subclass is fitted on its rows, binned or not, so no case bins.
CASES = {
    "region_curve_continuing": (ContinuingPSpline, {"boundary": _BOUNDARY}, None),
    # a subclass whose raw basis is the stock one, held at the range
    "region_curve_wrapped": (WrappedPSpline, {"boundary": _BOUNDARY}, None),
    "tensor_continuing": (ContinuingPSpline, {"boundary": _BOUNDARY}, ()),
    "tensor_callable": (CallableMarginPSpline, {"boundary": _BOUNDARY}, ()),
    "tensor_wrapped": (WrappedPSpline, {"boundary": _BOUNDARY}, ()),
    "tensor_one_argument": (OneArgumentTail, {"boundary": _BOUNDARY}, ()),
    "tensor_in_range_raw": (InRangeRaw, {}, ()),
    "tensor_in_range_ingredients": (InRangeIngredients, {"boundary": _BOUNDARY}, ()),
    "tensor_constructor_override": (ConstructorTailPSpline, {}, (4, 5)),
}


def custom_margin_model(case: str) -> sg.SuperGLM:
    """One case's model: a custom age spline under ``extend`` and its interaction, Poisson."""
    spline, options, tensor = CASES[case]
    age = spline(n_knots=3, extrapolation="extend", **options)
    if tensor is None:
        terms = [
            sg.term("age", age),
            sg.cat("region", base="A"),
            sg.interaction(sg.SplineCategorical("age", "region")),
        ]
    else:
        terms = [
            sg.term("age", age),
            sg.term("bonus", sg.PSpline(n_knots=3, extrapolation="extend")),
            sg.ti("age", "bonus", n_knots=tensor or None),
        ]
    return sg.SuperGLM(
        family="poisson",
        terms=terms,
        selection_penalty=0.0,
        spline_penalty=0.03,
        n_jobs=1,
    )
