"""Regressions for retained nullable bounds and side-aware censoring diagnostics."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import (
    GaussianLS,
    GeneralizedParetoLSS,
    SeparationError,
    SeparationWarning,
    SuperLSS,
    cat,
)
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.separation import censored_recession
from superglm.links import InverseLink, InverseSquaredLink, PowerLink, SqrtLink


def _gaussian_model():
    family = GaussianLS()
    return SuperLSS(family, family.location(), family.scale())


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("missing", [pd.NA, np.nan, None], ids=["NA", "NaN", "None"])
def test_missing_truncation_bound_on_a_dropped_row_matches_dropping_the_row(side, missing):
    X = pd.DataFrame({"one": np.ones(5)})
    y = np.arange(-2.0, 3.0)
    limit = -3.0 if side == "lower" else 3.0
    bounds = [missing, limit, limit, limit, limit]
    truncation = (bounds, None) if side == "lower" else (None, bounds)
    retained = (limit, None) if side == "lower" else (None, limit)

    expected = _gaussian_model().fit(X.iloc[1:], y[1:], truncation=retained)
    actual = _gaussian_model().fit(
        X, y, sample_weight=[0.0, 1.0, 1.0, 1.0, 1.0], truncation=truncation
    )

    np.testing.assert_allclose(
        actual.predict_parameters(X), expected.predict_parameters(X), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("missing", [pd.NA, np.nan, None], ids=["NA", "NaN", "None"])
def test_missing_retained_truncation_bound_names_the_original_input_row(side, missing):
    X = pd.DataFrame({"one": np.ones(5)})
    limit = -3.0 if side == "lower" else 3.0
    bounds = [limit, missing, limit, limit, limit]
    truncation = (bounds, None) if side == "lower" else (None, bounds)

    with pytest.raises(ValueError, match=r"1 row\(s\).*missing.*first at input row 1"):
        _gaussian_model().fit(
            X,
            np.arange(-2.0, 3.0),
            sample_weight=[0.0, 1.0, 1.0, 1.0, 1.0],
            truncation=truncation,
        )


def test_nullable_float_bounds_and_scalar_open_sides_keep_their_meaning():
    X = pd.DataFrame({"one": np.ones(5)})
    y = np.arange(-2.0, 3.0)
    expected = _gaussian_model().fit(X.iloc[1:], y[1:], truncation=(-3.0, None))
    actual = _gaussian_model().fit(
        X,
        y,
        sample_weight=[0.0, 1.0, 1.0, 1.0, 1.0],
        truncation=(pd.Series([pd.NA, -3.0, -3.0, -3.0, -3.0], dtype="Float64"), None),
    )
    np.testing.assert_allclose(
        actual.predict_parameters(X), expected.predict_parameters(X), rtol=1e-12, atol=1e-12
    )


def _censored_level_fixture(side):
    X = pd.DataFrame({"g": ["a"] * 40 + ["b"] * 10})
    y = np.r_[np.tile([1.0, 3.0], 20), np.ones(10)]
    return X, y, ["none"] * 40 + [side] * 10


@pytest.mark.parametrize("link", ["sqrt", PowerLink(0.5), PowerLink(1.0)])
def test_right_censored_location_separates_when_the_upper_range_end_is_only_at_infinity(link):
    X, y, flags = _censored_level_fixture("right")
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(cat("g", base="a"), link=link),
        family.scale(),
        separation="error",
    )
    with pytest.raises(SeparationError, match=r"'location:g'.*only right-censored rows: 'b'"):
        model.fit(X, y, censored=flags)


def test_left_censored_sqrt_location_keeps_its_finite_boundary_estimate():
    X, y, flags = _censored_level_fixture("left")
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(cat("g", base="a"), link="sqrt"),
        family.scale(),
        separation="error",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(X, y, censored=flags)
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    # The likelihood increases as mu_b falls to its attainable lower end, zero.
    assert fitted["location"].iloc[1] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(fitted.to_numpy()).all()


def test_right_censored_pareto_scale_on_a_positive_power_link_separates():
    X, y, flags = _censored_level_fixture("right")
    family = GeneralizedParetoLSS()
    model = SuperLSS(
        family,
        family.scale(cat("g", base="a"), link=PowerLink(0.5)),
        family.shape(),
        separation="error",
    )
    with pytest.raises(SeparationError, match=r"'scale:g'.*only right-censored rows: 'b'"):
        model.fit(X, y, censored=flags)


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (SqrtLink(), [False, True]),
        (PowerLink(0.5), [False, True]),
        (PowerLink(1.0), [True, True]),
        (PowerLink(-1.0), [True, False]),
        (InverseLink(), [True, False]),
        (InverseSquaredLink(), [True, False]),
    ],
)
def test_built_in_monotone_flags_follow_each_links_reachable_endpoint(link, expected):
    bounds = ResponseBounds.from_arrays(np.full(2, -np.inf), np.full(2, np.inf), np.array([-1, 1]))
    compiled = [
        SimpleNamespace(link=link, parameter_index=0),
        SimpleNamespace(link=GaussianLS().parameters[1].default_link, parameter_index=1),
    ]
    flags = censored_recession(GaussianLS(), bounds, compiled)
    np.testing.assert_array_equal(flags[:, 0], expected)


def test_custom_censored_flags_do_not_inherit_a_builtin_direction_assumption():
    class CustomGaussian(GaussianLS):
        def censored_recession(self, bounds):
            return np.ones((bounds.n_rows, 2), dtype=bool)

    bounds = ResponseBounds.from_arrays(np.full(2, -np.inf), np.full(2, np.inf), np.array([-1, 1]))
    compiled = [
        SimpleNamespace(link=SqrtLink(), parameter_index=0),
        SimpleNamespace(link=GaussianLS().parameters[1].default_link, parameter_index=1),
    ]
    flags = censored_recession(CustomGaussian(), bounds, compiled)
    np.testing.assert_array_equal(flags, [[False, True], [False, True]])
