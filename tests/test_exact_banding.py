"""Tests for exact contiguous banding of a fitted curve."""

import itertools
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, Polynomial, Spline, SuperGLM
from superglm.diagnostics.discretize import (
    _compute_edges,
    _term_se_at,
    _validated_discretization_weights,
)
from superglm.diagnostics.exact_banding import (
    MAX_EXACT_VALUES,
    _fewest_then_least,
    exact_bands,
)
from superglm.export._ppform import extract_ppform
from superglm.export.rating_tables import build_rating_table_payload

_EPS = float(np.finfo(np.float64).eps)


def _brute_force(s, w, tol):
    """(band count, weighted squared error) minimised lexicographically over every banding."""
    n = len(s)
    best = None
    for r in range(n):
        for cuts in itertools.combinations(range(1, n), r):
            starts = (0, *cuts)
            ends = (*cuts, n)
            total = 0.0
            feasible = True
            for a, b in zip(starts, ends, strict=True):
                mean = np.average(s[a:b], weights=w[a:b])
                if b - a > 1 and np.any(np.abs(s[a:b] - mean) > tol[a:b]):
                    feasible = False
                    break
                total += float(np.sum(w[a:b] * (s[a:b] - mean) ** 2))
            if feasible and (best is None or (len(starts), total) < best):
                best = (len(starts), total)
    return best


def test_matches_exhaustive_search_on_random_curves():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 11))
        s = np.cumsum(rng.normal(0.0, 0.3, n))
        w = rng.uniform(0.1, 3.0, n)
        tol = rng.uniform(0.0, 0.4, n)
        tol[rng.random(n) < 0.15] = 0.0
        result = exact_bands(s, w, tol, max_bands=n)
        count, sse = _brute_force(s, w, tol)
        # Each band cost is sum(w s^2) - W m^2; its rounding is a small multiple
        # of eps * sum(w s^2), and n <= 10 values bound the multiple by 256.
        scale = 256 * _EPS * (1.0 + float(np.sum(w * s * s)))
        assert len(result.starts) == count
        assert abs(result.sse - sse) <= scale


def test_two_obvious_bands():
    s = np.array([0.0, 0.05, 1.0, 1.05])
    result = exact_bands(s, np.ones(4), np.full(4, 0.1), max_bands=4)
    assert result.starts.tolist() == [0, 2]
    np.testing.assert_allclose(result.factors, [0.025, 1.025], rtol=0, atol=4 * _EPS)
    assert result.tolerance_factor == 1.0


def test_every_value_is_within_its_tolerance_of_its_band():
    rng = np.random.default_rng(1)
    s = np.cumsum(rng.normal(0.0, 0.05, 60))
    w = rng.uniform(0.1, 3.0, 60)
    tol = rng.uniform(0.01, 0.1, 60)
    result = exact_bands(s, w, tol, max_bands=60)
    ends = np.append(result.starts[1:], 60)
    for a, b, factor in zip(result.starts, ends, result.factors, strict=True):
        slack = 4 * _EPS * (b - a) * (1.0 + np.abs(s[a:b]).max())
        assert np.all(np.abs(s[a:b] - factor) <= tol[a:b] + slack)


def test_constant_curve_with_zero_tolerance_is_one_band():
    s = np.full(25, 0.3)
    result = exact_bands(s, np.linspace(0.5, 2.0, 25), np.zeros(25), max_bands=25)
    assert result.starts.tolist() == [0]


def test_zero_tolerance_values_can_always_stand_alone():
    s = np.array([0.0, 1.0, 2.0])
    result = exact_bands(s, np.ones(3), np.zeros(3), max_bands=3)
    assert result.starts.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("s", "w", "tol", "max_bands", "match"),
    [
        ([0.0, 1.0], [1.0], [0.1, 0.1], 2, "same length"),
        ([], [], [], 1, "at least one value"),
        ([0.0, np.nan], [1.0, 1.0], [0.1, 0.1], 2, "finite"),
        ([0.0, 1.0], [1.0, 0.0], [0.1, 0.1], 2, "positive"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, -0.1], 2, "nonnegative"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, 0.1], 0, "max_bands"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, 0.1], True, "max_bands"),
    ],
)
def test_rejects_bad_inputs(s, w, tol, max_bands, match):
    with pytest.raises(ValueError, match=match):
        exact_bands(np.asarray(s), np.asarray(w), np.asarray(tol), max_bands=max_bands)


def test_rejects_too_many_values():
    n = MAX_EXACT_VALUES + 1
    with pytest.raises(ValueError, match="at most 5000"):
        exact_bands(np.zeros(n), np.ones(n), np.zeros(n), max_bands=10)


def test_a_small_cap_widens_the_tolerance_just_enough():
    s = np.linspace(0.0, 1.0, 50) ** 2
    w = np.ones(50)
    tol = np.full(50, 0.01)
    result = exact_bands(s, w, tol, max_bands=5)
    assert len(result.starts) <= 5
    assert result.tolerance_factor > 1.0
    tighter = result.tolerance_factor / (1.0 + 2e-6)
    assert len(_fewest_then_least(s, w, tighter * tol)[0]) > 5
    ends = np.append(result.starts[1:], 50)
    for a, b, factor in zip(result.starts, ends, result.factors, strict=True):
        slack = 4 * _EPS * (b - a) * 2.0
        assert np.all(np.abs(s[a:b] - factor) <= result.tolerance_factor * tol[a:b] + slack)


def test_no_widening_when_the_cap_is_not_binding():
    s = np.linspace(0.0, 1.0, 50) ** 2
    result = exact_bands(s, np.ones(50), np.full(50, 0.01), max_bands=50)
    assert result.tolerance_factor == 1.0


def test_zero_tolerances_that_cannot_merge_are_refused():
    with pytest.raises(ValueError, match="cannot fit"):
        exact_bands(np.arange(6.0), np.ones(6), np.zeros(6), max_bands=3)


def test_rounding_slack_cannot_admit_a_band_that_breaks_a_tolerance():
    # Two values 7 ulps apart near 100, tolerances of 4.2 ulps, the weight on the
    # upper one: their mean lands 7 ulps from the lower value, past its 4.2. A
    # slack scaled by the curve's level (100) let that band through.
    u = np.spacing(100.0)
    s = np.array([100.0, 100.0 + 7.0 * u])
    result = exact_bands(s, np.array([1.0, 1e6]), np.full(2, 4.2 * u), max_bands=2)
    assert result.starts.tolist() == [0, 1]


def test_a_tiny_positive_tolerance_widens_by_its_finite_factor():
    # The least factor is 0.5 / 1e-30 = 5e29; a fixed 2**60 limit called it impossible.
    result = exact_bands(np.array([0.0, 1.0]), np.ones(2), np.full(2, 1e-30), max_bands=1)
    assert result.starts.tolist() == [0]
    assert result.tolerance_factor == pytest.approx(5e29, rel=2e-6)


def test_a_plateau_merges_whatever_its_tolerances():
    # A bound on the widest tolerance used to shrink a zero-width window past the
    # exact mean of two equal values, and the widening then reported a factor of 0.
    result = exact_bands(np.array([1.0, 1.0]), np.ones(2), np.array([0.0, 0.1]), max_bands=1)
    assert result.starts.tolist() == [0]
    assert result.tolerance_factor == 1.0


@pytest.mark.parametrize(
    ("s", "tol", "least"),
    [
        # sqrt(lower * upper) overflowed once the bounds passed about 1e154.
        (np.array([0.0, 1.0]), np.full(2, 1e-200), 5e199),
        # 2 R / tau overflowed, though a factor of 1e308 fits.
        (np.array([0.0, 1.0, 2.0]), np.array([1e-308, 1.0, 1e-308]), 1e308),
    ],
)
def test_the_widening_search_survives_factors_near_the_top_of_double(s, tol, least):
    result = exact_bands(s, np.ones(len(s)), tol, max_bands=1)
    assert result.starts.tolist() == [0]
    assert result.tolerance_factor == pytest.approx(least, rel=2e-6)


def test_weights_near_the_largest_double_do_not_overflow_the_mean():
    # w * d was 1e308 * -10 = -inf; only ratios of weights matter.
    result = exact_bands(
        np.array([0.0, 10.0]), np.array([1e308, 1e10]), np.array([1.0, 11.0]), max_bands=2
    )
    assert result.starts.tolist() == [0]
    assert np.isfinite(result.sse)


@pytest.mark.parametrize("top", [1e200, 1e308])
def test_a_curve_too_wide_for_its_moments_is_refused(top):
    # w * d * d overflows once |d| passes about 1.3e154; the band count survived
    # but sse came back NaN and ties fell to whichever NaN argmin met first.
    with pytest.raises(ValueError, match="below 2\\*\\*500"):
        exact_bands(np.array([0.0, top]), np.ones(2), np.full(2, top), max_bands=1)


def test_a_product_that_underflows_cannot_carry_a_band_past_its_tolerance():
    # w * d = 2**-1020 * -2**-60 rounds to zero, so the computed mean of {0, 2**-60}
    # was 0 and the band was accepted with the second value 2**-61 from the exact
    # mean against a tolerance of 2**-100.
    result = exact_bands(
        np.array([0.0, 2.0**-60, 1.0]),
        np.array([2.0**-1020, 2.0**-1020, 1.0]),
        np.array([2.0**-59, 2.0**-100, 0.0]),
        max_bands=3,
    )
    assert result.starts.tolist() == [0, 1, 2]


def test_the_acceptance_margin_certifies_ties_at_rounding_level():
    """Tolerances set to each value's exact distance from the exact mean, rounded.

    Rounding to nearest leaves about half of them short of feasibility by less
    than the computed mean's own rounding; the margin has to refuse those, so
    every band returned meets every tolerance in exact arithmetic.
    """
    rng = np.random.default_rng(23)
    for _ in range(400):
        k = int(rng.integers(2, 6))
        s = 100.0 + np.cumsum(rng.normal(0.0, 1e-3, k))
        w = rng.uniform(0.1, 3.0, k) ** 3
        exact_mean = sum(Fraction(a) * Fraction(b) for a, b in zip(w, s)) / sum(map(Fraction, w))
        tol = np.array([float(abs(exact_mean - Fraction(value))) for value in s])
        result = exact_bands(s, w, tol, max_bands=k)
        ends = np.append(result.starts[1:], k)
        for a, b in zip(result.starts, ends, strict=True):
            band_s = list(map(Fraction, s[a:b]))
            weights = list(map(Fraction, w[a:b]))
            mean = sum(x * y for x, y in zip(weights, band_s)) / sum(weights)
            assert all(abs(mean - value) <= Fraction(t) for value, t in zip(band_s, tol[a:b]))


def test_the_acceptance_margin_does_not_cost_bands_it_can_certify():
    # Eight times the margin clears any rounding, so one band must be taken.
    rng = np.random.default_rng(29)
    for _ in range(100):
        k = int(rng.integers(2, 6))
        s = 100.0 + np.cumsum(rng.normal(0.0, 1e-3, k))
        w = rng.uniform(0.1, 3.0, k) ** 3
        # In the frame of the last value, as the banding measures, so the
        # tolerances are not rounded at the curve's level of 100.
        d = s - s[-1]
        mean = np.average(d, weights=w)
        margin = 16.0 * _EPS * (k + 2) * np.abs(d).max()
        result = exact_bands(s, w, np.abs(d - mean) + margin, max_bands=k)
        assert result.starts.tolist() == [0]


def test_weights_spanning_more_than_a_double_are_refused():
    with pytest.raises(ValueError, match="span more than a double can average"):
        exact_bands(np.array([0.0, 1.0]), np.array([1e308, 1e-20]), np.zeros(2), max_bands=2)


@pytest.fixture(scope="module")
def banded_model():
    """Integer ages (63 values) with a steep young-driver effect, and a polynomial."""
    rng = np.random.default_rng(7)
    n = 4000
    age = rng.integers(18, 81, n).astype(float)
    density = rng.integers(0, 11, n).astype(float)
    eta = (
        -2.0 + 0.6 * np.exp(-(age - 18.0) / 6.0) + 0.02 * (age - 50.0) ** 2 / 50.0 + 0.05 * density
    )
    y = rng.poisson(np.exp(eta)).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame({"age": age, "density": density})
    model = SuperGLM(features={"age": Spline(n_knots=8), "density": Polynomial(degree=2)})
    model.fit(df, y, sample_weight=w)
    return model, df, y, w


def _row_of(table, value):
    last = len(table) - 1
    for k, row in table.iterrows():
        if row["bin_from"] <= value < row["bin_to"] or (k == last and value >= row["bin_from"]):
            return row
    raise AssertionError(f"no row holds {value}")


def test_exact_tables_follow_the_limit(banded_model):
    model, df, y, w = banded_model
    result = model.discretization_impact(
        df, y, sample_weight=w, n_bins=150, bin_strategy="exact", features=["age"]
    )
    table = result.tables["age"]
    diag = result.band_diagnostics["age"]
    assert len(table) == diag["bands"]
    assert set(table["bin_from"]) <= set(np.unique(df["age"]))
    assert diag["tolerance_factor"] == 1.0
    values = np.unique(df["age"].to_numpy())
    curve = extract_ppform(model, "age").evaluate(values)
    se = _term_se_at(model, "age", values)
    tol = np.minimum(se, np.log1p(0.10))
    for value, s_v, tol_v in zip(values, curve, tol, strict=True):
        row = _row_of(table, value)
        # ppform reproduces the fitted curve to its certified 1e-11.
        assert abs(s_v - row["log_relativity"]) <= tol_v + 1e-9


def test_band_factor_is_the_weighted_mean_of_the_curve(banded_model):
    model, df, y, w = banded_model
    result = model.discretization_impact(
        df, y, sample_weight=w, n_bins=150, bin_strategy="exact", features=["age"]
    )
    table = result.tables["age"]
    age = df["age"].to_numpy()
    curve = extract_ppform(model, "age").evaluate(age)
    # Bands average with the geometry mass the other strategies use: replication
    # mass under frequency weights, one unit per physical row under prior weights.
    _, geometry = _validated_discretization_weights(model, w, len(df))
    last = len(table) - 1
    for k, row in table.iterrows():
        inside = (age >= row["bin_from"]) & ((age < row["bin_to"]) | (k == last))
        expected = np.average(curve[inside], weights=geometry[inside])
        assert abs(row["log_relativity"] - expected) <= 1e-9


def test_every_value_is_its_own_band_when_the_limit_is_tiny(banded_model):
    model, df, y, w = banded_model
    result = model.discretization_impact(
        df,
        y,
        sample_weight=w,
        n_bins=100,
        bin_strategy="exact",
        band_max_error=1e-12,
        features=["age"],
    )
    table = result.tables["age"]
    assert len(table) == df["age"].nunique()
    last = table.iloc[-1]
    assert last["bin_from"] == last["bin_to"] == df["age"].max()
    assert last["n_obs"] == int((df["age"] == df["age"].max()).sum())


def test_a_small_cap_reports_the_widened_limit(banded_model):
    model, df, y, w = banded_model
    result = model.discretization_impact(
        df, y, sample_weight=w, n_bins=5, bin_strategy="exact", features=["age"]
    )
    assert len(result.tables["age"]) <= 5
    assert result.band_diagnostics["age"]["tolerance_factor"] > 1.0


def test_spline_and_polynomial_are_both_banded(banded_model):
    model, df, y, w = banded_model
    result = model.discretization_impact(df, y, sample_weight=w, n_bins=150, bin_strategy="exact")
    assert set(result.band_diagnostics) == {"age", "density"}
    assert set(result.tables) == {"age", "density"}


def test_zero_weight_rows_do_not_move_edges(banded_model):
    model, df, y, w = banded_model
    w0 = w.copy()
    w0[df["age"].to_numpy() == 18.0] = 0.0
    result = model.discretization_impact(
        df, y, sample_weight=w0, n_bins=150, bin_strategy="exact", features=["age"]
    )
    table = result.tables["age"]
    assert table["bin_from"].iloc[0] == 19.0
    assert table["n_obs"].sum() == len(df)


def test_term_se_at_matches_the_library_grid(banded_model):
    model, _, _, _ = banded_model
    cov, active = model._coef_covariance
    for name in ("age", "density"):
        spec = model._specs[name]
        grid = np.linspace(spec._lo, spec._hi, 50)
        expected = model._feature_se_from_cov(name, cov, active, n_points=50)
        np.testing.assert_allclose(_term_se_at(model, name, grid), expected, rtol=64 * _EPS, atol=0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, True, "1"])
def test_exact_rejects_bad_band_settings(banded_model, bad):
    model, df, y, w = banded_model
    with pytest.raises(ValueError, match="band_se"):
        model.discretization_impact(df, y, sample_weight=w, bin_strategy="exact", band_se=bad)
    with pytest.raises(ValueError, match="band_max_error"):
        model.discretization_impact(
            df, y, sample_weight=w, bin_strategy="exact", band_max_error=bad
        )


def test_band_settings_are_checked_under_every_strategy(banded_model):
    # One contract with the export, which checks them whatever is binned.
    model, df, y, w = banded_model
    with pytest.raises(ValueError, match="band_se must be a positive finite number"):
        model.discretization_impact(
            df, y, sample_weight=w, n_bins=10, bin_strategy="exposure_quantile", band_se=-1.0
        )
    result = model.discretization_impact(
        df, y, sample_weight=w, n_bins=10, bin_strategy="exposure_quantile"
    )
    assert result.band_diagnostics == {}


def test_exact_is_refused_where_there_is_no_curve():
    with pytest.raises(ValueError, match="no fitted curve"):
        _compute_edges(np.arange(5.0), np.ones(5), 3, "exact")


def test_too_many_values_names_the_feature():
    rng = np.random.default_rng(3)
    n = MAX_EXACT_VALUES + 500
    df = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n)})
    y = rng.poisson(1.0, n).astype(float)
    model = SuperGLM(features={"x": Spline(n_knots=5)})
    model.fit(df, y)
    with pytest.raises(ValueError, match="'x' has .* distinct values"):
        model.discretization_impact(df, y, bin_strategy="exact")


def test_payload_uses_the_exact_bands(banded_model):
    model, df, y, w = banded_model
    impact = model.discretization_impact(
        df, y, sample_weight=w, n_bins=150, bin_strategy="exact", band_max_error=0.05
    )
    payload = build_rating_table_payload(
        model,
        df,
        y,
        sample_weight=w,
        n_bins=150,
        impact_bins=(),
        bin_strategy="exact",
        band_max_error=0.05,
    )
    block = next(b for b in payload.main_effects if b.name == "age")
    assert len(block.table) == impact.band_diagnostics["age"]["bands"]


def test_payload_sweep_carries_the_settings(banded_model):
    model, df, y, w = banded_model
    payload = build_rating_table_payload(
        model,
        df,
        y,
        sample_weight=w,
        n_bins=150,
        impact_bins=(150,),
        bin_strategy="exact",
        band_max_error=1e-12,
    )
    ages = payload.discretization_impact.query("feature == 'age'")
    assert ages["actual_bins"].tolist() == [df["age"].nunique()]


def test_export_warns_when_the_cap_widens_the_limit(banded_model):
    model, df, y, w = banded_model
    with pytest.warns(UserWarning, match="widened"):
        build_rating_table_payload(
            model, df, y, sample_weight=w, n_bins=5, impact_bins=(), bin_strategy="exact"
        )


def test_export_is_quiet_when_the_limit_holds(banded_model):
    model, df, y, w = banded_model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_rating_table_payload(
            model, df, y, sample_weight=w, n_bins=150, impact_bins=(), bin_strategy="exact"
        )
    assert not any("widened" in str(item.message) for item in caught)


def test_impact_sheet_shows_the_band_limit(banded_model):
    model, df, y, w = banded_model
    with pytest.warns(UserWarning, match="widened"):
        payload = build_rating_table_payload(
            model, df, y, sample_weight=w, n_bins=5, impact_bins=(5,), bin_strategy="exact"
        )
    row = payload.discretization_impact.query("feature == 'age'").iloc[0]
    assert row["band_tolerance_factor"] > 1.0
    assert row["band_worst_error"] > 0.10


def test_band_error_is_the_band_factors_error_against_the_curve(banded_model):
    # A band log(1.1) above the curve is 10% off it, not the 9.1% the curve is off the band.
    model, df, y, w = banded_model
    result = model.discretization_impact(
        df, y, sample_weight=w, n_bins=5, bin_strategy="exact", features=["age"]
    )
    table = result.tables["age"]
    age = df["age"].to_numpy()
    curve = extract_ppform(model, "age").evaluate(age)
    band = np.array([_row_of(table, value)["log_relativity"] for value in age])
    _, geometry = _validated_discretization_weights(model, w, len(df))
    relative = np.abs(np.expm1(band - curve))
    diagnostics = result.band_diagnostics["age"]
    assert diagnostics["worst_error"] == pytest.approx(relative.max(), rel=1e-8)
    assert diagnostics["mean_error"] == pytest.approx(
        np.average(relative, weights=geometry), rel=1e-8
    )


def test_exact_bands_refuse_a_post_fit_repaired_term():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 200)
    y = -((x - 0.35) ** 2) + 0.05 * rng.normal(size=len(x))
    df = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        features={"x": Spline(kind="ps", n_knots=10, constraint=Constraint.postfit.convex)},
    ).fit(df, y)
    model.apply_shape_postfit(df)
    assert model._shape_repairs["x"]
    with pytest.raises(ValueError, match="post-fit shape repair"):
        model.discretization_impact(df, y, bin_strategy="exact")


def test_a_single_value_last_band_exports_a_closed_key(banded_model):
    # Its half-open key [x, x) would match nothing, not even the rows it rates.
    model, df, y, w = banded_model
    payload = build_rating_table_payload(
        model,
        df,
        y,
        sample_weight=w,
        n_bins=100,
        impact_bins=(),
        bin_strategy="exact",
        band_max_error=1e-12,
    )
    keys = next(b for b in payload.main_effects if b.name == "age").table["age"].tolist()
    top = float(df["age"].max())
    assert keys[-1] == f"[{top!r}, {top!r}]"
    assert all(key.endswith(")") for key in keys[:-1])


def test_a_binned_offset_is_refused_before_any_band_is_solved(monkeypatch):
    from superglm.diagnostics import exact_banding

    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 300)})
    exposure = rng.uniform(0.5, 2.0, 300)
    y = rng.poisson(exposure * np.exp(0.3 * df["x"].to_numpy())).astype(float)
    model = SuperGLM(features={"x": Spline(n_knots=5)}).fit(df, y, offset=np.log(exposure))

    def solved(*args, **kwargs):
        raise AssertionError("a band was solved before the refusal")

    monkeypatch.setattr(exact_banding, "exact_bands", solved)
    with pytest.raises(ValueError, match="binned offset has no fitted curve"):
        build_rating_table_payload(
            model,
            df,
            y,
            offset=np.log(exposure),
            impact_bins=(),
            bin_strategy="exact",
            offset_kind="binned",
        )


def test_band_settings_are_checked_even_when_nothing_is_binned(banded_model):
    model, df, y, w = banded_model
    with pytest.raises(ValueError, match="band_se must be a positive finite number"):
        build_rating_table_payload(
            model, df, y, sample_weight=w, impact_bins=(), continuous_kind="ppform", band_se=-1.0
        )


def test_exact_bands_refuse_a_model_carrying_editor_edits(banded_model):
    # Its standard errors are the fit's before the edit, as term_inference says.
    from superglm.editor import EditorSession

    model, df, y, w = banded_model
    session = EditorSession.from_model(model, terms=["age"])
    session.select_x("age", 30.0, 40.0)
    session.shift("age", 0.1)
    edited = session.to_model()
    with pytest.raises(ValueError, match="Editor coefficient edits"):
        edited.discretization_impact(df, y, sample_weight=w, bin_strategy="exact", features=["age"])


def test_only_a_last_single_value_band_gets_a_closed_key():
    # Repeated uniform edges give zero-width rows too; they stay half-open and
    # empty, so a consumer never sees two matching rows.
    from superglm.export.rating_tables import _continuous_block

    table = pd.DataFrame(
        {
            "bin_from": [0.0, 1.0, 1.0, 2.0],
            "bin_to": [1.0, 1.0, 2.0, 2.0],
            "relativity": [1.0, 1.1, 1.2, 1.3],
            "sample_weight": [1.0, 0.0, 1.0, 1.0],
        }
    )
    keys = _continuous_block("x", table, 0.0).table["x"].tolist()
    assert keys == ["[0.0, 1.0)", "[1.0, 1.0)", "[1.0, 2.0)", "[2.0, 2.0]"]


def test_a_weight_near_the_largest_double_keeps_the_table_finite():
    # One replication weight of 1e308 at a value whose log relativity is near 2:
    # its weight * value overflowed in the table's own average of that band.
    rng = np.random.default_rng(11)
    x = rng.integers(0, 21, 3000).astype(float) / 20.0
    y = rng.poisson(np.exp(-1.0 + 4.0 * x)).astype(float)
    df = pd.DataFrame({"x": x})
    model = SuperGLM(features={"x": Spline(n_knots=6)}, weight_semantics="frequency").fit(df, y)
    weights = np.ones(len(df))
    weights[np.flatnonzero(x == x.max())[0]] = 1e308
    result = model.discretization_impact(
        df, y, sample_weight=weights, n_bins=150, bin_strategy="exact"
    )
    assert np.isfinite(result.tables["x"]["log_relativity"]).all()


def test_exact_tables_export_the_certified_factors(banded_model, monkeypatch):
    # Re-averaging the rows could differ from the certified factor by a rounding,
    # which a tolerance below that rounding would then see as a breach.
    from superglm.diagnostics import exact_banding

    model, df, y, w = banded_model
    certified = []
    solve = exact_banding.exact_bands

    def recorded(*args, **kwargs):
        certified.append(solve(*args, **kwargs))
        return certified[-1]

    monkeypatch.setattr(exact_banding, "exact_bands", recorded)
    result = model.discretization_impact(
        df, y, sample_weight=w, n_bins=150, bin_strategy="exact", features=["age"]
    )
    np.testing.assert_array_equal(
        result.tables["age"]["log_relativity"].to_numpy(), certified[0].factors
    )
